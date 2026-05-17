#include "shim.h"

#include <OpenColorIO/OpenColorIO.h>
#include <lcms2.h>
#include <lcms2_plugin.h>

#include <algorithm>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace OCIO = OCIO_NAMESPACE;

struct OcioConfig {
    OCIO::ConstConfigRcPtr ptr;
    std::string scratch;
    std::vector<uint8_t> byte_scratch;
};

struct OcioProcessor {
    OCIO::ConstProcessorRcPtr ptr;
    std::string scratch;
};

struct OcioCpuProcessor {
    OCIO::ConstCPUProcessorRcPtr ptr;
    std::string scratch;
};

namespace {
thread_local std::string last_error;

void clear_error() {
    last_error.clear();
}

void set_error(const char *message) {
    last_error = message ? message : "unknown OCIO error";
}

void set_error(const std::exception &error) {
    set_error(error.what());
}

template <typename Fn>
auto catch_ptr(Fn &&fn) -> decltype(fn()) {
    clear_error();
    try {
        return fn();
    } catch (const std::exception &error) {
        set_error(error);
    } catch (...) {
        set_error("unknown OCIO exception");
    }
    return nullptr;
}

template <typename Fn>
int32_t catch_status(Fn &&fn) {
    clear_error();
    try {
        fn();
        return 0;
    } catch (const std::exception &error) {
        set_error(error);
    } catch (...) {
        set_error("unknown OCIO exception");
    }
    return 1;
}

template <typename T>
const char *copy_string(T *handle, const char *value) {
    if (!handle || !value) {
        return nullptr;
    }
    handle->scratch = value;
    return handle->scratch.c_str();
}

bool valid_config(OcioConfig *config) {
    if (!config || !config->ptr) {
        set_error("null OCIO config");
        return false;
    }
    return true;
}

bool valid_processor(OcioProcessor *processor) {
    if (!processor || !processor->ptr) {
        set_error("null OCIO processor");
        return false;
    }
    return true;
}

bool valid_cpu_processor(OcioCpuProcessor *processor) {
    if (!processor || !processor->ptr) {
        set_error("null OCIO CPU processor");
        return false;
    }
    return true;
}

void lcms_error_handler(cmsContext, cmsUInt32Number, const char *text) {
    set_error(text);
}

void add_three_gamma_curves(cmsPipeline *pipeline, cmsFloat64Number gamma) {
    cmsToneCurve *curve = cmsBuildGamma(nullptr, gamma);
    if (!curve) {
        throw std::runtime_error("failed to create ICC tone curve");
    }
    cmsToneCurve *curves[3] = {curve, curve, curve};
    cmsStage *stage = cmsStageAllocToneCurves(nullptr, 3, curves);
    cmsFreeToneCurve(curve);
    if (!stage || !cmsPipelineInsertStage(pipeline, cmsAT_END, stage)) {
        if (stage) {
            cmsStageFree(stage);
        }
        throw std::runtime_error("failed to add ICC tone curve stage");
    }
}

void add_identity_matrix(cmsPipeline *pipeline) {
    const cmsFloat64Number identity[] = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.0
    };
    cmsStage *stage = cmsStageAllocMatrix(nullptr, 3, 3, identity, nullptr);
    if (!stage || !cmsPipelineInsertStage(pipeline, cmsAT_END, stage)) {
        if (stage) {
            cmsStageFree(stage);
        }
        throw std::runtime_error("failed to add ICC matrix stage");
    }
}

struct IccSamplerData {
    OCIO::ConstCPUProcessorRcPtr processor;
    cmsHTRANSFORM to_pcs;
    cmsHTRANSFORM from_pcs;
};

cmsInt32Number device_to_pcs_sampler(const cmsUInt16Number in[], cmsUInt16Number out[], void *userdata) {
    auto *data = static_cast<IccSamplerData *>(userdata);
    float rgb[3] = {
        static_cast<float>(in[0]) / 65535.0f,
        static_cast<float>(in[1]) / 65535.0f,
        static_cast<float>(in[2]) / 65535.0f
    };
    data->processor->applyRGB(rgb);
    out[0] = static_cast<cmsUInt16Number>(std::clamp(rgb[0], 0.0f, 1.0f) * 65535.0f);
    out[1] = static_cast<cmsUInt16Number>(std::clamp(rgb[1], 0.0f, 1.0f) * 65535.0f);
    out[2] = static_cast<cmsUInt16Number>(std::clamp(rgb[2], 0.0f, 1.0f) * 65535.0f);
    cmsDoTransform(data->to_pcs, out, out, 1);
    return 1;
}

cmsInt32Number pcs_to_device_sampler(const cmsUInt16Number in[], cmsUInt16Number out[], void *userdata) {
    auto *data = static_cast<IccSamplerData *>(userdata);
    cmsDoTransform(data->from_pcs, in, out, 1);
    return 1;
}

std::vector<uint8_t> bake_icc(OCIO::ConstCPUProcessorRcPtr processor, int32_t cube_size, const char *description) {
    cmsSetLogErrorHandler(lcms_error_handler);

    cmsCIExyY white_point;
    if (!cmsWhitePointFromTemp(&white_point, 6505)) {
        throw std::runtime_error("failed to create ICC white point");
    }

    cmsHPROFILE lab_profile = cmsCreateLab4ProfileTHR(nullptr, &white_point);
    cmsHPROFILE display_profile = cmsCreate_sRGBProfileTHR(nullptr);
    cmsHPROFILE profile = cmsCreateRGBProfileTHR(nullptr, &white_point, nullptr, nullptr);
    if (!lab_profile || !display_profile || !profile) {
        if (lab_profile) cmsCloseProfile(lab_profile);
        if (display_profile) cmsCloseProfile(display_profile);
        if (profile) cmsCloseProfile(profile);
        throw std::runtime_error("failed to create ICC profile");
    }

    cmsSetProfileVersion(profile, 4.2);
    cmsSetDeviceClass(profile, cmsSigDisplayClass);
    cmsSetColorSpace(profile, cmsSigRgbData);
    cmsSetPCS(profile, cmsSigLabData);
    cmsSetHeaderRenderingIntent(profile, INTENT_PERCEPTUAL);

    cmsMLU *description_mlu = cmsMLUalloc(nullptr, 1);
    if (!description_mlu || !cmsMLUsetASCII(description_mlu, "en", "US", description ? description : "OCIO output")) {
        if (description_mlu) cmsMLUfree(description_mlu);
        cmsCloseProfile(lab_profile);
        cmsCloseProfile(display_profile);
        cmsCloseProfile(profile);
        throw std::runtime_error("failed to create ICC profile description");
    }
    cmsWriteTag(profile, cmsSigProfileDescriptionTag, description_mlu);
    cmsMLUfree(description_mlu);

    IccSamplerData data;
    data.processor = processor;
    data.to_pcs = cmsCreateTransform(display_profile, TYPE_RGB_16, lab_profile, TYPE_LabV2_16,
                                     INTENT_PERCEPTUAL, cmsFLAGS_NOOPTIMIZE | cmsFLAGS_NOCACHE);
    data.from_pcs = cmsCreateTransform(lab_profile, TYPE_LabV2_16, display_profile, TYPE_RGB_16,
                                       INTENT_PERCEPTUAL, cmsFLAGS_NOOPTIMIZE | cmsFLAGS_NOCACHE);
    if (!data.to_pcs || !data.from_pcs) {
        if (data.to_pcs) cmsDeleteTransform(data.to_pcs);
        if (data.from_pcs) cmsDeleteTransform(data.from_pcs);
        cmsCloseProfile(lab_profile);
        cmsCloseProfile(display_profile);
        cmsCloseProfile(profile);
        throw std::runtime_error("failed to create ICC PCS transforms");
    }

    const int32_t grid = cube_size > 1 ? cube_size : 32;
    cmsPipeline *a_to_b = cmsPipelineAlloc(nullptr, 3, 3);
    cmsPipeline *b_to_a = cmsPipelineAlloc(nullptr, 3, 3);
    if (!a_to_b || !b_to_a) {
        if (a_to_b) cmsPipelineFree(a_to_b);
        if (b_to_a) cmsPipelineFree(b_to_a);
        cmsDeleteTransform(data.to_pcs);
        cmsDeleteTransform(data.from_pcs);
        cmsCloseProfile(lab_profile);
        cmsCloseProfile(display_profile);
        cmsCloseProfile(profile);
        throw std::runtime_error("failed to create ICC LUT pipeline");
    }

    add_three_gamma_curves(a_to_b, 1.0);
    cmsStage *a_to_b_clut = cmsStageAllocCLut16bit(nullptr, grid, 3, 3, nullptr);
    if (!a_to_b_clut || !cmsStageSampleCLut16bit(a_to_b_clut, device_to_pcs_sampler, &data, 0)
        || !cmsPipelineInsertStage(a_to_b, cmsAT_END, a_to_b_clut)) {
        if (a_to_b_clut) cmsStageFree(a_to_b_clut);
        cmsPipelineFree(a_to_b);
        cmsPipelineFree(b_to_a);
        cmsDeleteTransform(data.to_pcs);
        cmsDeleteTransform(data.from_pcs);
        cmsCloseProfile(lab_profile);
        cmsCloseProfile(display_profile);
        cmsCloseProfile(profile);
        throw std::runtime_error("failed to sample ICC AToB LUT");
    }
    add_three_gamma_curves(a_to_b, 1.0);
    add_identity_matrix(a_to_b);
    add_three_gamma_curves(a_to_b, 1.0);

    add_three_gamma_curves(b_to_a, 1.0);
    add_identity_matrix(b_to_a);
    add_three_gamma_curves(b_to_a, 1.0);
    cmsStage *b_to_a_clut = cmsStageAllocCLut16bit(nullptr, grid, 3, 3, nullptr);
    if (!b_to_a_clut || !cmsStageSampleCLut16bit(b_to_a_clut, pcs_to_device_sampler, &data, 0)
        || !cmsPipelineInsertStage(b_to_a, cmsAT_END, b_to_a_clut)) {
        if (b_to_a_clut) cmsStageFree(b_to_a_clut);
        cmsPipelineFree(a_to_b);
        cmsPipelineFree(b_to_a);
        cmsDeleteTransform(data.to_pcs);
        cmsDeleteTransform(data.from_pcs);
        cmsCloseProfile(lab_profile);
        cmsCloseProfile(display_profile);
        cmsCloseProfile(profile);
        throw std::runtime_error("failed to sample ICC BToA LUT");
    }
    add_three_gamma_curves(b_to_a, 1.0);

    if (!cmsWriteTag(profile, cmsSigAToB0Tag, a_to_b) || !cmsWriteTag(profile, cmsSigBToA0Tag, b_to_a)) {
        cmsPipelineFree(a_to_b);
        cmsPipelineFree(b_to_a);
        cmsDeleteTransform(data.to_pcs);
        cmsDeleteTransform(data.from_pcs);
        cmsCloseProfile(lab_profile);
        cmsCloseProfile(display_profile);
        cmsCloseProfile(profile);
        throw std::runtime_error("failed to write ICC LUT tags");
    }
    cmsPipelineFree(a_to_b);
    cmsPipelineFree(b_to_a);

    cmsUInt32Number size = 0;
    if (!cmsSaveProfileToMem(profile, nullptr, &size) || size == 0) {
        cmsDeleteTransform(data.to_pcs);
        cmsDeleteTransform(data.from_pcs);
        cmsCloseProfile(lab_profile);
        cmsCloseProfile(display_profile);
        cmsCloseProfile(profile);
        throw std::runtime_error("failed to size ICC profile");
    }
    std::vector<uint8_t> bytes(size);
    if (!cmsSaveProfileToMem(profile, bytes.data(), &size)) {
        cmsDeleteTransform(data.to_pcs);
        cmsDeleteTransform(data.from_pcs);
        cmsCloseProfile(lab_profile);
        cmsCloseProfile(display_profile);
        cmsCloseProfile(profile);
        throw std::runtime_error("failed to write ICC profile");
    }
    bytes.resize(size);

    cmsDeleteTransform(data.to_pcs);
    cmsDeleteTransform(data.from_pcs);
    cmsCloseProfile(lab_profile);
    cmsCloseProfile(display_profile);
    cmsCloseProfile(profile);
    return bytes;
}
}

extern "C" {

const char *ocio_last_error(void) {
    return last_error.empty() ? nullptr : last_error.c_str();
}

void ocio_clear_error(void) {
    clear_error();
}

const char *ocio_get_version(void) {
    return OCIO::GetVersion();
}

int32_t ocio_get_version_hex(void) {
    return OCIO::GetVersionHex();
}

OcioConfig *ocio_config_create_from_file(const char *path) {
    return catch_ptr([&]() -> OcioConfig * {
        if (!path) {
            set_error("null config path");
            return nullptr;
        }
        return new OcioConfig{OCIO::Config::CreateFromFile(path), {}, {}};
    });
}

OcioConfig *ocio_config_create_from_builtin(const char *name) {
    return catch_ptr([&]() -> OcioConfig * {
        if (!name) {
            set_error("null builtin config name");
            return nullptr;
        }
        return new OcioConfig{OCIO::Config::CreateFromBuiltinConfig(name), {}, {}};
    });
}

void ocio_config_release(OcioConfig *config) {
    delete config;
}

int32_t ocio_config_validate(OcioConfig *config) {
    return catch_status([&]() {
        if (!valid_config(config)) {
            throw std::runtime_error(last_error);
        }
        config->ptr->validate();
    });
}

const char *ocio_config_cache_id(OcioConfig *config) {
    return catch_ptr([&]() -> const char * {
        if (!valid_config(config)) {
            return nullptr;
        }
        return copy_string(config, config->ptr->getCacheID());
    });
}

const char *ocio_config_get_role_color_space(OcioConfig *config, const char *role) {
    return catch_ptr([&]() -> const char * {
        if (!valid_config(config) || !role) {
            return nullptr;
        }
        return copy_string(config, config->ptr->getRoleColorSpace(role));
    });
}

int32_t ocio_config_get_num_color_spaces(OcioConfig *config) {
    clear_error();
    try {
        if (!valid_config(config)) {
            return -1;
        }
        return config->ptr->getNumColorSpaces();
    } catch (const std::exception &error) {
        set_error(error);
        return -1;
    }
}

const char *ocio_config_get_color_space_name(OcioConfig *config, int32_t index) {
    return catch_ptr([&]() -> const char * {
        if (!valid_config(config)) {
            return nullptr;
        }
        return copy_string(config, config->ptr->getColorSpaceNameByIndex(index));
    });
}

const char *ocio_config_get_color_space_from_filepath(OcioConfig *config, const char *path) {
    return catch_ptr([&]() -> const char * {
        if (!valid_config(config) || !path) {
            return nullptr;
        }
        return copy_string(config, config->ptr->getColorSpaceFromFilepath(path));
    });
}

const char *ocio_config_get_default_display(OcioConfig *config) {
    return catch_ptr([&]() -> const char * {
        if (!valid_config(config)) {
            return nullptr;
        }
        return copy_string(config, config->ptr->getDefaultDisplay());
    });
}

const char *ocio_config_get_default_view(OcioConfig *config, const char *display, const char *src_colorspace) {
    return catch_ptr([&]() -> const char * {
        if (!valid_config(config) || !display) {
            return nullptr;
        }
        const char *view = src_colorspace && src_colorspace[0]
            ? config->ptr->getDefaultView(display, src_colorspace)
            : config->ptr->getDefaultView(display);
        return copy_string(config, view);
    });
}

const char *ocio_config_get_display_view_color_space(OcioConfig *config, const char *display, const char *view) {
    return catch_ptr([&]() -> const char * {
        if (!valid_config(config) || !display || !view) {
            return nullptr;
        }
        std::string color_space = config->ptr->getDisplayViewColorSpaceName(display, view);
        if (color_space == "<USE_DISPLAY_NAME>") {
            color_space = display;
        }
        return copy_string(config, color_space.c_str());
    });
}

const char *ocio_config_get_color_space_interchange_attribute(OcioConfig *config, const char *color_space, const char *attribute) {
    return catch_ptr([&]() -> const char * {
        if (!valid_config(config) || !color_space || !attribute) {
            return nullptr;
        }
        auto cs = config->ptr->getColorSpace(color_space);
        if (!cs) {
            set_error("OCIO color space not found");
            return nullptr;
        }
        return copy_string(config, cs->getInterchangeAttribute(attribute));
    });
}

const char *ocio_config_resolve_file_location(OcioConfig *config, const char *path) {
    return catch_ptr([&]() -> const char * {
        if (!valid_config(config) || !path) {
            return nullptr;
        }
        auto context = config->ptr->getCurrentContext();
        return copy_string(config, context->resolveFileLocation(path));
    });
}

const uint8_t *ocio_config_bake_color_space_icc(OcioConfig *config, const char *src, const char *dst, const char *description, int32_t cube_size, size_t *size) {
    return catch_ptr([&]() -> const uint8_t * {
        if (size) {
            *size = 0;
        }
        if (!valid_config(config) || !src || !dst || !size) {
            return nullptr;
        }
        auto processor = config->ptr->getProcessor(src, dst)->getDefaultCPUProcessor();
        config->byte_scratch = bake_icc(processor, cube_size, description);
        *size = config->byte_scratch.size();
        return config->byte_scratch.data();
    });
}

OcioProcessor *ocio_config_get_processor(OcioConfig *config, const char *src, const char *dst) {
    return catch_ptr([&]() -> OcioProcessor * {
        if (!valid_config(config) || !src || !dst) {
            return nullptr;
        }
        return new OcioProcessor{config->ptr->getProcessor(src, dst), {}};
    });
}

OcioProcessor *ocio_config_get_display_view_processor(OcioConfig *config, const char *src, const char *display, const char *view) {
    return catch_ptr([&]() -> OcioProcessor * {
        if (!valid_config(config) || !src || !display || !view) {
            return nullptr;
        }
        return new OcioProcessor{
            config->ptr->getProcessor(src, display, view, OCIO::TRANSFORM_DIR_FORWARD),
            {}
        };
    });
}

void ocio_processor_release(OcioProcessor *processor) {
    delete processor;
}

int32_t ocio_processor_is_noop(OcioProcessor *processor) {
    clear_error();
    if (!valid_processor(processor)) {
        return 0;
    }
    return processor->ptr->isNoOp() ? 1 : 0;
}

int32_t ocio_processor_has_channel_crosstalk(OcioProcessor *processor) {
    clear_error();
    if (!valid_processor(processor)) {
        return 0;
    }
    return processor->ptr->hasChannelCrosstalk() ? 1 : 0;
}

const char *ocio_processor_cache_id(OcioProcessor *processor) {
    return catch_ptr([&]() -> const char * {
        if (!valid_processor(processor)) {
            return nullptr;
        }
        return copy_string(processor, processor->ptr->getCacheID());
    });
}

OcioCpuProcessor *ocio_processor_get_default_cpu(OcioProcessor *processor) {
    return catch_ptr([&]() -> OcioCpuProcessor * {
        if (!valid_processor(processor)) {
            return nullptr;
        }
        return new OcioCpuProcessor{processor->ptr->getDefaultCPUProcessor(), {}};
    });
}

void ocio_cpu_processor_release(OcioCpuProcessor *processor) {
    delete processor;
}

int32_t ocio_cpu_processor_apply_rgb(OcioCpuProcessor *processor, float *rgb) {
    return catch_status([&]() {
        if (!valid_cpu_processor(processor) || !rgb) {
            throw std::runtime_error(last_error.empty() ? "null RGB buffer" : last_error);
        }
        processor->ptr->applyRGB(rgb);
    });
}

int32_t ocio_cpu_processor_apply_rgba(OcioCpuProcessor *processor, float *rgba) {
    return catch_status([&]() {
        if (!valid_cpu_processor(processor) || !rgba) {
            throw std::runtime_error(last_error.empty() ? "null RGBA buffer" : last_error);
        }
        processor->ptr->applyRGBA(rgba);
    });
}

int32_t ocio_cpu_processor_apply_rgb_packed(OcioCpuProcessor *processor, float *data, long width, long height) {
    return catch_status([&]() {
        if (!valid_cpu_processor(processor) || !data) {
            throw std::runtime_error(last_error.empty() ? "null RGB image buffer" : last_error);
        }
        OCIO::PackedImageDesc desc(data, width, height, 3);
        processor->ptr->apply(desc);
    });
}

int32_t ocio_cpu_processor_apply_rgba_packed(OcioCpuProcessor *processor, float *data, long width, long height) {
    return catch_status([&]() {
        if (!valid_cpu_processor(processor) || !data) {
            throw std::runtime_error(last_error.empty() ? "null RGBA image buffer" : last_error);
        }
        OCIO::PackedImageDesc desc(data, width, height, 4);
        processor->ptr->apply(desc);
    });
}

int32_t ocio_cpu_processor_is_noop(OcioCpuProcessor *processor) {
    clear_error();
    if (!valid_cpu_processor(processor)) {
        return 0;
    }
    return processor->ptr->isNoOp() ? 1 : 0;
}

int32_t ocio_cpu_processor_is_identity(OcioCpuProcessor *processor) {
    clear_error();
    if (!valid_cpu_processor(processor)) {
        return 0;
    }
    return processor->ptr->isIdentity() ? 1 : 0;
}

int32_t ocio_cpu_processor_has_channel_crosstalk(OcioCpuProcessor *processor) {
    clear_error();
    if (!valid_cpu_processor(processor)) {
        return 0;
    }
    return processor->ptr->hasChannelCrosstalk() ? 1 : 0;
}

const char *ocio_cpu_processor_cache_id(OcioCpuProcessor *processor) {
    return catch_ptr([&]() -> const char * {
        if (!valid_cpu_processor(processor)) {
            return nullptr;
        }
        return copy_string(processor, processor->ptr->getCacheID());
    });
}

}
