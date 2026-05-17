#include "shim.h"

#include <OpenColorIO/OpenColorIO.h>

#include <exception>
#include <memory>
#include <string>

namespace OCIO = OCIO_NAMESPACE;

struct OcioConfig {
    OCIO::ConstConfigRcPtr ptr;
    std::string scratch;
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
        return new OcioConfig{OCIO::Config::CreateFromFile(path), {}};
    });
}

OcioConfig *ocio_config_create_from_builtin(const char *name) {
    return catch_ptr([&]() -> OcioConfig * {
        if (!name) {
            set_error("null builtin config name");
            return nullptr;
        }
        return new OcioConfig{OCIO::Config::CreateFromBuiltinConfig(name), {}};
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
