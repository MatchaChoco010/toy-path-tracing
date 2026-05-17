#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OcioConfig OcioConfig;
typedef struct OcioProcessor OcioProcessor;
typedef struct OcioCpuProcessor OcioCpuProcessor;

const char *ocio_last_error(void);
void ocio_clear_error(void);

const char *ocio_get_version(void);
int32_t ocio_get_version_hex(void);

OcioConfig *ocio_config_create_from_file(const char *path);
OcioConfig *ocio_config_create_from_builtin(const char *name);
void ocio_config_release(OcioConfig *config);
int32_t ocio_config_validate(OcioConfig *config);
const char *ocio_config_cache_id(OcioConfig *config);
const char *ocio_config_get_role_color_space(OcioConfig *config, const char *role);
int32_t ocio_config_get_num_color_spaces(OcioConfig *config);
const char *ocio_config_get_color_space_name(OcioConfig *config, int32_t index);
const char *ocio_config_get_color_space_from_filepath(OcioConfig *config, const char *path);
const char *ocio_config_get_default_display(OcioConfig *config);
const char *ocio_config_get_default_view(OcioConfig *config, const char *display, const char *src_colorspace);

OcioProcessor *ocio_config_get_processor(OcioConfig *config, const char *src, const char *dst);
OcioProcessor *ocio_config_get_display_view_processor(OcioConfig *config, const char *src, const char *display, const char *view);
void ocio_processor_release(OcioProcessor *processor);
int32_t ocio_processor_is_noop(OcioProcessor *processor);
int32_t ocio_processor_has_channel_crosstalk(OcioProcessor *processor);
const char *ocio_processor_cache_id(OcioProcessor *processor);
OcioCpuProcessor *ocio_processor_get_default_cpu(OcioProcessor *processor);

void ocio_cpu_processor_release(OcioCpuProcessor *processor);
int32_t ocio_cpu_processor_apply_rgb(OcioCpuProcessor *processor, float *rgb);
int32_t ocio_cpu_processor_apply_rgba(OcioCpuProcessor *processor, float *rgba);
int32_t ocio_cpu_processor_apply_rgb_packed(OcioCpuProcessor *processor, float *data, long width, long height);
int32_t ocio_cpu_processor_apply_rgba_packed(OcioCpuProcessor *processor, float *data, long width, long height);
int32_t ocio_cpu_processor_is_noop(OcioCpuProcessor *processor);
int32_t ocio_cpu_processor_is_identity(OcioCpuProcessor *processor);
int32_t ocio_cpu_processor_has_channel_crosstalk(OcioCpuProcessor *processor);
const char *ocio_cpu_processor_cache_id(OcioCpuProcessor *processor);

#ifdef __cplusplus
}
#endif
