use std::ffi::{c_char, c_float, c_int, c_long};

pub enum OcioConfig {}
pub enum OcioProcessor {}
pub enum OcioCpuProcessor {}

unsafe extern "C" {
    pub fn ocio_last_error() -> *const c_char;
    pub fn ocio_clear_error();

    pub fn ocio_get_version() -> *const c_char;
    pub fn ocio_get_version_hex() -> i32;

    pub fn ocio_config_create_from_file(path: *const c_char) -> *mut OcioConfig;
    pub fn ocio_config_create_from_builtin(name: *const c_char) -> *mut OcioConfig;
    pub fn ocio_config_release(config: *mut OcioConfig);
    pub fn ocio_config_validate(config: *mut OcioConfig) -> c_int;
    pub fn ocio_config_cache_id(config: *mut OcioConfig) -> *const c_char;
    pub fn ocio_config_get_role_color_space(
        config: *mut OcioConfig,
        role: *const c_char,
    ) -> *const c_char;
    pub fn ocio_config_get_num_color_spaces(config: *mut OcioConfig) -> c_int;
    pub fn ocio_config_get_color_space_name(config: *mut OcioConfig, index: c_int)
    -> *const c_char;
    pub fn ocio_config_get_color_space_from_filepath(
        config: *mut OcioConfig,
        path: *const c_char,
    ) -> *const c_char;
    pub fn ocio_config_get_default_display(config: *mut OcioConfig) -> *const c_char;
    pub fn ocio_config_get_default_view(
        config: *mut OcioConfig,
        display: *const c_char,
        src_colorspace: *const c_char,
    ) -> *const c_char;

    pub fn ocio_config_get_processor(
        config: *mut OcioConfig,
        src: *const c_char,
        dst: *const c_char,
    ) -> *mut OcioProcessor;
    pub fn ocio_config_get_display_view_processor(
        config: *mut OcioConfig,
        src: *const c_char,
        display: *const c_char,
        view: *const c_char,
    ) -> *mut OcioProcessor;
    pub fn ocio_processor_release(processor: *mut OcioProcessor);
    pub fn ocio_processor_is_noop(processor: *mut OcioProcessor) -> c_int;
    pub fn ocio_processor_has_channel_crosstalk(processor: *mut OcioProcessor) -> c_int;
    pub fn ocio_processor_cache_id(processor: *mut OcioProcessor) -> *const c_char;
    pub fn ocio_processor_get_default_cpu(processor: *mut OcioProcessor) -> *mut OcioCpuProcessor;

    pub fn ocio_cpu_processor_release(processor: *mut OcioCpuProcessor);
    pub fn ocio_cpu_processor_apply_rgb(
        processor: *mut OcioCpuProcessor,
        rgb: *mut c_float,
    ) -> c_int;
    pub fn ocio_cpu_processor_apply_rgba(
        processor: *mut OcioCpuProcessor,
        rgba: *mut c_float,
    ) -> c_int;
    pub fn ocio_cpu_processor_apply_rgb_packed(
        processor: *mut OcioCpuProcessor,
        data: *mut c_float,
        width: c_long,
        height: c_long,
    ) -> c_int;
    pub fn ocio_cpu_processor_apply_rgba_packed(
        processor: *mut OcioCpuProcessor,
        data: *mut c_float,
        width: c_long,
        height: c_long,
    ) -> c_int;
    pub fn ocio_cpu_processor_is_noop(processor: *mut OcioCpuProcessor) -> c_int;
    pub fn ocio_cpu_processor_is_identity(processor: *mut OcioCpuProcessor) -> c_int;
    pub fn ocio_cpu_processor_has_channel_crosstalk(processor: *mut OcioCpuProcessor) -> c_int;
    pub fn ocio_cpu_processor_cache_id(processor: *mut OcioCpuProcessor) -> *const c_char;
}
