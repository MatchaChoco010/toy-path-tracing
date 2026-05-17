use std::{
    error::Error as StdError,
    ffi::{CStr, CString, NulError},
    fmt,
    path::Path,
    ptr::NonNull,
};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    message: String,
}

impl Error {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    fn last_or(message: &str) -> Self {
        let last = unsafe { ocio_sys::ocio_last_error() };
        if last.is_null() {
            Self::new(message)
        } else {
            Self::new(
                unsafe { CStr::from_ptr(last) }
                    .to_string_lossy()
                    .into_owned(),
            )
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl StdError for Error {}

impl From<NulError> for Error {
    fn from(error: NulError) -> Self {
        Self::new(format!("string contains interior NUL: {error}"))
    }
}

pub fn version() -> String {
    unsafe { cstr_to_string(ocio_sys::ocio_get_version()) }.unwrap_or_default()
}

pub fn version_hex() -> i32 {
    unsafe { ocio_sys::ocio_get_version_hex() }
}

#[derive(Debug)]
pub struct OcioConfig {
    raw: NonNull<ocio_sys::OcioConfig>,
}

impl OcioConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path_to_cstring(path.as_ref())?;
        let raw = unsafe { ocio_sys::ocio_config_create_from_file(path.as_ptr()) };
        let raw = NonNull::new(raw).ok_or_else(|| Error::last_or("failed to load OCIO config"))?;
        Ok(Self { raw })
    }

    pub fn from_builtin(name: &str) -> Result<Self> {
        let name = CString::new(name)?;
        let raw = unsafe { ocio_sys::ocio_config_create_from_builtin(name.as_ptr()) };
        let raw = NonNull::new(raw)
            .ok_or_else(|| Error::last_or("failed to load built-in OCIO config"))?;
        Ok(Self { raw })
    }

    pub fn validate(&self) -> Result<()> {
        status(
            unsafe { ocio_sys::ocio_config_validate(self.raw.as_ptr()) },
            "OCIO config validation failed",
        )
    }

    pub fn cache_id(&self) -> Result<String> {
        unsafe {
            required_string(
                ocio_sys::ocio_config_cache_id(self.raw.as_ptr()),
                "missing config cache id",
            )
        }
    }

    pub fn role_color_space(&self, role: &str) -> Result<Option<String>> {
        let role = CString::new(role)?;
        unsafe {
            optional_string(ocio_sys::ocio_config_get_role_color_space(
                self.raw.as_ptr(),
                role.as_ptr(),
            ))
        }
    }

    pub fn color_spaces(&self) -> Result<Vec<String>> {
        let count = unsafe { ocio_sys::ocio_config_get_num_color_spaces(self.raw.as_ptr()) };
        if count < 0 {
            return Err(Error::last_or("failed to query OCIO color spaces"));
        }

        let mut names = Vec::with_capacity(count as usize);
        for index in 0..count {
            let name = unsafe {
                required_string(
                    ocio_sys::ocio_config_get_color_space_name(self.raw.as_ptr(), index),
                    "missing color space name",
                )?
            };
            names.push(name);
        }
        Ok(names)
    }

    pub fn color_space_from_filepath(&self, path: impl AsRef<Path>) -> Result<Option<String>> {
        let path = path_to_cstring(path.as_ref())?;
        unsafe {
            optional_string(ocio_sys::ocio_config_get_color_space_from_filepath(
                self.raw.as_ptr(),
                path.as_ptr(),
            ))
        }
    }

    pub fn default_display(&self) -> Result<String> {
        unsafe {
            required_string(
                ocio_sys::ocio_config_get_default_display(self.raw.as_ptr()),
                "OCIO config has no default display",
            )
        }
    }

    pub fn default_view(&self, display: &str, src_color_space: Option<&str>) -> Result<String> {
        let display = CString::new(display)?;
        let src_color_space = src_color_space.map(CString::new).transpose()?;
        let src_ptr = src_color_space
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(std::ptr::null());
        unsafe {
            required_string(
                ocio_sys::ocio_config_get_default_view(
                    self.raw.as_ptr(),
                    display.as_ptr(),
                    src_ptr,
                ),
                "OCIO config has no default view",
            )
        }
    }

    pub fn processor(&self, src: &str, dst: &str) -> Result<OcioProcessor> {
        let src = CString::new(src)?;
        let dst = CString::new(dst)?;
        let raw = unsafe {
            ocio_sys::ocio_config_get_processor(self.raw.as_ptr(), src.as_ptr(), dst.as_ptr())
        };
        let raw =
            NonNull::new(raw).ok_or_else(|| Error::last_or("failed to create OCIO processor"))?;
        Ok(OcioProcessor { raw })
    }

    pub fn display_view_processor(
        &self,
        src: &str,
        display: &str,
        view: &str,
    ) -> Result<OcioProcessor> {
        let src = CString::new(src)?;
        let display = CString::new(display)?;
        let view = CString::new(view)?;
        let raw = unsafe {
            ocio_sys::ocio_config_get_display_view_processor(
                self.raw.as_ptr(),
                src.as_ptr(),
                display.as_ptr(),
                view.as_ptr(),
            )
        };
        let raw = NonNull::new(raw)
            .ok_or_else(|| Error::last_or("failed to create OCIO display/view processor"))?;
        Ok(OcioProcessor { raw })
    }
}

impl Drop for OcioConfig {
    fn drop(&mut self) {
        unsafe { ocio_sys::ocio_config_release(self.raw.as_ptr()) }
    }
}

#[derive(Debug)]
pub struct OcioProcessor {
    raw: NonNull<ocio_sys::OcioProcessor>,
}

impl OcioProcessor {
    pub fn default_cpu_processor(&self) -> Result<OcioCpuProcessor> {
        let raw = unsafe { ocio_sys::ocio_processor_get_default_cpu(self.raw.as_ptr()) };
        let raw = NonNull::new(raw)
            .ok_or_else(|| Error::last_or("failed to create OCIO CPU processor"))?;
        Ok(OcioCpuProcessor { raw })
    }

    pub fn is_no_op(&self) -> bool {
        unsafe { ocio_sys::ocio_processor_is_noop(self.raw.as_ptr()) != 0 }
    }

    pub fn has_channel_crosstalk(&self) -> bool {
        unsafe { ocio_sys::ocio_processor_has_channel_crosstalk(self.raw.as_ptr()) != 0 }
    }

    pub fn cache_id(&self) -> Result<String> {
        unsafe {
            required_string(
                ocio_sys::ocio_processor_cache_id(self.raw.as_ptr()),
                "missing processor cache id",
            )
        }
    }
}

impl Drop for OcioProcessor {
    fn drop(&mut self) {
        unsafe { ocio_sys::ocio_processor_release(self.raw.as_ptr()) }
    }
}

#[derive(Debug)]
pub struct OcioCpuProcessor {
    raw: NonNull<ocio_sys::OcioCpuProcessor>,
}

impl OcioCpuProcessor {
    pub fn apply_rgb(&self, rgb: &mut [f32; 3]) -> Result<()> {
        status(
            unsafe { ocio_sys::ocio_cpu_processor_apply_rgb(self.raw.as_ptr(), rgb.as_mut_ptr()) },
            "failed to apply OCIO RGB transform",
        )
    }

    pub fn apply_rgba(&self, rgba: &mut [f32; 4]) -> Result<()> {
        status(
            unsafe {
                ocio_sys::ocio_cpu_processor_apply_rgba(self.raw.as_ptr(), rgba.as_mut_ptr())
            },
            "failed to apply OCIO RGBA transform",
        )
    }

    pub fn apply_rgb_packed(&self, pixels: &mut [f32], width: usize, height: usize) -> Result<()> {
        checked_packed_len(pixels.len(), width, height, 3)?;
        status(
            unsafe {
                ocio_sys::ocio_cpu_processor_apply_rgb_packed(
                    self.raw.as_ptr(),
                    pixels.as_mut_ptr(),
                    width as _,
                    height as _,
                )
            },
            "failed to apply OCIO packed RGB transform",
        )
    }

    pub fn apply_rgba_packed(&self, pixels: &mut [f32], width: usize, height: usize) -> Result<()> {
        checked_packed_len(pixels.len(), width, height, 4)?;
        status(
            unsafe {
                ocio_sys::ocio_cpu_processor_apply_rgba_packed(
                    self.raw.as_ptr(),
                    pixels.as_mut_ptr(),
                    width as _,
                    height as _,
                )
            },
            "failed to apply OCIO packed RGBA transform",
        )
    }

    pub fn is_no_op(&self) -> bool {
        unsafe { ocio_sys::ocio_cpu_processor_is_noop(self.raw.as_ptr()) != 0 }
    }

    pub fn is_identity(&self) -> bool {
        unsafe { ocio_sys::ocio_cpu_processor_is_identity(self.raw.as_ptr()) != 0 }
    }

    pub fn has_channel_crosstalk(&self) -> bool {
        unsafe { ocio_sys::ocio_cpu_processor_has_channel_crosstalk(self.raw.as_ptr()) != 0 }
    }

    pub fn cache_id(&self) -> Result<String> {
        unsafe {
            required_string(
                ocio_sys::ocio_cpu_processor_cache_id(self.raw.as_ptr()),
                "missing CPU processor cache id",
            )
        }
    }
}

impl Drop for OcioCpuProcessor {
    fn drop(&mut self) {
        unsafe { ocio_sys::ocio_cpu_processor_release(self.raw.as_ptr()) }
    }
}

unsafe fn cstr_to_string(ptr: *const std::ffi::c_char) -> Option<String> {
    if ptr.is_null() {
        None
    } else {
        Some(
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned(),
        )
    }
}

unsafe fn optional_string(ptr: *const std::ffi::c_char) -> Result<Option<String>> {
    if ptr.is_null() {
        let last = unsafe { ocio_sys::ocio_last_error() };
        if last.is_null() {
            Ok(None)
        } else {
            Err(Error::last_or("OCIO string query failed"))
        }
    } else {
        Ok(unsafe { cstr_to_string(ptr) }.filter(|value| !value.is_empty()))
    }
}

unsafe fn required_string(ptr: *const std::ffi::c_char, message: &str) -> Result<String> {
    unsafe { cstr_to_string(ptr) }.ok_or_else(|| Error::last_or(message))
}

fn status(status: i32, message: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::last_or(message))
    }
}

fn checked_packed_len(len: usize, width: usize, height: usize, channels: usize) -> Result<()> {
    let expected = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(channels))
        .ok_or_else(|| Error::new("image dimensions overflow"))?;
    if len == expected {
        Ok(())
    } else {
        Err(Error::new(format!(
            "packed image length mismatch: expected {expected} floats, got {len}"
        )))
    }
}

fn path_to_cstring(path: &Path) -> Result<CString> {
    CString::new(path.to_string_lossy().as_bytes()).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEFAULT_TEST_CONFIG: &str = "ocio://cg-config-v4.0.0_aces-v2.0_ocio-v2.5";

    #[test]
    fn reports_ocio_version() {
        assert!(version().starts_with("2.5."));
        assert!(version_hex() >= 0x02050000);
    }

    #[test]
    fn loads_builtin_config_and_applies_processor() {
        let config = OcioConfig::from_builtin(DEFAULT_TEST_CONFIG).expect("config");
        config.validate().expect("valid config");
        let processor = config
            .processor("sRGB - Texture", "ACEScg")
            .expect("processor");
        let cpu = processor.default_cpu_processor().expect("cpu");
        let mut rgb = [0.5, 0.25, 0.125];
        cpu.apply_rgb(&mut rgb).expect("apply");
        assert!(rgb.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn validates_packed_lengths() {
        let config = OcioConfig::from_builtin(DEFAULT_TEST_CONFIG).expect("config");
        let cpu = config
            .processor("ACEScg", "ACEScg")
            .unwrap()
            .default_cpu_processor()
            .unwrap();
        let mut pixels = vec![0.0; 5];
        let error = cpu.apply_rgb_packed(&mut pixels, 1, 1).unwrap_err();
        assert!(error.to_string().contains("length mismatch"));
    }
}
