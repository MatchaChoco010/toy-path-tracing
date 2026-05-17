use std::{
    collections::HashMap,
    error::Error as StdError,
    fmt, fs,
    path::Path,
    sync::{Arc, Mutex, OnceLock},
};

use glam::{Vec3, Vec4};

pub const DEFAULT_OCIO_CONFIG: &str = "ocio://cg-config-v4.0.0_aces-v2.0_ocio-v2.5";
pub const DEFAULT_RENDERING_SPACE: &str = "ACEScg";
pub const DEFAULT_TEXTURE_COLOR_SPACE: &str = "sRGB - Texture";
pub const DEFAULT_OUTPUT_DISPLAY: &str = "sRGB - Display";
pub const DEFAULT_OUTPUT_VIEW: &str = "ACES 2.0 - SDR 100 nits (Rec.709)";

const ICC_PROFILE_ATTRIBUTE: &str = "icc_profile_name";
const ICC_BAKE_TARGET_COLOR_SPACE: &str = DEFAULT_OUTPUT_DISPLAY;
const ICC_BAKE_CUBE_SIZE: i32 = 32;

static CURRENT_CONTEXT: OnceLock<Arc<OcioRenderContext>> = OnceLock::new();

#[derive(Debug)]
pub enum ColorManagementError {
    Ocio(ocio::Error),
    InvalidConfig(String),
    AlreadyInitialized,
}

impl fmt::Display for ColorManagementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ocio(error) => write!(f, "{error}"),
            Self::InvalidConfig(message) => f.write_str(message),
            Self::AlreadyInitialized => f.write_str("OCIO render context is already initialized"),
        }
    }
}

impl StdError for ColorManagementError {}

impl From<ocio::Error> for ColorManagementError {
    fn from(error: ocio::Error) -> Self {
        Self::Ocio(error)
    }
}

pub type Result<T> = std::result::Result<T, ColorManagementError>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ProcessorKey {
    ColorSpace {
        src: String,
        dst: String,
    },
    DisplayView {
        src: String,
        display: String,
        view: String,
    },
}

#[derive(Debug)]
struct CachedProcessor {
    cpu: ocio::OcioCpuProcessor,
    is_no_op: bool,
}

#[derive(Debug)]
pub struct OcioRenderContext {
    config_source: String,
    config: ocio::OcioConfig,
    rendering_space: String,
    texture_color_space: String,
    processors: Mutex<HashMap<ProcessorKey, Arc<CachedProcessor>>>,
}

impl OcioRenderContext {
    pub fn new(
        config_source: impl Into<String>,
        rendering_space: Option<String>,
        texture_color_space: impl Into<String>,
    ) -> Result<Self> {
        let config_source = config_source.into();
        let config = load_config(&config_source)?;
        config.validate()?;
        let rendering_space = resolve_rendering_space(&config, rendering_space)?;
        let texture_color_space = texture_color_space.into();
        if texture_color_space.trim().is_empty() {
            return Err(ColorManagementError::InvalidConfig(
                "--texture-color-space must not be empty".to_string(),
            ));
        }

        let context = Self {
            config_source,
            config,
            rendering_space,
            texture_color_space,
            processors: Mutex::new(HashMap::new()),
        };
        context.processor(&context.texture_color_space, &context.rendering_space)?;
        Ok(context)
    }

    pub fn config_source(&self) -> &str {
        &self.config_source
    }

    pub fn rendering_space(&self) -> &str {
        &self.rendering_space
    }

    pub fn texture_color_space(&self) -> &str {
        &self.texture_color_space
    }

    pub fn validate_color_space(&self, src: &str, dst: &str) -> Result<()> {
        self.processor(src, dst)?;
        Ok(())
    }

    pub fn validate_display_view(&self, display: &str, view: &str) -> Result<()> {
        self.display_view_processor(display, view)?;
        Ok(())
    }

    pub fn transform_rgb(&self, rgb: Vec3, src_color_space: &str) -> Result<Vec3> {
        self.transform_rgb_between(rgb, src_color_space, &self.rendering_space)
    }

    pub fn transform_rgba(&self, rgba: Vec4, src_color_space: &str) -> Result<Vec4> {
        self.transform_rgba_between(rgba, src_color_space, &self.rendering_space)
    }

    pub fn transform_rgb_between(&self, rgb: Vec3, src: &str, dst: &str) -> Result<Vec3> {
        if src == dst {
            return Ok(rgb);
        }
        let processor = self.processor(src, dst)?;
        apply_rgb(&processor, rgb)
    }

    pub fn transform_rgba_between(&self, rgba: Vec4, src: &str, dst: &str) -> Result<Vec4> {
        if src == dst {
            return Ok(rgba);
        }
        let processor = self.processor(src, dst)?;
        apply_rgba(&processor, rgba)
    }

    pub fn transform_rgb_pixels_to_rendering(
        &self,
        pixels: &mut [f32],
        width: usize,
        height: usize,
        src_color_space: &str,
    ) -> Result<()> {
        if src_color_space == self.rendering_space {
            return Ok(());
        }
        let processor = self.processor(src_color_space, &self.rendering_space)?;
        if processor.is_no_op {
            return Ok(());
        }
        Ok(processor.cpu.apply_rgb_packed(pixels, width, height)?)
    }

    pub fn transform_rgba_pixels_to_rendering(
        &self,
        pixels: &mut [f32],
        width: usize,
        height: usize,
        src_color_space: &str,
    ) -> Result<()> {
        if src_color_space == self.rendering_space {
            return Ok(());
        }
        let processor = self.processor(src_color_space, &self.rendering_space)?;
        if processor.is_no_op {
            return Ok(());
        }
        Ok(processor.cpu.apply_rgba_packed(pixels, width, height)?)
    }

    pub fn transform_output_display_view(
        &self,
        pixels: &mut [f32],
        width: usize,
        height: usize,
        display: &str,
        view: &str,
    ) -> Result<()> {
        let processor = self.display_view_processor(display, view)?;
        if processor.is_no_op {
            return Ok(());
        }
        Ok(processor.cpu.apply_rgb_packed(pixels, width, height)?)
    }

    pub fn icc_profile_for_display_view(&self, display: &str, view: &str) -> Result<Vec<u8>> {
        let color_space = self.config.display_view_color_space(display, view)?;
        if let Some(profile_name) = self
            .config
            .color_space_interchange_attribute(&color_space, ICC_PROFILE_ATTRIBUTE)?
        {
            let profile_path = self.config.resolve_file_location(&profile_name)?;
            let profile = fs::read(&profile_path).map_err(|error| {
                ColorManagementError::InvalidConfig(format!(
                    "failed to read OCIO ICC profile `{profile_path}`: {error}"
                ))
            })?;
            if profile.is_empty() {
                return Err(ColorManagementError::InvalidConfig(format!(
                    "OCIO ICC profile `{profile_path}` is empty"
                )));
            }
            return Ok(profile);
        }

        let description = format!("{display} / {view} ({color_space})");
        Ok(self.config.bake_color_space_icc(
            &color_space,
            ICC_BAKE_TARGET_COLOR_SPACE,
            &description,
            ICC_BAKE_CUBE_SIZE,
        )?)
    }

    fn display_view_processor(&self, display: &str, view: &str) -> Result<Arc<CachedProcessor>> {
        let key = ProcessorKey::DisplayView {
            src: self.rendering_space.clone(),
            display: display.to_string(),
            view: view.to_string(),
        };
        self.cached_processor(key, || {
            self.config
                .display_view_processor(&self.rendering_space, display, view)
        })
    }

    fn processor(&self, src: &str, dst: &str) -> Result<Arc<CachedProcessor>> {
        let key = ProcessorKey::ColorSpace {
            src: src.to_string(),
            dst: dst.to_string(),
        };
        self.cached_processor(key, || self.config.processor(src, dst))
    }

    fn cached_processor(
        &self,
        key: ProcessorKey,
        create: impl FnOnce() -> ocio::Result<ocio::OcioProcessor>,
    ) -> Result<Arc<CachedProcessor>> {
        if let Some(processor) = self.processors.lock().expect("processor cache").get(&key) {
            return Ok(Arc::clone(processor));
        }

        let processor = create()?;
        let cpu = processor.default_cpu_processor()?;
        let cached = Arc::new(CachedProcessor {
            is_no_op: cpu.is_no_op() || cpu.is_identity(),
            cpu,
        });
        self.processors
            .lock()
            .expect("processor cache")
            .insert(key, Arc::clone(&cached));
        Ok(cached)
    }
}

pub fn set_current(context: Arc<OcioRenderContext>) -> Result<()> {
    CURRENT_CONTEXT
        .set(context)
        .map_err(|_| ColorManagementError::AlreadyInitialized)
}

pub fn current() -> Option<&'static Arc<OcioRenderContext>> {
    CURRENT_CONTEXT.get()
}

pub fn map_materialx_color_space(name: &str) -> &str {
    match name {
        "srgb_texture" | "srgb" | "sRGB" => "sRGB - Texture",
        other => other,
    }
}

pub fn image_error(error: impl fmt::Display) -> image::ImageError {
    image::ImageError::IoError(std::io::Error::other(error.to_string()))
}

fn load_config(config_source: &str) -> Result<ocio::OcioConfig> {
    if config_source.starts_with("ocio://") {
        Ok(ocio::OcioConfig::from_builtin(config_source)?)
    } else {
        Ok(ocio::OcioConfig::from_file(Path::new(config_source))?)
    }
}

fn resolve_rendering_space(config: &ocio::OcioConfig, requested: Option<String>) -> Result<String> {
    if let Some(requested) = requested {
        if requested.trim().is_empty() {
            return Err(ColorManagementError::InvalidConfig(
                "--ocio-rendering-space must not be empty".to_string(),
            ));
        }
        return Ok(requested);
    }

    for role in ["rendering", "scene_linear"] {
        if let Some(color_space) = config.role_color_space(role)? {
            return Ok(color_space);
        }
    }

    if config
        .color_spaces()?
        .iter()
        .any(|name| name == DEFAULT_RENDERING_SPACE)
    {
        return Ok(DEFAULT_RENDERING_SPACE.to_string());
    }

    Err(ColorManagementError::InvalidConfig(
        "OCIO config has neither rendering/scene_linear role nor ACEScg color space".to_string(),
    ))
}

fn apply_rgb(processor: &CachedProcessor, rgb: Vec3) -> Result<Vec3> {
    if processor.is_no_op {
        return Ok(rgb);
    }
    let mut rgb = rgb.to_array();
    processor.cpu.apply_rgb(&mut rgb)?;
    Ok(Vec3::from_array(rgb))
}

fn apply_rgba(processor: &CachedProcessor, rgba: Vec4) -> Result<Vec4> {
    if processor.is_no_op {
        return Ok(rgba);
    }
    let mut rgba = rgba.to_array();
    processor.cpu.apply_rgba(&mut rgba)?;
    Ok(Vec4::from_array(rgba))
}
