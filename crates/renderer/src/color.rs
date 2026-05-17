use std::{error::Error as StdError, fmt, fs, path::Path, sync::Arc};

use glam::{Vec3, Vec4};

pub const DEFAULT_OCIO_CONFIG: &str = "ocio://cg-config-v4.0.0_aces-v2.0_ocio-v2.5";
pub const DEFAULT_RENDERING_SPACE: &str = "ACEScg";
pub const DEFAULT_TEXTURE_COLOR_SPACE: &str = "sRGB - Texture";
pub const DEFAULT_OUTPUT_DISPLAY: &str = "sRGB - Display";
pub const DEFAULT_OUTPUT_VIEW: &str = "ACES 2.0 - SDR 100 nits (Rec.709)";

const ICC_PROFILE_ATTRIBUTE: &str = "icc_profile_name";
const ICC_BAKE_TARGET_COLOR_SPACE: &str = DEFAULT_OUTPUT_DISPLAY;
const ICC_BAKE_CUBE_SIZE: i32 = 32;

#[derive(Debug)]
pub enum ColorManagementError {
    Ocio(ocio::Error),
    InvalidConfig(String),
}

impl fmt::Display for ColorManagementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ocio(error) => write!(f, "{error}"),
            Self::InvalidConfig(message) => f.write_str(message),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpaceRef<'a> {
    Rendering,
    Texture,
    Ocio(&'a str),
}

#[derive(Debug)]
pub struct OcioColorProcessor {
    is_no_op: bool,
    cpu: Option<ocio::OcioCpuProcessor>,
}

impl OcioColorProcessor {
    pub fn apply_rgb(&self, rgb: Vec3) -> Result<Vec3> {
        if self.is_no_op {
            return Ok(rgb);
        }
        let mut rgb = rgb.to_array();
        self.cpu
            .as_ref()
            .expect("OCIO processor")
            .apply_rgb(&mut rgb)?;
        Ok(Vec3::from_array(rgb))
    }

    pub fn apply_rgba(&self, rgba: Vec4) -> Result<Vec4> {
        if self.is_no_op {
            return Ok(rgba);
        }
        let mut rgba = rgba.to_array();
        self.cpu
            .as_ref()
            .expect("OCIO processor")
            .apply_rgba(&mut rgba)?;
        Ok(Vec4::from_array(rgba))
    }

    pub fn apply_rgb_packed(&self, pixels: &mut [f32], width: usize, height: usize) -> Result<()> {
        if self.is_no_op {
            return Ok(());
        }
        Ok(self
            .cpu
            .as_ref()
            .expect("OCIO processor")
            .apply_rgb_packed(pixels, width, height)?)
    }

    pub fn apply_rgba_packed(&self, pixels: &mut [f32], width: usize, height: usize) -> Result<()> {
        if self.is_no_op {
            return Ok(());
        }
        Ok(self
            .cpu
            .as_ref()
            .expect("OCIO processor")
            .apply_rgba_packed(pixels, width, height)?)
    }
}

#[derive(Debug)]
pub struct OcioColorPipeline {
    config_source: String,
    config: ocio::OcioConfig,
    rendering_space: String,
    texture_color_space: String,
}

impl OcioColorPipeline {
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

        let pipeline = Self {
            config_source,
            config,
            rendering_space,
            texture_color_space,
        };
        pipeline.color_space_processor(ColorSpaceRef::Texture, ColorSpaceRef::Rendering)?;
        Ok(pipeline)
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

    pub fn color_space_name<'a>(&'a self, color_space: ColorSpaceRef<'a>) -> &'a str {
        match color_space {
            ColorSpaceRef::Rendering => &self.rendering_space,
            ColorSpaceRef::Texture => &self.texture_color_space,
            ColorSpaceRef::Ocio(name) => name,
        }
    }

    pub fn validate_color_space(
        &self,
        from: ColorSpaceRef<'_>,
        to: ColorSpaceRef<'_>,
    ) -> Result<()> {
        self.color_space_processor(from, to)?;
        Ok(())
    }

    pub fn validate_display_view(&self, display: &str, view: &str) -> Result<()> {
        self.display_view_processor(display, view)?;
        Ok(())
    }

    pub fn transform_rgb(
        &self,
        rgb: Vec3,
        from: ColorSpaceRef<'_>,
        to: ColorSpaceRef<'_>,
    ) -> Result<Vec3> {
        let processor = self.color_space_processor(from, to)?;
        processor.apply_rgb(rgb)
    }

    pub fn transform_rgba(
        &self,
        rgba: Vec4,
        from: ColorSpaceRef<'_>,
        to: ColorSpaceRef<'_>,
    ) -> Result<Vec4> {
        let processor = self.color_space_processor(from, to)?;
        processor.apply_rgba(rgba)
    }

    pub fn transform_rgba_pixels(
        &self,
        pixels: &mut [f32],
        width: usize,
        height: usize,
        from: ColorSpaceRef<'_>,
        to: ColorSpaceRef<'_>,
    ) -> Result<()> {
        let processor = self.color_space_processor(from, to)?;
        processor.apply_rgba_packed(pixels, width, height)
    }

    pub fn transform_rgb_pixels_to_rendering(
        &self,
        pixels: &mut [f32],
        width: usize,
        height: usize,
        src_color_space: &str,
    ) -> Result<()> {
        let processor = self.color_space_processor(
            ColorSpaceRef::Ocio(src_color_space),
            ColorSpaceRef::Rendering,
        )?;
        processor.apply_rgb_packed(pixels, width, height)
    }

    pub fn transform_rgba_pixels_to_rendering(
        &self,
        pixels: &mut [f32],
        width: usize,
        height: usize,
        src_color_space: &str,
    ) -> Result<()> {
        let processor = self.color_space_processor(
            ColorSpaceRef::Ocio(src_color_space),
            ColorSpaceRef::Rendering,
        )?;
        processor.apply_rgba_packed(pixels, width, height)
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
        processor.apply_rgb_packed(pixels, width, height)
    }

    pub fn color_space_processor(
        &self,
        from: ColorSpaceRef<'_>,
        to: ColorSpaceRef<'_>,
    ) -> Result<Arc<OcioColorProcessor>> {
        let from = self.color_space_name(from);
        let to = self.color_space_name(to);
        if from == to {
            return no_op_processor();
        }
        self.processor(self.config.processor(from, to)?)
    }

    pub fn display_view_processor(
        &self,
        display: &str,
        view: &str,
    ) -> Result<Arc<OcioColorProcessor>> {
        self.processor(
            self.config
                .display_view_processor(&self.rendering_space, display, view)?,
        )
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

    fn processor(&self, processor: ocio::OcioProcessor) -> Result<Arc<OcioColorProcessor>> {
        let cpu = processor.default_cpu_processor()?;
        Ok(Arc::new(OcioColorProcessor {
            is_no_op: cpu.is_no_op() || cpu.is_identity(),
            cpu: Some(cpu),
        }))
    }
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

fn no_op_processor() -> Result<Arc<OcioColorProcessor>> {
    Ok(Arc::new(OcioColorProcessor {
        is_no_op: true,
        cpu: None,
    }))
}
