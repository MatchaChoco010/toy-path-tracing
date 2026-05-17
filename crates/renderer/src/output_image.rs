use std::{ffi::CStr, fs::File, io::BufWriter, path::Path, slice};

use glam::UVec2;
use image::{
    ExtendedColorType, ImageEncoder,
    codecs::{openexr::OpenExrEncoder, png::PngEncoder, webp::WebPEncoder},
};
use jc_libavif_sys as avif;

use crate::color::OcioColorPipeline;

#[derive(Debug, Clone)]
pub enum OutputTransform {
    Rendering,
    DisplayView { display: String, view: String },
}

pub fn save_output(
    path: &Path,
    resolution: UVec2,
    rendering_pixels: &[f32],
    ocio: &OcioColorPipeline,
    transform: &OutputTransform,
) -> Result<(), Box<dyn std::error::Error>> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match extension.as_str() {
        "exr" => {
            let pixels = match transform {
                OutputTransform::Rendering => rendering_pixels.to_vec(),
                OutputTransform::DisplayView { .. } => {
                    transform_display_pixels(rendering_pixels, resolution, ocio, transform)?
                }
            };
            write_exr(path, resolution, &pixels)?;
        }
        "png" => {
            let pixels = transform_display_pixels(rendering_pixels, resolution, ocio, transform)?;
            let icc_profile = icc_profile_for_transform(ocio, transform)?;
            write_png(path, resolution, &pixels, &icc_profile)?;
        }
        "webp" => {
            let pixels = transform_display_pixels(rendering_pixels, resolution, ocio, transform)?;
            let icc_profile = icc_profile_for_transform(ocio, transform)?;
            write_webp(path, resolution, &pixels, &icc_profile)?;
        }
        "avif" => {
            let pixels = transform_display_pixels(rendering_pixels, resolution, ocio, transform)?;
            let icc_profile = icc_profile_for_transform(ocio, transform)?;
            write_avif(path, resolution, &pixels, &icc_profile)?;
        }
        other => {
            return Err(format!(
                "unsupported output extension `{other}`; expected .exr, .png, .webp, or .avif"
            )
            .into());
        }
    }
    Ok(())
}

fn transform_display_pixels(
    rendering_pixels: &[f32],
    resolution: UVec2,
    ocio: &OcioColorPipeline,
    transform: &OutputTransform,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut pixels = rendering_pixels.to_vec();
    match transform {
        OutputTransform::Rendering => {
            return Err("non-EXR output requires --output-display and --output-view".into());
        }
        OutputTransform::DisplayView { display, view } => {
            ocio.transform_output_display_view(
                &mut pixels,
                resolution.x as usize,
                resolution.y as usize,
                display,
                view,
            )?;
        }
    }
    Ok(pixels)
}

fn icc_profile_for_transform(
    ocio: &OcioColorPipeline,
    transform: &OutputTransform,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    match transform {
        OutputTransform::Rendering => {
            Err("non-EXR output requires --output-display and --output-view".into())
        }
        OutputTransform::DisplayView { display, view } => {
            Ok(ocio.icc_profile_for_display_view(display, view)?)
        }
    }
}

fn write_exr(
    path: &Path,
    resolution: UVec2,
    pixels: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let file = BufWriter::new(File::create(path)?);
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(pixels));
    for value in pixels {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    OpenExrEncoder::new(file).write_image(
        &bytes,
        resolution.x,
        resolution.y,
        ExtendedColorType::Rgb32F,
    )?;
    Ok(())
}

fn write_png(
    path: &Path,
    resolution: UVec2,
    pixels: &[f32],
    icc_profile: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let file = BufWriter::new(File::create(path)?);
    let mut encoder = PngEncoder::new(file);
    encoder.set_icc_profile(icc_profile.to_vec())?;
    let mut bytes = Vec::with_capacity(pixels.len() * std::mem::size_of::<u16>());
    for value in pixels {
        let quantized = (value.clamp(0.0, 1.0) * 65535.0).round() as u16;
        bytes.extend_from_slice(&quantized.to_ne_bytes());
    }
    encoder.write_image(&bytes, resolution.x, resolution.y, ExtendedColorType::Rgb16)?;
    Ok(())
}

fn write_webp(
    path: &Path,
    resolution: UVec2,
    pixels: &[f32],
    icc_profile: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let file = BufWriter::new(File::create(path)?);
    let mut encoder = WebPEncoder::new_lossless(file);
    encoder.set_icc_profile(icc_profile.to_vec())?;
    encoder.write_image(
        &quantize_rgb8(pixels),
        resolution.x,
        resolution.y,
        ExtendedColorType::Rgb8,
    )?;
    Ok(())
}

fn write_avif(
    path: &Path,
    resolution: UVec2,
    pixels: &[f32],
    icc_profile: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let rgb = quantize_rgb8(pixels);
    let image = AvifImage::new(resolution.x, resolution.y)?;
    unsafe {
        avif_result(
            avif::avifImageSetProfileICC(image.raw, icc_profile.as_ptr(), icc_profile.len()),
            "failed to set AVIF ICC profile",
        )?;

        let mut rgb_image = avif::avifRGBImage::default();
        avif::avifRGBImageSetDefaults(&mut rgb_image, image.raw);
        rgb_image.depth = 8;
        rgb_image.format = avif::avifRGBFormat_AVIF_RGB_FORMAT_RGB;
        rgb_image.chromaDownsampling =
            avif::avifChromaDownsampling_AVIF_CHROMA_DOWNSAMPLING_BEST_QUALITY;
        rgb_image.pixels = rgb.as_ptr().cast_mut();
        rgb_image.rowBytes = resolution.x * 3;
        avif_result(
            avif::avifImageRGBToYUV(image.raw, &rgb_image),
            "failed to convert RGB pixels to AVIF YUV planes",
        )?;
    }

    let encoder = AvifEncoder::new()?;
    unsafe {
        (*encoder.raw).quality = 90;
        (*encoder.raw).qualityAlpha = 90;
        (*encoder.raw).speed = 6;
    }

    let mut output = AvifData::default();
    unsafe {
        avif_result(
            avif::avifEncoderWrite(encoder.raw, image.raw, &mut output.raw),
            "failed to encode AVIF",
        )?;
        let bytes = slice::from_raw_parts(output.raw.data, output.raw.size);
        std::fs::write(path, bytes)?;
    }
    Ok(())
}

fn quantize_rgb8(pixels: &[f32]) -> Vec<u8> {
    pixels
        .iter()
        .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect()
}

struct AvifImage {
    raw: *mut avif::avifImage,
}

impl AvifImage {
    fn new(width: u32, height: u32) -> Result<Self, Box<dyn std::error::Error>> {
        let raw = unsafe {
            avif::avifImageCreate(
                width,
                height,
                8,
                avif::avifPixelFormat_AVIF_PIXEL_FORMAT_YUV444,
            )
        };
        if raw.is_null() {
            return Err("failed to create AVIF image".into());
        }
        unsafe {
            (*raw).yuvRange = avif::avifRange_AVIF_RANGE_FULL;
            avif_result(
                avif::avifImageAllocatePlanes(raw, avif::avifPlanesFlag_AVIF_PLANES_YUV),
                "failed to allocate AVIF YUV planes",
            )?;
        }
        Ok(Self { raw })
    }
}

impl Drop for AvifImage {
    fn drop(&mut self) {
        unsafe {
            avif::avifImageDestroy(self.raw);
        }
    }
}

struct AvifEncoder {
    raw: *mut avif::avifEncoder,
}

impl AvifEncoder {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let raw = unsafe { avif::avifEncoderCreate() };
        if raw.is_null() {
            return Err("failed to create AVIF encoder".into());
        }
        Ok(Self { raw })
    }
}

impl Drop for AvifEncoder {
    fn drop(&mut self) {
        unsafe {
            avif::avifEncoderDestroy(self.raw);
        }
    }
}

#[derive(Default)]
struct AvifData {
    raw: avif::avifRWData,
}

impl Drop for AvifData {
    fn drop(&mut self) {
        unsafe {
            avif::avifRWDataFree(&mut self.raw);
        }
    }
}

fn avif_result(result: avif::avifResult, context: &str) -> Result<(), Box<dyn std::error::Error>> {
    if result == avif::avifResult_AVIF_RESULT_OK {
        return Ok(());
    }

    let message = unsafe {
        let ptr = avif::avifResultToString(result);
        if ptr.is_null() {
            "unknown libavif error".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    Err(format!("{context}: {message}").into())
}
