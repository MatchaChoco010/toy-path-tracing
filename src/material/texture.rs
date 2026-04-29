use std::{
    ops::{Add, Mul},
    path::Path,
    sync::Arc,
};

use glam::{Vec2, Vec3};

use crate::color::srgb_to_linear;

/// Pixel types that a [`Texture`] can hold.
///
/// Implemented for `Vec3` (colour / normal maps) and `f32` (scalar maps such
/// as opacity, metallic, roughness). Going through this trait lets the
/// sampling and mip-pyramid code be written once and shared between channel
/// counts, instead of carrying a 3x-bloated `Vec3` for every scalar texture.
pub trait TexturePixel:
    Copy + Default + Add<Output = Self> + Mul<f32, Output = Self> + PartialEq
{
    /// Maximum scalar value of the pixel. Used by emissive materials to
    /// bound radiance over an entire texture without re-walking the image.
    fn max_value(self) -> f32;

    /// `self * (1 - t) + other * t`. Defined on the trait so `Vec3` can use
    /// the SIMD `Vec3::lerp` while the scalar specialisation stays plain.
    fn lerp(self, other: Self, t: f32) -> Self;
}

impl TexturePixel for Vec3 {
    fn max_value(self) -> f32 {
        self.max_element()
    }

    fn lerp(self, other: Self, t: f32) -> Self {
        Vec3::lerp(self, other, t)
    }
}

impl TexturePixel for f32 {
    fn max_value(self) -> f32 {
        self
    }

    fn lerp(self, other: Self, t: f32) -> Self {
        self + (other - self) * t
    }
}

/// Filtered, mip-mapped image. Generic over the pixel type; defaults to
/// `Vec3` so existing call sites that wrote `Texture` keep meaning "RGB
/// texture".
#[derive(Debug, Clone, PartialEq)]
pub struct Texture<P: TexturePixel = Vec3> {
    levels: Vec<TextureLevel<P>>,
    /// Cached maximum of `pixel.max_value()` over the level-0 image.
    /// Pre-computed at construction time so callers like the light tree
    /// builder can ask for it once per emissive material rather than
    /// re-scanning every pixel per emissive triangle.
    max_value: f32,
}

#[derive(Debug, Clone, PartialEq)]
struct TextureLevel<P> {
    width: usize,
    height: usize,
    pixels: Vec<P>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureColorSpace {
    Linear,
    Srgb,
}

/// Convenience alias for single-channel scalar textures (opacity, metallic,
/// roughness, …).
pub type ScalarTexture = Texture<f32>;

const MAX_ANISOTROPY: f32 = 8.0;
const MIN_FILTER_WIDTH: f32 = 1.0e-8;
const EWA_ALPHA: f32 = 2.0;

impl<P: TexturePixel> Texture<P> {
    pub fn from_pixels(width: usize, height: usize, pixels: Vec<P>) -> Self {
        assert!(width > 0 && height > 0, "texture must be non-empty");
        assert_eq!(
            pixels.len(),
            width * height,
            "pixel buffer length does not match width * height"
        );

        let max_value = pixels
            .iter()
            .copied()
            .map(P::max_value)
            .fold(0.0_f32, f32::max);
        let base_level = TextureLevel {
            width,
            height,
            pixels,
        };

        Self {
            levels: build_mip_levels(base_level),
            max_value,
        }
    }

    pub fn width(&self) -> usize {
        self.level(0).width
    }

    pub fn height(&self) -> usize {
        self.level(0).height
    }

    pub fn pixels(&self) -> &[P] {
        &self.level(0).pixels
    }

    /// Maximum pixel value seen at level 0 (channel-wise max for `Vec3`).
    pub fn max_value(&self) -> f32 {
        self.max_value
    }

    pub fn sample(&self, uv: Vec2) -> P {
        self.bilerp_level(0, uv)
    }

    pub fn sample_filtered(&self, uv: Vec2, dstdx: Vec2, dstdy: Vec2) -> P {
        self.sample_ewa(uv, dstdx, dstdy)
    }

    fn sample_ewa(&self, uv: Vec2, mut dst0: Vec2, mut dst1: Vec2) -> P {
        if !dst0.is_finite() || !dst1.is_finite() {
            return self.sample(uv);
        }

        if dst0.length_squared() < dst1.length_squared() {
            std::mem::swap(&mut dst0, &mut dst1);
        }

        let longer_length = dst0.length();
        let mut shorter_length = dst1.length();

        if shorter_length > 0.0 && shorter_length * MAX_ANISOTROPY < longer_length {
            let scale = longer_length / (shorter_length * MAX_ANISOTROPY);
            dst1 *= scale;
            shorter_length *= scale;
        }

        if shorter_length <= 0.0 {
            return self.sample(uv);
        }

        let lod = ((self.level_count() - 1) as f32 + shorter_length.max(MIN_FILTER_WIDTH).log2())
            .max(0.0);
        let level = lod.floor() as usize;
        let t = lod - level as f32;
        let a = self.ewa_level(level, uv, dst0, dst1);
        let b = self.ewa_level(level + 1, uv, dst0, dst1);

        a.lerp(b, t)
    }

    #[cfg(test)]
    fn sample_trilinear(&self, uv: Vec2, dstdx: Vec2, dstdy: Vec2) -> P {
        if !dstdx.is_finite() || !dstdy.is_finite() {
            return self.sample(uv);
        }

        let width = 2.0
            * dstdx
                .abs()
                .max(dstdy.abs())
                .max_element()
                .max(MIN_FILTER_WIDTH);
        let lod = (self.level_count() - 1) as f32 + width.log2();

        if lod >= (self.level_count() - 1) as f32 {
            return self.pixel_level_wrapped(self.level_count() - 1, 0, 0);
        }

        let level = lod.floor().max(0.0) as usize;
        if level == 0 {
            return self.bilerp_level(0, uv);
        }

        self.bilerp_level(level, uv)
            .lerp(self.bilerp_level(level + 1, uv), lod - level as f32)
    }

    fn bilerp_level(&self, level: usize, uv: Vec2) -> P {
        let level = level.min(self.level_count() - 1);
        let texture_level = self.level(level);
        let u = wrap_unit(uv.x);
        let v = wrap_unit(uv.y);
        let x = u * texture_level.width as f32 - 0.5;
        let y = v * texture_level.height as f32 - 0.5;
        let x0 = x.floor() as isize;
        let y0 = y.floor() as isize;
        let tx = x - x0 as f32;
        let ty = y - y0 as f32;

        let c00 = self.pixel_level_wrapped(level, x0, y0);
        let c10 = self.pixel_level_wrapped(level, x0 + 1, y0);
        let c01 = self.pixel_level_wrapped(level, x0, y0 + 1);
        let c11 = self.pixel_level_wrapped(level, x0 + 1, y0 + 1);
        let cx0 = c00.lerp(c10, tx);
        let cx1 = c01.lerp(c11, tx);

        cx0.lerp(cx1, ty)
    }

    fn ewa_level(&self, level: usize, uv: Vec2, mut dst0: Vec2, mut dst1: Vec2) -> P {
        if level >= self.level_count() {
            return self.pixel_level_wrapped(self.level_count() - 1, 0, 0);
        }

        let texture_level = self.level(level);
        let st = Vec2::new(
            uv.x * texture_level.width as f32 - 0.5,
            uv.y * texture_level.height as f32 - 0.5,
        );
        dst0 *= Vec2::new(texture_level.width as f32, texture_level.height as f32);
        dst1 *= Vec2::new(texture_level.width as f32, texture_level.height as f32);

        let mut a = dst0.y * dst0.y + dst1.y * dst1.y + 1.0;
        let mut b = -2.0 * (dst0.x * dst0.y + dst1.x * dst1.y);
        let mut c = dst0.x * dst0.x + dst1.x * dst1.x + 1.0;
        let inv_f = 1.0 / (a * c - 0.25 * b * b);
        if !inv_f.is_finite() {
            return self.bilerp_level(level, uv);
        }
        a *= inv_f;
        b *= inv_f;
        c *= inv_f;

        let det = -b * b + 4.0 * a * c;
        if det <= 0.0 || !det.is_finite() {
            return self.bilerp_level(level, uv);
        }

        let inv_det = 1.0 / det;
        let u_sqrt = (det * c).max(0.0).sqrt();
        let v_sqrt = (a * det).max(0.0).sqrt();
        let s0 = (st.x - 2.0 * inv_det * u_sqrt).ceil() as isize;
        let s1 = (st.x + 2.0 * inv_det * u_sqrt).floor() as isize;
        let t0 = (st.y - 2.0 * inv_det * v_sqrt).ceil() as isize;
        let t1 = (st.y + 2.0 * inv_det * v_sqrt).floor() as isize;

        let mut sum = P::default();
        let mut weight_sum = 0.0;
        let cutoff = (-EWA_ALPHA).exp();

        for it in t0..=t1 {
            let tt = it as f32 - st.y;
            for is in s0..=s1 {
                let ss = is as f32 - st.x;
                let r2 = a * ss * ss + b * ss * tt + c * tt * tt;
                if r2 >= 1.0 {
                    continue;
                }

                let weight = (-EWA_ALPHA * r2).exp() - cutoff;
                sum = sum + self.pixel_level_wrapped(level, is, it) * weight;
                weight_sum += weight;
            }
        }

        if weight_sum > 0.0 {
            sum * (1.0 / weight_sum)
        } else {
            self.bilerp_level(level, uv)
        }
    }

    fn pixel_level_wrapped(&self, level: usize, x: isize, y: isize) -> P {
        let texture_level = self.level(level.min(self.level_count() - 1));
        let x = wrap_index(x, texture_level.width);
        let y = wrap_index(y, texture_level.height);

        texture_level.pixels[y * texture_level.width + x]
    }

    fn level(&self, level: usize) -> &TextureLevel<P> {
        &self.levels[level]
    }

    fn level_count(&self) -> usize {
        self.levels.len()
    }
}

impl Texture<Vec3> {
    pub fn from_file(path: impl AsRef<Path>) -> image::ImageResult<Self> {
        Self::from_file_with_color_space(path, TextureColorSpace::Linear)
    }

    pub fn from_srgb_file(path: impl AsRef<Path>) -> image::ImageResult<Self> {
        Self::from_file_with_color_space(path, TextureColorSpace::Srgb)
    }

    pub fn from_file_with_color_space(
        path: impl AsRef<Path>,
        color_space: TextureColorSpace,
    ) -> image::ImageResult<Self> {
        let image = image::open(path)?.into_rgba32f();
        let width = image.width() as usize;
        let height = image.height() as usize;
        let pixels = image
            .pixels()
            .map(|pixel| Vec3::new(pixel[0], pixel[1], pixel[2]))
            .map(|rgb| decode_color_space(rgb, color_space))
            .collect();

        Ok(Self::from_pixels(width, height, pixels))
    }

    /// Loads an image and returns the decoded colour texture together with
    /// an optional alpha texture. The alpha texture is only returned when
    /// the source image actually carries non-opaque pixels, so opaque
    /// images do not pay extra memory for an all-ones alpha pyramid.
    pub fn from_file_with_alpha(
        path: impl AsRef<Path>,
        color_space: TextureColorSpace,
    ) -> image::ImageResult<(Self, Option<ScalarTexture>)> {
        let image = image::open(path)?.into_rgba32f();
        let width = image.width() as usize;
        let height = image.height() as usize;
        let mut rgb_pixels = Vec::with_capacity(width * height);
        let mut alpha_pixels: Vec<f32> = Vec::with_capacity(width * height);
        let mut has_nontrivial_alpha = false;
        for pixel in image.pixels() {
            let rgb = decode_color_space(Vec3::new(pixel[0], pixel[1], pixel[2]), color_space);
            rgb_pixels.push(rgb);
            let alpha = pixel[3];
            if alpha < 1.0 - 1.0e-3 {
                has_nontrivial_alpha = true;
            }
            alpha_pixels.push(alpha);
        }
        let rgb = Self::from_pixels(width, height, rgb_pixels);
        let alpha = if has_nontrivial_alpha {
            Some(ScalarTexture::from_pixels(width, height, alpha_pixels))
        } else {
            None
        };

        Ok((rgb, alpha))
    }
}

impl Texture<f32> {
    /// Build a scalar texture from one channel of a tightly packed RGBA
    /// byte buffer. `channel` is 0=R, 1=G, 2=B, 3=A. Bistro packs its ORM
    /// map this way (G=Roughness, B=Metalness), and this constructor lets
    /// us extract each channel into its own scalar pyramid without the 3x
    /// memory penalty of splatting into a `Vec3` texture.
    pub fn from_rgba_channel(width: usize, height: usize, rgba: &[u8], channel: usize) -> Self {
        assert!(channel < 4, "channel index must be in 0..4");
        assert_eq!(
            rgba.len(),
            width * height * 4,
            "rgba buffer length does not match width * height * 4"
        );
        let mut pixels = Vec::with_capacity(width * height);
        for chunk in rgba.chunks_exact(4) {
            pixels.push(chunk[channel] as f32 / 255.0);
        }
        Self::from_pixels(width, height, pixels)
    }

    /// Loads a scalar texture from disk. Reads the R channel of the source
    /// image (after sRGB decoding has been skipped — scalar maps are always
    /// linear).
    pub fn from_file(path: impl AsRef<Path>) -> image::ImageResult<Self> {
        let image = image::open(path)?.into_rgba32f();
        let width = image.width() as usize;
        let height = image.height() as usize;
        let pixels = image.pixels().map(|pixel| pixel[0]).collect();
        Ok(Self::from_pixels(width, height, pixels))
    }
}

fn build_mip_levels<P: TexturePixel>(base_level: TextureLevel<P>) -> Vec<TextureLevel<P>> {
    let mut levels = vec![base_level];

    while levels
        .last()
        .is_some_and(|level| level.width > 1 || level.height > 1)
    {
        let previous = levels.last().expect("mip pyramid has at least one level");
        let width = previous.width.div_ceil(2).max(1);
        let height = previous.height.div_ceil(2).max(1);
        let mut pixels = Vec::with_capacity(width * height);

        for y in 0..height {
            for x in 0..width {
                let sx = (2 * x) as isize;
                let sy = (2 * y) as isize;
                let texel = (pixel_wrapped_in_level(previous, sx, sy)
                    + pixel_wrapped_in_level(previous, sx + 1, sy)
                    + pixel_wrapped_in_level(previous, sx, sy + 1)
                    + pixel_wrapped_in_level(previous, sx + 1, sy + 1))
                    * 0.25;
                pixels.push(texel);
            }
        }

        levels.push(TextureLevel {
            width,
            height,
            pixels,
        });
    }

    levels
}

fn pixel_wrapped_in_level<P: TexturePixel>(level: &TextureLevel<P>, x: isize, y: isize) -> P {
    let x = wrap_index(x, level.width);
    let y = wrap_index(y, level.height);

    level.pixels[y * level.width + x]
}

pub(super) fn load_optional_color_texture(
    path: Option<&Path>,
    color_space: TextureColorSpace,
) -> image::ImageResult<Option<Arc<Texture<Vec3>>>> {
    path.map(|path| Texture::from_file_with_color_space(path, color_space).map(Arc::new))
        .transpose()
}

pub(super) fn load_optional_scalar_texture(
    path: Option<&Path>,
) -> image::ImageResult<Option<Arc<ScalarTexture>>> {
    path.map(|path| ScalarTexture::from_file(path).map(Arc::new))
        .transpose()
}

fn decode_color_space(rgb: Vec3, color_space: TextureColorSpace) -> Vec3 {
    match color_space {
        TextureColorSpace::Linear => rgb,
        TextureColorSpace::Srgb => srgb_to_linear(rgb),
    }
}

fn wrap_unit(t: f32) -> f32 {
    if t.is_finite() {
        t.rem_euclid(1.0)
    } else {
        0.0
    }
}

fn wrap_index(index: isize, size: usize) -> usize {
    index.rem_euclid(size as isize) as usize
}

#[cfg(test)]
mod tests {
    use glam::{Vec2, Vec3};

    use super::{ScalarTexture, Texture, TextureColorSpace, decode_color_space};

    fn test_texture() -> Texture {
        Texture::from_pixels(
            2,
            2,
            vec![
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(1.0, 1.0, 1.0),
            ],
        )
    }

    #[test]
    fn samples_texel_centers() {
        let texture = test_texture();

        assert_eq!(
            texture.sample(Vec2::new(0.25, 0.25)),
            Vec3::new(1.0, 0.0, 0.0)
        );
        assert_eq!(
            texture.sample(Vec2::new(0.75, 0.75)),
            Vec3::new(1.0, 1.0, 1.0)
        );
    }

    #[test]
    fn bilinearly_interpolates_between_texels() {
        let texture = test_texture();

        assert!(
            texture
                .sample(Vec2::new(0.5, 0.5))
                .abs_diff_eq(Vec3::splat(0.5), 1.0e-6)
        );
    }

    #[test]
    fn wraps_uvs() {
        let texture = test_texture();

        assert_eq!(
            texture.sample(Vec2::new(1.25, -0.75)),
            Vec3::new(1.0, 0.0, 0.0)
        );
    }

    #[test]
    fn scalar_texture_samples_single_channel() {
        let texture = ScalarTexture::from_pixels(1, 1, vec![0.25_f32]);
        assert_eq!(texture.sample(Vec2::ZERO), 0.25);
    }

    #[test]
    fn scalar_texture_bilerp_returns_average() {
        let texture = ScalarTexture::from_pixels(2, 2, vec![0.0, 1.0, 1.0, 0.0]);
        assert!((texture.sample(Vec2::new(0.5, 0.5)) - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn from_rgba_channel_extracts_requested_byte() {
        let texture =
            ScalarTexture::from_rgba_channel(2, 1, &[10, 20, 30, 40, 110, 120, 130, 140], 2);
        assert!((texture.sample(Vec2::new(0.25, 0.5)) - 30.0 / 255.0).abs() < 1.0e-6);
        assert!((texture.sample(Vec2::new(0.75, 0.5)) - 130.0 / 255.0).abs() < 1.0e-6);
    }

    #[test]
    fn mip_pyramid_averages_to_top_level() {
        let texture = test_texture();

        assert_eq!(texture.level_count(), 2);
        assert!(
            texture
                .pixel_level_wrapped(1, 0, 0)
                .abs_diff_eq(Vec3::splat(0.5), 1.0e-6)
        );
    }

    #[test]
    fn trilinear_sampling_uses_coarser_mip_for_wide_footprint() {
        let texture = test_texture();

        let filtered = texture.sample_trilinear(Vec2::new(0.25, 0.25), Vec2::X, Vec2::Y);

        assert!(filtered.abs_diff_eq(Vec3::splat(0.5), 1.0e-6));
    }

    #[test]
    fn ewa_sampling_uses_coarser_mip_for_wide_footprint() {
        let texture = test_texture();

        let filtered = texture.sample_filtered(Vec2::new(0.25, 0.25), Vec2::X, Vec2::Y);

        assert!(filtered.abs_diff_eq(Vec3::splat(0.5), 1.0e-6));
    }

    #[test]
    fn srgb_decode_converts_color_channels_to_linear() {
        let decoded = decode_color_space(Vec3::splat(0.5), TextureColorSpace::Srgb);

        assert!(decoded.abs_diff_eq(Vec3::splat(0.21404114), 1.0e-6));
    }

    #[test]
    fn linear_color_space_keeps_values_unchanged() {
        let value = Vec3::new(0.25, 0.5, 0.75);

        assert_eq!(decode_color_space(value, TextureColorSpace::Linear), value);
    }
}
