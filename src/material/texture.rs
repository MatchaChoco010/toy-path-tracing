use std::path::Path;

use glam::{Vec2, Vec3};

#[derive(Debug, Clone, PartialEq)]
pub struct Texture {
    levels: Vec<TextureLevel>,
}

#[derive(Debug, Clone, PartialEq)]
struct TextureLevel {
    width: usize,
    height: usize,
    pixels: Vec<Vec3>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureColorSpace {
    Linear,
    Srgb,
}

const MAX_ANISOTROPY: f32 = 8.0;
const MIN_FILTER_WIDTH: f32 = 1.0e-8;
const EWA_ALPHA: f32 = 2.0;

impl Texture {
    pub fn from_pixels(width: usize, height: usize, pixels: Vec<Vec3>) -> Self {
        assert!(width > 0 && height > 0, "texture must be non-empty");
        assert_eq!(
            pixels.len(),
            width * height,
            "pixel buffer length does not match width * height"
        );

        let base_level = TextureLevel {
            width,
            height,
            pixels,
        };

        Self {
            levels: build_mip_levels(base_level),
        }
    }

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

    pub fn width(&self) -> usize {
        self.level(0).width
    }

    pub fn height(&self) -> usize {
        self.level(0).height
    }

    pub fn pixels(&self) -> &[Vec3] {
        &self.level(0).pixels
    }

    pub fn sample_rgb(&self, uv: Vec2) -> Vec3 {
        self.bilerp_level(0, uv)
    }

    pub fn sample_rgb_filtered(&self, uv: Vec2, dstdx: Vec2, dstdy: Vec2) -> Vec3 {
        self.sample_rgb_ewa(uv, dstdx, dstdy)
    }

    pub fn sample_scalar(&self, uv: Vec2) -> f32 {
        self.sample_rgb(uv).x
    }

    pub fn sample_scalar_filtered(&self, uv: Vec2, dstdx: Vec2, dstdy: Vec2) -> f32 {
        self.sample_rgb_filtered(uv, dstdx, dstdy).x
    }

    fn sample_rgb_ewa(&self, uv: Vec2, mut dst0: Vec2, mut dst1: Vec2) -> Vec3 {
        if !dst0.is_finite() || !dst1.is_finite() {
            return self.sample_rgb(uv);
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
            return self.sample_rgb(uv);
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
    fn sample_rgb_trilinear(&self, uv: Vec2, dstdx: Vec2, dstdy: Vec2) -> Vec3 {
        if !dstdx.is_finite() || !dstdy.is_finite() {
            return self.sample_rgb(uv);
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

    fn bilerp_level(&self, level: usize, uv: Vec2) -> Vec3 {
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

    fn ewa_level(&self, level: usize, uv: Vec2, mut dst0: Vec2, mut dst1: Vec2) -> Vec3 {
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

        let mut sum = Vec3::ZERO;
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
                sum += weight * self.pixel_level_wrapped(level, is, it);
                weight_sum += weight;
            }
        }

        if weight_sum > 0.0 {
            sum / weight_sum
        } else {
            self.bilerp_level(level, uv)
        }
    }

    fn pixel_level_wrapped(&self, level: usize, x: isize, y: isize) -> Vec3 {
        let texture_level = self.level(level.min(self.level_count() - 1));
        let x = wrap_index(x, texture_level.width);
        let y = wrap_index(y, texture_level.height);

        texture_level.pixels[y * texture_level.width + x]
    }

    fn level(&self, level: usize) -> &TextureLevel {
        &self.levels[level]
    }

    fn level_count(&self) -> usize {
        self.levels.len()
    }
}

fn build_mip_levels(base_level: TextureLevel) -> Vec<TextureLevel> {
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
                let texel = 0.25
                    * (pixel_wrapped_in_level(previous, sx, sy)
                        + pixel_wrapped_in_level(previous, sx + 1, sy)
                        + pixel_wrapped_in_level(previous, sx, sy + 1)
                        + pixel_wrapped_in_level(previous, sx + 1, sy + 1));
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

fn pixel_wrapped_in_level(level: &TextureLevel, x: isize, y: isize) -> Vec3 {
    let x = wrap_index(x, level.width);
    let y = wrap_index(y, level.height);

    level.pixels[y * level.width + x]
}

pub(super) fn load_optional_texture(
    path: Option<&Path>,
    color_space: TextureColorSpace,
) -> image::ImageResult<Option<Texture>> {
    path.map(|path| Texture::from_file_with_color_space(path, color_space))
        .transpose()
}

fn decode_color_space(rgb: Vec3, color_space: TextureColorSpace) -> Vec3 {
    match color_space {
        TextureColorSpace::Linear => rgb,
        TextureColorSpace::Srgb => Vec3::new(
            srgb_channel_to_linear(rgb.x),
            srgb_channel_to_linear(rgb.y),
            srgb_channel_to_linear(rgb.z),
        ),
    }
}

fn srgb_channel_to_linear(channel: f32) -> f32 {
    if channel <= 0.04045 {
        channel / 12.92
    } else {
        ((channel + 0.055) / 1.055).powf(2.4)
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

    use super::{Texture, TextureColorSpace, decode_color_space};

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
            texture.sample_rgb(Vec2::new(0.25, 0.25)),
            Vec3::new(1.0, 0.0, 0.0)
        );
        assert_eq!(
            texture.sample_rgb(Vec2::new(0.75, 0.75)),
            Vec3::new(1.0, 1.0, 1.0)
        );
    }

    #[test]
    fn bilinearly_interpolates_between_texels() {
        let texture = test_texture();

        assert!(
            texture
                .sample_rgb(Vec2::new(0.5, 0.5))
                .abs_diff_eq(Vec3::splat(0.5), 1.0e-6)
        );
    }

    #[test]
    fn wraps_uvs() {
        let texture = test_texture();

        assert_eq!(
            texture.sample_rgb(Vec2::new(1.25, -0.75)),
            Vec3::new(1.0, 0.0, 0.0)
        );
    }

    #[test]
    fn scalar_sampling_uses_red_channel() {
        let texture = Texture::from_pixels(1, 1, vec![Vec3::new(0.25, 0.5, 0.75)]);

        assert_eq!(texture.sample_scalar(Vec2::ZERO), 0.25);
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

        let filtered = texture.sample_rgb_trilinear(Vec2::new(0.25, 0.25), Vec2::X, Vec2::Y);

        assert!(filtered.abs_diff_eq(Vec3::splat(0.5), 1.0e-6));
    }

    #[test]
    fn ewa_sampling_uses_coarser_mip_for_wide_footprint() {
        let texture = test_texture();

        let filtered = texture.sample_rgb_filtered(Vec2::new(0.25, 0.25), Vec2::X, Vec2::Y);

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
