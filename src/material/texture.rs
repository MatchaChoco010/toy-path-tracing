use std::path::Path;

use glam::{Vec2, Vec3};

#[derive(Debug, Clone, PartialEq)]
pub struct Texture {
    width: usize,
    height: usize,
    pixels: Vec<Vec3>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureColorSpace {
    Linear,
    Srgb,
}

impl Texture {
    pub fn from_pixels(width: usize, height: usize, pixels: Vec<Vec3>) -> Self {
        assert!(width > 0 && height > 0, "texture must be non-empty");
        assert_eq!(
            pixels.len(),
            width * height,
            "pixel buffer length does not match width * height"
        );

        Self {
            width,
            height,
            pixels,
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
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn pixels(&self) -> &[Vec3] {
        &self.pixels
    }

    pub fn sample_rgb(&self, uv: Vec2) -> Vec3 {
        let u = wrap_unit(uv.x);
        let v = wrap_unit(uv.y);
        let x = u * self.width as f32 - 0.5;
        let y = v * self.height as f32 - 0.5;
        let x0 = x.floor() as isize;
        let y0 = y.floor() as isize;
        let tx = x - x0 as f32;
        let ty = y - y0 as f32;

        let c00 = self.pixel_wrapped(x0, y0);
        let c10 = self.pixel_wrapped(x0 + 1, y0);
        let c01 = self.pixel_wrapped(x0, y0 + 1);
        let c11 = self.pixel_wrapped(x0 + 1, y0 + 1);
        let cx0 = c00.lerp(c10, tx);
        let cx1 = c01.lerp(c11, tx);

        cx0.lerp(cx1, ty)
    }

    pub fn sample_scalar(&self, uv: Vec2) -> f32 {
        self.sample_rgb(uv).x
    }

    fn pixel_wrapped(&self, x: isize, y: isize) -> Vec3 {
        let x = wrap_index(x, self.width);
        let y = wrap_index(y, self.height);

        self.pixels[y * self.width + x]
    }
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
