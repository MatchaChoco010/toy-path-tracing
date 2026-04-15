use std::f32::consts::{PI, TAU};
use std::path::Path;

use glam::{Vec2, Vec3};

#[derive(Debug, Clone, PartialEq)]
pub struct EnvironmentLight {
    width: usize,
    height: usize,
    scale: f32,
    pixels: Vec<Vec3>,
    conditional_cdf: Vec<f32>,
    marginal_cdf: Vec<f32>,
    total_weight: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnvironmentLightSample {
    pub direction: Vec3,
    pub radiance: Vec3,
    pub uv: Vec2,
    pub pdf: f32,
}

impl EnvironmentLight {
    pub fn from_pixels(width: usize, height: usize, pixels: Vec<Vec3>, scale: f32) -> Self {
        assert!(width > 0 && height > 0, "environment map must be non-empty");
        assert_eq!(
            pixels.len(),
            width * height,
            "pixel buffer length does not match width * height"
        );

        let (conditional_cdf, marginal_cdf, total_weight) =
            build_distribution(width, height, &pixels);

        Self {
            width,
            height,
            scale,
            pixels,
            conditional_cdf,
            marginal_cdf,
            total_weight,
        }
    }

    pub fn from_hdr_file(path: impl AsRef<Path>, scale: f32) -> image::ImageResult<Self> {
        let dynamic = image::open(path)?;
        let rgb32f = dynamic.into_rgb32f();
        let width = rgb32f.width() as usize;
        let height = rgb32f.height() as usize;
        let pixels = rgb32f
            .pixels()
            .map(|p| Vec3::new(p[0], p[1], p[2]))
            .collect();

        Ok(Self::from_pixels(width, height, pixels, scale))
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn set_scale(&mut self, scale: f32) {
        self.scale = scale;
    }

    pub fn pixels(&self) -> &[Vec3] {
        &self.pixels
    }

    pub fn radiance(&self, direction: Vec3) -> Vec3 {
        let uv = direction_to_uv(direction);
        let i = pixel_coord(uv.x, self.width);
        let j = pixel_coord(uv.y, self.height);
        self.scale * self.pixels[j * self.width + i]
    }

    pub fn pdf(&self, direction: Vec3) -> f32 {
        if self.total_weight <= 0.0 {
            return 0.0;
        }
        let uv = direction_to_uv(direction);
        let sin_theta = (uv.y * PI).sin();
        if sin_theta <= 0.0 {
            return 0.0;
        }
        let i = pixel_coord(uv.x, self.width);
        let j = pixel_coord(uv.y, self.height);
        let weight = self.pixel_weight(i, j);
        if weight <= 0.0 {
            return 0.0;
        }
        let p2d = weight / self.total_weight * (self.width as f32) * (self.height as f32);
        p2d / (TAU * PI * sin_theta)
    }

    pub fn sample(&self, us: Vec2) -> Option<EnvironmentLightSample> {
        if self.total_weight <= 0.0 {
            return None;
        }

        let (j, dv) = sample_cdf(&self.marginal_cdf, us.y);
        let row_offset = j * (self.width + 1);
        let row = &self.conditional_cdf[row_offset..row_offset + self.width + 1];
        let (i, du) = sample_cdf(row, us.x);

        let u_cont = (i as f32 + du) / self.width as f32;
        let v_cont = (j as f32 + dv) / self.height as f32;
        let uv = Vec2::new(u_cont, v_cont);
        let direction = uv_to_direction(uv);

        let sin_theta = (v_cont * PI).sin();
        if sin_theta <= 0.0 {
            return None;
        }
        let weight = self.pixel_weight(i, j);
        if weight <= 0.0 {
            return None;
        }

        let p2d = weight / self.total_weight * (self.width as f32) * (self.height as f32);
        let pdf = p2d / (TAU * PI * sin_theta);
        let radiance = self.scale * self.pixels[j * self.width + i];

        Some(EnvironmentLightSample {
            direction,
            radiance,
            uv,
            pdf,
        })
    }

    fn pixel_weight(&self, i: usize, j: usize) -> f32 {
        let pixel = self.pixels[j * self.width + i];
        luminance(pixel).max(0.0) * row_sin_theta(j, self.height)
    }
}

fn luminance(v: Vec3) -> f32 {
    0.2126 * v.x + 0.7152 * v.y + 0.0722 * v.z
}

fn row_sin_theta(j: usize, height: usize) -> f32 {
    let v = (j as f32 + 0.5) / height as f32;
    (v * PI).sin()
}

fn pixel_coord(t: f32, size: usize) -> usize {
    let clamped = t.clamp(0.0, 1.0);
    let idx = (clamped * size as f32) as usize;
    idx.min(size - 1)
}

pub fn uv_to_direction(uv: Vec2) -> Vec3 {
    let phi = uv.x * TAU;
    let theta = uv.y * PI;
    let sin_theta = theta.sin();
    Vec3::new(sin_theta * phi.sin(), theta.cos(), sin_theta * phi.cos())
}

pub fn direction_to_uv(direction: Vec3) -> Vec2 {
    let dir = direction.normalize_or_zero();
    let y = dir.y.clamp(-1.0, 1.0);
    let theta = y.acos();
    let mut phi = dir.x.atan2(dir.z);
    if phi < 0.0 {
        phi += TAU;
    }
    Vec2::new((phi / TAU).clamp(0.0, 1.0), (theta / PI).clamp(0.0, 1.0))
}

fn build_distribution(
    width: usize,
    height: usize,
    pixels: &[Vec3],
) -> (Vec<f32>, Vec<f32>, f32) {
    let mut conditional_cdf = vec![0.0f32; height * (width + 1)];
    let mut row_integrals = vec![0.0f32; height];

    for j in 0..height {
        let sin_theta = row_sin_theta(j, height);
        let row_offset = j * (width + 1);
        for i in 0..width {
            let weight = luminance(pixels[j * width + i]).max(0.0) * sin_theta;
            conditional_cdf[row_offset + i + 1] = conditional_cdf[row_offset + i] + weight;
        }
        let row_sum = conditional_cdf[row_offset + width];
        row_integrals[j] = row_sum;
        if row_sum > 0.0 {
            let inv = 1.0 / row_sum;
            for k in 0..=width {
                conditional_cdf[row_offset + k] *= inv;
            }
        }
    }

    let mut marginal_cdf = vec![0.0f32; height + 1];
    for j in 0..height {
        marginal_cdf[j + 1] = marginal_cdf[j] + row_integrals[j];
    }
    let total = marginal_cdf[height];
    if total > 0.0 {
        let inv = 1.0 / total;
        for entry in marginal_cdf.iter_mut() {
            *entry *= inv;
        }
    }

    (conditional_cdf, marginal_cdf, total)
}

fn sample_cdf(cdf: &[f32], u: f32) -> (usize, f32) {
    debug_assert!(cdf.len() >= 2);
    let last = cdf.len() - 2;
    let u = u.clamp(0.0, 1.0);
    let idx = cdf
        .partition_point(|&c| c <= u)
        .saturating_sub(1)
        .min(last);
    let a = cdf[idx];
    let b = cdf[idx + 1];
    let du = if b > a {
        ((u - a) / (b - a)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (idx, du)
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use glam::{Vec2, Vec3};

    use super::{EnvironmentLight, direction_to_uv, uv_to_direction};

    fn uniform_environment(width: usize, height: usize, radiance: f32) -> EnvironmentLight {
        let pixels = vec![Vec3::splat(radiance); width * height];
        EnvironmentLight::from_pixels(width, height, pixels, 1.0)
    }

    #[test]
    fn direction_uv_roundtrip_on_axes() {
        for direction in [Vec3::Y, Vec3::NEG_Y, Vec3::Z, Vec3::NEG_Z, Vec3::X, Vec3::NEG_X] {
            let uv = direction_to_uv(direction);
            let reconstructed = uv_to_direction(uv);
            assert!(
                reconstructed.abs_diff_eq(direction, 1.0e-5),
                "direction {direction:?} round-tripped to {reconstructed:?}"
            );
        }
    }

    #[test]
    fn uv_to_direction_produces_unit_vector() {
        let dir = uv_to_direction(Vec2::new(0.37, 0.62));
        assert!((dir.length() - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn uniform_environment_has_inverse_sphere_pdf() {
        let env = uniform_environment(128, 64, 1.0);
        let direction = uv_to_direction(Vec2::new(0.25, 0.5));
        let pdf = env.pdf(direction);
        let expected = 1.0 / (4.0 * PI);

        assert!(
            (pdf - expected).abs() < 5.0e-3,
            "pdf {pdf} deviates from uniform sphere pdf {expected}"
        );
    }

    #[test]
    fn sample_pdf_matches_pdf_query() {
        let mut pixels = vec![Vec3::splat(0.05); 16 * 8];
        pixels[3 * 16 + 9] = Vec3::splat(10.0);
        let env = EnvironmentLight::from_pixels(16, 8, pixels, 1.0);

        for (ux, uy) in [(0.1, 0.1), (0.5, 0.5), (0.8, 0.3), (0.95, 0.85)] {
            let sample = env.sample(Vec2::new(ux, uy)).expect("sample should succeed");
            let queried = env.pdf(sample.direction);
            assert!(
                (sample.pdf - queried).abs() / sample.pdf.max(1.0e-6) < 1.0e-3,
                "pdf mismatch: sample.pdf={}, pdf(dir)={}",
                sample.pdf,
                queried
            );
            assert!((sample.direction.length() - 1.0).abs() < 1.0e-5);
            assert!(sample.pdf > 0.0);
        }
    }

    #[test]
    fn zero_radiance_environment_returns_no_sample() {
        let env = uniform_environment(8, 4, 0.0);
        assert!(env.sample(Vec2::new(0.5, 0.5)).is_none());
        assert_eq!(env.pdf(Vec3::Y), 0.0);
    }

    #[test]
    fn peaked_environment_concentrates_samples_on_bright_pixel() {
        let width = 32;
        let height = 16;
        let mut pixels = vec![Vec3::splat(1.0e-4); width * height];
        let bright_i = 10;
        let bright_j = 6;
        pixels[bright_j * width + bright_i] = Vec3::splat(1.0e6);
        let env = EnvironmentLight::from_pixels(width, height, pixels, 1.0);

        for (ux, uy) in [(0.1, 0.1), (0.4, 0.4), (0.7, 0.7), (0.95, 0.05)] {
            let sample = env.sample(Vec2::new(ux, uy)).expect("sample should succeed");
            let uv = direction_to_uv(sample.direction);
            let i = (uv.x * width as f32) as usize;
            let j = (uv.y * height as f32) as usize;
            assert_eq!(i, bright_i);
            assert_eq!(j, bright_j);
        }
    }

    #[test]
    fn integrated_pdf_over_sphere_is_one() {
        let env = uniform_environment(64, 32, 1.0);
        // Monte-Carlo integrate pdf on uniformly sampled sphere directions.
        let n = 4096;
        let mut sum = 0.0;
        for k in 0..n {
            let k = k as u32;
            let u1 = ((k.wrapping_mul(2654435761)) & 0xffffff) as f32 / 0x1000000 as f32;
            let u2 = ((k.wrapping_mul(40503)) & 0xffffff) as f32 / 0x1000000 as f32;
            let z = 1.0 - 2.0 * u1;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let phi = 2.0 * PI * u2;
            let dir = Vec3::new(r * phi.cos(), z, r * phi.sin());
            sum += env.pdf(dir);
        }
        // Expected: pdf = 1/(4π). Mean * 4π ≈ 1.
        let integral = sum / n as f32 * 4.0 * PI;
        assert!((integral - 1.0).abs() < 5.0e-2, "integrated pdf = {integral}");
    }
}
