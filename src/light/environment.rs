use std::f32::consts::{PI, TAU};
use std::path::Path;

use glam::{Vec2, Vec3};

use super::{LightLiSample, LightType};
use crate::scene::Scene;

#[derive(Debug, Clone, PartialEq)]
pub struct EnvironmentLight {
    width: usize,
    height: usize,
    scale: f32,
    pixels: Vec<Vec3>,
    distribution: EnvironmentDistribution,
    mis_compensated_distribution: EnvironmentDistribution,
}

#[derive(Debug, Clone, PartialEq)]
struct EnvironmentDistribution {
    texel_weights: Vec<f32>,
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

        let distribution =
            build_distribution(width, height, environment_weights(width, height, &pixels));
        let mis_compensated_distribution = build_distribution(
            width,
            height,
            mis_compensated_environment_weights(width, height, &pixels),
        );

        Self {
            width,
            height,
            scale,
            pixels,
            distribution,
            mis_compensated_distribution,
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
        self.pdf_with_distribution(&self.distribution, direction)
    }

    pub fn pdf_mis_compensated(&self, direction: Vec3) -> f32 {
        self.pdf_with_distribution(&self.mis_compensated_distribution, direction)
    }

    pub fn sample(&self, us: Vec2) -> Option<EnvironmentLightSample> {
        self.sample_with_distribution(&self.distribution, us)
    }

    pub fn sample_mis_compensated(&self, us: Vec2) -> Option<EnvironmentLightSample> {
        self.sample_with_distribution(&self.mis_compensated_distribution, us)
    }

    fn pdf_with_distribution(
        &self,
        distribution: &EnvironmentDistribution,
        direction: Vec3,
    ) -> f32 {
        if distribution.total_weight <= 0.0 {
            return 0.0;
        }
        let uv = direction_to_uv(direction);
        let sin_theta = (uv.y * PI).sin();
        if sin_theta <= 0.0 {
            return 0.0;
        }
        let i = pixel_coord(uv.x, self.width);
        let j = pixel_coord(uv.y, self.height);
        let weight = distribution.texel_weight(i, j, self.width);
        if weight <= 0.0 {
            return 0.0;
        }
        let p2d = weight / distribution.total_weight * (self.width as f32) * (self.height as f32);
        p2d / (TAU * PI * sin_theta)
    }

    fn sample_with_distribution(
        &self,
        distribution: &EnvironmentDistribution,
        us: Vec2,
    ) -> Option<EnvironmentLightSample> {
        if distribution.total_weight <= 0.0 {
            return None;
        }

        let (j, dv) = sample_cdf(&distribution.marginal_cdf, us.y);
        let row_offset = j * (self.width + 1);
        let row = &distribution.conditional_cdf[row_offset..row_offset + self.width + 1];
        let (i, du) = sample_cdf(row, us.x);

        let u_cont = (i as f32 + du) / self.width as f32;
        let v_cont = (j as f32 + dv) / self.height as f32;
        let uv = Vec2::new(u_cont, v_cont);
        let direction = uv_to_direction(uv);

        let sin_theta = (v_cont * PI).sin();
        if sin_theta <= 0.0 {
            return None;
        }
        let weight = distribution.texel_weight(i, j, self.width);
        if weight <= 0.0 {
            return None;
        }

        let p2d = weight / distribution.total_weight * (self.width as f32) * (self.height as f32);
        let pdf = p2d / (TAU * PI * sin_theta);
        let radiance = self.scale * self.pixels[j * self.width + i];

        Some(EnvironmentLightSample {
            direction,
            radiance,
            uv,
            pdf,
        })
    }
}

pub(super) fn sample_li(scene: &Scene, us: Vec2) -> Option<LightLiSample> {
    let env = scene.environment_light.as_ref()?;
    let sample = env.sample(us)?;
    if sample.pdf <= 0.0 {
        return None;
    }

    Some(LightLiSample {
        radiance: sample.radiance,
        wi: sample.direction,
        pdf: sample.pdf,
        distance: f32::INFINITY,
        light_type: LightType::Infinite,
        target_triangle: None,
    })
}

pub(super) fn sample_li_mis_compensated(scene: &Scene, us: Vec2) -> Option<LightLiSample> {
    let env = scene.environment_light.as_ref()?;
    let sample = env.sample_mis_compensated(us)?;
    if sample.pdf <= 0.0 {
        return None;
    }

    Some(LightLiSample {
        radiance: sample.radiance,
        wi: sample.direction,
        pdf: sample.pdf,
        distance: f32::INFINITY,
        light_type: LightType::Infinite,
        target_triangle: None,
    })
}

pub fn infinite_light_pdf_li(scene: &Scene, direction: Vec3) -> f32 {
    scene
        .environment_light
        .as_ref()
        .map(|env| env.pdf(direction))
        .unwrap_or(0.0)
}

pub fn infinite_light_pdf_li_mis_compensated(scene: &Scene, direction: Vec3) -> f32 {
    scene
        .environment_light
        .as_ref()
        .map(|env| env.pdf_mis_compensated(direction))
        .unwrap_or(0.0)
}

pub fn infinite_light_le(scene: &Scene, direction: Vec3) -> Vec3 {
    scene
        .environment_light
        .as_ref()
        .map(|env| env.radiance(direction))
        .unwrap_or(Vec3::ZERO)
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

fn uv_to_direction(uv: Vec2) -> Vec3 {
    let phi = uv.x * TAU;
    let theta = uv.y * PI;
    let sin_theta = theta.sin();
    Vec3::new(sin_theta * phi.sin(), theta.cos(), sin_theta * phi.cos())
}

fn direction_to_uv(direction: Vec3) -> Vec2 {
    let dir = direction.normalize_or_zero();
    let y = dir.y.clamp(-1.0, 1.0);
    let theta = y.acos();
    let mut phi = dir.x.atan2(dir.z);
    if phi < 0.0 {
        phi += TAU;
    }
    Vec2::new((phi / TAU).clamp(0.0, 1.0), (theta / PI).clamp(0.0, 1.0))
}

impl EnvironmentDistribution {
    fn texel_weight(&self, i: usize, j: usize, width: usize) -> f32 {
        self.texel_weights[j * width + i]
    }
}

fn environment_weights(width: usize, height: usize, pixels: &[Vec3]) -> Vec<f32> {
    let mut weights = vec![0.0; width * height];
    for j in 0..height {
        let sin_theta = row_sin_theta(j, height);
        for i in 0..width {
            weights[j * width + i] = positive_luminance(pixels[j * width + i]) * sin_theta;
        }
    }
    weights
}

fn mis_compensated_environment_weights(width: usize, height: usize, pixels: &[Vec3]) -> Vec<f32> {
    let mean_luminance = solid_angle_mean_luminance(width, height, pixels);
    let compensation_epsilon = mean_luminance.abs().max(1.0) * 1.0e-6;
    let mut weights = vec![0.0; width * height];
    for j in 0..height {
        let sin_theta = row_sin_theta(j, height);
        for i in 0..width {
            let l = positive_luminance(pixels[j * width + i]);
            // Normal-independent MIS compensation for equal BSDF/env sampling.
            // The texel-domain CDF still needs the lat-long sin(theta) jacobian.
            let compensated = l - mean_luminance;
            weights[j * width + i] = if compensated > compensation_epsilon {
                compensated * sin_theta
            } else {
                0.0
            };
        }
    }
    weights
}

fn solid_angle_mean_luminance(width: usize, height: usize, pixels: &[Vec3]) -> f32 {
    let mut weighted_sum = 0.0;
    let mut weight_sum = 0.0;
    for j in 0..height {
        let sin_theta = row_sin_theta(j, height);
        weight_sum += sin_theta * width as f32;
        for i in 0..width {
            weighted_sum += positive_luminance(pixels[j * width + i]) * sin_theta;
        }
    }
    if weight_sum > 0.0 {
        weighted_sum / weight_sum
    } else {
        0.0
    }
}

fn positive_luminance(v: Vec3) -> f32 {
    luminance(v).max(0.0)
}

fn build_distribution(
    width: usize,
    height: usize,
    mut texel_weights: Vec<f32>,
) -> EnvironmentDistribution {
    assert_eq!(texel_weights.len(), width * height);
    for weight in &mut texel_weights {
        *weight = weight.max(0.0);
    }

    let mut conditional_cdf = vec![0.0f32; height * (width + 1)];
    let mut row_integrals = vec![0.0f32; height];

    for j in 0..height {
        let row_offset = j * (width + 1);
        for i in 0..width {
            let weight = texel_weights[j * width + i];
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

    EnvironmentDistribution {
        texel_weights,
        conditional_cdf,
        marginal_cdf,
        total_weight: total,
    }
}

fn sample_cdf(cdf: &[f32], u: f32) -> (usize, f32) {
    debug_assert!(cdf.len() >= 2);
    let last = cdf.len() - 2;
    let u = u.clamp(0.0, 1.0);
    let idx = cdf.partition_point(|&c| c <= u).saturating_sub(1).min(last);
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

    use super::super::{LightKind, LightSampleContext, LightType, sample_light_li};
    use super::{
        EnvironmentLight, direction_to_uv, infinite_light_le, infinite_light_pdf_li,
        uv_to_direction,
    };
    use crate::scene::Scene;

    fn uniform_environment(width: usize, height: usize, radiance: f32) -> EnvironmentLight {
        let pixels = vec![Vec3::splat(radiance); width * height];
        EnvironmentLight::from_pixels(width, height, pixels, 1.0)
    }

    #[test]
    fn direction_uv_roundtrip_on_axes() {
        for direction in [
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
            Vec3::X,
            Vec3::NEG_X,
        ] {
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
            let sample = env
                .sample(Vec2::new(ux, uy))
                .expect("sample should succeed");
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
    fn uniform_environment_has_no_mis_compensated_samples() {
        let env = uniform_environment(8, 4, 1.0);

        assert!(env.sample_mis_compensated(Vec2::new(0.5, 0.5)).is_none());
        assert_eq!(env.pdf_mis_compensated(Vec3::Y), 0.0);
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
            let sample = env
                .sample(Vec2::new(ux, uy))
                .expect("sample should succeed");
            let uv = direction_to_uv(sample.direction);
            let i = (uv.x * width as f32) as usize;
            let j = (uv.y * height as f32) as usize;
            assert_eq!(i, bright_i);
            assert_eq!(j, bright_j);
        }
    }

    #[test]
    fn mis_compensated_environment_keeps_only_above_mean_texels() {
        let width = 4;
        let height = 2;
        let bright_i = 2;
        let bright_j = 1;
        let mut pixels = vec![Vec3::splat(1.0); width * height];
        pixels[bright_j * width + bright_i] = Vec3::splat(9.0);
        let env = EnvironmentLight::from_pixels(width, height, pixels, 1.0);

        let low_direction = uv_to_direction(Vec2::new(0.125, 0.25));
        let bright_direction = uv_to_direction(Vec2::new(
            (bright_i as f32 + 0.5) / width as f32,
            (bright_j as f32 + 0.5) / height as f32,
        ));

        assert_eq!(env.pdf_mis_compensated(low_direction), 0.0);
        assert!(env.pdf_mis_compensated(bright_direction) > 0.0);

        for (ux, uy) in [(0.1, 0.1), (0.4, 0.4), (0.7, 0.7), (0.95, 0.95)] {
            let sample = env
                .sample_mis_compensated(Vec2::new(ux, uy))
                .expect("compensated sample should choose the bright texel");
            let uv = direction_to_uv(sample.direction);
            let i = (uv.x * width as f32) as usize;
            let j = (uv.y * height as f32) as usize;
            assert_eq!(i, bright_i);
            assert_eq!(j, bright_j);
        }
    }

    #[test]
    fn mis_compensated_mean_uses_spherical_jacobian() {
        let width = 1;
        let height = 4;
        let mut pixels = vec![Vec3::ZERO; width * height];
        pixels[0] = Vec3::splat(4.0);
        pixels[1] = Vec3::splat(1.2);
        let env = EnvironmentLight::from_pixels(width, height, pixels, 1.0);

        let high_solid_angle_row_direction = uv_to_direction(Vec2::new(0.5, 0.375));

        assert!(
            env.pdf_mis_compensated(high_solid_angle_row_direction) > 0.0,
            "row 1 stays above the jacobian-weighted mean"
        );
    }

    #[test]
    fn mis_compensated_sample_pdf_matches_pdf_query() {
        let mut pixels = vec![Vec3::splat(0.2); 16 * 8];
        pixels[2 * 16 + 4] = Vec3::splat(8.0);
        pixels[5 * 16 + 11] = Vec3::splat(4.0);
        let env = EnvironmentLight::from_pixels(16, 8, pixels, 1.0);

        for (ux, uy) in [(0.1, 0.1), (0.5, 0.5), (0.8, 0.3), (0.95, 0.85)] {
            let sample = env
                .sample_mis_compensated(Vec2::new(ux, uy))
                .expect("compensated sample should succeed");
            let queried = env.pdf_mis_compensated(sample.direction);
            assert!(
                (sample.pdf - queried).abs() / sample.pdf.max(1.0e-6) < 1.0e-3,
                "pdf mismatch: sample.pdf={}, pdf(dir)={}",
                sample.pdf,
                queried
            );
            assert!(sample.pdf > 0.0);
        }
    }

    #[test]
    fn integrated_pdf_over_sphere_is_one() {
        let env = uniform_environment(64, 32, 1.0);
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
        let integral = sum / n as f32 * 4.0 * PI;
        assert!(
            (integral - 1.0).abs() < 5.0e-2,
            "integrated pdf = {integral}"
        );
    }

    #[test]
    fn sample_light_li_for_infinite_returns_environment_sample() {
        let mut scene = Scene::new();
        let pixels = vec![Vec3::splat(1.0); 32 * 16];
        scene.set_environment_light(EnvironmentLight::from_pixels(32, 16, pixels, 1.0));

        let ctx = LightSampleContext {
            p: Vec3::ZERO,
            ng: Vec3::Z,
            ns: Vec3::Z,
        };

        let li = sample_light_li(&scene, LightKind::Infinite, &ctx, 0.0, Vec2::new(0.3, 0.6))
            .expect("expected a sample");

        assert_eq!(li.light_type, LightType::Infinite);
        assert!(li.target_triangle.is_none());
        assert!(li.distance.is_infinite());
        assert!((li.wi.length() - 1.0).abs() < 1.0e-5);
        assert!(li.pdf > 0.0);
    }

    #[test]
    fn infinite_light_helpers_report_zero_when_missing() {
        let scene = Scene::new();
        assert_eq!(infinite_light_le(&scene, Vec3::Z), Vec3::ZERO);
        assert_eq!(infinite_light_pdf_li(&scene, Vec3::Z), 0.0);
    }
}
