use std::f32::consts::{PI, TAU};
use std::path::Path;

use glam::{Mat3, Vec2, Vec3};

use super::{LightLiSample, LightType};
use crate::scene::Scene;

#[derive(Debug, Clone, PartialEq)]
pub struct EnvironmentLight {
    width: usize,
    height: usize,
    scale: f32,
    rotate_y: f32,
    rotation: Mat3,
    inverse_rotation: Mat3,
    pixels: Vec<Vec3>,
    distribution: HierarchicalDistribution,
    mis_compensated_distribution: HierarchicalDistribution,
}

#[derive(Debug, Clone, PartialEq)]
struct HierarchicalDistribution {
    levels: Vec<MipLevel>,
    padded_width: usize,
    padded_height: usize,
    total_weight: f32,
}

#[derive(Debug, Clone, PartialEq)]
struct MipLevel {
    width: usize,
    height: usize,
    weights: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnvironmentLightSample {
    pub direction: Vec3,
    pub radiance: Vec3,
    pub uv: Vec2,
    pub pdf: f32,
}

impl EnvironmentLight {
    pub fn from_pixels(
        width: usize,
        height: usize,
        pixels: Vec<Vec3>,
        scale: f32,
        rotate_y: f32,
    ) -> Self {
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

        let rotation = Mat3::from_rotation_y(rotate_y);
        let inverse_rotation = Mat3::from_rotation_y(-rotate_y);

        Self {
            width,
            height,
            scale,
            rotate_y,
            rotation,
            inverse_rotation,
            pixels,
            distribution,
            mis_compensated_distribution,
        }
    }

    pub fn from_hdr_file(
        path: impl AsRef<Path>,
        scale: f32,
        rotate_y: f32,
    ) -> image::ImageResult<Self> {
        let dynamic = image::open(crate::paths::workspace_path(path))?;
        let rgb32f = dynamic.into_rgb32f();
        let width = rgb32f.width() as usize;
        let height = rgb32f.height() as usize;
        let pixels = rgb32f
            .pixels()
            .map(|p| Vec3::new(p[0], p[1], p[2]))
            .collect();

        Ok(Self::from_pixels(width, height, pixels, scale, rotate_y))
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

    pub fn rotate_y(&self) -> f32 {
        self.rotate_y
    }

    pub fn pixels(&self) -> &[Vec3] {
        &self.pixels
    }

    /// Total radiance integrated over the unit sphere, `∫ L(ω) dω`,
    /// scaled by `self.scale`. Comparable in units to a directional light's
    /// `intensity * luminance(color)` for the top-level light category CDF.
    pub fn total_power(&self) -> f32 {
        let texel_solid_angle = TAU * PI / (self.width as f32 * self.height as f32);
        self.distribution.total_weight * self.scale * texel_solid_angle
    }

    pub fn radiance(&self, direction: Vec3) -> Vec3 {
        let local_dir = self.inverse_rotation * direction;
        let uv = direction_to_uv(local_dir);
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
        distribution: &HierarchicalDistribution,
        direction: Vec3,
    ) -> f32 {
        if distribution.total_weight <= 0.0 {
            return 0.0;
        }
        let local_dir = self.inverse_rotation * direction;
        let uv = direction_to_uv(local_dir);
        let sin_theta = (uv.y * PI).sin();
        if sin_theta <= 0.0 {
            return 0.0;
        }
        let i = pixel_coord(uv.x, self.width);
        let j = pixel_coord(uv.y, self.height);
        let weight = distribution.texel_weight(i, j);
        if weight <= 0.0 {
            return 0.0;
        }
        let p2d = weight / distribution.total_weight * (self.width as f32) * (self.height as f32);
        p2d / (TAU * PI * sin_theta)
    }

    fn sample_with_distribution(
        &self,
        distribution: &HierarchicalDistribution,
        us: Vec2,
    ) -> Option<EnvironmentLightSample> {
        if distribution.total_weight <= 0.0 {
            return None;
        }

        let mut u = us.x.clamp(0.0, 1.0);
        let mut v = us.y.clamp(0.0, 1.0);
        let mut i = 0usize;
        let mut j = 0usize;

        for level_idx in 1..distribution.levels.len() {
            let parent = &distribution.levels[level_idx - 1];
            let child = &distribution.levels[level_idx];
            let (ni, nj, nu, nv) = hsw_step(parent.width, parent.height, child, i, j, u, v)?;
            i = ni;
            j = nj;
            u = nu;
            v = nv;
        }

        if i >= self.width || j >= self.height {
            return None;
        }

        let weight = distribution.texel_weight(i, j);
        if weight <= 0.0 {
            return None;
        }

        let u_cont = (i as f32 + u) / self.width as f32;
        let v_cont = (j as f32 + v) / self.height as f32;
        let uv = Vec2::new(
            u_cont.clamp(0.0, ONE_MINUS_EPS),
            v_cont.clamp(0.0, ONE_MINUS_EPS),
        );
        let local_direction = uv_to_direction(uv);
        let direction = self.rotation * local_direction;

        let sin_theta = (v_cont * PI).sin();
        if sin_theta <= 0.0 {
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

const ONE_MINUS_EPS: f32 = 1.0 - f32::EPSILON;

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

impl HierarchicalDistribution {
    fn leaf(&self) -> &MipLevel {
        self.levels
            .last()
            .expect("pyramid must have at least one level")
    }

    fn texel_weight(&self, i: usize, j: usize) -> f32 {
        let leaf = self.leaf();
        leaf.weights[j * leaf.width + i]
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
    pixel_weights: Vec<f32>,
) -> HierarchicalDistribution {
    debug_assert_eq!(pixel_weights.len(), width * height);

    let padded_width = width.next_power_of_two();
    let padded_height = height.next_power_of_two();

    let mut leaf = vec![0.0f32; padded_width * padded_height];
    for j in 0..height {
        for i in 0..width {
            leaf[j * padded_width + i] = pixel_weights[j * width + i].max(0.0);
        }
    }

    let mut levels = Vec::new();
    levels.push(MipLevel {
        width: padded_width,
        height: padded_height,
        weights: leaf,
    });

    while levels
        .last()
        .map(|l| l.width > 1 || l.height > 1)
        .unwrap_or(false)
    {
        let finer = levels.last().unwrap();
        let scale_i = if finer.width > 1 { 2 } else { 1 };
        let scale_j = if finer.height > 1 { 2 } else { 1 };
        let cw = finer.width / scale_i;
        let ch = finer.height / scale_j;
        let mut weights = vec![0.0f32; cw * ch];
        for j in 0..ch {
            for i in 0..cw {
                let mut sum = 0.0f32;
                for dj in 0..scale_j {
                    for di in 0..scale_i {
                        let fi = i * scale_i + di;
                        let fj = j * scale_j + dj;
                        sum += finer.weights[fj * finer.width + fi];
                    }
                }
                weights[j * cw + i] = sum;
            }
        }
        levels.push(MipLevel {
            width: cw,
            height: ch,
            weights,
        });
    }

    levels.reverse();
    let total_weight = levels[0].weights[0];

    HierarchicalDistribution {
        levels,
        padded_width,
        padded_height,
        total_weight,
    }
}

fn hsw_step(
    parent_w: usize,
    parent_h: usize,
    child: &MipLevel,
    parent_i: usize,
    parent_j: usize,
    mut u: f32,
    mut v: f32,
) -> Option<(usize, usize, f32, f32)> {
    let scale_i = child.width / parent_w;
    let scale_j = child.height / parent_h;
    let mut ci = parent_i * scale_i;
    let mut cj = parent_j * scale_j;

    if scale_j == 2 {
        let mut w_top = 0.0f32;
        let mut w_bot = 0.0f32;
        for di in 0..scale_i {
            w_top += child.weights[cj * child.width + ci + di];
            w_bot += child.weights[(cj + 1) * child.width + ci + di];
        }
        let total = w_top + w_bot;
        if total <= 0.0 {
            return None;
        }
        let p_top = w_top / total;
        if v < p_top {
            v = (v / p_top).clamp(0.0, ONE_MINUS_EPS);
        } else {
            v = ((v - p_top) / (1.0 - p_top)).clamp(0.0, ONE_MINUS_EPS);
            cj += 1;
        }
    }

    if scale_i == 2 {
        let w_left = child.weights[cj * child.width + ci];
        let w_right = child.weights[cj * child.width + ci + 1];
        let total = w_left + w_right;
        if total <= 0.0 {
            return None;
        }
        let p_left = w_left / total;
        if u < p_left {
            u = (u / p_left).clamp(0.0, ONE_MINUS_EPS);
        } else {
            u = ((u - p_left) / (1.0 - p_left)).clamp(0.0, ONE_MINUS_EPS);
            ci += 1;
        }
    }

    Some((ci, cj, u, v))
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use glam::{Vec2, Vec3};

    use super::super::LightType;
    use super::{
        EnvironmentLight, build_distribution, direction_to_uv, infinite_light_le,
        infinite_light_pdf_li, sample_li, uv_to_direction,
    };
    use crate::scene::Scene;

    fn uniform_environment(width: usize, height: usize, radiance: f32) -> EnvironmentLight {
        let pixels = vec![Vec3::splat(radiance); width * height];
        EnvironmentLight::from_pixels(width, height, pixels, 1.0, 0.0)
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
        let env = EnvironmentLight::from_pixels(16, 8, pixels, 1.0, 0.0);

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
        let env = EnvironmentLight::from_pixels(width, height, pixels, 1.0, 0.0);

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
        let env = EnvironmentLight::from_pixels(width, height, pixels, 1.0, 0.0);

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
        let env = EnvironmentLight::from_pixels(width, height, pixels, 1.0, 0.0);

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
        let env = EnvironmentLight::from_pixels(16, 8, pixels, 1.0, 0.0);

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
    fn sample_li_for_infinite_returns_environment_sample() {
        let mut scene = Scene::new();
        let pixels = vec![Vec3::splat(1.0); 32 * 16];
        scene.set_environment_light(EnvironmentLight::from_pixels(32, 16, pixels, 1.0, 0.0));

        let li = sample_li(&scene, Vec2::new(0.3, 0.6)).expect("expected a sample");

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

    #[test]
    fn pyramid_root_holds_total_weight_and_levels_collapse_to_root() {
        let weights = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let dist = build_distribution(3, 2, weights);

        assert_eq!(dist.padded_width, 4);
        assert_eq!(dist.padded_height, 2);
        assert!((dist.total_weight - (1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0)).abs() < 1.0e-5);
        assert_eq!(dist.levels.first().unwrap().width, 1);
        assert_eq!(dist.levels.first().unwrap().height, 1);
        assert_eq!(dist.levels.last().unwrap().width, 4);
        assert_eq!(dist.levels.last().unwrap().height, 2);

        let leaf = dist.levels.last().unwrap();
        assert_eq!(leaf.weights[3], 0.0);
        assert_eq!(leaf.weights[4 + 3], 0.0);
        assert_eq!(leaf.weights[0], 1.0);
        assert_eq!(leaf.weights[2], 3.0);
        assert_eq!(leaf.weights[4 + 2], 6.0);
    }

    #[test]
    fn non_power_of_two_uniform_environment_has_inverse_sphere_pdf() {
        let env = uniform_environment(120, 60, 1.0);
        let direction = uv_to_direction(Vec2::new(0.31, 0.47));
        let pdf = env.pdf(direction);
        let expected = 1.0 / (4.0 * PI);

        assert!(
            (pdf - expected).abs() < 1.0e-2,
            "pdf {pdf} deviates from uniform sphere pdf {expected}"
        );
    }

    #[test]
    fn non_power_of_two_environment_sample_lands_in_valid_region() {
        let width = 5;
        let height = 3;
        let mut pixels = vec![Vec3::splat(1.0e-3); width * height];
        pixels[width + 3] = Vec3::splat(8.0);
        let env = EnvironmentLight::from_pixels(width, height, pixels, 1.0, 0.0);

        for (ux, uy) in [(0.05, 0.05), (0.5, 0.5), (0.95, 0.5), (0.5, 0.95)] {
            let sample = env
                .sample(Vec2::new(ux, uy))
                .expect("sample should succeed");
            let uv = direction_to_uv(sample.direction);
            let i = (uv.x * width as f32) as usize;
            let j = (uv.y * height as f32) as usize;
            assert!(
                i < width && j < height,
                "sample fell into padding: i={i}, j={j}"
            );
            let queried = env.pdf(sample.direction);
            assert!(
                (sample.pdf - queried).abs() / sample.pdf.max(1.0e-6) < 1.0e-3,
                "pdf mismatch in non-power-of-two env"
            );
        }
    }

    #[test]
    fn rotate_y_rotates_radiance_and_keeps_pdf_consistent() {
        let width = 16;
        let height = 8;
        let mut pixels = vec![Vec3::splat(0.05); width * height];
        let bright_i = 4;
        let bright_j = 3;
        pixels[bright_j * width + bright_i] = Vec3::splat(20.0);

        let rotation_radians = std::f32::consts::FRAC_PI_2;
        let rotated_env =
            EnvironmentLight::from_pixels(width, height, pixels.clone(), 1.0, rotation_radians);
        let plain_env = EnvironmentLight::from_pixels(width, height, pixels, 1.0, 0.0);

        let bright_uv = Vec2::new(
            (bright_i as f32 + 0.5) / width as f32,
            (bright_j as f32 + 0.5) / height as f32,
        );
        let local_dir = uv_to_direction(bright_uv);
        let rotated_dir = glam::Mat3::from_rotation_y(rotation_radians) * local_dir;

        let plain_radiance = plain_env.radiance(local_dir);
        let rotated_radiance = rotated_env.radiance(rotated_dir);
        assert!(
            (plain_radiance - rotated_radiance).length() < 1.0e-3,
            "rotated radiance lookup must match plain radiance at the rotated direction"
        );

        for (ux, uy) in [(0.1, 0.2), (0.4, 0.6), (0.85, 0.55)] {
            let sample = rotated_env
                .sample(Vec2::new(ux, uy))
                .expect("sample should succeed");
            let queried = rotated_env.pdf(sample.direction);
            assert!(
                (sample.pdf - queried).abs() / sample.pdf.max(1.0e-6) < 1.0e-3,
                "pdf mismatch under rotation"
            );
            assert!((sample.direction.length() - 1.0).abs() < 1.0e-5);
        }
    }
}
