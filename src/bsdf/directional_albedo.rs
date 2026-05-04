use std::{
    collections::HashMap,
    f32::consts::{FRAC_PI_2, PI, TAU},
    sync::Arc,
};

use glam::{Vec2, Vec3};
use rayon::prelude::*;

use crate::math::fresnel_dielectric;

use super::dielectric_ggx::{DielectricGgxAllowedPaths, DielectricGgxBsdf};
use super::sheen::sheen_directional_albedo_estimate;

const MIN_ALPHA: f32 = 1.0e-4;
const EFFECTIVELY_SMOOTH_ALPHA: f32 = 1.0e-3;

const DIELECTRIC_GGX_LUT_SQRT_COS_RESOLUTION: usize = 64;
const DIELECTRIC_GGX_LUT_PHI_RESOLUTION: usize = 16;
const DIELECTRIC_GGX_LUT_ROUGHNESS_RESOLUTION: usize = 64;
const DIELECTRIC_GGX_LUT_ANISOTROPY_RESOLUTION: usize = 16;
const DIELECTRIC_GGX_LUT_LEN: usize = DIELECTRIC_GGX_LUT_SQRT_COS_RESOLUTION
    * DIELECTRIC_GGX_LUT_PHI_RESOLUTION
    * DIELECTRIC_GGX_LUT_ROUGHNESS_RESOLUTION
    * DIELECTRIC_GGX_LUT_ANISOTROPY_RESOLUTION;

const DIRECTIONAL_ALBEDO_SAMPLE_COUNT: usize = 64;
const UNIFORM_HEMISPHERE_PDF: f32 = 1.0 / (2.0 * PI);

// Directional albedo LUTs are cached per BSDF type, not per material.
// Different BSDFs can need different cache keys, dimensions, and lookup
// parameters, so each BSDF gets a concrete cache field and LUT type here.
// Layered materials should request the LUT for their top-layer BSDF when they
// are added to the scene, then keep an Arc to the immutable LUT for shading.
#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct DirectionalAlbedoCache {
    dielectric_ggx:
        HashMap<DielectricGgxDirectionalAlbedoKey, Arc<DielectricGgxDirectionalAlbedoLut>>,
    sheen: Option<Arc<SheenDirectionalAlbedoLut>>,
}

impl DirectionalAlbedoCache {
    pub(crate) fn get_or_build_dielectric_ggx(
        &mut self,
        eta: f32,
    ) -> Arc<DielectricGgxDirectionalAlbedoLut> {
        let eta = sanitize_dielectric_eta(eta);
        let key = DielectricGgxDirectionalAlbedoKey::from_eta(eta);

        if let Some(lut) = self.dielectric_ggx.get(&key) {
            return Arc::clone(lut);
        }

        let lut = Arc::new(DielectricGgxDirectionalAlbedoLut::build(eta));
        self.dielectric_ggx.insert(key, Arc::clone(&lut));
        lut
    }

    pub(crate) fn get_or_build_sheen(&mut self) -> Arc<SheenDirectionalAlbedoLut> {
        if let Some(lut) = self.sheen.as_ref() {
            return Arc::clone(lut);
        }
        let lut = Arc::new(SheenDirectionalAlbedoLut::build());
        self.sheen = Some(Arc::clone(&lut));
        lut
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct DielectricGgxDirectionalAlbedoKey(u32);

impl DielectricGgxDirectionalAlbedoKey {
    fn from_eta(eta: f32) -> Self {
        Self(sanitize_dielectric_eta(eta).to_bits())
    }
}

#[derive(Debug, PartialEq)]
pub(crate) struct DielectricGgxDirectionalAlbedoLut {
    eta: f32,
    values: Vec<f32>,
}

impl DielectricGgxDirectionalAlbedoLut {
    fn build(eta: f32) -> Self {
        let eta = sanitize_dielectric_eta(eta);
        let hemisphere_samples = generate_uniform_hemisphere_samples();
        let mut values = vec![0.0; DIELECTRIC_GGX_LUT_LEN];

        values
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, value)| {
                let indices = DielectricGgxLutIndices::from_linear_index(index);
                let sqrt_cos =
                    lut_cell_center(indices.sqrt_cos, DIELECTRIC_GGX_LUT_SQRT_COS_RESOLUTION);
                let cos_theta = sqrt_cos * sqrt_cos;
                let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
                let phi =
                    lut_cell_center(indices.phi, DIELECTRIC_GGX_LUT_PHI_RESOLUTION) * FRAC_PI_2;
                let wo = Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta);
                let roughness =
                    lut_cell_center(indices.roughness, DIELECTRIC_GGX_LUT_ROUGHNESS_RESOLUTION);
                let anisotropy = 2.0
                    * lut_cell_center(indices.anisotropy, DIELECTRIC_GGX_LUT_ANISOTROPY_RESOLUTION)
                    - 1.0;

                *value = estimate_dielectric_ggx_directional_albedo(
                    eta,
                    wo,
                    roughness,
                    anisotropy,
                    &hemisphere_samples,
                );
            });

        Self { eta, values }
    }

    pub(crate) fn lookup(&self, w_local: Vec3, roughness: f32, anisotropy: f32) -> f32 {
        if w_local.z <= 0.0 {
            return 0.0;
        }

        let roughness = roughness.clamp(0.0, 1.0);
        let anisotropy = anisotropy.clamp(-1.0, 1.0);
        let (alpha_x, alpha_y) = alpha_xy_from_roughness(roughness, anisotropy);
        if alpha_x.max(alpha_y) < EFFECTIVELY_SMOOTH_ALPHA {
            return fresnel_dielectric(w_local.z.clamp(0.0, 1.0), 1.0, self.eta);
        }

        let sqrt_cos = w_local.z.clamp(0.0, 1.0).sqrt();
        let phi = fold_phi_to_first_quadrant(w_local) / FRAC_PI_2;
        let roughness = roughness.clamp(0.0, 1.0);
        let anisotropy = 0.5 * (anisotropy + 1.0);

        let (sqrt_cos_lower, sqrt_cos_upper, sqrt_cos_blend) =
            compute_lut_axis_lerp(sqrt_cos, DIELECTRIC_GGX_LUT_SQRT_COS_RESOLUTION);
        let (phi_lower, phi_upper, phi_blend) =
            compute_lut_axis_lerp(phi, DIELECTRIC_GGX_LUT_PHI_RESOLUTION);
        let (roughness_lower, roughness_upper, roughness_blend) =
            compute_lut_axis_lerp(roughness, DIELECTRIC_GGX_LUT_ROUGHNESS_RESOLUTION);
        let (anisotropy_lower, anisotropy_upper, anisotropy_blend) =
            compute_lut_axis_lerp(anisotropy, DIELECTRIC_GGX_LUT_ANISOTROPY_RESOLUTION);

        let mut value = 0.0;
        for (sqrt_cos_index, sqrt_cos_weight) in [
            (sqrt_cos_lower, 1.0 - sqrt_cos_blend),
            (sqrt_cos_upper, sqrt_cos_blend),
        ] {
            for (phi_index, phi_weight) in [(phi_lower, 1.0 - phi_blend), (phi_upper, phi_blend)] {
                for (roughness_index, roughness_weight) in [
                    (roughness_lower, 1.0 - roughness_blend),
                    (roughness_upper, roughness_blend),
                ] {
                    for (anisotropy_index, anisotropy_weight) in [
                        (anisotropy_lower, 1.0 - anisotropy_blend),
                        (anisotropy_upper, anisotropy_blend),
                    ] {
                        value += sqrt_cos_weight
                            * phi_weight
                            * roughness_weight
                            * anisotropy_weight
                            * self.value_at_indices(
                                sqrt_cos_index,
                                phi_index,
                                roughness_index,
                                anisotropy_index,
                            );
                    }
                }
            }
        }

        value.clamp(0.0, 1.0)
    }

    fn value_at_indices(
        &self,
        sqrt_cos_index: usize,
        phi_index: usize,
        roughness_index: usize,
        anisotropy_index: usize,
    ) -> f32 {
        self.values[DielectricGgxLutIndices {
            sqrt_cos: sqrt_cos_index,
            phi: phi_index,
            roughness: roughness_index,
            anisotropy: anisotropy_index,
        }
        .to_linear_index()]
    }

    #[cfg(test)]
    pub(crate) fn constant_for_tests(eta: f32, value: f32) -> Self {
        Self {
            eta: sanitize_dielectric_eta(eta),
            values: vec![value.clamp(0.0, 1.0); DIELECTRIC_GGX_LUT_LEN],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DielectricGgxLutIndices {
    sqrt_cos: usize,
    phi: usize,
    roughness: usize,
    anisotropy: usize,
}

impl DielectricGgxLutIndices {
    fn from_linear_index(index: usize) -> Self {
        let anisotropy = index % DIELECTRIC_GGX_LUT_ANISOTROPY_RESOLUTION;
        let index = index / DIELECTRIC_GGX_LUT_ANISOTROPY_RESOLUTION;
        let roughness = index % DIELECTRIC_GGX_LUT_ROUGHNESS_RESOLUTION;
        let index = index / DIELECTRIC_GGX_LUT_ROUGHNESS_RESOLUTION;
        let phi = index % DIELECTRIC_GGX_LUT_PHI_RESOLUTION;
        let sqrt_cos = index / DIELECTRIC_GGX_LUT_PHI_RESOLUTION;

        Self {
            sqrt_cos,
            phi,
            roughness,
            anisotropy,
        }
    }

    fn to_linear_index(self) -> usize {
        (((self.sqrt_cos * DIELECTRIC_GGX_LUT_PHI_RESOLUTION + self.phi)
            * DIELECTRIC_GGX_LUT_ROUGHNESS_RESOLUTION
            + self.roughness)
            * DIELECTRIC_GGX_LUT_ANISOTROPY_RESOLUTION)
            + self.anisotropy
    }
}

fn estimate_dielectric_ggx_directional_albedo(
    eta: f32,
    wo: Vec3,
    roughness: f32,
    anisotropy: f32,
    hemisphere_samples: &[Vec3; DIRECTIONAL_ALBEDO_SAMPLE_COUNT],
) -> f32 {
    if wo.z <= 0.0 {
        return 0.0;
    }

    let (alpha_x, alpha_y) = alpha_xy_from_roughness(roughness, anisotropy);
    if alpha_x.max(alpha_y) < EFFECTIVELY_SMOOTH_ALPHA {
        return fresnel_dielectric(wo.z.clamp(0.0, 1.0), 1.0, eta);
    }

    let bsdf = DielectricGgxBsdf::new_with_allowed_paths(
        Vec3::ONE,
        eta,
        alpha_x,
        alpha_y,
        false,
        true,
        DielectricGgxAllowedPaths::Reflection,
    );
    let sum = hemisphere_samples
        .iter()
        .map(|&wi| bsdf.eval(wo, wi).x * wi.z.max(0.0) / UNIFORM_HEMISPHERE_PDF)
        .sum::<f32>();

    (sum / DIRECTIONAL_ALBEDO_SAMPLE_COUNT as f32).clamp(0.0, 1.0)
}

fn generate_uniform_hemisphere_samples() -> [Vec3; DIRECTIONAL_ALBEDO_SAMPLE_COUNT] {
    std::array::from_fn(|index| {
        let u = (index as f32 + 0.5) / DIRECTIONAL_ALBEDO_SAMPLE_COUNT as f32;
        let v = radical_inverse_vdc(index as u32);
        sample_uniform_hemisphere(Vec2::new(u, v))
    })
}

fn sample_uniform_hemisphere(us: Vec2) -> Vec3 {
    let z = us.x.clamp(0.0, 1.0);
    let r = (1.0 - z * z).max(0.0).sqrt();
    let phi = TAU * us.y;

    Vec3::new(r * phi.cos(), r * phi.sin(), z)
}

fn radical_inverse_vdc(bits: u32) -> f32 {
    bits.reverse_bits() as f32 * 2.328_306_4e-10
}

fn alpha_xy_from_roughness(roughness: f32, anisotropy: f32) -> (f32, f32) {
    let roughness = roughness.clamp(0.0, 1.0);
    let anisotropy = anisotropy.clamp(-1.0, 1.0);
    let alpha = roughness * roughness;
    let aspect = (1.0 - 0.9 * anisotropy.abs()).sqrt();
    let (alpha_x, alpha_y) = if anisotropy >= 0.0 {
        (alpha / aspect, alpha * aspect)
    } else {
        (alpha * aspect, alpha / aspect)
    };

    (alpha_x.clamp(MIN_ALPHA, 1.0), alpha_y.clamp(MIN_ALPHA, 1.0))
}

pub(crate) fn sanitize_dielectric_eta(eta: f32) -> f32 {
    if eta.is_finite() && eta > 0.0 {
        eta
    } else {
        1.5
    }
}

fn fold_phi_to_first_quadrant(w: Vec3) -> f32 {
    if w.x == 0.0 && w.y == 0.0 {
        return 0.0;
    }

    w.y.abs().atan2(w.x.abs())
}

fn compute_lut_axis_lerp(value: f32, resolution: usize) -> (usize, usize, f32) {
    debug_assert!(resolution > 0);

    if resolution == 1 {
        return (0, 0, 0.0);
    }

    let position = value.clamp(0.0, 1.0) * resolution as f32 - 0.5;
    if position <= 0.0 {
        return (0, 0, 0.0);
    }

    let max_index = resolution - 1;
    if position >= max_index as f32 {
        return (max_index, max_index, 0.0);
    }

    let lower = position.floor() as usize;
    let upper = lower + 1;
    (lower, upper, position - lower as f32)
}

fn lut_cell_center(index: usize, resolution: usize) -> f32 {
    (index as f32 + 0.5) / resolution as f32
}

const SHEEN_LUT_COS_RESOLUTION: usize = 32;
const SHEEN_LUT_ROUGHNESS_RESOLUTION: usize = 32;
const SHEEN_LUT_LEN: usize = SHEEN_LUT_COS_RESOLUTION * SHEEN_LUT_ROUGHNESS_RESOLUTION;
const SHEEN_LUT_SAMPLE_COUNT: usize = 256;

#[derive(Debug, PartialEq)]
pub(crate) struct SheenDirectionalAlbedoLut {
    values: Vec<f32>,
}

impl SheenDirectionalAlbedoLut {
    fn build() -> Self {
        let mut values = vec![0.0; SHEEN_LUT_LEN];
        values
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, value)| {
                let r_index = index / SHEEN_LUT_COS_RESOLUTION;
                let c_index = index % SHEEN_LUT_COS_RESOLUTION;
                let cos_theta = lut_cell_center(c_index, SHEEN_LUT_COS_RESOLUTION);
                let roughness = lut_cell_center(r_index, SHEEN_LUT_ROUGHNESS_RESOLUTION);
                *value =
                    sheen_directional_albedo_estimate(roughness, cos_theta, SHEEN_LUT_SAMPLE_COUNT);
            });
        Self { values }
    }

    pub(crate) fn lookup(&self, cos_theta: f32, roughness: f32) -> f32 {
        let cos_theta = cos_theta.clamp(0.0, 1.0);
        let roughness = roughness.clamp(0.0, 1.0);
        let (cos_lower, cos_upper, cos_blend) =
            compute_lut_axis_lerp(cos_theta, SHEEN_LUT_COS_RESOLUTION);
        let (r_lower, r_upper, r_blend) =
            compute_lut_axis_lerp(roughness, SHEEN_LUT_ROUGHNESS_RESOLUTION);
        let mut value = 0.0;
        for (cos_idx, cos_w) in [(cos_lower, 1.0 - cos_blend), (cos_upper, cos_blend)] {
            for (r_idx, r_w) in [(r_lower, 1.0 - r_blend), (r_upper, r_blend)] {
                value += cos_w * r_w * self.values[r_idx * SHEEN_LUT_COS_RESOLUTION + cos_idx];
            }
        }
        value.clamp(0.0, 1.0)
    }

    #[cfg(test)]
    pub(crate) fn constant_for_tests(value: f32) -> Self {
        Self {
            values: vec![value.clamp(0.0, 1.0); SHEEN_LUT_LEN],
        }
    }
}
