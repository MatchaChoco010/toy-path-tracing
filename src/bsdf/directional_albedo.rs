use std::{
    collections::HashMap,
    f32::consts::{FRAC_PI_2, PI, TAU},
    sync::Arc,
};

use glam::{Vec2, Vec3};
use rayon::prelude::*;

use crate::math::fresnel_dielectric;

use super::dielectric_ggx::{DielectricGgxAllowedPaths, DielectricGgxBsdf};
use super::mtlx::{DielectricBsdf as MtlxDielectricBsdf, ScatterMode as MtlxScatterMode};
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
    mtlx_dielectric_ggx: Option<Arc<MtlxDielectricGgxDirectionalAlbedoLut>>,
    mtlx_generalized_schlick_ggx: Option<Arc<MtlxGeneralizedSchlickGgxDirectionalAlbedoLut>>,
    conductor_ggx_energy_compensation: Option<Arc<ConductorGgxEnergyCompensationLut>>,
    dielectric_ggx_energy_compensation: Option<Arc<DielectricGgxEnergyCompensationLut>>,
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

    pub(crate) fn get_or_build_mtlx_dielectric_ggx(
        &mut self,
    ) -> Arc<MtlxDielectricGgxDirectionalAlbedoLut> {
        if let Some(lut) = self.mtlx_dielectric_ggx.as_ref() {
            return Arc::clone(lut);
        }
        let lut = Arc::new(MtlxDielectricGgxDirectionalAlbedoLut::build());
        self.mtlx_dielectric_ggx = Some(Arc::clone(&lut));
        lut
    }

    pub(crate) fn get_or_build_mtlx_generalized_schlick_ggx(
        &mut self,
    ) -> Arc<MtlxGeneralizedSchlickGgxDirectionalAlbedoLut> {
        if let Some(lut) = self.mtlx_generalized_schlick_ggx.as_ref() {
            return Arc::clone(lut);
        }
        let lut = Arc::new(MtlxGeneralizedSchlickGgxDirectionalAlbedoLut::build());
        self.mtlx_generalized_schlick_ggx = Some(Arc::clone(&lut));
        lut
    }

    pub(crate) fn get_or_build_conductor_ggx_energy_compensation(
        &mut self,
    ) -> Arc<ConductorGgxEnergyCompensationLut> {
        if let Some(lut) = self.conductor_ggx_energy_compensation.as_ref() {
            return Arc::clone(lut);
        }
        let lut = Arc::new(ConductorGgxEnergyCompensationLut::build());
        self.conductor_ggx_energy_compensation = Some(Arc::clone(&lut));
        lut
    }

    pub(crate) fn get_or_build_dielectric_ggx_energy_compensation(
        &mut self,
    ) -> Arc<DielectricGgxEnergyCompensationLut> {
        if let Some(lut) = self.dielectric_ggx_energy_compensation.as_ref() {
            return Arc::clone(lut);
        }
        let lut = Arc::new(DielectricGgxEnergyCompensationLut::build());
        self.dielectric_ggx_energy_compensation = Some(Arc::clone(&lut));
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
    pub(crate) fn build_for_tests(eta: f32) -> Self {
        Self::build(eta)
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

fn radical_inverse_base3(mut n: u32) -> f32 {
    let mut q = 0.0_f32;
    let mut bk = 1.0_f32 / 3.0;
    while n > 0 {
        q += (n % 3) as f32 * bk;
        n /= 3;
        bk /= 3.0;
    }
    q
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

const MTLX_DIELECTRIC_DA_COS_RESOLUTION: usize = 64;
const MTLX_DIELECTRIC_DA_ALPHA_RESOLUTION: usize = 64;
const MTLX_DIELECTRIC_DA_ETA_RESOLUTION: usize = 32;
const MTLX_DIELECTRIC_DA_LEN: usize = MTLX_DIELECTRIC_DA_COS_RESOLUTION
    * MTLX_DIELECTRIC_DA_ALPHA_RESOLUTION
    * MTLX_DIELECTRIC_DA_ETA_RESOLUTION;
const MTLX_DIELECTRIC_DA_SAMPLE_COUNT: usize = 64;

#[derive(Debug, PartialEq)]
pub(crate) struct MtlxDielectricGgxDirectionalAlbedoLut {
    values: Vec<f32>,
}

impl MtlxDielectricGgxDirectionalAlbedoLut {
    fn build() -> Self {
        let hemisphere_samples = generate_uniform_hemisphere_samples();
        let mut values = vec![0.0; MTLX_DIELECTRIC_DA_LEN];
        values
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, value)| {
                let cos_index = index % MTLX_DIELECTRIC_DA_COS_RESOLUTION;
                let rest = index / MTLX_DIELECTRIC_DA_COS_RESOLUTION;
                let alpha_index = rest % MTLX_DIELECTRIC_DA_ALPHA_RESOLUTION;
                let eta_index = rest / MTLX_DIELECTRIC_DA_ALPHA_RESOLUTION;

                let sqrt_cos = lut_cell_center(cos_index, MTLX_DIELECTRIC_DA_COS_RESOLUTION);
                let cos_theta = sqrt_cos * sqrt_cos;
                let alpha = lut_cell_center(alpha_index, MTLX_DIELECTRIC_DA_ALPHA_RESOLUTION);
                let eta_axis = lut_cell_center(eta_index, MTLX_DIELECTRIC_DA_ETA_RESOLUTION);
                let eta_rel = dielectric_ec_axis_to_eta(eta_axis);

                *value = estimate_mtlx_dielectric_ggx_directional_albedo(
                    cos_theta,
                    alpha,
                    eta_rel,
                    &hemisphere_samples,
                );
            });
        Self { values }
    }

    pub(crate) fn lookup(&self, cos_theta: f32, alpha_x: f32, alpha_y: f32, eta_rel: f32) -> f32 {
        if cos_theta <= 0.0 {
            return 0.0;
        }

        let sqrt_cos = cos_theta.clamp(0.0, 1.0).sqrt();
        let alpha_x = alpha_x.clamp(MIN_ALPHA, 1.0);
        let alpha_y = alpha_y.clamp(MIN_ALPHA, 1.0);
        let alpha = (alpha_x * alpha_y).sqrt().clamp(MIN_ALPHA, 1.0);
        let eta_axis = dielectric_ec_eta_to_axis(eta_rel);

        let (cos_lower, cos_upper, cos_blend) =
            compute_lut_axis_lerp(sqrt_cos, MTLX_DIELECTRIC_DA_COS_RESOLUTION);
        let (alpha_lower, alpha_upper, alpha_blend) =
            compute_lut_axis_lerp(alpha, MTLX_DIELECTRIC_DA_ALPHA_RESOLUTION);
        let (eta_lower, eta_upper, eta_blend) =
            compute_lut_axis_lerp(eta_axis, MTLX_DIELECTRIC_DA_ETA_RESOLUTION);

        let mut value = 0.0;
        for (eta_idx, eta_w) in [(eta_lower, 1.0 - eta_blend), (eta_upper, eta_blend)] {
            for (alpha_idx, alpha_w) in
                [(alpha_lower, 1.0 - alpha_blend), (alpha_upper, alpha_blend)]
            {
                for (cos_idx, cos_w) in [(cos_lower, 1.0 - cos_blend), (cos_upper, cos_blend)] {
                    let idx = (eta_idx * MTLX_DIELECTRIC_DA_ALPHA_RESOLUTION + alpha_idx)
                        * MTLX_DIELECTRIC_DA_COS_RESOLUTION
                        + cos_idx;
                    value += eta_w * alpha_w * cos_w * self.values[idx];
                }
            }
        }
        value.clamp(0.0, 1.0)
    }
}

fn estimate_mtlx_dielectric_ggx_directional_albedo(
    cos_theta: f32,
    alpha: f32,
    eta_rel: f32,
    hemisphere_samples: &[Vec3; DIRECTIONAL_ALBEDO_SAMPLE_COUNT],
) -> f32 {
    if cos_theta <= 0.0 || eta_rel <= 0.0 {
        return 0.0;
    }
    let alpha = alpha.clamp(MIN_ALPHA, 1.0);
    let eta_rel = eta_rel.clamp(DIELECTRIC_EC_ETA_MIN_INVERSE, DIELECTRIC_EC_ETA_MAX);
    let ior = if eta_rel >= 1.0 {
        eta_rel
    } else {
        1.0 / eta_rel
    };
    let front_face = eta_rel >= 1.0;
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let wo = Vec3::new(sin_theta, 0.0, cos_theta);
    let bsdf = MtlxDielectricBsdf::with_thin_film(
        1.0,
        Vec3::ONE,
        ior,
        Vec2::splat(alpha),
        MtlxScatterMode::Reflection,
        0.0,
        1.5,
        front_face,
    );
    let sum = hemisphere_samples
        .iter()
        .map(|&wi| bsdf.eval(wo, wi).x * wi.z.max(0.0) / UNIFORM_HEMISPHERE_PDF)
        .sum::<f32>();

    (sum / MTLX_DIELECTRIC_DA_SAMPLE_COUNT as f32).clamp(0.0, 1.0)
}

const MTLX_GENERALIZED_SCHLICK_DA_COS_RESOLUTION: usize = 64;
const MTLX_GENERALIZED_SCHLICK_DA_ALPHA_RESOLUTION: usize = 64;
const MTLX_GENERALIZED_SCHLICK_DA_LEN: usize =
    MTLX_GENERALIZED_SCHLICK_DA_COS_RESOLUTION * MTLX_GENERALIZED_SCHLICK_DA_ALPHA_RESOLUTION;
const MTLX_GENERALIZED_SCHLICK_DA_SAMPLE_COUNT: usize = 64;

#[derive(Debug, PartialEq)]
pub(crate) struct MtlxGeneralizedSchlickGgxDirectionalAlbedoLut {
    ab_values: Vec<Vec2>,
}

impl MtlxGeneralizedSchlickGgxDirectionalAlbedoLut {
    fn build() -> Self {
        let samples = generate_stratified_unit_square_samples_mtlx_schlick();
        let mut ab_values = vec![Vec2::ZERO; MTLX_GENERALIZED_SCHLICK_DA_LEN];
        ab_values
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, value)| {
                let cos_index = index % MTLX_GENERALIZED_SCHLICK_DA_COS_RESOLUTION;
                let alpha_index = index / MTLX_GENERALIZED_SCHLICK_DA_COS_RESOLUTION;
                let sqrt_cos =
                    lut_cell_center(cos_index, MTLX_GENERALIZED_SCHLICK_DA_COS_RESOLUTION);
                let cos_theta = sqrt_cos * sqrt_cos;
                let alpha =
                    lut_cell_center(alpha_index, MTLX_GENERALIZED_SCHLICK_DA_ALPHA_RESOLUTION);
                *value = estimate_mtlx_generalized_schlick_ggx_ab(cos_theta, alpha, &samples);
            });
        Self { ab_values }
    }

    pub(crate) fn lookup(
        &self,
        cos_theta: f32,
        alpha_x: f32,
        alpha_y: f32,
        color0: Vec3,
        color90: Vec3,
    ) -> Vec3 {
        if cos_theta <= 0.0 {
            return Vec3::ZERO;
        }
        let sqrt_cos = cos_theta.clamp(0.0, 1.0).sqrt();
        let alpha = (alpha_x.clamp(MIN_ALPHA, 1.0) * alpha_y.clamp(MIN_ALPHA, 1.0))
            .sqrt()
            .clamp(MIN_ALPHA, 1.0);
        let (cos_lower, cos_upper, cos_blend) =
            compute_lut_axis_lerp(sqrt_cos, MTLX_GENERALIZED_SCHLICK_DA_COS_RESOLUTION);
        let (alpha_lower, alpha_upper, alpha_blend) =
            compute_lut_axis_lerp(alpha, MTLX_GENERALIZED_SCHLICK_DA_ALPHA_RESOLUTION);

        let mut ab = Vec2::ZERO;
        for (alpha_idx, alpha_w) in [(alpha_lower, 1.0 - alpha_blend), (alpha_upper, alpha_blend)] {
            for (cos_idx, cos_w) in [(cos_lower, 1.0 - cos_blend), (cos_upper, cos_blend)] {
                let idx = alpha_idx * MTLX_GENERALIZED_SCHLICK_DA_COS_RESOLUTION + cos_idx;
                ab += self.ab_values[idx] * (alpha_w * cos_w);
            }
        }

        (color0.max(Vec3::ZERO) * ab.x + color90.max(Vec3::ZERO) * ab.y)
            .clamp(Vec3::ZERO, Vec3::ONE)
    }
}

fn estimate_mtlx_generalized_schlick_ggx_ab(
    cos_theta: f32,
    alpha: f32,
    samples: &[Vec2; MTLX_GENERALIZED_SCHLICK_DA_SAMPLE_COUNT],
) -> Vec2 {
    if cos_theta <= 0.0 {
        return Vec2::ZERO;
    }
    let alpha = alpha.clamp(MIN_ALPHA, 1.0);
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let wo = Vec3::new(sin_theta, 0.0, cos_theta);
    let g1_wo = crate::bsdf::smith_ggx::ggx_g1(wo, alpha, alpha);
    if g1_wo <= 0.0 {
        return Vec2::ZERO;
    }

    let sum = samples.iter().fold(Vec2::ZERO, |acc, &us| {
        let Some(wm) = crate::bsdf::smith_ggx::sample_wm_vndf(wo, alpha, alpha, us) else {
            return acc;
        };
        let v_dot_h = wo.dot(wm).clamp(0.0, 1.0);
        if v_dot_h <= 0.0 {
            return acc;
        }
        let wi = crate::bsdf::smith_ggx::reflect_local(wo, wm);
        if wi.z <= 0.0 {
            return acc;
        }
        let fc = (1.0 - v_dot_h).powi(5);
        let g2 = crate::bsdf::smith_ggx::ggx_g2_height_correlated(wo, wi, alpha, alpha);
        acc + Vec2::new(g2 * (1.0 - fc), g2 * fc)
    });

    (sum / (g1_wo * MTLX_GENERALIZED_SCHLICK_DA_SAMPLE_COUNT as f32)).clamp(Vec2::ZERO, Vec2::ONE)
}

fn generate_stratified_unit_square_samples_mtlx_schlick()
-> [Vec2; MTLX_GENERALIZED_SCHLICK_DA_SAMPLE_COUNT] {
    std::array::from_fn(|index| {
        let u = (index as f32 + 0.5) / MTLX_GENERALIZED_SCHLICK_DA_SAMPLE_COUNT as f32;
        let v = radical_inverse_vdc(index as u32);
        Vec2::new(u, v)
    })
}

const CONDUCTOR_EC_COS_RESOLUTION: usize = 64;
const CONDUCTOR_EC_ROUGHNESS_RESOLUTION: usize = 64;
const CONDUCTOR_EC_LEN: usize = CONDUCTOR_EC_COS_RESOLUTION * CONDUCTOR_EC_ROUGHNESS_RESOLUTION;
const CONDUCTOR_EC_SAMPLE_COUNT: usize = 256;

#[derive(Debug, PartialEq)]
pub(crate) struct ConductorGgxEnergyCompensationLut {
    e_values: Vec<f32>,
    e_avg_values: Vec<f32>,
}

impl ConductorGgxEnergyCompensationLut {
    fn build() -> Self {
        let stratum_samples = generate_stratified_unit_square_samples_ec();
        let mut e_values = vec![0.0; CONDUCTOR_EC_LEN];

        e_values
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, value)| {
                let cos_index = index % CONDUCTOR_EC_COS_RESOLUTION;
                let r_index = index / CONDUCTOR_EC_COS_RESOLUTION;
                let sqrt_cos = lut_cell_center(cos_index, CONDUCTOR_EC_COS_RESOLUTION);
                let cos_theta = sqrt_cos * sqrt_cos;
                let roughness = lut_cell_center(r_index, CONDUCTOR_EC_ROUGHNESS_RESOLUTION);
                *value = estimate_conductor_ggx_directional_albedo_white_fresnel(
                    cos_theta,
                    roughness,
                    &stratum_samples,
                );
            });

        let mut e_avg_values = vec![0.0; CONDUCTOR_EC_ROUGHNESS_RESOLUTION];
        for r_index in 0..CONDUCTOR_EC_ROUGHNESS_RESOLUTION {
            let mut weighted = 0.0;
            for cos_index in 0..CONDUCTOR_EC_COS_RESOLUTION {
                let sqrt_cos = lut_cell_center(cos_index, CONDUCTOR_EC_COS_RESOLUTION);
                let cos_theta = sqrt_cos * sqrt_cos;
                let e = e_values[r_index * CONDUCTOR_EC_COS_RESOLUTION + cos_index];
                weighted += e * cos_theta * sqrt_cos;
            }
            e_avg_values[r_index] =
                (4.0 / CONDUCTOR_EC_COS_RESOLUTION as f32 * weighted).clamp(0.0, 1.0);
        }

        Self {
            e_values,
            e_avg_values,
        }
    }

    pub(crate) fn lookup_e(&self, cos_theta: f32, roughness: f32) -> f32 {
        let cos_theta = cos_theta.clamp(0.0, 1.0);
        let roughness = roughness.clamp(0.0, 1.0);
        let sqrt_cos = cos_theta.sqrt();

        let (cos_lower, cos_upper, cos_blend) =
            compute_lut_axis_lerp(sqrt_cos, CONDUCTOR_EC_COS_RESOLUTION);
        let (r_lower, r_upper, r_blend) =
            compute_lut_axis_lerp(roughness, CONDUCTOR_EC_ROUGHNESS_RESOLUTION);

        let mut value = 0.0;
        for (cos_idx, cos_w) in [(cos_lower, 1.0 - cos_blend), (cos_upper, cos_blend)] {
            for (r_idx, r_w) in [(r_lower, 1.0 - r_blend), (r_upper, r_blend)] {
                value += cos_w * r_w * self.e_values[r_idx * CONDUCTOR_EC_COS_RESOLUTION + cos_idx];
            }
        }
        value.clamp(0.0, 1.0)
    }

    pub(crate) fn lookup_e_avg(&self, roughness: f32) -> f32 {
        let roughness = roughness.clamp(0.0, 1.0);
        let (lower, upper, blend) =
            compute_lut_axis_lerp(roughness, CONDUCTOR_EC_ROUGHNESS_RESOLUTION);
        let v = (1.0 - blend) * self.e_avg_values[lower] + blend * self.e_avg_values[upper];
        v.clamp(0.0, 1.0)
    }

    #[cfg(test)]
    pub(crate) fn build_for_tests() -> Self {
        Self::build()
    }

    #[cfg(test)]
    pub(crate) fn constant_for_tests(e: f32, e_avg: f32) -> Self {
        Self {
            e_values: vec![e.clamp(0.0, 1.0); CONDUCTOR_EC_LEN],
            e_avg_values: vec![e_avg.clamp(0.0, 1.0); CONDUCTOR_EC_ROUGHNESS_RESOLUTION],
        }
    }
}

fn estimate_conductor_ggx_directional_albedo_white_fresnel(
    cos_theta: f32,
    roughness: f32,
    stratum_samples: &[Vec2; CONDUCTOR_EC_SAMPLE_COUNT],
) -> f32 {
    if cos_theta <= 0.0 {
        return 0.0;
    }

    let alpha = (roughness * roughness).clamp(MIN_ALPHA, 1.0);
    if alpha < EFFECTIVELY_SMOOTH_ALPHA {
        return 1.0;
    }

    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let wo = Vec3::new(sin_theta, 0.0, cos_theta);
    let g1_wo = crate::bsdf::smith_ggx::ggx_g1(wo, alpha, alpha);
    if g1_wo <= 0.0 {
        return 0.0;
    }

    let sum: f32 = stratum_samples
        .iter()
        .map(|&us| {
            let Some(wm) = crate::bsdf::smith_ggx::sample_wm_vndf(wo, alpha, alpha, us) else {
                return 0.0;
            };
            let cos_wo_wm = wo.dot(wm);
            if cos_wo_wm <= 0.0 {
                return 0.0;
            }
            let wi = crate::bsdf::smith_ggx::reflect_local(wo, wm);
            if wi.z <= 0.0 {
                return 0.0;
            }
            let g2 = crate::bsdf::smith_ggx::ggx_g2_height_correlated(wo, wi, alpha, alpha);
            g2 / g1_wo
        })
        .sum();

    (sum / CONDUCTOR_EC_SAMPLE_COUNT as f32).clamp(0.0, 1.0)
}

fn generate_stratified_unit_square_samples_ec() -> [Vec2; CONDUCTOR_EC_SAMPLE_COUNT] {
    std::array::from_fn(|index| {
        let u = (index as f32 + 0.5) / CONDUCTOR_EC_SAMPLE_COUNT as f32;
        let v = radical_inverse_vdc(index as u32);
        Vec2::new(u, v)
    })
}

const DIELECTRIC_EC_COS_RESOLUTION: usize = 32;
const DIELECTRIC_EC_ROUGHNESS_RESOLUTION: usize = 32;
const DIELECTRIC_EC_ETA_RESOLUTION: usize = 32;
const DIELECTRIC_EC_E_LEN: usize = DIELECTRIC_EC_COS_RESOLUTION
    * DIELECTRIC_EC_ROUGHNESS_RESOLUTION
    * DIELECTRIC_EC_ETA_RESOLUTION;
const DIELECTRIC_EC_EAVG_LEN: usize =
    DIELECTRIC_EC_ROUGHNESS_RESOLUTION * DIELECTRIC_EC_ETA_RESOLUTION;
const DIELECTRIC_EC_BSDF_SAMPLE_COUNT: usize = 4096;

const DIELECTRIC_EC_ETA_MIN_INVERSE: f32 = 1.0 / 3.0;
const DIELECTRIC_EC_ETA_MAX: f32 = 3.0;

fn dielectric_ec_eta_to_axis(eta: f32) -> f32 {
    let eta = eta.clamp(DIELECTRIC_EC_ETA_MIN_INVERSE, DIELECTRIC_EC_ETA_MAX);
    if eta >= 1.0 {
        0.5 + 0.5 * ((eta - 1.0) / (DIELECTRIC_EC_ETA_MAX - 1.0))
    } else {
        let inv = 1.0 / eta;
        0.5 - 0.5 * ((inv - 1.0) / (DIELECTRIC_EC_ETA_MAX - 1.0))
    }
}

fn dielectric_ec_axis_to_eta(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    if t >= 0.5 {
        1.0 + (t - 0.5) * 2.0 * (DIELECTRIC_EC_ETA_MAX - 1.0)
    } else {
        let inv = 1.0 + (0.5 - t) * 2.0 * (DIELECTRIC_EC_ETA_MAX - 1.0);
        1.0 / inv
    }
}

#[derive(Debug, PartialEq)]
pub(crate) struct DielectricGgxEnergyCompensationLut {
    e_values: Vec<f32>,
    e_avg_values: Vec<f32>,
}

impl DielectricGgxEnergyCompensationLut {
    fn build() -> Self {
        let mut e_values = vec![0.0; DIELECTRIC_EC_E_LEN];

        e_values
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, value)| {
                let cos_index = index % DIELECTRIC_EC_COS_RESOLUTION;
                let rest = index / DIELECTRIC_EC_COS_RESOLUTION;
                let r_index = rest % DIELECTRIC_EC_ROUGHNESS_RESOLUTION;
                let eta_index = rest / DIELECTRIC_EC_ROUGHNESS_RESOLUTION;
                let sqrt_cos = lut_cell_center(cos_index, DIELECTRIC_EC_COS_RESOLUTION);
                let cos_theta = sqrt_cos * sqrt_cos;
                let roughness = lut_cell_center(r_index, DIELECTRIC_EC_ROUGHNESS_RESOLUTION);
                let eta_axis = lut_cell_center(eta_index, DIELECTRIC_EC_ETA_RESOLUTION);
                let eta = dielectric_ec_axis_to_eta(eta_axis);
                *value = estimate_dielectric_ggx_directional_albedo_full_sphere(
                    cos_theta, roughness, eta,
                );
            });

        let mut e_avg_values = vec![0.0; DIELECTRIC_EC_EAVG_LEN];
        for eta_index in 0..DIELECTRIC_EC_ETA_RESOLUTION {
            for r_index in 0..DIELECTRIC_EC_ROUGHNESS_RESOLUTION {
                let mut weighted = 0.0;
                for cos_index in 0..DIELECTRIC_EC_COS_RESOLUTION {
                    let sqrt_cos = lut_cell_center(cos_index, DIELECTRIC_EC_COS_RESOLUTION);
                    let cos_theta = sqrt_cos * sqrt_cos;
                    let e_index = (eta_index * DIELECTRIC_EC_ROUGHNESS_RESOLUTION + r_index)
                        * DIELECTRIC_EC_COS_RESOLUTION
                        + cos_index;
                    let e = e_values[e_index];
                    weighted += e * cos_theta * sqrt_cos;
                }
                let v = 4.0 / DIELECTRIC_EC_COS_RESOLUTION as f32 * weighted;
                e_avg_values[eta_index * DIELECTRIC_EC_ROUGHNESS_RESOLUTION + r_index] =
                    v.clamp(0.0, 1.0);
            }
        }

        Self {
            e_values,
            e_avg_values,
        }
    }

    pub(crate) fn lookup_e(&self, cos_theta: f32, roughness: f32, eta: f32) -> f32 {
        let cos_theta = cos_theta.clamp(0.0, 1.0);
        let roughness = roughness.clamp(0.0, 1.0);
        let sqrt_cos = cos_theta.sqrt();
        let eta_axis = dielectric_ec_eta_to_axis(eta);

        let (cos_lower, cos_upper, cos_blend) =
            compute_lut_axis_lerp(sqrt_cos, DIELECTRIC_EC_COS_RESOLUTION);
        let (r_lower, r_upper, r_blend) =
            compute_lut_axis_lerp(roughness, DIELECTRIC_EC_ROUGHNESS_RESOLUTION);
        let (e_lower, e_upper, e_blend) =
            compute_lut_axis_lerp(eta_axis, DIELECTRIC_EC_ETA_RESOLUTION);

        let mut value = 0.0;
        for (eta_idx, eta_w) in [(e_lower, 1.0 - e_blend), (e_upper, e_blend)] {
            for (r_idx, r_w) in [(r_lower, 1.0 - r_blend), (r_upper, r_blend)] {
                for (cos_idx, cos_w) in [(cos_lower, 1.0 - cos_blend), (cos_upper, cos_blend)] {
                    let idx = (eta_idx * DIELECTRIC_EC_ROUGHNESS_RESOLUTION + r_idx)
                        * DIELECTRIC_EC_COS_RESOLUTION
                        + cos_idx;
                    value += eta_w * r_w * cos_w * self.e_values[idx];
                }
            }
        }
        value.clamp(0.0, 1.0)
    }

    pub(crate) fn lookup_e_avg(&self, roughness: f32, eta: f32) -> f32 {
        let roughness = roughness.clamp(0.0, 1.0);
        let eta_axis = dielectric_ec_eta_to_axis(eta);
        let (r_lower, r_upper, r_blend) =
            compute_lut_axis_lerp(roughness, DIELECTRIC_EC_ROUGHNESS_RESOLUTION);
        let (e_lower, e_upper, e_blend) =
            compute_lut_axis_lerp(eta_axis, DIELECTRIC_EC_ETA_RESOLUTION);

        let mut value = 0.0;
        for (eta_idx, eta_w) in [(e_lower, 1.0 - e_blend), (e_upper, e_blend)] {
            for (r_idx, r_w) in [(r_lower, 1.0 - r_blend), (r_upper, r_blend)] {
                let idx = eta_idx * DIELECTRIC_EC_ROUGHNESS_RESOLUTION + r_idx;
                value += eta_w * r_w * self.e_avg_values[idx];
            }
        }
        value.clamp(0.0, 1.0)
    }

    #[cfg(test)]
    pub(crate) fn build_for_tests() -> Self {
        Self::build()
    }

    #[cfg(test)]
    pub(crate) fn constant_for_tests(e: f32, e_avg: f32) -> Self {
        Self {
            e_values: vec![e.clamp(0.0, 1.0); DIELECTRIC_EC_E_LEN],
            e_avg_values: vec![e_avg.clamp(0.0, 1.0); DIELECTRIC_EC_EAVG_LEN],
        }
    }
}

fn estimate_dielectric_ggx_directional_albedo_full_sphere(
    cos_theta: f32,
    roughness: f32,
    eta: f32,
) -> f32 {
    if cos_theta <= 0.0 || eta <= 0.0 {
        return 0.0;
    }

    let alpha = (roughness * roughness).clamp(MIN_ALPHA, 1.0);
    if alpha < EFFECTIVELY_SMOOTH_ALPHA {
        return 1.0;
    }

    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let wo = Vec3::new(sin_theta, 0.0, cos_theta);
    let inv_eta_sq = 1.0 / (eta * eta);
    let bsdf = super::dielectric_ggx::DielectricGgxBsdf::new_with_allowed_paths(
        Vec3::ONE,
        eta,
        alpha,
        alpha,
        false,
        true,
        super::dielectric_ggx::DielectricGgxAllowedPaths::ReflectionAndTransmission,
    );

    let mut sum = 0.0_f32;
    for index in 0..DIELECTRIC_EC_BSDF_SAMPLE_COUNT {
        let uc = (index as f32 + 0.5) / DIELECTRIC_EC_BSDF_SAMPLE_COUNT as f32;
        let u = radical_inverse_vdc(index as u32 + 1);
        let v = radical_inverse_base3(index as u32 + 1);
        let us = Vec2::new(u, v);
        let Some(sample) = bsdf.sample(wo, uc, us) else {
            continue;
        };
        if !sample.weight.x.is_finite() {
            continue;
        }
        let scale_irrad = if sample.flags.contains(super::BsdfFlags::TRANSMISSION) {
            inv_eta_sq
        } else {
            1.0
        };
        sum += sample.weight.x * scale_irrad;
    }
    (sum / DIELECTRIC_EC_BSDF_SAMPLE_COUNT as f32).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conductor_energy_compensation_lookup_returns_cell_center_values() {
        let lut = ConductorGgxEnergyCompensationLut::build_for_tests();
        for (cos_index, r_index) in [(0, 0), (7, 5), (31, 23), (63, 63)] {
            let sqrt_cos = lut_cell_center(cos_index, CONDUCTOR_EC_COS_RESOLUTION);
            let cos_theta = sqrt_cos * sqrt_cos;
            let roughness = lut_cell_center(r_index, CONDUCTOR_EC_ROUGHNESS_RESOLUTION);
            let expected = lut.e_values[r_index * CONDUCTOR_EC_COS_RESOLUTION + cos_index];
            let actual = lut.lookup_e(cos_theta, roughness);
            assert!(
                (actual - expected).abs() < 1.0e-5,
                "lookup mismatch at cos_index={cos_index}, r_index={r_index}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn conductor_energy_compensation_lookup_stays_in_bounds() {
        let lut = ConductorGgxEnergyCompensationLut::build_for_tests();
        for cos_theta in [0.0, 0.1, 0.5, 0.9, 1.0] {
            for roughness in [0.0, 0.05, 0.25, 0.75, 1.0] {
                let e = lut.lookup_e(cos_theta, roughness);
                assert!((0.0..=1.0).contains(&e));
            }
        }
    }
}
