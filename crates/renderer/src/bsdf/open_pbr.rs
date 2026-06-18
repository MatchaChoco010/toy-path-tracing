use std::{f32::consts::PI, sync::Arc};

use glam::{Vec2, Vec3};

use crate::{
    math::{
        OrthonormalBasis, cosine_weighted_hemisphere_pdf, fresnel_dielectric, refract,
        sample_cosine_weighted_hemisphere, sg::luminance,
    },
    sampler::MaterialSampleRandoms,
};

use super::dispersion::{cauchy_ior, sample_dispersion_wavelength_weighted};
use super::eon::EonBsdf;
use super::smith_ggx::{
    EFFECTIVELY_SMOOTH_ALPHA, MIN_ALPHA, ggx_d, ggx_g2_height_correlated, is_upper_hemisphere,
    pdf_wm_bounded_vndf, pdf_wm_vndf, reflect_local, reflection_half_vector,
    sample_wm_bounded_vndf, sample_wm_vndf,
};
use super::thin_film::{eval_thin_film_conductor, eval_thin_film_dielectric};
use super::{
    BsdfFlags, BsdfSample, ConductorGgxEnergyCompensationLut, DielectricGgxDirectionalAlbedoLut,
    DielectricGgxEnergyCompensationLut,
};

const COS_82: f32 = 1.0 / 7.0;

#[derive(Debug, Clone)]
pub struct OpenPbrBsdfParams {
    pub base_color: Vec3,
    pub base: f32,
    pub specular: f32,
    pub specular_color: Vec3,
    pub specular_alpha_x: f32,
    pub specular_alpha_y: f32,
    pub specular_eta: f32,
    pub transmission_eta: f32,
    pub metalness: f32,
    pub metal_n: Vec3,
    pub metal_k: Vec3,
    pub coat: f32,
    pub coat_color: Vec3,
    pub coat_darkening: Vec3,
    pub coat_alpha_x: f32,
    pub coat_alpha_y: f32,
    pub coat_eta: f32,
    pub fuzz: f32,
    pub fuzz_color: Vec3,
    pub fuzz_roughness: f32,
    pub transmission: f32,
    pub transmission_color: Vec3,
    pub transmission_alpha_x: f32,
    pub transmission_alpha_y: f32,
    pub transmission_dispersion_abbe: f32,
    pub diffuse_roughness: f32,
    pub subsurface: f32,
    pub subsurface_color: Vec3,
    pub thin_walled: bool,
    pub thin_film_weight: f32,
    pub thin_film_thickness: f32,
    pub thin_film_ior: f32,
    pub front_face: bool,
    pub coat_basis_in_base: Option<OrthonormalBasis>,
    pub fuzz_basis_in_base: Option<OrthonormalBasis>,
    pub path_throughput: Vec3,
    pub wavelength_lock: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct OpenPbrBsdf {
    p: OpenPbrBsdfParams,
    spec_lut: Arc<DielectricGgxDirectionalAlbedoLut>,
    coat_lut: Arc<DielectricGgxDirectionalAlbedoLut>,
    conductor_ec_lut: Arc<ConductorGgxEnergyCompensationLut>,
    dielectric_ec_lut: Arc<DielectricGgxEnergyCompensationLut>,
}

#[derive(Debug, Clone, Copy)]
struct LayerWeights {
    coat_amp: f32,
    metal: Vec3,
    spec_brdf: Vec3,
    spec_btdf: Vec3,
    fuzz: Vec3,
    diff_brdf: Vec3,
    diff_btdf: Vec3,
}

#[derive(Debug, Clone, Copy)]
struct LobeProbs {
    coat: f32,
    metal: f32,
    spec_brdf: f32,
    spec_btdf: f32,
    fuzz: f32,
    diff_brdf: f32,
    diff_btdf: f32,
    total: f32,
}

#[derive(Debug, Clone, Copy)]
struct DielectricMsParams {
    eta_o: f32,
    eta_rel: f32,
    roughness_eq: f32,
    e_avg_o: f32,
    ratio_r: f32,
    one_minus_e_avg_o: f32,
    one_minus_e_avg_t: f32,
}

#[derive(Debug, Clone, Copy)]
struct TwoLobeWeights {
    ss: f32,
    ms: f32,
    total: f32,
}

#[derive(Debug, Clone, Copy)]
enum ChosenLobe {
    Coat,
    Metal,
    SpecBrdf,
    SpecBtdf,
    Fuzz,
    DiffBrdf,
    DiffBtdf,
}

#[derive(Debug, Clone, Copy)]
struct SpecBtdfEval {
    scalar: f32,
    cos_wo_wm: f32,
}

#[derive(Debug, Clone, Copy)]
struct DispersionChannel {
    lambda_nm: f32,
    color: Vec3,
    probability: f32,
}

#[derive(Debug, Clone, Copy)]
struct OpenPbrFuzzBsdf {
    roughness: f32,
}

impl OpenPbrFuzzBsdf {
    fn new(roughness: f32) -> Self {
        Self {
            roughness: roughness.clamp(0.01, 1.0),
        }
    }

    fn eval(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        if wo.z <= 0.0 || wi.z <= 0.0 {
            return Vec3::ZERO;
        }
        let phi_wo = phi_of(wo);
        let wi_std = rotate_about_z(wi, -phi_wo);
        let a_inv = zeltner_ltc_a_inv(wo.z, self.roughness);
        let b_inv = zeltner_ltc_b_inv(wo.z, self.roughness);
        let r = zeltner_dir_albedo(wo.z, self.roughness);
        let value = eval_ltc(wi_std, a_inv, b_inv);
        let cos_i = wi.z.max(1.0e-6);
        Vec3::splat(r * value / cos_i)
    }

    fn pdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        if wo.z <= 0.0 || wi.z <= 0.0 {
            return 0.0;
        }
        let phi_wo = phi_of(wo);
        let wi_std = rotate_about_z(wi, -phi_wo);
        let a_inv = zeltner_ltc_a_inv(wo.z, self.roughness);
        let b_inv = zeltner_ltc_b_inv(wo.z, self.roughness);
        eval_ltc(wi_std, a_inv, b_inv)
    }

    fn sample(&self, wo: Vec3, us: Vec2) -> Option<Vec3> {
        if wo.z <= 0.0 {
            return None;
        }
        let a_inv = zeltner_ltc_a_inv(wo.z, self.roughness);
        let b_inv = zeltner_ltc_b_inv(wo.z, self.roughness);
        let wi_std = sample_ltc(a_inv, b_inv, us);
        if wi_std.z <= 0.0 {
            return None;
        }
        let phi_wo = phi_of(wo);
        let wi = rotate_about_z(wi_std, phi_wo);
        if wi.z <= 0.0 {
            return None;
        }
        Some(wi)
    }

    fn directional_albedo(&self, wo: Vec3) -> f32 {
        zeltner_dir_albedo(wo.z.clamp(0.0, 1.0), self.roughness)
    }
}

impl OpenPbrBsdf {
    pub(crate) fn new(
        params: OpenPbrBsdfParams,
        spec_lut: Arc<DielectricGgxDirectionalAlbedoLut>,
        coat_lut: Arc<DielectricGgxDirectionalAlbedoLut>,
        conductor_ec_lut: Arc<ConductorGgxEnergyCompensationLut>,
        dielectric_ec_lut: Arc<DielectricGgxEnergyCompensationLut>,
    ) -> Self {
        Self {
            p: params,
            spec_lut,
            coat_lut,
            conductor_ec_lut,
            dielectric_ec_lut,
        }
    }

    pub fn eval(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        if !is_upper_hemisphere(wo) {
            return Vec3::ZERO;
        }
        let weights = self.layer_weights(wo, Some(wi));
        let mut total = Vec3::ZERO;

        if weights.coat_amp > 0.0 && !self.coat_is_smooth() {
            let (wo_c, wi_c) = self.to_coat(wo, wi);
            total += weights.coat_amp * self.eval_coat(wo_c, wi_c);
        }
        if weights.metal.length_squared() > 0.0 && !self.metal_is_smooth() {
            total += weights.metal * self.eval_metal(wo, wi);
        }
        if weights.spec_brdf.length_squared() > 0.0 && !self.spec_brdf_is_smooth() {
            total += weights.spec_brdf * self.eval_spec_brdf(wo, wi);
        }
        if weights.spec_btdf.length_squared() > 0.0 && !self.spec_btdf_is_smooth() {
            total += weights.spec_btdf * self.eval_spec_btdf(wo, wi);
        }
        if weights.fuzz.length_squared() > 0.0 {
            total += weights.fuzz * self.eval_fuzz(wo, wi);
        }
        if weights.diff_brdf.length_squared() > 0.0 {
            total += weights.diff_brdf * self.eval_diff_brdf(wo, wi);
        }
        if weights.diff_btdf.length_squared() > 0.0 {
            total += weights.diff_btdf * self.eval_diff_btdf(wo, wi);
        }
        total
    }

    pub fn pdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        if !is_upper_hemisphere(wo) {
            return 0.0;
        }
        let probs = self.lobe_probabilities(wo);
        if probs.total <= 0.0 {
            return 0.0;
        }
        let mut pdf = 0.0;
        if probs.coat > 0.0 && !self.coat_is_smooth() {
            let (wo_c, wi_c) = self.to_coat(wo, wi);
            pdf += (probs.coat / probs.total) * self.pdf_coat(wo_c, wi_c);
        }
        if probs.metal > 0.0 && !self.metal_is_smooth() {
            pdf += (probs.metal / probs.total) * self.pdf_metal(wo, wi);
        }
        if probs.spec_brdf > 0.0 && !self.spec_brdf_is_smooth() {
            pdf += (probs.spec_brdf / probs.total) * self.pdf_specular_brdf(wo, wi);
        }
        if probs.spec_btdf > 0.0 && !self.spec_btdf_is_smooth() {
            pdf += (probs.spec_btdf / probs.total) * self.pdf_specular_btdf(wo, wi);
        }
        if probs.fuzz > 0.0 {
            pdf += (probs.fuzz / probs.total) * self.pdf_fuzz(wo, wi);
        }
        if probs.diff_brdf > 0.0 {
            pdf += (probs.diff_brdf / probs.total) * self.pdf_diff_brdf(wo, wi);
        }
        if probs.diff_btdf > 0.0 {
            pdf += (probs.diff_btdf / probs.total) * self.pdf_diff_btdf(wo, wi);
        }
        pdf
    }

    pub fn sample(&self, wo: Vec3, randoms: &MaterialSampleRandoms) -> Option<BsdfSample> {
        if !is_upper_hemisphere(wo) {
            return None;
        }
        let probs = self.lobe_probabilities(wo);
        if probs.total <= 0.0 {
            return None;
        }
        let chosen = pick_lobe(&probs, randoms.u_lobe * probs.total);
        let p_lobe = probs.lobe(chosen) / probs.total;
        if p_lobe <= 0.0 {
            return None;
        }

        let us = randoms.u_dir;
        match chosen {
            ChosenLobe::Coat => self.sample_coat(wo, us, p_lobe),
            ChosenLobe::Metal => self.sample_metal(wo, us, p_lobe),
            ChosenLobe::SpecBrdf => self.sample_spec_brdf(wo, us, p_lobe),
            ChosenLobe::SpecBtdf => self.sample_spec_btdf(wo, us, p_lobe, randoms.u_extra0),
            ChosenLobe::Fuzz => self.sample_fuzz(wo, us),
            ChosenLobe::DiffBrdf => self.sample_diff_brdf(wo, us),
            ChosenLobe::DiffBtdf => self.sample_diff_btdf(wo, us),
        }
    }

    fn layer_weights(&self, wo: Vec3, wi: Option<Vec3>) -> LayerWeights {
        let one = Vec3::ONE;
        let e_fuzz = self.lookup_fuzz_albedo(wo);
        let fuzz_amp_rgb = self.p.fuzz * self.p.fuzz_color;
        let fuzz_w = fuzz_amp_rgb;
        let below_fuzz_scalar = (1.0 - self.p.fuzz * e_fuzz).max(0.0);
        let below_fuzz = Vec3::splat(below_fuzz_scalar);

        let coat_amp = self.p.coat;
        let e_coat = if coat_amp > 0.0 {
            self.lookup_coat_albedo(wo)
        } else {
            0.0
        };
        let under_coat = if coat_amp > 0.0 {
            let absorb_tint = wi
                .map(|wi| self.coat_absorb_tint(wo, wi))
                .unwrap_or(self.p.coat_color);
            let coat_throughput =
                self.p.coat_darkening * absorb_tint * Vec3::splat((1.0 - e_coat).max(0.0));
            below_fuzz * (one * (1.0 - coat_amp) + coat_throughput * coat_amp)
        } else {
            below_fuzz
        };

        let metal = self.p.metalness * under_coat;
        let below_metal = (1.0 - self.p.metalness) * under_coat;

        let e_spec = self.lookup_spec_albedo(wo);
        let spec_amp_rgb = self.p.specular_color;
        let spec_brdf_w = below_metal * spec_amp_rgb;
        let leakage_spec = (one - spec_amp_rgb * e_spec).max(Vec3::ZERO);
        let below_spec = below_metal * leakage_spec;

        let trans = self.p.transmission;
        let transmission_tint = if self.p.thin_walled {
            Vec3::ONE
        } else {
            self.p.transmission_color
        };
        let spec_btdf_w = below_spec * trans * transmission_tint;
        let below_trans = below_spec * (1.0 - trans);

        let thin_factor = if self.p.thin_walled { 1.0 } else { 0.0 };
        let diff_brdf_w =
            below_trans * ((1.0 - self.p.subsurface) * self.p.base) * self.p.base_color;
        let diff_btdf_w = below_trans * (self.p.subsurface * thin_factor) * self.p.subsurface_color;

        LayerWeights {
            coat_amp: coat_amp * below_fuzz_scalar,
            metal,
            spec_brdf: spec_brdf_w,
            spec_btdf: spec_btdf_w,
            fuzz: fuzz_w,
            diff_brdf: diff_brdf_w,
            diff_btdf: diff_btdf_w,
        }
    }

    fn lobe_probabilities(&self, wo: Vec3) -> LobeProbs {
        let weights = self.layer_weights(wo, None);
        let coat = if self.p.coat > 0.0 {
            weights.coat_amp
        } else {
            0.0
        };
        let metal_color_proxy = self.p.base_color;
        let metal = luminance(weights.metal * (Vec3::splat(0.5) + 0.5 * metal_color_proxy));
        let spec_brdf = luminance(weights.spec_brdf);
        let spec_btdf = if self.p.transmission > 0.0 {
            luminance(weights.spec_btdf).max(0.0)
        } else {
            0.0
        };
        let fuzz = luminance(weights.fuzz).max(0.0);
        let diff_brdf = luminance(weights.diff_brdf).max(0.0);
        let diff_btdf = luminance(weights.diff_btdf).max(0.0);

        let total = coat + metal + spec_brdf + spec_btdf + fuzz + diff_brdf + diff_btdf;
        LobeProbs {
            coat,
            metal,
            spec_brdf,
            spec_btdf,
            fuzz,
            diff_brdf,
            diff_btdf,
            total,
        }
    }

    fn lookup_coat_albedo(&self, wo: Vec3) -> f32 {
        let (wo_c, _) = self.to_coat(wo, Vec3::Z);
        if wo_c.z <= 0.0 {
            return 0.0;
        }
        let ss = self.coat_lut.lookup(wo_c, self.coat_roughness_proxy(), 0.0);
        self.dielectric_reflection_albedo_with_ms(
            wo_c,
            self.p.coat_alpha_x,
            self.p.coat_alpha_y,
            self.p.coat_eta,
            ss,
        )
    }

    fn lookup_spec_albedo(&self, wo: Vec3) -> f32 {
        if wo.z <= 0.0 {
            return 0.0;
        }
        let ss = self.spec_lut.lookup(wo, self.spec_roughness_proxy(), 0.0);
        if self.p.thin_walled {
            return ss;
        }
        self.dielectric_reflection_albedo_with_ms(
            wo,
            self.p.specular_alpha_x,
            self.p.specular_alpha_y,
            self.p.specular_eta,
            ss,
        )
    }

    fn lookup_fuzz_albedo(&self, wo: Vec3) -> f32 {
        if self.p.fuzz <= 0.0 {
            return 0.0;
        }
        let wo_f = self.to_fuzz(wo, Vec3::Z).0;
        OpenPbrFuzzBsdf::new(self.p.fuzz_roughness).directional_albedo(wo_f)
    }

    fn dielectric_reflection_albedo_with_ms(
        &self,
        wo: Vec3,
        alpha_x: f32,
        alpha_y: f32,
        eta: f32,
        ss_albedo: f32,
    ) -> f32 {
        if wo.z <= 0.0 || alpha_x.max(alpha_y) < EFFECTIVELY_SMOOTH_ALPHA {
            return ss_albedo.clamp(0.0, 1.0);
        }
        let ms = self.dielectric_ms_params(alpha_x, alpha_y, eta);
        let e_o = self
            .dielectric_ec_lut
            .lookup_e(wo.z, ms.roughness_eq, ms.eta_o);
        (ss_albedo + self.dielectric_reflection_ms_factor(&ms) * (1.0 - e_o).max(0.0))
            .clamp(0.0, 1.0)
    }

    fn coat_absorb_tint(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        let (wo_c, wi_c) = self.to_coat(wo, wi);
        if wo_c.z <= 0.0 || wi_c.z <= 0.0 {
            return Vec3::ONE;
        }
        let mu_o = refracted_cos_in_dielectric(wo_c.z, self.p.coat_eta);
        let mu_i = refracted_cos_in_dielectric(wi_c.z, self.p.coat_eta);
        let exponent = 0.5 * (1.0 / mu_i.max(1.0e-4) + 1.0 / mu_o.max(1.0e-4));
        vec3_powf(self.p.coat_color, exponent)
    }

    fn coat_roughness_proxy(&self) -> f32 {
        let alpha = self.p.coat_alpha_x.max(self.p.coat_alpha_y);
        alpha.sqrt().clamp(0.0, 1.0)
    }

    fn spec_roughness_proxy(&self) -> f32 {
        let alpha = self.p.specular_alpha_x.max(self.p.specular_alpha_y);
        alpha.sqrt().clamp(0.0, 1.0)
    }

    fn to_coat(&self, wo: Vec3, wi: Vec3) -> (Vec3, Vec3) {
        match self.p.coat_basis_in_base {
            Some(b) => (b.world_to_local(wo), b.world_to_local(wi)),
            None => (wo, wi),
        }
    }

    fn coat_to_base(&self, v: Vec3) -> Vec3 {
        match self.p.coat_basis_in_base {
            Some(b) => b.local_to_world(v),
            None => v,
        }
    }

    fn to_fuzz(&self, wo: Vec3, wi: Vec3) -> (Vec3, Vec3) {
        match self.p.fuzz_basis_in_base {
            Some(b) => (b.world_to_local(wo), b.world_to_local(wi)),
            None => (wo, wi),
        }
    }

    fn fuzz_to_base(&self, v: Vec3) -> Vec3 {
        match self.p.fuzz_basis_in_base {
            Some(b) => b.local_to_world(v),
            None => v,
        }
    }

    fn coat_is_smooth(&self) -> bool {
        self.p.coat <= 0.0
            || self.p.coat_alpha_x.max(self.p.coat_alpha_y) < EFFECTIVELY_SMOOTH_ALPHA
    }

    fn metal_is_smooth(&self) -> bool {
        self.p.metalness <= 0.0
            || self.p.specular_alpha_x.max(self.p.specular_alpha_y) < EFFECTIVELY_SMOOTH_ALPHA
    }

    fn spec_brdf_is_smooth(&self) -> bool {
        self.p.specular <= 0.0
            || self.p.specular_alpha_x.max(self.p.specular_alpha_y) < EFFECTIVELY_SMOOTH_ALPHA
    }

    fn spec_btdf_is_smooth(&self) -> bool {
        self.p.transmission <= 0.0
            || self.p.transmission_alpha_x.max(self.p.transmission_alpha_y)
                < EFFECTIVELY_SMOOTH_ALPHA
    }

    fn thin_wall_transmission_alpha_xy(&self) -> (f32, f32) {
        let eta = self.p.transmission_eta.max(1.0);
        let scale = (3.7 * (eta - 1.0) * (eta - 0.5).powi(2) / eta.powi(3))
            .max(0.0)
            .sqrt();
        let alpha_scale = scale * scale;
        (
            (self.p.transmission_alpha_x * alpha_scale).clamp(MIN_ALPHA, 1.0),
            (self.p.transmission_alpha_y * alpha_scale).clamp(MIN_ALPHA, 1.0),
        )
    }

    fn conductor_roughness_eq(&self) -> f32 {
        (self.p.specular_alpha_x * self.p.specular_alpha_y).powf(0.25)
    }

    fn dielectric_roughness_eq(&self, alpha_x: f32, alpha_y: f32) -> f32 {
        (alpha_x * alpha_y).powf(0.25)
    }

    fn dielectric_reflection_ms_factor(&self, ms: &DielectricMsParams) -> f32 {
        let f_avg = f_avg_dielectric(ms.eta_o).clamp(0.0, 1.0);
        let denom = 1.0 - f_avg * (1.0 - ms.e_avg_o).max(0.0);
        (f_avg * f_avg * ms.e_avg_o / denom.max(MS_DENOM_EPS)).clamp(0.0, 1.0)
    }

    fn conductor_lobe_weights(&self) -> TwoLobeWeights {
        let e_avg = self
            .conductor_ec_lut
            .lookup_e_avg(self.conductor_roughness_eq())
            .clamp(0.0, 1.0);
        let ss = e_avg;
        let ms = (1.0 - e_avg).max(0.0);
        TwoLobeWeights {
            ss,
            ms,
            total: (ss + ms).max(1.0e-8),
        }
    }

    fn dielectric_ms_params(&self, alpha_x: f32, alpha_y: f32, eta: f32) -> DielectricMsParams {
        let eta_o = if self.p.front_face {
            eta.max(1.0e-4)
        } else {
            1.0 / eta.max(1.0e-4)
        };
        let eta_rel = 1.0 / eta_o;
        let roughness_eq = self.dielectric_roughness_eq(alpha_x, alpha_y);
        let e_avg_o = self.dielectric_ec_lut.lookup_e_avg(roughness_eq, eta_o);
        let e_avg_t = self.dielectric_ec_lut.lookup_e_avg(roughness_eq, eta_rel);
        let one_minus_e_avg_o = (1.0 - e_avg_o).max(MS_DENOM_EPS);
        let one_minus_e_avg_t = (1.0 - e_avg_t).max(MS_DENOM_EPS);
        let f_avg_o = f_avg_dielectric(eta_o);
        let f_avg_t = f_avg_dielectric(eta_rel);
        let a = (1.0 - f_avg_o) / one_minus_e_avg_o;
        let b = (1.0 - f_avg_t) * eta_o * eta_o / one_minus_e_avg_t;
        let x = if a + b > MS_DENOM_EPS {
            b / (a + b)
        } else {
            0.5
        };
        let ratio_r = (1.0 - x * (1.0 - f_avg_o)).clamp(0.0, 1.0);

        DielectricMsParams {
            eta_o,
            eta_rel,
            roughness_eq,
            e_avg_o,
            ratio_r,
            one_minus_e_avg_o,
            one_minus_e_avg_t,
        }
    }

    fn dielectric_reflection_lobe_weights(&self, ms: &DielectricMsParams) -> TwoLobeWeights {
        let f_avg = f_avg_dielectric(ms.eta_o).clamp(0.0, 1.0);
        let ss = (f_avg * ms.e_avg_o).clamp(0.0, 1.0);
        let ms_weight = (1.0 - ms.e_avg_o).max(0.0) * self.dielectric_reflection_ms_factor(ms);
        TwoLobeWeights {
            ss,
            ms: ms_weight,
            total: (ss + ms_weight).max(1.0e-8),
        }
    }

    fn dielectric_transmission_lobe_weights(&self, ms: &DielectricMsParams) -> TwoLobeWeights {
        let ss = ms.e_avg_o.clamp(0.0, 1.0);
        let ms_weight = (1.0 - ms.e_avg_o).max(0.0) * (1.0 - ms.ratio_r).clamp(0.0, 1.0);
        TwoLobeWeights {
            ss,
            ms: ms_weight,
            total: (ss + ms_weight).max(1.0e-8),
        }
    }

    fn eval_dielectric_reflection_ms(
        &self,
        wo: Vec3,
        wi: Vec3,
        alpha_x: f32,
        alpha_y: f32,
        eta: f32,
    ) -> f32 {
        if wo.z <= 0.0 || wi.z <= 0.0 {
            return 0.0;
        }
        let ms = self.dielectric_ms_params(alpha_x, alpha_y, eta);
        let cos_o = wo.z;
        let cos_i = wi.z;
        let e_o = self
            .dielectric_ec_lut
            .lookup_e(cos_o, ms.roughness_eq, ms.eta_o);
        let e_i = self
            .dielectric_ec_lut
            .lookup_e(cos_i, ms.roughness_eq, ms.eta_o);
        self.dielectric_reflection_ms_factor(&ms) * (1.0 - e_o) * (1.0 - e_i)
            / (PI * ms.one_minus_e_avg_o)
    }

    fn eval_dielectric_transmission_ms(
        &self,
        wo: Vec3,
        wi: Vec3,
        alpha_x: f32,
        alpha_y: f32,
        eta: f32,
    ) -> f32 {
        if wo.z <= 0.0 || wi.z >= 0.0 {
            return 0.0;
        }
        let ms = self.dielectric_ms_params(alpha_x, alpha_y, eta);
        let cos_o = wo.z;
        let cos_i = wi.z.abs();
        let e_o = self
            .dielectric_ec_lut
            .lookup_e(cos_o, ms.roughness_eq, ms.eta_o);
        let e_i = self
            .dielectric_ec_lut
            .lookup_e(cos_i, ms.roughness_eq, ms.eta_rel);
        let radiance_scale = ms.eta_o * ms.eta_o;
        (1.0 - ms.ratio_r) * (1.0 - e_o) * (1.0 - e_i) * radiance_scale
            / (PI * ms.one_minus_e_avg_t)
    }

    fn eval_coat(&self, wo_c: Vec3, wi_c: Vec3) -> Vec3 {
        if wo_c.z <= 0.0 || wi_c.z <= 0.0 {
            return Vec3::ZERO;
        }
        let Some(wm) = reflection_half_vector(wo_c, wi_c) else {
            return Vec3::ZERO;
        };
        let cos_o = wo_c.z;
        let cos_i = wi_c.z;
        let d = ggx_d(wm, self.p.coat_alpha_x, self.p.coat_alpha_y);
        let g = ggx_g2_height_correlated(wo_c, wi_c, self.p.coat_alpha_x, self.p.coat_alpha_y);
        if d <= 0.0 || g <= 0.0 {
            return Vec3::ZERO;
        }
        let f = fresnel_dielectric(wo_c.dot(wm).abs(), 1.0, self.p.coat_eta);
        let ss = d * g * f / (4.0 * cos_o * cos_i);
        let ms = self.eval_dielectric_reflection_ms(
            wo_c,
            wi_c,
            self.p.coat_alpha_x,
            self.p.coat_alpha_y,
            self.p.coat_eta,
        );
        Vec3::splat(ss + ms)
    }

    fn pdf_coat(&self, wo_c: Vec3, wi_c: Vec3) -> f32 {
        if wo_c.z <= 0.0 || wi_c.z <= 0.0 {
            return 0.0;
        }
        let ms =
            self.dielectric_ms_params(self.p.coat_alpha_x, self.p.coat_alpha_y, self.p.coat_eta);
        let weights = self.dielectric_reflection_lobe_weights(&ms);
        let pdf_ms = cosine_weighted_hemisphere_pdf(wi_c.z);
        let Some(wm) = reflection_half_vector(wo_c, wi_c) else {
            return weights.ms * pdf_ms / weights.total;
        };
        let pdf_wm = pdf_wm_bounded_vndf(wo_c, wm, self.p.coat_alpha_x, self.p.coat_alpha_y);
        let denom = 4.0 * wo_c.dot(wm).abs();
        if denom <= 0.0 {
            return weights.ms * pdf_ms / weights.total;
        }
        let pdf_ss = pdf_wm / denom;
        (weights.ss * pdf_ss + weights.ms * pdf_ms) / weights.total
    }

    fn eval_metal(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        if wo.z <= 0.0 || wi.z <= 0.0 {
            return Vec3::ZERO;
        }
        let Some(wm) = reflection_half_vector(wo, wi) else {
            return Vec3::ZERO;
        };
        let d = ggx_d(wm, self.p.specular_alpha_x, self.p.specular_alpha_y);
        let g = ggx_g2_height_correlated(wo, wi, self.p.specular_alpha_x, self.p.specular_alpha_y);
        if d <= 0.0 || g <= 0.0 {
            return Vec3::ZERO;
        }
        let f = self.metal_fresnel(wo.dot(wm).abs());
        let ss = f * (d * g / (4.0 * wo.z * wi.z));
        ss + self.eval_metal_ms(wo, wi)
    }

    fn metal_fresnel(&self, cos_theta: f32) -> Vec3 {
        let no_film = self.metal_f82_tint_fresnel(cos_theta);
        let film_weight = self.p.thin_film_weight.clamp(0.0, 1.0);
        if film_weight > 0.0 && self.p.thin_film_thickness > 0.0 {
            let uncoated_film = eval_thin_film_conductor(
                cos_theta,
                1.0,
                self.p.thin_film_ior,
                self.p.metal_n,
                self.p.metal_k,
                self.p.thin_film_thickness,
            );
            let film = if self.p.coat > 0.0 {
                let coated = eval_thin_film_conductor(
                    cos_theta,
                    self.p.coat_eta,
                    self.p.thin_film_ior,
                    self.p.metal_n,
                    self.p.metal_k,
                    self.p.thin_film_thickness,
                );
                lerp_vec3(uncoated_film, coated, self.p.coat.clamp(0.0, 1.0))
            } else {
                uncoated_film
            };
            lerp_vec3(no_film, film, film_weight)
        } else {
            no_film
        }
    }

    fn metal_f82_tint_fresnel(&self, cos_theta: f32) -> Vec3 {
        let f0 = (self.p.base * self.p.base_color).max(Vec3::ZERO);
        let f_schlick = schlick_vec3(f0, cos_theta);
        let f_schlick_82 = schlick_vec3(f0, COS_82);
        let f_target_82 = self.p.specular_color.clamp(Vec3::ZERO, Vec3::ONE) * f_schlick_82;
        let c = cos_theta.clamp(0.0, 1.0);
        let denom = COS_82 * (1.0 - COS_82).powi(6);
        let correction = c * (1.0 - c).powi(6) / denom.max(1.0e-8);
        (self.p.specular * (f_schlick - correction * (f_schlick_82 - f_target_82)))
            .clamp(Vec3::ZERO, Vec3::ONE)
    }

    fn eval_metal_ms(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        if wo.z <= 0.0 || wi.z <= 0.0 {
            return Vec3::ZERO;
        }
        let roughness_eq = self.conductor_roughness_eq();
        let e_o = self.conductor_ec_lut.lookup_e(wo.z, roughness_eq);
        let e_i = self.conductor_ec_lut.lookup_e(wi.z, roughness_eq);
        let e_avg = self.conductor_ec_lut.lookup_e_avg(roughness_eq);
        let one_minus_e_avg = (1.0 - e_avg).max(MS_DENOM_EPS);
        let f_avg = self.metal_fresnel_avg();
        let f_ms = compute_conductor_f_ms(f_avg, e_avg);
        f_ms * ((1.0 - e_o) * (1.0 - e_i) / (PI * one_minus_e_avg))
    }

    fn metal_fresnel_avg(&self) -> Vec3 {
        if self.p.thin_film_weight <= 0.0 || self.p.thin_film_thickness <= 0.0 {
            return self.metal_f82_tint_fresnel_avg();
        }

        let mut sum = Vec3::ZERO;
        const SAMPLES: usize = 16;
        for i in 0..SAMPLES {
            let mu = (i as f32 + 0.5) / SAMPLES as f32;
            sum += self.metal_fresnel(mu) * (2.0 * mu);
        }
        sum / SAMPLES as f32
    }

    fn metal_f82_tint_fresnel_avg(&self) -> Vec3 {
        let f0 = (self.p.base * self.p.base_color).max(Vec3::ZERO);
        let f_schlick_82 = schlick_vec3(f0, COS_82);
        let f_target_82 = self.p.specular_color.clamp(Vec3::ZERO, Vec3::ONE) * f_schlick_82;
        let denom = COS_82 * (1.0 - COS_82).powi(6);
        let b = (f_schlick_82 - f_target_82) / denom.max(1.0e-8);
        (self.p.specular * (f0 + (Vec3::ONE - f0) / 21.0 - b / 126.0)).clamp(Vec3::ZERO, Vec3::ONE)
    }

    fn eval_spec_brdf(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        if wo.z <= 0.0 || wi.z <= 0.0 {
            return Vec3::ZERO;
        }
        let Some(wm) = reflection_half_vector(wo, wi) else {
            return Vec3::ZERO;
        };
        let d = ggx_d(wm, self.p.specular_alpha_x, self.p.specular_alpha_y);
        let g = ggx_g2_height_correlated(wo, wi, self.p.specular_alpha_x, self.p.specular_alpha_y);
        if d <= 0.0 || g <= 0.0 {
            return Vec3::ZERO;
        }
        if self.p.thin_walled {
            let (r, _) = self.thin_wall_coefficients(wo.z.abs());
            return r * (d * g / (4.0 * wo.z * wi.z));
        }
        let f = self.spec_brdf_fresnel(wo.dot(wm).abs());
        let ss = f * (d * g / (4.0 * wo.z * wi.z));
        let ms = Vec3::splat(self.eval_dielectric_reflection_ms(
            wo,
            wi,
            self.p.specular_alpha_x,
            self.p.specular_alpha_y,
            self.p.specular_eta,
        ));
        ss + ms
    }

    fn spec_brdf_fresnel(&self, cos_theta: f32) -> Vec3 {
        let no_film = Vec3::splat(self.base_dielectric_fresnel(
            cos_theta,
            self.p.transmission_eta,
            self.p.specular_eta,
        ));
        let film_weight = self.p.thin_film_weight.clamp(0.0, 1.0);
        if film_weight > 0.0 && self.p.thin_film_thickness > 0.0 {
            let uncoated_film = eval_thin_film_dielectric(
                cos_theta,
                1.0,
                self.p.thin_film_ior,
                self.p.specular_eta,
                self.p.thin_film_thickness,
            );
            let film = if self.p.coat > 0.0 {
                let coated = eval_thin_film_dielectric(
                    cos_theta,
                    self.p.coat_eta,
                    self.p.thin_film_ior,
                    self.p.specular_eta,
                    self.p.thin_film_thickness,
                );
                lerp_vec3(uncoated_film, coated, self.p.coat.clamp(0.0, 1.0))
            } else {
                uncoated_film
            };
            lerp_vec3(no_film, film, film_weight)
        } else {
            no_film
        }
    }

    fn base_dielectric_fresnel(&self, cos_theta: f32, eta_physical: f32, eta_prime: f32) -> f32 {
        let uncoated = if self.p.front_face {
            weighted_fresnel_ratio(cos_theta, eta_physical, eta_prime)
        } else {
            weighted_fresnel_ratio(cos_theta, 1.0 / eta_physical, 1.0 / eta_prime)
        };
        let coat = self.p.coat.clamp(0.0, 1.0);
        if coat <= 0.0 {
            return uncoated;
        }
        let eta_bc = eta_physical / self.p.coat_eta.max(1.0e-4);
        let eta_bc_prime = eta_prime / self.p.coat_eta.max(1.0e-4);
        let (eta_bc_no_tir, eta_bc_prime_no_tir) = if self.p.coat_eta > eta_physical {
            (1.0 / eta_bc.max(1.0e-4), 1.0 / eta_bc_prime.max(1.0e-4))
        } else {
            (eta_bc, eta_bc_prime)
        };
        let coated = if self.p.front_face {
            weighted_fresnel_ratio(cos_theta, eta_bc_no_tir, eta_bc_prime_no_tir)
        } else {
            weighted_fresnel_ratio(
                cos_theta,
                1.0 / eta_bc_no_tir.max(1.0e-4),
                1.0 / eta_bc_prime_no_tir.max(1.0e-4),
            )
        };
        lerp(uncoated, coated, coat)
    }

    fn thin_wall_coefficients(&self, cos_theta_o: f32) -> (Vec3, Vec3) {
        let eta = self.p.transmission_eta.max(1.0e-4);
        let eta_prime = self.p.specular_eta.max(1.0e-4);
        let cos_o = cos_theta_o.clamp(0.0, 1.0);
        let sin2_o = (1.0 - cos_o * cos_o).max(0.0);
        let sin2_i = (sin2_o / (eta * eta)).clamp(0.0, 1.0);
        let cos_i = (1.0 - sin2_i).sqrt().max(1.0e-4);

        let f_o = self.spec_brdf_fresnel(cos_o).clamp(Vec3::ZERO, Vec3::ONE);
        let f_i = self
            .thin_wall_internal_fresnel(cos_i, eta, eta_prime)
            .clamp(Vec3::ZERO, Vec3::ONE);
        let a = vec3_powf(
            self.p.transmission_color.clamp(Vec3::ZERO, Vec3::ONE),
            1.0 / cos_i,
        );
        let a2 = a * a;
        let f_i2_a2 = a2 * f_i * f_i;
        let denom = (Vec3::ONE - f_i2_a2).max(Vec3::splat(1.0e-6));
        let one_minus_fo = (Vec3::ONE - f_o).max(Vec3::ZERO);
        let one_minus_fi = (Vec3::ONE - f_i).max(Vec3::ZERO);

        let reflection = f_o + one_minus_fo * one_minus_fi * f_i * a2 / denom;
        let transmission = one_minus_fo * one_minus_fi * a * (Vec3::ONE + f_i2_a2 / denom);
        (
            reflection.clamp(Vec3::ZERO, Vec3::ONE),
            transmission.clamp(Vec3::ZERO, Vec3::ONE),
        )
    }

    fn thin_wall_internal_fresnel(&self, cos_theta: f32, eta: f32, eta_prime: f32) -> Vec3 {
        let no_film = Vec3::splat(weighted_fresnel_ratio(
            cos_theta,
            1.0 / eta.max(1.0e-4),
            1.0 / eta_prime.max(1.0e-4),
        ));
        let film_weight = self.p.thin_film_weight.clamp(0.0, 1.0);
        if film_weight <= 0.0 || self.p.thin_film_thickness <= 0.0 {
            return no_film;
        }
        let film = eval_thin_film_dielectric(
            cos_theta,
            eta.max(1.0e-4),
            self.p.thin_film_ior,
            1.0,
            self.p.thin_film_thickness,
        );
        lerp_vec3(no_film, film, film_weight)
    }

    fn pdf_specular_brdf_ss(&self, wo: Vec3, wi: Vec3) -> f32 {
        if wo.z <= 0.0 || wi.z <= 0.0 {
            return 0.0;
        }
        let Some(wm) = reflection_half_vector(wo, wi) else {
            return 0.0;
        };
        let pdf_wm = pdf_wm_bounded_vndf(wo, wm, self.p.specular_alpha_x, self.p.specular_alpha_y);
        let denom = 4.0 * wo.dot(wm).abs();
        if denom <= 0.0 {
            return 0.0;
        }
        pdf_wm / denom
    }

    fn pdf_metal(&self, wo: Vec3, wi: Vec3) -> f32 {
        if wo.z <= 0.0 || wi.z <= 0.0 {
            return 0.0;
        }
        let weights = self.conductor_lobe_weights();
        let pdf_ss = self.pdf_specular_brdf_ss(wo, wi);
        let pdf_ms = cosine_weighted_hemisphere_pdf(wi.z);
        (weights.ss * pdf_ss + weights.ms * pdf_ms) / weights.total
    }

    fn pdf_specular_brdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        if wo.z <= 0.0 || wi.z <= 0.0 {
            return 0.0;
        }
        if self.p.thin_walled {
            return self.pdf_specular_brdf_ss(wo, wi);
        }
        let ms = self.dielectric_ms_params(
            self.p.specular_alpha_x,
            self.p.specular_alpha_y,
            self.p.specular_eta,
        );
        let weights = self.dielectric_reflection_lobe_weights(&ms);
        let pdf_ss = self.pdf_specular_brdf_ss(wo, wi);
        let pdf_ms = cosine_weighted_hemisphere_pdf(wi.z);
        (weights.ss * pdf_ss + weights.ms * pdf_ms) / weights.total
    }

    fn eval_spec_btdf(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        if self.p.thin_walled {
            return self.eval_thin_wall_spec_btdf(wo, wi);
        }
        if wo.z <= 0.0 || wi.z >= 0.0 {
            return Vec3::ZERO;
        }
        if self.dispersion_rgb_sharing_active() && self.p.wavelength_lock.is_none() {
            return self.eval_spec_btdf_rgb_shared(wo, wi);
        }
        let eta_rel = self.transmission_eta_rel();
        let Some(value) = self.eval_spec_btdf_scalar_with_eta(wo, wi, eta_rel) else {
            return Vec3::ZERO;
        };
        let eta_physical = self.transmission_eta_used();
        let eta_fresnel = self.transmission_fresnel_eta_used();
        let f_rgb = self.spec_btdf_fresnel_with_eta(value.cos_wo_wm, eta_physical, eta_fresnel);
        let ss = (Vec3::ONE - f_rgb).max(Vec3::ZERO) * value.scalar;
        let ms = Vec3::splat(self.eval_dielectric_transmission_ms(
            wo,
            wi,
            self.p.transmission_alpha_x,
            self.p.transmission_alpha_y,
            eta_fresnel,
        ));
        ss + ms
    }

    fn eval_thin_wall_spec_btdf(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        if wo.z <= 0.0 || wi.z >= 0.0 {
            return Vec3::ZERO;
        }
        let (_, t) = self.thin_wall_coefficients(wo.z.abs());
        if self.spec_btdf_is_smooth() {
            return Vec3::ZERO;
        }
        let wi_mirror = Vec3::new(wi.x, wi.y, -wi.z);
        let Some(wm) = reflection_half_vector(wo, wi_mirror) else {
            return Vec3::ZERO;
        };
        let (alpha_x, alpha_y) = self.thin_wall_transmission_alpha_xy();
        let d = ggx_d(wm, alpha_x, alpha_y);
        let g = ggx_g2_height_correlated(wo, wi_mirror, alpha_x, alpha_y);
        if d <= 0.0 || g <= 0.0 {
            return Vec3::ZERO;
        }
        t * (d * g / (4.0 * wo.z * wi.z.abs()))
    }

    fn eval_spec_btdf_scalar_with_eta(
        &self,
        wo: Vec3,
        wi: Vec3,
        eta_rel: f32,
    ) -> Option<SpecBtdfEval> {
        let wm_unnorm = eta_rel * wo + wi;
        if wm_unnorm.length_squared() < 1.0e-12 {
            return None;
        }
        let mut wm = wm_unnorm.normalize();
        if wm.z < 0.0 {
            wm = -wm;
        }
        let cos_wo_wm = wo.dot(wm);
        if cos_wo_wm <= 0.0 {
            return None;
        }
        let cos_wi_wm = wi.dot(wm);
        let den = cos_wi_wm + eta_rel * cos_wo_wm;
        if den.abs() < 1.0e-6 {
            return None;
        }
        let d = ggx_d(wm, self.p.transmission_alpha_x, self.p.transmission_alpha_y);
        let g = ggx_g2_height_correlated(
            wo,
            wi,
            self.p.transmission_alpha_x,
            self.p.transmission_alpha_y,
        );
        if d <= 0.0 || g <= 0.0 {
            return None;
        }
        let radiance_scale = 1.0 / (eta_rel * eta_rel);
        let scalar = d * g * (cos_wi_wm * cos_wo_wm).abs();
        let denom = den * den * wo.z.abs() * wi.z.abs();
        if denom <= 0.0 {
            return None;
        }
        Some(SpecBtdfEval {
            scalar: scalar * radiance_scale / denom,
            cos_wo_wm,
        })
    }

    fn spec_btdf_fresnel_with_eta(
        &self,
        cos_theta: f32,
        eta_physical: f32,
        eta_prime: f32,
    ) -> Vec3 {
        let no_film = Vec3::splat(self.base_dielectric_fresnel(cos_theta, eta_physical, eta_prime));
        let film_weight = self.p.thin_film_weight.clamp(0.0, 1.0);
        if film_weight > 0.0 && self.p.thin_film_thickness > 0.0 && !self.p.thin_walled {
            let uncoated_film = eval_thin_film_dielectric(
                cos_theta,
                1.0,
                self.p.thin_film_ior,
                eta_prime,
                self.p.thin_film_thickness,
            );
            let film = if self.p.coat > 0.0 {
                let coated = eval_thin_film_dielectric(
                    cos_theta,
                    self.p.coat_eta,
                    self.p.thin_film_ior,
                    eta_prime,
                    self.p.thin_film_thickness,
                );
                lerp_vec3(uncoated_film, coated, self.p.coat.clamp(0.0, 1.0))
            } else {
                uncoated_film
            };
            lerp_vec3(no_film, film, film_weight)
        } else {
            no_film
        }
    }

    fn pdf_specular_btdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        if self.p.thin_walled {
            if self.spec_btdf_is_smooth() || wo.z <= 0.0 || wi.z >= 0.0 {
                return 0.0;
            }
            let wi_mirror = Vec3::new(wi.x, wi.y, -wi.z);
            let Some(wm) = reflection_half_vector(wo, wi_mirror) else {
                return 0.0;
            };
            let (alpha_x, alpha_y) = self.thin_wall_transmission_alpha_xy();
            let pdf_wm = pdf_wm_bounded_vndf(wo, wm, alpha_x, alpha_y);
            let denom = 4.0 * wo.dot(wm).abs();
            return if denom > 0.0 { pdf_wm / denom } else { 0.0 };
        }
        if wo.z <= 0.0 || wi.z >= 0.0 {
            return 0.0;
        }
        if self.dispersion_rgb_sharing_active() && self.p.wavelength_lock.is_none() {
            return self.pdf_specular_btdf_rgb_mixture(wo, wi);
        }
        let eta_rel = self.transmission_eta_rel();
        self.pdf_specular_btdf_with_eta(wo, wi, eta_rel, self.transmission_eta_used())
    }

    fn pdf_specular_btdf_with_eta(&self, wo: Vec3, wi: Vec3, eta_rel: f32, eta: f32) -> f32 {
        if wo.z <= 0.0 || wi.z >= 0.0 {
            return 0.0;
        }
        let ms = self.dielectric_ms_params(
            self.p.transmission_alpha_x,
            self.p.transmission_alpha_y,
            eta,
        );
        let weights = self.dielectric_transmission_lobe_weights(&ms);
        let pdf_ss = self.pdf_specular_btdf_ss_with_eta(wo, wi, eta_rel);
        let pdf_ms = cosine_weighted_hemisphere_pdf(-wi.z);
        (weights.ss * pdf_ss + weights.ms * pdf_ms) / weights.total
    }

    fn pdf_specular_btdf_ss_with_eta(&self, wo: Vec3, wi: Vec3, eta_rel: f32) -> f32 {
        let wm_unnorm = eta_rel * wo + wi;
        if wm_unnorm.length_squared() < 1.0e-12 {
            return 0.0;
        }
        let mut wm = wm_unnorm.normalize();
        if wm.z < 0.0 {
            wm = -wm;
        }
        let cos_wo_wm = wo.dot(wm);
        if cos_wo_wm <= 0.0 {
            return 0.0;
        }
        let cos_wi_wm = wi.dot(wm);
        let den = cos_wi_wm + eta_rel * cos_wo_wm;
        if den.abs() < 1.0e-6 {
            return 0.0;
        }
        let pdf_wm = pdf_wm_vndf(
            wo,
            wm,
            self.p.transmission_alpha_x,
            self.p.transmission_alpha_y,
        );
        if pdf_wm <= 0.0 {
            return 0.0;
        }
        pdf_wm * cos_wi_wm.abs() / (den * den)
    }

    fn transmission_eta_used(&self) -> f32 {
        match self.p.wavelength_lock {
            Some(lambda) if self.p.transmission_dispersion_abbe > 0.0 => cauchy_ior(
                lambda,
                self.p.specular_eta,
                self.p.transmission_dispersion_abbe,
            ),
            _ => self.p.transmission_eta,
        }
    }

    fn transmission_fresnel_eta_used(&self) -> f32 {
        match self.p.wavelength_lock {
            Some(lambda) if self.p.transmission_dispersion_abbe > 0.0 => cauchy_ior(
                lambda,
                self.p.specular_eta,
                self.p.transmission_dispersion_abbe,
            ),
            _ => self.p.specular_eta,
        }
    }

    fn transmission_eta_rel(&self) -> f32 {
        let eta = self.transmission_eta_used();
        if self.p.front_face { 1.0 / eta } else { eta }
    }

    fn dispersion_rgb_sharing_active(&self) -> bool {
        self.p.transmission_dispersion_abbe > 0.0
            && self.p.front_face
            && !self.spec_btdf_is_smooth()
    }

    fn dispersion_channels(&self) -> [DispersionChannel; 3] {
        let throughput = self.p.path_throughput.max(Vec3::ZERO);
        let weights = if throughput.max_element() > 0.0 {
            throughput
        } else {
            Vec3::ONE
        };
        let sum = (weights.x + weights.y + weights.z).max(1.0e-8);
        [
            DispersionChannel {
                lambda_nm: 656.27,
                color: Vec3::X,
                probability: weights.x / sum,
            },
            DispersionChannel {
                lambda_nm: 587.56,
                color: Vec3::Y,
                probability: weights.y / sum,
            },
            DispersionChannel {
                lambda_nm: 486.13,
                color: Vec3::Z,
                probability: weights.z / sum,
            },
        ]
    }

    fn pick_dispersion_channel(&self, u: f32) -> DispersionChannel {
        let channels = self.dispersion_channels();
        let mut u = u.clamp(0.0, 1.0);
        for channel in channels {
            if u <= channel.probability {
                return channel;
            }
            u -= channel.probability;
        }
        channels[2]
    }

    fn eval_spec_btdf_rgb_shared(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        let mut value = Vec3::ZERO;
        for channel in self.dispersion_channels() {
            if channel.probability <= 0.0 {
                continue;
            }
            let eta = cauchy_ior(
                channel.lambda_nm,
                self.p.transmission_eta,
                self.p.transmission_dispersion_abbe,
            );
            let eta_fresnel = cauchy_ior(
                channel.lambda_nm,
                self.p.specular_eta,
                self.p.transmission_dispersion_abbe,
            );
            let eta_rel = if self.p.front_face { 1.0 / eta } else { eta };
            let ss = self
                .eval_spec_btdf_scalar_with_eta(wo, wi, eta_rel)
                .map(|eval| {
                    let f = self.spec_btdf_fresnel_with_eta(eval.cos_wo_wm, eta, eta_fresnel);
                    (1.0 - f.dot(channel.color)).max(0.0) * eval.scalar
                })
                .unwrap_or(0.0);
            let ms = self.eval_dielectric_transmission_ms(
                wo,
                wi,
                self.p.transmission_alpha_x,
                self.p.transmission_alpha_y,
                eta_fresnel,
            );
            value += channel.color * (ss + ms);
        }
        value
    }

    fn pdf_specular_btdf_rgb_mixture(&self, wo: Vec3, wi: Vec3) -> f32 {
        let mut pdf = 0.0;
        for channel in self.dispersion_channels() {
            if channel.probability <= 0.0 {
                continue;
            }
            let eta = cauchy_ior(
                channel.lambda_nm,
                self.p.transmission_eta,
                self.p.transmission_dispersion_abbe,
            );
            let eta_fresnel = cauchy_ior(
                channel.lambda_nm,
                self.p.specular_eta,
                self.p.transmission_dispersion_abbe,
            );
            let eta_rel = if self.p.front_face { 1.0 / eta } else { eta };
            pdf +=
                channel.probability * self.pdf_specular_btdf_with_eta(wo, wi, eta_rel, eta_fresnel);
        }
        pdf
    }

    fn eval_fuzz(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        let (wo_f, wi_f) = self.to_fuzz(wo, wi);
        OpenPbrFuzzBsdf::new(self.p.fuzz_roughness).eval(wo_f, wi_f)
    }

    fn pdf_fuzz(&self, wo: Vec3, wi: Vec3) -> f32 {
        let (wo_f, wi_f) = self.to_fuzz(wo, wi);
        OpenPbrFuzzBsdf::new(self.p.fuzz_roughness).pdf(wo_f, wi_f)
    }

    fn eval_diff_brdf(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        EonBsdf::new(Vec3::ONE, self.p.diffuse_roughness).eval(wo, wi)
    }

    fn pdf_diff_brdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        EonBsdf::new(Vec3::ONE, self.p.diffuse_roughness).pdf(wo, wi)
    }

    fn eval_diff_btdf(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        if !self.p.thin_walled {
            return Vec3::ZERO;
        }
        if wo.z <= 0.0 || wi.z >= 0.0 {
            return Vec3::ZERO;
        }
        let wi_flipped = Vec3::new(wi.x, wi.y, -wi.z);
        EonBsdf::new(Vec3::ONE, self.p.diffuse_roughness).eval(wo, wi_flipped)
    }

    fn pdf_diff_btdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        if !self.p.thin_walled {
            return 0.0;
        }
        if wo.z <= 0.0 || wi.z >= 0.0 {
            return 0.0;
        }
        let wi_flipped = Vec3::new(wi.x, wi.y, -wi.z);
        EonBsdf::new(Vec3::ONE, self.p.diffuse_roughness).pdf(wo, wi_flipped)
    }

    fn sample_coat(&self, wo: Vec3, us: Vec2, p_lobe: f32) -> Option<BsdfSample> {
        let (wo_c, _) = self.to_coat(wo, Vec3::Z);
        if wo_c.z <= 0.0 {
            return None;
        }
        if self.coat_is_smooth() {
            let wi_c = Vec3::new(-wo_c.x, -wo_c.y, wo_c.z);
            let f = fresnel_dielectric(wi_c.z.abs(), 1.0, self.p.coat_eta);
            let wi = self.coat_to_base(wi_c);
            let weight = Vec3::splat(self.layer_weights(wo, Some(wi)).coat_amp * f / p_lobe);
            return Some(BsdfSample {
                weight,
                wi,
                pdf: p_lobe,
                flags: BsdfFlags::DELTA | BsdfFlags::REFLECTION,
                eta: 1.0,
                wavelength_lock: None,
            });
        }
        let ms =
            self.dielectric_ms_params(self.p.coat_alpha_x, self.p.coat_alpha_y, self.p.coat_eta);
        let weights = self.dielectric_reflection_lobe_weights(&ms);
        let ss_probability = weights.ss / weights.total;
        if us.x >= ss_probability {
            let u = if ss_probability < 1.0 {
                (us.x - ss_probability) / (1.0 - ss_probability)
            } else {
                0.0
            };
            let wi_c = sample_cosine_weighted_hemisphere(Vec2::new(u, us.y));
            let wi = self.coat_to_base(wi_c);
            return self.finalize_rough_sample(
                wo,
                wi,
                BsdfFlags::GLOSSY | BsdfFlags::REFLECTION,
                1.0,
            );
        }
        let u = if ss_probability > 0.0 {
            us.x / ss_probability
        } else {
            0.0
        };
        let wm = sample_wm_bounded_vndf(
            wo_c,
            self.p.coat_alpha_x,
            self.p.coat_alpha_y,
            Vec2::new(u, us.y),
        )?;
        let wi_c = reflect_local(wo_c, wm);
        if wi_c.z <= 0.0 {
            return None;
        }
        let wi = self.coat_to_base(wi_c);
        self.finalize_rough_sample(wo, wi, BsdfFlags::GLOSSY | BsdfFlags::REFLECTION, 1.0)
    }

    fn sample_metal(&self, wo: Vec3, us: Vec2, p_lobe: f32) -> Option<BsdfSample> {
        if self.metal_is_smooth() {
            let wi = Vec3::new(-wo.x, -wo.y, wo.z);
            let f = self.metal_fresnel(wi.z.abs());
            let weights = self.layer_weights(wo, Some(wi));
            let weight = (weights.metal * f) / p_lobe;
            return Some(BsdfSample {
                weight,
                wi,
                pdf: p_lobe,
                flags: BsdfFlags::DELTA | BsdfFlags::REFLECTION,
                eta: 1.0,
                wavelength_lock: None,
            });
        }
        let weights = self.conductor_lobe_weights();
        let ss_probability = weights.ss / weights.total;
        if us.x >= ss_probability {
            let u = if ss_probability < 1.0 {
                (us.x - ss_probability) / (1.0 - ss_probability)
            } else {
                0.0
            };
            let wi = sample_cosine_weighted_hemisphere(Vec2::new(u, us.y));
            return self.finalize_rough_sample(
                wo,
                wi,
                BsdfFlags::GLOSSY | BsdfFlags::REFLECTION,
                1.0,
            );
        }
        let u = if ss_probability > 0.0 {
            us.x / ss_probability
        } else {
            0.0
        };
        let wm = sample_wm_bounded_vndf(
            wo,
            self.p.specular_alpha_x,
            self.p.specular_alpha_y,
            Vec2::new(u, us.y),
        )?;
        let wi = reflect_local(wo, wm);
        if wi.z <= 0.0 {
            return None;
        }
        self.finalize_rough_sample(wo, wi, BsdfFlags::GLOSSY | BsdfFlags::REFLECTION, 1.0)
    }

    fn sample_spec_brdf(&self, wo: Vec3, us: Vec2, p_lobe: f32) -> Option<BsdfSample> {
        if self.spec_brdf_is_smooth() {
            let wi = Vec3::new(-wo.x, -wo.y, wo.z);
            let f = if self.p.thin_walled {
                self.thin_wall_coefficients(wo.z.abs()).0
            } else {
                self.spec_brdf_fresnel(wi.z.abs())
            };
            let weights = self.layer_weights(wo, Some(wi));
            let weight = (weights.spec_brdf * f) / p_lobe;
            return Some(BsdfSample {
                weight,
                wi,
                pdf: p_lobe,
                flags: BsdfFlags::DELTA | BsdfFlags::REFLECTION,
                eta: 1.0,
                wavelength_lock: None,
            });
        }
        if self.p.thin_walled {
            let wm =
                sample_wm_bounded_vndf(wo, self.p.specular_alpha_x, self.p.specular_alpha_y, us)?;
            let wi = reflect_local(wo, wm);
            if wi.z <= 0.0 {
                return None;
            }
            return self.finalize_rough_sample(
                wo,
                wi,
                BsdfFlags::GLOSSY | BsdfFlags::REFLECTION,
                1.0,
            );
        }
        let ms = self.dielectric_ms_params(
            self.p.specular_alpha_x,
            self.p.specular_alpha_y,
            self.p.specular_eta,
        );
        let weights = self.dielectric_reflection_lobe_weights(&ms);
        let ss_probability = weights.ss / weights.total;
        if us.x >= ss_probability {
            let u = if ss_probability < 1.0 {
                (us.x - ss_probability) / (1.0 - ss_probability)
            } else {
                0.0
            };
            let wi = sample_cosine_weighted_hemisphere(Vec2::new(u, us.y));
            return self.finalize_rough_sample(
                wo,
                wi,
                BsdfFlags::GLOSSY | BsdfFlags::REFLECTION,
                1.0,
            );
        }
        let u = if ss_probability > 0.0 {
            us.x / ss_probability
        } else {
            0.0
        };
        let wm = sample_wm_bounded_vndf(
            wo,
            self.p.specular_alpha_x,
            self.p.specular_alpha_y,
            Vec2::new(u, us.y),
        )?;
        let wi = reflect_local(wo, wm);
        if wi.z <= 0.0 {
            return None;
        }
        self.finalize_rough_sample(wo, wi, BsdfFlags::GLOSSY | BsdfFlags::REFLECTION, 1.0)
    }

    fn sample_spec_btdf(&self, wo: Vec3, us: Vec2, p_lobe: f32, u_aux: f32) -> Option<BsdfSample> {
        if self.p.thin_walled {
            let (_, t) = self.thin_wall_coefficients(wo.z.abs());
            if self.spec_btdf_is_smooth() {
                let wi = -wo;
                let base_weight = self.layer_weights(wo, Some(wi)).spec_btdf;
                let weight = (base_weight * t) / p_lobe;
                return Some(BsdfSample {
                    weight,
                    wi,
                    pdf: p_lobe,
                    flags: BsdfFlags::DELTA | BsdfFlags::TRANSMISSION,
                    eta: 1.0,
                    wavelength_lock: None,
                });
            }
            let (alpha_x, alpha_y) = self.thin_wall_transmission_alpha_xy();
            let wm = sample_wm_bounded_vndf(wo, alpha_x, alpha_y, us)?;
            let wi_reflect = reflect_local(wo, wm);
            if wi_reflect.z <= 0.0 {
                return None;
            }
            let wi = Vec3::new(wi_reflect.x, wi_reflect.y, -wi_reflect.z);
            let base_weight = self.layer_weights(wo, Some(wi)).spec_btdf;
            let pdf_lobe_value = self.pdf_specular_btdf(wo, wi);
            let pdf_total = p_lobe * pdf_lobe_value;
            if pdf_total <= 0.0 {
                return None;
            }
            let f = self.eval_thin_wall_spec_btdf(wo, wi);
            let weight = base_weight * f * wi.z.abs() / pdf_total;
            return Some(BsdfSample {
                weight,
                wi,
                pdf: pdf_total,
                flags: BsdfFlags::GLOSSY | BsdfFlags::TRANSMISSION,
                eta: 1.0,
                wavelength_lock: None,
            });
        }

        if self.dispersion_rgb_sharing_active() && self.p.wavelength_lock.is_none() {
            return self.sample_spec_btdf_rgb_shared(wo, us, p_lobe, u_aux);
        }

        let dispersion_active = self.p.transmission_dispersion_abbe > 0.0;
        let (eta_used, dispersion_basis, fresh_lock) = if dispersion_active {
            if let Some(lambda) = self.p.wavelength_lock {
                let eta_lambda = cauchy_ior(
                    lambda,
                    self.p.transmission_eta,
                    self.p.transmission_dispersion_abbe,
                );
                (eta_lambda, Vec3::ONE, None)
            } else if self.p.front_face {
                let (lambda, basis) =
                    sample_dispersion_wavelength_weighted(u_aux, self.p.path_throughput);
                let eta_lambda = cauchy_ior(
                    lambda,
                    self.p.transmission_eta,
                    self.p.transmission_dispersion_abbe,
                );
                (eta_lambda, basis, Some(lambda))
            } else {
                (self.p.transmission_eta, Vec3::ONE, None)
            }
        } else {
            (self.p.transmission_eta, Vec3::ONE, None)
        };
        let eta_fresnel = match fresh_lock.or(self.p.wavelength_lock) {
            Some(lambda) if self.p.transmission_dispersion_abbe > 0.0 => cauchy_ior(
                lambda,
                self.p.specular_eta,
                self.p.transmission_dispersion_abbe,
            ),
            _ => self.p.specular_eta,
        };

        let eta_rel = if self.p.front_face {
            1.0 / eta_used
        } else {
            eta_used
        };
        if self.spec_btdf_is_smooth() {
            let wi = refract(wo, eta_rel)?;
            let f = self.base_dielectric_fresnel(wo.z.abs(), eta_used, eta_fresnel);
            let scale = 1.0 / (eta_rel * eta_rel);
            let base_weight = self.layer_weights(wo, Some(wi)).spec_btdf;
            let weight = (base_weight * dispersion_basis * (1.0 - f) * scale) / p_lobe;
            return Some(BsdfSample {
                weight,
                wi,
                pdf: p_lobe,
                flags: BsdfFlags::DELTA | BsdfFlags::TRANSMISSION,
                eta: eta_rel,
                wavelength_lock: fresh_lock,
            });
        }

        let ms = self.dielectric_ms_params(
            self.p.transmission_alpha_x,
            self.p.transmission_alpha_y,
            eta_fresnel,
        );
        let weights = self.dielectric_transmission_lobe_weights(&ms);
        let ss_probability = weights.ss / weights.total;
        if us.x >= ss_probability {
            let u = if ss_probability < 1.0 {
                (us.x - ss_probability) / (1.0 - ss_probability)
            } else {
                0.0
            };
            let wi_up = sample_cosine_weighted_hemisphere(Vec2::new(u, us.y));
            let wi = Vec3::new(wi_up.x, wi_up.y, -wi_up.z);
            return self.finalize_rough_sample_with_wavelength(
                wo,
                wi,
                BsdfFlags::GLOSSY | BsdfFlags::TRANSMISSION,
                eta_rel,
                fresh_lock,
            );
        }
        let u = if ss_probability > 0.0 {
            us.x / ss_probability
        } else {
            0.0
        };
        let wm = sample_wm_vndf(
            wo,
            self.p.transmission_alpha_x,
            self.p.transmission_alpha_y,
            Vec2::new(u, us.y),
        )?;
        let cos_wo_wm = wo.dot(wm);
        if cos_wo_wm <= 0.0 {
            return None;
        }
        let wi = refract_about(wo, wm, eta_rel)?;
        if wi.z >= 0.0 {
            return None;
        }
        let cos_wi_wm = wi.dot(wm);
        let den = cos_wi_wm + eta_rel * cos_wo_wm;
        if den.abs() < 1.0e-6 {
            return None;
        }
        self.finalize_rough_sample_with_wavelength(
            wo,
            wi,
            BsdfFlags::GLOSSY | BsdfFlags::TRANSMISSION,
            eta_rel,
            fresh_lock,
        )
    }

    fn sample_spec_btdf_rgb_shared(
        &self,
        wo: Vec3,
        us: Vec2,
        p_lobe: f32,
        u_channel: f32,
    ) -> Option<BsdfSample> {
        let channel = self.pick_dispersion_channel(u_channel);
        if channel.probability <= 0.0 {
            return None;
        }
        let eta_used = cauchy_ior(
            channel.lambda_nm,
            self.p.transmission_eta,
            self.p.transmission_dispersion_abbe,
        );
        let eta_fresnel = cauchy_ior(
            channel.lambda_nm,
            self.p.specular_eta,
            self.p.transmission_dispersion_abbe,
        );
        let eta_rel = if self.p.front_face {
            1.0 / eta_used
        } else {
            eta_used
        };

        let ms = self.dielectric_ms_params(
            self.p.transmission_alpha_x,
            self.p.transmission_alpha_y,
            eta_fresnel,
        );
        let weights = self.dielectric_transmission_lobe_weights(&ms);
        let ss_probability = weights.ss / weights.total;
        let wi = if us.x >= ss_probability {
            let u = if ss_probability < 1.0 {
                (us.x - ss_probability) / (1.0 - ss_probability)
            } else {
                0.0
            };
            let wi_up = sample_cosine_weighted_hemisphere(Vec2::new(u, us.y));
            Vec3::new(wi_up.x, wi_up.y, -wi_up.z)
        } else {
            let u = if ss_probability > 0.0 {
                us.x / ss_probability
            } else {
                0.0
            };
            let wm = sample_wm_vndf(
                wo,
                self.p.transmission_alpha_x,
                self.p.transmission_alpha_y,
                Vec2::new(u, us.y),
            )?;
            let cos_wo_wm = wo.dot(wm);
            if cos_wo_wm <= 0.0 {
                return None;
            }
            let wi = refract_about(wo, wm, eta_rel)?;
            if wi.z >= 0.0 {
                return None;
            }
            wi
        };
        if wi.z >= 0.0 {
            return None;
        }
        let base_weight = self.layer_weights(wo, Some(wi)).spec_btdf;
        let pdf_lobe_value = self.pdf_specular_btdf_rgb_mixture(wo, wi);
        let pdf_total = p_lobe * pdf_lobe_value;
        if pdf_total <= 0.0 {
            return None;
        }

        let mut f_rgb = Vec3::ZERO;
        for eval_channel in self.dispersion_channels() {
            if eval_channel.probability <= 0.0 {
                continue;
            }
            let eta = cauchy_ior(
                eval_channel.lambda_nm,
                self.p.transmission_eta,
                self.p.transmission_dispersion_abbe,
            );
            let eta_fresnel = cauchy_ior(
                eval_channel.lambda_nm,
                self.p.specular_eta,
                self.p.transmission_dispersion_abbe,
            );
            let eta_rel_channel = if self.p.front_face { 1.0 / eta } else { eta };
            let ss = self
                .eval_spec_btdf_scalar_with_eta(wo, wi, eta_rel_channel)
                .map(|eval| {
                    let f = self.spec_btdf_fresnel_with_eta(eval.cos_wo_wm, eta, eta_fresnel);
                    (1.0 - f.dot(eval_channel.color)).max(0.0) * eval.scalar
                })
                .unwrap_or(0.0);
            let ms = self.eval_dielectric_transmission_ms(
                wo,
                wi,
                self.p.transmission_alpha_x,
                self.p.transmission_alpha_y,
                eta_fresnel,
            );
            f_rgb += eval_channel.color * (ss + ms);
        }
        let weight = base_weight * f_rgb * (wi.z.abs() / pdf_total);

        Some(BsdfSample {
            weight,
            wi,
            pdf: pdf_total,
            flags: BsdfFlags::GLOSSY | BsdfFlags::TRANSMISSION,
            eta: eta_rel,
            wavelength_lock: Some(channel.lambda_nm),
        })
    }

    fn sample_fuzz(&self, wo: Vec3, us: Vec2) -> Option<BsdfSample> {
        let wo_f = self.to_fuzz(wo, Vec3::Z).0;
        let wi_f = OpenPbrFuzzBsdf::new(self.p.fuzz_roughness).sample(wo_f, us)?;
        let wi = self.fuzz_to_base(wi_f);
        self.finalize_rough_sample(wo, wi, BsdfFlags::GLOSSY | BsdfFlags::REFLECTION, 1.0)
    }

    fn sample_diff_brdf(&self, wo: Vec3, us: Vec2) -> Option<BsdfSample> {
        let bsdf = EonBsdf::new(Vec3::ONE, self.p.diffuse_roughness);
        let s = bsdf.sample(wo, us)?;
        self.finalize_rough_sample(wo, s.wi, s.flags, s.eta)
    }

    fn sample_diff_btdf(&self, wo: Vec3, us: Vec2) -> Option<BsdfSample> {
        if !self.p.thin_walled {
            return None;
        }
        let bsdf = EonBsdf::new(Vec3::ONE, self.p.diffuse_roughness);
        let s = bsdf.sample(wo, us)?;
        let wi = Vec3::new(s.wi.x, s.wi.y, -s.wi.z);
        self.finalize_rough_sample(wo, wi, BsdfFlags::DIFFUSE | BsdfFlags::TRANSMISSION, 1.0)
    }

    fn finalize_rough_sample(
        &self,
        wo: Vec3,
        wi: Vec3,
        flags: BsdfFlags,
        eta: f32,
    ) -> Option<BsdfSample> {
        self.finalize_rough_sample_with_wavelength(wo, wi, flags, eta, None)
    }

    fn finalize_rough_sample_with_wavelength(
        &self,
        wo: Vec3,
        wi: Vec3,
        flags: BsdfFlags,
        eta: f32,
        wavelength_lock: Option<f32>,
    ) -> Option<BsdfSample> {
        let pdf = self.pdf(wo, wi);
        if pdf <= 0.0 {
            return None;
        }
        let f = self.eval(wo, wi);
        if f.length_squared() == 0.0 {
            return None;
        }
        let cos = wi.z.abs();
        if cos <= 0.0 {
            return None;
        }
        Some(BsdfSample {
            weight: f * (cos / pdf),
            wi,
            pdf,
            flags,
            wavelength_lock,
            eta,
        })
    }
}

impl LobeProbs {
    fn lobe(&self, lobe: ChosenLobe) -> f32 {
        match lobe {
            ChosenLobe::Coat => self.coat,
            ChosenLobe::Metal => self.metal,
            ChosenLobe::SpecBrdf => self.spec_brdf,
            ChosenLobe::SpecBtdf => self.spec_btdf,
            ChosenLobe::Fuzz => self.fuzz,
            ChosenLobe::DiffBrdf => self.diff_brdf,
            ChosenLobe::DiffBtdf => self.diff_btdf,
        }
    }
}

fn pick_lobe(probs: &LobeProbs, mut u: f32) -> ChosenLobe {
    if u < probs.coat {
        return ChosenLobe::Coat;
    }
    u -= probs.coat;
    if u < probs.metal {
        return ChosenLobe::Metal;
    }
    u -= probs.metal;
    if u < probs.spec_brdf {
        return ChosenLobe::SpecBrdf;
    }
    u -= probs.spec_brdf;
    if u < probs.spec_btdf {
        return ChosenLobe::SpecBtdf;
    }
    u -= probs.spec_btdf;
    if u < probs.fuzz {
        return ChosenLobe::Fuzz;
    }
    u -= probs.fuzz;
    if u < probs.diff_brdf {
        return ChosenLobe::DiffBrdf;
    }
    ChosenLobe::DiffBtdf
}

fn refract_about(wo: Vec3, wm: Vec3, eta: f32) -> Option<Vec3> {
    let cos_o = wo.dot(wm);
    if cos_o <= 0.0 {
        return None;
    }
    let sin2_t = eta * eta * (1.0 - cos_o * cos_o).max(0.0);
    if sin2_t >= 1.0 {
        return None;
    }
    let cos_t = (1.0 - sin2_t).max(0.0).sqrt();
    let wi = (-eta * wo + (eta * cos_o - cos_t) * wm).normalize_or_zero();
    if wi.length_squared() == 0.0 {
        return None;
    }
    Some(wi)
}

fn phi_of(v: Vec3) -> f32 {
    let p = v.y.atan2(v.x);
    if p < 0.0 { p + 2.0 * PI } else { p }
}

fn rotate_about_z(v: Vec3, angle: f32) -> Vec3 {
    let (s, c) = angle.sin_cos();
    Vec3::new(c * v.x - s * v.y, s * v.x + c * v.y, v.z)
}

fn zeltner_dir_albedo(ndot_v: f32, roughness: f32) -> f32 {
    let x = ndot_v.clamp(0.0, 1.0);
    let y = roughness.clamp(0.01, 1.0);
    let s = y * (0.020_660_7 + 1.584_91 * y) / (0.037_942_4 + y * (1.322_27 + y));
    let m = y * (-0.193_854 + y * (-1.148_85 + y * (1.793_2 - 0.959_43 * y * y))) / (0.046_391 + y);
    let o =
        y * (0.000_654_023 + (-0.020_781_8 + 0.119_681 * y) * y) / (1.262_64 + y * (-1.920_21 + y));
    (-0.5 * sqr((x - m) / s)).exp() / (s * (2.0 * PI).sqrt()) + o
}

fn zeltner_ltc_a_inv(ndot_v: f32, roughness: f32) -> f32 {
    let x = ndot_v.clamp(0.0, 1.0);
    let y = roughness.clamp(0.01, 1.0);
    (2.581_26 * x + 0.813_703 * y) * y / (1.0 + 0.310_327 * x * x + 2.609_94 * x * y)
}

fn zeltner_ltc_b_inv(ndot_v: f32, roughness: f32) -> f32 {
    let x = ndot_v.clamp(0.0, 1.0);
    let y = roughness.clamp(0.01, 1.0);
    (1.0 - x).sqrt() * (y - 1.0) * y * y * y
        / (0.000_025_405_3 + 1.712_28 * x - 1.715_06 * x * y + 1.341_74 * y * y)
}

fn sqr(v: f32) -> f32 {
    v * v
}

fn eval_ltc(wi: Vec3, a_inv: f32, b_inv: f32) -> f32 {
    let mut wo_orig = Vec3::new(a_inv * wi.x + b_inv * wi.z, a_inv * wi.y, wi.z);
    let length = wo_orig.length();
    if length <= 0.0 {
        return 0.0;
    }
    wo_orig /= length;
    if wo_orig.z <= 0.0 {
        return 0.0;
    }
    let det = a_inv * a_inv;
    let jacobian = det / (length * length * length);
    (wo_orig.z / PI) * jacobian
}

fn sample_ltc(a_inv: f32, b_inv: f32, u: Vec2) -> Vec3 {
    let phi = 2.0 * PI * u.x;
    let cos_theta = u.y.sqrt();
    let sin_theta = (1.0 - u.y).max(0.0).sqrt();
    let wo_orig = Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta);
    let wi = Vec3::new(
        wo_orig.x / a_inv - wo_orig.z * b_inv / a_inv,
        wo_orig.y / a_inv,
        wo_orig.z,
    );
    wi.normalize_or_zero()
}

const MS_DENOM_EPS: f32 = 1.0e-6;

fn f_avg_dielectric(eta: f32) -> f32 {
    if eta >= 1.0 {
        ((eta - 1.0) / (4.08567 + 1.00071 * eta)).clamp(0.0, 1.0)
    } else {
        let e = eta.max(MS_DENOM_EPS);
        let v = 0.997118 + 0.1014 * e - 0.965241 * e * e - 0.130607 * e * e * e;
        v.clamp(0.0, 1.0)
    }
}

fn compute_conductor_f_ms(f_avg: Vec3, e_avg: f32) -> Vec3 {
    let one_minus_eavg = (1.0 - e_avg).max(0.0);
    let denom = Vec3::ONE - f_avg * one_minus_eavg;
    let denom_safe = Vec3::new(
        denom.x.max(MS_DENOM_EPS),
        denom.y.max(MS_DENOM_EPS),
        denom.z.max(MS_DENOM_EPS),
    );
    f_avg * f_avg * e_avg / denom_safe
}

fn schlick_vec3(f0: Vec3, cos_theta: f32) -> Vec3 {
    let one_minus = 1.0 - cos_theta.clamp(0.0, 1.0);
    f0 + (Vec3::ONE - f0) * one_minus.powi(5)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn lerp_vec3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a + (b - a) * t
}

fn weighted_fresnel_ratio(cos_theta: f32, eta_ti: f32, eta_ti_prime: f32) -> f32 {
    let eta_ti = eta_ti.max(1.0e-4);
    let eta_ti_prime = eta_ti_prime.max(1.0e-4);
    if eta_ti_prime >= 1.0 {
        return fresnel_dielectric(cos_theta, 1.0, eta_ti_prime);
    }
    let mu2_t = 1.0 - (1.0 - cos_theta.clamp(0.0, 1.0).powi(2)) / (eta_ti * eta_ti);
    if mu2_t <= 0.0 {
        return 1.0;
    }
    fresnel_dielectric(mu2_t.sqrt(), 1.0, 1.0 / eta_ti_prime)
}

fn refracted_cos_in_dielectric(mu: f32, eta: f32) -> f32 {
    let sin2_t = (1.0 - mu.clamp(0.0, 1.0).powi(2)).max(0.0) / (eta * eta).max(MS_DENOM_EPS);
    (1.0 - sin2_t).max(0.0).sqrt()
}

fn vec3_powf(v: Vec3, e: f32) -> Vec3 {
    Vec3::new(
        v.x.max(0.0).powf(e),
        v.y.max(0.0).powf(e),
        v.z.max(0.0).powf(e),
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use glam::{Vec2, Vec3};

    use crate::{
        bsdf::{
            BsdfFlags, ConductorGgxEnergyCompensationLut, DielectricGgxDirectionalAlbedoLut,
            DielectricGgxEnergyCompensationLut, artist_friendly_complex_ior,
        },
        math::refract,
    };

    use super::{COS_82, OpenPbrBsdf, OpenPbrBsdfParams, schlick_vec3};

    fn default_params() -> OpenPbrBsdfParams {
        let (n, k) = artist_friendly_complex_ior(Vec3::splat(0.8), Vec3::ONE);
        OpenPbrBsdfParams {
            base_color: Vec3::new(0.8, 0.6, 0.4),
            base: 0.8,
            specular: 1.0,
            specular_color: Vec3::ONE,
            specular_alpha_x: 0.04,
            specular_alpha_y: 0.04,
            specular_eta: 1.5,
            transmission_eta: 1.5,
            metalness: 0.0,
            metal_n: n,
            metal_k: k,
            coat: 0.0,
            coat_color: Vec3::ONE,
            coat_darkening: Vec3::ONE,
            coat_alpha_x: 0.01,
            coat_alpha_y: 0.01,
            coat_eta: 1.5,
            fuzz: 0.0,
            fuzz_color: Vec3::ONE,
            fuzz_roughness: 0.3,
            transmission: 0.0,
            transmission_color: Vec3::ONE,
            transmission_alpha_x: 0.04,
            transmission_alpha_y: 0.04,
            transmission_dispersion_abbe: 0.0,
            diffuse_roughness: 0.0,
            subsurface: 0.0,
            subsurface_color: Vec3::ONE,
            thin_walled: false,
            thin_film_weight: 0.0,
            thin_film_thickness: 0.0,
            thin_film_ior: 1.5,
            front_face: true,
            coat_basis_in_base: None,
            fuzz_basis_in_base: None,
            path_throughput: Vec3::ONE,
            wavelength_lock: None,
        }
    }

    fn test_bsdf(params: OpenPbrBsdfParams) -> OpenPbrBsdf {
        let spec_lut = Arc::new(DielectricGgxDirectionalAlbedoLut::constant_for_tests(
            1.5, 0.04,
        ));
        let coat_lut = Arc::new(DielectricGgxDirectionalAlbedoLut::constant_for_tests(
            1.5, 0.04,
        ));
        let conductor_ec_lut = Arc::new(ConductorGgxEnergyCompensationLut::constant_for_tests(
            0.9, 0.9,
        ));
        let dielectric_ec_lut = Arc::new(DielectricGgxEnergyCompensationLut::constant_for_tests(
            0.9, 0.9,
        ));
        OpenPbrBsdf::new(
            params,
            spec_lut,
            coat_lut,
            conductor_ec_lut,
            dielectric_ec_lut,
        )
    }

    fn test_bsdf_with_real_luts(params: OpenPbrBsdfParams) -> OpenPbrBsdf {
        OpenPbrBsdf::new(
            params,
            Arc::new(DielectricGgxDirectionalAlbedoLut::build_for_tests(1.5)),
            Arc::new(DielectricGgxDirectionalAlbedoLut::build_for_tests(1.5)),
            Arc::new(ConductorGgxEnergyCompensationLut::build_for_tests()),
            Arc::new(DielectricGgxEnergyCompensationLut::build_for_tests()),
        )
    }

    fn test_bsdf_with_real_luts_for_spec_eta(params: OpenPbrBsdfParams) -> OpenPbrBsdf {
        let spec_eta = params.specular_eta;
        OpenPbrBsdf::new(
            params,
            Arc::new(DielectricGgxDirectionalAlbedoLut::build_for_tests(spec_eta)),
            Arc::new(DielectricGgxDirectionalAlbedoLut::build_for_tests(1.5)),
            Arc::new(ConductorGgxEnergyCompensationLut::build_for_tests()),
            Arc::new(DielectricGgxEnergyCompensationLut::build_for_tests()),
        )
    }

    fn modulated_eta_for_specular_weight(eta: f32, specular_weight: f32) -> f32 {
        let f0 = ((eta - 1.0) / (eta + 1.0)).powi(2);
        let eps = (specular_weight * f0).clamp(0.0, 0.999_99).sqrt() * (eta - 1.0).signum();
        ((1.0 + eps) / (1.0 - eps).max(1.0e-6)).max(1.0e-4)
    }

    use crate::bsdf::integrate_upper_hemisphere_vec3;

    #[test]
    fn default_diffuse_dominant_evaluates_finite() {
        let bsdf = test_bsdf(default_params());
        let f = bsdf.eval(Vec3::Z, Vec3::new(0.2, 0.3, 0.9327379).normalize());
        assert!(f.is_finite());
    }

    #[test]
    fn rough_dielectric_specular_does_not_exceed_white_furnace_energy() {
        let mut params = default_params();
        params.base = 0.0;
        params.base_color = Vec3::ZERO;
        params.specular = 1.0;
        params.specular_color = Vec3::ONE;
        params.specular_eta = 1.5;
        params.transmission_eta = 1.5;
        params.transmission = 0.0;
        params.metalness = 0.0;

        for roughness in [0.35_f32, 0.65, 0.88, 1.0] {
            let alpha = roughness * roughness;
            params.specular_alpha_x = alpha;
            params.specular_alpha_y = alpha;
            let bsdf = test_bsdf_with_real_luts(params.clone());
            let wo = Vec3::new(0.25, -0.15, 0.956_556).normalize();
            let energy = integrate_upper_hemisphere_vec3(|wi| bsdf.eval(wo, wi) * wi.z);

            assert!(energy.is_finite());
            assert!(
                energy.max_element() <= 1.0 + 1.0e-2,
                "roughness={roughness}, dielectric specular energy={energy:?}"
            );
        }
    }

    #[test]
    fn rough_openpbr_diffuse_plus_specular_does_not_gain_energy() {
        let mut params = default_params();
        params.base = 1.0;
        params.base_color = Vec3::ONE;
        params.specular = 1.0;
        params.specular_color = Vec3::ONE;
        params.specular_eta = 1.5;
        params.transmission_eta = 1.5;
        params.transmission = 0.0;
        params.metalness = 0.0;
        params.diffuse_roughness = 1.0;

        for roughness in [0.35_f32, 0.65, 0.88, 1.0] {
            let alpha = roughness * roughness;
            params.specular_alpha_x = alpha;
            params.specular_alpha_y = alpha;
            let bsdf = test_bsdf_with_real_luts(params.clone());
            let wo = Vec3::new(0.25, -0.15, 0.956_556).normalize();
            let energy = integrate_upper_hemisphere_vec3(|wi| bsdf.eval(wo, wi) * wi.z);

            assert!(energy.is_finite());
            assert!(
                energy.max_element() <= 1.0 + 1.0e-2,
                "roughness={roughness}, diffuse+specular energy={energy:?}"
            );
        }
    }

    #[test]
    fn nonmetal_rough_specular_stays_near_dielectric_reflectance() {
        let mut params = default_params();
        params.base = 0.0;
        params.base_color = Vec3::ZERO;
        params.specular = 1.0;
        params.specular_color = Vec3::ONE;
        params.specular_eta = 1.5;
        params.transmission_eta = 1.5;
        params.transmission = 0.0;
        params.metalness = 0.0;
        params.diffuse_roughness = 1.0;
        params.specular_alpha_x = 0.88_f32 * 0.88;
        params.specular_alpha_y = 0.88_f32 * 0.88;
        let bsdf = test_bsdf_with_real_luts(params.clone());
        let wo = Vec3::new(0.25, -0.15, 0.956_556).normalize();
        let spec_energy = integrate_upper_hemisphere_vec3(|wi| bsdf.eval(wo, wi) * wi.z);

        params.base = 1.0;
        params.base_color = Vec3::new(0.28, 0.04, 0.1);
        let bsdf = test_bsdf_with_real_luts(params);
        let total_energy = integrate_upper_hemisphere_vec3(|wi| bsdf.eval(wo, wi) * wi.z);

        assert!(
            (0.005..=0.06).contains(&spec_energy.x)
                && spec_energy.abs_diff_eq(Vec3::splat(spec_energy.x), 1.0e-5),
            "rough dielectric specular should remain a small neutral dielectric contribution, got {spec_energy:?}"
        );
        assert!(
            total_energy.max_element() <= 1.0 + 1.0e-2,
            "red diffuse + rough dielectric specular should not gain energy, got {total_energy:?}"
        );
    }

    #[test]
    fn low_weight_nonmetal_rough_specular_remains_low_energy() {
        let mut params = default_params();
        params.base = 0.0;
        params.base_color = Vec3::ZERO;
        params.specular = 1.0;
        params.specular_color = Vec3::ONE;
        params.specular_eta = modulated_eta_for_specular_weight(1.5, 0.01);
        params.transmission_eta = params.specular_eta;
        params.transmission = 0.0;
        params.metalness = 0.0;
        params.diffuse_roughness = 1.0;
        params.specular_alpha_x = 0.88_f32 * 0.88;
        params.specular_alpha_y = 0.88_f32 * 0.88;

        let bsdf = test_bsdf_with_real_luts_for_spec_eta(params);
        let wo = Vec3::new(0.25, -0.15, 0.956_556).normalize();
        let spec_energy = integrate_upper_hemisphere_vec3(|wi| bsdf.eval(wo, wi) * wi.z);

        assert!(
            spec_energy.max_element() <= 0.004,
            "specular_weight=0.01 equivalent rough dielectric should stay tiny, got {spec_energy:?}"
        );
    }

    #[test]
    fn dielectric_transmission_ms_ratio_satisfies_reciprocity_constraint() {
        let bsdf = test_bsdf_with_real_luts(default_params());

        for eta in [1.1_f32, 1.5, 2.4] {
            for roughness in [0.35_f32, 0.65, 0.88, 1.0] {
                let alpha = roughness * roughness;
                let entering = bsdf.dielectric_ms_params(alpha, alpha, eta);
                let exiting = bsdf.dielectric_ms_params(alpha, alpha, 1.0 / eta);
                let enter_transmit = (1.0 - entering.ratio_r) / entering.one_minus_e_avg_o;
                let exit_transmit = (1.0 - exiting.ratio_r) * eta * eta / exiting.one_minus_e_avg_o;

                assert!(
                    (enter_transmit - exit_transmit).abs() <= 1.0e-4,
                    "eta={eta}, roughness={roughness}, entering={enter_transmit}, exiting={exit_transmit}"
                );
            }
        }
    }

    #[test]
    fn zero_weight_nonmetal_rough_specular_vanishes() {
        let mut params = default_params();
        params.base = 0.0;
        params.base_color = Vec3::ZERO;
        params.specular = 0.0;
        params.specular_color = Vec3::ONE;
        params.specular_eta = modulated_eta_for_specular_weight(1.5, 0.0);
        params.transmission_eta = params.specular_eta;
        params.transmission = 0.0;
        params.metalness = 0.0;
        params.specular_alpha_x = 0.88_f32 * 0.88;
        params.specular_alpha_y = 0.88_f32 * 0.88;

        let bsdf = test_bsdf_with_real_luts_for_spec_eta(params);
        let wo = Vec3::new(0.25, -0.15, 0.956_556).normalize();
        let spec_energy = integrate_upper_hemisphere_vec3(|wi| bsdf.eval(wo, wi) * wi.z);

        assert!(
            spec_energy.max_element() <= 1.0e-4,
            "specular_weight=0 equivalent rough dielectric should vanish, got {spec_energy:?}"
        );
    }

    #[test]
    fn rough_metal_specular_does_not_exceed_white_furnace_energy() {
        let mut params = default_params();
        params.base = 1.0;
        params.base_color = Vec3::ONE;
        params.specular = 1.0;
        params.specular_color = Vec3::ONE;
        params.metalness = 1.0;
        params.transmission = 0.0;

        for roughness in [0.35_f32, 0.65, 0.88, 1.0] {
            let alpha = roughness * roughness;
            params.specular_alpha_x = alpha;
            params.specular_alpha_y = alpha;
            let bsdf = test_bsdf_with_real_luts(params.clone());
            let wo = Vec3::new(0.25, -0.15, 0.956_556).normalize();
            let energy = integrate_upper_hemisphere_vec3(|wi| bsdf.eval(wo, wi) * wi.z);

            assert!(energy.is_finite());
            assert!(
                energy.max_element() <= 1.0 + 1.5e-2,
                "roughness={roughness}, metal specular energy={energy:?}"
            );
        }
    }

    #[test]
    fn pure_metallic_returns_specular_lobe() {
        let mut params = default_params();
        params.metalness = 1.0;
        params.specular_alpha_x = 0.2;
        params.specular_alpha_y = 0.2;
        let bsdf = test_bsdf(params);
        let mut rng = crate::sampler::AuxRng::from_seed(0);
        let randoms = crate::sampler::MaterialSampleRandoms::from_aux_rng(&mut rng);
        let sample = bsdf
            .sample(Vec3::new(0.2, -0.1, 0.9746794).normalize(), &randoms)
            .unwrap();
        assert!(sample.flags.contains(BsdfFlags::REFLECTION));
    }

    #[test]
    fn metallic_fresnel_uses_f82_tint() {
        let mut params = default_params();
        params.base = 0.8;
        params.base_color = Vec3::new(0.9, 0.45, 0.2);
        params.specular = 0.7;
        params.specular_color = Vec3::new(0.4, 1.0, 0.25);
        let bsdf = test_bsdf(params.clone());

        let f0 = params.specular * params.base * params.base_color;
        assert!(bsdf.metal_fresnel(1.0).abs_diff_eq(f0, 1.0e-6));

        let schlick_82 = schlick_vec3(params.base * params.base_color, COS_82);
        let expected_82 = params.specular * params.specular_color * schlick_82;
        assert!(bsdf.metal_fresnel(COS_82).abs_diff_eq(expected_82, 1.0e-5));
    }

    #[test]
    fn metallic_fresnel_avg_uses_f82_closed_form_without_thin_film() {
        let mut params = default_params();
        params.base = 0.8;
        params.base_color = Vec3::new(0.9, 0.45, 0.2);
        params.specular = 0.7;
        params.specular_color = Vec3::new(0.4, 1.0, 0.25);
        params.thin_film_weight = 0.0;
        let bsdf = test_bsdf(params.clone());

        let f0 = params.base * params.base_color;
        let f_schlick_82 = schlick_vec3(f0, COS_82);
        let f_target_82 = params.specular_color * f_schlick_82;
        let b = (f_schlick_82 - f_target_82) / (COS_82 * (1.0 - COS_82).powi(6));
        let expected = (params.specular * (f0 + (Vec3::ONE - f0) / 21.0 - b / 126.0))
            .clamp(Vec3::ZERO, Vec3::ONE);

        assert!(bsdf.metal_fresnel_avg().abs_diff_eq(expected, 1.0e-6));
    }

    #[test]
    fn thin_walled_specular_btdf_yields_pass_through() {
        let mut params = default_params();
        params.thin_walled = true;
        params.transmission = 1.0;
        params.transmission_alpha_x = 0.0;
        params.transmission_alpha_y = 0.0;
        params.specular = 0.0;
        params.base = 0.0;
        let bsdf = test_bsdf(params);
        let mut rng = crate::sampler::AuxRng::from_seed(0);
        let wo = Vec3::new(0.2, -0.3, 0.9327379).normalize();
        let mut got = false;
        for _ in 0..32 {
            let randoms = crate::sampler::MaterialSampleRandoms::from_aux_rng(&mut rng);
            if let Some(sample) = bsdf.sample(wo, &randoms)
                && sample.flags.contains(BsdfFlags::TRANSMISSION)
            {
                assert!(sample.wi.abs_diff_eq(-wo, 1.0e-5));
                got = true;
                break;
            }
        }
        assert!(got);
    }

    #[test]
    fn rough_dispersion_sample_shares_rgb_lobes() {
        let mut params = default_params();
        params.base = 0.0;
        params.specular = 0.0;
        params.transmission = 1.0;
        params.transmission_alpha_x = 0.45;
        params.transmission_alpha_y = 0.45;
        params.transmission_dispersion_abbe = 20.0;
        params.path_throughput = Vec3::new(0.8, 0.5, 0.3);
        let bsdf = test_bsdf(params);
        let sample = bsdf
            .sample_spec_btdf_rgb_shared(Vec3::Z, glam::Vec2::new(0.37, 0.82), 1.0, 0.41)
            .expect("rough dispersive BTDF should sample");

        assert!(sample.flags.contains(BsdfFlags::TRANSMISSION));
        assert!(sample.wavelength_lock.is_some());
        assert!(sample.weight.x > 0.0);
        assert!(sample.weight.y > 0.0);
        assert!(sample.weight.z > 0.0);
    }

    #[test]
    fn smooth_transmission_eta_controls_refraction_direction() {
        let mut params = default_params();
        params.base = 0.0;
        params.specular = 0.0;
        params.transmission = 1.0;
        params.transmission_alpha_x = 0.0;
        params.transmission_alpha_y = 0.0;
        params.specular_eta = 1.0;
        params.transmission_eta = 1.5;
        let bsdf = test_bsdf(params);
        let wo = Vec3::new(0.35, 0.0, 0.936_75).normalize();
        let expected = refract(wo, 1.0 / 1.5).unwrap();
        let sample = bsdf.sample_spec_btdf(wo, Vec2::ZERO, 1.0, 0.5).unwrap();
        assert!(sample.wi.abs_diff_eq(expected, 1.0e-5));
    }
}
