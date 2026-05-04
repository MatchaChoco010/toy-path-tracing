use std::sync::Arc;

use glam::{Vec2, Vec3};
use rand::{RngExt, rngs::ThreadRng};

use crate::math::{OrthonormalBasis, fresnel_dielectric, refract, sg::luminance};

use super::conductor_complex::fresnel_complex;
use super::dispersion::{cauchy_ior, sample_dispersion_wavelength};
use super::oren_nayar::OrenNayarBsdf;
use super::sheen::SheenBsdf;
use super::smith_ggx::{
    EFFECTIVELY_SMOOTH_ALPHA, ggx_d, ggx_g1, ggx_g2_height_correlated, is_upper_hemisphere,
    pdf_wm_bounded_vndf, pdf_wm_vndf, reflect_local, reflection_half_vector,
    sample_wm_bounded_vndf, sample_wm_vndf,
};
use super::thin_film::{eval_thin_film_conductor, eval_thin_film_dielectric};
use super::{BsdfFlags, BsdfSample, DielectricGgxDirectionalAlbedoLut, SheenDirectionalAlbedoLut};

#[derive(Debug, Clone)]
pub struct StandardSurfaceBsdfParams {
    pub base_color: Vec3,
    pub base: f32,
    pub specular: f32,
    pub specular_color: Vec3,
    pub specular_alpha_x: f32,
    pub specular_alpha_y: f32,
    pub specular_eta: f32,
    pub metalness: f32,
    pub metal_n: Vec3,
    pub metal_k: Vec3,
    pub coat: f32,
    pub coat_color: Vec3,
    pub coat_alpha_x: f32,
    pub coat_alpha_y: f32,
    pub coat_eta: f32,
    pub sheen: f32,
    pub sheen_color: Vec3,
    pub sheen_roughness: f32,
    pub transmission: f32,
    pub transmission_color: Vec3,
    pub transmission_alpha_x: f32,
    pub transmission_alpha_y: f32,
    pub transmission_dispersion_abbe: f32,
    pub diffuse_roughness: f32,
    pub subsurface: f32,
    pub subsurface_color: Vec3,
    pub thin_walled: bool,
    pub thin_film_thickness: f32,
    pub thin_film_ior: f32,
    pub front_face: bool,
    pub coat_basis_in_base: Option<OrthonormalBasis>,
    pub wavelength_lock: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct StandardSurfaceBsdf {
    p: StandardSurfaceBsdfParams,
    spec_lut: Arc<DielectricGgxDirectionalAlbedoLut>,
    coat_lut: Arc<DielectricGgxDirectionalAlbedoLut>,
    sheen_lut: Arc<SheenDirectionalAlbedoLut>,
}

#[derive(Debug, Clone, Copy)]
struct LayerWeights {
    coat_amp: f32,
    metal: Vec3,
    spec_brdf: Vec3,
    spec_btdf: Vec3,
    sheen: Vec3,
    diff_brdf: Vec3,
    diff_btdf: Vec3,
}

#[derive(Debug, Clone, Copy)]
struct LobeProbs {
    coat: f32,
    metal: f32,
    spec_brdf: f32,
    spec_btdf: f32,
    sheen: f32,
    diff_brdf: f32,
    diff_btdf: f32,
    total: f32,
}

#[derive(Debug, Clone, Copy)]
enum ChosenLobe {
    Coat,
    Metal,
    SpecBrdf,
    SpecBtdf,
    Sheen,
    DiffBrdf,
    DiffBtdf,
}

impl StandardSurfaceBsdf {
    pub(crate) fn new(
        params: StandardSurfaceBsdfParams,
        spec_lut: Arc<DielectricGgxDirectionalAlbedoLut>,
        coat_lut: Arc<DielectricGgxDirectionalAlbedoLut>,
        sheen_lut: Arc<SheenDirectionalAlbedoLut>,
    ) -> Self {
        Self {
            p: params,
            spec_lut,
            coat_lut,
            sheen_lut,
        }
    }

    pub fn eval(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        if !is_upper_hemisphere(wo) {
            return Vec3::ZERO;
        }
        let weights = self.layer_weights(wo);
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
        if weights.spec_btdf.length_squared() > 0.0
            && !self.p.thin_walled
            && !self.spec_btdf_is_smooth()
        {
            total += weights.spec_btdf * self.eval_spec_btdf(wo, wi);
        }
        if weights.sheen.length_squared() > 0.0 {
            total += weights.sheen * self.eval_sheen(wo, wi);
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
            pdf += (probs.metal / probs.total) * self.pdf_specular_brdf(wo, wi);
        }
        if probs.spec_brdf > 0.0 && !self.spec_brdf_is_smooth() {
            pdf += (probs.spec_brdf / probs.total) * self.pdf_specular_brdf(wo, wi);
        }
        if probs.spec_btdf > 0.0 && !self.p.thin_walled && !self.spec_btdf_is_smooth() {
            pdf += (probs.spec_btdf / probs.total) * self.pdf_specular_btdf(wo, wi);
        }
        if probs.sheen > 0.0 {
            pdf += (probs.sheen / probs.total) * self.pdf_sheen(wo, wi);
        }
        if probs.diff_brdf > 0.0 {
            pdf += (probs.diff_brdf / probs.total) * self.pdf_diff_brdf(wo, wi);
        }
        if probs.diff_btdf > 0.0 {
            pdf += (probs.diff_btdf / probs.total) * self.pdf_diff_btdf(wo, wi);
        }
        pdf
    }

    pub fn sample(&self, wo: Vec3, rng: &mut ThreadRng) -> Option<BsdfSample> {
        if !is_upper_hemisphere(wo) {
            return None;
        }
        let probs = self.lobe_probabilities(wo);
        if probs.total <= 0.0 {
            return None;
        }
        let chosen = pick_lobe(&probs, rng.random::<f32>() * probs.total);
        let p_lobe = probs.lobe(chosen) / probs.total;
        if p_lobe <= 0.0 {
            return None;
        }

        let us = Vec2::new(rng.random::<f32>(), rng.random::<f32>());
        match chosen {
            ChosenLobe::Coat => self.sample_coat(wo, us, p_lobe),
            ChosenLobe::Metal => self.sample_metal(wo, us, p_lobe),
            ChosenLobe::SpecBrdf => self.sample_spec_brdf(wo, us, p_lobe),
            ChosenLobe::SpecBtdf => self.sample_spec_btdf(wo, us, p_lobe, rng),
            ChosenLobe::Sheen => self.sample_sheen(wo, us),
            ChosenLobe::DiffBrdf => self.sample_diff_brdf(wo, us),
            ChosenLobe::DiffBtdf => self.sample_diff_btdf(wo, us),
        }
    }

    fn layer_weights(&self, wo: Vec3) -> LayerWeights {
        let one = Vec3::ONE;
        let coat_amp = self.p.coat;
        let e_coat = if coat_amp > 0.0 {
            self.lookup_coat_albedo(wo)
        } else {
            0.0
        };
        let under_coat = if coat_amp > 0.0 {
            one * (1.0 - coat_amp) + self.p.coat_color * (1.0 - e_coat) * coat_amp
        } else {
            one
        };

        let metal = self.p.metalness * under_coat;
        let below_metal = (1.0 - self.p.metalness) * under_coat;

        let e_spec = self.lookup_spec_albedo(wo);
        let spec_amp_rgb = self.p.specular * self.p.specular_color;
        let spec_brdf_w = below_metal * spec_amp_rgb;
        let leakage_spec = (one - spec_amp_rgb * e_spec).max(Vec3::ZERO);
        let below_spec = below_metal * leakage_spec;

        let trans = self.p.transmission;
        let spec_btdf_w = below_spec * trans * self.p.transmission_color;
        let below_trans = below_spec * (1.0 - trans);

        let e_sheen = self.lookup_sheen_albedo(wo);
        let sheen_amp_rgb = self.p.sheen * self.p.sheen_color;
        let sheen_w = below_trans * sheen_amp_rgb;
        let leakage_sheen = (1.0 - self.p.sheen * e_sheen).max(0.0);
        let below_sheen = below_trans * leakage_sheen;

        let thin_factor = if self.p.thin_walled { 1.0 } else { 0.0 };
        let diff_brdf_w =
            below_sheen * ((1.0 - self.p.subsurface) * self.p.base) * self.p.base_color;
        let diff_btdf_w = below_sheen * (self.p.subsurface * thin_factor) * self.p.subsurface_color;

        LayerWeights {
            coat_amp,
            metal,
            spec_brdf: spec_brdf_w,
            spec_btdf: spec_btdf_w,
            sheen: sheen_w,
            diff_brdf: diff_brdf_w,
            diff_btdf: diff_btdf_w,
        }
    }

    fn lobe_probabilities(&self, wo: Vec3) -> LobeProbs {
        let weights = self.layer_weights(wo);
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
        let sheen = luminance(weights.sheen).max(0.0);
        let diff_brdf = luminance(weights.diff_brdf).max(0.0);
        let diff_btdf = luminance(weights.diff_btdf).max(0.0);

        let total = coat + metal + spec_brdf + spec_btdf + sheen + diff_brdf + diff_btdf;
        LobeProbs {
            coat,
            metal,
            spec_brdf,
            spec_btdf,
            sheen,
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
        self.coat_lut.lookup(wo_c, self.coat_roughness_proxy(), 0.0)
    }

    fn lookup_spec_albedo(&self, wo: Vec3) -> f32 {
        if wo.z <= 0.0 {
            return 0.0;
        }
        self.spec_lut.lookup(wo, self.spec_roughness_proxy(), 0.0)
    }

    fn lookup_sheen_albedo(&self, wo: Vec3) -> f32 {
        if self.p.sheen <= 0.0 {
            return 0.0;
        }
        self.sheen_lut.lookup(wo.z.max(0.0), self.p.sheen_roughness)
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
        Vec3::splat(d * g * f / (4.0 * cos_o * cos_i))
    }

    fn pdf_coat(&self, wo_c: Vec3, wi_c: Vec3) -> f32 {
        if wo_c.z <= 0.0 || wi_c.z <= 0.0 {
            return 0.0;
        }
        let Some(wm) = reflection_half_vector(wo_c, wi_c) else {
            return 0.0;
        };
        let pdf_wm = pdf_wm_bounded_vndf(wo_c, wm, self.p.coat_alpha_x, self.p.coat_alpha_y);
        let denom = 4.0 * wo_c.dot(wm).abs();
        if denom <= 0.0 {
            return 0.0;
        }
        pdf_wm / denom
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
        f * (d * g / (4.0 * wo.z * wi.z))
    }

    fn metal_fresnel(&self, cos_theta: f32) -> Vec3 {
        if self.p.thin_film_thickness > 0.0 && !self.p.thin_walled {
            eval_thin_film_conductor(
                cos_theta,
                1.0,
                self.p.thin_film_ior,
                self.p.metal_n,
                self.p.metal_k,
                self.p.thin_film_thickness,
            )
        } else {
            fresnel_complex(cos_theta, self.p.metal_n, self.p.metal_k)
        }
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
        let f = self.spec_brdf_fresnel(wo.dot(wm).abs());
        f * (d * g / (4.0 * wo.z * wi.z))
    }

    fn spec_brdf_fresnel(&self, cos_theta: f32) -> Vec3 {
        if self.p.thin_film_thickness > 0.0 && !self.p.thin_walled {
            eval_thin_film_dielectric(
                cos_theta,
                1.0,
                self.p.thin_film_ior,
                self.p.specular_eta,
                self.p.thin_film_thickness,
            )
        } else {
            Vec3::splat(fresnel_dielectric(cos_theta, 1.0, self.p.specular_eta))
        }
    }

    fn pdf_specular_brdf(&self, wo: Vec3, wi: Vec3) -> f32 {
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

    fn eval_spec_btdf(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        if self.p.thin_walled {
            return Vec3::ZERO;
        }
        if wo.z <= 0.0 || wi.z >= 0.0 {
            return Vec3::ZERO;
        }
        let eta_rel = self.transmission_eta_rel();
        let wm_unnorm = eta_rel * wo + wi;
        if wm_unnorm.length_squared() < 1.0e-12 {
            return Vec3::ZERO;
        }
        let mut wm = wm_unnorm.normalize();
        if wm.z < 0.0 {
            wm = -wm;
        }
        let cos_wo_wm = wo.dot(wm);
        if cos_wo_wm <= 0.0 {
            return Vec3::ZERO;
        }
        let cos_wi_wm = wi.dot(wm);
        let den = cos_wi_wm + eta_rel * cos_wo_wm;
        if den.abs() < 1.0e-6 {
            return Vec3::ZERO;
        }
        let d = ggx_d(wm, self.p.transmission_alpha_x, self.p.transmission_alpha_y);
        let g = ggx_g2_height_correlated(
            wo,
            wi,
            self.p.transmission_alpha_x,
            self.p.transmission_alpha_y,
        );
        if d <= 0.0 || g <= 0.0 {
            return Vec3::ZERO;
        }
        let f_rgb = self.spec_btdf_fresnel(cos_wo_wm);
        let radiance_scale = 1.0 / (eta_rel * eta_rel);
        let one_minus_f = (Vec3::ONE - f_rgb).max(Vec3::ZERO);
        let scalar = d * g * (cos_wi_wm * cos_wo_wm).abs();
        let denom = den * den * wo.z.abs() * wi.z.abs();
        if denom <= 0.0 {
            return Vec3::ZERO;
        }
        one_minus_f * (scalar * radiance_scale / denom)
    }

    fn spec_btdf_fresnel(&self, cos_theta: f32) -> Vec3 {
        if self.p.thin_film_thickness > 0.0 && !self.p.thin_walled {
            eval_thin_film_dielectric(
                cos_theta,
                1.0,
                self.p.thin_film_ior,
                self.transmission_eta_used(),
                self.p.thin_film_thickness,
            )
        } else {
            Vec3::splat(fresnel_dielectric(
                cos_theta,
                self.fresnel_eta_i(),
                self.fresnel_eta_t(),
            ))
        }
    }

    fn pdf_specular_btdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        if self.p.thin_walled {
            return 0.0;
        }
        if wo.z <= 0.0 || wi.z >= 0.0 {
            return 0.0;
        }
        let eta_rel = self.transmission_eta_rel();
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

    fn fresnel_eta_i(&self) -> f32 {
        if self.p.front_face {
            1.0
        } else {
            self.transmission_eta_used()
        }
    }

    fn fresnel_eta_t(&self) -> f32 {
        if self.p.front_face {
            self.transmission_eta_used()
        } else {
            1.0
        }
    }

    fn transmission_eta_used(&self) -> f32 {
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

    fn eval_sheen(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        SheenBsdf::new(Vec3::ONE, self.p.sheen_roughness).eval(wo, wi)
    }

    fn pdf_sheen(&self, wo: Vec3, wi: Vec3) -> f32 {
        SheenBsdf::new(Vec3::ONE, self.p.sheen_roughness).pdf(wo, wi)
    }

    fn eval_diff_brdf(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        OrenNayarBsdf::new(Vec3::ONE, self.p.diffuse_roughness).eval(wo, wi)
    }

    fn pdf_diff_brdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        OrenNayarBsdf::new(Vec3::ONE, self.p.diffuse_roughness).pdf(wo, wi)
    }

    fn eval_diff_btdf(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        if !self.p.thin_walled {
            return Vec3::ZERO;
        }
        if wo.z <= 0.0 || wi.z >= 0.0 {
            return Vec3::ZERO;
        }
        let wi_flipped = Vec3::new(wi.x, wi.y, -wi.z);
        OrenNayarBsdf::new(Vec3::ONE, self.p.diffuse_roughness).eval(wo, wi_flipped)
    }

    fn pdf_diff_btdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        if !self.p.thin_walled {
            return 0.0;
        }
        if wo.z <= 0.0 || wi.z >= 0.0 {
            return 0.0;
        }
        let wi_flipped = Vec3::new(wi.x, wi.y, -wi.z);
        OrenNayarBsdf::new(Vec3::ONE, self.p.diffuse_roughness).pdf(wo, wi_flipped)
    }

    fn sample_coat(&self, wo: Vec3, us: Vec2, p_lobe: f32) -> Option<BsdfSample> {
        let (wo_c, _) = self.to_coat(wo, Vec3::Z);
        if wo_c.z <= 0.0 {
            return None;
        }
        if self.coat_is_smooth() {
            let wi_c = Vec3::new(-wo_c.x, -wo_c.y, wo_c.z);
            let f = fresnel_dielectric(wi_c.z.abs(), 1.0, self.p.coat_eta);
            let weight = Vec3::splat(self.p.coat * f / p_lobe);
            let wi = self.coat_to_base(wi_c);
            return Some(BsdfSample {
                weight,
                wi,
                pdf: p_lobe,
                flags: BsdfFlags::DELTA | BsdfFlags::REFLECTION,
                eta: 1.0,
                wavelength_lock: None,
            });
        }
        let wm = sample_wm_bounded_vndf(wo_c, self.p.coat_alpha_x, self.p.coat_alpha_y, us)?;
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
            let weights = self.layer_weights(wo);
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
        let wm = sample_wm_bounded_vndf(wo, self.p.specular_alpha_x, self.p.specular_alpha_y, us)?;
        let wi = reflect_local(wo, wm);
        if wi.z <= 0.0 {
            return None;
        }
        self.finalize_rough_sample(wo, wi, BsdfFlags::GLOSSY | BsdfFlags::REFLECTION, 1.0)
    }

    fn sample_spec_brdf(&self, wo: Vec3, us: Vec2, p_lobe: f32) -> Option<BsdfSample> {
        if self.spec_brdf_is_smooth() {
            let wi = Vec3::new(-wo.x, -wo.y, wo.z);
            let f = self.spec_brdf_fresnel(wi.z.abs());
            let weights = self.layer_weights(wo);
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
        let wm = sample_wm_bounded_vndf(wo, self.p.specular_alpha_x, self.p.specular_alpha_y, us)?;
        let wi = reflect_local(wo, wm);
        if wi.z <= 0.0 {
            return None;
        }
        self.finalize_rough_sample(wo, wi, BsdfFlags::GLOSSY | BsdfFlags::REFLECTION, 1.0)
    }

    fn sample_spec_btdf(
        &self,
        wo: Vec3,
        us: Vec2,
        p_lobe: f32,
        rng: &mut ThreadRng,
    ) -> Option<BsdfSample> {
        let weights = self.layer_weights(wo);
        let base_weight = weights.spec_btdf;

        if self.p.thin_walled {
            let wi = -wo;
            let weight = base_weight / p_lobe;
            return Some(BsdfSample {
                weight,
                wi,
                pdf: p_lobe,
                flags: BsdfFlags::DELTA | BsdfFlags::TRANSMISSION,
                eta: 1.0,
                wavelength_lock: None,
            });
        }

        let dispersion_active = self.p.transmission_dispersion_abbe > 0.0;
        let (eta_used, dispersion_basis, fresh_lock) = if dispersion_active {
            if let Some(lambda) = self.p.wavelength_lock {
                let eta_lambda = cauchy_ior(
                    lambda,
                    self.p.specular_eta,
                    self.p.transmission_dispersion_abbe,
                );
                (eta_lambda, Vec3::ONE, None)
            } else if self.p.front_face {
                let u_lambda = rng.random::<f32>();
                let (lambda, basis) = sample_dispersion_wavelength(u_lambda);
                let eta_lambda = cauchy_ior(
                    lambda,
                    self.p.specular_eta,
                    self.p.transmission_dispersion_abbe,
                );
                (eta_lambda, basis, Some(lambda))
            } else {
                (self.p.specular_eta, Vec3::ONE, None)
            }
        } else {
            (self.p.specular_eta, Vec3::ONE, None)
        };

        let eta_rel = if self.p.front_face {
            1.0 / eta_used
        } else {
            eta_used
        };
        let eta_i = if self.p.front_face { 1.0 } else { eta_used };
        let eta_t = if self.p.front_face { eta_used } else { 1.0 };

        if self.spec_btdf_is_smooth() {
            let wi = refract(wo, eta_rel)?;
            let f = fresnel_dielectric(wo.z.abs(), eta_i, eta_t);
            let scale = 1.0 / (eta_rel * eta_rel);
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

        let wm = sample_wm_vndf(
            wo,
            self.p.transmission_alpha_x,
            self.p.transmission_alpha_y,
            us,
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
        let g1 = ggx_g1(wo, self.p.transmission_alpha_x, self.p.transmission_alpha_y);
        if g1 <= 0.0 {
            return None;
        }
        let g2 = ggx_g2_height_correlated(
            wo,
            wi,
            self.p.transmission_alpha_x,
            self.p.transmission_alpha_y,
        );
        let f = fresnel_dielectric(cos_wo_wm, eta_i, eta_t);
        let scale = 1.0 / (eta_rel * eta_rel);
        let lobe_weight = base_weight * dispersion_basis * (scale * (1.0 - f) * g2 / g1);

        let pdf_wm = pdf_wm_vndf(
            wo,
            wm,
            self.p.transmission_alpha_x,
            self.p.transmission_alpha_y,
        );
        if pdf_wm <= 0.0 {
            return None;
        }
        let pdf_lobe_value = pdf_wm * cos_wi_wm.abs() / (den * den);
        if pdf_lobe_value <= 0.0 {
            return None;
        }

        let pdf_total = if dispersion_active {
            p_lobe * pdf_lobe_value
        } else {
            self.pdf(wo, wi).max(p_lobe * pdf_lobe_value)
        };
        if pdf_total <= 0.0 {
            return None;
        }

        Some(BsdfSample {
            weight: lobe_weight / p_lobe,
            wi,
            pdf: pdf_total,
            flags: BsdfFlags::GLOSSY | BsdfFlags::TRANSMISSION,
            eta: eta_rel,
            wavelength_lock: fresh_lock,
        })
    }

    fn sample_sheen(&self, wo: Vec3, us: Vec2) -> Option<BsdfSample> {
        let bsdf = SheenBsdf::new(Vec3::ONE, self.p.sheen_roughness);
        let s = bsdf.sample(wo, us)?;
        self.finalize_rough_sample(wo, s.wi, s.flags, s.eta)
    }

    fn sample_diff_brdf(&self, wo: Vec3, us: Vec2) -> Option<BsdfSample> {
        let bsdf = OrenNayarBsdf::new(Vec3::ONE, self.p.diffuse_roughness);
        let s = bsdf.sample(wo, us)?;
        self.finalize_rough_sample(wo, s.wi, s.flags, s.eta)
    }

    fn sample_diff_btdf(&self, wo: Vec3, us: Vec2) -> Option<BsdfSample> {
        if !self.p.thin_walled {
            return None;
        }
        let bsdf = OrenNayarBsdf::new(Vec3::ONE, self.p.diffuse_roughness);
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
            wavelength_lock: None,
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
            ChosenLobe::Sheen => self.sheen,
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
    if u < probs.sheen {
        return ChosenLobe::Sheen;
    }
    u -= probs.sheen;
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use glam::Vec3;

    use crate::bsdf::{
        BsdfFlags, DielectricGgxDirectionalAlbedoLut, SheenDirectionalAlbedoLut,
        artist_friendly_complex_ior,
    };

    use super::{StandardSurfaceBsdf, StandardSurfaceBsdfParams};

    fn default_params() -> StandardSurfaceBsdfParams {
        let (n, k) = artist_friendly_complex_ior(Vec3::splat(0.8), Vec3::ONE);
        StandardSurfaceBsdfParams {
            base_color: Vec3::new(0.8, 0.6, 0.4),
            base: 0.8,
            specular: 1.0,
            specular_color: Vec3::ONE,
            specular_alpha_x: 0.04,
            specular_alpha_y: 0.04,
            specular_eta: 1.5,
            metalness: 0.0,
            metal_n: n,
            metal_k: k,
            coat: 0.0,
            coat_color: Vec3::ONE,
            coat_alpha_x: 0.01,
            coat_alpha_y: 0.01,
            coat_eta: 1.5,
            sheen: 0.0,
            sheen_color: Vec3::ONE,
            sheen_roughness: 0.3,
            transmission: 0.0,
            transmission_color: Vec3::ONE,
            transmission_alpha_x: 0.04,
            transmission_alpha_y: 0.04,
            transmission_dispersion_abbe: 0.0,
            diffuse_roughness: 0.0,
            subsurface: 0.0,
            subsurface_color: Vec3::ONE,
            thin_walled: false,
            thin_film_thickness: 0.0,
            thin_film_ior: 1.5,
            front_face: true,
            coat_basis_in_base: None,
            wavelength_lock: None,
        }
    }

    fn test_bsdf(params: StandardSurfaceBsdfParams) -> StandardSurfaceBsdf {
        let spec_lut = Arc::new(DielectricGgxDirectionalAlbedoLut::constant_for_tests(
            1.5, 0.04,
        ));
        let coat_lut = Arc::new(DielectricGgxDirectionalAlbedoLut::constant_for_tests(
            1.5, 0.04,
        ));
        let sheen_lut = Arc::new(SheenDirectionalAlbedoLut::constant_for_tests(0.3));
        StandardSurfaceBsdf::new(params, spec_lut, coat_lut, sheen_lut)
    }

    #[test]
    fn default_diffuse_dominant_evaluates_finite() {
        let bsdf = test_bsdf(default_params());
        let f = bsdf.eval(Vec3::Z, Vec3::new(0.2, 0.3, 0.9327379).normalize());
        assert!(f.is_finite());
    }

    #[test]
    fn pure_metallic_returns_specular_lobe() {
        let mut params = default_params();
        params.metalness = 1.0;
        params.specular_alpha_x = 0.2;
        params.specular_alpha_y = 0.2;
        let bsdf = test_bsdf(params);
        let mut rng = rand::rng();
        let sample = bsdf
            .sample(Vec3::new(0.2, -0.1, 0.9746794).normalize(), &mut rng)
            .unwrap();
        assert!(sample.flags.contains(BsdfFlags::REFLECTION));
    }

    #[test]
    fn thin_walled_specular_btdf_yields_pass_through() {
        let mut params = default_params();
        params.thin_walled = true;
        params.transmission = 1.0;
        params.specular = 0.0;
        params.base = 0.0;
        let bsdf = test_bsdf(params);
        let mut rng = rand::rng();
        let wo = Vec3::new(0.2, -0.3, 0.9327379).normalize();
        let mut got = false;
        for _ in 0..32 {
            if let Some(sample) = bsdf.sample(wo, &mut rng)
                && sample.flags.contains(BsdfFlags::TRANSMISSION)
            {
                assert!(sample.wi.abs_diff_eq(-wo, 1.0e-5));
                got = true;
                break;
            }
        }
        assert!(got);
    }
}
