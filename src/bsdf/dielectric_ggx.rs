use std::f32::consts::PI;
use std::sync::Arc;

use glam::{Vec2, Vec3};

use crate::math::{
    cosine_weighted_hemisphere_pdf, fresnel_dielectric, refract, sample_cosine_weighted_hemisphere,
};

use super::DielectricGgxEnergyCompensationLut;
use super::smith_ggx::{
    EFFECTIVELY_SMOOTH_ALPHA, MIN_ALPHA, ggx_d, ggx_g1, ggx_g2_height_correlated,
    is_upper_hemisphere, pdf_wm_vndf, reflect_local, reflection_half_vector, sample_wm_vndf,
};
use super::{BsdfFlags, BsdfSample};

const DENOM_EPS: f32 = 1.0e-6;
const MS_DENOM_EPS: f32 = 1.0e-4;

#[derive(Debug, Clone, PartialEq)]
pub struct DielectricGgxBsdf {
    color: Vec3,
    eta: f32,
    alpha_x: f32,
    alpha_y: f32,
    thin: bool,
    front_face: bool,
    allowed_paths: DielectricGgxAllowedPaths,
    energy_compensation_lut: Option<Arc<DielectricGgxEnergyCompensationLut>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DielectricGgxAllowedPaths {
    Reflection,
    Transmission,
    ReflectionAndTransmission,
}

impl DielectricGgxBsdf {
    pub fn new(
        color: Vec3,
        eta: f32,
        alpha_x: f32,
        alpha_y: f32,
        thin: bool,
        front_face: bool,
    ) -> Self {
        Self::new_with_allowed_paths(
            color,
            eta,
            alpha_x,
            alpha_y,
            thin,
            front_face,
            DielectricGgxAllowedPaths::ReflectionAndTransmission,
        )
    }

    pub fn new_with_allowed_paths(
        color: Vec3,
        eta: f32,
        alpha_x: f32,
        alpha_y: f32,
        thin: bool,
        front_face: bool,
        allowed_paths: DielectricGgxAllowedPaths,
    ) -> Self {
        Self {
            color,
            eta,
            alpha_x: alpha_x.max(MIN_ALPHA),
            alpha_y: alpha_y.max(MIN_ALPHA),
            thin,
            front_face,
            allowed_paths,
            energy_compensation_lut: None,
        }
    }

    pub(crate) fn new_with_energy_compensation(
        color: Vec3,
        eta: f32,
        alpha_x: f32,
        alpha_y: f32,
        thin: bool,
        front_face: bool,
        lut: Arc<DielectricGgxEnergyCompensationLut>,
    ) -> Self {
        Self {
            color,
            eta,
            alpha_x: alpha_x.max(MIN_ALPHA),
            alpha_y: alpha_y.max(MIN_ALPHA),
            thin,
            front_face,
            allowed_paths: DielectricGgxAllowedPaths::ReflectionAndTransmission,
            energy_compensation_lut: Some(lut),
        }
    }

    fn effectively_smooth(&self) -> bool {
        self.alpha_x.max(self.alpha_y) < EFFECTIVELY_SMOOTH_ALPHA
    }

    fn fresnel_interface(&self) -> (f32, f32) {
        if self.front_face {
            (1.0, self.eta)
        } else {
            (self.eta, 1.0)
        }
    }

    fn eta_rel(&self) -> f32 {
        // eta_rel = n_o / n_i; matches math::refract's `eta` parameter convention.
        if self.front_face {
            1.0 / self.eta
        } else {
            self.eta
        }
    }

    pub fn eval(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        if !is_upper_hemisphere(wo) || self.eta <= 0.0 {
            return Vec3::ZERO;
        }
        if self.thin || self.effectively_smooth() {
            return Vec3::ZERO;
        }

        let f_ss = self.eval_single_scattering(wo, wi);
        let f_ms = self.eval_multi_scattering(wo, wi);
        f_ss + f_ms
    }

    fn eval_single_scattering(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        let (eta_i, eta_t) = self.fresnel_interface();
        let eta_rel = self.eta_rel();

        if wo.z * wi.z > 0.0 && self.allowed_paths.allows_reflection() {
            // Reflection branch.
            let Some(wm) = reflection_half_vector(wo, wi) else {
                return Vec3::ZERO;
            };
            let cos_wo_wm = wo.dot(wm);
            if cos_wo_wm <= 0.0 {
                return Vec3::ZERO;
            }
            let cos_o = wo.z.abs();
            let cos_i = wi.z.abs();
            if cos_o <= 0.0 || cos_i <= 0.0 {
                return Vec3::ZERO;
            }
            let d = ggx_d(wm, self.alpha_x, self.alpha_y);
            if d <= 0.0 {
                return Vec3::ZERO;
            }
            let g = ggx_g2_height_correlated(wo, wi, self.alpha_x, self.alpha_y);
            if g <= 0.0 {
                return Vec3::ZERO;
            }
            let f = fresnel_dielectric(cos_wo_wm, eta_i, eta_t);
            Vec3::splat(d * g * f / (4.0 * cos_o * cos_i))
        } else if wo.z * wi.z < 0.0 && self.allowed_paths.allows_transmission() {
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
            // Backfacing-microfacet check: only wi values that actually arise
            // from the forward refract map have cos_wi_wm with the same sign as
            // wi.z. Any other (wo, wi) pair is unreachable by the sampler, so
            // the BSDF must vanish there to keep eval/pdf consistent with the
            // sampling distribution.
            if cos_wi_wm * wi.z < 0.0 {
                return Vec3::ZERO;
            }
            let den = cos_wi_wm + eta_rel * cos_wo_wm;
            if den.abs() < DENOM_EPS {
                return Vec3::ZERO;
            }
            let cos_o = wo.z.abs();
            let cos_i = wi.z.abs();
            if cos_o <= 0.0 || cos_i <= 0.0 {
                return Vec3::ZERO;
            }
            let d = ggx_d(wm, self.alpha_x, self.alpha_y);
            if d <= 0.0 {
                return Vec3::ZERO;
            }
            let g = ggx_g2_height_correlated(wo, wi, self.alpha_x, self.alpha_y);
            if g <= 0.0 {
                return Vec3::ZERO;
            }
            let f = fresnel_dielectric(cos_wo_wm, eta_i, eta_t);
            let radiance_scale = 1.0 / (eta_rel * eta_rel);
            let numerator = d * (1.0 - f) * g * (cos_wi_wm * cos_wo_wm).abs();
            let denom = den * den * cos_o * cos_i;
            if denom <= 0.0 {
                return Vec3::ZERO;
            }
            self.color * (radiance_scale * numerator / denom)
        } else {
            Vec3::ZERO
        }
    }

    pub fn pdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        if !is_upper_hemisphere(wo) || self.eta <= 0.0 {
            return 0.0;
        }
        if self.thin || self.effectively_smooth() {
            return 0.0;
        }

        let pdf_ss = self.pdf_single_scattering(wo, wi);

        if !self.compensation_active() {
            return pdf_ss;
        }

        let ms = self.ms_params(wo);
        let pr_ss = ms.e_avg_o.clamp(0.0, 1.0);
        let pr_ms = (1.0 - pr_ss).max(0.0);
        let pdf_ms = self.pdf_multi_scattering(wi, &ms);
        pr_ss * pdf_ss + pr_ms * pdf_ms
    }

    fn pdf_single_scattering(&self, wo: Vec3, wi: Vec3) -> f32 {
        let (eta_i, eta_t) = self.fresnel_interface();
        let eta_rel = self.eta_rel();

        if wo.z * wi.z > 0.0 && self.allowed_paths.allows_reflection() {
            // Reflection.
            let Some(wm) = reflection_half_vector(wo, wi) else {
                return 0.0;
            };
            let cos_wo_wm = wo.dot(wm);
            if cos_wo_wm <= 0.0 {
                return 0.0;
            }
            let f = fresnel_dielectric(cos_wo_wm, eta_i, eta_t);
            let pdf_wm = pdf_wm_vndf(wo, wm, self.alpha_x, self.alpha_y);
            if pdf_wm <= 0.0 {
                return 0.0;
            }
            let branch_probability =
                if self.allowed_paths == DielectricGgxAllowedPaths::ReflectionAndTransmission {
                    f
                } else {
                    1.0
                };
            branch_probability * pdf_wm / (4.0 * cos_wo_wm)
        } else if wo.z * wi.z < 0.0 && self.allowed_paths.allows_transmission() {
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
            if cos_wi_wm * wi.z < 0.0 {
                return 0.0;
            }
            let den = cos_wi_wm + eta_rel * cos_wo_wm;
            if den.abs() < DENOM_EPS {
                return 0.0;
            }
            let f = fresnel_dielectric(cos_wo_wm, eta_i, eta_t);
            let pdf_wm = pdf_wm_vndf(wo, wm, self.alpha_x, self.alpha_y);
            if pdf_wm <= 0.0 {
                return 0.0;
            }
            let branch_probability =
                if self.allowed_paths == DielectricGgxAllowedPaths::ReflectionAndTransmission {
                    1.0 - f
                } else {
                    1.0
                };
            branch_probability * pdf_wm * cos_wi_wm.abs() / (den * den)
        } else {
            0.0
        }
    }

    pub fn sample(&self, wo: Vec3, uc: f32, us: Vec2) -> Option<BsdfSample> {
        if !is_upper_hemisphere(wo) || self.eta <= 0.0 {
            return None;
        }

        if self.thin {
            return self.sample_thin_delta(wo, uc);
        }

        if self.effectively_smooth() {
            return self.sample_smooth_delta(wo, uc);
        }

        if !self.compensation_active() {
            return self.sample_rough(wo, uc, us);
        }

        self.sample_rough_with_compensation(wo, uc, us)
    }

    fn sample_rough_with_compensation(&self, wo: Vec3, uc: f32, us: Vec2) -> Option<BsdfSample> {
        let ms = self.ms_params(wo);
        let pr_ss = ms.e_avg_o.clamp(0.0, 1.0);
        let pr_ms = (1.0 - pr_ss).max(0.0);
        if pr_ss + pr_ms <= 0.0 {
            return None;
        }

        if uc < pr_ss {
            // SS branch: re-map uc to [0, 1) for the inner F-based R/T pick.
            let uc_inner = if pr_ss > 0.0 { uc / pr_ss } else { 0.0 };
            let mut sample = self.sample_rough(wo, uc_inner, us)?;
            // Recompute total pdf and weight using full eval / pdf so that
            // the sample respects the multi-lobe MIS combination.
            let pdf_ss = self.pdf_single_scattering(wo, sample.wi);
            let pdf_ms = self.pdf_multi_scattering(sample.wi, &ms);
            let pdf_total = pr_ss * pdf_ss + pr_ms * pdf_ms;
            if pdf_total <= 0.0 {
                return None;
            }
            let f_total = self.eval_single_scattering(wo, sample.wi)
                + self.eval_multi_scattering_with(wo, sample.wi, &ms);
            sample.pdf = pdf_total;
            sample.weight = f_total * (sample.wi.z.abs() / pdf_total);
            Some(sample)
        } else {
            // MS branch: sub-pick R vs T via Ratio(eta_o).
            let uc_inner = if pr_ms > 0.0 {
                (uc - pr_ss) / pr_ms
            } else {
                0.0
            };
            let ratio_r = ms.ratio_r.clamp(0.0, 1.0);
            let (wi, ms_flags, eta_returned) = if uc_inner < ratio_r {
                let wi = sample_cosine_weighted_hemisphere(us);
                if !is_upper_hemisphere(wi) {
                    return None;
                }
                (wi, BsdfFlags::GLOSSY | BsdfFlags::REFLECTION, 1.0)
            } else {
                let wi_up = sample_cosine_weighted_hemisphere(us);
                let wi = Vec3::new(wi_up.x, wi_up.y, -wi_up.z);
                if wi.z >= 0.0 {
                    return None;
                }
                (wi, BsdfFlags::GLOSSY | BsdfFlags::TRANSMISSION, ms.eta_rel)
            };

            let pdf_ss = self.pdf_single_scattering(wo, wi);
            let pdf_ms = self.pdf_multi_scattering(wi, &ms);
            let pdf_total = pr_ss * pdf_ss + pr_ms * pdf_ms;
            if pdf_total <= 0.0 {
                return None;
            }
            let f_total =
                self.eval_single_scattering(wo, wi) + self.eval_multi_scattering_with(wo, wi, &ms);
            let weight = f_total * (wi.z.abs() / pdf_total);
            Some(BsdfSample {
                weight,
                wi,
                pdf: pdf_total,
                flags: ms_flags,
                eta: eta_returned,
                wavelength_lock: None,
            })
        }
    }

    fn sample_thin_delta(&self, wo: Vec3, uc: f32) -> Option<BsdfSample> {
        let mut reflectance = fresnel_dielectric(wo.z.abs(), 1.0, self.eta);
        let transmittance = 1.0 - reflectance;
        let denom = 1.0 - reflectance * reflectance;

        if denom <= 0.0 {
            reflectance = 1.0;
        } else {
            reflectance += transmittance * transmittance * reflectance / denom;
        }

        let transmittance = (1.0 - reflectance).max(0.0);
        let mut pr = reflectance.max(0.0);
        let mut pt = transmittance.max(0.0);
        if !self.allowed_paths.allows_reflection() {
            pr = 0.0;
        }
        if !self.allowed_paths.allows_transmission() {
            pt = 0.0;
        }
        let probability_sum = pr + pt;
        if probability_sum <= 0.0 {
            return None;
        }
        pr /= probability_sum;
        pt /= probability_sum;

        let reflect = uc < pr;
        let (wi, pdf, weight, flags) = if reflect {
            (
                reflected_direction(wo),
                pr,
                Vec3::splat(reflectance / pr),
                BsdfFlags::DELTA | BsdfFlags::REFLECTION,
            )
        } else {
            (
                -wo,
                pt,
                self.color * (transmittance / pt),
                BsdfFlags::DELTA | BsdfFlags::TRANSMISSION,
            )
        };

        Some(BsdfSample {
            weight,
            wi,
            pdf,
            flags,
            wavelength_lock: None,
            eta: 1.0,
        })
    }

    fn sample_smooth_delta(&self, wo: Vec3, uc: f32) -> Option<BsdfSample> {
        let (eta_i, eta_t) = self.fresnel_interface();
        let eta_rel = self.eta_rel();
        let reflectance = fresnel_dielectric(wo.z.abs(), eta_i, eta_t);
        let transmission_direction = refract(wo, eta_rel);
        let transmittance = if transmission_direction.is_some() {
            1.0 - reflectance
        } else {
            0.0
        };
        let mut pr = reflectance.max(0.0);
        let mut pt = transmittance.max(0.0);
        if !self.allowed_paths.allows_reflection() {
            pr = 0.0;
        }
        if !self.allowed_paths.allows_transmission() {
            pt = 0.0;
        }
        let probability_sum = pr + pt;
        if probability_sum <= 0.0 {
            return None;
        }
        pr /= probability_sum;
        pt /= probability_sum;

        if uc < pr {
            return Some(BsdfSample {
                weight: Vec3::splat(reflectance / pr),
                wi: reflected_direction(wo),
                pdf: pr,
                flags: BsdfFlags::DELTA | BsdfFlags::REFLECTION,
                eta: 1.0,
                wavelength_lock: None,
            });
        }

        let wi = transmission_direction?;
        let radiance_scale = 1.0 / (eta_rel * eta_rel);

        Some(BsdfSample {
            weight: self.color * (radiance_scale * transmittance / pt),
            wi,
            pdf: pt,
            flags: BsdfFlags::DELTA | BsdfFlags::TRANSMISSION,
            eta: eta_rel,
            wavelength_lock: None,
        })
    }

    fn sample_rough(&self, wo: Vec3, uc: f32, us: Vec2) -> Option<BsdfSample> {
        let (eta_i, eta_t) = self.fresnel_interface();
        let eta_rel = self.eta_rel();

        let wm = sample_wm_vndf(wo, self.alpha_x, self.alpha_y, us)?;
        let cos_wo_wm = wo.dot(wm);
        if cos_wo_wm <= 0.0 {
            return None;
        }
        let f = fresnel_dielectric(cos_wo_wm, eta_i, eta_t);
        let g1 = ggx_g1(wo, self.alpha_x, self.alpha_y);
        if g1 <= 0.0 {
            return None;
        }

        let mut pr = f.max(0.0);
        let mut pt = (1.0 - f).max(0.0);
        if !self.allowed_paths.allows_reflection() {
            pr = 0.0;
        }
        if !self.allowed_paths.allows_transmission() {
            pt = 0.0;
        }
        let probability_sum = pr + pt;
        if probability_sum <= 0.0 {
            return None;
        }
        pr /= probability_sum;
        pt /= probability_sum;

        if uc < pr {
            // Reflection branch.
            let wi = reflect_local(wo, wm);
            if !is_upper_hemisphere(wi) {
                return None;
            }
            let g2 = ggx_g2_height_correlated(wo, wi, self.alpha_x, self.alpha_y);
            let pdf_wm = pdf_wm_vndf(wo, wm, self.alpha_x, self.alpha_y);
            if pdf_wm <= 0.0 {
                return None;
            }
            let pdf = pr * pdf_wm / (4.0 * cos_wo_wm);
            if pdf <= 0.0 {
                return None;
            }
            let weight = Vec3::splat(f * g2 / (pr * g1));
            Some(BsdfSample {
                weight,
                wi,
                pdf,
                flags: BsdfFlags::GLOSSY | BsdfFlags::REFLECTION,
                eta: 1.0,
                wavelength_lock: None,
            })
        } else {
            // Transmission branch.
            let wi = refract_about_wm(wo, wm, eta_rel)?;
            if wi.z >= 0.0 {
                return None;
            }
            let cos_wi_wm = wi.dot(wm);
            let den = cos_wi_wm + eta_rel * cos_wo_wm;
            if den.abs() < DENOM_EPS {
                return None;
            }
            let g2 = ggx_g2_height_correlated(wo, wi, self.alpha_x, self.alpha_y);
            let pdf_wm = pdf_wm_vndf(wo, wm, self.alpha_x, self.alpha_y);
            if pdf_wm <= 0.0 {
                return None;
            }
            let pdf = pt * pdf_wm * cos_wi_wm.abs() / (den * den);
            if pdf <= 0.0 {
                return None;
            }
            let radiance_scale = 1.0 / (eta_rel * eta_rel);
            let weight = self.color * (radiance_scale * (1.0 - f) * g2 / (pt * g1));
            Some(BsdfSample {
                weight,
                wi,
                pdf,
                flags: BsdfFlags::GLOSSY | BsdfFlags::TRANSMISSION,
                eta: eta_rel,
                wavelength_lock: None,
            })
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct MsParams {
    eta_o: f32,
    eta_rel: f32,
    roughness_eq: f32,
    e_avg_o: f32,
    ratio_r: f32,
    one_minus_e_avg_o: f32,
    one_minus_e_avg_t: f32,
}

impl DielectricGgxBsdf {
    fn compensation_active(&self) -> bool {
        self.energy_compensation_lut.is_some()
            && self.allowed_paths == DielectricGgxAllowedPaths::ReflectionAndTransmission
    }

    fn ms_params(&self, _wo: Vec3) -> MsParams {
        let lut = self
            .energy_compensation_lut
            .as_ref()
            .expect("ms_params called without LUT");
        let eta_o = if self.front_face {
            self.eta
        } else {
            1.0 / self.eta
        };
        let eta_t = 1.0 / eta_o;
        let roughness_eq = (self.alpha_x * self.alpha_y).powf(0.25);

        let e_avg_o = lut.lookup_e_avg(roughness_eq, eta_o);
        let e_avg_t = lut.lookup_e_avg(roughness_eq, eta_t);
        let one_minus_e_avg_o = (1.0 - e_avg_o).max(MS_DENOM_EPS);
        let one_minus_e_avg_t = (1.0 - e_avg_t).max(MS_DENOM_EPS);

        let f_avg_o = f_avg_dielectric(eta_o);
        let f_avg_t = f_avg_dielectric(eta_t);
        let a = (1.0 - f_avg_o) / one_minus_e_avg_t;
        let b = (1.0 - f_avg_t) * eta_o * eta_o / one_minus_e_avg_o;
        let x = if a + b > MS_DENOM_EPS {
            b / (a + b)
        } else {
            0.5
        };
        let ratio_r = (1.0 - x * (1.0 - f_avg_o)).clamp(0.0, 1.0);

        MsParams {
            eta_o,
            eta_rel: eta_t,
            roughness_eq,
            e_avg_o,
            ratio_r,
            one_minus_e_avg_o,
            one_minus_e_avg_t,
        }
    }

    fn eval_multi_scattering(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        if !self.compensation_active() {
            return Vec3::ZERO;
        }
        let ms = self.ms_params(wo);
        self.eval_multi_scattering_with(wo, wi, &ms)
    }

    fn eval_multi_scattering_with(&self, wo: Vec3, wi: Vec3, ms: &MsParams) -> Vec3 {
        let lut = self
            .energy_compensation_lut
            .as_ref()
            .expect("eval_multi_scattering_with called without LUT");

        let cos_o = wo.z.abs();
        let cos_i = wi.z.abs();
        if cos_o <= 0.0 || cos_i <= 0.0 {
            return Vec3::ZERO;
        }

        let e_o = lut.lookup_e(cos_o, ms.roughness_eq, ms.eta_o);
        if wo.z * wi.z > 0.0 {
            let e_i = lut.lookup_e(cos_i, ms.roughness_eq, ms.eta_o);
            let f_ms_r = ms.ratio_r * (1.0 - e_o) * (1.0 - e_i) / (PI * ms.one_minus_e_avg_o);
            Vec3::splat(f_ms_r)
        } else {
            let e_i = lut.lookup_e(cos_i, ms.roughness_eq, ms.eta_rel);
            let radiance_scale = ms.eta_o * ms.eta_o;
            let f_ms_t = (1.0 - ms.ratio_r) * (1.0 - e_o) * (1.0 - e_i) * radiance_scale
                / (PI * ms.one_minus_e_avg_t);
            self.color * f_ms_t
        }
    }

    fn pdf_multi_scattering(&self, wi: Vec3, ms: &MsParams) -> f32 {
        if wi.z > 0.0 {
            ms.ratio_r * cosine_weighted_hemisphere_pdf(wi.z)
        } else if wi.z < 0.0 {
            (1.0 - ms.ratio_r) * cosine_weighted_hemisphere_pdf(-wi.z)
        } else {
            0.0
        }
    }
}

fn f_avg_dielectric(eta: f32) -> f32 {
    if eta >= 1.0 {
        ((eta - 1.0) / (4.08567 + 1.00071 * eta)).clamp(0.0, 1.0)
    } else {
        let e = eta.max(MS_DENOM_EPS);
        let v = 0.997118 + 0.1014 * e - 0.965241 * e * e - 0.130607 * e * e * e;
        v.clamp(0.0, 1.0)
    }
}

impl DielectricGgxAllowedPaths {
    fn allows_reflection(self) -> bool {
        matches!(self, Self::Reflection | Self::ReflectionAndTransmission)
    }

    fn allows_transmission(self) -> bool {
        matches!(self, Self::Transmission | Self::ReflectionAndTransmission)
    }
}

fn reflected_direction(wo: Vec3) -> Vec3 {
    Vec3::new(-wo.x, -wo.y, wo.z).normalize_or_zero()
}

fn refract_about_wm(wo: Vec3, wm: Vec3, eta: f32) -> Option<Vec3> {
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
    use glam::{Vec2, Vec3};

    use crate::{
        bsdf::{BsdfFlags, DielectricGgxAllowedPaths, DielectricGgxBsdf},
        math::{fresnel_dielectric, refract},
    };

    #[test]
    fn effectively_smooth_returns_delta_reflection_at_normal_incidence() {
        let bsdf = DielectricGgxBsdf::new(Vec3::ONE, 1.5, 1.0e-4, 1.0e-4, false, true);
        let wo = Vec3::Z;
        let sample = bsdf
            .sample(wo, 0.01, Vec2::splat(0.5))
            .expect("expected a delta reflection sample");

        assert_eq!(sample.wi, Vec3::Z);
        assert_eq!(sample.weight, Vec3::ONE);
        assert!((sample.pdf - 0.04).abs() < 1.0e-4);
        assert_eq!(sample.flags, BsdfFlags::DELTA | BsdfFlags::REFLECTION);
        assert_eq!(bsdf.eval(wo, sample.wi), Vec3::ZERO);
        assert_eq!(bsdf.pdf(wo, sample.wi), 0.0);
    }

    #[test]
    fn effectively_smooth_returns_delta_transmission_with_radiance_scaling() {
        let color = Vec3::new(0.3, 0.5, 0.7);
        let eta = 1.5;
        let bsdf = DielectricGgxBsdf::new(color, eta, 1.0e-4, 1.0e-4, false, true);
        let wo = Vec3::new(0.3, -0.4, 0.8660254).normalize();
        let sample = bsdf
            .sample(wo, 0.99, Vec2::splat(0.5))
            .expect("expected a delta transmission sample");
        let expected_wi = refract(wo, 1.0 / eta).expect("expected refraction");

        assert!(sample.wi.abs_diff_eq(expected_wi, 1.0e-6));
        assert!(sample.weight.abs_diff_eq(color * (eta * eta), 1.0e-6));
        assert_eq!(sample.flags, BsdfFlags::DELTA | BsdfFlags::TRANSMISSION);
        assert!((sample.pdf - (1.0 - fresnel_dielectric(wo.z, 1.0, eta))).abs() < 1.0e-5);
    }

    #[test]
    fn effectively_smooth_exiting_glass_scales_radiance_down() {
        let color = Vec3::new(0.3, 0.5, 0.7);
        let eta = 1.5;
        let bsdf = DielectricGgxBsdf::new(color, eta, 1.0e-4, 1.0e-4, false, false);
        let wo = Vec3::Z;
        let sample = bsdf
            .sample(wo, 0.99, Vec2::splat(0.5))
            .expect("expected a delta transmission sample");

        assert!(sample.weight.abs_diff_eq(color / (eta * eta), 1.0e-6));
        assert_eq!(sample.wi, -Vec3::Z);
        assert_eq!(sample.flags, BsdfFlags::DELTA | BsdfFlags::TRANSMISSION);
    }

    #[test]
    fn effectively_smooth_falls_back_to_reflection_under_total_internal_reflection() {
        let bsdf = DielectricGgxBsdf::new(Vec3::ONE, 1.5, 1.0e-4, 1.0e-4, false, false);
        let wo = Vec3::new(0.8, 0.0, 0.6).normalize();
        let sample = bsdf
            .sample(wo, 0.9, Vec2::splat(0.5))
            .expect("expected total internal reflection");

        assert!(sample.wi.abs_diff_eq(Vec3::new(-0.8, 0.0, 0.6), 1.0e-6));
        assert_eq!(sample.weight, Vec3::ONE);
        assert_eq!(sample.pdf, 1.0);
        assert_eq!(sample.flags, BsdfFlags::DELTA | BsdfFlags::REFLECTION);
    }

    #[test]
    fn thin_uses_aggregate_reflectance_and_flips_direction_on_transmission() {
        let color = Vec3::new(0.3, 0.5, 0.7);
        let eta = 1.5;
        let bsdf = DielectricGgxBsdf::new(color, eta, 0.1, 0.1, true, true);
        let wo = Vec3::new(0.3, -0.4, 0.8660254).normalize();
        let sample = bsdf
            .sample(wo, 0.9, Vec2::splat(0.5))
            .expect("expected a thin transmission sample");
        let base = fresnel_dielectric(wo.z, 1.0, eta);
        let expected_reflectance = base + (1.0 - base).powi(2) * base / (1.0 - base * base);

        assert!(sample.wi.abs_diff_eq(-wo, 1.0e-6));
        assert!(sample.weight.abs_diff_eq(color, 1.0e-6));
        assert_eq!(sample.flags, BsdfFlags::DELTA | BsdfFlags::TRANSMISSION);
        assert!((sample.pdf - (1.0 - expected_reflectance)).abs() < 1.0e-5);
    }

    #[test]
    fn thin_ignores_roughness_and_stays_delta() {
        let bsdf = DielectricGgxBsdf::new(Vec3::ONE, 1.5, 0.9, 0.9, true, true);
        let wo = Vec3::new(0.2, -0.1, 0.9746794).normalize();
        let sample = bsdf
            .sample(wo, 0.5, Vec2::new(0.3, 0.7))
            .expect("expected a thin delta sample");

        assert!(sample.flags.contains(BsdfFlags::DELTA));
        assert!(!sample.flags.contains(BsdfFlags::GLOSSY));
    }

    #[test]
    fn rough_reflection_sample_matches_eval_cos_over_pdf() {
        let bsdf = DielectricGgxBsdf::new(Vec3::ONE, 1.5, 0.3, 0.2, false, true);
        let wo = Vec3::new(0.2, -0.3, 0.9327379).normalize();

        // With uc = 0.0 we always pick the reflection branch.
        let sample = bsdf
            .sample(wo, 0.0, Vec2::new(0.35, 0.72))
            .expect("expected a reflection sample");

        assert!(
            sample
                .flags
                .contains(BsdfFlags::GLOSSY | BsdfFlags::REFLECTION)
        );
        assert!(sample.wi.z > 0.0);
        assert!(sample.pdf > 0.0);

        let f = bsdf.eval(wo, sample.wi);
        let expected = f * (sample.wi.z.abs() / sample.pdf);
        assert!(sample.weight.abs_diff_eq(expected, 5.0e-4));
    }

    #[test]
    fn rough_transmission_sample_matches_eval_cos_over_pdf() {
        let bsdf = DielectricGgxBsdf::new(Vec3::new(0.5, 0.7, 0.9), 1.5, 0.3, 0.2, false, true);
        let wo = Vec3::new(0.2, -0.3, 0.9327379).normalize();

        // With uc = 1.0 we always pick the transmission branch.
        let sample = bsdf
            .sample(wo, 0.999, Vec2::new(0.4, 0.6))
            .expect("expected a transmission sample");

        assert!(
            sample
                .flags
                .contains(BsdfFlags::GLOSSY | BsdfFlags::TRANSMISSION)
        );
        assert!(sample.wi.z < 0.0);
        assert!(sample.pdf > 0.0);

        let f = bsdf.eval(wo, sample.wi);
        let expected = f * (sample.wi.z.abs() / sample.pdf);
        assert!(sample.weight.abs_diff_eq(expected, 5.0e-4));
    }

    #[test]
    fn reflection_path_uses_only_reflection_branch() {
        let bsdf = DielectricGgxBsdf::new_with_allowed_paths(
            Vec3::ONE,
            1.5,
            0.3,
            0.2,
            false,
            true,
            DielectricGgxAllowedPaths::Reflection,
        );
        let wo = Vec3::new(0.2, -0.3, 0.9327379).normalize();
        let transmission_wi = Vec3::new(-0.1, 0.2, -0.9746794).normalize();

        assert_eq!(bsdf.eval(wo, transmission_wi), Vec3::ZERO);
        assert_eq!(bsdf.pdf(wo, transmission_wi), 0.0);

        let sample = bsdf
            .sample(wo, 0.999, Vec2::new(0.35, 0.72))
            .expect("expected a reflection-only sample");
        let expected = bsdf.eval(wo, sample.wi) * (sample.wi.z / sample.pdf);

        assert!(sample.flags.contains(BsdfFlags::REFLECTION));
        assert!(!sample.flags.contains(BsdfFlags::TRANSMISSION));
        assert!(sample.wi.z > 0.0);
        assert!(sample.weight.abs_diff_eq(expected, 5.0e-4));
    }

    #[test]
    fn transmission_path_uses_only_transmission_branch() {
        let bsdf = DielectricGgxBsdf::new_with_allowed_paths(
            Vec3::ONE,
            1.5,
            0.3,
            0.2,
            false,
            true,
            DielectricGgxAllowedPaths::Transmission,
        );
        let wo = Vec3::new(0.2, -0.3, 0.9327379).normalize();
        let reflection_wi = Vec3::new(-0.2, 0.3, 0.9327379).normalize();

        assert_eq!(bsdf.eval(wo, reflection_wi), Vec3::ZERO);
        assert_eq!(bsdf.pdf(wo, reflection_wi), 0.0);

        let sample = bsdf
            .sample(wo, 0.0, Vec2::new(0.35, 0.72))
            .expect("expected a transmission-only sample");
        let expected = bsdf.eval(wo, sample.wi) * (sample.wi.z.abs() / sample.pdf);

        assert!(sample.flags.contains(BsdfFlags::TRANSMISSION));
        assert!(!sample.flags.contains(BsdfFlags::REFLECTION));
        assert!(sample.wi.z < 0.0);
        assert!(sample.weight.abs_diff_eq(expected, 5.0e-4));
    }

    #[test]
    fn sample_returns_none_for_lower_hemisphere_wo() {
        let bsdf = DielectricGgxBsdf::new(Vec3::ONE, 1.5, 0.3, 0.3, false, true);

        assert!(bsdf.sample(-Vec3::Z, 0.5, Vec2::splat(0.5)).is_none());
        assert!(
            DielectricGgxBsdf::new(Vec3::ONE, 0.0, 0.3, 0.3, false, true)
                .sample(Vec3::Z, 0.5, Vec2::splat(0.5))
                .is_none()
        );
    }

    #[test]
    fn eval_returns_zero_when_wi_is_on_wrong_side_of_expected_event() {
        let bsdf = DielectricGgxBsdf::new(Vec3::ONE, 1.5, 0.3, 0.3, false, true);
        // wi == wo.z small positive but non-physical reflection pair; eval uses
        // generalized half vector logic and should not produce energy.
        assert_eq!(bsdf.eval(Vec3::Z, Vec3::ZERO), Vec3::ZERO);
    }

    fn van_der_corput(n: u32, base: u32) -> f32 {
        let mut q = 0.0_f32;
        let mut bk = 1.0 / base as f32;
        let mut nn = n;
        while nn > 0 {
            q += (nn % base) as f32 * bk;
            nn /= base;
            bk /= base as f32;
        }
        q
    }

    fn halton2(i: u32) -> Vec2 {
        Vec2::new(van_der_corput(i + 1, 2), van_der_corput(i + 1, 3))
    }

    fn estimate_irradiance_via_quadrature(bsdf: &DielectricGgxBsdf, wo: Vec3, eta: f32) -> f32 {
        let inv_eta_sq = 1.0 / (eta * eta);
        let n = 4096_u32;
        let mut sum = 0.0_f32;
        for index in 0..n {
            let uc = van_der_corput(index + 1, 5);
            let us = halton2(index);
            let Some(sample) = bsdf.sample(wo, uc, us) else {
                continue;
            };
            if !sample.weight.x.is_finite() {
                continue;
            }
            let scale = if sample.flags.contains(BsdfFlags::TRANSMISSION) {
                inv_eta_sq
            } else {
                1.0
            };
            sum += sample.weight.x * scale;
        }
        sum / n as f32
    }

    #[test]
    fn compensation_passes_white_furnace_full_sphere() {
        use std::sync::Arc;

        use crate::bsdf::DielectricGgxEnergyCompensationLut;

        let lut = Arc::new(DielectricGgxEnergyCompensationLut::build_for_tests());

        let cases = [
            (1.5_f32, 0.05_f32),
            (1.5, 0.15),
            (1.5, 0.3),
            (1.5, 0.6),
            (1.5, 0.9),
            (1.5, 1.0),
            (1.33, 0.7),
            (2.4, 0.5),
        ];
        let mut report = Vec::new();
        let mut max_err = 0.0_f32;
        for (eta, alpha) in cases {
            let bsdf = DielectricGgxBsdf::new_with_energy_compensation(
                Vec3::ONE,
                eta,
                alpha,
                alpha,
                false,
                true,
                Arc::clone(&lut),
            );
            let wo = Vec3::new(0.3, -0.2, 0.9327379).normalize();
            let energy = estimate_irradiance_via_quadrature(&bsdf, wo, eta);
            let err = (energy - 1.0).abs();
            max_err = max_err.max(err);
            report.push(format!("eta={eta} alpha={alpha} energy={energy:.4}"));
        }
        assert!(
            max_err < 0.02,
            "white furnace error {max_err:.4} exceeds tolerance:\n  {}",
            report.join("\n  "),
        );
    }

    #[test]
    fn compensation_off_matches_existing_eval() {
        let bsdf = DielectricGgxBsdf::new(Vec3::ONE, 1.5, 0.3, 0.3, false, true);
        let wo = Vec3::new(0.2, -0.3, 0.9327379).normalize();
        let wi = Vec3::new(-0.2, 0.3, 0.9327379).normalize();
        let f = bsdf.eval(wo, wi);
        let pdf = bsdf.pdf(wo, wi);
        assert!(f.is_finite());
        assert!(pdf.is_finite());
    }

    #[test]
    fn compensation_sample_weight_matches_eval_cos_over_pdf() {
        use std::sync::Arc;

        use crate::bsdf::DielectricGgxEnergyCompensationLut;

        let lut = Arc::new(DielectricGgxEnergyCompensationLut::build_for_tests());
        let bsdf = DielectricGgxBsdf::new_with_energy_compensation(
            Vec3::new(0.85, 0.95, 0.95),
            1.5,
            0.4,
            0.4,
            false,
            true,
            lut,
        );
        let wo = Vec3::new(0.2, -0.3, 0.9327379).normalize();

        for x in 0..3 {
            for y in 0..3 {
                for c in 0..3 {
                    let uc = (c as f32 + 0.5) / 3.0;
                    let us = Vec2::new((x as f32 + 0.5) / 3.0, (y as f32 + 0.5) / 3.0);
                    if let Some(sample) = bsdf.sample(wo, uc, us) {
                        let f = bsdf.eval(wo, sample.wi);
                        let expected = f * (sample.wi.z.abs() / sample.pdf);
                        assert!(
                            sample.weight.abs_diff_eq(expected, 8.0e-3),
                            "uc={uc} us={us:?} weight={:?} expected={:?}",
                            sample.weight,
                            expected,
                        );
                    }
                }
            }
        }
    }
}
