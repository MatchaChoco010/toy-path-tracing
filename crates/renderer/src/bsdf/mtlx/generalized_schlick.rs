use glam::{Vec2, Vec3};

use crate::bsdf::BsdfFlags;
use crate::bsdf::smith_ggx::{ggx_d, ggx_g2_height_correlated, pdf_wm_vndf, sample_wm_vndf};

use super::closure::MtlxLobeSample;
use super::dielectric::ScatterMode;

const ONE_MINUS_COS_82: f32 = 6.0 / 7.0;
const G_AT_COS_82: f32 = 0.056_647_927;
const TRANSMISSION_IOR: f32 = 1.5;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneralizedSchlickBsdf {
    pub weight: f32,
    pub color0: Vec3,
    pub color82: Vec3,
    pub color90: Vec3,
    pub exponent: f32,
    pub alpha_x: f32,
    pub alpha_y: f32,
    pub scatter_mode: ScatterMode,
    pub thinfilm_thickness: f32,
    pub thinfilm_ior: f32,
    pub front_face: bool,
}

impl GeneralizedSchlickBsdf {
    pub fn new(
        weight: f32,
        color0: Vec3,
        color82: Vec3,
        color90: Vec3,
        exponent: f32,
        roughness: Vec2,
        scatter_mode: ScatterMode,
    ) -> Self {
        Self::with_thin_film(
            weight,
            color0,
            color82,
            color90,
            exponent,
            roughness,
            scatter_mode,
            0.0,
            1.5,
            true,
        )
    }

    pub fn with_thin_film(
        weight: f32,
        color0: Vec3,
        color82: Vec3,
        color90: Vec3,
        exponent: f32,
        roughness: Vec2,
        scatter_mode: ScatterMode,
        thinfilm_thickness: f32,
        thinfilm_ior: f32,
        front_face: bool,
    ) -> Self {
        let ax = roughness.x.clamp(0.0, 1.0).max(0.001);
        let ay = roughness.y.clamp(0.0, 1.0).max(0.001);
        Self {
            weight: weight.clamp(0.0, 1.0),
            color0: color0.max(Vec3::ZERO),
            color82: color82.max(Vec3::ZERO),
            color90: color90.max(Vec3::ZERO),
            exponent,
            alpha_x: ax,
            alpha_y: ay,
            scatter_mode,
            thinfilm_thickness: thinfilm_thickness.max(0.0),
            thinfilm_ior: thinfilm_ior.max(1.0e-3),
            front_face,
        }
    }

    fn eta_outside_inside(&self) -> (f32, f32) {
        if self.front_face {
            (1.0, TRANSMISSION_IOR)
        } else {
            (TRANSMISSION_IOR, 1.0)
        }
    }

    pub fn fresnel(&self, cos_theta: f32) -> Vec3 {
        let f = self.fresnel_dry(cos_theta);
        if self.thinfilm_thickness > 0.0 {
            let f0_avg = (self.color0.x + self.color0.y + self.color0.z) / 3.0;
            let sqrt_f0 = f0_avg.clamp(0.0, 0.999).sqrt();
            let eta_base = ((1.0 + sqrt_f0) / (1.0 - sqrt_f0)).max(1.0e-3);
            let film = crate::bsdf::thin_film::eval_thin_film_dielectric(
                cos_theta,
                1.0,
                self.thinfilm_ior,
                eta_base,
                self.thinfilm_thickness,
            );
            let dry_avg = (f.x + f.y + f.z) / 3.0;
            let shape = if dry_avg > 1.0e-6 {
                f / dry_avg
            } else {
                Vec3::ONE
            };
            return (film * shape).max(Vec3::ZERO);
        }
        f
    }

    fn fresnel_dry(&self, cos_theta: f32) -> Vec3 {
        let c = cos_theta.clamp(0.0, 1.0);
        let one_minus = 1.0 - c;
        let f_schlick = self
            .color0
            .lerp(self.color90, one_minus.powf(self.exponent));
        let f_at_82 = self
            .color0
            .lerp(self.color90, ONE_MINUS_COS_82.powf(self.exponent));
        let alpha = (f_at_82 - self.color82) / G_AT_COS_82;
        let g_cos = one_minus.powi(6) * c;
        (f_schlick - alpha * g_cos).max(Vec3::ZERO)
    }

    pub fn eval(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        if wo.z <= 0.0 {
            return Vec3::ZERO;
        }
        let reflect = wi.z > 0.0;
        match self.scatter_mode {
            ScatterMode::Reflection if !reflect => return Vec3::ZERO,
            ScatterMode::Transmission if reflect => return Vec3::ZERO,
            _ => {}
        }
        if reflect {
            let h = (wo + wi).normalize_or_zero();
            if h.z <= 0.0 {
                return Vec3::ZERO;
            }
            let d = ggx_d(h, self.alpha_x, self.alpha_y);
            let g = ggx_g2_height_correlated(wo, wi, self.alpha_x, self.alpha_y);
            let f = self.fresnel(wo.dot(h).clamp(0.0, 1.0));
            let denom = (4.0 * wo.z * wi.z).max(1.0e-8);
            f * (self.weight * d * g / denom)
        } else {
            let (eta_o, eta_i) = self.eta_outside_inside();
            let h_unnorm = -(wo * eta_o + wi * eta_i);
            let h = h_unnorm.normalize_or_zero();
            let h = if h.z >= 0.0 { h } else { -h };
            let cos_oh = wo.dot(h);
            let cos_ih = wi.dot(h);
            if cos_oh <= 0.0 {
                return Vec3::ZERO;
            }
            let f_refl = self.fresnel(cos_oh.clamp(0.0, 1.0));
            let t_color = (Vec3::ONE - f_refl).max(Vec3::ZERO);
            let scale = match self.scatter_mode {
                ScatterMode::Both => t_color,
                ScatterMode::Transmission => Vec3::ONE,
                _ => Vec3::ZERO,
            };
            let d = ggx_d(h, self.alpha_x, self.alpha_y);
            let g = ggx_g2_height_correlated(wo, wi, self.alpha_x, self.alpha_y);
            let denom = (eta_o * cos_oh + eta_i * cos_ih).abs().max(1.0e-8);
            let denom2 = denom * denom;
            let cos_factor = (wo.z * wi.z.abs()).max(1.0e-8);
            let factor =
                (cos_ih.abs() * cos_oh.abs() * eta_i * eta_i * d * g) / (cos_factor * denom2);
            scale * self.weight * factor
        }
    }

    pub fn pdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        if wo.z <= 0.0 {
            return 0.0;
        }
        let reflect = wi.z > 0.0;
        match self.scatter_mode {
            ScatterMode::Reflection if !reflect => return 0.0,
            ScatterMode::Transmission if reflect => return 0.0,
            _ => {}
        }
        if reflect {
            let h = (wo + wi).normalize_or_zero();
            if h.z <= 0.0 {
                return 0.0;
            }
            let cos_oh = wo.dot(h).max(1.0e-8);
            let p_branch = match self.scatter_mode {
                ScatterMode::Both => {
                    let f = self.fresnel(cos_oh.clamp(0.0, 1.0));
                    (f.x + f.y + f.z) / 3.0
                }
                _ => 1.0,
            };
            p_branch * pdf_wm_vndf(wo, h, self.alpha_x, self.alpha_y) / (4.0 * cos_oh)
        } else {
            let (eta_o, eta_i) = self.eta_outside_inside();
            let h_unnorm = -(wo * eta_o + wi * eta_i);
            let h = h_unnorm.normalize_or_zero();
            let h = if h.z >= 0.0 { h } else { -h };
            let cos_oh = wo.dot(h).clamp(0.0, 1.0);
            let cos_ih = wi.dot(h);
            if cos_oh <= 0.0 {
                return 0.0;
            }
            let denom = (eta_o * cos_oh + eta_i * cos_ih).abs().max(1.0e-8);
            let jac = eta_i * eta_i * cos_ih.abs() / (denom * denom);
            let p_branch = match self.scatter_mode {
                ScatterMode::Both => {
                    let f = self.fresnel(cos_oh);
                    1.0 - (f.x + f.y + f.z) / 3.0
                }
                _ => 1.0,
            };
            p_branch * pdf_wm_vndf(wo, h, self.alpha_x, self.alpha_y) * jac
        }
    }

    pub fn sample(&self, wo: Vec3, us: Vec2, u_branch: f32) -> Option<MtlxLobeSample> {
        if wo.z <= 0.0 {
            return None;
        }
        let h = sample_wm_vndf(wo, self.alpha_x, self.alpha_y, us)?;
        let cos_oh = wo.dot(h).max(0.0);
        let want_reflect = match self.scatter_mode {
            ScatterMode::Reflection => true,
            ScatterMode::Transmission => false,
            ScatterMode::Both => {
                let f = self.fresnel(cos_oh);
                let f_avg = (f.x + f.y + f.z) / 3.0;
                u_branch < f_avg
            }
        };
        if want_reflect {
            let wi = -wo + 2.0 * h * cos_oh;
            if wi.z <= 0.0 {
                return None;
            }
            let f = self.eval(wo, wi);
            let pdf = self.pdf(wo, wi);
            if pdf <= 0.0 {
                return None;
            }
            Some(MtlxLobeSample {
                weight: f * wi.z / pdf,
                wi_local: wi,
                pdf,
                flags: BsdfFlags::GLOSSY | BsdfFlags::REFLECTION,
                eta: 1.0,
            })
        } else {
            let (eta_o, eta_i) = self.eta_outside_inside();
            let eta = eta_o / eta_i;
            let cos_oh = wo.dot(h);
            let k = 1.0 - eta * eta * (1.0 - cos_oh * cos_oh);
            if k < 0.0 {
                return None;
            }
            let wi = -wo * eta + h * (eta * cos_oh - k.sqrt());
            if wi.z >= 0.0 {
                return None;
            }
            let f = self.eval(wo, wi);
            let pdf = self.pdf(wo, wi);
            if pdf <= 0.0 {
                return None;
            }
            Some(MtlxLobeSample {
                weight: f * wi.z.abs() / pdf,
                wi_local: wi,
                pdf,
                flags: BsdfFlags::GLOSSY | BsdfFlags::TRANSMISSION,
                eta,
            })
        }
    }

    pub fn directional_albedo(&self, wo: Vec3) -> Vec3 {
        if matches!(self.scatter_mode, ScatterMode::Transmission) {
            return Vec3::ZERO;
        }
        self.fresnel(wo.z.clamp(0.0, 1.0)) * self.weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bsdf::BsdfFlags;

    #[test]
    fn schlick_t_mode_eval_zero_above_horizon() {
        let b = GeneralizedSchlickBsdf::new(
            1.0,
            Vec3::splat(0.04),
            Vec3::ONE,
            Vec3::ONE,
            5.0,
            Vec2::splat(0.05),
            ScatterMode::Transmission,
        );
        assert_eq!(b.eval(Vec3::Z, Vec3::Z), Vec3::ZERO);
    }

    #[test]
    fn schlick_t_mode_transmits_below_horizon() {
        let b = GeneralizedSchlickBsdf::new(
            1.0,
            Vec3::splat(0.04),
            Vec3::ONE,
            Vec3::ONE,
            5.0,
            Vec2::splat(0.05),
            ScatterMode::Transmission,
        );
        let wo = Vec3::Z;
        let s = b
            .sample(wo, Vec2::new(0.5, 0.5), 1.0)
            .expect("normal-entry T sample should succeed");
        assert!(s.flags.contains(BsdfFlags::TRANSMISSION));
        assert!(s.wi_local.z < 0.0);
    }

    #[test]
    fn schlick_exponent_below_one_is_preserved() {
        let exponent = 0.5;
        let color82 = Vec3::splat(ONE_MINUS_COS_82.powf(exponent));
        let b = GeneralizedSchlickBsdf::new(
            1.0,
            Vec3::ZERO,
            color82,
            Vec3::ONE,
            exponent,
            Vec2::splat(0.05),
            ScatterMode::Reflection,
        );
        let f = b.fresnel(0.5);
        assert!((f.x - (0.5_f32).sqrt()).abs() < 1.0e-6);
    }

    #[test]
    fn schlick_rt_mode_branches_on_fresnel() {
        let b = GeneralizedSchlickBsdf::new(
            1.0,
            Vec3::splat(0.04),
            Vec3::ONE,
            Vec3::ONE,
            5.0,
            Vec2::splat(0.1),
            ScatterMode::Both,
        );
        let wo = Vec3::Z;
        let mut refl = 0;
        let mut trans = 0;
        for i in 0..200 {
            for j in 0..200 {
                let us = Vec2::new(i as f32 / 200.0, j as f32 / 200.0);
                if let Some(s) = b.sample(wo, us, j as f32 / 200.0) {
                    if s.flags.contains(BsdfFlags::TRANSMISSION) {
                        trans += 1;
                    } else if s.flags.contains(BsdfFlags::REFLECTION) {
                        refl += 1;
                    }
                }
            }
        }
        assert!(
            refl > 0 && trans > 0,
            "RT mode must produce both lobes; got refl={} trans={}",
            refl,
            trans
        );
        assert!(
            trans > refl,
            "low-F0 RT should transmit more than reflect; got refl={} trans={}",
            refl,
            trans
        );
    }

    #[test]
    fn schlick_rt_mode_uses_independent_branch_sample() {
        let b = GeneralizedSchlickBsdf::new(
            1.0,
            Vec3::splat(0.04),
            Vec3::ONE,
            Vec3::ONE,
            5.0,
            Vec2::splat(0.05),
            ScatterMode::Both,
        );
        let wo = Vec3::Z;
        let us = Vec2::new(0.5, 0.5);

        let reflected = b
            .sample(wo, us, 0.0)
            .expect("reflection branch should sample");
        let transmitted = b
            .sample(wo, us, 0.99)
            .expect("transmission branch should sample");

        assert!(reflected.flags.contains(BsdfFlags::REFLECTION));
        assert!(transmitted.flags.contains(BsdfFlags::TRANSMISSION));
    }
}
