use glam::{Vec2, Vec3};

use crate::bsdf::BsdfFlags;
use crate::bsdf::smith_ggx::{ggx_d, ggx_g2_height_correlated, pdf_wm_vndf, sample_wm_vndf};

use super::closure::MtlxLobeSample;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScatterMode {
    Reflection,
    Transmission,
    Both,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DielectricBsdf {
    pub weight: f32,
    pub tint: Vec3,
    pub ior: f32,
    pub alpha_x: f32,
    pub alpha_y: f32,
    pub scatter_mode: ScatterMode,
    pub thinfilm_thickness: f32,
    pub thinfilm_ior: f32,
    pub front_face: bool,
}

impl DielectricBsdf {
    pub fn new(
        weight: f32,
        tint: Vec3,
        ior: f32,
        roughness: Vec2,
        scatter_mode: ScatterMode,
    ) -> Self {
        Self::with_thin_film(weight, tint, ior, roughness, scatter_mode, 0.0, 1.5, true)
    }

    pub fn with_thin_film(
        weight: f32,
        tint: Vec3,
        ior: f32,
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
            tint: tint.max(Vec3::ZERO),
            ior: ior.max(0.0),
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
            (1.0, self.ior.max(1.0e-3))
        } else {
            (self.ior.max(1.0e-3), 1.0)
        }
    }

    fn fresnel(&self, cos_theta: f32) -> f32 {
        if self.ior <= 0.0 {
            return 1.0;
        }
        let (eta_o, eta_i) = self.eta_outside_inside();
        fresnel_dielectric(cos_theta, eta_o, eta_i)
    }

    fn fresnel_rgb(&self, cos_theta: f32) -> Vec3 {
        if self.thinfilm_thickness > 0.0 {
            let (eta_o, eta_i) = self.eta_outside_inside();
            return crate::bsdf::thin_film::eval_thin_film_dielectric(
                cos_theta,
                eta_o,
                self.thinfilm_ior,
                eta_i,
                self.thinfilm_thickness,
            );
        }
        Vec3::splat(self.fresnel(cos_theta))
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
            let cos_oh = wo.dot(h).clamp(0.0, 1.0);
            let f_rgb = if self.ior > 0.0 {
                self.fresnel_rgb(cos_oh)
            } else {
                Vec3::ONE
            };
            let denom = (4.0 * wo.z * wi.z).max(1.0e-8);
            self.tint * self.weight * f_rgb * (d * g / denom)
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
            let scale = match self.scatter_mode {
                ScatterMode::Both => 1.0 - self.fresnel(cos_oh),
                ScatterMode::Transmission => 1.0,
                _ => 0.0,
            };
            let d = ggx_d(h, self.alpha_x, self.alpha_y);
            let g = ggx_g2_height_correlated(wo, wi, self.alpha_x, self.alpha_y);
            let denom = (eta_o * cos_oh + eta_i * cos_ih).abs().max(1.0e-8);
            let denom2 = denom * denom;
            let cos_factor = (wo.z * wi.z.abs()).max(1.0e-8);
            let factor =
                (cos_ih.abs() * cos_oh.abs() * eta_i * eta_i * d * g) / (cos_factor * denom2);
            self.tint * self.weight * scale * factor
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
                    let f_rgb = self.fresnel_rgb(cos_oh.clamp(0.0, 1.0));
                    (f_rgb.x + f_rgb.y + f_rgb.z) / 3.0
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
                    let f_rgb = self.fresnel_rgb(cos_oh);
                    1.0 - (f_rgb.x + f_rgb.y + f_rgb.z) / 3.0
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
                let f_rgb = self.fresnel_rgb(cos_oh);
                let f_avg = (f_rgb.x + f_rgb.y + f_rgb.z) / 3.0;
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
        let cos_o = wo.z.clamp(0.0, 1.0);
        let f_rgb = self.fresnel_rgb(cos_o);
        (self.tint * self.weight * f_rgb).clamp(Vec3::ZERO, Vec3::ONE)
    }
}

pub fn fresnel_dielectric(cos_theta: f32, eta_o: f32, eta_i: f32) -> f32 {
    let cti = cos_theta.clamp(0.0, 1.0);
    let sti2 = (1.0 - cti * cti).max(0.0);
    let stt2 = (eta_o / eta_i).powi(2) * sti2;
    if stt2 >= 1.0 {
        return 1.0;
    }
    let ctt = (1.0 - stt2).max(0.0).sqrt();
    let r_par = (eta_i * cti - eta_o * ctt) / (eta_i * cti + eta_o * ctt);
    let r_per = (eta_o * cti - eta_i * ctt) / (eta_o * cti + eta_i * ctt);
    (r_par * r_par + r_per * r_per) * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresnel_at_normal_matches_schlick_baseline() {
        let f = fresnel_dielectric(1.0, 1.0, 1.5);
        let f0: f32 = ((1.5 - 1.0) / (1.5 + 1.0_f32)).powi(2);
        assert!((f - f0).abs() < 1.0e-6);
    }

    #[test]
    fn fresnel_total_internal_reflection_is_one() {
        let f = fresnel_dielectric(0.5, 1.5, 1.0);
        assert!((f - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn fresnel_diamond_exit_total_internal_reflection() {
        let f = fresnel_dielectric(0.5, 2.4, 1.0);
        assert!((f - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn fresnel_diamond_normal_entry_low() {
        let f = fresnel_dielectric(1.0, 1.0, 2.4);
        let expected: f32 = ((2.4 - 1.0) / (2.4 + 1.0_f32)).powi(2);
        assert!((f - expected).abs() < 1.0e-6);
    }

    #[test]
    fn dielectric_reflection_eval_includes_no_cos_factor() {
        let b = DielectricBsdf::new(
            1.0,
            Vec3::ONE,
            1.5,
            Vec2::splat(0.2),
            ScatterMode::Reflection,
        );
        let wo = Vec3::new(0.3, 0.0, 0.9539392).normalize();
        let s = b.sample(wo, Vec2::new(0.4, 0.6), 0.2).unwrap();
        let f = b.eval(wo, s.wi_local);
        let expected = f * s.wi_local.z / s.pdf;
        assert!(s.weight.abs_diff_eq(expected, 1.0e-4));
    }

    #[test]
    fn diamond_back_face_at_grazing_returns_tir_none() {
        let b = DielectricBsdf::with_thin_film(
            1.0,
            Vec3::ONE,
            2.4,
            Vec2::splat(0.001),
            ScatterMode::Transmission,
            0.0,
            1.5,
            false,
        );
        let wo = Vec3::new(0.7, 0.0, 0.7142857).normalize();
        let mut none_count = 0;
        let mut total = 0;
        for i in 0..32 {
            for j in 0..32 {
                total += 1;
                let us = Vec2::new(i as f32 / 32.0, j as f32 / 32.0);
                if b.sample(wo, us, 0.5).is_none() {
                    none_count += 1;
                }
            }
        }
        assert!(
            none_count > total / 2,
            "expected mostly TIR but got {}/{}",
            none_count,
            total
        );
    }

    #[test]
    fn diamond_front_face_normal_entry_transmits() {
        let b = DielectricBsdf::with_thin_film(
            1.0,
            Vec3::ONE,
            2.4,
            Vec2::splat(0.001),
            ScatterMode::Transmission,
            0.0,
            1.5,
            true,
        );
        let wo = Vec3::new(0.0, 0.0, 1.0);
        let s = b
            .sample(wo, Vec2::new(0.5, 0.5), 1.0)
            .expect("normal-entry transmission should succeed");
        assert!(s.flags.contains(BsdfFlags::TRANSMISSION));
        assert!(s.wi_local.z < 0.0);
    }

    #[test]
    fn rt_mode_uses_independent_branch_sample() {
        let b = DielectricBsdf::new(1.0, Vec3::ONE, 1.5, Vec2::splat(0.05), ScatterMode::Both);
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
