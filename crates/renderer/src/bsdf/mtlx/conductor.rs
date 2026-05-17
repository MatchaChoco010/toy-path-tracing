use glam::{Vec2, Vec3};

use crate::bsdf::BsdfFlags;
use crate::bsdf::fresnel_complex;
use crate::bsdf::smith_ggx::{ggx_d, ggx_g2_height_correlated, pdf_wm_vndf, sample_wm_vndf};

use super::closure::MtlxLobeSample;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConductorBsdf {
    pub weight: f32,
    pub ior: Vec3,
    pub extinction: Vec3,
    pub alpha_x: f32,
    pub alpha_y: f32,
    pub thinfilm_thickness: f32,
    pub thinfilm_ior: f32,
}

impl ConductorBsdf {
    pub fn new(weight: f32, ior: Vec3, extinction: Vec3, roughness: Vec2) -> Self {
        Self::with_thin_film(weight, ior, extinction, roughness, 0.0, 1.5)
    }

    pub fn with_thin_film(
        weight: f32,
        ior: Vec3,
        extinction: Vec3,
        roughness: Vec2,
        thinfilm_thickness: f32,
        thinfilm_ior: f32,
    ) -> Self {
        let ax = roughness.x.clamp(0.0, 1.0).max(0.001);
        let ay = roughness.y.clamp(0.0, 1.0).max(0.001);
        Self {
            weight: weight.clamp(0.0, 1.0),
            ior: ior.max(Vec3::ZERO),
            extinction: extinction.max(Vec3::ZERO),
            alpha_x: ax,
            alpha_y: ay,
            thinfilm_thickness: thinfilm_thickness.max(0.0),
            thinfilm_ior: thinfilm_ior.max(1.0e-3),
        }
    }

    fn fresnel(&self, cos_theta: f32) -> Vec3 {
        if self.thinfilm_thickness > 0.0 {
            return crate::bsdf::thin_film::eval_thin_film_conductor(
                cos_theta,
                1.0,
                self.thinfilm_ior,
                self.ior,
                self.extinction,
                self.thinfilm_thickness,
            );
        }
        fresnel_complex(cos_theta, self.ior, self.extinction)
    }

    pub fn eval(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        if wo.z <= 0.0 || wi.z <= 0.0 {
            return Vec3::ZERO;
        }
        let h = (wo + wi).normalize_or_zero();
        if h.z <= 0.0 {
            return Vec3::ZERO;
        }
        let d = ggx_d(h, self.alpha_x, self.alpha_y);
        let g = ggx_g2_height_correlated(wo, wi, self.alpha_x, self.alpha_y);
        let f = self.fresnel(wo.dot(h).clamp(0.0, 1.0));
        let denom = (4.0 * wo.z * wi.z).max(1.0e-8);
        f * (self.weight * d * g / denom)
    }

    pub fn pdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        if wo.z <= 0.0 || wi.z <= 0.0 {
            return 0.0;
        }
        let h = (wo + wi).normalize_or_zero();
        if h.z <= 0.0 {
            return 0.0;
        }
        let cos_oh = wo.dot(h).max(1.0e-8);
        pdf_wm_vndf(wo, h, self.alpha_x, self.alpha_y) / (4.0 * cos_oh)
    }

    pub fn sample(&self, wo: Vec3, us: Vec2) -> Option<MtlxLobeSample> {
        if wo.z <= 0.0 {
            return None;
        }
        let h = sample_wm_vndf(wo, self.alpha_x, self.alpha_y, us)?;
        let cos_oh = wo.dot(h).max(0.0);
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
    }

    pub fn directional_albedo(&self, wo: Vec3) -> Vec3 {
        let cos_o = wo.z.clamp(0.0, 1.0);
        self.fresnel(cos_o) * self.weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_weight_matches_f_cos_over_pdf() {
        let b = ConductorBsdf::new(1.0, Vec3::splat(0.2), Vec3::splat(3.0), Vec2::splat(0.2));
        let wo = Vec3::new(0.0, 0.0, 1.0);
        let s = b.sample(wo, Vec2::new(0.4, 0.6)).unwrap();
        let f = b.eval(wo, s.wi_local);
        let expected = f * s.wi_local.z / s.pdf;
        assert!(s.weight.abs_diff_eq(expected, 1.0e-4));
    }
}
