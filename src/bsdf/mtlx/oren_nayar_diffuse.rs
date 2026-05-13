use std::f32::consts::PI;

use glam::{Vec2, Vec3};

use crate::bsdf::{BsdfFlags, EonBsdf, OrenNayarBsdf};

use super::closure::MtlxLobeSample;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrenNayarDiffuseBsdf {
    pub weight: f32,
    pub color: Vec3,
    pub roughness: f32,
    pub energy_compensation: bool,
}

impl OrenNayarDiffuseBsdf {
    pub fn new(weight: f32, color: Vec3, roughness: f32, energy_compensation: bool) -> Self {
        Self {
            weight: weight.clamp(0.0, 1.0),
            color: color.max(Vec3::ZERO),
            roughness: roughness.clamp(0.0, 1.0),
            energy_compensation,
        }
    }

    pub fn eval(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        if wo.z <= 0.0 || wi.z <= 0.0 {
            return Vec3::ZERO;
        }
        let weighted = self.color * self.weight;
        if self.roughness <= 1.0e-6 {
            return weighted / PI;
        }
        if self.energy_compensation {
            return EonBsdf::new(weighted, self.roughness).eval(wo, wi);
        }
        OrenNayarBsdf::new(weighted, self.roughness).eval(wo, wi)
    }

    pub fn pdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        if wo.z <= 0.0 || wi.z <= 0.0 {
            return 0.0;
        }
        wi.z / PI
    }

    pub fn sample(&self, wo: Vec3, us: Vec2) -> Option<MtlxLobeSample> {
        if wo.z <= 0.0 {
            return None;
        }
        let phi = 2.0 * PI * us.x;
        let cos_theta = us.y.sqrt();
        let sin_theta = (1.0 - us.y).max(0.0).sqrt();
        let wi = Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta);
        let pdf = wi.z / PI;
        if pdf <= 0.0 {
            return None;
        }
        let f = self.eval(wo, wi);
        Some(MtlxLobeSample {
            weight: f * wi.z / pdf,
            wi_local: wi,
            pdf,
            flags: BsdfFlags::DIFFUSE | BsdfFlags::REFLECTION,
            eta: 1.0,
        })
    }

    pub fn directional_albedo(&self, _wo: Vec3) -> Vec3 {
        self.color * self.weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lambert_limit_matches_one_over_pi() {
        let b = OrenNayarDiffuseBsdf::new(1.0, Vec3::ONE, 0.0, false);
        let v = b.eval(Vec3::Z, Vec3::Z);
        assert!((v.x - 1.0 / PI).abs() < 1.0e-6);
    }

    #[test]
    fn smooth_and_rough_lambert_limit_match_at_zero_roughness() {
        let smooth = OrenNayarDiffuseBsdf::new(1.0, Vec3::splat(0.6), 0.0, false);
        let rough = OrenNayarDiffuseBsdf::new(1.0, Vec3::splat(0.6), 1.0e-7, false);
        let wo = Vec3::new(0.2, -0.1, 0.9746794).normalize();
        let wi = Vec3::new(-0.3, 0.2, 0.932738).normalize();
        let fs = smooth.eval(wo, wi);
        let fr = rough.eval(wo, wi);
        assert!(fs.abs_diff_eq(fr, 1.0e-4));
    }

    #[test]
    fn sample_weight_matches_f_cos_over_pdf() {
        let b = OrenNayarDiffuseBsdf::new(1.0, Vec3::splat(0.6), 0.0, false);
        let wo = Vec3::new(0.0, 0.0, 1.0);
        let s = b.sample(wo, Vec2::new(0.3, 0.7)).unwrap();
        let f = b.eval(wo, s.wi_local);
        let expected = f * s.wi_local.z / s.pdf;
        assert!(s.weight.abs_diff_eq(expected, 1.0e-5));
    }
}
