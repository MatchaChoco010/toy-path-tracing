use std::f32::consts::PI;

use glam::{Vec2, Vec3};

use crate::bsdf::BsdfFlags;

use super::closure::MtlxLobeSample;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TranslucentBsdf {
    pub weight: f32,
    pub color: Vec3,
}

impl TranslucentBsdf {
    pub fn new(weight: f32, color: Vec3) -> Self {
        Self {
            weight: weight.clamp(0.0, 1.0),
            color: color.max(Vec3::ZERO),
        }
    }

    pub fn eval(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        if wo.z <= 0.0 || wi.z >= 0.0 {
            return Vec3::ZERO;
        }
        self.color * self.weight / PI
    }

    pub fn pdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        if wo.z <= 0.0 || wi.z >= 0.0 {
            return 0.0;
        }
        -wi.z / PI
    }

    pub fn sample(&self, wo: Vec3, us: Vec2) -> Option<MtlxLobeSample> {
        if wo.z <= 0.0 {
            return None;
        }
        let phi = 2.0 * PI * us.x;
        let cos_theta = us.y.sqrt();
        let sin_theta = (1.0 - us.y).max(0.0).sqrt();
        let wi = Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, -cos_theta);
        let pdf = -wi.z / PI;
        if pdf <= 0.0 {
            return None;
        }
        let f = self.eval(wo, wi);
        Some(MtlxLobeSample {
            weight: f * (-wi.z) / pdf,
            wi_local: wi,
            pdf,
            flags: BsdfFlags::DIFFUSE | BsdfFlags::TRANSMISSION,
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
    fn returns_zero_for_same_side() {
        let b = TranslucentBsdf::new(1.0, Vec3::ONE);
        assert_eq!(b.eval(Vec3::Z, Vec3::Z), Vec3::ZERO);
        assert_eq!(b.eval(Vec3::Z, Vec3::new(0.0, 0.0, 0.5)), Vec3::ZERO);
    }

    #[test]
    fn lambert_transmission_value() {
        let b = TranslucentBsdf::new(1.0, Vec3::splat(0.7));
        let v = b.eval(Vec3::Z, -Vec3::Z);
        assert!((v.x - 0.7 / PI).abs() < 1.0e-6);
    }

    #[test]
    fn sample_weight_matches_f_cos_over_pdf() {
        let b = TranslucentBsdf::new(1.0, Vec3::splat(0.6));
        let wo = Vec3::Z;
        let s = b.sample(wo, Vec2::new(0.3, 0.7)).unwrap();
        let f = b.eval(wo, s.wi_local);
        let expected = f * (-s.wi_local.z) / s.pdf;
        assert!(s.weight.abs_diff_eq(expected, 1.0e-5));
    }
}
