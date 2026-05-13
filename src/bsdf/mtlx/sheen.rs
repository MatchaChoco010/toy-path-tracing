use std::f32::consts::PI;

use glam::{Vec2, Vec3};

use crate::bsdf::{
    BsdfFlags, SheenBsdf as ContyKullaSheen, SheenDirectionalAlbedoLut,
    sheen_directional_albedo_estimate,
};

use super::closure::MtlxLobeSample;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SheenMode {
    ContyKulla,
    Zeltner,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SheenBsdfMtlx {
    pub weight: f32,
    pub color: Vec3,
    pub roughness: f32,
    pub mode: SheenMode,
}

impl SheenBsdfMtlx {
    pub fn new(weight: f32, color: Vec3, roughness: f32, mode: SheenMode) -> Self {
        Self {
            weight: weight.clamp(0.0, 1.0),
            color: color.max(Vec3::ZERO),
            roughness: roughness.clamp(0.0, 1.0),
            mode,
        }
    }

    fn conty_inner(&self) -> ContyKullaSheen {
        ContyKullaSheen::new(self.color * self.weight, self.roughness)
    }

    pub fn eval(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        if wo.z <= 0.0 || wi.z <= 0.0 {
            return Vec3::ZERO;
        }
        match self.mode {
            SheenMode::ContyKulla => self.conty_inner().eval(wo, wi),
            SheenMode::Zeltner => self.zeltner_eval(wo, wi),
        }
    }

    pub fn pdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        if wo.z <= 0.0 || wi.z <= 0.0 {
            return 0.0;
        }
        match self.mode {
            SheenMode::ContyKulla => 1.0 / (2.0 * PI),
            SheenMode::Zeltner => self.zeltner_pdf(wo, wi),
        }
    }

    pub fn sample(&self, wo: Vec3, us: Vec2) -> Option<MtlxLobeSample> {
        if wo.z <= 0.0 {
            return None;
        }
        match self.mode {
            SheenMode::ContyKulla => self.conty_sample(wo, us),
            SheenMode::Zeltner => self.zeltner_sample(wo, us),
        }
    }

    pub fn directional_albedo(&self, wo: Vec3) -> Vec3 {
        self.directional_albedo_with_lut(wo, None)
    }

    pub(crate) fn directional_albedo_with_lut(
        &self,
        wo: Vec3,
        lut: Option<&SheenDirectionalAlbedoLut>,
    ) -> Vec3 {
        match self.mode {
            SheenMode::ContyKulla => {
                let scalar = lut.map_or_else(
                    || sheen_directional_albedo_estimate(self.roughness, wo.z, 32),
                    |lut| lut.lookup(wo.z, self.roughness),
                );
                self.color * (self.weight * scalar)
            }
            SheenMode::Zeltner => {
                let r = zeltner_dir_albedo(wo.z.clamp(0.0, 1.0), self.roughness);
                self.color * (self.weight * r)
            }
        }
    }

    fn conty_sample(&self, wo: Vec3, us: Vec2) -> Option<MtlxLobeSample> {
        let z = us.y.clamp(0.0, 1.0);
        let r_xy = (1.0 - z * z).max(0.0).sqrt();
        let phi = 2.0 * PI * us.x;
        let wi = Vec3::new(phi.cos() * r_xy, phi.sin() * r_xy, z);
        let pdf = 1.0 / (2.0 * PI);
        let f = self.conty_inner().eval(wo, wi);
        Some(MtlxLobeSample {
            weight: f * wi.z / pdf,
            wi_local: wi,
            pdf,
            flags: BsdfFlags::GLOSSY | BsdfFlags::REFLECTION,
            eta: 1.0,
        })
    }

    fn zeltner_eval(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        let phi_wo = phi_of(wo);
        let wi_std = rotate_about_z(wi, -phi_wo);
        let roughness = self.roughness.clamp(0.01, 1.0);
        let a_inv = zeltner_ltc_a_inv(wo.z, roughness);
        let b_inv = zeltner_ltc_b_inv(wo.z, roughness);
        let r = zeltner_dir_albedo(wo.z, roughness);
        let value = eval_ltc(wi_std, a_inv, b_inv);
        let cos_i = wi.z.max(1.0e-6);
        self.color * (self.weight * r * value / cos_i)
    }

    fn zeltner_pdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        let phi_wo = phi_of(wo);
        let wi_std = rotate_about_z(wi, -phi_wo);
        let roughness = self.roughness.clamp(0.01, 1.0);
        let a_inv = zeltner_ltc_a_inv(wo.z, roughness);
        let b_inv = zeltner_ltc_b_inv(wo.z, roughness);
        eval_ltc(wi_std, a_inv, b_inv)
    }

    fn zeltner_sample(&self, wo: Vec3, us: Vec2) -> Option<MtlxLobeSample> {
        let roughness = self.roughness.clamp(0.01, 1.0);
        let a_inv = zeltner_ltc_a_inv(wo.z, roughness);
        let b_inv = zeltner_ltc_b_inv(wo.z, roughness);
        let wi_std = sample_ltc(a_inv, b_inv, us);
        if wi_std.z <= 0.0 {
            return None;
        }
        let phi_wo = phi_of(wo);
        let wi = rotate_about_z(wi_std, phi_wo);
        if wi.z <= 0.0 {
            return None;
        }
        let pdf = eval_ltc(wi_std, a_inv, b_inv);
        if pdf <= 0.0 {
            return None;
        }
        let f = self.zeltner_eval(wo, wi);
        Some(MtlxLobeSample {
            weight: f * wi.z / pdf,
            wi_local: wi,
            pdf,
            flags: BsdfFlags::GLOSSY | BsdfFlags::REFLECTION,
            eta: 1.0,
        })
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upper_hemisphere_only() {
        let b = SheenBsdfMtlx::new(1.0, Vec3::ONE, 0.3, SheenMode::ContyKulla);
        assert_eq!(b.eval(Vec3::Z, -Vec3::Z), Vec3::ZERO);
        assert_eq!(b.eval(-Vec3::Z, Vec3::Z), Vec3::ZERO);
    }

    #[test]
    fn conty_sample_weight_matches_f_cos_over_pdf() {
        let b = SheenBsdfMtlx::new(1.0, Vec3::splat(0.7), 0.4, SheenMode::ContyKulla);
        let wo = Vec3::new(0.2, 0.0, 0.9797959).normalize();
        let s = b.sample(wo, Vec2::new(0.4, 0.6)).unwrap();
        let f = b.eval(wo, s.wi_local);
        let expected = f * s.wi_local.z / s.pdf;
        assert!(s.weight.abs_diff_eq(expected, 1.0e-5));
    }

    #[test]
    fn zeltner_sample_weight_matches_f_cos_over_pdf() {
        let b = SheenBsdfMtlx::new(1.0, Vec3::splat(0.7), 0.5, SheenMode::Zeltner);
        let wo = Vec3::new(0.3, 0.0, 0.9539392).normalize();
        let s = b.sample(wo, Vec2::new(0.4, 0.6)).unwrap();
        let f = b.eval(wo, s.wi_local);
        let expected = f * s.wi_local.z / s.pdf;
        assert!(
            s.weight.abs_diff_eq(expected, 1.0e-3),
            "weight={} expected={}",
            s.weight,
            expected,
        );
    }

    #[test]
    fn zeltner_dir_albedo_matches_materialx_glsl_fit() {
        let b = SheenBsdfMtlx::new(1.0, Vec3::ONE, 0.5, SheenMode::Zeltner);
        let albedo = b.directional_albedo(Vec3::Z);
        let r = zeltner_dir_albedo(1.0, 0.5);
        assert!((albedo.x - r).abs() < 1.0e-5);
        assert!((zeltner_ltc_a_inv(0.6, 0.5) - 0.516_073_2).abs() < 1.0e-6);
        assert!((zeltner_ltc_b_inv(0.6, 0.5) + 0.046_596_706).abs() < 1.0e-6);
    }

    #[test]
    fn zeltner_distinct_from_conty_kulla_at_rough() {
        let conty = SheenBsdfMtlx::new(1.0, Vec3::ONE, 0.8, SheenMode::ContyKulla);
        let zeltner = SheenBsdfMtlx::new(1.0, Vec3::ONE, 0.8, SheenMode::Zeltner);
        let wo = Vec3::new(0.3, 0.0, 0.9539392).normalize();
        let wi = Vec3::new(-0.2, 0.1, 0.9746794).normalize();
        let f_conty = conty.eval(wo, wi);
        let f_zeltner = zeltner.eval(wo, wi);
        let diff = (f_zeltner.x - f_conty.x).abs();
        assert!(
            diff > 1.0e-4,
            "Zeltner LTC must differ from Conty-Kulla: conty={} zeltner={}",
            f_conty.x,
            f_zeltner.x,
        );
    }
}
