use std::f32::consts::PI;

use glam::{Vec2, Vec3};

use crate::bsdf::BsdfFlags;

use super::closure::MtlxLobeSample;

const P_MAX: usize = 3;
const SQRT_PI_OVER_8: f32 = 0.626_657_07;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChiangHairBsdf {
    pub h: f32,
    pub gamma_o: f32,
    pub eta: f32,
    pub sigma_a: Vec3,
    pub tint_r: Vec3,
    pub tint_tt: Vec3,
    pub tint_trt: Vec3,
    pub v: [f32; P_MAX + 1],
    pub s: f32,
    pub sin_2k_alpha: [f32; 3],
    pub cos_2k_alpha: [f32; 3],
    pub curve_direction: Vec3,
}

#[inline]
fn sqr(v: f32) -> f32 {
    v * v
}

#[inline]
fn safe_sqrt(x: f32) -> f32 {
    x.max(0.0).sqrt()
}

#[inline]
fn safe_asin(x: f32) -> f32 {
    x.clamp(-1.0, 1.0).asin()
}

fn i0(x: f32) -> f32 {
    let mut val = 0.0_f32;
    let mut x2i = 1.0_f32;
    let mut ifact: u64 = 1;
    let mut i4: u64 = 1;
    for i in 0..10 {
        if i > 1 {
            ifact *= i as u64;
        }
        val += x2i / (i4 as f32 * sqr(ifact as f32));
        x2i *= x * x;
        i4 *= 4;
    }
    val
}

fn log_i0(x: f32) -> f32 {
    if x > 12.0 {
        x + 0.5 * (-(2.0 * PI).ln() + (1.0 / x).ln() + 1.0 / (8.0 * x))
    } else {
        i0(x).ln()
    }
}

fn mp(cos_theta_i: f32, cos_theta_o: f32, sin_theta_i: f32, sin_theta_o: f32, v: f32) -> f32 {
    let a = cos_theta_i * cos_theta_o / v;
    let b = sin_theta_i * sin_theta_o / v;
    if v <= 0.1 {
        (log_i0(a) - b - 1.0 / v + std::f32::consts::LN_2 + (1.0 / (2.0 * v)).ln()).exp()
    } else {
        ((-b).exp() * i0(a)) / ((1.0 / v).sinh() * 2.0 * v)
    }
}

fn fr_dielectric(cos_theta_i: f32, eta_i: f32, eta_t: f32) -> f32 {
    let cos_i = cos_theta_i.clamp(-1.0, 1.0);
    let entering = cos_i > 0.0;
    let (eta_i, eta_t, cos_i) = if entering {
        (eta_i, eta_t, cos_i)
    } else {
        (eta_t, eta_i, cos_i.abs())
    };
    let sin_i = (1.0 - cos_i * cos_i).max(0.0).sqrt();
    let sin_t = eta_i / eta_t * sin_i;
    if sin_t >= 1.0 {
        return 1.0;
    }
    let cos_t = (1.0 - sin_t * sin_t).max(0.0).sqrt();
    let r_par = (eta_t * cos_i - eta_i * cos_t) / (eta_t * cos_i + eta_i * cos_t);
    let r_per = (eta_i * cos_i - eta_t * cos_t) / (eta_i * cos_i + eta_t * cos_t);
    0.5 * (r_par * r_par + r_per * r_per)
}

fn ap(cos_theta_o: f32, eta: f32, h: f32, t: Vec3) -> [Vec3; P_MAX + 1] {
    let mut a = [Vec3::ZERO; P_MAX + 1];
    let cos_gamma_o = safe_sqrt(1.0 - h * h);
    let cos_theta = cos_theta_o * cos_gamma_o;
    let f = fr_dielectric(cos_theta, 1.0, eta);
    a[0] = Vec3::splat(f);
    a[1] = sqr(1.0 - f) * t;
    for p in 2..P_MAX {
        a[p] = a[p - 1] * t * f;
    }
    let denom = (Vec3::ONE - t * f).max(Vec3::splat(1.0e-12));
    a[P_MAX] = a[P_MAX - 1] * f * t / denom;
    a
}

fn phi(p: usize, gamma_o: f32, gamma_t: f32) -> f32 {
    2.0 * p as f32 * gamma_t - 2.0 * gamma_o + p as f32 * PI
}

fn logistic(x: f32, s: f32) -> f32 {
    let x = x.abs();
    (-x / s).exp() / (s * sqr(1.0 + (-x / s).exp()))
}

fn logistic_cdf(x: f32, s: f32) -> f32 {
    1.0 / (1.0 + (-x / s).exp())
}

fn trimmed_logistic(x: f32, s: f32, a: f32, b: f32) -> f32 {
    logistic(x, s) / (logistic_cdf(b, s) - logistic_cdf(a, s))
}

fn np(phi_d: f32, p: usize, s: f32, gamma_o: f32, gamma_t: f32) -> f32 {
    let mut dphi = phi_d - phi(p, gamma_o, gamma_t);
    while dphi > PI {
        dphi -= 2.0 * PI;
    }
    while dphi < -PI {
        dphi += 2.0 * PI;
    }
    trimmed_logistic(dphi, s, -PI, PI)
}

fn sample_trimmed_logistic(u: f32, s: f32, a: f32, b: f32) -> f32 {
    let k = logistic_cdf(b, s) - logistic_cdf(a, s);
    let x = -s * (1.0 / (u * k + logistic_cdf(a, s)) - 1.0).ln();
    x.clamp(a, b)
}

fn compact_1by1(mut x: u32) -> u32 {
    x &= 0x55555555;
    x = (x ^ (x >> 1)) & 0x33333333;
    x = (x ^ (x >> 2)) & 0x0f0f0f0f;
    x = (x ^ (x >> 4)) & 0x00ff00ff;
    x = (x ^ (x >> 8)) & 0x0000ffff;
    x
}

fn demux_float(f: f32) -> Vec2 {
    let f = f.clamp(0.0, 1.0 - 1.0e-7);
    let v = (f * (1u64 << 32) as f32) as u64;
    let bits0 = compact_1by1(v as u32);
    let bits1 = compact_1by1((v >> 1) as u32);
    Vec2::new(
        bits0 as f32 / (1u32 << 16) as f32,
        bits1 as f32 / (1u32 << 16) as f32,
    )
}

impl ChiangHairBsdf {
    /// `roughness_*` inputs are (ν, s) — variance and logistic scale —
    /// already produced by chiang_hair_roughness or passed directly.
    /// SQRT_PI_OVER_8 is applied to `s` to match the PBRT v4 trimmed
    /// logistic CDF normalization.
    pub fn from_mtlx(
        tint_r: Vec3,
        tint_tt: Vec3,
        tint_trt: Vec3,
        ior: f32,
        roughness_r: Vec2,
        roughness_tt: Vec2,
        roughness_trt: Vec2,
        cuticle_angle: f32,
        absorption: Vec3,
        curve_direction: Vec3,
        h: f32,
    ) -> Self {
        let clamp_v = |x: f32| x.max(1.0e-6);
        let v_r = clamp_v(roughness_r.x);
        let v_tt = clamp_v(roughness_tt.x);
        let v_trt = clamp_v(roughness_trt.x);
        let v = [v_r, v_tt, v_trt, v_trt];
        let s = SQRT_PI_OVER_8 * roughness_r.y.max(1.0e-6);
        let alpha_rad = cuticle_angle;
        let mut sin_2k = [0.0; 3];
        let mut cos_2k = [0.0; 3];
        sin_2k[0] = alpha_rad.sin();
        cos_2k[0] = (1.0 - sin_2k[0] * sin_2k[0]).max(0.0).sqrt();
        for i in 1..3 {
            sin_2k[i] = 2.0 * cos_2k[i - 1] * sin_2k[i - 1];
            cos_2k[i] = sqr(cos_2k[i - 1]) - sqr(sin_2k[i - 1]);
        }
        let h = h.clamp(-1.0, 1.0);
        Self {
            h,
            gamma_o: safe_asin(h),
            eta: ior.max(1.001),
            sigma_a: absorption.max(Vec3::ZERO),
            tint_r: tint_r.max(Vec3::ZERO),
            tint_tt: tint_tt.max(Vec3::ZERO),
            tint_trt: tint_trt.max(Vec3::ZERO),
            v,
            s,
            sin_2k_alpha: sin_2k,
            cos_2k_alpha: cos_2k,
            curve_direction,
        }
    }

    fn hair_basis(&self) -> (Vec3, Vec3, Vec3) {
        let t = self.curve_direction.normalize_or(Vec3::X);
        let z = Vec3::Z;
        let mut u = z - t * z.dot(t);
        if u.length_squared() < 1.0e-6 {
            u = Vec3::Y - t * Vec3::Y.dot(t);
            if u.length_squared() < 1.0e-6 {
                u = Vec3::X;
            }
        }
        let u = u.normalize();
        let v = t.cross(u);
        (t, u, v)
    }

    fn local_to_hair(&self, w_local: Vec3) -> Vec3 {
        let (t, u, v) = self.hair_basis();
        Vec3::new(w_local.dot(t), w_local.dot(u), w_local.dot(v))
    }

    fn hair_to_local(&self, w_hair: Vec3) -> Vec3 {
        let (t, u, v) = self.hair_basis();
        w_hair.x * t + w_hair.y * u + w_hair.z * v
    }

    fn theta_op_for_p(&self, sin_theta_o: f32, cos_theta_o: f32, p: usize) -> (f32, f32) {
        match p {
            0 => (
                sin_theta_o * self.cos_2k_alpha[1] - cos_theta_o * self.sin_2k_alpha[1],
                cos_theta_o * self.cos_2k_alpha[1] + sin_theta_o * self.sin_2k_alpha[1],
            ),
            1 => (
                sin_theta_o * self.cos_2k_alpha[0] + cos_theta_o * self.sin_2k_alpha[0],
                cos_theta_o * self.cos_2k_alpha[0] - sin_theta_o * self.sin_2k_alpha[0],
            ),
            2 => (
                sin_theta_o * self.cos_2k_alpha[2] + cos_theta_o * self.sin_2k_alpha[2],
                cos_theta_o * self.cos_2k_alpha[2] - sin_theta_o * self.sin_2k_alpha[2],
            ),
            _ => (sin_theta_o, cos_theta_o),
        }
    }

    fn lobe_tints(&self) -> [Vec3; 4] {
        [self.tint_r, self.tint_tt, self.tint_trt, self.tint_trt]
    }

    pub fn eval(&self, wo_local: Vec3, wi_local: Vec3) -> Vec3 {
        let wo = self.local_to_hair(wo_local);
        let wi = self.local_to_hair(wi_local);
        let sin_theta_o = wo.x;
        let cos_theta_o = safe_sqrt(1.0 - sqr(sin_theta_o));
        let phi_o = wo.z.atan2(wo.y);
        let sin_theta_i = wi.x;
        let cos_theta_i = safe_sqrt(1.0 - sqr(sin_theta_i));
        let phi_i = wi.z.atan2(wi.y);
        let sin_theta_t = sin_theta_o / self.eta;
        let cos_theta_t = safe_sqrt(1.0 - sqr(sin_theta_t));
        let etap =
            (self.eta * self.eta - sqr(sin_theta_o)).max(0.0).sqrt() / cos_theta_o.max(1.0e-6);
        let sin_gamma_t = self.h / etap;
        let cos_gamma_t = safe_sqrt(1.0 - sqr(sin_gamma_t));
        let gamma_t = safe_asin(sin_gamma_t);
        let t = (-self.sigma_a * (2.0 * cos_gamma_t / cos_theta_t.max(1.0e-6))).exp();
        let phi_d = phi_i - phi_o;
        let ap_arr = ap(cos_theta_o, self.eta, self.h, t);
        let tints = self.lobe_tints();
        let mut fsum = Vec3::ZERO;
        for p in 0..P_MAX {
            let (sin_op, cos_op) = self.theta_op_for_p(sin_theta_o, cos_theta_o, p);
            let cos_op = cos_op.abs();
            let m = mp(cos_theta_i, cos_op, sin_theta_i, sin_op, self.v[p]);
            let n = np(phi_d, p, self.s, self.gamma_o, gamma_t);
            fsum += m * ap_arr[p] * tints[p] * n;
        }
        fsum += mp(
            cos_theta_i,
            cos_theta_o,
            sin_theta_i,
            sin_theta_o,
            self.v[P_MAX],
        ) * ap_arr[P_MAX]
            * tints[P_MAX]
            / (2.0 * PI);
        let abs_cos_i = wi_local.z.abs().max(1.0e-6);
        fsum / abs_cos_i
    }

    fn compute_ap_pdf(&self, cos_theta_o: f32) -> [f32; P_MAX + 1] {
        let sin_theta_o = safe_sqrt(1.0 - cos_theta_o * cos_theta_o);
        let sin_theta_t = sin_theta_o / self.eta;
        let cos_theta_t = safe_sqrt(1.0 - sqr(sin_theta_t));
        let etap =
            (self.eta * self.eta - sqr(sin_theta_o)).max(0.0).sqrt() / cos_theta_o.max(1.0e-6);
        let sin_gamma_t = self.h / etap;
        let cos_gamma_t = safe_sqrt(1.0 - sqr(sin_gamma_t));
        let t = (-self.sigma_a * (2.0 * cos_gamma_t / cos_theta_t.max(1.0e-6))).exp();
        let ap_arr = ap(cos_theta_o, self.eta, self.h, t);
        let tints = self.lobe_tints();
        let mut weights = [0.0_f32; P_MAX + 1];
        let mut sum = 0.0_f32;
        for p in 0..=P_MAX {
            let lum = (ap_arr[p] * tints[p])
                .dot(Vec3::new(0.2722287, 0.6740818, 0.0536895))
                .max(0.0);
            weights[p] = lum;
            sum += lum;
        }
        if sum <= 0.0 {
            for w in weights.iter_mut() {
                *w = 1.0 / (P_MAX + 1) as f32;
            }
        } else {
            for w in weights.iter_mut() {
                *w /= sum;
            }
        }
        weights
    }

    pub fn pdf(&self, wo_local: Vec3, wi_local: Vec3) -> f32 {
        let wo = self.local_to_hair(wo_local);
        let wi = self.local_to_hair(wi_local);
        let sin_theta_o = wo.x;
        let cos_theta_o = safe_sqrt(1.0 - sqr(sin_theta_o));
        let phi_o = wo.z.atan2(wo.y);
        let sin_theta_i = wi.x;
        let cos_theta_i = safe_sqrt(1.0 - sqr(sin_theta_i));
        let phi_i = wi.z.atan2(wi.y);
        let etap =
            (self.eta * self.eta - sqr(sin_theta_o)).max(0.0).sqrt() / cos_theta_o.max(1.0e-6);
        let sin_gamma_t = self.h / etap;
        let gamma_t = safe_asin(sin_gamma_t);
        let ap_pdf = self.compute_ap_pdf(cos_theta_o);
        let phi_d = phi_i - phi_o;
        let mut pdf = 0.0_f32;
        for (p, &ap_pdf_p) in ap_pdf.iter().enumerate().take(P_MAX) {
            let (sin_op, cos_op) = self.theta_op_for_p(sin_theta_o, cos_theta_o, p);
            let cos_op = cos_op.abs();
            pdf += mp(cos_theta_i, cos_op, sin_theta_i, sin_op, self.v[p])
                * ap_pdf_p
                * np(phi_d, p, self.s, self.gamma_o, gamma_t);
        }
        pdf += mp(
            cos_theta_i,
            cos_theta_o,
            sin_theta_i,
            sin_theta_o,
            self.v[P_MAX],
        ) * ap_pdf[P_MAX]
            * (1.0 / (2.0 * PI));
        pdf
    }

    pub fn sample(&self, wo_local: Vec3, us: Vec2) -> Option<MtlxLobeSample> {
        let wo = self.local_to_hair(wo_local);
        let sin_theta_o = wo.x;
        let cos_theta_o = safe_sqrt(1.0 - sqr(sin_theta_o));
        let phi_o = wo.z.atan2(wo.y);
        let u0 = demux_float(us.x);
        let u1 = demux_float(us.y);
        let ap_pdf = self.compute_ap_pdf(cos_theta_o);
        let mut p = P_MAX;
        let mut u00 = u0.x;
        for (pp, &ap_pdf_pp) in ap_pdf.iter().enumerate().take(P_MAX) {
            if u00 < ap_pdf_pp {
                p = pp;
                break;
            }
            u00 -= ap_pdf_pp;
        }
        let (sin_op, cos_op) = self.theta_op_for_p(sin_theta_o, cos_theta_o, p);
        let u10 = u1.x.max(1.0e-5);
        let cos_theta = 1.0 + self.v[p] * (u10 + (1.0 - u10) * (-2.0 / self.v[p]).exp()).ln();
        let sin_theta = safe_sqrt(1.0 - sqr(cos_theta));
        let cos_phi = (2.0 * PI * u1.y).cos();
        let sin_theta_i = -cos_theta * sin_op + sin_theta * cos_phi * cos_op;
        let cos_theta_i = safe_sqrt(1.0 - sqr(sin_theta_i));
        let etap =
            (self.eta * self.eta - sqr(sin_theta_o)).max(0.0).sqrt() / cos_theta_o.max(1.0e-6);
        let sin_gamma_t = self.h / etap;
        let gamma_t = safe_asin(sin_gamma_t);
        let dphi = if p < P_MAX {
            phi(p, self.gamma_o, gamma_t) + sample_trimmed_logistic(u0.y, self.s, -PI, PI)
        } else {
            2.0 * PI * u0.y
        };
        let phi_i = phi_o + dphi;
        let wi_hair = Vec3::new(
            sin_theta_i,
            cos_theta_i * phi_i.cos(),
            cos_theta_i * phi_i.sin(),
        );
        let wi_local = self.hair_to_local(wi_hair);
        let f = self.eval(wo_local, wi_local);
        let pdf = self.pdf(wo_local, wi_local);
        if pdf <= 0.0 {
            return None;
        }
        let weight = f * wi_local.z.abs() / pdf;
        Some(MtlxLobeSample {
            weight,
            wi_local,
            pdf,
            flags: BsdfFlags::GLOSSY | BsdfFlags::REFLECTION,
            eta: 1.0,
        })
    }

    pub fn directional_albedo(&self, _wo: Vec3) -> Vec3 {
        (self.tint_r + self.tint_tt + self.tint_trt) / 3.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bsdf() -> ChiangHairBsdf {
        ChiangHairBsdf::from_mtlx(
            Vec3::ONE,
            Vec3::ONE,
            Vec3::ONE,
            1.55,
            Vec2::splat(0.3),
            Vec2::splat(0.3),
            Vec2::splat(0.3),
            0.5,
            Vec3::splat(0.06),
            Vec3::X,
            0.0,
        )
    }

    #[test]
    fn pdf_is_non_negative() {
        let b = make_bsdf();
        let wo = Vec3::new(0.2, 0.5, 0.8392).normalize();
        let wi = Vec3::new(-0.1, 0.5, 0.8602).normalize();
        let pdf = b.pdf(wo, wi);
        assert!(pdf >= 0.0);
    }

    #[test]
    fn eval_returns_finite_non_negative() {
        let b = make_bsdf();
        let wo = Vec3::new(0.1, 0.4, 0.911).normalize();
        let wi = Vec3::new(-0.2, 0.6, 0.7727).normalize();
        let f = b.eval(wo, wi);
        assert!(f.x.is_finite() && f.x >= 0.0);
        assert!(f.y.is_finite() && f.y >= 0.0);
        assert!(f.z.is_finite() && f.z >= 0.0);
    }

    #[test]
    fn ap_pdf_normalizes() {
        let b = make_bsdf();
        let pdf = b.compute_ap_pdf(0.9);
        let sum: f32 = pdf.iter().sum();
        assert!((sum - 1.0).abs() < 1.0e-3);
    }

    #[test]
    fn cuticle_angle_matches_mdl_radians() {
        let b = make_bsdf();
        assert!((b.sin_2k_alpha[0] - 0.5_f32.sin()).abs() < 1.0e-6);
        assert!((b.cos_2k_alpha[0] - 0.5_f32.cos()).abs() < 1.0e-6);
    }

    #[test]
    fn sample_produces_valid_direction() {
        let b = make_bsdf();
        let wo = Vec3::new(0.0, 0.0, 1.0);
        let mut hits = 0;
        for i in 0..16 {
            for j in 0..16 {
                let us = Vec2::new((i as f32 + 0.5) / 16.0, (j as f32 + 0.5) / 16.0);
                if let Some(s) = b.sample(wo, us)
                    && s.pdf > 0.0
                    && s.wi_local.is_finite()
                {
                    hits += 1;
                }
            }
        }
        assert!(hits > 0, "expected some valid samples");
    }
}
