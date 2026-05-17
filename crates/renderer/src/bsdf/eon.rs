use std::f32::consts::PI;

use glam::{Vec2, Vec3};

use super::{BsdfFlags, BsdfSample};

const RCP_PI: f32 = 1.0 / PI;
const CONSTANT1_FON: f32 = 0.5 - 2.0 / (3.0 * PI);
const CONSTANT2_FON: f32 = 2.0 / 3.0 - 28.0 / (15.0 * PI);
const EPS: f32 = 1.0e-7;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EonBsdf {
    rho: Vec3,
    r: f32,
}

impl EonBsdf {
    pub fn new(rho: Vec3, roughness: f32) -> Self {
        Self {
            rho,
            r: roughness.clamp(0.0, 1.0),
        }
    }

    pub fn eval(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        if wo.z <= 0.0 || wi.z <= 0.0 {
            return Vec3::ZERO;
        }
        f_eon(self.rho, self.r, wo, wi)
    }

    pub fn pdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        if wo.z <= 0.0 || wi.z <= 0.0 {
            return 0.0;
        }
        pdf_eon(wo, wi, self.r)
    }

    pub fn sample(&self, wo: Vec3, us: Vec2) -> Option<BsdfSample> {
        if wo.z <= 0.0 {
            return None;
        }
        let (wi, pdf) = sample_eon(wo, self.r, us.x, us.y);
        if pdf <= 0.0 || wi.z <= 0.0 {
            return None;
        }
        let f = f_eon(self.rho, self.r, wo, wi);
        let weight = f * (wi.z / pdf);
        Some(BsdfSample {
            weight,
            wi,
            pdf,
            flags: BsdfFlags::DIFFUSE | BsdfFlags::REFLECTION,
            eta: 1.0,
            wavelength_lock: None,
        })
    }
}

// Approximate FON directional albedo (paper Eq. 14, 4th-order polynomial).
fn e_fon_approx(mu: f32, r: f32) -> f32 {
    let mucomp = 1.0 - mu;
    let g_over_pi = mucomp
        * (0.057_108_53 + mucomp * (0.491_881_88 + mucomp * (-0.332_181_45 + mucomp * 0.071_443)));
    (1.0 + r * g_over_pi) / (1.0 + CONSTANT1_FON * r)
}

// Single-scatter FON lobe + analytic energy-compensation lobe (paper Eq. 16).
fn f_eon(rho: Vec3, r: f32, wo: Vec3, wi: Vec3) -> Vec3 {
    let mu_o = wo.z;
    let mu_i = wi.z;
    let s = wo.dot(wi) - mu_i * mu_o;
    let s_over_t = if s > 0.0 { s / mu_o.max(mu_i) } else { s };
    let af = 1.0 / (1.0 + CONSTANT1_FON * r);
    let f_ss = rho * (RCP_PI * af * (1.0 + r * s_over_t));

    let e_fo = e_fon_approx(mu_o, r);
    let e_fi = e_fon_approx(mu_i, r);
    let avg_ef = af * (1.0 + CONSTANT2_FON * r);
    let one_minus_avg = 1.0 - avg_ef;
    let denom = Vec3::ONE - rho * one_minus_avg;
    let rho_ms = rho * rho * avg_ef / denom;
    let scale = RCP_PI * (1.0 - e_fo).max(EPS) * (1.0 - e_fi).max(EPS) / one_minus_avg.max(EPS);
    f_ss + rho_ms * scale
}

fn ltc_coeffs(mu: f32, r: f32) -> (f32, f32, f32, f32) {
    let a =
        1.0 + r * (0.303392 + (-0.518982 + 0.111709 * mu) * mu + (-0.276266 + 0.335918 * mu) * r);
    let b =
        r * (-1.16407 + 1.15859 * mu + (0.150815 - 0.150105 * mu) * r) / (mu * mu * mu - 1.43545);
    let c = 1.0 + r * (0.20013 + (-0.506373 + 0.261777 * mu) * mu);
    let d = r * (0.540852 + (-1.01625 + 0.475392 * mu) * mu) / (-1.0743 + (0.0725628 + mu) * mu);
    (a, b, c, d)
}

// LTC frame X axis = projection of wo onto the tangent plane (paper Listing 2).
fn ltc_basis_x(wo: Vec3) -> Vec2 {
    let len_sqr = wo.x * wo.x + wo.y * wo.y;
    if len_sqr > 0.0 {
        let inv_len = len_sqr.sqrt().recip();
        Vec2::new(wo.x * inv_len, wo.y * inv_len)
    } else {
        Vec2::new(1.0, 0.0)
    }
}

// LTC -> tangent local. Z axis is unchanged (both share the surface normal).
fn ltc_apply_from(wo: Vec3, v: Vec3) -> Vec3 {
    let x = ltc_basis_x(wo);
    let y = Vec2::new(-x.y, x.x);
    Vec3::new(x.x * v.x + y.x * v.y, x.y * v.x + y.y * v.y, v.z)
}

// Inverse rotation (orthonormal -> transpose).
fn ltc_apply_to(wo: Vec3, v: Vec3) -> Vec3 {
    let x = ltc_basis_x(wo);
    let y = Vec2::new(-x.y, x.x);
    Vec3::new(x.x * v.x + x.y * v.y, y.x * v.x + y.y * v.y, v.z)
}

fn cltc_sample(wo: Vec3, r: f32, u1: f32, u2: f32) -> (Vec3, f32) {
    let (a, b, c, d) = ltc_coeffs(wo.z, r);
    let radius = u1.sqrt();
    let phi = 2.0 * PI * u2;
    let x0 = radius * phi.cos();
    let y0 = radius * phi.sin();
    let vz = (d * d + 1.0).sqrt().recip();
    let s = 0.5 * (1.0 + vz);
    let edge = (1.0 - y0 * y0).max(0.0).sqrt();
    let x = -((1.0 - s) * edge + s * x0);
    let y = y0;
    let wh_z = (1.0 - (x * x + y * y)).max(0.0).sqrt();
    let wh = Vec3::new(x, y, wh_z);
    let pdf_wh = wh.z / (PI * s);
    let wi_ltc = Vec3::new(a * wh.x + b * wh.z, c * wh.y, d * wh.x + wh.z);
    let len = wi_ltc.length();
    let det_m = c * (a - b * d);
    let pdf_wi = pdf_wh * (len * len * len) / det_m;
    let wi_local = ltc_apply_from(wo, wi_ltc / len);
    (wi_local, pdf_wi)
}

fn cltc_pdf(wo: Vec3, wi_local: Vec3, r: f32) -> f32 {
    let wi_ltc = ltc_apply_to(wo, wi_local);
    let (a, b, c, d) = ltc_coeffs(wo.z, r);
    let det_m = c * (a - b * d);
    let wh = Vec3::new(
        c * (wi_ltc.x - b * wi_ltc.z),
        (a - b * d) * wi_ltc.y,
        -c * (d * wi_ltc.x - a * wi_ltc.z),
    );
    let lensq = wh.dot(wh).max(1.0e-30);
    let vz = (d * d + 1.0).sqrt().recip();
    let s = 0.5 * (1.0 + vz);
    let det_over_lensq = det_m / lensq;
    det_over_lensq * det_over_lensq * wh.z.max(0.0) / (PI * s)
}

fn p_uniform(mu: f32, r: f32) -> f32 {
    r.powf(0.1) * (0.162925 + (-0.372058 + (0.538233 - 0.290822 * mu) * mu) * mu)
}

fn uniform_lobe_sample(u1: f32, u2: f32) -> Vec3 {
    let sin_theta = (1.0 - u1 * u1).max(0.0).sqrt();
    let phi = 2.0 * PI * u2;
    Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), u1)
}

fn sample_eon(wo: Vec3, r: f32, u1: f32, u2: f32) -> (Vec3, f32) {
    let mu = wo.z;
    let p_u = p_uniform(mu, r);
    let p_c = 1.0 - p_u;
    let (wi, pdf_c) = if u1 < p_u {
        let u1n = u1 / p_u;
        let wi0 = uniform_lobe_sample(u1n, u2);
        (wi0, cltc_pdf(wo, wi0, r))
    } else {
        let denom = p_c.max(1.0e-30);
        let u1n = (u1 - p_u) / denom;
        cltc_sample(wo, r, u1n, u2)
    };
    let pdf_u = 1.0 / (2.0 * PI);
    let pdf = p_u * pdf_u + p_c * pdf_c;
    (wi, pdf)
}

fn pdf_eon(wo: Vec3, wi: Vec3, r: f32) -> f32 {
    let mu = wo.z;
    let p_u = p_uniform(mu, r);
    let p_c = 1.0 - p_u;
    let pdf_c = cltc_pdf(wo, wi, r);
    let pdf_u = 1.0 / (2.0 * PI);
    p_u * pdf_u + p_c * pdf_c
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use glam::{Vec2, Vec3};

    use super::{EonBsdf, f_eon};

    #[test]
    fn lambert_reduction_when_roughness_zero() {
        let bsdf = EonBsdf::new(Vec3::splat(0.7), 0.0);
        let wo = Vec3::new(0.2, 0.3, 0.9327379).normalize();
        let wi = Vec3::new(-0.1, 0.4, 0.910).normalize();
        let f = bsdf.eval(wo, wi);
        assert!(f.abs_diff_eq(Vec3::splat(0.7) / PI, 1.0e-5));
    }

    #[test]
    fn upper_hemisphere_only() {
        let bsdf = EonBsdf::new(Vec3::ONE, 0.5);
        assert_eq!(bsdf.eval(Vec3::Z, -Vec3::Z), Vec3::ZERO);
        assert_eq!(bsdf.eval(-Vec3::Z, Vec3::Z), Vec3::ZERO);
        assert_eq!(bsdf.pdf(Vec3::Z, -Vec3::Z), 0.0);
        assert!(bsdf.sample(-Vec3::Z, Vec2::splat(0.5)).is_none());
    }

    #[test]
    fn reciprocity() {
        let rho = Vec3::new(0.6, 0.4, 0.2);
        for &r in &[0.2_f32, 0.5, 0.9, 1.0] {
            for (wo, wi) in [
                (Vec3::Z, Vec3::new(0.3, -0.4, 0.866).normalize()),
                (
                    Vec3::new(0.5, 0.0, 0.866).normalize(),
                    Vec3::new(-0.5, 0.2, 0.84).normalize(),
                ),
                (
                    Vec3::new(0.7, 0.0, 0.71414).normalize(),
                    Vec3::new(0.0, 0.7, 0.71414).normalize(),
                ),
            ] {
                let a = f_eon(rho, r, wo, wi);
                let b = f_eon(rho, r, wi, wo);
                assert!(a.abs_diff_eq(b, 1.0e-5), "r={r} a={a:?} b={b:?}",);
            }
        }
    }

    #[test]
    fn sample_weight_matches_eval_cos_over_pdf() {
        let bsdf = EonBsdf::new(Vec3::new(0.7, 0.5, 0.3), 0.6);
        let wo = Vec3::new(0.2, -0.1, 0.9746794).normalize();
        // Cover both the uniform-lobe branch (u1 small) and the CLTC branch.
        for &(u1, u2) in &[(0.01_f32, 0.42), (0.37, 0.82), (0.6, 0.1), (0.95, 0.5)] {
            let sample = bsdf
                .sample(wo, Vec2::new(u1, u2))
                .expect("expected a valid sample");
            let f = bsdf.eval(wo, sample.wi);
            let expected = f * (sample.wi.z / sample.pdf);
            assert!(
                sample.weight.abs_diff_eq(expected, 1.0e-4),
                "u=({u1},{u2}) weight={:?} expected={:?}",
                sample.weight,
                expected,
            );
        }
    }

    #[test]
    fn sample_pdf_matches_pdf_query() {
        let bsdf = EonBsdf::new(Vec3::ONE, 0.4);
        let wo = Vec3::new(0.3, -0.2, 0.932).normalize();
        for &(u1, u2) in &[(0.05_f32, 0.3), (0.5, 0.7), (0.85, 0.15)] {
            let sample = bsdf
                .sample(wo, Vec2::new(u1, u2))
                .expect("expected a valid sample");
            let pdf_q = bsdf.pdf(wo, sample.wi);
            assert!(
                (sample.pdf - pdf_q).abs() / sample.pdf.max(1.0e-6) < 1.0e-4,
                "sample.pdf={} pdf_query={}",
                sample.pdf,
                pdf_q,
            );
        }
    }

    fn integrate_pdf(bsdf: &EonBsdf, wo: Vec3, n_z: usize, n_phi: usize) -> f32 {
        let dz = 1.0 / n_z as f32;
        let dphi = std::f32::consts::TAU / n_phi as f32;
        let mut total = 0.0_f32;
        for zi in 0..n_z {
            let z = (zi as f32 + 0.5) * dz;
            let r = (1.0 - z * z).max(0.0).sqrt();
            for pi in 0..n_phi {
                let phi = (pi as f32 + 0.5) * dphi;
                let wi = Vec3::new(r * phi.cos(), r * phi.sin(), z);
                total += bsdf.pdf(wo, wi) * dz * dphi;
            }
        }
        total
    }

    #[test]
    fn pdf_integrates_to_one() {
        // dω = sin θ dθ dφ = dz dφ for cos-parametrised hemisphere.
        let bsdf = EonBsdf::new(Vec3::ONE, 0.6);
        for &wo in &[
            Vec3::Z,
            Vec3::new(0.4, 0.0, 0.9165151).normalize(),
            Vec3::new(0.0, 0.8, 0.6).normalize(),
        ] {
            let total = integrate_pdf(&bsdf, wo, 256, 256);
            assert!(
                (total - 1.0).abs() < 0.005,
                "wo={wo:?} pdf integral={total}",
            );
        }
    }

    fn integrate_brdf_cos(bsdf: &EonBsdf, wo: Vec3, n_z: usize, n_phi: usize) -> Vec3 {
        let dz = 1.0 / n_z as f32;
        let dphi = std::f32::consts::TAU / n_phi as f32;
        let mut e = Vec3::ZERO;
        for zi in 0..n_z {
            let z = (zi as f32 + 0.5) * dz;
            let r = (1.0 - z * z).max(0.0).sqrt();
            for pi in 0..n_phi {
                let phi = (pi as f32 + 0.5) * dphi;
                let wi = Vec3::new(r * phi.cos(), r * phi.sin(), z);
                e += bsdf.eval(wo, wi) * wi.z * dz * dphi;
            }
        }
        e
    }

    #[test]
    fn white_furnace_is_energy_preserving_at_normal_incidence() {
        // ρ = 1, r = 1, wo = N: ∫ f cos dω should equal 1 within ~1% (the
        // analytic energy-compensation term is the whole point of EON).
        let bsdf = EonBsdf::new(Vec3::ONE, 1.0);
        let energy = integrate_brdf_cos(&bsdf, Vec3::Z, 256, 256);
        for c in [energy.x, energy.y, energy.z] {
            assert!(
                (c - 1.0).abs() < 0.01,
                "energy at ρ=r=1, wo=N should be ~1.0, got {c}",
            );
        }
    }

    #[test]
    fn white_furnace_is_energy_preserving_at_grazing_incidence() {
        let bsdf = EonBsdf::new(Vec3::ONE, 1.0);
        let wo = Vec3::new(0.6, 0.0, 0.8); // ~37° from normal
        let energy = integrate_brdf_cos(&bsdf, wo, 256, 256);
        for c in [energy.x, energy.y, energy.z] {
            assert!(
                (c - 1.0).abs() < 0.02,
                "energy at ρ=r=1, wo=({:?}) should be ~1.0, got {c}",
                wo,
            );
        }
    }
}
