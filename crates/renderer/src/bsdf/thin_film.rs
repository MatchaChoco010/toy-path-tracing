use std::f32::consts::PI;

use glam::{Mat3, Vec2, Vec3};

const XYZ_TO_REC709: Mat3 = Mat3::from_cols(
    Vec3::new(3.2404542, -0.969_266, 0.055_643_4),
    Vec3::new(-1.537_138_5, 1.875_010_8, -0.204_025_9),
    Vec3::new(-0.498_531_4, 0.041_556, 1.057_225_2),
);

pub fn eval_thin_film_dielectric(
    cos_theta_i: f32,
    eta_outside: f32,
    eta_film: f32,
    eta_base: f32,
    thickness_nm: f32,
) -> Vec3 {
    eval_thin_film_inner(
        cos_theta_i,
        eta_outside,
        eta_film,
        Vec3::splat(eta_base),
        Vec3::ZERO,
        thickness_nm,
    )
}

pub fn eval_thin_film_conductor(
    cos_theta_i: f32,
    eta_outside: f32,
    eta_film: f32,
    n_base: Vec3,
    k_base: Vec3,
    thickness_nm: f32,
) -> Vec3 {
    eval_thin_film_inner(
        cos_theta_i,
        eta_outside,
        eta_film,
        n_base,
        k_base,
        thickness_nm,
    )
}

fn eval_thin_film_inner(
    cos_theta_i: f32,
    eta_outside: f32,
    eta_film: f32,
    n_base: Vec3,
    k_base: Vec3,
    thickness_nm: f32,
) -> Vec3 {
    let cos1 = cos_theta_i.clamp(0.0, 1.0);
    let eta_outside = eta_outside.max(1.0e-3);
    let eta_film = eta_film.max(1.0e-3);
    let eta_ratio = eta_outside / eta_film;
    let sin1_sq = (1.0 - cos1 * cos1).max(0.0);
    let sin2_sq = eta_ratio * eta_ratio * sin1_sq;
    if sin2_sq >= 1.0 {
        return Vec3::ONE;
    }
    let cos2 = (1.0 - sin2_sq).max(0.0).sqrt();

    let (r12, phi12) = fresnel_dielectric_polarized(cos1, eta_outside, eta_film);
    let phi21 = Vec2::new(PI - phi12.x, PI - phi12.y);
    let r12_sq = r12 * r12;
    let t121_sq = (Vec2::ONE - r12_sq).max(Vec2::ZERO);

    let (r23_s, r23_p, phi23_s, phi23_p) =
        fresnel_polarized_per_channel(cos2, eta_film, n_base, k_base);
    let r23_s_sq = r23_s * r23_s;
    let r23_p_sq = r23_p * r23_p;

    let opd = 2.0 * eta_film * thickness_nm * cos2;
    let phi2_s = Vec3::splat(phi21.x) + phi23_s;
    let phi2_p = Vec3::splat(phi21.y) + phi23_p;

    let mut i_s = Vec3::splat(r12_sq.x);
    let mut i_p = Vec3::splat(r12_sq.y);

    let r_geo_s = vec3_sqrt((Vec3::splat(r12_sq.x) * r23_s_sq).max(Vec3::ZERO));
    let r_geo_p = vec3_sqrt((Vec3::splat(r12_sq.y) * r23_p_sq).max(Vec3::ZERO));

    let mut cm_s = Vec3::splat(t121_sq.x) * r23_s_sq
        / (Vec3::ONE - Vec3::splat(r12_sq.x) * r23_s_sq).max(Vec3::splat(1.0e-6));
    let mut cm_p = Vec3::splat(t121_sq.y) * r23_p_sq
        / (Vec3::ONE - Vec3::splat(r12_sq.y) * r23_p_sq).max(Vec3::splat(1.0e-6));

    i_s += cm_s;
    i_p += cm_p;

    cm_s = Vec3::splat(t121_sq.x) * r23_s_sq;
    cm_p = Vec3::splat(t121_sq.y) * r23_p_sq;

    let max_orders = 2;
    for m in 1..=max_orders {
        let s_s = eval_sensitivity(m as f32 * opd, m as f32 * phi2_s);
        let s_p = eval_sensitivity(m as f32 * opd, m as f32 * phi2_p);
        i_s += 2.0 * cm_s * s_s;
        i_p += 2.0 * cm_p * s_p;
        cm_s *= r_geo_s;
        cm_p *= r_geo_p;
    }

    let result = 0.5 * (i_s + i_p);
    Vec3::new(
        result.x.clamp(0.0, 1.0),
        result.y.clamp(0.0, 1.0),
        result.z.clamp(0.0, 1.0),
    )
}

fn eval_sensitivity(opd: f32, shift: Vec3) -> Vec3 {
    let phase = 2.0 * PI * opd * 1.0e-9;
    let val = Vec3::new(5.4856e-13, 4.4201e-13, 5.2481e-13);
    let pos = Vec3::new(1.6810e+06, 1.7953e+06, 2.2084e+06);
    let var = Vec3::new(4.3278e+09, 9.3046e+09, 6.6121e+09);

    let two_pi = 2.0 * PI;
    let amp = vec3_sqrt(var * two_pi);
    let cosines = Vec3::new(
        (pos.x * phase + shift.x).cos(),
        (pos.y * phase + shift.y).cos(),
        (pos.z * phase + shift.z).cos(),
    );
    let exps = Vec3::new(
        (-phase * phase * var.x).exp(),
        (-phase * phase * var.y).exp(),
        (-phase * phase * var.z).exp(),
    );
    let mut xyz = val * amp * cosines * exps;

    let extra_var = 4.5282e+09;
    let extra_amp = (two_pi * extra_var).sqrt();
    let extra_cos = (2.2399e+06 * phase + shift.x).cos();
    let extra_exp = (-extra_var * phase * phase).exp();
    xyz.x += 9.7470e-14 * extra_amp * extra_cos * extra_exp;
    xyz /= 1.0685e-7;

    XYZ_TO_REC709 * xyz
}

fn fresnel_dielectric_polarized(cos_i: f32, eta_i: f32, eta_t: f32) -> (Vec2, Vec2) {
    let cos_i = cos_i.clamp(0.0, 1.0);
    let sin_i_sq = (1.0 - cos_i * cos_i).max(0.0);
    let sin_t_sq = (eta_i / eta_t).powi(2) * sin_i_sq;
    if sin_t_sq >= 1.0 {
        return (Vec2::ONE, Vec2::new(0.0, 0.0));
    }
    let cos_t = (1.0 - sin_t_sq).max(0.0).sqrt();
    let r_s = (eta_i * cos_i - eta_t * cos_t) / (eta_i * cos_i + eta_t * cos_t);
    let r_p = (eta_t * cos_i - eta_i * cos_t) / (eta_t * cos_i + eta_i * cos_t);
    let phi_s = if r_s < 0.0 { PI } else { 0.0 };
    let phi_p = if r_p < 0.0 { PI } else { 0.0 };
    (Vec2::new(r_s.abs(), r_p.abs()), Vec2::new(phi_s, phi_p))
}

fn fresnel_polarized_per_channel(
    cos_i: f32,
    eta_film: f32,
    n_base: Vec3,
    k_base: Vec3,
) -> (Vec3, Vec3, Vec3, Vec3) {
    let mut r_s = Vec3::ZERO;
    let mut r_p = Vec3::ZERO;
    let mut phi_s = Vec3::ZERO;
    let mut phi_p = Vec3::ZERO;
    for i in 0..3 {
        let n3 = n_base.to_array()[i];
        let k3 = k_base.to_array()[i];
        let (rs, rp, ps, pp) = fresnel_complex_polarized(cos_i, eta_film, n3, k3);
        match i {
            0 => {
                r_s.x = rs;
                r_p.x = rp;
                phi_s.x = ps;
                phi_p.x = pp;
            }
            1 => {
                r_s.y = rs;
                r_p.y = rp;
                phi_s.y = ps;
                phi_p.y = pp;
            }
            _ => {
                r_s.z = rs;
                r_p.z = rp;
                phi_s.z = ps;
                phi_p.z = pp;
            }
        }
    }
    (r_s, r_p, phi_s, phi_p)
}

fn fresnel_complex_polarized(
    cos_i: f32,
    eta_i: f32,
    eta_t: f32,
    kappa_t: f32,
) -> (f32, f32, f32, f32) {
    let cos_i = cos_i.clamp(0.0, 1.0);
    let cos_i2 = cos_i * cos_i;
    let sin_i2 = (1.0 - cos_i2).max(0.0);

    let eta_r = eta_t / eta_i.max(1.0e-6);
    let kappa_r = kappa_t / eta_i.max(1.0e-6);
    let eta_r2 = eta_r * eta_r;
    let kappa_r2 = kappa_r * kappa_r;

    let inner = eta_r2 - kappa_r2 - sin_i2;
    let radicand = (inner * inner + 4.0 * eta_r2 * kappa_r2).max(0.0);
    let a2_plus_b2 = radicand.sqrt();
    let a2 = (0.5 * (a2_plus_b2 + inner)).max(0.0);
    let b2 = (0.5 * (a2_plus_b2 - inner)).max(0.0);
    let a = a2.sqrt();

    let t1 = a2_plus_b2 + cos_i2;
    let t2 = 2.0 * a * cos_i;
    let denom_s = (t1 + t2).max(1.0e-12);
    let r_s2 = (t1 - t2) / denom_s;

    let t3 = a2_plus_b2 * cos_i2 + sin_i2 * sin_i2;
    let t4 = t2 * sin_i2;
    let denom_p = (t3 + t4).max(1.0e-12);
    let r_p2 = r_s2 * (t3 - t4) / denom_p;

    let r_s = r_s2.max(0.0).sqrt();
    let r_p = r_p2.max(0.0).sqrt();

    let phi_s_num = 2.0 * b2.sqrt() * cos_i;
    let phi_s_den = cos_i2 - a2 - b2;
    let phi_s = phi_s_num.atan2(phi_s_den);

    let u = (eta_r2 - kappa_r2) * cos_i - a;
    let v = 2.0 * eta_r * kappa_r * cos_i - b2.sqrt();
    let phi_p = ((u * u - v * v) / (u * u + v * v + 1.0e-12)).atan2(2.0 * u * v);

    let _ = phi_p;
    let phi_p = if r_p2 > 0.0 {
        let r_p_amp = r_p;
        let r_s_amp = r_s.max(1.0e-6);
        let _ = r_p_amp;
        let _ = r_s_amp;
        let p_num = 2.0 * cos_i * (b2.sqrt() * (eta_r2 + kappa_r2) - 2.0 * a * eta_r * kappa_r);
        let p_den = (eta_r2 + kappa_r2) * (eta_r2 + kappa_r2) * cos_i2 - (a2 + b2);
        p_num.atan2(p_den)
    } else {
        0.0
    };

    (r_s, r_p, phi_s, phi_p)
}

fn vec3_sqrt(v: Vec3) -> Vec3 {
    Vec3::new(
        v.x.max(0.0).sqrt(),
        v.y.max(0.0).sqrt(),
        v.z.max(0.0).sqrt(),
    )
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::{eval_thin_film_conductor, eval_thin_film_dielectric};

    #[test]
    fn zero_thickness_returns_finite_rgb() {
        let f = eval_thin_film_dielectric(0.7, 1.0, 1.5, 1.5, 0.0);
        assert!(f.is_finite());
        assert!(f.x >= 0.0 && f.y >= 0.0 && f.z >= 0.0);
    }

    #[test]
    fn dielectric_film_changes_response_with_thickness() {
        let cos_i = 0.85_f32;
        let f0 = eval_thin_film_dielectric(cos_i, 1.0, 2.4, 1.5, 0.0);
        let f250 = eval_thin_film_dielectric(cos_i, 1.0, 2.4, 1.5, 250.0);
        let f500 = eval_thin_film_dielectric(cos_i, 1.0, 2.4, 1.5, 500.0);
        let diff_a = (f0 - f250).abs().max_element();
        let diff_b = (f250 - f500).abs().max_element();
        assert!(diff_a > 1.0e-3 || diff_b > 1.0e-3);
    }

    #[test]
    fn conductor_film_returns_finite_rgb() {
        let n = Vec3::new(0.18, 0.42, 1.37);
        let k = Vec3::new(3.42, 2.35, 1.77);
        let f = eval_thin_film_conductor(0.7, 1.0, 2.0, n, k, 300.0);
        assert!(f.is_finite());
        assert!(f.x >= 0.0 && f.y >= 0.0 && f.z >= 0.0);
        assert!(f.x <= 1.0 && f.y <= 1.0 && f.z <= 1.0);
    }
}
