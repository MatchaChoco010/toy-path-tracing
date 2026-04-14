use std::f32::consts::TAU;

use glam::{Vec2, Vec3};

pub const MIN_ALPHA: f32 = 1.0e-4;
pub const EFFECTIVELY_SMOOTH_ALPHA: f32 = 1.0e-3;

pub fn is_upper_hemisphere(w: Vec3) -> bool {
    w.z > 0.0
}

pub fn reflect_local(wo: Vec3, wm: Vec3) -> Vec3 {
    (-wo + 2.0 * wo.dot(wm) * wm).normalize_or_zero()
}

pub fn reflection_half_vector(wo: Vec3, wi: Vec3) -> Option<Vec3> {
    let wm = (wo + wi).normalize_or_zero();
    if wm.length_squared() == 0.0 {
        return None;
    }
    Some(if wm.z < 0.0 { -wm } else { wm })
}

pub fn ggx_d(wm: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
    if wm.z <= 0.0 {
        return 0.0;
    }

    let term = wm.x * wm.x / (alpha_x * alpha_x) + wm.y * wm.y / (alpha_y * alpha_y) + wm.z * wm.z;
    let denom = std::f32::consts::PI * alpha_x * alpha_y * term * term;
    if denom <= 0.0 { 0.0 } else { 1.0 / denom }
}

pub fn ggx_lambda(w: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
    let cos_theta = w.z.abs();
    if cos_theta <= 0.0 {
        return f32::INFINITY;
    }

    let term = 1.0
        + (alpha_x * alpha_x * w.x * w.x + alpha_y * alpha_y * w.y * w.y) / (cos_theta * cos_theta);
    0.5 * (-1.0 + term.sqrt())
}

pub fn ggx_g1(w: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
    let lambda = ggx_lambda(w, alpha_x, alpha_y);
    if !lambda.is_finite() {
        return 0.0;
    }
    1.0 / (1.0 + lambda)
}

pub fn ggx_g2_height_correlated(wo: Vec3, wi: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
    let lambda_o = ggx_lambda(wo, alpha_x, alpha_y);
    let lambda_i = ggx_lambda(wi, alpha_x, alpha_y);
    if !lambda_o.is_finite() || !lambda_i.is_finite() {
        return 0.0;
    }
    1.0 / (1.0 + lambda_o + lambda_i)
}

pub fn pdf_wm_vndf(wo: Vec3, wm: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
    if !is_upper_hemisphere(wo) || wm.z <= 0.0 {
        return 0.0;
    }

    let d = ggx_d(wm, alpha_x, alpha_y);
    let g1 = ggx_g1(wo, alpha_x, alpha_y);
    let dot_wm_wo = wo.dot(wm).max(0.0);
    if d <= 0.0 || g1 <= 0.0 || dot_wm_wo <= 0.0 {
        return 0.0;
    }

    d * g1 * dot_wm_wo / wo.z
}

// Heitz 2018, "Sampling the GGX Distribution of Visible Normals" (JCGT).
pub fn sample_wm_vndf(wo: Vec3, alpha_x: f32, alpha_y: f32, us: Vec2) -> Option<Vec3> {
    if wo.z <= 0.0 {
        return None;
    }

    let alpha_x = alpha_x.max(MIN_ALPHA);
    let alpha_y = alpha_y.max(MIN_ALPHA);

    // Stretch wo to standard (isotropic) space.
    let wo_std = Vec3::new(alpha_x * wo.x, alpha_y * wo.y, wo.z).normalize_or_zero();
    if wo_std.length_squared() == 0.0 {
        return None;
    }

    // Orthonormal basis in stretched space.
    let lensq = wo_std.x * wo_std.x + wo_std.y * wo_std.y;
    let t1 = if lensq > 0.0 {
        Vec3::new(-wo_std.y, wo_std.x, 0.0) / lensq.sqrt()
    } else {
        Vec3::X
    };
    let t2 = wo_std.cross(t1);

    // Parameterize projected area.
    let r = us.x.clamp(0.0, 1.0).sqrt();
    let phi = TAU * us.y;
    let p1 = r * phi.cos();
    let mut p2 = r * phi.sin();
    let s = 0.5 * (1.0 + wo_std.z);
    p2 = (1.0 - s) * (1.0 - p1 * p1).max(0.0).sqrt() + s * p2;

    // Reproject onto hemisphere.
    let nh = p1 * t1
        + p2 * t2
        + (1.0 - p1 * p1 - p2 * p2).max(0.0).sqrt() * wo_std;

    // Unstretch back to anisotropic space.
    let wm = Vec3::new(alpha_x * nh.x, alpha_y * nh.y, nh.z.max(0.0)).normalize_or_zero();

    if wm.length_squared() == 0.0 || wm.z <= 0.0 {
        return None;
    }

    Some(wm)
}

pub fn sample_wm_bounded_vndf(wo: Vec3, alpha_x: f32, alpha_y: f32, us: Vec2) -> Option<Vec3> {
    if wo.z <= 0.0 {
        return None;
    }

    let alpha_x = alpha_x.max(MIN_ALPHA);
    let alpha_y = alpha_y.max(MIN_ALPHA);
    let wo_std = Vec3::new(alpha_x * wo.x, alpha_y * wo.y, wo.z).normalize_or_zero();
    if wo_std.length_squared() == 0.0 {
        return None;
    }

    let phi = TAU * us.x;
    let a = alpha_x.min(alpha_y).clamp(0.0, 1.0);
    let s = 1.0 + wo.truncate().length();
    let a2 = a * a;
    let s2 = s * s;
    let k = (1.0 - a2) * s2 / (s2 + a2 * wo.z * wo.z);
    let lower_bound = if wo.z > 0.0 { -k * wo_std.z } else { -wo_std.z };
    let z = lower_bound.mul_add(us.y, 1.0 - us.y);
    let sin_theta = (1.0 - z * z).clamp(0.0, 1.0).sqrt();
    let o_std = Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), z);
    let wm_std = wo_std + o_std;
    let wm = Vec3::new(alpha_x * wm_std.x, alpha_y * wm_std.y, wm_std.z).normalize_or_zero();

    if wm.length_squared() == 0.0 || wm.z <= 0.0 {
        return None;
    }

    Some(wm)
}

pub fn pdf_wm_bounded_vndf(wo: Vec3, wm: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
    if wo.z <= 0.0 || wm.z <= 0.0 {
        return 0.0;
    }

    let alpha_x = alpha_x.max(MIN_ALPHA);
    let alpha_y = alpha_y.max(MIN_ALPHA);
    let a = alpha_x.min(alpha_y).clamp(0.0, 1.0);
    let s = 1.0 + wo.truncate().length();
    let a2 = a * a;
    let s2 = s * s;
    let k = (1.0 - a2) * s2 / (s2 + a2 * wo.z * wo.z);
    let t =
        (alpha_x * alpha_x * wo.x * wo.x + alpha_y * alpha_y * wo.y * wo.y + wo.z * wo.z).sqrt();
    let denom = k * wo.z + t;
    if denom <= 0.0 {
        return 0.0;
    }

    2.0 * ggx_d(wm, alpha_x, alpha_y) * wo.dot(wm).max(0.0) / denom
}

#[cfg(test)]
mod tests {
    use std::f32::consts::TAU;

    use glam::{Vec2, Vec3};

    use super::{
        ggx_d, pdf_wm_bounded_vndf, pdf_wm_vndf, sample_wm_bounded_vndf, sample_wm_vndf,
    };

    const HEMISPHERE_Z_SAMPLES: usize = 256;
    const HEMISPHERE_PHI_SAMPLES: usize = 256;

    fn integrate_hemisphere(f: impl Fn(Vec3) -> f32) -> f32 {
        let dz = 1.0 / HEMISPHERE_Z_SAMPLES as f32;
        let dphi = TAU / HEMISPHERE_PHI_SAMPLES as f32;
        let domega = dz * dphi;
        let mut integral = 0.0;

        for z_index in 0..HEMISPHERE_Z_SAMPLES {
            let z = (z_index as f32 + 0.5) * dz;
            let r = (1.0 - z * z).max(0.0).sqrt();

            for phi_index in 0..HEMISPHERE_PHI_SAMPLES {
                let phi = (phi_index as f32 + 0.5) * dphi;
                let w = Vec3::new(r * phi.cos(), r * phi.sin(), z);
                integral += f(w);
            }
        }

        integral * domega
    }

    #[test]
    fn ggx_ndf_is_normalized_over_projected_hemisphere() {
        let configs = [(0.2, 0.2), (0.35, 0.7), (0.8, 0.15)];

        for (alpha_x, alpha_y) in configs {
            let integral = integrate_hemisphere(|wm| ggx_d(wm, alpha_x, alpha_y) * wm.z);

            assert!(integral.is_finite());
            assert!(
                (integral - 1.0).abs() < 5.0e-3,
                "alpha_x={alpha_x}, alpha_y={alpha_y}, integral={integral}"
            );
        }
    }

    #[test]
    fn vndf_pdf_is_normalized() {
        let configs = [
            (Vec3::Z, 0.2, 0.2),
            (Vec3::new(0.3, -0.4, 0.8660254).normalize(), 0.35, 0.7),
            (Vec3::new(0.8, 0.0, 0.6).normalize(), 0.8, 0.15),
        ];

        for (wo, alpha_x, alpha_y) in configs {
            let integral = integrate_hemisphere(|wm| pdf_wm_vndf(wo, wm, alpha_x, alpha_y));

            assert!(integral.is_finite());
            assert!(
                (integral - 1.0).abs() < 5.0e-3,
                "wo={wo:?}, alpha_x={alpha_x}, alpha_y={alpha_y}, integral={integral}"
            );
        }
    }

    #[test]
    fn bounded_pdf_over_half_vectors_is_positive_for_valid_configuration() {
        let wo = Vec3::new(0.1, 0.2, 0.9746794).normalize();
        let wm = Vec3::new(0.05, -0.15, 0.9874209).normalize();
        let pdf = pdf_wm_bounded_vndf(wo, wm, 0.4, 0.25);

        assert!(pdf > 0.0);
    }

    #[test]
    fn vndf_sample_returns_upper_hemisphere_normal() {
        let cases = [
            (Vec3::new(0.3, -0.4, 0.8660254).normalize(), 0.2, 0.2),
            (Vec3::new(-0.2, 0.3, 0.9327379).normalize(), 0.35, 0.2),
            (Vec3::Z, 0.8, 0.15),
        ];

        for (wo, alpha_x, alpha_y) in cases {
            for y in 0..4 {
                for x in 0..4 {
                    let us = Vec2::new((x as f32 + 0.5) / 4.0, (y as f32 + 0.5) / 4.0);
                    let wm = sample_wm_vndf(wo, alpha_x, alpha_y, us)
                        .expect("expected a VNDF sample");

                    assert!(wm.is_finite());
                    assert!(wm.z > 0.0);
                    assert!((wm.length() - 1.0).abs() < 1.0e-4);
                    assert!(pdf_wm_vndf(wo, wm, alpha_x, alpha_y) > 0.0);
                }
            }
        }
    }

    #[test]
    fn bounded_vndf_sample_returns_upper_hemisphere_normal() {
        let cases = [
            (Vec3::new(0.3, -0.4, 0.8660254).normalize(), 0.2, 0.2),
            (Vec3::new(-0.2, 0.3, 0.9327379).normalize(), 0.35, 0.2),
        ];

        for (wo, alpha_x, alpha_y) in cases {
            for y in 0..4 {
                for x in 0..4 {
                    let us = Vec2::new((x as f32 + 0.5) / 4.0, (y as f32 + 0.5) / 4.0);
                    let wm = sample_wm_bounded_vndf(wo, alpha_x, alpha_y, us)
                        .expect("expected a bounded VNDF sample");

                    assert!(wm.is_finite());
                    assert!(wm.z > 0.0);
                }
            }
        }
    }
}
