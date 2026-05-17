use glam::{Vec2, Vec3};

use super::smith_ggx::{
    EFFECTIVELY_SMOOTH_ALPHA, MIN_ALPHA, ggx_d, ggx_g1, ggx_g2_height_correlated,
    is_upper_hemisphere, pdf_wm_bounded_vndf, pdf_wm_vndf, reflect_local, reflection_half_vector,
    sample_wm_bounded_vndf,
};
use super::{BsdfFlags, BsdfSample};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConductorComplexGgxBsdf {
    n: Vec3,
    k: Vec3,
    alpha_x: f32,
    alpha_y: f32,
}

impl ConductorComplexGgxBsdf {
    pub fn new(n: Vec3, k: Vec3, alpha_x: f32, alpha_y: f32) -> Self {
        Self {
            n,
            k,
            alpha_x: alpha_x.max(MIN_ALPHA),
            alpha_y: alpha_y.max(MIN_ALPHA),
        }
    }

    pub fn eval(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        if !is_upper_hemisphere(wo) || !is_upper_hemisphere(wi) || self.effectively_smooth() {
            return Vec3::ZERO;
        }
        let Some(wm) = reflection_half_vector(wo, wi) else {
            return Vec3::ZERO;
        };
        let cos_o = wo.z.max(0.0);
        let cos_i = wi.z.max(0.0);
        if cos_o <= 0.0 || cos_i <= 0.0 {
            return Vec3::ZERO;
        }
        let d = ggx_d(wm, self.alpha_x, self.alpha_y);
        if d <= 0.0 {
            return Vec3::ZERO;
        }
        let g = ggx_g2_height_correlated(wo, wi, self.alpha_x, self.alpha_y);
        if g <= 0.0 {
            return Vec3::ZERO;
        }
        let f = fresnel_complex(wo.dot(wm).abs(), self.n, self.k);
        f * (d * g / (4.0 * cos_o * cos_i))
    }

    pub fn pdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        if !is_upper_hemisphere(wo) || !is_upper_hemisphere(wi) || self.effectively_smooth() {
            return 0.0;
        }
        let Some(wm) = reflection_half_vector(wo, wi) else {
            return 0.0;
        };
        let pdf_wm = pdf_wm_bounded_vndf(wo, wm, self.alpha_x, self.alpha_y);
        let denom = 4.0 * wo.dot(wm).abs();
        if denom <= 0.0 {
            return 0.0;
        }
        pdf_wm / denom
    }

    pub fn sample(&self, wo: Vec3, us: Vec2) -> Option<BsdfSample> {
        if !is_upper_hemisphere(wo) {
            return None;
        }
        if self.effectively_smooth() {
            return self.sample_smooth(wo);
        }
        let wm = sample_wm_bounded_vndf(wo, self.alpha_x, self.alpha_y, us)?;
        let wi = reflect_local(wo, wm);
        if !is_upper_hemisphere(wi) {
            return None;
        }
        let pdf_wm = pdf_wm_bounded_vndf(wo, wm, self.alpha_x, self.alpha_y);
        let denom = 4.0 * wo.dot(wm).abs();
        if denom <= 0.0 {
            return None;
        }
        let pdf = pdf_wm / denom;
        if pdf <= 0.0 {
            return None;
        }
        let g1 = ggx_g1(wo, self.alpha_x, self.alpha_y);
        if g1 <= 0.0 {
            return None;
        }
        let g2 = ggx_g2_height_correlated(wo, wi, self.alpha_x, self.alpha_y);
        let vndf_pdf_wm = pdf_wm_vndf(wo, wm, self.alpha_x, self.alpha_y);
        if vndf_pdf_wm <= 0.0 {
            return None;
        }
        let f = fresnel_complex(wo.dot(wm).abs(), self.n, self.k);
        let weight = f * ((g2 / g1) * (vndf_pdf_wm / pdf_wm));
        Some(BsdfSample {
            weight,
            wi,
            pdf,
            flags: BsdfFlags::GLOSSY | BsdfFlags::REFLECTION,
            eta: 1.0,
            wavelength_lock: None,
        })
    }

    fn sample_smooth(&self, wo: Vec3) -> Option<BsdfSample> {
        let wi = Vec3::new(-wo.x, -wo.y, wo.z).normalize_or_zero();
        if !is_upper_hemisphere(wi) {
            return None;
        }
        let weight = fresnel_complex(wi.z.abs(), self.n, self.k);
        Some(BsdfSample {
            weight,
            wi,
            pdf: 1.0,
            flags: BsdfFlags::DELTA | BsdfFlags::REFLECTION,
            eta: 1.0,
            wavelength_lock: None,
        })
    }

    fn effectively_smooth(&self) -> bool {
        self.alpha_x.max(self.alpha_y) < EFFECTIVELY_SMOOTH_ALPHA
    }
}

pub fn artist_friendly_complex_ior(face_color: Vec3, edge_color: Vec3) -> (Vec3, Vec3) {
    let r = face_color.clamp(Vec3::ZERO, Vec3::splat(0.999));
    let g = edge_color.clamp(Vec3::ZERO, Vec3::ONE);
    let r_sqrt = vec3_sqrt(r);
    let one = Vec3::ONE;
    let n_max = (one + r_sqrt) / (one - r_sqrt).max(Vec3::splat(1.0e-6));
    let n_min = (one - r) / (one + r);
    let n = g * n_min + (one - g) * n_max;
    let np1 = n + one;
    let nm1 = n - one;
    let k2 = (np1 * np1 * r - nm1 * nm1) / (one - r).max(Vec3::splat(1.0e-6));
    let k2 = k2.max(Vec3::ZERO);
    let k = vec3_sqrt(k2);
    (n, k)
}

pub fn fresnel_complex(cos_theta_i: f32, n: Vec3, k: Vec3) -> Vec3 {
    Vec3::new(
        fresnel_complex_scalar(cos_theta_i, n.x, k.x),
        fresnel_complex_scalar(cos_theta_i, n.y, k.y),
        fresnel_complex_scalar(cos_theta_i, n.z, k.z),
    )
}

fn fresnel_complex_scalar(cos_theta_i: f32, n: f32, k: f32) -> f32 {
    let cos_theta = cos_theta_i.clamp(0.0, 1.0);
    let cos2 = cos_theta * cos_theta;
    let sin2 = (1.0 - cos2).max(0.0);
    let n2 = n * n;
    let k2 = k * k;
    let inner = (n2 - k2 - sin2).max(-1.0e10);
    let radicand = (inner * inner + 4.0 * n2 * k2).max(0.0);
    let a2_plus_b2 = radicand.sqrt();
    let a2 = (0.5 * (a2_plus_b2 + inner)).max(0.0);
    let a = a2.sqrt();
    let t1 = a2_plus_b2 + cos2;
    let t2 = 2.0 * a * cos_theta;
    let denom_s = (t1 + t2).max(1.0e-12);
    let rs = (t1 - t2) / denom_s;
    let t3 = a2_plus_b2 * cos2 + sin2 * sin2;
    let t4 = t2 * sin2;
    let denom_p = (t3 + t4).max(1.0e-12);
    let rp = rs * (t3 - t4) / denom_p;
    (0.5 * (rs + rp)).clamp(0.0, 1.0)
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
    use glam::{Vec2, Vec3};

    use super::{ConductorComplexGgxBsdf, artist_friendly_complex_ior, fresnel_complex};
    use crate::bsdf::BsdfFlags;

    #[test]
    fn artist_ior_recovers_face_reflectance_at_normal_incidence() {
        let face = Vec3::new(0.95, 0.78, 0.35);
        let edge = Vec3::ONE;
        let (n, k) = artist_friendly_complex_ior(face, edge);
        let f = fresnel_complex(1.0, n, k);
        assert!(f.abs_diff_eq(face, 5.0e-3));
    }

    #[test]
    fn smooth_sample_returns_complex_fresnel_weight() {
        let face = Vec3::new(0.95, 0.78, 0.35);
        let (n, k) = artist_friendly_complex_ior(face, Vec3::ONE);
        let bsdf = ConductorComplexGgxBsdf::new(n, k, 1.0e-4, 1.0e-4);
        let wo = Vec3::Z;
        let sample = bsdf.sample(wo, Vec2::splat(0.5)).unwrap();
        assert!(sample.flags.contains(BsdfFlags::DELTA));
        assert!(sample.weight.abs_diff_eq(face, 5.0e-3));
    }

    #[test]
    fn rough_sample_matches_eval_cos_over_pdf() {
        let (n, k) = artist_friendly_complex_ior(Vec3::new(0.8, 0.6, 0.2), Vec3::ONE);
        let bsdf = ConductorComplexGgxBsdf::new(n, k, 0.35, 0.2);
        let wo = Vec3::new(0.2, -0.3, 0.9327379).normalize();
        let sample = bsdf.sample(wo, Vec2::new(0.37, 0.82)).unwrap();
        let f = bsdf.eval(wo, sample.wi);
        let expected = f * (sample.wi.z / sample.pdf);
        assert!(sample.weight.abs_diff_eq(expected, 1.0e-4));
    }

    #[test]
    fn fresnel_grazing_approaches_unity() {
        let (n, k) = artist_friendly_complex_ior(Vec3::splat(0.5), Vec3::ONE);
        let f = fresnel_complex(0.0, n, k);
        assert!(f.x > 0.95 && f.y > 0.95 && f.z > 0.95);
    }
}
