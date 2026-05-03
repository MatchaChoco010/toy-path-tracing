use std::f32::consts::PI;

use glam::{Vec2, Vec3};
use rand::{RngExt, rngs::ThreadRng};

use crate::math::{sample_cosine_weighted_hemisphere, sg};

use super::gtr1::{d_gtr1, pdf_h_gtr1, sample_h_gtr1};
use super::smith_ggx::{
    MIN_ALPHA, ggx_d, ggx_g2_height_correlated, ggx_lambda, is_upper_hemisphere, pdf_wm_vndf,
    reflection_half_vector, sample_wm_vndf,
};
use super::{BsdfFlags, BsdfSample};

#[derive(Debug, Clone, Copy)]
pub struct DisneyBrdfBsdf {
    base_color: Vec3,
    metallic: f32,
    subsurface: f32,
    roughness: f32,
    sheen: f32,
    clearcoat: f32,
    c_spec0: Vec3,
    c_sheen: Vec3,
    alpha_x: f32,
    alpha_y: f32,
    alpha_cc: f32,
}

impl DisneyBrdfBsdf {
    pub fn new(
        base_color: Vec3,
        metallic: f32,
        subsurface: f32,
        specular: f32,
        specular_tint: f32,
        roughness: f32,
        anisotropic: f32,
        sheen: f32,
        sheen_tint: f32,
        clearcoat: f32,
        clearcoat_gloss: f32,
    ) -> Self {
        let base_color = base_color.clamp(Vec3::ZERO, Vec3::ONE);
        let metallic = metallic.clamp(0.0, 1.0);
        let subsurface = subsurface.clamp(0.0, 1.0);
        let specular = specular.clamp(0.0, 1.0);
        let specular_tint = specular_tint.clamp(0.0, 1.0);
        let roughness = roughness.clamp(0.0, 1.0);
        let anisotropic = anisotropic.clamp(0.0, 1.0);
        let sheen = sheen.clamp(0.0, 1.0);
        let sheen_tint = sheen_tint.clamp(0.0, 1.0);
        let clearcoat = clearcoat.clamp(0.0, 1.0);
        let clearcoat_gloss = clearcoat_gloss.clamp(0.0, 1.0);

        let lum = sg::luminance(base_color);
        let c_tint = if lum > 0.0 {
            base_color / lum
        } else {
            Vec3::ONE
        };
        let dielectric_f0 = 0.08 * specular * Vec3::ONE.lerp(c_tint, specular_tint);
        let c_spec0 = dielectric_f0.lerp(base_color, metallic);
        let c_sheen = Vec3::ONE.lerp(c_tint, sheen_tint);

        let alpha = (roughness * roughness).max(MIN_ALPHA);
        let aspect = (1.0 - 0.9 * anisotropic).sqrt();
        let alpha_x = (alpha / aspect).max(MIN_ALPHA);
        let alpha_y = (alpha * aspect).max(MIN_ALPHA);
        let alpha_cc = (0.1 * (1.0 - clearcoat_gloss) + 0.001 * clearcoat_gloss).max(1.0e-3);

        Self {
            base_color,
            metallic,
            subsurface,
            roughness,
            sheen,
            clearcoat,
            c_spec0,
            c_sheen,
            alpha_x,
            alpha_y,
            alpha_cc,
        }
    }

    pub fn eval(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        if !is_upper_hemisphere(wo) || !is_upper_hemisphere(wi) {
            return Vec3::ZERO;
        }
        let Some(h) = reflection_half_vector(wo, wi) else {
            return Vec3::ZERO;
        };
        self.eval_lobes(wo, wi, h)
    }

    pub fn pdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        if !is_upper_hemisphere(wo) || !is_upper_hemisphere(wi) {
            return 0.0;
        }
        let Some(h) = reflection_half_vector(wo, wi) else {
            return 0.0;
        };
        self.pdf_total(wo, wi, h)
    }

    pub fn sample(&self, wo: Vec3, rng: &mut ThreadRng) -> Option<BsdfSample> {
        if !is_upper_hemisphere(wo) {
            return None;
        }

        let (p_d, p_s, p_c) = self.lobe_probabilities();
        if p_d <= 0.0 && p_s <= 0.0 && p_c <= 0.0 {
            return None;
        }

        let u_lobe = rng.random::<f32>();
        let us = Vec2::new(rng.random::<f32>(), rng.random::<f32>());

        let wi = if u_lobe < p_d {
            sample_cosine_weighted_hemisphere(us)
        } else if u_lobe < p_d + p_s {
            let wm = sample_wm_vndf(wo, self.alpha_x, self.alpha_y, us)?;
            reflect_local(wo, wm)
        } else {
            let h = sample_h_gtr1(self.alpha_cc, us);
            reflect_local(wo, h)
        };

        if !is_upper_hemisphere(wi) {
            return None;
        }
        let h = reflection_half_vector(wo, wi)?;

        let pdf = self.pdf_total(wo, wi, h);
        if pdf <= 0.0 {
            return None;
        }
        let f = self.eval_lobes(wo, wi, h);
        if f.length_squared() == 0.0 {
            return None;
        }

        let cos_i = wi.z.max(0.0);
        if cos_i <= 0.0 {
            return None;
        }

        Some(BsdfSample {
            weight: f * (cos_i / pdf),
            wi,
            pdf,
            flags: BsdfFlags::GLOSSY | BsdfFlags::DIFFUSE | BsdfFlags::REFLECTION,
            eta: 1.0,
        })
    }

    fn lobe_probabilities(&self) -> (f32, f32, f32) {
        let diffuse = (sg::luminance(self.base_color) * (1.0 - self.metallic)).max(0.0);
        let specular = sg::luminance(self.c_spec0).max(0.0);
        let clearcoat = (0.25 * self.clearcoat * 0.04).max(0.0);
        let total = diffuse + specular + clearcoat;
        if total <= 0.0 {
            (0.0, 0.0, 0.0)
        } else {
            (diffuse / total, specular / total, clearcoat / total)
        }
    }

    fn eval_lobes(&self, wo: Vec3, wi: Vec3, h: Vec3) -> Vec3 {
        let n_dot_l = wi.z.max(0.0);
        let n_dot_v = wo.z.max(0.0);
        if n_dot_l <= 0.0 || n_dot_v <= 0.0 {
            return Vec3::ZERO;
        }
        let l_dot_h = wi.dot(h).max(0.0);

        let f_l = schlick5(n_dot_l);
        let f_v = schlick5(n_dot_v);
        let fd90 = 0.5 + 2.0 * l_dot_h * l_dot_h * self.roughness;
        let f_d = (1.0 + (fd90 - 1.0) * f_l) * (1.0 + (fd90 - 1.0) * f_v);

        let fss90 = l_dot_h * l_dot_h * self.roughness;
        let f_ss_inner = (1.0 + (fss90 - 1.0) * f_l) * (1.0 + (fss90 - 1.0) * f_v);
        let denom_ss = (n_dot_l + n_dot_v).max(1.0e-4);
        let ss = 1.25 * (f_ss_inner * (1.0 / denom_ss - 0.5) + 0.5);

        let diffuse_shape = (1.0 - self.subsurface) * f_d + self.subsurface * ss;
        let diffuse = self.base_color * (diffuse_shape / PI);

        let fh = schlick5(l_dot_h);
        let sheen = self.c_sheen * (self.sheen * fh);

        let layer_diffuse_sheen = (diffuse + sheen) * (1.0 - self.metallic);

        let d_s = ggx_d(h, self.alpha_x, self.alpha_y);
        let g_s = ggx_g2_height_correlated(wo, wi, self.alpha_x, self.alpha_y);
        let f_s = self.c_spec0.lerp(Vec3::ONE, fh);
        let denom_s = 4.0 * n_dot_v * n_dot_l;
        let primary_specular = if denom_s > 0.0 {
            f_s * (d_s * g_s / denom_s)
        } else {
            Vec3::ZERO
        };

        let d_r = d_gtr1(h.z, self.alpha_cc);
        let g_r = smith_g2_iso(wo, wi, 0.25);
        let f_r = 0.04_f32 * (1.0 - fh) + fh;
        let clearcoat = if denom_s > 0.0 {
            Vec3::splat(0.25 * self.clearcoat * d_r * g_r * f_r / denom_s)
        } else {
            Vec3::ZERO
        };

        layer_diffuse_sheen + primary_specular + clearcoat
    }

    fn pdf_total(&self, wo: Vec3, wi: Vec3, h: Vec3) -> f32 {
        let (p_d, p_s, p_c) = self.lobe_probabilities();
        if p_d <= 0.0 && p_s <= 0.0 && p_c <= 0.0 {
            return 0.0;
        }

        let pdf_d = wi.z.max(0.0) / PI;

        let pdf_s = if p_s > 0.0 {
            let pdf_wm = pdf_wm_vndf(wo, h, self.alpha_x, self.alpha_y);
            let denom = 4.0 * wo.dot(h).abs();
            if denom > 0.0 { pdf_wm / denom } else { 0.0 }
        } else {
            0.0
        };

        let pdf_c = if p_c > 0.0 {
            let pdf_h = pdf_h_gtr1(h.z, self.alpha_cc);
            let denom = 4.0 * wo.dot(h).abs();
            if denom > 0.0 { pdf_h / denom } else { 0.0 }
        } else {
            0.0
        };

        p_d * pdf_d + p_s * pdf_s + p_c * pdf_c
    }
}

fn reflect_local(wo: Vec3, n: Vec3) -> Vec3 {
    (-wo + 2.0 * wo.dot(n) * n).normalize_or_zero()
}

fn schlick5(x: f32) -> f32 {
    let m = (1.0 - x).clamp(0.0, 1.0);
    let m2 = m * m;
    m2 * m2 * m
}

fn smith_g2_iso(wo: Vec3, wi: Vec3, alpha: f32) -> f32 {
    let lambda_o = ggx_lambda(wo, alpha, alpha);
    let lambda_i = ggx_lambda(wi, alpha, alpha);
    if !lambda_o.is_finite() || !lambda_i.is_finite() {
        return 0.0;
    }
    1.0 / (1.0 + lambda_o + lambda_i)
}

#[cfg(test)]
mod tests {
    use std::f32::consts::TAU;

    use glam::Vec3;

    use super::DisneyBrdfBsdf;

    const HEMISPHERE_Z_SAMPLES: usize = 128;
    const HEMISPHERE_PHI_SAMPLES: usize = 128;

    fn integrate_hemisphere_vec3(f: impl Fn(Vec3) -> Vec3) -> Vec3 {
        let dz = 1.0 / HEMISPHERE_Z_SAMPLES as f32;
        let dphi = TAU / HEMISPHERE_PHI_SAMPLES as f32;
        let domega = dz * dphi;
        let mut acc = Vec3::ZERO;
        for zi in 0..HEMISPHERE_Z_SAMPLES {
            let z = (zi as f32 + 0.5) * dz;
            let r = (1.0 - z * z).max(0.0).sqrt();
            for pi in 0..HEMISPHERE_PHI_SAMPLES {
                let phi = (pi as f32 + 0.5) * dphi;
                let w = Vec3::new(r * phi.cos(), r * phi.sin(), z);
                acc += f(w);
            }
        }
        acc * domega
    }

    fn default_bsdf() -> DisneyBrdfBsdf {
        DisneyBrdfBsdf::new(
            Vec3::new(0.82, 0.67, 0.16),
            0.0,
            0.0,
            0.5,
            0.0,
            0.5,
            0.0,
            0.0,
            0.5,
            0.0,
            1.0,
        )
    }

    #[test]
    fn eval_is_zero_for_lower_hemisphere_inputs() {
        let bsdf = default_bsdf();
        assert_eq!(bsdf.eval(Vec3::Z, Vec3::NEG_Z), Vec3::ZERO);
        assert_eq!(bsdf.eval(Vec3::NEG_Z, Vec3::Z), Vec3::ZERO);
        assert_eq!(bsdf.pdf(Vec3::Z, Vec3::NEG_Z), 0.0);
    }

    #[test]
    fn reciprocity_holds_for_smooth_directions() {
        let bsdf = DisneyBrdfBsdf::new(
            Vec3::new(0.82, 0.67, 0.16),
            0.5,
            0.0,
            0.5,
            0.0,
            0.4,
            0.0,
            0.3,
            0.5,
            0.5,
            1.0,
        );
        let wo = Vec3::new(0.2, -0.3, 0.9327379).normalize();
        let wi = Vec3::new(-0.1, 0.4, 0.910).normalize();

        let f_io = bsdf.eval(wo, wi);
        let f_oi = bsdf.eval(wi, wo);
        assert!(f_io.abs_diff_eq(f_oi, 1.0e-3));
    }

    #[test]
    fn metallic_one_disables_diffuse_layer() {
        let bsdf = DisneyBrdfBsdf::new(
            Vec3::new(0.82, 0.67, 0.16),
            1.0,
            0.0,
            0.5,
            0.0,
            0.5,
            0.0,
            1.0,
            0.5,
            0.0,
            1.0,
        );
        let wo = Vec3::new(0.0, 0.0, 1.0);
        let metallic_energy = integrate_hemisphere_vec3(|wi| bsdf.eval(wo, wi) * wi.z.max(0.0));
        assert!(metallic_energy.max_element() < 1.0 + 5.0e-3);
    }

    #[test]
    fn sheen_contribution_matches_burley_2012_formula() {
        let no_sheen = DisneyBrdfBsdf::new(
            Vec3::new(0.45, 0.10, 0.08),
            0.0,
            0.0,
            0.0,
            0.0,
            0.7,
            0.0,
            0.0,
            0.5,
            0.0,
            1.0,
        );
        let with_sheen = DisneyBrdfBsdf::new(
            Vec3::new(0.45, 0.10, 0.08),
            0.0,
            0.0,
            0.0,
            0.0,
            0.7,
            0.0,
            1.0,
            0.5,
            0.0,
            1.0,
        );

        let wo = Vec3::new(0.866_025_4_f32, 0.0, 0.5).normalize();
        let wi = Vec3::new(-0.866_025_4_f32, 0.0, 0.5).normalize();

        let f_no = no_sheen.eval(wo, wi);
        let f_with = with_sheen.eval(wo, wi);
        let delta = f_with - f_no;

        // Csheen at sheenTint=0.5 with base (0.45, 0.10, 0.08), Rec.709 lum:
        //   lum = 0.1729; Ctint = (2.6027, 0.5783, 0.4626)
        //   Csheen = (1.8014, 0.7891, 0.7313)
        //   F_sheen at LdotH=0.5 = Csheen * 0.03125 = (0.05629, 0.02466, 0.02285)
        let expected_sheen = Vec3::new(0.056_29, 0.024_66, 0.022_85);
        assert!(
            delta.abs_diff_eq(expected_sheen, 5.0e-4),
            "delta={delta:?} expected={expected_sheen:?}"
        );
    }

    #[test]
    fn sheen_peaks_at_grazing_l_dot_h() {
        let bsdf = DisneyBrdfBsdf::new(
            Vec3::new(0.5, 0.5, 0.5),
            0.0,
            0.0,
            0.0,
            0.0,
            0.7,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
        );
        let bsdf_no = DisneyBrdfBsdf::new(
            Vec3::new(0.5, 0.5, 0.5),
            0.0,
            0.0,
            0.0,
            0.0,
            0.7,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        );

        let wo = Vec3::new(0.866_025_4_f32, 0.0, 0.5).normalize();
        let mut last_sheen_part = f32::INFINITY;
        for cos_offset in [0.5_f32, 0.6, 0.7, 0.8, 0.9, 0.95] {
            let sin = (1.0 - cos_offset * cos_offset).sqrt();
            let wi = Vec3::new(sin, 0.0, cos_offset).normalize();
            let f = bsdf.eval(wo, wi);
            let f_diffuse_only = bsdf_no.eval(wo, wi);
            let sheen_part = (f - f_diffuse_only).y;
            assert!(sheen_part >= 0.0);
            assert!(sheen_part < last_sheen_part + 1.0e-5);
            last_sheen_part = sheen_part;
        }
    }

    #[test]
    fn clearcoat_zero_makes_clearcoat_lobe_inactive() {
        let no_cc = DisneyBrdfBsdf::new(
            Vec3::new(0.82, 0.67, 0.16),
            0.0,
            0.0,
            0.5,
            0.0,
            0.5,
            0.0,
            0.0,
            0.5,
            0.0,
            1.0,
        );
        let with_cc = DisneyBrdfBsdf::new(
            Vec3::new(0.82, 0.67, 0.16),
            0.0,
            0.0,
            0.5,
            0.0,
            0.5,
            0.0,
            0.0,
            0.5,
            1.0,
            1.0,
        );

        let wo = Vec3::Z;
        let wi = Vec3::new(0.0, 0.1, 0.994_987_4).normalize();
        let f_with = with_cc.eval(wo, wi);
        let f_no = no_cc.eval(wo, wi);
        assert!(f_with.x > f_no.x);
    }

    #[test]
    fn sample_weight_matches_eval_cos_over_pdf() {
        let bsdf = DisneyBrdfBsdf::new(
            Vec3::new(0.82, 0.67, 0.16),
            0.3,
            0.0,
            0.5,
            0.0,
            0.4,
            0.0,
            0.0,
            0.5,
            0.4,
            1.0,
        );
        let wo = Vec3::new(0.2, -0.1, 0.974_679_4).normalize();
        let mut rng = rand::rng();

        let mut got_any = false;
        for _ in 0..64 {
            if let Some(sample) = bsdf.sample(wo, &mut rng) {
                let f = bsdf.eval(wo, sample.wi);
                let expected = f * (sample.wi.z.max(0.0) / sample.pdf);
                let max_component = sample
                    .weight
                    .abs()
                    .max_element()
                    .max(expected.abs().max_element());
                let tol = (max_component * 1.0e-3).max(1.0e-4);
                assert!(sample.weight.abs_diff_eq(expected, tol));
                got_any = true;
            }
        }
        assert!(got_any);
    }
}
