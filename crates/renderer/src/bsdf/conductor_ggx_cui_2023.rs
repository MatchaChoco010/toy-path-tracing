use std::f32::consts::{PI, TAU};

use glam::{Vec2, Vec3};

use crate::{math::schlick_fresnel, sampler::AuxRng};

use super::smith_ggx::{EFFECTIVELY_SMOOTH_ALPHA, MIN_ALPHA, ggx_d, is_upper_hemisphere};
use super::{BsdfFlags, BsdfSample};

const MAX_BOUNCES: usize = 10;
const RR_DEPTH: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConductorGgxCui2023Bsdf {
    base_color: Vec3,
    alpha_x: f32,
    alpha_y: f32,
}

impl ConductorGgxCui2023Bsdf {
    pub fn new(base_color: Vec3, alpha_x: f32, alpha_y: f32) -> Self {
        Self {
            base_color: base_color.clamp(Vec3::ZERO, Vec3::ONE),
            alpha_x: alpha_x.max(MIN_ALPHA),
            alpha_y: alpha_y.max(MIN_ALPHA),
        }
    }

    fn effectively_smooth(&self) -> bool {
        self.alpha_x.max(self.alpha_y) < EFFECTIVELY_SMOOTH_ALPHA
    }

    pub fn eval(&self, wo: Vec3, wi: Vec3, aux_rng: &mut AuxRng) -> Vec3 {
        if !is_upper_hemisphere(wo) || !is_upper_hemisphere(wi) || self.effectively_smooth() {
            return Vec3::ZERO;
        }

        let cos_o = wo.z;
        if cos_o <= 0.0 {
            return Vec3::ZERO;
        }

        let lambda_wo = signed_lambda(wo, self.alpha_x, self.alpha_y);
        let mut s = SegmentTerm::new(lambda_wo);
        let mut inv_pdf = signed_lambda(-wi, self.alpha_x, self.alpha_y);
        s.add_bounce(inv_pdf);

        let mut result = self.vertex(wi, wo) * s.get_sk();

        let mut weight = Vec3::ONE;
        let mut current_view = wi;

        for i in 1..MAX_BOUNCES {
            let us = Vec2::new(aux_rng.next_f32(), aux_rng.next_f32());
            let wm = sample_vndf_unrestricted(current_view, self.alpha_x, self.alpha_y, us);
            if wm.length_squared() == 0.0 {
                break;
            }

            let cos_im = current_view.dot(wm);
            let f = schlick_fresnel(self.base_color, cos_im.abs());
            weight *= f;

            let reflected = (2.0 * cos_im * wm - current_view).normalize_or_zero();
            if reflected.length_squared() == 0.0 {
                break;
            }

            let lambda_ref = signed_lambda(reflected, self.alpha_x, self.alpha_y);
            s.add_bounce(lambda_ref);

            current_view = -reflected;

            let sk = s.get_sk();

            if i >= RR_DEPTH {
                let q = sk.clamp(0.3, 0.95);
                let r = aux_rng.next_f32();
                if r >= q {
                    break;
                }
                weight /= q;
            }

            let v = self.vertex(current_view, wo);
            result += v * weight * inv_pdf.abs() * sk;

            inv_pdf *= lambda_ref;
        }

        result / cos_o.max(1.0e-6)
    }

    pub fn pdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        if !is_upper_hemisphere(wo) || !is_upper_hemisphere(wi) || self.effectively_smooth() {
            return 0.0;
        }

        let pdf_bounce = self.eval_bounce_pdf(wo, wi);

        let alpha = ((self.alpha_x + self.alpha_y) * 0.5).min(0.975);
        let denom = (wo.z + wi.z).max(1.0e-6);
        let pdf_multi =
            alpha * (h_approx(wo.z, alpha) * h_approx(wi.z, alpha) - 1.0) / (4.0 * PI * denom);

        (pdf_bounce + pdf_multi).max(0.0)
    }

    fn eval_bounce_pdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        let h = (wo + wi).normalize_or_zero();
        if h.length_squared() == 0.0 || h.z <= 0.0 {
            return 0.0;
        }
        let d = ggx_d(h, self.alpha_x, self.alpha_y);
        if d <= 0.0 {
            return 0.0;
        }
        let lambda_wo = signed_lambda(wo, self.alpha_x, self.alpha_y);
        let g1 = 1.0 / (1.0 + lambda_wo);
        if !g1.is_finite() || g1 <= 0.0 {
            return 0.0;
        }
        d * g1 / (4.0 * wo.z.max(1.0e-6))
    }

    pub fn sample(&self, wo: Vec3, aux_rng: &mut AuxRng) -> Option<BsdfSample> {
        if !is_upper_hemisphere(wo) {
            return None;
        }
        if self.effectively_smooth() {
            return self.sample_smooth(wo);
        }

        let mut current_view = wo;
        let mut weight = Vec3::ONE;
        let mut pdf_sample = 1.0;
        let mut up_point: Option<usize> = None;
        let mut bounce = 0usize;

        let mut dir_list: Vec<RayInfo> = Vec::with_capacity(MAX_BOUNCES + 1);
        dir_list.push(RayInfo::new(-wo, self.alpha_x, self.alpha_y));

        let final_dir;

        loop {
            let us = Vec2::new(aux_rng.next_f32(), aux_rng.next_f32());
            let wm = sample_vndf_unrestricted(current_view, self.alpha_x, self.alpha_y, us);
            if wm.length_squared() == 0.0 {
                return None;
            }
            let cos_im = current_view.dot(wm);
            let reflected = (2.0 * cos_im * wm - current_view).normalize_or_zero();
            if reflected.length_squared() == 0.0 {
                return None;
            }

            let lambda_unsigned = dir_list[bounce].lambda_unsigned;
            if lambda_unsigned == 0.0 {
                return None;
            }
            pdf_sample /= lambda_unsigned;
            if pdf_sample < 1.0e-10 {
                return None;
            }

            bounce += 1;

            if up_point.is_none() && reflected.z > 0.0 {
                up_point = Some(bounce);
            }

            dir_list.push(RayInfo::new(reflected, self.alpha_x, self.alpha_y));

            let f = schlick_fresnel(self.base_color, cos_im.abs());
            weight *= f;

            if reflected.z <= 0.0 {
                if bounce == MAX_BOUNCES {
                    return None;
                }
                current_view = -reflected;
            } else if bounce == MAX_BOUNCES {
                final_dir = reflected;
                break;
            } else {
                let lambda_signed = dir_list[bounce].lambda_signed;
                let g1 = 1.0 / (1.0 + lambda_signed);
                let r = aux_rng.next_f32();
                if r < g1 {
                    pdf_sample *= g1;
                    final_dir = reflected;
                    break;
                }
                pdf_sample *= 1.0 - g1;
                current_view = -reflected;
            }
        }

        let up = up_point.unwrap_or(bounce);
        let path_g = compute_path_g(&dir_list, 0, bounce, up);
        weight *= path_g;

        if pdf_sample < 1.0e-7 {
            return None;
        }
        weight /= pdf_sample;

        let mis_pdf = self.pdf(wo, final_dir);
        if !mis_pdf.is_finite() || mis_pdf <= 0.0 {
            return None;
        }
        if !weight.is_finite() {
            return None;
        }

        Some(BsdfSample {
            weight,
            wi: final_dir,
            pdf: mis_pdf,
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
        let weight = schlick_fresnel(self.base_color, wi.z.abs());
        Some(BsdfSample {
            weight,
            wi,
            pdf: 1.0,
            flags: BsdfFlags::DELTA | BsdfFlags::REFLECTION,
            eta: 1.0,
            wavelength_lock: None,
        })
    }

    fn vertex(&self, first: Vec3, second: Vec3) -> Vec3 {
        let h = (first + second).normalize_or_zero();
        if h.length_squared() == 0.0 || h.z <= 0.0 {
            return Vec3::ZERO;
        }
        let d = ggx_d(h, self.alpha_x, self.alpha_y);
        if d <= 0.0 {
            return Vec3::ZERO;
        }
        let cos = first.dot(h);
        let f = schlick_fresnel(self.base_color, cos.abs());
        f * (d / (4.0 * first.z.abs().max(1.0e-6)))
    }
}

fn signed_lambda(w: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
    RayInfo::new(w, alpha_x, alpha_y).lambda_signed
}

#[derive(Debug, Clone, Copy)]
struct RayInfo {
    lambda_unsigned: f32,
    lambda_signed: f32,
}

impl RayInfo {
    fn new(w: Vec3, alpha_x: f32, alpha_y: f32) -> Self {
        let cos_theta = w.z;
        let unsigned = if cos_theta.abs() >= 0.9999 {
            if cos_theta > 0.0 { 0.0 } else { 1.0 }
        } else {
            let sin2 = (1.0 - cos_theta * cos_theta).max(0.0);
            if sin2 <= 0.0 {
                0.0
            } else {
                let sin_theta = sin2.sqrt();
                let tan_theta = sin_theta / cos_theta;
                let inv_sin2 = 1.0 / sin2;
                let cos_phi2 = w.x * w.x * inv_sin2;
                let sin_phi2 = w.y * w.y * inv_sin2;
                let alpha = (cos_phi2 * alpha_x * alpha_x + sin_phi2 * alpha_y * alpha_y).sqrt();
                let a = 1.0 / (tan_theta * alpha);
                let sign_part = if a < 0.0 { 1.0 } else { -1.0 };
                0.5 * (sign_part + (1.0 + 1.0 / (a * a)).sqrt())
            }
        };
        let signed = unsigned.copysign(cos_theta);
        Self {
            lambda_unsigned: unsigned,
            lambda_signed: signed,
        }
    }
}

struct SegmentTerm {
    lambdao: f32,
    n: usize,
    m: f32,
    e: Vec<f32>,
    g: Vec<f32>,
    l: Vec<f32>,
}

impl SegmentTerm {
    fn new(lambda_o: f32) -> Self {
        Self {
            lambdao: lambda_o,
            n: 0,
            m: 1.0,
            e: Vec::with_capacity(MAX_BOUNCES + 1),
            g: Vec::with_capacity(MAX_BOUNCES + 1),
            l: Vec::with_capacity(MAX_BOUNCES + 1),
        }
    }

    fn add_bounce(&mut self, lambda: f32) {
        if lambda < 0.0 {
            let l_value = -lambda;
            self.l.push(l_value);
            let denom = self.lambdao + l_value;
            let e = if denom != 0.0 { 1.0 / denom } else { 0.0 };
            self.e.push(e);
            self.g.push(0.0);
            self.m *= e;
            self.n += 1;
        } else {
            if self.n == 0 {
                return;
            }
            let last_l = self.l[self.n - 1];
            let denom_last = lambda + last_l;
            if self.m == 0.0 {
                let updated = if denom_last != 0.0 {
                    self.g[self.n - 1] / denom_last
                } else {
                    0.0
                };
                self.g[self.n - 1] = updated;
            } else {
                self.g[self.n - 1] = if denom_last != 0.0 {
                    1.0 / denom_last
                } else {
                    0.0
                };
                self.m = 0.0;
            }
            if self.n >= 2 {
                for i in (0..self.n - 1).rev() {
                    let denom = lambda + self.l[i];
                    self.g[i] = if denom != 0.0 {
                        (self.g[i] + self.g[i + 1]) / denom
                    } else {
                        0.0
                    };
                }
            }
        }
    }

    fn get_sk(&self) -> f32 {
        if self.m != 0.0 {
            return self.m;
        }
        let mut s = 0.0;
        for i in (0..self.n).rev() {
            s = self.e[i] * (s + self.g[i]);
        }
        s
    }
}

fn compute_path_g(dir_list: &[RayInfo], begin: usize, end: usize, up_point: usize) -> f32 {
    let width = up_point - begin;
    let height = end - up_point + 1;
    if width == 0 || height == 0 {
        return 0.0;
    }

    let mut g_matrix = vec![0.0f32; width * height];
    let idx = |x: usize, y: usize| x * height + y;

    let mut p1 = (width - 1, height - 1);
    let denom = dir_list[begin + p1.0].lambda_unsigned + dir_list[end - p1.1].lambda_unsigned;
    g_matrix[idx(p1.0, p1.1)] = if denom != 0.0 { 1.0 / denom } else { 0.0 };

    while !(p1.0 == 0 && p1.1 == 0) {
        if p1.0 > 0 {
            p1.0 -= 1;
        } else {
            p1.1 -= 1;
        }
        let mut p2 = p1;
        loop {
            let right = if p2.0 + 1 < width {
                g_matrix[idx(p2.0 + 1, p2.1)]
            } else {
                0.0
            };
            let down = if p2.1 + 1 < height {
                g_matrix[idx(p2.0, p2.1 + 1)]
            } else {
                0.0
            };
            let denom =
                dir_list[begin + p2.0].lambda_unsigned + dir_list[end - p2.1].lambda_unsigned;
            g_matrix[idx(p2.0, p2.1)] = if denom != 0.0 {
                (right + down) / denom
            } else {
                0.0
            };

            if p2.0 + 1 >= width || p2.1 == 0 {
                break;
            }
            p2.0 += 1;
            p2.1 -= 1;
        }
    }

    g_matrix[0]
}

fn h_approx(u: f32, a: f32) -> f32 {
    let denom = 1.0 + 2.0 * (1.0 - a).max(0.0).sqrt() * u;
    if denom <= 0.0 {
        0.0
    } else {
        (1.0 + 2.0 * u) / denom
    }
}

fn sample_vndf_unrestricted(wo: Vec3, alpha_x: f32, alpha_y: f32, us: Vec2) -> Vec3 {
    let alpha_x = alpha_x.max(MIN_ALPHA);
    let alpha_y = alpha_y.max(MIN_ALPHA);

    let vh = Vec3::new(alpha_x * wo.x, alpha_y * wo.y, wo.z).normalize_or_zero();
    if vh.length_squared() == 0.0 {
        return Vec3::ZERO;
    }

    let lensq = vh.x * vh.x + vh.y * vh.y;
    let t1 = if lensq > 0.0 {
        Vec3::new(-vh.y, vh.x, 0.0) / lensq.sqrt()
    } else {
        Vec3::X
    };
    let t2 = vh.cross(t1);

    let r = us.x.clamp(0.0, 1.0).sqrt();
    let phi = TAU * us.y;
    let p1 = r * phi.cos();
    let mut p2 = r * phi.sin();
    let s = 0.5 * (1.0 + vh.z);
    p2 = (1.0 - s) * (1.0 - p1 * p1).max(0.0).sqrt() + s * p2;

    let nh = p1 * t1 + p2 * t2 + (1.0 - p1 * p1 - p2 * p2).max(0.0).sqrt() * vh;

    Vec3::new(alpha_x * nh.x, alpha_y * nh.y, nh.z.max(0.0)).normalize_or_zero()
}

#[cfg(test)]
mod tests {
    use std::f32::consts::TAU;

    use glam::Vec3;

    use super::{ConductorGgxCui2023Bsdf, h_approx};
    use crate::{
        bsdf::{BsdfFlags, ConductorGgxBsdf},
        sampler::AuxRng,
    };

    const HEMISPHERE_Z_SAMPLES: usize = 96;
    const HEMISPHERE_PHI_SAMPLES: usize = 96;

    fn integrate_hemisphere<F>(mut f: F) -> f32
    where
        F: FnMut(Vec3) -> f32,
    {
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
    fn smooth_alpha_returns_perfect_mirror_sample() {
        let bsdf = ConductorGgxCui2023Bsdf::new(Vec3::new(0.7, 0.5, 0.3), 1.0e-4, 1.0e-4);
        let wo = Vec3::new(0.3, -0.4, 0.866_025_4).normalize();
        let mut aux_rng = AuxRng::from_seed(0);

        let sample = bsdf
            .sample(wo, &mut aux_rng)
            .expect("expected smooth reflection");
        let expected = Vec3::new(-wo.x, -wo.y, wo.z).normalize();

        assert!(sample.wi.abs_diff_eq(expected, 1.0e-6));
        assert_eq!(sample.flags, BsdfFlags::DELTA | BsdfFlags::REFLECTION);
    }

    #[test]
    fn eval_returns_zero_for_lower_hemisphere_inputs() {
        let bsdf = ConductorGgxCui2023Bsdf::new(Vec3::ONE, 0.3, 0.3);
        let mut aux_rng = AuxRng::from_seed(0);

        assert_eq!(bsdf.eval(Vec3::Z, Vec3::NEG_Z, &mut aux_rng), Vec3::ZERO);
        assert_eq!(bsdf.eval(Vec3::NEG_Z, Vec3::Z, &mut aux_rng), Vec3::ZERO);
    }

    #[test]
    fn pdf_is_finite_and_within_reasonable_range() {
        let configs = [
            (Vec3::Z, 0.3, 0.3),
            (Vec3::new(0.3, -0.4, 0.866_025_4).normalize(), 0.5, 0.5),
            (Vec3::new(0.6, 0.0, 0.8).normalize(), 0.7, 0.7),
        ];

        for (wo, alpha_x, alpha_y) in configs {
            let bsdf = ConductorGgxCui2023Bsdf::new(Vec3::ONE, alpha_x, alpha_y);
            let integral = integrate_hemisphere(|wi| bsdf.pdf(wo, wi));

            assert!(integral.is_finite());
            assert!(
                integral > 0.5 && integral < 2.0,
                "wo={wo:?}, alpha=({alpha_x}, {alpha_y}), integral={integral}"
            );
        }
    }

    #[test]
    fn low_alpha_eval_matches_single_scattering_conductor() {
        let alpha = 5.0e-3;
        let multi = ConductorGgxCui2023Bsdf::new(Vec3::new(0.95, 0.78, 0.4), alpha, alpha);
        let single = ConductorGgxBsdf::new(Vec3::new(0.95, 0.78, 0.4), alpha, alpha);
        let wo = Vec3::new(0.2, -0.1, 0.974_679_4).normalize();
        let wi = Vec3::new(-wo.x, -wo.y, wo.z).normalize();

        let mut accum = Vec3::ZERO;
        let samples = 64;
        for i in 0..samples {
            let mut aux_rng = AuxRng::from_seed(i);
            accum += multi.eval(wo, wi, &mut aux_rng);
        }
        let multi_mean = accum / samples as f32;
        let single_value = single.eval(wo, wi);

        assert!(multi_mean.is_finite());
        assert!(single_value.is_finite());
        assert!(multi_mean.x >= single_value.x * 0.9);
        assert!(multi_mean.y >= single_value.y * 0.9);
        assert!(multi_mean.z >= single_value.z * 0.9);
    }

    #[test]
    fn h_approx_matches_normal_incidence() {
        let h = h_approx(1.0, 0.5);
        let expected = (1.0 + 2.0_f32) / (1.0 + 2.0 * (1.0_f32 - 0.5).sqrt());
        assert!((h - expected).abs() < 1.0e-6);
    }

    #[test]
    fn sample_returns_upper_hemisphere_glossy_reflection_for_typical_inputs() {
        let bsdf = ConductorGgxCui2023Bsdf::new(Vec3::new(0.9, 0.7, 0.4), 0.4, 0.4);
        let wo = Vec3::new(0.3, -0.4, 0.866_025_4).normalize();
        let mut hits = 0usize;
        for i in 0..32 {
            let mut aux_rng = AuxRng::from_seed(i);
            if let Some(sample) = bsdf.sample(wo, &mut aux_rng) {
                assert!(sample.wi.is_finite());
                assert!(sample.weight.is_finite());
                assert!(sample.pdf.is_finite());
                assert!(sample.wi.z > 0.0);
                assert!(sample.pdf > 0.0);
                assert!(sample.flags.contains(BsdfFlags::REFLECTION));
                hits += 1;
            }
        }
        assert!(hits >= 24, "too many failed samples: {hits}/32");
    }

    #[test]
    fn reciprocity_holds_in_expectation() {
        let bsdf = ConductorGgxCui2023Bsdf::new(Vec3::ONE, 0.45, 0.3);
        let wo = Vec3::new(0.2, 0.1, 0.974_679_4).normalize();
        let wi = Vec3::new(-0.3, 0.05, 0.952_628_5).normalize();
        let samples = 256;
        let mut sum_io = Vec3::ZERO;
        let mut sum_oi = Vec3::ZERO;
        for i in 0..samples {
            let mut aux_rng_io = AuxRng::from_seed(i);
            let mut aux_rng_oi = AuxRng::from_seed(i + samples);
            sum_io += bsdf.eval(wo, wi, &mut aux_rng_io);
            sum_oi += bsdf.eval(wi, wo, &mut aux_rng_oi);
        }
        let mean_io = sum_io / samples as f32;
        let mean_oi = sum_oi / samples as f32;

        let diff = (mean_io - mean_oi).length();
        let scale = mean_io.length().max(mean_oi.length()).max(1.0e-3);
        assert!(diff / scale < 0.25, "io={mean_io:?}, oi={mean_oi:?}");
    }
}
