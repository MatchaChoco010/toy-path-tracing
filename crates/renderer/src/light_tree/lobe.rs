// Pure-function lobe helpers used by `Material::light_tree_importance`.
//
// This module owns the shared SG math: per-shading-point precompute structs,
// the SG×lobe convolution helpers, and the (Phi, mu, sigma, nu, lambda)
// → (W, xi, kappa) reduction that turns a tree node into the SG light
// `W g(o; xi, kappa)` viewed from `(p, n)`.
//
// Materials own their own `light_tree_precompute(vtx)` and
// `light_tree_importance(precompute, W, lobe)`; these helpers just provide
// the SG / NDF-filtering arithmetic so each material doesn't reimplement it.
//
// MULTI-LOBE NOTES (read this before adding multi-glossy materials)
// -----------------------------------------------------------------
// If a material has multiple diffuse lobes, sum their importances:
//   I_diffuse = sum_i rho_i * <SG-cosine product integral>
// (linearity of the integral over BRDFs).
//
// If a material has multiple glossy reflection lobes (e.g. Standard Surface
// has a coat + base specular), the paper recommends merging them into a
// single proxy lobe by *flux-weighted averaging the roughness matrix A* of
// each layer (Tokuyoshi et al. 2024 §6 "Implementation"):
//
//     A_merged = sum_i (rho_i / sum_j rho_j) * A_i,
//     alpha_merged = (sqrt(A_merged.diag.x), sqrt(A_merged.diag.y))
//
// Then call `glossy_importance` once with the merged roughness via
// `make_glossy_lobe`. Use `merge_glossy_roughness` for the merge step.
//
// If a material has multiple refraction lobes (rare), the same approach
// works -- merge the (alpha, eta, perfect-refraction-direction) triples by
// reflectance-weighted averaging. For coat-on-base layered glass you'd
// likely want to sum the per-lobe BTDF importance instead, since the
// orientations differ enough that merging would mis-direct the proxy.
//
// In all cases: diffuse and glossy importances are *added*, never merged.
// Their integrals are over the same domain and share the SG light, so their
// importances stack linearly.

use std::f32::consts::PI;

use glam::{Mat2, Vec2, Vec3};

use crate::{
    bsdf::sanitize_dielectric_eta,
    math::{
        OrthonormalBasis, refract,
        sg::{
            SG_SHARPNESS_MAX, SgLobe, sg_clamped_cosine_product_integral_over_pi, sg_integral,
            sg_product, vmf_hemispherical_integral,
        },
    },
};

use super::LightTreeNode;

/// Per-shading-point precompute that a material returns from
/// `light_tree_precompute`. Captures the shading geometry (`p`, `n`,
/// `frame`) and the *lobe-specific* parts the material wants importance for.
///
/// A material with multiple lobes (e.g. SimplePBR's diffuse + glossy + BTDF)
/// fills several `Option`s; importance evaluation simply adds the
/// contributions.
#[derive(Debug, Clone, Copy)]
pub struct LightTreePrecompute {
    pub p: Vec3,
    pub n: Vec3,
    pub frame: OrthonormalBasis,
    pub diffuse: Option<DiffuseLobePrecompute>,
    pub glossy: Option<GlossyLobePrecompute>,
    pub btdf: Option<BtdfLobePrecompute>,
}

#[derive(Debug, Clone, Copy)]
pub struct DiffuseLobePrecompute {
    /// Luminance of the diffuse albedo (rho_d).
    pub rho: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct GlossyLobePrecompute {
    pub rho: f32,
    /// Tangent-space view direction.
    pub wi_ts: Vec3,
    /// JJ^T at h = n (reflection Jacobian).
    pub jj: Mat2,
    /// 4 * det(JJ^T) -- see Sup. §5.2 for why we keep this form.
    pub det_jj4: f32,
    /// Beta = alpha^2 / (1 - alpha^2). Diagonal of 2 Sigma_D.
    pub beta: Vec2,
    pub alpha2_max: f32,
    /// Perfect specular reflection direction in world space.
    pub xi_p: Vec3,
    /// Lobe sharpness used for the filtered visibility kernel.
    pub kappa_p: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct BtdfLobePrecompute {
    pub rho: f32,
    pub wi_ts: Vec3,
    pub jj: Mat2,
    pub det_jj4: f32,
    pub beta: Vec2,
    pub alpha2_max: f32,
    pub xi_t: Vec3,
    pub kappa_t: f32,
}

/// Build a `GlossyLobePrecompute` for an axis-aligned anisotropic GGX glossy
/// reflection lobe. Returns `None` if the view direction is below the
/// surface (no lobe to integrate).
pub fn make_glossy_lobe(
    rho: f32,
    frame: OrthonormalBasis,
    wo_world: Vec3,
    alpha_x: f32,
    alpha_y: f32,
) -> Option<GlossyLobePrecompute> {
    if rho <= 0.0 {
        return None;
    }
    let wi_ts = frame.world_to_local(wo_world);
    if wi_ts.z <= 0.0 {
        return None;
    }
    let vlen = (wi_ts.x * wi_ts.x + wi_ts.y * wi_ts.y).sqrt();
    let v = if vlen > 0.0 {
        Vec2::new(wi_ts.x / vlen, wi_ts.y / vlen)
    } else {
        Vec2::new(1.0, 0.0)
    };
    // Rotation that sends the tangent x-axis onto the projection of wi_ts.
    let r = Mat2::from_cols(Vec2::new(v.x, v.y), Vec2::new(-v.y, v.x));
    let d = Mat2::from_cols(
        Vec2::new(0.5, 0.0),
        Vec2::new(0.0, 0.5 / wi_ts.z.max(1.0e-4)),
    );
    let jacobian = r * d;
    let jj = jacobian * jacobian.transpose();
    // det(J J^T) for reflection at h=n is 1 / (16 * cos^2 theta_i). The
    // /4 form mirrors the supplementary §5.2 derivation.
    let det_jj4 = 1.0 / (4.0 * wi_ts.z * wi_ts.z);

    // alpha^2 / (1 - alpha^2). Clamp alpha < 1 first so beta stays finite.
    let alpha2_x = (alpha_x * alpha_x).clamp(0.0, 1.0 - 1.0e-4);
    let alpha2_y = (alpha_y * alpha_y).clamp(0.0, 1.0 - 1.0e-4);
    let beta = Vec2::new(alpha2_x / (1.0 - alpha2_x), alpha2_y / (1.0 - alpha2_y));
    let alpha2_max = alpha2_x.max(alpha2_y).max(1.0e-6);

    let n = frame.normal();
    let xi_p = (-wo_world + 2.0 * wo_world.dot(n) * n).normalize_or_zero();
    let kappa_p = ((1.0 - alpha2_max) / (2.0 * alpha2_max)).clamp(0.0, SG_SHARPNESS_MAX);

    Some(GlossyLobePrecompute {
        rho,
        wi_ts,
        jj,
        det_jj4,
        beta,
        alpha2_max,
        xi_p,
        kappa_p,
    })
}

/// Build a `BtdfLobePrecompute` for a dielectric GGX transmission lobe
/// (Tokuyoshi 2024 "Proxy A": pivot the SG×lobe convolution around the
/// perfect refraction direction with the supplementary §1 refraction
/// Jacobian at h = n).
///
/// Returns `None` for total internal reflection (no transmission lobe) or
/// when the view is below the surface.
pub fn make_btdf_lobe(
    rho: f32,
    frame: OrthonormalBasis,
    wo_world: Vec3,
    alpha_x: f32,
    alpha_y: f32,
    eta_rel: f32,
) -> Option<BtdfLobePrecompute> {
    if rho <= 0.0 {
        return None;
    }
    let eta_rel = sanitize_dielectric_eta(eta_rel);
    let wi_ts = frame.world_to_local(wo_world);
    if wi_ts.z <= 0.0 {
        return None;
    }
    let xi_t_ts = refract(wi_ts, eta_rel)?;
    let xi_t = frame.local_to_world(xi_t_ts);

    // Refraction Jacobian at h=n from sup. §1 reduces to a scaled rotation.
    let cos_o = wi_ts.z.max(1.0e-4);
    let cos_t = xi_t_ts.z.abs().max(1.0e-4);
    let denom = (cos_o + eta_rel * cos_t).max(1.0e-4);
    let j = eta_rel / denom;
    let vlen = (wi_ts.x * wi_ts.x + wi_ts.y * wi_ts.y).sqrt();
    let v = if vlen > 0.0 {
        Vec2::new(wi_ts.x / vlen, wi_ts.y / vlen)
    } else {
        Vec2::new(1.0, 0.0)
    };
    let r = Mat2::from_cols(Vec2::new(v.x, v.y), Vec2::new(-v.y, v.x));
    let d = Mat2::from_cols(Vec2::new(j, 0.0), Vec2::new(0.0, j));
    let jacobian = r * d;
    let jj = jacobian * jacobian.transpose();
    // det(J J^T) for refraction at h=n is j^4. The /4 prefactor matches the
    // glossy form so we can share `glossy_importance`'s det formula.
    let det_jj4 = 4.0 * j * j * j * j;

    let alpha2_x = (alpha_x * alpha_x).clamp(0.0, 1.0 - 1.0e-4);
    let alpha2_y = (alpha_y * alpha_y).clamp(0.0, 1.0 - 1.0e-4);
    let beta = Vec2::new(alpha2_x / (1.0 - alpha2_x), alpha2_y / (1.0 - alpha2_y));
    let alpha2_max = alpha2_x.max(alpha2_y).max(1.0e-6);

    let kappa_t = ((1.0 - alpha2_max) / (2.0 * alpha2_max)).clamp(0.0, SG_SHARPNESS_MAX);

    Some(BtdfLobePrecompute {
        rho,
        wi_ts,
        jj,
        det_jj4,
        beta,
        alpha2_max,
        xi_t,
        kappa_t,
    })
}

/// Project a node's (mu, sigma_s^2, nu, lambda, Phi, radius) onto the shading
/// point as an SG light `W g(o; xi, kappa)`. Returns `None` for nodes with
/// zero flux.
pub fn sg_light_for_node(p: Vec3, n: Vec3, node: &LightTreeNode) -> Option<(f32, SgLobe)> {
    if node.flux <= 0.0 {
        return None;
    }
    let d = node.mu - p;
    let r2 = d.dot(d);
    if r2 <= 0.0 {
        return None;
    }
    let inv_r = 1.0 / r2.sqrt();
    // Hybrid spatial variance (Sup. Eq. 6):
    //   c = max(n . (x - mu) / ||x - mu||, 0).
    // c > 0 when the cluster centre is below the surface (the
    // "bounding-sphere" branch the paper recommends for invisible cluster
    // centres). With d = mu - p, c = max(-(n . d) / ||d||, 0).
    let cos_below = (-(n.dot(d)) * inv_r).max(0.0);
    let bound_var = 0.5 * node.radius * node.radius;
    let mut sigma2 = node.sigma_s2 * (1.0 - cos_below) + bound_var * cos_below;
    let min_sigma2 = r2 / SG_SHARPNESS_MAX;
    if sigma2 < min_sigma2 {
        sigma2 = min_sigma2;
    }
    let spatial_axis = d * inv_r;
    let spatial_sharpness = (r2 / sigma2).clamp(0.0, SG_SHARPNESS_MAX);

    // Directional vMF viewed from the shading point: emission axis nu is
    // stored in the +emission direction (e.g. nu_bar = 0.5 * n_geom for
    // triangle leaves), but the shading point's incoming-light direction o
    // is opposite to the emission direction. Substituting o_emit = -o into
    // g(o_emit; nu, lambda) gives g(o; -nu, lambda); see paper §3.2.
    let lobe = sg_product(-node.nu, node.lambda, spatial_axis, spatial_sharpness);

    let sg_norm = sg_integral(node.lambda).max(1.0e-30);
    let w_amp = node.flux / (2.0 * PI * sigma2 * sg_norm);
    let w = w_amp * lobe.log_amplitude.exp();
    if !w.is_finite() || w <= 0.0 {
        return None;
    }
    Some((w, lobe))
}

/// Diffuse SG×cos product integral [Tokuyoshi 2024 Sec. 4]. The integrand
/// is positive by construction.
pub fn diffuse_importance(diffuse: DiffuseLobePrecompute, n: Vec3, w: f32, lobe: &SgLobe) -> f32 {
    if diffuse.rho <= 0.0 || w <= 0.0 {
        return 0.0;
    }
    let cosine = lobe.axis.dot(n).clamp(-1.0, 1.0);
    let integral = sg_clamped_cosine_product_integral_over_pi(cosine, lobe.sharpness) * PI;
    diffuse.rho * w * integral
}

/// Glossy SG×lobe importance with NDF filtering [Tokuyoshi 2024 Sec. 5,
/// Eq. 12]. Combines:
///   * Filtered NDF roughness Ā via the SG's directional variance.
///   * Filtered visibility V via the lower-frequency reflection-lobe SG.
///   * sg_integral(kappa) to normalise the SG.
pub fn glossy_importance(
    glossy: GlossyLobePrecompute,
    frame: OrthonormalBasis,
    n: Vec3,
    w: f32,
    lobe: &SgLobe,
) -> f32 {
    if glossy.rho <= 0.0 || w <= 0.0 {
        return 0.0;
    }
    let inv_kappa = 1.0 / lobe.sharpness.max(1.0e-30);

    let two_sigma = Mat2::from_cols(Vec2::new(glossy.beta.x, 0.0), Vec2::new(0.0, glossy.beta.y));
    let filtered_proj = two_sigma + glossy.jj * inv_kappa;
    let det = glossy.beta.x * glossy.beta.y
        + 2.0
            * inv_kappa
            * (glossy.beta.x * glossy.jj.col(1).y + glossy.beta.y * glossy.jj.col(0).x)
        + inv_kappa * inv_kappa * glossy.det_jj4 / 4.0;
    let tr = filtered_proj.col(0).x + filtered_proj.col(1).y;
    let denom = (1.0 + tr + det).max(1.0e-12);
    let a_bar = Mat2::from_cols(
        Vec2::new(
            (filtered_proj.col(0).x + det) / denom,
            filtered_proj.col(0).y / denom,
        ),
        Vec2::new(
            filtered_proj.col(1).x / denom,
            (filtered_proj.col(1).y + det) / denom,
        ),
    );

    let xi_ts = frame.world_to_local(lobe.axis);
    let h_unnorm = glossy.wi_ts + xi_ts;
    if h_unnorm.length_squared() <= 1.0e-20 {
        return 0.0;
    }
    let h = h_unnorm.normalize();
    let lobe_value = sggx_reflection_pdf(glossy.wi_ts, h, a_bar);

    let prod = glossy.xi_p * glossy.kappa_p + lobe.axis * lobe.sharpness;
    let prod_sharp = prod.length();
    if prod_sharp <= 0.0 {
        return 0.0;
    }
    let prod_dir = prod / prod_sharp;
    let v_visibility = vmf_hemispherical_integral(prod_dir.dot(n).clamp(-1.0, 1.0), prod_sharp);

    let sg_full = sg_integral(lobe.sharpness);
    glossy.rho * w * v_visibility * lobe_value * sg_full
}

/// BTDF (transmission) importance for the dielectric GGX "Proxy A" lobe.
pub fn btdf_importance(
    btdf: BtdfLobePrecompute,
    frame: OrthonormalBasis,
    n: Vec3,
    w: f32,
    lobe: &SgLobe,
) -> f32 {
    if btdf.rho <= 0.0 || w <= 0.0 {
        return 0.0;
    }
    let inv_kappa = 1.0 / lobe.sharpness.max(1.0e-30);

    let two_sigma = Mat2::from_cols(Vec2::new(btdf.beta.x, 0.0), Vec2::new(0.0, btdf.beta.y));
    let filtered_proj = two_sigma + btdf.jj * inv_kappa;
    let det = btdf.beta.x * btdf.beta.y
        + 2.0 * inv_kappa * (btdf.beta.x * btdf.jj.col(1).y + btdf.beta.y * btdf.jj.col(0).x)
        + inv_kappa * inv_kappa * btdf.det_jj4 / 4.0;
    let tr = filtered_proj.col(0).x + filtered_proj.col(1).y;
    let denom = (1.0 + tr + det).max(1.0e-12);
    let a_bar = Mat2::from_cols(
        Vec2::new(
            (filtered_proj.col(0).x + det) / denom,
            filtered_proj.col(0).y / denom,
        ),
        Vec2::new(
            filtered_proj.col(1).x / denom,
            (filtered_proj.col(1).y + det) / denom,
        ),
    );

    let xi_ts = frame.world_to_local(lobe.axis);
    let h_unnorm = btdf.wi_ts + xi_ts;
    if h_unnorm.length_squared() <= 1.0e-20 {
        return 0.0;
    }
    let h = h_unnorm.normalize();
    let lobe_value = sggx_reflection_pdf(btdf.wi_ts, h, a_bar);

    let prod = btdf.xi_t * btdf.kappa_t + lobe.axis * lobe.sharpness;
    let prod_sharp = prod.length();
    if prod_sharp <= 0.0 {
        return 0.0;
    }
    let prod_dir = prod / prod_sharp;
    // Transmission asks for visibility *below* the surface; mirror the
    // upper-hemisphere integrator.
    let v_upper = vmf_hemispherical_integral(prod_dir.dot(n).clamp(-1.0, 1.0), prod_sharp);
    let v_visibility = (1.0 - v_upper).clamp(0.0, 1.0);

    let sg_full = sg_integral(lobe.sharpness);
    btdf.rho * w * v_visibility * lobe_value * sg_full
}

fn sggx_reflection_pdf(wi: Vec3, h: Vec3, roughness_mat: Mat2) -> f32 {
    let det = (roughness_mat.col(0).x * roughness_mat.col(1).y
        - roughness_mat.col(0).y * roughness_mat.col(1).x)
        .max(1.0e-30);
    let adj = Mat2::from_cols(
        Vec2::new(roughness_mat.col(1).y, -roughness_mat.col(0).y),
        Vec2::new(-roughness_mat.col(1).x, roughness_mat.col(0).x),
    );
    let m_xy = Vec2::new(h.x, h.y);
    let length2 = m_xy.dot(adj * m_xy) / det + h.z * h.z;
    let sggx = 1.0 / (PI * det.sqrt() * (length2 * length2));
    let inv_norm = (wi.x * wi.x * roughness_mat.col(0).x
        + 2.0 * wi.x * wi.y * roughness_mat.col(0).y
        + wi.y * wi.y * roughness_mat.col(1).y
        + wi.z * wi.z)
        .max(0.0)
        .sqrt();
    sggx / (4.0 * inv_norm.max(1.0e-30))
}

/// Merge two glossy roughness matrices (each axis-aligned anisotropic) by
/// reflectance-weighted averaging. Used by future multi-glossy-lobe
/// materials. Returns merged (alpha_x, alpha_y).
///
/// `rho_a`, `rho_b` are the reflectance weights. Both should be >= 0.
pub fn merge_glossy_roughness(
    rho_a: f32,
    alpha_a: (f32, f32),
    rho_b: f32,
    alpha_b: (f32, f32),
) -> (f32, f32) {
    let total = (rho_a + rho_b).max(1.0e-30);
    let wa = rho_a / total;
    let wb = rho_b / total;
    let a2_x = wa * alpha_a.0 * alpha_a.0 + wb * alpha_b.0 * alpha_b.0;
    let a2_y = wa * alpha_a.1 * alpha_a.1 + wb * alpha_b.1 * alpha_b.1;
    (a2_x.sqrt(), a2_y.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::sg::SgLobe;

    #[test]
    fn merge_glossy_roughness_with_zero_weight_keeps_other() {
        let m = merge_glossy_roughness(1.0, (0.3, 0.4), 0.0, (0.8, 0.9));
        assert!((m.0 - 0.3).abs() < 1e-5);
        assert!((m.1 - 0.4).abs() < 1e-5);
    }

    #[test]
    fn merge_glossy_roughness_picks_weighted_average() {
        let m = merge_glossy_roughness(1.0, (0.2, 0.2), 1.0, (0.8, 0.8));
        // average of squares = (0.04 + 0.64)/2 = 0.34, sqrt -> ~0.583
        assert!((m.0 - 0.5830952).abs() < 1e-4);
    }

    #[test]
    fn diffuse_importance_for_lambert_above_surface_is_positive() {
        let lobe = SgLobe::new(Vec3::Z, 10.0, 0.0);
        let i = diffuse_importance(DiffuseLobePrecompute { rho: 0.5 }, Vec3::Z, 1.0, &lobe);
        assert!(i > 0.0);
    }

    #[test]
    fn diffuse_importance_zero_when_albedo_zero() {
        let lobe = SgLobe::new(Vec3::Z, 10.0, 0.0);
        let i = diffuse_importance(DiffuseLobePrecompute { rho: 0.0 }, Vec3::Z, 1.0, &lobe);
        assert_eq!(i, 0.0);
    }
}
