use std::f32::consts::{PI, TAU};

use glam::{Vec2, Vec3};

pub mod orthonormal_basis;
pub mod sg;

pub use orthonormal_basis::OrthonormalBasis;

const MACHINE_EPSILON: f32 = f32::EPSILON * 0.5;

pub fn sample_tent_2d(us: Vec2) -> Vec2 {
    Vec2::new(sample_tent_1d(us.x), sample_tent_1d(us.y))
}

pub fn sample_tent_1d(u: f32) -> f32 {
    let u = u.clamp(0.0, 1.0);
    if u < 0.5 {
        (2.0 * u).sqrt() - 1.0
    } else {
        1.0 - (2.0 - 2.0 * u).sqrt()
    }
}

pub fn sample_uniform_disk_polar(us: Vec2) -> Vec2 {
    let r = us.x.clamp(0.0, 1.0).sqrt();
    let phi = TAU * us.y;
    Vec2::new(r * phi.cos(), r * phi.sin())
}

pub fn sample_cosine_weighted_hemisphere(us: Vec2) -> Vec3 {
    let disk = sample_uniform_disk_polar(us);
    let z = (1.0 - disk.length_squared()).max(0.0).sqrt();
    Vec3::new(disk.x, disk.y, z)
}

pub fn cosine_weighted_hemisphere_pdf(cos_theta: f32) -> f32 {
    cos_theta.max(0.0) / PI
}

pub fn face_forward(normal: Vec3, reference: Vec3) -> Vec3 {
    if normal.dot(reference) < 0.0 {
        -normal
    } else {
        normal
    }
}

pub fn fresnel_dielectric(mut cos_theta_i: f32, mut eta_i: f32, mut eta_t: f32) -> f32 {
    cos_theta_i = cos_theta_i.clamp(-1.0, 1.0);

    if cos_theta_i < 0.0 {
        std::mem::swap(&mut eta_i, &mut eta_t);
        cos_theta_i = -cos_theta_i;
    }

    let sin_theta_i = (1.0 - cos_theta_i * cos_theta_i).max(0.0).sqrt();
    let sin_theta_t = (eta_i / eta_t) * sin_theta_i;
    if sin_theta_t >= 1.0 {
        return 1.0;
    }

    let cos_theta_t = (1.0 - sin_theta_t * sin_theta_t).max(0.0).sqrt();
    let r_parallel =
        (eta_t * cos_theta_i - eta_i * cos_theta_t) / (eta_t * cos_theta_i + eta_i * cos_theta_t);
    let r_perpendicular =
        (eta_i * cos_theta_i - eta_t * cos_theta_t) / (eta_i * cos_theta_i + eta_t * cos_theta_t);

    0.5 * (r_parallel * r_parallel + r_perpendicular * r_perpendicular)
}

pub fn schlick_fresnel(f0: Vec3, cos_theta: f32) -> Vec3 {
    let cos_theta = cos_theta.clamp(0.0, 1.0);
    let one_minus_cos_theta = 1.0 - cos_theta;
    f0 + (Vec3::ONE - f0) * one_minus_cos_theta.powi(5)
}

pub fn reflect(wo: Vec3, normal: Vec3) -> Vec3 {
    (-wo + 2.0 * wo.dot(normal) * normal).normalize_or_zero()
}

pub fn refract(wo: Vec3, eta: f32) -> Option<Vec3> {
    if wo.z <= 0.0 {
        return None;
    }

    let sin2_theta_o = (1.0 - wo.z * wo.z).max(0.0);
    let sin2_theta_t = eta * eta * sin2_theta_o;
    if sin2_theta_t >= 1.0 {
        return None;
    }

    let cos_theta_t = (1.0 - sin2_theta_t).max(0.0).sqrt();
    let wi = Vec3::new(-eta * wo.x, -eta * wo.y, -cos_theta_t).normalize_or_zero();

    if wi.length_squared() == 0.0 {
        return None;
    }

    Some(wi)
}

pub fn interpolate_vec2(barycentric: Vec3, v0: Vec2, v1: Vec2, v2: Vec2) -> Vec2 {
    barycentric.x * v0 + barycentric.y * v1 + barycentric.z * v2
}

pub fn interpolate_vec3(barycentric: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> Vec3 {
    barycentric.x * v0 + barycentric.y * v1 + barycentric.z * v2
}

pub fn compute_surface_partials(positions: [Vec3; 3], uvs: [Vec2; 3]) -> Option<(Vec3, Vec3)> {
    let [p0, p1, p2] = positions;
    let [uv0, uv1, uv2] = uvs;
    let dp1 = p1 - p0;
    let dp2 = p2 - p0;
    let duv1 = uv1 - uv0;
    let duv2 = uv2 - uv0;
    let determinant = duv1.x * duv2.y - duv1.y * duv2.x;

    if determinant.abs() <= 1.0e-8 {
        return None;
    }

    let inv_determinant = 1.0 / determinant;
    let dpdu = (duv2.y * dp1 - duv1.y * dp2) * inv_determinant;
    let dpdv = (-duv2.x * dp1 + duv1.x * dp2) * inv_determinant;

    if dpdu.length_squared() == 0.0 || dpdv.length_squared() == 0.0 {
        return None;
    }

    Some((dpdu, dpdv))
}

pub fn difference_of_products(a: f32, b: f32, c: f32, d: f32) -> f32 {
    let cd = c * d;
    let difference_of_products = a.mul_add(b, -cd);
    let error = (-c).mul_add(d, cd);
    difference_of_products + error
}

pub fn gamma(n: i32) -> f32 {
    let n = n as f32;
    (n * MACHINE_EPSILON) / (1.0 - n * MACHINE_EPSILON)
}

pub fn max_component_index(v: Vec3) -> usize {
    if v.x > v.y {
        if v.x > v.z { 0 } else { 2 }
    } else if v.y > v.z {
        1
    } else {
        2
    }
}

pub fn permute_vec3(v: Vec3, x: usize, y: usize, z: usize) -> Vec3 {
    let a = v.to_array();
    Vec3::new(a[x], a[y], a[z])
}

pub fn reinhard(color: Vec3) -> Vec3 {
    color / (Vec3::ONE + color)
}

pub fn balance_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    let pdf_sum = pdf_a + pdf_b;

    if pdf_a <= 0.0 || pdf_sum <= 0.0 {
        return 0.0;
    }

    pdf_a / pdf_sum
}

pub fn russian_roulette_probability(throughput: Vec3) -> f32 {
    throughput.max_element().clamp(0.05, 1.0)
}

// Cubic Hermite interpolation with zero derivatives at both endpoints.
// S(edge0) = 0, S(edge1) = 1, S'(edge0) = S'(edge1) = 0, giving a C¹
// transition without the visible seams a linear ramp produces.
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use glam::{Vec2, Vec3};

    use super::{
        balance_heuristic, compute_surface_partials, cosine_weighted_hemisphere_pdf,
        difference_of_products, face_forward, fresnel_dielectric, gamma, interpolate_vec2,
        interpolate_vec3, max_component_index, permute_vec3, reflect, refract, reinhard,
        russian_roulette_probability, sample_cosine_weighted_hemisphere, sample_tent_1d,
        schlick_fresnel, smoothstep,
    };

    #[test]
    fn cosine_weighted_hemisphere_sample_stays_above_surface() {
        let wi = sample_cosine_weighted_hemisphere(Vec2::new(0.25, 0.75));

        assert!(wi.z >= 0.0);
        assert!((wi.length() - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn cosine_weighted_hemisphere_pdf_matches_expected_value() {
        let pdf = cosine_weighted_hemisphere_pdf(0.5);

        assert!((pdf - 0.5 / std::f32::consts::PI).abs() < 1.0e-6);
    }

    #[test]
    fn tent_sample_is_centered() {
        assert!(sample_tent_1d(0.5).abs() < 1.0e-6);
    }

    #[test]
    fn face_forward_aligns_to_reference() {
        assert_eq!(face_forward(Vec3::NEG_Z, Vec3::Z), Vec3::Z);
    }

    #[test]
    fn fresnel_dielectric_matches_normal_incidence_reflectance() {
        let reflectance = fresnel_dielectric(1.0, 1.0, 1.5);

        assert!((reflectance - 0.04).abs() < 1.0e-6);
    }

    #[test]
    fn fresnel_dielectric_returns_total_internal_reflection() {
        let cos_theta_i = (1.0_f32 - 0.8_f32 * 0.8_f32).sqrt();
        let reflectance = fresnel_dielectric(cos_theta_i, 1.5, 1.0);

        assert_eq!(reflectance, 1.0);
    }

    #[test]
    fn schlick_matches_f0_at_normal_incidence_and_one_at_grazing() {
        let f0 = Vec3::new(0.2, 0.5, 0.8);

        assert!(schlick_fresnel(f0, 1.0).abs_diff_eq(f0, 1.0e-6));
        assert!(schlick_fresnel(f0, 0.0).abs_diff_eq(Vec3::ONE, 1.0e-6));
    }

    #[test]
    fn reflect_mirrors_direction_around_surface_normal() {
        let wo = Vec3::new(0.3, -0.4, 0.8660254).normalize();
        let wi = reflect(wo, Vec3::Z);

        assert!(wi.abs_diff_eq(Vec3::new(-wo.x, -wo.y, wo.z).normalize(), 1.0e-6));
    }

    #[test]
    fn refract_returns_lower_hemisphere_direction() {
        let wo = Vec3::new(0.3, -0.4, 0.8660254).normalize();
        let wi = refract(wo, 1.0 / 1.5).expect("expected refraction");

        assert!(wi.z < 0.0);
        assert!((wi.x + wo.x / 1.5).abs() < 1.0e-6);
        assert!((wi.y + wo.y / 1.5).abs() < 1.0e-6);
    }

    #[test]
    fn refract_returns_none_for_total_internal_reflection() {
        let wo = Vec3::new(0.8, 0.0, 0.6).normalize();

        assert!(refract(wo, 1.5).is_none());
    }

    #[test]
    fn interpolation_helpers_match_barycentric_combination() {
        let barycentric = Vec3::new(0.5, 0.25, 0.25);

        assert!(
            interpolate_vec2(barycentric, Vec2::ZERO, Vec2::X, Vec2::Y)
                .abs_diff_eq(Vec2::splat(0.25), 1.0e-6)
        );
        assert!(
            interpolate_vec3(barycentric, Vec3::ZERO, Vec3::X, Vec3::Y)
                .abs_diff_eq(Vec3::new(0.25, 0.25, 0.0), 1.0e-6)
        );
    }

    #[test]
    fn surface_partials_follow_uv_axes() {
        let positions = [Vec3::ZERO, Vec3::X, Vec3::Y];
        let uvs = [Vec2::ZERO, Vec2::X, Vec2::Y];
        let (dpdu, dpdv) = compute_surface_partials(positions, uvs).expect("expected valid UVs");

        assert!(dpdu.abs_diff_eq(Vec3::X, 1.0e-6));
        assert!(dpdv.abs_diff_eq(Vec3::Y, 1.0e-6));
    }

    #[test]
    fn numeric_helpers_remain_stable() {
        assert!((difference_of_products(3.0, 5.0, 2.0, 7.0) - 1.0).abs() < 1.0e-6);
        assert!(gamma(3) > 0.0);
        assert_eq!(max_component_index(Vec3::new(1.0, 3.0, 2.0)), 1);
        assert_eq!(
            permute_vec3(Vec3::new(1.0, 2.0, 3.0), 2, 0, 1),
            Vec3::new(3.0, 1.0, 2.0)
        );
    }

    #[test]
    fn reinhard_maps_white_to_half_gray() {
        assert!(reinhard(Vec3::ONE).abs_diff_eq(Vec3::splat(0.5), 1.0e-6));
    }

    #[test]
    fn balance_heuristic_returns_normalized_weight() {
        assert!((balance_heuristic(2.0, 3.0) - 0.4).abs() < 1.0e-6);
        assert_eq!(balance_heuristic(0.0, 3.0), 0.0);
    }

    #[test]
    fn russian_roulette_probability_clamps_to_safe_range() {
        assert!((russian_roulette_probability(Vec3::splat(0.01)) - 0.05).abs() < 1.0e-6);
        assert!((russian_roulette_probability(Vec3::splat(0.5)) - 0.5).abs() < 1.0e-6);
        assert!((russian_roulette_probability(Vec3::splat(10.0)) - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn smoothstep_pins_endpoints_and_midpoint() {
        assert_eq!(smoothstep(0.0, 1.0, -0.5), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 0.0), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 1.0), 1.0);
        assert_eq!(smoothstep(0.0, 1.0, 1.5), 1.0);
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn smoothstep_is_monotonic_and_symmetric() {
        let n = 32;
        let mut prev = smoothstep(0.0, 1.0, 0.0);
        for i in 1..=n {
            let x = i as f32 / n as f32;
            let curr = smoothstep(0.0, 1.0, x);
            assert!(curr >= prev, "smoothstep must be monotonic non-decreasing");
            // Symmetric around 0.5: S(x) + S(1 - x) = 1.
            let mirror = smoothstep(0.0, 1.0, 1.0 - x);
            assert!((curr + mirror - 1.0).abs() < 1.0e-6);
            prev = curr;
        }
    }

    #[test]
    fn smoothstep_handles_offset_edges() {
        assert!((smoothstep(2.0, 4.0, 3.0) - 0.5).abs() < 1.0e-6);
        assert_eq!(smoothstep(2.0, 4.0, 1.5), 0.0);
        assert_eq!(smoothstep(2.0, 4.0, 5.0), 1.0);
    }
}
