use std::f32::consts::{PI, TAU};

use glam::{Vec2, Vec3};

const MACHINE_EPSILON: f32 = f32::EPSILON * 0.5;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrthonormalBasis {
    tangent: Vec3,
    bitangent: Vec3,
    normal: Vec3,
}

impl OrthonormalBasis {
    pub fn from_normal(normal: Vec3) -> Self {
        let normal = if normal.length_squared() > 0.0 {
            normal.normalize()
        } else {
            Vec3::Z
        };
        // "Building an Orthonormal Basis, Revisited" (Duff et al.) avoids the
        // cancellation issues that show up with cross-product based frames.
        let sign = 1.0_f32.copysign(normal.z);
        let a = -1.0 / (sign + normal.z);
        let b = normal.x * normal.y * a;
        let tangent = Vec3::new(
            1.0 + sign * normal.x * normal.x * a,
            sign * b,
            -sign * normal.x,
        );
        let bitangent = Vec3::new(b, sign + normal.y * normal.y * a, -normal.y);

        Self {
            tangent,
            bitangent,
            normal,
        }
    }

    pub fn from_normal_and_tangent(normal: Vec3, tangent_hint: Vec3) -> Self {
        let normal = if normal.length_squared() > 0.0 {
            normal.normalize()
        } else {
            Vec3::Z
        };
        let tangent = (tangent_hint - tangent_hint.dot(normal) * normal).normalize_or_zero();

        if tangent.length_squared() == 0.0 {
            return Self::from_normal(normal);
        }

        let bitangent = normal.cross(tangent).normalize_or_zero();
        let tangent = bitangent.cross(normal).normalize_or_zero();

        Self {
            tangent,
            bitangent,
            normal,
        }
    }

    pub fn local_to_world(self, local_direction: Vec3) -> Vec3 {
        (local_direction.x * self.tangent
            + local_direction.y * self.bitangent
            + local_direction.z * self.normal)
            .normalize()
    }

    pub fn world_to_local(self, world_direction: Vec3) -> Vec3 {
        Vec3::new(
            world_direction.dot(self.tangent),
            world_direction.dot(self.bitangent),
            world_direction.dot(self.normal),
        )
    }

    pub fn tangent(self) -> Vec3 {
        self.tangent
    }

    pub fn bitangent(self) -> Vec3 {
        self.bitangent
    }

    pub fn normal(self) -> Vec3 {
        self.normal
    }
}

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
    throughput.max_element().clamp(0.05, 0.95)
}

#[cfg(test)]
mod tests {
    use glam::{Vec2, Vec3};

    use super::{
        OrthonormalBasis, balance_heuristic, compute_surface_partials,
        cosine_weighted_hemisphere_pdf, difference_of_products, face_forward, gamma,
        interpolate_vec2, interpolate_vec3, max_component_index, permute_vec3, reinhard,
        russian_roulette_probability, sample_cosine_weighted_hemisphere, sample_tent_1d,
    };

    #[test]
    fn basis_maps_normal_to_local_z() {
        let normal = Vec3::new(0.3, -0.4, 0.8660254).normalize();
        let basis = OrthonormalBasis::from_normal(normal);

        let local = basis.world_to_local(normal);

        assert!(local.abs_diff_eq(Vec3::Z, 1.0e-5));
    }

    #[test]
    fn basis_round_trips_direction() {
        let basis = OrthonormalBasis::from_normal(Vec3::new(-0.2, 0.9, 0.38).normalize());
        let local = Vec3::new(0.4, -0.3, 0.8660254).normalize();

        let world = basis.local_to_world(local);
        let reconstructed = basis.world_to_local(world);

        assert!(reconstructed.abs_diff_eq(local, 1.0e-5));
    }

    #[test]
    fn basis_respects_tangent_hint() {
        let basis = OrthonormalBasis::from_normal_and_tangent(Vec3::Z, Vec3::X);

        assert!(basis.tangent().abs_diff_eq(Vec3::X, 1.0e-6));
        assert!(basis.normal().abs_diff_eq(Vec3::Z, 1.0e-6));
    }

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
        assert!((russian_roulette_probability(Vec3::splat(10.0)) - 0.95).abs() < 1.0e-6);
    }
}
