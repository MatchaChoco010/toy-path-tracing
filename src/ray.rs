use glam::{Mat4, Vec3};

use crate::math::{difference_of_products, gamma, max_component_index, permute_vec3};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction }
    }

    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + t * self.direction
    }

    pub fn transformed(&self, transform: Mat4) -> Self {
        Self {
            origin: transform.transform_point3(self.origin),
            direction: transform.transform_vector3(self.direction),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TriangleIntersection {
    pub t: f32,
    pub barycentric: Vec3,
}

impl TriangleIntersection {
    pub fn interpolate<T>(&self, v0: T, v1: T, v2: T) -> T
    where
        T: Copy + core::ops::Mul<f32, Output = T> + core::ops::Add<Output = T>,
    {
        v0 * self.barycentric.x + v1 * self.barycentric.y + v2 * self.barycentric.z
    }
}

pub fn intersect_triangle(
    ray: &Ray,
    t_max: f32,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
) -> Option<TriangleIntersection> {
    if (v2 - v0).cross(v1 - v0).length_squared() == 0.0 {
        return None;
    }

    let mut p0t = v0 - ray.origin;
    let mut p1t = v1 - ray.origin;
    let mut p2t = v2 - ray.origin;

    let kz = max_component_index(ray.direction.abs());
    let mut kx = kz + 1;
    if kx == 3 {
        kx = 0;
    }
    let mut ky = kx + 1;
    if ky == 3 {
        ky = 0;
    }

    let d = permute_vec3(ray.direction, kx, ky, kz);
    if d.z == 0.0 {
        return None;
    }

    p0t = permute_vec3(p0t, kx, ky, kz);
    p1t = permute_vec3(p1t, kx, ky, kz);
    p2t = permute_vec3(p2t, kx, ky, kz);

    let sx = -d.x / d.z;
    let sy = -d.y / d.z;
    let sz = 1.0 / d.z;

    p0t.x += sx * p0t.z;
    p0t.y += sy * p0t.z;
    p1t.x += sx * p1t.z;
    p1t.y += sy * p1t.z;
    p2t.x += sx * p2t.z;
    p2t.y += sy * p2t.z;

    let mut e0 = difference_of_products(p1t.x, p2t.y, p1t.y, p2t.x);
    let mut e1 = difference_of_products(p2t.x, p0t.y, p2t.y, p0t.x);
    let mut e2 = difference_of_products(p0t.x, p1t.y, p0t.y, p1t.x);

    if e0 == 0.0 || e1 == 0.0 || e2 == 0.0 {
        e0 = (f64::from(p2t.y) * f64::from(p1t.x) - f64::from(p2t.x) * f64::from(p1t.y)) as f32;
        e1 = (f64::from(p0t.y) * f64::from(p2t.x) - f64::from(p0t.x) * f64::from(p2t.y)) as f32;
        e2 = (f64::from(p1t.y) * f64::from(p0t.x) - f64::from(p1t.x) * f64::from(p0t.y)) as f32;
    }

    if (e0 < 0.0 || e1 < 0.0 || e2 < 0.0) && (e0 > 0.0 || e1 > 0.0 || e2 > 0.0) {
        return None;
    }
    let det = e0 + e1 + e2;
    if det == 0.0 {
        return None;
    }

    p0t.z *= sz;
    p1t.z *= sz;
    p2t.z *= sz;
    let t_scaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;

    if det < 0.0 && (t_scaled >= 0.0 || t_scaled < t_max * det) {
        return None;
    } else if det > 0.0 && (t_scaled <= 0.0 || t_scaled > t_max * det) {
        return None;
    }

    let inv_det = 1.0 / det;
    let b0 = e0 * inv_det;
    let b1 = e1 * inv_det;
    let b2 = e2 * inv_det;
    let t = t_scaled * inv_det;
    debug_assert!(!t.is_nan());

    let max_zt = Vec3::new(p0t.z, p1t.z, p2t.z).abs().max_element();
    let delta_z = gamma(3) * max_zt;

    let max_xt = Vec3::new(p0t.x, p1t.x, p2t.x).abs().max_element();
    let max_yt = Vec3::new(p0t.y, p1t.y, p2t.y).abs().max_element();
    let delta_x = gamma(5) * (max_xt + max_zt);
    let delta_y = gamma(5) * (max_yt + max_zt);

    let delta_e = 2.0 * (gamma(2) * max_xt * max_yt + delta_y * max_xt + delta_x * max_yt);

    let max_e = Vec3::new(e0, e1, e2).abs().max_element();
    let delta_t =
        3.0 * (gamma(3) * max_e * max_zt + delta_e * max_zt + delta_z * max_e) * inv_det.abs();
    if t <= delta_t {
        return None;
    }

    Some(TriangleIntersection {
        t,
        barycentric: Vec3::new(b0, b1, b2),
    })
}

pub fn intersect_triangle_unbounded(
    ray: &Ray,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
) -> Option<TriangleIntersection> {
    intersect_triangle(ray, f32::INFINITY, v0, v1, v2)
}

#[cfg(test)]
mod tests {
    use super::{Ray, intersect_triangle, intersect_triangle_unbounded};
    use glam::Vec3;

    fn unit_triangle() -> (Vec3, Vec3, Vec3) {
        (Vec3::ZERO, Vec3::X, Vec3::Y)
    }

    #[test]
    fn intersect_triangle_returns_hit() {
        let (v0, v1, v2) = unit_triangle();
        let ray = Ray::new(Vec3::new(0.25, 0.25, 1.0), Vec3::NEG_Z);

        let hit = intersect_triangle_unbounded(&ray, v0, v1, v2).expect("expected hit");

        assert!((hit.t - 1.0).abs() < 1.0e-6);
        assert!((hit.barycentric.x - 0.5).abs() < 1.0e-6);
        assert!((hit.barycentric.y - 0.25).abs() < 1.0e-6);
        assert!((hit.barycentric.z - 0.25).abs() < 1.0e-6);
        assert!(
            hit.interpolate(v0, v1, v2)
                .abs_diff_eq(Vec3::new(0.25, 0.25, 0.0), 1.0e-6)
        );
    }

    #[test]
    fn intersect_triangle_returns_none_when_outside_triangle() {
        let (v0, v1, v2) = unit_triangle();
        let ray = Ray::new(Vec3::new(1.25, 1.25, 1.0), Vec3::NEG_Z);

        assert!(intersect_triangle_unbounded(&ray, v0, v1, v2).is_none());
    }

    #[test]
    fn intersect_triangle_returns_none_for_parallel_ray() {
        let (v0, v1, v2) = unit_triangle();
        let ray = Ray::new(Vec3::new(0.25, 0.25, 1.0), Vec3::X);

        assert!(intersect_triangle_unbounded(&ray, v0, v1, v2).is_none());
    }

    #[test]
    fn intersect_triangle_returns_none_for_degenerate_triangle() {
        let ray = Ray::new(Vec3::new(0.25, 0.0, 1.0), Vec3::NEG_Z);
        let v0 = Vec3::ZERO;
        let v1 = Vec3::X;
        let v2 = 2.0 * Vec3::X;

        assert!(intersect_triangle_unbounded(&ray, v0, v1, v2).is_none());
    }

    #[test]
    fn intersect_triangle_hits_on_edge() {
        let (v0, v1, v2) = unit_triangle();
        let ray = Ray::new(Vec3::new(0.5, 0.0, 1.0), Vec3::NEG_Z);

        let hit = intersect_triangle_unbounded(&ray, v0, v1, v2).expect("expected edge hit");

        assert!((hit.barycentric.x - 0.5).abs() < 1.0e-6);
        assert!((hit.barycentric.y - 0.5).abs() < 1.0e-6);
        assert!(hit.barycentric.z.abs() < 1.0e-6);
    }

    #[test]
    fn intersect_triangle_hits_on_vertex() {
        let (v0, v1, v2) = unit_triangle();
        let ray = Ray::new(Vec3::new(0.0, 0.0, 1.0), Vec3::NEG_Z);

        let hit = intersect_triangle_unbounded(&ray, v0, v1, v2).expect("expected vertex hit");

        assert!((hit.barycentric.x - 1.0).abs() < 1.0e-6);
        assert!(hit.barycentric.y.abs() < 1.0e-6);
        assert!(hit.barycentric.z.abs() < 1.0e-6);
    }

    #[test]
    fn intersect_triangle_respects_t_max() {
        let (v0, v1, v2) = unit_triangle();
        let ray = Ray::new(Vec3::new(0.25, 0.25, 2.0), Vec3::NEG_Z);

        assert!(intersect_triangle(&ray, 1.5, v0, v1, v2).is_none());
        assert!(intersect_triangle(&ray, 2.5, v0, v1, v2).is_some());
    }

    #[test]
    fn intersect_triangle_rejects_hit_at_ray_origin() {
        let (v0, v1, v2) = unit_triangle();
        let ray = Ray::new(Vec3::new(0.25, 0.25, 0.0), Vec3::NEG_Z);

        assert!(intersect_triangle_unbounded(&ray, v0, v1, v2).is_none());
    }

    #[test]
    fn intersect_triangle_accepts_backface_hit() {
        let (v0, v1, v2) = unit_triangle();
        let ray = Ray::new(Vec3::new(0.25, 0.25, -1.0), Vec3::Z);

        let hit = intersect_triangle_unbounded(&ray, v0, v1, v2).expect("expected hit");

        assert!((hit.t - 1.0).abs() < 1.0e-6);
    }
}
