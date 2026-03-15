use glam::{UVec2, Vec2, Vec3};

use crate::ray::Ray;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PinholeCamera {
    pub eye: Vec3,
    pub look_at: Vec3,
    pub up: Vec3,
    pub fov_y: f32,
    forward: Vec3,
    right: Vec3,
    image_up: Vec3,
    tan_half_fov_y: f32,
}

impl PinholeCamera {
    pub fn new(eye: Vec3, look_at: Vec3, up: Vec3, fov_y: f32) -> Self {
        let forward = (look_at - eye).normalize();
        let right = forward.cross(up).normalize();
        let image_up = right.cross(forward).normalize();
        let tan_half_fov_y = (0.5 * fov_y).tan();

        Self {
            eye,
            look_at,
            up,
            fov_y,
            forward,
            right,
            image_up,
            tan_half_fov_y,
        }
    }

    pub fn generate_ray(&self, resolution: UVec2, pixel: UVec2, us: Vec2) -> Ray {
        let resolution = resolution.as_vec2();
        let pixel = pixel.as_vec2() + 0.5 + 0.5 * sample_tent_2d(us);
        let aspect_ratio = resolution.x / resolution.y;

        let sensor_x = ((2.0 * pixel.x / resolution.x) - 1.0) * aspect_ratio * self.tan_half_fov_y;
        let sensor_y = (1.0 - (2.0 * pixel.y / resolution.y)) * self.tan_half_fov_y;

        let direction =
            (self.forward + sensor_x * self.right + sensor_y * self.image_up).normalize();

        Ray::new(self.eye, direction)
    }
}

fn sample_tent_2d(us: Vec2) -> Vec2 {
    Vec2::new(sample_tent_1d(us.x), sample_tent_1d(us.y))
}

fn sample_tent_1d(u: f32) -> f32 {
    let u = u.clamp(0.0, 1.0);
    if u < 0.5 {
        (2.0 * u).sqrt() - 1.0
    } else {
        1.0 - (2.0 - 2.0 * u).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use glam::{UVec2, Vec2, Vec3};

    use super::PinholeCamera;

    #[test]
    fn generate_ray_points_forward_through_center_pixel() {
        let camera = PinholeCamera::new(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::ZERO,
            Vec3::Y,
            60.0_f32.to_radians(),
        );

        let ray = camera.generate_ray(UVec2::new(3, 3), UVec2::new(1, 1), Vec2::splat(0.5));

        assert!(ray.origin.abs_diff_eq(Vec3::new(0.0, 0.0, 5.0), 1.0e-6));
        assert!(ray.direction.abs_diff_eq(Vec3::NEG_Z, 1.0e-6));
    }

    #[test]
    fn generate_ray_returns_normalized_direction() {
        let camera = PinholeCamera::new(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::ZERO,
            Vec3::Y,
            45.0_f32.to_radians(),
        );

        let ray = camera.generate_ray(UVec2::new(512, 512), UVec2::new(0, 0), Vec2::splat(0.5));

        assert!((ray.direction.length() - 1.0).abs() < 1.0e-6);
    }
}
