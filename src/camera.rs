use glam::{UVec2, Vec2, Vec3};

use crate::{
    math::sample_tent_2d,
    ray::{Ray, RayCone, RayDifferential},
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PinholeCamera {
    pub eye: Vec3,
    pub look_at: Vec3,
    pub up: Vec3,
    pub fov_y: f32,
    pub exposure: f32,
    forward: Vec3,
    right: Vec3,
    image_up: Vec3,
    tan_half_fov_y: f32,
}

impl PinholeCamera {
    pub fn new(eye: Vec3, look_at: Vec3, up: Vec3, fov_y: f32, exposure: f32) -> Self {
        let forward = (look_at - eye).normalize();
        let right = forward.cross(up).normalize();
        let image_up = right.cross(forward).normalize();
        let tan_half_fov_y = (0.5 * fov_y).tan();

        Self {
            eye,
            look_at,
            up,
            fov_y,
            exposure,
            forward,
            right,
            image_up,
            tan_half_fov_y,
        }
    }

    pub fn generate_ray(&self, resolution: UVec2, pixel: UVec2, us: Vec2) -> Ray {
        let pixel = pixel.as_vec2() + 0.5 + 0.5 * sample_tent_2d(us);
        let direction = self.direction_for_sample(resolution, pixel);

        Ray::new(self.eye, direction)
    }

    pub fn generate_ray_differential(
        &self,
        resolution: UVec2,
        pixel: UVec2,
        us: Vec2,
        samples_per_pixel: u32,
    ) -> Ray {
        let pixel = pixel.as_vec2() + 0.5 + 0.5 * sample_tent_2d(us);
        let direction = self.direction_for_sample(resolution, pixel);
        let rx_direction = self.direction_for_sample(resolution, pixel + Vec2::X);
        let ry_direction = self.direction_for_sample(resolution, pixel + Vec2::Y);
        let scale = differential_scale(samples_per_pixel);
        let differential = RayDifferential {
            rx_origin: self.eye,
            ry_origin: self.eye,
            rx_direction: (direction + scale * (rx_direction - direction)).normalize_or_zero(),
            ry_direction: (direction + scale * (ry_direction - direction)).normalize_or_zero(),
        };

        Ray::new(self.eye, direction)
            .with_differential(differential)
            .with_cone(RayCone::new(
                0.0,
                self.pixel_spread_angle(resolution) * scale,
            ))
    }

    fn direction_for_sample(&self, resolution: UVec2, pixel: Vec2) -> Vec3 {
        let resolution = resolution.as_vec2();
        let aspect_ratio = resolution.x / resolution.y;
        let sensor_x = ((2.0 * pixel.x / resolution.x) - 1.0) * aspect_ratio * self.tan_half_fov_y;
        let sensor_y = (1.0 - (2.0 * pixel.y / resolution.y)) * self.tan_half_fov_y;

        (self.forward + sensor_x * self.right + sensor_y * self.image_up).normalize()
    }

    fn pixel_spread_angle(&self, resolution: UVec2) -> f32 {
        let resolution = resolution.as_vec2();
        let aspect_ratio = resolution.x / resolution.y;
        let dx = 2.0 * aspect_ratio * self.tan_half_fov_y / resolution.x;
        let dy = 2.0 * self.tan_half_fov_y / resolution.y;

        dx.max(dy).atan()
    }
}

fn differential_scale(samples_per_pixel: u32) -> f32 {
    (1.0 / (samples_per_pixel.max(1) as f32).sqrt()).max(0.125)
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
            1.0,
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
            1.0,
        );

        let ray = camera.generate_ray(UVec2::new(512, 512), UVec2::new(0, 0), Vec2::splat(0.5));

        assert!((ray.direction.length() - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn generate_ray_differential_offsets_neighbor_directions() {
        let camera = PinholeCamera::new(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::ZERO,
            Vec3::Y,
            45.0_f32.to_radians(),
            1.0,
        );

        let ray = camera.generate_ray_differential(
            UVec2::new(512, 512),
            UVec2::ZERO,
            Vec2::splat(0.5),
            1,
        );
        let differential = ray
            .differential
            .expect("camera ray should carry differentials");

        assert_eq!(differential.rx_origin, ray.origin);
        assert_eq!(differential.ry_origin, ray.origin);
        assert!(differential.rx_direction != ray.direction);
        assert!(differential.ry_direction != ray.direction);
        assert!(ray.cone.spread_angle > 0.0);
    }
}
