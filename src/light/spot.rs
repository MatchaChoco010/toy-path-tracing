use std::f32::consts::{PI, TAU};

use glam::Vec3;

use super::{LightLiSample, LightSampleContext, LightType};
use crate::math::smoothstep;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpotLightIndex(pub usize);

// Spotlight modelled as an isotropic point light with a directional mask.
// `intensity` is the total emitted power of the underlying point light in
// watts (W); the cone angles simply mask which directions are illuminated.
// Changing the cone narrows/widens the lit region without altering the
// brightness at points that are already inside the cone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpotLight {
    pub position: Vec3,
    /// Cone axis direction (light → scene), unit vector.
    pub direction: Vec3,
    /// Normalized RGB tint.
    pub color: Vec3,
    /// Total emitted power of the underlying isotropic point light, in watts (W).
    pub intensity: f32,
    pub cos_total_width: f32,
    pub cos_falloff_start: f32,
}

impl SpotLight {
    pub fn new(
        position: Vec3,
        direction: Vec3,
        color: Vec3,
        intensity: f32,
        total_width_rad: f32,
        falloff_start_rad: f32,
    ) -> Self {
        let total_width = total_width_rad.clamp(0.0, PI);
        let falloff_start = falloff_start_rad.clamp(0.0, total_width);
        Self {
            position,
            direction: direction.normalize(),
            color,
            intensity,
            cos_total_width: total_width.cos(),
            cos_falloff_start: falloff_start.cos(),
        }
    }

    pub fn falloff(&self, direction_from_light: Vec3) -> f32 {
        let cos_theta = direction_from_light.dot(self.direction);
        smoothstep(self.cos_total_width, self.cos_falloff_start, cos_theta)
    }
}

pub(super) fn sample_li(light: &SpotLight, ctx: &LightSampleContext) -> Option<LightLiSample> {
    let to_light = light.position - ctx.p;
    let distance_squared = to_light.length_squared();
    if distance_squared <= 0.0 {
        return None;
    }
    let distance = distance_squared.sqrt();
    let wi = to_light / distance;
    let falloff = light.falloff(-wi);
    if falloff <= 0.0 {
        return None;
    }
    // Isotropic point light: I = P / (4π); Li at surface = I / r² * mask.
    let radiance = light.color * (light.intensity * falloff / (2.0 * TAU * distance_squared));
    Some(LightLiSample {
        radiance,
        wi,
        pdf: 1.0,
        distance,
        light_type: LightType::DeltaPosition,
        target_triangle: None,
    })
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use glam::{Vec2, Vec3};

    use super::super::{LightKind, LightSampleContext, LightType, sample_light_li};
    use super::SpotLight;
    use crate::scene::Scene;

    #[test]
    fn spot_light_falloff_is_full_inside_cone() {
        let spot = SpotLight::new(
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::NEG_Z,
            Vec3::ONE,
            1.0,
            (30.0_f32).to_radians(),
            (20.0_f32).to_radians(),
        );
        assert!((spot.falloff(Vec3::NEG_Z) - 1.0).abs() < 1.0e-5);
        let dir = Vec3::new(0.1, 0.0, -1.0).normalize();
        assert!((spot.falloff(dir) - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn spot_light_falloff_is_zero_outside_cone() {
        let spot = SpotLight::new(
            Vec3::ZERO,
            Vec3::NEG_Z,
            Vec3::ONE,
            1.0,
            (30.0_f32).to_radians(),
            (20.0_f32).to_radians(),
        );
        let outside = Vec3::new(1.0, 0.0, -1.0).normalize();
        assert_eq!(spot.falloff(outside), 0.0);
    }

    #[test]
    fn spot_light_falloff_is_smooth_between_cones() {
        let spot = SpotLight::new(
            Vec3::ZERO,
            Vec3::NEG_Z,
            Vec3::ONE,
            1.0,
            (30.0_f32).to_radians(),
            (20.0_f32).to_radians(),
        );
        let midpoint_cos = 0.5 * (spot.cos_total_width + spot.cos_falloff_start);
        let sin_theta = (1.0 - midpoint_cos * midpoint_cos).sqrt();
        let dir = Vec3::new(sin_theta, 0.0, -midpoint_cos);
        let falloff = spot.falloff(dir);
        assert!((falloff - 0.5).abs() < 1.0e-5);
    }

    #[test]
    fn spot_light_axis_radiance_is_independent_of_cone_width() {
        // Same power, different cone width -> axis radiance should match,
        // because the mask only changes which directions are lit.
        let wide = SpotLight::new(
            Vec3::new(0.0, 0.0, 2.0),
            Vec3::NEG_Z,
            Vec3::ONE,
            4.0,
            (40.0_f32).to_radians(),
            (30.0_f32).to_radians(),
        );
        let narrow = SpotLight::new(
            Vec3::new(0.0, 0.0, 2.0),
            Vec3::NEG_Z,
            Vec3::ONE,
            4.0,
            (10.0_f32).to_radians(),
            (5.0_f32).to_radians(),
        );

        let mut wide_scene = Scene::new();
        wide_scene.add_spot_light(wide);
        let mut narrow_scene = Scene::new();
        narrow_scene.add_spot_light(narrow);

        let ctx = LightSampleContext {
            p: Vec3::ZERO,
            ng: Vec3::Z,
            ns: Vec3::Z,
        };
        let wide_li =
            sample_light_li(&wide_scene, LightKind::DeltaSpot(0), &ctx, 0.0, Vec2::ZERO).unwrap();
        let narrow_li = sample_light_li(
            &narrow_scene,
            LightKind::DeltaSpot(0),
            &ctx,
            0.0,
            Vec2::ZERO,
        )
        .unwrap();
        assert!(wide_li.radiance.abs_diff_eq(narrow_li.radiance, 1.0e-5));
    }

    #[test]
    fn spot_light_sample_returns_none_outside_cone() {
        let mut scene = Scene::new();
        scene.add_spot_light(SpotLight::new(
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::NEG_Z,
            Vec3::ONE,
            1.0,
            (10.0_f32).to_radians(),
            (5.0_f32).to_radians(),
        ));

        let ctx = LightSampleContext {
            p: Vec3::new(10.0, 0.0, 0.0),
            ng: Vec3::Z,
            ns: Vec3::Z,
        };
        let li = sample_light_li(&scene, LightKind::DeltaSpot(0), &ctx, 0.0, Vec2::ZERO);
        assert!(li.is_none());
    }

    #[test]
    fn spot_light_sample_inside_cone_matches_point_light_formula() {
        let mut scene = Scene::new();
        // P = 16π so Li at r=2 on axis equals color (same as point light).
        scene.add_spot_light(SpotLight::new(
            Vec3::new(0.0, 0.0, 2.0),
            Vec3::NEG_Z,
            Vec3::new(1.0, 0.5, 0.25),
            16.0 * PI,
            (30.0_f32).to_radians(),
            (20.0_f32).to_radians(),
        ));
        let ctx = LightSampleContext {
            p: Vec3::ZERO,
            ng: Vec3::Z,
            ns: Vec3::Z,
        };
        let li = sample_light_li(&scene, LightKind::DeltaSpot(0), &ctx, 0.0, Vec2::ZERO)
            .expect("spot light should sample along its axis");

        assert_eq!(li.light_type, LightType::DeltaPosition);
        assert_eq!(li.pdf, 1.0);
        assert!(li.radiance.abs_diff_eq(Vec3::new(1.0, 0.5, 0.25), 1.0e-5));
    }
}
