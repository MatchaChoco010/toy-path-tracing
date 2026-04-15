use glam::Vec3;

use super::{LightLiSample, LightType};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DirectionalLightIndex(pub usize);

// Parallel light source at infinity, authored as (color, intensity).
// `intensity` is the perpendicular irradiance (W/m²) incident on a surface
// whose normal aligns with the incoming light. For a delta-direction
// distribution this is numerically identical to the light's emitted radiance,
// so it can be fed straight into the BSDF integrand. Matches UE/VRay where
// sun-like lights are dialled in by illuminance rather than radiance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectionalLight {
    /// Direction the light is traveling along (unit vector).
    pub direction: Vec3,
    /// Normalized RGB tint.
    pub color: Vec3,
    /// Perpendicular irradiance in W/m².
    pub intensity: f32,
}

impl DirectionalLight {
    pub fn new(direction: Vec3, color: Vec3, intensity: f32) -> Self {
        Self {
            direction: direction.normalize(),
            color,
            intensity,
        }
    }
}

pub(super) fn sample_li(light: &DirectionalLight) -> Option<LightLiSample> {
    Some(LightLiSample {
        radiance: light.color * light.intensity,
        wi: -light.direction,
        pdf: 1.0,
        distance: f32::INFINITY,
        light_type: LightType::DeltaDirection,
        target_triangle: None,
    })
}

#[cfg(test)]
mod tests {
    use glam::{Vec2, Vec3};

    use super::super::{LightKind, LightSampleContext, LightType, sample_light_li};
    use super::DirectionalLight;
    use crate::scene::Scene;

    #[test]
    fn directional_light_sample_returns_irradiance_at_infinity() {
        let mut scene = Scene::new();
        scene.add_directional_light(DirectionalLight::new(
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::ONE,
            3.0,
        ));

        let ctx = LightSampleContext {
            p: Vec3::new(5.0, 7.0, 11.0),
            ng: Vec3::Z,
            ns: Vec3::Z,
        };
        let li = sample_light_li(
            &scene,
            LightKind::DeltaDirectional(0),
            &ctx,
            0.0,
            Vec2::ZERO,
        )
        .expect("directional light must sample");

        assert_eq!(li.light_type, LightType::DeltaDirection);
        assert_eq!(li.pdf, 1.0);
        assert!(li.distance.is_infinite());
        assert!(li.wi.abs_diff_eq(Vec3::Z, 1.0e-5));
        assert!(li.radiance.abs_diff_eq(Vec3::splat(3.0), 1.0e-5));
    }

    #[test]
    fn directional_light_sample_scales_by_color() {
        let mut scene = Scene::new();
        scene.add_directional_light(DirectionalLight::new(
            Vec3::NEG_Z,
            Vec3::new(0.2, 0.6, 1.0),
            5.0,
        ));
        let ctx = LightSampleContext {
            p: Vec3::ZERO,
            ng: Vec3::Z,
            ns: Vec3::Z,
        };
        let li = sample_light_li(
            &scene,
            LightKind::DeltaDirectional(0),
            &ctx,
            0.0,
            Vec2::ZERO,
        )
        .expect("directional light must sample");
        assert!(li.radiance.abs_diff_eq(Vec3::new(1.0, 3.0, 5.0), 1.0e-5));
    }

    #[test]
    fn directional_light_normalizes_direction() {
        let light = DirectionalLight::new(Vec3::new(0.0, 0.0, -4.0), Vec3::ONE, 1.0);
        assert!((light.direction.length() - 1.0).abs() < 1.0e-5);
    }
}
