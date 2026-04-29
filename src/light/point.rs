use std::f32::consts::PI;

use glam::Vec3;

use super::{LightLiSample, LightSampleContext, LightType};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PointLightIndex(pub usize);

// Isotropic point light authored as (color, intensity).
// `color` is the normalized spectral tint; `intensity` is the total emitted
// power in watts (W). Matches the convention used by UE / VRay / Blender
// where the bulb's wattage is the primary brightness knob.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointLight {
    pub position: Vec3,
    /// Normalized RGB tint.
    pub color: Vec3,
    /// Total emitted power in watts (W).
    pub intensity: f32,
}

impl PointLight {
    pub fn new(position: Vec3, color: Vec3, intensity: f32) -> Self {
        Self {
            position,
            color,
            intensity,
        }
    }
}

pub fn sample_li(light: &PointLight, ctx: &LightSampleContext) -> Option<LightLiSample> {
    let to_light = light.position - ctx.p;
    let distance_squared = to_light.length_squared();
    if distance_squared <= 0.0 {
        return None;
    }
    let distance = distance_squared.sqrt();
    let wi = to_light / distance;
    // P (W) -> I (W/sr) by dividing over the full sphere: I = P / (4π).
    // Li contributed at the surface is I / r².
    let radiance = light.color * (light.intensity / (4.0 * PI * distance_squared));
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

    use glam::Vec3;

    use super::super::{LightSampleContext, LightType};
    use super::PointLight;
    use super::sample_li;

    #[test]
    fn point_light_sample_returns_inverse_square_radiance() {
        // Pick power so the expected Li is numerically simple: at r=2, Li = P / (4π·4).
        // P = 16π -> Li = 1.
        let light = PointLight::new(Vec3::new(0.0, 0.0, 2.0), Vec3::ONE, 16.0 * PI);

        let ctx = LightSampleContext {
            p: Vec3::ZERO,
            ng: Vec3::Z,
            ns: Vec3::Z,
        };
        let li = sample_li(&light, &ctx).expect("point light must sample");

        assert_eq!(li.light_type, LightType::DeltaPosition);
        assert!(li.target_triangle.is_none());
        assert_eq!(li.pdf, 1.0);
        assert!((li.distance - 2.0).abs() < 1.0e-5);
        assert!(li.wi.abs_diff_eq(Vec3::Z, 1.0e-5));
        assert!(li.radiance.abs_diff_eq(Vec3::splat(1.0), 1.0e-5));
    }

    #[test]
    fn point_light_sample_scales_by_color() {
        let light = PointLight::new(
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.5, 0.25),
            4.0 * PI,
        );
        let ctx = LightSampleContext {
            p: Vec3::ZERO,
            ng: Vec3::Z,
            ns: Vec3::Z,
        };
        let li = sample_li(&light, &ctx).expect("point light must sample");
        assert!(li.radiance.abs_diff_eq(Vec3::new(1.0, 0.5, 0.25), 1.0e-5));
    }

    #[test]
    fn point_light_sample_returns_none_when_sharing_position() {
        let light = PointLight::new(Vec3::ZERO, Vec3::ONE, 4.0);
        let ctx = LightSampleContext {
            p: Vec3::ZERO,
            ng: Vec3::Z,
            ns: Vec3::Z,
        };
        let li = sample_li(&light, &ctx);
        assert!(li.is_none());
    }
}
