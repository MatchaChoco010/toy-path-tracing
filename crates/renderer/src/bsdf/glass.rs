use glam::Vec3;

use crate::math::{fresnel_dielectric, refract};

use super::{BsdfFlags, BsdfSample};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlassBsdf {
    eta: f32,
    color: Vec3,
    thin: bool,
    front_face: bool,
}

impl GlassBsdf {
    pub fn new(eta: f32, color: Vec3, thin: bool, front_face: bool) -> Self {
        Self {
            eta,
            color,
            thin,
            front_face,
        }
    }

    pub fn eval(&self, _wo: Vec3, _wi: Vec3) -> Vec3 {
        Vec3::ZERO
    }

    pub fn pdf(&self, _wo: Vec3, _wi: Vec3) -> f32 {
        0.0
    }

    pub fn sample(&self, wo: Vec3, uc: f32) -> Option<BsdfSample> {
        if wo.z <= 0.0 || self.eta <= 0.0 {
            return None;
        }

        if self.thin {
            return self.sample_thin(wo, uc);
        }

        self.sample_thick(wo, uc)
    }

    fn sample_thin(&self, wo: Vec3, uc: f32) -> Option<BsdfSample> {
        let mut reflectance = fresnel_dielectric(wo.z.abs(), 1.0, self.eta);
        let transmittance = 1.0 - reflectance;
        let denominator = 1.0 - reflectance * reflectance;

        if denominator <= 0.0 {
            reflectance = 1.0;
        } else {
            reflectance += transmittance * transmittance * reflectance / denominator;
        }

        let transmittance = (1.0 - reflectance).max(0.0);
        let (reflection_probability, transmission_probability) =
            normalized_probabilities(reflectance, transmittance)?;

        let reflect = uc < reflection_probability;
        let wi = if reflect {
            reflected_direction(wo)
        } else {
            -wo
        };
        let pdf = if reflect {
            reflection_probability
        } else {
            transmission_probability
        };
        let weight = if reflect { Vec3::ONE } else { self.color };
        let flags = BsdfFlags::DELTA
            | if reflect {
                BsdfFlags::REFLECTION
            } else {
                BsdfFlags::TRANSMISSION
            };

        Some(BsdfSample {
            weight,
            wi,
            pdf,
            flags,
            wavelength_lock: None,
            eta: 1.0,
        })
    }

    fn sample_thick(&self, wo: Vec3, uc: f32) -> Option<BsdfSample> {
        let (eta_i, eta_t) = if self.front_face {
            (1.0, self.eta)
        } else {
            (self.eta, 1.0)
        };
        let eta = eta_i / eta_t;
        let reflectance = fresnel_dielectric(wo.z.abs(), eta_i, eta_t);
        let transmission_direction = refract(wo, eta);
        let transmittance = if transmission_direction.is_some() {
            1.0 - reflectance
        } else {
            0.0
        };
        let (reflection_probability, transmission_probability) =
            normalized_probabilities(reflectance, transmittance)?;

        if uc < reflection_probability {
            return Some(BsdfSample {
                weight: Vec3::ONE,
                wi: reflected_direction(wo),
                pdf: reflection_probability,
                flags: BsdfFlags::DELTA | BsdfFlags::REFLECTION,
                eta: 1.0,
                wavelength_lock: None,
            });
        }

        let wi = transmission_direction?;
        let radiance_scale = 1.0 / (eta * eta);

        Some(BsdfSample {
            weight: self.color * radiance_scale,
            wi,
            pdf: transmission_probability,
            flags: BsdfFlags::DELTA | BsdfFlags::TRANSMISSION,
            eta,
            wavelength_lock: None,
        })
    }
}

fn reflected_direction(wo: Vec3) -> Vec3 {
    Vec3::new(-wo.x, -wo.y, wo.z).normalize_or_zero()
}

fn normalized_probabilities(reflectance: f32, transmittance: f32) -> Option<(f32, f32)> {
    let reflection = reflectance.max(0.0);
    let transmission = transmittance.max(0.0);
    let probability_sum = reflection + transmission;

    if probability_sum <= 0.0 {
        return None;
    }

    Some((reflection / probability_sum, transmission / probability_sum))
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::{
        bsdf::{BsdfFlags, GlassBsdf},
        math::{fresnel_dielectric, refract},
    };

    #[test]
    fn eval_and_pdf_are_zero() {
        let bsdf = GlassBsdf::new(1.5, Vec3::new(0.3, 0.5, 0.7), false, true);

        assert_eq!(bsdf.eval(Vec3::Z, Vec3::Z), Vec3::ZERO);
        assert_eq!(bsdf.pdf(Vec3::Z, Vec3::Z), 0.0);
    }

    #[test]
    fn thick_sample_can_choose_reflection() {
        let bsdf = GlassBsdf::new(1.5, Vec3::new(0.3, 0.5, 0.7), false, true);
        let wo = Vec3::new(0.0, 0.0, 1.0);
        let sample = bsdf.sample(wo, 0.01).expect("expected a reflection sample");

        assert_eq!(sample.wi, Vec3::Z);
        assert_eq!(sample.weight, Vec3::ONE);
        assert!((sample.pdf - 0.04).abs() < 1.0e-6);
        assert_eq!(sample.flags, BsdfFlags::DELTA | BsdfFlags::REFLECTION);
    }

    #[test]
    fn thick_sample_can_choose_transmission_with_radiance_scaling() {
        let color = Vec3::new(0.3, 0.5, 0.7);
        let eta = 1.5;
        let bsdf = GlassBsdf::new(eta, color, false, true);
        let wo = Vec3::new(0.3, -0.4, 0.8660254).normalize();
        let sample = bsdf
            .sample(wo, 0.9)
            .expect("expected a transmission sample");
        let expected_wi = refract(wo, 1.0 / eta).expect("expected refraction");

        assert!(sample.wi.abs_diff_eq(expected_wi, 1.0e-6));
        assert!(sample.weight.abs_diff_eq(color * (eta * eta), 1.0e-6));
        assert_eq!(sample.flags, BsdfFlags::DELTA | BsdfFlags::TRANSMISSION);
        assert!((sample.pdf - (1.0 - fresnel_dielectric(wo.z, 1.0, eta))).abs() < 1.0e-6);
    }

    #[test]
    fn thick_sample_exiting_glass_scales_radiance_down() {
        let color = Vec3::new(0.3, 0.5, 0.7);
        let eta = 1.5;
        let bsdf = GlassBsdf::new(eta, color, false, false);
        let wo = Vec3::Z;
        let sample = bsdf
            .sample(wo, 0.99)
            .expect("expected a transmission sample");

        assert!(sample.weight.abs_diff_eq(color / (eta * eta), 1.0e-6));
        assert_eq!(sample.wi, -Vec3::Z);
        assert_eq!(sample.flags, BsdfFlags::DELTA | BsdfFlags::TRANSMISSION);
    }

    #[test]
    fn thick_sample_falls_back_to_reflection_for_total_internal_reflection() {
        let bsdf = GlassBsdf::new(1.5, Vec3::ONE, false, false);
        let wo = Vec3::new(0.8, 0.0, 0.6).normalize();
        let sample = bsdf
            .sample(wo, 0.9)
            .expect("expected total internal reflection");

        assert!(sample.wi.abs_diff_eq(Vec3::new(-0.8, 0.0, 0.6), 1.0e-6));
        assert_eq!(sample.weight, Vec3::ONE);
        assert_eq!(sample.pdf, 1.0);
        assert_eq!(sample.flags, BsdfFlags::DELTA | BsdfFlags::REFLECTION);
    }

    #[test]
    fn thin_sample_uses_sheet_reflection_probability_and_flips_direction_on_transmission() {
        let color = Vec3::new(0.3, 0.5, 0.7);
        let eta = 1.5;
        let bsdf = GlassBsdf::new(eta, color, true, true);
        let wo = Vec3::new(0.3, -0.4, 0.8660254).normalize();
        let sample = bsdf
            .sample(wo, 0.9)
            .expect("expected a thin transmission sample");
        let base_reflectance = fresnel_dielectric(wo.z, 1.0, eta);
        let expected_reflectance = base_reflectance
            + (1.0 - base_reflectance).powi(2) * base_reflectance
                / (1.0 - base_reflectance * base_reflectance);

        assert!(sample.wi.abs_diff_eq(-wo, 1.0e-6));
        assert!(sample.weight.abs_diff_eq(color, 1.0e-6));
        assert_eq!(sample.flags, BsdfFlags::DELTA | BsdfFlags::TRANSMISSION);
        assert!((sample.pdf - (1.0 - expected_reflectance)).abs() < 1.0e-6);
    }

    #[test]
    fn sample_returns_none_for_invalid_configuration() {
        let bsdf = GlassBsdf::new(1.5, Vec3::ONE, false, true);

        assert!(bsdf.sample(-Vec3::Z, 0.5).is_none());
        assert!(
            GlassBsdf::new(0.0, Vec3::ONE, false, true)
                .sample(Vec3::Z, 0.5)
                .is_none()
        );
    }
}
