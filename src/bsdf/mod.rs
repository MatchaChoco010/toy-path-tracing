mod normalized_lambert;

use glam::Vec3;

pub use normalized_lambert::NormalizedLambertBsdf;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BsdfSample {
    pub weight: Vec3,
    pub wi: Vec3,
    pub pdf: f32,
}
