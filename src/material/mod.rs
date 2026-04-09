mod emissive;
mod normalized_lambert;

use glam::{Vec2, Vec3};

use crate::{math::OrthonormalBasis, scene::TriangleRef};

pub use emissive::EmissiveMaterial;
pub use normalized_lambert::NormalizedLambertMaterial;

#[derive(Debug, Clone, PartialEq)]
pub enum Material {
    NormalizedLambert(NormalizedLambertMaterial),
    Emissive(EmissiveMaterial),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShadingVertex {
    pub triangle: TriangleRef,
    pub p: Vec3,
    pub uv: Vec2,
    pub ng: Vec3,
    pub ns: Vec3,
    pub dpdu: Vec3,
    pub dpdv: Vec3,
    pub frame: OrthonormalBasis,
    pub front_face: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaterialSample {
    pub weight: Vec3,
    pub wi: Vec3,
    pub pdf: f32,
}

impl Material {
    pub fn sample(
        &self,
        shading_vertex: &ShadingVertex,
        us: Vec2,
        wo: Vec3,
    ) -> Option<MaterialSample> {
        match self {
            Self::NormalizedLambert(material) => material.sample(shading_vertex, us, wo),
            Self::Emissive(material) => material.sample(shading_vertex, us, wo),
        }
    }

    pub fn le(&self, shading_vertex: &ShadingVertex) -> Option<Vec3> {
        match self {
            Self::NormalizedLambert(material) => material.le(shading_vertex),
            Self::Emissive(material) => material.le(shading_vertex),
        }
    }

    pub fn may_emit(&self) -> bool {
        match self {
            Self::NormalizedLambert(material) => material.may_emit(),
            Self::Emissive(material) => material.may_emit(),
        }
    }

    pub fn max_emission(&self) -> f32 {
        match self {
            Self::NormalizedLambert(material) => material.max_emission(),
            Self::Emissive(material) => material.max_emission(),
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::{EmissiveMaterial, Material, NormalizedLambertMaterial};

    #[test]
    fn emissive_material_reports_emission_capability() {
        let material = Material::Emissive(EmissiveMaterial::new(Vec3::new(0.25, 2.0, 1.0), 3.0));

        assert!(material.may_emit());
        assert_eq!(material.max_emission(), 6.0);
    }

    #[test]
    fn lambert_material_reports_no_emission_capability() {
        let material = Material::NormalizedLambert(NormalizedLambertMaterial::new(Vec3::ONE));

        assert!(!material.may_emit());
        assert_eq!(material.max_emission(), 0.0);
    }
}
