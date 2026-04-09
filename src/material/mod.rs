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
    pub wo: Vec3,
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
    pub fn sample(&self, shading_vertex: &ShadingVertex, us: Vec2) -> Option<MaterialSample> {
        match self {
            Self::NormalizedLambert(material) => material.sample(shading_vertex, us),
            Self::Emissive(material) => material.sample(shading_vertex, us),
        }
    }

    pub fn le(&self, shading_vertex: &ShadingVertex) -> Option<Vec3> {
        match self {
            Self::NormalizedLambert(material) => material.le(shading_vertex),
            Self::Emissive(material) => material.le(shading_vertex),
        }
    }

    pub fn eval(&self, shading_vertex: &ShadingVertex, wi: Vec3) -> Vec3 {
        match self {
            Self::NormalizedLambert(material) => material.eval(shading_vertex, wi),
            Self::Emissive(material) => material.eval(shading_vertex, wi),
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

    use crate::{
        math::OrthonormalBasis,
        scene::{InstanceIndex, TriangleRef},
    };

    use super::{EmissiveMaterial, Material, NormalizedLambertMaterial, ShadingVertex};

    fn test_shading_vertex(wo: Vec3) -> ShadingVertex {
        ShadingVertex {
            triangle: TriangleRef {
                instance_index: InstanceIndex(0),
                triangle_index: 0,
            },
            p: Vec3::ZERO,
            uv: glam::Vec2::ZERO,
            ng: Vec3::Z,
            ns: Vec3::Z,
            wo,
            dpdu: Vec3::X,
            dpdv: Vec3::Y,
            frame: OrthonormalBasis::from_normal(Vec3::Z),
            front_face: true,
        }
    }

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

    #[test]
    fn emissive_material_eval_is_always_zero() {
        let material = Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 2.0));
        let shading_vertex = test_shading_vertex(Vec3::Z);

        assert_eq!(material.eval(&shading_vertex, Vec3::Z), Vec3::ZERO);
    }

    #[test]
    fn lambert_material_eval_delegates_to_bsdf() {
        let material =
            Material::NormalizedLambert(NormalizedLambertMaterial::new(Vec3::new(0.3, 0.5, 0.7)));
        let shading_vertex = test_shading_vertex(Vec3::Z);
        let f = material.eval(&shading_vertex, Vec3::Z);

        assert!(f.abs_diff_eq(Vec3::new(0.3, 0.5, 0.7) / std::f32::consts::PI, 1.0e-6));
    }
}
