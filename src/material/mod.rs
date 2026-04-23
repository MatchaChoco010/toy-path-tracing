mod conductor_ggx;
mod dielectric_ggx;
mod emissive;
mod glass;
mod mirror;
mod normalized_lambert;
mod texture;

use glam::{Vec2, Vec3};
use rand::rngs::ThreadRng;

use crate::{bsdf::BsdfFlags, math::OrthonormalBasis, scene::TriangleRef};

pub use conductor_ggx::ConductorGgxMaterial;
pub use dielectric_ggx::DielectricGgxMaterial;
pub use emissive::EmissiveMaterial;
pub use glass::GlassMaterial;
pub use mirror::MirrorMaterial;
pub use normalized_lambert::NormalizedLambertMaterial;
pub use texture::{Texture, TextureColorSpace};

#[derive(Debug, Clone, PartialEq)]
pub enum Material {
    NormalizedLambert(NormalizedLambertMaterial),
    Mirror(MirrorMaterial),
    ConductorGgx(ConductorGgxMaterial),
    DielectricGgx(DielectricGgxMaterial),
    Glass(GlassMaterial),
    Emissive(EmissiveMaterial),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShadingVertex {
    pub triangle: TriangleRef,
    pub p: Vec3,
    pub uv: Vec2,
    pub dudx: f32,
    pub dvdx: f32,
    pub dudy: f32,
    pub dvdy: f32,
    pub ng: Vec3,
    pub ns: Vec3,
    pub wo: Vec3,
    pub dpdu: Vec3,
    pub dpdv: Vec3,
    pub dpdx: Vec3,
    pub dpdy: Vec3,
    pub dndu: Vec3,
    pub dndv: Vec3,
    pub frame: OrthonormalBasis,
    pub front_face: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaterialSample {
    pub weight: Vec3,
    pub wi: Vec3,
    pub pdf: f32,
    pub flags: BsdfFlags,
    pub eta: f32,
    pub cone_spread: f32,
}

impl ShadingVertex {
    pub fn uv_dx(&self) -> Vec2 {
        Vec2::new(self.dudx, self.dvdx)
    }

    pub fn uv_dy(&self) -> Vec2 {
        Vec2::new(self.dudy, self.dvdy)
    }
}

impl Material {
    pub fn sample(
        &self,
        shading_vertex: &ShadingVertex,
        rng: &mut ThreadRng,
    ) -> Option<MaterialSample> {
        match self {
            Self::NormalizedLambert(material) => material.sample(shading_vertex, rng),
            Self::Mirror(material) => material.sample(shading_vertex, rng),
            Self::ConductorGgx(material) => material.sample(shading_vertex, rng),
            Self::DielectricGgx(material) => material.sample(shading_vertex, rng),
            Self::Glass(material) => material.sample(shading_vertex, rng),
            Self::Emissive(material) => material.sample(shading_vertex, rng),
        }
    }

    pub fn le(&self, shading_vertex: &ShadingVertex) -> Option<Vec3> {
        match self {
            Self::NormalizedLambert(material) => material.le(shading_vertex),
            Self::Mirror(material) => material.le(shading_vertex),
            Self::ConductorGgx(material) => material.le(shading_vertex),
            Self::DielectricGgx(material) => material.le(shading_vertex),
            Self::Glass(material) => material.le(shading_vertex),
            Self::Emissive(material) => material.le(shading_vertex),
        }
    }

    pub fn eval(&self, shading_vertex: &ShadingVertex, wi: Vec3) -> Vec3 {
        match self {
            Self::NormalizedLambert(material) => material.eval(shading_vertex, wi),
            Self::Mirror(material) => material.eval(shading_vertex, wi),
            Self::ConductorGgx(material) => material.eval(shading_vertex, wi),
            Self::DielectricGgx(material) => material.eval(shading_vertex, wi),
            Self::Glass(material) => material.eval(shading_vertex, wi),
            Self::Emissive(material) => material.eval(shading_vertex, wi),
        }
    }

    pub fn pdf(&self, shading_vertex: &ShadingVertex, wi: Vec3) -> f32 {
        match self {
            Self::NormalizedLambert(material) => material.pdf(shading_vertex, wi),
            Self::Mirror(material) => material.pdf(shading_vertex, wi),
            Self::ConductorGgx(material) => material.pdf(shading_vertex, wi),
            Self::DielectricGgx(material) => material.pdf(shading_vertex, wi),
            Self::Glass(material) => material.pdf(shading_vertex, wi),
            Self::Emissive(material) => material.pdf(shading_vertex, wi),
        }
    }

    pub fn may_emit(&self) -> bool {
        match self {
            Self::NormalizedLambert(material) => material.may_emit(),
            Self::Mirror(material) => material.may_emit(),
            Self::ConductorGgx(material) => material.may_emit(),
            Self::DielectricGgx(material) => material.may_emit(),
            Self::Glass(material) => material.may_emit(),
            Self::Emissive(material) => material.may_emit(),
        }
    }

    pub fn max_emission(&self) -> f32 {
        match self {
            Self::NormalizedLambert(material) => material.max_emission(),
            Self::Mirror(material) => material.max_emission(),
            Self::ConductorGgx(material) => material.max_emission(),
            Self::DielectricGgx(material) => material.max_emission(),
            Self::Glass(material) => material.max_emission(),
            Self::Emissive(material) => material.max_emission(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use glam::Vec3;

    use crate::{
        bsdf::BsdfFlags,
        math::OrthonormalBasis,
        scene::{InstanceIndex, TriangleRef},
    };

    use super::{
        ConductorGgxMaterial, DielectricGgxMaterial, EmissiveMaterial, GlassMaterial, Material,
        MirrorMaterial, NormalizedLambertMaterial, ShadingVertex,
    };

    fn test_shading_vertex(wo: Vec3) -> ShadingVertex {
        ShadingVertex {
            triangle: TriangleRef {
                instance_index: InstanceIndex(0),
                triangle_index: 0,
            },
            p: Vec3::ZERO,
            uv: glam::Vec2::ZERO,
            dudx: 0.0,
            dvdx: 0.0,
            dudy: 0.0,
            dvdy: 0.0,
            ng: Vec3::Z,
            ns: Vec3::Z,
            wo,
            dpdu: Vec3::X,
            dpdv: Vec3::Y,
            dpdx: Vec3::ZERO,
            dpdy: Vec3::ZERO,
            dndu: Vec3::ZERO,
            dndv: Vec3::ZERO,
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
    fn mirror_material_reports_no_emission_capability() {
        let material = Material::Mirror(MirrorMaterial::new(Vec3::ONE));

        assert!(!material.may_emit());
        assert_eq!(material.max_emission(), 0.0);
    }

    #[test]
    fn conductor_material_reports_no_emission_capability() {
        let material = Material::ConductorGgx(ConductorGgxMaterial::new(Vec3::ONE, 0.5, 0.0));

        assert!(!material.may_emit());
        assert_eq!(material.max_emission(), 0.0);
    }

    #[test]
    fn glass_material_reports_no_emission_capability() {
        let material = Material::Glass(GlassMaterial::new(1.5, Vec3::ONE, false));

        assert!(!material.may_emit());
        assert_eq!(material.max_emission(), 0.0);
    }

    #[test]
    fn dielectric_ggx_material_reports_no_emission_capability() {
        let material =
            Material::DielectricGgx(DielectricGgxMaterial::new(Vec3::ONE, 1.5, 0.3, 0.0, false));

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

    #[test]
    fn lambert_material_pdf_delegates_to_bsdf() {
        let material = Material::NormalizedLambert(NormalizedLambertMaterial::new(Vec3::ONE));
        let shading_vertex = test_shading_vertex(Vec3::Z);
        let wi = Vec3::new(0.2, 0.3, 0.9327379).normalize();

        let pdf = material.pdf(&shading_vertex, wi);

        assert!((pdf - wi.z / PI).abs() < 1.0e-6);
    }

    #[test]
    fn lambert_material_sample_returns_diffuse_flag() {
        let material = Material::NormalizedLambert(NormalizedLambertMaterial::new(Vec3::ONE));
        let shading_vertex = test_shading_vertex(Vec3::Z);
        let mut rng = rand::rng();

        let sample = material
            .sample(&shading_vertex, &mut rng)
            .expect("expected a valid sample");

        assert_eq!(sample.flags, BsdfFlags::DIFFUSE | BsdfFlags::REFLECTION);
    }

    #[test]
    fn mirror_material_sample_returns_delta_flag() {
        let color = Vec3::new(0.3, 0.5, 0.7);
        let material = Material::Mirror(MirrorMaterial::new(color));
        let wo = Vec3::new(0.3, -0.4, 0.8660254).normalize();
        let shading_vertex = test_shading_vertex(wo);
        let mut rng = rand::rng();

        let sample = material
            .sample(&shading_vertex, &mut rng)
            .expect("expected a valid sample");

        let expected_wi = Vec3::new(-wo.x, -wo.y, wo.z).normalize();
        assert!(sample.wi.abs_diff_eq(expected_wi, 1.0e-6));
        assert_eq!(sample.weight, color);
        assert_eq!(sample.pdf, 1.0);
        assert_eq!(sample.flags, BsdfFlags::DELTA | BsdfFlags::REFLECTION);
    }

    #[test]
    fn conductor_material_sample_returns_glossy_or_delta_reflection_flag() {
        let material = Material::ConductorGgx(ConductorGgxMaterial::new(
            Vec3::new(0.9, 0.7, 0.3),
            0.0,
            0.0,
        ));
        let shading_vertex = test_shading_vertex(Vec3::new(0.3, -0.4, 0.8660254).normalize());
        let mut rng = rand::rng();

        let sample = material
            .sample(&shading_vertex, &mut rng)
            .expect("expected a valid sample");

        assert!(sample.flags.contains(BsdfFlags::REFLECTION));
        assert!(
            sample.flags.contains(BsdfFlags::GLOSSY) || sample.flags.contains(BsdfFlags::DELTA)
        );
    }

    #[test]
    fn mirror_material_eval_and_pdf_are_zero() {
        let material = Material::Mirror(MirrorMaterial::new(Vec3::ONE));
        let shading_vertex = test_shading_vertex(Vec3::Z);

        assert_eq!(material.eval(&shading_vertex, Vec3::Z), Vec3::ZERO);
        assert_eq!(material.pdf(&shading_vertex, Vec3::Z), 0.0);
    }

    #[test]
    fn glass_material_sample_can_return_transmission_flag() {
        let color = Vec3::new(0.3, 0.5, 0.7);
        let material = Material::Glass(GlassMaterial::new(1.5, color, false));
        let wo = Vec3::new(0.3, -0.4, 0.8660254).normalize();
        let shading_vertex = test_shading_vertex(wo);
        let mut rng = rand::rng();

        // Transmission probability at this angle is ~95%; retry a few times to
        // avoid a flaky test when the RNG happens to pick the reflection branch.
        let sample = (0..64)
            .find_map(|_| {
                let s = material.sample(&shading_vertex, &mut rng)?;
                s.flags.contains(BsdfFlags::TRANSMISSION).then_some(s)
            })
            .expect("expected a transmission sample within retry budget");

        assert!(sample.wi.z < 0.0);
        assert!(sample.weight.abs_diff_eq(color * 2.25, 1.0e-6));
        assert_eq!(sample.flags, BsdfFlags::DELTA | BsdfFlags::TRANSMISSION);
    }

    #[test]
    fn glass_material_eval_and_pdf_are_zero() {
        let material = Material::Glass(GlassMaterial::new(1.5, Vec3::ONE, false));
        let shading_vertex = test_shading_vertex(Vec3::Z);

        assert_eq!(material.eval(&shading_vertex, Vec3::Z), Vec3::ZERO);
        assert_eq!(material.pdf(&shading_vertex, Vec3::Z), 0.0);
    }

    #[test]
    fn emissive_material_pdf_is_always_zero_for_mis() {
        let material = Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 2.0));
        let shading_vertex = test_shading_vertex(Vec3::Z);

        assert_eq!(material.pdf(&shading_vertex, Vec3::Z), 0.0);
        assert_eq!(material.pdf(&shading_vertex, -Vec3::Z), 0.0);
    }
}
