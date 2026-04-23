use std::path::Path;

use glam::{Vec2, Vec3};
use rand::{RngExt, rngs::ThreadRng};

use crate::bsdf::NormalizedLambertBsdf;

use super::{
    MaterialSample, ShadingVertex, Texture, TextureColorSpace, texture::load_optional_texture,
};

#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedLambertMaterial {
    pub rho: Vec3,
    pub rho_texture: Option<Texture>,
}

impl NormalizedLambertMaterial {
    pub fn new(rho: Vec3) -> Self {
        Self {
            rho,
            rho_texture: None,
        }
    }

    pub fn try_new_with_texture_path(
        rho: Vec3,
        rho_texture_path: Option<&Path>,
    ) -> image::ImageResult<Self> {
        Ok(Self {
            rho,
            rho_texture: load_optional_texture(rho_texture_path, TextureColorSpace::Srgb)?,
        })
    }

    pub fn sample(
        &self,
        shading_vertex: &ShadingVertex,
        rng: &mut ThreadRng,
    ) -> Option<MaterialSample> {
        let wo_local = shading_vertex
            .frame
            .world_to_local(shading_vertex.wo)
            .normalize_or_zero();
        let us = Vec2::new(rng.random::<f32>(), rng.random::<f32>());
        let bsdf = NormalizedLambertBsdf::new(self.rho_at(shading_vertex));
        let sample = bsdf.sample(wo_local, us)?;
        let wi = shading_vertex.frame.local_to_world(sample.wi);

        Some(MaterialSample {
            weight: sample.weight,
            wi,
            pdf: sample.pdf,
            flags: sample.flags,
        })
    }

    pub fn eval(&self, shading_vertex: &ShadingVertex, wi: Vec3) -> Vec3 {
        let wo_local = shading_vertex
            .frame
            .world_to_local(shading_vertex.wo)
            .normalize_or_zero();
        let wi_local = shading_vertex.frame.world_to_local(wi).normalize_or_zero();
        let bsdf = NormalizedLambertBsdf::new(self.rho_at(shading_vertex));
        bsdf.eval(wo_local, wi_local)
    }

    pub fn pdf(&self, shading_vertex: &ShadingVertex, wi: Vec3) -> f32 {
        let wo_local = shading_vertex
            .frame
            .world_to_local(shading_vertex.wo)
            .normalize_or_zero();
        let wi_local = shading_vertex.frame.world_to_local(wi).normalize_or_zero();
        let bsdf = NormalizedLambertBsdf::new(self.rho_at(shading_vertex));
        bsdf.pdf(wo_local, wi_local)
    }

    pub fn le(&self, _shading_vertex: &ShadingVertex) -> Option<Vec3> {
        None
    }

    pub fn may_emit(&self) -> bool {
        false
    }

    pub fn max_emission(&self) -> f32 {
        0.0
    }

    fn rho_at(&self, shading_vertex: &ShadingVertex) -> Vec3 {
        self.rho
            * self
                .rho_texture
                .as_ref()
                .map(|texture| texture.sample_rgb(shading_vertex.uv))
                .unwrap_or(Vec3::ONE)
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use glam::{Vec2, Vec3};

    use crate::{
        material::{NormalizedLambertMaterial, ShadingVertex, Texture},
        math::OrthonormalBasis,
        scene::{InstanceIndex, TriangleRef},
    };

    fn test_shading_vertex(uv: Vec2) -> ShadingVertex {
        ShadingVertex {
            triangle: TriangleRef {
                instance_index: InstanceIndex(0),
                triangle_index: 0,
            },
            p: Vec3::ZERO,
            uv,
            ng: Vec3::Z,
            ns: Vec3::Z,
            wo: Vec3::Z,
            dpdu: Vec3::X,
            dpdv: Vec3::Y,
            frame: OrthonormalBasis::from_normal(Vec3::Z),
            front_face: true,
        }
    }

    #[test]
    fn texture_modulates_rho() {
        let material = NormalizedLambertMaterial {
            rho: Vec3::new(0.5, 0.5, 0.5),
            rho_texture: Some(Texture::from_pixels(1, 1, vec![Vec3::new(0.2, 0.4, 0.6)])),
        };
        let vtx = test_shading_vertex(Vec2::ZERO);

        assert!(
            material
                .eval(&vtx, Vec3::Z)
                .abs_diff_eq(Vec3::new(0.1, 0.2, 0.3) / PI, 1.0e-6)
        );
    }

    #[test]
    fn none_texture_keeps_existing_rho() {
        let material = NormalizedLambertMaterial::try_new_with_texture_path(Vec3::ONE, None)
            .expect("None texture should not try to load an image");

        assert_eq!(material, NormalizedLambertMaterial::new(Vec3::ONE));
    }
}
