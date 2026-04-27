use std::path::Path;

use glam::{Vec2, Vec3};
use rand::{RngExt, rngs::ThreadRng};

use crate::bsdf::NormalizedLambertBsdf;

use super::{
    GEOMETRIC_NORMAL_COS_EPSILON, MaterialSample, NormalMap, ShadingVertex, Texture,
    TextureColorSpace, normal_map::load_optional_normal_map, texture::load_optional_texture,
};

const DIFFUSE_CONE_SPREAD: f32 = 0.5;

#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedLambertMaterial {
    pub rho: Vec3,
    pub rho_texture: Option<Texture>,
    pub normal_map: Option<NormalMap>,
    pub normal_strength: f32,
}

impl NormalizedLambertMaterial {
    pub fn new(rho: Vec3) -> Self {
        Self {
            rho,
            rho_texture: None,
            normal_map: None,
            normal_strength: 1.0,
        }
    }

    pub fn try_new_with_texture_path(
        rho: Vec3,
        rho_texture_path: Option<&Path>,
        normal_map_path: Option<&Path>,
    ) -> image::ImageResult<Self> {
        Ok(Self {
            rho,
            rho_texture: load_optional_texture(rho_texture_path, TextureColorSpace::Srgb)?,
            normal_map: load_optional_normal_map(normal_map_path)?,
            normal_strength: 1.0,
        })
    }

    pub(crate) fn prepare_shading_vertex(&self, shading_vertex: &ShadingVertex) -> ShadingVertex {
        self.normal_map
            .as_ref()
            .map(|normal_map| normal_map.apply(shading_vertex, self.normal_strength))
            .unwrap_or(*shading_vertex)
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

        let sample = MaterialSample {
            weight: sample.weight,
            wi,
            pdf: sample.pdf,
            flags: sample.flags,
            eta: sample.eta,
            cone_spread: DIFFUSE_CONE_SPREAD,
        };

        if sample.wi.dot(shading_vertex.ng) <= GEOMETRIC_NORMAL_COS_EPSILON {
            return None;
        }

        Some(sample)
    }

    pub fn eval(&self, shading_vertex: &ShadingVertex, wi: Vec3) -> Vec3 {
        if shading_vertex.wo.dot(shading_vertex.ng) <= 0.0 || wi.dot(shading_vertex.ng) <= 0.0 {
            return Vec3::ZERO;
        }

        let wo_local = shading_vertex
            .frame
            .world_to_local(shading_vertex.wo)
            .normalize_or_zero();
        let wi_local = shading_vertex.frame.world_to_local(wi).normalize_or_zero();
        let bsdf = NormalizedLambertBsdf::new(self.rho_at(shading_vertex));
        bsdf.eval(wo_local, wi_local)
    }

    pub fn pdf(&self, shading_vertex: &ShadingVertex, wi: Vec3) -> f32 {
        if shading_vertex.wo.dot(shading_vertex.ng) <= 0.0 || wi.dot(shading_vertex.ng) <= 0.0 {
            return 0.0;
        }

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
                .map(|texture| {
                    texture.sample_rgb_filtered(
                        shading_vertex.uv,
                        shading_vertex.uv_dx(),
                        shading_vertex.uv_dy(),
                    )
                })
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
            dudx: 0.0,
            dvdx: 0.0,
            dudy: 0.0,
            dvdy: 0.0,
            ng: Vec3::Z,
            ns: Vec3::Z,
            wo: Vec3::Z,
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
    fn texture_modulates_rho() {
        let material = NormalizedLambertMaterial {
            rho: Vec3::new(0.5, 0.5, 0.5),
            rho_texture: Some(Texture::from_pixels(1, 1, vec![Vec3::new(0.2, 0.4, 0.6)])),
            normal_map: None,
            normal_strength: 1.0,
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
        let material = NormalizedLambertMaterial::try_new_with_texture_path(Vec3::ONE, None, None)
            .expect("None texture should not try to load an image");

        assert_eq!(material, NormalizedLambertMaterial::new(Vec3::ONE));
    }
}
