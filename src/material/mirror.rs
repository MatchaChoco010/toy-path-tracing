use std::path::Path;

use glam::Vec3;
use rand::rngs::ThreadRng;

use crate::bsdf::MirrorBsdf;

use super::{
    MaterialSample, NormalMap, ShadingVertex, Texture, TextureColorSpace,
    normal_map::load_optional_normal_map, sample_matches_geometric_side,
    texture::load_optional_texture,
};

#[derive(Debug, Clone, PartialEq)]
pub struct MirrorMaterial {
    pub color: Vec3,
    pub color_texture: Option<Texture>,
    pub normal_map: Option<NormalMap>,
    pub normal_strength: f32,
}

impl MirrorMaterial {
    pub fn new(color: Vec3) -> Self {
        Self {
            color,
            color_texture: None,
            normal_map: None,
            normal_strength: 1.0,
        }
    }

    pub fn try_new_with_texture_path(
        color: Vec3,
        color_texture_path: Option<&Path>,
        normal_map_path: Option<&Path>,
    ) -> image::ImageResult<Self> {
        Ok(Self {
            color,
            color_texture: load_optional_texture(color_texture_path, TextureColorSpace::Srgb)?,
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
        _rng: &mut ThreadRng,
    ) -> Option<MaterialSample> {
        let wo_local = shading_vertex
            .frame
            .world_to_local(shading_vertex.wo)
            .normalize_or_zero();
        let bsdf = MirrorBsdf::new(self.color_at(shading_vertex));
        let sample = bsdf.sample(wo_local)?;
        let wi = shading_vertex.frame.local_to_world(sample.wi);

        let sample = MaterialSample {
            weight: sample.weight,
            wi,
            pdf: sample.pdf,
            flags: sample.flags,
            eta: sample.eta,
            cone_spread: 0.0,
        };

        sample_matches_geometric_side(&sample, shading_vertex.ng).then_some(sample)
    }

    pub fn eval(&self, _shading_vertex: &ShadingVertex, _wi: Vec3) -> Vec3 {
        Vec3::ZERO
    }

    pub fn pdf(&self, _shading_vertex: &ShadingVertex, _wi: Vec3) -> f32 {
        0.0
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

    fn color_at(&self, shading_vertex: &ShadingVertex) -> Vec3 {
        self.color
            * self
                .color_texture
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
    use glam::{Vec2, Vec3};

    use crate::{
        material::{MirrorMaterial, ShadingVertex, Texture},
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
    fn texture_modulates_color() {
        let material = MirrorMaterial {
            color: Vec3::new(0.5, 0.5, 0.5),
            color_texture: Some(Texture::from_pixels(1, 1, vec![Vec3::new(0.2, 0.4, 0.6)])),
            normal_map: None,
            normal_strength: 1.0,
        };
        let vtx = test_shading_vertex(Vec2::ZERO);
        let mut rng = rand::rng();
        let sample = material
            .sample(&vtx, &mut rng)
            .expect("expected a valid mirror sample");

        assert!(sample.weight.abs_diff_eq(Vec3::new(0.1, 0.2, 0.3), 1.0e-6));
    }

    #[test]
    fn sample_returns_none_when_shading_normal_reflects_below_geometry() {
        let material = MirrorMaterial::new(Vec3::ONE);
        let mut vtx = test_shading_vertex(Vec2::ZERO);
        let ns = Vec3::new(0.8660254, 0.0, 0.5).normalize();
        vtx.ns = ns;
        vtx.frame = OrthonormalBasis::from_normal_and_tangent(ns, vtx.dpdu);
        let mut rng = rand::rng();

        assert!(material.sample(&vtx, &mut rng).is_none());
    }
}
