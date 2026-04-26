use std::path::Path;

use glam::Vec3;

use crate::math::OrthonormalBasis;

use super::{ShadingVertex, Texture, TextureColorSpace};

#[derive(Debug, Clone, PartialEq)]
pub struct NormalMap {
    texture: Texture,
}

impl NormalMap {
    pub fn from_texture(texture: Texture) -> Self {
        Self { texture }
    }

    pub fn from_file(path: impl AsRef<Path>) -> image::ImageResult<Self> {
        Ok(Self::from_texture(Texture::from_file_with_color_space(
            path,
            TextureColorSpace::Linear,
        )?))
    }

    pub fn apply(&self, shading_vertex: &ShadingVertex, strength: f32) -> ShadingVertex {
        let mut mapped = *shading_vertex;
        let local_normal = self.local_normal_at(shading_vertex, strength);

        if local_normal.length_squared() == 0.0 {
            return mapped;
        }

        let ns = shading_vertex.frame.local_to_world(local_normal);
        if !ns.is_finite() || ns.length_squared() == 0.0 {
            return mapped;
        }

        mapped.ns = ns;
        mapped.frame = OrthonormalBasis::from_normal_and_tangent(ns, shading_vertex.dpdu);
        mapped
    }

    fn local_normal_at(&self, shading_vertex: &ShadingVertex, strength: f32) -> Vec3 {
        let rgb = self.texture.sample_rgb_filtered(
            shading_vertex.uv,
            shading_vertex.uv_dx(),
            shading_vertex.uv_dy(),
        );
        let encoded_normal = 2.0 * rgb - Vec3::ONE;
        let strength = strength.max(0.0);

        Vec3::new(
            encoded_normal.x * strength,
            encoded_normal.y * strength,
            encoded_normal.z,
        )
        .normalize_or_zero()
    }
}

pub(super) fn load_optional_normal_map(
    path: Option<&Path>,
) -> image::ImageResult<Option<NormalMap>> {
    path.map(NormalMap::from_file).transpose()
}

#[cfg(test)]
mod tests {
    use glam::{Vec2, Vec3};

    use crate::{
        material::{NormalMap, ShadingVertex, Texture},
        math::OrthonormalBasis,
        scene::{InstanceIndex, TriangleRef},
    };

    fn test_shading_vertex() -> ShadingVertex {
        ShadingVertex {
            triangle: TriangleRef {
                instance_index: InstanceIndex(0),
                triangle_index: 0,
            },
            p: Vec3::ZERO,
            uv: Vec2::ZERO,
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
    fn normal_map_replaces_shading_normal() {
        let local_normal = Vec3::new(0.6, 0.0, 0.8).normalize();
        let pixel = 0.5 * (local_normal + Vec3::ONE);
        let normal_map = NormalMap::from_texture(Texture::from_pixels(1, 1, vec![pixel]));
        let mapped = normal_map.apply(&test_shading_vertex(), 1.0);

        assert!(mapped.ns.abs_diff_eq(local_normal, 1.0e-6));
        assert!(mapped.frame.normal().abs_diff_eq(local_normal, 1.0e-6));
        assert_eq!(mapped.ng, Vec3::Z);
    }

    #[test]
    fn strength_scales_tangent_space_xy_components() {
        let local_normal = Vec3::new(0.6, 0.0, 0.8).normalize();
        let pixel = 0.5 * (local_normal + Vec3::ONE);
        let normal_map = NormalMap::from_texture(Texture::from_pixels(1, 1, vec![pixel]));
        let mapped = normal_map.apply(&test_shading_vertex(), 0.5);
        let expected =
            Vec3::new(local_normal.x * 0.5, local_normal.y * 0.5, local_normal.z).normalize();

        assert!(mapped.ns.abs_diff_eq(expected, 1.0e-6));
        assert!(mapped.frame.normal().abs_diff_eq(expected, 1.0e-6));
    }

    #[test]
    fn zero_strength_returns_flat_shading_normal() {
        let local_normal = Vec3::new(0.6, 0.0, 0.8).normalize();
        let pixel = 0.5 * (local_normal + Vec3::ONE);
        let normal_map = NormalMap::from_texture(Texture::from_pixels(1, 1, vec![pixel]));
        let mapped = normal_map.apply(&test_shading_vertex(), 0.0);

        assert!(mapped.ns.abs_diff_eq(Vec3::Z, 1.0e-6));
        assert!(mapped.frame.normal().abs_diff_eq(Vec3::Z, 1.0e-6));
    }
}
