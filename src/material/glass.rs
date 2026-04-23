use std::path::Path;

use glam::Vec3;
use rand::{RngExt, rngs::ThreadRng};

use crate::{
    bsdf::{BsdfFlags, GlassBsdf},
    math::OrthonormalBasis,
};

use super::{
    MaterialSample, ShadingVertex, Texture, TextureColorSpace, texture::load_optional_texture,
};

#[derive(Debug, Clone, PartialEq)]
pub struct GlassMaterial {
    pub eta: f32,
    pub color: Vec3,
    pub color_texture: Option<Texture>,
    pub thin: bool,
}

impl GlassMaterial {
    pub fn new(eta: f32, color: Vec3, thin: bool) -> Self {
        Self {
            eta,
            color,
            color_texture: None,
            thin,
        }
    }

    pub fn try_new_with_texture_path(
        eta: f32,
        color: Vec3,
        color_texture_path: Option<&Path>,
        thin: bool,
    ) -> image::ImageResult<Self> {
        Ok(Self {
            eta,
            color,
            color_texture: load_optional_texture(color_texture_path, TextureColorSpace::Srgb)?,
            thin,
        })
    }

    pub fn sample(
        &self,
        shading_vertex: &ShadingVertex,
        rng: &mut ThreadRng,
    ) -> Option<MaterialSample> {
        let uc = rng.random::<f32>();
        let sample = self.sample_with_frame(shading_vertex, shading_vertex.frame, uc)?;

        if sample_matches_geometric_side(&sample, shading_vertex.ng) {
            return Some(sample);
        }

        // Shading normals can bend a transmission or reflection event across
        // the actual surface. Fall back to the geometric frame only in that
        // case so smooth normal interpolation still shapes the result when the
        // sampled direction is physically plausible.
        let geometric_frame = OrthonormalBasis::from_normal(shading_vertex.ng);
        let sample = self.sample_with_frame(shading_vertex, geometric_frame, uc)?;

        if !sample_matches_geometric_side(&sample, shading_vertex.ng) {
            return None;
        }

        Some(sample)
    }

    fn sample_with_frame(
        &self,
        shading_vertex: &ShadingVertex,
        frame: OrthonormalBasis,
        uc: f32,
    ) -> Option<MaterialSample> {
        let wo_local = frame.world_to_local(shading_vertex.wo).normalize_or_zero();
        let bsdf = GlassBsdf::new(
            self.eta,
            self.color_at(shading_vertex),
            self.thin,
            shading_vertex.front_face,
        );
        let sample = bsdf.sample(wo_local, uc)?;
        let wi = frame.local_to_world(sample.wi);

        Some(MaterialSample {
            weight: sample.weight,
            wi,
            pdf: sample.pdf,
            flags: sample.flags,
        })
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
                .map(|texture| texture.sample_rgb(shading_vertex.uv))
                .unwrap_or(Vec3::ONE)
    }
}

fn sample_matches_geometric_side(sample: &MaterialSample, geometric_normal: Vec3) -> bool {
    let side = sample.wi.dot(geometric_normal);
    let epsilon = 1.0e-6;

    if sample.flags.contains(BsdfFlags::TRANSMISSION) {
        return side < -epsilon;
    }

    if sample.flags.contains(BsdfFlags::REFLECTION) {
        return side > epsilon;
    }

    true
}

#[cfg(test)]
mod tests {
    use glam::{Vec2, Vec3};

    use crate::{
        material::{GlassMaterial, ShadingVertex, Texture},
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
    fn texture_modulates_transmission_color() {
        let material = GlassMaterial {
            eta: 1.5,
            color: Vec3::new(0.5, 0.5, 0.5),
            color_texture: Some(Texture::from_pixels(1, 1, vec![Vec3::new(0.2, 0.4, 0.6)])),
            thin: false,
        };
        let vtx = test_shading_vertex(Vec2::ZERO);
        let sample = material
            .sample_with_frame(&vtx, vtx.frame, 1.0)
            .expect("expected a transmission sample");
        let radiance_scale = 1.0 / (1.0 / material.eta).powi(2);

        assert!(
            sample
                .weight
                .abs_diff_eq(Vec3::new(0.1, 0.2, 0.3) * radiance_scale, 1.0e-6)
        );
    }
}
