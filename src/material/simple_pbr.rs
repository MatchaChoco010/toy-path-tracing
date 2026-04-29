use std::{path::Path, sync::Arc};

use glam::{Vec2, Vec3};
use rand::{RngExt, rngs::ThreadRng};

use crate::bsdf::{
    BsdfFlags, ConductorGgxBsdf, DielectricGgxAllowedPaths, DielectricGgxBsdf,
    DielectricGgxDirectionalAlbedoLut, NormalizedLambertBsdf, sanitize_dielectric_eta,
};

use super::{
    GEOMETRIC_NORMAL_COS_EPSILON, MaterialSample, NormalMap, ScalarTexture, ShadingVertex, Texture,
    TextureColorSpace,
    normal_map::load_optional_normal_map,
    texture::{load_optional_color_texture, load_optional_scalar_texture},
};

const MIN_ALPHA: f32 = 1.0e-4;

#[derive(Debug, Clone, PartialEq)]
pub struct SimplePbrMaterial {
    pub metallic: f32,
    pub roughness: f32,
    pub eta: f32,
    pub base_color: Vec3,
    pub anisotropy: f32,
    pub base_color_texture: Option<Arc<Texture>>,
    pub metallic_texture: Option<Arc<ScalarTexture>>,
    pub roughness_texture: Option<Arc<ScalarTexture>>,
    pub normal_map: Option<NormalMap>,
    pub normal_strength: f32,
    pub opacity: f32,
    pub opacity_texture: Option<Arc<ScalarTexture>>,
    dielectric_ggx_directional_albedo_lut: Option<Arc<DielectricGgxDirectionalAlbedoLut>>,
}

impl SimplePbrMaterial {
    pub fn new(base_color: Vec3, metallic: f32, roughness: f32, eta: f32, anisotropy: f32) -> Self {
        Self {
            metallic,
            roughness,
            eta: sanitize_dielectric_eta(eta),
            base_color,
            anisotropy,
            base_color_texture: None,
            metallic_texture: None,
            roughness_texture: None,
            normal_map: None,
            normal_strength: 1.0,
            opacity: 1.0,
            opacity_texture: None,
            dielectric_ggx_directional_albedo_lut: None,
        }
    }

    pub fn try_new_with_texture_paths(
        base_color: Vec3,
        metallic: f32,
        roughness: f32,
        eta: f32,
        anisotropy: f32,
        base_color_texture_path: Option<&Path>,
        metallic_texture_path: Option<&Path>,
        roughness_texture_path: Option<&Path>,
        normal_map_path: Option<&Path>,
    ) -> image::ImageResult<Self> {
        Ok(Self {
            metallic,
            roughness,
            eta: sanitize_dielectric_eta(eta),
            base_color,
            anisotropy,
            base_color_texture: load_optional_color_texture(
                base_color_texture_path,
                TextureColorSpace::Srgb,
            )?,
            metallic_texture: load_optional_scalar_texture(metallic_texture_path)?,
            roughness_texture: load_optional_scalar_texture(roughness_texture_path)?,
            normal_map: load_optional_normal_map(normal_map_path)?,
            normal_strength: 1.0,
            opacity: 1.0,
            opacity_texture: None,
            dielectric_ggx_directional_albedo_lut: None,
        })
    }

    pub fn opacity_at_uv(&self, shading_vertex: &ShadingVertex) -> f32 {
        let texture_factor = self
            .opacity_texture
            .as_ref()
            .map(|texture| {
                texture.sample_filtered(
                    shading_vertex.uv,
                    shading_vertex.uv_dx(),
                    shading_vertex.uv_dy(),
                )
            })
            .unwrap_or(1.0);
        (self.opacity * texture_factor).clamp(0.0, 1.0)
    }

    pub fn has_alpha_test(&self) -> bool {
        self.opacity < 1.0 || self.opacity_texture.is_some()
    }

    pub fn any_hit(&self, shading_vertex: &ShadingVertex, u: f32) -> bool {
        let alpha = self.opacity_at_uv(shading_vertex);
        alpha >= 1.0 || u < alpha
    }

    pub(crate) fn install_dielectric_ggx_directional_albedo_lut(
        &mut self,
        lut: Arc<DielectricGgxDirectionalAlbedoLut>,
    ) {
        self.dielectric_ggx_directional_albedo_lut = Some(lut);
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
        let u_component = rng.random::<f32>();
        let u_layer = rng.random::<f32>();
        let us = Vec2::new(rng.random::<f32>(), rng.random::<f32>());
        let sample = self.sample_impl(shading_vertex, u_component, u_layer, us)?;

        if sample.wi.dot(shading_vertex.ng) <= GEOMETRIC_NORMAL_COS_EPSILON {
            return None;
        }

        Some(sample)
    }

    fn sample_impl(
        &self,
        shading_vertex: &ShadingVertex,
        u_component: f32,
        u_layer: f32,
        us: Vec2,
    ) -> Option<MaterialSample> {
        if shading_vertex.wo.dot(shading_vertex.ng) <= 0.0 {
            return None;
        }

        let wo_local = shading_vertex
            .frame
            .world_to_local(shading_vertex.wo)
            .normalize_or_zero();
        if wo_local.z <= 0.0 {
            return None;
        }

        let params = self.params_at(shading_vertex);
        let (sample, selected_probability) = if u_component < params.metallic {
            let bsdf = ConductorGgxBsdf::new(params.base_color, params.alpha_x, params.alpha_y);
            (bsdf.sample(wo_local, us)?, params.metallic)
        } else {
            let coating_weight =
                self.lookup_directional_albedo(wo_local, params.roughness, params.anisotropy);
            let diffuse_weight = 1.0 - coating_weight;

            if u_layer < coating_weight {
                let bsdf = DielectricGgxBsdf::new_with_allowed_paths(
                    Vec3::ONE,
                    params.eta,
                    params.alpha_x,
                    params.alpha_y,
                    false,
                    true,
                    DielectricGgxAllowedPaths::Reflection,
                );
                (
                    bsdf.sample(wo_local, u_layer, us)?,
                    (1.0 - params.metallic) * coating_weight,
                )
            } else {
                let bsdf = NormalizedLambertBsdf::new(params.base_color);
                (
                    bsdf.sample(wo_local, us)?,
                    (1.0 - params.metallic) * diffuse_weight,
                )
            }
        };

        let wi = shading_vertex.frame.local_to_world(sample.wi);
        let wi_local = sample.wi;
        let cone_spread = if sample.flags.contains(BsdfFlags::GLOSSY) {
            2.0 * params.roughness.clamp(0.0, 1.0)
        } else {
            0.0
        };

        if sample.flags.contains(BsdfFlags::DELTA) {
            if selected_probability <= 0.0 {
                return None;
            }
            return Some(MaterialSample {
                weight: sample.weight / selected_probability,
                wi,
                pdf: selected_probability * sample.pdf,
                flags: sample.flags,
                eta: sample.eta,
                cone_spread,
            });
        }

        let pdf = self.pdf(shading_vertex, wi);
        if pdf <= 0.0 {
            return None;
        }

        let f = self.eval(shading_vertex, wi);
        if f.length_squared() == 0.0 {
            return None;
        }

        let cos_i = wi_local.z.max(0.0);
        if cos_i <= 0.0 {
            return None;
        }

        Some(MaterialSample {
            weight: f * (cos_i / pdf),
            wi,
            pdf,
            flags: sample.flags,
            eta: sample.eta,
            cone_spread,
        })
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
        if wo_local.z <= 0.0 || wi_local.z <= 0.0 {
            return Vec3::ZERO;
        }

        let params = self.params_at(shading_vertex);
        let conductor = ConductorGgxBsdf::new(params.base_color, params.alpha_x, params.alpha_y);
        let coating = DielectricGgxBsdf::new_with_allowed_paths(
            Vec3::ONE,
            params.eta,
            params.alpha_x,
            params.alpha_y,
            false,
            true,
            DielectricGgxAllowedPaths::Reflection,
        );
        let diffuse = NormalizedLambertBsdf::new(params.base_color);
        let coating_weight =
            self.lookup_directional_albedo(wo_local, params.roughness, params.anisotropy);
        let diffuse_weight = 1.0 - coating_weight;

        params.metallic * conductor.eval(wo_local, wi_local)
            + (1.0 - params.metallic)
                * (coating_weight * coating.eval(wo_local, wi_local)
                    + diffuse_weight * diffuse.eval(wo_local, wi_local))
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
        if wo_local.z <= 0.0 || wi_local.z <= 0.0 {
            return 0.0;
        }

        let params = self.params_at(shading_vertex);
        let conductor = ConductorGgxBsdf::new(params.base_color, params.alpha_x, params.alpha_y);
        let coating = DielectricGgxBsdf::new_with_allowed_paths(
            Vec3::ONE,
            params.eta,
            params.alpha_x,
            params.alpha_y,
            false,
            true,
            DielectricGgxAllowedPaths::Reflection,
        );
        let diffuse = NormalizedLambertBsdf::new(params.base_color);
        let coating_weight =
            self.lookup_directional_albedo(wo_local, params.roughness, params.anisotropy);
        let diffuse_weight = 1.0 - coating_weight;

        params.metallic * conductor.pdf(wo_local, wi_local)
            + (1.0 - params.metallic)
                * (coating_weight * coating.pdf(wo_local, wi_local)
                    + diffuse_weight * diffuse.pdf(wo_local, wi_local))
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

    fn lookup_directional_albedo(&self, w_local: Vec3, roughness: f32, anisotropy: f32) -> f32 {
        self.get_dielectric_ggx_directional_albedo_lut()
            .lookup(w_local, roughness, anisotropy)
    }

    fn get_dielectric_ggx_directional_albedo_lut(&self) -> &DielectricGgxDirectionalAlbedoLut {
        self.dielectric_ggx_directional_albedo_lut.as_deref().expect(
            "SimplePBR materials must be added to Scene before shading so the dielectric GGX directional albedo LUT is installed",
        )
    }

    fn params_at(&self, shading_vertex: &ShadingVertex) -> SimplePbrParams {
        let roughness = self.roughness_at(shading_vertex).clamp(0.0, 1.0);
        let anisotropy = self.anisotropy.clamp(-1.0, 1.0);
        let (alpha_x, alpha_y) = alpha_xy_from_roughness(roughness, anisotropy);

        SimplePbrParams {
            metallic: self.metallic_at(shading_vertex).clamp(0.0, 1.0),
            roughness,
            eta: self.eta,
            base_color: self
                .base_color_at(shading_vertex)
                .clamp(Vec3::ZERO, Vec3::ONE),
            anisotropy,
            alpha_x,
            alpha_y,
        }
    }

    fn base_color_at(&self, shading_vertex: &ShadingVertex) -> Vec3 {
        self.base_color
            * self
                .base_color_texture
                .as_ref()
                .map(|texture| {
                    texture.sample_filtered(
                        shading_vertex.uv,
                        shading_vertex.uv_dx(),
                        shading_vertex.uv_dy(),
                    )
                })
                .unwrap_or(Vec3::ONE)
    }

    fn roughness_at(&self, shading_vertex: &ShadingVertex) -> f32 {
        self.roughness
            * self
                .roughness_texture
                .as_ref()
                .map(|texture| {
                    texture.sample_filtered(
                        shading_vertex.uv,
                        shading_vertex.uv_dx(),
                        shading_vertex.uv_dy(),
                    )
                })
                .unwrap_or(1.0)
    }

    fn metallic_at(&self, shading_vertex: &ShadingVertex) -> f32 {
        self.metallic
            * self
                .metallic_texture
                .as_ref()
                .map(|texture| {
                    texture.sample_filtered(
                        shading_vertex.uv,
                        shading_vertex.uv_dx(),
                        shading_vertex.uv_dy(),
                    )
                })
                .unwrap_or(1.0)
    }

    #[cfg(test)]
    fn with_directional_albedo_lut_for_tests(
        mut self,
        lut: Arc<DielectricGgxDirectionalAlbedoLut>,
    ) -> Self {
        self.dielectric_ggx_directional_albedo_lut = Some(lut);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SimplePbrParams {
    metallic: f32,
    roughness: f32,
    eta: f32,
    base_color: Vec3,
    anisotropy: f32,
    alpha_x: f32,
    alpha_y: f32,
}

fn alpha_xy_from_roughness(roughness: f32, anisotropy: f32) -> (f32, f32) {
    let roughness = roughness.clamp(0.0, 1.0);
    let anisotropy = anisotropy.clamp(-1.0, 1.0);
    let alpha = roughness * roughness;
    let aspect = (1.0 - 0.9 * anisotropy.abs()).sqrt();
    let (alpha_x, alpha_y) = if anisotropy >= 0.0 {
        (alpha / aspect, alpha * aspect)
    } else {
        (alpha * aspect, alpha / aspect)
    };

    (alpha_x.clamp(MIN_ALPHA, 1.0), alpha_y.clamp(MIN_ALPHA, 1.0))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use glam::{Vec2, Vec3};

    use crate::{
        bsdf::{BsdfFlags, DielectricGgxDirectionalAlbedoLut},
        material::ShadingVertex,
        math::OrthonormalBasis,
        scene::{InstanceIndex, TriangleRef},
    };

    use super::SimplePbrMaterial;

    fn test_shading_vertex(wo: Vec3) -> ShadingVertex {
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

    fn test_material(metallic: f32) -> SimplePbrMaterial {
        SimplePbrMaterial::new(Vec3::new(0.8, 0.6, 0.4), metallic, 0.65, 1.5, 0.2)
            .with_directional_albedo_lut_for_tests(Arc::new(
                DielectricGgxDirectionalAlbedoLut::constant_for_tests(1.5, 0.25),
            ))
    }

    #[test]
    fn rough_diffuse_layer_sample_matches_eval_cos_over_pdf() {
        let material = test_material(0.0);
        let vtx = test_shading_vertex(Vec3::new(0.2, -0.1, 0.9746794).normalize());

        let sample = material
            .sample_impl(&vtx, 0.9, 0.9, Vec2::new(0.37, 0.82))
            .expect("expected a diffuse layer sample");
        let f = material.eval(&vtx, sample.wi);
        let wi_local = vtx.frame.world_to_local(sample.wi).normalize_or_zero();
        let expected = f * (wi_local.z.max(0.0) / sample.pdf);

        assert_eq!(sample.flags, BsdfFlags::DIFFUSE | BsdfFlags::REFLECTION);
        assert!(sample.weight.abs_diff_eq(expected, 1.0e-5));
    }

    #[test]
    fn metallic_one_samples_only_metal_component() {
        let material = test_material(1.0);
        let vtx = test_shading_vertex(Vec3::new(0.2, -0.1, 0.9746794).normalize());

        let sample = material
            .sample_impl(&vtx, 0.9, 0.9, Vec2::new(0.25, 0.75))
            .expect("expected a metal sample");

        assert!(sample.flags.contains(BsdfFlags::REFLECTION));
        assert!(
            sample.flags.contains(BsdfFlags::GLOSSY) || sample.flags.contains(BsdfFlags::DELTA)
        );
    }

    #[test]
    fn install_directional_albedo_lut_uses_shared_lut() {
        let lut = Arc::new(DielectricGgxDirectionalAlbedoLut::constant_for_tests(
            1.5, 0.25,
        ));
        let mut first = SimplePbrMaterial::new(Vec3::ONE, 0.0, 0.5, 1.5, 0.0);
        let mut second = SimplePbrMaterial::new(Vec3::ONE, 0.0, 0.5, 1.5, 0.0);

        first.install_dielectric_ggx_directional_albedo_lut(Arc::clone(&lut));
        second.install_dielectric_ggx_directional_albedo_lut(lut);

        assert!(Arc::ptr_eq(
            first
                .dielectric_ggx_directional_albedo_lut
                .as_ref()
                .unwrap(),
            second
                .dielectric_ggx_directional_albedo_lut
                .as_ref()
                .unwrap()
        ));
    }
}
