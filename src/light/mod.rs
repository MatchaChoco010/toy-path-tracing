mod area;
mod directional;
mod environment;
mod point;
mod spot;

use glam::{Vec2, Vec3};

use crate::{
    material::ShadingVertex,
    scene::{Scene, TriangleRef},
};

pub use area::area_light_pdf_li;
pub use directional::{DirectionalLight, DirectionalLightIndex};
pub use environment::{
    EnvironmentLight, EnvironmentLightSample, infinite_light_le, infinite_light_pdf_li,
    infinite_light_pdf_li_mis_compensated,
};
pub use point::{PointLight, PointLightIndex};
pub use spot::{SpotLight, SpotLightIndex};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightType {
    DeltaPosition,
    DeltaDirection,
    Area,
    Infinite,
}

impl LightType {
    pub fn is_delta(self) -> bool {
        matches!(self, Self::DeltaPosition | Self::DeltaDirection)
    }

    pub fn is_infinite(self) -> bool {
        matches!(self, Self::Infinite | Self::DeltaDirection)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LightKind {
    Area,
    Infinite,
    DeltaPoint(usize),
    DeltaDirectional(usize),
    DeltaSpot(usize),
}

impl LightKind {
    pub fn light_type(self) -> LightType {
        match self {
            Self::Area => LightType::Area,
            Self::Infinite => LightType::Infinite,
            Self::DeltaPoint(_) | Self::DeltaSpot(_) => LightType::DeltaPosition,
            Self::DeltaDirectional(_) => LightType::DeltaDirection,
        }
    }

    pub fn is_delta(self) -> bool {
        self.light_type().is_delta()
    }

    pub fn is_infinite(self) -> bool {
        self.light_type().is_infinite()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightSampleContext {
    pub p: Vec3,
    pub ng: Vec3,
    pub ns: Vec3,
}

impl LightSampleContext {
    pub fn from_vertex(vtx: &ShadingVertex) -> Self {
        Self {
            p: vtx.p,
            ng: vtx.ng,
            ns: vtx.ns,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightLiSample {
    pub radiance: Vec3,
    pub wi: Vec3,
    pub pdf: f32,
    pub distance: f32,
    pub light_type: LightType,
    pub target_triangle: Option<TriangleRef>,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct LightSampler {
    entries: Vec<LightEntry>,
    cdf: Vec<f32>,
    total_weight: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct LightEntry {
    kind: LightKind,
    weight: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampledLight {
    pub kind: LightKind,
    pub pmf: f32,
}

impl LightSampler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_weighted_kinds(entries: &[(LightKind, f32)]) -> Self {
        let entries: Vec<LightEntry> = entries
            .iter()
            .filter(|(_, w)| *w > 0.0)
            .map(|(k, w)| LightEntry {
                kind: *k,
                weight: *w,
            })
            .collect();

        let mut cdf = Vec::with_capacity(entries.len() + 1);
        cdf.push(0.0f32);
        let mut total = 0.0f32;
        for entry in &entries {
            total += entry.weight;
            cdf.push(total);
        }
        if total > 0.0 {
            let inv = 1.0 / total;
            for c in &mut cdf {
                *c *= inv;
            }
        }

        Self {
            entries,
            cdf,
            total_weight: total,
        }
    }

    pub fn build_from_scene(scene: &Scene) -> Self {
        let mut entries = Vec::new();
        if scene.area_light_weight_sum > 0.0 {
            entries.push((LightKind::Area, 1.0));
        }
        if scene.environment_light.is_some() {
            entries.push((LightKind::Infinite, 1.0));
        }
        for i in 0..scene.point_lights.len() {
            entries.push((LightKind::DeltaPoint(i), 1.0));
        }
        for i in 0..scene.directional_lights.len() {
            entries.push((LightKind::DeltaDirectional(i), 1.0));
        }
        for i in 0..scene.spot_lights.len() {
            entries.push((LightKind::DeltaSpot(i), 1.0));
        }
        Self::from_weighted_kinds(&entries)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty() || self.total_weight <= 0.0
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn sample(&self, u: f32) -> Option<SampledLight> {
        if self.is_empty() {
            return None;
        }
        let u = u.clamp(0.0, 1.0);
        let last = self.entries.len() - 1;
        let idx = self
            .cdf
            .partition_point(|&c| c <= u)
            .saturating_sub(1)
            .min(last);
        let entry = self.entries[idx];
        Some(SampledLight {
            kind: entry.kind,
            pmf: entry.weight / self.total_weight,
        })
    }

    pub fn pmf(&self, kind: LightKind) -> f32 {
        if self.total_weight <= 0.0 {
            return 0.0;
        }
        self.entries
            .iter()
            .find(|e| e.kind == kind)
            .map(|e| e.weight / self.total_weight)
            .unwrap_or(0.0)
    }

    pub fn contains(&self, kind: LightKind) -> bool {
        self.entries.iter().any(|e| e.kind == kind)
    }

    pub fn kinds(&self) -> impl Iterator<Item = LightKind> + '_ {
        self.entries.iter().map(|e| e.kind)
    }
}

pub fn sample_light_li(
    scene: &Scene,
    kind: LightKind,
    ctx: &LightSampleContext,
    u_aux: f32,
    us: Vec2,
) -> Option<LightLiSample> {
    match kind {
        LightKind::Area => area::sample_li(scene, ctx, u_aux, us),
        LightKind::Infinite => environment::sample_li(scene, us),
        LightKind::DeltaPoint(i) => point::sample_li(&scene.point_lights[i], ctx),
        LightKind::DeltaDirectional(i) => directional::sample_li(&scene.directional_lights[i]),
        LightKind::DeltaSpot(i) => spot::sample_li(&scene.spot_lights[i], ctx),
    }
}

pub fn sample_light_li_mis_compensated(
    scene: &Scene,
    kind: LightKind,
    ctx: &LightSampleContext,
    u_aux: f32,
    us: Vec2,
) -> Option<LightLiSample> {
    match kind {
        LightKind::Infinite => environment::sample_li_mis_compensated(scene, us),
        _ => sample_light_li(scene, kind, ctx, u_aux, us),
    }
}

#[cfg(test)]
mod test_helpers {
    use glam::{Vec2, Vec3};

    use super::EnvironmentLight;
    use crate::mesh::{Mesh, Vertex};

    pub fn unit_mesh(z: f32) -> Mesh {
        Mesh::new(
            vec![
                Vertex {
                    position: Vec3::new(0.0, 0.0, z),
                    normal: Vec3::Z,
                    uv: Vec2::ZERO,
                },
                Vertex {
                    position: Vec3::new(1.0, 0.0, z),
                    normal: Vec3::Z,
                    uv: Vec2::X,
                },
                Vertex {
                    position: Vec3::new(0.0, 1.0, z),
                    normal: Vec3::Z,
                    uv: Vec2::Y,
                },
            ],
            vec![0, 1, 2],
        )
    }

    pub fn uniform_environment(radiance: f32) -> EnvironmentLight {
        let pixels = vec![Vec3::splat(radiance); 32 * 16];
        EnvironmentLight::from_pixels(32, 16, pixels, 1.0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use glam::Vec3;

    use super::test_helpers::{uniform_environment, unit_mesh};
    use super::{DirectionalLight, LightKind, LightSampler, LightType, PointLight, SpotLight};
    use crate::{
        material::{EmissiveMaterial, Material},
        scene::Scene,
    };

    #[test]
    fn light_type_delta_and_infinite_classification() {
        assert!(LightType::DeltaPosition.is_delta());
        assert!(LightType::DeltaDirection.is_delta());
        assert!(!LightType::Area.is_delta());
        assert!(!LightType::Infinite.is_delta());

        assert!(!LightType::DeltaPosition.is_infinite());
        assert!(LightType::DeltaDirection.is_infinite());
        assert!(!LightType::Area.is_infinite());
        assert!(LightType::Infinite.is_infinite());
    }

    #[test]
    fn light_kind_maps_to_light_type() {
        assert_eq!(LightKind::Area.light_type(), LightType::Area);
        assert_eq!(LightKind::Infinite.light_type(), LightType::Infinite);
        assert!(!LightKind::Area.is_delta());
        assert!(LightKind::Infinite.is_infinite());
    }

    #[test]
    fn light_kind_classifies_delta_variants_correctly() {
        assert_eq!(
            LightKind::DeltaPoint(0).light_type(),
            LightType::DeltaPosition
        );
        assert_eq!(
            LightKind::DeltaSpot(2).light_type(),
            LightType::DeltaPosition
        );
        assert_eq!(
            LightKind::DeltaDirectional(1).light_type(),
            LightType::DeltaDirection
        );

        assert!(LightKind::DeltaPoint(0).is_delta());
        assert!(LightKind::DeltaSpot(0).is_delta());
        assert!(LightKind::DeltaDirectional(0).is_delta());
        assert!(!LightKind::DeltaPoint(0).is_infinite());
        assert!(!LightKind::DeltaSpot(0).is_infinite());
        assert!(LightKind::DeltaDirectional(0).is_infinite());
    }

    #[test]
    fn empty_sampler_returns_no_sample() {
        let sampler = LightSampler::new();
        assert!(sampler.is_empty());
        assert!(sampler.sample(0.5).is_none());
        assert_eq!(sampler.pmf(LightKind::Area), 0.0);
    }

    #[test]
    fn uniform_sampler_across_two_kinds_returns_equal_pmf() {
        let sampler = LightSampler::from_weighted_kinds(&[
            (LightKind::Area, 1.0),
            (LightKind::Infinite, 1.0),
        ]);

        assert_eq!(sampler.len(), 2);
        assert!((sampler.pmf(LightKind::Area) - 0.5).abs() < 1.0e-6);
        assert!((sampler.pmf(LightKind::Infinite) - 0.5).abs() < 1.0e-6);

        let first = sampler.sample(0.1).unwrap();
        let second = sampler.sample(0.9).unwrap();
        assert_ne!(first.kind, second.kind);
        assert!((first.pmf - 0.5).abs() < 1.0e-6);
        assert!((second.pmf - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn weighted_sampler_biases_toward_heavy_kind() {
        let sampler = LightSampler::from_weighted_kinds(&[
            (LightKind::Area, 9.0),
            (LightKind::Infinite, 1.0),
        ]);
        assert!((sampler.pmf(LightKind::Area) - 0.9).abs() < 1.0e-6);
        assert!((sampler.pmf(LightKind::Infinite) - 0.1).abs() < 1.0e-6);
    }

    #[test]
    fn build_from_scene_includes_only_present_lights() {
        let mut scene = Scene::new();
        let empty = LightSampler::build_from_scene(&scene);
        assert!(empty.is_empty());

        scene.set_environment_light(uniform_environment(1.0));
        let env_only = LightSampler::build_from_scene(&scene);
        assert_eq!(env_only.len(), 1);
        assert!(env_only.contains(LightKind::Infinite));
        assert!(!env_only.contains(LightKind::Area));

        let light_mesh = scene.add_mesh(unit_mesh(1.0));
        let light_material =
            scene.add_material(Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 1.0)));
        scene.add_instance(light_mesh, light_material, glam::Mat4::IDENTITY);

        let both = LightSampler::build_from_scene(&scene);
        assert_eq!(both.len(), 2);
        assert!(both.contains(LightKind::Area));
        assert!(both.contains(LightKind::Infinite));
    }

    #[test]
    fn build_from_scene_lists_each_delta_light_individually() {
        let mut scene = Scene::new();
        scene.add_point_light(PointLight::new(Vec3::X, Vec3::ONE, 1.0));
        scene.add_point_light(PointLight::new(Vec3::Y, Vec3::ONE, 1.0));
        scene.add_directional_light(DirectionalLight::new(Vec3::NEG_Z, Vec3::ONE, 1.0));
        scene.add_spot_light(SpotLight::new(
            Vec3::ZERO,
            Vec3::NEG_Z,
            Vec3::ONE,
            1.0,
            PI / 6.0,
            PI / 8.0,
        ));

        let sampler = LightSampler::build_from_scene(&scene);
        assert_eq!(sampler.len(), 4);
        assert!(sampler.contains(LightKind::DeltaPoint(0)));
        assert!(sampler.contains(LightKind::DeltaPoint(1)));
        assert!(sampler.contains(LightKind::DeltaDirectional(0)));
        assert!(sampler.contains(LightKind::DeltaSpot(0)));
        assert!((sampler.pmf(LightKind::DeltaPoint(0)) - 0.25).abs() < 1.0e-5);
        assert!((sampler.pmf(LightKind::DeltaDirectional(0)) - 0.25).abs() < 1.0e-5);
    }

    #[test]
    fn build_from_scene_mixes_area_environment_and_delta_lights() {
        let mut scene = Scene::new();
        scene.set_environment_light(uniform_environment(1.0));
        let light_mesh = scene.add_mesh(unit_mesh(1.0));
        let light_material =
            scene.add_material(Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 1.0)));
        scene.add_instance(light_mesh, light_material, glam::Mat4::IDENTITY);
        scene.add_point_light(PointLight::new(Vec3::Y, Vec3::splat(1.0), 2.0));

        let sampler = LightSampler::build_from_scene(&scene);
        assert_eq!(sampler.len(), 3);
        assert!(sampler.contains(LightKind::Area));
        assert!(sampler.contains(LightKind::Infinite));
        assert!(sampler.contains(LightKind::DeltaPoint(0)));
    }
}
