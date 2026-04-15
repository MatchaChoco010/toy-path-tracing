use glam::{Vec2, Vec3};

use crate::{
    material::ShadingVertex,
    scene::{Scene, TriangleRef},
};

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
}

impl LightKind {
    pub fn light_type(self) -> LightType {
        match self {
            Self::Area => LightType::Area,
            Self::Infinite => LightType::Infinite,
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
        LightKind::Area => sample_area_light(scene, ctx, u_aux, us),
        LightKind::Infinite => sample_infinite_light(scene, us),
    }
}

fn sample_area_light(
    scene: &Scene,
    ctx: &LightSampleContext,
    u_aux: f32,
    us: Vec2,
) -> Option<LightLiSample> {
    let point = scene.sample_area_light_point(u_aux, us)?;
    if point.pdf_area <= 0.0 {
        return None;
    }

    let to_light = point.p - ctx.p;
    let distance_squared = to_light.length_squared();
    if distance_squared <= 0.0 {
        return None;
    }
    let distance = distance_squared.sqrt();
    let wi = to_light / distance;

    let light_material = scene.instance_material(point.triangle.instance_index);
    if !light_material.may_emit() {
        return None;
    }

    let lvtx = scene.shading_vertex_from_triangle_sample(point.triangle, point.barycentric, wi);
    let le = light_material.le(&lvtx)?;

    let cos_light = lvtx.ng.dot(-wi).max(0.0);
    if cos_light <= 0.0 {
        return None;
    }

    let pdf_solid_angle = point.pdf_area * distance_squared / cos_light;

    Some(LightLiSample {
        radiance: le,
        wi,
        pdf: pdf_solid_angle,
        distance,
        light_type: LightType::Area,
        target_triangle: Some(point.triangle),
    })
}

fn sample_infinite_light(scene: &Scene, us: Vec2) -> Option<LightLiSample> {
    let env = scene.environment_light.as_ref()?;
    let sample = env.sample(us)?;
    if sample.pdf <= 0.0 {
        return None;
    }

    Some(LightLiSample {
        radiance: sample.radiance,
        wi: sample.direction,
        pdf: sample.pdf,
        distance: f32::INFINITY,
        light_type: LightType::Infinite,
        target_triangle: None,
    })
}

pub fn area_light_pdf_li(scene: &Scene, vtx: &ShadingVertex, lvtx: &ShadingVertex) -> f32 {
    scene.area_light_pdf_solid_angle(vtx, lvtx).unwrap_or(0.0)
}

pub fn infinite_light_pdf_li(scene: &Scene, direction: Vec3) -> f32 {
    scene
        .environment_light
        .as_ref()
        .map(|env| env.pdf(direction))
        .unwrap_or(0.0)
}

pub fn infinite_light_le(scene: &Scene, direction: Vec3) -> Vec3 {
    scene
        .environment_light
        .as_ref()
        .map(|env| env.radiance(direction))
        .unwrap_or(Vec3::ZERO)
}

#[cfg(test)]
mod tests {
    use glam::{Vec2, Vec3};

    use super::{
        LightKind, LightSampleContext, LightSampler, LightType, infinite_light_le,
        infinite_light_pdf_li, sample_light_li,
    };
    use crate::{
        environment_light::EnvironmentLight,
        material::{EmissiveMaterial, Material, NormalizedLambertMaterial},
        mesh::{Mesh, Vertex},
        scene::Scene,
    };

    fn unit_mesh(z: f32) -> Mesh {
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

    fn uniform_environment(radiance: f32) -> EnvironmentLight {
        let pixels = vec![Vec3::splat(radiance); 32 * 16];
        EnvironmentLight::from_pixels(32, 16, pixels, 1.0)
    }

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
    fn empty_sampler_returns_no_sample() {
        let sampler = LightSampler::new();
        assert!(sampler.is_empty());
        assert!(sampler.sample(0.5).is_none());
        assert_eq!(sampler.pmf(LightKind::Area), 0.0);
    }

    #[test]
    fn uniform_sampler_across_two_kinds_returns_equal_pmf() {
        let sampler =
            LightSampler::from_weighted_kinds(&[(LightKind::Area, 1.0), (LightKind::Infinite, 1.0)]);

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
        let sampler =
            LightSampler::from_weighted_kinds(&[(LightKind::Area, 9.0), (LightKind::Infinite, 1.0)]);
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
    fn sample_light_li_for_area_matches_solid_angle_pdf() {
        let mut scene = Scene::new();
        let floor_mesh = scene.add_mesh(unit_mesh(0.0));
        let light_mesh = scene.add_mesh(unit_mesh(1.0));
        let floor_material = scene.add_material(Material::NormalizedLambert(
            NormalizedLambertMaterial::new(Vec3::splat(0.8)),
        ));
        let light_material =
            scene.add_material(Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 10.0)));
        scene.add_instance(floor_mesh, floor_material, glam::Mat4::IDENTITY);
        scene.add_instance(light_mesh, light_material, glam::Mat4::IDENTITY);
        scene.build_bvh();

        let ctx = LightSampleContext {
            p: Vec3::new(0.25, 0.25, 0.0),
            ng: Vec3::Z,
            ns: Vec3::Z,
        };

        let li = sample_light_li(
            &scene,
            LightKind::Area,
            &ctx,
            0.5,
            Vec2::new(0.25, 0.5),
        )
        .expect("expected a sample");

        assert_eq!(li.light_type, LightType::Area);
        assert!(li.target_triangle.is_some());
        assert!((li.pdf - 2.0).abs() < 1.0e-4);
        assert!(li.radiance.abs_diff_eq(Vec3::splat(10.0), 1.0e-5));
        assert!((li.distance - 1.0).abs() < 1.0e-5);
        assert!(li.wi.abs_diff_eq(Vec3::Z, 1.0e-5));
    }

    #[test]
    fn sample_light_li_for_infinite_returns_environment_sample() {
        let mut scene = Scene::new();
        scene.set_environment_light(uniform_environment(1.0));

        let ctx = LightSampleContext {
            p: Vec3::ZERO,
            ng: Vec3::Z,
            ns: Vec3::Z,
        };

        let li = sample_light_li(
            &scene,
            LightKind::Infinite,
            &ctx,
            0.0,
            Vec2::new(0.3, 0.6),
        )
        .expect("expected a sample");

        assert_eq!(li.light_type, LightType::Infinite);
        assert!(li.target_triangle.is_none());
        assert!(li.distance.is_infinite());
        assert!((li.wi.length() - 1.0).abs() < 1.0e-5);
        assert!(li.pdf > 0.0);
    }

    #[test]
    fn infinite_light_helpers_report_zero_when_missing() {
        let scene = Scene::new();
        assert_eq!(infinite_light_le(&scene, Vec3::Z), Vec3::ZERO);
        assert_eq!(infinite_light_pdf_li(&scene, Vec3::Z), 0.0);
    }
}
