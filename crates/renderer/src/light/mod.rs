// Top-level light selection.
//
// Light kinds are partitioned into three top-level categories:
//   * `LightCategory::Tree` -- the SG light tree from `crate::light_tree`,
//     which holds emissive triangles, point lights, and spot lights.
//   * `LightCategory::Environment` -- the IBL environment, treated as one
//     entry whose internal hierarchical distribution is unchanged.
//   * `LightCategory::Directional(i)` -- one entry per directional light, kept
//     out of the tree because directional lights have no spatial mean and
//     are best sampled with a delta lookup.
//
// `LightSampler` is a flat CDF over those categories. Within `Tree` we use
// the SG light tree's stochastic descent to pick a leaf; within `Environment`
// we forward to the existing mip-pyramid sampler; for `Directional(i)` the
// sample is deterministic.
//
// `selection_pmf` carried back on every sample multiplies the top-level
// category pmf and (for `Tree`) the per-leaf pmf coming out of the
// hierarchical descent. The continuous part of the density (solid-angle PDF
// for area lights, env PDF for the environment, 1.0 for delta lights) lives
// on `LightLiSample::pdf`. NEE / MIS just multiplies them.

mod area;
mod directional;
mod environment;
mod point;
mod spot;

use glam::{Vec2, Vec3};

use crate::{
    light_tree::{
        LightTreeLeafKind, LightTreeQuery, build_query, pdf_for_leaf_kind, sample_light_tree,
    },
    material::{Material, ShadingVertex},
    math::sg,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LightCategory {
    Tree,
    Environment,
    Directional(usize),
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct LightSampler {
    entries: Vec<LightEntry>,
    cdf: Vec<f32>,
    total_weight: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct LightEntry {
    category: LightCategory,
    weight: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampledCategory {
    pub category: LightCategory,
    pub pmf: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampledLight {
    pub category: LightCategory,
    pub leaf: Option<LightTreeLeafKind>,
    pub sample: LightLiSample,
    /// Discrete PMF of selecting this exact leaf (top-level * tree-leaf
    /// internal pmf for `Tree`; top-level only for env/directional).
    pub selection_pmf: f32,
}

impl LightSampler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn build_from_scene(scene: &Scene) -> Self {
        let mut entries: Vec<LightEntry> = Vec::new();

        let tree_weight = scene
            .light_tree
            .as_ref()
            .map(|t| t.root_flux().max(0.0))
            .unwrap_or(0.0);
        if tree_weight > 0.0 {
            entries.push(LightEntry {
                category: LightCategory::Tree,
                weight: tree_weight,
            });
        }

        if let Some(env) = scene.environment_light.as_ref() {
            let w = env.total_power().max(0.0);
            if w > 0.0 {
                entries.push(LightEntry {
                    category: LightCategory::Environment,
                    weight: w,
                });
            }
        }

        for (i, dir) in scene.directional_lights.iter().enumerate() {
            let w = (sg::luminance(dir.color) * dir.intensity).max(0.0);
            if w > 0.0 {
                entries.push(LightEntry {
                    category: LightCategory::Directional(i),
                    weight: w,
                });
            }
        }

        let mut cdf = Vec::with_capacity(entries.len() + 1);
        cdf.push(0.0_f32);
        let mut total = 0.0_f32;
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

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty() || self.total_weight <= 0.0
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn sample_category(&self, u: f32) -> Option<SampledCategory> {
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
        Some(SampledCategory {
            category: entry.category,
            pmf: entry.weight / self.total_weight,
        })
    }

    pub fn category_pmf(&self, category: LightCategory) -> f32 {
        if self.total_weight <= 0.0 {
            return 0.0;
        }
        self.entries
            .iter()
            .find(|e| e.category == category)
            .map(|e| e.weight / self.total_weight)
            .unwrap_or(0.0)
    }

    pub fn contains(&self, category: LightCategory) -> bool {
        self.entries.iter().any(|e| e.category == category)
    }

    pub fn categories(&self) -> impl Iterator<Item = LightCategory> + '_ {
        self.entries.iter().map(|e| e.category)
    }
}

/// Top-level NEE entry point. Picks a category, then samples a leaf inside.
///
/// `tree_query` must be present (and built via `light_tree::build_query`)
/// when the surface has any non-delta lobe; if the query is `None`, the
/// `Tree` category is skipped (we still try environment / directional).
///
/// Caller supplies four uniform samples:
///   * `u_root`  — top-level category selection
///   * `u_tree`  — hierarchical descent (only used by `Tree`; otherwise dead)
///   * `u_aux`   — area-light triangle selection (only used by `Tree::Triangle`)
///   * `us`      — 2-D barycentric / environment uv
///
/// Returns `None` if no light is selectable (e.g. empty scene or all
/// importances zero from the shading point's perspective).
pub fn sample_light(
    scene: &Scene,
    ctx: &LightSampleContext,
    tree_query: Option<&LightTreeQuery>,
    u_root: f32,
    u_tree: f32,
    u_aux: f32,
    us: Vec2,
    mtlx_scratch: &mut crate::material::MtlxScratch,
) -> Option<SampledLight> {
    sample_light_inner(
        scene,
        ctx,
        tree_query,
        u_root,
        u_tree,
        u_aux,
        us,
        false,
        mtlx_scratch,
    )
}

pub fn sample_light_mis_compensated(
    scene: &Scene,
    ctx: &LightSampleContext,
    tree_query: Option<&LightTreeQuery>,
    u_root: f32,
    u_tree: f32,
    u_aux: f32,
    us: Vec2,
    mtlx_scratch: &mut crate::material::MtlxScratch,
) -> Option<SampledLight> {
    sample_light_inner(
        scene,
        ctx,
        tree_query,
        u_root,
        u_tree,
        u_aux,
        us,
        true,
        mtlx_scratch,
    )
}

pub fn sample_light_mis_compensated_lazy(
    scene: &Scene,
    ctx: &LightSampleContext,
    tree_vtx: &ShadingVertex,
    tree_material: &Material,
    u_root: f32,
    u_tree: f32,
    u_aux: f32,
    us: Vec2,
    mtlx_scratch: &mut crate::material::MtlxScratch,
) -> Option<SampledLight> {
    let category = scene.light_sampler.sample_category(u_root)?;
    match category.category {
        LightCategory::Tree => {
            let tree = scene.light_tree.as_ref()?;
            let query = build_query(tree_vtx, tree_material, mtlx_scratch)?;
            let leaf = sample_light_tree(tree, &query, u_tree)?;
            let li = sample_li_for_leaf(scene, leaf.leaf, ctx, u_aux, us, mtlx_scratch)?;
            Some(SampledLight {
                category: category.category,
                leaf: Some(leaf.leaf),
                sample: li,
                selection_pmf: category.pmf * leaf.pmf,
            })
        }
        LightCategory::Environment => {
            let li = environment::sample_li_mis_compensated(scene, us)?;
            Some(SampledLight {
                category: category.category,
                leaf: None,
                sample: li,
                selection_pmf: category.pmf,
            })
        }
        LightCategory::Directional(i) => {
            let li = directional::sample_li(&scene.directional_lights[i])?;
            Some(SampledLight {
                category: category.category,
                leaf: None,
                sample: li,
                selection_pmf: category.pmf,
            })
        }
    }
}

fn sample_light_inner(
    scene: &Scene,
    ctx: &LightSampleContext,
    tree_query: Option<&LightTreeQuery>,
    u_root: f32,
    u_tree: f32,
    u_aux: f32,
    us: Vec2,
    mis_compensated: bool,
    mtlx_scratch: &mut crate::material::MtlxScratch,
) -> Option<SampledLight> {
    let category = scene.light_sampler.sample_category(u_root)?;
    match category.category {
        LightCategory::Tree => {
            let tree = scene.light_tree.as_ref()?;
            let query = tree_query?;
            let leaf = sample_light_tree(tree, query, u_tree)?;
            let li = sample_li_for_leaf(scene, leaf.leaf, ctx, u_aux, us, mtlx_scratch)?;
            Some(SampledLight {
                category: category.category,
                leaf: Some(leaf.leaf),
                sample: li,
                selection_pmf: category.pmf * leaf.pmf,
            })
        }
        LightCategory::Environment => {
            let li = if mis_compensated {
                environment::sample_li_mis_compensated(scene, us)?
            } else {
                environment::sample_li(scene, us)?
            };
            Some(SampledLight {
                category: category.category,
                leaf: None,
                sample: li,
                selection_pmf: category.pmf,
            })
        }
        LightCategory::Directional(i) => {
            let li = directional::sample_li(&scene.directional_lights[i])?;
            Some(SampledLight {
                category: category.category,
                leaf: None,
                sample: li,
                selection_pmf: category.pmf,
            })
        }
    }
}

fn sample_li_for_leaf(
    scene: &Scene,
    leaf: LightTreeLeafKind,
    ctx: &LightSampleContext,
    _u_aux: f32,
    us: Vec2,
    mtlx_scratch: &mut crate::material::MtlxScratch,
) -> Option<LightLiSample> {
    match leaf {
        LightTreeLeafKind::Triangle(tri) => {
            area::sample_li_for_triangle(scene, tri, ctx, us, mtlx_scratch)
        }
        LightTreeLeafKind::Point(PointLightIndex(i)) => {
            point::sample_li(&scene.point_lights[i], ctx)
        }
        LightTreeLeafKind::Spot(SpotLightIndex(i)) => spot::sample_li(&scene.spot_lights[i], ctx),
    }
}

/// PDF of selecting a leaf via NEE that we *would have produced* at this
/// shading point. Used by MIS for the BSDF-sampled-light path.
///
/// `light_type` is the leaf's logical light kind, needed to figure out the
/// continuous PDF (solid-angle for area, env PDF for environment).
pub fn pdf_for_triangle_hit(
    scene: &Scene,
    tree_query: Option<&LightTreeQuery>,
    vtx: &ShadingVertex,
    lvtx: &ShadingVertex,
) -> f32 {
    let Some(tree) = scene.light_tree.as_ref() else {
        return 0.0;
    };
    let Some(query) = tree_query else {
        return 0.0;
    };
    let leaf_pmf = pdf_for_leaf_kind(tree, query, LightTreeLeafKind::Triangle(lvtx.triangle));
    if leaf_pmf <= 0.0 {
        return 0.0;
    }
    let cat_pmf = scene.light_sampler.category_pmf(LightCategory::Tree);
    let Some(area_pdf_solid_angle) = scene.area_light_pdf_solid_angle(vtx, lvtx) else {
        return 0.0;
    };
    cat_pmf * leaf_pmf * area_pdf_solid_angle
}

pub fn pdf_for_environment_hit(scene: &Scene, direction: Vec3, mis_compensated: bool) -> f32 {
    let cat_pmf = scene.light_sampler.category_pmf(LightCategory::Environment);
    if cat_pmf <= 0.0 {
        return 0.0;
    }
    let env_pdf = if mis_compensated {
        infinite_light_pdf_li_mis_compensated(scene, direction)
    } else {
        infinite_light_pdf_li(scene, direction)
    };
    cat_pmf * env_pdf
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
    use super::*;
    use crate::material::{EmissiveMaterial, Material};
    use glam::Mat4;

    use super::test_helpers::{uniform_environment, unit_mesh};

    #[test]
    fn light_type_classifications() {
        assert!(LightType::DeltaPosition.is_delta());
        assert!(LightType::DeltaDirection.is_delta());
        assert!(LightType::DeltaDirection.is_infinite());
        assert!(LightType::Infinite.is_infinite());
        assert!(!LightType::Area.is_delta());
    }

    #[test]
    fn empty_sampler() {
        let sampler = LightSampler::new();
        assert!(sampler.is_empty());
        assert!(sampler.sample_category(0.5).is_none());
        assert_eq!(sampler.category_pmf(LightCategory::Tree), 0.0);
    }

    #[test]
    fn build_from_scene_with_only_environment() {
        let mut scene = Scene::new();
        scene.set_environment_light(uniform_environment(1.0));
        scene.build_light_tree();
        let sampler = LightSampler::build_from_scene(&scene);
        assert_eq!(sampler.len(), 1);
        assert!(sampler.contains(LightCategory::Environment));
    }

    #[test]
    fn build_from_scene_with_tree_and_env() {
        let mut scene = Scene::new();
        scene.set_environment_light(uniform_environment(1.0));
        let mesh = scene.add_mesh(unit_mesh(1.0));
        let mat = scene.add_material(Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 1.0)));
        scene.add_instance(mesh, mat, Mat4::IDENTITY);
        scene.build_light_tree();
        let sampler = LightSampler::build_from_scene(&scene);
        assert!(sampler.contains(LightCategory::Tree));
        assert!(sampler.contains(LightCategory::Environment));
    }

    #[test]
    fn directional_lights_each_get_their_own_entry() {
        let mut scene = Scene::new();
        scene.add_directional_light(DirectionalLight::new(Vec3::NEG_Z, Vec3::ONE, 1.0));
        scene.add_directional_light(DirectionalLight::new(Vec3::NEG_X, Vec3::ONE, 2.0));
        scene.build_light_tree();
        let sampler = LightSampler::build_from_scene(&scene);
        assert_eq!(sampler.len(), 2);
        assert!(sampler.contains(LightCategory::Directional(0)));
        assert!(sampler.contains(LightCategory::Directional(1)));
        let p0 = sampler.category_pmf(LightCategory::Directional(0));
        let p1 = sampler.category_pmf(LightCategory::Directional(1));
        assert!(p1 > p0);
    }
}
