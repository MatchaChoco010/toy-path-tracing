// Hierarchical light sampling tree based on
// [Tokuyoshi, Ikeda, Kulkarni, Harada 2024 "Hierarchical Light Sampling with
//  Accurate Spherical Gaussian Lighting"] (SIGGRAPH Asia 2024).
//
// The tree clusters area, point and spot lights into a binary hierarchy. Each
// internal node carries enough information to evaluate the SG-based importance
// approximation
//
//      I ~  ∫ L(x, o) f(i, o) |o.n| do
//        ~  W * ∫ g(o; xi, kappa) f(i, o) |o.n| do
//
// at any shading point x, then `sample` traverses the tree by stochastically
// descending children with probability proportional to importance, while
// `pdf_for_leaf` reverses that process for MIS.
//
// Build (`builder.rs`): top-down binned SAOH (Conty & Kulla 2018) plus
// bottom-up aggregation of (mu, sigma_s^2, nu_bar, Phi) using Eqs. 2, 4, 5 of
// the paper. Parallelized with `rayon::join` for sub-trees that exceed
// `PARALLEL_BUILD_THRESHOLD` primitives.
//
// Traversal (`traversal.rs`): hierarchical sample warping
// [McCool & Harwood 1997; Clarberg et al. 2005].
//
// Importance (`importance.rs`): per-material dispatch. Diffuse lobes use the
// new SG-cosine product integral [Tokuyoshi 2024 Sec. 4]; glossy lobes use NDF
// filtering [Sec. 5] with the `J` Jacobian at h = n; dielectric BTDFs reuse
// the glossy formulation around the perfect refraction direction (Proxy A,
// derived from the supplementary's refraction Jacobian).

mod builder;
mod importance;
pub mod lobe;
mod traversal;

use glam::Vec3;

use crate::{
    light::{PointLightIndex, SpotLightIndex},
    scene::Bounds,
    scene::TriangleRef,
};

pub use builder::build_light_tree;
pub use importance::{LightTreeQuery, build_query, evaluate_node_importance};
pub use lobe::{
    BtdfLobePrecompute, DiffuseLobePrecompute, GlossyLobePrecompute, LightTreePrecompute,
    btdf_importance, diffuse_importance, glossy_importance, make_btdf_lobe, make_glossy_lobe,
    merge_glossy_roughness,
};
pub use traversal::{LightTreeSample, pdf_for_leaf, pdf_for_leaf_kind, sample_light_tree};

/// What a leaf in the light tree refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LightTreeLeafKind {
    Triangle(TriangleRef),
    Point(PointLightIndex),
    Spot(SpotLightIndex),
}

/// One node of the SG light tree. Internal nodes have `leaf == None`,
/// leaves have `leaf == Some(...)` and `left == right == INVALID_NODE`.
///
/// The data needed *per shading point* (W, xi, kappa) is computed on demand
/// from `(mu, sigma_s2, nu, lambda, flux, radius)` via `importance::sg_light_for_node`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightTreeNode {
    pub flux: f32,
    pub mu: Vec3,
    pub sigma_s2: f32,
    /// Normalised vMF axis. Zero-vector for empty / uniform clusters.
    pub nu: Vec3,
    pub lambda: f32,
    pub aabb: Bounds,
    /// Bounding-sphere radius about `mu`. Used in the conservative-variance
    /// blend (Eq. 6 of the supplementary).
    pub radius: f32,
    /// SAOH orientation cone — kept around so that bottom-up updates are easy
    /// to audit. Not consulted at traversal time.
    pub cone_axis: Vec3,
    pub cone_theta_o: f32,
    pub cone_theta_e: f32,
    pub leaf: Option<LightTreeLeafKind>,
    pub left: u32,
    pub right: u32,
    pub parent: u32,
}

pub const INVALID_NODE: u32 = u32::MAX;

impl LightTreeNode {
    pub fn is_leaf(&self) -> bool {
        self.leaf.is_some()
    }
}

/// Linearised tree.
///
/// `nodes[root]` is the root. `triangle_leaves` / `point_leaves` /
/// `spot_leaves` are reverse lookups used to compute the PDF of a leaf
/// reached via BSDF sampling (the MIS path).
#[derive(Debug, Clone, PartialEq)]
pub struct LightTree {
    pub nodes: Vec<LightTreeNode>,
    pub root: u32,
    pub triangle_leaves: std::collections::HashMap<TriangleRef, u32>,
    pub point_leaves: Vec<u32>,
    pub spot_leaves: Vec<u32>,
}

impl LightTree {
    pub fn root_flux(&self) -> f32 {
        if self.nodes.is_empty() {
            0.0
        } else {
            self.nodes[self.root as usize].flux
        }
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}
