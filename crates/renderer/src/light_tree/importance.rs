// Thin dispatcher between traversal and the Material trait surface.
//
// `build_query` and `evaluate_node_importance` here own *no* SG math --
// they bundle the shading point with a `&Material` and forward to the
// material's own `light_tree_precompute` / `light_tree_importance` methods.
// All the actual SG×lobe arithmetic lives in `lobe.rs` and is invoked from
// the per-material implementations.
//
// This keeps the encapsulation boundary aligned with the existing Material
// API (`eval`, `pdf`, `sample`, ...): every per-shading-point question is
// answered by the material itself, not by a switch in this file.

use crate::material::{Material, ShadingVertex};

use super::{
    LightTreeNode,
    lobe::{LightTreePrecompute, sg_light_for_node},
};

#[derive(Debug, Clone, Copy)]
pub struct LightTreeQuery<'a> {
    pub material: &'a Material,
    pub precompute: LightTreePrecompute,
}

pub fn build_query<'a>(
    vtx: &ShadingVertex,
    material: &'a Material,
    mtlx_scratch: &crate::material::MtlxScratch,
) -> Option<LightTreeQuery<'a>> {
    let precompute = material.light_tree_precompute(vtx, mtlx_scratch)?;
    Some(LightTreeQuery {
        material,
        precompute,
    })
}

/// Evaluate the SG-based importance of `node` from the standpoint of
/// `query`. Used by the stochastic descent and by the reverse PDF lookup
/// for MIS.
pub fn evaluate_node_importance(query: &LightTreeQuery, node: &LightTreeNode) -> f32 {
    let Some((w, lobe)) = sg_light_for_node(query.precompute.p, query.precompute.n, node) else {
        return 0.0;
    };
    query
        .material
        .light_tree_importance(&query.precompute, w, &lobe)
}
