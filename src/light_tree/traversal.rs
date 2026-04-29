// Hierarchical sample warping over the SG light tree.
//
// `sample_light_tree(query, u)` descends from the root by stochastically
// picking children with probability proportional to the SG-based importance,
// re-normalising the carried sample `u` after each step (McCool & Harwood
// 1997 / Clarberg et al. 2005). One uniform 1-D random sample is enough to
// descend any depth.
//
// `pdf_for_leaf(query, leaf_node_index)` walks the same tree from the leaf
// up to the root and reproduces the importance ratio at every step so that
// MIS can compute the BSDF-side reverse PDF.

use super::importance::{LightTreeQuery, evaluate_node_importance};
use super::{INVALID_NODE, LightTree, LightTreeLeafKind, LightTreeNode};

#[derive(Debug, Clone, Copy)]
pub struct LightTreeSample {
    pub leaf: LightTreeLeafKind,
    pub leaf_node: u32,
    /// Probability mass of *selecting this leaf among the tree's leaves*. The
    /// caller multiplies this with whatever 2-D PDF the leaf produces (e.g.
    /// triangle solid-angle PDF).
    pub pmf: f32,
}

pub fn sample_light_tree(
    tree: &LightTree,
    query: &LightTreeQuery,
    mut u: f32,
) -> Option<LightTreeSample> {
    if tree.is_empty() {
        return None;
    }
    u = u.clamp(0.0, 1.0 - f32::EPSILON);
    let mut node_idx = tree.root;
    let mut pmf = 1.0_f32;

    loop {
        let node = &tree.nodes[node_idx as usize];
        if let Some(leaf) = node.leaf {
            return Some(LightTreeSample {
                leaf,
                leaf_node: node_idx,
                pmf,
            });
        }
        let left_idx = node.left;
        let right_idx = node.right;
        let left = &tree.nodes[left_idx as usize];
        let right = &tree.nodes[right_idx as usize];

        let il = evaluate_node_importance(query, left);
        let ir = evaluate_node_importance(query, right);
        let total = il + ir;
        if total <= 0.0 {
            // Both children look uniformly dark from the shading point. Fall
            // back to flux-proportional descent so we can still reach a leaf
            // (paper sup. discusses this implicitly: importance must be
            // strictly positive when the integrand is). Using node flux as
            // the safety net mirrors Conty & Kulla 2018's behaviour.
            let lf = left.flux.max(0.0);
            let rf = right.flux.max(0.0);
            let total_flux = lf + rf;
            if total_flux <= 0.0 {
                return None;
            }
            let p_left = lf / total_flux;
            if u < p_left {
                u /= p_left.max(f32::MIN_POSITIVE);
                pmf *= p_left;
                node_idx = left_idx;
            } else {
                u = (u - p_left) / (1.0 - p_left).max(f32::MIN_POSITIVE);
                pmf *= 1.0 - p_left;
                node_idx = right_idx;
            }
            continue;
        }

        let p_left = il / total;
        if u < p_left {
            u /= p_left.max(f32::MIN_POSITIVE);
            pmf *= p_left;
            node_idx = left_idx;
        } else {
            u = (u - p_left) / (1.0 - p_left).max(f32::MIN_POSITIVE);
            pmf *= 1.0 - p_left;
            node_idx = right_idx;
        }
        u = u.clamp(0.0, 1.0 - f32::EPSILON);
    }
}

/// Walk from the leaf up to the root, multiplying the per-level probability
/// of picking the side that contains the leaf. Returns the leaf-selection
/// PMF that matches `sample_light_tree`.
pub fn pdf_for_leaf(tree: &LightTree, query: &LightTreeQuery, leaf_node: u32) -> f32 {
    if tree.is_empty() || leaf_node == INVALID_NODE {
        return 0.0;
    }
    if !tree.nodes[leaf_node as usize].is_leaf() {
        return 0.0;
    }

    let mut pmf = 1.0_f32;
    let mut child = leaf_node;
    let mut parent = tree.nodes[leaf_node as usize].parent;
    while parent != INVALID_NODE {
        let pnode = &tree.nodes[parent as usize];
        let left = &tree.nodes[pnode.left as usize];
        let right = &tree.nodes[pnode.right as usize];
        let il = evaluate_node_importance(query, left);
        let ir = evaluate_node_importance(query, right);
        let total = il + ir;
        if total <= 0.0 {
            // Match the safety-net branch in `sample_light_tree`.
            let lf = left.flux.max(0.0);
            let rf = right.flux.max(0.0);
            let total_flux = lf + rf;
            if total_flux <= 0.0 {
                return 0.0;
            }
            let p_left = lf / total_flux;
            let p = if child == pnode.left {
                p_left
            } else {
                1.0 - p_left
            };
            pmf *= p;
        } else {
            let p_left = il / total;
            let p = if child == pnode.left {
                p_left
            } else {
                1.0 - p_left
            };
            pmf *= p;
        }
        child = parent;
        parent = pnode.parent;
    }
    pmf
}

/// Convenience wrapper. Resolves the leaf's node index from the tree's
/// per-leaf reverse maps.
pub fn pdf_for_leaf_kind(tree: &LightTree, query: &LightTreeQuery, leaf: LightTreeLeafKind) -> f32 {
    let node_idx = match leaf {
        LightTreeLeafKind::Triangle(tri) => {
            *tree.triangle_leaves.get(&tri).unwrap_or(&INVALID_NODE)
        }
        LightTreeLeafKind::Point(crate::light::PointLightIndex(i)) => {
            tree.point_leaves.get(i).copied().unwrap_or(INVALID_NODE)
        }
        LightTreeLeafKind::Spot(crate::light::SpotLightIndex(i)) => {
            tree.spot_leaves.get(i).copied().unwrap_or(INVALID_NODE)
        }
    };
    pdf_for_leaf(tree, query, node_idx)
}

#[allow(unused)]
fn debug_node(node: &LightTreeNode) -> String {
    format!(
        "leaf={:?} flux={} mu={:?} sigma={} radius={}",
        node.leaf, node.flux, node.mu, node.sigma_s2, node.radius
    )
}
