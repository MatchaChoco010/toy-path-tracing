// Binned SAOH light-tree builder.
//
// Top-down binary build with Conty & Kulla 2018's "surface area orientation
// heuristic" cost. Parallelised with `rayon::join` whenever a sub-tree has
// more leaves than `PARALLEL_BUILD_THRESHOLD`.
//
// The leaf set is a heterogeneous mix of:
//   * emissive triangle leaves (luminance-weighted Phi, vMF axis ~ 0.5 * n_geom),
//   * point-light leaves (sigma=0, lambda=0 — uniform direction),
//   * spot-light leaves (sigma=0, vMF fitted to the cone half-angle).
//
// All leaves share the same node layout so the traversal and importance code
// can ignore the kind. Shading-time importance dispatch is deferred to
// `importance.rs`, which is the only place that needs to know whether a node
// represents a triangle (lookup the material) versus a point/spot (no
// material interaction).

use std::collections::HashMap;
use std::f32::consts::PI;

use glam::Vec3;
use rayon::prelude::*;

use crate::{
    light::{PointLight, PointLightIndex, SpotLight, SpotLightIndex},
    material::Material,
    math::sg,
    scene::Bounds,
    scene::{AreaLightTriangle, Scene},
};

use super::{INVALID_NODE, LightTree, LightTreeLeafKind, LightTreeNode};

const NUM_BINS: usize = 12;
const TARGET_LEAF_SIZE: usize = 1;
const PARALLEL_BUILD_THRESHOLD: usize = 1024;

/// Per-leaf cluster parameters built ahead of the recursive split.
#[derive(Debug, Clone, Copy)]
struct LeafBuild {
    kind: LightTreeLeafKind,
    flux: f32,
    mu: Vec3,
    sigma_s2: f32,
    /// Un-normalised vMF axis (length encodes sharpness via Banerjee 2005).
    nu_bar: Vec3,
    aabb: Bounds,
    radius: f32,
    cone_axis: Vec3,
    cone_theta_o: f32,
    cone_theta_e: f32,
    /// Centroid used as the SAOH binning key.
    centroid: Vec3,
}

/// Build an SG light tree from the scene's emissive triangles, point and spot
/// lights. Returns `None` if there are no leaves.
pub fn build_light_tree(scene: &Scene) -> Option<LightTree> {
    let mut leaves: Vec<LeafBuild> = collect_triangle_leaves(scene);
    leaves.extend(collect_point_leaves(scene));
    leaves.extend(collect_spot_leaves(scene));
    leaves.retain(|l| l.flux > 0.0);

    if leaves.is_empty() {
        return None;
    }

    let mut nodes: Vec<LightTreeNode> = Vec::with_capacity(2 * leaves.len() - 1);
    let root = build_recursive(&mut leaves[..], &mut nodes, INVALID_NODE);

    fixup_parents(&mut nodes, root);

    let mut triangle_leaves = HashMap::new();
    let mut point_leaves = vec![INVALID_NODE; scene.point_lights.len()];
    let mut spot_leaves = vec![INVALID_NODE; scene.spot_lights.len()];
    for (idx, node) in nodes.iter().enumerate() {
        if let Some(kind) = node.leaf {
            match kind {
                LightTreeLeafKind::Triangle(tri) => {
                    triangle_leaves.insert(tri, idx as u32);
                }
                LightTreeLeafKind::Point(PointLightIndex(i)) => {
                    point_leaves[i] = idx as u32;
                }
                LightTreeLeafKind::Spot(SpotLightIndex(i)) => {
                    spot_leaves[i] = idx as u32;
                }
            }
        }
    }

    Some(LightTree {
        nodes,
        root,
        triangle_leaves,
        point_leaves,
        spot_leaves,
    })
}

fn collect_triangle_leaves(scene: &Scene) -> Vec<LeafBuild> {
    scene
        .area_light_triangles
        .par_iter()
        .filter_map(|tri| triangle_leaf(scene, tri))
        .collect()
}

fn collect_point_leaves(scene: &Scene) -> Vec<LeafBuild> {
    scene
        .point_lights
        .iter()
        .enumerate()
        .map(|(i, l)| point_leaf(PointLightIndex(i), l))
        .collect()
}

fn collect_spot_leaves(scene: &Scene) -> Vec<LeafBuild> {
    scene
        .spot_lights
        .iter()
        .enumerate()
        .map(|(i, l)| spot_leaf(SpotLightIndex(i), l))
        .collect()
}

fn triangle_leaf(scene: &Scene, area_light: &AreaLightTriangle) -> Option<LeafBuild> {
    let triangle = area_light.triangle;
    let [p0, p1, p2] = scene.triangle_positions(triangle);
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    let geom_normal_unnorm = e1.cross(e2);
    let area_doubled = geom_normal_unnorm.length();
    if area_doubled <= 0.0 {
        return None;
    }
    let n_geom = geom_normal_unnorm / area_doubled;
    let area = 0.5 * area_doubled;

    let material: &Material = scene.instance_material(triangle.instance_index);
    if !material.may_emit() {
        return None;
    }
    // Luminance-based importance: collapse the per-channel emission into a
    // single scalar so that the tree's PDF is well-defined regardless of
    // colour. `max_emission` returns the peak channel; use it as the
    // representative magnitude and weight by area for radiant flux.
    let phi = material.max_emission() * area;
    if phi <= 0.0 {
        return None;
    }

    let mu = (p0 + p1 + p2) / 3.0;
    // Spatial variance for a uniformly-sampled point on the triangle:
    //   sigma_s^2 = (||e1||^2 + ||e2||^2 - e1 . e2) / 18
    // (see Sec. 3.1.2 of the paper).
    let sigma_s2 = (e1.length_squared() + e2.length_squared() - e1.dot(e2)) / 18.0;

    // Bounding sphere centred on mu, containing the triangle's three vertices.
    let r = (p0 - mu)
        .length()
        .max((p1 - mu).length())
        .max((p2 - mu).length());

    let aabb = Bounds {
        min: p0.min(p1).min(p2),
        max: p0.max(p1).max(p2),
    };

    // Triangle leaf vMF: nu_bar = 0.5 * n_geom (paper Sec. 3.1.1, "rough fit
    // to Lambert's cosine"). This gives lambda ~ 1.83.
    let nu_bar = 0.5 * n_geom;

    Some(LeafBuild {
        kind: LightTreeLeafKind::Triangle(triangle),
        flux: phi,
        mu,
        sigma_s2,
        nu_bar,
        aabb,
        radius: r,
        cone_axis: n_geom,
        cone_theta_o: 0.0,
        cone_theta_e: PI * 0.5, // Lambertian: emission cone is the full hemisphere.
        centroid: mu,
    })
}

fn point_leaf(index: PointLightIndex, light: &PointLight) -> LeafBuild {
    // An isotropic point light emits uniformly; vMF axis-length is zero so
    // lambda will resolve to 0 (uniform). Phi = color_lum * intensity (W).
    let phi = sg::luminance(light.color) * light.intensity;
    let aabb = Bounds {
        min: light.position,
        max: light.position,
    };
    LeafBuild {
        kind: LightTreeLeafKind::Point(index),
        flux: phi.max(0.0),
        mu: light.position,
        sigma_s2: 0.0,
        nu_bar: Vec3::ZERO,
        aabb,
        radius: 0.0,
        cone_axis: Vec3::Z,
        // SAOH treats undirected emitters as sphere-covering: theta_o = pi
        // makes the orientation measure 4 pi.
        cone_theta_o: PI,
        cone_theta_e: 0.0,
        centroid: light.position,
    }
}

fn spot_leaf(index: SpotLightIndex, light: &SpotLight) -> LeafBuild {
    // Phi: the full radiated power of the underlying point light, in W.
    let phi = sg::luminance(light.color) * light.intensity;
    // For a uniform distribution over a cone of half-angle theta_total, the
    // mean direction has length (1 + cos theta) / 2. We use this as the vMF
    // axis-length so the SG-fitting picks up the spotlight's directionality.
    // theta_total can be recovered from cos_total_width.
    let cos_total = light.cos_total_width.clamp(-1.0, 1.0);
    let mean_len = ((1.0 + cos_total) * 0.5).clamp(0.0, 1.0);
    let nu_bar = mean_len * light.direction;
    let aabb = Bounds {
        min: light.position,
        max: light.position,
    };
    let theta_total = cos_total.acos();
    LeafBuild {
        kind: LightTreeLeafKind::Spot(index),
        flux: phi.max(0.0),
        mu: light.position,
        sigma_s2: 0.0,
        nu_bar,
        aabb,
        radius: 0.0,
        cone_axis: light.direction,
        cone_theta_o: 0.0,
        cone_theta_e: theta_total,
        centroid: light.position,
    }
}

#[derive(Debug, Clone, Copy)]
struct OrientationCone {
    axis: Vec3,
    theta_o: f32,
    theta_e: f32,
    /// Negative `theta_o` marks the empty cone.
    valid: bool,
}

impl OrientationCone {
    const EMPTY: Self = Self {
        axis: Vec3::Z,
        theta_o: 0.0,
        theta_e: 0.0,
        valid: false,
    };

    fn from_leaf(leaf: &LeafBuild) -> Self {
        Self {
            axis: leaf.cone_axis,
            theta_o: leaf.cone_theta_o,
            theta_e: leaf.cone_theta_e,
            valid: true,
        }
    }

    fn measure(self) -> f32 {
        if !self.valid {
            return 0.0;
        }
        // [Conty & Kulla 2018, Eq. 1]
        let theta_o = self.theta_o.clamp(0.0, PI);
        let theta_e = self.theta_e.clamp(0.0, PI * 0.5);
        let theta_w = (theta_o + theta_e).min(PI);
        let cos_o = theta_o.cos();
        let sin_o = theta_o.sin();
        2.0 * PI * (1.0 - cos_o)
            + 0.5
                * PI
                * (2.0 * theta_w * sin_o - (theta_o - 2.0 * theta_w).cos() - 2.0 * theta_o * sin_o
                    + cos_o)
    }

    fn merge(a: Self, b: Self) -> Self {
        if !a.valid {
            return b;
        }
        if !b.valid {
            return a;
        }
        let theta_e = a.theta_e.max(b.theta_e);
        // Pick the larger cone as the base; b is contained in a if applicable.
        let (a, b) = if a.theta_o >= b.theta_o {
            (a, b)
        } else {
            (b, a)
        };
        let cos_d = a.axis.dot(b.axis).clamp(-1.0, 1.0);
        let theta_d = cos_d.acos();
        if (theta_d + b.theta_o).min(PI) <= a.theta_o {
            return Self {
                axis: a.axis,
                theta_o: a.theta_o,
                theta_e,
                valid: true,
            };
        }
        let theta_o = ((a.theta_o + theta_d + b.theta_o) * 0.5).min(PI);
        if theta_o >= PI - 1e-5 {
            return Self {
                axis: a.axis,
                theta_o: PI,
                theta_e,
                valid: true,
            };
        }
        let rotation_axis = a.axis.cross(b.axis);
        let new_axis = if rotation_axis.length_squared() > 1.0e-12 {
            let theta_r = theta_o - a.theta_o;
            let k = rotation_axis.normalize();
            let cos_r = theta_r.cos();
            let sin_r = theta_r.sin();
            (a.axis * cos_r + k.cross(a.axis) * sin_r + k * k.dot(a.axis) * (1.0 - cos_r))
                .normalize_or_zero()
        } else {
            // Anti-parallel axes: any rotation axis works to get a hemisphere.
            a.axis
        };
        Self {
            axis: if new_axis.length_squared() > 0.0 {
                new_axis
            } else {
                a.axis
            },
            theta_o,
            theta_e,
            valid: true,
        }
    }
}

/// Aggregate cluster parameters of a contiguous leaf range. Returns the
/// node-level (flux, mu, sigma_s2, nu_bar, aabb, radius, cone) used for both
/// SAOH cost evaluation and final node output.
fn aggregate(leaves: &[LeafBuild]) -> AggregatedCluster {
    let mut acc = AggregatedCluster::EMPTY;
    for leaf in leaves {
        acc.merge_leaf(leaf);
    }
    acc.finish_radius();
    acc
}

#[derive(Debug, Clone, Copy)]
struct AggregatedCluster {
    flux: f32,
    mu: Vec3,
    sigma_s2: f32,
    nu_bar: Vec3,
    aabb: Bounds,
    radius: f32,
    cone: OrientationCone,
}

impl AggregatedCluster {
    const EMPTY: Self = Self {
        flux: 0.0,
        mu: Vec3::ZERO,
        sigma_s2: 0.0,
        nu_bar: Vec3::ZERO,
        aabb: Bounds::EMPTY,
        radius: 0.0,
        cone: OrientationCone::EMPTY,
    };

    fn merge_leaf(&mut self, leaf: &LeafBuild) {
        let new_flux = self.flux + leaf.flux;
        if new_flux <= 0.0 {
            return;
        }
        if self.flux <= 0.0 {
            self.flux = leaf.flux;
            self.mu = leaf.mu;
            self.sigma_s2 = leaf.sigma_s2;
            self.nu_bar = leaf.nu_bar;
            self.aabb = leaf.aabb;
            self.cone = OrientationCone::from_leaf(leaf);
            return;
        }
        let (mu, sigma_s2, nu_bar, total) = sg::merge_cluster_params(
            self.flux,
            self.mu,
            self.sigma_s2,
            self.nu_bar,
            leaf.flux,
            leaf.mu,
            leaf.sigma_s2,
            leaf.nu_bar,
        );
        self.flux = total;
        self.mu = mu;
        self.sigma_s2 = sigma_s2;
        self.nu_bar = nu_bar;
        self.aabb = self.aabb.union(leaf.aabb);
        self.cone = OrientationCone::merge(self.cone, OrientationCone::from_leaf(leaf));
    }

    fn merge_with(&mut self, other: &Self) {
        if other.flux <= 0.0 {
            return;
        }
        if self.flux <= 0.0 {
            *self = *other;
            return;
        }
        let (mu, sigma_s2, nu_bar, total) = sg::merge_cluster_params(
            self.flux,
            self.mu,
            self.sigma_s2,
            self.nu_bar,
            other.flux,
            other.mu,
            other.sigma_s2,
            other.nu_bar,
        );
        self.flux = total;
        self.mu = mu;
        self.sigma_s2 = sigma_s2;
        self.nu_bar = nu_bar;
        self.aabb = self.aabb.union(other.aabb);
        self.cone = OrientationCone::merge(self.cone, other.cone);
    }

    fn finish_radius(&mut self) {
        if self.flux <= 0.0 {
            self.radius = 0.0;
            return;
        }
        // Bounding-sphere radius about mu containing the cluster's AABB.
        let extents = (self.aabb.max - self.mu)
            .abs()
            .max((self.aabb.min - self.mu).abs());
        self.radius = extents.length();
    }

    fn surface_area(&self) -> f32 {
        self.aabb.surface_area().max(0.0)
    }

    fn cone_measure(&self) -> f32 {
        self.cone.measure()
    }
}

fn build_recursive(leaves: &mut [LeafBuild], nodes: &mut Vec<LightTreeNode>, parent: u32) -> u32 {
    if leaves.len() <= TARGET_LEAF_SIZE {
        // Leaf node: should be exactly one primitive, but cope with empties.
        let leaf = leaves[0];
        return push_leaf(nodes, parent, leaf);
    }

    let total = aggregate(leaves);

    let split_result = find_best_binned_split(leaves, &total);

    let split_index = match split_result {
        Some(s) => s,
        None => {
            // No improving split — fall back to a median split along the
            // longest centroid extent so we still make progress.
            let extent = leaves.iter().map(|l| l.centroid).fold(
                (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
                |(mn, mx), c| (mn.min(c), mx.max(c)),
            );
            let dims = extent.1 - extent.0;
            let axis = if dims.x >= dims.y && dims.x >= dims.z {
                0
            } else if dims.y >= dims.z {
                1
            } else {
                2
            };
            leaves.sort_unstable_by(|a, b| {
                a.centroid[axis]
                    .partial_cmp(&b.centroid[axis])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            leaves.len() / 2
        }
    };

    let leaves_len = leaves.len();
    let split_index = split_index.clamp(1, leaves_len - 1);

    // Reserve this internal node up front so child indices link back correctly.
    let node_idx = nodes.len() as u32;
    nodes.push(empty_internal_node(parent));

    let go_parallel = leaves_len >= PARALLEL_BUILD_THRESHOLD;

    let (left_slice, right_slice) = leaves.split_at_mut(split_index);

    let (left_idx, right_idx) = if go_parallel {
        // Parallel path: build sub-trees into local buffers, then splice in.
        let (left_pair, right_pair) = rayon::join(
            || {
                let mut local = Vec::new();
                let lroot = build_recursive(left_slice, &mut local, INVALID_NODE);
                (local, lroot)
            },
            || {
                let mut local = Vec::new();
                let rroot = build_recursive(right_slice, &mut local, INVALID_NODE);
                (local, rroot)
            },
        );
        let (left_local, lroot_local) = left_pair;
        let (right_local, rroot_local) = right_pair;
        let lbase = nodes.len() as u32;
        nodes.extend(left_local);
        let rbase = nodes.len() as u32;
        nodes.extend(right_local);
        let lroot = lroot_local + lbase;
        let rroot = rroot_local + rbase;
        // Patch internal child references that referenced the local buffers.
        let nodes_len = nodes.len() as u32;
        patch_subtree(nodes, lbase, rbase, lbase);
        patch_subtree(nodes, rbase, nodes_len, rbase);
        (lroot, rroot)
    } else {
        let lroot = build_recursive(left_slice, nodes, INVALID_NODE);
        let rroot = build_recursive(right_slice, nodes, INVALID_NODE);
        (lroot, rroot)
    };

    {
        let node = &mut nodes[node_idx as usize];
        node.left = left_idx;
        node.right = right_idx;
        let lambda = sg::vmf_axis_length_to_sharpness(total.nu_bar.length());
        let nu = if total.nu_bar.length_squared() > 0.0 {
            total.nu_bar.normalize()
        } else {
            Vec3::Z
        };
        node.flux = total.flux;
        node.mu = total.mu;
        node.sigma_s2 = total.sigma_s2;
        node.nu = nu;
        node.lambda = lambda;
        node.aabb = total.aabb;
        node.radius = total.radius;
        node.cone_axis = total.cone.axis;
        node.cone_theta_o = total.cone.theta_o;
        node.cone_theta_e = total.cone.theta_e;
    }

    node_idx
}

fn patch_subtree(nodes: &mut [LightTreeNode], begin: u32, end: u32, offset: u32) {
    if offset == 0 {
        return;
    }
    for i in begin..end {
        let n = &mut nodes[i as usize];
        if !n.is_leaf() {
            // child indices were emitted relative to a local buffer; shift
            // them so they reference the merged `nodes` buffer.
            n.left += offset;
            n.right += offset;
        }
        if n.parent != INVALID_NODE {
            n.parent += offset;
        }
    }
}

fn empty_internal_node(parent: u32) -> LightTreeNode {
    LightTreeNode {
        flux: 0.0,
        mu: Vec3::ZERO,
        sigma_s2: 0.0,
        nu: Vec3::Z,
        lambda: 0.0,
        aabb: Bounds::EMPTY,
        radius: 0.0,
        cone_axis: Vec3::Z,
        cone_theta_o: 0.0,
        cone_theta_e: 0.0,
        leaf: None,
        left: INVALID_NODE,
        right: INVALID_NODE,
        parent,
    }
}

fn push_leaf(nodes: &mut Vec<LightTreeNode>, parent: u32, leaf: LeafBuild) -> u32 {
    let idx = nodes.len() as u32;
    let lambda = sg::vmf_axis_length_to_sharpness(leaf.nu_bar.length());
    let nu = if leaf.nu_bar.length_squared() > 0.0 {
        leaf.nu_bar.normalize()
    } else {
        Vec3::Z
    };
    nodes.push(LightTreeNode {
        flux: leaf.flux,
        mu: leaf.mu,
        sigma_s2: leaf.sigma_s2,
        nu,
        lambda,
        aabb: leaf.aabb,
        radius: leaf.radius,
        cone_axis: leaf.cone_axis,
        cone_theta_o: leaf.cone_theta_o,
        cone_theta_e: leaf.cone_theta_e,
        leaf: Some(leaf.kind),
        left: INVALID_NODE,
        right: INVALID_NODE,
        parent,
    });
    idx
}

fn fixup_parents(nodes: &mut [LightTreeNode], root: u32) {
    // After flat construction, set parent pointers from each node's children.
    nodes[root as usize].parent = INVALID_NODE;
    let len = nodes.len();
    for i in 0..len {
        let (left, right, idx) = {
            let n = &nodes[i];
            (n.left, n.right, i as u32)
        };
        if left != INVALID_NODE {
            nodes[left as usize].parent = idx;
        }
        if right != INVALID_NODE {
            nodes[right as usize].parent = idx;
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Bin {
    flux: f32,
    cluster: AggregatedCluster,
    count: u32,
}

impl Bin {
    const EMPTY: Self = Self {
        flux: 0.0,
        cluster: AggregatedCluster::EMPTY,
        count: 0,
    };
}

fn find_best_binned_split(leaves: &mut [LeafBuild], total: &AggregatedCluster) -> Option<usize> {
    let n = leaves.len();
    if n < 2 {
        return None;
    }
    // Compute centroid AABB to define binning extents.
    let mut centroid_min = Vec3::splat(f32::INFINITY);
    let mut centroid_max = Vec3::splat(f32::NEG_INFINITY);
    for l in leaves.iter() {
        centroid_min = centroid_min.min(l.centroid);
        centroid_max = centroid_max.max(l.centroid);
    }
    let centroid_extent = centroid_max - centroid_min;
    // Degenerate case: all centroids coincide -> SAOH degenerates.
    if centroid_extent.max_element() <= 0.0 {
        return None;
    }

    let parent_area = total.surface_area().max(1.0e-30);
    let parent_cone = total.cone_measure().max(1.0e-30);

    let mut best_cost = f32::INFINITY;
    let mut best_axis = 0usize;
    let mut best_split = 0usize;

    for axis in 0..3 {
        if centroid_extent[axis] <= 0.0 {
            continue;
        }

        let mut bins = [Bin::EMPTY; NUM_BINS];
        let inv_extent = NUM_BINS as f32 / centroid_extent[axis];
        let lo = centroid_min[axis];

        for leaf in leaves.iter() {
            let key = ((leaf.centroid[axis] - lo) * inv_extent) as usize;
            let bin_idx = key.min(NUM_BINS - 1);
            let bin = &mut bins[bin_idx];
            bin.flux += leaf.flux;
            bin.cluster.merge_leaf(leaf);
            bin.count += 1;
        }

        // Prefix and suffix accumulators.
        let mut prefix = [AggregatedCluster::EMPTY; NUM_BINS];
        let mut prefix_count = [0u32; NUM_BINS];
        let mut acc = AggregatedCluster::EMPTY;
        let mut acc_count = 0u32;
        for i in 0..NUM_BINS {
            acc.merge_with(&bins[i].cluster);
            acc_count += bins[i].count;
            prefix[i] = acc;
            prefix_count[i] = acc_count;
        }
        let mut suffix = [AggregatedCluster::EMPTY; NUM_BINS];
        let mut suffix_count = [0u32; NUM_BINS];
        let mut sacc = AggregatedCluster::EMPTY;
        let mut scount = 0u32;
        for i in (0..NUM_BINS).rev() {
            sacc.merge_with(&bins[i].cluster);
            scount += bins[i].count;
            suffix[i] = sacc;
            suffix_count[i] = scount;
        }
        // Need to finalise radii so AABBs are consistent.
        for i in 0..NUM_BINS {
            prefix[i].finish_radius();
            suffix[i].finish_radius();
        }

        for split in 1..NUM_BINS {
            let l = &prefix[split - 1];
            let r = &suffix[split];
            if prefix_count[split - 1] == 0 || suffix_count[split] == 0 {
                continue;
            }
            // SAOH cost without parent-normalisation (constant here).
            let cost = l.flux * l.surface_area() * l.cone_measure()
                + r.flux * r.surface_area() * r.cone_measure();
            // Avoid splitting if both sides report zero work.
            if !cost.is_finite() {
                continue;
            }
            // Normalise so the cost is comparable across axes.
            let normalised = cost / (parent_area * parent_cone);
            if normalised < best_cost {
                best_cost = normalised;
                best_axis = axis;
                best_split = prefix_count[split - 1] as usize;
            }
        }
    }

    if !best_cost.is_finite() {
        return None;
    }

    leaves.sort_unstable_by(|a, b| {
        a.centroid[best_axis]
            .partial_cmp(&b.centroid[best_axis])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Some(best_split.clamp(1, n - 1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::{EmissiveMaterial, Material, NormalizedLambertMaterial};
    use crate::scene::{Mesh, Vertex};
    use glam::{Mat4, Vec2};

    fn unit_emissive_mesh(z: f32) -> Mesh {
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

    #[test]
    fn empty_scene_returns_no_tree() {
        let scene = Scene::new();
        assert!(build_light_tree(&scene).is_none());
    }

    #[test]
    fn single_emissive_triangle_yields_single_leaf_tree() {
        let mut scene = Scene::new();
        let mesh = scene.add_mesh(unit_emissive_mesh(0.0));
        let mat = scene.add_material(Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 1.0)));
        scene.add_instance(mesh, mat, Mat4::IDENTITY);
        let tree = build_light_tree(&scene).expect("tree must build");
        assert_eq!(tree.nodes.len(), 1);
        let n = &tree.nodes[tree.root as usize];
        assert!(n.is_leaf());
        assert!((n.flux - 0.5).abs() < 1.0e-3);
    }

    #[test]
    fn two_triangles_yield_internal_root() {
        let mut scene = Scene::new();
        let m1 = scene.add_mesh(unit_emissive_mesh(0.0));
        let m2 = scene.add_mesh(unit_emissive_mesh(5.0));
        let mat = scene.add_material(Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 1.0)));
        scene.add_instance(m1, mat, Mat4::IDENTITY);
        scene.add_instance(m2, mat, Mat4::IDENTITY);
        let tree = build_light_tree(&scene).expect("tree must build");
        assert!(tree.nodes.len() >= 3);
        let n = &tree.nodes[tree.root as usize];
        assert!(!n.is_leaf());
        let mut leaf_flux = 0.0;
        for node in &tree.nodes {
            if node.is_leaf() {
                leaf_flux += node.flux;
            }
        }
        assert!((leaf_flux - n.flux).abs() < 1.0e-4);
    }

    #[test]
    fn non_emissive_mesh_is_ignored() {
        let mut scene = Scene::new();
        let mesh = scene.add_mesh(unit_emissive_mesh(0.0));
        let mat = scene.add_material(Material::NormalizedLambert(NormalizedLambertMaterial::new(
            Vec3::splat(0.5),
        )));
        scene.add_instance(mesh, mat, Mat4::IDENTITY);
        assert!(build_light_tree(&scene).is_none());
    }

    #[test]
    fn point_light_only_scene_builds_tree() {
        let mut scene = Scene::new();
        scene.add_point_light(crate::light::PointLight::new(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::ONE,
            10.0,
        ));
        let tree = build_light_tree(&scene).expect("tree must build");
        assert_eq!(tree.nodes.len(), 1);
        let n = &tree.nodes[tree.root as usize];
        assert!(n.is_leaf());
        assert!(n.lambda.abs() < 1.0e-5);
    }

    #[test]
    fn spot_light_lambda_grows_with_narrower_cone() {
        let wide = SpotLight::new(
            Vec3::ZERO,
            -Vec3::Z,
            Vec3::ONE,
            1.0,
            (60.0_f32).to_radians(),
            (30.0_f32).to_radians(),
        );
        let narrow = SpotLight::new(
            Vec3::ZERO,
            -Vec3::Z,
            Vec3::ONE,
            1.0,
            (5.0_f32).to_radians(),
            (3.0_f32).to_radians(),
        );

        let wide_leaf = spot_leaf(SpotLightIndex(0), &wide);
        let narrow_leaf = spot_leaf(SpotLightIndex(1), &narrow);
        let wide_lambda = sg::vmf_axis_length_to_sharpness(wide_leaf.nu_bar.length());
        let narrow_lambda = sg::vmf_axis_length_to_sharpness(narrow_leaf.nu_bar.length());
        assert!(narrow_lambda > wide_lambda);
    }
}
