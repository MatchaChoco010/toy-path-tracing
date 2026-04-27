use glam::Vec3;
use rayon::prelude::*;
use wide::{CmpLe, f32x4};

use crate::{mesh::Bounds, ray::Ray};

const BIN_COUNT: usize = 16;
const TARGET_LEAF_PRIMS: usize = 4;
const MAX_LEAF_PRIMS: usize = 16;
const COST_TRAVERSE: f32 = 1.0;
const COST_INTERSECT: f32 = 1.0;
const PARALLEL_BUILD_THRESHOLD: usize = 1024;
const PARALLEL_BIN_THRESHOLD: usize = 8192;

const LEAF_BIT: u32 = 1 << 31;
const COUNT_SHIFT: u32 = 24;
const COUNT_MASK_BITS: u32 = 0x7F;
const OFFSET_MASK_BITS: u32 = (1 << 24) - 1;

pub const EMPTY_CHILD: u32 = LEAF_BIT;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Child {
    Empty,
    Leaf { offset: u32, count: u32 },
    Interior { node: u32 },
}

pub fn encode_leaf(offset: u32, count: u32) -> u32 {
    debug_assert!(count <= COUNT_MASK_BITS);
    debug_assert!(offset <= OFFSET_MASK_BITS);
    LEAF_BIT | (count << COUNT_SHIFT) | offset
}

pub fn encode_interior(node_index: u32) -> u32 {
    debug_assert!(node_index < (1 << 31));
    node_index
}

pub fn decode_child(c: u32) -> Child {
    if c & LEAF_BIT != 0 {
        let count = (c >> COUNT_SHIFT) & COUNT_MASK_BITS;
        if count == 0 {
            return Child::Empty;
        }
        let offset = c & OFFSET_MASK_BITS;
        Child::Leaf { offset, count }
    } else {
        Child::Interior { node: c }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QbvhNode {
    pub min_x: f32x4,
    pub min_y: f32x4,
    pub min_z: f32x4,
    pub max_x: f32x4,
    pub max_y: f32x4,
    pub max_z: f32x4,
    pub children: [u32; 4],
    pub valid_mask: u8,
}

impl PartialEq for QbvhNode {
    fn eq(&self, other: &Self) -> bool {
        self.valid_mask == other.valid_mask
            && self.children == other.children
            && self.min_x.to_array() == other.min_x.to_array()
            && self.min_y.to_array() == other.min_y.to_array()
            && self.min_z.to_array() == other.min_z.to_array()
            && self.max_x.to_array() == other.max_x.to_array()
            && self.max_y.to_array() == other.max_y.to_array()
            && self.max_z.to_array() == other.max_z.to_array()
    }
}

impl QbvhNode {
    pub fn empty() -> Self {
        Self {
            min_x: f32x4::splat(f32::INFINITY),
            min_y: f32x4::splat(f32::INFINITY),
            min_z: f32x4::splat(f32::INFINITY),
            max_x: f32x4::splat(f32::NEG_INFINITY),
            max_y: f32x4::splat(f32::NEG_INFINITY),
            max_z: f32x4::splat(f32::NEG_INFINITY),
            children: [EMPTY_CHILD; 4],
            valid_mask: 0,
        }
    }

    pub fn set_child(&mut self, slot: usize, child_ref: u32, bounds: Bounds) {
        let mut min_x = self.min_x.to_array();
        let mut min_y = self.min_y.to_array();
        let mut min_z = self.min_z.to_array();
        let mut max_x = self.max_x.to_array();
        let mut max_y = self.max_y.to_array();
        let mut max_z = self.max_z.to_array();
        min_x[slot] = bounds.min.x;
        min_y[slot] = bounds.min.y;
        min_z[slot] = bounds.min.z;
        max_x[slot] = bounds.max.x;
        max_y[slot] = bounds.max.y;
        max_z[slot] = bounds.max.z;
        self.min_x = f32x4::from(min_x);
        self.min_y = f32x4::from(min_y);
        self.min_z = f32x4::from(min_z);
        self.max_x = f32x4::from(max_x);
        self.max_y = f32x4::from(max_y);
        self.max_z = f32x4::from(max_z);
        self.children[slot] = child_ref;
        self.valid_mask |= 1u8 << slot;
    }

    pub fn child_bounds(&self, slot: usize) -> Bounds {
        Bounds {
            min: Vec3::new(
                self.min_x.as_array_ref()[slot],
                self.min_y.as_array_ref()[slot],
                self.min_z.as_array_ref()[slot],
            ),
            max: Vec3::new(
                self.max_x.as_array_ref()[slot],
                self.max_y.as_array_ref()[slot],
                self.max_z.as_array_ref()[slot],
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qbvh {
    pub nodes: Vec<QbvhNode>,
    pub primitive_indices: Vec<usize>,
}

#[derive(Debug, Clone, Copy)]
struct BuilderPrimitive {
    index: usize,
    bounds: Bounds,
    centroid: Vec3,
}

pub fn build_qbvh(primitive_bounds: &[Bounds]) -> Option<Qbvh> {
    if primitive_bounds.is_empty() {
        return None;
    }

    let mut primitives = primitive_bounds
        .iter()
        .copied()
        .enumerate()
        .map(|(index, bounds)| BuilderPrimitive {
            index,
            bounds,
            centroid: bounds.center(),
        })
        .collect::<Vec<_>>();

    let build_root = build_qbvh_node(&mut primitives);
    let primitive_indices = primitives
        .into_iter()
        .map(|primitive| primitive.index)
        .collect();
    let mut nodes = Vec::new();
    flatten_build_node(&build_root, &mut nodes);

    Some(Qbvh {
        nodes,
        primitive_indices,
    })
}

#[derive(Debug)]
enum BuildNode {
    Leaf {
        bounds: Bounds,
        count: u32,
    },
    Interior {
        bounds: Bounds,
        children: Vec<BuildNode>,
    },
}

fn build_qbvh_node(primitives: &mut [BuilderPrimitive]) -> BuildNode {
    let bounds = bounds_of(primitives);
    let count = primitives.len() as u32;

    if primitives.len() <= TARGET_LEAF_PRIMS {
        return BuildNode::Leaf { bounds, count };
    }

    let split1 = match find_best_binary_split(primitives, &bounds) {
        Some(split) => split,
        None => {
            return make_leaf_or_force_split(primitives, bounds);
        }
    };
    let leaf_cost_here = leaf_cost(primitives.len() as u32);
    if split1.cost >= leaf_cost_here && primitives.len() <= MAX_LEAF_PRIMS {
        return BuildNode::Leaf { bounds, count };
    }

    let mid1 = partition_in_place(primitives, &split1);
    if mid1 == 0 || mid1 == primitives.len() {
        return make_leaf_or_force_split(primitives, bounds);
    }

    let (left, right) = primitives.split_at_mut(mid1);

    let parallel = primitives_len_for(left, right) >= PARALLEL_BUILD_THRESHOLD;
    let (left_children, right_children) = if parallel {
        rayon::join(|| split_into_children(left), || split_into_children(right))
    } else {
        (split_into_children(left), split_into_children(right))
    };

    let mut children = left_children;
    children.extend(right_children);

    BuildNode::Interior { bounds, children }
}

fn primitives_len_for(left: &[BuilderPrimitive], right: &[BuilderPrimitive]) -> usize {
    left.len() + right.len()
}

fn split_into_children(primitives: &mut [BuilderPrimitive]) -> Vec<BuildNode> {
    if primitives.is_empty() {
        return Vec::new();
    }

    let bounds = bounds_of(primitives);

    if primitives.len() <= TARGET_LEAF_PRIMS {
        return vec![build_qbvh_node(primitives)];
    }

    let split = match find_best_binary_split(primitives, &bounds) {
        Some(split) => split,
        None => return vec![build_qbvh_node(primitives)],
    };
    let leaf_cost_here = leaf_cost(primitives.len() as u32);
    if split.cost >= leaf_cost_here && primitives.len() <= MAX_LEAF_PRIMS {
        return vec![build_qbvh_node(primitives)];
    }

    let mid = partition_in_place(primitives, &split);
    if mid == 0 || mid == primitives.len() {
        return vec![build_qbvh_node(primitives)];
    }

    let (left, right) = primitives.split_at_mut(mid);

    let parallel = primitives_len_for(left, right) >= PARALLEL_BUILD_THRESHOLD;
    let (l, r) = if parallel {
        rayon::join(|| build_qbvh_node(left), || build_qbvh_node(right))
    } else {
        (build_qbvh_node(left), build_qbvh_node(right))
    };
    vec![l, r]
}

fn make_leaf_or_force_split(
    primitives: &mut [BuilderPrimitive],
    bounds: Bounds,
) -> BuildNode {
    if primitives.len() <= MAX_LEAF_PRIMS {
        return BuildNode::Leaf {
            bounds,
            count: primitives.len() as u32,
        };
    }

    let mid = primitives.len() / 2;
    let extent = bounds.extent();
    let axis = if extent.x >= extent.y && extent.x >= extent.z {
        0
    } else if extent.y >= extent.z {
        1
    } else {
        2
    };
    primitives.sort_unstable_by(|a, b| {
        component(a.centroid, axis)
            .partial_cmp(&component(b.centroid, axis))
            .unwrap_or(core::cmp::Ordering::Equal)
    });

    let (left, right) = primitives.split_at_mut(mid);

    let parallel = primitives_len_for(left, right) >= PARALLEL_BUILD_THRESHOLD;
    let (left_children, right_children) = if parallel {
        rayon::join(|| split_into_children(left), || split_into_children(right))
    } else {
        (split_into_children(left), split_into_children(right))
    };

    let mut children = left_children;
    children.extend(right_children);

    BuildNode::Interior { bounds, children }
}

#[derive(Debug, Clone, Copy)]
struct BinSplit {
    axis: usize,
    centroid_min: f32,
    centroid_max: f32,
    bin_split: usize,
    cost: f32,
}

fn find_best_binary_split(
    primitives: &[BuilderPrimitive],
    parent_bounds: &Bounds,
) -> Option<BinSplit> {
    if primitives.len() <= 1 {
        return None;
    }
    let centroid_bounds = centroid_bounds_of(primitives);
    let parent_area = parent_bounds.surface_area().max(1.0e-30);
    let mut best: Option<BinSplit> = None;

    let parallel_binning = primitives.len() >= PARALLEL_BIN_THRESHOLD;

    for axis in 0..3 {
        let cmin = component(centroid_bounds.min, axis);
        let cmax = component(centroid_bounds.max, axis);
        if cmax - cmin <= 0.0 {
            continue;
        }
        let bins = if parallel_binning {
            bin_primitives_parallel(primitives, axis, cmin, cmax)
        } else {
            bin_primitives(primitives, axis, cmin, cmax)
        };
        let Some((bin_split, cost)) = evaluate_bins(&bins, parent_area) else {
            continue;
        };
        if best.is_none() || cost < best.as_ref().unwrap().cost {
            best = Some(BinSplit {
                axis,
                centroid_min: cmin,
                centroid_max: cmax,
                bin_split,
                cost,
            });
        }
    }

    best
}

fn bin_primitives_parallel(
    primitives: &[BuilderPrimitive],
    axis: usize,
    centroid_min: f32,
    centroid_max: f32,
) -> [Bin; BIN_COUNT] {
    let chunk_size = (primitives.len() / rayon::current_num_threads().max(1)).max(1024);
    primitives
        .par_chunks(chunk_size)
        .map(|chunk| bin_primitives(chunk, axis, centroid_min, centroid_max))
        .reduce(|| [Bin::EMPTY; BIN_COUNT], merge_bins)
}

fn merge_bins(a: [Bin; BIN_COUNT], b: [Bin; BIN_COUNT]) -> [Bin; BIN_COUNT] {
    let mut result = a;
    for i in 0..BIN_COUNT {
        result[i].bounds = result[i].bounds.union(b[i].bounds);
        result[i].count += b[i].count;
    }
    result
}

#[derive(Debug, Clone, Copy)]
struct Bin {
    bounds: Bounds,
    count: u32,
}

impl Bin {
    const EMPTY: Self = Self {
        bounds: Bounds::EMPTY,
        count: 0,
    };
}

fn bin_primitives(
    primitives: &[BuilderPrimitive],
    axis: usize,
    centroid_min: f32,
    centroid_max: f32,
) -> [Bin; BIN_COUNT] {
    let mut bins = [Bin::EMPTY; BIN_COUNT];
    let extent = (centroid_max - centroid_min).max(1.0e-30);
    let inv = (BIN_COUNT as f32) / extent;
    for prim in primitives {
        let c = component(prim.centroid, axis);
        let bin_index = bin_index_for_centroid(c, centroid_min, inv);
        bins[bin_index].bounds = bins[bin_index].bounds.union(prim.bounds);
        bins[bin_index].count += 1;
    }
    bins
}

fn bin_index_for_centroid(centroid_value: f32, centroid_min: f32, inv_extent: f32) -> usize {
    let raw = (centroid_value - centroid_min) * inv_extent;
    let clamped = raw.clamp(0.0, (BIN_COUNT - 1) as f32);
    clamped as usize
}

fn evaluate_bins(bins: &[Bin; BIN_COUNT], parent_area: f32) -> Option<(usize, f32)> {
    let mut left_bounds = [Bounds::EMPTY; BIN_COUNT];
    let mut right_bounds = [Bounds::EMPTY; BIN_COUNT];
    let mut left_count = [0u32; BIN_COUNT];
    let mut right_count = [0u32; BIN_COUNT];

    let mut acc_b = Bounds::EMPTY;
    let mut acc_c = 0u32;
    for i in 0..BIN_COUNT {
        acc_b = acc_b.union(bins[i].bounds);
        acc_c += bins[i].count;
        left_bounds[i] = acc_b;
        left_count[i] = acc_c;
    }
    let mut acc_b = Bounds::EMPTY;
    let mut acc_c = 0u32;
    for i in (0..BIN_COUNT).rev() {
        acc_b = acc_b.union(bins[i].bounds);
        acc_c += bins[i].count;
        right_bounds[i] = acc_b;
        right_count[i] = acc_c;
    }

    let total = left_count[BIN_COUNT - 1];
    if total <= 1 {
        return None;
    }

    let mut best_cost = f32::INFINITY;
    let mut best_split: Option<usize> = None;

    for split in 0..(BIN_COUNT - 1) {
        let nl = left_count[split];
        let nr = total - nl;
        if nl == 0 || nr == 0 {
            continue;
        }
        let al = left_bounds[split].surface_area();
        let ar = right_bounds[split + 1].surface_area();
        let cost = COST_TRAVERSE
            + COST_INTERSECT * (al * nl as f32 + ar * nr as f32) / parent_area;
        if cost < best_cost {
            best_cost = cost;
            best_split = Some(split);
        }
    }

    best_split.map(|split| (split, best_cost))
}

fn leaf_cost(n: u32) -> f32 {
    COST_INTERSECT * n as f32
}

fn partition_in_place(primitives: &mut [BuilderPrimitive], split: &BinSplit) -> usize {
    let extent = (split.centroid_max - split.centroid_min).max(1.0e-30);
    let inv = (BIN_COUNT as f32) / extent;
    let split_bin = split.bin_split;
    let axis = split.axis;
    let cmin = split.centroid_min;

    let mut left = 0;
    let mut right = primitives.len();
    while left < right {
        let c = component(primitives[left].centroid, axis);
        let bin = bin_index_for_centroid(c, cmin, inv);
        if bin <= split_bin {
            left += 1;
        } else {
            right -= 1;
            primitives.swap(left, right);
        }
    }
    left
}

fn flatten_build_node(root: &BuildNode, nodes: &mut Vec<QbvhNode>) -> u32 {
    flatten_recursive(root, 0, nodes).0
}

fn flatten_recursive(
    node: &BuildNode,
    primitive_offset: u32,
    nodes: &mut Vec<QbvhNode>,
) -> (u32, u32) {
    match node {
        BuildNode::Leaf { count, .. } => (encode_leaf(primitive_offset, *count), *count),
        BuildNode::Interior { children, .. } => {
            let node_index = nodes.len() as u32;
            nodes.push(QbvhNode::empty());

            let mut new_node = QbvhNode::empty();
            let mut current_offset = primitive_offset;
            for (i, child) in children.iter().enumerate() {
                let child_bounds = bounds_of_subtree(child);
                let (child_ref, child_count) = flatten_recursive(child, current_offset, nodes);
                new_node.set_child(i, child_ref, child_bounds);
                current_offset += child_count;
            }

            nodes[node_index as usize] = new_node;
            let total = current_offset - primitive_offset;
            (encode_interior(node_index), total)
        }
    }
}

fn bounds_of_subtree(node: &BuildNode) -> Bounds {
    match node {
        BuildNode::Leaf { bounds, .. } | BuildNode::Interior { bounds, .. } => *bounds,
    }
}

fn bounds_of(primitives: &[BuilderPrimitive]) -> Bounds {
    let mut bounds = Bounds::EMPTY;
    for prim in primitives {
        bounds = bounds.union(prim.bounds);
    }
    bounds
}

fn centroid_bounds_of(primitives: &[BuilderPrimitive]) -> Bounds {
    let mut bounds = Bounds::EMPTY;
    for prim in primitives {
        bounds = bounds.union(Bounds {
            min: prim.centroid,
            max: prim.centroid,
        });
    }
    bounds
}

fn component(value: Vec3, axis: usize) -> f32 {
    match axis {
        0 => value.x,
        1 => value.y,
        _ => value.z,
    }
}

#[derive(Debug, Clone, Copy)]
struct RaySimd {
    origin_x: f32x4,
    origin_y: f32x4,
    origin_z: f32x4,
    inv_dir_x: f32x4,
    inv_dir_y: f32x4,
    inv_dir_z: f32x4,
}

impl RaySimd {
    fn new(ray: &Ray) -> Self {
        Self {
            origin_x: f32x4::splat(ray.origin.x),
            origin_y: f32x4::splat(ray.origin.y),
            origin_z: f32x4::splat(ray.origin.z),
            inv_dir_x: f32x4::splat(safe_inv(ray.direction.x)),
            inv_dir_y: f32x4::splat(safe_inv(ray.direction.y)),
            inv_dir_z: f32x4::splat(safe_inv(ray.direction.z)),
        }
    }
}

fn safe_inv(d: f32) -> f32 {
    if d == 0.0 {
        f32::INFINITY
    } else {
        1.0 / d
    }
}

fn intersect_4_aabbs(node: &QbvhNode, ray: &RaySimd, t_max: f32) -> (u32, [f32; 4]) {
    let t1x = (node.min_x - ray.origin_x) * ray.inv_dir_x;
    let t2x = (node.max_x - ray.origin_x) * ray.inv_dir_x;
    let t1y = (node.min_y - ray.origin_y) * ray.inv_dir_y;
    let t2y = (node.max_y - ray.origin_y) * ray.inv_dir_y;
    let t1z = (node.min_z - ray.origin_z) * ray.inv_dir_z;
    let t2z = (node.max_z - ray.origin_z) * ray.inv_dir_z;

    let tmin_x = t1x.fast_min(t2x);
    let tmax_x = t1x.fast_max(t2x);
    let tmin_y = t1y.fast_min(t2y);
    let tmax_y = t1y.fast_max(t2y);
    let tmin_z = t1z.fast_min(t2z);
    let tmax_z = t1z.fast_max(t2z);

    let zero = f32x4::ZERO;
    let t_max_simd = f32x4::splat(t_max);

    let tmin = tmin_x.fast_max(tmin_y).fast_max(tmin_z).fast_max(zero);
    let tmax = tmax_x.fast_min(tmax_y).fast_min(tmax_z).fast_min(t_max_simd);

    let mask = tmin.cmp_le(tmax);
    let hit_bits = (mask.move_mask() as u32) & (node.valid_mask as u32);
    let tmin_arr = tmin.to_array();

    (hit_bits, tmin_arr)
}

const TRAVERSAL_STACK_DEPTH: usize = 128;

pub fn traverse_qbvh<L>(qbvh: &Qbvh, ray: &Ray, initial_t_max: f32, mut leaf_callback: L) -> f32
where
    L: FnMut(u32, u32, f32) -> f32,
{
    if qbvh.nodes.is_empty() {
        if qbvh.primitive_indices.is_empty() {
            return initial_t_max;
        }
        return leaf_callback(0, qbvh.primitive_indices.len() as u32, initial_t_max);
    }

    let ray_simd = RaySimd::new(ray);
    let mut closest_t = initial_t_max;

    let mut stack: [u32; TRAVERSAL_STACK_DEPTH] = [0; TRAVERSAL_STACK_DEPTH];
    let mut stack_len = 0usize;
    stack[stack_len] = 0;
    stack_len += 1;

    while stack_len > 0 {
        stack_len -= 1;
        let node_index = stack[stack_len];
        let node = &qbvh.nodes[node_index as usize];

        let (hit_bits, tmin_arr) = intersect_4_aabbs(node, &ray_simd, closest_t);
        if hit_bits == 0 {
            continue;
        }

        let mut hits: [(f32, u32); 4] = [(0.0, 0); 4];
        let mut hit_count = 0usize;
        for slot in 0..4 {
            if hit_bits & (1 << slot) == 0 {
                continue;
            }
            hits[hit_count] = (tmin_arr[slot], node.children[slot]);
            hit_count += 1;
        }

        for i in 1..hit_count {
            let mut j = i;
            while j > 0 && hits[j - 1].0 > hits[j].0 {
                hits.swap(j - 1, j);
                j -= 1;
            }
        }

        for k in (0..hit_count).rev() {
            let child_ref = hits[k].1;
            match decode_child(child_ref) {
                Child::Empty => {}
                Child::Leaf { offset, count } => {
                    closest_t = leaf_callback(offset, count, closest_t);
                }
                Child::Interior { node } => {
                    debug_assert!(stack_len < TRAVERSAL_STACK_DEPTH);
                    stack[stack_len] = node;
                    stack_len += 1;
                }
            }
        }
    }

    closest_t
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn make_bounds(min: Vec3, max: Vec3) -> Bounds {
        Bounds { min, max }
    }

    #[test]
    fn empty_input_returns_none() {
        assert!(build_qbvh(&[]).is_none());
    }

    #[test]
    fn single_primitive_builds_root_leaf() {
        let bounds = vec![make_bounds(Vec3::ZERO, Vec3::ONE)];
        let qbvh = build_qbvh(&bounds).expect("expected qbvh");
        assert!(qbvh.nodes.is_empty());
        assert_eq!(qbvh.primitive_indices, vec![0]);
    }

    #[test]
    fn child_encoding_round_trip() {
        let leaf = encode_leaf(123, 4);
        match decode_child(leaf) {
            Child::Leaf { offset, count } => {
                assert_eq!(offset, 123);
                assert_eq!(count, 4);
            }
            _ => panic!("expected leaf"),
        }
        let interior = encode_interior(7);
        match decode_child(interior) {
            Child::Interior { node } => assert_eq!(node, 7),
            _ => panic!("expected interior"),
        }
        match decode_child(EMPTY_CHILD) {
            Child::Empty => {}
            _ => panic!("expected empty"),
        }
    }

    #[test]
    fn intersect_4_aabbs_matches_scalar() {
        let mut node = QbvhNode::empty();
        node.set_child(0, encode_leaf(0, 1), make_bounds(Vec3::ZERO, Vec3::ONE));
        node.set_child(
            1,
            encode_leaf(1, 1),
            make_bounds(Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.0)),
        );
        node.set_child(
            2,
            encode_leaf(2, 1),
            make_bounds(Vec3::new(0.0, 2.0, 0.0), Vec3::new(1.0, 3.0, 1.0)),
        );

        let ray = Ray::new(Vec3::new(0.5, 0.5, 5.0), Vec3::NEG_Z);
        let ray_simd = RaySimd::new(&ray);
        let (hit_bits, _tmin) = intersect_4_aabbs(&node, &ray_simd, f32::INFINITY);
        assert_eq!(hit_bits & 0b0001, 0b0001);
        assert_eq!(hit_bits & 0b0010, 0);
        assert_eq!(hit_bits & 0b0100, 0);
        assert_eq!(hit_bits & 0b1000, 0);
    }

    #[test]
    fn intersect_4_aabbs_reports_per_lane_tmin() {
        let mut node = QbvhNode::empty();
        node.set_child(0, encode_leaf(0, 1), make_bounds(Vec3::ZERO, Vec3::ONE));
        node.set_child(
            1,
            encode_leaf(1, 1),
            make_bounds(
                Vec3::new(0.0, 0.0, -2.0),
                Vec3::new(1.0, 1.0, -1.0),
            ),
        );

        let ray = Ray::new(Vec3::new(0.5, 0.5, 5.0), Vec3::NEG_Z);
        let ray_simd = RaySimd::new(&ray);
        let (hit_bits, tmin) = intersect_4_aabbs(&node, &ray_simd, f32::INFINITY);
        assert_eq!(hit_bits & 0b0001, 0b0001);
        assert_eq!(hit_bits & 0b0010, 0b0010);
        assert!((tmin[0] - 4.0).abs() < 1.0e-5);
        assert!((tmin[1] - 6.0).abs() < 1.0e-5);
    }

    #[test]
    fn build_separates_clusters() {
        let bounds: Vec<Bounds> = (0..32)
            .map(|i| {
                let x = (i % 4) as f32;
                let y = ((i / 4) % 4) as f32;
                let z = (i / 16) as f32;
                make_bounds(
                    Vec3::new(x, y, z),
                    Vec3::new(x + 0.5, y + 0.5, z + 0.5),
                )
            })
            .collect();
        let qbvh = build_qbvh(&bounds).expect("expected qbvh");

        verify_invariants(&qbvh, &bounds);
    }

    #[test]
    fn build_handles_coincident_primitives() {
        let bounds: Vec<Bounds> = (0..20)
            .map(|_| make_bounds(Vec3::ZERO, Vec3::ONE))
            .collect();
        let qbvh = build_qbvh(&bounds).expect("expected qbvh");
        verify_invariants(&qbvh, &bounds);
    }

    #[test]
    fn build_with_parallel_thresholds_preserves_invariants() {
        let n = PARALLEL_BIN_THRESHOLD * 2 + 17;
        let bounds: Vec<Bounds> = (0..n)
            .map(|i| {
                let x = ((i * 73) % 211) as f32;
                let y = ((i * 37) % 191) as f32;
                let z = ((i * 53) % 173) as f32;
                make_bounds(
                    Vec3::new(x, y, z),
                    Vec3::new(x + 0.5, y + 0.5, z + 0.5),
                )
            })
            .collect();
        let qbvh = build_qbvh(&bounds).expect("expected qbvh");
        verify_invariants(&qbvh, &bounds);
    }

    #[test]
    fn traverse_matches_naive_intersection_for_random_rays() {
        let n = 200usize;
        let bounds: Vec<Bounds> = (0..n)
            .map(|i| {
                let x = ((i * 19) % 41) as f32;
                let y = ((i * 11) % 37) as f32;
                let z = ((i * 7) % 31) as f32;
                make_bounds(
                    Vec3::new(x, y, z),
                    Vec3::new(x + 0.4, y + 0.4, z + 0.4),
                )
            })
            .collect();
        let qbvh = build_qbvh(&bounds).expect("expected qbvh");

        let rays: Vec<Ray> = (0..32)
            .map(|i| {
                let ox = ((i * 5) % 41) as f32 + 0.1;
                let oy = ((i * 9) % 37) as f32 + 0.1;
                Ray::new(Vec3::new(ox, oy, 50.0), Vec3::NEG_Z)
            })
            .collect();

        for ray in &rays {
            let mut visited_indices: Vec<usize> = Vec::new();
            traverse_qbvh(&qbvh, ray, f32::INFINITY, |offset, count, t_max| {
                for k in offset..offset + count {
                    visited_indices.push(qbvh.primitive_indices[k as usize]);
                }
                t_max
            });
            visited_indices.sort_unstable();
            visited_indices.dedup();

            let mut expected: Vec<usize> = (0..n)
                .filter(|&i| {
                    let b = &bounds[i];
                    ray.origin.x >= b.min.x
                        && ray.origin.x <= b.max.x
                        && ray.origin.y >= b.min.y
                        && ray.origin.y <= b.max.y
                })
                .collect();
            expected.sort_unstable();

            for &i in &expected {
                assert!(
                    visited_indices.contains(&i),
                    "ray {:?} should visit primitive {} but did not",
                    ray.origin,
                    i
                );
            }
        }
    }

    fn slab_intersect_scalar(bounds: &Bounds, ray: &Ray, t_max: f32) -> Option<f32> {
        let inv = Vec3::new(
            safe_inv(ray.direction.x),
            safe_inv(ray.direction.y),
            safe_inv(ray.direction.z),
        );
        let t1 = (bounds.min - ray.origin) * inv;
        let t2 = (bounds.max - ray.origin) * inv;
        let tmin_v = t1.min(t2);
        let tmax_v = t1.max(t2);
        let tmin = tmin_v.x.max(tmin_v.y).max(tmin_v.z).max(0.0);
        let tmax = tmax_v.x.min(tmax_v.y).min(tmax_v.z).min(t_max);
        if tmin <= tmax { Some(tmin) } else { None }
    }

    fn brute_force_closest(bounds: &[Bounds], ray: &Ray) -> Option<(usize, f32)> {
        let mut best: Option<(usize, f32)> = None;
        for (i, b) in bounds.iter().enumerate() {
            if let Some(t) = slab_intersect_scalar(b, ray, f32::INFINITY) {
                if best.map_or(true, |(_, bt)| t < bt) {
                    best = Some((i, t));
                }
            }
        }
        best
    }

    fn qbvh_closest(qbvh: &Qbvh, bounds: &[Bounds], ray: &Ray) -> Option<(usize, f32)> {
        let mut best: Option<(usize, f32)> = None;
        traverse_qbvh(qbvh, ray, f32::INFINITY, |offset, count, t_max| {
            let mut new_t_max = t_max;
            for k in offset..offset + count {
                let idx = qbvh.primitive_indices[k as usize];
                if let Some(t) = slab_intersect_scalar(&bounds[idx], ray, new_t_max) {
                    if best.is_none_or(|(_, bt)| t < bt) {
                        best = Some((idx, t));
                        new_t_max = t;
                    }
                }
            }
            new_t_max
        });
        best
    }

    #[test]
    fn traverse_closest_matches_brute_force_on_grid() {
        let bounds: Vec<Bounds> = (0..200)
            .map(|i| {
                let x = ((i * 19) % 41) as f32;
                let y = ((i * 11) % 37) as f32;
                let z = ((i * 7) % 31) as f32;
                make_bounds(
                    Vec3::new(x, y, z),
                    Vec3::new(x + 0.4, y + 0.4, z + 0.4),
                )
            })
            .collect();
        let qbvh = build_qbvh(&bounds).expect("qbvh");

        let mut scene_bounds = Bounds::EMPTY;
        for b in &bounds {
            scene_bounds = scene_bounds.union(*b);
        }
        let center = scene_bounds.center();

        let directions = [
            Vec3::NEG_Z,
            Vec3::Z,
            Vec3::NEG_X,
            Vec3::X,
            Vec3::NEG_Y,
            Vec3::Y,
            Vec3::new(1.0, 1.0, 1.0).normalize(),
            Vec3::new(-1.0, 0.5, 0.3).normalize(),
            Vec3::new(0.2, -0.7, 0.6).normalize(),
        ];
        let mut compared = 0usize;
        let mut hit_count = 0usize;
        for dir in &directions {
            for i in 0..50 {
                let jitter = Vec3::new(
                    ((i * 13) % 47) as f32 * 0.5 - 12.0,
                    ((i * 17) % 43) as f32 * 0.5 - 11.0,
                    ((i * 7) % 41) as f32 * 0.5 - 10.0,
                );
                let origin = center + jitter - *dir * 80.0;
                let ray = Ray::new(origin, *dir);
                let bf = brute_force_closest(&bounds, &ray);
                let qb = qbvh_closest(&qbvh, &bounds, &ray);
                match (bf, qb) {
                    (None, None) => {}
                    (Some((bi, bt)), Some((qi, qt))) => {
                        let tol = 1.0e-3 * bt.abs().max(1.0);
                        assert!(
                            (bt - qt).abs() <= tol,
                            "t mismatch: brute={} qbvh={} ray={:?}",
                            bt,
                            qt,
                            ray.origin
                        );
                        if bi != qi {
                            assert!(
                                (bt - qt).abs() <= tol,
                                "different prim at same t requires near-tie"
                            );
                        }
                        hit_count += 1;
                    }
                    other => panic!("brute/qbvh disagreement: {:?}", other),
                }
                compared += 1;
            }
        }
        assert!(compared > 0);
        assert!(hit_count > 0, "expected at least one ray to hit");
    }

    #[test]
    fn traverse_few_primitives_root_leaf_path() {
        for n in 1..=4 {
            let bounds: Vec<Bounds> = (0..n)
                .map(|i| {
                    make_bounds(
                        Vec3::splat(i as f32),
                        Vec3::splat(i as f32 + 0.5),
                    )
                })
                .collect();
            let qbvh = build_qbvh(&bounds).expect("qbvh");
            assert!(qbvh.nodes.is_empty(), "expected nodes empty for n={}", n);
            let ray = Ray::new(Vec3::new(0.25, 0.25, -10.0), Vec3::Z);
            let bf = brute_force_closest(&bounds, &ray);
            let qb = qbvh_closest(&qbvh, &bounds, &ray);
            assert_eq!(bf.map(|(i, _)| i), qb.map(|(i, _)| i));
        }
    }

    #[test]
    fn build_handles_many_coincident_primitives() {
        let bounds: Vec<Bounds> = (0..120)
            .map(|_| make_bounds(Vec3::ZERO, Vec3::ONE))
            .collect();
        let qbvh = build_qbvh(&bounds).expect("qbvh");
        verify_invariants(&qbvh, &bounds);
        let ray = Ray::new(Vec3::new(0.5, 0.5, -5.0), Vec3::Z);
        let qb = qbvh_closest(&qbvh, &bounds, &ray);
        assert!(qb.is_some());
    }

    #[test]
    fn traverse_handles_axis_aligned_rays() {
        let bounds: Vec<Bounds> = (0..32)
            .map(|i| {
                let x = (i % 8) as f32;
                let y = ((i / 8) % 4) as f32;
                make_bounds(
                    Vec3::new(x, y, 0.0),
                    Vec3::new(x + 0.5, y + 0.5, 0.5),
                )
            })
            .collect();
        let qbvh = build_qbvh(&bounds).expect("qbvh");

        let cases: [(Vec3, Vec3); 6] = [
            (Vec3::new(-10.0, 0.25, 0.25), Vec3::X),
            (Vec3::new(10.0, 0.25, 0.25), Vec3::NEG_X),
            (Vec3::new(0.25, -10.0, 0.25), Vec3::Y),
            (Vec3::new(0.25, 10.0, 0.25), Vec3::NEG_Y),
            (Vec3::new(0.25, 0.25, -10.0), Vec3::Z),
            (Vec3::new(0.25, 0.25, 10.0), Vec3::NEG_Z),
        ];
        for (origin, dir) in &cases {
            let ray = Ray::new(*origin, *dir);
            let bf = brute_force_closest(&bounds, &ray);
            let qb = qbvh_closest(&qbvh, &bounds, &ray);
            assert_eq!(bf.map(|(i, _)| i), qb.map(|(i, _)| i));
            if let (Some((_, bt)), Some((_, qt))) = (bf, qb) {
                assert!((bt - qt).abs() <= 1.0e-3 * bt.abs().max(1.0));
            }
        }
    }

    #[test]
    fn traverse_handles_ray_origin_inside_aabb() {
        let bounds: Vec<Bounds> = vec![
            make_bounds(Vec3::ZERO, Vec3::splat(2.0)),
            make_bounds(Vec3::splat(5.0), Vec3::splat(6.0)),
            make_bounds(Vec3::splat(-3.0), Vec3::splat(-2.0)),
            make_bounds(Vec3::splat(10.0), Vec3::splat(11.0)),
            make_bounds(Vec3::splat(20.0), Vec3::splat(21.0)),
        ];
        let qbvh = build_qbvh(&bounds).expect("qbvh");
        let ray = Ray::new(Vec3::ONE, Vec3::X);
        let qb = qbvh_closest(&qbvh, &bounds, &ray);
        assert_eq!(qb.map(|(i, _)| i), Some(0));
        if let Some((_, t)) = qb {
            assert!(t.abs() < 1.0e-5, "ray inside should hit at t=0, got {}", t);
        }
    }

    #[test]
    fn build_with_skewed_distribution_preserves_invariants_and_closest_hit() {
        let n = 500usize;
        let bounds: Vec<Bounds> = (0..n)
            .map(|i| {
                let t = i as f32 / n as f32;
                let x = t * t * 100.0;
                let y = (i as f32).sin() * 0.5;
                let z = (i as f32 * 0.137).cos() * 0.5;
                make_bounds(
                    Vec3::new(x, y, z),
                    Vec3::new(x + 0.2, y + 0.2, z + 0.2),
                )
            })
            .collect();
        let qbvh = build_qbvh(&bounds).expect("qbvh");
        verify_invariants(&qbvh, &bounds);

        let directions = [
            Vec3::X,
            Vec3::NEG_X,
            Vec3::new(1.0, 0.2, 0.0).normalize(),
            Vec3::new(-1.0, -0.1, 0.05).normalize(),
        ];
        let mut hits = 0usize;
        for dir in &directions {
            for i in 0..30 {
                let origin = Vec3::new(-50.0 + i as f32 * 0.3, (i as f32).sin() * 0.4, 0.0);
                let ray = Ray::new(origin, *dir);
                let bf = brute_force_closest(&bounds, &ray);
                let qb = qbvh_closest(&qbvh, &bounds, &ray);
                assert_eq!(bf.map(|(i, _)| i), qb.map(|(i, _)| i));
                if bf.is_some() {
                    hits += 1;
                }
            }
        }
        assert!(hits > 0);
    }

    #[test]
    fn traverse_finds_correct_leaf_for_simple_grid() {
        let bounds: Vec<Bounds> = (0..16)
            .map(|i| {
                let x = i as f32 * 2.0;
                make_bounds(Vec3::new(x, 0.0, 0.0), Vec3::new(x + 1.0, 1.0, 1.0))
            })
            .collect();
        let qbvh = build_qbvh(&bounds).expect("expected qbvh");

        let ray = Ray::new(Vec3::new(20.5, 0.5, 5.0), Vec3::NEG_Z);

        let mut visited_indices = Vec::new();
        traverse_qbvh(&qbvh, &ray, f32::INFINITY, |offset, count, t_max| {
            for k in offset..offset + count {
                visited_indices.push(qbvh.primitive_indices[k as usize]);
            }
            t_max
        });
        assert!(visited_indices.contains(&10));
    }

    fn collect_subtree_bounds(
        qbvh: &Qbvh,
        primitive_bounds: &[Bounds],
        child_ref: u32,
        out: &mut Bounds,
    ) {
        match decode_child(child_ref) {
            Child::Empty => {}
            Child::Leaf { offset, count } => {
                for k in offset..offset + count {
                    let idx = qbvh.primitive_indices[k as usize];
                    *out = out.union(primitive_bounds[idx]);
                }
            }
            Child::Interior { node } => {
                let n = &qbvh.nodes[node as usize];
                for slot in 0..4 {
                    collect_subtree_bounds(qbvh, primitive_bounds, n.children[slot], out);
                }
            }
        }
    }

    fn verify_invariants(qbvh: &Qbvh, primitive_bounds: &[Bounds]) {
        for node in &qbvh.nodes {
            for slot in 0..4 {
                let child_ref = node.children[slot];
                if matches!(decode_child(child_ref), Child::Empty) {
                    continue;
                }
                let cb = node.child_bounds(slot);
                let mut subtree_bounds = Bounds::EMPTY;
                collect_subtree_bounds(qbvh, primitive_bounds, child_ref, &mut subtree_bounds);
                assert!(
                    bounds_contains(cb, subtree_bounds),
                    "stored child bounds must contain all descendant primitive bounds"
                );
                if let Child::Leaf { count, .. } = decode_child(child_ref) {
                    assert!(count >= 1 && count <= MAX_LEAF_PRIMS as u32);
                }
            }
        }

        let touched = total_primitives_in_qbvh(qbvh);
        assert_eq!(touched, primitive_bounds.len());

        let mut seen = vec![false; primitive_bounds.len()];
        for &idx in &qbvh.primitive_indices {
            assert!(!seen[idx]);
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&b| b));
    }

    fn bounds_contains(outer: Bounds, inner: Bounds) -> bool {
        const EPS: f32 = 1.0e-4;
        outer.min.x <= inner.min.x + EPS
            && outer.min.y <= inner.min.y + EPS
            && outer.min.z <= inner.min.z + EPS
            && outer.max.x + EPS >= inner.max.x
            && outer.max.y + EPS >= inner.max.y
            && outer.max.z + EPS >= inner.max.z
    }

    fn total_primitives_in_qbvh(qbvh: &Qbvh) -> usize {
        if qbvh.nodes.is_empty() {
            return qbvh.primitive_indices.len();
        }
        let mut total = 0usize;
        for node in &qbvh.nodes {
            for slot in 0..4 {
                if let Child::Leaf { count, .. } = decode_child(node.children[slot]) {
                    total += count as usize;
                }
            }
        }
        total
    }
}
