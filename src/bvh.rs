use crate::{mesh::Bounds, ray::Ray};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LinearBvhNode {
    Leaf {
        bounds: Bounds,
        primitive_offset: usize,
        primitive_count: usize,
    },
    Interior {
        bounds: Bounds,
        right_child_offset: usize,
    },
}

impl LinearBvhNode {
    pub fn bounds(&self) -> Bounds {
        match *self {
            Self::Leaf { bounds, .. } | Self::Interior { bounds, .. } => bounds,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeshBvh {
    pub nodes: Vec<LinearBvhNode>,
    pub triangle_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SceneBvh {
    pub nodes: Vec<LinearBvhNode>,
    pub instance_indices: Vec<usize>,
}

#[derive(Debug, Clone, Copy)]
struct BuilderPrimitive {
    index: usize,
    bounds: Bounds,
}

#[derive(Debug)]
enum BuildNode {
    Leaf {
        bounds: Bounds,
        primitive_offset: usize,
        primitive_count: usize,
    },
    Interior {
        bounds: Bounds,
        left: Box<BuildNode>,
        right: Box<BuildNode>,
    },
}

pub fn build_mesh_bvh(triangle_bounds: &[Bounds]) -> Option<MeshBvh> {
    let (nodes, ordered_indices) = build_linear_bvh(triangle_bounds)?;

    Some(MeshBvh {
        nodes,
        triangle_indices: ordered_indices,
    })
}

pub fn build_scene_bvh(instance_bounds: &[Bounds]) -> Option<SceneBvh> {
    let (nodes, ordered_indices) = build_linear_bvh(instance_bounds)?;

    Some(SceneBvh {
        nodes,
        instance_indices: ordered_indices,
    })
}

pub fn intersect_bounds(ray: &Ray, t_max: f32, bounds: Bounds) -> Option<f32> {
    let mut t_min = 0.0_f32;
    let mut t_max = t_max;

    for axis in 0..3 {
        let origin = component(ray.origin, axis);
        let direction = component(ray.direction, axis);
        let min = component(bounds.min, axis);
        let max = component(bounds.max, axis);

        if direction == 0.0 {
            if origin < min || origin > max {
                return None;
            }

            continue;
        }

        let inv_direction = 1.0 / direction;
        let mut t_near = (min - origin) * inv_direction;
        let mut t_far = (max - origin) * inv_direction;

        if t_near > t_far {
            core::mem::swap(&mut t_near, &mut t_far);
        }

        t_min = t_min.max(t_near);
        t_max = t_max.min(t_far);

        if t_min > t_max {
            return None;
        }
    }

    Some(t_min)
}

fn build_linear_bvh(primitive_bounds: &[Bounds]) -> Option<(Vec<LinearBvhNode>, Vec<usize>)> {
    if primitive_bounds.is_empty() {
        return None;
    }

    let mut primitives = primitive_bounds
        .iter()
        .copied()
        .enumerate()
        .map(|(index, bounds)| BuilderPrimitive { index, bounds })
        .collect::<Vec<_>>();
    let root = build_node(&mut primitives, 0, primitive_bounds.len());
    let ordered_indices = primitives
        .into_iter()
        .map(|primitive| primitive.index)
        .collect();
    let mut nodes = Vec::new();
    flatten_node(&root, &mut nodes);

    Some((nodes, ordered_indices))
}

fn build_node(primitives: &mut [BuilderPrimitive], start: usize, end: usize) -> BuildNode {
    let bounds = bounds_of_primitives(&primitives[start..end]);
    let primitive_count = end - start;

    if primitive_count <= 1 {
        return BuildNode::Leaf {
            bounds,
            primitive_offset: start,
            primitive_count,
        };
    }

    let leaf_cost = primitive_count as f32;
    let parent_surface_area = bounds.surface_area().max(1.0e-12);
    let mut best_cost = f32::INFINITY;
    let mut best_split = None;
    let mut best_order = Vec::new();

    for axis in 0..3 {
        let mut sorted = primitives[start..end].to_vec();
        sorted.sort_by(|left, right| {
            component(left.bounds.center(), axis)
                .partial_cmp(&component(right.bounds.center(), axis))
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        if let Some((split_index, cost)) = evaluate_split(&sorted, parent_surface_area) {
            if cost < best_cost {
                best_cost = cost;
                best_split = Some(start + split_index);
                best_order = sorted;
            }
        }
    }

    if best_cost >= leaf_cost || best_split.is_none() {
        return BuildNode::Leaf {
            bounds,
            primitive_offset: start,
            primitive_count,
        };
    }

    primitives[start..end].copy_from_slice(&best_order);
    let mid = best_split.expect("split index must exist when a split is selected");
    let left = build_node(primitives, start, mid);
    let right = build_node(primitives, mid, end);

    BuildNode::Interior {
        bounds,
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn evaluate_split(sorted: &[BuilderPrimitive], parent_surface_area: f32) -> Option<(usize, f32)> {
    if sorted.len() <= 1 {
        return None;
    }

    let mut left_bounds = vec![sorted[0].bounds; sorted.len()];
    let mut right_bounds = vec![sorted[sorted.len() - 1].bounds; sorted.len()];

    for index in 1..sorted.len() {
        left_bounds[index] = left_bounds[index - 1].union(sorted[index].bounds);
    }
    for index in (0..sorted.len() - 1).rev() {
        right_bounds[index] = right_bounds[index + 1].union(sorted[index].bounds);
    }

    let mut best_cost = f32::INFINITY;
    let mut best_split = 0;

    for split_index in 1..sorted.len() {
        let left_area = left_bounds[split_index - 1].surface_area();
        let right_area = right_bounds[split_index].surface_area();
        let left_count = split_index as f32;
        let right_count = (sorted.len() - split_index) as f32;
        let cost = 1.0 + (left_count * left_area + right_count * right_area) / parent_surface_area;

        if cost < best_cost {
            best_cost = cost;
            best_split = split_index;
        }
    }

    Some((best_split, best_cost))
}

fn flatten_node(node: &BuildNode, nodes: &mut Vec<LinearBvhNode>) -> usize {
    let node_index = nodes.len();
    nodes.push(LinearBvhNode::Leaf {
        bounds: Bounds::EMPTY,
        primitive_offset: 0,
        primitive_count: 0,
    });

    match node {
        BuildNode::Leaf {
            bounds,
            primitive_offset,
            primitive_count,
        } => {
            nodes[node_index] = LinearBvhNode::Leaf {
                bounds: *bounds,
                primitive_offset: *primitive_offset,
                primitive_count: *primitive_count,
            };
        }
        BuildNode::Interior {
            bounds,
            left,
            right,
        } => {
            flatten_node(left, nodes);
            let right_child_offset = flatten_node(right, nodes);
            nodes[node_index] = LinearBvhNode::Interior {
                bounds: *bounds,
                right_child_offset,
            };
        }
    }

    node_index
}

fn bounds_of_primitives(primitives: &[BuilderPrimitive]) -> Bounds {
    let mut bounds = primitives[0].bounds;

    for primitive in &primitives[1..] {
        bounds = bounds.union(primitive.bounds);
    }

    bounds
}

fn component(value: glam::Vec3, axis: usize) -> f32 {
    match axis {
        0 => value.x,
        1 => value.y,
        _ => value.z,
    }
}
