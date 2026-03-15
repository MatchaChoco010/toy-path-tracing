use glam::{Mat3, Mat4, Vec3};
use std::fmt;

use crate::{
    bvh::{LinearBvhNode, SceneBvh, build_scene_bvh, intersect_bounds},
    mesh::{Bounds, Mesh},
    ray::{Ray, intersect_triangle},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshIndex(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstanceIndex(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialIndex(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TriangleRef {
    pub instance_index: InstanceIndex,
    pub triangle_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Material {
    Diffuse { rho: Vec3 },
    Emissive { color: Vec3, strength: f32 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Instance {
    pub mesh_index: MeshIndex,
    pub material_index: MaterialIndex,
    pub local_to_world: Mat4,
    pub world_to_local: Mat4,
    pub normal_to_world: Mat3,
    pub world_bounds: Bounds,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneHit {
    pub triangle: TriangleRef,
    pub t: f32,
    pub barycentric: Vec3,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Scene {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub instances: Vec<Instance>,
    pub triangles: Vec<TriangleRef>,
    pub bvh: Option<SceneBvh>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClosestHitError {
    BvhNotBuilt,
}

impl fmt::Display for ClosestHitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BvhNotBuilt => write!(f, "scene BVH has not been built yet"),
        }
    }
}

impl std::error::Error for ClosestHitError {}

#[derive(Debug, Clone, Copy, PartialEq)]
struct MeshHit {
    triangle_index: usize,
    t: f32,
    barycentric: Vec3,
}

impl Scene {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_mesh(&mut self, mesh: Mesh) -> MeshIndex {
        let mesh_index = MeshIndex(self.meshes.len());
        self.meshes.push(mesh);
        self.bvh = None;
        mesh_index
    }

    pub fn add_material(&mut self, material: Material) -> MaterialIndex {
        let material_index = MaterialIndex(self.materials.len());
        self.materials.push(material);
        material_index
    }

    pub fn add_instance(
        &mut self,
        mesh_index: MeshIndex,
        material_index: MaterialIndex,
        local_to_world: Mat4,
    ) -> InstanceIndex {
        let mesh = &self.meshes[mesh_index.0];
        let world_to_local = local_to_world.inverse();
        let normal_to_world = Mat3::from_mat4(world_to_local.transpose());
        let world_bounds = transform_bounds(mesh.bounds, local_to_world);
        let instance_index = InstanceIndex(self.instances.len());
        let triangle_count = mesh.triangle_count();

        self.instances.push(Instance {
            mesh_index,
            material_index,
            local_to_world,
            world_to_local,
            normal_to_world,
            world_bounds,
        });
        self.triangles
            .extend((0..triangle_count).map(|triangle_index| TriangleRef {
                instance_index,
                triangle_index,
            }));
        self.bvh = None;

        instance_index
    }

    pub fn build_bvh(&mut self) {
        for mesh in &mut self.meshes {
            mesh.build_bvh();
        }

        let instance_bounds = self
            .instances
            .iter()
            .map(|instance| instance.world_bounds)
            .collect::<Vec<_>>();
        self.bvh = build_scene_bvh(&instance_bounds);
    }

    pub fn closest_hit(&self, ray: &Ray) -> Result<Option<SceneHit>, ClosestHitError> {
        let bvh = self.bvh.as_ref().ok_or(ClosestHitError::BvhNotBuilt)?;
        let mut closest_world_t = f32::INFINITY;
        let mut closest_hit = None;
        let mut stack = vec![0_usize];

        while let Some(node_index) = stack.pop() {
            let node = bvh.nodes[node_index];
            if intersect_bounds(ray, closest_world_t, node.bounds()).is_none() {
                continue;
            }

            match node {
                LinearBvhNode::Leaf {
                    primitive_offset,
                    primitive_count,
                    ..
                } => {
                    for ordered_index in primitive_offset..primitive_offset + primitive_count {
                        let instance_index = InstanceIndex(bvh.instance_indices[ordered_index]);
                        let instance = self.instances[instance_index.0];
                        let local_ray = ray.transformed(instance.world_to_local);
                        let mesh = &self.meshes[instance.mesh_index.0];

                        if let Some(mesh_hit) = closest_mesh_hit(mesh, &local_ray, closest_world_t)
                        {
                            closest_world_t = mesh_hit.t;
                            closest_hit = Some(SceneHit {
                                triangle: TriangleRef {
                                    instance_index,
                                    triangle_index: mesh_hit.triangle_index,
                                },
                                t: mesh_hit.t,
                                barycentric: mesh_hit.barycentric,
                            });
                        }
                    }
                }
                LinearBvhNode::Interior {
                    right_child_offset, ..
                } => {
                    let left_child_index = node_index + 1;
                    let right_child_index = right_child_offset;
                    let left_hit = intersect_bounds(
                        ray,
                        closest_world_t,
                        bvh.nodes[left_child_index].bounds(),
                    );
                    let right_hit = intersect_bounds(
                        ray,
                        closest_world_t,
                        bvh.nodes[right_child_index].bounds(),
                    );

                    match (left_hit, right_hit) {
                        (Some(left_t), Some(right_t)) => {
                            if left_t <= right_t {
                                stack.push(right_child_index);
                                stack.push(left_child_index);
                            } else {
                                stack.push(left_child_index);
                                stack.push(right_child_index);
                            }
                        }
                        (Some(_), None) => stack.push(left_child_index),
                        (None, Some(_)) => stack.push(right_child_index),
                        (None, None) => {}
                    }
                }
            }
        }

        Ok(closest_hit)
    }

    pub fn triangle_normals(&self, triangle: TriangleRef) -> [Vec3; 3] {
        let instance = self.instances[triangle.instance_index.0];
        let normals = self.meshes[instance.mesh_index.0].triangle_normals(triangle.triangle_index);

        normals.map(|normal| {
            instance
                .normal_to_world
                .mul_vec3(normal)
                .normalize_or_zero()
        })
    }

    pub fn triangle_positions(&self, triangle: TriangleRef) -> [Vec3; 3] {
        let instance = self.instances[triangle.instance_index.0];
        let positions =
            self.meshes[instance.mesh_index.0].triangle_positions(triangle.triangle_index);

        positions.map(|position| instance.local_to_world.transform_point3(position))
    }

    pub fn material(&self, material_index: MaterialIndex) -> Material {
        self.materials[material_index.0]
    }

    pub fn instance_material(&self, instance_index: InstanceIndex) -> Material {
        let material_index = self.instances[instance_index.0].material_index;
        self.material(material_index)
    }

    pub fn bounds(&self) -> Option<Bounds> {
        let mut instances = self.instances.iter();
        let first = instances.next()?;
        let mut bounds = first.world_bounds;

        for instance in instances {
            bounds = bounds.union(instance.world_bounds);
        }

        Some(bounds)
    }
}

fn closest_mesh_hit(mesh: &Mesh, ray: &Ray, t_max: f32) -> Option<MeshHit> {
    let bvh = mesh
        .bvh
        .as_ref()
        .expect("mesh BVH must be built before traversal");
    let mut closest_t = t_max;
    let mut closest_hit = None;
    let mut stack = vec![0_usize];

    while let Some(node_index) = stack.pop() {
        let node = bvh.nodes[node_index];
        if intersect_bounds(ray, closest_t, node.bounds()).is_none() {
            continue;
        }

        match node {
            LinearBvhNode::Leaf {
                primitive_offset,
                primitive_count,
                ..
            } => {
                for ordered_index in primitive_offset..primitive_offset + primitive_count {
                    let triangle_index = bvh.triangle_indices[ordered_index];
                    let [v0, v1, v2] = mesh.triangle_positions(triangle_index);

                    if let Some(hit) = intersect_triangle(ray, closest_t, v0, v1, v2) {
                        closest_t = hit.t;
                        closest_hit = Some(MeshHit {
                            triangle_index,
                            t: hit.t,
                            barycentric: hit.barycentric,
                        });
                    }
                }
            }
            LinearBvhNode::Interior {
                right_child_offset, ..
            } => {
                let left_child_index = node_index + 1;
                let right_child_index = right_child_offset;
                let left_hit =
                    intersect_bounds(ray, closest_t, bvh.nodes[left_child_index].bounds());
                let right_hit =
                    intersect_bounds(ray, closest_t, bvh.nodes[right_child_index].bounds());

                match (left_hit, right_hit) {
                    (Some(left_t), Some(right_t)) => {
                        if left_t <= right_t {
                            stack.push(right_child_index);
                            stack.push(left_child_index);
                        } else {
                            stack.push(left_child_index);
                            stack.push(right_child_index);
                        }
                    }
                    (Some(_), None) => stack.push(left_child_index),
                    (None, Some(_)) => stack.push(right_child_index),
                    (None, None) => {}
                }
            }
        }
    }

    closest_hit
}

fn transform_bounds(bounds: Bounds, transform: Mat4) -> Bounds {
    let mut corners = [
        Vec3::new(bounds.min.x, bounds.min.y, bounds.min.z),
        Vec3::new(bounds.min.x, bounds.min.y, bounds.max.z),
        Vec3::new(bounds.min.x, bounds.max.y, bounds.min.z),
        Vec3::new(bounds.min.x, bounds.max.y, bounds.max.z),
        Vec3::new(bounds.max.x, bounds.min.y, bounds.min.z),
        Vec3::new(bounds.max.x, bounds.min.y, bounds.max.z),
        Vec3::new(bounds.max.x, bounds.max.y, bounds.min.z),
        Vec3::new(bounds.max.x, bounds.max.y, bounds.max.z),
    ]
    .into_iter()
    .map(|corner| transform.transform_point3(corner));

    let first = corners.next().expect("bounds must have corners");
    let mut min = first;
    let mut max = first;

    for corner in corners {
        min = min.min(corner);
        max = max.max(corner);
    }

    Bounds { min, max }
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Vec3};

    use super::{ClosestHitError, InstanceIndex, Material, Scene, TriangleRef};
    use crate::{
        mesh::{Mesh, Vertex},
        ray::Ray,
    };

    fn unit_mesh(z: f32) -> Mesh {
        Mesh::new(
            vec![
                Vertex {
                    position: Vec3::new(0.0, 0.0, z),
                    normal: Vec3::Z,
                },
                Vertex {
                    position: Vec3::new(1.0, 0.0, z),
                    normal: Vec3::Z,
                },
                Vertex {
                    position: Vec3::new(0.0, 1.0, z),
                    normal: Vec3::Z,
                },
            ],
            vec![0, 1, 2],
        )
    }

    fn stacked_mesh() -> Mesh {
        Mesh::new(
            vec![
                Vertex {
                    position: Vec3::new(0.0, 0.0, 0.0),
                    normal: Vec3::Z,
                },
                Vertex {
                    position: Vec3::new(1.0, 0.0, 0.0),
                    normal: Vec3::Z,
                },
                Vertex {
                    position: Vec3::new(0.0, 1.0, 0.0),
                    normal: Vec3::Z,
                },
                Vertex {
                    position: Vec3::new(0.0, 0.0, -1.0),
                    normal: Vec3::Z,
                },
                Vertex {
                    position: Vec3::new(1.0, 0.0, -1.0),
                    normal: Vec3::Z,
                },
                Vertex {
                    position: Vec3::new(0.0, 1.0, -1.0),
                    normal: Vec3::Z,
                },
            ],
            vec![0, 1, 2, 3, 4, 5],
        )
    }

    fn default_material(scene: &mut Scene) -> super::MaterialIndex {
        scene.add_material(Material::Diffuse {
            rho: Vec3::splat(0.5),
        })
    }

    #[test]
    fn add_instance_populates_triangle_refs() {
        let mut scene = Scene::new();
        let mesh_index = scene.add_mesh(unit_mesh(0.0));
        let material_index = default_material(&mut scene);
        scene.add_instance(mesh_index, material_index, Mat4::IDENTITY);
        scene.add_instance(
            mesh_index,
            material_index,
            Mat4::from_translation(Vec3::new(0.0, 0.0, 1.0)),
        );

        assert_eq!(
            scene.triangles,
            vec![
                TriangleRef {
                    instance_index: InstanceIndex(0),
                    triangle_index: 0,
                },
                TriangleRef {
                    instance_index: InstanceIndex(1),
                    triangle_index: 0,
                },
            ]
        );
    }

    #[test]
    fn closest_hit_returns_the_nearest_triangle() {
        let mut scene = Scene::new();
        let mesh_index = scene.add_mesh(unit_mesh(0.0));
        let material_index = default_material(&mut scene);
        scene.add_instance(mesh_index, material_index, Mat4::IDENTITY);
        scene.add_instance(
            mesh_index,
            material_index,
            Mat4::from_translation(Vec3::new(0.0, 0.0, -1.0)),
        );
        scene.build_bvh();

        let ray = Ray::new(Vec3::new(0.25, 0.25, 2.0), Vec3::NEG_Z);
        let hit = scene
            .closest_hit(&ray)
            .expect("BVH should be built")
            .expect("expected hit");

        assert_eq!(
            hit.triangle,
            TriangleRef {
                instance_index: InstanceIndex(0),
                triangle_index: 0,
            }
        );
        assert!((hit.t - 2.0).abs() < 1.0e-6);
    }

    #[test]
    fn closest_hit_handles_scaled_instances() {
        let mut scene = Scene::new();
        let mesh_index = scene.add_mesh(unit_mesh(0.0));
        let material_index = default_material(&mut scene);
        scene.add_instance(
            mesh_index,
            material_index,
            Mat4::from_scale(Vec3::splat(2.0)),
        );
        scene.build_bvh();

        let ray = Ray::new(Vec3::new(0.5, 0.5, 1.0), Vec3::NEG_Z);
        let hit = scene
            .closest_hit(&ray)
            .expect("BVH should be built")
            .expect("expected hit");

        assert!((hit.t - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn closest_hit_requires_bvh_build() {
        let mut scene = Scene::new();
        let mesh_index = scene.add_mesh(unit_mesh(0.0));
        let material_index = default_material(&mut scene);
        scene.add_instance(mesh_index, material_index, Mat4::IDENTITY);

        let ray = Ray::new(Vec3::new(0.25, 0.25, 1.0), Vec3::NEG_Z);
        let error = scene
            .closest_hit(&ray)
            .expect_err("expected missing BVH error");

        assert_eq!(error, ClosestHitError::BvhNotBuilt);
    }

    #[test]
    fn build_bvh_populates_scene_and_mesh_bvhs() {
        let mut scene = Scene::new();
        let mesh_index = scene.add_mesh(stacked_mesh());
        let material_index = default_material(&mut scene);
        scene.add_instance(mesh_index, material_index, Mat4::IDENTITY);

        scene.build_bvh();

        assert!(scene.bvh.is_some());
        assert!(scene.meshes[mesh_index.0].bvh.is_some());
    }

    #[test]
    fn closest_hit_returns_none_when_ray_misses_scene() {
        let mut scene = Scene::new();
        let mesh_index = scene.add_mesh(unit_mesh(0.0));
        let material_index = default_material(&mut scene);
        scene.add_instance(mesh_index, material_index, Mat4::IDENTITY);
        scene.build_bvh();

        let ray = Ray::new(Vec3::new(2.0, 2.0, 1.0), Vec3::NEG_Z);
        let hit = scene.closest_hit(&ray).expect("BVH should be built");

        assert!(hit.is_none());
    }

    #[test]
    fn adding_instance_after_build_invalidates_scene_bvh() {
        let mut scene = Scene::new();
        let mesh_index = scene.add_mesh(unit_mesh(0.0));
        let material_index = default_material(&mut scene);
        scene.add_instance(mesh_index, material_index, Mat4::IDENTITY);
        scene.build_bvh();

        scene.add_instance(
            mesh_index,
            material_index,
            Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0)),
        );

        assert!(scene.bvh.is_none());
    }

    #[test]
    fn adding_mesh_after_build_invalidates_scene_bvh() {
        let mut scene = Scene::new();
        let mesh_index = scene.add_mesh(unit_mesh(0.0));
        let material_index = default_material(&mut scene);
        scene.add_instance(mesh_index, material_index, Mat4::IDENTITY);
        scene.build_bvh();

        scene.add_mesh(unit_mesh(-1.0));

        assert!(scene.bvh.is_none());
    }

    #[test]
    fn closest_hit_traverses_multi_triangle_mesh_bvh() {
        let mut scene = Scene::new();
        let mesh_index = scene.add_mesh(stacked_mesh());
        let material_index = default_material(&mut scene);
        scene.add_instance(mesh_index, material_index, Mat4::IDENTITY);
        scene.build_bvh();

        let ray = Ray::new(Vec3::new(0.25, 0.25, 2.0), Vec3::NEG_Z);
        let hit = scene
            .closest_hit(&ray)
            .expect("BVH should be built")
            .expect("expected hit");

        assert_eq!(hit.triangle.triangle_index, 0);
        assert!((hit.t - 2.0).abs() < 1.0e-6);
    }

    #[test]
    fn instance_material_returns_assigned_material() {
        let mut scene = Scene::new();
        let mesh_index = scene.add_mesh(unit_mesh(0.0));
        let material_index = scene.add_material(Material::Emissive {
            color: Vec3::ONE,
            strength: 12.0,
        });
        scene.add_instance(mesh_index, material_index, Mat4::IDENTITY);

        assert_eq!(
            scene.instance_material(InstanceIndex(0)),
            Material::Emissive {
                color: Vec3::ONE,
                strength: 12.0,
            }
        );
    }
}
