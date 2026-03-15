use glam::{Mat3, Mat4, Quat, Vec3};

use crate::{
    mesh::{Bounds, Mesh},
    ray::{Ray, intersect_triangle},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshIndex(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstanceIndex(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TriangleRef {
    pub instance_index: InstanceIndex,
    pub triangle_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Instance {
    pub mesh_index: MeshIndex,
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
    pub instances: Vec<Instance>,
    pub triangles: Vec<TriangleRef>,
}

impl Scene {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_mesh(&mut self, mesh: Mesh) -> MeshIndex {
        let mesh_index = MeshIndex(self.meshes.len());
        self.meshes.push(mesh);
        mesh_index
    }

    pub fn add_instance(
        &mut self,
        mesh_index: MeshIndex,
        translation: Vec3,
        rotation: Quat,
        scale: Vec3,
    ) -> InstanceIndex {
        let mesh = &self.meshes[mesh_index.0];
        let pivot = Vec3::new(
            mesh.bounds.center().x,
            mesh.bounds.min.y,
            mesh.bounds.center().z,
        );
        let local_to_world = Mat4::from_translation(translation)
            * Mat4::from_quat(rotation)
            * Mat4::from_scale(scale)
            * Mat4::from_translation(-pivot);
        let world_to_local = local_to_world.inverse();
        let normal_to_world = Mat3::from_mat4(world_to_local.transpose());
        let world_bounds = transform_bounds(mesh.bounds, local_to_world);
        let instance_index = InstanceIndex(self.instances.len());
        let triangle_count = mesh.triangle_count();

        self.instances.push(Instance {
            mesh_index,
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

        instance_index
    }

    pub fn closest_hit(&self, ray: &Ray) -> Option<SceneHit> {
        let mut closest_world_t = f32::INFINITY;
        let mut closest_hit = None;
        let mut current_instance_index = None;
        let mut current_local_ray = *ray;

        for triangle in &self.triangles {
            if current_instance_index != Some(triangle.instance_index) {
                let instance = self.instances[triangle.instance_index.0];
                current_local_ray = ray.transformed(instance.world_to_local);
                current_instance_index = Some(triangle.instance_index);
            }

            let instance = self.instances[triangle.instance_index.0];
            let mesh = &self.meshes[instance.mesh_index.0];
            let [v0, v1, v2] = mesh.triangle_positions(triangle.triangle_index);

            if let Some(hit) = intersect_triangle(&current_local_ray, f32::INFINITY, v0, v1, v2) {
                let local_hit_position = current_local_ray.at(hit.t);
                let world_hit_position =
                    instance.local_to_world.transform_point3(local_hit_position);
                let world_t = (world_hit_position - ray.origin).dot(ray.direction)
                    / ray.direction.length_squared();

                if world_t > 0.0 && world_t < closest_world_t {
                    closest_world_t = world_t;
                    closest_hit = Some(SceneHit {
                        triangle: *triangle,
                        t: world_t,
                        barycentric: hit.barycentric,
                    });
                }
            }
        }

        closest_hit
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
    use glam::{Quat, Vec3};

    use super::{InstanceIndex, Scene, TriangleRef};
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

    #[test]
    fn add_instance_populates_triangle_refs() {
        let mut scene = Scene::new();
        let mesh_index = scene.add_mesh(unit_mesh(0.0));
        scene.add_instance(mesh_index, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE);
        scene.add_instance(
            mesh_index,
            Vec3::new(0.0, 0.0, 1.0),
            Quat::IDENTITY,
            Vec3::ONE,
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
        scene.add_instance(mesh_index, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE);
        scene.add_instance(
            mesh_index,
            Vec3::new(0.0, 0.0, -1.0),
            Quat::IDENTITY,
            Vec3::ONE,
        );

        let ray = Ray::new(Vec3::new(0.25, 0.25, 2.0), Vec3::NEG_Z);
        let hit = scene.closest_hit(&ray).expect("expected hit");

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
        scene.add_instance(mesh_index, Vec3::ZERO, Quat::IDENTITY, Vec3::splat(2.0));

        let ray = Ray::new(Vec3::new(0.5, 0.5, 1.0), Vec3::NEG_Z);
        let hit = scene.closest_hit(&ray).expect("expected hit");

        assert!((hit.t - 1.0).abs() < 1.0e-6);
    }
}
