use glam::Vec3;

use crate::{
    mesh::Mesh,
    ray::{Ray, intersect_triangle},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TriangleRef {
    pub mesh_index: usize,
    pub triangle_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneHit {
    pub triangle: TriangleRef,
    pub t: f32,
    pub barycentric: Vec3,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    pub min: Vec3,
    pub max: Vec3,
}

impl Bounds {
    pub fn center(&self) -> Vec3 {
        0.5 * (self.min + self.max)
    }

    pub fn extent(&self) -> Vec3 {
        self.max - self.min
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Scene {
    pub meshes: Vec<Mesh>,
    pub triangles: Vec<TriangleRef>,
}

impl Scene {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_mesh(&mut self, mesh: Mesh) {
        let mesh_index = self.meshes.len();
        let triangle_count = mesh.triangle_count();

        self.meshes.push(mesh);
        self.triangles
            .extend((0..triangle_count).map(|triangle_index| TriangleRef {
                mesh_index,
                triangle_index,
            }));
    }

    pub fn closest_hit(&self, ray: &Ray) -> Option<SceneHit> {
        let mut closest_t = f32::INFINITY;
        let mut closest_hit = None;

        for triangle in &self.triangles {
            let [v0, v1, v2] =
                self.meshes[triangle.mesh_index].triangle_positions(triangle.triangle_index);

            if let Some(hit) = intersect_triangle(ray, closest_t, v0, v1, v2) {
                closest_t = hit.t;
                closest_hit = Some(SceneHit {
                    triangle: *triangle,
                    t: hit.t,
                    barycentric: hit.barycentric,
                });
            }
        }

        closest_hit
    }

    pub fn triangle_normals(&self, triangle: TriangleRef) -> [Vec3; 3] {
        self.meshes[triangle.mesh_index].triangle_normals(triangle.triangle_index)
    }

    pub fn bounds(&self) -> Option<Bounds> {
        let mut vertices = self.meshes.iter().flat_map(|mesh| mesh.vertices.iter());
        let first = vertices.next()?;
        let mut min = first.position;
        let mut max = first.position;

        for vertex in vertices {
            min = min.min(vertex.position);
            max = max.max(vertex.position);
        }

        Some(Bounds { min, max })
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::{Scene, TriangleRef};
    use crate::{
        mesh::{Mesh, Vertex},
        ray::Ray,
    };

    fn unit_mesh(z: f32) -> Mesh {
        Mesh {
            vertices: vec![
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
            indices: vec![0, 1, 2],
        }
    }

    #[test]
    fn add_mesh_populates_triangle_refs() {
        let mut scene = Scene::new();
        scene.add_mesh(unit_mesh(0.0));
        scene.add_mesh(unit_mesh(1.0));

        assert_eq!(
            scene.triangles,
            vec![
                TriangleRef {
                    mesh_index: 0,
                    triangle_index: 0,
                },
                TriangleRef {
                    mesh_index: 1,
                    triangle_index: 0,
                },
            ]
        );
    }

    #[test]
    fn closest_hit_returns_the_nearest_triangle() {
        let mut scene = Scene::new();
        scene.add_mesh(unit_mesh(0.0));
        scene.add_mesh(unit_mesh(-1.0));

        let ray = Ray::new(Vec3::new(0.25, 0.25, 2.0), Vec3::NEG_Z);
        let hit = scene.closest_hit(&ray).expect("expected hit");

        assert_eq!(
            hit.triangle,
            TriangleRef {
                mesh_index: 0,
                triangle_index: 0,
            }
        );
        assert!((hit.t - 2.0).abs() < 1.0e-6);
    }
}
