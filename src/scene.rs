use glam::{Mat3, Mat4, Vec2, Vec3};
use std::fmt;

use crate::{
    bvh::{LinearBvhNode, SceneBvh, build_scene_bvh, intersect_bounds},
    material::{Material, ShadingVertex},
    math::{
        OrthonormalBasis, compute_surface_partials, face_forward, interpolate_vec2,
        interpolate_vec3,
    },
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
pub struct AreaLightTriangle {
    pub triangle: TriangleRef,
    pub area: f32,
    pub weight: f32,
    pub prefix_weight: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrianglePointSample {
    pub triangle: TriangleRef,
    pub barycentric: Vec3,
    pub p: Vec3,
    pub pdf_area: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AreaLightPointSample {
    pub triangle: TriangleRef,
    pub barycentric: Vec3,
    pub p: Vec3,
    pub triangle_selection_probability: f32,
    pub pdf_area: f32,
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
    pub area_light_triangles: Vec<AreaLightTriangle>,
    pub area_light_weight_sum: f32,
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
        self.register_area_light_triangles(instance_index);
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

    pub fn triangle_uvs(&self, triangle: TriangleRef) -> [Vec2; 3] {
        let instance = self.instances[triangle.instance_index.0];
        self.meshes[instance.mesh_index.0].triangle_uvs(triangle.triangle_index)
    }

    pub fn triangle_positions(&self, triangle: TriangleRef) -> [Vec3; 3] {
        let instance = self.instances[triangle.instance_index.0];
        let positions =
            self.meshes[instance.mesh_index.0].triangle_positions(triangle.triangle_index);

        positions.map(|position| instance.local_to_world.transform_point3(position))
    }

    pub fn triangle_area(&self, triangle: TriangleRef) -> f32 {
        let [p0, p1, p2] = self.triangle_positions(triangle);
        0.5 * (p1 - p0).cross(p2 - p0).length()
    }

    pub fn material(&self, material_index: MaterialIndex) -> &Material {
        &self.materials[material_index.0]
    }

    pub fn instance_material(&self, instance_index: InstanceIndex) -> &Material {
        let material_index = self.instances[instance_index.0].material_index;
        self.material(material_index)
    }

    pub fn shading_vertex_from_triangle_sample(
        &self,
        triangle: TriangleRef,
        barycentric: Vec3,
        incident_direction: Vec3,
    ) -> ShadingVertex {
        let [p0, p1, p2] = self.triangle_positions(triangle);
        let [n0, n1, n2] = self.triangle_normals(triangle);
        let [uv0, uv1, uv2] = self.triangle_uvs(triangle);
        let p = interpolate_vec3(barycentric, p0, p1, p2);
        let uv = interpolate_vec2(barycentric, uv0, uv1, uv2);
        let geometric_normal = (p1 - p0).cross(p2 - p0).normalize_or_zero();
        let shading_normal = interpolate_vec3(barycentric, n0, n1, n2).normalize_or_zero();
        let front_face = geometric_normal.dot(-incident_direction) >= 0.0;
        let ng = face_forward(geometric_normal, -incident_direction);
        let ns = face_forward(
            if shading_normal.length_squared() > 0.0 {
                shading_normal
            } else {
                ng
            },
            ng,
        );
        let (mut dpdu, mut dpdv) = compute_surface_partials([p0, p1, p2], [uv0, uv1, uv2])
            .unwrap_or_else(|| {
                let frame = OrthonormalBasis::from_normal(ns);
                (frame.tangent(), frame.bitangent())
            });
        let frame = OrthonormalBasis::from_normal_and_tangent(ns, dpdu);

        if dpdu.length_squared() == 0.0 {
            dpdu = frame.tangent();
        }
        if dpdv.length_squared() == 0.0 {
            dpdv = frame.bitangent();
        }

        ShadingVertex {
            triangle,
            p,
            uv,
            ng,
            ns,
            dpdu,
            dpdv,
            frame,
            front_face,
        }
    }

    pub fn shading_vertex(&self, hit: SceneHit, incident_direction: Vec3) -> ShadingVertex {
        self.shading_vertex_from_triangle_sample(hit.triangle, hit.barycentric, incident_direction)
    }

    pub fn area_light_triangle_probability(&self, triangle: TriangleRef) -> Option<f32> {
        let area_light = self
            .area_light_triangles
            .iter()
            .find(|area_light| area_light.triangle == triangle)?;

        if self.area_light_weight_sum <= 0.0 {
            return None;
        }

        Some(area_light.weight / self.area_light_weight_sum)
    }

    pub fn area_light_pdf_area(&self, triangle: TriangleRef) -> Option<f32> {
        let area_light = self
            .area_light_triangles
            .iter()
            .find(|area_light| area_light.triangle == triangle)?;
        let triangle_selection_probability = self.area_light_triangle_probability(triangle)?;

        if area_light.area <= 0.0 {
            return None;
        }

        Some(triangle_selection_probability / area_light.area)
    }

    pub fn sample_triangle_point(&self, triangle: TriangleRef, us: Vec2) -> TrianglePointSample {
        let [p0, p1, p2] = self.triangle_positions(triangle);
        let su0 = us.x.clamp(0.0, 1.0).sqrt();
        let u1 = us.y.clamp(0.0, 1.0);
        let barycentric = Vec3::new(1.0 - su0, u1 * su0, (1.0 - u1) * su0);
        let p = interpolate_vec3(barycentric, p0, p1, p2);
        let area = self.triangle_area(triangle);
        let pdf_area = if area > 0.0 { 1.0 / area } else { 0.0 };

        TrianglePointSample {
            triangle,
            barycentric,
            p,
            pdf_area,
        }
    }

    pub fn sample_area_light_point(
        &self,
        u_triangle: f32,
        us: Vec2,
    ) -> Option<AreaLightPointSample> {
        let area_light = self.sample_area_light_triangle(u_triangle)?;
        let triangle_selection_probability = area_light.weight / self.area_light_weight_sum;
        let triangle_sample = self.sample_triangle_point(area_light.triangle, us);

        Some(AreaLightPointSample {
            triangle: triangle_sample.triangle,
            barycentric: triangle_sample.barycentric,
            p: triangle_sample.p,
            triangle_selection_probability,
            pdf_area: triangle_selection_probability / area_light.area,
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

    fn register_area_light_triangles(&mut self, instance_index: InstanceIndex) {
        let material = self.instance_material(instance_index);
        if !material.may_emit() {
            return;
        }

        let max_emission = material.max_emission();
        if max_emission <= 0.0 {
            return;
        }

        let mesh_index = self.instances[instance_index.0].mesh_index;
        let triangle_count = self.meshes[mesh_index.0].triangle_count();

        for triangle_index in 0..triangle_count {
            let triangle = TriangleRef {
                instance_index,
                triangle_index,
            };
            let area = self.triangle_area(triangle);
            let weight = area * max_emission;

            if weight <= 0.0 {
                continue;
            }

            self.area_light_weight_sum += weight;
            self.area_light_triangles.push(AreaLightTriangle {
                triangle,
                area,
                weight,
                prefix_weight: self.area_light_weight_sum,
            });
        }
    }

    fn sample_area_light_triangle(&self, u_triangle: f32) -> Option<&AreaLightTriangle> {
        if self.area_light_weight_sum <= 0.0 {
            return None;
        }

        let target_weight = u_triangle.clamp(0.0, 1.0) * self.area_light_weight_sum;
        let index = self
            .area_light_triangles
            .partition_point(|area_light| area_light.prefix_weight < target_weight)
            .min(self.area_light_triangles.len().checked_sub(1)?);

        self.area_light_triangles.get(index)
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
    use glam::{Mat4, Vec2, Vec3};

    use super::{
        AreaLightTriangle, ClosestHitError, InstanceIndex, MaterialIndex, Scene, SceneHit,
        TriangleRef,
    };
    use crate::{
        material::{EmissiveMaterial, Material, NormalizedLambertMaterial},
        mesh::{Mesh, Vertex},
        ray::Ray,
    };

    fn unit_mesh(z: f32) -> Mesh {
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

    fn stacked_mesh() -> Mesh {
        Mesh::new(
            vec![
                Vertex {
                    position: Vec3::new(0.0, 0.0, 0.0),
                    normal: Vec3::Z,
                    uv: Vec2::ZERO,
                },
                Vertex {
                    position: Vec3::new(1.0, 0.0, 0.0),
                    normal: Vec3::Z,
                    uv: Vec2::X,
                },
                Vertex {
                    position: Vec3::new(0.0, 1.0, 0.0),
                    normal: Vec3::Z,
                    uv: Vec2::Y,
                },
                Vertex {
                    position: Vec3::new(0.0, 0.0, -1.0),
                    normal: Vec3::Z,
                    uv: Vec2::ZERO,
                },
                Vertex {
                    position: Vec3::new(1.0, 0.0, -1.0),
                    normal: Vec3::Z,
                    uv: Vec2::X,
                },
                Vertex {
                    position: Vec3::new(0.0, 1.0, -1.0),
                    normal: Vec3::Z,
                    uv: Vec2::Y,
                },
            ],
            vec![0, 1, 2, 3, 4, 5],
        )
    }

    fn default_material(scene: &mut Scene) -> MaterialIndex {
        scene.add_material(Material::NormalizedLambert(NormalizedLambertMaterial::new(
            Vec3::splat(0.5),
        )))
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
        let material_index =
            scene.add_material(Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 12.0)));
        scene.add_instance(mesh_index, material_index, Mat4::IDENTITY);

        assert_eq!(
            scene.instance_material(InstanceIndex(0)),
            &Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 12.0))
        );
    }

    #[test]
    fn shading_vertex_interpolates_surface_data() {
        let mut scene = Scene::new();
        let mesh_index = scene.add_mesh(unit_mesh(0.0));
        let material_index = default_material(&mut scene);
        scene.add_instance(mesh_index, material_index, Mat4::IDENTITY);
        let hit = SceneHit {
            triangle: TriangleRef {
                instance_index: InstanceIndex(0),
                triangle_index: 0,
            },
            t: 1.0,
            barycentric: Vec3::new(0.5, 0.25, 0.25),
        };

        let shading_vertex = scene.shading_vertex(hit, Vec3::NEG_Z);

        assert!(
            shading_vertex
                .p
                .abs_diff_eq(Vec3::new(0.25, 0.25, 0.0), 1.0e-6)
        );
        assert_eq!(
            shading_vertex.triangle,
            TriangleRef {
                instance_index: InstanceIndex(0),
                triangle_index: 0,
            }
        );
        assert!(shading_vertex.uv.abs_diff_eq(Vec2::new(0.25, 0.25), 1.0e-6));
        assert!(shading_vertex.ng.abs_diff_eq(Vec3::Z, 1.0e-6));
        assert!(shading_vertex.ns.abs_diff_eq(Vec3::Z, 1.0e-6));
        assert!(shading_vertex.frame.normal().abs_diff_eq(Vec3::Z, 1.0e-6));
        assert!(shading_vertex.front_face);
        assert!(shading_vertex.dpdu.length_squared() > 0.0);
        assert!(shading_vertex.dpdv.length_squared() > 0.0);
    }

    #[test]
    fn emissive_instance_populates_area_light_distribution() {
        let mut scene = Scene::new();
        let mesh_index = scene.add_mesh(unit_mesh(0.0));
        let material_index =
            scene.add_material(Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 12.0)));

        scene.add_instance(mesh_index, material_index, Mat4::IDENTITY);

        assert_eq!(
            scene.area_light_triangles,
            vec![AreaLightTriangle {
                triangle: TriangleRef {
                    instance_index: InstanceIndex(0),
                    triangle_index: 0,
                },
                area: 0.5,
                weight: 6.0,
                prefix_weight: 6.0,
            }]
        );
        assert_eq!(scene.area_light_weight_sum, 6.0);
    }

    #[test]
    fn sample_triangle_point_returns_barycentric_point_and_area_pdf() {
        let mut scene = Scene::new();
        let mesh_index = scene.add_mesh(unit_mesh(0.0));
        let material_index = default_material(&mut scene);
        scene.add_instance(mesh_index, material_index, Mat4::IDENTITY);

        let sample = scene.sample_triangle_point(
            TriangleRef {
                instance_index: InstanceIndex(0),
                triangle_index: 0,
            },
            Vec2::new(1.0, 0.25),
        );

        assert!(
            sample
                .barycentric
                .abs_diff_eq(Vec3::new(0.0, 0.25, 0.75), 1.0e-6)
        );
        assert!(sample.p.abs_diff_eq(Vec3::new(0.25, 0.75, 0.0), 1.0e-6));
        assert!((sample.pdf_area - 2.0).abs() < 1.0e-6);
    }

    #[test]
    fn sample_area_light_point_uses_triangle_weight_distribution() {
        let mut scene = Scene::new();
        let mesh_index = scene.add_mesh(unit_mesh(0.0));
        let material_index =
            scene.add_material(Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 10.0)));
        scene.add_instance(mesh_index, material_index, Mat4::IDENTITY);
        scene.add_instance(
            mesh_index,
            material_index,
            Mat4::from_scale(Vec3::splat(2.0)),
        );

        let first_sample = scene
            .sample_area_light_point(0.1, Vec2::new(0.0, 0.0))
            .expect("expected area light sample");
        let second_sample = scene
            .sample_area_light_point(0.9, Vec2::new(0.0, 0.0))
            .expect("expected area light sample");

        assert_eq!(
            first_sample.triangle,
            TriangleRef {
                instance_index: InstanceIndex(0),
                triangle_index: 0,
            }
        );
        assert_eq!(
            second_sample.triangle,
            TriangleRef {
                instance_index: InstanceIndex(1),
                triangle_index: 0,
            }
        );
        assert!((first_sample.triangle_selection_probability - 0.2).abs() < 1.0e-6);
        assert!((second_sample.triangle_selection_probability - 0.8).abs() < 1.0e-6);
        assert!((first_sample.pdf_area - 0.4).abs() < 1.0e-6);
        assert!((second_sample.pdf_area - 0.4).abs() < 1.0e-6);
    }
}
