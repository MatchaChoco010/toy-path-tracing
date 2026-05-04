use glam::{Mat3, Mat4, Vec2, Vec3};
use rand::{RngExt, rngs::ThreadRng};
use std::fmt;

use crate::{
    bsdf::DirectionalAlbedoCache,
    light::{
        DirectionalLight, DirectionalLightIndex, EnvironmentLight, LightSampler, PointLight,
        PointLightIndex, SpotLight, SpotLightIndex,
    },
    light_tree::{LightTree, build_light_tree},
    material::{Material, ShadingVertex},
    math::{
        OrthonormalBasis, compute_surface_partials, difference_of_products, face_forward,
        interpolate_vec2, interpolate_vec3,
    },
    mesh::{Bounds, Mesh},
    qbvh::{Qbvh, build_qbvh, traverse_qbvh},
    ray::{Ray, intersect_triangle},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshIndex(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstanceIndex(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialIndex(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    directional_albedo_cache: DirectionalAlbedoCache,
    pub instances: Vec<Instance>,
    pub triangles: Vec<TriangleRef>,
    pub area_light_triangles: Vec<AreaLightTriangle>,
    pub area_light_weight_sum: f32,
    pub qbvh: Option<Qbvh>,
    pub environment_light: Option<EnvironmentLight>,
    pub point_lights: Vec<PointLight>,
    pub directional_lights: Vec<DirectionalLight>,
    pub spot_lights: Vec<SpotLight>,
    pub light_sampler: LightSampler,
    pub light_tree: Option<LightTree>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClosestHitError {
    QbvhNotBuilt,
}

impl fmt::Display for ClosestHitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QbvhNotBuilt => write!(f, "scene QBVH has not been built yet"),
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
        self.qbvh = None;
        mesh_index
    }

    pub fn add_material(&mut self, mut material: Material) -> MaterialIndex {
        if let Material::SimplePBR(simple_pbr) = &mut material {
            let lut = self
                .directional_albedo_cache
                .get_or_build_dielectric_ggx(simple_pbr.eta);
            simple_pbr.install_dielectric_ggx_directional_albedo_lut(lut);
        }
        if let Material::StandardSurface(standard) = &mut material {
            standard.validate_and_warn();
            let spec_eta = standard.requires_specular_eta();
            let coat_eta = standard.requires_coat_eta();
            let spec_lut = self
                .directional_albedo_cache
                .get_or_build_dielectric_ggx(spec_eta);
            let coat_lut = self
                .directional_albedo_cache
                .get_or_build_dielectric_ggx(coat_eta);
            let sheen_lut = self.directional_albedo_cache.get_or_build_sheen();
            standard.install_spec_lut(spec_lut);
            standard.install_coat_lut(coat_lut);
            standard.install_sheen_lut(sheen_lut);
        }

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
        self.light_tree = None;
        self.rebuild_light_sampler();
        self.qbvh = None;

        instance_index
    }

    pub fn set_environment_light(&mut self, light: EnvironmentLight) {
        self.environment_light = Some(light);
        self.rebuild_light_sampler();
    }

    pub fn clear_environment_light(&mut self) {
        self.environment_light = None;
        self.rebuild_light_sampler();
    }

    pub fn environment_light(&self) -> Option<&EnvironmentLight> {
        self.environment_light.as_ref()
    }

    pub fn add_point_light(&mut self, light: PointLight) -> PointLightIndex {
        let index = PointLightIndex(self.point_lights.len());
        self.point_lights.push(light);
        self.light_tree = None;
        self.rebuild_light_sampler();
        index
    }

    pub fn add_directional_light(&mut self, light: DirectionalLight) -> DirectionalLightIndex {
        let index = DirectionalLightIndex(self.directional_lights.len());
        self.directional_lights.push(light);
        self.rebuild_light_sampler();
        index
    }

    pub fn add_spot_light(&mut self, light: SpotLight) -> SpotLightIndex {
        let index = SpotLightIndex(self.spot_lights.len());
        self.spot_lights.push(light);
        self.light_tree = None;
        self.rebuild_light_sampler();
        index
    }

    pub fn rebuild_light_sampler(&mut self) {
        self.light_sampler = LightSampler::build_from_scene(self);
    }

    /// Build the SG hierarchical light tree
    /// [Tokuyoshi et al. 2024] from the scene's emissive triangles, point and
    /// spot lights, then rebuild the top-level `LightSampler` so that its
    /// `Tree` entry weight reflects the new tree's root flux.
    pub fn build_light_tree(&mut self) {
        self.light_tree = build_light_tree(self);
        self.rebuild_light_sampler();
    }

    pub fn build_qbvh(&mut self) {
        for mesh in &mut self.meshes {
            mesh.build_qbvh();
        }

        let instance_bounds = self
            .instances
            .iter()
            .map(|instance| instance.world_bounds)
            .collect::<Vec<_>>();
        self.qbvh = build_qbvh(&instance_bounds);
    }

    pub fn closest_hit(
        &self,
        ray: &Ray,
        rng: &mut ThreadRng,
    ) -> Result<Option<SceneHit>, ClosestHitError> {
        if self.instances.is_empty() {
            return Ok(None);
        }
        let qbvh = self.qbvh.as_ref().ok_or(ClosestHitError::QbvhNotBuilt)?;
        let mut closest_hit: Option<SceneHit> = None;

        traverse_qbvh(qbvh, ray, f32::INFINITY, |offset, count, current_t_max| {
            let mut t_max = current_t_max;
            for ordered_index in offset..offset + count {
                let instance_index = InstanceIndex(qbvh.primitive_indices[ordered_index as usize]);
                let instance = self.instances[instance_index.0];
                let local_ray = ray.transformed(instance.world_to_local);
                let mesh = &self.meshes[instance.mesh_index.0];
                let material = self.instance_material(instance_index);
                let has_alpha_test = material.has_alpha_test();

                let mesh_hit = closest_mesh_hit_with_filter(mesh, &local_ray, t_max, |hit| {
                    if !has_alpha_test {
                        return true;
                    }
                    let triangle = TriangleRef {
                        instance_index,
                        triangle_index: hit.triangle_index,
                    };
                    let shading_vertex =
                        self.alpha_test_shading_vertex(triangle, hit.barycentric, ray);
                    let u: f32 = rng.random();
                    material.any_hit(&shading_vertex, u)
                });

                if let Some(mesh_hit) = mesh_hit {
                    t_max = mesh_hit.t;
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
            t_max
        });

        Ok(closest_hit)
    }

    /// Builds a [`ShadingVertex`] suitable for an `any_hit` query without
    /// running any material-specific normal-mapping. Use this only inside
    /// the alpha-test path; downstream shading should still go through
    /// [`Self::shading_vertex`] so the BSDF sees the prepared vertex.
    fn alpha_test_shading_vertex(
        &self,
        triangle: TriangleRef,
        barycentric: Vec3,
        ray: &Ray,
    ) -> ShadingVertex {
        self.base_shading_vertex_from_triangle_sample_impl(
            triangle,
            barycentric,
            ray.direction,
            Some(ray),
        )
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
        let shading_vertex = self.base_shading_vertex_from_triangle_sample_impl(
            triangle,
            barycentric,
            incident_direction,
            None,
        );
        self.instance_material(triangle.instance_index)
            .prepare_shading_vertex(&shading_vertex)
    }

    fn base_shading_vertex_from_triangle_sample_impl(
        &self,
        triangle: TriangleRef,
        barycentric: Vec3,
        incident_direction: Vec3,
        ray: Option<&Ray>,
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
        let raw_ns = if shading_normal.length_squared() > 0.0 {
            shading_normal
        } else {
            ng
        };
        let ns = face_forward(raw_ns, ng);
        let (mut dndu, mut dndv) = compute_surface_partials([n0, n1, n2], [uv0, uv1, uv2])
            .unwrap_or((Vec3::ZERO, Vec3::ZERO));
        if ns.dot(raw_ns) < 0.0 {
            dndu = -dndu;
            dndv = -dndv;
        }
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
        let differentials = compute_shading_differentials(p, ng, dpdu, dpdv, ray);

        ShadingVertex {
            triangle,
            p,
            uv,
            dudx: differentials.dudx,
            dvdx: differentials.dvdx,
            dudy: differentials.dudy,
            dvdy: differentials.dvdy,
            ng,
            ns,
            wo: (-incident_direction).normalize_or_zero(),
            dpdu,
            dpdv,
            dpdx: differentials.dpdx,
            dpdy: differentials.dpdy,
            dndu,
            dndv,
            frame,
            front_face,
            wavelength_lock: None,
        }
    }

    pub fn shading_vertex(&self, hit: SceneHit, ray: &Ray) -> ShadingVertex {
        let shading_vertex = self.base_shading_vertex_from_triangle_sample_impl(
            hit.triangle,
            hit.barycentric,
            ray.direction,
            Some(ray),
        );
        self.instance_material(hit.triangle.instance_index)
            .prepare_shading_vertex(&shading_vertex)
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

    /// Area-domain PDF of the *uniform* point sampler within an emissive
    /// triangle.
    ///
    /// Triangle *selection* is the SG light tree's responsibility: NEE picks
    /// a leaf via the tree's stochastic descent (PMF = `leaf_pmf`), and
    /// inside the chosen triangle we pick a point uniformly. So the area PDF
    /// here is simply `1/area`. Any caller computing a total density should
    /// multiply this by the leaf-selection PMF separately.
    ///
    /// Folding the old power-CDF `triangle_selection_probability` in here
    /// (the previous behaviour) would double-count selection: the forward
    /// NEE path uses `1/area`, so the MIS reverse PDF must as well.
    pub fn area_light_pdf_area(&self, triangle: TriangleRef) -> Option<f32> {
        let area_light = self
            .area_light_triangles
            .iter()
            .find(|area_light| area_light.triangle == triangle)?;

        if area_light.area <= 0.0 {
            return None;
        }

        Some(1.0 / area_light.area)
    }

    pub fn area_light_pdf_solid_angle(
        &self,
        vtx: &ShadingVertex,
        lvtx: &ShadingVertex,
    ) -> Option<f32> {
        let pdf_area = self.area_light_pdf_area(lvtx.triangle)?;
        let to_light = lvtx.p - vtx.p;
        let distance_squared = to_light.length_squared();

        if distance_squared <= 0.0 {
            return None;
        }

        let distance = distance_squared.sqrt();
        let wi = to_light / distance;
        let cos_light = lvtx.ng.dot(-wi).max(0.0);

        if cos_light <= 0.0 {
            return None;
        }

        Some(pdf_area * distance_squared / cos_light)
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
}

fn closest_mesh_hit_with_filter<F>(
    mesh: &Mesh,
    ray: &Ray,
    t_max: f32,
    mut accept: F,
) -> Option<MeshHit>
where
    F: FnMut(&MeshHit) -> bool,
{
    let qbvh = mesh
        .qbvh
        .as_ref()
        .expect("mesh QBVH must be built before traversal");
    let mut closest_hit: Option<MeshHit> = None;

    traverse_qbvh(qbvh, ray, t_max, |offset, count, current_t_max| {
        let mut t_max = current_t_max;
        for ordered_index in offset..offset + count {
            let triangle_index = qbvh.primitive_indices[ordered_index as usize];
            let [v0, v1, v2] = mesh.triangle_positions(triangle_index);

            if let Some(hit) = intersect_triangle(ray, t_max, v0, v1, v2) {
                let candidate = MeshHit {
                    triangle_index,
                    t: hit.t,
                    barycentric: hit.barycentric,
                };
                if accept(&candidate) {
                    t_max = hit.t;
                    closest_hit = Some(candidate);
                }
            }
        }
        t_max
    });

    closest_hit
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ShadingDifferentials {
    dpdx: Vec3,
    dpdy: Vec3,
    dudx: f32,
    dvdx: f32,
    dudy: f32,
    dvdy: f32,
}

impl ShadingDifferentials {
    fn zero() -> Self {
        Self {
            dpdx: Vec3::ZERO,
            dpdy: Vec3::ZERO,
            dudx: 0.0,
            dvdx: 0.0,
            dudy: 0.0,
            dvdy: 0.0,
        }
    }
}

fn compute_shading_differentials(
    p: Vec3,
    n: Vec3,
    dpdu: Vec3,
    dpdv: Vec3,
    ray: Option<&Ray>,
) -> ShadingDifferentials {
    let Some(ray) = ray else {
        return ShadingDifferentials::zero();
    };

    if let Some(differential) = ray.differential {
        let plane_d = -n.dot(p);
        let rx_denominator = n.dot(differential.rx_direction);
        let ry_denominator = n.dot(differential.ry_direction);

        if rx_denominator.abs() > 1.0e-8 && ry_denominator.abs() > 1.0e-8 {
            let tx = (-n.dot(differential.rx_origin) - plane_d) / rx_denominator;
            let ty = (-n.dot(differential.ry_origin) - plane_d) / ry_denominator;
            let px = differential.rx_origin + tx * differential.rx_direction;
            let py = differential.ry_origin + ty * differential.ry_direction;
            let dpdx = px - p;
            let dpdy = py - p;

            if tx.is_finite()
                && ty.is_finite()
                && dpdx.is_finite()
                && dpdy.is_finite()
                && let Some(differentials) = differentials_from_dp(dpdu, dpdv, dpdx, dpdy)
            {
                return differentials;
            }
        }
    }

    cone_shading_differentials(p, n, dpdu, dpdv, ray).unwrap_or_else(ShadingDifferentials::zero)
}

fn differentials_from_dp(
    dpdu: Vec3,
    dpdv: Vec3,
    dpdx: Vec3,
    dpdy: Vec3,
) -> Option<ShadingDifferentials> {
    let ata00 = dpdu.dot(dpdu);
    let ata01 = dpdu.dot(dpdv);
    let ata11 = dpdv.dot(dpdv);
    let determinant = difference_of_products(ata00, ata11, ata01, ata01);

    if determinant.abs() <= 1.0e-12 || !determinant.is_finite() {
        return None;
    }

    let inv_det = 1.0 / determinant;
    let atb0x = dpdu.dot(dpdx);
    let atb1x = dpdv.dot(dpdx);
    let atb0y = dpdu.dot(dpdy);
    let atb1y = dpdv.dot(dpdy);
    let dudx = difference_of_products(ata11, atb0x, ata01, atb1x) * inv_det;
    let dvdx = difference_of_products(ata00, atb1x, ata01, atb0x) * inv_det;
    let dudy = difference_of_products(ata11, atb0y, ata01, atb1y) * inv_det;
    let dvdy = difference_of_products(ata00, atb1y, ata01, atb0y) * inv_det;

    Some(ShadingDifferentials {
        dpdx,
        dpdy,
        dudx: finite_clamped_derivative(dudx),
        dvdx: finite_clamped_derivative(dvdx),
        dudy: finite_clamped_derivative(dudy),
        dvdy: finite_clamped_derivative(dvdy),
    })
}

fn cone_shading_differentials(
    p: Vec3,
    n: Vec3,
    dpdu: Vec3,
    dpdv: Vec3,
    ray: &Ray,
) -> Option<ShadingDifferentials> {
    let ray_t = ray_parameter_to_point(ray, p)?;
    let width = ray.cone.width_at(ray_t);
    if width <= 0.0 {
        return None;
    }

    let cos_theta = ray.direction.normalize_or_zero().dot(n).abs().max(1.0e-3);
    let projected_width = width / cos_theta;
    let dudx = projected_width / dpdu.length().max(1.0e-6);
    let dvdy = projected_width / dpdv.length().max(1.0e-6);
    let dpdx = dpdu * dudx;
    let dpdy = dpdv * dvdy;

    Some(ShadingDifferentials {
        dpdx,
        dpdy,
        dudx,
        dvdx: 0.0,
        dudy: 0.0,
        dvdy,
    })
}

fn ray_parameter_to_point(ray: &Ray, p: Vec3) -> Option<f32> {
    let direction_length_squared = ray.direction.length_squared();
    if direction_length_squared <= 0.0 {
        return None;
    }

    let t = (p - ray.origin).dot(ray.direction) / direction_length_squared;
    t.is_finite().then_some(t)
}

fn finite_clamped_derivative(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(-1.0e8, 1.0e8)
    } else {
        0.0
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
    use std::sync::Arc;

    use glam::{Mat4, Vec2, Vec3};

    use super::{
        AreaLightTriangle, ClosestHitError, InstanceIndex, MaterialIndex, Scene, SceneHit,
        TriangleRef,
    };
    use crate::{
        material::{EmissiveMaterial, Material, NormalMap, NormalizedLambertMaterial, Texture},
        mesh::{Mesh, Vertex},
        ray::{Ray, RayCone, RayDifferential},
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
        scene.build_qbvh();
        scene.build_light_tree();

        let ray = Ray::new(Vec3::new(0.25, 0.25, 2.0), Vec3::NEG_Z);
        let hit = scene
            .closest_hit(&ray, &mut rand::rng())
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
        scene.build_qbvh();
        scene.build_light_tree();

        let ray = Ray::new(Vec3::new(0.5, 0.5, 1.0), Vec3::NEG_Z);
        let hit = scene
            .closest_hit(&ray, &mut rand::rng())
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
            .closest_hit(&ray, &mut rand::rng())
            .expect_err("expected missing BVH error");

        assert_eq!(error, ClosestHitError::QbvhNotBuilt);
    }

    #[test]
    fn build_qbvh_populates_scene_and_mesh_qbvhs() {
        let mut scene = Scene::new();
        let mesh_index = scene.add_mesh(stacked_mesh());
        let material_index = default_material(&mut scene);
        scene.add_instance(mesh_index, material_index, Mat4::IDENTITY);

        scene.build_qbvh();
        scene.build_light_tree();

        assert!(scene.qbvh.is_some());
        assert!(scene.meshes[mesh_index.0].qbvh.is_some());
    }

    #[test]
    fn closest_hit_returns_none_when_ray_misses_scene() {
        let mut scene = Scene::new();
        let mesh_index = scene.add_mesh(unit_mesh(0.0));
        let material_index = default_material(&mut scene);
        scene.add_instance(mesh_index, material_index, Mat4::IDENTITY);
        scene.build_qbvh();
        scene.build_light_tree();

        let ray = Ray::new(Vec3::new(2.0, 2.0, 1.0), Vec3::NEG_Z);
        let hit = scene
            .closest_hit(&ray, &mut rand::rng())
            .expect("BVH should be built");

        assert!(hit.is_none());
    }

    #[test]
    fn adding_instance_after_build_invalidates_scene_bvh() {
        let mut scene = Scene::new();
        let mesh_index = scene.add_mesh(unit_mesh(0.0));
        let material_index = default_material(&mut scene);
        scene.add_instance(mesh_index, material_index, Mat4::IDENTITY);
        scene.build_qbvh();
        scene.build_light_tree();

        scene.add_instance(
            mesh_index,
            material_index,
            Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0)),
        );

        assert!(scene.qbvh.is_none());
    }

    #[test]
    fn adding_mesh_after_build_invalidates_scene_bvh() {
        let mut scene = Scene::new();
        let mesh_index = scene.add_mesh(unit_mesh(0.0));
        let material_index = default_material(&mut scene);
        scene.add_instance(mesh_index, material_index, Mat4::IDENTITY);
        scene.build_qbvh();
        scene.build_light_tree();

        scene.add_mesh(unit_mesh(-1.0));

        assert!(scene.qbvh.is_none());
    }

    #[test]
    fn closest_hit_traverses_multi_triangle_mesh_bvh() {
        let mut scene = Scene::new();
        let mesh_index = scene.add_mesh(stacked_mesh());
        let material_index = default_material(&mut scene);
        scene.add_instance(mesh_index, material_index, Mat4::IDENTITY);
        scene.build_qbvh();
        scene.build_light_tree();

        let ray = Ray::new(Vec3::new(0.25, 0.25, 2.0), Vec3::NEG_Z);
        let hit = scene
            .closest_hit(&ray, &mut rand::rng())
            .expect("BVH should be built")
            .expect("expected hit");

        assert_eq!(hit.triangle.triangle_index, 0);
        assert!((hit.t - 2.0).abs() < 1.0e-6);
    }

    #[test]
    fn closest_hit_skips_zero_opacity_material_and_returns_geometry_behind_it() {
        use crate::material::SimplePbrMaterial;

        let mut scene = Scene::new();
        let front_mesh = scene.add_mesh(unit_mesh(0.0));
        let back_mesh = scene.add_mesh(unit_mesh(-1.0));

        let mut transparent = SimplePbrMaterial::new(Vec3::ONE, 0.0, 0.5, 1.5, 0.0);
        transparent.opacity = 0.0;
        let transparent_material = scene.add_material(Material::SimplePBR(transparent));
        let opaque_material = scene.add_material(Material::NormalizedLambert(
            NormalizedLambertMaterial::new(Vec3::splat(0.5)),
        ));

        scene.add_instance(front_mesh, transparent_material, Mat4::IDENTITY);
        scene.add_instance(back_mesh, opaque_material, Mat4::IDENTITY);
        scene.build_qbvh();
        scene.build_light_tree();

        let ray = Ray::new(Vec3::new(0.25, 0.25, 2.0), Vec3::NEG_Z);
        let hit = scene
            .closest_hit(&ray, &mut rand::rng())
            .expect("BVH should be built")
            .expect("ray must reach the opaque triangle behind the transparent one");

        assert_eq!(
            hit.triangle,
            TriangleRef {
                instance_index: InstanceIndex(1),
                triangle_index: 0,
            }
        );
        assert!((hit.t - 3.0).abs() < 1.0e-6);
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

        let ray = Ray::new(Vec3::new(0.25, 0.25, 1.0), Vec3::NEG_Z);
        let shading_vertex = scene.shading_vertex(hit, &ray);

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
        assert!(shading_vertex.wo.abs_diff_eq(Vec3::Z, 1.0e-6));
        assert!(shading_vertex.frame.normal().abs_diff_eq(Vec3::Z, 1.0e-6));
        assert!(shading_vertex.front_face);
        assert!(shading_vertex.dpdu.length_squared() > 0.0);
        assert!(shading_vertex.dpdv.length_squared() > 0.0);
    }

    #[test]
    fn shading_vertex_applies_material_normal_map() {
        let mut scene = Scene::new();
        let mesh_index = scene.add_mesh(unit_mesh(0.0));
        let local_normal = Vec3::new(0.6, 0.0, 0.8).normalize();
        let normal_texel = 0.5 * (local_normal + Vec3::ONE);
        let material_index =
            scene.add_material(Material::NormalizedLambert(NormalizedLambertMaterial {
                rho: Vec3::ONE,
                rho_texture: None,
                normal_map: Some(NormalMap::from_texture(Arc::new(Texture::from_pixels(
                    1,
                    1,
                    vec![normal_texel],
                )))),
                normal_strength: 1.0,
                opacity: 1.0,
                opacity_texture: None,
            }));
        scene.add_instance(mesh_index, material_index, Mat4::IDENTITY);
        let hit = SceneHit {
            triangle: TriangleRef {
                instance_index: InstanceIndex(0),
                triangle_index: 0,
            },
            t: 1.0,
            barycentric: Vec3::new(0.5, 0.25, 0.25),
        };

        let ray = Ray::new(Vec3::new(0.25, 0.25, 1.0), Vec3::NEG_Z);
        let shading_vertex = scene.shading_vertex(hit, &ray);

        assert!(shading_vertex.ng.abs_diff_eq(Vec3::Z, 1.0e-6));
        assert!(shading_vertex.ns.abs_diff_eq(local_normal, 1.0e-6));
        assert!(
            shading_vertex
                .frame
                .normal()
                .abs_diff_eq(local_normal, 1.0e-6)
        );
    }

    #[test]
    fn shading_vertex_computes_uv_differentials_from_ray_differential() {
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
        let ray =
            Ray::new(Vec3::new(0.25, 0.25, 1.0), Vec3::NEG_Z).with_differential(RayDifferential {
                rx_origin: Vec3::new(0.35, 0.25, 1.0),
                ry_origin: Vec3::new(0.25, 0.45, 1.0),
                rx_direction: Vec3::NEG_Z,
                ry_direction: Vec3::NEG_Z,
            });

        let shading_vertex = scene.shading_vertex(hit, &ray);

        assert!(
            shading_vertex
                .dpdx
                .abs_diff_eq(Vec3::new(0.1, 0.0, 0.0), 1.0e-6)
        );
        assert!(
            shading_vertex
                .dpdy
                .abs_diff_eq(Vec3::new(0.0, 0.2, 0.0), 1.0e-6)
        );
        assert!((shading_vertex.dudx - 0.1).abs() < 1.0e-6);
        assert_eq!(shading_vertex.dvdx, 0.0);
        assert_eq!(shading_vertex.dudy, 0.0);
        assert!((shading_vertex.dvdy - 0.2).abs() < 1.0e-6);
    }

    #[test]
    fn shading_vertex_falls_back_to_ray_cone_without_differentials() {
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
        let ray =
            Ray::new(Vec3::new(0.25, 0.25, 1.0), Vec3::NEG_Z).with_cone(RayCone::new(0.1, 0.0));

        let shading_vertex = scene.shading_vertex(hit, &ray);

        assert!((shading_vertex.dudx - 0.1).abs() < 1.0e-6);
        assert_eq!(shading_vertex.dvdx, 0.0);
        assert_eq!(shading_vertex.dudy, 0.0);
        assert!((shading_vertex.dvdy - 0.1).abs() < 1.0e-6);
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
    fn area_light_pdf_solid_angle_converts_area_density_with_jacobian() {
        let mut scene = Scene::new();
        let floor_mesh = scene.add_mesh(unit_mesh(0.0));
        let light_mesh = scene.add_mesh(unit_mesh(2.0));
        let floor_material = default_material(&mut scene);
        let light_material =
            scene.add_material(Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 10.0)));
        scene.add_instance(floor_mesh, floor_material, Mat4::IDENTITY);
        scene.add_instance(light_mesh, light_material, Mat4::IDENTITY);

        let vtx = scene.shading_vertex_from_triangle_sample(
            TriangleRef {
                instance_index: InstanceIndex(0),
                triangle_index: 0,
            },
            Vec3::new(0.5, 0.25, 0.25),
            Vec3::NEG_Z,
        );
        let lvtx = scene.shading_vertex_from_triangle_sample(
            TriangleRef {
                instance_index: InstanceIndex(1),
                triangle_index: 0,
            },
            Vec3::new(0.5, 0.25, 0.25),
            Vec3::Z,
        );

        let pdf = scene
            .area_light_pdf_solid_angle(&vtx, &lvtx)
            .expect("expected valid area light pdf");

        assert!((pdf - 8.0).abs() < 1.0e-6);
    }

    #[test]
    fn area_light_pdf_solid_angle_returns_none_for_non_emissive_triangle() {
        let mut scene = Scene::new();
        let mesh_index = scene.add_mesh(unit_mesh(0.0));
        let material_index = default_material(&mut scene);
        scene.add_instance(mesh_index, material_index, Mat4::IDENTITY);
        scene.add_instance(
            mesh_index,
            material_index,
            Mat4::from_translation(Vec3::new(0.0, 0.0, 1.0)),
        );

        let vtx = scene.shading_vertex_from_triangle_sample(
            TriangleRef {
                instance_index: InstanceIndex(0),
                triangle_index: 0,
            },
            Vec3::new(0.5, 0.25, 0.25),
            Vec3::NEG_Z,
        );
        let lvtx = scene.shading_vertex_from_triangle_sample(
            TriangleRef {
                instance_index: InstanceIndex(1),
                triangle_index: 0,
            },
            Vec3::new(0.5, 0.25, 0.25),
            Vec3::Z,
        );

        assert_eq!(scene.area_light_pdf_solid_angle(&vtx, &lvtx), None);
    }
}
