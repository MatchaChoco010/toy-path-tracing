use clap::ValueEnum;
use glam::Vec3;

use crate::{
    bsdf::BsdfFlags,
    light::LightLiSample,
    material::{MaterialSample, MtlxScratch, ShadingVertex},
    math::ray::{Ray, RayCone, RayDifferential},
    sampler::{AuxRng, PathSampler},
    scene::{Scene, TriangleRef},
};

pub mod mis;
pub mod nee;
pub mod pt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum IntegratorKind {
    Mis,
    Pt,
    Nee,
}

impl IntegratorKind {
    pub fn trace_radiance(
        self,
        scene: &Scene,
        initial_ray: Ray,
        sampler: &PathSampler,
        max_depth: u32,
        mtlx_scratch: &mut MtlxScratch,
    ) -> Vec3 {
        let cp = mtlx_scratch.checkpoint();
        let radiance = match self {
            Self::Mis => mis::trace_radiance(scene, initial_ray, sampler, max_depth, mtlx_scratch),
            Self::Pt => pt::trace_radiance(scene, initial_ray, sampler, max_depth, mtlx_scratch),
            Self::Nee => nee::trace_radiance(scene, initial_ray, sampler, max_depth, mtlx_scratch),
        };
        mtlx_scratch.restore(cp);
        radiance
    }
}

const RAY_EPSILON: f32 = 1.0e-4;
const SHADOW_TOLERANCE: f32 = 1.0e-3;

pub(super) fn spawn_ray(origin: Vec3, geometric_normal: Vec3, direction: Vec3) -> Ray {
    Ray::new(
        offset_ray_origin(origin, geometric_normal, direction),
        direction,
    )
}

pub(super) fn spawn_scattered_ray(
    incoming_ray: &Ray,
    hit_t: f32,
    vtx: &ShadingVertex,
    sample: &MaterialSample,
) -> Ray {
    let mut ray = spawn_ray(vtx.p, vtx.ng, sample.wi);
    let width = incoming_ray.cone.width_at(hit_t);
    let spread_angle = incoming_ray.cone.spread_angle + sample.cone_spread;
    ray.cone = RayCone::new(width, spread_angle);

    if sample.flags.contains(BsdfFlags::DELTA) {
        ray.differential = specular_ray_differential(incoming_ray, vtx, sample);
    }

    ray
}

fn offset_ray_origin(origin: Vec3, geometric_normal: Vec3, direction: Vec3) -> Vec3 {
    let normal_offset = if direction.dot(geometric_normal) >= 0.0 {
        geometric_normal
    } else {
        -geometric_normal
    };

    origin + RAY_EPSILON * normal_offset
}

fn specular_ray_differential(
    incoming_ray: &Ray,
    vtx: &ShadingVertex,
    sample: &MaterialSample,
) -> Option<RayDifferential> {
    let incoming_differential = incoming_ray.differential?;
    let wo = vtx.wo;
    let wi = sample.wi.normalize_or_zero();
    let dndx = vtx.dndu * vtx.dudx + vtx.dndv * vtx.dvdx;
    let dndy = vtx.dndu * vtx.dudy + vtx.dndv * vtx.dvdy;
    let dwodx = -incoming_differential.rx_direction - wo;
    let dwody = -incoming_differential.ry_direction - wo;
    let rx_origin = offset_ray_origin(vtx.p + vtx.dpdx, vtx.ng, wi);
    let ry_origin = offset_ray_origin(vtx.p + vtx.dpdy, vtx.ng, wi);

    let (rx_direction, ry_direction) = if sample.flags.contains(BsdfFlags::REFLECTION) {
        reflected_differential_directions(wo, wi, vtx.ns, dndx, dndy, dwodx, dwody)
    } else if sample.flags.contains(BsdfFlags::TRANSMISSION) {
        transmitted_differential_directions(TransmissionDifferentialInput {
            wo,
            wi,
            n: vtx.ns,
            dndx,
            dndy,
            dwodx,
            dwody,
            eta: sample.eta,
        })?
    } else {
        return None;
    };

    let differential = RayDifferential {
        rx_origin,
        ry_origin,
        rx_direction,
        ry_direction,
    };

    if differential_is_reasonable(differential) {
        Some(differential)
    } else {
        None
    }
}

fn reflected_differential_directions(
    wo: Vec3,
    wi: Vec3,
    n: Vec3,
    dndx: Vec3,
    dndy: Vec3,
    dwodx: Vec3,
    dwody: Vec3,
) -> (Vec3, Vec3) {
    let dwo_dot_n_dx = dwodx.dot(n) + wo.dot(dndx);
    let dwo_dot_n_dy = dwody.dot(n) + wo.dot(dndy);
    let wo_dot_n = wo.dot(n);
    let rx_direction = wi - dwodx + 2.0 * (wo_dot_n * dndx + dwo_dot_n_dx * n);
    let ry_direction = wi - dwody + 2.0 * (wo_dot_n * dndy + dwo_dot_n_dy * n);

    (rx_direction, ry_direction)
}

struct TransmissionDifferentialInput {
    wo: Vec3,
    wi: Vec3,
    n: Vec3,
    dndx: Vec3,
    dndy: Vec3,
    dwodx: Vec3,
    dwody: Vec3,
    eta: f32,
}

fn transmitted_differential_directions(
    input: TransmissionDifferentialInput,
) -> Option<(Vec3, Vec3)> {
    let TransmissionDifferentialInput {
        wo,
        wi,
        mut n,
        mut dndx,
        mut dndy,
        dwodx,
        dwody,
        eta,
    } = input;

    if (eta - 1.0).abs() <= 1.0e-6 {
        return Some((wi - dwodx, wi - dwody));
    }

    if eta <= 0.0 {
        return None;
    }

    if wo.dot(n) < 0.0 {
        n = -n;
        dndx = -dndx;
        dndy = -dndy;
    }

    let wi_dot_n = wi.dot(n);
    if wi_dot_n.abs() <= 1.0e-6 {
        return None;
    }

    let dwo_dot_n_dx = dwodx.dot(n) + wo.dot(dndx);
    let dwo_dot_n_dy = dwody.dot(n) + wo.dot(dndy);
    let wo_dot_n = wo.dot(n);
    let mu = wo_dot_n * eta - wi_dot_n.abs();
    let dmu_scale = eta + eta * eta * wo_dot_n / wi_dot_n;
    let dmudx = dwo_dot_n_dx * dmu_scale;
    let dmudy = dwo_dot_n_dy * dmu_scale;
    let rx_direction = wi - eta * dwodx + mu * dndx + dmudx * n;
    let ry_direction = wi - eta * dwody + mu * dndy + dmudy * n;

    Some((rx_direction, ry_direction))
}

fn differential_is_reasonable(differential: RayDifferential) -> bool {
    const MAX_DIFFERENTIAL_LENGTH_SQUARED: f32 = 1.0e16;

    differential.rx_origin.is_finite()
        && differential.ry_origin.is_finite()
        && differential.rx_direction.is_finite()
        && differential.ry_direction.is_finite()
        && differential.rx_origin.length_squared() <= MAX_DIFFERENTIAL_LENGTH_SQUARED
        && differential.ry_origin.length_squared() <= MAX_DIFFERENTIAL_LENGTH_SQUARED
        && differential.rx_direction.length_squared() <= MAX_DIFFERENTIAL_LENGTH_SQUARED
        && differential.ry_direction.length_squared() <= MAX_DIFFERENTIAL_LENGTH_SQUARED
}

pub(super) fn unoccluded(
    scene: &Scene,
    vtx: &ShadingVertex,
    li: &LightLiSample,
    aux_rng: &mut AuxRng,
    mtlx_scratch: &mut crate::material::MtlxScratch,
) -> bool {
    unoccluded_ray(
        scene,
        vtx,
        li.wi,
        li.distance,
        li.target_triangle,
        aux_rng,
        mtlx_scratch,
    )
}

pub(super) fn unoccluded_ray(
    scene: &Scene,
    vtx: &ShadingVertex,
    wi: Vec3,
    distance: f32,
    target_triangle: Option<TriangleRef>,
    aux_rng: &mut AuxRng,
    mtlx_scratch: &mut crate::material::MtlxScratch,
) -> bool {
    let shadow_ray = spawn_ray(vtx.p, vtx.ng, wi);
    let hit = scene
        .closest_hit(&shadow_ray, aux_rng, mtlx_scratch)
        .expect("scene.build_qbvh() must be called before traversal");

    match hit {
        None => true,
        Some(hit) => {
            if let Some(target) = target_triangle {
                hit.triangle == target
            } else if distance.is_infinite() {
                false
            } else {
                hit.t >= distance * (1.0 - SHADOW_TOLERANCE)
            }
        }
    }
}

#[cfg(test)]
pub(super) mod test_helpers {
    use glam::{Vec2, Vec3};

    use crate::{
        material::{EmissiveMaterial, Material, MirrorMaterial},
        math::ray::Ray,
        scene::Scene,
        scene::{Mesh, Vertex},
    };

    pub(super) fn mirror_to_light_scene() -> (Scene, Ray, Vec3) {
        let mut scene = Scene::new();
        let mirror_color_linear = Vec3::new(0.25, 0.5, 0.75);
        let light_strength = 4.0;
        let mirror_material =
            scene.add_material(Material::Mirror(MirrorMaterial::new(mirror_color_linear)));
        let light_material = scene.add_material(Material::Emissive(EmissiveMaterial::new(
            Vec3::ONE,
            light_strength,
        )));
        let mirror_mesh = scene.add_mesh(mirror_triangle_mesh());
        let light_mesh = scene.add_mesh(light_triangle_mesh());

        scene.add_instance(mirror_mesh, mirror_material, glam::Mat4::IDENTITY);
        scene.add_instance(light_mesh, light_material, glam::Mat4::IDENTITY);
        scene.build_qbvh();
        scene.build_light_tree();

        let mirror_hit = Vec3::new(0.25, 0.20, 0.0);
        let light_hit = Vec3::new(0.65, 0.20, 1.0);
        let reflected_direction = (light_hit - mirror_hit).normalize();
        let incoming_direction = Vec3::new(
            reflected_direction.x,
            reflected_direction.y,
            -reflected_direction.z,
        )
        .normalize();
        let ray = Ray::new(mirror_hit - incoming_direction, incoming_direction);

        (scene, ray, mirror_color_linear * light_strength)
    }

    fn mirror_triangle_mesh() -> Mesh {
        Mesh::new(
            vec![
                Vertex {
                    position: Vec3::new(-1.0, -1.0, 0.0),
                    normal: Vec3::Z,
                    uv: Vec2::ZERO,
                },
                Vertex {
                    position: Vec3::new(2.0, -1.0, 0.0),
                    normal: Vec3::Z,
                    uv: Vec2::X,
                },
                Vertex {
                    position: Vec3::new(-1.0, 2.0, 0.0),
                    normal: Vec3::Z,
                    uv: Vec2::Y,
                },
            ],
            vec![0, 1, 2],
        )
    }

    fn light_triangle_mesh() -> Mesh {
        Mesh::new(
            vec![
                Vertex {
                    position: Vec3::new(0.45, 0.0, 1.0),
                    normal: Vec3::Z,
                    uv: Vec2::ZERO,
                },
                Vertex {
                    position: Vec3::new(0.95, 0.0, 1.0),
                    normal: Vec3::Z,
                    uv: Vec2::X,
                },
                Vertex {
                    position: Vec3::new(0.45, 0.5, 1.0),
                    normal: Vec3::Z,
                    uv: Vec2::Y,
                },
            ],
            vec![0, 1, 2],
        )
    }
}

#[cfg(test)]
mod tests {
    use glam::{Vec2, Vec3};

    use super::{spawn_ray, spawn_scattered_ray};
    use crate::{
        bsdf::BsdfFlags,
        material::{MaterialSample, ShadingVertex},
        math::OrthonormalBasis,
        math::ray::{Ray, RayCone, RayDifferential},
        scene::{InstanceIndex, TriangleRef},
    };

    #[test]
    fn spawn_ray_offsets_along_the_sampled_hemisphere() {
        let reflection_ray = spawn_ray(Vec3::ZERO, Vec3::Z, Vec3::Z);
        let transmission_ray = spawn_ray(Vec3::ZERO, Vec3::Z, Vec3::NEG_Z);

        assert_eq!(reflection_ray.origin, Vec3::new(0.0, 0.0, 1.0e-4));
        assert_eq!(transmission_ray.origin, Vec3::new(0.0, 0.0, -1.0e-4));
    }

    #[test]
    fn spawn_scattered_ray_propagates_delta_reflection_differentials() {
        let incoming_ray = Ray::new(Vec3::new(0.0, 0.0, 1.0), Vec3::NEG_Z)
            .with_differential(RayDifferential {
                rx_origin: Vec3::new(0.1, 0.0, 1.0),
                ry_origin: Vec3::new(0.0, 0.1, 1.0),
                rx_direction: Vec3::NEG_Z,
                ry_direction: Vec3::NEG_Z,
            })
            .with_cone(RayCone::new(0.0, 0.01));
        let vtx = test_shading_vertex();
        let sample = MaterialSample {
            weight: Vec3::ONE,
            wi: Vec3::Z,
            pdf: 1.0,
            flags: BsdfFlags::DELTA | BsdfFlags::REFLECTION,
            eta: 1.0,
            cone_spread: 0.0,
            wavelength_lock: None,
        };

        let ray = spawn_scattered_ray(&incoming_ray, 1.0, &vtx, &sample);
        let differential = ray
            .differential
            .expect("delta reflection should keep ray differentials");

        assert!((ray.cone.width - 0.01).abs() < 1.0e-6);
        assert!(
            differential
                .rx_origin
                .abs_diff_eq(Vec3::new(0.1, 0.0, 1.0e-4), 1.0e-6)
        );
        assert!(
            differential
                .ry_origin
                .abs_diff_eq(Vec3::new(0.0, 0.1, 1.0e-4), 1.0e-6)
        );
        assert!(differential.rx_direction.abs_diff_eq(Vec3::Z, 1.0e-6));
        assert!(differential.ry_direction.abs_diff_eq(Vec3::Z, 1.0e-6));
    }

    #[test]
    fn spawn_scattered_ray_drops_glossy_differentials_and_expands_cone() {
        let incoming_ray = Ray::new(Vec3::new(0.0, 0.0, 1.0), Vec3::NEG_Z)
            .with_differential(RayDifferential {
                rx_origin: Vec3::new(0.1, 0.0, 1.0),
                ry_origin: Vec3::new(0.0, 0.1, 1.0),
                rx_direction: Vec3::NEG_Z,
                ry_direction: Vec3::NEG_Z,
            })
            .with_cone(RayCone::new(0.0, 0.01));
        let vtx = test_shading_vertex();
        let sample = MaterialSample {
            weight: Vec3::ONE,
            wi: Vec3::Z,
            pdf: 1.0,
            flags: BsdfFlags::GLOSSY | BsdfFlags::REFLECTION,
            eta: 1.0,
            cone_spread: 0.2,
            wavelength_lock: None,
        };

        let ray = spawn_scattered_ray(&incoming_ray, 1.0, &vtx, &sample);

        assert!(ray.differential.is_none());
        assert!((ray.cone.width - 0.01).abs() < 1.0e-6);
        assert!((ray.cone.spread_angle - 0.21).abs() < 1.0e-6);
    }

    fn test_shading_vertex() -> ShadingVertex {
        ShadingVertex {
            triangle: TriangleRef {
                instance_index: InstanceIndex(0),
                triangle_index: 0,
            },
            p: Vec3::ZERO,
            uv: Vec2::ZERO,
            dudx: 0.1,
            dvdx: 0.0,
            dudy: 0.0,
            dvdy: 0.1,
            ng: Vec3::Z,
            ns: Vec3::Z,
            wo: Vec3::Z,
            dpdu: Vec3::X,
            dpdv: Vec3::Y,
            dpdx: Vec3::X * 0.1,
            dpdy: Vec3::Y * 0.1,
            dndu: Vec3::ZERO,
            dndv: Vec3::ZERO,
            frame: OrthonormalBasis::from_normal(Vec3::Z),
            front_face: true,
            path_throughput: Vec3::ONE,
            wavelength_lock: None,
            object_to_world: glam::Mat4::IDENTITY,
            world_to_object: glam::Mat4::IDENTITY,
            object_normal_to_world: glam::Mat3::IDENTITY,
            mtlx_regs: None,
            mtlx_dalbedo: None,
            mtlx_precomputed_for: None,
        }
    }
}
