use std::{cell::RefCell, f32::consts::PI};

use glam::Vec3;

use crate::{
    bsdf::{BsdfFlags, TransportMode},
    light::{
        LightSampleContext, LightType, infinite_light_le, pdf_for_environment_hit,
        sample_light_mis_compensated_lazy,
    },
    material::{Material, MtlxScratch, ShadingVertex},
    math::ray::Ray,
    math::{
        OrthonormalBasis, balance_heuristic, cosine_weighted_hemisphere_pdf,
        russian_roulette_probability, sample_cosine_weighted_hemisphere,
    },
    sampler::{AuxRng, PathSampler},
    scene::{AreaLightTriangle, Scene, TriangleRef},
};

use super::{SHADOW_TOLERANCE, spawn_ray, spawn_scattered_ray, unoccluded};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VertexType {
    Light,
    Surface,
}

#[derive(Debug, Clone)]
struct PathVertex {
    vertex_type: VertexType,
    vtx: ShadingVertex,
    beta: Vec3,
    pdf_fwd_area: f32,
    pdf_rev_area: f32,
    delta: bool,
    connectible: bool,
}

#[derive(Default)]
struct PathBuffers {
    camera_path: Vec<PathVertex>,
    light_path: Vec<PathVertex>,
}

thread_local! {
    static PATH_BUFFERS: RefCell<PathBuffers> = RefCell::new(PathBuffers::default());
}

pub fn trace_radiance(
    scene: &Scene,
    initial_ray: Ray,
    sampler: &PathSampler,
    max_depth: u32,
    mtlx_scratch: &mut MtlxScratch,
) -> Vec3 {
    PATH_BUFFERS.with(|buffers| {
        let mut buffers = buffers.borrow_mut();
        trace_radiance_with_buffers(
            scene,
            initial_ray,
            sampler,
            max_depth,
            mtlx_scratch,
            &mut buffers,
        )
    })
}

fn trace_radiance_with_buffers(
    scene: &Scene,
    initial_ray: Ray,
    sampler: &PathSampler,
    max_depth: u32,
    mtlx_scratch: &mut MtlxScratch,
    buffers: &mut PathBuffers,
) -> Vec3 {
    reserve_path_capacity(&mut buffers.camera_path, max_depth as usize);
    reserve_path_capacity(&mut buffers.light_path, max_depth as usize);

    let mut aux_rng = AuxRng::from_seed(sampler.initial_aux_rng_seed());
    let sample_non_area_lights = scene_has_non_area_lights(scene);

    let mut radiance = generate_camera_subpath(
        scene,
        initial_ray,
        sampler,
        max_depth,
        sample_non_area_lights,
        &mut buffers.camera_path,
        &mut aux_rng,
        mtlx_scratch,
    );

    generate_light_subpath(
        scene,
        sampler,
        max_depth,
        &mut buffers.light_path,
        &mut aux_rng,
        mtlx_scratch,
    );

    for (li, light_v) in buffers.light_path.iter().enumerate() {
        for (ci, camera_v) in buffers.camera_path.iter().enumerate() {
            if !light_v.connectible || !camera_v.connectible {
                continue;
            }
            let s = li + 1;
            let t = ci + 1;
            let weight = bdpt_mis_weight(
                scene,
                &buffers.light_path[..s],
                &buffers.camera_path[..t],
                mtlx_scratch,
            );
            radiance +=
                weight * connect_vertices(scene, light_v, camera_v, &mut aux_rng, mtlx_scratch);
        }
    }

    radiance
}

fn reserve_path_capacity(path: &mut Vec<PathVertex>, capacity: usize) {
    if path.capacity() < capacity {
        path.reserve(capacity - path.capacity());
    }
}

fn scene_has_non_area_lights(scene: &Scene) -> bool {
    scene.environment_light.is_some()
        || !scene.point_lights.is_empty()
        || !scene.directional_lights.is_empty()
        || !scene.spot_lights.is_empty()
}

fn generate_camera_subpath(
    scene: &Scene,
    initial_ray: Ray,
    sampler: &PathSampler,
    max_depth: u32,
    sample_non_area_lights: bool,
    vertices: &mut Vec<PathVertex>,
    aux_rng: &mut AuxRng,
    mtlx_scratch: &mut MtlxScratch,
) -> Vec3 {
    vertices.clear();
    let mut radiance = Vec3::ZERO;
    let mut throughput = Vec3::ONE;
    let mut ray = initial_ray;
    let mut count_surface_emission_at_hit = true;
    let mut wavelength_lock = None;
    let mut pending_pdf_fwd_dir: Option<(Vec3, f32)> = None;
    let mut last_bsdf_pdf = 0.0;
    let rr_start_depth = 4;

    for depth in 0..max_depth {
        let randoms = sampler.path_vertex_randoms(depth);
        *aux_rng = AuxRng::from_seed(randoms.aux_rng_seed);
        let hit = scene
            .closest_hit(&ray, aux_rng, mtlx_scratch)
            .expect("scene.build_qbvh() must be called before traversal");

        let Some(hit) = hit else {
            radiance += emitted_radiance_from_infinite_hit(
                scene,
                throughput,
                last_bsdf_pdf,
                count_surface_emission_at_hit,
                ray.direction,
            );
            break;
        };

        let mut vtx = scene.shading_vertex(hit, &ray);
        vtx.path_throughput = throughput;
        vtx.wavelength_lock = wavelength_lock;
        let material = scene.instance_material(hit.triangle.instance_index);
        material.precompute_shading(&mut vtx, mtlx_scratch);
        let pdf_fwd_area = pending_pdf_fwd_dir
            .take()
            .map(|(from, pdf)| solid_angle_pdf_to_area(pdf, from, vtx.p, vtx.ng))
            .unwrap_or(0.0);

        let vertex_index = vertices.len();
        vertices.push(PathVertex {
            vertex_type: VertexType::Surface,
            vtx: vtx.clone(),
            beta: throughput,
            pdf_fwd_area,
            pdf_rev_area: 0.0,
            delta: false,
            connectible: !material.is_pure_emitter(),
        });

        if let Some(le) = material.le(&vtx, mtlx_scratch) {
            let weight = if count_surface_emission_at_hit {
                1.0
            } else {
                camera_light_hit_mis_weight(scene, vertices)
            };
            radiance += throughput * le * weight;
        }

        if sample_non_area_lights && !material.is_pure_emitter() {
            let light_randoms = randoms.light;
            radiance += throughput
                * direct_non_area_light_contribution(
                    scene,
                    material,
                    &vtx,
                    light_randoms.u_category,
                    light_randoms.u_tree,
                    light_randoms.u_light_aux,
                    light_randoms.u_surface,
                    aux_rng,
                    mtlx_scratch,
                );
        }

        let Some(sample) = material.sample(
            &vtx,
            mtlx_scratch,
            &randoms.material,
            aux_rng,
            TransportMode::Radiance,
        ) else {
            break;
        };
        let is_delta_sample = sample.flags.contains(BsdfFlags::DELTA);
        vertices[vertex_index].delta = is_delta_sample;
        vertices[vertex_index].connectible =
            !vertices[vertex_index].delta && !material.is_pure_emitter();

        if vertex_index > 0 {
            let prev = &vertices[vertex_index - 1];
            let pdf_rev_area =
                solid_angle_pdf_to_area(sample.pdf_rev, vtx.p, prev.vtx.p, prev.vtx.ng);
            vertices[vertex_index - 1].pdf_rev_area = pdf_rev_area;
        }

        let next_ray = spawn_scattered_ray(&ray, hit.t, &vtx, &sample);

        throughput *= sample.weight;
        if let Some(lambda) = sample.wavelength_lock {
            wavelength_lock = Some(lambda);
        }

        if depth + 1 >= rr_start_depth {
            let survive_probability = russian_roulette_probability(throughput);
            if randoms.u_rr > survive_probability {
                break;
            }
            throughput /= survive_probability;
        }

        ray = next_ray;
        pending_pdf_fwd_dir = Some((vtx.p, sample.pdf));
        last_bsdf_pdf = sample.pdf;
        count_surface_emission_at_hit = is_delta_sample;
    }

    radiance
}

fn generate_light_subpath(
    scene: &Scene,
    sampler: &PathSampler,
    max_depth: u32,
    vertices: &mut Vec<PathVertex>,
    aux_rng: &mut AuxRng,
    mtlx_scratch: &mut MtlxScratch,
) {
    vertices.clear();
    if max_depth == 0 {
        return;
    }

    let randoms = sampler.path_vertex_randoms(0);
    let Some((area_light, triangle_pmf)) =
        sample_area_light_triangle(scene, randoms.light.u_category)
    else {
        return;
    };
    let point = scene.sample_triangle_point(area_light.triangle, randoms.light.u_surface);
    let pdf_area = triangle_pmf * point.pdf_area;
    if pdf_area <= 0.0 {
        return;
    }

    let frame =
        OrthonormalBasis::from_normal(triangle_geometric_normal(scene, area_light.triangle));
    let local_dir = sample_cosine_weighted_hemisphere(randoms.material.u_dir);
    let emit_dir = frame.local_to_world(local_dir).normalize_or_zero();
    let pdf_dir = cosine_weighted_hemisphere_pdf(local_dir.z);
    if emit_dir.length_squared() == 0.0 || pdf_dir <= 0.0 {
        return;
    }

    let mut light_vtx = scene.shading_vertex_from_triangle_sample(
        area_light.triangle,
        point.barycentric,
        -emit_dir,
    );
    light_vtx.wo = emit_dir;
    let light_material = scene.instance_material(area_light.triangle.instance_index);
    light_material.precompute_shading(&mut light_vtx, mtlx_scratch);
    let Some(le) = light_material.le(&light_vtx, mtlx_scratch) else {
        return;
    };
    let cos_emit = light_vtx.ng.dot(emit_dir).max(0.0);
    if cos_emit <= 0.0 {
        return;
    }

    let endpoint_beta = le / pdf_area;
    vertices.push(PathVertex {
        vertex_type: VertexType::Light,
        vtx: light_vtx.clone(),
        beta: endpoint_beta,
        pdf_fwd_area: pdf_area,
        pdf_rev_area: 0.0,
        delta: false,
        connectible: true,
    });

    let mut ray = spawn_ray(light_vtx.p, light_vtx.ng, emit_dir);
    let mut throughput = endpoint_beta * (cos_emit / pdf_dir);
    let mut wavelength_lock = None;
    let mut pending_pdf_fwd_dir = Some((light_vtx.p, pdf_dir));

    for depth in 1..max_depth {
        let randoms = sampler.path_vertex_randoms(depth);
        *aux_rng = AuxRng::from_seed(randoms.aux_rng_seed);
        let hit = scene
            .closest_hit(&ray, aux_rng, mtlx_scratch)
            .expect("scene.build_qbvh() must be called before traversal");
        let Some(hit) = hit else {
            break;
        };

        let mut vtx = scene.shading_vertex(hit, &ray);
        vtx.path_throughput = throughput;
        vtx.wavelength_lock = wavelength_lock;
        let material = scene.instance_material(hit.triangle.instance_index);
        material.precompute_shading(&mut vtx, mtlx_scratch);
        let pdf_fwd_area = pending_pdf_fwd_dir
            .take()
            .map(|(from, pdf)| solid_angle_pdf_to_area(pdf, from, vtx.p, vtx.ng))
            .unwrap_or(0.0);

        let vertex_index = vertices.len();
        vertices.push(PathVertex {
            vertex_type: VertexType::Surface,
            vtx: vtx.clone(),
            beta: throughput,
            pdf_fwd_area,
            pdf_rev_area: 0.0,
            delta: false,
            connectible: !material.is_pure_emitter(),
        });

        let Some(sample) = material.sample(
            &vtx,
            mtlx_scratch,
            &randoms.material,
            aux_rng,
            TransportMode::Importance,
        ) else {
            break;
        };
        vertices[vertex_index].delta = sample.flags.contains(BsdfFlags::DELTA);
        vertices[vertex_index].connectible =
            !vertices[vertex_index].delta && !material.is_pure_emitter();

        if vertex_index > 0 {
            let prev = &vertices[vertex_index - 1];
            let pdf_rev_area =
                solid_angle_pdf_to_area(sample.pdf_rev, vtx.p, prev.vtx.p, prev.vtx.ng);
            vertices[vertex_index - 1].pdf_rev_area = pdf_rev_area;
        }

        let next_ray = spawn_scattered_ray(&ray, hit.t, &vtx, &sample);

        throughput *= sample.weight;
        if let Some(lambda) = sample.wavelength_lock {
            wavelength_lock = Some(lambda);
        }

        ray = next_ray;
        pending_pdf_fwd_dir = Some((vtx.p, sample.pdf));
    }
}

fn direct_non_area_light_contribution(
    scene: &Scene,
    material: &Material,
    vtx: &ShadingVertex,
    u_root: f32,
    u_tree: f32,
    u_aux: f32,
    us: glam::Vec2,
    aux_rng: &mut AuxRng,
    mtlx_scratch: &mut MtlxScratch,
) -> Vec3 {
    let ctx = LightSampleContext::from_vertex(vtx);

    let Some(sampled) = sample_light_mis_compensated_lazy(
        scene,
        &ctx,
        vtx,
        material,
        u_root,
        u_tree,
        u_aux,
        us,
        mtlx_scratch,
    ) else {
        return Vec3::ZERO;
    };
    let li = sampled.sample;

    if li.light_type == LightType::Area {
        return Vec3::ZERO;
    }
    if li.pdf <= 0.0 || sampled.selection_pmf <= 0.0 {
        return Vec3::ZERO;
    }

    let is_delta_light = li.light_type.is_delta();
    let (f, bsdf_pdf) = if is_delta_light {
        (
            material.eval(vtx, mtlx_scratch, li.wi, aux_rng, TransportMode::Radiance),
            0.0,
        )
    } else {
        material.eval_pdf(vtx, mtlx_scratch, li.wi, aux_rng, TransportMode::Radiance)
    };
    if f.length_squared() == 0.0 {
        return Vec3::ZERO;
    }

    let cos_surface = vtx.ng.dot(li.wi).abs();
    if cos_surface <= 0.0 {
        return Vec3::ZERO;
    }

    if !unoccluded(scene, vtx, &li, aux_rng, mtlx_scratch) {
        return Vec3::ZERO;
    }

    let pdf_total = sampled.selection_pmf * li.pdf;
    if pdf_total <= 0.0 {
        return Vec3::ZERO;
    }

    if is_delta_light {
        return li.radiance * f * (cos_surface / pdf_total);
    }

    let mis_weight = balance_heuristic(pdf_total, bsdf_pdf);
    li.radiance * f * (cos_surface * mis_weight / pdf_total)
}

fn emitted_radiance_from_infinite_hit(
    scene: &Scene,
    throughput: Vec3,
    bsdf_pdf: f32,
    count_full_emission: bool,
    direction: Vec3,
) -> Vec3 {
    let le = infinite_light_le(scene, direction);
    if le == Vec3::ZERO {
        return Vec3::ZERO;
    }

    if count_full_emission {
        return throughput * le;
    }

    let light_pdf = pdf_for_environment_hit(scene, direction, true);
    let mis_weight = balance_heuristic(bsdf_pdf, light_pdf);
    throughput * le * mis_weight
}

fn connect_vertices(
    scene: &Scene,
    light_v: &PathVertex,
    camera_v: &PathVertex,
    aux_rng: &mut AuxRng,
    mtlx_scratch: &mut MtlxScratch,
) -> Vec3 {
    let delta = camera_v.vtx.p - light_v.vtx.p;
    let dist2 = delta.length_squared();
    if dist2 <= 0.0 {
        return Vec3::ZERO;
    }
    let dist = dist2.sqrt();
    let w_lc = delta / dist;
    let w_cl = -w_lc;

    if !visible_between(
        scene,
        light_v.vtx.p,
        light_v.vtx.ng,
        w_lc,
        dist,
        Some(camera_v.vtx.triangle),
        aux_rng,
        mtlx_scratch,
    ) {
        return Vec3::ZERO;
    }

    let camera_material = scene.instance_material(camera_v.vtx.triangle.instance_index);
    let f_camera = camera_material.eval(
        &camera_v.vtx,
        mtlx_scratch,
        w_cl,
        aux_rng,
        TransportMode::Radiance,
    );
    if f_camera.length_squared() == 0.0 {
        return Vec3::ZERO;
    }

    let f_light = match light_v.vertex_type {
        VertexType::Light => {
            if light_v.vtx.ng.dot(w_lc) <= 0.0 {
                return Vec3::ZERO;
            }
            Vec3::ONE
        }
        VertexType::Surface => {
            let light_material = scene.instance_material(light_v.vtx.triangle.instance_index);
            light_material.eval(
                &light_v.vtx,
                mtlx_scratch,
                w_lc,
                aux_rng,
                TransportMode::Importance,
            )
        }
    };
    if f_light.length_squared() == 0.0 {
        return Vec3::ZERO;
    }

    let geometry = light_v.vtx.ng.dot(w_lc).abs() * camera_v.vtx.ng.dot(w_cl).abs() / dist2;
    if geometry <= 0.0 {
        return Vec3::ZERO;
    }

    light_v.beta * f_light * geometry * f_camera * camera_v.beta
}

fn sample_area_light_triangle(scene: &Scene, u: f32) -> Option<(AreaLightTriangle, f32)> {
    if scene.area_light_weight_sum <= 0.0 {
        return None;
    }
    let target = u.clamp(0.0, 1.0) * scene.area_light_weight_sum;
    let index = scene
        .area_light_triangles
        .partition_point(|area_light| area_light.prefix_weight < target)
        .min(scene.area_light_triangles.len().saturating_sub(1));
    let area_light = *scene.area_light_triangles.get(index)?;
    let pmf = area_light.weight / scene.area_light_weight_sum;
    Some((area_light, pmf))
}

fn triangle_geometric_normal(scene: &Scene, triangle: TriangleRef) -> Vec3 {
    let [p0, p1, p2] = scene.triangle_positions(triangle);
    (p1 - p0).cross(p2 - p0).normalize_or_zero()
}

fn visible_between(
    scene: &Scene,
    origin: Vec3,
    origin_ng: Vec3,
    direction: Vec3,
    distance: f32,
    target_triangle: Option<TriangleRef>,
    aux_rng: &mut AuxRng,
    mtlx_scratch: &mut MtlxScratch,
) -> bool {
    let shadow_ray = spawn_ray(origin, origin_ng, direction);
    let hit = scene
        .closest_hit(&shadow_ray, aux_rng, mtlx_scratch)
        .expect("scene.build_qbvh() must be called before traversal");

    match hit {
        None => true,
        Some(hit) => {
            if target_triangle.is_some_and(|target| hit.triangle == target) {
                true
            } else {
                hit.t >= distance * (1.0 - SHADOW_TOLERANCE)
            }
        }
    }
}

fn solid_angle_pdf_to_area(pdf_solid_angle: f32, from: Vec3, to: Vec3, to_ng: Vec3) -> f32 {
    if pdf_solid_angle <= 0.0 {
        return 0.0;
    }
    let delta = to - from;
    let dist2 = delta.length_squared();
    if dist2 <= 0.0 {
        return 0.0;
    }
    let wi = delta.normalize_or_zero();
    let cos_to = to_ng.dot(-wi).abs();
    if cos_to <= 0.0 {
        return 0.0;
    }
    pdf_solid_angle * cos_to / dist2
}

fn area_light_origin_pdf_area(scene: &Scene, vertex: &PathVertex) -> f32 {
    let Some(area_light) = scene
        .area_light_triangles
        .iter()
        .find(|area_light| area_light.triangle == vertex.vtx.triangle)
    else {
        return 0.0;
    };

    if scene.area_light_weight_sum <= 0.0 || area_light.area <= 0.0 {
        return 0.0;
    }

    area_light.weight / scene.area_light_weight_sum / area_light.area
}

fn light_emission_pdf_area(light: &PathVertex, target: &PathVertex) -> f32 {
    let delta = target.vtx.p - light.vtx.p;
    let dist2 = delta.length_squared();
    if dist2 <= 0.0 {
        return 0.0;
    }

    let direction = delta / dist2.sqrt();
    let cos_emit = light.vtx.ng.dot(direction).max(0.0);
    if cos_emit <= 0.0 {
        return 0.0;
    }

    solid_angle_pdf_to_area(cos_emit / PI, light.vtx.p, target.vtx.p, target.vtx.ng)
}

fn bsdf_pdf_area(
    scene: &Scene,
    from: &PathVertex,
    previous: Option<&PathVertex>,
    target: &PathVertex,
    mtlx_scratch: &MtlxScratch,
) -> f32 {
    let delta = target.vtx.p - from.vtx.p;
    if delta.length_squared() <= 0.0 {
        return 0.0;
    }

    let mut vtx = from.vtx.clone();
    if let Some(previous) = previous {
        let to_previous = previous.vtx.p - from.vtx.p;
        if to_previous.length_squared() <= 0.0 {
            return 0.0;
        }
        vtx.wo = to_previous.normalize_or_zero();
    }

    let wi = delta.normalize_or_zero();
    let material = scene.instance_material(from.vtx.triangle.instance_index);
    let pdf_dir = material.pdf(&vtx, mtlx_scratch, wi);
    solid_angle_pdf_to_area(pdf_dir, from.vtx.p, target.vtx.p, target.vtx.ng)
}

fn vertex_pdf_area(
    scene: &Scene,
    from: &PathVertex,
    previous: Option<&PathVertex>,
    target: &PathVertex,
    mtlx_scratch: &MtlxScratch,
) -> f32 {
    match from.vertex_type {
        VertexType::Light => light_emission_pdf_area(from, target),
        VertexType::Surface => bsdf_pdf_area(scene, from, previous, target, mtlx_scratch),
    }
}

fn camera_light_hit_mis_weight(scene: &Scene, camera_path: &[PathVertex]) -> f32 {
    if camera_path.len() <= 1 {
        return 1.0;
    }

    let pt_index = camera_path.len() - 1;
    let pt_minus_index = pt_index.checked_sub(1);
    let pt_pdf_rev = area_light_origin_pdf_area(scene, &camera_path[pt_index]);
    let pt_minus_pdf_rev = pt_minus_index
        .map(|i| light_emission_pdf_area(&camera_path[pt_index], &camera_path[i]))
        .unwrap_or(0.0);

    let mut sum_ri = 0.0_f32;
    let mut ri = 1.0_f32;

    for i in (1..camera_path.len()).rev() {
        let pdf_rev_area = if i == pt_index {
            pt_pdf_rev
        } else if Some(i) == pt_minus_index {
            pt_minus_pdf_rev
        } else {
            camera_path[i].pdf_rev_area
        };

        ri *= remap0(pdf_rev_area) / remap0(camera_path[i].pdf_fwd_area);
        if !camera_path[i].delta && !camera_path[i - 1].delta {
            sum_ri += ri;
        }
    }

    1.0 / (1.0 + sum_ri.max(0.0))
}

fn bdpt_mis_weight(
    scene: &Scene,
    light_path: &[PathVertex],
    camera_path: &[PathVertex],
    mtlx_scratch: &MtlxScratch,
) -> f32 {
    if light_path.len() + camera_path.len() <= 2 {
        return 1.0;
    }

    let qs_index = light_path.len() - 1;
    let qs_minus_index = qs_index.checked_sub(1);
    let pt_index = camera_path.len() - 1;
    let pt_minus_index = pt_index.checked_sub(1);

    let qs = &light_path[qs_index];
    let qs_minus = qs_minus_index.map(|i| &light_path[i]);
    let pt = &camera_path[pt_index];
    let pt_minus = pt_minus_index.map(|i| &camera_path[i]);

    let pt_pdf_rev = vertex_pdf_area(scene, qs, qs_minus, pt, mtlx_scratch);
    let pt_minus_pdf_rev = pt_minus.map_or(0.0, |pt_minus| {
        vertex_pdf_area(scene, pt, Some(qs), pt_minus, mtlx_scratch)
    });
    let qs_pdf_rev = vertex_pdf_area(scene, pt, pt_minus, qs, mtlx_scratch);
    let qs_minus_pdf_rev = qs_minus.map_or(0.0, |qs_minus| {
        vertex_pdf_area(scene, qs, Some(pt), qs_minus, mtlx_scratch)
    });

    let mut sum_ri = 0.0_f32;
    let mut ri = 1.0_f32;

    for i in (1..camera_path.len()).rev() {
        let pdf_rev_area = if i == pt_index {
            pt_pdf_rev
        } else if Some(i) == pt_minus_index {
            pt_minus_pdf_rev
        } else {
            camera_path[i].pdf_rev_area
        };

        ri *= remap0(pdf_rev_area) / remap0(camera_path[i].pdf_fwd_area);
        if !camera_path[i].delta && !camera_path[i - 1].delta {
            sum_ri += ri;
        }
    }

    ri = 1.0;
    for i in (0..light_path.len()).rev() {
        let pdf_rev_area = if i == qs_index {
            qs_pdf_rev
        } else if Some(i) == qs_minus_index {
            qs_minus_pdf_rev
        } else {
            light_path[i].pdf_rev_area
        };

        ri *= remap0(pdf_rev_area) / remap0(light_path[i].pdf_fwd_area);
        let previous_is_delta = i > 0 && light_path[i - 1].delta;
        if !light_path[i].delta && !previous_is_delta {
            sum_ri += ri;
        }
    }

    1.0 / (1.0 + sum_ri.max(0.0))
}

fn remap0(x: f32) -> f32 {
    if x == 0.0 { 1.0 } else { x }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use glam::{Mat4, Vec2, Vec3};

    use super::super::test_helpers::{
        floor_with_area_light_scene, floor_with_directional_light_scene,
        floor_with_point_light_scene, floor_with_spot_light_scene, mirror_to_light_scene,
        sample_floor_vtx, triangle_ref, unit_mesh,
    };
    use super::{
        direct_non_area_light_contribution, emitted_radiance_from_infinite_hit,
        scene_has_non_area_lights, trace_radiance,
    };
    use crate::{
        bsdf::TransportMode,
        light::{
            EnvironmentLight, LightSampleContext, LightType, sample_light_mis_compensated_lazy,
        },
        material::{Material, MtlxScratch, NormalizedLambertMaterial},
        math::OrthonormalBasis,
        math::ray::Ray,
        sampler::{AuxRng, PathSampler},
        scene::Scene,
    };

    #[test]
    fn scene_has_non_area_lights_detects_environment_and_delta_lights() {
        assert!(!scene_has_non_area_lights(&floor_with_area_light_scene(
            0.8, 10.0
        )));
        assert!(scene_has_non_area_lights(&floor_with_point_light_scene(
            0.8, 1.0
        )));
        assert!(scene_has_non_area_lights(
            &floor_with_directional_light_scene(0.8, 1.0)
        ));
        assert!(scene_has_non_area_lights(&floor_with_spot_light_scene(
            0.8,
            1.0,
            30f32.to_radians(),
            20f32.to_radians(),
            Vec3::NEG_Z,
        )));

        let mut scene = Scene::new();
        scene.set_environment_light(EnvironmentLight::from_pixels(
            4,
            2,
            vec![Vec3::ONE; 8],
            1.0,
            0.0,
        ));
        assert!(scene_has_non_area_lights(&scene));
    }

    #[test]
    fn direct_non_area_light_sampling_skips_area_light_samples() {
        let scene = floor_with_area_light_scene(0.8, 10.0);
        let vtx = sample_floor_vtx(&scene, Vec3::NEG_Z);
        let material = scene.instance_material(triangle_ref(0).instance_index);
        let radiance = direct_non_area_light_contribution(
            &scene,
            material,
            &vtx,
            0.0,
            0.0,
            0.5,
            Vec2::new(0.25, 0.5),
            &mut AuxRng::default(),
            &mut MtlxScratch::default(),
        );

        assert_eq!(radiance, Vec3::ZERO);
    }

    #[test]
    fn direct_non_area_light_sampling_matches_point_light_estimator() {
        let scene = floor_with_point_light_scene(0.8, 16.0 * PI);
        let vtx = sample_floor_vtx(&scene, Vec3::NEG_Z);
        let material = scene.instance_material(triangle_ref(0).instance_index);
        let radiance = direct_non_area_light_contribution(
            &scene,
            material,
            &vtx,
            0.0,
            0.0,
            0.0,
            Vec2::splat(0.5),
            &mut AuxRng::default(),
            &mut MtlxScratch::default(),
        );

        assert!(radiance.abs_diff_eq(Vec3::splat(0.8 / PI), 1.0e-3));
    }

    #[test]
    fn direct_non_area_light_sampling_matches_directional_light_estimator() {
        let scene = floor_with_directional_light_scene(0.8, 2.0);
        let vtx = sample_floor_vtx(&scene, Vec3::NEG_Z);
        let material = scene.instance_material(triangle_ref(0).instance_index);
        let radiance = direct_non_area_light_contribution(
            &scene,
            material,
            &vtx,
            0.0,
            0.0,
            0.0,
            Vec2::ZERO,
            &mut AuxRng::default(),
            &mut MtlxScratch::default(),
        );

        assert!(radiance.abs_diff_eq(Vec3::splat(2.0 * 0.8 / PI), 1.0e-3));
    }

    #[test]
    fn direct_non_area_light_sampling_matches_spot_light_estimator() {
        let scene = floor_with_spot_light_scene(
            0.8,
            16.0 * PI,
            30f32.to_radians(),
            20f32.to_radians(),
            Vec3::NEG_Z,
        );
        let vtx = sample_floor_vtx(&scene, Vec3::NEG_Z);
        let material = scene.instance_material(triangle_ref(0).instance_index);
        let radiance = direct_non_area_light_contribution(
            &scene,
            material,
            &vtx,
            0.0,
            0.0,
            0.0,
            Vec2::ZERO,
            &mut AuxRng::default(),
            &mut MtlxScratch::default(),
        );

        assert!(radiance.abs_diff_eq(Vec3::splat(0.8 / PI), 1.0e-3));
    }

    #[test]
    fn direct_non_area_light_sampling_includes_environment_light() {
        let mut scene = Scene::new();
        let floor_mesh = scene.add_mesh(unit_mesh(0.0));
        let floor_material = scene.add_material(Material::NormalizedLambert(
            NormalizedLambertMaterial::new(Vec3::splat(0.8)),
        ));
        scene.add_instance(floor_mesh, floor_material, Mat4::from_rotation_x(-0.5 * PI));
        let mut pixels = vec![Vec3::ZERO; 4 * 2];
        pixels[0] = Vec3::splat(4.0);
        scene.set_environment_light(EnvironmentLight::from_pixels(4, 2, pixels, 1.0, 0.0));
        scene.build_qbvh();
        scene.build_light_tree();

        let env_dir = scene
            .environment_light()
            .and_then(|env| env.sample_mis_compensated(Vec2::splat(0.5)))
            .expect("non-uniform environment must provide a compensated sample")
            .direction;
        let mut vtx = sample_floor_vtx(&scene, -env_dir);
        vtx.ng = env_dir;
        vtx.ns = env_dir;
        vtx.wo = env_dir;
        vtx.frame = OrthonormalBasis::from_normal(env_dir);
        let material = scene.instance_material(triangle_ref(0).instance_index);
        let mut scratch = MtlxScratch::default();
        let sampled = sample_light_mis_compensated_lazy(
            &scene,
            &LightSampleContext::from_vertex(&vtx),
            &vtx,
            material,
            0.0,
            0.0,
            0.0,
            Vec2::splat(0.5),
            &mut scratch,
        )
        .expect("environment light must be sampled");
        assert_eq!(sampled.sample.light_type, LightType::Infinite);
        assert!(sampled.sample.radiance.x > 0.0);
        let f = material.eval(
            &vtx,
            &scratch,
            sampled.sample.wi,
            &mut AuxRng::default(),
            TransportMode::Radiance,
        );
        assert!(f.x > 0.0);
        let radiance = direct_non_area_light_contribution(
            &scene,
            material,
            &vtx,
            0.0,
            0.0,
            0.0,
            Vec2::splat(0.5),
            &mut AuxRng::default(),
            &mut scratch,
        );

        assert!(radiance.x > 0.0);
        assert!(radiance.y > 0.0);
        assert!(radiance.z > 0.0);
    }

    #[test]
    fn direct_non_area_light_sampling_returns_zero_when_delta_light_is_occluded() {
        let mut scene = floor_with_point_light_scene(0.8, 16.0 * PI);
        let blocker_mesh = scene.add_mesh(unit_mesh(1.0));
        let blocker_material = scene.add_material(Material::NormalizedLambert(
            NormalizedLambertMaterial::new(Vec3::splat(0.5)),
        ));
        scene.add_instance(blocker_mesh, blocker_material, Mat4::IDENTITY);
        scene.build_qbvh();
        scene.build_light_tree();

        let vtx = sample_floor_vtx(&scene, Vec3::NEG_Z);
        let material = scene.instance_material(triangle_ref(0).instance_index);
        let radiance = direct_non_area_light_contribution(
            &scene,
            material,
            &vtx,
            0.0,
            0.0,
            0.0,
            Vec2::ZERO,
            &mut AuxRng::default(),
            &mut MtlxScratch::default(),
        );

        assert_eq!(radiance, Vec3::ZERO);
    }

    #[test]
    fn emitted_radiance_from_infinite_hit_uses_primary_and_delta_full_weight() {
        let mut scene = Scene::new();
        let env_radiance = Vec3::new(0.2, 0.4, 0.8);
        scene.set_environment_light(EnvironmentLight::from_pixels(
            16,
            8,
            vec![env_radiance; 16 * 8],
            1.0,
            0.0,
        ));
        scene.build_qbvh();
        scene.build_light_tree();

        let radiance = emitted_radiance_from_infinite_hit(&scene, Vec3::ONE, 0.0, true, Vec3::Y);

        assert!(radiance.abs_diff_eq(env_radiance, 1.0e-5));
    }

    #[test]
    fn emitted_radiance_from_infinite_hit_applies_mis_for_non_delta_bsdf() {
        let mut scene = Scene::new();
        let env_radiance = Vec3::splat(2.0);
        scene.set_environment_light(EnvironmentLight::from_pixels(
            16,
            8,
            vec![env_radiance; 16 * 8],
            1.0,
            0.0,
        ));
        scene.build_qbvh();
        scene.build_light_tree();

        let zero = emitted_radiance_from_infinite_hit(&scene, Vec3::ONE, 0.0, false, Vec3::Y);
        assert_eq!(zero, Vec3::ZERO);

        let high_bsdf =
            emitted_radiance_from_infinite_hit(&scene, Vec3::ONE, 1000.0, false, Vec3::Y);
        assert!(high_bsdf.x > env_radiance.x * 0.9);
    }

    #[test]
    fn trace_radiance_counts_light_after_delta_bounce() {
        let (scene, ray, expected) = mirror_to_light_scene();
        let sampler = PathSampler::new(glam::UVec2::ZERO, 0, 1, glam::UVec2::new(1, 1));

        let radiance = trace_radiance(&scene, ray, &sampler, 2, &mut MtlxScratch::default());

        assert!(radiance.abs_diff_eq(expected, 1.0e-3));
    }

    #[test]
    fn trace_radiance_direct_hit_on_environment_reads_primary_radiance() {
        let mut scene = Scene::new();
        let env_radiance = Vec3::new(0.1, 0.4, 0.9);
        scene.set_environment_light(EnvironmentLight::from_pixels(
            16,
            8,
            vec![env_radiance; 16 * 8],
            1.0,
            0.0,
        ));
        scene.build_qbvh();
        scene.build_light_tree();

        let ray = Ray::new(Vec3::ZERO, Vec3::Y);
        let sampler = PathSampler::new(glam::UVec2::ZERO, 0, 1, glam::UVec2::new(1, 1));
        let radiance = trace_radiance(&scene, ray, &sampler, 4, &mut MtlxScratch::default());

        assert!(radiance.abs_diff_eq(env_radiance, 1.0e-5));
    }

    #[test]
    fn trace_radiance_gets_positive_delta_light_contribution() {
        let scene = floor_with_point_light_scene(0.8, 16.0 * PI);
        let ray = Ray::new(Vec3::new(0.25, 0.25, 1.0), Vec3::NEG_Z);
        let sampler = PathSampler::new(glam::UVec2::ZERO, 0, 1, glam::UVec2::new(1, 1));
        let radiance = trace_radiance(&scene, ray, &sampler, 1, &mut MtlxScratch::default());

        assert!(radiance.x > 0.0);
        assert!(radiance.y > 0.0);
        assert!(radiance.z > 0.0);
    }
}
