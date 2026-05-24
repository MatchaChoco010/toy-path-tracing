//! mori-knob の床に retroreflective 反射材つき三角コーンと通常白プラスチック反射材の三角コーンを並べる。

use std::{error::Error, path::Path};

use glam::{Mat4, Vec2, Vec3};

use crate::{
    material::{EmissiveMaterial, Material, NormalizedLambertMaterial},
    scene::PinholeCamera,
    scene::Scene,
    scene::load_gltf,
    scene::load_obj,
    scene::mtlx_loader::{load_mtlx_material, load_standard_library},
    scene::{Bounds, Mesh, Vertex},
};

pub fn create_scene_67(
    ocio: &crate::color::OcioColorPipeline,
) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();

    let library = load_standard_library(Path::new("lib/materialx/libraries"))?;
    let material_path = Path::new("assets/mtlx/traffic_cone_retroreflective.mtlx");

    let floor_scale = 0.7_f32;
    let floor_data = load_obj(Path::new("assets/mori-knob/floor.obj"))?;
    let floor_ground_y = floor_data.bounds.min.y * floor_scale;
    let floor_mesh = scene.add_mesh(floor_data);
    let cone_mesh_data = load_gltf(Path::new("assets/models/cone.glb"))?;
    let reflector_mesh_data = load_gltf(Path::new("assets/models/cone-reflector.glb"))?;
    let cone_bounds = cone_mesh_data.bounds.union(reflector_mesh_data.bounds);
    let cone_mesh = scene.add_mesh(cone_mesh_data);
    let reflector_mesh = scene.add_mesh(reflector_mesh_data);
    let light_mesh = scene.add_mesh(camera_back_light_mesh());

    let floor_material = scene.add_material(Material::NormalizedLambert(
        NormalizedLambertMaterial::new(Vec3::splat(0.55)),
    ));
    let light_material =
        scene.add_material(Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 22.0)));
    let orange = scene.add_material(Material::Mtlx(load_mtlx_material(
        &library,
        material_path,
        "ConeOrangePlastic",
        ocio,
    )?));
    let white = scene.add_material(Material::Mtlx(load_mtlx_material(
        &library,
        material_path,
        "ConeWhitePlastic",
        ocio,
    )?));
    let retro = scene.add_material(Material::Mtlx(load_mtlx_material(
        &library,
        material_path,
        "ConeRetroreflectiveWhite",
        ocio,
    )?));

    scene.add_instance(
        floor_mesh,
        floor_material,
        Mat4::from_scale(Vec3::splat(floor_scale)),
    );
    scene.add_instance(
        light_mesh,
        light_material,
        Mat4::from_translation(Vec3::new(0.0, 0.55, -3.0)) * Mat4::from_scale(Vec3::splat(0.9)),
    );

    let left = cone_transform(cone_bounds, Vec3::new(-0.45, floor_ground_y, 0.0));
    let right = cone_transform(cone_bounds, Vec3::new(0.45, floor_ground_y, 0.0));
    scene.add_instance(cone_mesh, orange, left);
    scene.add_instance(reflector_mesh, retro, left);
    scene.add_instance(cone_mesh, orange, right);
    scene.add_instance(reflector_mesh, white, right);

    let camera = PinholeCamera::new(
        Vec3::new(0.0, 0.45, -2.4),
        Vec3::new(0.0, 0.05, 0.0),
        Vec3::Y,
        40.0_f32.to_radians(),
        1.0,
    );

    Ok((scene, camera))
}

fn cone_transform(bounds: Bounds, position: Vec3) -> Mat4 {
    let target_height = 0.9_f32;
    let scale = target_height / bounds.extent().y.max(1.0e-3);
    let anchor = Vec3::new(bounds.center().x, bounds.min.y, bounds.center().z);
    Mat4::from_translation(position)
        * Mat4::from_scale(Vec3::splat(scale))
        * Mat4::from_translation(-anchor)
}

fn camera_back_light_mesh() -> Mesh {
    let normal = Vec3::Z;
    Mesh::new(
        vec![
            Vertex {
                position: Vec3::new(-0.5, -0.5, 0.0),
                normal,
                uv: Vec2::new(0.0, 0.0),
            },
            Vertex {
                position: Vec3::new(0.5, -0.5, 0.0),
                normal,
                uv: Vec2::new(1.0, 0.0),
            },
            Vertex {
                position: Vec3::new(0.5, 0.5, 0.0),
                normal,
                uv: Vec2::new(1.0, 1.0),
            },
            Vertex {
                position: Vec3::new(-0.5, 0.5, 0.0),
                normal,
                uv: Vec2::new(0.0, 1.0),
            },
        ],
        vec![0, 1, 2, 0, 2, 3],
    )
}
