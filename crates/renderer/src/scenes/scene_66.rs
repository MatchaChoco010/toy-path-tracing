//! sphere を 3 x 3 グリッドに配置し、MaterialX 1.39.5 の flake3d ノードを使った coated metallic flake を比較する。

use std::{error::Error, path::Path};

use glam::{Mat4, Vec3};

use crate::{
    light::EnvironmentLight,
    material::{EmissiveMaterial, Material, NormalizedLambertMaterial},
    scene::PinholeCamera,
    scene::Scene,
    scene::load_gltf,
    scene::load_obj,
    scene::mtlx_loader::{load_mtlx_material, load_standard_library},
    scenes::uniform_scale_for_height,
};

pub fn create_scene_66(
    ocio: &crate::color::OcioColorPipeline,
) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();

    let library = load_standard_library(Path::new("lib/materialx/libraries"))?;
    let material_path = Path::new("assets/mtlx/standard_surface_colored_flake_metal.mtlx");

    let floor_scale = 0.55_f32;
    let floor = load_obj(Path::new("assets/mori-knob/floor.obj"))?;
    let light = load_obj(Path::new("assets/mori-knob/light.obj"))?;
    let floor_ground_y = floor.bounds.min.y * floor_scale;
    let sphere = load_gltf(Path::new("assets/models/sphere.glb"))?;
    let sphere_scale = uniform_scale_for_height(&sphere, 0.56);
    let sphere_pivot = Vec3::new(
        sphere.bounds.center().x,
        sphere.bounds.min.y,
        sphere.bounds.center().z,
    );

    let floor_mesh = scene.add_mesh(floor);
    let light_mesh = scene.add_mesh(light);
    let sphere_mesh = scene.add_mesh(sphere);

    let floor_material = scene.add_material(Material::NormalizedLambert(
        NormalizedLambertMaterial::new(Vec3::splat(0.32)),
    ));
    let light_material =
        scene.add_material(Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 45.0)));

    let world_root = Mat4::from_scale(Vec3::splat(floor_scale));
    scene.add_instance(floor_mesh, floor_material, world_root);
    scene.add_instance(light_mesh, light_material, world_root);

    let materials = [
        "FlakeMetal_size_0_005",
        "FlakeMetal_size_0_01",
        "FlakeMetal_size_0_02",
        "FlakeMetal_rough_005",
        "FlakeMetal_rough_025",
        "FlakeMetal_rough_050",
        "FlakeMetal_coverage_010",
        "FlakeMetal_coverage_050",
        "FlakeMetal_coverage_100",
    ];

    let spacing = 0.78_f32;
    for row in 0..3 {
        for col in 0..3 {
            let index = row * 3 + col;
            let offset_x = (1.0 - col as f32) * spacing;
            let offset_z = (row as f32 - 1.0) * spacing;
            let transform = Mat4::from_translation(Vec3::new(offset_x, floor_ground_y, offset_z))
                * Mat4::from_scale(Vec3::splat(sphere_scale))
                * Mat4::from_translation(-sphere_pivot);
            let mtlx_material =
                load_mtlx_material(&library, material_path, materials[index], ocio)?;
            let sphere_material = scene.add_material(Material::Mtlx(mtlx_material));
            scene.add_instance(sphere_mesh, sphere_material, transform);
        }
    }

    let env = EnvironmentLight::from_hdr_file(
        "assets/sky/kloofendal_48d_partly_cloudy_puresky_4k.hdr",
        5.0,
        0.0,
    )?;
    scene.set_environment_light(env);

    let camera = PinholeCamera::new(
        Vec3::new(0.0, 1.78, -2.12),
        Vec3::new(0.0, 0.0, -0.16),
        Vec3::Y,
        42.0_f32.to_radians(),
        1.0,
    );

    Ok((scene, camera))
}
