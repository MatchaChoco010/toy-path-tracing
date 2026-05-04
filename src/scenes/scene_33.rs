use std::{error::Error, path::Path};

use glam::{Mat4, Vec3};

use crate::{
    camera::PinholeCamera,
    light::EnvironmentLight,
    material::{ConductorGgxMaterial, EmissiveMaterial, Material, StandardSurfaceMaterial},
    mesh::{load_gltf, load_obj},
    scene::Scene,
};

use super::uniform_scale_for_height;

pub fn create_scene_33() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    create_conductor_roughness_row(|color, roughness| {
        Material::ConductorGgx(ConductorGgxMaterial::new(color, roughness, 0.0))
    })
}

pub(super) fn create_conductor_roughness_row<F>(
    make_material: F,
) -> Result<(Scene, PinholeCamera), Box<dyn Error>>
where
    F: Fn(Vec3, f32) -> Material,
{
    let mut scene = Scene::new();

    let floor = load_obj(Path::new("assets/mori-knob/floor.obj"))?;
    let floor_mesh = scene.add_mesh(floor);
    let floor_material = scene.add_material(Material::StandardSurface(
        StandardSurfaceMaterial::new(Vec3::splat(0.72))
            .with_specular_roughness(0.6)
            .with_diffuse_roughness(0.3),
    ));

    let stage_scale = 1.4_f32;
    let world_root = Mat4::from_scale(Vec3::splat(stage_scale));
    scene.add_instance(floor_mesh, floor_material, world_root);

    let sphere = load_gltf(Path::new("assets/models/sphere.glb"))?;
    let sphere_height = 0.55_f32;
    let sphere_scale = uniform_scale_for_height(&sphere, sphere_height);
    let sphere_pivot = Vec3::new(
        sphere.bounds.center().x,
        sphere.bounds.min.y,
        sphere.bounds.center().z,
    );
    let sphere_mesh = scene.add_mesh(sphere);

    let sphere_count = 9_usize;
    let spacing = 0.65_f32;
    let silver = Vec3::splat(0.92);
    let row_z = 0.0_f32;
    let center_offset = (sphere_count as f32 - 1.0) * 0.5;

    for i in 0..sphere_count {
        let roughness = i as f32 / (sphere_count as f32 - 1.0);
        let material = scene.add_material(make_material(silver, roughness));
        let x = (center_offset - i as f32) * spacing;
        let transform = Mat4::from_translation(Vec3::new(x, 0.0, row_z))
            * Mat4::from_scale(Vec3::splat(sphere_scale))
            * Mat4::from_translation(-sphere_pivot);
        scene.add_instance(sphere_mesh, material, transform);
    }

    let env = EnvironmentLight::from_hdr_file(
        "assets/sky/studio_small_08_4k.hdr",
        0.2,
        std::f32::consts::PI * -0.5,
    )?;
    scene.set_environment_light(env);

    let light = load_obj(Path::new("assets/mori-knob/light.obj"))?;
    let light_mesh = scene.add_mesh(light);
    let light_material =
        scene.add_material(Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 10.0)));
    scene.add_instance(light_mesh, light_material, world_root);

    let camera_eye = Vec3::new(0.0, 0.75, -4.5);
    let camera_target = Vec3::new(0.0, sphere_height * 0.5, row_z);
    let camera = PinholeCamera::new(
        camera_eye,
        camera_target,
        Vec3::Y,
        55.0_f32.to_radians(),
        1.0,
    );

    Ok((scene, camera))
}
