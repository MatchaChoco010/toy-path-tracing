use std::{error::Error, path::Path};

use glam::{Mat4, Vec3};

use crate::{
    light::EnvironmentLight,
    material::{Material, OpenPbrMaterial},
    scene::PinholeCamera,
    scene::{MaterialIndex, MeshIndex, Scene},
    scene::{load_gltf, load_obj},
};

use super::super::uniform_scale_for_height;

pub(in crate::scenes) fn create_single_openpbr_bunny_scene(
    bunny_material: OpenPbrMaterial,
) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    create_single_openpbr_bunny_scene_impl(
        bunny_material,
        Path::new("assets/models/bunny.glb"),
        Some(std::f32::consts::PI * -0.45),
    )
}

pub(in crate::scenes) fn create_single_openpbr_low_bunny_scene(
    bunny_material: OpenPbrMaterial,
) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    create_single_openpbr_bunny_scene_impl(
        bunny_material,
        Path::new("assets/models/bunny-low.glb"),
        Some(std::f32::consts::PI * -0.5),
    )
}

fn create_single_openpbr_bunny_scene_impl(
    bunny_material: OpenPbrMaterial,
    bunny_path: &Path,
    environment_rotation: Option<f32>,
) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();

    let floor = load_obj(Path::new("assets/mori-knob/floor.obj"))?;
    let light = load_obj(Path::new("assets/mori-knob/light.obj"))?;
    let bunny = load_gltf(bunny_path)?;

    let bunny_height = 0.94_f32;
    let bunny_scale = uniform_scale_for_height(&bunny, bunny_height);
    let bunny_ground_pivot = Vec3::new(
        bunny.bounds.center().x,
        bunny.bounds.min.y,
        bunny.bounds.center().z,
    );

    let floor_mesh = scene.add_mesh(floor);
    let light_mesh = scene.add_mesh(light);
    let bunny_mesh = scene.add_mesh(bunny);

    let floor_material = scene.add_material(Material::OpenPbr(
        OpenPbrMaterial::new(Vec3::splat(0.018))
            .with_specular_roughness(0.85)
            .with_base_diffuse_roughness(0.85),
    ));
    let light_material = scene.add_material(Material::OpenPbr(
        OpenPbrMaterial::new(Vec3::ZERO)
            .with_base_weight(0.0)
            .with_specular_weight(0.0)
            .with_emission_color(Vec3::ONE)
            .with_emission_luminance(15.0),
    ));

    let room = Mat4::from_scale(Vec3::splat(0.55));
    scene.add_instance(floor_mesh, floor_material, room);
    scene.add_instance(light_mesh, light_material, room);

    let bunny_material = scene.add_material(Material::OpenPbr(bunny_material));
    add_bunny(
        &mut scene,
        bunny_mesh,
        bunny_material,
        Vec3::new(0.0, -0.26, 0.0),
        bunny_scale,
        bunny_ground_pivot,
    );

    if let Some(environment_rotation) = environment_rotation {
        let env = EnvironmentLight::from_hdr_file(
            "assets/sky/brown_photostudio_02_4k.hdr",
            1.2,
            environment_rotation,
        )?;
        scene.set_environment_light(env);
    }

    let camera = PinholeCamera::new(
        Vec3::new(0.0, 0.78, -1.95),
        Vec3::new(0.0, 0.22, 0.0),
        Vec3::Y,
        40.0_f32.to_radians(),
        1.0,
    );

    Ok((scene, camera))
}

fn add_bunny(
    scene: &mut Scene,
    mesh: MeshIndex,
    material: MaterialIndex,
    position: Vec3,
    scale: f32,
    ground_pivot: Vec3,
) {
    let transform = Mat4::from_translation(position)
        * Mat4::from_rotation_y(std::f32::consts::PI)
        * Mat4::from_scale(Vec3::splat(scale))
        * Mat4::from_translation(-ground_pivot);
    scene.add_instance(mesh, material, transform);
}
