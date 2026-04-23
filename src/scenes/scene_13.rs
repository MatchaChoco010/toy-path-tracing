use glam::{Mat4, Vec3};
use std::{error::Error, path::Path};

use crate::{
    camera::PinholeCamera,
    light::EnvironmentLight,
    material::{Material, NormalizedLambertMaterial},
    mesh::load_mesh,
    scene::Scene,
    scenes::{game_rotation_degrees, uniform_scale_for_height},
};

pub fn create_scene_13() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    create_sky_bunny_scene("assets/sky/brown_photostudio_02_4k.hdr", 2.0)
}

pub(super) fn create_sky_bunny_scene(
    sky_path: &'static str,
    sky_scale: f32,
) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();

    let floor_material = scene.add_material(Material::NormalizedLambert(
        NormalizedLambertMaterial::new(Vec3::splat(0.62)),
    ));
    let floor_mesh = scene.add_mesh(load_mesh(Path::new("assets/gltf/floor.glb"))?);
    let floor_transform = Mat4::from_scale(Vec3::new(12.0, 1.0, 12.0));
    scene.add_instance(floor_mesh, floor_material, floor_transform);

    let bunny_material = scene.add_material(Material::NormalizedLambert(
        NormalizedLambertMaterial::new(Vec3::new(0.74, 0.76, 0.72)),
    ));
    let bunny = load_mesh(Path::new("assets/gltf/bunny.glb"))?;
    let bunny_scale = uniform_scale_for_height(&bunny, 2.35);
    let bunny_pivot = Vec3::new(
        bunny.bounds.center().x,
        bunny.bounds.min.y,
        bunny.bounds.center().z,
    );
    let bunny_mesh = scene.add_mesh(bunny);
    let bunny_transform = Mat4::from_translation(Vec3::new(0.0, 0.0, -0.15))
        * Mat4::from_quat(game_rotation_degrees(0.0, -20.0, 0.0))
        * Mat4::from_scale(Vec3::splat(bunny_scale))
        * Mat4::from_translation(-bunny_pivot);
    scene.add_instance(bunny_mesh, bunny_material, bunny_transform);

    let env = EnvironmentLight::from_hdr_file(sky_path, sky_scale)?;
    scene.set_environment_light(env);

    let camera = PinholeCamera::new(
        Vec3::new(0.0, 2.0, 7.0),
        Vec3::new(0.0, 1.15, -0.1),
        Vec3::Y,
        38.0_f32.to_radians(),
    );

    Ok((scene, camera))
}
