use glam::{Mat4, Vec3};
use std::{error::Error, path::Path};

use crate::{
    camera::PinholeCamera,
    light::EnvironmentLight,
    material::{Material, NormalizedLambertMaterial, SimplePbrMaterial},
    mesh::{load_gltf, load_obj},
    scene::Scene,
};

use super::{game_rotation_degrees, uniform_scale_for_height};

pub fn create_scene_20() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();

    let world_rotation = Mat4::from_rotation_y(45.0_f32.to_radians());

    let floor_material = scene.add_material(Material::NormalizedLambert(
        NormalizedLambertMaterial::new(Vec3::splat(0.9)),
    ));
    let floor_mesh = scene.add_mesh(load_gltf(Path::new("assets/models/floor.glb"))?);
    let floor_transform = world_rotation * Mat4::from_scale(Vec3::new(18.0, 1.0, 18.0));
    scene.add_instance(floor_mesh, floor_material, floor_transform);

    let dragon_material = scene.add_material(Material::SimplePBR(
        SimplePbrMaterial::try_new_with_texture_paths(
            Vec3::ONE,
            1.0,
            1.0,
            1.5,
            0.0,
            Some(Path::new("assets/models/dragon-BaseColor.png")),
            Some(Path::new("assets/models/dragon-Metallic.png")),
            Some(Path::new("assets/models/dragon-Roughness.png")),
            Some(Path::new("assets/models/dragon-Normal.png")),
        )?,
    ));

    let dragon = load_obj(Path::new("assets/models/dragon.obj"))?;
    let dragon_scale = uniform_scale_for_height(&dragon, 2.35);
    let dragon_pivot = Vec3::new(
        dragon.bounds.center().x,
        dragon.bounds.min.y,
        dragon.bounds.center().z,
    );
    let dragon_mesh = scene.add_mesh(dragon);
    let dragon_transform = world_rotation
        * Mat4::from_quat(game_rotation_degrees(0.0, -35.0, 0.0))
        * Mat4::from_scale(Vec3::splat(dragon_scale))
        * Mat4::from_translation(-dragon_pivot);
    scene.add_instance(dragon_mesh, dragon_material, dragon_transform);

    let env = EnvironmentLight::from_hdr_file(
        "assets/sky/kloofendal_48d_partly_cloudy_puresky_4k.hdr",
        0.5,
        0.0,
    )?;
    scene.set_environment_light(env);

    let camera_eye = world_rotation.transform_point3(Vec3::new(-3.9, 2.7, -6.2));
    let camera_target = world_rotation.transform_point3(Vec3::new(0.0, 0.95, 0.0));
    let camera = PinholeCamera::new(
        camera_eye,
        camera_target,
        Vec3::Y,
        38.0_f32.to_radians(),
        1.0,
    );

    Ok((scene, camera))
}
