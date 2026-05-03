// Side-by-side comparison of SimplePBR (scene 20 setup) and Disney BRDF
// using the same dragon model, identical baseColor / metallic / roughness /
// normal textures, and the same lighting environment. The Disney dragon
// adds a light clearcoat layer to make the difference legible.
//
// Left dragon (negative X) -- SimplePBR.
// Right dragon (positive X) -- Disney BRDF.

use glam::{Mat4, Vec3};
use std::{error::Error, path::Path};

use crate::{
    camera::PinholeCamera,
    light::EnvironmentLight,
    material::{DisneyBrdfMaterial, Material, NormalizedLambertMaterial, SimplePbrMaterial},
    mesh::{load_gltf, load_obj},
    scene::Scene,
};

use super::{game_rotation_degrees, uniform_scale_for_height};

pub fn create_scene_26() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();

    let floor_material = scene.add_material(Material::NormalizedLambert(
        NormalizedLambertMaterial::new(Vec3::splat(0.9)),
    ));
    let floor_mesh = scene.add_mesh(load_gltf(Path::new("assets/gltf/floor.glb"))?);
    let floor_transform = Mat4::from_scale(Vec3::new(20.0, 1.0, 20.0));
    scene.add_instance(floor_mesh, floor_material, floor_transform);

    let simple_pbr_material = scene.add_material(Material::SimplePBR(
        SimplePbrMaterial::try_new_with_texture_paths(
            Vec3::ONE,
            1.0,
            1.0,
            1.5,
            0.0,
            Some(Path::new("assets/gltf/dragon-BaseColor.png")),
            Some(Path::new("assets/gltf/dragon-Metallic.png")),
            Some(Path::new("assets/gltf/dragon-Roughness.png")),
            Some(Path::new("assets/gltf/dragon-Normal.png")),
        )?,
    ));

    let disney_material = scene.add_material(Material::DisneyBrdf(
        DisneyBrdfMaterial::try_new_with_texture_paths(
            Vec3::ONE,
            1.0,
            1.0,
            Some(Path::new("assets/gltf/dragon-BaseColor.png")),
            Some(Path::new("assets/gltf/dragon-Metallic.png")),
            Some(Path::new("assets/gltf/dragon-Roughness.png")),
            Some(Path::new("assets/gltf/dragon-Normal.png")),
        )?
        .with_specular(0.5)
        .with_clearcoat(0.4)
        .with_clearcoat_gloss(0.9),
    ));

    let dragon = load_obj(Path::new("assets/gltf/dragon.obj"))?;
    let dragon_scale = uniform_scale_for_height(&dragon, 1.8);
    let dragon_pivot = Vec3::new(
        dragon.bounds.center().x,
        dragon.bounds.min.y,
        dragon.bounds.center().z,
    );
    let dragon_mesh = scene.add_mesh(dragon);

    // Both dragons face the camera (rotate to head toward +Z).
    let face_camera = Mat4::from_quat(game_rotation_degrees(0.0, 90.0, 0.0));
    let dragon_local = face_camera
        * Mat4::from_scale(Vec3::splat(dragon_scale))
        * Mat4::from_translation(-dragon_pivot);

    let half_offset = 1.6_f32;
    let simple_pbr_transform =
        Mat4::from_translation(Vec3::new(-half_offset, 0.0, 0.0)) * dragon_local;
    scene.add_instance(dragon_mesh, simple_pbr_material, simple_pbr_transform);

    let disney_transform = Mat4::from_translation(Vec3::new(half_offset, 0.0, 0.0)) * dragon_local;
    scene.add_instance(dragon_mesh, disney_material, disney_transform);

    let env = EnvironmentLight::from_hdr_file(
        "assets/sky/kloofendal_48d_partly_cloudy_puresky_4k.hdr",
        0.5,
        0.0,
    )?;
    scene.set_environment_light(env);

    // Front camera: a touch elevated so the back of each dragon is visible.
    let camera_eye = Vec3::new(0.0, 1.6, 6.2);
    let camera_target = Vec3::new(0.0, 0.7, 0.0);
    let camera = PinholeCamera::new(
        camera_eye,
        camera_target,
        Vec3::Y,
        42.0_f32.to_radians(),
        1.0,
    );

    Ok((scene, camera))
}
