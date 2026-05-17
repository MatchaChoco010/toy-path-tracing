//! Lambert 床と puresky 環境光のもと、テクスチャ付き Lambert バニーと Conductor GGX 球を配置する。

use glam::{Mat4, Vec3};
use std::{error::Error, path::Path};

use crate::{
    camera::PinholeCamera,
    light::EnvironmentLight,
    material::{ConductorGgxMaterial, Material, NormalizedLambertMaterial},
    mesh::load_gltf,
    scene::Scene,
};

use super::{game_rotation_degrees, uniform_scale_for_height};

pub fn create_scene_16() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();

    let floor_material = scene.add_material(Material::NormalizedLambert(
        NormalizedLambertMaterial::new(Vec3::splat(0.62)),
    ));
    let floor_mesh = scene.add_mesh(load_gltf(Path::new("assets/models/floor.glb"))?);
    let floor_transform = Mat4::from_scale(Vec3::new(12.0, 1.0, 12.0));
    scene.add_instance(floor_mesh, floor_material, floor_transform);

    let bunny_material = scene.add_material(Material::NormalizedLambert(
        NormalizedLambertMaterial::try_new_with_texture_path(
            Vec3::ONE,
            Some(Path::new("assets/models/bunny-color.png")),
            None,
        )?,
    ));
    let bunny = load_gltf(Path::new("assets/models/bunny.glb"))?;
    let bunny_scale = uniform_scale_for_height(&bunny, 2.25);
    let bunny_pivot = Vec3::new(
        bunny.bounds.center().x,
        bunny.bounds.min.y,
        bunny.bounds.center().z,
    );
    let bunny_mesh = scene.add_mesh(bunny);
    let bunny_transform = Mat4::from_translation(Vec3::new(0.85, 0.0, 0.20))
        * Mat4::from_quat(game_rotation_degrees(0.0, 200.0, 0.0))
        * Mat4::from_scale(Vec3::splat(bunny_scale))
        * Mat4::from_translation(-bunny_pivot);
    scene.add_instance(bunny_mesh, bunny_material, bunny_transform);

    let sphere_material = scene.add_material(Material::ConductorGgx(
        ConductorGgxMaterial::try_new_with_texture_paths(
            Vec3::ONE,
            1.0,
            0.0,
            Some(Path::new("assets/models/sphere-color.png")),
            Some(Path::new("assets/models/sphere-roughness.png")),
            None,
        )?,
    ));
    let sphere = load_gltf(Path::new("assets/models/sphere.glb"))?;
    let sphere_scale = uniform_scale_for_height(&sphere, 1.05);
    let sphere_pivot = Vec3::new(
        sphere.bounds.center().x,
        sphere.bounds.min.y,
        sphere.bounds.center().z,
    );
    let sphere_mesh = scene.add_mesh(sphere);
    let sphere_transform = Mat4::from_translation(Vec3::new(-1.15, 0.0, 0.85))
        * Mat4::from_scale(Vec3::splat(sphere_scale))
        * Mat4::from_translation(-sphere_pivot);
    scene.add_instance(sphere_mesh, sphere_material, sphere_transform);

    let env = EnvironmentLight::from_hdr_file(
        "assets/sky/kloofendal_48d_partly_cloudy_puresky_4k.hdr",
        0.5,
        0.0,
    )?;
    scene.set_environment_light(env);

    let camera = PinholeCamera::new(
        Vec3::new(0.0, 2.0, -7.2),
        Vec3::new(-0.05, 1.05, 0.45),
        Vec3::Y,
        38.0_f32.to_radians(),
        1.0,
    );

    Ok((scene, camera))
}
