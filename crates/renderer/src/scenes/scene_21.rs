//! Lambert 床と puresky 環境光のもと、SimplePBR ドラゴン、金色 Conductor GGX バニー、Glass 球、NormalizedLambert バニーを並べる。

use glam::{Mat4, Vec3};
use std::{error::Error, path::Path};

use crate::{
    light::EnvironmentLight,
    material::{
        ConductorGgxMaterial, GlassMaterial, Material, NormalizedLambertMaterial, SimplePbrMaterial,
    },
    scene::PinholeCamera,
    scene::Scene,
    scene::{load_gltf, load_obj},
};

use super::{game_rotation_degrees, uniform_scale_for_height};

pub fn create_scene_21(
    _ocio: &crate::color::OcioColorPipeline,
) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();

    let floor_material = scene.add_material(Material::NormalizedLambert(
        NormalizedLambertMaterial::new(Vec3::splat(0.9)),
    ));
    let floor_mesh = scene.add_mesh(load_gltf(Path::new("assets/models/floor.glb"))?);
    let floor_transform = Mat4::from_scale(Vec3::new(18.0, 1.0, 18.0));
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
            _ocio,
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
    let dragon_transform = Mat4::from_quat(game_rotation_degrees(0.0, -35.0, 0.0))
        * Mat4::from_scale(Vec3::splat(dragon_scale))
        * Mat4::from_translation(-dragon_pivot);
    scene.add_instance(dragon_mesh, dragon_material, dragon_transform);

    let gold_metal_material = scene.add_material(Material::ConductorGgx(
        ConductorGgxMaterial::new(Vec3::new(1.0, 0.78, 0.34), 0.4, 0.0),
    ));
    let lambert_bunny_material = scene.add_material(Material::NormalizedLambert(
        NormalizedLambertMaterial::new(Vec3::new(0.62, 0.7, 0.85)),
    ));
    let glass_material = scene.add_material(Material::Glass(GlassMaterial::new(
        1.5,
        Vec3::splat(0.97),
        false,
    )));

    let bunny = load_gltf(Path::new("assets/models/bunny.glb"))?;
    let bunny_scale = uniform_scale_for_height(&bunny, 1.25);
    let bunny_pivot = Vec3::new(
        bunny.bounds.center().x,
        bunny.bounds.min.y,
        bunny.bounds.center().z,
    );
    let bunny_mesh = scene.add_mesh(bunny);

    let metal_bunny_transform = Mat4::from_translation(Vec3::new(-1.7, 0.0, -1.7))
        * Mat4::from_quat(game_rotation_degrees(0.0, -55.0, 0.0))
        * Mat4::from_scale(Vec3::splat(bunny_scale))
        * Mat4::from_translation(-bunny_pivot);
    scene.add_instance(bunny_mesh, gold_metal_material, metal_bunny_transform);

    let lambert_bunny_transform = Mat4::from_translation(Vec3::new(-2.6, 0.0, 0.5))
        * Mat4::from_quat(game_rotation_degrees(0.0, 35.0, 0.0))
        * Mat4::from_scale(Vec3::splat(bunny_scale))
        * Mat4::from_translation(-bunny_pivot);
    scene.add_instance(bunny_mesh, lambert_bunny_material, lambert_bunny_transform);

    let sphere = load_gltf(Path::new("assets/models/sphere.glb"))?;
    let sphere_scale = uniform_scale_for_height(&sphere, 1.25);
    let sphere_pivot = Vec3::new(
        sphere.bounds.center().x,
        sphere.bounds.min.y,
        sphere.bounds.center().z,
    );
    let sphere_mesh = scene.add_mesh(sphere);

    let glass_sphere_transform = Mat4::from_translation(Vec3::new(2.2, 0.0, -2.1))
        * Mat4::from_scale(Vec3::splat(sphere_scale))
        * Mat4::from_translation(-sphere_pivot);
    scene.add_instance(sphere_mesh, glass_material, glass_sphere_transform);

    let env = EnvironmentLight::from_hdr_file(
        "assets/sky/kloofendal_48d_partly_cloudy_puresky_4k.hdr",
        0.5,
        1.2,
    )?;
    scene.set_environment_light(env);

    let camera = PinholeCamera::new(
        Vec3::new(-3.9, 2.7, -6.2),
        Vec3::new(0.0, 0.95, 0.0),
        Vec3::Y,
        50.0_f32.to_radians(),
        1.0,
    );

    Ok((scene, camera))
}
