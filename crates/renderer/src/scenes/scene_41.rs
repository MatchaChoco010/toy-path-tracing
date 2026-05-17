//! Scene 40 と同じ配置で、Glavenus STL モデルのマテリアルを EON に差し替える。

use std::{error::Error, path::Path};

use glam::{Mat4, Vec3};

use crate::{
    light::EnvironmentLight,
    material::{EmissiveMaterial, EonMaterial, Material, NormalizedLambertMaterial},
    scene::PinholeCamera,
    scene::Scene,
    scene::{Bounds, load_obj, load_stl},
};

pub fn create_scene_41(
    _ocio: &crate::color::OcioColorPipeline,
) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();

    let floor_mesh = scene.add_mesh(load_obj(Path::new("assets/mori-knob/floor.obj"))?);
    let light_mesh = scene.add_mesh(load_obj(Path::new("assets/mori-knob/light.obj"))?);

    let floor_material = scene.add_material(Material::NormalizedLambert(
        NormalizedLambertMaterial::new(Vec3::splat(0.62)),
    ));
    let light_material =
        scene.add_material(Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 8.0)));

    let world_root = Mat4::from_scale(Vec3::splat(0.55));
    scene.add_instance(floor_mesh, floor_material, world_root);
    scene.add_instance(light_mesh, light_material, world_root);

    let glavenus_part_paths: [&str; 9] = [
        "assets/glavenus/base.stl",
        "assets/glavenus/body.stl",
        "assets/glavenus/body-horn-l.stl",
        "assets/glavenus/body-horn-r.stl",
        "assets/glavenus/head.stl",
        "assets/glavenus/leg-l.stl",
        "assets/glavenus/leg-r.stl",
        "assets/glavenus/tail-1.stl",
        "assets/glavenus/tail-2.stl",
    ];
    let glavenus_parts: Vec<_> = glavenus_part_paths
        .iter()
        .map(|path| load_stl(Path::new(path)))
        .collect::<Result<_, _>>()?;

    let mut combined = Bounds::EMPTY;
    for mesh in &glavenus_parts {
        combined = combined.union(mesh.bounds);
    }

    let model_height = 1.5_f32;
    let model_scale = model_height / combined.extent().y.max(1.0e-3);
    let pivot = Vec3::new(
        (combined.min.x + combined.max.x) * 0.5,
        combined.min.y,
        (combined.min.z + combined.max.z) * 0.5,
    );
    let glavenus_transform = world_root
        * Mat4::from_translation(Vec3::new(0.0, -0.5, 0.0))
        * Mat4::from_scale(Vec3::splat(model_scale))
        * Mat4::from_rotation_y(std::f32::consts::PI * 0.62)
        * Mat4::from_translation(-pivot);

    let base_material = scene.add_material(Material::Eon(EonMaterial::new(Vec3::splat(0.95), 1.0)));
    let body_material = scene.add_material(Material::Eon(EonMaterial::new(
        Vec3::new(0.95, 0.62, 0.45),
        1.0,
    )));

    for (part_index, mesh) in glavenus_parts.into_iter().enumerate() {
        let mesh_index = scene.add_mesh(mesh);
        let material = if part_index == 0 {
            base_material
        } else {
            body_material
        };
        scene.add_instance(mesh_index, material, glavenus_transform);
    }

    let env = EnvironmentLight::from_hdr_file("assets/sky/brown_photostudio_02_4k.hdr", 1.0, 0.0)?;
    scene.set_environment_light(env);

    let camera = PinholeCamera::new(
        Vec3::new(0.0, 0.15, -2.2),
        Vec3::new(0.0, 0.1, 0.0),
        Vec3::Y,
        30.0_f32.to_radians(),
        1.0,
    );

    Ok((scene, camera))
}
