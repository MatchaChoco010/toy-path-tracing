use std::{error::Error, path::Path};

use glam::{Mat4, Vec3};

use crate::{
    camera::PinholeCamera,
    light::EnvironmentLight,
    material::{Material, OpenPbrMaterial},
    mesh::load_obj,
    scene::Scene,
};

pub(super) fn create_openpbr_mori_knob_scene(
    knob_material: OpenPbrMaterial,
) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();

    let floor = scene.add_mesh(load_obj(Path::new("assets/mori-knob/floor.obj"))?);
    let base = scene.add_mesh(load_obj(Path::new("assets/mori-knob/base.obj"))?);
    let knob = scene.add_mesh(load_obj(Path::new("assets/mori-knob/knob.obj"))?);
    let light = scene.add_mesh(load_obj(Path::new("assets/mori-knob/light.obj"))?);

    let floor_material = scene.add_material(Material::OpenPbr(
        OpenPbrMaterial::new(Vec3::splat(0.018))
            .with_specular_roughness(0.85)
            .with_base_diffuse_roughness(0.9),
    ));
    let base_material = scene.add_material(Material::OpenPbr(
        OpenPbrMaterial::new(Vec3::splat(0.018))
            .with_specular_roughness(0.78)
            .with_base_diffuse_roughness(0.92),
    ));
    let light_material = scene.add_material(Material::OpenPbr(
        OpenPbrMaterial::new(Vec3::ZERO)
            .with_base_weight(0.0)
            .with_specular_weight(0.0)
            .with_emission_color(Vec3::ONE)
            .with_emission_luminance(15.0),
    ));
    let knob_material = scene.add_material(Material::OpenPbr(knob_material));

    let room = Mat4::from_scale(Vec3::splat(0.85));
    scene.add_instance(floor, floor_material, room);
    scene.add_instance(base, base_material, room);
    scene.add_instance(knob, knob_material, room);
    scene.add_instance(light, light_material, room);

    let env = EnvironmentLight::from_hdr_file(
        "assets/sky/brown_photostudio_02_4k.hdr",
        1.2,
        std::f32::consts::PI * -0.45,
    )?;
    scene.set_environment_light(env);

    let camera = PinholeCamera::new(
        Vec3::new(0.0, 0.7, -1.6),
        Vec3::new(0.0, -0.05, 0.04),
        Vec3::Y,
        40.0_f32.to_radians(),
        1.0,
    );

    Ok((scene, camera))
}
