use std::{error::Error, path::Path};

use glam::{Mat4, Vec3};

use crate::{
    camera::PinholeCamera,
    light::EnvironmentLight,
    material::{EmissiveMaterial, Material, StandardSurfaceMaterial},
    mesh::load_obj,
    scene::Scene,
};

pub fn create_scene_30() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();

    let floor = load_obj(Path::new("assets/mori-knob/floor.obj"))?;
    let base = load_obj(Path::new("assets/mori-knob/base.obj"))?;
    let knob = load_obj(Path::new("assets/mori-knob/knob.obj"))?;
    let light = load_obj(Path::new("assets/mori-knob/light.obj"))?;

    let floor_mesh = scene.add_mesh(floor);
    let base_mesh = scene.add_mesh(base);
    let knob_mesh = scene.add_mesh(knob);
    let light_mesh = scene.add_mesh(light);

    let floor_material = scene.add_material(Material::StandardSurface(
        StandardSurfaceMaterial::new(Vec3::splat(0.72))
            .with_specular_roughness(0.6)
            .with_diffuse_roughness(0.3),
    ));
    let base_material = scene.add_material(Material::StandardSurface(
        StandardSurfaceMaterial::new(Vec3::splat(0.18))
            .with_specular(0.1)
            .with_specular_roughness(0.5),
    ));

    let knob_makers: [fn() -> StandardSurfaceMaterial; 9] = [
        knob_polished_gold,
        knob_iridescent_metal,
        knob_brushed_copper,
        knob_non_dispersive_glass,
        knob_smooth_dispersive_glass,
        knob_rough_dispersive_glass,
        knob_red_velvet_sheen,
        knob_coated_plastic,
        knob_matte_ceramic,
    ];

    let cell_scale = 0.55_f32;
    let spacing = 0.78_f32;

    let world_root = Mat4::from_scale(Vec3::splat(cell_scale));
    let light_material =
        scene.add_material(Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 40.0)));
    scene.add_instance(floor_mesh, floor_material, world_root);
    scene.add_instance(light_mesh, light_material, world_root);

    for row in 0..3 {
        for col in 0..3 {
            let index = row * 3 + col;
            let offset_x = (col as f32 - 1.0) * spacing;
            let offset_z = (row as f32 - 1.0) * spacing;
            let cell = Mat4::from_translation(Vec3::new(offset_x, 0.0, offset_z))
                * Mat4::from_scale(Vec3::splat(cell_scale));
            scene.add_instance(base_mesh, base_material, cell);
            let knob_material = scene.add_material(Material::StandardSurface(knob_makers[index]()));
            scene.add_instance(knob_mesh, knob_material, cell);
        }
    }

    let env = EnvironmentLight::from_hdr_file(
        "assets/sky/studio_small_08_4k.hdr",
        2.0,
        std::f32::consts::PI * 0.5,
    )?;
    scene.set_environment_light(env);

    let camera_eye = Vec3::new(0.8, 2.0, -2.9);
    let camera_target = Vec3::new(-0.05, -0.2, -0.05);
    let camera = PinholeCamera::new(
        camera_eye,
        camera_target,
        Vec3::Y,
        42.0_f32.to_radians(),
        1.0,
    );

    Ok((scene, camera))
}

fn knob_polished_gold() -> StandardSurfaceMaterial {
    StandardSurfaceMaterial::new(Vec3::new(0.95, 0.78, 0.42))
        .with_metalness(1.0)
        .with_specular_color(Vec3::new(0.99, 0.95, 0.85))
        .with_specular_roughness(0.08)
}

fn knob_iridescent_metal() -> StandardSurfaceMaterial {
    StandardSurfaceMaterial::new(Vec3::new(0.85, 0.85, 0.88))
        .with_metalness(1.0)
        .with_specular_color(Vec3::ONE)
        .with_specular_roughness(0.15)
        .with_thin_film_thickness(340.0)
        .with_thin_film_ior(1.35)
}

fn knob_brushed_copper() -> StandardSurfaceMaterial {
    StandardSurfaceMaterial::new(Vec3::new(0.92, 0.55, 0.35))
        .with_metalness(1.0)
        .with_specular_color(Vec3::new(0.98, 0.85, 0.7))
        .with_specular_roughness(0.35)
        .with_specular_anisotropy(0.85)
}

fn knob_non_dispersive_glass() -> StandardSurfaceMaterial {
    StandardSurfaceMaterial::new(Vec3::new(0.05, 0.06, 0.08))
        .with_specular_roughness(0.05)
        .with_specular_ior(1.55)
        .with_transmission(1.0)
        .with_transmission_color(Vec3::new(0.7, 0.85, 0.98))
        .with_base(0.0)
}

fn knob_smooth_dispersive_glass() -> StandardSurfaceMaterial {
    StandardSurfaceMaterial::new(Vec3::ONE)
        .with_specular_roughness(0.0)
        .with_specular_ior(1.55)
        .with_transmission(1.0)
        .with_transmission_color(Vec3::ONE)
        .with_transmission_dispersion(20.0)
        .with_base(0.0)
}

fn knob_rough_dispersive_glass() -> StandardSurfaceMaterial {
    StandardSurfaceMaterial::new(Vec3::ONE)
        .with_specular_roughness(0.5)
        .with_specular_ior(1.55)
        .with_transmission(1.0)
        .with_transmission_color(Vec3::ONE)
        .with_transmission_dispersion(20.0)
        .with_base(0.0)
}

fn knob_red_velvet_sheen() -> StandardSurfaceMaterial {
    StandardSurfaceMaterial::new(Vec3::new(0.55, 0.05, 0.08))
        .with_specular(0.0)
        .with_diffuse_roughness(0.6)
        .with_sheen(1.0)
        .with_sheen_color(Vec3::new(0.95, 0.45, 0.45))
        .with_sheen_roughness(0.25)
}

fn knob_coated_plastic() -> StandardSurfaceMaterial {
    StandardSurfaceMaterial::new(Vec3::new(0.05, 0.35, 0.55))
        .with_specular(0.5)
        .with_specular_roughness(0.4)
        .with_coat(1.0)
        .with_coat_color(Vec3::ONE)
        .with_coat_roughness(0.05)
        .with_coat_ior(1.5)
        .with_coat_affect_color(0.6)
}

fn knob_matte_ceramic() -> StandardSurfaceMaterial {
    StandardSurfaceMaterial::new(Vec3::new(0.85, 0.78, 0.65))
        .with_specular(0.15)
        .with_specular_roughness(0.7)
        .with_diffuse_roughness(0.85)
}
