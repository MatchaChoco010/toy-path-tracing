//! HDRI 環境光のもと、Disney BRDF 球を 11 列 x 10 段で並べ、各段でパラメータをスイープする。

use glam::{Mat4, Vec3};
use std::{error::Error, path::Path};

use crate::{
    camera::PinholeCamera,
    light::EnvironmentLight,
    material::{DisneyBrdfMaterial, Material},
    mesh::load_gltf,
    scene::Scene,
};

pub fn create_scene_25() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();

    let sphere = load_gltf(Path::new("assets/models/sphere.glb"))?;
    let sphere_extent = sphere.bounds.extent();
    let sphere_target_diameter = 0.6_f32;
    let sphere_scale = sphere_target_diameter / sphere_extent.y.max(1.0e-3);
    let sphere_center = sphere.bounds.center();
    let sphere_mesh = scene.add_mesh(sphere);

    // 11 columns x 10 rows. Spacing keeps spheres well separated.
    let column_count = 11;
    let row_count = 10;
    let spacing = 0.72_f32;
    let column_origin = -((column_count - 1) as f32) * spacing * 0.5;
    let row_origin = -((row_count - 1) as f32) * spacing * 0.5;

    // Each row uses a row-specific baseline matched to Figure 16 (e.g. the
    // metallic row starts gold, the sheen row dark red, etc.).
    let rows: [(&str, RowMaker); 10] = [
        ("subsurface", make_subsurface_row),
        ("metallic", make_metallic_row),
        ("specular", make_specular_row),
        ("specularTint", make_specular_tint_row),
        ("roughness", make_roughness_row),
        ("anisotropic", make_anisotropic_row),
        ("sheen", make_sheen_row),
        ("sheenTint", make_sheen_tint_row),
        ("clearcoat", make_clearcoat_row),
        ("clearcoatGloss", make_clearcoat_gloss_row),
    ];

    for (row_idx, (_, builder)) in rows.iter().enumerate() {
        let y = row_origin + spacing * (row_count - 1 - row_idx) as f32;
        for column_idx in 0..column_count {
            let t = column_idx as f32 / (column_count - 1) as f32;
            let x = column_origin + spacing * column_idx as f32;
            let material = builder(t);
            let material_id = scene.add_material(Material::DisneyBrdf(material));
            let transform = Mat4::from_translation(Vec3::new(x, y, 0.0))
                * Mat4::from_scale(Vec3::splat(sphere_scale))
                * Mat4::from_translation(-sphere_center);
            scene.add_instance(sphere_mesh, material_id, transform);
        }
    }

    let env = EnvironmentLight::from_hdr_file(
        "assets/sky/studio_small_08_4k.hdr",
        0.6,
        std::f32::consts::PI * 0.5,
    )?;
    scene.set_environment_light(env);

    let camera_eye = Vec3::new(0.0, 0.0, 22.0);
    let camera_target = Vec3::ZERO;
    let camera = PinholeCamera::new(
        camera_eye,
        camera_target,
        Vec3::Y,
        22.5_f32.to_radians(),
        1.0,
    );

    Ok((scene, camera))
}

type RowMaker = fn(f32) -> DisneyBrdfMaterial;

fn make_subsurface_row(t: f32) -> DisneyBrdfMaterial {
    DisneyBrdfMaterial::new(Vec3::new(1.0, 1.0, 1.0))
        .with_subsurface(t)
        .with_roughness(0.7)
}

fn make_metallic_row(t: f32) -> DisneyBrdfMaterial {
    DisneyBrdfMaterial::new(Vec3::new(1.0, 0.8, 0.1))
        .with_metallic(t)
        .with_roughness(0.0)
}

fn make_specular_row(t: f32) -> DisneyBrdfMaterial {
    DisneyBrdfMaterial::new(Vec3::new(1.0, 0.0, 0.0))
        .with_specular(t)
        .with_roughness(0.2)
}

fn make_specular_tint_row(t: f32) -> DisneyBrdfMaterial {
    DisneyBrdfMaterial::new(Vec3::new(1.0, 0.0, 0.0))
        .with_specular_tint(t)
        .with_roughness(0.2)
}

fn make_roughness_row(t: f32) -> DisneyBrdfMaterial {
    DisneyBrdfMaterial::new(Vec3::new(0.3, 0.3, 0.7)).with_roughness(t)
}

fn make_anisotropic_row(t: f32) -> DisneyBrdfMaterial {
    DisneyBrdfMaterial::new(Vec3::new(0.5, 0.0, 0.25))
        .with_roughness(0.2)
        .with_anisotropic(t)
}

fn make_sheen_row(t: f32) -> DisneyBrdfMaterial {
    DisneyBrdfMaterial::new(Vec3::new(0.5, 0.15, 0.05))
        .with_sheen(t)
        .with_sheen_tint(0.0)
        .with_specular(0.0)
}

fn make_sheen_tint_row(t: f32) -> DisneyBrdfMaterial {
    DisneyBrdfMaterial::new(Vec3::new(0.5, 0.15, 0.05))
        .with_sheen(1.0)
        .with_sheen_tint(t)
        .with_specular(0.0)
}

fn make_clearcoat_row(t: f32) -> DisneyBrdfMaterial {
    DisneyBrdfMaterial::new(Vec3::new(0.0, 0.2, 0.3))
        .with_roughness(0.8)
        .with_clearcoat(t)
        .with_clearcoat_gloss(1.0)
}

fn make_clearcoat_gloss_row(t: f32) -> DisneyBrdfMaterial {
    DisneyBrdfMaterial::new(Vec3::new(0.0, 0.2, 0.3))
        .with_roughness(0.8)
        .with_clearcoat(1.0)
        .with_clearcoat_gloss(t)
}
