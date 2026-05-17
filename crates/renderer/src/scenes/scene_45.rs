//! mori-knob を 4 x 4 グリッドに配置し、各ノブに別々の MaterialX マテリアルを割り当てる。

use std::{error::Error, path::Path};

use glam::{Mat4, Vec3};

use crate::{
    light::EnvironmentLight,
    material::{EmissiveMaterial, Material, NormalizedLambertMaterial},
    scene::PinholeCamera,
    scene::Scene,
    scene::load_obj,
    scene::mtlx_loader::{load_mtlx_material, load_standard_library},
};

pub fn create_scene_45(
    _ocio: &crate::color::OcioColorPipeline,
) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();

    let library_root = Path::new("lib/materialx/libraries");
    let library = load_standard_library(library_root)?;

    let floor = load_obj(Path::new("assets/mori-knob/floor.obj"))?;
    let base = load_obj(Path::new("assets/mori-knob/base.obj"))?;
    let knob = load_obj(Path::new("assets/mori-knob/knob.obj"))?;
    let light = load_obj(Path::new("assets/mori-knob/light.obj"))?;

    let floor_mesh = scene.add_mesh(floor);
    let base_mesh = scene.add_mesh(base);
    let knob_mesh = scene.add_mesh(knob);
    let light_mesh = scene.add_mesh(light);

    let floor_material = scene.add_material(Material::NormalizedLambert(
        NormalizedLambertMaterial::new(Vec3::splat(0.42)),
    ));
    let base_material = scene.add_material(Material::NormalizedLambert(
        NormalizedLambertMaterial::new(Vec3::splat(0.18)),
    ));
    let acryl_base_material =
        scene.add_material(Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 25.0)));

    let entries: [(&str, &str); 16] = [
        (
            "assets/mtlx/Argentinian_Layered_Onyx_4k_8b_hSJXwd2/Argentinian_Layered_Onyx.mtlx",
            "Argentinian_Layered_Onyx",
        ),
        ("assets/mtlx/Car_Paint_4k_8b/Car_Paint.mtlx", "Car_Paint"),
        (
            "assets/mtlx/Copper_Satin_4k_8b/Copper_Satin.mtlx",
            "Copper_Satin",
        ),
        (
            "assets/mtlx/Emerald_Peaks_Wallpaper_4k_8b_nJvUmTK/Emerald_Peaks_Wallpaper.mtlx",
            "Emerald_Peaks_Wallpaper",
        ),
        (
            "assets/mtlx/Acryl_Plastic_1k_8b_kylYFM6/Acryl_Plastic.mtlx",
            "Acryl_Plastic",
        ),
        (
            "assets/mtlx/Anodized_Titanium_4k_8b/Anodized_Titanium.mtlx",
            "Anodized_Titanium",
        ),
        (
            "assets/mtlx/Sky_Velvet_Linen_Fabric_4k_8b_26oHIiJ/Sky_Velvet_Linen_Fabric.mtlx",
            "Sky_Velvet_Linen_Fabric",
        ),
        (
            "assets/mtlx/TH_Broken_Cobblestone_Floor_4k_8b_PtAUnYc/TH_Broken_Cobblestone_Floor.mtlx",
            "TH_Broken_Cobblestone_Floor",
        ),
        (
            "assets/mtlx/TH_Wood_Table_4k_8b/TH_Wood_Table.mtlx",
            "TH_Wood_Table",
        ),
        (
            "assets/mtlx/Verdi_Almi_Marble_4k_8b_cEBpse8/Verdi_Almi_Marble.mtlx",
            "Verdi_Almi_Marble",
        ),
        (
            "assets/mtlx/Bronze_Oxydized_4k_8b_F9NVlqV/Bronze_Oxydized.mtlx",
            "Bronze_Oxydized",
        ),
        ("assets/mtlx/shader_ops.mtlx", "material_checker_opacity"),
        (
            "assets/mtlx/standard_surface_brick_procedural/standard_surface_brick_procedural.mtlx",
            "M_BrickPattern",
        ),
        ("assets/mtlx/Diamond_1k_8b/Diamond.mtlx", "Diamond"),
        (
            "assets/mtlx/standard_surface_onyx_hextiled/standard_surface_onyx_hextiled.mtlx",
            "M_OnyxHextiled",
        ),
        ("assets/mtlx/standard_surface_velvet.mtlx", "Velvet"),
    ];

    let cell_scale = 0.55_f32;
    let spacing = 0.78_f32;
    let world_root = Mat4::from_scale(Vec3::splat(cell_scale));
    let light_material =
        scene.add_material(Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 20.0)));
    scene.add_instance(floor_mesh, floor_material, world_root);
    scene.add_instance(light_mesh, light_material, world_root);

    let acryl_index = 4;
    for row in 0..4 {
        for col in 0..4 {
            let index = row * 4 + col;
            let (path, material_name) = entries[index];
            let offset_x = (col as f32 - 1.5) * spacing;
            let offset_z = (row as f32 - 1.5) * spacing;
            let cell = Mat4::from_translation(Vec3::new(offset_x, 0.0, offset_z))
                * Mat4::from_scale(Vec3::splat(cell_scale));
            let cell_base_material = if index == acryl_index {
                acryl_base_material
            } else {
                base_material
            };
            scene.add_instance(base_mesh, cell_base_material, cell);
            let mtlx_material =
                load_mtlx_material(&library, Path::new(path), material_name, _ocio)?;
            let knob_material = scene.add_material(Material::Mtlx(mtlx_material));
            scene.add_instance(knob_mesh, knob_material, cell);
        }
    }

    let env = EnvironmentLight::from_hdr_file(
        "assets/sky/brown_photostudio_02_4k.hdr",
        2.5,
        std::f32::consts::PI * -0.5,
    )?;
    scene.set_environment_light(env);

    let camera_eye = Vec3::new(0.0, 6.5, -5.5);
    let camera_target = Vec3::new(0.0, -0.2, 0.0);
    let camera = PinholeCamera::new(
        camera_eye,
        camera_target,
        Vec3::Y,
        22.5_f32.to_radians(),
        1.0,
    );

    Ok((scene, camera))
}
