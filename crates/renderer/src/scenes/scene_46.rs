//! mori-knob の床とライト、brown_photostudio HDRI のもと、OpenPBR の多様なマテリアルを 4 x 4 のバニーで比較する。

use std::{error::Error, path::Path, sync::Arc};

use glam::{Mat4, Vec3};

use crate::{
    camera::PinholeCamera,
    light::EnvironmentLight,
    material::{Material, NormalMap, OpenPbrMaterial, ScalarTexture, Texture},
    mesh::{load_gltf, load_obj},
    scene::{MaterialIndex, MeshIndex, Scene},
};

use super::uniform_scale_for_height;

pub fn create_scene_46() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();

    let floor = load_obj(Path::new("assets/mori-knob/floor.obj"))?;
    let light = load_obj(Path::new("assets/mori-knob/light.obj"))?;
    let bunny = load_gltf(Path::new("assets/models/bunny.glb"))?;

    let bunny_height = 0.48_f32;
    let bunny_scale = uniform_scale_for_height(&bunny, bunny_height);
    let bunny_ground_pivot = Vec3::new(
        bunny.bounds.center().x,
        bunny.bounds.min.y,
        bunny.bounds.center().z,
    );

    let floor_mesh = scene.add_mesh(floor);
    let light_mesh = scene.add_mesh(light);
    let bunny_mesh = scene.add_mesh(bunny);

    let floor_material = scene.add_material(Material::OpenPbr(
        OpenPbrMaterial::new(Vec3::splat(0.018))
            .with_specular_roughness(0.85)
            .with_base_diffuse_roughness(0.85),
    ));
    let light_material = scene.add_material(Material::OpenPbr(
        OpenPbrMaterial::new(Vec3::ZERO)
            .with_base_weight(0.0)
            .with_specular_weight(0.0)
            .with_emission_color(Vec3::ONE)
            .with_emission_luminance(35.0),
    ));

    let room_scale = 0.55_f32;
    let room = Mat4::from_scale(Vec3::splat(room_scale));
    scene.add_instance(floor_mesh, floor_material, room);
    scene.add_instance(light_mesh, light_material, room);

    let wood_texture = Arc::new(Texture::from_srgb_file(
        "assets/models/bunny_wood_texture_BaseColor.png",
    )?);
    let wood_roughness = Arc::new(ScalarTexture::from_file(
        "assets/models/bunny_wood_texture_Roughness.png",
    )?);
    let wood_normal = NormalMap::from_file("assets/models/bunny_wood_texture_Normal.png")?;
    let titanium_thickness = Arc::new(ScalarTexture::from_file(
        "assets/models/bunny_perlin_noise.png",
    )?);
    let soap_thickness = Arc::new(ScalarTexture::from_file(
        "assets/models/bunny_soap_thin_film_thickness.png",
    )?);

    let materials = [
        glass_low_roughness(),
        glass_high_roughness(),
        clear_glass_no_dispersion(),
        clear_glass_strong_dispersion(),
        gold_metal(),
        green_tinted_copper(),
        titanium_with_noisy_thin_film(titanium_thickness),
        soap_thin_walled(soap_thickness),
        wood_no_coat(
            wood_texture.clone(),
            wood_roughness.clone(),
            wood_normal.clone(),
        ),
        wood_coat_darkening(wood_texture, wood_roughness, wood_normal),
        purple_coated_metal(false),
        purple_coated_metal(true),
        fuzz_metal(1.0 / 3.0),
        fuzz_metal(1.0),
        rough_diffuse(),
        velvet_diffuse(),
    ];

    let spacing_x = 0.88_f32;
    let spacing_z = 0.82_f32;
    for row in 0..4 {
        for col in 0..4 {
            let index = row * 4 + col;
            let x = (col as f32 - 1.5) * spacing_x;
            let z = (row as f32 - 1.5) * spacing_z;
            let material = scene.add_material(Material::OpenPbr(materials[index].clone()));
            add_bunny(
                &mut scene,
                bunny_mesh,
                material,
                Vec3::new(x, -0.26, z),
                bunny_scale,
                bunny_ground_pivot,
            );
        }
    }

    let env = EnvironmentLight::from_hdr_file(
        "assets/sky/brown_photostudio_02_4k.hdr",
        1.8,
        std::f32::consts::PI * -0.45,
    )?;
    scene.set_environment_light(env);

    let camera_eye = Vec3::new(0.0, 2.45, -3.32);
    let camera_target = Vec3::new(0.0, -0.22, 0.0);
    let camera = PinholeCamera::new(
        camera_eye,
        camera_target,
        Vec3::Y,
        47.0_f32.to_radians(),
        4.0 / 3.0,
    );

    Ok((scene, camera))
}

fn add_bunny(
    scene: &mut Scene,
    mesh: MeshIndex,
    material: MaterialIndex,
    position: Vec3,
    scale: f32,
    ground_pivot: Vec3,
) {
    let transform = Mat4::from_translation(position)
        * Mat4::from_rotation_y(std::f32::consts::PI)
        * Mat4::from_scale(Vec3::splat(scale))
        * Mat4::from_translation(-ground_pivot);
    scene.add_instance(mesh, material, transform);
}

fn glass_low_roughness() -> OpenPbrMaterial {
    OpenPbrMaterial::new(Vec3::splat(0.02))
        .with_base_weight(0.0)
        .with_specular_ior(1.4)
        .with_specular_roughness(0.1)
        .with_transmission_weight(1.0)
        .with_transmission_color(Vec3::new(0.94, 0.98, 1.0))
}

fn glass_high_roughness() -> OpenPbrMaterial {
    OpenPbrMaterial::new(Vec3::splat(0.02))
        .with_base_weight(0.0)
        .with_specular_ior(1.4)
        .with_specular_roughness(0.42)
        .with_transmission_weight(1.0)
        .with_transmission_color(Vec3::new(0.92, 0.97, 1.0))
}

fn clear_glass_no_dispersion() -> OpenPbrMaterial {
    OpenPbrMaterial::new(Vec3::ZERO)
        .with_base_weight(0.0)
        .with_specular_ior(1.4)
        .with_specular_roughness(0.0)
        .with_transmission_weight(1.0)
        .with_transmission_color(Vec3::ONE)
        .with_transmission_dispersion_scale(0.0)
}

fn clear_glass_strong_dispersion() -> OpenPbrMaterial {
    OpenPbrMaterial::new(Vec3::ZERO)
        .with_base_weight(0.0)
        .with_specular_ior(1.4)
        .with_specular_roughness(0.0)
        .with_transmission_weight(1.0)
        .with_transmission_color(Vec3::ONE)
        .with_transmission_dispersion_abbe_number(20.0)
        .with_transmission_dispersion_scale(1.0)
}

fn gold_metal() -> OpenPbrMaterial {
    OpenPbrMaterial::new(Vec3::new(1.0, 0.74, 0.28))
        .with_base_metalness(1.0)
        .with_specular_color(Vec3::new(1.0, 0.92, 0.72))
        .with_specular_roughness(0.16)
}

fn green_tinted_copper() -> OpenPbrMaterial {
    OpenPbrMaterial::new(Vec3::new(0.95, 0.48, 0.22))
        .with_base_metalness(1.0)
        .with_specular_color(Vec3::new(0.0, 1.0, 0.0))
        .with_specular_roughness(0.24)
}

fn titanium_with_noisy_thin_film(thickness: Arc<ScalarTexture>) -> OpenPbrMaterial {
    OpenPbrMaterial::new(Vec3::new(0.62, 0.64, 0.68))
        .with_base_metalness(1.0)
        .with_specular_color(Vec3::new(0.88, 0.9, 0.96))
        .with_specular_roughness(0.18)
        .with_thin_film_weight(1.0)
        .with_thin_film_ior(2.35)
        .with_thin_film_thickness_texture(thickness, 160.0, 520.0)
}

fn soap_thin_walled(thickness: Arc<ScalarTexture>) -> OpenPbrMaterial {
    OpenPbrMaterial::new(Vec3::new(0.92, 0.96, 1.0))
        .with_base_weight(0.0)
        .with_specular_ior(1.33)
        .with_specular_roughness(0.0)
        .with_transmission_weight(1.0)
        .with_transmission_color(Vec3::new(0.94, 0.98, 1.0))
        .with_geometry_thin_walled(true)
        .with_thin_film_weight(1.0)
        .with_thin_film_ior(1.33)
        .with_thin_film_thickness_texture(thickness, 10.0, 1000.0)
}

fn wood_no_coat(
    texture: Arc<Texture>,
    roughness: Arc<ScalarTexture>,
    normal_map: NormalMap,
) -> OpenPbrMaterial {
    let mut material = OpenPbrMaterial::new(Vec3::ONE)
        .with_specular_weight(0.28)
        .with_specular_roughness(0.45)
        .with_base_diffuse_roughness(0.55);
    material.base_color_texture = Some(texture);
    material.specular_roughness_texture = Some(roughness);
    material.normal_map = Some(normal_map);
    material.normal_strength = 0.65;
    material
}

fn wood_coat_darkening(
    texture: Arc<Texture>,
    roughness: Arc<ScalarTexture>,
    normal_map: NormalMap,
) -> OpenPbrMaterial {
    let mut material = OpenPbrMaterial::new(Vec3::ONE)
        .with_specular_weight(0.35)
        .with_specular_roughness(0.42)
        .with_base_diffuse_roughness(0.45)
        .with_coat_weight(1.0)
        .with_coat_color(Vec3::ONE)
        .with_coat_roughness(0.08)
        .with_coat_ior(1.65)
        .with_coat_darkening(1.0);
    material.base_color_texture = Some(texture);
    material.specular_roughness_texture = Some(roughness);
    material.normal_map = Some(normal_map.clone());
    material.coat_normal_map = Some(normal_map);
    material.normal_strength = 0.45;
    material.coat_normal_strength = 0.25;
    material
}

fn purple_coated_metal(rough_coat: bool) -> OpenPbrMaterial {
    let (roughness, ior) = if rough_coat {
        (0.42, 2.15)
    } else {
        (0.03, 1.55)
    };
    OpenPbrMaterial::new(Vec3::new(0.74, 0.72, 0.8))
        .with_base_metalness(1.0)
        .with_specular_color(Vec3::new(0.9, 0.88, 1.0))
        .with_specular_roughness(0.2)
        .with_coat_weight(1.0)
        .with_coat_color(Vec3::new(0.62, 0.22, 1.0))
        .with_coat_roughness(roughness)
        .with_coat_ior(ior)
        .with_coat_darkening(0.75)
}

fn fuzz_metal(fuzz_roughness: f32) -> OpenPbrMaterial {
    OpenPbrMaterial::new(Vec3::new(0.72, 0.7, 0.76))
        .with_base_metalness(1.0)
        .with_specular_color(Vec3::new(0.9, 0.88, 0.95))
        .with_specular_roughness(0.22)
        .with_fuzz(0.75, Vec3::new(0.8, 0.72, 1.0), fuzz_roughness)
}

fn rough_diffuse() -> OpenPbrMaterial {
    OpenPbrMaterial::new(Vec3::new(0.72, 0.52, 0.36))
        .with_specular_weight(0.12)
        .with_specular_roughness(0.82)
        .with_base_diffuse_roughness(0.92)
}

fn velvet_diffuse() -> OpenPbrMaterial {
    OpenPbrMaterial::new(Vec3::new(0.28, 0.04, 0.1))
        .with_specular_weight(0.05)
        .with_specular_roughness(0.88)
        .with_base_diffuse_roughness(0.95)
        .with_fuzz(1.0, Vec3::new(1.0, 0.32, 0.62), 0.5)
}
