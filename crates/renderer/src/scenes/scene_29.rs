//! puresky 環境光のもと、同じドラゴンモデルを左 SimplePBR / 右 Standard Surface で比較する。

use std::{error::Error, path::Path, sync::Arc};

use glam::{Mat4, Vec3};

use crate::{
    light::EnvironmentLight,
    material::{
        Material, NormalMap, NormalizedLambertMaterial, ScalarTexture, SimplePbrMaterial,
        StandardSurfaceMaterial, Texture, TextureColorSpace,
    },
    scene::PinholeCamera,
    scene::Scene,
    scene::{load_gltf, load_obj},
};

use super::{game_rotation_degrees, uniform_scale_for_height};

pub fn create_scene_29(
    _ocio: &crate::color::OcioColorPipeline,
) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();

    let floor_material = scene.add_material(Material::NormalizedLambert(
        NormalizedLambertMaterial::new(Vec3::splat(0.9)),
    ));
    let floor_mesh = scene.add_mesh(load_gltf(Path::new("assets/models/floor.glb"))?);
    let floor_transform = Mat4::from_scale(Vec3::new(20.0, 1.0, 20.0));
    scene.add_instance(floor_mesh, floor_material, floor_transform);

    let simple_pbr_material = scene.add_material(Material::SimplePBR(
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

    let base_color_tex = Arc::new(Texture::from_file_with_color_space(
        "assets/models/dragon-BaseColor.png",
        TextureColorSpace::Srgb,
        _ocio,
    )?);
    let metalness_tex = Arc::new(ScalarTexture::from_file(
        "assets/models/dragon-Metallic.png",
    )?);
    let roughness_tex = Arc::new(ScalarTexture::from_file(
        "assets/models/dragon-Roughness.png",
    )?);
    let normal_map = NormalMap::from_file("assets/models/dragon-Normal.png")?;
    let standard_material = scene.add_material(Material::StandardSurface(
        StandardSurfaceMaterial::new(Vec3::ONE)
            .with_metalness(1.0)
            .with_specular_roughness(1.0)
            .with_base_color_texture(base_color_tex)
            .with_metalness_texture(metalness_tex)
            .with_specular_roughness_texture(roughness_tex)
            .with_normal_map(normal_map),
    ));

    let dragon = load_obj(Path::new("assets/models/dragon.obj"))?;
    let dragon_scale = uniform_scale_for_height(&dragon, 1.8);
    let dragon_pivot = Vec3::new(
        dragon.bounds.center().x,
        dragon.bounds.min.y,
        dragon.bounds.center().z,
    );
    let dragon_mesh = scene.add_mesh(dragon);

    let face_camera = Mat4::from_quat(game_rotation_degrees(0.0, 90.0, 0.0));
    let dragon_local = face_camera
        * Mat4::from_scale(Vec3::splat(dragon_scale))
        * Mat4::from_translation(-dragon_pivot);

    let half_offset = 1.6_f32;
    let simple_pbr_transform =
        Mat4::from_translation(Vec3::new(-half_offset, 0.0, 0.0)) * dragon_local;
    scene.add_instance(dragon_mesh, simple_pbr_material, simple_pbr_transform);

    let standard_transform =
        Mat4::from_translation(Vec3::new(half_offset, 0.0, 0.0)) * dragon_local;
    scene.add_instance(dragon_mesh, standard_material, standard_transform);

    let env = EnvironmentLight::from_hdr_file(
        "assets/sky/kloofendal_48d_partly_cloudy_puresky_4k.hdr",
        0.5,
        0.0,
    )?;
    scene.set_environment_light(env);

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
