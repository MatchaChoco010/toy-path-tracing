//! OpenPBR の単体バニーで、暗めの rough diffuse にパーリンノイズの fuzz roughness を載せた velvet 風マテリアルを表示する。

use std::{error::Error, sync::Arc};

use glam::Vec3;

use crate::{
    material::{OpenPbrMaterial, ScalarTexture},
    scene::PinholeCamera,
    scene::Scene,
};

use super::helper::create_single_openpbr_bunny_scene;

pub fn create_scene_56(
    _ocio: &crate::color::OcioColorPipeline,
) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let fuzz_roughness = Arc::new(ScalarTexture::from_file(
        "assets/models/bunny_perlin_noise.png",
    )?);
    create_single_openpbr_bunny_scene(
        OpenPbrMaterial::new(Vec3::new(0.28, 0.04, 0.1))
            .with_specular_weight(0.05)
            .with_specular_roughness(0.88)
            .with_base_diffuse_roughness(0.95)
            .with_fuzz(1.0, Vec3::new(1.0, 0.32, 0.62), 0.8)
            .with_fuzz_roughness_texture(fuzz_roughness),
    )
}
