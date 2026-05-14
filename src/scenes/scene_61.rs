//! OpenPBR の mori-knob で、銅色 metal にパーリンノイズの fuzz weight を載せたマテリアルを表示する。

use std::{error::Error, sync::Arc};

use glam::Vec3;

use crate::{
    camera::PinholeCamera,
    material::{OpenPbrMaterial, ScalarTexture},
    scene::Scene,
};

use super::openpbr_mori_knob::create_openpbr_mori_knob_scene;

pub fn create_scene_61() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let fuzz_weight = Arc::new(ScalarTexture::from_file(
        "assets/models/mori_knob_perlin_noise.png",
    )?);
    create_openpbr_mori_knob_scene(
        OpenPbrMaterial::new(Vec3::new(0.95, 0.48, 0.22))
            .with_base_metalness(1.0)
            .with_specular_roughness(0.1)
            .with_fuzz(0.8, Vec3::new(0.98, 0.98, 0.98), 0.88)
            .with_fuzz_weight_texture(fuzz_weight),
    )
}
