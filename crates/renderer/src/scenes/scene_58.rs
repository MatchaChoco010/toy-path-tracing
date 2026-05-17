//! OpenPBR の mori-knob で、暗い赤紫の rough diffuse にパーリンノイズの fuzz weight を使った velvet 風マテリアルを表示する。

use std::{error::Error, sync::Arc};

use glam::Vec3;

use crate::{
    camera::PinholeCamera,
    material::{OpenPbrMaterial, ScalarTexture},
    scene::Scene,
};

use super::openpbr_mori_knob::create_openpbr_mori_knob_scene;

pub fn create_scene_58() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let fuzz_weight = Arc::new(ScalarTexture::from_file(
        "assets/models/mori_knob_perlin_noise.png",
    )?);
    create_openpbr_mori_knob_scene(
        OpenPbrMaterial::new(Vec3::new(0.28, 0.04, 0.1))
            .with_specular_weight(0.05)
            .with_specular_roughness(0.88)
            .with_base_diffuse_roughness(0.95)
            .with_fuzz(1.0, Vec3::new(1.0, 0.32, 0.62), 0.7)
            .with_fuzz_weight_texture(fuzz_weight),
    )
}
