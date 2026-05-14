//! OpenPBR の mori-knob で、パーリンノイズを thin-film thickness に使った淡い titanium 風 metal を表示する。

use std::{error::Error, sync::Arc};

use glam::Vec3;

use crate::{
    camera::PinholeCamera,
    material::{OpenPbrMaterial, ScalarTexture},
    scene::Scene,
};

use super::openpbr_mori_knob::create_openpbr_mori_knob_scene;

pub fn create_scene_62() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let thickness = Arc::new(ScalarTexture::from_file(
        "assets/models/mori_knob_perlin_noise.png",
    )?);
    create_openpbr_mori_knob_scene(
        OpenPbrMaterial::new(Vec3::new(0.62, 0.64, 0.68))
            .with_base_metalness(1.0)
            .with_specular_color(Vec3::new(0.88, 0.9, 0.96))
            .with_specular_roughness(0.05)
            .with_thin_film_weight(1.0)
            .with_thin_film_ior(2.2)
            .with_thin_film_thickness_texture(thickness, 100.0, 420.0),
    )
}
