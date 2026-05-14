//! OpenPBR の mori-knob で、パーリンノイズを thin-film thickness に使った thin-walled soap film 風透明マテリアルを表示する。

use std::{error::Error, sync::Arc};

use glam::Vec3;

use crate::{
    camera::PinholeCamera,
    material::{OpenPbrMaterial, ScalarTexture},
    scene::Scene,
};

use super::openpbr_mori_knob::create_openpbr_mori_knob_scene;

pub fn create_scene_59() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let thickness = Arc::new(ScalarTexture::from_file(
        "assets/models/mori_knob_perlin_noise.png",
    )?);
    create_openpbr_mori_knob_scene(
        OpenPbrMaterial::new(Vec3::new(0.92, 0.96, 1.0))
            .with_base_weight(0.0)
            .with_specular_ior(1.33)
            .with_specular_roughness(0.0)
            .with_transmission_weight(1.0)
            .with_transmission_color(Vec3::new(0.94, 0.98, 1.0))
            .with_geometry_thin_walled(true)
            .with_thin_film_weight(1.0)
            .with_thin_film_ior(1.33)
            .with_thin_film_thickness_texture(thickness, 10.0, 1000.0),
    )
}
