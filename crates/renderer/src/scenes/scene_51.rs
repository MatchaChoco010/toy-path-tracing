//! OpenPBR の単体バニーで、UV の上下方向 thickness テクスチャを使った thin-walled soap film 風透明マテリアルを表示する。

use std::{error::Error, sync::Arc};

use glam::Vec3;

use crate::{
    material::{OpenPbrMaterial, ScalarTexture},
    scene::PinholeCamera,
    scene::Scene,
};

use super::helper::create_single_openpbr_bunny_scene;

pub fn create_scene_51(
    _ocio: &crate::color::OcioColorPipeline,
) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let thickness = Arc::new(ScalarTexture::from_file(
        "assets/models/bunny_soap_thin_film_thickness.png",
    )?);
    create_single_openpbr_bunny_scene(
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
