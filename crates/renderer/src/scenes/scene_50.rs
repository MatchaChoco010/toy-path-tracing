//! OpenPBR の単体バニーで、bunny 用パーリンノイズを thin-film thickness に使った titanium 風 metal を表示する。

use std::{error::Error, sync::Arc};

use glam::Vec3;

use crate::{
    camera::PinholeCamera,
    material::{OpenPbrMaterial, ScalarTexture},
    scene::Scene,
};

use super::openpbr_single_bunny::create_single_openpbr_bunny_scene;

pub fn create_scene_50() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let thickness = Arc::new(ScalarTexture::from_file(
        "assets/models/bunny_perlin_noise.png",
    )?);
    create_single_openpbr_bunny_scene(
        OpenPbrMaterial::new(Vec3::new(0.62, 0.64, 0.68))
            .with_base_metalness(1.0)
            .with_specular_color(Vec3::new(0.88, 0.9, 0.96))
            .with_specular_roughness(0.18)
            .with_thin_film_weight(1.0)
            .with_thin_film_ior(2.35)
            .with_thin_film_thickness_texture(thickness, 160.0, 520.0),
    )
}
