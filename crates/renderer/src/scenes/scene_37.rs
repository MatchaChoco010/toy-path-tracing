//! Scene 36 と同じ配置で、Kulla & Conty 2017 energy compensation を有効にした版。

use std::error::Error;

use crate::{scene::PinholeCamera, scene::Scene};

use super::scene_36::create_metal_glass_roughness_rows;

pub fn create_scene_37(
    _ocio: &crate::color::OcioColorPipeline,
) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    create_metal_glass_roughness_rows(true)
}
