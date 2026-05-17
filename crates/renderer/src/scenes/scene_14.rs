//! Scene 13 と同じ diffuse バニーを puresky HDRI 環境光で照らす。

use std::error::Error;

use crate::{scene::PinholeCamera, scene::Scene};

pub fn create_scene_14(
    _ocio: &crate::color::OcioColorPipeline,
) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    super::scene_13::create_sky_bunny_scene(
        "assets/sky/kloofendal_48d_partly_cloudy_puresky_4k.hdr",
        0.5,
    )
}
