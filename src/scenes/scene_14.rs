use std::error::Error;

use crate::{camera::PinholeCamera, scene::Scene};

pub fn create_scene_14() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    super::scene_13::create_sky_bunny_scene(
        "assets/sky/kloofendal_48d_partly_cloudy_puresky_4k.hdr",
        0.5,
    )
}
