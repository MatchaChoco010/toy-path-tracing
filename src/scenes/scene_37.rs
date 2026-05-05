use std::error::Error;

use crate::{camera::PinholeCamera, scene::Scene};

use super::scene_36::create_metal_glass_roughness_rows;

pub fn create_scene_37() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    create_metal_glass_roughness_rows(true)
}
