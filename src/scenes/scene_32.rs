use std::error::Error;

use crate::{camera::PinholeCamera, scene::Scene};

use super::scene_31::create_paper_plane_scene;

pub fn create_scene_32() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    create_paper_plane_scene(0.5)
}
