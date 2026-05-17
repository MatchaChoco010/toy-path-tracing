pub(super) mod gltf_scene_loader;
pub(super) mod obj_scene_loader;

mod openpbr_mori_knob;
mod openpbr_single_bunny;

pub(super) use openpbr_mori_knob::create_openpbr_mori_knob_scene;
pub(super) use openpbr_single_bunny::{
    create_single_openpbr_bunny_scene, create_single_openpbr_low_bunny_scene,
};
