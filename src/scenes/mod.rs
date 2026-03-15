mod scene_0;
mod scene_1;

use glam::{EulerRot, Quat, Vec3};
use std::error::Error;

use crate::{
    camera::PinholeCamera,
    mesh::{Bounds, Mesh},
    scene::{MeshIndex, Scene},
};

pub fn load_scene(scene_index: u32) -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    match scene_index {
        1 => scene_1::create_scene_1(),
        _ => scene_0::create_scene_0(),
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct InstanceSpec {
    pub translation: Vec3,
    pub scale_multiplier: f32,
    pub rotation_degrees: Vec3,
}

pub(super) fn create_camera(
    bounds: Bounds,
    eye_offset: Vec3,
    look_at_offset: Vec3,
    fov_y: f32,
) -> PinholeCamera {
    let center = bounds.center();
    let extent = bounds.extent();
    let radius = 0.5 * extent.length().max(1.0e-3);
    let distance = radius / (0.5 * fov_y).tan();
    let scaled_eye_offset = Vec3::new(
        eye_offset.x * extent.x.max(radius),
        eye_offset.y * extent.y.max(radius),
        eye_offset.z * distance,
    );
    let look_at = center
        + Vec3::new(
            look_at_offset.x * extent.x,
            look_at_offset.y * extent.y,
            look_at_offset.z * extent.z,
        );

    PinholeCamera::new(center + scaled_eye_offset, look_at, Vec3::Y, fov_y)
}

pub(super) fn uniform_scale_for_height(mesh: &Mesh, target_height: f32) -> f32 {
    target_height / mesh.bounds.extent().y.max(1.0e-3)
}

pub(super) fn game_rotation_degrees(x_degrees: f32, y_degrees: f32, z_degrees: f32) -> Quat {
    Quat::from_euler(
        EulerRot::YXZ,
        y_degrees.to_radians(),
        x_degrees.to_radians(),
        z_degrees.to_radians(),
    )
}

pub(super) fn add_instances(
    scene: &mut Scene,
    mesh_index: MeshIndex,
    base_scale: f32,
    instances: &[InstanceSpec],
) {
    for instance in instances {
        scene.add_instance(
            mesh_index,
            instance.translation,
            game_rotation_degrees(
                instance.rotation_degrees.x,
                instance.rotation_degrees.y,
                instance.rotation_degrees.z,
            ),
            Vec3::splat(base_scale * instance.scale_multiplier),
        );
    }
}
