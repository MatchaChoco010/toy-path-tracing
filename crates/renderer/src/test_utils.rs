use glam::{Mat4, Vec2, Vec3};

use crate::{
    material::ShadingVertex,
    math::OrthonormalBasis,
    scene::{InstanceIndex, TriangleRef},
};

pub fn approx_f(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}

pub fn approx_v3(a: Vec3, b: Vec3, eps: f32) -> bool {
    a.abs_diff_eq(b, eps)
}

pub fn shading_vertex_on_z(wo: Vec3) -> ShadingVertex {
    ShadingVertex {
        triangle: TriangleRef {
            instance_index: InstanceIndex(0),
            triangle_index: 0,
        },
        p: Vec3::ZERO,
        uv: Vec2::ZERO,
        dudx: 0.0,
        dvdx: 0.0,
        dudy: 0.0,
        dvdy: 0.0,
        ng: Vec3::Z,
        ns: Vec3::Z,
        wo,
        dpdu: Vec3::X,
        dpdv: Vec3::Y,
        dpdx: Vec3::ZERO,
        dpdy: Vec3::ZERO,
        dndu: Vec3::ZERO,
        dndv: Vec3::ZERO,
        frame: OrthonormalBasis::from_normal(Vec3::Z),
        front_face: true,
        path_throughput: Vec3::ONE,
        wavelength_lock: None,
        object_to_world: Mat4::IDENTITY,
        world_to_object: Mat4::IDENTITY,
        object_normal_to_world: glam::Mat3::IDENTITY,
        mtlx_regs: None,
        mtlx_dalbedo: None,
        mtlx_precomputed_for: None,
    }
}
