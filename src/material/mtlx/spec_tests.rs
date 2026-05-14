#![cfg(test)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use glam::{Vec2, Vec3, Vec4};

use crate::material::pattern::noise::{cellnoise2d, hsv_to_rgb};
use crate::material::{ScalarTexture, ShadingVertex, Texture};
use crate::math::OrthonormalBasis;
use crate::scene::{InstanceIndex, TriangleRef};
use crate::scene_loader::mtlx_loader::flatten::{
    FlatGraph, FlatInput, FlatNode, FlatNodeInput, FlatNodeKind,
    GeometricKind as FlatGeometricKind, flatten_material,
};
use crate::scene_loader::mtlx_loader::library::load_standard_library;
use crate::scene_loader::mtlx_loader::parser::parse_str;
use crate::scene_loader::mtlx_loader::types::{MtlxType, MtlxValue};

use super::MtlxScratch;
use super::compiled::{
    AddressMode, ArithOp, BlendOp, ChiangHairRoughnessOutput, ClosureKind, ClosureNode,
    CombineKind, CompareOp, CompiledMaterial, FilterType, GeomSpace, GeometricKind, ImageKind,
    ImageTexture, Instruction, LogicalOp, MaskOp, MergeOp, NoiseKind, NoiseOutput, Operand,
    ParamRef, UdimTile, UdimTiles, UnaryOp, Value, ValueType, WorleyStyle,
};

fn dummy_sv() -> ShadingVertex {
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
        wo: Vec3::Z,
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
        object_to_world: glam::Mat4::IDENTITY,
        world_to_object: glam::Mat4::IDENTITY,
        object_normal_to_world: glam::Mat3::IDENTITY,
        mtlx_regs: None,
        mtlx_dalbedo: None,
        mtlx_precomputed_for: None,
    }
}

fn lib_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("lib/materialx/libraries")
}

fn run(
    instructions: Vec<Instruction>,
    operand_pool: Vec<Operand>,
    value_pool: Vec<Value>,
    num_registers: u32,
) -> Vec<Value> {
    run_with_sv(
        instructions,
        operand_pool,
        value_pool,
        num_registers,
        dummy_sv(),
    )
}

fn run_with_sv(
    instructions: Vec<Instruction>,
    operand_pool: Vec<Operand>,
    value_pool: Vec<Value>,
    num_registers: u32,
    sv: ShadingVertex,
) -> Vec<Value> {
    let mut scratch = MtlxScratch::default();
    let compiled = CompiledMaterial {
        instructions,
        operand_pool,
        value_pool,
        opacity_instructions: Vec::new(),
        opacity_operand_pool: Vec::new(),
        opacity_closure_nodes: Vec::new(),
        opacity_num_registers: 0,
        num_registers,
        closure_nodes: vec![ClosureNode::Zero],
        root: 0,
        passthrough: false,
        max_emission: 0.0,
        may_emit: false,
        has_opacity_test: false,
        thin_walled: false,
        sheen_lut: None,
        mtlx_dielectric_lut: None,
        mtlx_generalized_schlick_lut: None,
    };
    let handle = scratch.alloc_regs(num_registers as usize);
    super::runtime::run_instructions(&compiled, &sv, &mut scratch, handle);
    scratch.regs_slice(handle).to_vec()
}

fn eval_compiled_le(compiled: &CompiledMaterial) -> Vec3 {
    let mut scratch = MtlxScratch::default();
    let handle = scratch.alloc_regs(compiled.num_registers as usize);
    super::runtime::run_instructions(compiled, &dummy_sv(), &mut scratch, handle);
    super::runtime::evaluate_le(compiled, scratch.regs_slice(handle), &dummy_sv())
        .unwrap_or(Vec3::ZERO)
}

fn run_arith(op: ArithOp, ty: ValueType, a: Value, b: Value) -> Value {
    let regs = run(
        vec![Instruction::Arith {
            dst: 0,
            op,
            ty,
            a: Operand::Const(0),
            b: Operand::Const(1),
        }],
        Vec::new(),
        vec![a, b],
        1,
    );
    regs[0]
}

fn run_unary(op: UnaryOp, ty: ValueType, v: Value) -> Value {
    let regs = run(
        vec![Instruction::Unary {
            dst: 0,
            op,
            ty,
            src: Operand::Const(0),
        }],
        Vec::new(),
        vec![v],
        1,
    );
    regs[0]
}

#[test]
fn spec_facingratio_matches_official_nodegraph_formula() {
    let cases = [
        (true, false, Vec3::Z, Vec3::Z, 1.0),
        (false, false, Vec3::Z, Vec3::Z, -1.0),
        (true, false, -Vec3::Z, Vec3::Z, 1.0),
        (false, false, -Vec3::Z, Vec3::Z, 1.0),
        (true, true, Vec3::Z, Vec3::Z, 0.0),
    ];
    for (faceforward, invert, view, normal, expected) in cases {
        let regs = run(
            vec![Instruction::FacingRatio {
                dst: 0,
                view: Operand::Const(0),
                normal: Operand::Const(1),
                invert,
                faceforward,
            }],
            Vec::new(),
            vec![Value::Vector3(view), Value::Vector3(normal)],
            1,
        );
        assert!(approx_f(regs[0].as_float(), expected, 1.0e-6));
    }
}

#[test]
fn spec_viewdirection_matches_materialx_camera_to_surface_direction() {
    let regs = run(
        vec![Instruction::LoadGeom {
            dst: 0,
            kind: GeometricKind::ViewDirection(GeomSpace::World),
        }],
        Vec::new(),
        Vec::new(),
        1,
    );

    assert!(approx_v3(regs[0].as_vector3(), Vec3::NEG_Z, 1.0e-6));
}

#[test]
fn spec_geometric_normalized_vectors_use_raw_normalize() {
    for kind in [
        GeometricKind::Tangent(GeomSpace::World),
        GeometricKind::Bitangent(GeomSpace::World),
        GeometricKind::ViewDirection(GeomSpace::World),
    ] {
        let mut sv = dummy_sv();
        sv.dpdu = Vec3::ZERO;
        sv.dpdv = Vec3::ZERO;
        sv.wo = Vec3::ZERO;
        let regs = run_with_sv(
            vec![Instruction::LoadGeom { dst: 0, kind }],
            Vec::new(),
            Vec::new(),
            1,
            sv,
        );
        assert!(regs[0].as_vector3().is_nan());
    }
}

#[test]
fn spec_geometric_normal_object_space_uses_raw_transform_normal() {
    let mut sv = dummy_sv();
    sv.ns = Vec3::X;
    sv.object_to_world = glam::Mat4::from_scale(Vec3::new(2.0, 3.0, 4.0));
    sv.world_to_object = sv.object_to_world.inverse();
    sv.object_normal_to_world = glam::Mat3::from_mat4(sv.world_to_object.transpose());
    let regs = run_with_sv(
        vec![Instruction::LoadGeom {
            dst: 0,
            kind: GeometricKind::Normal(GeomSpace::Object),
        }],
        Vec::new(),
        Vec::new(),
        1,
        sv,
    );
    assert!(approx_v3(
        regs[0].as_vector3(),
        Vec3::new(2.0, 0.0, 0.0),
        1.0e-6
    ));
}

fn compile_geometric_color_graph(
    kind: FlatGeometricKind,
    inputs: Vec<FlatNodeInput>,
) -> Result<CompiledMaterial, String> {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Geometric { kind, index: 0 },
                output_type: MtlxType::Vector3,
                inputs,
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission_color".to_string(),
                    ty: MtlxType::Color3,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: "geometric_space".to_string(),
    };
    super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .map_err(|err| err.to_string())
}

#[test]
fn spec_geometric_space_defaults_and_static_string_validation() {
    let compiled = compile_geometric_color_graph(FlatGeometricKind::ViewDirection, Vec::new())
        .expect("viewdirection default space should compile");
    let view_kind = compiled.instructions.iter().find_map(|instr| match instr {
        Instruction::LoadGeom { kind, .. } => Some(*kind),
        _ => None,
    });
    assert_eq!(
        view_kind,
        Some(GeometricKind::ViewDirection(GeomSpace::World))
    );

    let dynamic_space = FlatNodeInput {
        name: "space".to_string(),
        ty: MtlxType::String,
        colorspace: None,
        unit: None,
        unittype: None,
        binding: FlatInput::Node {
            node: 0,
            output: None,
        },
    };
    let err = compile_geometric_color_graph(FlatGeometricKind::Position, vec![dynamic_space])
        .expect_err("dynamic geometric space should not silently fall back to object");
    assert!(err.contains("geometric.space must be a static string value"));
}

fn run_convert(from: ValueType, to: ValueType, v: Value) -> Value {
    let regs = run(
        vec![Instruction::Convert {
            dst: 0,
            from,
            to,
            src: Operand::Const(0),
        }],
        Vec::new(),
        vec![v],
        1,
    );
    regs[0]
}

fn run_dot(ty: ValueType, a: Value, b: Value) -> Value {
    let regs = run(
        vec![Instruction::DotProduct {
            dst: 0,
            ty,
            a: Operand::Const(0),
            b: Operand::Const(1),
        }],
        Vec::new(),
        vec![a, b],
        1,
    );
    regs[0]
}

fn run_cross(a: Value, b: Value) -> Value {
    let regs = run(
        vec![Instruction::CrossProduct {
            dst: 0,
            a: Operand::Const(0),
            b: Operand::Const(1),
        }],
        Vec::new(),
        vec![a, b],
        1,
    );
    regs[0]
}

fn run_distance(ty: ValueType, a: Value, b: Value) -> Value {
    let regs = run(
        vec![Instruction::Distance {
            dst: 0,
            ty,
            a: Operand::Const(0),
            b: Operand::Const(1),
        }],
        Vec::new(),
        vec![a, b],
        1,
    );
    regs[0]
}

fn run_reflect(i: Value, n: Value) -> Value {
    let regs = run(
        vec![Instruction::Reflect {
            dst: 0,
            i: Operand::Const(0),
            n: Operand::Const(1),
        }],
        Vec::new(),
        vec![i, n],
        1,
    );
    regs[0]
}

fn run_refract(i: Value, n: Value, eta: Value) -> Value {
    let regs = run(
        vec![Instruction::Refract {
            dst: 0,
            i: Operand::Const(0),
            n: Operand::Const(1),
            eta: Operand::Const(2),
        }],
        Vec::new(),
        vec![i, n, eta],
        1,
    );
    regs[0]
}

fn run_rotate2d(v: Value, amount: Value) -> Value {
    let regs = run(
        vec![Instruction::Rotate2d {
            dst: 0,
            v: Operand::Const(0),
            amount: Operand::Const(1),
        }],
        Vec::new(),
        vec![v, amount],
        1,
    );
    regs[0]
}

fn run_rotate3d(v: Value, axis: Value, amount: Value) -> Value {
    let regs = run(
        vec![Instruction::Rotate3d {
            dst: 0,
            v: Operand::Const(0),
            axis: Operand::Const(1),
            amount: Operand::Const(2),
        }],
        Vec::new(),
        vec![v, axis, amount],
        1,
    );
    regs[0]
}

fn run_mix(ty: ValueType, bg: Value, fg: Value, mix: Value) -> Value {
    let regs = run(
        vec![Instruction::MixValue {
            dst: 0,
            ty,
            bg: Operand::Const(0),
            fg: Operand::Const(1),
            mix: Operand::Const(2),
        }],
        Vec::new(),
        vec![bg, fg, mix],
        1,
    );
    regs[0]
}

fn run_clamp(ty: ValueType, v: Value, lo: Value, hi: Value) -> Value {
    let regs = run(
        vec![Instruction::Clamp {
            dst: 0,
            ty,
            v: Operand::Const(0),
            lo: Operand::Const(1),
            hi: Operand::Const(2),
        }],
        Vec::new(),
        vec![v, lo, hi],
        1,
    );
    regs[0]
}

fn run_smoothstep(ty: ValueType, v: Value, lo: Value, hi: Value) -> Value {
    let regs = run(
        vec![Instruction::Smoothstep {
            dst: 0,
            ty,
            v: Operand::Const(0),
            lo: Operand::Const(1),
            hi: Operand::Const(2),
        }],
        Vec::new(),
        vec![v, lo, hi],
        1,
    );
    regs[0]
}

fn run_hextilednormalmap_missing(default: Vec3) -> Value {
    let regs = run(
        vec![Instruction::HextiledNormalMap {
            dst: 0,
            texture: None,
            flip_g: false,
            operands_start: 0,
        }],
        vec![
            Operand::Const(0),
            Operand::Const(1),
            Operand::Const(2),
            Operand::Const(3),
            Operand::Const(4),
            Operand::Const(5),
            Operand::Const(6),
            Operand::Const(7),
            Operand::Const(8),
            Operand::Const(9),
            Operand::Const(10),
            Operand::Const(11),
            Operand::Const(12),
            Operand::Const(13),
        ],
        vec![
            Value::Vector2(Vec2::ZERO),
            Value::Vector2(Vec2::ONE),
            Value::Float(1.0),
            Value::Vector2(Vec2::new(0.0, 360.0)),
            Value::Float(1.0),
            Value::Vector2(Vec2::new(0.5, 2.0)),
            Value::Float(1.0),
            Value::Vector2(Vec2::new(0.0, 1.0)),
            Value::Float(0.5),
            Value::Float(1.0),
            Value::Vector3(default),
            Value::Vector3(Vec3::ZERO),
            Value::Vector3(Vec3::ZERO),
            Value::Vector3(Vec3::ZERO),
        ],
        1,
    );
    regs[0]
}

fn run_roughness_anisotropy(roughness: f32, anisotropy: f32) -> Value {
    let regs = run(
        vec![Instruction::RoughnessAnisotropy {
            dst: 0,
            r: Operand::Const(0),
            a: Operand::Const(1),
        }],
        Vec::new(),
        vec![Value::Float(roughness), Value::Float(anisotropy)],
        1,
    );
    regs[0]
}

fn run_glossiness_anisotropy(glossiness: f32, anisotropy: f32) -> Value {
    let regs = run(
        vec![Instruction::GlossinessAnisotropy {
            dst: 0,
            g: Operand::Const(0),
            a: Operand::Const(1),
        }],
        Vec::new(),
        vec![Value::Float(glossiness), Value::Float(anisotropy)],
        1,
    );
    regs[0]
}

fn run_roughness_dual(roughness: Vec2) -> Value {
    let regs = run(
        vec![Instruction::RoughnessDual {
            dst: 0,
            src: Operand::Const(0),
        }],
        Vec::new(),
        vec![Value::Vector2(roughness)],
        1,
    );
    regs[0]
}

fn run_compare(op: CompareOp, v1: Value, v2: Value, in_true: Value, in_false: Value) -> Value {
    let regs = run(
        vec![Instruction::Compare {
            dst: 0,
            op,
            v1: Operand::Const(0),
            v2: Operand::Const(1),
            in_true: Operand::Const(2),
            in_false: Operand::Const(3),
        }],
        Vec::new(),
        vec![v1, v2, in_true, in_false],
        1,
    );
    regs[0]
}

fn run_compare_bool(op: CompareOp, v1: Value, v2: Value) -> Value {
    let regs = run(
        vec![Instruction::CompareBool {
            dst: 0,
            op,
            v1: Operand::Const(0),
            v2: Operand::Const(1),
        }],
        Vec::new(),
        vec![v1, v2],
        1,
    );
    regs[0]
}

fn run_logical(op: LogicalOp, a: Value, b: Value) -> Value {
    let regs = run(
        vec![Instruction::Logical {
            dst: 0,
            op,
            a: Operand::Const(0),
            b: Operand::Const(1),
        }],
        Vec::new(),
        vec![a, b],
        1,
    );
    regs[0]
}

fn run_combine(kind: CombineKind, vals: &[Value]) -> Value {
    let mut value_pool = Vec::new();
    let mut operand_pool = Vec::new();
    for v in vals {
        operand_pool.push(Operand::Const(value_pool.len() as u32));
        value_pool.push(*v);
    }
    let regs = run(
        vec![Instruction::Combine {
            dst: 0,
            kind,
            operands_start: 0,
        }],
        operand_pool,
        value_pool,
        1,
    );
    regs[0]
}

fn run_switch(ty: ValueType, which: Value, branches: [Value; 10]) -> Value {
    let mut value_pool = vec![which];
    value_pool.extend(branches);
    let operand_pool = (1..=10).map(Operand::Const).collect();
    let regs = run(
        vec![Instruction::Switch {
            dst: 0,
            ty,
            which: Operand::Const(0),
            branches_start: 0,
        }],
        operand_pool,
        value_pool,
        1,
    );
    regs[0]
}

fn run_extract(in_ty: ValueType, v: Value, idx: i32) -> Value {
    let regs = run(
        vec![Instruction::Extract {
            dst: 0,
            in_ty,
            src: Operand::Const(0),
            idx: Operand::Const(1),
        }],
        Vec::new(),
        vec![v, Value::Integer(idx)],
        1,
    );
    regs[0]
}

fn run_blend(op: BlendOp, ty: ValueType, bg: Value, fg: Value, mix: Value) -> Value {
    let regs = run(
        vec![Instruction::Blend {
            dst: 0,
            op,
            ty,
            bg: Operand::Const(0),
            fg: Operand::Const(1),
            mix: Operand::Const(2),
        }],
        Vec::new(),
        vec![bg, fg, mix],
        1,
    );
    regs[0]
}

fn run_merge(op: MergeOp, bg: Value, fg: Value, mix: Value) -> Value {
    let regs = run(
        vec![Instruction::Merge {
            dst: 0,
            op,
            bg: Operand::Const(0),
            fg: Operand::Const(1),
            mix: Operand::Const(2),
        }],
        Vec::new(),
        vec![bg, fg, mix],
        1,
    );
    regs[0]
}

fn run_mask(op: MaskOp, ty: ValueType, v: Value, mask: Value) -> Value {
    let regs = run(
        vec![Instruction::Mask {
            dst: 0,
            op,
            ty,
            v: Operand::Const(0),
            mask: Operand::Const(1),
        }],
        Vec::new(),
        vec![v, mask],
        1,
    );
    regs[0]
}

fn run_premult(v: Value) -> Value {
    let regs = run(
        vec![Instruction::Premult {
            dst: 0,
            src: Operand::Const(0),
        }],
        Vec::new(),
        vec![v],
        1,
    );
    regs[0]
}

fn run_unpremult(v: Value) -> Value {
    let regs = run(
        vec![Instruction::Unpremult {
            dst: 0,
            src: Operand::Const(0),
        }],
        Vec::new(),
        vec![v],
        1,
    );
    regs[0]
}

fn run_contrast(ty: ValueType, v: Value, amount: Value, pivot: Value) -> Value {
    let regs = run(
        vec![Instruction::Contrast {
            dst: 0,
            ty,
            v: Operand::Const(0),
            amount: Operand::Const(1),
            pivot: Operand::Const(2),
        }],
        Vec::new(),
        vec![v, amount, pivot],
        1,
    );
    regs[0]
}

fn run_range(
    ty: ValueType,
    doclamp: bool,
    v: Value,
    inlow: Value,
    inhigh: Value,
    gamma: Value,
    outlow: Value,
    outhigh: Value,
) -> Value {
    let value_pool = vec![v, inlow, inhigh, gamma, outlow, outhigh];
    let operand_pool = (0..6).map(Operand::Const).collect();
    let regs = run(
        vec![Instruction::Range {
            dst: 0,
            ty,
            doclamp,
            operands_start: 0,
        }],
        operand_pool,
        value_pool,
        1,
    );
    regs[0]
}

fn run_saturate(ty: ValueType, c: Value, amount: Value, lumacoeffs: Value) -> Value {
    let regs = run(
        vec![Instruction::Saturate {
            dst: 0,
            ty,
            c: Operand::Const(0),
            amount: Operand::Const(1),
            lumacoeffs: Operand::Const(2),
        }],
        Vec::new(),
        vec![c, amount, lumacoeffs],
        1,
    );
    regs[0]
}

fn run_colorcorrect(ty: ValueType, values: [Value; 9]) -> Value {
    let operand_pool = (0..9).map(Operand::Const).collect();
    let regs = run(
        vec![Instruction::ColorCorrect {
            dst: 0,
            ty,
            operands_start: 0,
        }],
        operand_pool,
        values.to_vec(),
        1,
    );
    regs[0]
}

fn run_remap(
    ty: ValueType,
    v: Value,
    inlow: Value,
    inhigh: Value,
    outlow: Value,
    outhigh: Value,
) -> Value {
    let value_pool = vec![v, inlow, inhigh, outlow, outhigh];
    let operand_pool = (0..5).map(Operand::Const).collect();
    let regs = run(
        vec![Instruction::Remap {
            dst: 0,
            ty,
            operands_start: 0,
        }],
        operand_pool,
        value_pool,
        1,
    );
    regs[0]
}

fn approx_f(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}

fn approx_v2(a: Vec2, b: Vec2, eps: f32) -> bool {
    a.abs_diff_eq(b, eps)
}

fn approx_v3(a: Vec3, b: Vec3, eps: f32) -> bool {
    a.abs_diff_eq(b, eps)
}

#[test]
fn spec_add_float() {
    let r = run_arith(
        ArithOp::Add,
        ValueType::Float,
        Value::Float(1.5),
        Value::Float(2.25),
    );
    assert_eq!(r.as_float(), 3.75);
}

#[test]
fn spec_add_color3() {
    let r = run_arith(
        ArithOp::Add,
        ValueType::Color3,
        Value::Color3(Vec3::new(0.1, 0.2, 0.3)),
        Value::Color3(Vec3::new(0.4, 0.5, 0.6)),
    );
    assert!(approx_v3(r.as_color3(), Vec3::new(0.5, 0.7, 0.9), 1.0e-6));
}

#[test]
fn spec_subtract_returns_in1_minus_in2() {
    let r = run_arith(
        ArithOp::Subtract,
        ValueType::Float,
        Value::Float(5.0),
        Value::Float(2.0),
    );
    assert_eq!(r.as_float(), 3.0);
}

#[test]
fn spec_multiply_componentwise_for_vectors() {
    let r = run_arith(
        ArithOp::Multiply,
        ValueType::Vector3,
        Value::Vector3(Vec3::new(2.0, 3.0, 4.0)),
        Value::Vector3(Vec3::new(0.5, 0.25, 0.125)),
    );
    assert!(approx_v3(r.as_vector3(), Vec3::new(1.0, 0.75, 0.5), 1.0e-6));
}

#[test]
fn spec_divide_returns_in1_over_in2() {
    let r = run_arith(
        ArithOp::Divide,
        ValueType::Float,
        Value::Float(10.0),
        Value::Float(4.0),
    );
    assert_eq!(r.as_float(), 2.5);
}

#[test]
fn spec_modulo_is_non_negative() {
    let r = run_arith(
        ArithOp::Modulo,
        ValueType::Float,
        Value::Float(-1.5),
        Value::Float(2.0),
    );
    let v = r.as_float();
    assert!(v >= 0.0, "modulo of negative should be non-negative: {}", v);
    assert!(approx_f(v, 0.5, 1.0e-6));
}

#[test]
fn spec_modulo_matches_mdl_floor_formula_for_negative_divisor() {
    let r = run_arith(
        ArithOp::Modulo,
        ValueType::Float,
        Value::Float(1.0),
        Value::Float(-2.0),
    );
    assert!(approx_f(r.as_float(), -1.0, 1.0e-6));
}

#[test]
fn spec_min_max() {
    let mn = run_arith(
        ArithOp::Min,
        ValueType::Float,
        Value::Float(3.0),
        Value::Float(2.0),
    );
    let mx = run_arith(
        ArithOp::Max,
        ValueType::Float,
        Value::Float(3.0),
        Value::Float(2.0),
    );
    assert_eq!(mn.as_float(), 2.0);
    assert_eq!(mx.as_float(), 3.0);
}

#[test]
fn spec_min_max_vector_and_float_rhs_are_componentwise() {
    let mn = run_arith(
        ArithOp::Min,
        ValueType::Vector3,
        Value::Vector3(Vec3::new(-1.0, 0.5, 2.0)),
        Value::Float(0.25),
    );
    let mx = run_arith(
        ArithOp::Max,
        ValueType::Color4,
        Value::Color4(Vec4::new(-1.0, 0.5, 2.0, 0.0)),
        Value::Float(0.25),
    );
    assert!(approx_v3(
        mn.as_vector3(),
        Vec3::new(-1.0, 0.25, 0.25),
        1.0e-6
    ));
    assert!((mx.as_color4() - Vec4::new(0.25, 0.5, 2.0, 0.25)).length() < 1.0e-6);
}

#[test]
fn spec_power_pow_in1_in2() {
    let r = run_arith(
        ArithOp::Power,
        ValueType::Float,
        Value::Float(2.0),
        Value::Float(10.0),
    );
    assert_eq!(r.as_float(), 1024.0);
}

#[test]
fn spec_safepower_preserves_negative_sign() {
    let r = run_arith(
        ArithOp::SafePower,
        ValueType::Float,
        Value::Float(-2.0),
        Value::Float(3.0),
    );
    assert!(approx_f(r.as_float(), -8.0, 1.0e-6));
}

#[test]
fn spec_atan2_in_radians() {
    let r = run_arith(
        ArithOp::Atan2,
        ValueType::Float,
        Value::Float(1.0),
        Value::Float(1.0),
    );
    assert!(approx_f(r.as_float(), std::f32::consts::FRAC_PI_4, 1.0e-6));
}

#[test]
fn spec_sin_cos_radians() {
    let s = run_unary(
        UnaryOp::Sin,
        ValueType::Float,
        Value::Float(std::f32::consts::FRAC_PI_2),
    );
    let c = run_unary(UnaryOp::Cos, ValueType::Float, Value::Float(0.0));
    assert!(approx_f(s.as_float(), 1.0, 1.0e-6));
    assert!(approx_f(c.as_float(), 1.0, 1.0e-6));
}

#[test]
fn spec_sqrt_ln_exp() {
    assert!(approx_f(
        run_unary(UnaryOp::Sqrt, ValueType::Float, Value::Float(9.0)).as_float(),
        3.0,
        1.0e-6
    ));
    assert!(approx_f(
        run_unary(
            UnaryOp::Ln,
            ValueType::Float,
            Value::Float(std::f32::consts::E)
        )
        .as_float(),
        1.0,
        1.0e-6
    ));
    assert!(approx_f(
        run_unary(UnaryOp::Exp, ValueType::Float, Value::Float(0.0)).as_float(),
        1.0,
        1.0e-6
    ));
}

#[test]
fn spec_unary_domains_match_mdl_without_clamping() {
    assert!(
        run_unary(UnaryOp::Sqrt, ValueType::Float, Value::Float(-1.0))
            .as_float()
            .is_nan()
    );
    assert!(
        run_unary(UnaryOp::Ln, ValueType::Float, Value::Float(-1.0))
            .as_float()
            .is_nan()
    );
    assert!(
        run_unary(UnaryOp::Asin, ValueType::Float, Value::Float(2.0))
            .as_float()
            .is_nan()
    );
    assert!(
        run_unary(UnaryOp::Acos, ValueType::Float, Value::Float(2.0))
            .as_float()
            .is_nan()
    );
}

#[test]
fn spec_absval_sign_floor_ceil_round_fract() {
    let absv = run_unary(UnaryOp::Abs, ValueType::Float, Value::Float(-3.5));
    let sign_pos = run_unary(UnaryOp::Sign, ValueType::Float, Value::Float(2.0));
    let sign_neg = run_unary(UnaryOp::Sign, ValueType::Float, Value::Float(-0.001));
    let sign_zero = run_unary(UnaryOp::Sign, ValueType::Float, Value::Float(0.0));
    let fl = run_unary(UnaryOp::Floor, ValueType::Float, Value::Float(2.7));
    let ce = run_unary(UnaryOp::Ceil, ValueType::Float, Value::Float(2.1));
    let rd = run_unary(UnaryOp::Round, ValueType::Float, Value::Float(2.5));
    let rd_neg = run_unary(UnaryOp::Round, ValueType::Float, Value::Float(-2.5));
    let rd_even = run_unary(UnaryOp::Round, ValueType::Float, Value::Float(3.5));
    let fr = run_unary(UnaryOp::Fract, ValueType::Float, Value::Float(2.7));
    let fr_neg = run_unary(UnaryOp::Fract, ValueType::Float, Value::Float(-1.25));
    assert_eq!(absv.as_float(), 3.5);
    assert_eq!(sign_pos.as_float(), 1.0);
    assert_eq!(sign_neg.as_float(), -1.0);
    assert_eq!(sign_zero.as_float(), 0.0);
    assert_eq!(fl.as_float(), 2.0);
    assert_eq!(ce.as_float(), 3.0);
    assert!(approx_f(rd.as_float(), 2.0, 1.0e-5));
    assert!(approx_f(rd_neg.as_float(), -2.0, 1.0e-5));
    assert!(approx_f(rd_even.as_float(), 4.0, 1.0e-5));
    assert!(approx_f(fr.as_float(), 0.7, 1.0e-5));
    assert!(approx_f(fr_neg.as_float(), 0.75, 1.0e-5));
}

#[test]
fn spec_invert_subtracts_from_one() {
    let r = run_unary(UnaryOp::Invert, ValueType::Float, Value::Float(0.3));
    assert!(approx_f(r.as_float(), 0.7, 1.0e-6));
}

#[test]
fn spec_invert_via_arith_subtract() {
    let r = run_arith(
        ArithOp::Subtract,
        ValueType::Color3,
        Value::Color3(Vec3::splat(0.4)),
        Value::Color3(Vec3::splat(0.1)),
    );
    let c = r.as_color3();
    assert!((c - Vec3::splat(0.3)).length() < 1e-6);
}

#[test]
fn spec_trianglewave_matches_stdlib_nodegraph() {
    let r = |x: f32| run_unary(UnaryOp::Trianglewave, ValueType::Float, Value::Float(x)).as_float();
    assert!((r(0.0) - 0.0).abs() < 1e-6);
    assert!((r(0.25) - 0.25).abs() < 1e-6);
    assert!((r(0.5) - 0.5).abs() < 1e-6);
    assert!((r(0.75) - 0.25).abs() < 1e-6);
    assert!((r(1.0) - 0.0).abs() < 1e-6);
    assert!((r(-0.5) - 0.5).abs() < 1e-6);
    assert!((r(-0.25) - 0.25).abs() < 1e-6);
}

#[test]
fn spec_normalize_returns_unit_vector() {
    let r = run_unary(
        UnaryOp::Normalize,
        ValueType::Vector3,
        Value::Vector3(Vec3::new(2.0, 0.0, 0.0)),
    );
    assert!(approx_v3(r.as_vector3(), Vec3::X, 1.0e-6));
}

#[test]
fn spec_normalize_zero_matches_raw_mdl_normalize() {
    let r = run_unary(
        UnaryOp::Normalize,
        ValueType::Vector3,
        Value::Vector3(Vec3::ZERO),
    );
    assert!(r.as_vector3().is_nan());
}

#[test]
fn spec_magnitude_returns_vector_length() {
    let r = run_unary(
        UnaryOp::Length,
        ValueType::Vector3,
        Value::Vector3(Vec3::new(3.0, 4.0, 0.0)),
    );
    assert!(approx_f(r.as_float(), 5.0, 1.0e-6));
}

#[test]
fn spec_normalize_vector4_includes_w_component() {
    let r = run_unary(
        UnaryOp::Normalize,
        ValueType::Vector4,
        Value::Vector4(Vec4::new(2.0, 0.0, 0.0, 0.0)),
    );
    assert!((r.as_color4() - Vec4::new(1.0, 0.0, 0.0, 0.0)).length() < 1.0e-6);
    let r2 = run_unary(
        UnaryOp::Normalize,
        ValueType::Vector4,
        Value::Vector4(Vec4::new(0.0, 0.0, 0.0, 5.0)),
    );
    assert!((r2.as_color4() - Vec4::new(0.0, 0.0, 0.0, 1.0)).length() < 1.0e-6);
}

#[test]
fn spec_magnitude_vector4_includes_w_component() {
    let r = run_unary(
        UnaryOp::Length,
        ValueType::Vector4,
        Value::Vector4(Vec4::new(0.0, 0.0, 0.0, 5.0)),
    );
    assert!((r.as_float() - 5.0).abs() < 1.0e-6);
}

#[test]
fn spec_luminance_color4_preserves_alpha() {
    let r = run_unary(
        UnaryOp::Luminance,
        ValueType::Color4,
        Value::Color4(Vec4::new(1.0, 0.0, 0.0, 0.42)),
    );
    let v = r.as_color4();
    assert!((v.w - 0.42).abs() < 1e-6);
    assert!((v.x - 0.2722287).abs() < 1e-5);
}

#[test]
fn spec_rgbtohsv_color4_preserves_alpha() {
    let r = run_unary(
        UnaryOp::RgbToHsv,
        ValueType::Color4,
        Value::Color4(Vec4::new(1.0, 0.0, 0.0, 0.7)),
    );
    let v = r.as_color4();
    assert!((v.w - 0.7).abs() < 1e-6);
}

#[test]
fn spec_rgbtohsv_low_chroma_matches_mdl_thresholds() {
    let r = run_unary(
        UnaryOp::RgbToHsv,
        ValueType::Color3,
        Value::Color3(Vec3::new(0.001, 0.00100001, 0.001)),
    );
    let hsv = r.as_color3();
    assert!(approx_f(hsv.x, 1.0 / 3.0, 1.0e-3));
    assert!(hsv.y > 0.0);
}

#[test]
fn spec_hsvtorgb_color4_preserves_alpha() {
    let r = run_unary(
        UnaryOp::HsvToRgb,
        ValueType::Color4,
        Value::Color4(Vec4::new(0.0, 1.0, 1.0, 0.55)),
    );
    let v = r.as_color4();
    assert!((v.w - 0.55).abs() < 1e-6);
}

#[test]
fn spec_luminance_uses_acescg_coefficients() {
    let r = run_unary(
        UnaryOp::Luminance,
        ValueType::Color3,
        Value::Color3(Vec3::new(1.0, 0.0, 0.0)),
    );
    assert!(approx_f(r.as_color3().x, 0.2722287, 1.0e-4));
}

#[test]
fn spec_luminance_instruction_uses_custom_lumacoeffs() {
    let regs = run(
        vec![Instruction::LuminanceWithCoeffs {
            dst: 0,
            ty: ValueType::Color4,
            c: Operand::Const(0),
            lumacoeffs: Operand::Const(1),
        }],
        Vec::new(),
        vec![
            Value::Color4(Vec4::new(0.2, 0.4, 0.8, 0.6)),
            Value::Color3(Vec3::new(0.0, 1.0, 0.0)),
        ],
        1,
    );
    let v = regs[0].as_color4();
    assert!((v - Vec4::new(0.4, 0.4, 0.4, 0.6)).length() < 1.0e-6);
}

#[test]
fn spec_dotproduct_returns_scalar() {
    let r = run_dot(
        ValueType::Vector3,
        Value::Vector3(Vec3::new(1.0, 2.0, 3.0)),
        Value::Vector3(Vec3::new(4.0, 5.0, 6.0)),
    );
    assert!(approx_f(r.as_float(), 32.0, 1.0e-6));
}

#[test]
fn spec_dotproduct_vector4_includes_w_component() {
    let r = run_dot(
        ValueType::Vector4,
        Value::Vector4(Vec4::new(1.0, 0.0, 0.0, 2.0)),
        Value::Vector4(Vec4::new(0.0, 1.0, 0.0, 3.0)),
    );
    assert!((r.as_float() - 6.0).abs() < 1.0e-6);
}

#[test]
fn spec_crossproduct_orthogonal() {
    let r = run_cross(Value::Vector3(Vec3::X), Value::Vector3(Vec3::Y));
    assert!(approx_v3(r.as_vector3(), Vec3::Z, 1.0e-6));
}

#[test]
fn spec_distance_is_euclidean() {
    let r = run_distance(
        ValueType::Vector3,
        Value::Vector3(Vec3::ZERO),
        Value::Vector3(Vec3::new(3.0, 4.0, 0.0)),
    );
    assert!(approx_f(r.as_float(), 5.0, 1.0e-6));
}

#[test]
fn spec_distance_vector4_includes_w_component() {
    let r = run_distance(
        ValueType::Vector4,
        Value::Vector4(Vec4::ZERO),
        Value::Vector4(Vec4::new(0.0, 0.0, 0.0, 5.0)),
    );
    assert!((r.as_float() - 5.0).abs() < 1.0e-6);
}

#[test]
fn spec_reflect_against_z_normal() {
    let r = run_reflect(
        Value::Vector3(Vec3::new(1.0, 0.0, -1.0)),
        Value::Vector3(Vec3::Z),
    );
    assert!(approx_v3(r.as_vector3(), Vec3::new(1.0, 0.0, 1.0), 1.0e-6));
}

#[test]
fn spec_refract_uses_unclamped_dot_like_nodegraph() {
    let r = run_refract(
        Value::Vector3(Vec3::new(2.0, 0.0, 0.0)),
        Value::Vector3(Vec3::X),
        Value::Float(0.5),
    );
    let expected = Vec3::new(-(1.75f32).sqrt(), 0.0, 0.0);
    assert!(approx_v3(r.as_vector3(), expected, 1.0e-6));
}

#[test]
fn spec_refract_total_internal_reflection_returns_zero() {
    let r = run_refract(
        Value::Vector3(Vec3::new(1.0, 0.0, 0.0)),
        Value::Vector3(Vec3::Z),
        Value::Float(2.0),
    );
    assert!(approx_v3(r.as_vector3(), Vec3::ZERO, 1.0e-6));
}

#[test]
fn spec_rotate2d_matches_osl_mdl_cw_convention() {
    let r = run_rotate2d(Value::Vector2(Vec2::X), Value::Float(90.0));
    let v = r.as_vector2();
    assert!(v.abs_diff_eq(Vec2::new(0.0, -1.0), 1.0e-5));
}

#[test]
fn spec_rotate3d_about_z_90_degrees() {
    let r = run_rotate3d(
        Value::Vector3(Vec3::X),
        Value::Vector3(Vec3::Z),
        Value::Float(90.0),
    );
    let v = r.as_vector3();
    assert!(v.abs_diff_eq(Vec3::new(0.0, 1.0, 0.0), 1.0e-5));
}

#[test]
fn spec_rotate3d_uses_raw_axis_like_mdl() {
    let r = run_rotate3d(
        Value::Vector3(Vec3::X),
        Value::Vector3(Vec3::new(0.0, 0.0, 2.0)),
        Value::Float(90.0),
    );
    let v = r.as_vector3();
    assert!(v.abs_diff_eq(Vec3::new(0.0, 2.0, 0.0), 1.0e-5));
}

#[test]
fn spec_mix_lerps_bg_to_fg() {
    let r = run_mix(
        ValueType::Float,
        Value::Float(1.0),
        Value::Float(3.0),
        Value::Float(0.25),
    );
    assert!(approx_f(r.as_float(), 1.5, 1.0e-6));
}

#[test]
fn spec_mix_vector3_uses_component_mix() {
    let r = run_mix(
        ValueType::Vector3,
        Value::Vector3(Vec3::ZERO),
        Value::Vector3(Vec3::new(1.0, 2.0, 4.0)),
        Value::Vector3(Vec3::new(0.0, 0.5, 1.0)),
    );
    assert!(approx_v3(r.as_vector3(), Vec3::new(0.0, 1.0, 4.0), 1.0e-6));
}

#[test]
fn spec_mix_vector4_preserves_vector_type() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Combinator {
                    category: "mix".to_string(),
                },
                output_type: MtlxType::Vector4,
                inputs: vec![
                    FlatNodeInput {
                        name: "bg".to_string(),
                        ty: MtlxType::Vector4,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Vector4(Vec4::ZERO)),
                    },
                    FlatNodeInput {
                        name: "fg".to_string(),
                        ty: MtlxType::Vector4,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Vector4(Vec4::new(
                            1.0, 2.0, 4.0, 8.0,
                        ))),
                    },
                    FlatNodeInput {
                        name: "mix".to_string(),
                        ty: MtlxType::Vector4,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Vector4(Vec4::new(
                            0.0, 0.25, 0.5, 1.0,
                        ))),
                    },
                ],
            },
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "extract".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![
                    FlatNodeInput {
                        name: "in".to_string(),
                        ty: MtlxType::Vector4,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Node {
                            node: 0,
                            output: None,
                        },
                    },
                    FlatNodeInput {
                        name: "index".to_string(),
                        ty: MtlxType::Integer,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Integer(3)),
                    },
                ],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 2,
                        output: None,
                    },
                }],
            },
        ],
        root: 3,
        back_root: None,
        material_name: "mix_vector4".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("vector4 mix should compile as vector4");
    let le = eval_compiled_le(&compiled);
    assert!(approx_v3(
        le,
        Vec3::splat(8.0 / std::f32::consts::PI),
        1.0e-6
    ));
}

#[test]
fn spec_mix_color4_supports_float_and_per_channel_mix() {
    let bg = Value::Color4(Vec4::new(0.0, 0.2, 0.4, 0.6));
    let fg = Value::Color4(Vec4::new(1.0, 0.8, 0.6, 0.4));
    let uniform = run_mix(ValueType::Color4, bg, fg, Value::Float(0.25)).as_color4();
    let per_channel = run_mix(
        ValueType::Color4,
        bg,
        fg,
        Value::Color4(Vec4::new(0.0, 0.25, 0.5, 1.0)),
    )
    .as_color4();
    assert!((uniform - Vec4::new(0.25, 0.35, 0.45, 0.55)).length() < 1.0e-6);
    assert!((per_channel - Vec4::new(0.0, 0.35, 0.5, 0.4)).length() < 1.0e-6);
}

#[test]
fn spec_ramp_nodegraph_flattens_and_evaluates_linear_interval() {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let src = r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodegraph name="NG_my">
    <ramp name="r" type="color4">
      <input name="texcoord" type="vector2" value="0.25,0.0"/>
      <input name="interpolation" type="integer" value="0"/>
      <input name="color1" type="color4" value="0.0,0.0,0.0,1.0"/>
      <input name="color2" type="color4" value="1.0,1.0,1.0,1.0"/>
    </ramp>
    <extract name="red" type="float">
      <input name="in" type="color4" nodename="r"/>
      <input name="index" type="integer" value="0"/>
    </extract>
    <surface_unlit name="srf" type="surfaceshader">
      <input name="emission" type="float" nodename="red"/>
    </surface_unlit>
    <output name="out" type="surfaceshader" nodename="srf"/>
  </nodegraph>
  <surfacematerial name="MyMat" type="material">
    <input name="surfaceshader" type="surfaceshader" nodegraph="NG_my"/>
  </surfacematerial>
</materialx>"#;
            let lib = load_standard_library(&lib_root()).expect("library");
            let doc = parse_str(src, Path::new("inline.mtlx")).expect("parse");
            let graph = flatten_material(&lib, &doc, "MyMat").expect("flatten");
            assert!(!graph.nodes.iter().any(|node| {
                matches!(
                    &node.kind,
                    FlatNodeKind::Pattern { category } if category == "ramp" || category == "ramp_gradient"
                )
            }));
            let compiled = super::compile::compile(
                &graph,
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
            )
            .expect("ramp nodegraph should compile");
            let mut scratch = MtlxScratch::default();
            let handle = scratch.alloc_regs(compiled.num_registers as usize);
            super::runtime::run_instructions(&compiled, &dummy_sv(), &mut scratch, handle);
            let le = super::runtime::evaluate_le(&compiled, scratch.regs_slice(handle), &dummy_sv())
                .expect("surface_unlit should emit");
            assert!(approx_v3(
                le,
                Vec3::splat(0.25 / std::f32::consts::PI),
                1.0e-6
            ));
        })
        .expect("spawn")
        .join()
        .expect("ramp compile test panicked");
}

#[test]
fn spec_clamp_returns_input_in_range() {
    assert_eq!(
        run_clamp(
            ValueType::Float,
            Value::Float(0.3),
            Value::Float(0.0),
            Value::Float(1.0)
        )
        .as_float(),
        0.3
    );
    assert_eq!(
        run_clamp(
            ValueType::Float,
            Value::Float(-1.0),
            Value::Float(0.0),
            Value::Float(1.0)
        )
        .as_float(),
        0.0
    );
    assert_eq!(
        run_clamp(
            ValueType::Float,
            Value::Float(2.0),
            Value::Float(0.0),
            Value::Float(1.0)
        )
        .as_float(),
        1.0
    );
}

#[test]
fn spec_clamp_uses_bounds_order_like_mdl() {
    let r = run_clamp(
        ValueType::Float,
        Value::Float(0.5),
        Value::Float(1.0),
        Value::Float(0.0),
    );
    assert!(approx_f(r.as_float(), 0.0, 1.0e-6));
}

#[test]
fn spec_smoothstep_cubic_hermite() {
    let s = |t: f32| {
        run_smoothstep(
            ValueType::Float,
            Value::Float(t),
            Value::Float(0.0),
            Value::Float(1.0),
        )
        .as_float()
    };
    assert!((s(0.0) - 0.0).abs() < 1e-6);
    assert!((s(1.0) - 1.0).abs() < 1e-6);
    assert!((s(0.5) - 0.5).abs() < 1e-6);
}

#[test]
fn spec_compare_greater_selects_in_true_or_in_false() {
    let r = run_compare(
        CompareOp::Greater,
        Value::Float(2.0),
        Value::Float(1.0),
        Value::Float(99.0),
        Value::Float(-1.0),
    );
    assert_eq!(r.as_float(), 99.0);
    let r = run_compare(
        CompareOp::Greater,
        Value::Float(0.5),
        Value::Float(1.0),
        Value::Float(99.0),
        Value::Float(-1.0),
    );
    assert_eq!(r.as_float(), -1.0);
}

#[test]
fn spec_compare_greatereq_selects_in_true_on_equality() {
    let r = run_compare(
        CompareOp::GreaterEq,
        Value::Float(1.0),
        Value::Float(1.0),
        Value::Color3(Vec3::ONE),
        Value::Color3(Vec3::ZERO),
    );
    assert!(approx_v3(r.as_color3(), Vec3::ONE, 1.0e-6));
}

#[test]
fn spec_compare_equal_uses_exact_mdl_equality() {
    let r = run_compare(
        CompareOp::Equal,
        Value::Float(1.0),
        Value::Float(1.0 + 5.0e-7),
        Value::Float(99.0),
        Value::Float(-1.0),
    );
    assert_eq!(r.as_float(), -1.0);

    let r = run_compare_bool(
        CompareOp::Equal,
        Value::Float(1.0),
        Value::Float(1.0 + 5.0e-7),
    );
    assert!(!r.as_bool());
}

#[test]
fn spec_compare_bool_returns_bool() {
    let r = run_compare_bool(CompareOp::Greater, Value::Float(2.0), Value::Float(1.0));
    assert!(r.as_bool());
    let r = run_compare_bool(CompareOp::GreaterEq, Value::Float(1.0), Value::Float(1.0));
    assert!(r.as_bool());
    let r = run_compare_bool(CompareOp::Equal, Value::Float(1.0), Value::Float(1.0));
    assert!(r.as_bool());
}

#[test]
fn spec_switch_floor_and_clamp_selector() {
    let branches = [
        Value::Float(10.0),
        Value::Float(20.0),
        Value::Float(30.0),
        Value::Float(40.0),
        Value::Float(50.0),
        Value::Float(60.0),
        Value::Float(70.0),
        Value::Float(80.0),
        Value::Float(90.0),
        Value::Float(100.0),
    ];
    assert_eq!(
        run_switch(ValueType::Float, Value::Float(1.75), branches).as_float(),
        20.0
    );
    assert_eq!(
        run_switch(ValueType::Float, Value::Float(10.0), branches).as_float(),
        100.0
    );
    assert_eq!(
        run_switch(ValueType::Float, Value::Integer(-3), branches).as_float(),
        10.0
    );
}

#[test]
fn spec_ifelse_picks_true_or_false_branch() {
    let regs = run(
        vec![Instruction::IfElse {
            dst: 0,
            cond: Operand::Const(0),
            in_true: Operand::Const(1),
            in_false: Operand::Const(2),
        }],
        Vec::new(),
        vec![Value::Bool(true), Value::Float(7.0), Value::Float(9.0)],
        1,
    );
    assert_eq!(regs[0].as_float(), 7.0);
}

#[test]
fn spec_logical_and_or_xor() {
    assert!(run_logical(LogicalOp::And, Value::Bool(true), Value::Bool(true)).as_bool());
    assert!(!run_logical(LogicalOp::And, Value::Bool(true), Value::Bool(false)).as_bool());
    assert!(run_logical(LogicalOp::Or, Value::Bool(false), Value::Bool(true)).as_bool());
    assert!(run_logical(LogicalOp::Xor, Value::Bool(true), Value::Bool(false)).as_bool());
    assert!(!run_logical(LogicalOp::Xor, Value::Bool(true), Value::Bool(true)).as_bool());
    assert!(run_logical(LogicalOp::Not, Value::Bool(false), Value::Bool(false)).as_bool());
}

#[test]
fn spec_convert_float_to_color3_broadcasts() {
    let r = run_convert(ValueType::Float, ValueType::Color3, Value::Float(0.5));
    assert!(approx_v3(r.as_color3(), Vec3::splat(0.5), 1.0e-6));
}

#[test]
fn spec_convert_boolean_integer_scalar_rules() {
    let b_to_f = run_convert(ValueType::Boolean, ValueType::Float, Value::Bool(true));
    let b_to_i = run_convert(ValueType::Boolean, ValueType::Integer, Value::Bool(true));
    let i_to_b0 = run_convert(ValueType::Integer, ValueType::Boolean, Value::Integer(0));
    let i_to_b1 = run_convert(ValueType::Integer, ValueType::Boolean, Value::Integer(-3));
    assert_eq!(b_to_f.as_float(), 1.0);
    assert_eq!(b_to_i.as_integer(), 1);
    assert!(!i_to_b0.as_bool());
    assert!(i_to_b1.as_bool());
}

#[test]
fn spec_convert_runtime_float_to_integer_truncates_like_mdl_constructor() {
    let pos = run_convert(ValueType::Float, ValueType::Integer, Value::Float(2.9));
    let neg = run_convert(ValueType::Float, ValueType::Integer, Value::Float(-2.9));
    assert_eq!(pos.as_integer(), 2);
    assert_eq!(neg.as_integer(), -2);
}

#[test]
fn spec_convert_color3_to_color4_adds_alpha_one() {
    let r = run_convert(
        ValueType::Color3,
        ValueType::Color4,
        Value::Color3(Vec3::new(0.2, 0.4, 0.6)),
    );
    let v = r.as_color4();
    assert!((v - Vec4::new(0.2, 0.4, 0.6, 1.0)).length() < 1.0e-6);
}

#[test]
fn spec_convert_vector3_to_vector4_adds_w_one() {
    let r = run_convert(
        ValueType::Vector3,
        ValueType::Vector4,
        Value::Vector3(Vec3::new(0.2, 0.4, 0.6)),
    );
    let v = r.as_color4();
    assert!((v - Vec4::new(0.2, 0.4, 0.6, 1.0)).length() < 1.0e-6);
}

#[test]
fn spec_convert_color4_vector4_preserves_fourth_channel() {
    let color = run_convert(
        ValueType::Vector4,
        ValueType::Color4,
        Value::Vector4(Vec4::new(0.2, 0.4, 0.6, 0.8)),
    );
    let vector = run_convert(
        ValueType::Color4,
        ValueType::Vector4,
        Value::Color4(Vec4::new(0.1, 0.3, 0.5, 0.7)),
    );
    assert!((color.as_color4() - Vec4::new(0.2, 0.4, 0.6, 0.8)).length() < 1.0e-6);
    assert!((vector.as_color4() - Vec4::new(0.1, 0.3, 0.5, 0.7)).length() < 1.0e-6);
}

#[test]
fn spec_combine2_vector2_from_floats() {
    let r = run_combine(
        CombineKind::Vector2FromFloats,
        &[Value::Float(0.25), Value::Float(0.75)],
    );
    assert!(r.as_vector2().abs_diff_eq(Vec2::new(0.25, 0.75), 1.0e-6));
}

#[test]
fn spec_combine3_color3_from_floats() {
    let r = run_combine(
        CombineKind::Color3FromFloats,
        &[Value::Float(0.1), Value::Float(0.2), Value::Float(0.3)],
    );
    assert!(approx_v3(r.as_color3(), Vec3::new(0.1, 0.2, 0.3), 1.0e-6));
}

#[test]
fn spec_combine4_color4_from_color3_and_float() {
    let r = run_combine(
        CombineKind::Color4FromColor3Float,
        &[Value::Color3(Vec3::new(0.1, 0.2, 0.3)), Value::Float(0.5)],
    );
    let v = r.as_color4();
    assert!((v - Vec4::new(0.1, 0.2, 0.3, 0.5)).length() < 1.0e-6);
}

#[test]
fn spec_combine_remaining_overloads_copy_channels() {
    let v3 = run_combine(
        CombineKind::Vector3FromFloats,
        &[Value::Float(0.1), Value::Float(0.2), Value::Float(0.3)],
    );
    let c4 = run_combine(
        CombineKind::Color4FromFloats,
        &[
            Value::Float(0.1),
            Value::Float(0.2),
            Value::Float(0.3),
            Value::Float(0.4),
        ],
    );
    let v4 = run_combine(
        CombineKind::Vector4FromFloats,
        &[
            Value::Float(0.5),
            Value::Float(0.6),
            Value::Float(0.7),
            Value::Float(0.8),
        ],
    );
    let v4_vf = run_combine(
        CombineKind::Vector4FromVector3Float,
        &[Value::Vector3(Vec3::new(1.0, 2.0, 3.0)), Value::Float(4.0)],
    );
    let v4_vv = run_combine(
        CombineKind::Vector4FromVector2Vector2,
        &[
            Value::Vector2(Vec2::new(1.0, 2.0)),
            Value::Vector2(Vec2::new(3.0, 4.0)),
        ],
    );

    assert!(approx_v3(v3.as_vector3(), Vec3::new(0.1, 0.2, 0.3), 1.0e-6));
    assert!((c4.as_color4() - Vec4::new(0.1, 0.2, 0.3, 0.4)).length() < 1.0e-6);
    assert!((v4.as_color4() - Vec4::new(0.5, 0.6, 0.7, 0.8)).length() < 1.0e-6);
    assert!((v4_vf.as_color4() - Vec4::new(1.0, 2.0, 3.0, 4.0)).length() < 1.0e-6);
    assert!((v4_vv.as_color4() - Vec4::new(1.0, 2.0, 3.0, 4.0)).length() < 1.0e-6);
}

#[test]
fn spec_combine2_overload_uses_declared_default_input_type() {
    let compile_kind = |out_type: MtlxType, in1_type: MtlxType, in2_type: MtlxType| {
        let graph = FlatGraph {
            nodes: vec![
                FlatNode {
                    kind: FlatNodeKind::Pattern {
                        category: "combine2".to_string(),
                    },
                    output_type: out_type.clone(),
                    inputs: vec![
                        FlatNodeInput {
                            name: "in1".to_string(),
                            ty: in1_type,
                            colorspace: None,
                            unit: None,
                            unittype: None,
                            binding: FlatInput::Empty,
                        },
                        FlatNodeInput {
                            name: "in2".to_string(),
                            ty: in2_type,
                            colorspace: None,
                            unit: None,
                            unittype: None,
                            binding: FlatInput::Empty,
                        },
                    ],
                },
                FlatNode {
                    kind: FlatNodeKind::Pattern {
                        category: "extract".to_string(),
                    },
                    output_type: MtlxType::Float,
                    inputs: vec![
                        FlatNodeInput {
                            name: "in".to_string(),
                            ty: out_type,
                            colorspace: None,
                            unit: None,
                            unittype: None,
                            binding: FlatInput::Node {
                                node: 0,
                                output: None,
                            },
                        },
                        FlatNodeInput {
                            name: "index".to_string(),
                            ty: MtlxType::Integer,
                            colorspace: None,
                            unit: None,
                            unittype: None,
                            binding: FlatInput::Value(MtlxValue::Integer(0)),
                        },
                    ],
                },
                FlatNode {
                    kind: FlatNodeKind::SurfaceUnlit,
                    output_type: MtlxType::Surfaceshader,
                    inputs: vec![FlatNodeInput {
                        name: "emission".to_string(),
                        ty: MtlxType::Float,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Node {
                            node: 1,
                            output: None,
                        },
                    }],
                },
                FlatNode {
                    kind: FlatNodeKind::SurfaceMaterial,
                    output_type: MtlxType::Material,
                    inputs: vec![FlatNodeInput {
                        name: "surfaceshader".to_string(),
                        ty: MtlxType::Surfaceshader,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Node {
                            node: 2,
                            output: None,
                        },
                    }],
                },
            ],
            root: 3,
            back_root: None,
            material_name: "combine2_declared_defaults".to_string(),
        };
        super::compile::compile(
            &graph,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        )
        .expect("combine2 declared default inputs should compile");
    };

    compile_kind(MtlxType::Color4, MtlxType::Color3, MtlxType::Float);
    compile_kind(MtlxType::Vector4, MtlxType::Vector3, MtlxType::Float);
    compile_kind(MtlxType::Vector4, MtlxType::Vector2, MtlxType::Vector2);
}

#[test]
fn spec_extract_returns_indexed_channel() {
    let v = Value::Color3(Vec3::new(0.1, 0.2, 0.3));
    assert_eq!(run_extract(ValueType::Color3, v, 0).as_float(), 0.1);
    assert_eq!(run_extract(ValueType::Color3, v, 1).as_float(), 0.2);
    assert_eq!(run_extract(ValueType::Color3, v, 2).as_float(), 0.3);
    assert_eq!(run_extract(ValueType::Color3, v, -1).as_float(), 0.3);
    assert_eq!(run_extract(ValueType::Color3, v, 99).as_float(), 0.3);
    let v2 = Value::Vector2(Vec2::new(0.4, 0.5));
    assert_eq!(run_extract(ValueType::Vector2, v2, -1).as_float(), 0.5);
    let c4 = Value::Color4(Vec4::new(0.1, 0.2, 0.3, 0.4));
    assert_eq!(run_extract(ValueType::Color4, c4, 99).as_float(), 0.4);
}

#[test]
fn spec_extractrowvector_reads_matrix_rows() {
    let m3 = glam::Mat3::from_cols(
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(4.0, 5.0, 6.0),
        Vec3::new(7.0, 8.0, 9.0),
    );
    let regs = run(
        vec![
            Instruction::LoadMat3Const { dst: 0, value: m3 },
            Instruction::ExtractRowVector {
                dst: 1,
                dim4: false,
                src: Operand::Reg(0),
                index: 1,
            },
        ],
        Vec::new(),
        Vec::new(),
        2,
    );
    assert!(approx_v3(
        regs[1].as_vector3(),
        Vec3::new(2.0, 5.0, 8.0),
        1.0e-6
    ));

    let m4 = glam::Mat4::from_cols(
        Vec4::new(1.0, 2.0, 3.0, 4.0),
        Vec4::new(5.0, 6.0, 7.0, 8.0),
        Vec4::new(9.0, 10.0, 11.0, 12.0),
        Vec4::new(13.0, 14.0, 15.0, 16.0),
    );
    let regs = run(
        vec![
            Instruction::LoadMat4Const { dst: 0, value: m4 },
            Instruction::ExtractRowVector {
                dst: 1,
                dim4: true,
                src: Operand::Reg(0),
                index: 2,
            },
        ],
        Vec::new(),
        Vec::new(),
        2,
    );
    assert!(
        regs[1]
            .as_color4()
            .abs_diff_eq(Vec4::new(3.0, 7.0, 11.0, 15.0), 1.0e-6)
    );
}

#[test]
fn spec_extractrowvector_static_bad_index_errors() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "extractrowvector".to_string(),
                },
                output_type: MtlxType::Vector3,
                inputs: vec![
                    FlatNodeInput {
                        name: "in".to_string(),
                        ty: MtlxType::Matrix33,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Matrix33(glam::Mat3::IDENTITY)),
                    },
                    FlatNodeInput {
                        name: "index".to_string(),
                        ty: MtlxType::Integer,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Integer(3)),
                    },
                ],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission_color".to_string(),
                    ty: MtlxType::Color3,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: "extractrow_bad_index".to_string(),
    };
    let err = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect_err("bad static extractrowvector index must be rejected");
    assert!(err.to_string().contains("out of range"));
}

#[test]
fn spec_extractrowvector_vector4_compiles_as_vector4() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "extractrowvector".to_string(),
                },
                output_type: MtlxType::Vector4,
                inputs: vec![
                    FlatNodeInput {
                        name: "in".to_string(),
                        ty: MtlxType::Matrix44,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Matrix44(glam::Mat4::IDENTITY)),
                    },
                    FlatNodeInput {
                        name: "index".to_string(),
                        ty: MtlxType::Integer,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Integer(3)),
                    },
                ],
            },
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "extract".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![
                    FlatNodeInput {
                        name: "in".to_string(),
                        ty: MtlxType::Vector4,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Node {
                            node: 0,
                            output: None,
                        },
                    },
                    FlatNodeInput {
                        name: "index".to_string(),
                        ty: MtlxType::Integer,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Integer(3)),
                    },
                ],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 2,
                        output: None,
                    },
                }],
            },
        ],
        root: 3,
        back_root: None,
        material_name: "extractrow_vector4".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("vector4 extractrowvector should compile");
    assert!(
        compiled
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instruction::ExtractRowVector { dim4: true, .. }))
    );
}

#[test]
fn spec_separate4_vector4_outw_extracts_w_channel() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "separate4".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![FlatNodeInput {
                    name: "in".to_string(),
                    ty: MtlxType::Vector4,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Value(MtlxValue::Vector4(Vec4::new(1.0, 2.0, 3.0, 4.0))),
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: Some("outw".to_string()),
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: "separate4_vector4".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("separate4 vector4 should compile");
    let mut scratch = MtlxScratch::default();
    let handle = scratch.alloc_regs(compiled.num_registers as usize);
    super::runtime::run_instructions(&compiled, &dummy_sv(), &mut scratch, handle);
    let le = super::runtime::evaluate_le(&compiled, scratch.regs_slice(handle), &dummy_sv())
        .expect("surface_unlit should emit");
    assert!(approx_v3(
        le,
        Vec3::splat(4.0 / std::f32::consts::PI),
        1.0e-6
    ));
}

#[test]
fn spec_separate_output_names_are_type_specific() {
    let compile_separate = |category: &str, input_ty: MtlxType, input: MtlxValue, output: &str| {
        let graph = FlatGraph {
            nodes: vec![
                FlatNode {
                    kind: FlatNodeKind::Pattern {
                        category: category.to_string(),
                    },
                    output_type: MtlxType::Float,
                    inputs: vec![FlatNodeInput {
                        name: "in".to_string(),
                        ty: input_ty,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(input),
                    }],
                },
                FlatNode {
                    kind: FlatNodeKind::SurfaceUnlit,
                    output_type: MtlxType::Surfaceshader,
                    inputs: vec![FlatNodeInput {
                        name: "emission".to_string(),
                        ty: MtlxType::Float,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Node {
                            node: 0,
                            output: Some(output.to_string()),
                        },
                    }],
                },
                FlatNode {
                    kind: FlatNodeKind::SurfaceMaterial,
                    output_type: MtlxType::Material,
                    inputs: vec![FlatNodeInput {
                        name: "surfaceshader".to_string(),
                        ty: MtlxType::Surfaceshader,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Node {
                            node: 1,
                            output: None,
                        },
                    }],
                },
            ],
            root: 2,
            back_root: None,
            material_name: format!("{category}_{output}"),
        };
        super::compile::compile(
            &graph,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        )
    };

    let err = compile_separate(
        "separate3",
        MtlxType::Vector3,
        MtlxValue::Vector3(Vec3::ONE),
        "outb",
    )
    .expect_err("vector3 separate3 must not expose color output names");
    assert!(err.to_string().contains("separate3 output `outb`"));

    let err = compile_separate(
        "separate4",
        MtlxType::Color4,
        MtlxValue::Color4(Vec4::ONE),
        "outw",
    )
    .expect_err("color4 separate4 must not expose vector output names");
    assert!(err.to_string().contains("separate4 output `outw`"));
}

#[test]
fn spec_blend_plus_adds_fg_scaled_by_mix() {
    let r = run_blend(
        BlendOp::Plus,
        ValueType::Color3,
        Value::Color3(Vec3::splat(0.1)),
        Value::Color3(Vec3::splat(0.4)),
        Value::Float(1.0),
    );
    assert!(approx_v3(r.as_color3(), Vec3::splat(0.5), 1.0e-6));
}

#[test]
fn spec_blend_minus_subtracts_fg() {
    let r = run_blend(
        BlendOp::Minus,
        ValueType::Color3,
        Value::Color3(Vec3::splat(0.5)),
        Value::Color3(Vec3::splat(0.2)),
        Value::Float(1.0),
    );
    assert!(approx_v3(r.as_color3(), Vec3::splat(0.3), 1.0e-6));
}

#[test]
fn spec_blend_remaining_ops_match_standard_formulas() {
    let difference = run_blend(
        BlendOp::Difference,
        ValueType::Float,
        Value::Float(0.25),
        Value::Float(0.75),
        Value::Float(1.0),
    );
    let burn = run_blend(
        BlendOp::Burn,
        ValueType::Float,
        Value::Float(0.25),
        Value::Float(0.5),
        Value::Float(1.0),
    );
    let dodge = run_blend(
        BlendOp::Dodge,
        ValueType::Float,
        Value::Float(0.25),
        Value::Float(0.5),
        Value::Float(1.0),
    );
    let screen = run_blend(
        BlendOp::Screen,
        ValueType::Float,
        Value::Float(0.25),
        Value::Float(0.5),
        Value::Float(1.0),
    );
    let overlay_low = run_blend(
        BlendOp::Overlay,
        ValueType::Float,
        Value::Float(0.25),
        Value::Float(0.5),
        Value::Float(1.0),
    );
    let overlay_high = run_blend(
        BlendOp::Overlay,
        ValueType::Float,
        Value::Float(0.75),
        Value::Float(0.5),
        Value::Float(1.0),
    );
    assert!(approx_f(difference.as_float(), 0.5, 1.0e-6));
    assert!(approx_f(burn.as_float(), -0.5, 1.0e-6));
    assert!(approx_f(dodge.as_float(), 0.5, 1.0e-6));
    assert!(approx_f(screen.as_float(), 0.625, 1.0e-6));
    assert!(approx_f(overlay_low.as_float(), 0.25, 1.0e-6));
    assert!(approx_f(overlay_high.as_float(), 0.75, 1.0e-6));
}

#[test]
fn spec_blend_burn_dodge_mdl_epsilon_branches_return_zero() {
    let burn = run_blend(
        BlendOp::Burn,
        ValueType::Float,
        Value::Float(0.25),
        Value::Float(0.0),
        Value::Float(1.0),
    );
    let dodge = run_blend(
        BlendOp::Dodge,
        ValueType::Float,
        Value::Float(0.25),
        Value::Float(1.0),
        Value::Float(1.0),
    );
    assert!(approx_f(burn.as_float(), 0.0, 1.0e-6));
    assert!(approx_f(dodge.as_float(), 0.0, 1.0e-6));
}

#[test]
fn spec_blend_vector_output_is_rejected() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "plus".to_string(),
                },
                output_type: MtlxType::Vector3,
                inputs: vec![
                    FlatNodeInput {
                        name: "bg".to_string(),
                        ty: MtlxType::Vector3,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Vector3(Vec3::ONE)),
                    },
                    FlatNodeInput {
                        name: "fg".to_string(),
                        ty: MtlxType::Vector3,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Vector3(Vec3::ONE)),
                    },
                ],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission_color".to_string(),
                    ty: MtlxType::Color3,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: "blend_vector_rejected".to_string(),
    };
    let err = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect_err("vector blend output is not a MaterialX blend overload");
    assert!(err.to_string().contains("blend node `plus`"));
}

#[test]
fn spec_merge_over_porter_duff() {
    let bg = Value::Color4(Vec4::new(0.5, 0.5, 0.5, 1.0));
    let fg = Value::Color4(Vec4::new(1.0, 0.0, 0.0, 1.0));
    let r = run_merge(MergeOp::Over, bg, fg, Value::Float(1.0));
    let v = r.as_color4();
    assert!((v - Vec4::new(1.0, 0.0, 0.0, 1.0)).length() < 1.0e-6);
}

#[test]
fn spec_merge_remaining_ops_match_mdl_formulas() {
    let bg = Value::Color4(Vec4::new(0.2, 0.4, 0.6, 0.25));
    let fg = Value::Color4(Vec4::new(0.8, 0.5, 0.1, 0.5));
    let disjoint = run_merge(MergeOp::Disjointover, bg, fg, Value::Float(1.0)).as_color4();
    let inside = run_merge(MergeOp::In, bg, fg, Value::Float(1.0)).as_color4();
    let mask = run_merge(MergeOp::Mask, bg, fg, Value::Float(1.0)).as_color4();
    let matte = run_merge(MergeOp::Matte, bg, fg, Value::Float(1.0)).as_color4();
    let out = run_merge(MergeOp::Out, bg, fg, Value::Float(1.0)).as_color4();
    assert!((disjoint - Vec4::new(1.0, 0.9, 0.7, 0.75)).length() < 1.0e-6);
    assert!((inside - Vec4::new(0.2, 0.125, 0.025, 0.125)).length() < 1.0e-6);
    assert!((mask - Vec4::new(0.1, 0.2, 0.3, 0.125)).length() < 1.0e-6);
    assert!((matte - Vec4::new(0.5, 0.45, 0.35, 0.625)).length() < 1.0e-6);
    assert!((out - Vec4::new(0.6, 0.375, 0.075, 0.375)).length() < 1.0e-6);
}

#[test]
fn spec_merge_and_mask_vector_outputs_are_rejected() {
    let graph_for = |category: &str, output_type: MtlxType| FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: category.to_string(),
                },
                output_type,
                inputs: vec![],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission_color".to_string(),
                    ty: MtlxType::Color3,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: format!("{category}_vector_rejected"),
    };

    let err = super::compile::compile(
        &graph_for("over", MtlxType::Vector4),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect_err("merge vector output is not a MaterialX overload");
    assert!(err.to_string().contains("merge node `over`"));

    let err = super::compile::compile(
        &graph_for("inside", MtlxType::Vector3),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect_err("mask vector output is not a MaterialX overload");
    assert!(err.to_string().contains("mask node `inside`"));
}

#[test]
fn spec_mask_inside_scales_by_mask() {
    let r = run_mask(
        MaskOp::Inside,
        ValueType::Color3,
        Value::Color3(Vec3::ONE),
        Value::Float(0.4),
    );
    assert!(approx_v3(r.as_color3(), Vec3::splat(0.4), 1.0e-6));
}

#[test]
fn spec_mask_outside_uses_one_minus_mask() {
    let r = run_mask(
        MaskOp::Outside,
        ValueType::Color3,
        Value::Color3(Vec3::ONE),
        Value::Float(0.3),
    );
    assert!(approx_v3(r.as_color3(), Vec3::splat(0.7), 1.0e-6));
}

#[test]
fn spec_mask_color4_scales_alpha_channel_too() {
    let inside = run_mask(
        MaskOp::Inside,
        ValueType::Color4,
        Value::Color4(Vec4::new(0.2, 0.4, 0.6, 0.8)),
        Value::Float(0.5),
    );
    let outside = run_mask(
        MaskOp::Outside,
        ValueType::Color4,
        Value::Color4(Vec4::new(0.2, 0.4, 0.6, 0.8)),
        Value::Float(0.25),
    );
    assert!((inside.as_color4() - Vec4::new(0.1, 0.2, 0.3, 0.4)).length() < 1.0e-6);
    assert!((outside.as_color4() - Vec4::new(0.15, 0.3, 0.45, 0.6)).length() < 1.0e-6);
}

#[test]
fn spec_premult_multiplies_rgb_by_alpha() {
    let r = run_premult(Value::Color4(Vec4::new(1.0, 0.5, 0.25, 0.5)));
    let v = r.as_color4();
    assert!((v - Vec4::new(0.5, 0.25, 0.125, 0.5)).length() < 1.0e-6);
}

#[test]
fn spec_unpremult_divides_rgb_by_alpha() {
    let r = run_unpremult(Value::Color4(Vec4::new(0.5, 0.25, 0.125, 0.5)));
    let v = r.as_color4();
    assert!((v - Vec4::new(1.0, 0.5, 0.25, 0.5)).length() < 1.0e-6);
}

#[test]
fn spec_unpremult_zero_alpha_is_safe() {
    let r = run_unpremult(Value::Color4(Vec4::new(0.0, 0.0, 0.0, 0.0)));
    let v = r.as_color4();
    assert!(v.w == 0.0);
}

#[test]
fn spec_unpremult_zero_alpha_passes_rgb_through() {
    let r = run_unpremult(Value::Color4(Vec4::new(0.2, 0.4, 0.6, 0.0)));
    let v = r.as_color4();
    assert!((v - Vec4::new(0.2, 0.4, 0.6, 0.0)).length() < 1.0e-6);
}

#[test]
fn spec_unpremult_tiny_nonzero_alpha_divides() {
    let r = run_unpremult(Value::Color4(Vec4::new(2.0e-7, 4.0e-7, 6.0e-7, 1.0e-7)));
    let v = r.as_color4();
    assert!((v - Vec4::new(2.0, 4.0, 6.0, 1.0e-7)).length() < 1.0e-5);
}

#[test]
fn spec_premult_unpremult_default_input_alpha_is_one() {
    for category in ["premult", "unpremult"] {
        let graph = FlatGraph {
            nodes: vec![
                FlatNode {
                    kind: FlatNodeKind::Pattern {
                        category: category.to_string(),
                    },
                    output_type: MtlxType::Color4,
                    inputs: vec![],
                },
                FlatNode {
                    kind: FlatNodeKind::Pattern {
                        category: "extract".to_string(),
                    },
                    output_type: MtlxType::Float,
                    inputs: vec![
                        FlatNodeInput {
                            name: "in".to_string(),
                            ty: MtlxType::Color4,
                            colorspace: None,
                            unit: None,
                            unittype: None,
                            binding: FlatInput::Node {
                                node: 0,
                                output: None,
                            },
                        },
                        FlatNodeInput {
                            name: "index".to_string(),
                            ty: MtlxType::Integer,
                            colorspace: None,
                            unit: None,
                            unittype: None,
                            binding: FlatInput::Value(MtlxValue::Integer(3)),
                        },
                    ],
                },
                FlatNode {
                    kind: FlatNodeKind::SurfaceUnlit,
                    output_type: MtlxType::Surfaceshader,
                    inputs: vec![FlatNodeInput {
                        name: "emission".to_string(),
                        ty: MtlxType::Float,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Node {
                            node: 1,
                            output: None,
                        },
                    }],
                },
                FlatNode {
                    kind: FlatNodeKind::SurfaceMaterial,
                    output_type: MtlxType::Material,
                    inputs: vec![FlatNodeInput {
                        name: "surfaceshader".to_string(),
                        ty: MtlxType::Surfaceshader,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Node {
                            node: 2,
                            output: None,
                        },
                    }],
                },
            ],
            root: 3,
            back_root: None,
            material_name: format!("{category}_default"),
        };
        let compiled = super::compile::compile(
            &graph,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        )
        .expect("premult/unpremult default should compile");
        let mut scratch = MtlxScratch::default();
        let handle = scratch.alloc_regs(compiled.num_registers as usize);
        super::runtime::run_instructions(&compiled, &dummy_sv(), &mut scratch, handle);
        let le = super::runtime::evaluate_le(&compiled, scratch.regs_slice(handle), &dummy_sv())
            .expect("surface_unlit should emit");
        assert!(approx_v3(
            le,
            Vec3::splat(1.0 / std::f32::consts::PI),
            1.0e-6
        ));
    }
}

#[test]
fn spec_contrast_amplifies_around_pivot() {
    let r = run_contrast(
        ValueType::Float,
        Value::Float(0.6),
        Value::Float(2.0),
        Value::Float(0.5),
    );
    assert!(approx_f(r.as_float(), 0.7, 1.0e-6));
}

#[test]
fn spec_contrast_color4_float_amount_and_pivot_broadcast() {
    let r = run_contrast(
        ValueType::Color4,
        Value::Color4(Vec4::new(0.25, 0.5, 0.75, 1.0)),
        Value::Float(2.0),
        Value::Float(0.5),
    );
    let v = r.as_color4();
    assert!((v - Vec4::new(0.0, 0.5, 1.0, 1.5)).length() < 1.0e-6);
}

#[test]
fn spec_range_no_clamp() {
    let r = run_range(
        ValueType::Float,
        false,
        Value::Float(0.5),
        Value::Float(0.0),
        Value::Float(1.0),
        Value::Float(1.0),
        Value::Float(0.0),
        Value::Float(2.0),
    );
    assert!(approx_f(r.as_float(), 1.0, 1.0e-6));
}

#[test]
fn spec_range_doclamp_clamps_input_to_inlow_inhigh() {
    let r = run_range(
        ValueType::Float,
        true,
        Value::Float(2.0),
        Value::Float(0.0),
        Value::Float(1.0),
        Value::Float(1.0),
        Value::Float(0.0),
        Value::Float(2.0),
    );
    assert!(approx_f(r.as_float(), 2.0, 1.0e-6));
}

#[test]
fn spec_range_allows_reversed_input_range() {
    let r = run_range(
        ValueType::Float,
        false,
        Value::Float(0.25),
        Value::Float(1.0),
        Value::Float(0.0),
        Value::Float(1.0),
        Value::Float(0.0),
        Value::Float(1.0),
    );
    assert!(approx_f(r.as_float(), 0.75, 1.0e-6));
}

#[test]
fn spec_range_doclamp_uses_output_bounds_order() {
    let r = run_range(
        ValueType::Float,
        true,
        Value::Float(0.5),
        Value::Float(0.0),
        Value::Float(1.0),
        Value::Float(1.0),
        Value::Float(1.0),
        Value::Float(0.0),
    );
    assert!(approx_f(r.as_float(), 0.0, 1.0e-6));
}

#[test]
fn spec_range_doclamp_rejects_dynamic_boolean() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "and".to_string(),
                },
                output_type: MtlxType::Boolean,
                inputs: vec![
                    FlatNodeInput {
                        name: "in1".to_string(),
                        ty: MtlxType::Boolean,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Boolean(true)),
                    },
                    FlatNodeInput {
                        name: "in2".to_string(),
                        ty: MtlxType::Boolean,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Boolean(true)),
                    },
                ],
            },
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "range".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![FlatNodeInput {
                    name: "doclamp".to_string(),
                    ty: MtlxType::Boolean,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 2,
                        output: None,
                    },
                }],
            },
        ],
        root: 3,
        back_root: None,
        material_name: "range_dynamic_doclamp".to_string(),
    };
    let err = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect_err("dynamic range.doclamp should not silently become false");
    assert!(
        err.to_string()
            .contains("range.doclamp must be a static boolean value")
    );
}

#[test]
fn spec_static_boolean_rejects_numeric_string_literal() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "range".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![FlatNodeInput {
                    name: "doclamp".to_string(),
                    ty: MtlxType::Boolean,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::String("1".to_string()),
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: "range_numeric_doclamp".to_string(),
    };
    let err = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect_err("numeric string boolean must be rejected");
    assert!(
        err.to_string()
            .contains("range.doclamp `1`: must be a static boolean value")
    );
}

#[test]
fn spec_remap_linear() {
    let r = run_remap(
        ValueType::Float,
        Value::Float(0.5),
        Value::Float(0.0),
        Value::Float(1.0),
        Value::Float(-1.0),
        Value::Float(1.0),
    );
    assert!(approx_f(r.as_float(), 0.0, 1.0e-6));
}

#[test]
fn spec_hsvadjust_default_amount_is_noop() {
    let input = Vec3::new(0.2, 0.4, 0.6);
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "hsvadjust".to_string(),
                },
                output_type: MtlxType::Color3,
                inputs: vec![FlatNodeInput {
                    name: "in".to_string(),
                    ty: MtlxType::Color3,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Value(MtlxValue::Color3(input)),
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission_color".to_string(),
                    ty: MtlxType::Color3,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: "hsvadjust_default".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("hsvadjust default graph should compile");

    let mut scratch = MtlxScratch::default();
    let handle = scratch.alloc_regs(compiled.num_registers as usize);
    super::runtime::run_instructions(&compiled, &dummy_sv(), &mut scratch, handle);
    let le = super::runtime::evaluate_le(&compiled, scratch.regs_slice(handle), &dummy_sv())
        .expect("surface_unlit should emit");
    assert!(approx_v3(le, input * (1.0 / std::f32::consts::PI), 1.0e-6));
}

#[test]
fn spec_hsvadjust_does_not_clamp_saturation_or_value() {
    let regs = run(
        vec![Instruction::HsvAdjust {
            dst: 0,
            ty: ValueType::Color3,
            c: Operand::Const(0),
            amount: Operand::Const(1),
        }],
        Vec::new(),
        vec![
            Value::Color3(Vec3::new(1.0, 0.0, 0.0)),
            Value::Vector3(Vec3::new(0.0, 2.0, -1.0)),
        ],
        1,
    );
    assert!(approx_v3(
        regs[0].as_color3(),
        Vec3::new(-1.0, 1.0, 1.0),
        1.0e-6
    ));
}

#[test]
fn spec_latlongimage_uses_viewdir_rotation_nodegraph() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "latlongimage".to_string(),
                },
                output_type: MtlxType::Color3,
                inputs: vec![
                    FlatNodeInput {
                        name: "file".to_string(),
                        ty: MtlxType::Filename,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Filename(String::new())),
                    },
                    FlatNodeInput {
                        name: "viewdir".to_string(),
                        ty: MtlxType::Vector3,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Vector3(Vec3::X)),
                    },
                    FlatNodeInput {
                        name: "rotation".to_string(),
                        ty: MtlxType::Float,
                        colorspace: None,
                        unit: None,
                        unittype: Some("angle".to_string()),
                        binding: FlatInput::Value(MtlxValue::Float(90.0)),
                    },
                ],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission_color".to_string(),
                    ty: MtlxType::Color3,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: "latlongimage".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("latlongimage graph should compile");

    let uv_reg = compiled.instructions.iter().find_map(|instr| match instr {
        Instruction::LatlongUv { dst, .. } => Some(*dst),
        _ => None,
    });
    let image_instr = compiled.instructions.iter().find_map(|instr| match instr {
        Instruction::Image {
            kind,
            uaddress,
            vaddress,
            filter,
            texcoord,
            ..
        } => Some((*kind, *uaddress, *vaddress, *filter, *texcoord)),
        _ => None,
    });
    let Some((kind, uaddress, vaddress, filter, texcoord)) = image_instr else {
        panic!("latlongimage must emit image instruction");
    };
    assert_eq!(kind, ImageKind::LatLongImage);
    assert_eq!(uaddress, AddressMode::Periodic);
    assert_eq!(vaddress, AddressMode::Mirror);
    assert_eq!(filter, FilterType::Linear);
    match texcoord {
        Operand::Reg(r) => {
            if uv_reg == Some(r) {
                return;
            }
            let uv = compiled.instructions.iter().find_map(|instr| match instr {
                Instruction::LoadConst {
                    dst,
                    value_pool_idx,
                } if *dst == r => Some(compiled.value_pool[*value_pool_idx as usize].as_vector2()),
                _ => None,
            });
            let Some(uv) = uv else {
                panic!("latlongimage texcoord should come from LatlongUv or folded constant");
            };
            assert!((uv - Vec2::splat(0.5)).abs().max_element() < 1.0e-6);
        }
        Operand::Const(idx) => {
            let uv = compiled.value_pool[idx as usize].as_vector2();
            assert!((uv - Vec2::splat(0.5)).abs().max_element() < 1.0e-6);
        }
    }
}

fn compile_image_filter_graph(
    image_inputs: Vec<FlatNodeInput>,
) -> Result<CompiledMaterial, String> {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "image".to_string(),
                },
                output_type: MtlxType::Color3,
                inputs: image_inputs,
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission_color".to_string(),
                    ty: MtlxType::Color3,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: "image_filter".to_string(),
    };
    super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .map_err(|err| err.to_string())
}

fn string_input(name: &str, value: &str) -> FlatNodeInput {
    FlatNodeInput {
        name: name.to_string(),
        ty: MtlxType::String,
        colorspace: None,
        unit: None,
        unittype: None,
        binding: FlatInput::Value(MtlxValue::String(value.to_string())),
    }
}

#[test]
fn spec_image_cubic_and_animated_inputs_warn_without_silent_filter_fallback() {
    let compiled = compile_image_filter_graph(vec![
        string_input("filtertype", "cubic"),
        string_input("framerange", "1001-1005"),
        FlatNodeInput {
            name: "frameoffset".to_string(),
            ty: MtlxType::Integer,
            colorspace: None,
            unit: None,
            unittype: None,
            binding: FlatInput::Value(MtlxValue::Integer(1)),
        },
        string_input("frameendaction", "periodic"),
    ])
    .expect("cubic filtering and animated image inputs should compile with warnings");

    let filter = compiled.instructions.iter().find_map(|instr| match instr {
        Instruction::Image { filter, .. } => Some(*filter),
        _ => None,
    });
    assert_eq!(filter, Some(FilterType::Linear));
}

#[test]
fn spec_image_string_enums_reject_dynamic_or_malformed_values() {
    let dynamic_file = FlatNodeInput {
        name: "file".to_string(),
        ty: MtlxType::Filename,
        colorspace: None,
        unit: None,
        unittype: None,
        binding: FlatInput::Node {
            node: 0,
            output: None,
        },
    };
    let err = compile_image_filter_graph(vec![dynamic_file])
        .expect_err("dynamic image file should not silently become an empty filename");
    assert!(err.contains("image.file must be a static string value"));

    let dynamic_filter = FlatNodeInput {
        name: "filtertype".to_string(),
        ty: MtlxType::String,
        colorspace: None,
        unit: None,
        unittype: None,
        binding: FlatInput::Node {
            node: 0,
            output: None,
        },
    };
    let err = compile_image_filter_graph(vec![dynamic_filter])
        .expect_err("dynamic filtertype should not silently fall back to linear");
    assert!(err.contains("image.filtertype must be a static string value"));

    let bad_offset = FlatNodeInput {
        name: "frameoffset".to_string(),
        ty: MtlxType::Integer,
        colorspace: None,
        unit: None,
        unittype: None,
        binding: FlatInput::Value(MtlxValue::String("bad".to_string())),
    };
    let err = compile_image_filter_graph(vec![bad_offset])
        .expect_err("malformed frameoffset should not be silently ignored");
    assert!(err.contains("image.frameoffset must be an integer value"));
}

#[test]
fn spec_image_float_reads_red_channel_not_luminance() {
    let texture = Texture::from_pixels(1, 1, vec![Vec3::new(0.2, 0.8, 0.8)]);
    let regs = run(
        vec![Instruction::Image {
            dst: 0,
            texture: ImageTexture::Color(Arc::new(texture)),
            kind: ImageKind::Image,
            output: ValueType::Float,
            color_space: crate::material::TextureColorSpace::Linear,
            uaddress: AddressMode::Periodic,
            vaddress: AddressMode::Periodic,
            filter: FilterType::Closest,
            texcoord: Operand::Const(0),
            tiling: Operand::Const(1),
            offset: Operand::Const(2),
            default: Operand::Const(3),
        }],
        Vec::new(),
        vec![
            Value::Vector2(Vec2::splat(0.5)),
            Value::Vector2(Vec2::ONE),
            Value::Vector2(Vec2::ZERO),
            Value::Float(0.0),
        ],
        1,
    );
    match regs[0] {
        Value::Float(v) => assert!((v - 0.2).abs() < 1.0e-6),
        other => panic!("expected float image sample, got {:?}", other),
    }
}

#[test]
fn spec_udim_float_uses_scalar_tile_red_channel() {
    let rgb = Texture::from_pixels(1, 1, vec![Vec3::new(0.1, 0.9, 0.9)]);
    let scalar = ScalarTexture::from_pixels(1, 1, vec![0.73]);
    let tiles = UdimTiles {
        tiles: HashMap::from([(
            1001,
            UdimTile {
                rgb: Arc::new(rgb),
                alpha: None,
                scalar: Some(Arc::new(scalar)),
            },
        )]),
    };
    let regs = run(
        vec![Instruction::Image {
            dst: 0,
            texture: ImageTexture::Udim {
                tiles: Arc::new(tiles),
            },
            kind: ImageKind::Image,
            output: ValueType::Float,
            color_space: crate::material::TextureColorSpace::Linear,
            uaddress: AddressMode::Periodic,
            vaddress: AddressMode::Periodic,
            filter: FilterType::Closest,
            texcoord: Operand::Const(0),
            tiling: Operand::Const(1),
            offset: Operand::Const(2),
            default: Operand::Const(3),
        }],
        Vec::new(),
        vec![
            Value::Vector2(Vec2::splat(0.5)),
            Value::Vector2(Vec2::ONE),
            Value::Vector2(Vec2::ZERO),
            Value::Float(0.0),
        ],
        1,
    );
    match regs[0] {
        Value::Float(v) => assert!((v - 0.73).abs() < 1.0e-6),
        other => panic!("expected scalar UDIM sample, got {:?}", other),
    }
}

#[test]
fn spec_hextiledimage_color4_preserves_sampled_alpha() {
    let rgb = Texture::from_pixels(1, 1, vec![Vec3::new(0.25, 0.5, 0.75)]);
    let alpha = ScalarTexture::from_pixels(1, 1, vec![0.6]);
    let regs = run(
        vec![Instruction::HextiledImage {
            dst: 0,
            texture: ImageTexture::ColorAlpha {
                rgb: Arc::new(rgb),
                alpha: Arc::new(alpha),
            },
            output: ValueType::Color4,
            default_color: Vec4::ZERO,
            color_space: crate::material::TextureColorSpace::Linear,
            operands_start: 0,
        }],
        vec![
            Operand::Const(0),
            Operand::Const(1),
            Operand::Const(2),
            Operand::Const(3),
            Operand::Const(4),
            Operand::Const(5),
            Operand::Const(6),
            Operand::Const(7),
            Operand::Const(8),
            Operand::Const(9),
            Operand::Const(10),
        ],
        vec![
            Value::Vector2(Vec2::splat(0.5)),
            Value::Vector2(Vec2::ONE),
            Value::Float(0.0),
            Value::Vector2(Vec2::ZERO),
            Value::Float(1.0),
            Value::Vector2(Vec2::ONE),
            Value::Float(0.0),
            Value::Vector2(Vec2::ZERO),
            Value::Float(0.5),
            Value::Float(0.0),
            Value::Color3(Vec3::new(0.2722287, 0.6740818, 0.0536895)),
        ],
        1,
    );
    match regs[0] {
        Value::Color4(v) => assert!((v.w - 0.6).abs() < 1.0e-6),
        other => panic!("expected color4 hextiledimage sample, got {:?}", other),
    }
}

#[test]
fn spec_empty_surface_root_is_passthrough() {
    let graph = FlatGraph {
        nodes: vec![FlatNode {
            kind: FlatNodeKind::SurfaceMaterial,
            output_type: MtlxType::Material,
            inputs: vec![FlatNodeInput {
                name: "surfaceshader".to_string(),
                ty: MtlxType::Surfaceshader,
                colorspace: None,
                unit: None,
                unittype: None,
                binding: FlatInput::Empty,
            }],
        }],
        root: 0,
        back_root: None,
        material_name: "empty".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("empty surface material should compile");
    assert!(compiled.passthrough);
    assert_eq!(compiled.root, 0);
}

#[test]
fn spec_saturate_color4_preserves_alpha() {
    let r = run_saturate(
        ValueType::Color4,
        Value::Color4(Vec4::new(0.25, 0.5, 0.75, 0.33)),
        Value::Float(0.0),
        Value::Color3(Vec3::new(0.2126, 0.7152, 0.0722)),
    );
    assert!(approx_f(r.as_color4().w, 0.33, 1.0e-6));
}

#[test]
fn spec_saturate_uses_luminance_mix_without_clamping_amount() {
    let r = run_saturate(
        ValueType::Color3,
        Value::Color3(Vec3::new(0.2, 0.4, 0.8)),
        Value::Float(1.5),
        Value::Color3(Vec3::new(0.0, 1.0, 0.0)),
    );
    assert!(approx_v3(r.as_color3(), Vec3::new(0.1, 0.4, 1.0), 1.0e-6));
}

#[test]
fn spec_colorcorrect_lift_gain_contrast_matches_nodegraph() {
    let r = run_colorcorrect(
        ValueType::Color3,
        [
            Value::Color3(Vec3::splat(0.4)),
            Value::Float(0.0),
            Value::Float(1.0),
            Value::Float(1.0),
            Value::Float(0.2),
            Value::Float(1.0),
            Value::Float(1.0),
            Value::Float(0.5),
            Value::Float(0.0),
        ],
    );
    assert!(approx_v3(r.as_color3(), Vec3::splat(0.52), 1.0e-6));
}

#[test]
fn spec_colorcorrect_color4_preserves_alpha() {
    let r = run_colorcorrect(
        ValueType::Color4,
        [
            Value::Color4(Vec4::new(0.25, 0.5, 0.75, 0.4)),
            Value::Float(0.0),
            Value::Float(1.0),
            Value::Float(1.0),
            Value::Float(0.0),
            Value::Float(1.0),
            Value::Float(1.0),
            Value::Float(0.5),
            Value::Float(0.0),
        ],
    );
    assert!(approx_f(r.as_color4().w, 0.4, 1.0e-6));
}

#[test]
fn spec_colorcorrect_saturation_uses_luminance_mix_nodegraph() {
    let r = run_colorcorrect(
        ValueType::Color3,
        [
            Value::Color3(Vec3::new(1.0, 0.0, 0.0)),
            Value::Float(0.0),
            Value::Float(0.0),
            Value::Float(1.0),
            Value::Float(0.0),
            Value::Float(1.0),
            Value::Float(1.0),
            Value::Float(0.5),
            Value::Float(0.0),
        ],
    );
    assert!(approx_v3(r.as_color3(), Vec3::splat(0.2722287), 1.0e-6));
}

#[test]
fn spec_roughness_anisotropy_matches_mdl_formula() {
    let r = run_roughness_anisotropy(0.5, 0.75).as_vector2();

    assert!(approx_v2(r, Vec2::new(0.5, 0.125), 1.0e-6));
}

#[test]
fn spec_glossiness_anisotropy_inverts_then_squares_roughness() {
    let r = run_glossiness_anisotropy(0.5, 0.0).as_vector2();

    assert!(approx_v2(r, Vec2::splat(0.25), 1.0e-6));
}

#[test]
fn spec_roughness_dual_accepts_vector2_and_mirrors_negative_y() {
    let r = run_roughness_dual(Vec2::new(0.5, -1.0)).as_vector2();

    assert!(approx_v2(r, Vec2::splat(0.25), 1.0e-6));
}

#[test]
fn spec_blackbody_matches_generated_glsl_planckian_locus() {
    let regs = run(
        vec![
            Instruction::Blackbody {
                dst: 0,
                temp: Operand::Const(0),
            },
            Instruction::Blackbody {
                dst: 1,
                temp: Operand::Const(1),
            },
            Instruction::Blackbody {
                dst: 2,
                temp: Operand::Const(2),
            },
        ],
        vec![],
        vec![
            Value::Float(5000.0),
            Value::Float(1000.0),
            Value::Float(30000.0),
        ],
        3,
    );

    assert!(approx_v3(
        regs[0].as_color3(),
        Vec3::new(1.2123184, 0.9608892, 0.7628533),
        1.0e-5
    ));
    assert!(approx_v3(
        regs[1].as_color3(),
        Vec3::new(2.964251, 0.5212498, 0.0),
        1.0e-5
    ));
    assert!(approx_v3(
        regs[2].as_color3(),
        Vec3::new(0.7272615, 0.9875422, 1.9270417),
        1.0e-5
    ));
}

#[test]
fn spec_chiang_hair_roughness_matches_mdl_variance_formula() {
    let regs = run(
        vec![
            Instruction::ChiangHairRoughness {
                dst: 0,
                which: ChiangHairRoughnessOutput::R,
                longitudinal: Operand::Const(0),
                azimuthal: Operand::Const(1),
                scale_tt: Operand::Const(2),
                scale_trt: Operand::Const(3),
            },
            Instruction::ChiangHairRoughness {
                dst: 1,
                which: ChiangHairRoughnessOutput::TT,
                longitudinal: Operand::Const(0),
                azimuthal: Operand::Const(1),
                scale_tt: Operand::Const(2),
                scale_trt: Operand::Const(3),
            },
            Instruction::ChiangHairRoughness {
                dst: 2,
                which: ChiangHairRoughnessOutput::TRT,
                longitudinal: Operand::Const(0),
                azimuthal: Operand::Const(1),
                scale_tt: Operand::Const(2),
                scale_trt: Operand::Const(3),
            },
        ],
        vec![],
        vec![
            Value::Float(0.1),
            Value::Float(0.2),
            Value::Float(0.5),
            Value::Float(2.0),
        ],
        3,
    );
    let lr = 0.1_f32;
    let ar = 0.2_f32;
    let v = 0.726 * lr + 0.812 * lr * lr + 3.7 * lr.powi(20);
    let v = v * v;
    let s = 0.265 * ar + 1.194 * ar * ar + 5.372 * ar.powi(22);

    assert!(approx_v2(regs[0].as_vector2(), Vec2::new(v, s), 1.0e-6));
    assert!(approx_v2(
        regs[1].as_vector2(),
        Vec2::new(v * 0.25, s),
        1.0e-6
    ));
    assert!(approx_v2(
        regs[2].as_vector2(),
        Vec2::new(v * 4.0, s),
        1.0e-6
    ));
}

#[test]
fn spec_deon_hair_absorption_from_melanin_matches_mdl_log_mapping() {
    let regs = run(
        vec![Instruction::DeonHairAbsorptionFromMelanin {
            dst: 0,
            operands_start: 0,
        }],
        vec![
            Operand::Const(0),
            Operand::Const(1),
            Operand::Const(2),
            Operand::Const(3),
        ],
        vec![
            Value::Float(0.25),
            Value::Float(0.5),
            Value::Color3(Vec3::new(0.657704, 0.498077, 0.254107)),
            Value::Color3(Vec3::new(0.829444, 0.67032, 0.349938)),
        ],
        1,
    );
    let melanin = -(1.0_f32 - 0.25).ln();
    let eumelanin = melanin * 0.5;
    let pheomelanin = melanin * 0.5;
    let eum = Vec3::new(0.657704_f32, 0.498077, 0.254107);
    let phe = Vec3::new(0.829444_f32, 0.67032, 0.349938);
    let expected = eumelanin * Vec3::new(-eum.x.ln(), -eum.y.ln(), -eum.z.ln())
        + pheomelanin * Vec3::new(-phe.x.ln(), -phe.y.ln(), -phe.z.ln());

    assert!(approx_v3(regs[0].as_color3(), expected, 1.0e-6));
}

#[test]
fn spec_deon_hair_absorption_from_melanin_does_not_clamp_redness() {
    let regs = run(
        vec![Instruction::DeonHairAbsorptionFromMelanin {
            dst: 0,
            operands_start: 0,
        }],
        vec![
            Operand::Const(0),
            Operand::Const(1),
            Operand::Const(2),
            Operand::Const(3),
        ],
        vec![
            Value::Float(0.25),
            Value::Float(1.5),
            Value::Color3(Vec3::new(0.657704, 0.498077, 0.254107)),
            Value::Color3(Vec3::new(0.829444, 0.67032, 0.349938)),
        ],
        1,
    );
    let melanin = -(1.0_f32 - 0.25).ln();
    let eumelanin = melanin * -0.5;
    let pheomelanin = melanin * 1.5;
    let eum = Vec3::new(0.657704_f32, 0.498077, 0.254107);
    let phe = Vec3::new(0.829444_f32, 0.67032, 0.349938);
    let expected = (eumelanin * Vec3::new(-eum.x.ln(), -eum.y.ln(), -eum.z.ln())
        + pheomelanin * Vec3::new(-phe.x.ln(), -phe.y.ln(), -phe.z.ln()))
    .max(Vec3::ZERO);

    assert!(approx_v3(regs[0].as_color3(), expected, 1.0e-6));
}

#[test]
fn spec_chiang_hair_absorption_from_color_clamps_color_like_mdl() {
    let regs = run(
        vec![Instruction::ChiangHairAbsorptionFromColor {
            dst: 0,
            color: Operand::Const(0),
            beta: Operand::Const(1),
        }],
        vec![],
        vec![
            Value::Color3(Vec3::new(2.0, 1.0, 0.00001)),
            Value::Float(0.2),
        ],
        1,
    );
    let b = 0.2_f32;
    let factor = 5.969 - 0.215 * b + 2.532 * b * b - 10.73 * b.powi(3)
        + 5.574 * b.powi(4)
        + 0.245 * b.powi(5);
    let z = (0.001_f32.ln() / factor).powi(2);

    assert!(approx_v3(
        regs[0].as_color3(),
        Vec3::new(0.0, 0.0, z),
        1.0e-6
    ));
}

#[test]
fn spec_chiang_hair_absorption_from_color_does_not_clamp_beta() {
    let regs = run(
        vec![Instruction::ChiangHairAbsorptionFromColor {
            dst: 0,
            color: Operand::Const(0),
            beta: Operand::Const(1),
        }],
        vec![],
        vec![Value::Color3(Vec3::splat(0.5)), Value::Float(0.0)],
        1,
    );
    let expected = Vec3::splat((0.5_f32.ln() / 5.969).powi(2));
    assert!(approx_v3(regs[0].as_color3(), expected, 1.0e-6));
}

#[test]
fn spec_hextilednormalmap_missing_file_returns_default() {
    let default = Vec3::new(0.1, 0.2, 0.9);
    let r = run_hextilednormalmap_missing(default);

    assert!(approx_v3(r.as_vector3(), default, 1.0e-6));
}

#[test]
fn spec_normalmap_with_frame_matches_mdl_linear_frame_sum() {
    let regs = run(
        vec![Instruction::NormalmapWithFrame {
            dst: 0,
            operands_start: 0,
        }],
        vec![
            Operand::Const(0),
            Operand::Const(1),
            Operand::Const(2),
            Operand::Const(3),
            Operand::Const(4),
        ],
        vec![
            Value::Vector3(Vec3::new(1.0, 1.0, 0.5)),
            Value::Vector2(Vec2::ONE),
            Value::Vector3(Vec3::Z),
            Value::Vector3(Vec3::new(2.0, 0.0, 0.0)),
            Value::Vector3(Vec3::Y),
        ],
        1,
    );
    let expected = Vec3::new(2.0, 1.0, 0.0).normalize();
    assert!(approx_v3(regs[0].as_vector3(), expected, 1.0e-6));
}

#[test]
fn spec_normalmap_default_frame_uses_normalized_geom_tangents() {
    let mut sv = dummy_sv();
    sv.dpdu = Vec3::new(2.0, 0.0, 0.0);
    sv.dpdv = Vec3::Y;
    let regs = run_with_sv(
        vec![Instruction::Normalmap {
            dst: 0,
            raw: Operand::Const(0),
            scale: Operand::Const(1),
        }],
        Vec::new(),
        vec![
            Value::Vector3(Vec3::new(1.0, 0.5, 1.0)),
            Value::Vector2(Vec2::ONE),
        ],
        1,
        sv,
    );
    assert!(approx_v3(
        regs[0].as_vector3(),
        Vec3::new(1.0, 0.0, 1.0).normalize(),
        1.0e-6
    ));
}

#[test]
fn spec_normalmap_zero_result_uses_raw_normalize() {
    let regs = run(
        vec![Instruction::NormalmapWithFrame {
            dst: 0,
            operands_start: 0,
        }],
        vec![
            Operand::Const(0),
            Operand::Const(1),
            Operand::Const(2),
            Operand::Const(3),
            Operand::Const(4),
        ],
        vec![
            Value::Vector3(Vec3::new(0.5, 0.5, 0.5)),
            Value::Vector2(Vec2::ONE),
            Value::Vector3(Vec3::ZERO),
            Value::Vector3(Vec3::ZERO),
            Value::Vector3(Vec3::ZERO),
        ],
        1,
    );
    assert!(regs[0].as_vector3().is_nan());
}

#[test]
fn spec_hextilednormalmap_explicit_zero_normal_is_not_default_frame() {
    let texture = Texture::from_pixels(1, 1, vec![Vec3::new(0.5, 0.5, 1.0)]);
    let regs = run(
        vec![Instruction::HextiledNormalMap {
            dst: 0,
            texture: Some(Arc::new(texture)),
            flip_g: false,
            operands_start: 0,
        }],
        vec![
            Operand::Const(0),
            Operand::Const(1),
            Operand::Const(2),
            Operand::Const(3),
            Operand::Const(4),
            Operand::Const(5),
            Operand::Const(6),
            Operand::Const(7),
            Operand::Const(8),
            Operand::Const(9),
            Operand::Const(10),
            Operand::Const(11),
            Operand::Const(12),
            Operand::Const(13),
        ],
        vec![
            Value::Vector2(Vec2::ZERO),
            Value::Vector2(Vec2::ONE),
            Value::Float(1.0),
            Value::Vector2(Vec2::new(0.0, 360.0)),
            Value::Float(1.0),
            Value::Vector2(Vec2::new(0.5, 2.0)),
            Value::Float(1.0),
            Value::Vector2(Vec2::new(0.0, 1.0)),
            Value::Float(0.5),
            Value::Float(1.0),
            Value::Vector3(Vec3::new(0.5, 0.5, 1.0)),
            Value::Vector3(Vec3::ZERO),
            Value::Vector3(Vec3::X),
            Value::Vector3(Vec3::Y),
        ],
        1,
    );
    assert!(regs[0].as_vector3().is_nan());
}

#[test]
fn spec_bump_constant_height_matches_flat_heighttonormal_graph() {
    let mut sv = dummy_sv();
    sv.ns = Vec3::new(0.0, 0.0, 2.0);
    let regs = run_with_sv(
        vec![Instruction::Bump {
            dst: 0,
            height: Operand::Const(0),
            scale: Operand::Const(1),
        }],
        Vec::new(),
        vec![Value::Float(2.0), Value::Float(5.0)],
        1,
        sv,
    );
    assert!(approx_v3(regs[0].as_vector3(), Vec3::Z, 1.0e-6));
}

#[test]
fn spec_bump_with_frame_explicit_zero_normal_uses_raw_normalize() {
    let regs = run(
        vec![Instruction::BumpWithFrame {
            dst: 0,
            operands_start: 0,
        }],
        vec![
            Operand::Const(0),
            Operand::Const(1),
            Operand::Const(2),
            Operand::Const(3),
        ],
        vec![
            Value::Float(2.0),
            Value::Float(5.0),
            Value::Vector3(Vec3::ZERO),
            Value::Vector3(Vec3::X),
        ],
        1,
    );
    assert!(regs[0].as_vector3().is_nan());
}

#[test]
fn spec_surface_unlit_defaults_emit_white() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
        ],
        root: 1,
        back_root: None,
        material_name: "surface_unlit_default".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("surface_unlit default graph should compile");

    assert!(compiled.may_emit);
    assert!(approx_f(compiled.max_emission, 1.0, 1.0e-6));
    assert!(compiled.thin_walled);
}

#[test]
fn spec_surface_unlit_saturates_transmission_like_mdl() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "transmission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Value(MtlxValue::Float(2.0)),
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
        ],
        root: 1,
        back_root: None,
        material_name: "surface_unlit_transmission_saturate".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("surface_unlit graph should compile");
    let mut scratch = MtlxScratch::default();
    let handle = scratch.alloc_regs(compiled.num_registers as usize);
    super::runtime::run_instructions(&compiled, &dummy_sv(), &mut scratch, handle);

    let le = super::runtime::evaluate_le(&compiled, scratch.regs_slice(handle), &dummy_sv());

    assert_eq!(le, None);
}

#[test]
fn spec_surface_thin_walled_requires_boolean() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Surface,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "thin_walled".to_string(),
                    ty: MtlxType::Boolean,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Value(MtlxValue::Float(1.0)),
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
        ],
        root: 1,
        back_root: None,
        material_name: "surface_bad_thin_walled".to_string(),
    };

    let err = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .unwrap_err();

    assert!(format!("{:?}", err).contains("surface.thin_walled must be boolean"));
}

#[test]
fn spec_ramplr_missing_right_value_defaults_to_zero() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "ramplr".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![
                    FlatNodeInput {
                        name: "valuel".to_string(),
                        ty: MtlxType::Float,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Float(0.25)),
                    },
                    FlatNodeInput {
                        name: "texcoord".to_string(),
                        ty: MtlxType::Vector2,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Vector2(Vec2::new(1.0, 0.0))),
                    },
                ],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: "ramplr_default".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("ramplr graph should compile");
    let mut scratch = MtlxScratch::default();
    let handle = scratch.alloc_regs(compiled.num_registers as usize);
    super::runtime::run_instructions(&compiled, &dummy_sv(), &mut scratch, handle);
    let le = super::runtime::evaluate_le(&compiled, scratch.regs_slice(handle), &dummy_sv());

    assert!(approx_v3(le.unwrap_or(Vec3::ZERO), Vec3::ZERO, 1.0e-6));
}

#[test]
fn spec_ramp4_mixes_top_to_bottom_like_nodegraph() {
    let regs = run(
        vec![Instruction::Ramp4 {
            dst: 0,
            ty: ValueType::Float,
            texcoord: Operand::Const(0),
            tl: Operand::Const(1),
            tr: Operand::Const(2),
            bl: Operand::Const(3),
            br: Operand::Const(4),
        }],
        Vec::new(),
        vec![
            Value::Vector2(Vec2::new(0.25, 0.0)),
            Value::Float(10.0),
            Value::Float(20.0),
            Value::Float(30.0),
            Value::Float(40.0),
        ],
        1,
    );

    assert!(approx_f(regs[0].as_float(), 12.5, 1.0e-6));
}

#[test]
fn spec_splittb_uses_mdl_x_axis_step() {
    let regs = run(
        vec![Instruction::Splittb {
            dst: 0,
            ty: ValueType::Float,
            texcoord: Operand::Const(0),
            center: Operand::Const(1),
            t: Operand::Const(2),
            b: Operand::Const(3),
        }],
        Vec::new(),
        vec![
            Value::Vector2(Vec2::new(0.75, 0.25)),
            Value::Float(0.5),
            Value::Float(1.0),
            Value::Float(9.0),
        ],
        1,
    );

    assert!(approx_f(regs[0].as_float(), 9.0, 1.0e-6));
}

#[test]
fn spec_fractal2d_vector2_uses_scalar_offset_channel() {
    let p = Vec2::new(0.37, 0.61);
    let regs = run(
        vec![Instruction::Noise {
            dst: 0,
            kind: NoiseKind::Fractal2d,
            output: NoiseOutput::Vector2,
            operands_start: 0,
        }],
        vec![
            Operand::Const(0),
            Operand::Const(1),
            Operand::Const(2),
            Operand::Const(3),
            Operand::Const(4),
            Operand::Const(5),
            Operand::Const(6),
        ],
        vec![
            Value::Vector2(p),
            Value::Vector2(Vec2::new(2.0, 3.0)),
            Value::Float(100.0),
            Value::Integer(2),
            Value::Float(2.0),
            Value::Float(0.5),
            Value::Float(1.0),
        ],
        1,
    );
    let expected = Vec2::new(
        2.0 * crate::material::pattern::noise::fbm2d(p, 2, 2.0, 0.5),
        3.0 * crate::material::pattern::noise::fbm2d(p + Vec2::new(19.0, 193.0), 2, 2.0, 0.5),
    );

    assert!(approx_v2(regs[0].as_vector2(), expected, 1.0e-6));
}

#[test]
fn spec_fractal3d_vector2_uses_scalar_offset_channel() {
    let p = Vec3::new(0.37, 0.61, 0.23);
    let regs = run(
        vec![Instruction::Noise {
            dst: 0,
            kind: NoiseKind::Fractal3d,
            output: NoiseOutput::Vector2,
            operands_start: 0,
        }],
        vec![
            Operand::Const(0),
            Operand::Const(1),
            Operand::Const(2),
            Operand::Const(3),
            Operand::Const(4),
            Operand::Const(5),
            Operand::Const(6),
        ],
        vec![
            Value::Vector3(p),
            Value::Vector2(Vec2::new(2.0, 3.0)),
            Value::Float(100.0),
            Value::Integer(2),
            Value::Float(2.0),
            Value::Float(0.5),
            Value::Float(1.0),
        ],
        1,
    );
    let expected = Vec2::new(
        2.0 * crate::material::pattern::noise::fbm3d(p, 2, 2.0, 0.5),
        3.0 * crate::material::pattern::noise::fbm3d(p + Vec3::new(19.0, 193.0, 17.0), 2, 2.0, 0.5),
    );

    assert!(approx_v2(regs[0].as_vector2(), expected, 1.0e-6));
}

#[test]
fn spec_fractal_zero_octaves_matches_mdl_empty_loop() {
    let regs = run(
        vec![Instruction::Noise {
            dst: 0,
            kind: NoiseKind::Fractal2d,
            output: NoiseOutput::Vector3,
            operands_start: 0,
        }],
        vec![
            Operand::Const(0),
            Operand::Const(1),
            Operand::Const(2),
            Operand::Const(3),
            Operand::Const(4),
            Operand::Const(5),
            Operand::Const(6),
        ],
        vec![
            Value::Vector2(Vec2::new(0.37, 0.61)),
            Value::Vector3(Vec3::new(2.0, 3.0, 4.0)),
            Value::Float(100.0),
            Value::Integer(0),
            Value::Float(2.0),
            Value::Float(0.5),
            Value::Float(1.0),
        ],
        1,
    );

    assert!(approx_v3(regs[0].as_vector3(), Vec3::ZERO, 1.0e-6));
}

#[test]
fn spec_worleynoise_integer_style_one_compiles_to_solid() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "worleynoise2d".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![
                    FlatNodeInput {
                        name: "style".to_string(),
                        ty: MtlxType::Integer,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Integer(1)),
                    },
                    FlatNodeInput {
                        name: "texcoord".to_string(),
                        ty: MtlxType::Vector2,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Vector2(Vec2::new(0.2, 0.4))),
                    },
                ],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: "worley_style".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("worleynoise style graph should compile");

    let saw_solid = compiled.instructions.iter().any(|inst| {
        matches!(
            inst,
            Instruction::Worley {
                style: WorleyStyle::Solid,
                ..
            }
        )
    });
    assert!(saw_solid);
}

#[test]
fn spec_unifiednoise_worley_uses_integer_style() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "unifiednoise2d".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![
                    FlatNodeInput {
                        name: "type".to_string(),
                        ty: MtlxType::Integer,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Integer(2)),
                    },
                    FlatNodeInput {
                        name: "style".to_string(),
                        ty: MtlxType::Integer,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Integer(1)),
                    },
                    FlatNodeInput {
                        name: "texcoord".to_string(),
                        ty: MtlxType::Vector2,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Vector2(Vec2::new(0.2, 0.4))),
                    },
                ],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: "unified_worley_style".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("unifiednoise style graph should compile");

    let saw_solid = compiled.instructions.iter().any(|inst| {
        matches!(
            inst,
            Instruction::Worley {
                style: WorleyStyle::Solid,
                ..
            }
        )
    });
    assert!(saw_solid);
}

#[test]
fn spec_worleynoise_invalid_integer_style_errors() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "worleynoise2d".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![FlatNodeInput {
                    name: "style".to_string(),
                    ty: MtlxType::Integer,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Value(MtlxValue::Integer(2)),
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: "worley_bad_style".to_string(),
    };
    let err = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect_err("invalid worleynoise style should fail");

    assert!(err.to_string().contains("worleynoise2d.style"));
}

#[test]
fn spec_unifiednoise_connected_type_errors() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "unifiednoise2d".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![FlatNodeInput {
                    name: "type".to_string(),
                    ty: MtlxType::Integer,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: "unified_bad_type".to_string(),
    };
    let err = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect_err("connected unifiednoise type should fail");

    assert!(err.to_string().contains("unifiednoise2d.type"));
}

#[test]
fn spec_boolean_logical_nodes_compile() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "and".to_string(),
                },
                output_type: MtlxType::Boolean,
                inputs: vec![
                    FlatNodeInput {
                        name: "in1".to_string(),
                        ty: MtlxType::Boolean,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Boolean(true)),
                    },
                    FlatNodeInput {
                        name: "in2".to_string(),
                        ty: MtlxType::Boolean,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Boolean(false)),
                    },
                ],
            },
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "ifelse".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![
                    FlatNodeInput {
                        name: "cond".to_string(),
                        ty: MtlxType::Boolean,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Node {
                            node: 0,
                            output: None,
                        },
                    },
                    FlatNodeInput {
                        name: "in1".to_string(),
                        ty: MtlxType::Float,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Float(1.0)),
                    },
                    FlatNodeInput {
                        name: "in2".to_string(),
                        ty: MtlxType::Float,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Float(0.0)),
                    },
                ],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 2,
                        output: None,
                    },
                }],
            },
        ],
        root: 3,
        back_root: None,
        material_name: "logical_compile".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("logical graph should compile");
    assert!(approx_v3(eval_compiled_le(&compiled), Vec3::ZERO, 1.0e-6));
}

#[test]
fn spec_logical_not_default_is_true() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "not".to_string(),
                },
                output_type: MtlxType::Boolean,
                inputs: vec![],
            },
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "ifelse".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![
                    FlatNodeInput {
                        name: "cond".to_string(),
                        ty: MtlxType::Boolean,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Node {
                            node: 0,
                            output: None,
                        },
                    },
                    FlatNodeInput {
                        name: "in1".to_string(),
                        ty: MtlxType::Float,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Float(1.0)),
                    },
                    FlatNodeInput {
                        name: "in2".to_string(),
                        ty: MtlxType::Float,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Float(0.0)),
                    },
                ],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 2,
                        output: None,
                    },
                }],
            },
        ],
        root: 3,
        back_root: None,
        material_name: "logical_not_default".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("logical not default should compile");
    let mut scratch = MtlxScratch::default();
    let handle = scratch.alloc_regs(compiled.num_registers as usize);
    super::runtime::run_instructions(&compiled, &dummy_sv(), &mut scratch, handle);
    let le = super::runtime::evaluate_le(&compiled, scratch.regs_slice(handle), &dummy_sv())
        .expect("surface_unlit should emit");

    assert!(approx_v3(
        le,
        Vec3::splat(1.0 / std::f32::consts::PI),
        1.0e-6
    ));
}

#[test]
fn spec_geomcolor_defaults_to_black() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "geompropvalue".to_string(),
                },
                output_type: MtlxType::Color3,
                inputs: vec![FlatNodeInput {
                    name: "geomprop".to_string(),
                    ty: MtlxType::String,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Value(MtlxValue::String("geomcolor".to_string())),
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![
                    FlatNodeInput {
                        name: "emission".to_string(),
                        ty: MtlxType::Float,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Float(1.0)),
                    },
                    FlatNodeInput {
                        name: "emission_color".to_string(),
                        ty: MtlxType::Color3,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Node {
                            node: 0,
                            output: None,
                        },
                    },
                ],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: "geomcolor_default".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("geomcolor graph should compile");
    assert!(compiled.instructions.iter().any(|inst| {
        matches!(
            inst,
            Instruction::LoadGeom {
                kind: super::compiled::GeometricKind::Geomcolor,
                ..
            }
        )
    }));
    let mut scratch = MtlxScratch::default();
    let handle = scratch.alloc_regs(compiled.num_registers as usize);
    super::runtime::run_instructions(&compiled, &dummy_sv(), &mut scratch, handle);
    let le = super::runtime::evaluate_le(&compiled, scratch.regs_slice(handle), &dummy_sv());

    assert!(approx_v3(le.unwrap_or(Vec3::ZERO), Vec3::ZERO, 1.0e-6));
}

#[test]
fn spec_geompropvalue_boolean_default_compiles() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "geompropvalue".to_string(),
                },
                output_type: MtlxType::Boolean,
                inputs: vec![FlatNodeInput {
                    name: "default".to_string(),
                    ty: MtlxType::Boolean,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Value(MtlxValue::Boolean(true)),
                }],
            },
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "ifelse".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![
                    FlatNodeInput {
                        name: "cond".to_string(),
                        ty: MtlxType::Boolean,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Node {
                            node: 0,
                            output: None,
                        },
                    },
                    FlatNodeInput {
                        name: "in1".to_string(),
                        ty: MtlxType::Float,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Float(1.0)),
                    },
                    FlatNodeInput {
                        name: "in2".to_string(),
                        ty: MtlxType::Float,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Float(0.0)),
                    },
                ],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 2,
                        output: None,
                    },
                }],
            },
        ],
        root: 3,
        back_root: None,
        material_name: "geomprop_bool".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("geompropvalue boolean default should compile");
    assert!(approx_v3(
        eval_compiled_le(&compiled),
        Vec3::splat(1.0 / std::f32::consts::PI),
        1.0e-6
    ));
}

#[test]
fn spec_geompropvalue_geomprop_requires_static_string() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "geompropvalue".to_string(),
                },
                output_type: MtlxType::Color3,
                inputs: vec![FlatNodeInput {
                    name: "geomprop".to_string(),
                    ty: MtlxType::String,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission_color".to_string(),
                    ty: MtlxType::Color3,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: "geomprop_dynamic".to_string(),
    };
    let err = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect_err("dynamic geomprop should not silently use the default value");
    assert!(
        err.to_string()
            .contains("geompropvalue.geomprop must be a static string value")
    );
}

#[test]
fn spec_ifgreater_integer_output_compiles() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "ifgreater".to_string(),
                },
                output_type: MtlxType::Integer,
                inputs: vec![
                    FlatNodeInput {
                        name: "value1".to_string(),
                        ty: MtlxType::Integer,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Integer(2)),
                    },
                    FlatNodeInput {
                        name: "value2".to_string(),
                        ty: MtlxType::Integer,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Integer(1)),
                    },
                    FlatNodeInput {
                        name: "in1".to_string(),
                        ty: MtlxType::Integer,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Integer(3)),
                    },
                ],
            },
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "convert".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![FlatNodeInput {
                    name: "in".to_string(),
                    ty: MtlxType::Integer,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 2,
                        output: None,
                    },
                }],
            },
        ],
        root: 3,
        back_root: None,
        material_name: "ifgreater_integer".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("ifgreater integer should compile");
    assert!(approx_v3(
        eval_compiled_le(&compiled),
        Vec3::splat(3.0 / std::f32::consts::PI),
        1.0e-6
    ));
}

#[test]
fn spec_ifgreater_matrix_output_compiles() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "ifgreater".to_string(),
                },
                output_type: MtlxType::Matrix33,
                inputs: vec![],
            },
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "determinant".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![FlatNodeInput {
                    name: "in".to_string(),
                    ty: MtlxType::Matrix33,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 2,
                        output: None,
                    },
                }],
            },
        ],
        root: 3,
        back_root: None,
        material_name: "ifgreater_matrix".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("ifgreater matrix should compile");
    assert!(
        compiled
            .instructions
            .iter()
            .any(|inst| matches!(inst, Instruction::Compare { .. }))
    );
}

#[test]
fn spec_ifequal_default_values_select_in1() {
    let input = Vec3::new(0.3, 0.2, 0.1);
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "ifequal".to_string(),
                },
                output_type: MtlxType::Color3,
                inputs: vec![FlatNodeInput {
                    name: "in1".to_string(),
                    ty: MtlxType::Color3,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Value(MtlxValue::Color3(input)),
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission_color".to_string(),
                    ty: MtlxType::Color3,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: "ifequal_default".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("ifequal default graph should compile");

    let mut scratch = MtlxScratch::default();
    let handle = scratch.alloc_regs(compiled.num_registers as usize);
    super::runtime::run_instructions(&compiled, &dummy_sv(), &mut scratch, handle);
    let le = super::runtime::evaluate_le(&compiled, scratch.regs_slice(handle), &dummy_sv())
        .expect("surface_unlit should emit");
    assert!(approx_v3(le, input * (1.0 / std::f32::consts::PI), 1.0e-6));
}

#[test]
fn spec_closure_ifequal_uses_exact_mdl_equality() {
    let compiled = CompiledMaterial {
        instructions: Vec::new(),
        operand_pool: Vec::new(),
        value_pool: Vec::new(),
        opacity_instructions: Vec::new(),
        opacity_operand_pool: Vec::new(),
        opacity_closure_nodes: Vec::new(),
        opacity_num_registers: 0,
        num_registers: 0,
        closure_nodes: vec![
            ClosureNode::IfEqual {
                value1: ParamRef::Float(1.0),
                value2: ParamRef::Float(1.0000005),
                then_branch: 1,
                else_branch: 2,
                kind: ClosureKind::Bsdf,
            },
            ClosureNode::BurleyDiffuse {
                weight: ParamRef::Float(1.0),
                color: ParamRef::Color3(Vec3::X),
                roughness: ParamRef::Float(0.0),
                normal: None,
            },
            ClosureNode::BurleyDiffuse {
                weight: ParamRef::Float(1.0),
                color: ParamRef::Color3(Vec3::Y),
                roughness: ParamRef::Float(0.0),
                normal: None,
            },
        ],
        root: 0,
        passthrough: false,
        max_emission: 0.0,
        may_emit: false,
        has_opacity_test: false,
        thin_walled: false,
        sheen_lut: None,
        mtlx_dielectric_lut: None,
        mtlx_generalized_schlick_lut: None,
    };

    let sv = dummy_sv();
    let f = super::runtime::eval_closure(&compiled, &[], &sv, Vec3::Z, Vec3::Z);
    assert!(f.y > 0.0);
    assert_eq!(f.x, 0.0);
}

#[test]
fn spec_switch_matrix_output_compiles() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "switch".to_string(),
                },
                output_type: MtlxType::Matrix33,
                inputs: vec![FlatNodeInput {
                    name: "which".to_string(),
                    ty: MtlxType::Integer,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Value(MtlxValue::Integer(0)),
                }],
            },
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "determinant".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![FlatNodeInput {
                    name: "in".to_string(),
                    ty: MtlxType::Matrix33,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 2,
                        output: None,
                    },
                }],
            },
        ],
        root: 3,
        back_root: None,
        material_name: "switch_matrix".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("switch matrix should compile");
    assert!(
        compiled
            .instructions
            .iter()
            .any(|inst| matches!(inst, Instruction::Switch { .. }))
    );
}

#[test]
fn spec_convert_boolean_to_integer_output_compiles() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "convert".to_string(),
                },
                output_type: MtlxType::Integer,
                inputs: vec![FlatNodeInput {
                    name: "in".to_string(),
                    ty: MtlxType::Boolean,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Value(MtlxValue::Boolean(true)),
                }],
            },
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "convert".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![FlatNodeInput {
                    name: "in".to_string(),
                    ty: MtlxType::Integer,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 2,
                        output: None,
                    },
                }],
            },
        ],
        root: 3,
        back_root: None,
        material_name: "convert_bool_int".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("boolean to integer convert should compile");
    assert!(approx_v3(
        eval_compiled_le(&compiled),
        Vec3::splat(1.0 / std::f32::consts::PI),
        1.0e-6
    ));
}

#[test]
fn spec_convert_float_to_integer_node_errors() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "convert".to_string(),
                },
                output_type: MtlxType::Integer,
                inputs: vec![FlatNodeInput {
                    name: "in".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Value(MtlxValue::Float(2.9)),
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Integer,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: "convert_float_int".to_string(),
    };
    let err = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect_err("float to integer convert is not a MaterialX convert overload");
    assert!(err.to_string().contains("convert from Float to Integer"));
}

#[test]
fn spec_creatematrix_vector3_matrix44_compiles_with_vec3_rows() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "creatematrix".to_string(),
                },
                output_type: MtlxType::Matrix44,
                inputs: vec![FlatNodeInput {
                    name: "in1".to_string(),
                    ty: MtlxType::Vector3,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Value(MtlxValue::Vector3(Vec3::X)),
                }],
            },
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "determinant".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![FlatNodeInput {
                    name: "in".to_string(),
                    ty: MtlxType::Matrix44,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 2,
                        output: None,
                    },
                }],
            },
        ],
        root: 3,
        back_root: None,
        material_name: "creatematrix_vec3_mat44".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("creatematrix vector3 matrix44 should compile");
    assert!(
        compiled
            .instructions
            .iter()
            .any(|inst| matches!(inst, Instruction::CreateMatrix4FromVec3 { .. }))
    );
}

#[test]
fn spec_creatematrix_vector3_matrix44_sets_mdl_w_components() {
    let regs = run(
        vec![
            Instruction::CreateMatrix4FromVec3 {
                dst: 0,
                rows_start: 0,
            },
            Instruction::ExtractRowVector {
                dst: 1,
                dim4: true,
                src: Operand::Reg(0),
                index: 0,
            },
            Instruction::ExtractRowVector {
                dst: 2,
                dim4: true,
                src: Operand::Reg(0),
                index: 3,
            },
        ],
        vec![
            Operand::Const(0),
            Operand::Const(1),
            Operand::Const(2),
            Operand::Const(3),
        ],
        vec![
            Value::Vector3(Vec3::new(1.0, 2.0, 3.0)),
            Value::Vector3(Vec3::new(0.0, 1.0, 0.0)),
            Value::Vector3(Vec3::new(0.0, 0.0, 1.0)),
            Value::Vector3(Vec3::new(4.0, 5.0, 6.0)),
        ],
        3,
    );
    assert!(
        regs[1]
            .as_color4()
            .abs_diff_eq(Vec4::new(1.0, 2.0, 3.0, 0.0), 1.0e-6)
    );
    assert!(
        regs[2]
            .as_color4()
            .abs_diff_eq(Vec4::new(4.0, 5.0, 6.0, 1.0), 1.0e-6)
    );
}

#[test]
fn spec_dot_boolean_output_compiles() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "dot".to_string(),
                },
                output_type: MtlxType::Boolean,
                inputs: vec![FlatNodeInput {
                    name: "in".to_string(),
                    ty: MtlxType::Boolean,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Value(MtlxValue::Boolean(true)),
                }],
            },
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "convert".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![FlatNodeInput {
                    name: "in".to_string(),
                    ty: MtlxType::Boolean,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 2,
                        output: None,
                    },
                }],
            },
        ],
        root: 3,
        back_root: None,
        material_name: "dot_boolean".to_string(),
    };

    super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("dot boolean should compile");
}

#[test]
fn spec_dot_matrix44_output_compiles() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "dot".to_string(),
                },
                output_type: MtlxType::Matrix44,
                inputs: vec![FlatNodeInput {
                    name: "in".to_string(),
                    ty: MtlxType::Matrix44,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Value(MtlxValue::Matrix44(glam::Mat4::IDENTITY)),
                }],
            },
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "determinant".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![FlatNodeInput {
                    name: "in".to_string(),
                    ty: MtlxType::Matrix44,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 2,
                        output: None,
                    },
                }],
            },
        ],
        root: 3,
        back_root: None,
        material_name: "dot_matrix44".to_string(),
    };

    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("dot matrix44 should compile");
    assert!(
        compiled
            .instructions
            .iter()
            .any(|inst| matches!(inst, Instruction::Determinant { dim4: true, .. }))
    );
}

#[test]
fn spec_floor_integer_output_compiles_to_integer_convert() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "floor".to_string(),
                },
                output_type: MtlxType::Integer,
                inputs: vec![FlatNodeInput {
                    name: "in".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Value(MtlxValue::Float(1.8)),
                }],
            },
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "convert".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![FlatNodeInput {
                    name: "in".to_string(),
                    ty: MtlxType::Integer,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 2,
                        output: None,
                    },
                }],
            },
        ],
        root: 3,
        back_root: None,
        material_name: "floor_integer".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("floor integer output should compile");

    assert!(approx_v3(
        eval_compiled_le(&compiled),
        Vec3::splat(1.0 / std::f32::consts::PI),
        1.0e-6
    ));
}

#[test]
fn spec_dotproduct_vector4_compile_preserves_input_dimension() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "dotproduct".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![
                    FlatNodeInput {
                        name: "in1".to_string(),
                        ty: MtlxType::Vector4,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Vector4(Vec4::ONE)),
                    },
                    FlatNodeInput {
                        name: "in2".to_string(),
                        ty: MtlxType::Vector4,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Vector4(Vec4::ONE)),
                    },
                ],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: "dot4".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("dotproduct vector4 should compile");

    assert!(approx_v3(
        eval_compiled_le(&compiled),
        Vec3::splat(4.0 / std::f32::consts::PI),
        1.0e-6
    ));
}

#[test]
fn spec_magnitude_vector2_compile_preserves_input_dimension() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "magnitude".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![FlatNodeInput {
                    name: "in".to_string(),
                    ty: MtlxType::Vector2,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Value(MtlxValue::Vector2(Vec2::ONE)),
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: "magnitude2".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("magnitude vector2 should compile");

    assert!(approx_v3(
        eval_compiled_le(&compiled),
        Vec3::splat(2.0_f32.sqrt() / std::f32::consts::PI),
        1.0e-6
    ));
}

#[test]
fn spec_transformnormal_default_is_z_axis() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "transformnormal".to_string(),
                },
                output_type: MtlxType::Vector3,
                inputs: vec![],
            },
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "magnitude".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![FlatNodeInput {
                    name: "in".to_string(),
                    ty: MtlxType::Vector3,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 2,
                        output: None,
                    },
                }],
            },
        ],
        root: 3,
        back_root: None,
        material_name: "transformnormal_default".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("transformnormal default graph should compile");

    let mut scratch = MtlxScratch::default();
    let handle = scratch.alloc_regs(compiled.num_registers as usize);
    super::runtime::run_instructions(&compiled, &dummy_sv(), &mut scratch, handle);
    let le = super::runtime::evaluate_le(&compiled, scratch.regs_slice(handle), &dummy_sv())
        .expect("surface_unlit should emit");
    assert!(approx_v3(
        le,
        Vec3::splat(1.0 / std::f32::consts::PI),
        1.0e-6
    ));
}

#[test]
fn spec_transform_space_defaults_are_object_to_world() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "transformpoint".to_string(),
                },
                output_type: MtlxType::Vector3,
                inputs: vec![],
            },
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "magnitude".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![FlatNodeInput {
                    name: "in".to_string(),
                    ty: MtlxType::Vector3,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 2,
                        output: None,
                    },
                }],
            },
        ],
        root: 3,
        back_root: None,
        material_name: "transform_defaults".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("transformpoint default graph should compile");
    assert!(compiled.instructions.iter().any(|instr| matches!(
        instr,
        Instruction::TransformPoint {
            from: GeomSpace::Object,
            to: GeomSpace::World,
            ..
        }
    )));
}

fn compile_transformpoint_graph(inputs: Vec<FlatNodeInput>) -> Result<CompiledMaterial, String> {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "transformpoint".to_string(),
                },
                output_type: MtlxType::Vector3,
                inputs,
            },
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "magnitude".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![FlatNodeInput {
                    name: "in".to_string(),
                    ty: MtlxType::Vector3,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 2,
                        output: None,
                    },
                }],
            },
        ],
        root: 3,
        back_root: None,
        material_name: "transformpoint_spaces".to_string(),
    };
    super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .map_err(|err| err.to_string())
}

#[test]
fn spec_transform_spaces_accept_empty_defaults_but_reject_dynamic_strings() {
    let compiled = compile_transformpoint_graph(vec![
        string_input("fromspace", ""),
        string_input("tospace", ""),
    ])
    .expect("empty transform spaces should use MDL object-to-world defaults");
    assert!(compiled.instructions.iter().any(|instr| matches!(
        instr,
        Instruction::TransformPoint {
            from: GeomSpace::Object,
            to: GeomSpace::World,
            ..
        }
    )));

    let err = compile_transformpoint_graph(vec![FlatNodeInput {
        name: "fromspace".to_string(),
        ty: MtlxType::String,
        colorspace: None,
        unit: None,
        unittype: None,
        binding: FlatInput::Node {
            node: 0,
            output: None,
        },
    }])
    .expect_err("dynamic fromspace should not silently fall back to object");
    assert!(err.contains("transformpoint.fromspace must be a static string value"));
}

#[test]
fn spec_transformcolor_spaces_reject_dynamic_strings() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "transformcolor".to_string(),
                },
                output_type: MtlxType::Color3,
                inputs: vec![FlatNodeInput {
                    name: "fromspace".to_string(),
                    ty: MtlxType::String,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission_color".to_string(),
                    ty: MtlxType::Color3,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: "transformcolor_dynamic_space".to_string(),
    };
    let err = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect_err("dynamic transformcolor fromspace should not silently become empty");
    assert!(
        err.to_string()
            .contains("transformcolor.fromspace must be a static string value")
    );
}

#[test]
fn spec_transformmatrix_matches_mdl_vector_append_rules() {
    let m3 = glam::Mat3::from_cols(
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(10.0, 20.0, 1.0),
    );
    let regs = run(
        vec![
            Instruction::LoadMat3Const { dst: 0, value: m3 },
            Instruction::TransformMatrix {
                dst: 1,
                out_ty: ValueType::Vector2,
                dim4: false,
                mat: Operand::Reg(0),
                v: Operand::Const(0),
            },
        ],
        Vec::new(),
        vec![Value::Vector2(Vec2::new(1.0, 2.0))],
        2,
    );
    assert!(
        regs[1]
            .as_vector2()
            .abs_diff_eq(Vec2::new(11.0, 22.0), 1.0e-6)
    );

    let m4 = glam::Mat4::from_cols(
        Vec4::new(1.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        Vec4::new(10.0, 20.0, 30.0, 1.0),
    );
    let regs = run(
        vec![
            Instruction::LoadMat4Const { dst: 0, value: m4 },
            Instruction::TransformMatrix {
                dst: 1,
                out_ty: ValueType::Vector3,
                dim4: true,
                mat: Operand::Reg(0),
                v: Operand::Const(0),
            },
        ],
        Vec::new(),
        vec![Value::Vector3(Vec3::new(1.0, 2.0, 3.0))],
        2,
    );
    let Value::Vector3(v) = regs[1] else {
        panic!("vector3M4 transformmatrix must write a vector3");
    };
    assert!(v.abs_diff_eq(Vec3::new(11.0, 22.0, 33.0), 1.0e-6));
}

#[test]
fn spec_transformmatrix_declared_matrix44_default_selects_m4_overload() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "transformmatrix".to_string(),
                },
                output_type: MtlxType::Vector3,
                inputs: vec![
                    FlatNodeInput {
                        name: "in".to_string(),
                        ty: MtlxType::Vector3,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Vector3(Vec3::new(1.0, 2.0, 3.0))),
                    },
                    FlatNodeInput {
                        name: "mat".to_string(),
                        ty: MtlxType::Matrix44,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Empty,
                    },
                ],
            },
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "extract".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![
                    FlatNodeInput {
                        name: "in".to_string(),
                        ty: MtlxType::Vector3,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Node {
                            node: 0,
                            output: None,
                        },
                    },
                    FlatNodeInput {
                        name: "index".to_string(),
                        ty: MtlxType::Integer,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Integer(0)),
                    },
                ],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 2,
                        output: None,
                    },
                }],
            },
        ],
        root: 3,
        back_root: None,
        material_name: "transformmatrix_m4_default".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("transformmatrix vector3M4 default should compile");
    assert!(compiled.instructions.iter().any(|instr| matches!(
        instr,
        Instruction::TransformMatrix {
            out_ty: ValueType::Vector3,
            dim4: true,
            ..
        }
    )));
}

#[test]
fn spec_matrix_transpose_output_compiles() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "transpose".to_string(),
                },
                output_type: MtlxType::Matrix33,
                inputs: vec![],
            },
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "determinant".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![FlatNodeInput {
                    name: "in".to_string(),
                    ty: MtlxType::Matrix33,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 2,
                        output: None,
                    },
                }],
            },
        ],
        root: 3,
        back_root: None,
        material_name: "matrix_transpose".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("matrix transpose graph should compile");

    assert!(
        compiled
            .instructions
            .iter()
            .any(|inst| matches!(inst, Instruction::Transpose { dim4: false, .. }))
    );
}

#[test]
fn spec_determinant_declared_matrix44_default_selects_m4_overload() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "determinant".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![FlatNodeInput {
                    name: "in".to_string(),
                    ty: MtlxType::Matrix44,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Empty,
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: "determinant_m4_default".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("determinant matrix44 default should compile");
    assert!(
        compiled
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instruction::Determinant { dim4: true, .. }))
    );
}

#[test]
fn spec_matrix_add_compiles_and_evaluates() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Combinator {
                    category: "add".to_string(),
                },
                output_type: MtlxType::Matrix33,
                inputs: vec![
                    FlatNodeInput {
                        name: "in1".to_string(),
                        ty: MtlxType::Matrix33,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Matrix33(glam::Mat3::IDENTITY)),
                    },
                    FlatNodeInput {
                        name: "in2".to_string(),
                        ty: MtlxType::Matrix33,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Matrix33(glam::Mat3::IDENTITY)),
                    },
                ],
            },
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "determinant".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![FlatNodeInput {
                    name: "in".to_string(),
                    ty: MtlxType::Matrix33,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 2,
                        output: None,
                    },
                }],
            },
        ],
        root: 3,
        back_root: None,
        material_name: "matrix_add".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("matrix add should compile");

    let mut scratch = MtlxScratch::default();
    let handle = scratch.alloc_regs(compiled.num_registers as usize);
    super::runtime::run_instructions(&compiled, &dummy_sv(), &mut scratch, handle);
    let le = super::runtime::evaluate_le(&compiled, scratch.regs_slice(handle), &dummy_sv())
        .expect("surface_unlit should emit");
    assert!(approx_v3(
        le,
        Vec3::splat(8.0 / std::f32::consts::PI),
        1.0e-5
    ));
}

#[test]
fn spec_matrix_divide_multiplies_by_inverse() {
    let regs = run(
        vec![
            Instruction::LoadMat3Const {
                dst: 0,
                value: glam::Mat3::from_diagonal(Vec3::splat(2.0)),
            },
            Instruction::LoadMat3Const {
                dst: 1,
                value: glam::Mat3::from_diagonal(Vec3::splat(2.0)),
            },
            Instruction::Arith {
                dst: 2,
                op: ArithOp::Divide,
                ty: ValueType::Matrix33,
                a: Operand::Reg(0),
                b: Operand::Reg(1),
            },
            Instruction::Determinant {
                dst: 3,
                dim4: false,
                src: Operand::Reg(2),
            },
        ],
        Vec::new(),
        Vec::new(),
        4,
    );
    assert!(approx_f(regs[3].as_float(), 1.0, 1.0e-6));
}

#[test]
fn spec_add_bsdf_matches_mdl_equal_mix() {
    let compiled = CompiledMaterial {
        instructions: vec![],
        operand_pool: vec![],
        value_pool: vec![],
        opacity_instructions: Vec::new(),
        opacity_operand_pool: Vec::new(),
        opacity_closure_nodes: Vec::new(),
        opacity_num_registers: 0,
        num_registers: 0,
        closure_nodes: vec![
            ClosureNode::Add {
                a: 1,
                b: 2,
                kind: super::compiled::ClosureKind::Bsdf,
            },
            ClosureNode::BurleyDiffuse {
                weight: super::compiled::ParamRef::Float(1.0),
                color: super::compiled::ParamRef::Color3(Vec3::X),
                roughness: super::compiled::ParamRef::Float(0.0),
                normal: None,
            },
            ClosureNode::BurleyDiffuse {
                weight: super::compiled::ParamRef::Float(1.0),
                color: super::compiled::ParamRef::Color3(Vec3::Z),
                roughness: super::compiled::ParamRef::Float(0.0),
                normal: None,
            },
        ],
        root: 0,
        passthrough: false,
        max_emission: 0.0,
        may_emit: false,
        has_opacity_test: false,
        thin_walled: false,
        sheen_lut: None,
        mtlx_dielectric_lut: None,
        mtlx_generalized_schlick_lut: None,
    };
    let sv = dummy_sv();
    let f = super::runtime::eval_closure(&compiled, &[], &sv, Vec3::Z, Vec3::Z);
    let expected = Vec3::new(0.5, 0.0, 0.5) / std::f32::consts::PI;
    assert!(approx_v3(f, expected, 1.0e-6));

    let albedo = super::runtime::directional_albedo_closure(&compiled, &[], &sv, Vec3::Z);
    assert!(approx_v3(albedo, Vec3::ONE, 1.0e-6));
}

#[test]
fn spec_add_edf_matches_mdl_unbounded_shape_and_intensity_add() {
    let compiled = CompiledMaterial {
        instructions: vec![],
        operand_pool: vec![],
        value_pool: vec![],
        opacity_instructions: Vec::new(),
        opacity_operand_pool: Vec::new(),
        opacity_closure_nodes: Vec::new(),
        opacity_num_registers: 0,
        num_registers: 0,
        closure_nodes: vec![
            ClosureNode::Surface {
                bsdf: 4,
                edf: 1,
                opacity: ParamRef::Float(1.0),
                thin_walled: false,
            },
            ClosureNode::Add {
                a: 2,
                b: 3,
                kind: ClosureKind::Edf,
            },
            ClosureNode::UniformEdf {
                color: ParamRef::Color3(Vec3::splat(2.0)),
            },
            ClosureNode::UniformEdf {
                color: ParamRef::Color3(Vec3::splat(4.0)),
            },
            ClosureNode::Zero,
        ],
        root: 0,
        passthrough: false,
        max_emission: 0.0,
        may_emit: false,
        has_opacity_test: false,
        thin_walled: false,
        sheen_lut: None,
        mtlx_dielectric_lut: None,
        mtlx_generalized_schlick_lut: None,
    };

    let le = super::runtime::evaluate_le(&compiled, &[], &dummy_sv()).expect("surface should emit");
    assert!(approx_v3(
        le,
        Vec3::splat(12.0 / std::f32::consts::PI),
        1.0e-6
    ));
}

#[test]
fn spec_add_edf_max_emission_matches_mdl_shape_intensity_bound() {
    let value_input = |name: &str, value: MtlxValue| FlatNodeInput {
        name: name.to_string(),
        ty: MtlxType::Color3,
        colorspace: None,
        unit: None,
        unittype: None,
        binding: FlatInput::Value(value),
    };
    let node_input = |name: &str, ty: MtlxType, node: usize| FlatNodeInput {
        name: name.to_string(),
        ty,
        colorspace: None,
        unit: None,
        unittype: None,
        binding: FlatInput::Node {
            node: node as u32,
            output: None,
        },
    };
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Shading {
                    category: "uniform_edf".to_string(),
                },
                output_type: MtlxType::Edf,
                inputs: vec![value_input("color", MtlxValue::Color3(Vec3::splat(2.0)))],
            },
            FlatNode {
                kind: FlatNodeKind::Shading {
                    category: "uniform_edf".to_string(),
                },
                output_type: MtlxType::Edf,
                inputs: vec![value_input("color", MtlxValue::Color3(Vec3::splat(4.0)))],
            },
            FlatNode {
                kind: FlatNodeKind::Combinator {
                    category: "add".to_string(),
                },
                output_type: MtlxType::Edf,
                inputs: vec![
                    node_input("in1", MtlxType::Edf, 0),
                    node_input("in2", MtlxType::Edf, 1),
                ],
            },
            FlatNode {
                kind: FlatNodeKind::Surface,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![node_input("edf", MtlxType::Edf, 2)],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![node_input("surfaceshader", MtlxType::Surfaceshader, 3)],
            },
        ],
        root: 4,
        back_root: None,
        material_name: "add_edf_max_emission".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("add_edf graph should compile");

    assert!(compiled.may_emit);
    assert!(approx_f(compiled.max_emission, 12.0, 1.0e-6));
}

#[test]
fn spec_mix_bsdf_clamps_mix_like_mdl() {
    let compiled = CompiledMaterial {
        instructions: vec![],
        operand_pool: vec![],
        value_pool: vec![],
        opacity_instructions: Vec::new(),
        opacity_operand_pool: Vec::new(),
        opacity_closure_nodes: Vec::new(),
        opacity_num_registers: 0,
        num_registers: 0,
        closure_nodes: vec![
            ClosureNode::Mix {
                bg: 1,
                fg: 2,
                mix: super::compiled::ParamRef::Float(2.0),
                kind: super::compiled::ClosureKind::Bsdf,
            },
            ClosureNode::BurleyDiffuse {
                weight: super::compiled::ParamRef::Float(1.0),
                color: super::compiled::ParamRef::Color3(Vec3::Z),
                roughness: super::compiled::ParamRef::Float(0.0),
                normal: None,
            },
            ClosureNode::BurleyDiffuse {
                weight: super::compiled::ParamRef::Float(1.0),
                color: super::compiled::ParamRef::Color3(Vec3::X),
                roughness: super::compiled::ParamRef::Float(0.0),
                normal: None,
            },
        ],
        root: 0,
        passthrough: false,
        max_emission: 0.0,
        may_emit: false,
        has_opacity_test: false,
        thin_walled: false,
        sheen_lut: None,
        mtlx_dielectric_lut: None,
        mtlx_generalized_schlick_lut: None,
    };
    let sv = dummy_sv();
    let f = super::runtime::eval_closure(&compiled, &[], &sv, Vec3::Z, Vec3::Z);
    assert!(approx_v3(f, Vec3::X / std::f32::consts::PI, 1.0e-6));

    let albedo = super::runtime::directional_albedo_closure(&compiled, &[], &sv, Vec3::Z);
    assert!(approx_v3(albedo, Vec3::ONE, 1.0e-6));
}

#[test]
fn spec_conical_edf_uses_normal_socket() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Shading {
                    category: "conical_edf".to_string(),
                },
                output_type: MtlxType::Edf,
                inputs: vec![
                    FlatNodeInput {
                        name: "inner_angle".to_string(),
                        ty: MtlxType::Float,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Float(0.0)),
                    },
                    FlatNodeInput {
                        name: "outer_angle".to_string(),
                        ty: MtlxType::Float,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Float(0.0)),
                    },
                    FlatNodeInput {
                        name: "normal".to_string(),
                        ty: MtlxType::Vector3,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Vector3(Vec3::X)),
                    },
                ],
            },
            FlatNode {
                kind: FlatNodeKind::Surface,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "edf".to_string(),
                    ty: MtlxType::Edf,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: "conical_edf_normal".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("conical_edf graph should compile");

    let mut scratch = MtlxScratch::default();
    let handle = scratch.alloc_regs(compiled.num_registers as usize);
    super::runtime::run_instructions(&compiled, &dummy_sv(), &mut scratch, handle);
    let le = super::runtime::evaluate_le(&compiled, scratch.regs_slice(handle), &dummy_sv());
    assert!(le.is_none());
}

#[test]
fn spec_heighttonormal_constant_height_is_flat() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "heighttonormal".to_string(),
                },
                output_type: MtlxType::Vector3,
                inputs: vec![
                    FlatNodeInput {
                        name: "in".to_string(),
                        ty: MtlxType::Float,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Float(2.0)),
                    },
                    FlatNodeInput {
                        name: "scale".to_string(),
                        ty: MtlxType::Float,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Float(5.0)),
                    },
                ],
            },
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "extract".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![
                    FlatNodeInput {
                        name: "in".to_string(),
                        ty: MtlxType::Vector3,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Node {
                            node: 0,
                            output: None,
                        },
                    },
                    FlatNodeInput {
                        name: "index".to_string(),
                        ty: MtlxType::Integer,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Integer(0)),
                    },
                ],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 2,
                        output: None,
                    },
                }],
            },
        ],
        root: 3,
        back_root: None,
        material_name: "heighttonormal_constant".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("constant heighttonormal should compile");
    let mut scratch = MtlxScratch::default();
    let handle = scratch.alloc_regs(compiled.num_registers as usize);
    super::runtime::run_instructions(&compiled, &dummy_sv(), &mut scratch, handle);
    let le = super::runtime::evaluate_le(&compiled, scratch.regs_slice(handle), &dummy_sv())
        .expect("surface_unlit should emit");
    assert!(approx_v3(
        le,
        Vec3::splat(0.5 / std::f32::consts::PI),
        1.0e-6
    ));
}

#[test]
fn spec_heighttonormal_dynamic_height_errors() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "add".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![
                    FlatNodeInput {
                        name: "in1".to_string(),
                        ty: MtlxType::Float,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Float(1.0)),
                    },
                    FlatNodeInput {
                        name: "in2".to_string(),
                        ty: MtlxType::Float,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Float(1.0)),
                    },
                ],
            },
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "heighttonormal".to_string(),
                },
                output_type: MtlxType::Vector3,
                inputs: vec![FlatNodeInput {
                    name: "in".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission_color".to_string(),
                    ty: MtlxType::Color3,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 2,
                        output: None,
                    },
                }],
            },
        ],
        root: 3,
        back_root: None,
        material_name: "heighttonormal_dynamic".to_string(),
    };

    let err = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .unwrap_err();
    assert!(format!("{err:?}").contains("heighttonormal"));
}

#[test]
fn spec_bump_dynamic_height_errors() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "add".to_string(),
                },
                output_type: MtlxType::Float,
                inputs: vec![
                    FlatNodeInput {
                        name: "in1".to_string(),
                        ty: MtlxType::Float,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Float(1.0)),
                    },
                    FlatNodeInput {
                        name: "in2".to_string(),
                        ty: MtlxType::Float,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Float(1.0)),
                    },
                ],
            },
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "bump".to_string(),
                },
                output_type: MtlxType::Vector3,
                inputs: vec![FlatNodeInput {
                    name: "height".to_string(),
                    ty: MtlxType::Float,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission_color".to_string(),
                    ty: MtlxType::Color3,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 2,
                        output: None,
                    },
                }],
            },
        ],
        root: 3,
        back_root: None,
        material_name: "bump_dynamic".to_string(),
    };

    let err = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .unwrap_err();
    assert!(format!("{err:?}").contains("bump"));
}

#[test]
fn spec_heighttonormal_invalid_scale_errors() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "heighttonormal".to_string(),
                },
                output_type: MtlxType::Vector3,
                inputs: vec![
                    FlatNodeInput {
                        name: "in".to_string(),
                        ty: MtlxType::Float,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::Float(0.0)),
                    },
                    FlatNodeInput {
                        name: "scale".to_string(),
                        ty: MtlxType::String,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Value(MtlxValue::String("bad".to_string())),
                    },
                ],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission_color".to_string(),
                    ty: MtlxType::Color3,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: "heighttonormal_bad_scale".to_string(),
    };
    let err = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .unwrap_err();
    assert!(format!("{err:?}").contains("bad"));
}

#[test]
fn spec_blur_errors_instead_of_passthrough() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "blur".to_string(),
                },
                output_type: MtlxType::Color3,
                inputs: vec![FlatNodeInput {
                    name: "in".to_string(),
                    ty: MtlxType::Color3,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Value(MtlxValue::Color3(Vec3::ONE)),
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission_color".to_string(),
                    ty: MtlxType::Color3,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: "blur_unsupported".to_string(),
    };

    let err = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .unwrap_err();
    assert!(format!("{err:?}").contains("blur"));
}

#[test]
fn spec_multi_output_pattern_requires_valid_output_name() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "artistic_ior".to_string(),
                },
                output_type: MtlxType::Color3,
                inputs: vec![],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission_color".to_string(),
                    ty: MtlxType::Color3,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: "multi_output_missing".to_string(),
    };
    let err = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("artistic_ior requires output"));

    let mut bad = graph;
    if let FlatInput::Node { output, .. } = &mut bad.nodes[1].inputs[0].binding {
        *output = Some("bad".to_string());
    }
    let err = super::compile::compile(
        &bad,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("artistic_ior output `bad`"));
}

#[test]
fn spec_artistic_ior_compile_defaults_match_nodedef() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Pattern {
                    category: "artistic_ior".to_string(),
                },
                output_type: MtlxType::Color3,
                inputs: vec![],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceUnlit,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "emission_color".to_string(),
                    ty: MtlxType::Color3,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: Some("ior".to_string()),
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: "artistic_ior_defaults".to_string(),
    };
    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("compile artistic_ior");
    let refl = Vec3::new(0.944, 0.776, 0.373);
    let edge = Vec3::new(0.998, 0.981, 0.751);
    let (ior, _) = super::runtime::artistic_ior(refl, edge);
    assert!(approx_v3(
        eval_compiled_le(&compiled),
        ior * (1.0 / std::f32::consts::PI),
        1.0e-6
    ));
}

#[test]
fn spec_randomfloat_float_matches_nodegraph_cellnoise() {
    let regs = run(
        vec![Instruction::RandomFloat {
            dst: 0,
            integer_input: false,
            operands_start: 0,
        }],
        vec![
            Operand::Const(0),
            Operand::Const(1),
            Operand::Const(2),
            Operand::Const(3),
        ],
        vec![
            Value::Float(0.375),
            Value::Integer(7),
            Value::Float(-2.0),
            Value::Float(5.0),
        ],
        1,
    );
    let noise = cellnoise2d(Vec2::new(0.375 * 4096.0, 7.0));
    let expected = -2.0 + 7.0 * noise;
    assert!(approx_f(regs[0].as_float(), expected, 1.0e-6));
}

#[test]
fn spec_randomfloat_integer_matches_nodegraph_cellnoise() {
    let regs = run(
        vec![Instruction::RandomFloat {
            dst: 0,
            integer_input: true,
            operands_start: 0,
        }],
        vec![
            Operand::Const(0),
            Operand::Const(1),
            Operand::Const(2),
            Operand::Const(3),
        ],
        vec![
            Value::Integer(9),
            Value::Integer(3),
            Value::Float(10.0),
            Value::Float(12.0),
        ],
        1,
    );
    let noise = cellnoise2d(Vec2::new(9.0, 3.0));
    let expected = 10.0 + 2.0 * noise;
    assert!(approx_f(regs[0].as_float(), expected, 1.0e-6));
}

#[test]
fn spec_randomcolor_matches_nodegraph_seed_offsets() {
    let regs = run(
        vec![Instruction::RandomColor {
            dst: 0,
            operands_start: 0,
        }],
        vec![
            Operand::Const(0),
            Operand::Const(1),
            Operand::Const(2),
            Operand::Const(3),
            Operand::Const(4),
            Operand::Const(5),
            Operand::Const(6),
            Operand::Const(7),
        ],
        vec![
            Value::Float(0.25),
            Value::Integer(5),
            Value::Float(0.1),
            Value::Float(0.9),
            Value::Float(0.2),
            Value::Float(0.8),
            Value::Float(0.3),
            Value::Float(0.7),
        ],
        1,
    );
    let x = 0.25 * 4096.0;
    let hue = 0.1 + 0.8 * cellnoise2d(Vec2::new(x, (5.0_f32 + 413.3).ceil()));
    let sat = 0.2 + 0.6 * cellnoise2d(Vec2::new(x, (5.0_f32 + 1522.4).ceil()));
    let val = 0.3 + 0.4 * cellnoise2d(Vec2::new(x, (5.0_f32 + 1813.8).ceil()));
    let expected = hsv_to_rgb(hue, sat, val);
    assert!(approx_v3(regs[0].as_color3(), expected, 1.0e-6));
}

#[test]
fn spec_checkerboard_subtracts_uvoffset() {
    let c1 = Vec3::new(0.1, 0.2, 0.3);
    let c2 = Vec3::new(0.8, 0.7, 0.6);
    let regs = run(
        vec![Instruction::Checkerboard {
            dst: 0,
            color1: Operand::Const(0),
            color2: Operand::Const(1),
            uvtiling: Operand::Const(2),
            uvoffset: Operand::Const(3),
            texcoord: Operand::Const(4),
        }],
        Vec::new(),
        vec![
            Value::Color3(c1),
            Value::Color3(c2),
            Value::Vector2(Vec2::new(2.0, 1.0)),
            Value::Vector2(Vec2::new(0.7, 0.0)),
            Value::Vector2(Vec2::new(0.6, 0.0)),
        ],
        1,
    );
    assert!(approx_v3(regs[0].as_color3(), c1, 1.0e-6));
}

#[test]
fn spec_dielectric_bsdf_compile_defaults_match_1394() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Shading {
                    category: "dielectric_bsdf".to_string(),
                },
                output_type: MtlxType::Bsdf,
                inputs: vec![],
            },
            FlatNode {
                kind: FlatNodeKind::Surface,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "bsdf".to_string(),
                    ty: MtlxType::Bsdf,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: "dielectric_defaults".to_string(),
    };

    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("dielectric_bsdf defaults should compile");

    let ClosureNode::Surface { bsdf, .. } = &compiled.closure_nodes[compiled.root as usize] else {
        panic!("root should be a surface closure");
    };
    let ClosureNode::Dielectric {
        weight,
        tint,
        ior,
        roughness,
        scatter_mode,
        thinfilm_thickness,
        thinfilm_ior,
        normal,
        tangent,
    } = &compiled.closure_nodes[*bsdf as usize]
    else {
        panic!("surface bsdf should be dielectric_bsdf");
    };
    assert!(matches!(weight, super::compiled::ParamRef::Float(v) if approx_f(*v, 1.0, 1.0e-6)));
    assert!(
        matches!(tint, super::compiled::ParamRef::Color3(v) if approx_v3(*v, Vec3::ONE, 1.0e-6))
    );
    assert!(matches!(ior, super::compiled::ParamRef::Float(v) if approx_f(*v, 1.5, 1.0e-6)));
    assert!(
        matches!(roughness, super::compiled::ParamRef::Vector2(v) if v.abs_diff_eq(Vec2::splat(0.05), 1.0e-6))
    );
    assert_eq!(*scatter_mode, crate::bsdf::mtlx::ScatterMode::Reflection);
    assert!(
        matches!(thinfilm_thickness, super::compiled::ParamRef::Float(v) if approx_f(*v, 0.0, 1.0e-6))
    );
    assert!(
        matches!(thinfilm_ior, super::compiled::ParamRef::Float(v) if approx_f(*v, 1.5, 1.0e-6))
    );
    assert!(normal.is_none());
    assert!(tangent.is_none());
}

#[test]
fn spec_dielectric_bsdf_invalid_scatter_mode_errors() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Shading {
                    category: "dielectric_bsdf".to_string(),
                },
                output_type: MtlxType::Bsdf,
                inputs: vec![FlatNodeInput {
                    name: "scatter_mode".to_string(),
                    ty: MtlxType::String,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Value(MtlxValue::String("bad".to_string())),
                }],
            },
            FlatNode {
                kind: FlatNodeKind::Surface,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "bsdf".to_string(),
                    ty: MtlxType::Bsdf,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: "dielectric_bad_scatter".to_string(),
    };

    let err = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect_err("invalid dielectric_bsdf scatter_mode should error");
    assert!(err.to_string().contains("scatter_mode"));
}

#[test]
fn spec_conductor_bsdf_compile_defaults_match_1394() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Shading {
                    category: "conductor_bsdf".to_string(),
                },
                output_type: MtlxType::Bsdf,
                inputs: vec![],
            },
            FlatNode {
                kind: FlatNodeKind::Surface,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "bsdf".to_string(),
                    ty: MtlxType::Bsdf,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: "conductor_defaults".to_string(),
    };

    let compiled = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect("conductor_bsdf defaults should compile");

    let ClosureNode::Surface { bsdf, .. } = &compiled.closure_nodes[compiled.root as usize] else {
        panic!("root should be a surface closure");
    };
    let ClosureNode::Conductor {
        weight,
        ior,
        extinction,
        roughness,
        thinfilm_thickness,
        thinfilm_ior,
        normal,
        tangent,
    } = &compiled.closure_nodes[*bsdf as usize]
    else {
        panic!("surface bsdf should be conductor_bsdf");
    };
    assert!(matches!(weight, super::compiled::ParamRef::Float(v) if approx_f(*v, 1.0, 1.0e-6)));
    assert!(
        matches!(ior, super::compiled::ParamRef::Color3(v) if approx_v3(*v, Vec3::new(0.183, 0.421, 1.373), 1.0e-6))
    );
    assert!(
        matches!(extinction, super::compiled::ParamRef::Color3(v) if approx_v3(*v, Vec3::new(3.424, 2.346, 1.770), 1.0e-6))
    );
    assert!(
        matches!(roughness, super::compiled::ParamRef::Vector2(v) if v.abs_diff_eq(Vec2::splat(0.05), 1.0e-6))
    );
    assert!(
        matches!(thinfilm_thickness, super::compiled::ParamRef::Float(v) if approx_f(*v, 0.0, 1.0e-6))
    );
    assert!(
        matches!(thinfilm_ior, super::compiled::ParamRef::Float(v) if approx_f(*v, 1.5, 1.0e-6))
    );
    assert!(normal.is_none());
    assert!(tangent.is_none());
}

#[test]
fn spec_conductor_bsdf_invalid_distribution_errors() {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Shading {
                    category: "conductor_bsdf".to_string(),
                },
                output_type: MtlxType::Bsdf,
                inputs: vec![FlatNodeInput {
                    name: "distribution".to_string(),
                    ty: MtlxType::String,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Value(MtlxValue::String("beckmann".to_string())),
                }],
            },
            FlatNode {
                kind: FlatNodeKind::Surface,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "bsdf".to_string(),
                    ty: MtlxType::Bsdf,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: "conductor_bad_distribution".to_string(),
    };

    let err = super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
    .expect_err("invalid conductor_bsdf distribution should error");
    assert!(err.to_string().contains("distribution"));
}

fn compile_single_bsdf_node(
    category: &str,
    inputs: Vec<FlatNodeInput>,
    material_name: &str,
) -> Result<CompiledMaterial, super::compile::CompileError> {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Shading {
                    category: category.to_string(),
                },
                output_type: MtlxType::Bsdf,
                inputs,
            },
            FlatNode {
                kind: FlatNodeKind::Surface,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "bsdf".to_string(),
                    ty: MtlxType::Bsdf,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: material_name.to_string(),
    };

    super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
}

fn compile_single_edf_node(
    category: &str,
    inputs: Vec<FlatNodeInput>,
    material_name: &str,
) -> Result<CompiledMaterial, super::compile::CompileError> {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Shading {
                    category: category.to_string(),
                },
                output_type: MtlxType::Edf,
                inputs,
            },
            FlatNode {
                kind: FlatNodeKind::Surface,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "edf".to_string(),
                    ty: MtlxType::Edf,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 0,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 1,
                        output: None,
                    },
                }],
            },
        ],
        root: 2,
        back_root: None,
        material_name: material_name.to_string(),
    };

    super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
}

fn compile_vdf_layer_base(
    category: &str,
    inputs: Vec<FlatNodeInput>,
    material_name: &str,
) -> Result<CompiledMaterial, super::compile::CompileError> {
    let graph = FlatGraph {
        nodes: vec![
            FlatNode {
                kind: FlatNodeKind::Shading {
                    category: "burley_diffuse_bsdf".to_string(),
                },
                output_type: MtlxType::Bsdf,
                inputs: vec![],
            },
            FlatNode {
                kind: FlatNodeKind::Shading {
                    category: category.to_string(),
                },
                output_type: MtlxType::Vdf,
                inputs,
            },
            FlatNode {
                kind: FlatNodeKind::Combinator {
                    category: "layer".to_string(),
                },
                output_type: MtlxType::Bsdf,
                inputs: vec![
                    FlatNodeInput {
                        name: "top".to_string(),
                        ty: MtlxType::Bsdf,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Node {
                            node: 0,
                            output: None,
                        },
                    },
                    FlatNodeInput {
                        name: "base".to_string(),
                        ty: MtlxType::Vdf,
                        colorspace: None,
                        unit: None,
                        unittype: None,
                        binding: FlatInput::Node {
                            node: 1,
                            output: None,
                        },
                    },
                ],
            },
            FlatNode {
                kind: FlatNodeKind::Surface,
                output_type: MtlxType::Surfaceshader,
                inputs: vec![FlatNodeInput {
                    name: "bsdf".to_string(),
                    ty: MtlxType::Bsdf,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 2,
                        output: None,
                    },
                }],
            },
            FlatNode {
                kind: FlatNodeKind::SurfaceMaterial,
                output_type: MtlxType::Material,
                inputs: vec![FlatNodeInput {
                    name: "surfaceshader".to_string(),
                    ty: MtlxType::Surfaceshader,
                    colorspace: None,
                    unit: None,
                    unittype: None,
                    binding: FlatInput::Node {
                        node: 3,
                        output: None,
                    },
                }],
            },
        ],
        root: 4,
        back_root: None,
        material_name: material_name.to_string(),
    };

    super::compile::compile(
        &graph,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
}

#[test]
fn spec_bsdf_string_enums_reject_dynamic_values() {
    let dynamic_scatter = FlatNodeInput {
        name: "scatter_mode".to_string(),
        ty: MtlxType::String,
        colorspace: None,
        unit: None,
        unittype: None,
        binding: FlatInput::Node {
            node: 0,
            output: None,
        },
    };
    let err = compile_single_bsdf_node(
        "dielectric_bsdf",
        vec![dynamic_scatter],
        "dielectric_dynamic_scatter",
    )
    .expect_err("dynamic dielectric scatter mode should not silently fall back to R");
    assert!(
        err.to_string()
            .contains("dielectric_bsdf.scatter_mode must be a static string value")
    );

    let dynamic_mode = FlatNodeInput {
        name: "mode".to_string(),
        ty: MtlxType::String,
        colorspace: None,
        unit: None,
        unittype: None,
        binding: FlatInput::Node {
            node: 0,
            output: None,
        },
    };
    let err = compile_single_bsdf_node("sheen_bsdf", vec![dynamic_mode], "sheen_dynamic_mode")
        .expect_err("dynamic sheen mode should not silently fall back to conty_kulla");
    assert!(
        err.to_string()
            .contains("sheen_bsdf.mode must be a static string value")
    );
}

#[test]
fn spec_generalized_schlick_bsdf_compile_defaults_match_1394() {
    let compiled = compile_single_bsdf_node("generalized_schlick_bsdf", vec![], "gs_defaults")
        .expect("generalized_schlick_bsdf defaults should compile");

    let ClosureNode::Surface { bsdf, .. } = &compiled.closure_nodes[compiled.root as usize] else {
        panic!("root should be a surface closure");
    };
    let ClosureNode::GeneralizedSchlick {
        weight,
        color0,
        color82,
        color90,
        exponent,
        roughness,
        scatter_mode,
        thinfilm_thickness,
        thinfilm_ior,
        normal,
        tangent,
    } = &compiled.closure_nodes[*bsdf as usize]
    else {
        panic!("surface bsdf should be generalized_schlick_bsdf");
    };
    assert!(matches!(weight, super::compiled::ParamRef::Float(v) if approx_f(*v, 1.0, 1.0e-6)));
    assert!(
        matches!(color0, super::compiled::ParamRef::Color3(v) if approx_v3(*v, Vec3::ONE, 1.0e-6))
    );
    assert!(
        matches!(color82, super::compiled::ParamRef::Color3(v) if approx_v3(*v, Vec3::ONE, 1.0e-6))
    );
    assert!(
        matches!(color90, super::compiled::ParamRef::Color3(v) if approx_v3(*v, Vec3::ONE, 1.0e-6))
    );
    assert!(matches!(exponent, super::compiled::ParamRef::Float(v) if approx_f(*v, 5.0, 1.0e-6)));
    assert!(
        matches!(roughness, super::compiled::ParamRef::Vector2(v) if v.abs_diff_eq(Vec2::splat(0.05), 1.0e-6))
    );
    assert_eq!(*scatter_mode, crate::bsdf::mtlx::ScatterMode::Reflection);
    assert!(
        matches!(thinfilm_thickness, super::compiled::ParamRef::Float(v) if approx_f(*v, 0.0, 1.0e-6))
    );
    assert!(
        matches!(thinfilm_ior, super::compiled::ParamRef::Float(v) if approx_f(*v, 1.5, 1.0e-6))
    );
    assert!(normal.is_none());
    assert!(tangent.is_none());
}

#[test]
fn spec_generalized_schlick_bsdf_invalid_scatter_mode_errors() {
    let err = compile_single_bsdf_node(
        "generalized_schlick_bsdf",
        vec![FlatNodeInput {
            name: "scatter_mode".to_string(),
            ty: MtlxType::String,
            colorspace: None,
            unit: None,
            unittype: None,
            binding: FlatInput::Value(MtlxValue::String("bad".to_string())),
        }],
        "gs_bad_scatter",
    )
    .expect_err("invalid generalized_schlick_bsdf scatter_mode should error");
    assert!(err.to_string().contains("scatter_mode"));
}

#[test]
fn spec_subsurface_bsdf_warns_and_falls_back_to_burley_diffuse() {
    let compiled = compile_single_bsdf_node("subsurface_bsdf", vec![], "subsurface_fallback")
        .expect("subsurface_bsdf fallback should compile");

    let ClosureNode::Surface { bsdf, .. } = &compiled.closure_nodes[compiled.root as usize] else {
        panic!("root should be a surface closure");
    };
    let ClosureNode::BurleyDiffuse {
        weight,
        color,
        roughness,
        normal,
    } = &compiled.closure_nodes[*bsdf as usize]
    else {
        panic!("subsurface fallback should be burley_diffuse_bsdf");
    };
    assert!(matches!(weight, super::compiled::ParamRef::Float(v) if approx_f(*v, 1.0, 1.0e-6)));
    assert!(
        matches!(color, super::compiled::ParamRef::Color3(v) if approx_v3(*v, Vec3::splat(0.18), 1.0e-6))
    );
    assert!(matches!(roughness, super::compiled::ParamRef::Float(v) if approx_f(*v, 0.5, 1.0e-6)));
    assert!(normal.is_none());
}

#[test]
fn spec_subsurface_bsdf_ignored_inputs_are_type_checked() {
    let err = compile_single_bsdf_node(
        "subsurface_bsdf",
        vec![FlatNodeInput {
            name: "radius".to_string(),
            ty: MtlxType::String,
            colorspace: None,
            unit: None,
            unittype: None,
            binding: FlatInput::Value(MtlxValue::String("bad".to_string())),
        }],
        "subsurface_bad_radius",
    )
    .expect_err("invalid subsurface_bsdf radius should error");
    assert!(err.to_string().contains("radius"));

    let err = compile_single_bsdf_node(
        "subsurface_bsdf",
        vec![FlatNodeInput {
            name: "anisotropy".to_string(),
            ty: MtlxType::String,
            colorspace: None,
            unit: None,
            unittype: None,
            binding: FlatInput::Value(MtlxValue::String("bad".to_string())),
        }],
        "subsurface_bad_anisotropy",
    )
    .expect_err("invalid subsurface_bsdf anisotropy should error");
    assert!(err.to_string().contains("anisotropy"));
}

#[test]
fn spec_sheen_bsdf_compile_defaults_match_1394() {
    let compiled = compile_single_bsdf_node("sheen_bsdf", vec![], "sheen_defaults")
        .expect("sheen_bsdf defaults should compile");

    let ClosureNode::Surface { bsdf, .. } = &compiled.closure_nodes[compiled.root as usize] else {
        panic!("root should be a surface closure");
    };
    let ClosureNode::Sheen {
        weight,
        color,
        roughness,
        mode,
        normal,
    } = &compiled.closure_nodes[*bsdf as usize]
    else {
        panic!("surface bsdf should be sheen_bsdf");
    };
    assert!(matches!(weight, super::compiled::ParamRef::Float(v) if approx_f(*v, 1.0, 1.0e-6)));
    assert!(
        matches!(color, super::compiled::ParamRef::Color3(v) if approx_v3(*v, Vec3::ONE, 1.0e-6))
    );
    assert!(matches!(roughness, super::compiled::ParamRef::Float(v) if approx_f(*v, 0.3, 1.0e-6)));
    assert_eq!(*mode, crate::bsdf::mtlx::SheenMode::ContyKulla);
    assert!(normal.is_none());
}

#[test]
fn spec_sheen_bsdf_invalid_mode_errors() {
    let err = compile_single_bsdf_node(
        "sheen_bsdf",
        vec![FlatNodeInput {
            name: "mode".to_string(),
            ty: MtlxType::String,
            colorspace: None,
            unit: None,
            unittype: None,
            binding: FlatInput::Value(MtlxValue::String("bad".to_string())),
        }],
        "sheen_bad_mode",
    )
    .expect_err("invalid sheen_bsdf mode should error");
    assert!(err.to_string().contains("mode"));
}

#[test]
fn spec_chiang_hair_bsdf_compile_defaults_match_1394() {
    let compiled = compile_single_bsdf_node("chiang_hair_bsdf", vec![], "chiang_defaults")
        .expect("chiang_hair_bsdf defaults should compile");

    let ClosureNode::Surface { bsdf, .. } = &compiled.closure_nodes[compiled.root as usize] else {
        panic!("root should be a surface closure");
    };
    let ClosureNode::ChiangHair {
        tint_r,
        tint_tt,
        tint_trt,
        absorption,
        ior,
        roughness_r,
        roughness_tt,
        roughness_trt,
        cuticle_angle,
        normal,
        curve_direction,
    } = &compiled.closure_nodes[*bsdf as usize]
    else {
        panic!("surface bsdf should be chiang_hair_bsdf");
    };
    assert!(
        matches!(tint_r, super::compiled::ParamRef::Color3(v) if approx_v3(*v, Vec3::ONE, 1.0e-6))
    );
    assert!(
        matches!(tint_tt, super::compiled::ParamRef::Color3(v) if approx_v3(*v, Vec3::ONE, 1.0e-6))
    );
    assert!(
        matches!(tint_trt, super::compiled::ParamRef::Color3(v) if approx_v3(*v, Vec3::ONE, 1.0e-6))
    );
    assert!(
        matches!(absorption, super::compiled::ParamRef::Color3(v) if approx_v3(*v, Vec3::ZERO, 1.0e-6))
    );
    assert!(matches!(ior, super::compiled::ParamRef::Float(v) if approx_f(*v, 1.55, 1.0e-6)));
    assert!(
        matches!(roughness_r, super::compiled::ParamRef::Vector2(v) if v.abs_diff_eq(Vec2::splat(0.1), 1.0e-6))
    );
    assert!(
        matches!(roughness_tt, super::compiled::ParamRef::Vector2(v) if v.abs_diff_eq(Vec2::splat(0.05), 1.0e-6))
    );
    assert!(
        matches!(roughness_trt, super::compiled::ParamRef::Vector2(v) if v.abs_diff_eq(Vec2::splat(0.2), 1.0e-6))
    );
    assert!(
        matches!(cuticle_angle, super::compiled::ParamRef::Float(v) if approx_f(*v, 0.5, 1.0e-6))
    );
    assert!(normal.is_none());
    assert!(matches!(
        curve_direction,
        super::compiled::ParamRef::Local(_)
    ));
}

#[test]
fn spec_chiang_hair_bsdf_normal_input_is_checked() {
    let err = compile_single_bsdf_node(
        "chiang_hair_bsdf",
        vec![FlatNodeInput {
            name: "normal".to_string(),
            ty: MtlxType::String,
            colorspace: None,
            unit: None,
            unittype: None,
            binding: FlatInput::Value(MtlxValue::String("bad".to_string())),
        }],
        "chiang_bad_normal",
    )
    .expect_err("invalid chiang_hair_bsdf normal should error");
    assert!(err.to_string().contains("normal"));
}

#[test]
fn spec_uniform_edf_compile_defaults_match_1394() {
    let compiled = compile_single_edf_node("uniform_edf", vec![], "uniform_edf_defaults")
        .expect("uniform_edf defaults should compile");

    let ClosureNode::Surface { edf, .. } = &compiled.closure_nodes[compiled.root as usize] else {
        panic!("root should be a surface closure");
    };
    let ClosureNode::UniformEdf { color } = &compiled.closure_nodes[*edf as usize] else {
        panic!("surface edf should be uniform_edf");
    };
    assert!(
        matches!(color, super::compiled::ParamRef::Color3(v) if approx_v3(*v, Vec3::ONE, 1.0e-6))
    );
}

#[test]
fn spec_conical_edf_compile_defaults_match_1394() {
    let compiled = compile_single_edf_node("conical_edf", vec![], "conical_edf_defaults")
        .expect("conical_edf defaults should compile");

    let ClosureNode::Surface { edf, .. } = &compiled.closure_nodes[compiled.root as usize] else {
        panic!("root should be a surface closure");
    };
    let ClosureNode::ConicalEdf {
        color,
        inner_angle,
        outer_angle,
        normal,
    } = &compiled.closure_nodes[*edf as usize]
    else {
        panic!("surface edf should be conical_edf");
    };
    assert!(
        matches!(color, super::compiled::ParamRef::Color3(v) if approx_v3(*v, Vec3::ONE, 1.0e-6))
    );
    assert!(
        matches!(inner_angle, super::compiled::ParamRef::Float(v) if approx_f(*v, 60.0, 1.0e-6))
    );
    assert!(
        matches!(outer_angle, super::compiled::ParamRef::Float(v) if approx_f(*v, 0.0, 1.0e-6))
    );
    assert!(normal.is_none());
}

#[test]
fn spec_measured_edf_warns_and_checks_file_socket() {
    let err = compile_single_edf_node(
        "measured_edf",
        vec![FlatNodeInput {
            name: "file".to_string(),
            ty: MtlxType::Float,
            colorspace: None,
            unit: None,
            unittype: None,
            binding: FlatInput::Value(MtlxValue::Float(1.0)),
        }],
        "measured_edf_bad_file",
    )
    .expect_err("invalid measured_edf file should error");
    assert!(err.to_string().contains("file"));

    let compiled = compile_single_edf_node("measured_edf", vec![], "measured_edf_fallback")
        .expect("measured_edf fallback should compile");
    let ClosureNode::Surface { edf, .. } = &compiled.closure_nodes[compiled.root as usize] else {
        panic!("root should be a surface closure");
    };
    assert!(matches!(
        compiled.closure_nodes[*edf as usize],
        ClosureNode::UniformEdf { .. }
    ));
}

#[test]
fn spec_generalized_schlick_edf_compile_defaults_match_1394() {
    let compiled = compile_single_edf_node(
        "generalized_schlick_edf",
        vec![],
        "generalized_schlick_edf_defaults",
    )
    .expect("generalized_schlick_edf defaults should compile");

    let ClosureNode::Surface { edf, .. } = &compiled.closure_nodes[compiled.root as usize] else {
        panic!("root should be a surface closure");
    };
    let ClosureNode::GeneralizedSchlickEdf {
        base,
        color0,
        color90,
        exponent,
    } = &compiled.closure_nodes[*edf as usize]
    else {
        panic!("surface edf should be generalized_schlick_edf");
    };
    assert_eq!(*base, 0);
    assert!(
        matches!(color0, super::compiled::ParamRef::Color3(v) if approx_v3(*v, Vec3::ONE, 1.0e-6))
    );
    assert!(
        matches!(color90, super::compiled::ParamRef::Color3(v) if approx_v3(*v, Vec3::ONE, 1.0e-6))
    );
    assert!(matches!(exponent, super::compiled::ParamRef::Float(v) if approx_f(*v, 5.0, 1.0e-6)));
}

#[test]
fn spec_vdf_nodes_warn_to_zero_but_validate_inputs() {
    compile_vdf_layer_base("absorption_vdf", vec![], "absorption_vdf_zero")
        .expect("absorption_vdf zero fallback should compile");

    let err = compile_vdf_layer_base(
        "anisotropic_vdf",
        vec![FlatNodeInput {
            name: "anisotropy".to_string(),
            ty: MtlxType::String,
            colorspace: None,
            unit: None,
            unittype: None,
            binding: FlatInput::Value(MtlxValue::String("bad".to_string())),
        }],
        "anisotropic_vdf_bad_anisotropy",
    )
    .expect_err("invalid anisotropic_vdf anisotropy should error");
    assert!(err.to_string().contains("anisotropy"));
}

#[test]
fn allocator_slot_count_matches_simple_chain() {
    let regs = run(
        vec![Instruction::Arith {
            dst: 0,
            op: ArithOp::Add,
            ty: ValueType::Float,
            a: Operand::Const(0),
            b: Operand::Const(1),
        }],
        Vec::new(),
        vec![Value::Float(1.0), Value::Float(2.0)],
        1,
    );
    assert_eq!(regs[0].as_float(), 3.0);
}
