use std::cell::Cell;

use glam::{Vec2, Vec3, Vec4};

use crate::bsdf::mtlx::{
    BurleyDiffuseBsdf, ChiangHairBsdf, ConductorBsdf, DielectricBsdf, GeneralizedSchlickBsdf,
    GoochShadeKernel, MtlxLobeSample, OrenNayarDiffuseBsdf, ScatterMode, SheenBsdfMtlx,
    TranslucentBsdf,
};
use crate::material::ShadingVertex;
use crate::material::pattern::noise::{
    cellnoise2d, cellnoise2d_vec3, cellnoise3d, cellnoise3d_vec3, fbm2d, fbm2d_vec3, fbm3d,
    fbm3d_vec3, flake3d, hsv_to_rgb, perlin2d, perlin2d_vec3, perlin3d, perlin3d_vec3,
    random_color, random_float, rgb_to_hsv, worley2d, worley2d_solid, worley2d_solid_vec3,
    worley2d_top2, worley2d_top3, worley3d, worley3d_solid, worley3d_solid_vec3, worley3d_top2,
    worley3d_top3,
};
use crate::math::OrthonormalBasis;
use crate::sampler::MaterialSampleRandoms;

use super::compiled::{
    AddressMode, ArithOp, ArtisticIorOutput, ChiangHairRoughnessOutput, ClosureNode, CombineKind,
    CompareOp, CompiledMaterial, FlakeOutput, GeometricKind, ImageTexture, Instruction, LogicalOp,
    MaskOp, NoiseKind, Operand, ParamRef, TriplanarFilter, UnaryOp, Value, ValueType, WorleyStyle,
};

type DalbedoCache<'a> = Option<&'a [Cell<Option<Vec3>>]>;
use super::{MtlxScratch, RegsHandle};

pub(crate) const MDL_FLOAT_EPS: f32 = 1.0e-6;

/// SSA register-machine 評価器の入口。`handle` は呼び出し側 (precompute_shading)
/// が `scratch.alloc_regs` で確保した region を指す。bytecode 実行中に
/// matrix3/matrix4_pool に push される行列の lifetime も同じ handle に紐づく。
pub fn run_instructions(
    compiled: &CompiledMaterial,
    sv: &ShadingVertex,
    scratch: &mut MtlxScratch,
    handle: RegsHandle,
) {
    run_instruction_stream(
        sv,
        scratch,
        handle,
        &compiled.instructions,
        &compiled.operand_pool,
        &compiled.value_pool,
        &compiled.color_processors,
    );
}

pub fn run_opacity_instructions(
    compiled: &CompiledMaterial,
    sv: &ShadingVertex,
    scratch: &mut MtlxScratch,
    handle: RegsHandle,
) {
    run_instruction_stream(
        sv,
        scratch,
        handle,
        &compiled.opacity_instructions,
        &compiled.opacity_operand_pool,
        &compiled.value_pool,
        &compiled.color_processors,
    );
}

fn run_instruction_stream(
    sv: &ShadingVertex,
    scratch: &mut MtlxScratch,
    handle: RegsHandle,
    instrs: &[Instruction],
    op_pool: &[Operand],
    value_pool: &[Value],
    color_processors: &[std::sync::Arc<crate::color::OcioColorProcessor>],
) {
    // field-destructure で regs slice と matrix pools の同時 mut borrow を可能にする。
    let MtlxScratch {
        regs_pool,
        matrix3_pool,
        matrix4_pool,
        ..
    } = scratch;
    let regs_start = handle.offset as usize;
    let regs_end = regs_start + handle.len as usize;
    let regs = &mut regs_pool[regs_start..regs_end];

    for instr in instrs {
        execute_instruction(
            instr,
            sv,
            regs,
            op_pool,
            value_pool,
            color_processors,
            matrix3_pool,
            matrix4_pool,
        );
    }
}

#[inline(always)]
fn push_mat3(pool: &mut Vec<glam::Mat3>, m: glam::Mat3) -> Value {
    let idx = pool.len() as u32;
    pool.push(m);
    Value::Matrix33Ref(idx)
}

#[inline(always)]
fn push_mat4(pool: &mut Vec<glam::Mat4>, m: glam::Mat4) -> Value {
    let idx = pool.len() as u32;
    pool.push(m);
    Value::Matrix44Ref(idx)
}

#[inline(always)]
fn read_mat3(pool: &[glam::Mat3], v: Value) -> glam::Mat3 {
    match v {
        Value::Matrix33Ref(idx) => pool[idx as usize],
        other => panic!("read_mat3 called on {:?}", other),
    }
}

#[inline(always)]
fn read_mat4(pool: &[glam::Mat4], v: Value) -> glam::Mat4 {
    match v {
        Value::Matrix44Ref(idx) => pool[idx as usize],
        other => panic!("read_mat4 called on {:?}", other),
    }
}

#[inline(always)]
fn read_operand(op: Operand, regs: &[Value], value_pool: &[Value]) -> Value {
    match op {
        Operand::Reg(i) => unsafe { *regs.get_unchecked(i as usize) },
        Operand::Const(i) => unsafe { *value_pool.get_unchecked(i as usize) },
    }
}

#[inline(always)]
fn write_reg(regs: &mut [Value], dst: u16, v: Value) {
    unsafe {
        *regs.get_unchecked_mut(dst as usize) = v;
    }
}

#[inline]
fn read_geometric(kind: GeometricKind, sv: &ShadingVertex) -> Value {
    match kind {
        GeometricKind::Position(space) => {
            let p = match space {
                super::compiled::GeomSpace::World => sv.p,
                super::compiled::GeomSpace::Object | super::compiled::GeomSpace::Model => {
                    sv.world_to_object.transform_point3(sv.p)
                }
            };
            Value::Vector3(p)
        }
        GeometricKind::Normal(space) => {
            let n = match space {
                super::compiled::GeomSpace::World => sv.ns,
                super::compiled::GeomSpace::Object | super::compiled::GeomSpace::Model => {
                    sv.object_normal_to_world.inverse().mul_vec3(sv.ns)
                }
            };
            Value::Vector3(n)
        }
        GeometricKind::Tangent(space) => {
            let t = match space {
                super::compiled::GeomSpace::World => sv.dpdu.normalize(),
                super::compiled::GeomSpace::Object | super::compiled::GeomSpace::Model => {
                    sv.world_to_object.transform_vector3(sv.dpdu).normalize()
                }
            };
            Value::Vector3(t)
        }
        GeometricKind::Bitangent(space) => {
            let b = match space {
                super::compiled::GeomSpace::World => sv.dpdv.normalize(),
                super::compiled::GeomSpace::Object | super::compiled::GeomSpace::Model => {
                    sv.world_to_object.transform_vector3(sv.dpdv).normalize()
                }
            };
            Value::Vector3(b)
        }
        GeometricKind::Texcoord => Value::Vector2(sv.uv),
        GeometricKind::Geomcolor => Value::Color3(Vec3::ZERO),
        GeometricKind::Frame => Value::Float(1.0),
        GeometricKind::Time => Value::Float(0.0),
        GeometricKind::ViewDirection(space) => {
            let view_direction = -sv.wo;
            let v = match space {
                super::compiled::GeomSpace::World => view_direction.normalize(),
                super::compiled::GeomSpace::Object | super::compiled::GeomSpace::Model => sv
                    .world_to_object
                    .transform_vector3(view_direction)
                    .normalize(),
            };
            Value::Vector3(v)
        }
    }
}

#[inline(always)]
fn execute_instruction(
    instr: &Instruction,
    sv: &ShadingVertex,
    regs: &mut [Value],
    op_pool: &[Operand],
    value_pool: &[Value],
    color_processors: &[std::sync::Arc<crate::color::OcioColorProcessor>],
    matrix3_pool: &mut Vec<glam::Mat3>,
    matrix4_pool: &mut Vec<glam::Mat4>,
) {
    match instr {
        Instruction::LoadConst {
            dst,
            value_pool_idx,
        } => {
            let v = unsafe { *value_pool.get_unchecked(*value_pool_idx as usize) };
            write_reg(regs, *dst, v);
        }
        Instruction::LoadGeom { dst, kind } => {
            write_reg(regs, *dst, read_geometric(*kind, sv));
        }
        Instruction::LoadMat3Const { dst, value } => {
            let v = push_mat3(matrix3_pool, *value);
            write_reg(regs, *dst, v);
        }
        Instruction::LoadMat4Const { dst, value } => {
            let v = push_mat4(matrix4_pool, *value);
            write_reg(regs, *dst, v);
        }

        Instruction::Arith { dst, op, ty, a, b } => {
            let av = read_operand(*a, regs, value_pool);
            let bv = read_operand(*b, regs, value_pool);
            if matches!(ty, ValueType::Matrix33) {
                let v = arith_mat3(av, bv, *op, matrix3_pool);
                write_reg(regs, *dst, push_mat3(matrix3_pool, v));
            } else if matches!(ty, ValueType::Matrix44) {
                let v = arith_mat4(av, bv, *op, matrix4_pool);
                write_reg(regs, *dst, push_mat4(matrix4_pool, v));
            } else {
                write_reg(regs, *dst, arith(av, bv, *op, *ty));
            }
        }
        Instruction::Unary { dst, op, ty, src } => {
            let v = read_operand(*src, regs, value_pool);
            write_reg(regs, *dst, unary(v, *op, *ty));
        }
        Instruction::Convert { dst, from, to, src } => {
            let v = read_operand(*src, regs, value_pool);
            write_reg(regs, *dst, convert_value(v, *from, *to));
        }
        Instruction::Logical { dst, op, a, b } => {
            let r = match op {
                LogicalOp::Not => !read_operand(*a, regs, value_pool).as_bool(),
                LogicalOp::And => {
                    read_operand(*a, regs, value_pool).as_bool()
                        && read_operand(*b, regs, value_pool).as_bool()
                }
                LogicalOp::Or => {
                    read_operand(*a, regs, value_pool).as_bool()
                        || read_operand(*b, regs, value_pool).as_bool()
                }
                LogicalOp::Xor => {
                    read_operand(*a, regs, value_pool).as_bool()
                        != read_operand(*b, regs, value_pool).as_bool()
                }
            };
            write_reg(regs, *dst, Value::Bool(r));
        }
        Instruction::CompareBool { dst, op, v1, v2 } => {
            let a = read_operand(*v1, regs, value_pool).as_float();
            let b = read_operand(*v2, regs, value_pool).as_float();
            let cond = match op {
                CompareOp::Greater => a > b,
                CompareOp::GreaterEq => a >= b,
                CompareOp::Equal => a == b,
            };
            write_reg(regs, *dst, Value::Bool(cond));
        }
        Instruction::Compare {
            dst,
            op,
            v1,
            v2,
            in_true,
            in_false,
        } => {
            let a = read_operand(*v1, regs, value_pool).as_float();
            let b = read_operand(*v2, regs, value_pool).as_float();
            let cond = match op {
                CompareOp::Greater => a > b,
                CompareOp::GreaterEq => a >= b,
                CompareOp::Equal => a == b,
            };
            let v = if cond {
                read_operand(*in_true, regs, value_pool)
            } else {
                read_operand(*in_false, regs, value_pool)
            };
            write_reg(regs, *dst, v);
        }
        Instruction::IfElse {
            dst,
            cond,
            in_true,
            in_false,
        } => {
            let c = read_operand(*cond, regs, value_pool).as_bool();
            let v = if c {
                read_operand(*in_true, regs, value_pool)
            } else {
                read_operand(*in_false, regs, value_pool)
            };
            write_reg(regs, *dst, v);
        }
        Instruction::MixValue {
            dst,
            ty,
            bg,
            fg,
            mix,
        } => {
            let bv = read_operand(*bg, regs, value_pool);
            let fv = read_operand(*fg, regs, value_pool);
            let mv = read_operand(*mix, regs, value_pool);
            write_reg(regs, *dst, mix_value(bv, fv, mv, *ty));
        }
        Instruction::Clamp { dst, ty, v, lo, hi } => {
            let vv = read_operand(*v, regs, value_pool);
            let lv = read_operand(*lo, regs, value_pool);
            let hv = read_operand(*hi, regs, value_pool);
            write_reg(regs, *dst, clamp_value(vv, lv, hv, *ty));
        }
        Instruction::Smoothstep { dst, ty, v, lo, hi } => {
            let vv = read_operand(*v, regs, value_pool);
            let lv = read_operand(*lo, regs, value_pool);
            let hv = read_operand(*hi, regs, value_pool);
            write_reg(regs, *dst, smoothstep_value(vv, lv, hv, *ty));
        }
        Instruction::Extract {
            dst,
            in_ty,
            src,
            idx,
        } => {
            let sv_val = read_operand(*src, regs, value_pool);
            let i = read_operand(*idx, regs, value_pool).as_integer();
            write_reg(regs, *dst, extract_value(sv_val, *in_ty, i));
        }
        Instruction::ExtractRowVector {
            dst,
            dim4,
            src,
            index,
        } => {
            let sv_val = read_operand(*src, regs, value_pool);
            if *dim4 {
                let row = read_mat4(matrix4_pool, sv_val).row(*index as usize);
                write_reg(regs, *dst, Value::Vector4(row));
            } else {
                let row = read_mat3(matrix3_pool, sv_val).row(*index as usize);
                write_reg(regs, *dst, Value::Vector3(row));
            }
        }
        Instruction::Reflect { dst, i, n } => {
            let iv = read_operand(*i, regs, value_pool).as_vector3();
            let nv = read_operand(*n, regs, value_pool).as_vector3();
            let r = iv - 2.0 * iv.dot(nv) * nv;
            write_reg(regs, *dst, Value::Vector3(r));
        }
        Instruction::Refract { dst, i, n, eta } => {
            let iv = read_operand(*i, regs, value_pool).as_vector3();
            let nv = read_operand(*n, regs, value_pool).as_vector3();
            let e = read_operand(*eta, regs, value_pool).as_float();
            let cosi = (-iv).dot(nv);
            let k = 1.0 - e * e * (1.0 - cosi * cosi);
            let r = if k < 0.0 {
                Vec3::ZERO
            } else {
                e * iv + (e * cosi - k.sqrt()) * nv
            };
            write_reg(regs, *dst, Value::Vector3(r));
        }
        Instruction::Rotate2d { dst, v, amount } => {
            // OSL mx_rotate_vector2 / MDL mx_rotate2d_vector2: (c*x+s*y, -s*x+c*y)
            let vv = read_operand(*v, regs, value_pool).as_vector2();
            let a_deg = read_operand(*amount, regs, value_pool).as_float();
            let a = a_deg.to_radians();
            let (s, c) = a.sin_cos();
            let r = Vec2::new(c * vv.x + s * vv.y, -s * vv.x + c * vv.y);
            write_reg(regs, *dst, Value::Vector2(r));
        }
        Instruction::Rotate3d {
            dst,
            v,
            axis,
            amount,
        } => {
            let vv = read_operand(*v, regs, value_pool).as_vector3();
            let ax = read_operand(*axis, regs, value_pool).as_vector3();
            let a_deg = read_operand(*amount, regs, value_pool).as_float();
            let a = a_deg.to_radians();
            let (s, c) = a.sin_cos();
            let one_minus_c = 1.0 - c;
            let r = vv * c + ax.cross(vv) * s + ax * ax.dot(vv) * one_minus_c;
            write_reg(regs, *dst, Value::Vector3(r));
        }
        Instruction::DotProduct { dst, ty, a, b } => {
            let av = read_operand(*a, regs, value_pool);
            let bv = read_operand(*b, regs, value_pool);
            let r = match ty {
                ValueType::Vector2 => av.as_vector2().dot(bv.as_vector2()),
                ValueType::Vector4 | ValueType::Color4 => av.as_color4().dot(bv.as_color4()),
                _ => av.as_vector3().dot(bv.as_vector3()),
            };
            write_reg(regs, *dst, Value::Float(r));
        }
        Instruction::CrossProduct { dst, a, b } => {
            let av = read_operand(*a, regs, value_pool).as_vector3();
            let bv = read_operand(*b, regs, value_pool).as_vector3();
            write_reg(regs, *dst, Value::Vector3(av.cross(bv)));
        }
        Instruction::Distance { dst, ty, a, b } => {
            let av = read_operand(*a, regs, value_pool);
            let bv = read_operand(*b, regs, value_pool);
            let d = match ty {
                ValueType::Vector2 => (av.as_vector2() - bv.as_vector2()).length(),
                ValueType::Vector4 | ValueType::Color4 => {
                    (av.as_color4() - bv.as_color4()).length()
                }
                _ => (av.as_vector3() - bv.as_vector3()).length(),
            };
            write_reg(regs, *dst, Value::Float(d));
        }
        Instruction::FacingRatio {
            dst,
            view,
            normal,
            invert,
            faceforward,
        } => {
            let v = read_operand(*view, regs, value_pool).as_vector3();
            let n = read_operand(*normal, regs, value_pool).as_vector3();
            let dot = v.dot(n);
            let mut f = if *faceforward { dot.abs() } else { -dot };
            if *invert {
                f = 1.0 - f;
            }
            write_reg(regs, *dst, Value::Float(f));
        }
        Instruction::LuminanceWithCoeffs {
            dst,
            ty,
            c,
            lumacoeffs,
        } => {
            let cv = read_operand(*c, regs, value_pool);
            let lc = read_operand(*lumacoeffs, regs, value_pool).as_color3();
            let lum = match ty {
                ValueType::Color4 | ValueType::Vector4 => {
                    let v4 = cv.as_color4();
                    Vec3::new(v4.x, v4.y, v4.z).dot(lc)
                }
                _ => cv.as_color3().dot(lc),
            };
            let out = match ty {
                ValueType::Color4 => {
                    let v4 = cv.as_color4();
                    Value::Color4(Vec4::new(lum, lum, lum, v4.w))
                }
                ValueType::Vector4 => {
                    let v4 = cv.as_color4();
                    Value::Vector4(Vec4::new(lum, lum, lum, v4.w))
                }
                _ => Value::Color3(Vec3::splat(lum)),
            };
            write_reg(regs, *dst, out);
        }
        Instruction::TransformPoint { dst, from, to, v } => {
            let vv = read_operand(*v, regs, value_pool).as_vector3();
            let r = transform_point_between_spaces(vv, *from, *to, sv);
            write_reg(regs, *dst, Value::Vector3(r));
        }
        Instruction::TransformVector { dst, from, to, v } => {
            let vv = read_operand(*v, regs, value_pool).as_vector3();
            let r = transform_vector_between_spaces(vv, *from, *to, sv);
            write_reg(regs, *dst, Value::Vector3(r));
        }
        Instruction::TransformNormal { dst, from, to, v } => {
            let vv = read_operand(*v, regs, value_pool).as_vector3();
            let r = transform_normal_between_spaces(vv, *from, *to, sv);
            write_reg(regs, *dst, Value::Vector3(r));
        }
        Instruction::TransformMatrix {
            dst,
            out_ty,
            dim4,
            mat,
            v,
        } => {
            let mv = read_operand(*mat, regs, value_pool);
            match (*dim4, *out_ty) {
                (false, ValueType::Vector2) => {
                    let m = read_mat3(matrix3_pool, mv);
                    let vv = read_operand(*v, regs, value_pool).as_vector2();
                    let r = m * Vec3::new(vv.x, vv.y, 1.0);
                    write_reg(regs, *dst, Value::Vector2(Vec2::new(r.x, r.y)));
                }
                (false, ValueType::Vector3) => {
                    let m = read_mat3(matrix3_pool, mv);
                    let vv = read_operand(*v, regs, value_pool).as_vector3();
                    let r = m * vv;
                    write_reg(regs, *dst, Value::Vector3(r));
                }
                (true, ValueType::Vector3) => {
                    let m = read_mat4(matrix4_pool, mv);
                    let vv = read_operand(*v, regs, value_pool).as_vector3();
                    let r = m * Vec4::new(vv.x, vv.y, vv.z, 1.0);
                    write_reg(regs, *dst, Value::Vector3(Vec3::new(r.x, r.y, r.z)));
                }
                (true, ValueType::Vector4) => {
                    let m = read_mat4(matrix4_pool, mv);
                    let vv = read_operand(*v, regs, value_pool).as_color4();
                    let r = m * vv;
                    write_reg(regs, *dst, Value::Vector4(r));
                }
                _ => panic!(
                    "transformmatrix unsupported output {:?} with dim4={}",
                    out_ty, dim4
                ),
            }
        }
        Instruction::Transpose { dst, dim4, src } => {
            let sv_val = read_operand(*src, regs, value_pool);
            if *dim4 {
                let m = read_mat4(matrix4_pool, sv_val);
                let v = push_mat4(matrix4_pool, m.transpose());
                write_reg(regs, *dst, v);
            } else {
                let m = read_mat3(matrix3_pool, sv_val);
                let v = push_mat3(matrix3_pool, m.transpose());
                write_reg(regs, *dst, v);
            }
        }
        Instruction::Determinant { dst, dim4, src } => {
            let sv_val = read_operand(*src, regs, value_pool);
            let d = if *dim4 {
                read_mat4(matrix4_pool, sv_val).determinant()
            } else {
                read_mat3(matrix3_pool, sv_val).determinant()
            };
            write_reg(regs, *dst, Value::Float(d));
        }
        Instruction::InvertMatrix { dst, dim4, src } => {
            let sv_val = read_operand(*src, regs, value_pool);
            if *dim4 {
                let m = read_mat4(matrix4_pool, sv_val);
                let v = push_mat4(matrix4_pool, m.inverse());
                write_reg(regs, *dst, v);
            } else {
                let m = read_mat3(matrix3_pool, sv_val);
                let v = push_mat3(matrix3_pool, m.inverse());
                write_reg(regs, *dst, v);
            }
        }
        Instruction::CreateMatrix3 { dst, rows_start } => {
            let r0 = read_operand(op_pool[*rows_start as usize], regs, value_pool).as_vector3();
            let r1 = read_operand(op_pool[*rows_start as usize + 1], regs, value_pool).as_vector3();
            let r2 = read_operand(op_pool[*rows_start as usize + 2], regs, value_pool).as_vector3();
            let m = glam::Mat3::from_cols(r0, r1, r2).transpose();
            let v = push_mat3(matrix3_pool, m);
            write_reg(regs, *dst, v);
        }
        Instruction::CreateMatrix4 { dst, rows_start } => {
            let r0 = read_operand(op_pool[*rows_start as usize], regs, value_pool).as_color4();
            let r1 = read_operand(op_pool[*rows_start as usize + 1], regs, value_pool).as_color4();
            let r2 = read_operand(op_pool[*rows_start as usize + 2], regs, value_pool).as_color4();
            let r3 = read_operand(op_pool[*rows_start as usize + 3], regs, value_pool).as_color4();
            let m = glam::Mat4::from_cols(r0, r1, r2, r3).transpose();
            let v = push_mat4(matrix4_pool, m);
            write_reg(regs, *dst, v);
        }
        Instruction::CreateMatrix4FromVec3 { dst, rows_start } => {
            let r0 = read_operand(op_pool[*rows_start as usize], regs, value_pool).as_vector3();
            let r1 = read_operand(op_pool[*rows_start as usize + 1], regs, value_pool).as_vector3();
            let r2 = read_operand(op_pool[*rows_start as usize + 2], regs, value_pool).as_vector3();
            let r3 = read_operand(op_pool[*rows_start as usize + 3], regs, value_pool).as_vector3();
            let m = glam::Mat4::from_cols(
                Vec4::new(r0.x, r0.y, r0.z, 0.0),
                Vec4::new(r1.x, r1.y, r1.z, 0.0),
                Vec4::new(r2.x, r2.y, r2.z, 0.0),
                Vec4::new(r3.x, r3.y, r3.z, 1.0),
            )
            .transpose();
            let v = push_mat4(matrix4_pool, m);
            write_reg(regs, *dst, v);
        }

        Instruction::Combine {
            dst,
            kind,
            operands_start,
        } => {
            let v = execute_combine_ssa(*kind, *operands_start, op_pool, regs, value_pool);
            write_reg(regs, *dst, v);
        }
        Instruction::Switch {
            dst,
            ty,
            which,
            branches_start,
        } => {
            let i = read_operand(*which, regs, value_pool)
                .as_integer()
                .clamp(0, 9) as usize;
            let op = op_pool[*branches_start as usize + i];
            let v0 = read_operand(op, regs, value_pool);
            let v = convert_value(v0, value_type_of(v0), *ty);
            write_reg(regs, *dst, v);
        }

        Instruction::Passthrough => {}

        Instruction::Blackbody { dst, temp } => {
            let t = read_operand(*temp, regs, value_pool).as_float();
            let rgb = blackbody(t);
            write_reg(regs, *dst, Value::Color3(rgb));
        }
        Instruction::ArtisticIor {
            dst,
            which,
            refl,
            edge,
        } => {
            let r = read_operand(*refl, regs, value_pool).as_color3();
            let e = read_operand(*edge, regs, value_pool).as_color3();
            let (ior, ext) = artistic_ior(r, e);
            let v = match which {
                ArtisticIorOutput::Ior => ior,
                ArtisticIorOutput::Extinction => ext,
            };
            write_reg(regs, *dst, Value::Color3(v));
        }
        Instruction::RoughnessAnisotropy { dst, r, a } => {
            let r_val = read_operand(*r, regs, value_pool).as_float();
            let a_val = read_operand(*a, regs, value_pool).as_float();
            write_reg(
                regs,
                *dst,
                Value::Vector2(roughness_anisotropy_mdl(r_val, a_val)),
            );
        }
        Instruction::GlossinessAnisotropy { dst, g, a } => {
            let g_val = read_operand(*g, regs, value_pool).as_float();
            let a_val = read_operand(*a, regs, value_pool).as_float();
            write_reg(
                regs,
                *dst,
                Value::Vector2(roughness_anisotropy_mdl(1.0 - g_val, a_val)),
            );
        }
        Instruction::RoughnessDual { dst, src } => {
            let mut r = read_operand(*src, regs, value_pool).as_vector2();
            if r.y < 0.0 {
                r.y = r.x;
            }
            write_reg(
                regs,
                *dst,
                Value::Vector2(Vec2::new(
                    (r.x * r.x).clamp(MDL_FLOAT_EPS, 1.0),
                    (r.y * r.y).clamp(MDL_FLOAT_EPS, 1.0),
                )),
            );
        }
        Instruction::ChiangHairRoughness {
            dst,
            which,
            longitudinal,
            azimuthal,
            scale_tt,
            scale_trt,
        } => {
            let l = read_operand(*longitudinal, regs, value_pool).as_float();
            let a = read_operand(*azimuthal, regs, value_pool).as_float();
            let stt = read_operand(*scale_tt, regs, value_pool).as_float();
            let strt = read_operand(*scale_trt, regs, value_pool).as_float();
            let lr = l.clamp(1.0e-3, 1.0);
            let ar = a.clamp(1.0e-3, 1.0);
            let v = 0.726 * lr + 0.812 * lr * lr + 3.7 * lr.powi(20);
            let v = v * v;
            let s = 0.265 * ar + 1.194 * ar * ar + 5.372 * ar.powi(22);
            let roughness = match which {
                ChiangHairRoughnessOutput::R => Vec2::new(v, s),
                ChiangHairRoughnessOutput::TT => Vec2::new(v * stt * stt, s),
                ChiangHairRoughnessOutput::TRT => Vec2::new(v * strt * strt, s),
            };
            write_reg(regs, *dst, Value::Vector2(roughness));
        }
        Instruction::DeonHairAbsorptionFromMelanin {
            dst,
            operands_start,
        } => {
            let s = *operands_start as usize;
            let conc = read_operand(op_pool[s], regs, value_pool).as_float();
            let redness = read_operand(op_pool[s + 1], regs, value_pool).as_float();
            let eum = read_operand(op_pool[s + 2], regs, value_pool).as_color3();
            let phe = read_operand(op_pool[s + 3], regs, value_pool).as_color3();
            let melanin = -(1.0 - conc).max(0.0001).ln();
            let eumelanin = melanin * (1.0 - redness);
            let pheomelanin = melanin * redness;
            let eum_absorb = Vec3::new(-eum.x.ln(), -eum.y.ln(), -eum.z.ln());
            let phe_absorb = Vec3::new(-phe.x.ln(), -phe.y.ln(), -phe.z.ln());
            let absorb = (eumelanin * eum_absorb + pheomelanin * phe_absorb).max(Vec3::ZERO);
            write_reg(regs, *dst, Value::Color3(absorb));
        }
        Instruction::ChiangHairAbsorptionFromColor { dst, color, beta } => {
            let c = read_operand(*color, regs, value_pool).as_color3();
            let b = read_operand(*beta, regs, value_pool).as_float();
            let factor = 5.969 - 0.215 * b + 2.532 * b * b - 10.73 * b.powi(3)
                + 5.574 * b.powi(4)
                + 0.245 * b.powi(5);
            let c = c.clamp(Vec3::splat(0.001), Vec3::ONE);
            let log_c = Vec3::new(c.x.ln(), c.y.ln(), c.z.ln());
            let absorb = (log_c / factor).powf(2.0);
            write_reg(regs, *dst, Value::Color3(absorb));
        }
        Instruction::TransformColor { dst, op, ty, src } => {
            let v = read_operand(*src, regs, value_pool);
            let out = match op {
                super::compiled::ColorXform::Identity => v,
                super::compiled::ColorXform::TextureToRendering
                | super::compiled::ColorXform::RenderingToTexture => v,
                super::compiled::ColorXform::Ocio { processor } => {
                    apply_ocio_color_xform(v, *ty, &color_processors[*processor as usize])
                }
            };
            write_reg(regs, *dst, out);
        }

        Instruction::Premult { dst, src } => {
            let v = read_operand(*src, regs, value_pool).as_color4();
            let a = v.w;
            write_reg(
                regs,
                *dst,
                Value::Color4(Vec4::new(v.x * a, v.y * a, v.z * a, a)),
            );
        }
        Instruction::Unpremult { dst, src } => {
            let v = read_operand(*src, regs, value_pool).as_color4();
            if v.w == 0.0 {
                write_reg(regs, *dst, Value::Color4(v));
            } else {
                write_reg(
                    regs,
                    *dst,
                    Value::Color4(Vec4::new(v.x / v.w, v.y / v.w, v.z / v.w, v.w)),
                );
            }
        }
        Instruction::Blend {
            dst,
            op,
            ty,
            bg,
            fg,
            mix,
        } => {
            let bv = read_operand(*bg, regs, value_pool);
            let fv = read_operand(*fg, regs, value_pool);
            let mv = read_operand(*mix, regs, value_pool).as_float();
            write_reg(regs, *dst, execute_blend(*op, *ty, bv, fv, mv));
        }
        Instruction::Merge {
            dst,
            op,
            bg,
            fg,
            mix,
        } => {
            let bv = read_operand(*bg, regs, value_pool).as_color4();
            let fv = read_operand(*fg, regs, value_pool).as_color4();
            let mv = read_operand(*mix, regs, value_pool).as_float();
            write_reg(regs, *dst, execute_merge(*op, bv, fv, mv));
        }
        Instruction::Mask {
            dst,
            op,
            ty,
            v,
            mask,
        } => {
            let vv = read_operand(*v, regs, value_pool);
            let mv = read_operand(*mask, regs, value_pool).as_float();
            let m = match op {
                MaskOp::Inside => mv,
                MaskOp::Outside => 1.0 - mv,
            };
            write_reg(regs, *dst, scale_value(vv, m, *ty));
        }
        Instruction::Contrast {
            dst,
            ty,
            v,
            amount,
            pivot,
        } => {
            let vv = read_operand(*v, regs, value_pool);
            let av = read_operand(*amount, regs, value_pool);
            let pv = read_operand(*pivot, regs, value_pool);
            write_reg(regs, *dst, apply_contrast_v(vv, av, pv, *ty));
        }
        Instruction::Range {
            dst,
            ty,
            doclamp,
            operands_start,
        } => {
            let s = *operands_start as usize;
            let v = read_operand(op_pool[s], regs, value_pool);
            let inlo = read_operand(op_pool[s + 1], regs, value_pool);
            let inhi = read_operand(op_pool[s + 2], regs, value_pool);
            let gamma = read_operand(op_pool[s + 3], regs, value_pool);
            let outlo = read_operand(op_pool[s + 4], regs, value_pool);
            let outhi = read_operand(op_pool[s + 5], regs, value_pool);
            write_reg(
                regs,
                *dst,
                apply_range_g(v, inlo, inhi, gamma, outlo, outhi, *doclamp, *ty),
            );
        }
        Instruction::Remap {
            dst,
            ty,
            operands_start,
        } => {
            let s = *operands_start as usize;
            let v = read_operand(op_pool[s], regs, value_pool);
            let inlo = read_operand(op_pool[s + 1], regs, value_pool);
            let inhi = read_operand(op_pool[s + 2], regs, value_pool);
            let outlo = read_operand(op_pool[s + 3], regs, value_pool);
            let outhi = read_operand(op_pool[s + 4], regs, value_pool);
            let one = match ty {
                ValueType::Float | ValueType::Integer => Value::Float(1.0),
                ValueType::Color3 => Value::Color3(Vec3::ONE),
                ValueType::Vector2 => Value::Vector2(Vec2::ONE),
                ValueType::Vector3 => Value::Vector3(Vec3::ONE),
                ValueType::Color4 => Value::Color4(Vec4::ONE),
                ValueType::Vector4 => Value::Vector4(Vec4::ONE),
                _ => Value::Float(1.0),
            };
            write_reg(
                regs,
                *dst,
                apply_range_g(v, inlo, inhi, one, outlo, outhi, false, *ty),
            );
        }
        Instruction::HsvAdjust { dst, ty, c, amount } => {
            let cv = read_operand(*c, regs, value_pool);
            let av = read_operand(*amount, regs, value_pool).as_color3();
            let c4 = cv.as_color4();
            let hsv = rgb_to_hsv(Vec3::new(c4.x, c4.y, c4.z));
            let mut h = hsv.x + av.x;
            h = h - h.floor();
            let s = hsv.y * av.y;
            let v = hsv.z * av.z;
            write_reg(
                regs,
                *dst,
                typed_color_with_alpha(hsv_to_rgb(h, s, v), c4.w, *ty),
            );
        }
        Instruction::Saturate {
            dst,
            ty,
            c,
            amount,
            lumacoeffs,
        } => {
            let cv = read_operand(*c, regs, value_pool).as_color4();
            let av = read_operand(*amount, regs, value_pool).as_float();
            let lc = read_operand(*lumacoeffs, regs, value_pool).as_color3();
            let rgb = Vec3::new(cv.x, cv.y, cv.z);
            let lum = rgb.dot(lc);
            let out = Vec3::splat(lum).lerp(rgb, av);
            write_reg(regs, *dst, typed_color_with_alpha(out, cv.w, *ty));
        }
        Instruction::ColorCorrect {
            dst,
            ty,
            operands_start,
        } => {
            let s = *operands_start as usize;
            let cv = read_operand(op_pool[s], regs, value_pool).as_color4();
            let hue = read_operand(op_pool[s + 1], regs, value_pool).as_float();
            let sat = read_operand(op_pool[s + 2], regs, value_pool).as_float();
            let gamma = read_operand(op_pool[s + 3], regs, value_pool).as_float();
            let lift = read_operand(op_pool[s + 4], regs, value_pool).as_float();
            let gain = read_operand(op_pool[s + 5], regs, value_pool).as_float();
            let contrast = read_operand(op_pool[s + 6], regs, value_pool).as_float();
            let pivot = read_operand(op_pool[s + 7], regs, value_pool).as_float();
            let exposure = read_operand(op_pool[s + 8], regs, value_pool).as_float();
            let rgb = Vec3::new(cv.x, cv.y, cv.z);
            let mut hsv = rgb_to_hsv(rgb);
            hsv.x += hue;
            let mut rgb = hsv_to_rgb(hsv.x, hsv.y, hsv.z);
            let lum = rgb.dot(Vec3::new(0.2722287, 0.6740818, 0.0536895));
            rgb = Vec3::splat(lum).lerp(rgb, sat);
            let apply_gamma = |x: f32| {
                if gamma == 1.0 {
                    x
                } else {
                    x.signum() * x.abs().powf(1.0 / gamma)
                }
            };
            rgb = Vec3::new(apply_gamma(rgb.x), apply_gamma(rgb.y), apply_gamma(rgb.z));
            rgb = rgb * (1.0 - lift) + Vec3::splat(lift);
            rgb *= gain;
            rgb = (rgb - Vec3::splat(pivot)) * contrast + Vec3::splat(pivot);
            rgb *= 2.0_f32.powf(exposure);
            write_reg(regs, *dst, typed_color_with_alpha(rgb, cv.w, *ty));
        }
        Instruction::Checkerboard {
            dst,
            color1,
            color2,
            uvtiling,
            uvoffset,
            texcoord,
        } => {
            let c1 = read_operand(*color1, regs, value_pool).as_color3();
            let c2 = read_operand(*color2, regs, value_pool).as_color3();
            let tiling = read_operand(*uvtiling, regs, value_pool).as_vector2();
            let offset = read_operand(*uvoffset, regs, value_pool).as_vector2();
            let uv = read_operand(*texcoord, regs, value_pool).as_vector2();
            let st = uv * tiling - offset;
            let ix = st.x.floor() as i32;
            let iy = st.y.floor() as i32;
            let cell = (ix + iy).rem_euclid(2);
            let out = if cell == 0 { c1 } else { c2 };
            write_reg(regs, *dst, Value::Color3(out));
        }

        Instruction::Image {
            dst,
            texture,
            kind: _,
            output,
            color_space: _,
            uaddress,
            vaddress,
            filter,
            texcoord,
            tiling,
            offset,
            default,
        } => {
            let tc = read_operand(*texcoord, regs, value_pool).as_vector2();
            let tl = read_operand(*tiling, regs, value_pool).as_vector2();
            let of = read_operand(*offset, regs, value_pool).as_vector2();
            let de = read_operand(*default, regs, value_pool);
            let uv_pre = tc * tl - of;
            let (uv, out_of_range) = apply_address_modes(uv_pre, *uaddress, *vaddress);
            let v = if out_of_range {
                de
            } else {
                sample_image_texture(texture, uv, sv, *output, de, *filter)
            };
            write_reg(regs, *dst, v);
        }
        Instruction::HextiledImage {
            dst,
            texture,
            output,
            default_color,
            color_space: _,
            operands_start,
        } => {
            let s = *operands_start as usize;
            let texcoord = read_operand(op_pool[s], regs, value_pool).as_vector2();
            let tiling = read_operand(op_pool[s + 1], regs, value_pool).as_vector2();
            let rotation = read_operand(op_pool[s + 2], regs, value_pool).as_float();
            let rotation_range = read_operand(op_pool[s + 3], regs, value_pool).as_vector2();
            let scale = read_operand(op_pool[s + 4], regs, value_pool).as_float();
            let scale_range = read_operand(op_pool[s + 5], regs, value_pool).as_vector2();
            let offset = read_operand(op_pool[s + 6], regs, value_pool).as_float();
            let offset_range = read_operand(op_pool[s + 7], regs, value_pool).as_vector2();
            let falloff = read_operand(op_pool[s + 8], regs, value_pool).as_float();
            let falloff_contrast = read_operand(op_pool[s + 9], regs, value_pool).as_float();
            let lumacoeffs = read_operand(op_pool[s + 10], regs, value_pool).as_color3();
            let coord = texcoord * tiling;
            let tile = crate::material::pattern::hextile::hextile_coord(
                coord,
                rotation,
                rotation_range,
                scale,
                scale_range,
                offset,
                offset_range,
            );
            let mut samples = [Vec4::ZERO; 3];
            let mut cw = Vec3::ZERO;
            for (i, sample) in samples.iter_mut().enumerate() {
                *sample = hextiled_color_sample(texture, sv, tile.coords[i], *default_color);
                let component = Vec3::new(sample.x, sample.y, sample.z).dot(lumacoeffs);
                match i {
                    0 => cw.x = component,
                    1 => cw.y = component,
                    _ => cw.z = component,
                }
            }
            cw = Vec3::ONE.lerp(cw, falloff_contrast);
            let w =
                crate::material::pattern::hextile::compute_blend_weights(cw, tile.weights, falloff);
            let rgb = w.x * samples[0].truncate()
                + w.y * samples[1].truncate()
                + w.z * samples[2].truncate();
            let mut alpha = (samples[0].w + samples[1].w + samples[2].w) / 3.0;
            if falloff != 0.5 {
                alpha = crate::material::pattern::hextile::schlick_gain(alpha, falloff);
            }
            let v = if matches!(*output, ValueType::Color4 | ValueType::Vector4) {
                typed_image_rgba(rgb, alpha, *output)
            } else {
                typed_image_rgb(rgb, *output)
            };
            write_reg(regs, *dst, v);
        }
        Instruction::HextiledNormalMap {
            dst,
            texture,
            flip_g,
            operands_start,
        } => {
            let s = *operands_start as usize;
            let texcoord = read_operand(op_pool[s], regs, value_pool).as_vector2();
            let tiling = read_operand(op_pool[s + 1], regs, value_pool).as_vector2();
            let rotation = read_operand(op_pool[s + 2], regs, value_pool).as_float();
            let rotation_range = read_operand(op_pool[s + 3], regs, value_pool).as_vector2();
            let scale = read_operand(op_pool[s + 4], regs, value_pool).as_float();
            let scale_range = read_operand(op_pool[s + 5], regs, value_pool).as_vector2();
            let offset = read_operand(op_pool[s + 6], regs, value_pool).as_float();
            let offset_range = read_operand(op_pool[s + 7], regs, value_pool).as_vector2();
            let falloff = read_operand(op_pool[s + 8], regs, value_pool).as_float();
            let strength = read_operand(op_pool[s + 9], regs, value_pool).as_float();
            let default = read_operand(op_pool[s + 10], regs, value_pool).as_vector3();
            if let Some(tex) = texture {
                let normal_override = read_operand(op_pool[s + 11], regs, value_pool).as_vector3();
                let tangent_override = read_operand(op_pool[s + 12], regs, value_pool).as_vector3();
                let bitangent_override =
                    read_operand(op_pool[s + 13], regs, value_pool).as_vector3();
                let coord = texcoord * tiling;
                let tile = crate::material::pattern::hextile::hextile_coord(
                    coord,
                    rotation,
                    rotation_range,
                    scale,
                    scale_range,
                    offset,
                    offset_range,
                );
                let mut normals = [Vec3::ZERO; 3];
                for (i, n) in normals.iter_mut().enumerate() {
                    let raw = tex.sample(tile.coords[i]);
                    let mut tan_n = raw * 2.0 - Vec3::ONE;
                    if *flip_g {
                        tan_n.y = -tan_n.y;
                    }
                    let rot = match i {
                        0 => tile.rotations.x,
                        1 => tile.rotations.y,
                        _ => tile.rotations.z,
                    };
                    let t = rotate_about_axis(tangent_override, normal_override, -rot) * strength;
                    let b = rotate_about_axis(bitangent_override, normal_override, -rot) * strength;
                    *n = (t * tan_n.x + b * tan_n.y + normal_override * tan_n.z).normalize();
                }
                let w = crate::material::pattern::hextile::compute_blend_weights(
                    Vec3::ONE,
                    tile.weights,
                    falloff,
                );
                let n_world = crate::material::pattern::hextile::gradient_blend_3_normals(
                    normal_override,
                    normals[0],
                    w.x,
                    normals[1],
                    w.y,
                    normals[2],
                    w.z,
                );
                write_reg(regs, *dst, Value::Vector3(n_world));
            } else {
                write_reg(regs, *dst, Value::Vector3(default));
            }
        }

        Instruction::Place2d {
            dst,
            trs,
            texcoord,
            pivot,
            scale,
            rotate,
            offset,
        } => {
            let tc = read_operand(*texcoord, regs, value_pool).as_vector2();
            let pv = read_operand(*pivot, regs, value_pool).as_vector2();
            let sc = read_operand(*scale, regs, value_pool).as_vector2();
            let ro = read_operand(*rotate, regs, value_pool).as_float();
            let of = read_operand(*offset, regs, value_pool).as_vector2();
            let safe_div =
                |a: Vec2, b: Vec2| Vec2::new(a.x / b.x.max(1.0e-30), a.y / b.y.max(1.0e-30));
            let rotate2d_uv = |v: Vec2, deg: f32| -> Vec2 {
                let (s, c) = deg.to_radians().sin_cos();
                Vec2::new(c * v.x + s * v.y, -s * v.x + c * v.y)
            };
            let result = if *trs {
                let centered = tc - pv - of;
                let rotated = rotate2d_uv(centered, ro);
                safe_div(rotated, sc) + pv
            } else {
                let centered = tc - pv;
                let scaled = safe_div(centered, sc);
                let rotated = rotate2d_uv(scaled, ro);
                rotated - of + pv
            };
            write_reg(regs, *dst, Value::Vector2(result));
        }
        Instruction::LatlongUv {
            dst,
            viewdir,
            rotation,
        } => {
            let v = read_operand(*viewdir, regs, value_pool).as_vector3();
            let r = read_operand(*rotation, regs, value_pool).as_float();
            let phi = v.x.atan2(v.z);
            let u = phi * (-1.0 / (2.0 * std::f32::consts::PI)) + 0.5 + r / 360.0;
            let theta = v.y.clamp(-1.0, 1.0).asin();
            let vv = theta * (1.0 / std::f32::consts::PI) + 0.5;
            write_reg(regs, *dst, Value::Vector2(Vec2::new(u, vv)));
        }

        Instruction::Noise {
            dst,
            kind,
            output,
            operands_start,
        } => {
            use super::compiled::NoiseOutput;
            let s = *operands_start as usize;
            let texcoord = read_operand(op_pool[s], regs, value_pool);
            let amp_val = read_operand(op_pool[s + 1], regs, value_pool);
            let pivot = read_operand(op_pool[s + 2], regs, value_pool).as_float();
            let octaves = read_operand(op_pool[s + 3], regs, value_pool)
                .as_integer()
                .max(0) as u32;
            let lacunarity = read_operand(op_pool[s + 4], regs, value_pool).as_float();
            let diminish = read_operand(op_pool[s + 5], regs, value_pool).as_float();
            let jitter = read_operand(op_pool[s + 6], regs, value_pool).as_float();
            let multi = matches!(
                output,
                NoiseOutput::Vector2 | NoiseOutput::Vector3 | NoiseOutput::Vector4
            );
            if multi {
                let amp4 = match amp_val {
                    Value::Vector2(v) => Vec4::new(v.x, v.y, 0.0, 0.0),
                    Value::Vector3(v) | Value::Color3(v) => Vec4::new(v.x, v.y, v.z, 1.0),
                    Value::Vector4(v) | Value::Color4(v) => v,
                    other => Vec4::splat(other.as_float()),
                };
                let amp_vec = Vec3::new(amp4.x, amp4.y, amp4.z);
                let raw_vec3 = match kind {
                    NoiseKind::Perlin2d => perlin2d_vec3(texcoord.as_vector2()),
                    NoiseKind::Perlin3d => perlin3d_vec3(texcoord.as_vector3()),
                    NoiseKind::Fractal2d => {
                        fbm2d_vec3(texcoord.as_vector2(), octaves, lacunarity, diminish)
                    }
                    NoiseKind::Fractal3d => {
                        fbm3d_vec3(texcoord.as_vector3(), octaves, lacunarity, diminish)
                    }
                    NoiseKind::Cellnoise2d => cellnoise2d_vec3(texcoord.as_vector2()),
                    NoiseKind::Cellnoise3d => cellnoise3d_vec3(texcoord.as_vector3()),
                    NoiseKind::Worleynoise2d | NoiseKind::Worleynoise3d => Vec3::ZERO,
                };
                let needs_pivot = matches!(kind, NoiseKind::Perlin2d | NoiseKind::Perlin3d);
                let v3 = if needs_pivot {
                    Vec3::splat(pivot) + amp_vec * raw_vec3
                } else {
                    amp_vec * raw_vec3
                };
                let val = match output {
                    NoiseOutput::Vector2 => {
                        let v2 = match kind {
                            NoiseKind::Fractal2d => Vec2::new(
                                amp4.x
                                    * fbm2d(texcoord.as_vector2(), octaves, lacunarity, diminish),
                                amp4.y
                                    * fbm2d(
                                        texcoord.as_vector2() + Vec2::new(19.0, 193.0),
                                        octaves,
                                        lacunarity,
                                        diminish,
                                    ),
                            ),
                            NoiseKind::Fractal3d => Vec2::new(
                                amp4.x
                                    * fbm3d(texcoord.as_vector3(), octaves, lacunarity, diminish),
                                amp4.y
                                    * fbm3d(
                                        texcoord.as_vector3() + Vec3::new(19.0, 193.0, 17.0),
                                        octaves,
                                        lacunarity,
                                        diminish,
                                    ),
                            ),
                            _ => Vec2::new(v3.x, v3.y),
                        };
                        Value::Vector2(v2)
                    }
                    NoiseOutput::Vector3 => Value::Vector3(v3),
                    NoiseOutput::Vector4 => {
                        let w_raw = match kind {
                            NoiseKind::Perlin2d => {
                                perlin2d(texcoord.as_vector2() + Vec2::new(19.0, 73.0))
                            }
                            NoiseKind::Perlin3d => {
                                perlin3d(texcoord.as_vector3() + Vec3::new(19.0, 73.0, 29.0))
                            }
                            NoiseKind::Fractal2d => fbm2d(
                                texcoord.as_vector2() + Vec2::new(19.0, 193.0),
                                octaves,
                                lacunarity,
                                diminish,
                            ),
                            NoiseKind::Fractal3d => fbm3d(
                                texcoord.as_vector3() + Vec3::new(19.0, 193.0, 17.0),
                                octaves,
                                lacunarity,
                                diminish,
                            ),
                            NoiseKind::Cellnoise2d => {
                                cellnoise2d(texcoord.as_vector2() + Vec2::new(19.0, 73.0))
                            }
                            NoiseKind::Cellnoise3d => {
                                cellnoise3d(texcoord.as_vector3() + Vec3::new(19.0, 73.0, 29.0))
                            }
                            NoiseKind::Worleynoise2d | NoiseKind::Worleynoise3d => 0.0,
                        };
                        let w = if needs_pivot {
                            pivot + amp4.w * w_raw
                        } else {
                            amp4.w * w_raw
                        };
                        Value::Vector4(Vec4::new(v3.x, v3.y, v3.z, w))
                    }
                    _ => unreachable!(),
                };
                write_reg(regs, *dst, val);
            } else {
                let amplitude = amp_val.as_float();
                let v = match kind {
                    NoiseKind::Perlin2d => pivot + amplitude * perlin2d(texcoord.as_vector2()),
                    NoiseKind::Perlin3d => pivot + amplitude * perlin3d(texcoord.as_vector3()),
                    NoiseKind::Cellnoise2d => cellnoise2d(texcoord.as_vector2()),
                    NoiseKind::Cellnoise3d => cellnoise3d(texcoord.as_vector3()),
                    NoiseKind::Worleynoise2d => worley2d(texcoord.as_vector2(), jitter),
                    NoiseKind::Worleynoise3d => worley3d(texcoord.as_vector3(), jitter),
                    NoiseKind::Fractal2d => {
                        amplitude * fbm2d(texcoord.as_vector2(), octaves, lacunarity, diminish)
                    }
                    NoiseKind::Fractal3d => {
                        amplitude * fbm3d(texcoord.as_vector3(), octaves, lacunarity, diminish)
                    }
                };
                write_reg(regs, *dst, Value::Float(v));
            }
        }
        Instruction::Worley {
            dst,
            dim3,
            output,
            style,
            operands_start,
        } => {
            use super::compiled::NoiseOutput;
            let s = *operands_start as usize;
            let coord = read_operand(op_pool[s], regs, value_pool);
            let jitter = read_operand(op_pool[s + 1], regs, value_pool).as_float();
            let val = match (output, style) {
                (NoiseOutput::Float, WorleyStyle::Distance) => Value::Float(if *dim3 {
                    worley3d(coord.as_vector3(), jitter)
                } else {
                    worley2d(coord.as_vector2(), jitter)
                }),
                (NoiseOutput::Float, WorleyStyle::Solid) => Value::Float(if *dim3 {
                    worley3d_solid(coord.as_vector3(), jitter)
                } else {
                    worley2d_solid(coord.as_vector2(), jitter)
                }),
                (NoiseOutput::Vector2, WorleyStyle::Distance) => {
                    let d = if *dim3 {
                        worley3d_top2(coord.as_vector3(), jitter, 0)
                    } else {
                        worley2d_top2(coord.as_vector2(), jitter, 0)
                    };
                    Value::Vector2(d)
                }
                (NoiseOutput::Vector3, WorleyStyle::Distance) => {
                    let d = if *dim3 {
                        worley3d_top3(coord.as_vector3(), jitter, 0)
                    } else {
                        worley2d_top3(coord.as_vector2(), jitter, 0)
                    };
                    Value::Vector3(d)
                }
                (NoiseOutput::Vector2, WorleyStyle::Solid) => {
                    let v3 = if *dim3 {
                        worley3d_solid_vec3(coord.as_vector3(), jitter)
                    } else {
                        worley2d_solid_vec3(coord.as_vector2(), jitter)
                    };
                    Value::Vector2(Vec2::new(v3.x, v3.y))
                }
                (NoiseOutput::Vector3, WorleyStyle::Solid) => {
                    let v3 = if *dim3 {
                        worley3d_solid_vec3(coord.as_vector3(), jitter)
                    } else {
                        worley2d_solid_vec3(coord.as_vector2(), jitter)
                    };
                    Value::Vector3(v3)
                }
                (NoiseOutput::Vector4, _) => {
                    let d = if *dim3 {
                        worley3d_top3(coord.as_vector3(), jitter, 0)
                    } else {
                        worley2d_top3(coord.as_vector2(), jitter, 0)
                    };
                    Value::Vector4(Vec4::new(d.x, d.y, d.z, 1.0))
                }
            };
            write_reg(regs, *dst, val);
        }
        Instruction::Cellnoise {
            dst,
            dim3,
            output,
            coord,
        } => {
            use super::compiled::NoiseOutput;
            let c = read_operand(*coord, regs, value_pool);
            let val = match output {
                NoiseOutput::Float => {
                    let v = if *dim3 {
                        cellnoise3d(c.as_vector3())
                    } else {
                        cellnoise2d(c.as_vector2())
                    };
                    Value::Float(v)
                }
                NoiseOutput::Vector2 => {
                    let v3 = if *dim3 {
                        cellnoise3d_vec3(c.as_vector3())
                    } else {
                        cellnoise2d_vec3(c.as_vector2())
                    };
                    Value::Vector2(Vec2::new(v3.x, v3.y))
                }
                NoiseOutput::Vector3 => {
                    let v3 = if *dim3 {
                        cellnoise3d_vec3(c.as_vector3())
                    } else {
                        cellnoise2d_vec3(c.as_vector2())
                    };
                    Value::Vector3(v3)
                }
                NoiseOutput::Vector4 => {
                    let v3 = if *dim3 {
                        cellnoise3d_vec3(c.as_vector3())
                    } else {
                        cellnoise2d_vec3(c.as_vector2())
                    };
                    let w = if *dim3 {
                        cellnoise3d(c.as_vector3() + Vec3::new(19.0, 73.0, 29.0))
                    } else {
                        cellnoise2d(c.as_vector2() + Vec2::new(19.0, 73.0))
                    };
                    Value::Vector4(Vec4::new(v3.x, v3.y, v3.z, w))
                }
            };
            write_reg(regs, *dst, val);
        }
        Instruction::Flake {
            dst,
            dim3,
            output,
            operands_start,
        } => {
            let s = *operands_start as usize;
            let size = read_operand(op_pool[s], regs, value_pool).as_float();
            let roughness = read_operand(op_pool[s + 1], regs, value_pool).as_float();
            let coverage = read_operand(op_pool[s + 2], regs, value_pool).as_float();
            let coord = read_operand(op_pool[s + 3], regs, value_pool);
            let position = if *dim3 {
                coord.as_vector3()
            } else {
                let uv = coord.as_vector2();
                Vec3::new(uv.x, uv.y, 0.0)
            };
            let normal = read_operand(op_pool[s + 4], regs, value_pool).as_vector3();
            let tangent = read_operand(op_pool[s + 5], regs, value_pool).as_vector3();
            let bitangent = read_operand(op_pool[s + 6], regs, value_pool).as_vector3();
            let flake = flake3d(
                size, roughness, coverage, position, normal, tangent, bitangent,
            );
            let val = match output {
                FlakeOutput::Id => Value::Integer(flake.id),
                FlakeOutput::Rand => Value::Float(flake.rand),
                FlakeOutput::Presence => Value::Float(flake.presence),
                FlakeOutput::Normal => Value::Vector3(flake.normal),
            };
            write_reg(regs, *dst, val);
        }
        Instruction::RandomFloat {
            dst,
            integer_input,
            operands_start,
        } => {
            let s = *operands_start as usize;
            let input = read_operand(op_pool[s], regs, value_pool).as_float();
            let seed = read_operand(op_pool[s + 1], regs, value_pool).as_integer();
            let lo = read_operand(op_pool[s + 2], regs, value_pool).as_float();
            let hi = read_operand(op_pool[s + 3], regs, value_pool).as_float();
            write_reg(
                regs,
                *dst,
                Value::Float(random_float(input, seed, lo, hi, *integer_input)),
            );
        }
        Instruction::RandomColor {
            dst,
            operands_start,
        } => {
            let s = *operands_start as usize;
            let input = read_operand(op_pool[s], regs, value_pool).as_float();
            let seed = read_operand(op_pool[s + 1], regs, value_pool).as_integer();
            let h_lo = read_operand(op_pool[s + 2], regs, value_pool).as_float();
            let h_hi = read_operand(op_pool[s + 3], regs, value_pool).as_float();
            let s_lo = read_operand(op_pool[s + 4], regs, value_pool).as_float();
            let s_hi = read_operand(op_pool[s + 5], regs, value_pool).as_float();
            let v_lo = read_operand(op_pool[s + 6], regs, value_pool).as_float();
            let v_hi = read_operand(op_pool[s + 7], regs, value_pool).as_float();
            write_reg(
                regs,
                *dst,
                Value::Color3(random_color(
                    input, seed, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi,
                )),
            );
        }

        Instruction::Ramplr {
            dst,
            ty,
            texcoord,
            l,
            r,
        } => {
            let tc = read_operand(*texcoord, regs, value_pool).as_vector2();
            let lv = read_operand(*l, regs, value_pool);
            let rv = read_operand(*r, regs, value_pool);
            let t = tc.x.clamp(0.0, 1.0);
            write_reg(regs, *dst, mix_value(lv, rv, Value::Float(t), *ty));
        }
        Instruction::Ramptb {
            dst,
            ty,
            texcoord,
            t,
            b,
        } => {
            let tc = read_operand(*texcoord, regs, value_pool).as_vector2();
            let tv = read_operand(*t, regs, value_pool);
            let bv = read_operand(*b, regs, value_pool);
            let u = tc.y.clamp(0.0, 1.0);
            write_reg(regs, *dst, mix_value(tv, bv, Value::Float(u), *ty));
        }
        Instruction::Ramp4 {
            dst,
            ty,
            texcoord,
            tl,
            tr,
            bl,
            br,
        } => {
            let tc = read_operand(*texcoord, regs, value_pool).as_vector2();
            let tlv = read_operand(*tl, regs, value_pool);
            let trv = read_operand(*tr, regs, value_pool);
            let blv = read_operand(*bl, regs, value_pool);
            let brv = read_operand(*br, regs, value_pool);
            let u = tc.x.clamp(0.0, 1.0);
            let v = tc.y.clamp(0.0, 1.0);
            let top = mix_value(tlv, trv, Value::Float(u), *ty);
            let bot = mix_value(blv, brv, Value::Float(u), *ty);
            write_reg(regs, *dst, mix_value(top, bot, Value::Float(v), *ty));
        }
        Instruction::Splitlr {
            dst,
            ty: _,
            texcoord,
            center,
            l,
            r,
        } => {
            let tc = read_operand(*texcoord, regs, value_pool).as_vector2();
            let c = read_operand(*center, regs, value_pool).as_float();
            let lv = read_operand(*l, regs, value_pool);
            let rv = read_operand(*r, regs, value_pool);
            write_reg(regs, *dst, if tc.x < c { lv } else { rv });
        }
        Instruction::Splittb {
            dst,
            ty: _,
            texcoord,
            center,
            t,
            b,
        } => {
            let tc = read_operand(*texcoord, regs, value_pool).as_vector2();
            let c = read_operand(*center, regs, value_pool).as_float();
            let tv = read_operand(*t, regs, value_pool);
            let bv = read_operand(*b, regs, value_pool);
            write_reg(regs, *dst, if tc.x < c { tv } else { bv });
        }

        Instruction::TriplanarBlend {
            dst,
            ty,
            filter,
            operands_start,
        } => {
            let s = *operands_start as usize;
            let inx = read_operand(op_pool[s], regs, value_pool);
            let iny = read_operand(op_pool[s + 1], regs, value_pool);
            let inz = read_operand(op_pool[s + 2], regs, value_pool);
            let normal = read_operand(op_pool[s + 3], regs, value_pool).as_vector3();
            let blend = read_operand(op_pool[s + 4], regs, value_pool).as_float();
            let abs_n = normal.normalize_or(Vec3::Z).abs();
            let w = match filter {
                TriplanarFilter::Closest => {
                    if abs_n.x >= abs_n.y && abs_n.x >= abs_n.z {
                        Vec3::new(1.0, 0.0, 0.0)
                    } else if abs_n.y >= abs_n.z {
                        Vec3::new(0.0, 1.0, 0.0)
                    } else {
                        Vec3::new(0.0, 0.0, 1.0)
                    }
                }
                TriplanarFilter::Linear => {
                    let sum = abs_n.x + abs_n.y + abs_n.z;
                    let w0 = if sum > 0.0 {
                        abs_n / sum
                    } else {
                        Vec3::splat(1.0 / 3.0)
                    };
                    let exp = 1.0 / blend.max(0.03);
                    let bp = Vec3::new(w0.x.powf(exp), w0.y.powf(exp), w0.z.powf(exp));
                    let bp_sum = bp.x + bp.y + bp.z;
                    if bp_sum > 0.0 {
                        bp / bp_sum
                    } else {
                        Vec3::splat(1.0 / 3.0)
                    }
                }
            };
            let result = match ty {
                ValueType::Float | ValueType::Integer => {
                    Value::Float(inx.as_float() * w.x + iny.as_float() * w.y + inz.as_float() * w.z)
                }
                ValueType::Color3 | ValueType::Vector3 => {
                    let cx = inx.as_color3();
                    let cy = iny.as_color3();
                    let cz = inz.as_color3();
                    let r = cx * w.x + cy * w.y + cz * w.z;
                    if matches!(ty, ValueType::Vector3) {
                        Value::Vector3(r)
                    } else {
                        Value::Color3(r)
                    }
                }
                ValueType::Color4 | ValueType::Vector4 => {
                    let cx = inx.as_color4();
                    let cy = iny.as_color4();
                    let cz = inz.as_color4();
                    let r = cx * w.x + cy * w.y + cz * w.z;
                    if matches!(ty, ValueType::Vector4) {
                        Value::Vector4(r)
                    } else {
                        Value::Color4(r)
                    }
                }
                _ => panic!("TriplanarBlend: unsupported ty {:?}", ty),
            };
            write_reg(regs, *dst, result);
        }

        Instruction::CurveUniformLinear { dst, knotvalues, t } => {
            let tv = read_operand(*t, regs, value_pool).as_float();
            let n = knotvalues.len();
            let s = tv * (n as f32 - 1.0);
            let k = s.floor() as isize;
            let u = s - k as f32;
            let k_clamped = k.clamp(0, (n - 1) as isize) as usize;
            let k1 = (k_clamped + 1).min(n - 1);
            write_reg(
                regs,
                *dst,
                Value::Float(knotvalues[k_clamped] * (1.0 - u) + knotvalues[k1] * u),
            );
        }
        Instruction::CurveUniformCubic { dst, knotvalues, t } => {
            let tv = read_operand(*t, regs, value_pool).as_float();
            write_reg(regs, *dst, Value::Float(catmull_rom_eval(knotvalues, tv)));
        }
        Instruction::CurveInverseCubic { dst, knots, x } => {
            let xv = read_operand(*x, regs, value_pool).as_float();
            write_reg(regs, *dst, Value::Float(catmull_rom_inverse(knots, xv)));
        }

        Instruction::Normalmap { dst, raw, scale } => {
            let raw_v = read_operand(*raw, regs, value_pool).as_vector3();
            let scale_v = read_operand(*scale, regs, value_pool).as_vector2();
            let v = if raw_v == Vec3::ZERO {
                Vec3::Z
            } else {
                raw_v * 2.0 - Vec3::ONE
            };
            let n = sv.ns;
            let t = sv.dpdu.normalize();
            let b = sv.dpdv.normalize();
            let result = (t * v.x * scale_v.x + b * v.y * scale_v.y + n * v.z).normalize();
            write_reg(regs, *dst, Value::Vector3(result));
        }
        Instruction::NormalmapWithFrame {
            dst,
            operands_start,
        } => {
            let s = *operands_start as usize;
            let raw_v = read_operand(op_pool[s], regs, value_pool).as_vector3();
            let scale_v = read_operand(op_pool[s + 1], regs, value_pool).as_vector2();
            let n_override = read_operand(op_pool[s + 2], regs, value_pool).as_vector3();
            let t_override = read_operand(op_pool[s + 3], regs, value_pool).as_vector3();
            let b_override = read_operand(op_pool[s + 4], regs, value_pool).as_vector3();
            let v = if raw_v == Vec3::ZERO {
                Vec3::Z
            } else {
                raw_v * 2.0 - Vec3::ONE
            };
            let result =
                (t_override * v.x * scale_v.x + b_override * v.y * scale_v.y + n_override * v.z)
                    .normalize();
            write_reg(regs, *dst, Value::Vector3(result));
        }
        Instruction::Bump { dst, height, scale } => {
            let _ = read_operand(*height, regs, value_pool).as_float();
            let _ = read_operand(*scale, regs, value_pool).as_float();
            write_reg(regs, *dst, Value::Vector3(sv.ns.normalize()));
        }
        Instruction::BumpWithFrame {
            dst,
            operands_start,
        } => {
            let s = *operands_start as usize;
            let _ = read_operand(op_pool[s], regs, value_pool).as_float();
            let _ = read_operand(op_pool[s + 1], regs, value_pool).as_float();
            let n_override = read_operand(op_pool[s + 2], regs, value_pool).as_vector3();
            let _ = read_operand(op_pool[s + 3], regs, value_pool).as_vector3();
            write_reg(regs, *dst, Value::Vector3(n_override.normalize()));
        }
        Instruction::HeightToNormal { .. } => {
            panic!("heighttonormal requires heightfield derivative/sample-grid evaluation")
        }
    }
}

fn apply_ocio_color_xform(
    v: Value,
    ty: ValueType,
    processor: &crate::color::OcioColorProcessor,
) -> Value {
    match ty {
        ValueType::Color3 => processor
            .apply_rgb(v.as_color3())
            .map(Value::Color3)
            .unwrap_or(v),
        ValueType::Color4 => {
            let c = v.as_color4();
            processor.apply_rgba(c).map(Value::Color4).unwrap_or(v)
        }
        _ => v,
    }
}

pub(crate) fn smoothstep_value(v: Value, lo: Value, hi: Value, ty: ValueType) -> Value {
    let smoothstep = |x: f32, l: f32, h: f32| {
        let t = ((x - l) / (h - l)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    match ty {
        ValueType::Float | ValueType::Integer => {
            Value::Float(smoothstep(v.as_float(), lo.as_float(), hi.as_float()))
        }
        ValueType::Vector2 => {
            let v = v.as_vector2();
            let l = lo.as_vector2();
            let h = hi.as_vector2();
            Value::Vector2(Vec2::new(
                smoothstep(v.x, l.x, h.x),
                smoothstep(v.y, l.y, h.y),
            ))
        }
        ValueType::Color3 | ValueType::Vector3 => {
            let v = v.as_vector3();
            let l = lo.as_vector3();
            let h = hi.as_vector3();
            let r = Vec3::new(
                smoothstep(v.x, l.x, h.x),
                smoothstep(v.y, l.y, h.y),
                smoothstep(v.z, l.z, h.z),
            );
            if matches!(ty, ValueType::Color3) {
                Value::Color3(r)
            } else {
                Value::Vector3(r)
            }
        }
        ValueType::Color4 | ValueType::Vector4 => {
            let v = v.as_color4();
            let l = lo.as_color4();
            let h = hi.as_color4();
            let r = Vec4::new(
                smoothstep(v.x, l.x, h.x),
                smoothstep(v.y, l.y, h.y),
                smoothstep(v.z, l.z, h.z),
                smoothstep(v.w, l.w, h.w),
            );
            if matches!(ty, ValueType::Color4) {
                Value::Color4(r)
            } else {
                Value::Vector4(r)
            }
        }
        _ => panic!("smoothstep unsupported type {:?}", ty),
    }
}

fn extract_value(src: Value, in_ty: ValueType, idx: i32) -> Value {
    let f = match in_ty {
        ValueType::Vector2 => {
            let v = src.as_vector2();
            let i = if idx == 0 { 0 } else { 1 };
            [v.x, v.y][i]
        }
        ValueType::Color3 | ValueType::Vector3 => {
            let v = src.as_vector3();
            let i = match idx {
                0 => 0,
                1 => 1,
                _ => 2,
            };
            [v.x, v.y, v.z][i]
        }
        ValueType::Color4 | ValueType::Vector4 => {
            let v = src.as_color4();
            let i = match idx {
                0 => 0,
                1 => 1,
                2 => 2,
                _ => 3,
            };
            [v.x, v.y, v.z, v.w][i]
        }
        _ => src.as_float(),
    };
    Value::Float(f)
}

pub(crate) fn value_type_of(v: Value) -> ValueType {
    match v {
        Value::Float(_) => ValueType::Float,
        Value::Integer(_) => ValueType::Integer,
        Value::Bool(_) => ValueType::Boolean,
        Value::Color3(_) => ValueType::Color3,
        Value::Color4(_) => ValueType::Color4,
        Value::Vector2(_) => ValueType::Vector2,
        Value::Vector3(_) => ValueType::Vector3,
        Value::Vector4(_) => ValueType::Vector4,
        Value::Matrix33Ref(_) => ValueType::Matrix33,
        Value::Matrix44Ref(_) => ValueType::Matrix44,
        Value::Empty => ValueType::Float,
    }
}

fn execute_combine_ssa(
    kind: CombineKind,
    operands_start: u32,
    op_pool: &[Operand],
    regs: &[Value],
    value_pool: &[Value],
) -> Value {
    let s = operands_start as usize;
    match kind {
        CombineKind::Vector2FromFloats => {
            let x = read_operand(op_pool[s], regs, value_pool).as_float();
            let y = read_operand(op_pool[s + 1], regs, value_pool).as_float();
            Value::Vector2(Vec2::new(x, y))
        }
        CombineKind::Color3FromFloats => {
            let x = read_operand(op_pool[s], regs, value_pool).as_float();
            let y = read_operand(op_pool[s + 1], regs, value_pool).as_float();
            let z = read_operand(op_pool[s + 2], regs, value_pool).as_float();
            Value::Color3(Vec3::new(x, y, z))
        }
        CombineKind::Vector3FromFloats => {
            let x = read_operand(op_pool[s], regs, value_pool).as_float();
            let y = read_operand(op_pool[s + 1], regs, value_pool).as_float();
            let z = read_operand(op_pool[s + 2], regs, value_pool).as_float();
            Value::Vector3(Vec3::new(x, y, z))
        }
        CombineKind::Color4FromFloats => {
            let x = read_operand(op_pool[s], regs, value_pool).as_float();
            let y = read_operand(op_pool[s + 1], regs, value_pool).as_float();
            let z = read_operand(op_pool[s + 2], regs, value_pool).as_float();
            let w = read_operand(op_pool[s + 3], regs, value_pool).as_float();
            Value::Color4(Vec4::new(x, y, z, w))
        }
        CombineKind::Vector4FromFloats => {
            let x = read_operand(op_pool[s], regs, value_pool).as_float();
            let y = read_operand(op_pool[s + 1], regs, value_pool).as_float();
            let z = read_operand(op_pool[s + 2], regs, value_pool).as_float();
            let w = read_operand(op_pool[s + 3], regs, value_pool).as_float();
            Value::Vector4(Vec4::new(x, y, z, w))
        }
        CombineKind::Color4FromColor3Float => {
            let rgb = read_operand(op_pool[s], regs, value_pool).as_color3();
            let a = read_operand(op_pool[s + 1], regs, value_pool).as_float();
            Value::Color4(Vec4::new(rgb.x, rgb.y, rgb.z, a))
        }
        CombineKind::Vector4FromVector3Float => {
            let xyz = read_operand(op_pool[s], regs, value_pool).as_vector3();
            let w = read_operand(op_pool[s + 1], regs, value_pool).as_float();
            Value::Vector4(Vec4::new(xyz.x, xyz.y, xyz.z, w))
        }
        CombineKind::Vector4FromVector2Vector2 => {
            let xy = read_operand(op_pool[s], regs, value_pool).as_vector2();
            let zw = read_operand(op_pool[s + 1], regs, value_pool).as_vector2();
            Value::Vector4(Vec4::new(xy.x, xy.y, zw.x, zw.y))
        }
    }
}

#[inline]
pub(crate) fn convert_value(v: Value, from: ValueType, to: ValueType) -> Value {
    if from == to {
        return v;
    }
    match (from, to) {
        (_, ValueType::Float) => Value::Float(v.as_float()),
        (_, ValueType::Integer) => Value::Integer(v.as_integer()),
        (_, ValueType::Boolean) => Value::Bool(v.as_bool()),
        (ValueType::Float | ValueType::Integer | ValueType::Boolean, ValueType::Color3) => {
            Value::Color3(Vec3::splat(v.as_float()))
        }
        (ValueType::Float | ValueType::Integer | ValueType::Boolean, ValueType::Vector3) => {
            Value::Vector3(Vec3::splat(v.as_float()))
        }
        (ValueType::Float | ValueType::Integer | ValueType::Boolean, ValueType::Color4) => {
            Value::Color4(Vec4::splat(v.as_float()))
        }
        (ValueType::Float | ValueType::Integer | ValueType::Boolean, ValueType::Vector4) => {
            Value::Vector4(Vec4::splat(v.as_float()))
        }
        (ValueType::Float | ValueType::Integer | ValueType::Boolean, ValueType::Vector2) => {
            Value::Vector2(Vec2::splat(v.as_float()))
        }
        (_, ValueType::Color3) => Value::Color3(v.as_color3()),
        (_, ValueType::Vector3) => Value::Vector3(v.as_vector3()),
        (_, ValueType::Vector2) => Value::Vector2(v.as_vector2()),
        (ValueType::Color4 | ValueType::Vector4, ValueType::Color4) => Value::Color4(v.as_color4()),
        (ValueType::Color4 | ValueType::Vector4, ValueType::Vector4) => {
            Value::Vector4(v.as_color4())
        }
        (_, ValueType::Color4) => {
            let c3 = v.as_color3();
            Value::Color4(Vec4::new(c3.x, c3.y, c3.z, 1.0))
        }
        (_, ValueType::Vector4) => {
            let c3 = v.as_vector3();
            Value::Vector4(Vec4::new(c3.x, c3.y, c3.z, 1.0))
        }
        (ValueType::Matrix33, ValueType::Matrix33) | (ValueType::Matrix44, ValueType::Matrix44) => {
            v
        }
        (_, ValueType::Matrix33) | (_, ValueType::Matrix44) => {
            panic!("convert_value: cannot convert {:?} to {:?}", from, to)
        }
    }
}

fn execute_blend_unsupported(ty: ValueType) -> ! {
    panic!("execute_blend: unsupported ValueType {:?}", ty)
}

fn transform_point_between_spaces(
    v: Vec3,
    from: super::compiled::GeomSpace,
    to: super::compiled::GeomSpace,
    sv: &ShadingVertex,
) -> Vec3 {
    use super::compiled::GeomSpace;
    if from == to {
        return v;
    }
    let world = match from {
        GeomSpace::World => v,
        GeomSpace::Object | GeomSpace::Model => sv.object_to_world.transform_point3(v),
    };
    match to {
        GeomSpace::World => world,
        GeomSpace::Object | GeomSpace::Model => sv.world_to_object.transform_point3(world),
    }
}

fn transform_vector_between_spaces(
    v: Vec3,
    from: super::compiled::GeomSpace,
    to: super::compiled::GeomSpace,
    sv: &ShadingVertex,
) -> Vec3 {
    use super::compiled::GeomSpace;
    if from == to {
        return v;
    }
    let world = match from {
        GeomSpace::World => v,
        GeomSpace::Object | GeomSpace::Model => sv.object_to_world.transform_vector3(v),
    };
    match to {
        GeomSpace::World => world,
        GeomSpace::Object | GeomSpace::Model => sv.world_to_object.transform_vector3(world),
    }
}

fn transform_normal_between_spaces(
    v: Vec3,
    from: super::compiled::GeomSpace,
    to: super::compiled::GeomSpace,
    sv: &ShadingVertex,
) -> Vec3 {
    use super::compiled::GeomSpace;
    if from == to {
        return v;
    }
    let world = match from {
        GeomSpace::World => v,
        GeomSpace::Object | GeomSpace::Model => sv.object_normal_to_world.mul_vec3(v),
    };
    match to {
        GeomSpace::World => world,
        GeomSpace::Object | GeomSpace::Model => {
            let world_to_object_normal = glam::Mat3::from_mat4(sv.object_to_world.transpose());
            world_to_object_normal.mul_vec3(world)
        }
    }
}

pub(crate) fn execute_blend(
    op: super::compiled::BlendOp,
    ty: ValueType,
    bg: Value,
    fg: Value,
    mix: f32,
) -> Value {
    use super::compiled::BlendOp;
    const FLOAT_EPS: f32 = 1.0e-6;
    let lerp_f = |b: f32, fch: f32| -> f32 {
        match op {
            BlendOp::Burn => {
                if fch.abs() < FLOAT_EPS {
                    0.0
                } else {
                    mix * (1.0 - (1.0 - b) / fch) + (1.0 - mix) * b
                }
            }
            BlendOp::Dodge => {
                if (1.0 - fch).abs() < FLOAT_EPS {
                    0.0
                } else {
                    mix * (b / (1.0 - fch)) + (1.0 - mix) * b
                }
            }
            BlendOp::Plus => mix * (b + fch) + (1.0 - mix) * b,
            BlendOp::Minus => mix * (b - fch) + (1.0 - mix) * b,
            BlendOp::Difference => mix * (b - fch).abs() + (1.0 - mix) * b,
            BlendOp::Screen => mix * (1.0 - (1.0 - fch) * (1.0 - b)) + (1.0 - mix) * b,
            BlendOp::Overlay => {
                let v = if b < 0.5 {
                    2.0 * fch * b
                } else {
                    1.0 - 2.0 * (1.0 - fch) * (1.0 - b)
                };
                mix * v + (1.0 - mix) * b
            }
        }
    };
    match ty {
        ValueType::Float | ValueType::Integer => Value::Float(lerp_f(bg.as_float(), fg.as_float())),
        ValueType::Color3 | ValueType::Vector3 => {
            let b = bg.as_color3();
            let g = fg.as_color3();
            let r = Vec3::new(lerp_f(b.x, g.x), lerp_f(b.y, g.y), lerp_f(b.z, g.z));
            if matches!(ty, ValueType::Vector3) {
                Value::Vector3(r)
            } else {
                Value::Color3(r)
            }
        }
        ValueType::Color4 | ValueType::Vector4 => {
            let b = bg.as_color4();
            let g = fg.as_color4();
            let r = Vec4::new(
                lerp_f(b.x, g.x),
                lerp_f(b.y, g.y),
                lerp_f(b.z, g.z),
                lerp_f(b.w, g.w),
            );
            if matches!(ty, ValueType::Color4) {
                Value::Color4(r)
            } else {
                Value::Vector4(r)
            }
        }
        ValueType::Vector2 => {
            let b = bg.as_vector2();
            let g = fg.as_vector2();
            Value::Vector2(Vec2::new(lerp_f(b.x, g.x), lerp_f(b.y, g.y)))
        }
        ValueType::Boolean | ValueType::Matrix33 | ValueType::Matrix44 => {
            let _ = (op, bg, fg, mix);
            execute_blend_unsupported(ty)
        }
    }
}

pub(crate) fn execute_merge(op: super::compiled::MergeOp, bg: Vec4, fg: Vec4, mix: f32) -> Value {
    use super::compiled::MergeOp;
    let (rgb_f, a_f): (Vec3, f32) = match op {
        MergeOp::Disjointover => {
            let b_rgb = Vec3::new(bg.x, bg.y, bg.z);
            let f_rgb = Vec3::new(fg.x, fg.y, fg.z);
            let b = bg.w;
            let f = fg.w;
            let rgb = if f + b <= 1.0 {
                f_rgb + b_rgb
            } else if b.abs() < 1.0e-6 {
                Vec3::ZERO
            } else {
                f_rgb + b_rgb * ((1.0 - f) / b)
            };
            (rgb, (f + b).min(1.0))
        }
        MergeOp::In => (Vec3::new(fg.x, fg.y, fg.z) * bg.w, fg.w * bg.w),
        MergeOp::Mask => (Vec3::new(bg.x, bg.y, bg.z) * fg.w, bg.w * fg.w),
        MergeOp::Matte => {
            let f = fg.w;
            let rgb = Vec3::new(fg.x, fg.y, fg.z) * f + Vec3::new(bg.x, bg.y, bg.z) * (1.0 - f);
            (rgb, fg.w + bg.w * (1.0 - f))
        }
        MergeOp::Out => (
            Vec3::new(fg.x, fg.y, fg.z) * (1.0 - bg.w),
            fg.w * (1.0 - bg.w),
        ),
        MergeOp::Over => {
            let f = fg.w;
            let rgb = Vec3::new(fg.x, fg.y, fg.z) + Vec3::new(bg.x, bg.y, bg.z) * (1.0 - f);
            (rgb, fg.w + bg.w * (1.0 - f))
        }
    };
    let bg3 = Vec3::new(bg.x, bg.y, bg.z);
    let final_rgb = bg3 * (1.0 - mix) + rgb_f * mix;
    let final_a = bg.w * (1.0 - mix) + a_f * mix;
    Value::Color4(Vec4::new(final_rgb.x, final_rgb.y, final_rgb.z, final_a))
}

pub(crate) fn scale_value(v: Value, m: f32, ty: ValueType) -> Value {
    match ty {
        ValueType::Float | ValueType::Integer => Value::Float(v.as_float() * m),
        ValueType::Color3 | ValueType::Vector3 => {
            let c = v.as_color3() * m;
            if matches!(ty, ValueType::Vector3) {
                Value::Vector3(c)
            } else {
                Value::Color3(c)
            }
        }
        ValueType::Color4 | ValueType::Vector4 => {
            let c = v.as_color4() * m;
            if matches!(ty, ValueType::Color4) {
                Value::Color4(c)
            } else {
                Value::Vector4(c)
            }
        }
        ValueType::Vector2 => Value::Vector2(v.as_vector2() * m),
        ValueType::Boolean | ValueType::Matrix33 | ValueType::Matrix44 => {
            panic!("scale_value: unsupported ValueType {:?}", ty)
        }
    }
}

pub(crate) fn apply_contrast_v(v: Value, amount: Value, pivot: Value, ty: ValueType) -> Value {
    let f = |x: f32, a: f32, p: f32| (x - p) * a + p;
    match ty {
        ValueType::Float | ValueType::Integer => {
            Value::Float(f(v.as_float(), amount.as_float(), pivot.as_float()))
        }
        ValueType::Color3 | ValueType::Vector3 => {
            let c = v.as_color3();
            let a = amount.as_color3();
            let p = pivot.as_color3();
            let r = Vec3::new(f(c.x, a.x, p.x), f(c.y, a.y, p.y), f(c.z, a.z, p.z));
            if matches!(ty, ValueType::Vector3) {
                Value::Vector3(r)
            } else {
                Value::Color3(r)
            }
        }
        ValueType::Color4 | ValueType::Vector4 => {
            let c = v.as_color4();
            let a = amount.as_color4();
            let p = pivot.as_color4();
            let r = Vec4::new(
                f(c.x, a.x, p.x),
                f(c.y, a.y, p.y),
                f(c.z, a.z, p.z),
                f(c.w, a.w, p.w),
            );
            if matches!(ty, ValueType::Color4) {
                Value::Color4(r)
            } else {
                Value::Vector4(r)
            }
        }
        ValueType::Vector2 => {
            let c = v.as_vector2();
            let a = amount.as_vector2();
            let p = pivot.as_vector2();
            Value::Vector2(Vec2::new(f(c.x, a.x, p.x), f(c.y, a.y, p.y)))
        }
        ValueType::Boolean | ValueType::Matrix33 | ValueType::Matrix44 => {
            panic!("apply_contrast_v: unsupported ValueType {:?}", ty)
        }
    }
}

pub(crate) fn apply_range_g(
    v: Value,
    inlow: Value,
    inhigh: Value,
    gamma: Value,
    outlow: Value,
    outhigh: Value,
    doclamp: bool,
    ty: ValueType,
) -> Value {
    let clamp_ordered = |x: f32, lo: f32, hi: f32| x.max(lo).min(hi);
    let f = |x: f32, il: f32, ih: f32, g: f32, ol: f32, oh: f32| -> f32 {
        let mut t = (x - il) / (ih - il);
        if g != 1.0 {
            let s = if t >= 0.0 { 1.0 } else { -1.0 };
            t = s * t.abs().powf(1.0 / g);
        }
        let mut o = ol + t * (oh - ol);
        if doclamp {
            o = clamp_ordered(o, ol, oh);
        }
        o
    };
    match ty {
        ValueType::Float | ValueType::Integer => Value::Float(f(
            v.as_float(),
            inlow.as_float(),
            inhigh.as_float(),
            gamma.as_float(),
            outlow.as_float(),
            outhigh.as_float(),
        )),
        ValueType::Color3 | ValueType::Vector3 => {
            let c = v.as_color3();
            let il = inlow.as_color3();
            let ih = inhigh.as_color3();
            let g = gamma.as_color3();
            let ol = outlow.as_color3();
            let oh = outhigh.as_color3();
            let r = Vec3::new(
                f(c.x, il.x, ih.x, g.x, ol.x, oh.x),
                f(c.y, il.y, ih.y, g.y, ol.y, oh.y),
                f(c.z, il.z, ih.z, g.z, ol.z, oh.z),
            );
            if matches!(ty, ValueType::Vector3) {
                Value::Vector3(r)
            } else {
                Value::Color3(r)
            }
        }
        ValueType::Color4 | ValueType::Vector4 => {
            let c = v.as_color4();
            let il = inlow.as_color4();
            let ih = inhigh.as_color4();
            let g = gamma.as_color4();
            let ol = outlow.as_color4();
            let oh = outhigh.as_color4();
            let r = Vec4::new(
                f(c.x, il.x, ih.x, g.x, ol.x, oh.x),
                f(c.y, il.y, ih.y, g.y, ol.y, oh.y),
                f(c.z, il.z, ih.z, g.z, ol.z, oh.z),
                f(c.w, il.w, ih.w, g.w, ol.w, oh.w),
            );
            if matches!(ty, ValueType::Color4) {
                Value::Color4(r)
            } else {
                Value::Vector4(r)
            }
        }
        ValueType::Vector2 => {
            let c = v.as_vector2();
            let il = inlow.as_vector2();
            let ih = inhigh.as_vector2();
            let g = gamma.as_vector2();
            let ol = outlow.as_vector2();
            let oh = outhigh.as_vector2();
            Value::Vector2(Vec2::new(
                f(c.x, il.x, ih.x, g.x, ol.x, oh.x),
                f(c.y, il.y, ih.y, g.y, ol.y, oh.y),
            ))
        }
        ValueType::Boolean | ValueType::Matrix33 | ValueType::Matrix44 => {
            panic!("apply_range_g: unsupported ValueType {:?}", ty)
        }
    }
}

fn apply_address_modes(uv: Vec2, ua: AddressMode, va: AddressMode) -> (Vec2, bool) {
    fn apply(c: f32, mode: AddressMode) -> (f32, bool) {
        match mode {
            AddressMode::Constant => {
                if !(0.0..=1.0).contains(&c) {
                    (c, true)
                } else {
                    (c, false)
                }
            }
            AddressMode::Clamp => (c.clamp(0.0, 1.0), false),
            AddressMode::Periodic => (c.rem_euclid(1.0), false),
            AddressMode::Mirror => {
                let two = (c.rem_euclid(2.0)).abs();
                let folded = if two > 1.0 { 2.0 - two } else { two };
                (folded, false)
            }
        }
    }
    let (u, ux) = apply(uv.x, ua);
    let (v, vx) = apply(uv.y, va);
    (Vec2::new(u, v), ux || vx)
}

fn sample_image_texture(
    texture: &ImageTexture,
    uv: Vec2,
    sv: &ShadingVertex,
    output: ValueType,
    default: Value,
    filter: super::compiled::FilterType,
) -> Value {
    use super::compiled::FilterType;
    match texture {
        ImageTexture::Color(t) => {
            let v = match filter {
                FilterType::Closest => t.sample_nearest(uv),
                FilterType::Linear => t.sample_mip_bilinear(uv, sv.uv_dx(), sv.uv_dy()),
            };
            typed_image_rgb(v, output)
        }
        ImageTexture::ColorAlpha { rgb, alpha } => {
            let (rgb_val, alpha_val) = match filter {
                FilterType::Closest => (rgb.sample_nearest(uv), alpha.sample_nearest(uv)),
                FilterType::Linear => (
                    rgb.sample_mip_bilinear(uv, sv.uv_dx(), sv.uv_dy()),
                    alpha.sample_mip_bilinear(uv, sv.uv_dx(), sv.uv_dy()),
                ),
            };
            typed_image_rgba(rgb_val, alpha_val, output)
        }
        ImageTexture::Scalar(t) => {
            let v = match filter {
                FilterType::Closest => t.sample_nearest(uv),
                FilterType::Linear => t.sample_mip_bilinear(uv, sv.uv_dx(), sv.uv_dy()),
            };
            typed_color(Vec3::splat(v), output)
        }
        ImageTexture::Udim { tiles } => {
            let (udim_id, frac_uv) = udim_id_and_frac_uv(uv);
            let Some(tile) = tiles.tiles.get(&udim_id) else {
                return default;
            };
            if matches!(output, ValueType::Float | ValueType::Integer)
                && let Some(scalar) = &tile.scalar
            {
                let v = match filter {
                    FilterType::Closest => scalar.sample_nearest(frac_uv),
                    FilterType::Linear => {
                        scalar.sample_mip_bilinear(frac_uv, sv.uv_dx(), sv.uv_dy())
                    }
                };
                return Value::Float(v);
            }
            let rgb = match filter {
                FilterType::Closest => tile.rgb.sample_nearest(frac_uv),
                FilterType::Linear => tile
                    .rgb
                    .sample_mip_bilinear(frac_uv, sv.uv_dx(), sv.uv_dy()),
            };
            if let Some(alpha_tex) = &tile.alpha {
                let alpha = match filter {
                    FilterType::Closest => alpha_tex.sample_nearest(frac_uv),
                    FilterType::Linear => {
                        alpha_tex.sample_mip_bilinear(frac_uv, sv.uv_dx(), sv.uv_dy())
                    }
                };
                typed_image_rgba(rgb, alpha, output)
            } else {
                typed_image_rgb(rgb, output)
            }
        }
        ImageTexture::Missing => default,
    }
}

/// Spec §Filename Substitutions: `UDIM = 1001 + floor(u) + floor(v)*10`,
/// with the fractional part of the UV used to sample within the tile.
pub fn udim_id_and_frac_uv(uv: Vec2) -> (u32, Vec2) {
    let u_int = uv.x.floor();
    let v_int = uv.y.floor();
    let u_idx = (u_int as i32).clamp(0, 9) as u32;
    let v_idx = (v_int as i32).clamp(0, 99) as u32;
    let udim = 1001 + u_idx + v_idx * 10;
    let frac = Vec2::new(uv.x - u_int, uv.y - v_int);
    (udim, frac)
}

fn rotate_about_axis(v: Vec3, axis: Vec3, angle: f32) -> Vec3 {
    let c = angle.cos();
    let s = angle.sin();
    v * c + axis.cross(v) * s + axis * axis.dot(v) * (1.0 - c)
}

pub(crate) fn roughness_anisotropy_mdl(roughness: f32, anisotropy: f32) -> Vec2 {
    let roughness_sqr = (roughness * roughness).clamp(MDL_FLOAT_EPS, 1.0);
    if anisotropy > 0.0 {
        let aspect = (1.0 - anisotropy.clamp(0.0, 0.98)).sqrt();
        Vec2::new((roughness_sqr / aspect).min(1.0), roughness_sqr * aspect)
    } else {
        Vec2::splat(roughness_sqr)
    }
}

fn hextiled_color_sample(
    texture: &ImageTexture,
    sv: &ShadingVertex,
    coord: Vec2,
    default: Vec4,
) -> Vec4 {
    match texture {
        ImageTexture::Color(t) => {
            let rgb = t.sample_mip_bilinear(coord, sv.uv_dx(), sv.uv_dy());
            Vec4::new(rgb.x, rgb.y, rgb.z, 0.0)
        }
        ImageTexture::ColorAlpha { rgb, alpha } => {
            let c = rgb.sample_mip_bilinear(coord, sv.uv_dx(), sv.uv_dy());
            let a = alpha.sample_mip_bilinear(coord, sv.uv_dx(), sv.uv_dy());
            Vec4::new(c.x, c.y, c.z, a)
        }
        ImageTexture::Scalar(t) => {
            let v = t.sample_mip_bilinear(coord, sv.uv_dx(), sv.uv_dy());
            Vec4::new(v, v, v, 0.0)
        }
        ImageTexture::Udim { tiles } => {
            let (udim_id, frac_uv) = udim_id_and_frac_uv(coord);
            tiles
                .tiles
                .get(&udim_id)
                .map(|tile| {
                    let rgb = tile
                        .rgb
                        .sample_mip_bilinear(frac_uv, sv.uv_dx(), sv.uv_dy());
                    let alpha = tile
                        .alpha
                        .as_ref()
                        .map(|a| a.sample_mip_bilinear(frac_uv, sv.uv_dx(), sv.uv_dy()))
                        .unwrap_or(0.0);
                    Vec4::new(rgb.x, rgb.y, rgb.z, alpha)
                })
                .unwrap_or(default)
        }
        ImageTexture::Missing => default,
    }
}

fn typed_color(v: Vec3, ty: ValueType) -> Value {
    match ty {
        // ACEScg lumacoeffs per MaterialX 1.39 default for color3 → float.
        ValueType::Float => Value::Float(0.2722287 * v.x + 0.6740818 * v.y + 0.0536895 * v.z),
        ValueType::Color3 => Value::Color3(v),
        ValueType::Vector2 => Value::Vector2(Vec2::new(v.x, v.y)),
        ValueType::Vector3 => Value::Vector3(v),
        ValueType::Color4 => Value::Color4(Vec4::new(v.x, v.y, v.z, 1.0)),
        ValueType::Vector4 => Value::Vector4(Vec4::new(v.x, v.y, v.z, 1.0)),
        ValueType::Integer | ValueType::Boolean | ValueType::Matrix33 | ValueType::Matrix44 => {
            panic!("typed_color: unsupported ValueType {:?}", ty)
        }
    }
}

fn typed_image_rgb(v: Vec3, ty: ValueType) -> Value {
    match ty {
        ValueType::Float => Value::Float(v.x),
        ValueType::Integer => Value::Integer(v.x as i32),
        ValueType::Color3 => Value::Color3(v),
        ValueType::Vector2 => Value::Vector2(Vec2::new(v.x, v.y)),
        ValueType::Vector3 => Value::Vector3(v),
        ValueType::Color4 => Value::Color4(Vec4::new(v.x, v.y, v.z, 0.0)),
        ValueType::Vector4 => Value::Vector4(Vec4::new(v.x, v.y, v.z, 0.0)),
        ValueType::Boolean | ValueType::Matrix33 | ValueType::Matrix44 => {
            panic!("typed_image_rgb: unsupported ValueType {:?}", ty)
        }
    }
}

fn typed_image_rgba(rgb: Vec3, alpha: f32, ty: ValueType) -> Value {
    match ty {
        ValueType::Float => Value::Float(rgb.x),
        ValueType::Integer => Value::Integer(rgb.x as i32),
        ValueType::Color3 => Value::Color3(rgb),
        ValueType::Vector2 => Value::Vector2(Vec2::new(rgb.x, rgb.y)),
        ValueType::Vector3 => Value::Vector3(rgb),
        ValueType::Color4 => Value::Color4(Vec4::new(rgb.x, rgb.y, rgb.z, alpha)),
        ValueType::Vector4 => Value::Vector4(Vec4::new(rgb.x, rgb.y, rgb.z, alpha)),
        ValueType::Boolean | ValueType::Matrix33 | ValueType::Matrix44 => {
            panic!("typed_image_rgba: unsupported ValueType {:?}", ty)
        }
    }
}

pub(crate) fn typed_color_with_alpha(rgb: Vec3, alpha: f32, ty: ValueType) -> Value {
    match ty {
        ValueType::Color4 => Value::Color4(Vec4::new(rgb.x, rgb.y, rgb.z, alpha)),
        ValueType::Vector4 => Value::Vector4(Vec4::new(rgb.x, rgb.y, rgb.z, alpha)),
        // Lower-rank outputs ignore the alpha channel (matches OSL behaviour
        // where image_color3 reads only the RGB part of an RGBA texture).
        _ => typed_color(rgb, ty),
    }
}

fn arith_mat3(a: Value, b: Value, op: ArithOp, pool: &[glam::Mat3]) -> glam::Mat3 {
    let av = read_mat3(pool, a);
    match b {
        Value::Float(_) | Value::Integer(_) => {
            let s = b.as_float();
            match op {
                ArithOp::Add => av + glam::Mat3::from_cols_array(&[s; 9]),
                ArithOp::Subtract => av - glam::Mat3::from_cols_array(&[s; 9]),
                ArithOp::Multiply => av * s,
                ArithOp::Divide => av * (1.0 / s),
                _ => panic!("arith_mat3: unsupported op {:?}", op),
            }
        }
        Value::Matrix33Ref(_) => {
            let bv = read_mat3(pool, b);
            match op {
                ArithOp::Add => av + bv,
                ArithOp::Subtract => av - bv,
                ArithOp::Multiply => av * bv,
                ArithOp::Divide => av * bv.inverse(),
                _ => panic!("arith_mat3: unsupported op {:?}", op),
            }
        }
        other => panic!("arith_mat3: rhs has unsupported type {:?}", other),
    }
}

fn arith_mat4(a: Value, b: Value, op: ArithOp, pool: &[glam::Mat4]) -> glam::Mat4 {
    let av = read_mat4(pool, a);
    match b {
        Value::Float(_) | Value::Integer(_) => {
            let s = b.as_float();
            match op {
                ArithOp::Add => av + glam::Mat4::from_cols_array(&[s; 16]),
                ArithOp::Subtract => av - glam::Mat4::from_cols_array(&[s; 16]),
                ArithOp::Multiply => av * s,
                ArithOp::Divide => av * (1.0 / s),
                _ => panic!("arith_mat4: unsupported op {:?}", op),
            }
        }
        Value::Matrix44Ref(_) => {
            let bv = read_mat4(pool, b);
            match op {
                ArithOp::Add => av + bv,
                ArithOp::Subtract => av - bv,
                ArithOp::Multiply => av * bv,
                ArithOp::Divide => av * bv.inverse(),
                _ => panic!("arith_mat4: unsupported op {:?}", op),
            }
        }
        other => panic!("arith_mat4: rhs has unsupported type {:?}", other),
    }
}

#[inline]
pub(crate) fn arith(a: Value, b: Value, op: ArithOp, ty: ValueType) -> Value {
    match ty {
        ValueType::Float | ValueType::Integer => {
            Value::Float(arith_scalar(a.as_float(), b.as_float(), op))
        }
        ValueType::Vector2 => {
            let av = a.as_vector2();
            let bv = arith_rhs_vec2(b);
            Value::Vector2(arith_vec2(av, bv, op))
        }
        ValueType::Color4 | ValueType::Vector4 => {
            let av = a.as_color4();
            let bv = arith_rhs_vec4(b);
            let r = arith_vec4(av, bv, op);
            if matches!(ty, ValueType::Color4) {
                Value::Color4(r)
            } else {
                Value::Vector4(r)
            }
        }
        ValueType::Color3 | ValueType::Vector3 => {
            let av = a.as_color3();
            let bv = arith_rhs_vec3(b);
            let r = arith_vec3(av, bv, op);
            if matches!(ty, ValueType::Vector3) {
                Value::Vector3(r)
            } else {
                Value::Color3(r)
            }
        }
        ValueType::Boolean | ValueType::Matrix33 | ValueType::Matrix44 => {
            panic!("arith: unsupported ValueType {:?}", ty)
        }
    }
}

#[inline(always)]
fn arith_scalar(x: f32, y: f32, op: ArithOp) -> f32 {
    match op {
        ArithOp::Add => x + y,
        ArithOp::Subtract => x - y,
        ArithOp::Multiply => x * y,
        // MaterialX 1.39 StandardNodes spec for `divide`: "dividing a
        // channel value by 0 results in floating-point NaN". Don't silently
        // substitute a finite value.
        ArithOp::Divide => x / y,
        ArithOp::Modulo => x - y * (x / y).floor(),
        ArithOp::Min => x.min(y),
        ArithOp::Max => x.max(y),
        ArithOp::Power => x.powf(y),
        ArithOp::SafePower => x.signum() * x.abs().powf(y),
        ArithOp::Atan2 => x.atan2(y),
    }
}

#[inline(always)]
fn arith_rhs_vec2(b: Value) -> Vec2 {
    match b {
        Value::Float(v) => Vec2::splat(v),
        Value::Integer(v) => Vec2::splat(v as f32),
        Value::Vector2(v) => v,
        other => panic!("arith Vector2: rhs has unsupported type {:?}", other),
    }
}

#[inline(always)]
fn arith_rhs_vec3(b: Value) -> Vec3 {
    match b {
        Value::Float(v) => Vec3::splat(v),
        Value::Integer(v) => Vec3::splat(v as f32),
        Value::Color3(v) | Value::Vector3(v) => v,
        other => panic!("arith Color3/Vector3: rhs has unsupported type {:?}", other),
    }
}

#[inline(always)]
fn arith_rhs_vec4(b: Value) -> Vec4 {
    match b {
        Value::Float(v) => Vec4::splat(v),
        Value::Integer(v) => Vec4::splat(v as f32),
        Value::Color4(v) | Value::Vector4(v) => v,
        other => panic!("arith Color4/Vector4: rhs has unsupported type {:?}", other),
    }
}

#[inline(always)]
fn arith_vec2(x: Vec2, y: Vec2, op: ArithOp) -> Vec2 {
    match op {
        ArithOp::Add => x + y,
        ArithOp::Subtract => x - y,
        ArithOp::Multiply => x * y,
        ArithOp::Divide => x / y,
        ArithOp::Modulo => x - y * (x / y).floor(),
        ArithOp::Min => x.min(y),
        ArithOp::Max => x.max(y),
        ArithOp::Power => Vec2::new(x.x.powf(y.x), x.y.powf(y.y)),
        ArithOp::SafePower => Vec2::new(
            x.x.signum() * x.x.abs().powf(y.x),
            x.y.signum() * x.y.abs().powf(y.y),
        ),
        ArithOp::Atan2 => Vec2::new(x.x.atan2(y.x), x.y.atan2(y.y)),
    }
}

#[inline(always)]
fn arith_vec3(x: Vec3, y: Vec3, op: ArithOp) -> Vec3 {
    match op {
        ArithOp::Add => x + y,
        ArithOp::Subtract => x - y,
        ArithOp::Multiply => x * y,
        ArithOp::Divide => x / y,
        ArithOp::Modulo => x - y * (x / y).floor(),
        ArithOp::Min => x.min(y),
        ArithOp::Max => x.max(y),
        ArithOp::Power => Vec3::new(x.x.powf(y.x), x.y.powf(y.y), x.z.powf(y.z)),
        ArithOp::SafePower => Vec3::new(
            x.x.signum() * x.x.abs().powf(y.x),
            x.y.signum() * x.y.abs().powf(y.y),
            x.z.signum() * x.z.abs().powf(y.z),
        ),
        ArithOp::Atan2 => Vec3::new(x.x.atan2(y.x), x.y.atan2(y.y), x.z.atan2(y.z)),
    }
}

#[inline(always)]
fn arith_vec4(x: Vec4, y: Vec4, op: ArithOp) -> Vec4 {
    match op {
        ArithOp::Add => x + y,
        ArithOp::Subtract => x - y,
        ArithOp::Multiply => x * y,
        ArithOp::Divide => x / y,
        ArithOp::Modulo => x - y * (x / y).floor(),
        ArithOp::Min => x.min(y),
        ArithOp::Max => x.max(y),
        ArithOp::Power => Vec4::new(x.x.powf(y.x), x.y.powf(y.y), x.z.powf(y.z), x.w.powf(y.w)),
        ArithOp::SafePower => Vec4::new(
            x.x.signum() * x.x.abs().powf(y.x),
            x.y.signum() * x.y.abs().powf(y.y),
            x.z.signum() * x.z.abs().powf(y.z),
            x.w.signum() * x.w.abs().powf(y.w),
        ),
        ArithOp::Atan2 => Vec4::new(
            x.x.atan2(y.x),
            x.y.atan2(y.y),
            x.z.atan2(y.z),
            x.w.atan2(y.w),
        ),
    }
}

#[inline]
pub(crate) fn unary(v: Value, op: UnaryOp, ty: ValueType) -> Value {
    let scalar = |x: f32| match op {
        UnaryOp::Sin => x.sin(),
        UnaryOp::Cos => x.cos(),
        UnaryOp::Tan => x.tan(),
        UnaryOp::Asin => x.asin(),
        UnaryOp::Acos => x.acos(),
        UnaryOp::Sqrt => x.sqrt(),
        UnaryOp::Ln => x.ln(),
        UnaryOp::Exp => x.exp(),
        UnaryOp::Abs => x.abs(),
        UnaryOp::Sign => {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        UnaryOp::Floor => x.floor(),
        UnaryOp::Ceil => x.ceil(),
        UnaryOp::Round => x.round_ties_even(),
        UnaryOp::Fract => x - x.floor(),
        UnaryOp::Invert => 1.0 - x,
        UnaryOp::Trianglewave => 0.5 - (x.abs().rem_euclid(1.0) - 0.5).abs(),
        UnaryOp::Normalize
        | UnaryOp::Magnitude
        | UnaryOp::Length
        | UnaryOp::Luminance
        | UnaryOp::RgbToHsv
        | UnaryOp::HsvToRgb => x,
    };
    match op {
        // StandardNodes spec: "the fourth channel in vector4 streams is not
        // treated any differently, e.g. not as a homogeneous w value", so
        // length/normalize must include all components for vector4.
        UnaryOp::Length | UnaryOp::Magnitude => match ty {
            ValueType::Vector4 => Value::Float(v.as_color4().length()),
            ValueType::Vector2 => Value::Float(v.as_vector2().length()),
            _ => Value::Float(v.as_vector3().length()),
        },
        UnaryOp::Normalize => match ty {
            ValueType::Vector4 => Value::Vector4(v.as_color4().normalize()),
            ValueType::Vector2 => Value::Vector2(v.as_vector2().normalize()),
            _ => Value::Vector3(v.as_vector3().normalize()),
        },
        UnaryOp::Luminance => {
            // StandardNodes spec: default lumacoeffs are ACEScg (AP1); for
            // color4 inputs the alpha channel must be preserved.
            let c = v.as_color3();
            let lum = 0.2722287 * c.x + 0.6740818 * c.y + 0.0536895 * c.z;
            match ty {
                ValueType::Color4 | ValueType::Vector4 => {
                    let alpha = v.as_color4().w;
                    let r = Vec4::new(lum, lum, lum, alpha);
                    if matches!(ty, ValueType::Color4) {
                        Value::Color4(r)
                    } else {
                        Value::Vector4(r)
                    }
                }
                _ => Value::Color3(Vec3::splat(lum)),
            }
        }
        UnaryOp::RgbToHsv => match ty {
            ValueType::Color4 | ValueType::Vector4 => {
                let c4 = v.as_color4();
                let hsv = rgb_to_hsv(Vec3::new(c4.x, c4.y, c4.z));
                let r = Vec4::new(hsv.x, hsv.y, hsv.z, c4.w);
                if matches!(ty, ValueType::Color4) {
                    Value::Color4(r)
                } else {
                    Value::Vector4(r)
                }
            }
            _ => Value::Color3(rgb_to_hsv(v.as_color3())),
        },
        UnaryOp::HsvToRgb => match ty {
            ValueType::Color4 | ValueType::Vector4 => {
                let c4 = v.as_color4();
                let rgb = hsv_to_rgb(c4.x, c4.y, c4.z);
                let r = Vec4::new(rgb.x, rgb.y, rgb.z, c4.w);
                if matches!(ty, ValueType::Color4) {
                    Value::Color4(r)
                } else {
                    Value::Vector4(r)
                }
            }
            _ => {
                let c = v.as_color3();
                Value::Color3(hsv_to_rgb(c.x, c.y, c.z))
            }
        },
        UnaryOp::Sin
        | UnaryOp::Cos
        | UnaryOp::Tan
        | UnaryOp::Asin
        | UnaryOp::Acos
        | UnaryOp::Sqrt
        | UnaryOp::Ln
        | UnaryOp::Exp
        | UnaryOp::Abs
        | UnaryOp::Sign
        | UnaryOp::Floor
        | UnaryOp::Ceil
        | UnaryOp::Round
        | UnaryOp::Fract
        | UnaryOp::Invert
        | UnaryOp::Trianglewave => match ty {
            ValueType::Float | ValueType::Integer => Value::Float(scalar(v.as_float())),
            ValueType::Vector2 => {
                let a = v.as_vector2();
                Value::Vector2(Vec2::new(scalar(a.x), scalar(a.y)))
            }
            ValueType::Color4 | ValueType::Vector4 => {
                let a = v.as_color4();
                let r = Vec4::new(scalar(a.x), scalar(a.y), scalar(a.z), scalar(a.w));
                if matches!(ty, ValueType::Color4) {
                    Value::Color4(r)
                } else {
                    Value::Vector4(r)
                }
            }
            ValueType::Color3 | ValueType::Vector3 => {
                let a = v.as_color3();
                let r = Vec3::new(scalar(a.x), scalar(a.y), scalar(a.z));
                if matches!(ty, ValueType::Vector3) {
                    Value::Vector3(r)
                } else {
                    Value::Color3(r)
                }
            }
            ValueType::Boolean | ValueType::Matrix33 | ValueType::Matrix44 => {
                panic!("unary: unsupported ValueType {:?}", ty)
            }
        },
    }
}

#[inline]
pub(crate) fn mix_value(bg: Value, fg: Value, m: Value, ty: ValueType) -> Value {
    match ty {
        ValueType::Float | ValueType::Integer => {
            let mt = m.as_float();
            Value::Float(bg.as_float() + (fg.as_float() - bg.as_float()) * mt)
        }
        ValueType::Vector2 => {
            let a = bg.as_vector2();
            let b = fg.as_vector2();
            let mt = m.as_vector2();
            Value::Vector2(a + (b - a) * mt)
        }
        ValueType::Color4 | ValueType::Vector4 => {
            let a = bg.as_color4();
            let b = fg.as_color4();
            let mt = m.as_color4();
            let r = a + (b - a) * mt;
            if matches!(ty, ValueType::Color4) {
                Value::Color4(r)
            } else {
                Value::Vector4(r)
            }
        }
        ValueType::Color3 | ValueType::Vector3 => {
            let a = bg.as_color3();
            let b = fg.as_color3();
            let mt = m.as_color3();
            let r = a + (b - a) * mt;
            if matches!(ty, ValueType::Vector3) {
                Value::Vector3(r)
            } else {
                Value::Color3(r)
            }
        }
        ValueType::Boolean | ValueType::Matrix33 | ValueType::Matrix44 => {
            panic!("mix_value: unsupported ValueType {:?}", ty)
        }
    }
}

#[inline]
pub(crate) fn clamp_value(v: Value, lo: Value, hi: Value, ty: ValueType) -> Value {
    let c = |x: f32, l: f32, h: f32| x.max(l).min(h);
    match ty {
        ValueType::Float | ValueType::Integer => {
            Value::Float(c(v.as_float(), lo.as_float(), hi.as_float()))
        }
        ValueType::Vector2 => {
            let v = v.as_vector2();
            let l = lo.as_vector2();
            let h = hi.as_vector2();
            Value::Vector2(Vec2::new(c(v.x, l.x, h.x), c(v.y, l.y, h.y)))
        }
        ValueType::Color4 | ValueType::Vector4 => {
            let v = v.as_color4();
            let l = lo.as_color4();
            let h = hi.as_color4();
            let r = Vec4::new(
                c(v.x, l.x, h.x),
                c(v.y, l.y, h.y),
                c(v.z, l.z, h.z),
                c(v.w, l.w, h.w),
            );
            if matches!(ty, ValueType::Color4) {
                Value::Color4(r)
            } else {
                Value::Vector4(r)
            }
        }
        ValueType::Color3 | ValueType::Vector3 => {
            let v = v.as_color3();
            let l = lo.as_color3();
            let h = hi.as_color3();
            let r = Vec3::new(c(v.x, l.x, h.x), c(v.y, l.y, h.y), c(v.z, l.z, h.z));
            if matches!(ty, ValueType::Vector3) {
                Value::Vector3(r)
            } else {
                Value::Color3(r)
            }
        }
        ValueType::Boolean | ValueType::Matrix33 | ValueType::Matrix44 => {
            panic!("clamp_value: unsupported ValueType {:?}", ty)
        }
    }
}

pub(crate) fn blackbody(temp: f32) -> Vec3 {
    let temperature = temp.clamp(1667.0, 25000.0);
    let t = 1000.0 / temperature;
    let t2 = t * t;
    let t3 = t2 * t;
    let xc = if temperature < 4000.0 {
        -0.266_123_9 * t3 - 0.234_358 * t2 + 0.877_695_6 * t + 0.179_91
    } else {
        -3.025_847 * t3 + 2.107_037_9 * t2 + 0.222_634_7 * t + 0.240_39
    };
    let xc2 = xc * xc;
    let xc3 = xc2 * xc;
    let yc = if temperature < 2222.0 {
        -1.106_381_4 * xc3 - 1.348_110_2 * xc2 + 2.185_558_3 * xc - 0.202_196_83
    } else if temperature < 4000.0 {
        -0.954_947_6 * xc3 - 1.374_185_9 * xc2 + 2.091_37 * xc - 0.167_488_67
    } else {
        3.081_758 * xc3 - 5.873_387 * xc2 + 3.751_13 * xc - 0.370_014_82
    };
    if yc <= 0.0 {
        Vec3::ONE
    } else {
        let x = xc / yc;
        let y = 1.0;
        let z = (1.0 - xc - yc) / yc;
        Vec3::new(
            3.2406 * x - 1.5372 * y - 0.4986 * z,
            -0.9689 * x + 1.8758 * y + 0.0415 * z,
            0.0557 * x - 0.2040 * y + 1.0570 * z,
        )
        .max(Vec3::ZERO)
    }
}

fn catmull_rom_eval(values: &[f32], t: f32) -> f32 {
    let n = values.len();
    if n < 2 {
        return values.first().copied().unwrap_or(0.0);
    }
    let s = t.clamp(0.0, 1.0) * (n as f32 - 1.0);
    let k = (s.floor() as isize).clamp(0, (n - 2) as isize);
    let u = s - k as f32;
    let kk = k;
    let pm1 = values[kk.saturating_sub(1).max(0) as usize];
    let p0 = values[kk as usize];
    let p1 = values[(kk + 1) as usize];
    let p2 = values[((kk + 2).min((n - 1) as isize)) as usize];
    let u2 = u * u;
    let u3 = u2 * u;
    0.5 * ((2.0 * p0)
        + (-pm1 + p1) * u
        + (2.0 * pm1 - 5.0 * p0 + 4.0 * p1 - p2) * u2
        + (-pm1 + 3.0 * p0 - 3.0 * p1 + p2) * u3)
}

fn catmull_rom_inverse(knots: &[f32], x: f32) -> f32 {
    let n = knots.len();
    if n < 2 {
        return 0.0;
    }
    if x <= knots[0] {
        return 0.0;
    }
    if x >= knots[n - 1] {
        return 1.0;
    }
    let mut lo = 0.0_f32;
    let mut hi = 1.0_f32;
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        let v = catmull_rom_eval(knots, mid);
        if v < x {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo) < 1.0e-6 {
            break;
        }
    }
    0.5 * (lo + hi)
}

pub(crate) fn artistic_ior(reflectivity: Vec3, edge: Vec3) -> (Vec3, Vec3) {
    let r = reflectivity.clamp(Vec3::ZERO, Vec3::splat(0.99));
    let r_sqrt = Vec3::new(r.x.sqrt(), r.y.sqrt(), r.z.sqrt());
    let n_min = (Vec3::ONE - r) / (Vec3::ONE + r);
    let n_max = (Vec3::ONE + r_sqrt) / (Vec3::ONE - r_sqrt);
    let ior = Vec3::new(
        n_max.x * (1.0 - edge.x) + n_min.x * edge.x,
        n_max.y * (1.0 - edge.y) + n_min.y * edge.y,
        n_max.z * (1.0 - edge.z) + n_min.z * edge.z,
    );
    let np1 = ior + Vec3::ONE;
    let nm1 = ior - Vec3::ONE;
    let one_minus_r = (Vec3::ONE - r).max(Vec3::splat(1e-6));
    let k2 = (np1 * np1 * r - nm1 * nm1) / one_minus_r;
    let k2 = k2.max(Vec3::ZERO);
    let extinction = Vec3::new(k2.x.sqrt(), k2.y.sqrt(), k2.z.sqrt());
    (ior, extinction)
}

// ============================================================================
// Closure tree evaluation
// ============================================================================

#[inline]
fn read_param(p: &ParamRef, locals: &[Value]) -> Value {
    match p {
        ParamRef::Float(v) => Value::Float(*v),
        ParamRef::Integer(v) => Value::Integer(*v),
        ParamRef::Bool(v) => Value::Bool(*v),
        ParamRef::Color3(v) => Value::Color3(*v),
        ParamRef::Color4(v) => Value::Color4(*v),
        ParamRef::Vector2(v) => Value::Vector2(*v),
        ParamRef::Vector3(v) => Value::Vector3(*v),
        ParamRef::Vector4(v) => Value::Vector4(*v),
        ParamRef::Matrix33(_) | ParamRef::Matrix44(_) => {
            panic!(
                "read_param: matrix ParamRef has no on-stack Value form; closure nodes never expose matrix-typed parameters"
            )
        }
        ParamRef::Local(i) => locals[*i as usize],
    }
}

#[inline]
fn read_float(p: &ParamRef, locals: &[Value]) -> f32 {
    read_param(p, locals).as_float()
}

#[inline]
fn read_color3(p: &ParamRef, locals: &[Value]) -> Vec3 {
    read_param(p, locals).as_color3()
}

#[inline]
fn read_vec2(p: &ParamRef, locals: &[Value]) -> Vec2 {
    read_param(p, locals).as_vector2()
}

#[inline]
fn read_vec3(p: &ParamRef, locals: &[Value]) -> Vec3 {
    read_param(p, locals).as_vector3()
}

/// MaterialX BSDF nodes may carry their own `normal`/`tangent` inputs; when
/// present, the BSDF must evaluate against a re-oriented frame, not the
/// shading vertex frame.
fn override_frame_for_wo(
    sv: &ShadingVertex,
    locals: &[Value],
    normal: &Option<ParamRef>,
    tangent: &Option<ParamRef>,
    wo_local_orig: Vec3,
) -> (OrthonormalBasis, Vec3) {
    let n = match normal {
        Some(p) => read_vec3(p, locals).normalize_or_zero(),
        None => return (sv.frame, wo_local_orig),
    };
    if n.length_squared() < 1.0e-6 {
        return (sv.frame, wo_local_orig);
    }
    let new_frame = match tangent {
        Some(p) => {
            let t = read_vec3(p, locals).normalize_or_zero();
            if t.length_squared() < 1.0e-6 {
                OrthonormalBasis::from_normal(n)
            } else {
                OrthonormalBasis::from_normal_and_tangent(n, t)
            }
        }
        None => OrthonormalBasis::from_normal(n),
    };
    let wo_world = sv.frame.local_to_world(wo_local_orig);
    let wo_new = new_frame.world_to_local(wo_world).normalize_or_zero();
    (new_frame, wo_new)
}

fn rebase_wi_into_frame(
    sv: &ShadingVertex,
    new_frame: &OrthonormalBasis,
    wi_local_orig: Vec3,
) -> Vec3 {
    let wi_world = sv.frame.local_to_world(wi_local_orig);
    new_frame.world_to_local(wi_world).normalize_or_zero()
}

fn rebase_wi_out_of_frame(
    sv: &ShadingVertex,
    new_frame: &OrthonormalBasis,
    wi_local_new: Vec3,
) -> Vec3 {
    let wi_world = new_frame.local_to_world(wi_local_new);
    sv.frame.world_to_local(wi_world).normalize_or_zero()
}

pub fn sample_closure(
    compiled: &CompiledMaterial,
    locals: &[Value],
    sv: &ShadingVertex,
    randoms: &MaterialSampleRandoms,
) -> Option<MtlxLobeSample> {
    sample_closure_cached(compiled, locals, sv, randoms, &[])
}

pub fn sample_closure_cached(
    compiled: &CompiledMaterial,
    locals: &[Value],
    sv: &ShadingVertex,
    randoms: &MaterialSampleRandoms,
    dalbedo_cache: &[Cell<Option<Vec3>>],
) -> Option<MtlxLobeSample> {
    let wo_local = sv.frame.world_to_local(sv.wo).normalize_or_zero();
    sample_closure_idx(
        compiled,
        locals,
        sv,
        *randoms,
        compiled.root,
        wo_local,
        Some(dalbedo_cache),
    )
}

pub fn eval_closure(
    compiled: &CompiledMaterial,
    locals: &[Value],
    sv: &ShadingVertex,
    wo_local: Vec3,
    wi_local: Vec3,
) -> Vec3 {
    eval_closure_cached(compiled, locals, sv, wo_local, wi_local, &[])
}

pub fn eval_closure_cached(
    compiled: &CompiledMaterial,
    locals: &[Value],
    sv: &ShadingVertex,
    wo_local: Vec3,
    wi_local: Vec3,
    dalbedo_cache: &[Cell<Option<Vec3>>],
) -> Vec3 {
    eval_closure_idx(
        compiled,
        locals,
        sv,
        compiled.root,
        wo_local,
        wi_local,
        Some(dalbedo_cache),
    )
}

pub fn pdf_closure(
    compiled: &CompiledMaterial,
    locals: &[Value],
    sv: &ShadingVertex,
    wo_local: Vec3,
    wi_local: Vec3,
) -> f32 {
    pdf_closure_cached(compiled, locals, sv, wo_local, wi_local, &[])
}

pub fn pdf_closure_cached(
    compiled: &CompiledMaterial,
    locals: &[Value],
    sv: &ShadingVertex,
    wo_local: Vec3,
    wi_local: Vec3,
    dalbedo_cache: &[Cell<Option<Vec3>>],
) -> f32 {
    pdf_closure_idx(
        compiled,
        locals,
        sv,
        compiled.root,
        wo_local,
        wi_local,
        Some(dalbedo_cache),
    )
}

pub fn eval_pdf_closure(
    compiled: &CompiledMaterial,
    locals: &[Value],
    sv: &ShadingVertex,
    wo_local: Vec3,
    wi_local: Vec3,
) -> (Vec3, f32) {
    eval_pdf_closure_cached(compiled, locals, sv, wo_local, wi_local, &[])
}

pub fn eval_pdf_closure_cached(
    compiled: &CompiledMaterial,
    locals: &[Value],
    sv: &ShadingVertex,
    wo_local: Vec3,
    wi_local: Vec3,
    dalbedo_cache: &[Cell<Option<Vec3>>],
) -> (Vec3, f32) {
    eval_pdf_closure_idx(
        compiled,
        locals,
        sv,
        compiled.root,
        wo_local,
        wi_local,
        Some(dalbedo_cache),
    )
}

pub fn directional_albedo_closure(
    compiled: &CompiledMaterial,
    locals: &[Value],
    sv: &ShadingVertex,
    wo_local: Vec3,
) -> Vec3 {
    directional_albedo_idx(compiled, locals, sv, compiled.root, wo_local, None)
}

#[derive(Debug, Clone, Copy)]
struct LightTreeClosureSummary {
    frame: OrthonormalBasis,
    n: Vec3,
    diffuse_rho: f32,
    glossy_rho: f32,
    glossy_alpha: (f32, f32),
    btdf_rho: f32,
    btdf_alpha: (f32, f32),
    btdf_eta: f32,
}

impl LightTreeClosureSummary {
    fn new(sv: &ShadingVertex) -> Self {
        Self {
            frame: sv.frame,
            n: sv.ns,
            diffuse_rho: 0.0,
            glossy_rho: 0.0,
            glossy_alpha: (0.001, 0.001),
            btdf_rho: 0.0,
            btdf_alpha: (0.001, 0.001),
            btdf_eta: 1.0,
        }
    }

    fn from_frame(frame: OrthonormalBasis) -> Self {
        Self {
            frame,
            n: frame.normal(),
            diffuse_rho: 0.0,
            glossy_rho: 0.0,
            glossy_alpha: (0.001, 0.001),
            btdf_rho: 0.0,
            btdf_alpha: (0.001, 0.001),
            btdf_eta: 1.0,
        }
    }

    fn add_diffuse(&mut self, rho: f32) {
        self.diffuse_rho += rho.max(0.0);
    }

    fn add_glossy(&mut self, rho: f32, alpha: Vec2) {
        let rho = rho.max(0.0);
        if rho <= 0.0 {
            return;
        }
        let alpha = (alpha.x.clamp(0.001, 1.0), alpha.y.clamp(0.001, 1.0));
        if self.glossy_rho > 0.0 {
            self.glossy_alpha = crate::light_tree::merge_glossy_roughness(
                self.glossy_rho,
                self.glossy_alpha,
                rho,
                alpha,
            );
        } else {
            self.glossy_alpha = alpha;
        }
        self.glossy_rho += rho;
    }

    fn add_btdf(&mut self, rho: f32, alpha: Vec2, eta: f32) {
        let rho = rho.max(0.0);
        if rho <= 0.0 {
            return;
        }
        let alpha = (alpha.x.clamp(0.001, 1.0), alpha.y.clamp(0.001, 1.0));
        if self.btdf_rho > 0.0 {
            let total = self.btdf_rho + rho;
            self.btdf_alpha = crate::light_tree::merge_glossy_roughness(
                self.btdf_rho,
                self.btdf_alpha,
                rho,
                alpha,
            );
            self.btdf_eta = (self.btdf_eta * self.btdf_rho + eta * rho) / total;
        } else {
            self.btdf_alpha = alpha;
            self.btdf_eta = eta;
        }
        self.btdf_rho += rho;
    }

    fn scale(mut self, s: f32) -> Self {
        let s = s.max(0.0);
        self.diffuse_rho *= s;
        self.glossy_rho *= s;
        self.btdf_rho *= s;
        self
    }

    fn add_scaled(&mut self, other: Self, s: f32) {
        let other = other.scale(s);
        self.add_diffuse(other.diffuse_rho);
        if other.glossy_rho > 0.0 {
            self.add_glossy(
                other.glossy_rho,
                Vec2::new(other.glossy_alpha.0, other.glossy_alpha.1),
            );
        }
        if other.btdf_rho > 0.0 {
            self.add_btdf(
                other.btdf_rho,
                Vec2::new(other.btdf_alpha.0, other.btdf_alpha.1),
                other.btdf_eta,
            );
        }
    }

    fn into_precompute(self, sv: &ShadingVertex) -> Option<crate::light_tree::LightTreePrecompute> {
        let diffuse =
            (self.diffuse_rho > 0.0).then_some(crate::light_tree::DiffuseLobePrecompute {
                rho: self.diffuse_rho,
            });
        let glossy = crate::light_tree::make_glossy_lobe(
            self.glossy_rho,
            self.frame,
            sv.wo,
            self.glossy_alpha.0,
            self.glossy_alpha.1,
        );
        let btdf = crate::light_tree::make_btdf_lobe(
            self.btdf_rho,
            self.frame,
            sv.wo,
            self.btdf_alpha.0,
            self.btdf_alpha.1,
            self.btdf_eta,
        );
        if diffuse.is_none() && glossy.is_none() && btdf.is_none() {
            return None;
        }
        Some(crate::light_tree::LightTreePrecompute {
            p: sv.p,
            n: self.n,
            frame: self.frame,
            diffuse,
            glossy,
            btdf,
        })
    }
}

pub fn light_tree_precompute_closure(
    compiled: &CompiledMaterial,
    locals: &[Value],
    sv: &ShadingVertex,
    wo_local: Vec3,
) -> Option<crate::light_tree::LightTreePrecompute> {
    light_tree_precompute_closure_cached(compiled, locals, sv, wo_local, &[])
}

pub fn light_tree_precompute_closure_cached(
    compiled: &CompiledMaterial,
    locals: &[Value],
    sv: &ShadingVertex,
    wo_local: Vec3,
    dalbedo_cache: &[Cell<Option<Vec3>>],
) -> Option<crate::light_tree::LightTreePrecompute> {
    light_tree_summary_idx(
        compiled,
        locals,
        sv,
        compiled.root,
        wo_local,
        Some(dalbedo_cache),
    )
    .into_precompute(sv)
}

#[inline(always)]
fn sheen_lut(compiled: &CompiledMaterial) -> &crate::bsdf::SheenDirectionalAlbedoLut {
    compiled
        .sheen_lut
        .as_deref()
        .expect("MaterialX runtime requires Scene-installed Sheen directional albedo LUT")
}

fn remap_choice_u(u: f32, p_upper: f32, chose_upper: bool) -> f32 {
    let p_upper = p_upper.clamp(0.0, 1.0);
    let u = u.clamp(0.0, 1.0 - f32::EPSILON);
    if chose_upper {
        if p_upper <= 0.0 { 0.0 } else { u / p_upper }
    } else if p_upper >= 1.0 {
        0.0
    } else {
        (u - p_upper) / (1.0 - p_upper)
    }
    .clamp(0.0, 1.0 - f32::EPSILON)
}

#[inline(always)]
fn mtlx_dielectric_lut(
    compiled: &CompiledMaterial,
) -> &crate::bsdf::MtlxDielectricGgxDirectionalAlbedoLut {
    compiled
        .mtlx_dielectric_lut
        .as_deref()
        .expect("MaterialX runtime requires Scene-installed dielectric directional albedo LUT")
}

#[inline(always)]
fn mtlx_generalized_schlick_lut(
    compiled: &CompiledMaterial,
) -> &crate::bsdf::MtlxGeneralizedSchlickGgxDirectionalAlbedoLut {
    compiled.mtlx_generalized_schlick_lut.as_deref().expect(
        "MaterialX runtime requires Scene-installed generalized Schlick directional albedo LUT",
    )
}

#[inline(always)]
fn mtlx_dielectric_directional_albedo(
    compiled: &CompiledMaterial,
    wo: Vec3,
    weight: f32,
    tint: Vec3,
    ior: f32,
    roughness: Vec2,
    scatter_mode: ScatterMode,
    thinfilm_thickness: f32,
    thinfilm_ior: f32,
    front_face: bool,
) -> Vec3 {
    if matches!(scatter_mode, ScatterMode::Transmission) {
        return Vec3::ZERO;
    }
    if thinfilm_thickness > 0.0 {
        return DielectricBsdf::with_thin_film(
            weight,
            tint,
            ior,
            roughness,
            false,
            scatter_mode,
            thinfilm_thickness,
            thinfilm_ior,
            front_face,
        )
        .directional_albedo(wo);
    }
    let ior = ior.max(1.0e-3);
    let eta_rel = if front_face { ior } else { 1.0 / ior };
    let albedo = mtlx_dielectric_lut(compiled).lookup(wo.z, roughness.x, roughness.y, eta_rel);
    (tint.max(Vec3::ZERO) * weight.clamp(0.0, 1.0) * albedo).clamp(Vec3::ZERO, Vec3::ONE)
}

#[inline(always)]
fn mtlx_generalized_schlick_directional_albedo(
    compiled: &CompiledMaterial,
    wo: Vec3,
    weight: f32,
    color0: Vec3,
    color82: Vec3,
    color90: Vec3,
    exponent: f32,
    roughness: Vec2,
    scatter_mode: ScatterMode,
    thinfilm_thickness: f32,
    thinfilm_ior: f32,
    front_face: bool,
) -> Vec3 {
    if matches!(scatter_mode, ScatterMode::Transmission) {
        return Vec3::ZERO;
    }
    if thinfilm_thickness > 0.0 {
        return GeneralizedSchlickBsdf::with_thin_film(
            weight,
            color0,
            color82,
            color90,
            exponent,
            roughness,
            false,
            scatter_mode,
            thinfilm_thickness,
            thinfilm_ior,
            front_face,
        )
        .directional_albedo(wo);
    }
    let albedo = mtlx_generalized_schlick_lut(compiled).lookup(
        wo.z,
        roughness.x,
        roughness.y,
        color0,
        color90,
    );
    (albedo * weight.clamp(0.0, 1.0)).clamp(Vec3::ZERO, Vec3::ONE)
}

pub fn evaluate_le(
    compiled: &CompiledMaterial,
    locals: &[Value],
    sv: &ShadingVertex,
) -> Option<Vec3> {
    fn walk(compiled: &CompiledMaterial, locals: &[Value], sv: &ShadingVertex, idx: u32) -> Vec3 {
        match compiled.closure(idx) {
            ClosureNode::Surface { edf, .. } => {
                evaluate_edf(compiled, locals, *edf, sv.wo, &sv.frame)
            }
            ClosureNode::GoochShade {
                warm,
                cool,
                specular_intensity,
                shininess,
                light_direction,
            } => {
                let kernel = GoochShadeKernel {
                    warm: read_color3(warm, locals),
                    cool: read_color3(cool, locals),
                    specular_intensity: read_float(specular_intensity, locals),
                    shininess: read_float(shininess, locals),
                    light_direction: read_param(light_direction, locals).as_vector3(),
                };
                kernel.eval(sv.ns, sv.wo)
            }
            ClosureNode::Mix { bg, fg, mix, kind } => {
                let m = read_float(mix, locals).clamp(0.0, 1.0);
                if matches!(kind, super::compiled::ClosureKind::Surface) {
                    let op_fg = surface_opacity_at(compiled, locals, *fg);
                    let op_bg = surface_opacity_at(compiled, locals, *bg);
                    let w_fg = m * op_fg;
                    let w_bg = (1.0 - m) * op_bg;
                    let total = w_fg + w_bg;
                    if total <= 1.0e-6 {
                        return Vec3::ZERO;
                    }
                    let p_fg = w_fg / total;
                    walk(compiled, locals, sv, *bg) * (1.0 - p_fg)
                        + walk(compiled, locals, sv, *fg) * p_fg
                } else {
                    walk(compiled, locals, sv, *bg) * (1.0 - m)
                        + walk(compiled, locals, sv, *fg) * m
                }
            }
            ClosureNode::Layer { top, base } => {
                walk(compiled, locals, sv, *top) + walk(compiled, locals, sv, *base)
            }
            ClosureNode::Add { a, b, .. } => {
                walk(compiled, locals, sv, *a) + walk(compiled, locals, sv, *b)
            }
            ClosureNode::Multiply { inner, scale, .. } => {
                walk(compiled, locals, sv, *inner) * read_color3(scale, locals)
            }
            ClosureNode::Switch {
                which, branches, ..
            } => {
                let i = read_param(which, locals).as_integer().clamp(0, 9) as usize;
                walk(compiled, locals, sv, branches[i])
            }
            ClosureNode::IfGreater {
                value1,
                value2,
                then_branch,
                else_branch,
                ..
            } => {
                let v1 = read_float(value1, locals);
                let v2 = read_float(value2, locals);
                let pick = if v1 > v2 { *then_branch } else { *else_branch };
                walk(compiled, locals, sv, pick)
            }
            ClosureNode::IfGreaterEq {
                value1,
                value2,
                then_branch,
                else_branch,
                ..
            } => {
                let v1 = read_float(value1, locals);
                let v2 = read_float(value2, locals);
                let pick = if v1 >= v2 { *then_branch } else { *else_branch };
                walk(compiled, locals, sv, pick)
            }
            ClosureNode::IfEqual {
                value1,
                value2,
                then_branch,
                else_branch,
                ..
            } => {
                let v1 = read_float(value1, locals);
                let v2 = read_float(value2, locals);
                let pick = if v1 == v2 { *then_branch } else { *else_branch };
                walk(compiled, locals, sv, pick)
            }
            ClosureNode::Zero
            | ClosureNode::OrenNayarDiffuse { .. }
            | ClosureNode::BurleyDiffuse { .. }
            | ClosureNode::Translucent { .. }
            | ClosureNode::Dielectric { .. }
            | ClosureNode::Conductor { .. }
            | ClosureNode::GeneralizedSchlick { .. }
            | ClosureNode::Sheen { .. }
            | ClosureNode::ChiangHair { .. }
            | ClosureNode::ThinFilm { .. }
            | ClosureNode::UniformEdf { .. }
            | ClosureNode::ConicalEdf { .. }
            | ClosureNode::GeneralizedSchlickEdf { .. } => Vec3::ZERO,
        }
    }
    let r = walk(compiled, locals, sv, compiled.root);
    if r.x <= 0.0 && r.y <= 0.0 && r.z <= 0.0 {
        None
    } else {
        Some(r)
    }
}

fn sample_closure_idx(
    compiled: &CompiledMaterial,
    locals: &[Value],
    sv: &ShadingVertex,
    randoms: MaterialSampleRandoms,
    idx: u32,
    wo: Vec3,
    dalbedo_cache: DalbedoCache<'_>,
) -> Option<MtlxLobeSample> {
    let node = compiled.closure(idx);
    match node {
        ClosureNode::Zero => None,
        ClosureNode::OrenNayarDiffuse {
            weight,
            color,
            roughness,
            energy_compensation,
            normal,
        } => {
            let (new_frame, wo_use) = override_frame_for_wo(sv, locals, normal, &None, wo);
            let bsdf = OrenNayarDiffuseBsdf::new(
                read_float(weight, locals),
                read_color3(color, locals),
                read_float(roughness, locals),
                *energy_compensation,
            );
            let us = randoms.u_dir;
            bsdf.sample(wo_use, us).map(|s| MtlxLobeSample {
                wi_local: rebase_wi_out_of_frame(sv, &new_frame, s.wi_local),
                ..s
            })
        }
        ClosureNode::BurleyDiffuse {
            weight,
            color,
            roughness,
            normal,
        } => {
            let (new_frame, wo_use) = override_frame_for_wo(sv, locals, normal, &None, wo);
            let bsdf = BurleyDiffuseBsdf::new(
                read_float(weight, locals),
                read_color3(color, locals),
                read_float(roughness, locals),
            );
            let us = randoms.u_dir;
            bsdf.sample(wo_use, us).map(|s| MtlxLobeSample {
                wi_local: rebase_wi_out_of_frame(sv, &new_frame, s.wi_local),
                ..s
            })
        }
        ClosureNode::Translucent {
            weight,
            color,
            normal,
        } => {
            let (new_frame, wo_use) = override_frame_for_wo(sv, locals, normal, &None, wo);
            let bsdf = TranslucentBsdf::new(read_float(weight, locals), read_color3(color, locals));
            let us = randoms.u_dir;
            bsdf.sample(wo_use, us).map(|s| MtlxLobeSample {
                wi_local: rebase_wi_out_of_frame(sv, &new_frame, s.wi_local),
                ..s
            })
        }
        ClosureNode::Dielectric {
            weight,
            tint,
            ior,
            roughness,
            retroreflective,
            scatter_mode,
            thinfilm_thickness,
            thinfilm_ior,
            normal,
            tangent,
        } => {
            let (new_frame, wo_use) = override_frame_for_wo(sv, locals, normal, tangent, wo);
            let bsdf = DielectricBsdf::with_thin_film(
                read_float(weight, locals),
                read_color3(tint, locals),
                read_float(ior, locals),
                read_vec2(roughness, locals),
                *retroreflective,
                *scatter_mode,
                read_float(thinfilm_thickness, locals),
                read_float(thinfilm_ior, locals),
                compiled.thin_walled || sv.front_face,
            );
            let us = randoms.u_dir;
            let u_branch = randoms.u_layer;
            bsdf.sample(wo_use, us, u_branch).map(|s| MtlxLobeSample {
                wi_local: rebase_wi_out_of_frame(sv, &new_frame, s.wi_local),
                ..s
            })
        }
        ClosureNode::Conductor {
            weight,
            ior,
            extinction,
            roughness,
            retroreflective,
            thinfilm_thickness,
            thinfilm_ior,
            normal,
            tangent,
        } => {
            let (new_frame, wo_use) = override_frame_for_wo(sv, locals, normal, tangent, wo);
            let bsdf = ConductorBsdf::with_thin_film(
                read_float(weight, locals),
                read_color3(ior, locals),
                read_color3(extinction, locals),
                read_vec2(roughness, locals),
                *retroreflective,
                read_float(thinfilm_thickness, locals),
                read_float(thinfilm_ior, locals),
            );
            let us = randoms.u_dir;
            bsdf.sample(wo_use, us).map(|s| MtlxLobeSample {
                wi_local: rebase_wi_out_of_frame(sv, &new_frame, s.wi_local),
                ..s
            })
        }
        ClosureNode::GeneralizedSchlick {
            weight,
            color0,
            color82,
            color90,
            exponent,
            roughness,
            retroreflective,
            scatter_mode,
            thinfilm_thickness,
            thinfilm_ior,
            normal,
            tangent,
        } => {
            let (new_frame, wo_use) = override_frame_for_wo(sv, locals, normal, tangent, wo);
            let bsdf = GeneralizedSchlickBsdf::with_thin_film(
                read_float(weight, locals),
                read_color3(color0, locals),
                read_color3(color82, locals),
                read_color3(color90, locals),
                read_float(exponent, locals),
                read_vec2(roughness, locals),
                *retroreflective,
                *scatter_mode,
                read_float(thinfilm_thickness, locals),
                read_float(thinfilm_ior, locals),
                compiled.thin_walled || sv.front_face,
            );
            let us = randoms.u_dir;
            let u_branch = randoms.u_layer;
            bsdf.sample(wo_use, us, u_branch).map(|s| MtlxLobeSample {
                wi_local: rebase_wi_out_of_frame(sv, &new_frame, s.wi_local),
                ..s
            })
        }
        ClosureNode::Sheen {
            weight,
            color,
            roughness,
            mode,
            normal,
        } => {
            let (new_frame, wo_use) = override_frame_for_wo(sv, locals, normal, &None, wo);
            let bsdf = SheenBsdfMtlx::new(
                read_float(weight, locals),
                read_color3(color, locals),
                read_float(roughness, locals),
                *mode,
            );
            let us = randoms.u_dir;
            bsdf.sample(wo_use, us).map(|s| MtlxLobeSample {
                wi_local: rebase_wi_out_of_frame(sv, &new_frame, s.wi_local),
                ..s
            })
        }
        ClosureNode::ChiangHair {
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
        } => {
            let h = -1.0 + 2.0 * sv.uv.y;
            let (new_frame, wo_use) = override_frame_for_wo(sv, locals, normal, &None, wo);
            let curve_direction_use = rebase_wi_into_frame(
                sv,
                &new_frame,
                read_param(curve_direction, locals).as_vector3(),
            );
            let bsdf = ChiangHairBsdf::from_mtlx(
                read_color3(tint_r, locals),
                read_color3(tint_tt, locals),
                read_color3(tint_trt, locals),
                read_float(ior, locals),
                read_vec2(roughness_r, locals),
                read_vec2(roughness_tt, locals),
                read_vec2(roughness_trt, locals),
                read_float(cuticle_angle, locals),
                read_color3(absorption, locals),
                curve_direction_use,
                h,
            );
            let us = randoms.u_dir;
            bsdf.sample(wo_use, us).map(|s| MtlxLobeSample {
                wi_local: rebase_wi_out_of_frame(sv, &new_frame, s.wi_local),
                ..s
            })
        }
        ClosureNode::ThinFilm { .. } => None,
        ClosureNode::Mix { bg, fg, mix, kind } => {
            let m = read_float(mix, locals).clamp(0.0, 1.0);
            let p_fg = if matches!(kind, super::compiled::ClosureKind::Surface) {
                let op_fg = surface_opacity_at(compiled, locals, *fg);
                let op_bg = surface_opacity_at(compiled, locals, *bg);
                let w_fg = m * op_fg;
                let w_bg = (1.0 - m) * op_bg;
                let total = w_fg + w_bg;
                if total <= 1.0e-6 {
                    return None;
                }
                w_fg / total
            } else {
                m
            };
            if randoms.u_lobe < p_fg {
                let child_randoms = randoms.with_lobe(remap_choice_u(randoms.u_lobe, p_fg, true));
                sample_closure_idx(compiled, locals, sv, child_randoms, *fg, wo, dalbedo_cache).map(
                    |s| MtlxLobeSample {
                        pdf: s.pdf * p_fg,
                        ..s
                    },
                )
            } else {
                let child_randoms = randoms.with_lobe(remap_choice_u(randoms.u_lobe, p_fg, false));
                sample_closure_idx(compiled, locals, sv, child_randoms, *bg, wo, dalbedo_cache).map(
                    |s| MtlxLobeSample {
                        pdf: s.pdf * (1.0 - p_fg),
                        ..s
                    },
                )
            }
        }
        ClosureNode::Layer { top, base } => {
            let top_node = compiled.closure(*top);
            if let ClosureNode::ThinFilm { .. } = top_node {
                return sample_closure_idx(compiled, locals, sv, randoms, *base, wo, dalbedo_cache);
            }
            let r_top =
                directional_albedo_idx_scalar(compiled, locals, sv, *top, wo, dalbedo_cache)
                    .clamp(0.0, 1.0);
            if randoms.u_lobe < r_top {
                let child_randoms = randoms.with_lobe(remap_choice_u(randoms.u_lobe, r_top, true));
                sample_closure_idx(compiled, locals, sv, child_randoms, *top, wo, dalbedo_cache)
                    .map(|s| MtlxLobeSample {
                        pdf: s.pdf * r_top,
                        ..s
                    })
            } else {
                let child_randoms = randoms.with_lobe(remap_choice_u(randoms.u_lobe, r_top, false));
                sample_closure_idx(
                    compiled,
                    locals,
                    sv,
                    child_randoms,
                    *base,
                    wo,
                    dalbedo_cache,
                )
                .map(|s| MtlxLobeSample {
                    pdf: s.pdf * (1.0 - r_top),
                    ..s
                })
            }
        }
        ClosureNode::Add {
            a,
            b,
            kind: super::compiled::ClosureKind::Bsdf,
        } => {
            if randoms.u_lobe < 0.5 {
                let child_randoms = randoms.with_lobe(remap_choice_u(randoms.u_lobe, 0.5, true));
                sample_closure_idx(compiled, locals, sv, child_randoms, *a, wo, dalbedo_cache).map(
                    |s| MtlxLobeSample {
                        pdf: s.pdf * 0.5,
                        ..s
                    },
                )
            } else {
                let child_randoms = randoms.with_lobe(remap_choice_u(randoms.u_lobe, 0.5, false));
                sample_closure_idx(compiled, locals, sv, child_randoms, *b, wo, dalbedo_cache).map(
                    |s| MtlxLobeSample {
                        pdf: s.pdf * 0.5,
                        ..s
                    },
                )
            }
        }
        ClosureNode::Add { a, b, .. } => {
            let wa = directional_albedo_idx_scalar(compiled, locals, sv, *a, wo, dalbedo_cache)
                .max(1e-3);
            let wb = directional_albedo_idx_scalar(compiled, locals, sv, *b, wo, dalbedo_cache)
                .max(1e-3);
            let total = wa + wb;
            let p_a = wa / total;
            if randoms.u_lobe < p_a {
                let child_randoms = randoms.with_lobe(remap_choice_u(randoms.u_lobe, p_a, true));
                sample_closure_idx(compiled, locals, sv, child_randoms, *a, wo, dalbedo_cache).map(
                    |s| MtlxLobeSample {
                        pdf: s.pdf * p_a,
                        ..s
                    },
                )
            } else {
                let child_randoms = randoms.with_lobe(remap_choice_u(randoms.u_lobe, p_a, false));
                sample_closure_idx(compiled, locals, sv, child_randoms, *b, wo, dalbedo_cache).map(
                    |s| MtlxLobeSample {
                        pdf: s.pdf * (1.0 - p_a),
                        ..s
                    },
                )
            }
        }
        ClosureNode::Multiply { inner, scale, .. } => {
            let s = read_color3(scale, locals);
            sample_closure_idx(compiled, locals, sv, randoms, *inner, wo, dalbedo_cache).map(|sm| {
                MtlxLobeSample {
                    weight: sm.weight * s,
                    ..sm
                }
            })
        }
        ClosureNode::IfGreater {
            value1,
            value2,
            then_branch,
            else_branch,
            ..
        } => {
            let v1 = read_float(value1, locals);
            let v2 = read_float(value2, locals);
            let pick = if v1 > v2 { *then_branch } else { *else_branch };
            sample_closure_idx(compiled, locals, sv, randoms, pick, wo, dalbedo_cache)
        }
        ClosureNode::IfGreaterEq {
            value1,
            value2,
            then_branch,
            else_branch,
            ..
        } => {
            let v1 = read_float(value1, locals);
            let v2 = read_float(value2, locals);
            let pick = if v1 >= v2 { *then_branch } else { *else_branch };
            sample_closure_idx(compiled, locals, sv, randoms, pick, wo, dalbedo_cache)
        }
        ClosureNode::IfEqual {
            value1,
            value2,
            then_branch,
            else_branch,
            ..
        } => {
            let v1 = read_float(value1, locals);
            let v2 = read_float(value2, locals);
            let pick = if v1 == v2 { *then_branch } else { *else_branch };
            sample_closure_idx(compiled, locals, sv, randoms, pick, wo, dalbedo_cache)
        }
        ClosureNode::Switch {
            which, branches, ..
        } => {
            let i = read_param(which, locals).as_integer().clamp(0, 9) as usize;
            sample_closure_idx(
                compiled,
                locals,
                sv,
                randoms,
                branches[i],
                wo,
                dalbedo_cache,
            )
        }
        ClosureNode::Surface { bsdf, .. } => {
            sample_closure_idx(compiled, locals, sv, randoms, *bsdf, wo, dalbedo_cache)
        }
        ClosureNode::UniformEdf { .. }
        | ClosureNode::ConicalEdf { .. }
        | ClosureNode::GeneralizedSchlickEdf { .. }
        | ClosureNode::GoochShade { .. } => None,
    }
}

fn eval_closure_idx(
    compiled: &CompiledMaterial,
    locals: &[Value],
    sv: &ShadingVertex,
    idx: u32,
    wo: Vec3,
    wi: Vec3,
    dalbedo_cache: DalbedoCache<'_>,
) -> Vec3 {
    let node = compiled.closure(idx);
    match node {
        ClosureNode::Zero => Vec3::ZERO,
        ClosureNode::OrenNayarDiffuse {
            weight,
            color,
            roughness,
            energy_compensation,
            normal,
        } => {
            let (nf, wo_use) = override_frame_for_wo(sv, locals, normal, &None, wo);
            let wi_use = rebase_wi_into_frame(sv, &nf, wi);
            OrenNayarDiffuseBsdf::new(
                read_float(weight, locals),
                read_color3(color, locals),
                read_float(roughness, locals),
                *energy_compensation,
            )
            .eval(wo_use, wi_use)
        }
        ClosureNode::BurleyDiffuse {
            weight,
            color,
            roughness,
            normal,
        } => {
            let (nf, wo_use) = override_frame_for_wo(sv, locals, normal, &None, wo);
            let wi_use = rebase_wi_into_frame(sv, &nf, wi);
            BurleyDiffuseBsdf::new(
                read_float(weight, locals),
                read_color3(color, locals),
                read_float(roughness, locals),
            )
            .eval(wo_use, wi_use)
        }
        ClosureNode::Translucent {
            weight,
            color,
            normal,
        } => {
            let (nf, wo_use) = override_frame_for_wo(sv, locals, normal, &None, wo);
            let wi_use = rebase_wi_into_frame(sv, &nf, wi);
            TranslucentBsdf::new(read_float(weight, locals), read_color3(color, locals))
                .eval(wo_use, wi_use)
        }
        ClosureNode::Dielectric {
            weight,
            tint,
            ior,
            roughness,
            retroreflective,
            scatter_mode,
            thinfilm_thickness,
            thinfilm_ior,
            normal,
            tangent,
        } => {
            let (nf, wo_use) = override_frame_for_wo(sv, locals, normal, tangent, wo);
            let wi_use = rebase_wi_into_frame(sv, &nf, wi);
            DielectricBsdf::with_thin_film(
                read_float(weight, locals),
                read_color3(tint, locals),
                read_float(ior, locals),
                read_vec2(roughness, locals),
                *retroreflective,
                *scatter_mode,
                read_float(thinfilm_thickness, locals),
                read_float(thinfilm_ior, locals),
                compiled.thin_walled || sv.front_face,
            )
            .eval(wo_use, wi_use)
        }
        ClosureNode::Conductor {
            weight,
            ior,
            extinction,
            roughness,
            retroreflective,
            thinfilm_thickness,
            thinfilm_ior,
            normal,
            tangent,
        } => {
            let (nf, wo_use) = override_frame_for_wo(sv, locals, normal, tangent, wo);
            let wi_use = rebase_wi_into_frame(sv, &nf, wi);
            ConductorBsdf::with_thin_film(
                read_float(weight, locals),
                read_color3(ior, locals),
                read_color3(extinction, locals),
                read_vec2(roughness, locals),
                *retroreflective,
                read_float(thinfilm_thickness, locals),
                read_float(thinfilm_ior, locals),
            )
            .eval(wo_use, wi_use)
        }
        ClosureNode::GeneralizedSchlick {
            weight,
            color0,
            color82,
            color90,
            exponent,
            roughness,
            retroreflective,
            scatter_mode,
            thinfilm_thickness,
            thinfilm_ior,
            normal,
            tangent,
        } => {
            let (nf, wo_use) = override_frame_for_wo(sv, locals, normal, tangent, wo);
            let wi_use = rebase_wi_into_frame(sv, &nf, wi);
            GeneralizedSchlickBsdf::with_thin_film(
                read_float(weight, locals),
                read_color3(color0, locals),
                read_color3(color82, locals),
                read_color3(color90, locals),
                read_float(exponent, locals),
                read_vec2(roughness, locals),
                *retroreflective,
                *scatter_mode,
                read_float(thinfilm_thickness, locals),
                read_float(thinfilm_ior, locals),
                compiled.thin_walled || sv.front_face,
            )
            .eval(wo_use, wi_use)
        }
        ClosureNode::Sheen {
            weight,
            color,
            roughness,
            mode,
            normal,
        } => {
            let (nf, wo_use) = override_frame_for_wo(sv, locals, normal, &None, wo);
            let wi_use = rebase_wi_into_frame(sv, &nf, wi);
            SheenBsdfMtlx::new(
                read_float(weight, locals),
                read_color3(color, locals),
                read_float(roughness, locals),
                *mode,
            )
            .eval(wo_use, wi_use)
        }
        ClosureNode::ChiangHair {
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
        } => {
            let h = -1.0 + 2.0 * sv.uv.y;
            let (nf, wo_use) = override_frame_for_wo(sv, locals, normal, &None, wo);
            let wi_use = rebase_wi_into_frame(sv, &nf, wi);
            let curve_direction_use =
                rebase_wi_into_frame(sv, &nf, read_param(curve_direction, locals).as_vector3());
            ChiangHairBsdf::from_mtlx(
                read_color3(tint_r, locals),
                read_color3(tint_tt, locals),
                read_color3(tint_trt, locals),
                read_float(ior, locals),
                read_vec2(roughness_r, locals),
                read_vec2(roughness_tt, locals),
                read_vec2(roughness_trt, locals),
                read_float(cuticle_angle, locals),
                read_color3(absorption, locals),
                curve_direction_use,
                h,
            )
            .eval(wo_use, wi_use)
        }
        ClosureNode::ThinFilm { .. } => Vec3::ZERO,
        ClosureNode::Mix { bg, fg, mix, kind } => {
            let m = read_float(mix, locals);
            let m = if matches!(kind, super::compiled::ClosureKind::Surface) {
                m
            } else {
                m.clamp(0.0, 1.0)
            };
            if matches!(kind, super::compiled::ClosureKind::Surface) {
                let op_fg = surface_opacity_at(compiled, locals, *fg);
                let op_bg = surface_opacity_at(compiled, locals, *bg);
                let w_fg = m * op_fg;
                let w_bg = (1.0 - m) * op_bg;
                let total = w_fg + w_bg;
                if total <= 1.0e-6 {
                    return Vec3::ZERO;
                }
                let p_fg = w_fg / total;
                eval_closure_idx(compiled, locals, sv, *bg, wo, wi, dalbedo_cache) * (1.0 - p_fg)
                    + eval_closure_idx(compiled, locals, sv, *fg, wo, wi, dalbedo_cache) * p_fg
            } else {
                eval_closure_idx(compiled, locals, sv, *bg, wo, wi, dalbedo_cache) * (1.0 - m)
                    + eval_closure_idx(compiled, locals, sv, *fg, wo, wi, dalbedo_cache) * m
            }
        }
        ClosureNode::Layer { top, base } => {
            let top_node = compiled.closure(*top);
            if let ClosureNode::ThinFilm { .. } = top_node {
                return eval_closure_idx(compiled, locals, sv, *base, wo, wi, dalbedo_cache);
            }
            let r = directional_albedo_idx(compiled, locals, sv, *top, wo, dalbedo_cache);
            eval_closure_idx(compiled, locals, sv, *top, wo, wi, dalbedo_cache)
                + eval_closure_idx(compiled, locals, sv, *base, wo, wi, dalbedo_cache)
                    * (Vec3::ONE - r)
        }
        ClosureNode::Add {
            a,
            b,
            kind: super::compiled::ClosureKind::Bsdf,
        } => {
            (eval_closure_idx(compiled, locals, sv, *a, wo, wi, dalbedo_cache)
                + eval_closure_idx(compiled, locals, sv, *b, wo, wi, dalbedo_cache))
                * 0.5
        }
        ClosureNode::Add { a, b, .. } => {
            eval_closure_idx(compiled, locals, sv, *a, wo, wi, dalbedo_cache)
                + eval_closure_idx(compiled, locals, sv, *b, wo, wi, dalbedo_cache)
        }
        ClosureNode::Multiply { inner, scale, .. } => {
            eval_closure_idx(compiled, locals, sv, *inner, wo, wi, dalbedo_cache)
                * read_color3(scale, locals)
        }
        ClosureNode::IfGreater {
            value1,
            value2,
            then_branch,
            else_branch,
            ..
        } => {
            let v1 = read_float(value1, locals);
            let v2 = read_float(value2, locals);
            let pick = if v1 > v2 { *then_branch } else { *else_branch };
            eval_closure_idx(compiled, locals, sv, pick, wo, wi, dalbedo_cache)
        }
        ClosureNode::IfGreaterEq {
            value1,
            value2,
            then_branch,
            else_branch,
            ..
        } => {
            let v1 = read_float(value1, locals);
            let v2 = read_float(value2, locals);
            let pick = if v1 >= v2 { *then_branch } else { *else_branch };
            eval_closure_idx(compiled, locals, sv, pick, wo, wi, dalbedo_cache)
        }
        ClosureNode::IfEqual {
            value1,
            value2,
            then_branch,
            else_branch,
            ..
        } => {
            let v1 = read_float(value1, locals);
            let v2 = read_float(value2, locals);
            let pick = if v1 == v2 { *then_branch } else { *else_branch };
            eval_closure_idx(compiled, locals, sv, pick, wo, wi, dalbedo_cache)
        }
        ClosureNode::Switch {
            which, branches, ..
        } => {
            let i = read_param(which, locals).as_integer().clamp(0, 9) as usize;
            eval_closure_idx(compiled, locals, sv, branches[i], wo, wi, dalbedo_cache)
        }
        ClosureNode::Surface { bsdf, .. } => {
            eval_closure_idx(compiled, locals, sv, *bsdf, wo, wi, dalbedo_cache)
        }
        ClosureNode::UniformEdf { .. }
        | ClosureNode::ConicalEdf { .. }
        | ClosureNode::GeneralizedSchlickEdf { .. }
        | ClosureNode::GoochShade { .. } => Vec3::ZERO,
    }
}

fn pdf_closure_idx(
    compiled: &CompiledMaterial,
    locals: &[Value],
    sv: &ShadingVertex,
    idx: u32,
    wo: Vec3,
    wi: Vec3,
    dalbedo_cache: DalbedoCache<'_>,
) -> f32 {
    use std::f32::consts::PI;
    let node = compiled.closure(idx);
    match node {
        ClosureNode::Zero => 0.0,
        ClosureNode::OrenNayarDiffuse { normal, .. } => {
            let (nf, wo_use) = override_frame_for_wo(sv, locals, normal, &None, wo);
            let wi_use = rebase_wi_into_frame(sv, &nf, wi);
            if wo_use.z <= 0.0 || wi_use.z <= 0.0 {
                0.0
            } else {
                wi_use.z / PI
            }
        }
        ClosureNode::BurleyDiffuse { normal, .. } => {
            let (nf, wo_use) = override_frame_for_wo(sv, locals, normal, &None, wo);
            let wi_use = rebase_wi_into_frame(sv, &nf, wi);
            if wo_use.z <= 0.0 || wi_use.z <= 0.0 {
                0.0
            } else {
                wi_use.z / PI
            }
        }
        ClosureNode::Sheen {
            weight,
            color,
            roughness,
            mode,
            normal,
        } => {
            let (nf, wo_use) = override_frame_for_wo(sv, locals, normal, &None, wo);
            let wi_use = rebase_wi_into_frame(sv, &nf, wi);
            SheenBsdfMtlx::new(
                read_float(weight, locals),
                read_color3(color, locals),
                read_float(roughness, locals),
                *mode,
            )
            .pdf(wo_use, wi_use)
        }
        ClosureNode::Translucent { normal, .. } => {
            let (nf, wo_use) = override_frame_for_wo(sv, locals, normal, &None, wo);
            let wi_use = rebase_wi_into_frame(sv, &nf, wi);
            if wo_use.z <= 0.0 || wi_use.z >= 0.0 {
                0.0
            } else {
                -wi_use.z / PI
            }
        }
        ClosureNode::Dielectric {
            ior,
            roughness,
            retroreflective,
            scatter_mode,
            thinfilm_thickness,
            thinfilm_ior,
            normal,
            tangent,
            ..
        } => {
            let (nf, wo_use) = override_frame_for_wo(sv, locals, normal, tangent, wo);
            let wi_use = rebase_wi_into_frame(sv, &nf, wi);
            DielectricBsdf::with_thin_film(
                1.0,
                Vec3::ONE,
                read_float(ior, locals),
                read_vec2(roughness, locals),
                *retroreflective,
                *scatter_mode,
                read_float(thinfilm_thickness, locals),
                read_float(thinfilm_ior, locals),
                compiled.thin_walled || sv.front_face,
            )
            .pdf(wo_use, wi_use)
        }
        ClosureNode::Conductor {
            roughness,
            normal,
            tangent,
            ..
        } => {
            let (nf, wo_use) = override_frame_for_wo(sv, locals, normal, tangent, wo);
            let wi_use = rebase_wi_into_frame(sv, &nf, wi);
            let bsdf = ConductorBsdf::new(1.0, Vec3::ONE, Vec3::ZERO, read_vec2(roughness, locals));
            bsdf.pdf(wo_use, wi_use)
        }
        ClosureNode::GeneralizedSchlick {
            weight,
            color0,
            color82,
            color90,
            exponent,
            roughness,
            retroreflective,
            scatter_mode,
            thinfilm_thickness,
            thinfilm_ior,
            normal,
            tangent,
        } => {
            let (nf, wo_use) = override_frame_for_wo(sv, locals, normal, tangent, wo);
            let wi_use = rebase_wi_into_frame(sv, &nf, wi);
            GeneralizedSchlickBsdf::with_thin_film(
                read_float(weight, locals),
                read_color3(color0, locals),
                read_color3(color82, locals),
                read_color3(color90, locals),
                read_float(exponent, locals),
                read_vec2(roughness, locals),
                *retroreflective,
                *scatter_mode,
                read_float(thinfilm_thickness, locals),
                read_float(thinfilm_ior, locals),
                compiled.thin_walled || sv.front_face,
            )
            .pdf(wo_use, wi_use)
        }
        ClosureNode::ChiangHair {
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
        } => {
            let h = -1.0 + 2.0 * sv.uv.y;
            let (nf, wo_use) = override_frame_for_wo(sv, locals, normal, &None, wo);
            let wi_use = rebase_wi_into_frame(sv, &nf, wi);
            let curve_direction_use =
                rebase_wi_into_frame(sv, &nf, read_param(curve_direction, locals).as_vector3());
            ChiangHairBsdf::from_mtlx(
                read_color3(tint_r, locals),
                read_color3(tint_tt, locals),
                read_color3(tint_trt, locals),
                read_float(ior, locals),
                read_vec2(roughness_r, locals),
                read_vec2(roughness_tt, locals),
                read_vec2(roughness_trt, locals),
                read_float(cuticle_angle, locals),
                read_color3(absorption, locals),
                curve_direction_use,
                h,
            )
            .pdf(wo_use, wi_use)
        }
        ClosureNode::ThinFilm { .. } => 0.0,
        ClosureNode::Mix { bg, fg, mix, kind } => {
            let m = read_float(mix, locals);
            let m = if matches!(kind, super::compiled::ClosureKind::Surface) {
                m
            } else {
                m.clamp(0.0, 1.0)
            };
            if matches!(kind, super::compiled::ClosureKind::Surface) {
                let op_fg = surface_opacity_at(compiled, locals, *fg);
                let op_bg = surface_opacity_at(compiled, locals, *bg);
                let w_fg = m * op_fg;
                let w_bg = (1.0 - m) * op_bg;
                let total = w_fg + w_bg;
                if total <= 1.0e-6 {
                    return 0.0;
                }
                let p_fg = w_fg / total;
                pdf_closure_idx(compiled, locals, sv, *bg, wo, wi, dalbedo_cache) * (1.0 - p_fg)
                    + pdf_closure_idx(compiled, locals, sv, *fg, wo, wi, dalbedo_cache) * p_fg
            } else {
                pdf_closure_idx(compiled, locals, sv, *bg, wo, wi, dalbedo_cache) * (1.0 - m)
                    + pdf_closure_idx(compiled, locals, sv, *fg, wo, wi, dalbedo_cache) * m
            }
        }
        ClosureNode::Layer { top, base } => {
            let top_node = compiled.closure(*top);
            if let ClosureNode::ThinFilm { .. } = top_node {
                return pdf_closure_idx(compiled, locals, sv, *base, wo, wi, dalbedo_cache);
            }
            let r_top =
                directional_albedo_idx_scalar(compiled, locals, sv, *top, wo, dalbedo_cache)
                    .clamp(0.0, 1.0);
            pdf_closure_idx(compiled, locals, sv, *top, wo, wi, dalbedo_cache) * r_top
                + pdf_closure_idx(compiled, locals, sv, *base, wo, wi, dalbedo_cache)
                    * (1.0 - r_top)
        }
        ClosureNode::Add {
            a,
            b,
            kind: super::compiled::ClosureKind::Bsdf,
        } => {
            (pdf_closure_idx(compiled, locals, sv, *a, wo, wi, dalbedo_cache)
                + pdf_closure_idx(compiled, locals, sv, *b, wo, wi, dalbedo_cache))
                * 0.5
        }
        ClosureNode::Add { a, b, .. } => {
            let wa = directional_albedo_idx_scalar(compiled, locals, sv, *a, wo, dalbedo_cache)
                .max(1e-3);
            let wb = directional_albedo_idx_scalar(compiled, locals, sv, *b, wo, dalbedo_cache)
                .max(1e-3);
            let total = wa + wb;
            pdf_closure_idx(compiled, locals, sv, *a, wo, wi, dalbedo_cache) * wa / total
                + pdf_closure_idx(compiled, locals, sv, *b, wo, wi, dalbedo_cache) * wb / total
        }
        ClosureNode::Multiply { inner, .. } => {
            pdf_closure_idx(compiled, locals, sv, *inner, wo, wi, dalbedo_cache)
        }
        ClosureNode::IfGreater {
            value1,
            value2,
            then_branch,
            else_branch,
            ..
        } => {
            let v1 = read_float(value1, locals);
            let v2 = read_float(value2, locals);
            let pick = if v1 > v2 { *then_branch } else { *else_branch };
            pdf_closure_idx(compiled, locals, sv, pick, wo, wi, dalbedo_cache)
        }
        ClosureNode::IfGreaterEq {
            value1,
            value2,
            then_branch,
            else_branch,
            ..
        } => {
            let v1 = read_float(value1, locals);
            let v2 = read_float(value2, locals);
            let pick = if v1 >= v2 { *then_branch } else { *else_branch };
            pdf_closure_idx(compiled, locals, sv, pick, wo, wi, dalbedo_cache)
        }
        ClosureNode::IfEqual {
            value1,
            value2,
            then_branch,
            else_branch,
            ..
        } => {
            let v1 = read_float(value1, locals);
            let v2 = read_float(value2, locals);
            let pick = if v1 == v2 { *then_branch } else { *else_branch };
            pdf_closure_idx(compiled, locals, sv, pick, wo, wi, dalbedo_cache)
        }
        ClosureNode::Switch {
            which, branches, ..
        } => {
            let i = read_param(which, locals).as_integer().clamp(0, 9) as usize;
            pdf_closure_idx(compiled, locals, sv, branches[i], wo, wi, dalbedo_cache)
        }
        ClosureNode::Surface { bsdf, .. } => {
            pdf_closure_idx(compiled, locals, sv, *bsdf, wo, wi, dalbedo_cache)
        }
        ClosureNode::UniformEdf { .. }
        | ClosureNode::ConicalEdf { .. }
        | ClosureNode::GeneralizedSchlickEdf { .. }
        | ClosureNode::GoochShade { .. } => 0.0,
    }
}

fn eval_pdf_closure_idx(
    compiled: &CompiledMaterial,
    locals: &[Value],
    sv: &ShadingVertex,
    idx: u32,
    wo: Vec3,
    wi: Vec3,
    dalbedo_cache: DalbedoCache<'_>,
) -> (Vec3, f32) {
    let node = compiled.closure(idx);
    match node {
        ClosureNode::Zero => (Vec3::ZERO, 0.0),
        ClosureNode::OrenNayarDiffuse {
            weight,
            color,
            roughness,
            energy_compensation,
            normal,
        } => {
            let (nf, wo_use) = override_frame_for_wo(sv, locals, normal, &None, wo);
            let wi_use = rebase_wi_into_frame(sv, &nf, wi);
            let bsdf = OrenNayarDiffuseBsdf::new(
                read_float(weight, locals),
                read_color3(color, locals),
                read_float(roughness, locals),
                *energy_compensation,
            );
            let pdf = if wo_use.z <= 0.0 || wi_use.z <= 0.0 {
                0.0
            } else {
                wi_use.z / std::f32::consts::PI
            };
            (bsdf.eval(wo_use, wi_use), pdf)
        }
        ClosureNode::BurleyDiffuse {
            weight,
            color,
            roughness,
            normal,
        } => {
            let (nf, wo_use) = override_frame_for_wo(sv, locals, normal, &None, wo);
            let wi_use = rebase_wi_into_frame(sv, &nf, wi);
            let bsdf = BurleyDiffuseBsdf::new(
                read_float(weight, locals),
                read_color3(color, locals),
                read_float(roughness, locals),
            );
            let pdf = if wo_use.z <= 0.0 || wi_use.z <= 0.0 {
                0.0
            } else {
                wi_use.z / std::f32::consts::PI
            };
            (bsdf.eval(wo_use, wi_use), pdf)
        }
        ClosureNode::Translucent {
            weight,
            color,
            normal,
        } => {
            let (nf, wo_use) = override_frame_for_wo(sv, locals, normal, &None, wo);
            let wi_use = rebase_wi_into_frame(sv, &nf, wi);
            let bsdf = TranslucentBsdf::new(read_float(weight, locals), read_color3(color, locals));
            let pdf = if wo_use.z <= 0.0 || wi_use.z >= 0.0 {
                0.0
            } else {
                -wi_use.z / std::f32::consts::PI
            };
            (bsdf.eval(wo_use, wi_use), pdf)
        }
        ClosureNode::Dielectric {
            weight,
            tint,
            ior,
            roughness,
            retroreflective,
            scatter_mode,
            thinfilm_thickness,
            thinfilm_ior,
            normal,
            tangent,
        } => {
            let (nf, wo_use) = override_frame_for_wo(sv, locals, normal, tangent, wo);
            let wi_use = rebase_wi_into_frame(sv, &nf, wi);
            let bsdf = DielectricBsdf::with_thin_film(
                read_float(weight, locals),
                read_color3(tint, locals),
                read_float(ior, locals),
                read_vec2(roughness, locals),
                *retroreflective,
                *scatter_mode,
                read_float(thinfilm_thickness, locals),
                read_float(thinfilm_ior, locals),
                compiled.thin_walled || sv.front_face,
            );
            (bsdf.eval(wo_use, wi_use), bsdf.pdf(wo_use, wi_use))
        }
        ClosureNode::Conductor {
            weight,
            ior,
            extinction,
            roughness,
            retroreflective,
            thinfilm_thickness,
            thinfilm_ior,
            normal,
            tangent,
        } => {
            let (nf, wo_use) = override_frame_for_wo(sv, locals, normal, tangent, wo);
            let wi_use = rebase_wi_into_frame(sv, &nf, wi);
            let rough = read_vec2(roughness, locals);
            let bsdf = ConductorBsdf::with_thin_film(
                read_float(weight, locals),
                read_color3(ior, locals),
                read_color3(extinction, locals),
                rough,
                *retroreflective,
                read_float(thinfilm_thickness, locals),
                read_float(thinfilm_ior, locals),
            );
            let pdf_bsdf = ConductorBsdf::with_thin_film(
                1.0,
                Vec3::ONE,
                Vec3::ZERO,
                rough,
                *retroreflective,
                0.0,
                1.5,
            );
            (bsdf.eval(wo_use, wi_use), pdf_bsdf.pdf(wo_use, wi_use))
        }
        ClosureNode::GeneralizedSchlick {
            weight,
            color0,
            color82,
            color90,
            exponent,
            roughness,
            retroreflective,
            scatter_mode,
            thinfilm_thickness,
            thinfilm_ior,
            normal,
            tangent,
        } => {
            let (nf, wo_use) = override_frame_for_wo(sv, locals, normal, tangent, wo);
            let wi_use = rebase_wi_into_frame(sv, &nf, wi);
            let bsdf = GeneralizedSchlickBsdf::with_thin_film(
                read_float(weight, locals),
                read_color3(color0, locals),
                read_color3(color82, locals),
                read_color3(color90, locals),
                read_float(exponent, locals),
                read_vec2(roughness, locals),
                *retroreflective,
                *scatter_mode,
                read_float(thinfilm_thickness, locals),
                read_float(thinfilm_ior, locals),
                compiled.thin_walled || sv.front_face,
            );
            (bsdf.eval(wo_use, wi_use), bsdf.pdf(wo_use, wi_use))
        }
        ClosureNode::Sheen {
            weight,
            color,
            roughness,
            mode,
            normal,
        } => {
            let (nf, wo_use) = override_frame_for_wo(sv, locals, normal, &None, wo);
            let wi_use = rebase_wi_into_frame(sv, &nf, wi);
            let bsdf = SheenBsdfMtlx::new(
                read_float(weight, locals),
                read_color3(color, locals),
                read_float(roughness, locals),
                *mode,
            );
            (bsdf.eval(wo_use, wi_use), bsdf.pdf(wo_use, wi_use))
        }
        ClosureNode::ChiangHair {
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
        } => {
            let h = -1.0 + 2.0 * sv.uv.y;
            let (nf, wo_use) = override_frame_for_wo(sv, locals, normal, &None, wo);
            let wi_use = rebase_wi_into_frame(sv, &nf, wi);
            let curve_direction_use =
                rebase_wi_into_frame(sv, &nf, read_param(curve_direction, locals).as_vector3());
            let bsdf = ChiangHairBsdf::from_mtlx(
                read_color3(tint_r, locals),
                read_color3(tint_tt, locals),
                read_color3(tint_trt, locals),
                read_float(ior, locals),
                read_vec2(roughness_r, locals),
                read_vec2(roughness_tt, locals),
                read_vec2(roughness_trt, locals),
                read_float(cuticle_angle, locals),
                read_color3(absorption, locals),
                curve_direction_use,
                h,
            );
            (bsdf.eval(wo_use, wi_use), bsdf.pdf(wo_use, wi_use))
        }
        ClosureNode::ThinFilm { .. }
        | ClosureNode::UniformEdf { .. }
        | ClosureNode::ConicalEdf { .. }
        | ClosureNode::GeneralizedSchlickEdf { .. }
        | ClosureNode::GoochShade { .. } => (Vec3::ZERO, 0.0),
        ClosureNode::Mix { bg, fg, mix, kind } => {
            let m = read_float(mix, locals);
            let m = if matches!(kind, super::compiled::ClosureKind::Surface) {
                m
            } else {
                m.clamp(0.0, 1.0)
            };
            let p_fg = if matches!(kind, super::compiled::ClosureKind::Surface) {
                let op_fg = surface_opacity_at(compiled, locals, *fg);
                let op_bg = surface_opacity_at(compiled, locals, *bg);
                let w_fg = m * op_fg;
                let w_bg = (1.0 - m) * op_bg;
                let total = w_fg + w_bg;
                if total <= 1.0e-6 {
                    return (Vec3::ZERO, 0.0);
                }
                w_fg / total
            } else {
                m
            };
            let (f_bg, pdf_bg) =
                eval_pdf_closure_idx(compiled, locals, sv, *bg, wo, wi, dalbedo_cache);
            let (f_fg, pdf_fg) =
                eval_pdf_closure_idx(compiled, locals, sv, *fg, wo, wi, dalbedo_cache);
            (
                f_bg * (1.0 - p_fg) + f_fg * p_fg,
                pdf_bg * (1.0 - p_fg) + pdf_fg * p_fg,
            )
        }
        ClosureNode::Layer { top, base } => {
            if let ClosureNode::ThinFilm { .. } = compiled.closure(*top) {
                return eval_pdf_closure_idx(compiled, locals, sv, *base, wo, wi, dalbedo_cache);
            }
            let r = directional_albedo_idx(compiled, locals, sv, *top, wo, dalbedo_cache);
            let r_scalar = ((r.x + r.y + r.z) / 3.0).clamp(0.0, 1.0);
            let (f_top, pdf_top) =
                eval_pdf_closure_idx(compiled, locals, sv, *top, wo, wi, dalbedo_cache);
            let (f_base, pdf_base) =
                eval_pdf_closure_idx(compiled, locals, sv, *base, wo, wi, dalbedo_cache);
            (
                f_top + f_base * (Vec3::ONE - r),
                pdf_top * r_scalar + pdf_base * (1.0 - r_scalar),
            )
        }
        ClosureNode::Add {
            a,
            b,
            kind: super::compiled::ClosureKind::Bsdf,
        } => {
            let (fa, pa) = eval_pdf_closure_idx(compiled, locals, sv, *a, wo, wi, dalbedo_cache);
            let (fb, pb) = eval_pdf_closure_idx(compiled, locals, sv, *b, wo, wi, dalbedo_cache);
            ((fa + fb) * 0.5, (pa + pb) * 0.5)
        }
        ClosureNode::Add { a, b, .. } => {
            let wa = directional_albedo_idx_scalar(compiled, locals, sv, *a, wo, dalbedo_cache)
                .max(1e-3);
            let wb = directional_albedo_idx_scalar(compiled, locals, sv, *b, wo, dalbedo_cache)
                .max(1e-3);
            let total = wa + wb;
            let (fa, pa) = eval_pdf_closure_idx(compiled, locals, sv, *a, wo, wi, dalbedo_cache);
            let (fb, pb) = eval_pdf_closure_idx(compiled, locals, sv, *b, wo, wi, dalbedo_cache);
            (fa + fb, pa * wa / total + pb * wb / total)
        }
        ClosureNode::Multiply { inner, scale, .. } => {
            let (f, pdf) =
                eval_pdf_closure_idx(compiled, locals, sv, *inner, wo, wi, dalbedo_cache);
            (f * read_color3(scale, locals), pdf)
        }
        ClosureNode::IfGreater {
            value1,
            value2,
            then_branch,
            else_branch,
            ..
        } => {
            let pick = if read_float(value1, locals) > read_float(value2, locals) {
                *then_branch
            } else {
                *else_branch
            };
            eval_pdf_closure_idx(compiled, locals, sv, pick, wo, wi, dalbedo_cache)
        }
        ClosureNode::IfGreaterEq {
            value1,
            value2,
            then_branch,
            else_branch,
            ..
        } => {
            let pick = if read_float(value1, locals) >= read_float(value2, locals) {
                *then_branch
            } else {
                *else_branch
            };
            eval_pdf_closure_idx(compiled, locals, sv, pick, wo, wi, dalbedo_cache)
        }
        ClosureNode::IfEqual {
            value1,
            value2,
            then_branch,
            else_branch,
            ..
        } => {
            let pick = if read_float(value1, locals) == read_float(value2, locals) {
                *then_branch
            } else {
                *else_branch
            };
            eval_pdf_closure_idx(compiled, locals, sv, pick, wo, wi, dalbedo_cache)
        }
        ClosureNode::Switch {
            which, branches, ..
        } => {
            let i = read_param(which, locals).as_integer().clamp(0, 9) as usize;
            eval_pdf_closure_idx(compiled, locals, sv, branches[i], wo, wi, dalbedo_cache)
        }
        ClosureNode::Surface { bsdf, .. } => {
            eval_pdf_closure_idx(compiled, locals, sv, *bsdf, wo, wi, dalbedo_cache)
        }
    }
}

fn light_tree_summary_idx(
    compiled: &CompiledMaterial,
    locals: &[Value],
    sv: &ShadingVertex,
    idx: u32,
    wo: Vec3,
    dalbedo_cache: DalbedoCache<'_>,
) -> LightTreeClosureSummary {
    match compiled.closure(idx) {
        ClosureNode::Zero
        | ClosureNode::ThinFilm { .. }
        | ClosureNode::UniformEdf { .. }
        | ClosureNode::ConicalEdf { .. }
        | ClosureNode::GeneralizedSchlickEdf { .. }
        | ClosureNode::GoochShade { .. } => LightTreeClosureSummary::new(sv),
        ClosureNode::OrenNayarDiffuse {
            weight,
            color,
            normal,
            ..
        }
        | ClosureNode::BurleyDiffuse {
            weight,
            color,
            normal,
            ..
        } => {
            let (frame, _) = override_frame_for_wo(sv, locals, normal, &None, wo);
            let mut out = LightTreeClosureSummary::from_frame(frame);
            out.add_diffuse(crate::math::sg::luminance(
                read_color3(color, locals) * read_float(weight, locals),
            ));
            out
        }
        ClosureNode::Translucent {
            weight,
            color,
            normal,
        } => {
            let (frame, _) = override_frame_for_wo(sv, locals, normal, &None, wo);
            let mut out = LightTreeClosureSummary::from_frame(frame);
            out.add_diffuse(crate::math::sg::luminance(
                read_color3(color, locals) * read_float(weight, locals),
            ));
            out
        }
        ClosureNode::Dielectric {
            weight,
            tint,
            ior,
            roughness,
            retroreflective: _,
            scatter_mode,
            thinfilm_thickness,
            thinfilm_ior,
            normal,
            tangent,
        } => {
            let (frame, wo_use) = override_frame_for_wo(sv, locals, normal, tangent, wo);
            let rough = read_vec2(roughness, locals);
            let albedo = mtlx_dielectric_directional_albedo(
                compiled,
                wo_use,
                read_float(weight, locals),
                read_color3(tint, locals),
                read_float(ior, locals),
                rough,
                *scatter_mode,
                read_float(thinfilm_thickness, locals),
                read_float(thinfilm_ior, locals),
                compiled.thin_walled || sv.front_face,
            );
            let mut out = LightTreeClosureSummary::from_frame(frame);
            let refl_rho = crate::math::sg::luminance(albedo);
            if !matches!(scatter_mode, ScatterMode::Transmission) {
                out.add_glossy(refl_rho, rough);
            }
            if !matches!(scatter_mode, ScatterMode::Reflection) && !compiled.thin_walled {
                let tint_rho = crate::math::sg::luminance(read_color3(tint, locals))
                    * read_float(weight, locals);
                let trans_rho = if matches!(scatter_mode, ScatterMode::Transmission) {
                    tint_rho
                } else {
                    (tint_rho - refl_rho).max(0.0)
                };
                let eta = read_float(ior, locals).max(1.0e-3);
                let eta_rel = if sv.front_face { 1.0 / eta } else { eta };
                out.add_btdf(trans_rho, rough, eta_rel);
            }
            out
        }
        ClosureNode::Conductor {
            weight,
            ior,
            extinction,
            roughness,
            retroreflective,
            thinfilm_thickness,
            thinfilm_ior,
            normal,
            tangent,
        } => {
            let (frame, wo_use) = override_frame_for_wo(sv, locals, normal, tangent, wo);
            let rough = read_vec2(roughness, locals);
            let bsdf = ConductorBsdf::with_thin_film(
                read_float(weight, locals),
                read_color3(ior, locals),
                read_color3(extinction, locals),
                rough,
                *retroreflective,
                read_float(thinfilm_thickness, locals),
                read_float(thinfilm_ior, locals),
            );
            let mut out = LightTreeClosureSummary::from_frame(frame);
            out.add_glossy(
                crate::math::sg::luminance(bsdf.directional_albedo(wo_use)),
                rough,
            );
            out
        }
        ClosureNode::GeneralizedSchlick {
            weight,
            color0,
            color82,
            color90,
            exponent,
            roughness,
            retroreflective: _,
            scatter_mode,
            thinfilm_thickness,
            thinfilm_ior,
            normal,
            tangent,
        } => {
            let (frame, wo_use) = override_frame_for_wo(sv, locals, normal, tangent, wo);
            let rough = read_vec2(roughness, locals);
            let albedo = mtlx_generalized_schlick_directional_albedo(
                compiled,
                wo_use,
                read_float(weight, locals),
                read_color3(color0, locals),
                read_color3(color82, locals),
                read_color3(color90, locals),
                read_float(exponent, locals),
                rough,
                *scatter_mode,
                read_float(thinfilm_thickness, locals),
                read_float(thinfilm_ior, locals),
                compiled.thin_walled || sv.front_face,
            );
            let mut out = LightTreeClosureSummary::from_frame(frame);
            let refl_rho = crate::math::sg::luminance(albedo);
            if !matches!(scatter_mode, ScatterMode::Transmission) {
                out.add_glossy(refl_rho, rough);
            }
            if !matches!(scatter_mode, ScatterMode::Reflection) && !compiled.thin_walled {
                let trans_rho = (read_float(weight, locals) - refl_rho).max(0.0);
                let eta_rel = if sv.front_face { 1.0 / 1.5 } else { 1.5 };
                out.add_btdf(trans_rho, rough, eta_rel);
            }
            out
        }
        ClosureNode::Sheen {
            weight,
            color,
            roughness,
            mode,
            normal,
        } => {
            let (frame, wo_use) = override_frame_for_wo(sv, locals, normal, &None, wo);
            let bsdf = SheenBsdfMtlx::new(
                read_float(weight, locals),
                read_color3(color, locals),
                read_float(roughness, locals),
                *mode,
            );
            let mut out = LightTreeClosureSummary::from_frame(frame);
            out.add_diffuse(crate::math::sg::luminance(
                bsdf.directional_albedo_with_lut(wo_use, Some(sheen_lut(compiled))),
            ));
            out
        }
        ClosureNode::ChiangHair {
            tint_r,
            tint_tt,
            tint_trt,
            normal,
            ..
        } => {
            let (frame, _) = override_frame_for_wo(sv, locals, normal, &None, wo);
            let mut out = LightTreeClosureSummary::from_frame(frame);
            out.add_diffuse(crate::math::sg::luminance(
                (read_color3(tint_r, locals)
                    + read_color3(tint_tt, locals)
                    + read_color3(tint_trt, locals))
                    / 3.0,
            ));
            out
        }
        ClosureNode::Mix { bg, fg, mix, .. } => {
            let m = read_float(mix, locals).clamp(0.0, 1.0);
            let bg = light_tree_summary_idx(compiled, locals, sv, *bg, wo, dalbedo_cache);
            let fg = light_tree_summary_idx(compiled, locals, sv, *fg, wo, dalbedo_cache);
            let mut out = LightTreeClosureSummary::new(sv);
            out.add_scaled(bg, 1.0 - m);
            out.add_scaled(fg, m);
            out
        }
        ClosureNode::Layer { top, base } => {
            if let ClosureNode::ThinFilm { .. } = compiled.closure(*top) {
                return light_tree_summary_idx(compiled, locals, sv, *base, wo, dalbedo_cache);
            }
            let r_top =
                directional_albedo_idx_scalar(compiled, locals, sv, *top, wo, dalbedo_cache)
                    .clamp(0.0, 1.0);
            let top = light_tree_summary_idx(compiled, locals, sv, *top, wo, dalbedo_cache);
            let base = light_tree_summary_idx(compiled, locals, sv, *base, wo, dalbedo_cache);
            let mut out = LightTreeClosureSummary::new(sv);
            out.add_scaled(top, r_top);
            out.add_scaled(base, 1.0 - r_top);
            out
        }
        ClosureNode::Add { a, b, kind } => {
            let a = light_tree_summary_idx(compiled, locals, sv, *a, wo, dalbedo_cache);
            let b = light_tree_summary_idx(compiled, locals, sv, *b, wo, dalbedo_cache);
            let mut out = LightTreeClosureSummary::new(sv);
            if matches!(kind, super::compiled::ClosureKind::Bsdf) {
                out.add_scaled(a, 0.5);
                out.add_scaled(b, 0.5);
            } else {
                out.add_scaled(a, 1.0);
                out.add_scaled(b, 1.0);
            }
            out
        }
        ClosureNode::Multiply { inner, scale, .. } => {
            let s = crate::math::sg::luminance(read_color3(scale, locals));
            light_tree_summary_idx(compiled, locals, sv, *inner, wo, dalbedo_cache).scale(s)
        }
        ClosureNode::IfGreater {
            value1,
            value2,
            then_branch,
            else_branch,
            ..
        } => {
            let pick = if read_float(value1, locals) > read_float(value2, locals) {
                *then_branch
            } else {
                *else_branch
            };
            light_tree_summary_idx(compiled, locals, sv, pick, wo, dalbedo_cache)
        }
        ClosureNode::IfGreaterEq {
            value1,
            value2,
            then_branch,
            else_branch,
            ..
        } => {
            let pick = if read_float(value1, locals) >= read_float(value2, locals) {
                *then_branch
            } else {
                *else_branch
            };
            light_tree_summary_idx(compiled, locals, sv, pick, wo, dalbedo_cache)
        }
        ClosureNode::IfEqual {
            value1,
            value2,
            then_branch,
            else_branch,
            ..
        } => {
            let pick = if read_float(value1, locals) == read_float(value2, locals) {
                *then_branch
            } else {
                *else_branch
            };
            light_tree_summary_idx(compiled, locals, sv, pick, wo, dalbedo_cache)
        }
        ClosureNode::Switch {
            which, branches, ..
        } => {
            let i = read_param(which, locals).as_integer().clamp(0, 9) as usize;
            light_tree_summary_idx(compiled, locals, sv, branches[i], wo, dalbedo_cache)
        }
        ClosureNode::Surface { bsdf, .. } => {
            light_tree_summary_idx(compiled, locals, sv, *bsdf, wo, dalbedo_cache)
        }
    }
}

fn directional_albedo_idx(
    compiled: &CompiledMaterial,
    locals: &[Value],
    sv: &ShadingVertex,
    idx: u32,
    wo: Vec3,
    dalbedo_cache: DalbedoCache<'_>,
) -> Vec3 {
    if let Some(cache) = dalbedo_cache
        && let Some(cell) = cache.get(idx as usize)
        && let Some(value) = cell.get()
    {
        return value;
    }
    let r = directional_albedo_idx_inner(compiled, locals, sv, idx, wo, dalbedo_cache);
    if let Some(cache) = dalbedo_cache
        && let Some(cell) = cache.get(idx as usize)
    {
        cell.set(Some(r));
    }
    r
}

fn directional_albedo_idx_inner(
    compiled: &CompiledMaterial,
    locals: &[Value],
    sv: &ShadingVertex,
    idx: u32,
    wo: Vec3,
    dalbedo_cache: DalbedoCache<'_>,
) -> Vec3 {
    let node = compiled.closure(idx);
    match node {
        ClosureNode::Zero => Vec3::ZERO,
        ClosureNode::OrenNayarDiffuse { .. }
        | ClosureNode::BurleyDiffuse { .. }
        | ClosureNode::Translucent { .. }
        | ClosureNode::Conductor { .. }
        | ClosureNode::ChiangHair { .. }
        | ClosureNode::GoochShade { .. } => Vec3::ONE,
        ClosureNode::Dielectric {
            weight,
            tint,
            ior,
            roughness,
            scatter_mode,
            thinfilm_thickness,
            thinfilm_ior,
            normal,
            tangent,
            ..
        } => {
            let (_, wo_use) = override_frame_for_wo(sv, locals, normal, tangent, wo);
            mtlx_dielectric_directional_albedo(
                compiled,
                wo_use,
                read_float(weight, locals),
                read_color3(tint, locals),
                read_float(ior, locals),
                read_vec2(roughness, locals),
                *scatter_mode,
                read_float(thinfilm_thickness, locals),
                read_float(thinfilm_ior, locals),
                compiled.thin_walled || sv.front_face,
            )
        }
        ClosureNode::GeneralizedSchlick {
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
            ..
        } => {
            let (_, wo_use) = override_frame_for_wo(sv, locals, normal, tangent, wo);
            mtlx_generalized_schlick_directional_albedo(
                compiled,
                wo_use,
                read_float(weight, locals),
                read_color3(color0, locals),
                read_color3(color82, locals),
                read_color3(color90, locals),
                read_float(exponent, locals),
                read_vec2(roughness, locals),
                *scatter_mode,
                read_float(thinfilm_thickness, locals),
                read_float(thinfilm_ior, locals),
                compiled.thin_walled || sv.front_face,
            )
        }
        ClosureNode::Sheen {
            weight,
            color,
            roughness,
            mode,
            normal,
        } => {
            let (_, wo_use) = override_frame_for_wo(sv, locals, normal, &None, wo);
            SheenBsdfMtlx::new(
                read_float(weight, locals),
                read_color3(color, locals),
                read_float(roughness, locals),
                *mode,
            )
            .directional_albedo_with_lut(wo_use, Some(sheen_lut(compiled)))
        }
        ClosureNode::ThinFilm { .. } => Vec3::ZERO,
        ClosureNode::Mix { bg, fg, mix, kind } => {
            let m = read_float(mix, locals);
            let m = if matches!(kind, super::compiled::ClosureKind::Surface) {
                m
            } else {
                m.clamp(0.0, 1.0)
            };
            directional_albedo_idx(compiled, locals, sv, *bg, wo, dalbedo_cache) * (1.0 - m)
                + directional_albedo_idx(compiled, locals, sv, *fg, wo, dalbedo_cache) * m
        }
        ClosureNode::Layer { top, base } => {
            let top_node = compiled.closure(*top);
            if let ClosureNode::ThinFilm { .. } = top_node {
                return directional_albedo_idx(compiled, locals, sv, *base, wo, dalbedo_cache);
            }
            let r = directional_albedo_idx(compiled, locals, sv, *top, wo, dalbedo_cache);
            r + directional_albedo_idx(compiled, locals, sv, *base, wo, dalbedo_cache)
                * (Vec3::ONE - r)
        }
        ClosureNode::Add {
            a,
            b,
            kind: super::compiled::ClosureKind::Bsdf,
        } => {
            (directional_albedo_idx(compiled, locals, sv, *a, wo, dalbedo_cache)
                + directional_albedo_idx(compiled, locals, sv, *b, wo, dalbedo_cache))
                * 0.5
        }
        ClosureNode::Add { a, b, .. } => {
            directional_albedo_idx(compiled, locals, sv, *a, wo, dalbedo_cache)
                + directional_albedo_idx(compiled, locals, sv, *b, wo, dalbedo_cache)
        }
        ClosureNode::Multiply { inner, scale, .. } => {
            directional_albedo_idx(compiled, locals, sv, *inner, wo, dalbedo_cache)
                * read_color3(scale, locals)
        }
        ClosureNode::IfGreater {
            value1,
            value2,
            then_branch,
            else_branch,
            ..
        } => {
            let v1 = read_float(value1, locals);
            let v2 = read_float(value2, locals);
            let pick = if v1 > v2 { *then_branch } else { *else_branch };
            directional_albedo_idx(compiled, locals, sv, pick, wo, dalbedo_cache)
        }
        ClosureNode::IfGreaterEq {
            value1,
            value2,
            then_branch,
            else_branch,
            ..
        } => {
            let v1 = read_float(value1, locals);
            let v2 = read_float(value2, locals);
            let pick = if v1 >= v2 { *then_branch } else { *else_branch };
            directional_albedo_idx(compiled, locals, sv, pick, wo, dalbedo_cache)
        }
        ClosureNode::IfEqual {
            value1,
            value2,
            then_branch,
            else_branch,
            ..
        } => {
            let v1 = read_float(value1, locals);
            let v2 = read_float(value2, locals);
            let pick = if v1 == v2 { *then_branch } else { *else_branch };
            directional_albedo_idx(compiled, locals, sv, pick, wo, dalbedo_cache)
        }
        ClosureNode::Switch {
            which, branches, ..
        } => {
            let i = read_param(which, locals).as_integer().clamp(0, 9) as usize;
            directional_albedo_idx(compiled, locals, sv, branches[i], wo, dalbedo_cache)
        }
        ClosureNode::Surface { bsdf, .. } => {
            directional_albedo_idx(compiled, locals, sv, *bsdf, wo, dalbedo_cache)
        }
        ClosureNode::UniformEdf { .. }
        | ClosureNode::ConicalEdf { .. }
        | ClosureNode::GeneralizedSchlickEdf { .. } => Vec3::ZERO,
    }
}

fn directional_albedo_idx_scalar(
    compiled: &CompiledMaterial,
    locals: &[Value],
    sv: &ShadingVertex,
    idx: u32,
    wo: Vec3,
    dalbedo_cache: DalbedoCache<'_>,
) -> f32 {
    let v = directional_albedo_idx(compiled, locals, sv, idx, wo, dalbedo_cache);
    ((v.x + v.y + v.z) / 3.0).clamp(0.0, 1.0)
}

#[derive(Clone, Copy)]
struct EdfTerms {
    shape: Vec3,
    intensity: Vec3,
}

impl EdfTerms {
    fn zero() -> Self {
        Self {
            shape: Vec3::ZERO,
            intensity: Vec3::ZERO,
        }
    }

    fn radiance(self) -> Vec3 {
        self.shape * self.intensity
    }
}

fn evaluate_edf(
    compiled: &CompiledMaterial,
    locals: &[Value],
    idx: u32,
    wo_world: Vec3,
    frame: &OrthonormalBasis,
) -> Vec3 {
    evaluate_edf_terms(compiled, locals, idx, wo_world, frame).radiance()
}

fn evaluate_edf_terms(
    compiled: &CompiledMaterial,
    locals: &[Value],
    idx: u32,
    wo_world: Vec3,
    frame: &OrthonormalBasis,
) -> EdfTerms {
    let inv_pi = 1.0 / std::f32::consts::PI;
    match compiled.closure(idx) {
        ClosureNode::Zero => EdfTerms::zero(),
        ClosureNode::UniformEdf { color } => EdfTerms {
            shape: Vec3::splat(inv_pi),
            intensity: read_color3(color, locals),
        },
        ClosureNode::ConicalEdf {
            color,
            inner_angle,
            outer_angle,
            normal,
        } => {
            let n = normal.as_ref().map_or(frame.local_to_world(Vec3::Z), |p| {
                read_param(p, locals).as_vector3().normalize_or_zero()
            });
            let cos_theta = wo_world.normalize_or_zero().dot(n).clamp(-1.0, 1.0);
            let inner = read_float(inner_angle, locals).max(0.0).to_radians();
            let outer = read_float(outer_angle, locals).max(0.0).to_radians();
            let spread = inner.max(outer);
            let cos_spread = spread.cos();
            let attenuation = if cos_theta < cos_spread {
                0.0
            } else if outer <= inner {
                1.0
            } else {
                let cos_inner = inner.cos();
                let cos_outer = outer.cos();
                ((cos_theta - cos_outer) / (cos_inner - cos_outer)).clamp(0.0, 1.0)
            };
            EdfTerms {
                shape: Vec3::splat(attenuation * inv_pi),
                intensity: read_color3(color, locals),
            }
        }
        ClosureNode::GeneralizedSchlickEdf {
            base,
            color0,
            color90,
            exponent,
        } => {
            let base_e = evaluate_edf_terms(compiled, locals, *base, wo_world, frame);
            let n = frame.local_to_world(Vec3::Z);
            let cos = wo_world.normalize_or_zero().dot(n).clamp(0.0, 1.0);
            let one_minus = 1.0 - cos;
            let exp = read_float(exponent, locals);
            let f = one_minus.powf(exp);
            let c0 = read_color3(color0, locals);
            let c90 = read_color3(color90, locals);
            EdfTerms {
                shape: base_e.shape * c0.lerp(c90, f),
                intensity: base_e.intensity,
            }
        }
        ClosureNode::Mix { bg, fg, mix, .. } => {
            let m = read_float(mix, locals).clamp(0.0, 1.0);
            let bg = evaluate_edf_terms(compiled, locals, *bg, wo_world, frame);
            let fg = evaluate_edf_terms(compiled, locals, *fg, wo_world, frame);
            EdfTerms {
                shape: bg.shape * (1.0 - m) + fg.shape * m,
                intensity: bg.intensity * (1.0 - m) + fg.intensity * m,
            }
        }
        ClosureNode::Add { a, b, .. } => {
            let a = evaluate_edf_terms(compiled, locals, *a, wo_world, frame);
            let b = evaluate_edf_terms(compiled, locals, *b, wo_world, frame);
            EdfTerms {
                shape: a.shape + b.shape,
                intensity: a.intensity + b.intensity,
            }
        }
        ClosureNode::Multiply { inner, scale, .. } => {
            let inner = evaluate_edf_terms(compiled, locals, *inner, wo_world, frame);
            EdfTerms {
                shape: inner.shape,
                intensity: inner.intensity * read_color3(scale, locals),
            }
        }
        ClosureNode::IfGreater {
            value1,
            value2,
            then_branch,
            else_branch,
            ..
        } => {
            let v1 = read_float(value1, locals);
            let v2 = read_float(value2, locals);
            let pick = if v1 > v2 { *then_branch } else { *else_branch };
            evaluate_edf_terms(compiled, locals, pick, wo_world, frame)
        }
        ClosureNode::IfGreaterEq {
            value1,
            value2,
            then_branch,
            else_branch,
            ..
        } => {
            let v1 = read_float(value1, locals);
            let v2 = read_float(value2, locals);
            let pick = if v1 >= v2 { *then_branch } else { *else_branch };
            evaluate_edf_terms(compiled, locals, pick, wo_world, frame)
        }
        ClosureNode::IfEqual {
            value1,
            value2,
            then_branch,
            else_branch,
            ..
        } => {
            let v1 = read_float(value1, locals);
            let v2 = read_float(value2, locals);
            let pick = if v1 == v2 { *then_branch } else { *else_branch };
            evaluate_edf_terms(compiled, locals, pick, wo_world, frame)
        }
        ClosureNode::Switch {
            which, branches, ..
        } => {
            let i = read_param(which, locals).as_integer().clamp(0, 9) as usize;
            evaluate_edf_terms(compiled, locals, branches[i], wo_world, frame)
        }
        ClosureNode::Surface { edf, .. } => {
            evaluate_edf_terms(compiled, locals, *edf, wo_world, frame)
        }
        ClosureNode::OrenNayarDiffuse { .. }
        | ClosureNode::BurleyDiffuse { .. }
        | ClosureNode::Translucent { .. }
        | ClosureNode::Dielectric { .. }
        | ClosureNode::Conductor { .. }
        | ClosureNode::GeneralizedSchlick { .. }
        | ClosureNode::Sheen { .. }
        | ClosureNode::ChiangHair { .. }
        | ClosureNode::ThinFilm { .. }
        | ClosureNode::Layer { .. }
        | ClosureNode::GoochShade { .. } => EdfTerms::zero(),
    }
}

fn surface_opacity_at(compiled: &CompiledMaterial, locals: &[Value], idx: u32) -> f32 {
    surface_opacity_at_nodes(&compiled.closure_nodes, locals, idx)
}

fn surface_opacity_at_nodes(nodes: &[ClosureNode], locals: &[Value], idx: u32) -> f32 {
    match &nodes[idx as usize] {
        ClosureNode::Surface { opacity, .. } => read_float(opacity, locals).clamp(0.0, 1.0),
        ClosureNode::Mix { bg, fg, mix, kind } => {
            let m = read_float(mix, locals).clamp(0.0, 1.0);
            if matches!(kind, super::compiled::ClosureKind::Surface) {
                let op_fg = surface_opacity_at_nodes(nodes, locals, *fg);
                let op_bg = surface_opacity_at_nodes(nodes, locals, *bg);
                op_bg * (1.0 - m) + op_fg * m
            } else {
                1.0
            }
        }
        ClosureNode::Layer { top, base } => {
            let a = surface_opacity_at_nodes(nodes, locals, *top);
            let b = surface_opacity_at_nodes(nodes, locals, *base);
            a.max(b)
        }
        ClosureNode::Zero => 0.0,
        ClosureNode::OrenNayarDiffuse { .. }
        | ClosureNode::BurleyDiffuse { .. }
        | ClosureNode::Translucent { .. }
        | ClosureNode::Dielectric { .. }
        | ClosureNode::Conductor { .. }
        | ClosureNode::GeneralizedSchlick { .. }
        | ClosureNode::Sheen { .. }
        | ClosureNode::ChiangHair { .. }
        | ClosureNode::ThinFilm { .. }
        | ClosureNode::Add { .. }
        | ClosureNode::Multiply { .. }
        | ClosureNode::IfGreater { .. }
        | ClosureNode::IfGreaterEq { .. }
        | ClosureNode::IfEqual { .. }
        | ClosureNode::Switch { .. }
        | ClosureNode::UniformEdf { .. }
        | ClosureNode::ConicalEdf { .. }
        | ClosureNode::GeneralizedSchlickEdf { .. }
        | ClosureNode::GoochShade { .. } => panic!(
            "surface_opacity_at: closure {} is not a Surface/Mix/Layer node",
            idx
        ),
    }
}

pub fn opacity(compiled: &CompiledMaterial, locals: &[Value]) -> f32 {
    surface_opacity_at(compiled, locals, compiled.root)
}

pub fn opacity_for_alpha_test(compiled: &CompiledMaterial, locals: &[Value]) -> f32 {
    surface_opacity_at_nodes(&compiled.opacity_closure_nodes, locals, compiled.root)
}

pub fn is_thin_walled(compiled: &CompiledMaterial) -> bool {
    compiled.thin_walled
}

pub fn thin_walled_transmittance(compiled: &CompiledMaterial, locals: &[Value]) -> Vec3 {
    fn walk(compiled: &CompiledMaterial, locals: &[Value], idx: u32) -> Vec3 {
        match compiled.closure(idx) {
            ClosureNode::Surface { bsdf, .. } => walk(compiled, locals, *bsdf),
            ClosureNode::Dielectric { weight, tint, .. } => {
                read_color3(tint, locals) * read_float(weight, locals)
            }
            ClosureNode::Translucent { weight, color, .. } => {
                read_color3(color, locals) * read_float(weight, locals)
            }
            ClosureNode::Layer { base, .. } => walk(compiled, locals, *base),
            ClosureNode::Mix { bg, fg, mix, .. } => {
                let m = read_float(mix, locals).clamp(0.0, 1.0);
                walk(compiled, locals, *bg) * (1.0 - m) + walk(compiled, locals, *fg) * m
            }
            ClosureNode::Multiply { inner, scale, .. } => {
                walk(compiled, locals, *inner) * read_color3(scale, locals)
            }
            ClosureNode::Zero => Vec3::ZERO,
            ClosureNode::OrenNayarDiffuse { weight, color, .. }
            | ClosureNode::BurleyDiffuse { weight, color, .. } => {
                read_color3(color, locals) * read_float(weight, locals)
            }
            ClosureNode::Conductor { weight, .. } => Vec3::splat(read_float(weight, locals)),
            ClosureNode::GeneralizedSchlick { weight, color0, .. } => {
                read_color3(color0, locals) * read_float(weight, locals)
            }
            ClosureNode::Sheen { weight, color, .. } => {
                read_color3(color, locals) * read_float(weight, locals)
            }
            ClosureNode::ChiangHair { .. } | ClosureNode::ThinFilm { .. } => Vec3::ONE,
            ClosureNode::Add {
                a,
                b,
                kind: super::compiled::ClosureKind::Bsdf,
            } => (walk(compiled, locals, *a) + walk(compiled, locals, *b)) * 0.5,
            ClosureNode::Add { a, b, .. } => {
                walk(compiled, locals, *a) + walk(compiled, locals, *b)
            }
            ClosureNode::IfGreater {
                value1,
                value2,
                then_branch,
                else_branch,
                ..
            } => {
                let v1 = read_float(value1, locals);
                let v2 = read_float(value2, locals);
                let pick = if v1 > v2 { *then_branch } else { *else_branch };
                walk(compiled, locals, pick)
            }
            ClosureNode::IfGreaterEq {
                value1,
                value2,
                then_branch,
                else_branch,
                ..
            } => {
                let v1 = read_float(value1, locals);
                let v2 = read_float(value2, locals);
                let pick = if v1 >= v2 { *then_branch } else { *else_branch };
                walk(compiled, locals, pick)
            }
            ClosureNode::IfEqual {
                value1,
                value2,
                then_branch,
                else_branch,
                ..
            } => {
                let v1 = read_float(value1, locals);
                let v2 = read_float(value2, locals);
                let pick = if v1 == v2 { *then_branch } else { *else_branch };
                walk(compiled, locals, pick)
            }
            ClosureNode::Switch {
                which, branches, ..
            } => {
                let i = read_param(which, locals).as_integer().clamp(0, 9) as usize;
                walk(compiled, locals, branches[i])
            }
            ClosureNode::UniformEdf { .. }
            | ClosureNode::ConicalEdf { .. }
            | ClosureNode::GeneralizedSchlickEdf { .. }
            | ClosureNode::GoochShade { .. } => Vec3::ZERO,
        }
    }
    walk(compiled, locals, compiled.root)
}
