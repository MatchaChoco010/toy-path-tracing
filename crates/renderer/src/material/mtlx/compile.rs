use std::collections::HashMap;
use std::sync::Arc;

use glam::{Mat3, Mat4, Vec2, Vec3, Vec4};

use crate::bsdf::mtlx::{ScatterMode, SheenMode};
use crate::color::{self, ColorSpaceRef, OcioColorPipeline};
use crate::material::pattern::noise::{hsv_to_rgb, rgb_to_hsv};
use crate::material::{ScalarTexture, Texture, TextureColorSpace};
use crate::scene::mtlx_loader::flatten::{
    FlatGraph, FlatInput, FlatNode, FlatNodeId, FlatNodeKind, GeometricKind as FgKind,
};
use crate::scene::mtlx_loader::{MtlxType, MtlxValue};

use super::compiled::{
    AddressMode, ArithOp, ArtisticIorOutput, BlendOp, ChiangHairRoughnessOutput, ClosureKind,
    ClosureNode, ColorXform, CombineKind, CompareOp, CompiledMaterial, FilterType, FlakeOutput,
    GeomSpace, GeometricKind, ImageKind, ImageTexture, Instruction, LogicalOp, MaskOp, MergeOp,
    NoiseKind, NoiseOutput, Operand, ParamRef, TriplanarFilter, UdimTiles, UnaryOp, Value,
    ValueType, WorleyStyle,
};

#[derive(Debug)]
pub enum CompileError {
    Missing(String),
    Unsupported(String),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing(s) => write!(f, "missing: {}", s),
            Self::Unsupported(s) => write!(f, "unsupported: {}", s),
        }
    }
}

impl std::error::Error for CompileError {}

pub fn compile(
    graph: &FlatGraph,
    color_textures: HashMap<Arc<str>, Arc<Texture>>,
    alpha_textures: HashMap<Arc<str>, Arc<ScalarTexture>>,
    udim_textures: HashMap<Arc<str>, Arc<UdimTiles>>,
    scalar_textures: HashMap<Arc<str>, Arc<ScalarTexture>>,
) -> Result<CompiledMaterial, CompileError> {
    let ocio = OcioColorPipeline::new(
        color::DEFAULT_OCIO_CONFIG,
        Some(color::DEFAULT_RENDERING_SPACE.to_string()),
        color::DEFAULT_TEXTURE_COLOR_SPACE,
    )
    .map_err(|error| CompileError::Unsupported(error.to_string()))?;
    compile_with_ocio(
        graph,
        &ocio,
        color_textures,
        alpha_textures,
        udim_textures,
        scalar_textures,
    )
}

pub fn compile_with_ocio(
    graph: &FlatGraph,
    ocio: &OcioColorPipeline,
    color_textures: HashMap<Arc<str>, Arc<Texture>>,
    alpha_textures: HashMap<Arc<str>, Arc<ScalarTexture>>,
    udim_textures: HashMap<Arc<str>, Arc<UdimTiles>>,
    scalar_textures: HashMap<Arc<str>, Arc<ScalarTexture>>,
) -> Result<CompiledMaterial, CompileError> {
    let mut builder = Builder {
        graph,
        color_textures: &color_textures,
        alpha_textures: &alpha_textures,
        udim_textures: &udim_textures,
        scalar_textures: &scalar_textures,
        ocio,
        color_processors: Vec::new(),
        instructions: Vec::new(),
        operand_pool: Vec::new(),
        value_pool: Vec::new(),
        closure_nodes: vec![ClosureNode::Zero],
        register_for: HashMap::new(),
        closure_for: HashMap::new(),
        next_vreg: 0,
    };

    let root_node = &graph.nodes[graph.root as usize];
    let root_idx = match &root_node.kind {
        FlatNodeKind::SurfaceMaterial => {
            let surface_input = root_node
                .inputs
                .iter()
                .find(|i| i.name == "surfaceshader")
                .ok_or_else(|| CompileError::Missing("surfaceshader".into()))?;
            builder.compile_closure_input(&surface_input.binding, ClosureKind::Surface)?
        }
        _ => {
            return Err(CompileError::Unsupported(format!(
                "non-material root: {:?}",
                root_node.kind
            )));
        }
    };

    fold_constant_instructions(
        &mut builder.instructions,
        &builder.operand_pool,
        &mut builder.value_pool,
        &builder.color_processors,
        builder.next_vreg,
    );

    let constants = local_constants(
        &builder.instructions,
        &builder.value_pool,
        builder.next_vreg,
    );
    inline_closure_constant_params(&mut builder.closure_nodes, &constants);
    simplify_closure_nodes(&mut builder.closure_nodes, root_idx);

    let passthrough = root_idx == 0 || is_volume_only(graph);
    let max_emission = closure_max_emission(&builder.closure_nodes, root_idx);
    let may_emit = max_emission > 0.0;
    let has_opacity_test = closure_has_opacity_test(
        &builder.closure_nodes,
        root_idx,
        &builder.instructions,
        &builder.value_pool,
        builder.next_vreg,
    );
    let thin_walled = closure_is_thin_walled(&builder.closure_nodes, root_idx);

    // Phase 5: SSA -> slot-allocated bytecode lowering (linear-scan).
    let raw_instructions = builder.instructions;
    let raw_operand_pool = builder.operand_pool;
    let raw_closure_nodes = builder.closure_nodes;
    let color_processors = builder.color_processors;
    let mut value_pool = builder.value_pool;
    let pre_alloc_vregs = builder.next_vreg;

    let (opacity_instructions, opacity_operand_pool, opacity_closure_nodes, opacity_num_registers) =
        if has_opacity_test {
            let mut instructions = raw_instructions.clone();
            let mut operand_pool = raw_operand_pool.clone();
            let mut closure_nodes = raw_closure_nodes.clone();
            let mut live_out = vec![false; pre_alloc_vregs as usize];
            closure_mark_opacity_locals(&closure_nodes, root_idx, &mut live_out);
            eliminate_dead_instructions_with_live(&mut instructions, &operand_pool, &mut live_out);
            let opacity_folded = fold_constant_instructions(
                &mut instructions,
                &operand_pool,
                &mut value_pool,
                &color_processors,
                pre_alloc_vregs,
            );
            if opacity_folded > 0 {
                let mut live_out = vec![false; pre_alloc_vregs as usize];
                closure_mark_opacity_locals(&closure_nodes, root_idx, &mut live_out);
                eliminate_dead_instructions_with_live(
                    &mut instructions,
                    &operand_pool,
                    &mut live_out,
                );
            }
            let num_registers = allocate_registers(
                &mut instructions,
                &mut operand_pool,
                &mut closure_nodes,
                root_idx,
                pre_alloc_vregs,
            );
            (instructions, operand_pool, closure_nodes, num_registers)
        } else {
            (Vec::new(), Vec::new(), Vec::new(), 0)
        };

    let mut instructions = raw_instructions;
    let mut operand_pool = raw_operand_pool;
    let mut closure_nodes = raw_closure_nodes;
    eliminate_dead_instructions(
        &mut instructions,
        &operand_pool,
        &closure_nodes,
        root_idx,
        pre_alloc_vregs,
    );
    let full_folded = fold_constant_instructions(
        &mut instructions,
        &operand_pool,
        &mut value_pool,
        &color_processors,
        pre_alloc_vregs,
    );
    if full_folded > 0 {
        eliminate_dead_instructions(
            &mut instructions,
            &operand_pool,
            &closure_nodes,
            root_idx,
            pre_alloc_vregs,
        );
    }
    let num_registers = allocate_registers(
        &mut instructions,
        &mut operand_pool,
        &mut closure_nodes,
        root_idx,
        pre_alloc_vregs,
    );
    Ok(CompiledMaterial {
        instructions,
        operand_pool,
        value_pool,
        color_processors,
        opacity_instructions,
        opacity_operand_pool,
        opacity_closure_nodes,
        opacity_num_registers,
        num_registers,
        closure_nodes,
        root: root_idx,
        passthrough,
        max_emission,
        may_emit,
        has_opacity_test,
        thin_walled,
        sheen_lut: None,
        mtlx_dielectric_lut: None,
        mtlx_generalized_schlick_lut: None,
    })
}

/// Bytecode 命令の dst register (生成 vreg) を返す。Passthrough 等の dst-less
/// 命令は `None`。
fn instr_dst(instr: &Instruction) -> Option<u16> {
    use Instruction::*;
    Some(match instr {
        Passthrough => return None,
        LoadConst { dst, .. }
        | LoadGeom { dst, .. }
        | LoadMat3Const { dst, .. }
        | LoadMat4Const { dst, .. }
        | Arith { dst, .. }
        | Unary { dst, .. }
        | Convert { dst, .. }
        | Logical { dst, .. }
        | CompareBool { dst, .. }
        | Compare { dst, .. }
        | IfElse { dst, .. }
        | MixValue { dst, .. }
        | Clamp { dst, .. }
        | Smoothstep { dst, .. }
        | Extract { dst, .. }
        | ExtractRowVector { dst, .. }
        | Reflect { dst, .. }
        | Refract { dst, .. }
        | Rotate2d { dst, .. }
        | Rotate3d { dst, .. }
        | DotProduct { dst, .. }
        | CrossProduct { dst, .. }
        | Distance { dst, .. }
        | FacingRatio { dst, .. }
        | LuminanceWithCoeffs { dst, .. }
        | Combine { dst, .. }
        | CreateMatrix3 { dst, .. }
        | CreateMatrix4 { dst, .. }
        | CreateMatrix4FromVec3 { dst, .. }
        | Switch { dst, .. }
        | Image { dst, .. }
        | HextiledImage { dst, .. }
        | HextiledNormalMap { dst, .. }
        | TransformPoint { dst, .. }
        | TransformVector { dst, .. }
        | TransformNormal { dst, .. }
        | TransformMatrix { dst, .. }
        | Transpose { dst, .. }
        | Determinant { dst, .. }
        | InvertMatrix { dst, .. }
        | Place2d { dst, .. }
        | LatlongUv { dst, .. }
        | Noise { dst, .. }
        | Worley { dst, .. }
        | Cellnoise { dst, .. }
        | Flake { dst, .. }
        | RandomFloat { dst, .. }
        | RandomColor { dst, .. }
        | Ramplr { dst, .. }
        | Ramptb { dst, .. }
        | Ramp4 { dst, .. }
        | Splitlr { dst, .. }
        | Splittb { dst, .. }
        | Blackbody { dst, .. }
        | ArtisticIor { dst, .. }
        | ChiangHairRoughness { dst, .. }
        | DeonHairAbsorptionFromMelanin { dst, .. }
        | ChiangHairAbsorptionFromColor { dst, .. }
        | RoughnessAnisotropy { dst, .. }
        | GlossinessAnisotropy { dst, .. }
        | RoughnessDual { dst, .. }
        | TransformColor { dst, .. }
        | TriplanarBlend { dst, .. }
        | CurveUniformLinear { dst, .. }
        | CurveUniformCubic { dst, .. }
        | CurveInverseCubic { dst, .. }
        | Normalmap { dst, .. }
        | NormalmapWithFrame { dst, .. }
        | Bump { dst, .. }
        | BumpWithFrame { dst, .. }
        | HeightToNormal { dst, .. }
        | Blend { dst, .. }
        | Merge { dst, .. }
        | Mask { dst, .. }
        | Premult { dst, .. }
        | Unpremult { dst, .. }
        | Contrast { dst, .. }
        | Range { dst, .. }
        | Remap { dst, .. }
        | HsvAdjust { dst, .. }
        | Saturate { dst, .. }
        | ColorCorrect { dst, .. }
        | Checkerboard { dst, .. } => *dst,
    })
}

/// 命令の operand_pool 参照範囲を返す。`(pool_start, count)`。
fn instr_pool_range(instr: &Instruction) -> Option<(u32, usize)> {
    use Instruction::*;
    match instr {
        Combine {
            operands_start,
            kind,
            ..
        } => {
            let n = match kind {
                CombineKind::Vector2FromFloats
                | CombineKind::Color4FromColor3Float
                | CombineKind::Vector4FromVector3Float
                | CombineKind::Vector4FromVector2Vector2 => 2,
                CombineKind::Color3FromFloats | CombineKind::Vector3FromFloats => 3,
                CombineKind::Color4FromFloats | CombineKind::Vector4FromFloats => 4,
            };
            Some((*operands_start, n))
        }
        CreateMatrix3 { rows_start, .. } => Some((*rows_start, 3)),
        CreateMatrix4 { rows_start, .. } | CreateMatrix4FromVec3 { rows_start, .. } => {
            Some((*rows_start, 4))
        }
        Switch { branches_start, .. } => Some((*branches_start, 10)),
        HextiledImage { operands_start, .. } => Some((*operands_start, 11)),
        HextiledNormalMap { operands_start, .. } => Some((*operands_start, 14)),
        Noise { operands_start, .. } => Some((*operands_start, 7)),
        Worley { operands_start, .. } => Some((*operands_start, 2)),
        Flake { operands_start, .. } => Some((*operands_start, 7)),
        RandomFloat { operands_start, .. } => Some((*operands_start, 4)),
        RandomColor { operands_start, .. } => Some((*operands_start, 8)),
        DeonHairAbsorptionFromMelanin { operands_start, .. } => Some((*operands_start, 4)),
        TriplanarBlend { operands_start, .. } => Some((*operands_start, 5)),
        NormalmapWithFrame { operands_start, .. } => Some((*operands_start, 5)),
        BumpWithFrame { operands_start, .. } => Some((*operands_start, 4)),
        Range { operands_start, .. } => Some((*operands_start, 6)),
        Remap { operands_start, .. } => Some((*operands_start, 5)),
        ColorCorrect { operands_start, .. } => Some((*operands_start, 9)),
        _ => None,
    }
}

/// inline operand fields (`Operand::Reg(_)`) を visit。
fn instr_visit_inline_operands(instr: &Instruction, mut f: impl FnMut(u16)) {
    use Instruction::*;
    let mut visit = |op: &Operand| {
        if let Operand::Reg(r) = op {
            f(*r);
        }
    };
    match instr {
        Passthrough
        | LoadConst { .. }
        | LoadGeom { .. }
        | LoadMat3Const { .. }
        | LoadMat4Const { .. }
        | Flake { .. } => {}
        Arith { a, b, .. }
        | Logical { a, b, .. }
        | DotProduct { a, b, .. }
        | CrossProduct { a, b, .. }
        | Distance { a, b, .. } => {
            visit(a);
            visit(b);
        }
        Unary { src, .. }
        | Convert { src, .. }
        | RoughnessDual { src, .. }
        | TransformColor { src, .. }
        | Transpose { src, .. }
        | Determinant { src, .. }
        | InvertMatrix { src, .. }
        | Premult { src, .. }
        | Unpremult { src, .. } => visit(src),
        CompareBool { v1, v2, .. } => {
            visit(v1);
            visit(v2);
        }
        Compare {
            v1,
            v2,
            in_true,
            in_false,
            ..
        } => {
            visit(v1);
            visit(v2);
            visit(in_true);
            visit(in_false);
        }
        IfElse {
            cond,
            in_true,
            in_false,
            ..
        } => {
            visit(cond);
            visit(in_true);
            visit(in_false);
        }
        MixValue { bg, fg, mix, .. } | Merge { bg, fg, mix, .. } | Blend { bg, fg, mix, .. } => {
            visit(bg);
            visit(fg);
            visit(mix);
        }
        Clamp { v, lo, hi, .. } | Smoothstep { v, lo, hi, .. } => {
            visit(v);
            visit(lo);
            visit(hi);
        }
        Contrast {
            v, amount, pivot, ..
        } => {
            visit(v);
            visit(amount);
            visit(pivot);
        }
        Extract { src, idx, .. } => {
            visit(src);
            visit(idx);
        }
        ExtractRowVector { src, .. } => visit(src),
        Reflect { i, n, .. } => {
            visit(i);
            visit(n);
        }
        Refract { i, n, eta, .. } => {
            visit(i);
            visit(n);
            visit(eta);
        }
        Rotate2d { v, amount, .. } => {
            visit(v);
            visit(amount);
        }
        Rotate3d {
            v, axis, amount, ..
        } => {
            visit(v);
            visit(axis);
            visit(amount);
        }
        FacingRatio { view, normal, .. } => {
            visit(view);
            visit(normal);
        }
        LuminanceWithCoeffs { c, lumacoeffs, .. } => {
            visit(c);
            visit(lumacoeffs);
        }
        Switch { which, .. } => visit(which),
        Image {
            texcoord,
            tiling,
            offset,
            default,
            ..
        } => {
            visit(texcoord);
            visit(tiling);
            visit(offset);
            visit(default);
        }
        TransformPoint { v, .. } | TransformVector { v, .. } | TransformNormal { v, .. } => {
            visit(v);
        }
        TransformMatrix { mat, v, .. } => {
            visit(mat);
            visit(v);
        }
        Place2d {
            texcoord,
            pivot,
            scale,
            rotate,
            offset,
            ..
        } => {
            visit(texcoord);
            visit(pivot);
            visit(scale);
            visit(rotate);
            visit(offset);
        }
        LatlongUv {
            viewdir, rotation, ..
        } => {
            visit(viewdir);
            visit(rotation);
        }
        Cellnoise { coord, .. } => visit(coord),
        Ramplr { texcoord, l, r, .. } => {
            visit(texcoord);
            visit(l);
            visit(r);
        }
        Ramptb { texcoord, t, b, .. } => {
            visit(texcoord);
            visit(t);
            visit(b);
        }
        Ramp4 {
            texcoord,
            tl,
            tr,
            bl,
            br,
            ..
        } => {
            visit(texcoord);
            visit(tl);
            visit(tr);
            visit(bl);
            visit(br);
        }
        Splitlr {
            texcoord,
            center,
            l,
            r,
            ..
        } => {
            visit(texcoord);
            visit(center);
            visit(l);
            visit(r);
        }
        Splittb {
            texcoord,
            center,
            t,
            b,
            ..
        } => {
            visit(texcoord);
            visit(center);
            visit(t);
            visit(b);
        }
        Blackbody { temp, .. } => visit(temp),
        ArtisticIor { refl, edge, .. } => {
            visit(refl);
            visit(edge);
        }
        ChiangHairRoughness {
            longitudinal,
            azimuthal,
            scale_tt,
            scale_trt,
            ..
        } => {
            visit(longitudinal);
            visit(azimuthal);
            visit(scale_tt);
            visit(scale_trt);
        }
        ChiangHairAbsorptionFromColor { color, beta, .. } => {
            visit(color);
            visit(beta);
        }
        RoughnessAnisotropy { r, a, .. } => {
            visit(r);
            visit(a);
        }
        GlossinessAnisotropy { g, a, .. } => {
            visit(g);
            visit(a);
        }
        CurveUniformLinear { t, .. } | CurveUniformCubic { t, .. } => visit(t),
        CurveInverseCubic { x, .. } => visit(x),
        Normalmap { raw, scale, .. } => {
            visit(raw);
            visit(scale);
        }
        Bump { height, scale, .. } | HeightToNormal { height, scale, .. } => {
            visit(height);
            visit(scale);
        }
        Mask { v, mask, .. } => {
            visit(v);
            visit(mask);
        }
        HsvAdjust { c, amount, .. } => {
            visit(c);
            visit(amount);
        }
        Saturate {
            c,
            amount,
            lumacoeffs,
            ..
        } => {
            visit(c);
            visit(amount);
            visit(lumacoeffs);
        }
        Checkerboard {
            color1,
            color2,
            uvtiling,
            uvoffset,
            texcoord,
            ..
        } => {
            visit(color1);
            visit(color2);
            visit(uvtiling);
            visit(uvoffset);
            visit(texcoord);
        }
        Combine { .. }
        | CreateMatrix3 { .. }
        | CreateMatrix4 { .. }
        | CreateMatrix4FromVec3 { .. }
        | HextiledImage { .. }
        | HextiledNormalMap { .. }
        | Noise { .. }
        | Worley { .. }
        | RandomFloat { .. }
        | RandomColor { .. }
        | DeonHairAbsorptionFromMelanin { .. }
        | TriplanarBlend { .. }
        | NormalmapWithFrame { .. }
        | BumpWithFrame { .. }
        | Range { .. }
        | Remap { .. }
        | ColorCorrect { .. } => {}
    }
}

fn rewrite_op(op: &mut Operand, mapping: &[u16]) {
    if let Operand::Reg(r) = op {
        *r = mapping[*r as usize];
    }
}

fn rewrite_instruction_inline(instr: &mut Instruction, mapping: &[u16]) {
    use Instruction::*;
    // dst
    match instr {
        Passthrough => {}
        LoadConst { dst, .. }
        | LoadGeom { dst, .. }
        | LoadMat3Const { dst, .. }
        | LoadMat4Const { dst, .. }
        | Arith { dst, .. }
        | Unary { dst, .. }
        | Convert { dst, .. }
        | Logical { dst, .. }
        | CompareBool { dst, .. }
        | Compare { dst, .. }
        | IfElse { dst, .. }
        | MixValue { dst, .. }
        | Clamp { dst, .. }
        | Smoothstep { dst, .. }
        | Extract { dst, .. }
        | ExtractRowVector { dst, .. }
        | Reflect { dst, .. }
        | Refract { dst, .. }
        | Rotate2d { dst, .. }
        | Rotate3d { dst, .. }
        | DotProduct { dst, .. }
        | CrossProduct { dst, .. }
        | Distance { dst, .. }
        | FacingRatio { dst, .. }
        | LuminanceWithCoeffs { dst, .. }
        | Combine { dst, .. }
        | CreateMatrix3 { dst, .. }
        | CreateMatrix4 { dst, .. }
        | CreateMatrix4FromVec3 { dst, .. }
        | Switch { dst, .. }
        | Image { dst, .. }
        | HextiledImage { dst, .. }
        | HextiledNormalMap { dst, .. }
        | TransformPoint { dst, .. }
        | TransformVector { dst, .. }
        | TransformNormal { dst, .. }
        | TransformMatrix { dst, .. }
        | Transpose { dst, .. }
        | Determinant { dst, .. }
        | InvertMatrix { dst, .. }
        | Place2d { dst, .. }
        | LatlongUv { dst, .. }
        | Noise { dst, .. }
        | Worley { dst, .. }
        | Cellnoise { dst, .. }
        | Flake { dst, .. }
        | RandomFloat { dst, .. }
        | RandomColor { dst, .. }
        | Ramplr { dst, .. }
        | Ramptb { dst, .. }
        | Ramp4 { dst, .. }
        | Splitlr { dst, .. }
        | Splittb { dst, .. }
        | Blackbody { dst, .. }
        | ArtisticIor { dst, .. }
        | ChiangHairRoughness { dst, .. }
        | DeonHairAbsorptionFromMelanin { dst, .. }
        | ChiangHairAbsorptionFromColor { dst, .. }
        | RoughnessAnisotropy { dst, .. }
        | GlossinessAnisotropy { dst, .. }
        | RoughnessDual { dst, .. }
        | TransformColor { dst, .. }
        | TriplanarBlend { dst, .. }
        | CurveUniformLinear { dst, .. }
        | CurveUniformCubic { dst, .. }
        | CurveInverseCubic { dst, .. }
        | Normalmap { dst, .. }
        | NormalmapWithFrame { dst, .. }
        | Bump { dst, .. }
        | BumpWithFrame { dst, .. }
        | HeightToNormal { dst, .. }
        | Blend { dst, .. }
        | Merge { dst, .. }
        | Mask { dst, .. }
        | Premult { dst, .. }
        | Unpremult { dst, .. }
        | Contrast { dst, .. }
        | Range { dst, .. }
        | Remap { dst, .. }
        | HsvAdjust { dst, .. }
        | Saturate { dst, .. }
        | ColorCorrect { dst, .. }
        | Checkerboard { dst, .. } => {
            *dst = mapping[*dst as usize];
        }
    }
    // inline operands
    match instr {
        Passthrough
        | LoadConst { .. }
        | LoadGeom { .. }
        | LoadMat3Const { .. }
        | LoadMat4Const { .. }
        | Flake { .. } => {}
        Arith { a, b, .. }
        | Logical { a, b, .. }
        | DotProduct { a, b, .. }
        | CrossProduct { a, b, .. }
        | Distance { a, b, .. } => {
            rewrite_op(a, mapping);
            rewrite_op(b, mapping);
        }
        Unary { src, .. }
        | Convert { src, .. }
        | RoughnessDual { src, .. }
        | TransformColor { src, .. }
        | Transpose { src, .. }
        | Determinant { src, .. }
        | InvertMatrix { src, .. }
        | Premult { src, .. }
        | Unpremult { src, .. } => {
            rewrite_op(src, mapping);
        }
        CompareBool { v1, v2, .. } => {
            rewrite_op(v1, mapping);
            rewrite_op(v2, mapping);
        }
        Compare {
            v1,
            v2,
            in_true,
            in_false,
            ..
        } => {
            rewrite_op(v1, mapping);
            rewrite_op(v2, mapping);
            rewrite_op(in_true, mapping);
            rewrite_op(in_false, mapping);
        }
        IfElse {
            cond,
            in_true,
            in_false,
            ..
        } => {
            rewrite_op(cond, mapping);
            rewrite_op(in_true, mapping);
            rewrite_op(in_false, mapping);
        }
        MixValue { bg, fg, mix, .. } | Merge { bg, fg, mix, .. } | Blend { bg, fg, mix, .. } => {
            rewrite_op(bg, mapping);
            rewrite_op(fg, mapping);
            rewrite_op(mix, mapping);
        }
        Clamp { v, lo, hi, .. } | Smoothstep { v, lo, hi, .. } => {
            rewrite_op(v, mapping);
            rewrite_op(lo, mapping);
            rewrite_op(hi, mapping);
        }
        Contrast {
            v, amount, pivot, ..
        } => {
            rewrite_op(v, mapping);
            rewrite_op(amount, mapping);
            rewrite_op(pivot, mapping);
        }
        Extract { src, idx, .. } => {
            rewrite_op(src, mapping);
            rewrite_op(idx, mapping);
        }
        ExtractRowVector { src, .. } => rewrite_op(src, mapping),
        Reflect { i, n, .. } => {
            rewrite_op(i, mapping);
            rewrite_op(n, mapping);
        }
        Refract { i, n, eta, .. } => {
            rewrite_op(i, mapping);
            rewrite_op(n, mapping);
            rewrite_op(eta, mapping);
        }
        Rotate2d { v, amount, .. } => {
            rewrite_op(v, mapping);
            rewrite_op(amount, mapping);
        }
        Rotate3d {
            v, axis, amount, ..
        } => {
            rewrite_op(v, mapping);
            rewrite_op(axis, mapping);
            rewrite_op(amount, mapping);
        }
        FacingRatio { view, normal, .. } => {
            rewrite_op(view, mapping);
            rewrite_op(normal, mapping);
        }
        LuminanceWithCoeffs { c, lumacoeffs, .. } => {
            rewrite_op(c, mapping);
            rewrite_op(lumacoeffs, mapping);
        }
        Switch { which, .. } => rewrite_op(which, mapping),
        Image {
            texcoord,
            tiling,
            offset,
            default,
            ..
        } => {
            rewrite_op(texcoord, mapping);
            rewrite_op(tiling, mapping);
            rewrite_op(offset, mapping);
            rewrite_op(default, mapping);
        }
        TransformPoint { v, .. } | TransformVector { v, .. } | TransformNormal { v, .. } => {
            rewrite_op(v, mapping);
        }
        TransformMatrix { mat, v, .. } => {
            rewrite_op(mat, mapping);
            rewrite_op(v, mapping);
        }
        Place2d {
            texcoord,
            pivot,
            scale,
            rotate,
            offset,
            ..
        } => {
            rewrite_op(texcoord, mapping);
            rewrite_op(pivot, mapping);
            rewrite_op(scale, mapping);
            rewrite_op(rotate, mapping);
            rewrite_op(offset, mapping);
        }
        LatlongUv {
            viewdir, rotation, ..
        } => {
            rewrite_op(viewdir, mapping);
            rewrite_op(rotation, mapping);
        }
        Cellnoise { coord, .. } => rewrite_op(coord, mapping),
        Ramplr { texcoord, l, r, .. } => {
            rewrite_op(texcoord, mapping);
            rewrite_op(l, mapping);
            rewrite_op(r, mapping);
        }
        Ramptb { texcoord, t, b, .. } => {
            rewrite_op(texcoord, mapping);
            rewrite_op(t, mapping);
            rewrite_op(b, mapping);
        }
        Ramp4 {
            texcoord,
            tl,
            tr,
            bl,
            br,
            ..
        } => {
            rewrite_op(texcoord, mapping);
            rewrite_op(tl, mapping);
            rewrite_op(tr, mapping);
            rewrite_op(bl, mapping);
            rewrite_op(br, mapping);
        }
        Splitlr {
            texcoord,
            center,
            l,
            r,
            ..
        } => {
            rewrite_op(texcoord, mapping);
            rewrite_op(center, mapping);
            rewrite_op(l, mapping);
            rewrite_op(r, mapping);
        }
        Splittb {
            texcoord,
            center,
            t,
            b,
            ..
        } => {
            rewrite_op(texcoord, mapping);
            rewrite_op(center, mapping);
            rewrite_op(t, mapping);
            rewrite_op(b, mapping);
        }
        Blackbody { temp, .. } => rewrite_op(temp, mapping),
        ArtisticIor { refl, edge, .. } => {
            rewrite_op(refl, mapping);
            rewrite_op(edge, mapping);
        }
        ChiangHairRoughness {
            longitudinal,
            azimuthal,
            scale_tt,
            scale_trt,
            ..
        } => {
            rewrite_op(longitudinal, mapping);
            rewrite_op(azimuthal, mapping);
            rewrite_op(scale_tt, mapping);
            rewrite_op(scale_trt, mapping);
        }
        ChiangHairAbsorptionFromColor { color, beta, .. } => {
            rewrite_op(color, mapping);
            rewrite_op(beta, mapping);
        }
        RoughnessAnisotropy { r, a, .. } => {
            rewrite_op(r, mapping);
            rewrite_op(a, mapping);
        }
        GlossinessAnisotropy { g, a, .. } => {
            rewrite_op(g, mapping);
            rewrite_op(a, mapping);
        }
        CurveUniformLinear { t, .. } | CurveUniformCubic { t, .. } => rewrite_op(t, mapping),
        CurveInverseCubic { x, .. } => rewrite_op(x, mapping),
        Normalmap { raw, scale, .. } => {
            rewrite_op(raw, mapping);
            rewrite_op(scale, mapping);
        }
        Bump { height, scale, .. } | HeightToNormal { height, scale, .. } => {
            rewrite_op(height, mapping);
            rewrite_op(scale, mapping);
        }
        Mask { v, mask, .. } => {
            rewrite_op(v, mapping);
            rewrite_op(mask, mapping);
        }
        HsvAdjust { c, amount, .. } => {
            rewrite_op(c, mapping);
            rewrite_op(amount, mapping);
        }
        Saturate {
            c,
            amount,
            lumacoeffs,
            ..
        } => {
            rewrite_op(c, mapping);
            rewrite_op(amount, mapping);
            rewrite_op(lumacoeffs, mapping);
        }
        Checkerboard {
            color1,
            color2,
            uvtiling,
            uvoffset,
            texcoord,
            ..
        } => {
            rewrite_op(color1, mapping);
            rewrite_op(color2, mapping);
            rewrite_op(uvtiling, mapping);
            rewrite_op(uvoffset, mapping);
            rewrite_op(texcoord, mapping);
        }
        Combine { .. }
        | CreateMatrix3 { .. }
        | CreateMatrix4 { .. }
        | CreateMatrix4FromVec3 { .. }
        | HextiledImage { .. }
        | HextiledNormalMap { .. }
        | Noise { .. }
        | Worley { .. }
        | RandomFloat { .. }
        | RandomColor { .. }
        | DeonHairAbsorptionFromMelanin { .. }
        | TriplanarBlend { .. }
        | NormalmapWithFrame { .. }
        | BumpWithFrame { .. }
        | Range { .. }
        | Remap { .. }
        | ColorCorrect { .. } => {}
    }
}

fn closure_for_each_local(node: &ClosureNode, mut f: impl FnMut(u32)) {
    fn visit_p(p: &ParamRef, f: &mut dyn FnMut(u32)) {
        if let ParamRef::Local(idx) = p {
            f(*idx);
        }
    }
    fn visit_op(p: &Option<ParamRef>, f: &mut dyn FnMut(u32)) {
        if let Some(p) = p {
            visit_p(p, f);
        }
    }
    let f: &mut dyn FnMut(u32) = &mut f;
    match node {
        ClosureNode::Zero => {}
        ClosureNode::OrenNayarDiffuse {
            weight,
            color,
            roughness,
            normal,
            ..
        } => {
            visit_p(weight, f);
            visit_p(color, f);
            visit_p(roughness, f);
            visit_op(normal, f);
        }
        ClosureNode::BurleyDiffuse {
            weight,
            color,
            roughness,
            normal,
        } => {
            visit_p(weight, f);
            visit_p(color, f);
            visit_p(roughness, f);
            visit_op(normal, f);
        }
        ClosureNode::Translucent {
            weight,
            color,
            normal,
        } => {
            visit_p(weight, f);
            visit_p(color, f);
            visit_op(normal, f);
        }
        ClosureNode::Dielectric {
            weight,
            tint,
            ior,
            roughness,
            thinfilm_thickness,
            thinfilm_ior,
            normal,
            tangent,
            ..
        } => {
            visit_p(weight, f);
            visit_p(tint, f);
            visit_p(ior, f);
            visit_p(roughness, f);
            visit_p(thinfilm_thickness, f);
            visit_p(thinfilm_ior, f);
            visit_op(normal, f);
            visit_op(tangent, f);
        }
        ClosureNode::Conductor {
            weight,
            ior,
            extinction,
            roughness,
            retroreflective: _,
            thinfilm_thickness,
            thinfilm_ior,
            normal,
            tangent,
        } => {
            visit_p(weight, f);
            visit_p(ior, f);
            visit_p(extinction, f);
            visit_p(roughness, f);
            visit_p(thinfilm_thickness, f);
            visit_p(thinfilm_ior, f);
            visit_op(normal, f);
            visit_op(tangent, f);
        }
        ClosureNode::GeneralizedSchlick {
            weight,
            color0,
            color82,
            color90,
            exponent,
            roughness,
            thinfilm_thickness,
            thinfilm_ior,
            normal,
            tangent,
            ..
        } => {
            visit_p(weight, f);
            visit_p(color0, f);
            visit_p(color82, f);
            visit_p(color90, f);
            visit_p(exponent, f);
            visit_p(roughness, f);
            visit_p(thinfilm_thickness, f);
            visit_p(thinfilm_ior, f);
            visit_op(normal, f);
            visit_op(tangent, f);
        }
        ClosureNode::Sheen {
            weight,
            color,
            roughness,
            normal,
            ..
        } => {
            visit_p(weight, f);
            visit_p(color, f);
            visit_p(roughness, f);
            visit_op(normal, f);
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
            visit_p(tint_r, f);
            visit_p(tint_tt, f);
            visit_p(tint_trt, f);
            visit_p(absorption, f);
            visit_p(ior, f);
            visit_p(roughness_r, f);
            visit_p(roughness_tt, f);
            visit_p(roughness_trt, f);
            visit_p(cuticle_angle, f);
            visit_op(normal, f);
            visit_p(curve_direction, f);
        }
        ClosureNode::ThinFilm { thickness, ior } => {
            visit_p(thickness, f);
            visit_p(ior, f);
        }
        ClosureNode::UniformEdf { color } => visit_p(color, f),
        ClosureNode::ConicalEdf {
            color,
            inner_angle,
            outer_angle,
            normal,
        } => {
            visit_p(color, f);
            visit_p(inner_angle, f);
            visit_p(outer_angle, f);
            visit_op(normal, f);
        }
        ClosureNode::GeneralizedSchlickEdf {
            color0,
            color90,
            exponent,
            ..
        } => {
            visit_p(color0, f);
            visit_p(color90, f);
            visit_p(exponent, f);
        }
        ClosureNode::Mix { mix, .. } => visit_p(mix, f),
        ClosureNode::Layer { .. } | ClosureNode::Add { .. } => {}
        ClosureNode::Multiply { scale, .. } => visit_p(scale, f),
        ClosureNode::IfGreater { value1, value2, .. }
        | ClosureNode::IfGreaterEq { value1, value2, .. }
        | ClosureNode::IfEqual { value1, value2, .. } => {
            visit_p(value1, f);
            visit_p(value2, f);
        }
        ClosureNode::Switch { which, .. } => visit_p(which, f),
        ClosureNode::Surface { opacity, .. } => visit_p(opacity, f),
        ClosureNode::GoochShade {
            warm,
            cool,
            specular_intensity,
            shininess,
            light_direction,
        } => {
            visit_p(warm, f);
            visit_p(cool, f);
            visit_p(specular_intensity, f);
            visit_p(shininess, f);
            visit_p(light_direction, f);
        }
    }
}

fn closure_for_each_child(node: &ClosureNode, mut f: impl FnMut(u32)) {
    match node {
        ClosureNode::Mix { bg, fg, .. } => {
            f(*bg);
            f(*fg);
        }
        ClosureNode::Layer { top, base } => {
            f(*top);
            f(*base);
        }
        ClosureNode::Add { a, b, .. } => {
            f(*a);
            f(*b);
        }
        ClosureNode::Multiply { inner, .. } => f(*inner),
        ClosureNode::IfGreater {
            then_branch,
            else_branch,
            ..
        }
        | ClosureNode::IfGreaterEq {
            then_branch,
            else_branch,
            ..
        }
        | ClosureNode::IfEqual {
            then_branch,
            else_branch,
            ..
        } => {
            f(*then_branch);
            f(*else_branch);
        }
        ClosureNode::Switch { branches, .. } => {
            for branch in branches {
                f(*branch);
            }
        }
        ClosureNode::Surface { bsdf, edf, .. } => {
            f(*bsdf);
            f(*edf);
        }
        ClosureNode::GeneralizedSchlickEdf { base, .. } => f(*base),
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
        | ClosureNode::GoochShade { .. } => {}
    }
}

fn closure_mark_reachable_locals(closure_nodes: &[ClosureNode], root: u32, live_out: &mut [bool]) {
    fn mark(closure_nodes: &[ClosureNode], idx: u32, seen: &mut [bool], live_out: &mut [bool]) {
        let idx = idx as usize;
        if idx >= closure_nodes.len() || seen[idx] {
            return;
        }
        seen[idx] = true;
        let node = &closure_nodes[idx];
        closure_for_each_local(node, |local| {
            let local = local as usize;
            if local < live_out.len() {
                live_out[local] = true;
            }
        });
        closure_for_each_child(node, |child| mark(closure_nodes, child, seen, live_out));
    }

    let mut seen = vec![false; closure_nodes.len()];
    mark(closure_nodes, root, &mut seen, live_out);
}

fn mark_param_local(param: &ParamRef, live_out: &mut [bool]) {
    if let ParamRef::Local(local) = param {
        let local = *local as usize;
        if local < live_out.len() {
            live_out[local] = true;
        }
    }
}

fn closure_mark_opacity_locals(closure_nodes: &[ClosureNode], root: u32, live_out: &mut [bool]) {
    fn mark(closure_nodes: &[ClosureNode], idx: u32, seen: &mut [bool], live_out: &mut [bool]) {
        let idx = idx as usize;
        if idx >= closure_nodes.len() || seen[idx] {
            return;
        }
        seen[idx] = true;
        match &closure_nodes[idx] {
            ClosureNode::Surface { opacity, .. } => mark_param_local(opacity, live_out),
            ClosureNode::Mix { bg, fg, mix, kind } => {
                mark_param_local(mix, live_out);
                if matches!(kind, ClosureKind::Surface) {
                    mark(closure_nodes, *bg, seen, live_out);
                    mark(closure_nodes, *fg, seen, live_out);
                }
            }
            ClosureNode::Layer { top, base } => {
                mark(closure_nodes, *top, seen, live_out);
                mark(closure_nodes, *base, seen, live_out);
            }
            ClosureNode::Zero => {}
            _ => {}
        }
    }

    let mut seen = vec![false; closure_nodes.len()];
    mark(closure_nodes, root, &mut seen, live_out);
}

fn closure_rewrite_locals(node: &mut ClosureNode, mapping: &[u16]) {
    fn rew(p: &mut ParamRef, mapping: &[u16]) {
        if let ParamRef::Local(idx) = p {
            *idx = mapping[*idx as usize] as u32;
        }
    }
    fn orew(p: &mut Option<ParamRef>, mapping: &[u16]) {
        if let Some(p) = p {
            rew(p, mapping);
        }
    }
    let rew = |p: &mut ParamRef| rew(p, mapping);
    let orew = |p: &mut Option<ParamRef>| orew(p, mapping);
    match node {
        ClosureNode::Zero => {}
        ClosureNode::OrenNayarDiffuse {
            weight,
            color,
            roughness,
            normal,
            ..
        } => {
            rew(weight);
            rew(color);
            rew(roughness);
            orew(normal);
        }
        ClosureNode::BurleyDiffuse {
            weight,
            color,
            roughness,
            normal,
        } => {
            rew(weight);
            rew(color);
            rew(roughness);
            orew(normal);
        }
        ClosureNode::Translucent {
            weight,
            color,
            normal,
        } => {
            rew(weight);
            rew(color);
            orew(normal);
        }
        ClosureNode::Dielectric {
            weight,
            tint,
            ior,
            roughness,
            thinfilm_thickness,
            thinfilm_ior,
            normal,
            tangent,
            ..
        } => {
            rew(weight);
            rew(tint);
            rew(ior);
            rew(roughness);
            rew(thinfilm_thickness);
            rew(thinfilm_ior);
            orew(normal);
            orew(tangent);
        }
        ClosureNode::Conductor {
            weight,
            ior,
            extinction,
            roughness,
            retroreflective: _,
            thinfilm_thickness,
            thinfilm_ior,
            normal,
            tangent,
        } => {
            rew(weight);
            rew(ior);
            rew(extinction);
            rew(roughness);
            rew(thinfilm_thickness);
            rew(thinfilm_ior);
            orew(normal);
            orew(tangent);
        }
        ClosureNode::GeneralizedSchlick {
            weight,
            color0,
            color82,
            color90,
            exponent,
            roughness,
            thinfilm_thickness,
            thinfilm_ior,
            normal,
            tangent,
            ..
        } => {
            rew(weight);
            rew(color0);
            rew(color82);
            rew(color90);
            rew(exponent);
            rew(roughness);
            rew(thinfilm_thickness);
            rew(thinfilm_ior);
            orew(normal);
            orew(tangent);
        }
        ClosureNode::Sheen {
            weight,
            color,
            roughness,
            normal,
            ..
        } => {
            rew(weight);
            rew(color);
            rew(roughness);
            orew(normal);
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
            rew(tint_r);
            rew(tint_tt);
            rew(tint_trt);
            rew(absorption);
            rew(ior);
            rew(roughness_r);
            rew(roughness_tt);
            rew(roughness_trt);
            rew(cuticle_angle);
            orew(normal);
            rew(curve_direction);
        }
        ClosureNode::ThinFilm { thickness, ior } => {
            rew(thickness);
            rew(ior);
        }
        ClosureNode::UniformEdf { color } => rew(color),
        ClosureNode::ConicalEdf {
            color,
            inner_angle,
            outer_angle,
            normal,
        } => {
            rew(color);
            rew(inner_angle);
            rew(outer_angle);
            orew(normal);
        }
        ClosureNode::GeneralizedSchlickEdf {
            color0,
            color90,
            exponent,
            ..
        } => {
            rew(color0);
            rew(color90);
            rew(exponent);
        }
        ClosureNode::Mix { mix, .. } => rew(mix),
        ClosureNode::Layer { .. } | ClosureNode::Add { .. } => {}
        ClosureNode::Multiply { scale, .. } => rew(scale),
        ClosureNode::IfGreater { value1, value2, .. }
        | ClosureNode::IfGreaterEq { value1, value2, .. }
        | ClosureNode::IfEqual { value1, value2, .. } => {
            rew(value1);
            rew(value2);
        }
        ClosureNode::Switch { which, .. } => rew(which),
        ClosureNode::Surface { opacity, .. } => rew(opacity),
        ClosureNode::GoochShade {
            warm,
            cool,
            specular_intensity,
            shininess,
            light_direction,
        } => {
            rew(warm);
            rew(cool);
            rew(specular_intensity);
            rew(shininess);
            rew(light_direction);
        }
    }
}

fn inline_closure_constant_params(nodes: &mut [ClosureNode], constants: &[Option<Value>]) {
    fn value_to_param(v: Value) -> Option<ParamRef> {
        Some(match v {
            Value::Float(v) => ParamRef::Float(v),
            Value::Integer(v) => ParamRef::Integer(v),
            Value::Bool(v) => ParamRef::Bool(v),
            Value::Color3(v) => ParamRef::Color3(v),
            Value::Color4(v) => ParamRef::Color4(v),
            Value::Vector2(v) => ParamRef::Vector2(v),
            Value::Vector3(v) => ParamRef::Vector3(v),
            Value::Vector4(v) => ParamRef::Vector4(v),
            Value::Matrix33Ref(_) | Value::Matrix44Ref(_) | Value::Empty => return None,
        })
    }
    fn inline_param(p: &mut ParamRef, constants: &[Option<Value>]) {
        if let ParamRef::Local(idx) = *p
            && let Some(Some(v)) = constants.get(idx as usize)
            && let Some(inline) = value_to_param(*v)
        {
            *p = inline;
        }
    }
    fn inline_opt(p: &mut Option<ParamRef>, constants: &[Option<Value>]) {
        if let Some(p) = p {
            inline_param(p, constants);
        }
    }

    for node in nodes {
        match node {
            ClosureNode::Zero => {}
            ClosureNode::OrenNayarDiffuse {
                weight,
                color,
                roughness,
                normal,
                ..
            }
            | ClosureNode::BurleyDiffuse {
                weight,
                color,
                roughness,
                normal,
            } => {
                inline_param(weight, constants);
                inline_param(color, constants);
                inline_param(roughness, constants);
                inline_opt(normal, constants);
            }
            ClosureNode::Translucent {
                weight,
                color,
                normal,
            } => {
                inline_param(weight, constants);
                inline_param(color, constants);
                inline_opt(normal, constants);
            }
            ClosureNode::Dielectric {
                weight,
                tint,
                ior,
                roughness,
                thinfilm_thickness,
                thinfilm_ior,
                normal,
                tangent,
                ..
            } => {
                inline_param(weight, constants);
                inline_param(tint, constants);
                inline_param(ior, constants);
                inline_param(roughness, constants);
                inline_param(thinfilm_thickness, constants);
                inline_param(thinfilm_ior, constants);
                inline_opt(normal, constants);
                inline_opt(tangent, constants);
            }
            ClosureNode::Conductor {
                weight,
                ior,
                extinction,
                roughness,
                retroreflective: _,
                thinfilm_thickness,
                thinfilm_ior,
                normal,
                tangent,
            } => {
                inline_param(weight, constants);
                inline_param(ior, constants);
                inline_param(extinction, constants);
                inline_param(roughness, constants);
                inline_param(thinfilm_thickness, constants);
                inline_param(thinfilm_ior, constants);
                inline_opt(normal, constants);
                inline_opt(tangent, constants);
            }
            ClosureNode::GeneralizedSchlick {
                weight,
                color0,
                color82,
                color90,
                exponent,
                roughness,
                thinfilm_thickness,
                thinfilm_ior,
                normal,
                tangent,
                ..
            } => {
                inline_param(weight, constants);
                inline_param(color0, constants);
                inline_param(color82, constants);
                inline_param(color90, constants);
                inline_param(exponent, constants);
                inline_param(roughness, constants);
                inline_param(thinfilm_thickness, constants);
                inline_param(thinfilm_ior, constants);
                inline_opt(normal, constants);
                inline_opt(tangent, constants);
            }
            ClosureNode::Sheen {
                weight,
                color,
                roughness,
                normal,
                ..
            } => {
                inline_param(weight, constants);
                inline_param(color, constants);
                inline_param(roughness, constants);
                inline_opt(normal, constants);
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
                inline_param(tint_r, constants);
                inline_param(tint_tt, constants);
                inline_param(tint_trt, constants);
                inline_param(absorption, constants);
                inline_param(ior, constants);
                inline_param(roughness_r, constants);
                inline_param(roughness_tt, constants);
                inline_param(roughness_trt, constants);
                inline_param(cuticle_angle, constants);
                inline_opt(normal, constants);
                inline_param(curve_direction, constants);
            }
            ClosureNode::ThinFilm { thickness, ior } => {
                inline_param(thickness, constants);
                inline_param(ior, constants);
            }
            ClosureNode::UniformEdf { color } => inline_param(color, constants),
            ClosureNode::ConicalEdf {
                color,
                inner_angle,
                outer_angle,
                normal,
            } => {
                inline_param(color, constants);
                inline_param(inner_angle, constants);
                inline_param(outer_angle, constants);
                inline_opt(normal, constants);
            }
            ClosureNode::GeneralizedSchlickEdf {
                color0,
                color90,
                exponent,
                ..
            } => {
                inline_param(color0, constants);
                inline_param(color90, constants);
                inline_param(exponent, constants);
            }
            ClosureNode::Mix { mix, .. } => inline_param(mix, constants),
            ClosureNode::Layer { .. } | ClosureNode::Add { .. } => {}
            ClosureNode::Multiply { scale, .. } => inline_param(scale, constants),
            ClosureNode::IfGreater { value1, value2, .. }
            | ClosureNode::IfGreaterEq { value1, value2, .. }
            | ClosureNode::IfEqual { value1, value2, .. } => {
                inline_param(value1, constants);
                inline_param(value2, constants);
            }
            ClosureNode::Switch { which, .. } => inline_param(which, constants),
            ClosureNode::Surface { opacity, .. } => inline_param(opacity, constants),
            ClosureNode::GoochShade {
                warm,
                cool,
                specular_intensity,
                shininess,
                light_direction,
            } => {
                inline_param(warm, constants);
                inline_param(cool, constants);
                inline_param(specular_intensity, constants);
                inline_param(shininess, constants);
                inline_param(light_direction, constants);
            }
        }
    }
}

fn simplify_closure_nodes(nodes: &mut [ClosureNode], root: u32) {
    fn simplify_idx(nodes: &mut [ClosureNode], idx: u32, state: &mut [u8]) {
        let i = idx as usize;
        if state[i] != 0 {
            return;
        }
        state[i] = 1;
        let node = nodes[i].clone();
        match &node {
            ClosureNode::Mix { bg, fg, .. } => {
                simplify_idx(nodes, *bg, state);
                simplify_idx(nodes, *fg, state);
            }
            ClosureNode::Layer { top, base } => {
                simplify_idx(nodes, *top, state);
                simplify_idx(nodes, *base, state);
            }
            ClosureNode::Add { a, b, .. } => {
                simplify_idx(nodes, *a, state);
                simplify_idx(nodes, *b, state);
            }
            ClosureNode::Multiply { inner, .. } => simplify_idx(nodes, *inner, state),
            ClosureNode::IfGreater {
                then_branch,
                else_branch,
                ..
            }
            | ClosureNode::IfGreaterEq {
                then_branch,
                else_branch,
                ..
            }
            | ClosureNode::IfEqual {
                then_branch,
                else_branch,
                ..
            } => {
                simplify_idx(nodes, *then_branch, state);
                simplify_idx(nodes, *else_branch, state);
            }
            ClosureNode::Switch { branches, .. } => {
                for branch in branches {
                    simplify_idx(nodes, *branch, state);
                }
            }
            ClosureNode::Surface { bsdf, edf, .. } => {
                simplify_idx(nodes, *bsdf, state);
                simplify_idx(nodes, *edf, state);
            }
            ClosureNode::GeneralizedSchlickEdf { base, .. } => simplify_idx(nodes, *base, state),
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
            | ClosureNode::GoochShade { .. } => {}
        }

        nodes[i] = simplify_closure_node(&node, nodes);
        state[i] = 2;
    }

    if (root as usize) >= nodes.len() {
        return;
    }
    let mut state = vec![0u8; nodes.len()];
    simplify_idx(nodes, root, &mut state);
}

fn simplify_closure_node(node: &ClosureNode, nodes: &[ClosureNode]) -> ClosureNode {
    match node {
        ClosureNode::OrenNayarDiffuse { weight, .. }
        | ClosureNode::BurleyDiffuse { weight, .. }
        | ClosureNode::Translucent { weight, .. }
        | ClosureNode::Dielectric { weight, .. }
        | ClosureNode::Conductor { weight, .. }
        | ClosureNode::GeneralizedSchlick { weight, .. }
        | ClosureNode::Sheen { weight, .. }
            if param_float_is_zero(weight) =>
        {
            ClosureNode::Zero
        }
        ClosureNode::Mix { bg, fg, mix, kind } => {
            if param_float_is_zero(mix) {
                nodes[*bg as usize].clone()
            } else if param_float_is_one(mix)
                || (!matches!(kind, ClosureKind::Surface) && param_float_is_at_least_one(mix))
                || closure_is_zero(nodes, *bg)
            {
                nodes[*fg as usize].clone()
            } else if closure_is_zero(nodes, *fg) {
                nodes[*bg as usize].clone()
            } else {
                node.clone()
            }
        }
        ClosureNode::Layer { top, base } => {
            if closure_is_zero(nodes, *top) {
                nodes[*base as usize].clone()
            } else if closure_is_zero(nodes, *base) {
                nodes[*top as usize].clone()
            } else {
                node.clone()
            }
        }
        ClosureNode::Add { a, b, kind } => {
            if !matches!(kind, ClosureKind::Bsdf) && closure_is_zero(nodes, *a) {
                nodes[*b as usize].clone()
            } else if !matches!(kind, ClosureKind::Bsdf) && closure_is_zero(nodes, *b) {
                nodes[*a as usize].clone()
            } else {
                node.clone()
            }
        }
        ClosureNode::Multiply { inner, scale, .. } => {
            if closure_is_zero(nodes, *inner) || param_color_is_zero(scale) {
                ClosureNode::Zero
            } else if param_color_is_one(scale) {
                nodes[*inner as usize].clone()
            } else {
                node.clone()
            }
        }
        _ => node.clone(),
    }
}

fn closure_is_zero(nodes: &[ClosureNode], idx: u32) -> bool {
    matches!(nodes[idx as usize], ClosureNode::Zero)
}

fn param_float_is_zero(p: &ParamRef) -> bool {
    matches!(p, ParamRef::Float(v) if v.abs() <= 1.0e-8)
}

fn param_float_is_one(p: &ParamRef) -> bool {
    matches!(p, ParamRef::Float(v) if (*v - 1.0).abs() <= 1.0e-8)
}

fn param_float_is_at_least_one(p: &ParamRef) -> bool {
    matches!(p, ParamRef::Float(v) if *v >= 1.0)
}

fn param_color_is_zero(p: &ParamRef) -> bool {
    match p {
        ParamRef::Float(v) => v.abs() <= 1.0e-8,
        ParamRef::Color3(v) | ParamRef::Vector3(v) => v.length_squared() <= 1.0e-16,
        ParamRef::Color4(v) | ParamRef::Vector4(v) => v.length_squared() <= 1.0e-16,
        _ => false,
    }
}

fn param_color_is_one(p: &ParamRef) -> bool {
    match p {
        ParamRef::Float(v) => (*v - 1.0).abs() <= 1.0e-8,
        ParamRef::Color3(v) | ParamRef::Vector3(v) => v.abs_diff_eq(Vec3::ONE, 1.0e-8),
        ParamRef::Color4(v) | ParamRef::Vector4(v) => v.abs_diff_eq(Vec4::ONE, 1.0e-8),
        _ => false,
    }
}

fn eliminate_dead_instructions(
    instructions: &mut Vec<Instruction>,
    operand_pool: &[Operand],
    closure_nodes: &[ClosureNode],
    root: u32,
    num_vregs: u32,
) {
    if instructions.is_empty() || num_vregs == 0 {
        return;
    }

    let mut needed = vec![false; num_vregs as usize];
    closure_mark_reachable_locals(closure_nodes, root, &mut needed);
    eliminate_dead_instructions_with_live(instructions, operand_pool, &mut needed);
}

fn eliminate_dead_instructions_with_live(
    instructions: &mut Vec<Instruction>,
    operand_pool: &[Operand],
    needed: &mut [bool],
) {
    if instructions.is_empty() || needed.is_empty() {
        return;
    }
    let mut keep = vec![false; instructions.len()];
    for i in (0..instructions.len()).rev() {
        let instr = &instructions[i];
        let keep_instr = instr_dst(instr).is_none_or(|dst| {
            let dst = dst as usize;
            dst < needed.len() && needed[dst]
        });
        if !keep_instr {
            continue;
        }
        keep[i] = true;
        instr_visit_inline_operands(instr, |r| {
            let r = r as usize;
            if r < needed.len() {
                needed[r] = true;
            }
        });
        if let Some((start, count)) = instr_pool_range(instr) {
            for op in &operand_pool[start as usize..start as usize + count] {
                if let Operand::Reg(r) = op {
                    let r = *r as usize;
                    if r < needed.len() {
                        needed[r] = true;
                    }
                }
            }
        }
    }

    let mut i = 0;
    instructions.retain(|_| {
        let k = keep[i];
        i += 1;
        k
    });
}

/// Linear-scan register allocator: SSA virtual register id を physical slot id
/// に lower する。
fn allocate_registers(
    instructions: &mut [Instruction],
    operand_pool: &mut [Operand],
    closure_nodes: &mut [ClosureNode],
    root: u32,
    num_vregs: u32,
) -> u32 {
    if num_vregs == 0 || instructions.is_empty() {
        return 0;
    }
    let n_vregs = num_vregs as usize;
    let n_instr = instructions.len();

    // Step 1: live-out 判定 (closure-referenced vreg は bytecode 末尾以降も live)
    let mut live_out = vec![false; n_vregs];
    closure_mark_reachable_locals(closure_nodes, root, &mut live_out);

    // Step 2: 各 vreg の last bytecode use index を計算
    let mut last_use = vec![0u32; n_vregs];
    for (i, instr) in instructions.iter().enumerate() {
        let i = i as u32;
        instr_visit_inline_operands(instr, |r| {
            let r = r as usize;
            if r < n_vregs && last_use[r] < i {
                last_use[r] = i;
            }
        });
        if let Some((start, count)) = instr_pool_range(instr) {
            for op in &operand_pool[start as usize..start as usize + count] {
                if let Operand::Reg(r) = op {
                    let r = *r as usize;
                    if r < n_vregs && last_use[r] < i {
                        last_use[r] = i;
                    }
                }
            }
        }
    }
    // live-out vreg は last_use = sentinel (= n_instr) で永続化
    for r in 0..n_vregs {
        if live_out[r] {
            last_use[r] = n_instr as u32;
        }
    }

    // Step 3: linear-scan 割当
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let mut mapping: Vec<u16> = vec![u16::MAX; n_vregs];
    let mut next_slot: u16 = 0;
    let mut active: Vec<(u16, u32)> = Vec::new(); // (slot, vreg)
    let mut free_slots: BinaryHeap<Reverse<u16>> = BinaryHeap::new();

    for (i, instr) in instructions.iter().enumerate() {
        let i = i as u32;
        // この命令以降使われない vreg の slot を解放 (live-out は除く)
        let mut idx = 0;
        while idx < active.len() {
            let (slot, vreg) = active[idx];
            if last_use[vreg as usize] < i {
                free_slots.push(Reverse(slot));
                active.swap_remove(idx);
            } else {
                idx += 1;
            }
        }
        // dst に slot を割り当て
        if let Some(d) = instr_dst(instr) {
            let d = d as usize;
            if mapping[d] == u16::MAX {
                let slot = if let Some(Reverse(s)) = free_slots.pop() {
                    s
                } else {
                    let s = next_slot;
                    next_slot = next_slot.checked_add(1).expect("too many SSA slots");
                    s
                };
                mapping[d] = slot;
                active.push((slot, d as u32));
            }
        }
    }

    // 未割当の vreg (= 命令未参照だが closure 参照されている等) があれば新 slot 採番
    for slot in mapping.iter_mut().take(n_vregs) {
        if *slot == u16::MAX {
            *slot = next_slot;
            next_slot = next_slot.checked_add(1).expect("too many SSA slots");
        }
    }

    // Step 4: 全 reference を rewrite
    for instr in instructions.iter_mut() {
        rewrite_instruction_inline(instr, &mapping);
    }
    for op in operand_pool.iter_mut() {
        rewrite_op(op, &mapping);
    }
    for node in closure_nodes.iter_mut() {
        closure_rewrite_locals(node, &mapping);
    }

    next_slot as u32
}

fn closure_is_thin_walled(nodes: &[ClosureNode], root: u32) -> bool {
    fn walk(nodes: &[ClosureNode], idx: u32, seen: &mut Vec<bool>) -> bool {
        if seen[idx as usize] {
            return false;
        }
        seen[idx as usize] = true;
        match &nodes[idx as usize] {
            ClosureNode::Surface { thin_walled, .. } => *thin_walled,
            ClosureNode::Mix { bg, fg, .. } => walk(nodes, *bg, seen) || walk(nodes, *fg, seen),
            ClosureNode::Layer { top, base } => walk(nodes, *top, seen) || walk(nodes, *base, seen),
            _ => false,
        }
    }
    let mut seen = vec![false; nodes.len()];
    walk(nodes, root, &mut seen)
}

fn closure_has_opacity_test(
    nodes: &[ClosureNode],
    root: u32,
    instructions: &[Instruction],
    value_pool: &[Value],
    num_vregs: u32,
) -> bool {
    let constants = local_constants(instructions, value_pool, num_vregs);
    fn walk(
        nodes: &[ClosureNode],
        idx: u32,
        seen: &mut Vec<bool>,
        constants: &[Option<Value>],
    ) -> bool {
        if seen[idx as usize] {
            return false;
        }
        seen[idx as usize] = true;
        match &nodes[idx as usize] {
            ClosureNode::Surface {
                opacity, bsdf, edf, ..
            } => {
                let opaque =
                    param_float_value(opacity, constants).is_some_and(|f| (f - 1.0).abs() < 1.0e-6);
                if !opaque {
                    return true;
                }
                walk(nodes, *bsdf, seen, constants) || walk(nodes, *edf, seen, constants)
            }
            ClosureNode::Mix { bg, fg, .. } => {
                walk(nodes, *bg, seen, constants) || walk(nodes, *fg, seen, constants)
            }
            ClosureNode::Layer { top, base } => {
                walk(nodes, *top, seen, constants) || walk(nodes, *base, seen, constants)
            }
            ClosureNode::Add { a, b, .. } => {
                walk(nodes, *a, seen, constants) || walk(nodes, *b, seen, constants)
            }
            ClosureNode::Multiply { inner, .. } => walk(nodes, *inner, seen, constants),
            ClosureNode::IfGreater {
                then_branch,
                else_branch,
                ..
            }
            | ClosureNode::IfGreaterEq {
                then_branch,
                else_branch,
                ..
            }
            | ClosureNode::IfEqual {
                then_branch,
                else_branch,
                ..
            } => {
                walk(nodes, *then_branch, seen, constants)
                    || walk(nodes, *else_branch, seen, constants)
            }
            ClosureNode::Switch { branches, .. } => {
                branches.iter().any(|b| walk(nodes, *b, seen, constants))
            }
            _ => false,
        }
    }
    let mut seen = vec![false; nodes.len()];
    walk(nodes, root, &mut seen, &constants)
}

fn fold_constant_instructions(
    instructions: &mut [Instruction],
    operand_pool: &[Operand],
    value_pool: &mut Vec<Value>,
    color_processors: &[Arc<color::OcioColorProcessor>],
    num_vregs: u32,
) -> usize {
    let mut constants = vec![None; num_vregs as usize];
    let mut folded = 0;
    for instr in instructions {
        if let Some(dst) = instr_dst(instr)
            && let Some(slot) = constants.get_mut(dst as usize)
        {
            *slot = None;
        }
        if let Instruction::LoadConst {
            dst,
            value_pool_idx,
        } = instr
        {
            if (*dst as usize) < constants.len() {
                constants[*dst as usize] = value_pool.get(*value_pool_idx as usize).copied();
            }
            continue;
        }
        if let Some((dst, value)) = constant_instruction_value(
            instr,
            operand_pool,
            &constants,
            value_pool,
            color_processors,
        ) && inline_constant_value(value)
        {
            let value_pool_idx = value_pool.len() as u32;
            value_pool.push(value);
            *instr = Instruction::LoadConst {
                dst,
                value_pool_idx,
            };
            if (dst as usize) < constants.len() {
                constants[dst as usize] = Some(value);
            }
            folded += 1;
        }
    }
    folded
}

fn inline_constant_value(v: Value) -> bool {
    !matches!(
        v,
        Value::Matrix33Ref(_) | Value::Matrix44Ref(_) | Value::Empty
    )
}

fn constant_instruction_value(
    instr: &Instruction,
    operand_pool: &[Operand],
    constants: &[Option<Value>],
    value_pool: &[Value],
    color_processors: &[Arc<color::OcioColorProcessor>],
) -> Option<(u16, Value)> {
    use Instruction::*;
    let c = |op| operand_constant(op, constants, value_pool);
    let pool = |start: u32, offset: usize| operand_pool.get(start as usize + offset).copied();
    Some(match instr {
        LoadConst { .. }
        | LoadGeom { .. }
        | LoadMat3Const { .. }
        | LoadMat4Const { .. }
        | Image { .. }
        | HextiledImage { .. }
        | HextiledNormalMap { .. }
        | TransformPoint { .. }
        | TransformVector { .. }
        | TransformNormal { .. }
        | TransformMatrix { .. }
        | Transpose { .. }
        | Determinant { .. }
        | InvertMatrix { .. }
        | Noise { .. }
        | Worley { .. }
        | Cellnoise { .. }
        | RandomFloat { .. }
        | RandomColor { .. }
        | TriplanarBlend { .. }
        | CurveUniformLinear { .. }
        | CurveUniformCubic { .. }
        | CurveInverseCubic { .. }
        | Normalmap { .. }
        | NormalmapWithFrame { .. }
        | Bump { .. }
        | BumpWithFrame { .. }
        | HeightToNormal { .. }
        | Passthrough => return None,
        Arith { dst, op, ty, a, b } => {
            if matches!(ty, ValueType::Matrix33 | ValueType::Matrix44) {
                return None;
            }
            (*dst, super::runtime::arith(c(*a)?, c(*b)?, *op, *ty))
        }
        Unary { dst, op, ty, src } => (*dst, super::runtime::unary(c(*src)?, *op, *ty)),
        Convert { dst, from, to, src } => {
            if matches!(to, ValueType::Matrix33 | ValueType::Matrix44) {
                return None;
            }
            (*dst, super::runtime::convert_value(c(*src)?, *from, *to))
        }
        Logical { dst, op, a, b } => {
            let av = c(*a)?.as_bool();
            let bv = c(*b).map(Value::as_bool).unwrap_or(false);
            let v = match op {
                LogicalOp::Not => !av,
                LogicalOp::And => av && bv,
                LogicalOp::Or => av || bv,
                LogicalOp::Xor => av != bv,
            };
            (*dst, Value::Bool(v))
        }
        CompareBool { dst, op, v1, v2 } => {
            let a = c(*v1)?.as_float();
            let b = c(*v2)?.as_float();
            let v = match op {
                CompareOp::Greater => a > b,
                CompareOp::GreaterEq => a >= b,
                CompareOp::Equal => a == b,
            };
            (*dst, Value::Bool(v))
        }
        Compare {
            dst,
            op,
            v1,
            v2,
            in_true,
            in_false,
        } => {
            let a = c(*v1)?.as_float();
            let b = c(*v2)?.as_float();
            let cond = match op {
                CompareOp::Greater => a > b,
                CompareOp::GreaterEq => a >= b,
                CompareOp::Equal => a == b,
            };
            (*dst, c(if cond { *in_true } else { *in_false })?)
        }
        IfElse {
            dst,
            cond,
            in_true,
            in_false,
        } => (
            *dst,
            c(if c(*cond)?.as_bool() {
                *in_true
            } else {
                *in_false
            })?,
        ),
        MixValue {
            dst,
            ty,
            bg,
            fg,
            mix,
        } => (
            *dst,
            super::runtime::mix_value(c(*bg)?, c(*fg)?, c(*mix)?, *ty),
        ),
        Clamp { dst, ty, v, lo, hi } => (
            *dst,
            super::runtime::clamp_value(c(*v)?, c(*lo)?, c(*hi)?, *ty),
        ),
        Smoothstep { dst, ty, v, lo, hi } => (
            *dst,
            super::runtime::smoothstep_value(c(*v)?, c(*lo)?, c(*hi)?, *ty),
        ),
        Extract {
            dst,
            in_ty,
            src,
            idx,
        } => (
            *dst,
            extract_constant(c(*src)?, *in_ty, c(*idx)?.as_integer()),
        ),
        ExtractRowVector { .. } => return None,
        Reflect { dst, i, n } => {
            let iv = c(*i)?.as_vector3();
            let nv = c(*n)?.as_vector3();
            (*dst, Value::Vector3(iv - 2.0 * iv.dot(nv) * nv))
        }
        Refract { dst, i, n, eta } => {
            let iv = c(*i)?.as_vector3();
            let nv = c(*n)?.as_vector3();
            let e = c(*eta)?.as_float();
            let cosi = (-iv).dot(nv);
            let k = 1.0 - e * e * (1.0 - cosi * cosi);
            let r = if k < 0.0 {
                Vec3::ZERO
            } else {
                e * iv + (e * cosi - k.sqrt()) * nv
            };
            (*dst, Value::Vector3(r))
        }
        Rotate2d { dst, v, amount } => {
            let vv = c(*v)?.as_vector2();
            let a = c(*amount)?.as_float().to_radians();
            let (s, co) = a.sin_cos();
            (
                *dst,
                Value::Vector2(Vec2::new(co * vv.x + s * vv.y, -s * vv.x + co * vv.y)),
            )
        }
        Rotate3d {
            dst,
            v,
            axis,
            amount,
        } => {
            let vv = c(*v)?.as_vector3();
            let ax = c(*axis)?.as_vector3();
            let a = c(*amount)?.as_float().to_radians();
            let (s, co) = a.sin_cos();
            (
                *dst,
                Value::Vector3(vv * co + ax.cross(vv) * s + ax * ax.dot(vv) * (1.0 - co)),
            )
        }
        DotProduct { dst, ty, a, b } => {
            let av = c(*a)?;
            let bv = c(*b)?;
            let r = match ty {
                ValueType::Vector2 => av.as_vector2().dot(bv.as_vector2()),
                ValueType::Vector4 | ValueType::Color4 => av.as_color4().dot(bv.as_color4()),
                _ => av.as_vector3().dot(bv.as_vector3()),
            };
            (*dst, Value::Float(r))
        }
        CrossProduct { dst, a, b } => (
            *dst,
            Value::Vector3(c(*a)?.as_vector3().cross(c(*b)?.as_vector3())),
        ),
        Distance { dst, ty, a, b } => {
            let av = c(*a)?;
            let bv = c(*b)?;
            let d = match ty {
                ValueType::Vector2 => (av.as_vector2() - bv.as_vector2()).length(),
                ValueType::Vector4 | ValueType::Color4 => {
                    (av.as_color4() - bv.as_color4()).length()
                }
                _ => (av.as_vector3() - bv.as_vector3()).length(),
            };
            (*dst, Value::Float(d))
        }
        FacingRatio {
            dst,
            view,
            normal,
            invert,
            faceforward,
        } => {
            let dot = c(*view)?.as_vector3().dot(c(*normal)?.as_vector3());
            let mut f = if *faceforward { dot.abs() } else { -dot };
            if *invert {
                f = 1.0 - f;
            }
            (*dst, Value::Float(f))
        }
        LuminanceWithCoeffs {
            dst,
            ty,
            c: color,
            lumacoeffs,
        } => (*dst, luminance_constant(c(*color)?, c(*lumacoeffs)?, *ty)),
        Combine {
            dst,
            kind,
            operands_start,
        } => (
            *dst,
            combine_constant(*kind, *operands_start, operand_pool, constants, value_pool)?,
        ),
        CreateMatrix3 { .. } | CreateMatrix4 { .. } | CreateMatrix4FromVec3 { .. } => return None,
        Switch {
            dst,
            ty,
            which,
            branches_start,
        } => {
            let i = c(*which)?.as_integer().clamp(0, 9) as usize;
            let v = c(pool(*branches_start, i)?)?;
            (
                *dst,
                super::runtime::convert_value(v, super::runtime::value_type_of(v), *ty),
            )
        }
        _ => {
            return constant_instruction_value_tail(
                instr,
                operand_pool,
                constants,
                value_pool,
                color_processors,
            );
        }
    })
}

fn constant_instruction_value_tail(
    instr: &Instruction,
    operand_pool: &[Operand],
    constants: &[Option<Value>],
    value_pool: &[Value],
    color_processors: &[Arc<color::OcioColorProcessor>],
) -> Option<(u16, Value)> {
    use Instruction::*;
    let c = |op| operand_constant(op, constants, value_pool);
    let pool = |start: u32, offset: usize| operand_pool.get(start as usize + offset).copied();
    Some(match instr {
        Place2d {
            dst,
            trs,
            texcoord,
            pivot,
            scale,
            rotate,
            offset,
        } => {
            let tc = c(*texcoord)?.as_vector2();
            let pv = c(*pivot)?.as_vector2();
            let sc = c(*scale)?.as_vector2();
            let ro = c(*rotate)?.as_float();
            let of = c(*offset)?.as_vector2();
            let safe_div =
                |a: Vec2, b: Vec2| Vec2::new(a.x / b.x.max(1.0e-30), a.y / b.y.max(1.0e-30));
            let rotate2d_uv = |v: Vec2, deg: f32| {
                let (s, co) = deg.to_radians().sin_cos();
                Vec2::new(co * v.x + s * v.y, -s * v.x + co * v.y)
            };
            let result = if *trs {
                safe_div(rotate2d_uv(tc - pv - of, ro), sc) + pv
            } else {
                rotate2d_uv(safe_div(tc - pv, sc), ro) - of + pv
            };
            (*dst, Value::Vector2(result))
        }
        LatlongUv {
            dst,
            viewdir,
            rotation,
        } => {
            let v = c(*viewdir)?.as_vector3();
            let r = c(*rotation)?.as_float();
            let u = v.x.atan2(v.z) * (-1.0 / (2.0 * std::f32::consts::PI)) + 0.5 + r / 360.0;
            let vv = v.y.clamp(-1.0, 1.0).asin() * (1.0 / std::f32::consts::PI) + 0.5;
            (*dst, Value::Vector2(Vec2::new(u, vv)))
        }
        Ramplr {
            dst,
            ty,
            texcoord,
            l,
            r,
        } => {
            let t = c(*texcoord)?.as_vector2().x.clamp(0.0, 1.0);
            (
                *dst,
                super::runtime::mix_value(c(*l)?, c(*r)?, Value::Float(t), *ty),
            )
        }
        Ramptb {
            dst,
            ty,
            texcoord,
            t,
            b,
        } => {
            let u = c(*texcoord)?.as_vector2().y.clamp(0.0, 1.0);
            (
                *dst,
                super::runtime::mix_value(c(*t)?, c(*b)?, Value::Float(u), *ty),
            )
        }
        Ramp4 {
            dst,
            ty,
            texcoord,
            tl,
            tr,
            bl,
            br,
        } => {
            let tc = c(*texcoord)?.as_vector2();
            let u = tc.x.clamp(0.0, 1.0);
            let v = tc.y.clamp(0.0, 1.0);
            let top = super::runtime::mix_value(c(*tl)?, c(*tr)?, Value::Float(u), *ty);
            let bot = super::runtime::mix_value(c(*bl)?, c(*br)?, Value::Float(u), *ty);
            (
                *dst,
                super::runtime::mix_value(top, bot, Value::Float(v), *ty),
            )
        }
        Splitlr {
            dst,
            texcoord,
            center,
            l,
            r,
            ..
        } => (
            *dst,
            if c(*texcoord)?.as_vector2().x < c(*center)?.as_float() {
                c(*l)?
            } else {
                c(*r)?
            },
        ),
        Splittb {
            dst,
            texcoord,
            center,
            t,
            b,
            ..
        } => (
            *dst,
            if c(*texcoord)?.as_vector2().x < c(*center)?.as_float() {
                c(*t)?
            } else {
                c(*b)?
            },
        ),
        Blackbody { dst, temp } => (
            *dst,
            Value::Color3(super::runtime::blackbody(c(*temp)?.as_float())),
        ),
        ArtisticIor {
            dst,
            which,
            refl,
            edge,
        } => {
            let (ior, ext) =
                super::runtime::artistic_ior(c(*refl)?.as_color3(), c(*edge)?.as_color3());
            (
                *dst,
                Value::Color3(match which {
                    ArtisticIorOutput::Ior => ior,
                    ArtisticIorOutput::Extinction => ext,
                }),
            )
        }
        ChiangHairRoughness {
            dst,
            which,
            longitudinal,
            azimuthal,
            scale_tt,
            scale_trt,
        } => {
            let l = c(*longitudinal)?.as_float();
            let a = c(*azimuthal)?.as_float();
            let stt = c(*scale_tt)?.as_float();
            let strt = c(*scale_trt)?.as_float();
            let lr = l.clamp(1.0e-3, 1.0);
            let ar = a.clamp(1.0e-3, 1.0);
            let v = (0.726 * lr + 0.812 * lr * lr + 3.7 * lr.powi(20)).powi(2);
            let s = 0.265 * ar + 1.194 * ar * ar + 5.372 * ar.powi(22);
            let roughness = match which {
                ChiangHairRoughnessOutput::R => Vec2::new(v, s),
                ChiangHairRoughnessOutput::TT => Vec2::new(v * stt * stt, s),
                ChiangHairRoughnessOutput::TRT => Vec2::new(v * strt * strt, s),
            };
            (*dst, Value::Vector2(roughness))
        }
        DeonHairAbsorptionFromMelanin {
            dst,
            operands_start,
        } => {
            let conc = c(pool(*operands_start, 0)?)?.as_float();
            let redness = c(pool(*operands_start, 1)?)?.as_float();
            let eum = c(pool(*operands_start, 2)?)?.as_color3();
            let phe = c(pool(*operands_start, 3)?)?.as_color3();
            let melanin = -(1.0 - conc).max(0.0001).ln();
            let eumelanin = melanin * (1.0 - redness);
            let pheomelanin = melanin * redness;
            let eum_absorb = Vec3::new(-eum.x.ln(), -eum.y.ln(), -eum.z.ln());
            let phe_absorb = Vec3::new(-phe.x.ln(), -phe.y.ln(), -phe.z.ln());
            (
                *dst,
                Value::Color3((eumelanin * eum_absorb + pheomelanin * phe_absorb).max(Vec3::ZERO)),
            )
        }
        ChiangHairAbsorptionFromColor { dst, color, beta } => {
            let cval = c(*color)?.as_color3().clamp(Vec3::splat(0.001), Vec3::ONE);
            let b = c(*beta)?.as_float();
            let factor = 5.969 - 0.215 * b + 2.532 * b * b - 10.73 * b.powi(3)
                + 5.574 * b.powi(4)
                + 0.245 * b.powi(5);
            let log_c = Vec3::new(cval.x.ln(), cval.y.ln(), cval.z.ln());
            (*dst, Value::Color3((log_c / factor).powf(2.0)))
        }
        RoughnessAnisotropy { dst, r, a } => (
            *dst,
            Value::Vector2(super::runtime::roughness_anisotropy_mdl(
                c(*r)?.as_float(),
                c(*a)?.as_float(),
            )),
        ),
        GlossinessAnisotropy { dst, g, a } => (
            *dst,
            Value::Vector2(super::runtime::roughness_anisotropy_mdl(
                1.0 - c(*g)?.as_float(),
                c(*a)?.as_float(),
            )),
        ),
        RoughnessDual { dst, src } => {
            let mut r = c(*src)?.as_vector2();
            if r.y < 0.0 {
                r.y = r.x;
            }
            (
                *dst,
                Value::Vector2(Vec2::new(
                    (r.x * r.x).clamp(super::runtime::MDL_FLOAT_EPS, 1.0),
                    (r.y * r.y).clamp(super::runtime::MDL_FLOAT_EPS, 1.0),
                )),
            )
        }
        TransformColor { dst, op, ty, src } => {
            let v = c(*src)?;
            let out = match op {
                ColorXform::Identity => v,
                ColorXform::TextureToRendering | ColorXform::RenderingToTexture => v,
                ColorXform::Ocio { processor } => {
                    ocio_xform_constant(v, *ty, &color_processors[*processor as usize])
                }
            };
            (*dst, out)
        }
        Blend {
            dst,
            op,
            ty,
            bg,
            fg,
            mix,
        } => (
            *dst,
            super::runtime::execute_blend(*op, *ty, c(*bg)?, c(*fg)?, c(*mix)?.as_float()),
        ),
        Merge {
            dst,
            op,
            bg,
            fg,
            mix,
        } => (
            *dst,
            super::runtime::execute_merge(
                *op,
                c(*bg)?.as_color4(),
                c(*fg)?.as_color4(),
                c(*mix)?.as_float(),
            ),
        ),
        Mask {
            dst,
            op,
            ty,
            v,
            mask,
        } => {
            let m = match op {
                MaskOp::Inside => c(*mask)?.as_float(),
                MaskOp::Outside => 1.0 - c(*mask)?.as_float(),
            };
            (*dst, super::runtime::scale_value(c(*v)?, m, *ty))
        }
        Premult { dst, src } => {
            let v = c(*src)?.as_color4();
            (
                *dst,
                Value::Color4(Vec4::new(v.x * v.w, v.y * v.w, v.z * v.w, v.w)),
            )
        }
        Unpremult { dst, src } => {
            let v = c(*src)?.as_color4();
            let out = if v.w == 0.0 {
                v
            } else {
                Vec4::new(v.x / v.w, v.y / v.w, v.z / v.w, v.w)
            };
            (*dst, Value::Color4(out))
        }
        Contrast {
            dst,
            ty,
            v,
            amount,
            pivot,
        } => (
            *dst,
            super::runtime::apply_contrast_v(c(*v)?, c(*amount)?, c(*pivot)?, *ty),
        ),
        Range {
            dst,
            ty,
            doclamp,
            operands_start,
        } => (
            *dst,
            super::runtime::apply_range_g(
                c(pool(*operands_start, 0)?)?,
                c(pool(*operands_start, 1)?)?,
                c(pool(*operands_start, 2)?)?,
                c(pool(*operands_start, 3)?)?,
                c(pool(*operands_start, 4)?)?,
                c(pool(*operands_start, 5)?)?,
                *doclamp,
                *ty,
            ),
        ),
        Remap {
            dst,
            ty,
            operands_start,
        } => {
            let one = match ty {
                ValueType::Float | ValueType::Integer => Value::Float(1.0),
                ValueType::Color3 => Value::Color3(Vec3::ONE),
                ValueType::Vector2 => Value::Vector2(Vec2::ONE),
                ValueType::Vector3 => Value::Vector3(Vec3::ONE),
                ValueType::Color4 => Value::Color4(Vec4::ONE),
                ValueType::Vector4 => Value::Vector4(Vec4::ONE),
                _ => Value::Float(1.0),
            };
            (
                *dst,
                super::runtime::apply_range_g(
                    c(pool(*operands_start, 0)?)?,
                    c(pool(*operands_start, 1)?)?,
                    c(pool(*operands_start, 2)?)?,
                    one,
                    c(pool(*operands_start, 3)?)?,
                    c(pool(*operands_start, 4)?)?,
                    false,
                    *ty,
                ),
            )
        }
        HsvAdjust {
            dst,
            ty,
            c: color,
            amount,
        } => {
            let cv = c(*color)?.as_color4();
            let av = c(*amount)?.as_color3();
            let hsv = rgb_to_hsv(Vec3::new(cv.x, cv.y, cv.z));
            let h = hsv.x + av.x - (hsv.x + av.x).floor();
            (
                *dst,
                super::runtime::typed_color_with_alpha(
                    hsv_to_rgb(h, hsv.y * av.y, hsv.z * av.z),
                    cv.w,
                    *ty,
                ),
            )
        }
        Saturate {
            dst,
            ty,
            c: color,
            amount,
            lumacoeffs,
        } => {
            let cv = c(*color)?.as_color4();
            let av = c(*amount)?.as_float();
            let lc = c(*lumacoeffs)?.as_color3();
            let rgb = Vec3::new(cv.x, cv.y, cv.z);
            let lum = rgb.dot(lc);
            (
                *dst,
                super::runtime::typed_color_with_alpha(Vec3::splat(lum).lerp(rgb, av), cv.w, *ty),
            )
        }
        ColorCorrect {
            dst,
            ty,
            operands_start,
        } => (
            *dst,
            color_correct_constant(*ty, *operands_start, operand_pool, constants, value_pool)?,
        ),
        Checkerboard {
            dst,
            color1,
            color2,
            uvtiling,
            uvoffset,
            texcoord,
        } => {
            let c1 = c(*color1)?.as_color3();
            let c2 = c(*color2)?.as_color3();
            let st = c(*texcoord)?.as_vector2() * c(*uvtiling)?.as_vector2()
                - c(*uvoffset)?.as_vector2();
            let cell = ((st.x.floor() as i32) + (st.y.floor() as i32)).rem_euclid(2);
            (*dst, Value::Color3(if cell == 0 { c1 } else { c2 }))
        }
        _ => return None,
    })
}

fn luminance_constant(c: Value, lumacoeffs: Value, ty: ValueType) -> Value {
    let lc = lumacoeffs.as_color3();
    let lum = match ty {
        ValueType::Color4 | ValueType::Vector4 => {
            let v4 = c.as_color4();
            Vec3::new(v4.x, v4.y, v4.z).dot(lc)
        }
        _ => c.as_color3().dot(lc),
    };
    match ty {
        ValueType::Color4 => {
            let v4 = c.as_color4();
            Value::Color4(Vec4::new(lum, lum, lum, v4.w))
        }
        ValueType::Vector4 => {
            let v4 = c.as_color4();
            Value::Vector4(Vec4::new(lum, lum, lum, v4.w))
        }
        _ => Value::Color3(Vec3::splat(lum)),
    }
}

fn combine_constant(
    kind: CombineKind,
    operands_start: u32,
    operand_pool: &[Operand],
    constants: &[Option<Value>],
    value_pool: &[Value],
) -> Option<Value> {
    let s = operands_start as usize;
    let c = |offset| operand_constant(*operand_pool.get(s + offset)?, constants, value_pool);
    Some(match kind {
        CombineKind::Vector2FromFloats => {
            Value::Vector2(Vec2::new(c(0)?.as_float(), c(1)?.as_float()))
        }
        CombineKind::Color3FromFloats => Value::Color3(Vec3::new(
            c(0)?.as_float(),
            c(1)?.as_float(),
            c(2)?.as_float(),
        )),
        CombineKind::Vector3FromFloats => Value::Vector3(Vec3::new(
            c(0)?.as_float(),
            c(1)?.as_float(),
            c(2)?.as_float(),
        )),
        CombineKind::Color4FromFloats => Value::Color4(Vec4::new(
            c(0)?.as_float(),
            c(1)?.as_float(),
            c(2)?.as_float(),
            c(3)?.as_float(),
        )),
        CombineKind::Vector4FromFloats => Value::Vector4(Vec4::new(
            c(0)?.as_float(),
            c(1)?.as_float(),
            c(2)?.as_float(),
            c(3)?.as_float(),
        )),
        CombineKind::Color4FromColor3Float => {
            let rgb = c(0)?.as_color3();
            Value::Color4(Vec4::new(rgb.x, rgb.y, rgb.z, c(1)?.as_float()))
        }
        CombineKind::Vector4FromVector3Float => {
            let xyz = c(0)?.as_vector3();
            Value::Vector4(Vec4::new(xyz.x, xyz.y, xyz.z, c(1)?.as_float()))
        }
        CombineKind::Vector4FromVector2Vector2 => {
            let xy = c(0)?.as_vector2();
            let zw = c(1)?.as_vector2();
            Value::Vector4(Vec4::new(xy.x, xy.y, zw.x, zw.y))
        }
    })
}

fn materialx_color_space_to_ocio(name: &str, rendering_space: &str) -> String {
    match name {
        "" | "linear" | "scene_linear" => rendering_space.to_string(),
        "none" => rendering_space.to_string(),
        "g22" => "g22_rec709".to_string(),
        other => color::map_materialx_color_space(other).to_string(),
    }
}

fn ocio_xform_constant(v: Value, ty: ValueType, processor: &color::OcioColorProcessor) -> Value {
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

fn color_correct_constant(
    ty: ValueType,
    operands_start: u32,
    operand_pool: &[Operand],
    constants: &[Option<Value>],
    value_pool: &[Value],
) -> Option<Value> {
    let s = operands_start as usize;
    let c = |offset| operand_constant(*operand_pool.get(s + offset)?, constants, value_pool);
    let cv = c(0)?.as_color4();
    let hue = c(1)?.as_float();
    let sat = c(2)?.as_float();
    let gamma = c(3)?.as_float();
    let lift = c(4)?.as_float();
    let gain = c(5)?.as_float();
    let contrast = c(6)?.as_float();
    let pivot = c(7)?.as_float();
    let exposure = c(8)?.as_float();
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
    Some(super::runtime::typed_color_with_alpha(rgb, cv.w, ty))
}

fn local_constants(
    instructions: &[Instruction],
    value_pool: &[Value],
    num_vregs: u32,
) -> Vec<Option<Value>> {
    let mut constants = vec![None; num_vregs as usize];
    for instr in instructions {
        if let Some(dst) = instr_dst(instr)
            && let Some(slot) = constants.get_mut(dst as usize)
        {
            *slot = None;
        }
        match instr {
            Instruction::LoadConst {
                dst,
                value_pool_idx,
            } if (*dst as usize) < constants.len() => {
                constants[*dst as usize] = value_pool.get(*value_pool_idx as usize).copied();
            }
            Instruction::LoadConst {
                dst: _,
                value_pool_idx: _,
            } => {}
            Instruction::Arith { dst, op, ty, a, b } => {
                if (*dst as usize) < constants.len()
                    && !matches!(ty, ValueType::Matrix33 | ValueType::Matrix44)
                    && let (Some(av), Some(bv)) = (
                        operand_constant(*a, &constants, value_pool),
                        operand_constant(*b, &constants, value_pool),
                    )
                {
                    constants[*dst as usize] = Some(super::runtime::arith(av, bv, *op, *ty));
                }
            }
            Instruction::Convert { dst, from, to, src } => {
                if (*dst as usize) < constants.len()
                    && let Some(v) = operand_constant(*src, &constants, value_pool)
                {
                    constants[*dst as usize] = Some(super::runtime::convert_value(v, *from, *to));
                }
            }
            Instruction::LuminanceWithCoeffs {
                dst,
                ty,
                c,
                lumacoeffs,
            } => {
                if (*dst as usize) < constants.len()
                    && let (Some(cv), Some(lc)) = (
                        operand_constant(*c, &constants, value_pool),
                        operand_constant(*lumacoeffs, &constants, value_pool),
                    )
                {
                    let lc = lc.as_color3();
                    let lum = match ty {
                        ValueType::Color4 | ValueType::Vector4 => {
                            let v4 = cv.as_color4();
                            Vec3::new(v4.x, v4.y, v4.z).dot(lc)
                        }
                        _ => cv.as_color3().dot(lc),
                    };
                    constants[*dst as usize] = Some(match ty {
                        ValueType::Color4 => {
                            let v4 = cv.as_color4();
                            Value::Color4(Vec4::new(lum, lum, lum, v4.w))
                        }
                        ValueType::Vector4 => {
                            let v4 = cv.as_color4();
                            Value::Vector4(Vec4::new(lum, lum, lum, v4.w))
                        }
                        _ => Value::Color3(Vec3::splat(lum)),
                    });
                }
            }
            Instruction::Extract {
                dst,
                in_ty,
                src,
                idx,
            } => {
                if (*dst as usize) < constants.len()
                    && let (Some(src), Some(idx)) = (
                        operand_constant(*src, &constants, value_pool),
                        operand_constant(*idx, &constants, value_pool),
                    )
                {
                    constants[*dst as usize] =
                        Some(extract_constant(src, *in_ty, idx.as_integer()));
                }
            }
            _ => {}
        }
    }
    constants
}

fn operand_constant(
    op: Operand,
    constants: &[Option<Value>],
    value_pool: &[Value],
) -> Option<Value> {
    match op {
        Operand::Reg(r) => constants.get(r as usize).and_then(|v| *v),
        Operand::Const(idx) => value_pool.get(idx as usize).copied(),
    }
}

fn extract_constant(src: Value, in_ty: ValueType, idx: i32) -> Value {
    let f = match in_ty {
        ValueType::Vector2 => {
            let v = src.as_vector2();
            if idx == 0 { v.x } else { v.y }
        }
        ValueType::Color3 | ValueType::Vector3 => {
            let v = src.as_vector3();
            match idx {
                0 => v.x,
                1 => v.y,
                _ => v.z,
            }
        }
        ValueType::Color4 | ValueType::Vector4 => {
            let v = src.as_color4();
            match idx {
                0 => v.x,
                1 => v.y,
                2 => v.z,
                _ => v.w,
            }
        }
        _ => src.as_float(),
    };
    Value::Float(f)
}

fn param_float_value(param: &ParamRef, constants: &[Option<Value>]) -> Option<f32> {
    match param {
        ParamRef::Float(v) => Some(*v),
        ParamRef::Integer(v) => Some(*v as f32),
        ParamRef::Bool(v) => Some(if *v { 1.0 } else { 0.0 }),
        ParamRef::Local(idx) => {
            constants
                .get(*idx as usize)
                .and_then(|v| *v)
                .and_then(|v| match v {
                    Value::Float(f) => Some(f),
                    Value::Integer(i) => Some(i as f32),
                    Value::Bool(b) => Some(if b { 1.0 } else { 0.0 }),
                    _ => None,
                })
        }
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct OutputKey {
    node: FlatNodeId,
    output_index: u8,
}

struct Builder<'a> {
    graph: &'a FlatGraph,
    color_textures: &'a HashMap<Arc<str>, Arc<Texture>>,
    alpha_textures: &'a HashMap<Arc<str>, Arc<ScalarTexture>>,
    udim_textures: &'a HashMap<Arc<str>, Arc<UdimTiles>>,
    scalar_textures: &'a HashMap<Arc<str>, Arc<ScalarTexture>>,
    ocio: &'a OcioColorPipeline,
    color_processors: Vec<Arc<color::OcioColorProcessor>>,

    instructions: Vec<Instruction>,
    operand_pool: Vec<Operand>,
    value_pool: Vec<Value>,
    closure_nodes: Vec<ClosureNode>,

    register_for: HashMap<OutputKey, u32>,
    closure_for: HashMap<OutputKey, u32>,

    next_vreg: u32,
}

impl<'a> Builder<'a> {
    fn alloc_vreg(&mut self) -> u32 {
        let idx = self.next_vreg;
        self.next_vreg += 1;
        idx
    }

    fn intern_value(&mut self, v: Value) -> Operand {
        let idx = self.value_pool.len() as u32;
        self.value_pool.push(v);
        Operand::Const(idx)
    }

    fn push_operands(&mut self, ops: &[Operand]) -> u32 {
        let start = self.operand_pool.len() as u32;
        self.operand_pool.extend_from_slice(ops);
        start
    }

    fn param_to_operand(&mut self, p: &ParamRef) -> Operand {
        match p {
            ParamRef::Float(v) => self.intern_value(Value::Float(*v)),
            ParamRef::Integer(v) => self.intern_value(Value::Integer(*v)),
            ParamRef::Bool(v) => self.intern_value(Value::Bool(*v)),
            ParamRef::Color3(v) => self.intern_value(Value::Color3(*v)),
            ParamRef::Color4(v) => self.intern_value(Value::Color4(*v)),
            ParamRef::Vector2(v) => self.intern_value(Value::Vector2(*v)),
            ParamRef::Vector3(v) => self.intern_value(Value::Vector3(*v)),
            ParamRef::Vector4(v) => self.intern_value(Value::Vector4(*v)),
            ParamRef::Local(idx) => Operand::Reg(*idx as u16),
            ParamRef::Matrix33(m) => {
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::LoadMat3Const {
                    dst: dst as u16,
                    value: *m,
                });
                Operand::Reg(dst as u16)
            }
            ParamRef::Matrix44(m) => {
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::LoadMat4Const {
                    dst: dst as u16,
                    value: *m,
                });
                Operand::Reg(dst as u16)
            }
        }
    }

    fn emit_arith(&mut self, op: ArithOp, ty: ValueType, a: Operand, b: Operand) -> u32 {
        let dst = self.alloc_vreg();
        self.instructions.push(Instruction::Arith {
            dst: dst as u16,
            op,
            ty,
            a,
            b,
        });
        dst
    }

    fn emit_unary(&mut self, op: UnaryOp, ty: ValueType, src: Operand) -> u32 {
        let dst = self.alloc_vreg();
        self.instructions.push(Instruction::Unary {
            dst: dst as u16,
            op,
            ty,
            src,
        });
        dst
    }

    fn emit_convert(&mut self, from: ValueType, to: ValueType, src: Operand) -> u32 {
        let dst = self.alloc_vreg();
        self.instructions.push(Instruction::Convert {
            dst: dst as u16,
            from,
            to,
            src,
        });
        dst
    }

    fn emit_load_geom(&mut self, kind: GeometricKind) -> u32 {
        let dst = self.alloc_vreg();
        self.instructions.push(Instruction::LoadGeom {
            dst: dst as u16,
            kind,
        });
        dst
    }

    fn emit_load_const(&mut self, v: Value) -> u32 {
        let pool_idx = self.value_pool.len() as u32;
        self.value_pool.push(v);
        let dst = self.alloc_vreg();
        self.instructions.push(Instruction::LoadConst {
            dst: dst as u16,
            value_pool_idx: pool_idx,
        });
        dst
    }

    fn operand_to_vreg(&mut self, op: Operand) -> u32 {
        match op {
            Operand::Reg(r) => r as u32,
            Operand::Const(pool_idx) => {
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::LoadConst {
                    dst: dst as u16,
                    value_pool_idx: pool_idx,
                });
                dst
            }
        }
    }

    fn push_closure(&mut self, node: ClosureNode) -> u32 {
        let idx = self.closure_nodes.len() as u32;
        self.closure_nodes.push(node);
        idx
    }

    fn input_binding<'b>(node: &'b FlatNode, name: &str) -> Option<&'b FlatInput> {
        for input in &node.inputs {
            if input.name == name {
                return Some(&input.binding);
            }
        }
        None
    }

    fn input_declared_value_type(node: &FlatNode, name: &str, default: ValueType) -> ValueType {
        node.inputs
            .iter()
            .find(|input| input.name == name)
            .and_then(|input| ValueType::from_mtlx(&input.ty))
            .unwrap_or(default)
    }

    fn convert_node_supported(from: ValueType, to: ValueType) -> bool {
        use ValueType::*;
        if from == to {
            return true;
        }
        matches!(
            (from, to),
            (Boolean | Integer, Float)
                | (Boolean, Integer)
                | (Integer, Boolean)
                | (
                    Float | Integer | Boolean,
                    Color3 | Color4 | Vector2 | Vector3 | Vector4
                )
                | (
                    Color3 | Color4 | Vector2 | Vector3 | Vector4,
                    Color3 | Color4 | Vector2 | Vector3 | Vector4,
                )
        )
    }

    fn input_static_string<'b>(
        node: &'b FlatNode,
        category: &str,
        name: &str,
    ) -> Result<Option<&'b str>, CompileError> {
        let Some(binding) = Self::input_binding(node, name) else {
            return Ok(None);
        };
        match binding {
            FlatInput::Value(MtlxValue::String(s))
            | FlatInput::Value(MtlxValue::Filename(s))
            | FlatInput::String(s) => Ok(Some(s.as_str())),
            FlatInput::Value(other) => Err(CompileError::Unsupported(format!(
                "{}.{} must be a static string value, got {:?}",
                category, name, other
            ))),
            FlatInput::Node { .. } | FlatInput::GeomProp(_) | FlatInput::Empty => {
                Err(CompileError::Unsupported(format!(
                    "{}.{} must be a static string value",
                    category, name
                )))
            }
        }
    }

    fn input_static_integer(
        node: &FlatNode,
        category: &str,
        name: &str,
    ) -> Result<Option<i32>, CompileError> {
        let Some(binding) = Self::input_binding(node, name) else {
            return Ok(None);
        };
        match binding {
            FlatInput::Value(MtlxValue::Integer(v)) => Ok(Some(*v)),
            FlatInput::Value(other) => Err(CompileError::Unsupported(format!(
                "{}.{} must be a static integer enum value, got {:?}",
                category, name, other
            ))),
            FlatInput::String(s) => s.parse::<i32>().map(Some).map_err(|_| {
                CompileError::Unsupported(format!(
                    "{}.{} `{}`: must be an integer enum value",
                    category, name, s
                ))
            }),
            _ => Err(CompileError::Unsupported(format!(
                "{}.{} must be a static integer enum value",
                category, name
            ))),
        }
    }

    fn input_static_bool(
        node: &FlatNode,
        category: &str,
        name: &str,
    ) -> Result<Option<bool>, CompileError> {
        let Some(binding) = Self::input_binding(node, name) else {
            return Ok(None);
        };
        match binding {
            FlatInput::Value(MtlxValue::Boolean(v)) => Ok(Some(*v)),
            FlatInput::String(s) => match s.as_str() {
                "true" => Ok(Some(true)),
                "false" => Ok(Some(false)),
                _ => Err(CompileError::Unsupported(format!(
                    "{}.{} `{}`: must be a static boolean value",
                    category, name, s
                ))),
            },
            FlatInput::Value(other) => Err(CompileError::Unsupported(format!(
                "{}.{} must be a static boolean value, got {:?}",
                category, name, other
            ))),
            _ => Err(CompileError::Unsupported(format!(
                "{}.{} must be a static boolean value",
                category, name
            ))),
        }
    }

    fn input_worley_style(node: &FlatNode, category: &str) -> Result<WorleyStyle, CompileError> {
        let Some(binding) = Self::input_binding(node, "style") else {
            return Ok(WorleyStyle::Distance);
        };
        match binding {
            FlatInput::Value(MtlxValue::Integer(0)) => Ok(WorleyStyle::Distance),
            FlatInput::Value(MtlxValue::Integer(1)) => Ok(WorleyStyle::Solid),
            FlatInput::Value(MtlxValue::Integer(other)) => Err(CompileError::Unsupported(format!(
                "{}.style `{}`: must be 0/Distance or 1/Solid",
                category, other
            ))),
            FlatInput::Value(MtlxValue::String(s))
            | FlatInput::Value(MtlxValue::Filename(s))
            | FlatInput::String(s) => {
                if s.eq_ignore_ascii_case("distance") || s == "0" {
                    Ok(WorleyStyle::Distance)
                } else if s.eq_ignore_ascii_case("solid") || s == "1" {
                    Ok(WorleyStyle::Solid)
                } else {
                    Err(CompileError::Unsupported(format!(
                        "{}.style `{}`: must be 0/Distance or 1/Solid",
                        category, s
                    )))
                }
            }
            FlatInput::Value(other) => Err(CompileError::Unsupported(format!(
                "{}.style must be a static integer enum value, got {:?}",
                category, other
            ))),
            _ => Err(CompileError::Unsupported(format!(
                "{}.style must be a static integer enum value",
                category
            ))),
        }
    }

    fn image_color_space(node: &FlatNode, name: &str) -> TextureColorSpace {
        if !matches!(node.output_type, MtlxType::Color3 | MtlxType::Color4) {
            return TextureColorSpace::Linear;
        }
        for input in &node.inputs {
            if input.name == name {
                return match input.colorspace.as_deref() {
                    None => TextureColorSpace::DefaultColor,
                    Some("none") | Some("linear") | Some("scene_linear") => {
                        TextureColorSpace::Linear
                    }
                    Some(other) => {
                        TextureColorSpace::ocio(color::map_materialx_color_space(other).to_string())
                    }
                };
            }
        }
        TextureColorSpace::DefaultColor
    }

    fn color_xform_to_rendering(&mut self, src: &str) -> Result<ColorXform, CompileError> {
        let rendering_space = self.ocio.rendering_space().to_string();
        let src = materialx_color_space_to_ocio(src, &rendering_space);
        if src == rendering_space {
            return Ok(ColorXform::Identity);
        }
        let processor = self
            .ocio
            .color_space_processor(ColorSpaceRef::Ocio(&src), ColorSpaceRef::Rendering)
            .map_err(|error| CompileError::Unsupported(format!("{src}: {error}")))?;
        Ok(self.intern_color_processor(processor))
    }

    fn color_xform_between(&mut self, from: &str, to: &str) -> Result<ColorXform, CompileError> {
        let rendering_space = self.ocio.rendering_space().to_string();
        let from = materialx_color_space_to_ocio(from, &rendering_space);
        let to = materialx_color_space_to_ocio(to, &rendering_space);
        if from == to {
            return Ok(ColorXform::Identity);
        }
        let processor = self
            .ocio
            .color_space_processor(ColorSpaceRef::Ocio(&from), ColorSpaceRef::Ocio(&to))
            .map_err(|error| CompileError::Unsupported(format!("{from} -> {to}: {error}")))?;
        Ok(self.intern_color_processor(processor))
    }

    fn intern_color_processor(&mut self, processor: Arc<color::OcioColorProcessor>) -> ColorXform {
        let index = self.color_processors.len();
        self.color_processors.push(processor);
        ColorXform::Ocio {
            processor: index as u16,
        }
    }

    fn warn_animated_image_inputs(node: &FlatNode, category: &str) -> Result<(), CompileError> {
        if let Some(binding) = Self::input_binding(node, "framerange") {
            match binding {
                FlatInput::Value(MtlxValue::String(range)) | FlatInput::String(range) => {
                    if !range.is_empty() {
                        tracing::warn!(
                            "warning: {}.framerange is not implemented; animated image frame range is ignored",
                            category
                        );
                    }
                }
                FlatInput::Node { .. } | FlatInput::GeomProp(_) => {
                    tracing::warn!(
                        "warning: dynamic {}.framerange is not implemented; animated image frame range is ignored",
                        category
                    );
                }
                FlatInput::Empty => {}
                FlatInput::Value(other) => {
                    return Err(CompileError::Unsupported(format!(
                        "{}.framerange must be a string value, got {:?}",
                        category, other
                    )));
                }
            }
        }
        if let Some(binding) = Self::input_binding(node, "frameendaction") {
            match binding {
                FlatInput::Value(MtlxValue::String(action)) | FlatInput::String(action) => {
                    if !matches!(
                        action.as_str(),
                        "" | "constant" | "clamp" | "periodic" | "mirror"
                    ) {
                        return Err(CompileError::Unsupported(format!(
                            "{}.frameendaction `{}`",
                            category, action
                        )));
                    }
                    if !action.is_empty() && action != "constant" {
                        tracing::warn!(
                            "warning: {}.frameendaction=`{}` is not implemented; animated image frame range is ignored",
                            category,
                            action
                        );
                    }
                }
                FlatInput::Node { .. } | FlatInput::GeomProp(_) => {
                    tracing::warn!(
                        "warning: dynamic {}.frameendaction is not implemented; animated image frame range is ignored",
                        category
                    );
                }
                FlatInput::Empty => {}
                FlatInput::Value(other) => {
                    return Err(CompileError::Unsupported(format!(
                        "{}.frameendaction must be a string value, got {:?}",
                        category, other
                    )));
                }
            }
        }
        if let Some(binding) = Self::input_binding(node, "frameoffset") {
            match binding {
                FlatInput::Value(MtlxValue::Integer(0)) => {}
                FlatInput::Value(MtlxValue::Float(v)) if *v == 0.0 => {}
                FlatInput::Value(MtlxValue::Integer(_)) | FlatInput::Value(MtlxValue::Float(_)) => {
                    tracing::warn!(
                        "warning: {}.frameoffset is not implemented; animated image frame offset is ignored",
                        category
                    );
                }
                FlatInput::Node { .. } | FlatInput::GeomProp(_) => {
                    tracing::warn!(
                        "warning: dynamic {}.frameoffset is not implemented; animated image frame offset is ignored",
                        category
                    );
                }
                FlatInput::String(s) => match s.parse::<i32>() {
                    Ok(0) => {}
                    Ok(_) => tracing::warn!(
                        "warning: {}.frameoffset is not implemented; animated image frame offset is ignored",
                        category
                    ),
                    Err(_) => {
                        return Err(CompileError::Unsupported(format!(
                            "{}.frameoffset must be an integer value, got `{}`",
                            category, s
                        )));
                    }
                },
                FlatInput::Empty => {}
                FlatInput::Value(other) => {
                    return Err(CompileError::Unsupported(format!(
                        "{}.frameoffset must be an integer value, got {:?}",
                        category, other
                    )));
                }
            }
        }
        Ok(())
    }

    fn emit_logical_pattern(
        &mut self,
        node: &FlatNode,
        category: &str,
    ) -> Result<u32, CompileError> {
        if category == "not" {
            let a = self
                .input_value_param(node, "in", Some(ValueType::Boolean))?
                .unwrap_or(ParamRef::Bool(false));
            let a_op = self.param_to_operand(&a);
            let dst = self.alloc_vreg();
            self.instructions.push(Instruction::Logical {
                dst: dst as u16,
                op: LogicalOp::Not,
                a: a_op,
                b: Operand::Reg(0),
            });
            return Ok(dst);
        }

        let a = self
            .input_value_param(node, "in1", Some(ValueType::Boolean))?
            .unwrap_or(ParamRef::Bool(false));
        let b = self
            .input_value_param(node, "in2", Some(ValueType::Boolean))?
            .unwrap_or(ParamRef::Bool(false));
        let op = match category {
            "and" => LogicalOp::And,
            "or" => LogicalOp::Or,
            "xor" => LogicalOp::Xor,
            _ => unreachable!("logical category is checked before emit_logical_pattern"),
        };
        let a_op = self.param_to_operand(&a);
        let b_op = self.param_to_operand(&b);
        let dst = self.alloc_vreg();
        self.instructions.push(Instruction::Logical {
            dst: dst as u16,
            op,
            a: a_op,
            b: b_op,
        });
        Ok(dst)
    }

    fn emit_geompropvalue_pattern(
        &mut self,
        node: &FlatNode,
        category: &str,
        out_ty: ValueType,
    ) -> Result<u32, CompileError> {
        let geomprop = Self::input_static_string(node, category, "geomprop")?.unwrap_or("");
        let kind = match geomprop {
            "Pworld" => Some(GeometricKind::Position(GeomSpace::World)),
            "Pobject" => Some(GeometricKind::Position(GeomSpace::Object)),
            "Nworld" => Some(GeometricKind::Normal(GeomSpace::World)),
            "Nobject" => Some(GeometricKind::Normal(GeomSpace::Object)),
            "Tworld" => Some(GeometricKind::Tangent(GeomSpace::World)),
            "Tobject" => Some(GeometricKind::Tangent(GeomSpace::Object)),
            "Bworld" => Some(GeometricKind::Bitangent(GeomSpace::World)),
            "Bobject" => Some(GeometricKind::Bitangent(GeomSpace::Object)),
            "UV0" | "texcoord" => Some(GeometricKind::Texcoord),
            "geomcolor" => Some(GeometricKind::Geomcolor),
            _ => None,
        };
        if let Some(k) = kind {
            let actual = geometric_kind_value_type(&k);
            if actual != out_ty {
                return Err(CompileError::Unsupported(format!(
                    "{} `{}` has type {:?}, not {:?}",
                    category, geomprop, actual, out_ty
                )));
            }
            return Ok(self.ensure_geometric_kind_local(k));
        }

        let default = self
            .input_value_param(node, "default", Some(out_ty))?
            .unwrap_or(zero_param(Some(out_ty)));
        let op = self.param_to_operand(&default);
        Ok(self.operand_to_vreg(op))
    }

    fn input_value_param(
        &mut self,
        node: &FlatNode,
        name: &str,
        expected: Option<ValueType>,
    ) -> Result<Option<ParamRef>, CompileError> {
        match Self::input_binding(node, name) {
            Some(b) => self
                .compile_value_param(b, expected)
                .map(Some)
                .map_err(|err| input_error(name, err)),
            None => Ok(None),
        }
    }

    fn input_closure(
        &mut self,
        node: &FlatNode,
        name: &str,
        kind: ClosureKind,
    ) -> Result<u32, CompileError> {
        match Self::input_binding(node, name) {
            Some(b) => self.compile_closure_input(b, kind),
            None => Ok(0),
        }
    }

    /// Resolve a `ParamRef` for a parameter that flows into a closure node
    /// or a pattern operand. Inline values + downstream constant nodes are
    /// returned by-value so closure params and `push_param` can avoid the
    /// `PushConstant; StoreLocal; LoadLocal` round-trip — that triplet was the
    /// dominant cost (~77% of all bytecode instructions for textured
    /// MaterialX materials).
    fn compile_value_param(
        &mut self,
        binding: &FlatInput,
        expected: Option<ValueType>,
    ) -> Result<ParamRef, CompileError> {
        match binding {
            FlatInput::Empty => Ok(zero_param(expected)),
            FlatInput::Value(v) => constant_param(v),
            FlatInput::String(s) => Err(CompileError::Unsupported(format!(
                "string value `{}` cannot be used as numeric/vector/matrix parameter",
                s
            ))),
            FlatInput::GeomProp(prop) => {
                let kind = geometric_kind_from_prop(prop);
                if let FgKind::Geompropvalue(prop) = &kind {
                    return Err(CompileError::Unsupported(format!(
                        "custom defaultgeomprop `{}` is not supported; use an explicit geompropvalue default or a standard geometric property",
                        prop
                    )));
                }
                let idx = self.ensure_geometric_local(&kind);
                Ok(ParamRef::Local(idx))
            }
            FlatInput::Node { node, output } => {
                if let FlatNodeKind::Constant { value } = &self.graph.nodes[*node as usize].kind {
                    return constant_param(value);
                }
                let key = OutputKey {
                    node: *node,
                    output_index: output_index(output.as_deref()),
                };
                let idx = self.ensure_pattern_local(key, output.as_deref())?;
                Ok(ParamRef::Local(idx))
            }
        }
    }

    fn compile_closure_input(
        &mut self,
        binding: &FlatInput,
        kind: ClosureKind,
    ) -> Result<u32, CompileError> {
        match binding {
            FlatInput::Empty => Ok(0),
            FlatInput::Value(v) => Err(CompileError::Unsupported(format!(
                "literal {:?} cannot be used as a {:?} closure input",
                v, kind
            ))),
            FlatInput::String(s) => Err(CompileError::Unsupported(format!(
                "string `{}` cannot be used as a {:?} closure input",
                s, kind
            ))),
            FlatInput::GeomProp(prop) => Err(CompileError::Unsupported(format!(
                "geomprop `{}` cannot be used as a {:?} closure input",
                prop, kind
            ))),
            FlatInput::Node { node, output } => {
                let key = OutputKey {
                    node: *node,
                    output_index: output_index(output.as_deref()),
                };
                if let Some(idx) = self.closure_for.get(&key) {
                    return Ok(*idx);
                }
                let idx = self.compile_node_closure(*node, output.as_deref(), kind)?;
                self.closure_for.insert(key, idx);
                Ok(idx)
            }
        }
    }

    fn ensure_geometric_local(&mut self, fg: &FgKind) -> u32 {
        // Used only for defaultgeomprop synthesis where the binding is e.g.
        // texcoord/UV0 with no space distinction. P/N/T/B paths capture
        // the implied space at flatten time and never reach this helper.
        let kind = match fg {
            FgKind::Position => GeometricKind::Position(GeomSpace::Object),
            FgKind::Normal => GeometricKind::Normal(GeomSpace::Object),
            FgKind::Tangent => GeometricKind::Tangent(GeomSpace::Object),
            FgKind::Bitangent => GeometricKind::Bitangent(GeomSpace::Object),
            FgKind::Texcoord => GeometricKind::Texcoord,
            FgKind::Geomcolor => GeometricKind::Geomcolor,
            FgKind::ViewDirection => GeometricKind::ViewDirection(GeomSpace::World),
            FgKind::Geompropvalue(prop) => {
                unreachable!(
                    "custom geomprop `{}` must be handled before geometric lowering",
                    prop
                )
            }
        };
        self.ensure_geometric_kind_local(kind)
    }

    fn ensure_geometric_kind_local(&mut self, kind: GeometricKind) -> u32 {
        let kind_id = match kind {
            GeometricKind::Position(space) => 16 * geom_space_id(space),
            GeometricKind::Normal(space) => 1 + 16 * geom_space_id(space),
            GeometricKind::Tangent(space) => 2 + 16 * geom_space_id(space),
            GeometricKind::Bitangent(space) => 3 + 16 * geom_space_id(space),
            GeometricKind::Texcoord => 4,
            GeometricKind::Geomcolor => 5,
            GeometricKind::Frame => 6,
            GeometricKind::Time => 7,
            GeometricKind::ViewDirection(space) => 8 + 16 * geom_space_id(space),
        };
        let key = OutputKey {
            node: u32::MAX - kind_id,
            output_index: 0,
        };
        if let Some(&idx) = self.register_for.get(&key) {
            return idx;
        }
        let idx = self.emit_load_geom(kind);
        self.register_for.insert(key, idx);
        idx
    }

    fn ensure_pattern_local(
        &mut self,
        key: OutputKey,
        output: Option<&str>,
    ) -> Result<u32, CompileError> {
        if let Some(&idx) = self.register_for.get(&key) {
            return Ok(idx);
        }

        let node_id = key.node;
        let node_ptr: *const FlatNode = &self.graph.nodes[node_id as usize];
        let node = unsafe { &*node_ptr };

        let idx = match &node.kind {
            FlatNodeKind::Constant { value } => match value {
                MtlxValue::Matrix33(m) => {
                    let dst = self.alloc_vreg();
                    self.instructions.push(Instruction::LoadMat3Const {
                        dst: dst as u16,
                        value: *m,
                    });
                    dst
                }
                MtlxValue::Matrix44(m) => {
                    let dst = self.alloc_vreg();
                    self.instructions.push(Instruction::LoadMat4Const {
                        dst: dst as u16,
                        value: *m,
                    });
                    dst
                }
                _ => self.emit_load_const(constant_value(value)?),
            },
            FlatNodeKind::Geometric { kind, .. } => {
                let default_space = match kind {
                    FgKind::ViewDirection => "world",
                    _ => "object",
                };
                let space = Self::input_static_string(node, "geometric", "space")?;
                let gs = parse_geom_space(space.or(Some(default_space)))?;
                let gk = match kind {
                    FgKind::Position => GeometricKind::Position(gs),
                    FgKind::Normal => GeometricKind::Normal(gs),
                    FgKind::Tangent => GeometricKind::Tangent(gs),
                    FgKind::Bitangent => GeometricKind::Bitangent(gs),
                    FgKind::Texcoord => GeometricKind::Texcoord,
                    FgKind::Geomcolor => GeometricKind::Geomcolor,
                    FgKind::ViewDirection => GeometricKind::ViewDirection(gs),
                    FgKind::Geompropvalue(prop) => {
                        return Err(CompileError::Unsupported(format!(
                            "custom defaultgeomprop `{}` is not supported; use an explicit geompropvalue default or a standard geometric property",
                            prop
                        )));
                    }
                };
                self.ensure_geometric_kind_local(gk)
            }
            FlatNodeKind::Pattern { category } => {
                let category = category.clone();
                self.emit_pattern(node, &category, output)?
            }
            FlatNodeKind::Combinator { category } => {
                let category = category.clone();
                self.emit_pattern(node, &category, output)?
            }
            FlatNodeKind::Surface | FlatNodeKind::SurfaceUnlit => {
                return Err(CompileError::Unsupported("surface used as value".into()));
            }
            FlatNodeKind::SurfaceMaterial => {
                return Err(CompileError::Unsupported(
                    "surfacematerial used as value".into(),
                ));
            }
            FlatNodeKind::Shading { .. } | FlatNodeKind::Displacement | FlatNodeKind::Light => {
                return Err(CompileError::Unsupported(format!(
                    "closure-type node used as value: {:?}",
                    node.kind
                )));
            }
        };
        self.register_for.insert(key, idx);
        Ok(idx)
    }

    fn compile_node_closure(
        &mut self,
        id: FlatNodeId,
        _output: Option<&str>,
        kind: ClosureKind,
    ) -> Result<u32, CompileError> {
        let node_ptr: *const FlatNode = &self.graph.nodes[id as usize];
        let node = unsafe { &*node_ptr };
        match &node.kind {
            FlatNodeKind::Surface => self.compile_surface(node),
            FlatNodeKind::SurfaceUnlit => self.compile_surface_unlit(node),
            FlatNodeKind::Shading { category } => {
                let cat = category.clone();
                self.compile_shading_leaf(node, &cat)
            }
            FlatNodeKind::Combinator { category } => {
                let cat = category.clone();
                self.compile_combinator(node, &cat, kind)
            }
            FlatNodeKind::Pattern { category } => {
                let cat = category.clone();
                self.compile_pattern_closure(node, &cat, kind)
            }
            FlatNodeKind::Constant { .. }
            | FlatNodeKind::Geometric { .. }
            | FlatNodeKind::SurfaceMaterial => Err(CompileError::Unsupported(format!(
                "node kind {:?} cannot be used as a {:?} closure",
                node.kind, kind
            ))),
            FlatNodeKind::Displacement => {
                tracing::warn!(
                    "warning: displacement node is not supported; ignoring displacement"
                );
                Ok(0)
            }
            FlatNodeKind::Light => Err(CompileError::Unsupported(
                "light nodes are not supported in MaterialX surface materials".into(),
            )),
        }
    }

    fn compile_surface(&mut self, node: &FlatNode) -> Result<u32, CompileError> {
        let is_unlit = Self::input_binding(node, "emission").is_some()
            || Self::input_binding(node, "emission_color").is_some()
            || Self::input_binding(node, "transmission").is_some()
            || Self::input_binding(node, "transmission_color").is_some();
        if is_unlit {
            return self.compile_surface_unlit(node);
        }
        let bsdf = self.input_closure(node, "bsdf", ClosureKind::Bsdf)?;
        let edf = self.input_closure(node, "edf", ClosureKind::Edf)?;
        let opacity = self
            .input_value_param(node, "opacity", Some(ValueType::Float))?
            .unwrap_or(ParamRef::Float(1.0));
        let thin_walled =
            match self.input_value_param(node, "thin_walled", Some(ValueType::Boolean))? {
                Some(ParamRef::Bool(v)) => v,
                Some(other) => {
                    return Err(CompileError::Unsupported(format!(
                        "surface.thin_walled must be boolean, got {:?}",
                        other
                    )));
                }
                None => false,
            };
        Ok(self.push_closure(ClosureNode::Surface {
            bsdf,
            edf,
            opacity,
            thin_walled,
        }))
    }

    fn compile_surface_unlit(&mut self, node: &FlatNode) -> Result<u32, CompileError> {
        let emission = self
            .input_value_param(node, "emission", Some(ValueType::Float))?
            .unwrap_or(ParamRef::Float(1.0));
        let emission_color = self
            .input_value_param(node, "emission_color", Some(ValueType::Color3))?
            .unwrap_or(ParamRef::Color3(Vec3::ONE));
        let transmission = self
            .input_value_param(node, "transmission", Some(ValueType::Float))?
            .unwrap_or(ParamRef::Float(0.0));
        let transmission_color = self
            .input_value_param(node, "transmission_color", Some(ValueType::Color3))?
            .unwrap_or(ParamRef::Color3(Vec3::ONE));
        let opacity = self
            .input_value_param(node, "opacity", Some(ValueType::Float))?
            .unwrap_or(ParamRef::Float(1.0));

        // OSL/MDL mx_surface_unlit attenuates emission by (1 - trans) so that
        // setting transmission to 1 yields a fully translucent thin sheet with
        // no emission contribution.
        let trans_sat_local = self.b_clamp(
            ValueType::Float,
            &transmission,
            &ParamRef::Float(0.0),
            &ParamRef::Float(1.0),
        );
        let trans_sat = ParamRef::Local(trans_sat_local);

        let one_op = self.param_to_operand(&ParamRef::Float(1.0));
        let trans_op = self.param_to_operand(&trans_sat);
        let one_minus_trans_local =
            self.emit_arith(ArithOp::Subtract, ValueType::Float, one_op, trans_op);

        let ec_op = self.param_to_operand(&emission_color);
        let e_op = self.param_to_operand(&emission);
        let ec_e = self.emit_arith(ArithOp::Multiply, ValueType::Color3, ec_op, e_op);
        let omt = self.param_to_operand(&ParamRef::Local(one_minus_trans_local));
        let edf_color_local = self.emit_arith(
            ArithOp::Multiply,
            ValueType::Color3,
            Operand::Reg(ec_e as u16),
            omt,
        );
        let edf_idx = self.push_closure(ClosureNode::UniformEdf {
            color: ParamRef::Local(edf_color_local),
        });

        let tc_op = self.param_to_operand(&transmission_color);
        let trans2_op = self.param_to_operand(&trans_sat);
        let bsdf_color_local =
            self.emit_arith(ArithOp::Multiply, ValueType::Color3, tc_op, trans2_op);
        let bsdf_idx = self.push_closure(ClosureNode::Translucent {
            weight: ParamRef::Float(1.0),
            color: ParamRef::Local(bsdf_color_local),
            normal: None,
        });

        Ok(self.push_closure(ClosureNode::Surface {
            bsdf: bsdf_idx,
            edf: edf_idx,
            opacity,
            thin_walled: true,
        }))
    }

    fn read_optional_geom_vec3(
        &mut self,
        node: &FlatNode,
        name: &str,
    ) -> Result<Option<ParamRef>, CompileError> {
        self.input_value_param(node, name, Some(ValueType::Vector3))
    }

    fn compile_shading_leaf(
        &mut self,
        node: &FlatNode,
        category: &str,
    ) -> Result<u32, CompileError> {
        let p_float = |p: Option<ParamRef>, default: f32| p.unwrap_or(ParamRef::Float(default));
        let p_color = |p: Option<ParamRef>, default: Vec3| p.unwrap_or(ParamRef::Color3(default));
        let p_vec2 = |p: Option<ParamRef>, default: Vec2| p.unwrap_or(ParamRef::Vector2(default));

        match category {
            "oren_nayar_diffuse_bsdf" => {
                let weight = p_float(
                    self.input_value_param(node, "weight", Some(ValueType::Float))?,
                    1.0,
                );
                let color = p_color(
                    self.input_value_param(node, "color", Some(ValueType::Color3))?,
                    Vec3::splat(0.18),
                );
                let roughness = p_float(
                    self.input_value_param(node, "roughness", Some(ValueType::Float))?,
                    0.0,
                );
                let energy_compensation =
                    Self::input_static_bool(node, category, "energy_compensation")?
                        .unwrap_or(false);
                let normal = self.read_optional_geom_vec3(node, "normal")?;
                Ok(self.push_closure(ClosureNode::OrenNayarDiffuse {
                    weight,
                    color,
                    roughness,
                    energy_compensation,
                    normal,
                }))
            }
            "burley_diffuse_bsdf" => {
                let weight = p_float(
                    self.input_value_param(node, "weight", Some(ValueType::Float))?,
                    1.0,
                );
                let color = p_color(
                    self.input_value_param(node, "color", Some(ValueType::Color3))?,
                    Vec3::splat(0.18),
                );
                let roughness = p_float(
                    self.input_value_param(node, "roughness", Some(ValueType::Float))?,
                    0.0,
                );
                let normal = self.read_optional_geom_vec3(node, "normal")?;
                Ok(self.push_closure(ClosureNode::BurleyDiffuse {
                    weight,
                    color,
                    roughness,
                    normal,
                }))
            }
            "translucent_bsdf" => {
                let weight = p_float(
                    self.input_value_param(node, "weight", Some(ValueType::Float))?,
                    1.0,
                );
                let color = p_color(
                    self.input_value_param(node, "color", Some(ValueType::Color3))?,
                    Vec3::ONE,
                );
                let normal = self.read_optional_geom_vec3(node, "normal")?;
                Ok(self.push_closure(ClosureNode::Translucent { weight, color, normal }))
            }
            "dielectric_bsdf" => {
                if let Some(s) = Self::input_static_string(node, category, "distribution")?
                    && !s.eq_ignore_ascii_case("ggx")
                {
                    return Err(CompileError::Unsupported(format!(
                        "dielectric_bsdf.distribution `{}`: only `ggx` is supported in MaterialX 1.39",
                        s
                    )));
                }
                let weight = p_float(
                    self.input_value_param(node, "weight", Some(ValueType::Float))?,
                    1.0,
                );
                let tint = p_color(
                    self.input_value_param(node, "tint", Some(ValueType::Color3))?,
                    Vec3::ONE,
                );
                let ior = p_float(
                    self.input_value_param(node, "ior", Some(ValueType::Float))?,
                    1.5,
                );
                let roughness = p_vec2(
                    self.input_value_param(node, "roughness", Some(ValueType::Vector2))?,
                    Vec2::splat(0.05),
                );
                let retroreflective =
                    Self::input_static_bool(node, category, "retroreflective")?.unwrap_or(false);
                let scatter_mode = match Self::input_static_string(node, category, "scatter_mode")? {
                    None | Some("R") => ScatterMode::Reflection,
                    Some("T") => ScatterMode::Transmission,
                    Some("RT") => ScatterMode::Both,
                    Some(other) => {
                        return Err(CompileError::Unsupported(format!(
                            "dielectric_bsdf.scatter_mode `{}`: must be one of `R`, `T`, `RT`",
                            other
                        )));
                    }
                };
                let thinfilm_thickness = p_float(
                    self.input_value_param(node, "thinfilm_thickness", Some(ValueType::Float))?,
                    0.0,
                );
                let thinfilm_ior = p_float(
                    self.input_value_param(node, "thinfilm_ior", Some(ValueType::Float))?,
                    1.5,
                );
                let normal = self.read_optional_geom_vec3(node, "normal")?;
                let tangent = self.read_optional_geom_vec3(node, "tangent")?;
                Ok(self.push_closure(ClosureNode::Dielectric {
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
                }))
            }
            "conductor_bsdf" => {
                if let Some(s) = Self::input_static_string(node, category, "distribution")?
                    && !s.eq_ignore_ascii_case("ggx")
                {
                    return Err(CompileError::Unsupported(format!(
                        "conductor_bsdf.distribution `{}`: only `ggx` is supported in MaterialX 1.39",
                        s
                    )));
                }
                let weight = p_float(
                    self.input_value_param(node, "weight", Some(ValueType::Float))?,
                    1.0,
                );
                let ior = p_color(
                    self.input_value_param(node, "ior", Some(ValueType::Color3))?,
                    Vec3::new(0.183, 0.421, 1.373),
                );
                let extinction = p_color(
                    self.input_value_param(node, "extinction", Some(ValueType::Color3))?,
                    Vec3::new(3.424, 2.346, 1.770),
                );
                let roughness = p_vec2(
                    self.input_value_param(node, "roughness", Some(ValueType::Vector2))?,
                    Vec2::splat(0.05),
                );
                let retroreflective =
                    Self::input_static_bool(node, category, "retroreflective")?.unwrap_or(false);
                let thinfilm_thickness = p_float(
                    self.input_value_param(node, "thinfilm_thickness", Some(ValueType::Float))?,
                    0.0,
                );
                let thinfilm_ior = p_float(
                    self.input_value_param(node, "thinfilm_ior", Some(ValueType::Float))?,
                    1.5,
                );
                let normal = self.read_optional_geom_vec3(node, "normal")?;
                let tangent = self.read_optional_geom_vec3(node, "tangent")?;
                Ok(self.push_closure(ClosureNode::Conductor {
                    weight,
                    ior,
                    extinction,
                    roughness,
                    retroreflective,
                    thinfilm_thickness,
                    thinfilm_ior,
                    normal,
                    tangent,
                }))
            }
            "generalized_schlick_bsdf" => {
                if let Some(s) = Self::input_static_string(node, category, "distribution")?
                    && !s.eq_ignore_ascii_case("ggx")
                {
                    return Err(CompileError::Unsupported(format!(
                        "generalized_schlick_bsdf.distribution `{}`: only `ggx` is supported in MaterialX 1.39",
                        s
                    )));
                }
                let weight = p_float(
                    self.input_value_param(node, "weight", Some(ValueType::Float))?,
                    1.0,
                );
                let color0 = p_color(
                    self.input_value_param(node, "color0", Some(ValueType::Color3))?,
                    Vec3::ONE,
                );
                let color82 = p_color(
                    self.input_value_param(node, "color82", Some(ValueType::Color3))?,
                    Vec3::ONE,
                );
                let color90 = p_color(
                    self.input_value_param(node, "color90", Some(ValueType::Color3))?,
                    Vec3::ONE,
                );
                let exponent = p_float(
                    self.input_value_param(node, "exponent", Some(ValueType::Float))?,
                    5.0,
                );
                let roughness = p_vec2(
                    self.input_value_param(node, "roughness", Some(ValueType::Vector2))?,
                    Vec2::splat(0.05),
                );
                let retroreflective =
                    Self::input_static_bool(node, category, "retroreflective")?.unwrap_or(false);
                let scatter_mode = match Self::input_static_string(node, category, "scatter_mode")? {
                    None | Some("R") => ScatterMode::Reflection,
                    Some("T") => ScatterMode::Transmission,
                    Some("RT") => ScatterMode::Both,
                    Some(other) => {
                        return Err(CompileError::Unsupported(format!(
                            "generalized_schlick_bsdf.scatter_mode `{}`: must be one of `R`, `T`, `RT`",
                            other
                        )));
                    }
                };
                let thinfilm_thickness = p_float(
                    self.input_value_param(node, "thinfilm_thickness", Some(ValueType::Float))?,
                    0.0,
                );
                let thinfilm_ior = p_float(
                    self.input_value_param(node, "thinfilm_ior", Some(ValueType::Float))?,
                    1.5,
                );
                let normal = self.read_optional_geom_vec3(node, "normal")?;
                let tangent = self.read_optional_geom_vec3(node, "tangent")?;
                Ok(self.push_closure(ClosureNode::GeneralizedSchlick {
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
                }))
            }
            "sheen_bsdf" => {
                let weight = p_float(
                    self.input_value_param(node, "weight", Some(ValueType::Float))?,
                    1.0,
                );
                let color = p_color(
                    self.input_value_param(node, "color", Some(ValueType::Color3))?,
                    Vec3::ONE,
                );
                let roughness = p_float(
                    self.input_value_param(node, "roughness", Some(ValueType::Float))?,
                    0.3,
                );
                let mode = match Self::input_static_string(node, category, "mode")? {
                    None => SheenMode::ContyKulla,
                    Some(s) if s.eq_ignore_ascii_case("conty_kulla") => SheenMode::ContyKulla,
                    Some(s) if s.eq_ignore_ascii_case("zeltner") => SheenMode::Zeltner,
                    Some(other) => {
                        return Err(CompileError::Unsupported(format!(
                            "sheen_bsdf.mode `{}`: must be `conty_kulla` or `zeltner`",
                            other
                        )));
                    }
                };
                let normal = self.read_optional_geom_vec3(node, "normal")?;
                Ok(self.push_closure(ClosureNode::Sheen {
                    weight,
                    color,
                    roughness,
                    mode,
                    normal,
                }))
            }
            "subsurface_bsdf" => {
                tracing::warn!(
                    "warning: subsurface_bsdf is not fully supported; falling back to burley_diffuse_bsdf with roughness=0.5 (radius/anisotropy ignored)"
                );
                let weight = p_float(
                    self.input_value_param(node, "weight", Some(ValueType::Float))?,
                    1.0,
                );
                let color = p_color(
                    self.input_value_param(node, "color", Some(ValueType::Color3))?,
                    Vec3::splat(0.18),
                );
                let _radius = p_color(
                    self.input_value_param(node, "radius", Some(ValueType::Color3))?,
                    Vec3::ONE,
                );
                let _anisotropy = p_float(
                    self.input_value_param(node, "anisotropy", Some(ValueType::Float))?,
                    0.0,
                );
                let normal = self.read_optional_geom_vec3(node, "normal")?;
                Ok(self.push_closure(ClosureNode::BurleyDiffuse {
                    weight,
                    color,
                    roughness: ParamRef::Float(0.5),
                    normal,
                }))
            }
            "thin_film_bsdf" => Err(CompileError::Unsupported(
                "thin_film_bsdf was removed in MaterialX 1.39; use the `thinfilm_thickness` and `thinfilm_ior` inputs on dielectric_bsdf / conductor_bsdf / generalized_schlick_bsdf instead".into(),
            )),
            "chiang_hair_bsdf" => {
                let tint_r = p_color(
                    self.input_value_param(node, "tint_R", Some(ValueType::Color3))?,
                    Vec3::ONE,
                );
                let tint_tt = p_color(
                    self.input_value_param(node, "tint_TT", Some(ValueType::Color3))?,
                    Vec3::ONE,
                );
                let tint_trt = p_color(
                    self.input_value_param(node, "tint_TRT", Some(ValueType::Color3))?,
                    Vec3::ONE,
                );
                let absorption = p_color(
                    self.input_value_param(
                        node,
                        "absorption_coefficient",
                        Some(ValueType::Vector3),
                    )?,
                    Vec3::ZERO,
                );
                let ior = p_float(
                    self.input_value_param(node, "ior", Some(ValueType::Float))?,
                    1.55,
                );
                let roughness_r = p_vec2(
                    self.input_value_param(node, "roughness_R", Some(ValueType::Vector2))?,
                    Vec2::splat(0.1),
                );
                let roughness_tt = p_vec2(
                    self.input_value_param(node, "roughness_TT", Some(ValueType::Vector2))?,
                    Vec2::splat(0.05),
                );
                let roughness_trt = p_vec2(
                    self.input_value_param(node, "roughness_TRT", Some(ValueType::Vector2))?,
                    Vec2::splat(0.2),
                );
                let cuticle_angle = p_float(
                    self.input_value_param(node, "cuticle_angle", Some(ValueType::Float))?,
                    0.5,
                );
                let normal = self.read_optional_geom_vec3(node, "normal")?;
                let curve_direction = self
                    .input_value_param(node, "curve_direction", Some(ValueType::Vector3))?
                    .unwrap_or_else(|| {
                        let idx =
                            self.ensure_geometric_kind_local(GeometricKind::Tangent(GeomSpace::World));
                        ParamRef::Local(idx)
                    });
                Ok(self.push_closure(ClosureNode::ChiangHair {
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
                }))
            }
            "uniform_edf" => {
                let color = p_color(
                    self.input_value_param(node, "color", Some(ValueType::Color3))?,
                    Vec3::ONE,
                );
                Ok(self.push_closure(ClosureNode::UniformEdf { color }))
            }
            "conical_edf" => {
                let color = p_color(
                    self.input_value_param(node, "color", Some(ValueType::Color3))?,
                    Vec3::ONE,
                );
                let inner_angle = p_float(
                    self.input_value_param(node, "inner_angle", Some(ValueType::Float))?,
                    60.0,
                );
                let outer_angle = p_float(
                    self.input_value_param(node, "outer_angle", Some(ValueType::Float))?,
                    0.0,
                );
                let normal = self.read_optional_geom_vec3(node, "normal")?;
                Ok(self.push_closure(ClosureNode::ConicalEdf {
                    color,
                    inner_angle,
                    outer_angle,
                    normal,
                }))
            }
            "measured_edf" => {
                tracing::warn!(
                    "warning: measured_edf (IES profiles) is not supported; falling back to uniform_edf with the same color"
                );
                let color = p_color(
                    self.input_value_param(node, "color", Some(ValueType::Color3))?,
                    Vec3::ONE,
                );
                let _normal = self.read_optional_geom_vec3(node, "normal")?;
                if let Some(binding) = Self::input_binding(node, "file") {
                    match binding {
                        FlatInput::Value(MtlxValue::Filename(_))
                        | FlatInput::Value(MtlxValue::String(_))
                        | FlatInput::String(_) => {}
                        other => {
                            return Err(CompileError::Unsupported(format!(
                                "measured_edf.file must be a filename, got {:?}",
                                other
                            )));
                        }
                    }
                }
                Ok(self.push_closure(ClosureNode::UniformEdf { color }))
            }
            "generalized_schlick_edf" => {
                let base = self.input_closure(node, "base", ClosureKind::Edf)?;
                let color0 = p_color(
                    self.input_value_param(node, "color0", Some(ValueType::Color3))?,
                    Vec3::ONE,
                );
                let color90 = p_color(
                    self.input_value_param(node, "color90", Some(ValueType::Color3))?,
                    Vec3::ONE,
                );
                let exponent = p_float(
                    self.input_value_param(node, "exponent", Some(ValueType::Float))?,
                    5.0,
                );
                Ok(self.push_closure(ClosureNode::GeneralizedSchlickEdf {
                    base,
                    color0,
                    color90,
                    exponent,
                }))
            }
            "absorption_vdf" => {
                let _absorption = self.input_value_param(
                    node,
                    "absorption",
                    Some(ValueType::Vector3),
                )?;
                tracing::warn!(
                    "warning: VDF node `{}` is not supported; treating as zero (no volume absorption/scattering)",
                    category
                );
                Ok(0)
            }
            "anisotropic_vdf" => {
                let _absorption = self.input_value_param(
                    node,
                    "absorption",
                    Some(ValueType::Vector3),
                )?;
                let _scattering = self.input_value_param(
                    node,
                    "scattering",
                    Some(ValueType::Vector3),
                )?;
                let _anisotropy =
                    self.input_value_param(node, "anisotropy", Some(ValueType::Float))?;
                tracing::warn!(
                    "warning: VDF node `{}` is not supported; treating as zero (no volume absorption/scattering)",
                    category
                );
                Ok(0)
            }
            "gooch_shade" => {
                let warm = p_color(
                    self.input_value_param(node, "warm_color", Some(ValueType::Color3))?,
                    Vec3::new(0.8, 0.8, 0.7),
                );
                let cool = p_color(
                    self.input_value_param(node, "cool_color", Some(ValueType::Color3))?,
                    Vec3::new(0.3, 0.3, 0.8),
                );
                let specular_intensity = p_float(
                    self.input_value_param(node, "specular_intensity", Some(ValueType::Float))?,
                    1.0,
                );
                let shininess = p_float(
                    self.input_value_param(node, "shininess", Some(ValueType::Float))?,
                    64.0,
                );
                let light_direction = p_color(
                    self.input_value_param(node, "light_direction", Some(ValueType::Vector3))?,
                    Vec3::new(1.0, -0.5, -0.5),
                );
                Ok(self.push_closure(ClosureNode::GoochShade {
                    warm,
                    cool,
                    specular_intensity,
                    shininess,
                    light_direction,
                }))
            }
            other => Err(CompileError::Unsupported(format!("shading `{}`", other))),
        }
    }

    fn compile_combinator(
        &mut self,
        node: &FlatNode,
        category: &str,
        kind: ClosureKind,
    ) -> Result<u32, CompileError> {
        match category {
            "mix" => {
                let bg = self.input_closure(node, "bg", kind)?;
                let fg = self.input_closure(node, "fg", kind)?;
                let mix = self
                    .input_value_param(node, "mix", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(0.0));
                Ok(self.push_closure(ClosureNode::Mix { bg, fg, mix, kind }))
            }
            "layer" => {
                let top = self.input_closure(node, "top", ClosureKind::Bsdf)?;
                let base = self.input_closure(node, "base", ClosureKind::Bsdf)?;
                Ok(self.push_closure(ClosureNode::Layer { top, base }))
            }
            "add" => {
                let a = self.input_closure(node, "in1", kind)?;
                let b = self.input_closure(node, "in2", kind)?;
                Ok(self.push_closure(ClosureNode::Add { a, b, kind }))
            }
            "multiply" => {
                let inner = self.input_closure(node, "in1", kind)?;
                let scale = self
                    .input_value_param(node, "in2", Some(ValueType::Color3))?
                    .unwrap_or(ParamRef::Color3(Vec3::ONE));
                Ok(self.push_closure(ClosureNode::Multiply { inner, scale, kind }))
            }
            other => Err(CompileError::Unsupported(format!("combinator `{}`", other))),
        }
    }

    fn compile_pattern_closure(
        &mut self,
        node: &FlatNode,
        category: &str,
        kind: ClosureKind,
    ) -> Result<u32, CompileError> {
        match category {
            "ifgreater" | "ifgreatereq" | "ifequal" => {
                let (v1_default, v2_default) = match category {
                    "ifequal" => (0.0_f32, 0.0_f32),
                    _ => (1.0, 0.0),
                };
                let value1 = self
                    .input_value_param(node, "value1", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(v1_default));
                let value2 = self
                    .input_value_param(node, "value2", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(v2_default));
                let then_branch = self.input_closure(node, "in1", kind)?;
                let else_branch = self.input_closure(node, "in2", kind)?;
                let cn = match category {
                    "ifgreater" => ClosureNode::IfGreater {
                        value1,
                        value2,
                        then_branch,
                        else_branch,
                        kind,
                    },
                    "ifgreatereq" => ClosureNode::IfGreaterEq {
                        value1,
                        value2,
                        then_branch,
                        else_branch,
                        kind,
                    },
                    _ => ClosureNode::IfEqual {
                        value1,
                        value2,
                        then_branch,
                        else_branch,
                        kind,
                    },
                };
                Ok(self.push_closure(cn))
            }
            "switch" => {
                let which = self
                    .input_value_param(node, "which", Some(ValueType::Integer))?
                    .unwrap_or(ParamRef::Integer(0));
                let mut branches = [0u32; 10];
                for (i, branch) in branches.iter_mut().enumerate() {
                    let name = format!("in{}", i + 1);
                    *branch = self.input_closure(node, &name, kind)?;
                }
                Ok(self.push_closure(ClosureNode::Switch {
                    which,
                    branches,
                    kind,
                }))
            }
            "convert" | "dot" => {
                if let Some(input) = node.inputs.iter().find(|i| i.name == "in") {
                    self.compile_closure_input(&input.binding, kind)
                } else {
                    Ok(0)
                }
            }
            other => Err(CompileError::Unsupported(format!(
                "closure leaf node category `{}` (kind {:?})",
                other, kind
            ))),
        }
    }

    fn b_arith(&mut self, op: ArithOp, ty: ValueType, a: &ParamRef, b: &ParamRef) -> u32 {
        let a_op = self.param_to_operand(a);
        let b_op = self.param_to_operand(b);
        self.emit_arith(op, ty, a_op, b_op)
    }

    fn b_unary(&mut self, op: UnaryOp, ty: ValueType, src: &ParamRef) -> u32 {
        let s = self.param_to_operand(src);
        self.emit_unary(op, ty, s)
    }

    fn b_convert(&mut self, from: ValueType, to: ValueType, src: &ParamRef) -> u32 {
        let s = self.param_to_operand(src);
        self.emit_convert(from, to, s)
    }

    fn b_mix(&mut self, ty: ValueType, bg: &ParamRef, fg: &ParamRef, mix: &ParamRef) -> u32 {
        let bg_op = self.param_to_operand(bg);
        let fg_op = self.param_to_operand(fg);
        let mix_op = self.param_to_operand(mix);
        let dst = self.alloc_vreg();
        self.instructions.push(Instruction::MixValue {
            dst: dst as u16,
            ty,
            bg: bg_op,
            fg: fg_op,
            mix: mix_op,
        });
        dst
    }

    fn b_clamp(&mut self, ty: ValueType, v: &ParamRef, lo: &ParamRef, hi: &ParamRef) -> u32 {
        let v_op = self.param_to_operand(v);
        let lo_op = self.param_to_operand(lo);
        let hi_op = self.param_to_operand(hi);
        let dst = self.alloc_vreg();
        self.instructions.push(Instruction::Clamp {
            dst: dst as u16,
            ty,
            v: v_op,
            lo: lo_op,
            hi: hi_op,
        });
        dst
    }

    fn emit_creatematrix_pattern(
        &mut self,
        node: &FlatNode,
        out_ty: ValueType,
    ) -> Result<u32, CompileError> {
        let dim4 = matches!(out_ty, ValueType::Matrix44);
        let rows = if dim4 { 4 } else { 3 };
        let vec3_rows = dim4
            && matches!(
                Self::input_declared_value_type(node, "in1", ValueType::Vector4),
                ValueType::Vector3
            );
        let row_ty = if !dim4 || vec3_rows {
            ValueType::Vector3
        } else {
            ValueType::Vector4
        };
        let mut ops = Vec::with_capacity(rows);
        for i in 0..rows {
            let name = format!("in{}", i + 1);
            let p = self
                .input_value_param(node, &name, Some(row_ty))?
                .unwrap_or_else(|| match (row_ty, i) {
                    (ValueType::Vector4, 0) => ParamRef::Vector4(Vec4::new(1.0, 0.0, 0.0, 0.0)),
                    (ValueType::Vector4, 1) => ParamRef::Vector4(Vec4::new(0.0, 1.0, 0.0, 0.0)),
                    (ValueType::Vector4, 2) => ParamRef::Vector4(Vec4::new(0.0, 0.0, 1.0, 0.0)),
                    (ValueType::Vector4, _) => ParamRef::Vector4(Vec4::new(0.0, 0.0, 0.0, 1.0)),
                    (_, 0) => ParamRef::Vector3(Vec3::X),
                    (_, 1) => ParamRef::Vector3(Vec3::Y),
                    (_, 2) => ParamRef::Vector3(Vec3::Z),
                    _ => ParamRef::Vector3(Vec3::ZERO),
                });
            ops.push(self.param_to_operand(&p));
        }
        let rows_start = self.push_operands(&ops);
        let dst = self.alloc_vreg();
        let instr = if !dim4 {
            Instruction::CreateMatrix3 {
                dst: dst as u16,
                rows_start,
            }
        } else if vec3_rows {
            Instruction::CreateMatrix4FromVec3 {
                dst: dst as u16,
                rows_start,
            }
        } else {
            Instruction::CreateMatrix4 {
                dst: dst as u16,
                rows_start,
            }
        };
        self.instructions.push(instr);
        Ok(dst)
    }

    fn emit_pattern(
        &mut self,
        node: &FlatNode,
        category: &str,
        _output: Option<&str>,
    ) -> Result<u32, CompileError> {
        let out_ty = ValueType::from_mtlx(&node.output_type).ok_or_else(|| {
            CompileError::Unsupported(format!(
                "pattern node `{}` output type {:?} not supported",
                category, node.output_type
            ))
        })?;

        if matches!(out_ty, ValueType::Boolean)
            && matches!(category, "ifgreater" | "ifgreatereq" | "ifequal")
        {
            let value_ty = Self::input_declared_value_type(node, "value1", ValueType::Float);
            let v1 = self
                .input_value_param(node, "value1", Some(value_ty))?
                .unwrap_or(if matches!(category, "ifequal") {
                    zero_param(Some(value_ty))
                } else {
                    one_param(Some(value_ty))
                });
            let v2 = self
                .input_value_param(node, "value2", Some(value_ty))?
                .unwrap_or(zero_param(Some(value_ty)));
            let op = match category {
                "ifgreater" => CompareOp::Greater,
                "ifgreatereq" => CompareOp::GreaterEq,
                _ => CompareOp::Equal,
            };
            let v1_op = self.param_to_operand(&v1);
            let v2_op = self.param_to_operand(&v2);
            let dst = self.alloc_vreg();
            self.instructions.push(Instruction::CompareBool {
                dst: dst as u16,
                op,
                v1: v1_op,
                v2: v2_op,
            });
            return Ok(dst);
        }

        if matches!(
            out_ty,
            ValueType::Integer | ValueType::Matrix33 | ValueType::Matrix44
        ) && matches!(category, "ifgreater" | "ifgreatereq" | "ifequal")
        {
            let value_ty = Self::input_declared_value_type(node, "value1", ValueType::Float);
            let v1 = self
                .input_value_param(node, "value1", Some(value_ty))?
                .unwrap_or(if matches!(category, "ifequal") {
                    zero_param(Some(value_ty))
                } else {
                    one_param(Some(value_ty))
                });
            let v2 = self
                .input_value_param(node, "value2", Some(value_ty))?
                .unwrap_or(zero_param(Some(value_ty)));
            let in_true = self
                .input_value_param(node, "in1", Some(out_ty))?
                .unwrap_or(zero_param(Some(out_ty)));
            let in_false = self
                .input_value_param(node, "in2", Some(out_ty))?
                .unwrap_or(zero_param(Some(out_ty)));
            let op = match category {
                "ifgreater" => CompareOp::Greater,
                "ifgreatereq" => CompareOp::GreaterEq,
                _ => CompareOp::Equal,
            };
            let v1_op = self.param_to_operand(&v1);
            let v2_op = self.param_to_operand(&v2);
            let t_op = self.param_to_operand(&in_true);
            let f_op = self.param_to_operand(&in_false);
            let dst = self.alloc_vreg();
            self.instructions.push(Instruction::Compare {
                dst: dst as u16,
                op,
                v1: v1_op,
                v2: v2_op,
                in_true: t_op,
                in_false: f_op,
            });
            return Ok(dst);
        }

        if matches!(
            out_ty,
            ValueType::Boolean | ValueType::Integer | ValueType::Matrix33 | ValueType::Matrix44
        ) && category == "ifelse"
        {
            let in_true = self
                .input_value_param(node, "in1", Some(out_ty))?
                .unwrap_or(zero_param(Some(out_ty)));
            let in_false = self
                .input_value_param(node, "in2", Some(out_ty))?
                .unwrap_or(zero_param(Some(out_ty)));
            let cond = self
                .input_value_param(node, "cond", Some(ValueType::Boolean))?
                .or(self.input_value_param(node, "value", Some(ValueType::Boolean))?)
                .unwrap_or(ParamRef::Bool(false));
            let t_op = self.param_to_operand(&in_true);
            let f_op = self.param_to_operand(&in_false);
            let c_op = self.param_to_operand(&cond);
            let dst = self.alloc_vreg();
            self.instructions.push(Instruction::IfElse {
                dst: dst as u16,
                cond: c_op,
                in_true: t_op,
                in_false: f_op,
            });
            return Ok(dst);
        }

        if matches!(out_ty, ValueType::Matrix33 | ValueType::Matrix44) && category == "switch" {
            let which_ty = Self::input_declared_value_type(node, "which", ValueType::Float);
            let which = self
                .input_value_param(node, "which", Some(which_ty))?
                .unwrap_or(zero_param(Some(which_ty)));
            let which_op = self.param_to_operand(&which);
            let mut ops = Vec::with_capacity(10);
            for i in 0..10 {
                let name = format!("in{}", i + 1);
                let p = self
                    .input_value_param(node, &name, Some(out_ty))?
                    .unwrap_or(zero_param(Some(out_ty)));
                ops.push(self.param_to_operand(&p));
            }
            let branches_start = self.push_operands(&ops);
            let dst = self.alloc_vreg();
            self.instructions.push(Instruction::Switch {
                dst: dst as u16,
                ty: out_ty,
                which: which_op,
                branches_start,
            });
            return Ok(dst);
        }

        if matches!(out_ty, ValueType::Matrix33 | ValueType::Matrix44) && category == "creatematrix"
        {
            return self.emit_creatematrix_pattern(node, out_ty);
        }

        if matches!(out_ty, ValueType::Boolean) && matches!(category, "and" | "or" | "xor" | "not")
        {
            return self.emit_logical_pattern(node, category);
        }

        if matches!(out_ty, ValueType::Boolean | ValueType::Integer) && category == "convert" {
            let from = match Self::input_binding(node, "in") {
                Some(FlatInput::Node { node: id, .. }) => {
                    ValueType::from_mtlx(&self.graph.nodes[*id as usize].output_type)
                        .unwrap_or(out_ty)
                }
                Some(FlatInput::Value(v)) => mtlx_value_type(v).unwrap_or(out_ty),
                _ => out_ty,
            };
            if !Self::convert_node_supported(from, out_ty) {
                return Err(CompileError::Unsupported(format!(
                    "convert from {:?} to {:?}",
                    from, out_ty
                )));
            }
            let v = self
                .input_value_param(node, "in", Some(from))?
                .unwrap_or(zero_param(Some(from)));
            return Ok(self.b_convert(from, out_ty, &v));
        }

        if category == "dot"
            && matches!(
                out_ty,
                ValueType::Boolean | ValueType::Integer | ValueType::Matrix33 | ValueType::Matrix44
            )
        {
            let default = match out_ty {
                ValueType::Matrix33 => ParamRef::Matrix33(Mat3::IDENTITY),
                ValueType::Matrix44 => ParamRef::Matrix44(Mat4::IDENTITY),
                _ => zero_param(Some(out_ty)),
            };
            let v = self
                .input_value_param(node, "in", Some(out_ty))?
                .unwrap_or(default);
            let op = self.param_to_operand(&v);
            return Ok(self.operand_to_vreg(op));
        }

        if matches!(category, "geompropvalue" | "geompropvalueuniform") {
            return self.emit_geompropvalue_pattern(node, category, out_ty);
        }

        if matches!(out_ty, ValueType::Matrix33 | ValueType::Matrix44) {
            return match category {
                "add" | "subtract" | "multiply" | "divide" => {
                    let op = match category {
                        "add" => ArithOp::Add,
                        "subtract" => ArithOp::Subtract,
                        "multiply" => ArithOp::Multiply,
                        _ => ArithOp::Divide,
                    };
                    let mat_identity = if matches!(out_ty, ValueType::Matrix44) {
                        ParamRef::Matrix44(Mat4::IDENTITY)
                    } else {
                        ParamRef::Matrix33(Mat3::IDENTITY)
                    };
                    let mat_zero = if matches!(out_ty, ValueType::Matrix44) {
                        ParamRef::Matrix44(Mat4::ZERO)
                    } else {
                        ParamRef::Matrix33(Mat3::ZERO)
                    };
                    let a = self
                        .input_value_param(node, "in1", Some(out_ty))?
                        .unwrap_or(mat_identity);
                    let rhs_ty = Self::input_declared_value_type(node, "in2", out_ty);
                    let rhs_default = if matches!(rhs_ty, ValueType::Float) {
                        if matches!(op, ArithOp::Multiply | ArithOp::Divide) {
                            ParamRef::Float(1.0)
                        } else {
                            ParamRef::Float(0.0)
                        }
                    } else if matches!(op, ArithOp::Multiply | ArithOp::Divide) {
                        if matches!(out_ty, ValueType::Matrix44) {
                            ParamRef::Matrix44(Mat4::IDENTITY)
                        } else {
                            ParamRef::Matrix33(Mat3::IDENTITY)
                        }
                    } else {
                        mat_zero
                    };
                    let b = self
                        .input_value_param(node, "in2", Some(rhs_ty))?
                        .unwrap_or(rhs_default);
                    Ok(self.b_arith(op, out_ty, &a, &b))
                }
                "transpose" => {
                    let dim4 = matches!(out_ty, ValueType::Matrix44);
                    let mat_ty = if dim4 {
                        ValueType::Matrix44
                    } else {
                        ValueType::Matrix33
                    };
                    let v = self
                        .input_value_param(node, "in", Some(mat_ty))?
                        .unwrap_or(if dim4 {
                            ParamRef::Matrix44(Mat4::IDENTITY)
                        } else {
                            ParamRef::Matrix33(Mat3::IDENTITY)
                        });
                    let v_op = self.param_to_operand(&v);
                    let dst = self.alloc_vreg();
                    self.instructions.push(Instruction::Transpose {
                        dst: dst as u16,
                        dim4,
                        src: v_op,
                    });
                    Ok(dst)
                }
                "invertmatrix" => {
                    let dim4 = matches!(out_ty, ValueType::Matrix44);
                    let mat_ty = if dim4 {
                        ValueType::Matrix44
                    } else {
                        ValueType::Matrix33
                    };
                    let v = self
                        .input_value_param(node, "in", Some(mat_ty))?
                        .unwrap_or(if dim4 {
                            ParamRef::Matrix44(Mat4::IDENTITY)
                        } else {
                            ParamRef::Matrix33(Mat3::IDENTITY)
                        });
                    let v_op = self.param_to_operand(&v);
                    let dst = self.alloc_vreg();
                    self.instructions.push(Instruction::InvertMatrix {
                        dst: dst as u16,
                        dim4,
                        src: v_op,
                    });
                    Ok(dst)
                }
                _ => Err(CompileError::Unsupported(format!(
                    "pattern node `{}` output type {:?} not supported",
                    category, out_ty
                ))),
            };
        }

        if matches!(out_ty, ValueType::Integer) && matches!(category, "floor" | "ceil" | "round") {
            let op = match category {
                "floor" => UnaryOp::Floor,
                "ceil" => UnaryOp::Ceil,
                _ => UnaryOp::Round,
            };
            let v = self
                .input_value_param(node, "in", Some(ValueType::Float))?
                .unwrap_or(ParamRef::Float(0.0));
            let tmp = self.b_unary(op, ValueType::Float, &v);
            return Ok(self.b_convert(ValueType::Float, ValueType::Integer, &ParamRef::Local(tmp)));
        }

        let out_color_ty = match out_ty {
            ValueType::Color4 | ValueType::Vector4 => ValueType::Color4,
            ValueType::Vector2 => ValueType::Vector2,
            ValueType::Float => ValueType::Float,
            ValueType::Vector3 => ValueType::Vector3,
            ValueType::Color3 => ValueType::Color3,
            ValueType::Integer | ValueType::Boolean | ValueType::Matrix33 | ValueType::Matrix44 => {
                return Err(CompileError::Unsupported(format!(
                    "pattern node `{}` output type {:?} not supported",
                    category, out_ty
                )));
            }
        };

        match category {
            "constant" => {
                let p = self
                    .input_value_param(node, "value", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let op = self.param_to_operand(&p);
                Ok(self.operand_to_vreg(op))
            }
            "frame" => Ok(self.ensure_geometric_kind_local(GeometricKind::Frame)),
            "time" => Ok(self.ensure_geometric_kind_local(GeometricKind::Time)),
            "viewdirection" => {
                let space = parse_geom_space(Self::input_static_string(node, category, "space")?)?;
                Ok(self.ensure_geometric_kind_local(GeometricKind::ViewDirection(space)))
            }
            "add" | "subtract" | "multiply" | "divide" | "modulo" | "min" | "max" | "power"
            | "safepower" | "atan2" => {
                let op = match category {
                    "add" => ArithOp::Add,
                    "subtract" => ArithOp::Subtract,
                    "multiply" => ArithOp::Multiply,
                    "divide" => ArithOp::Divide,
                    "modulo" => ArithOp::Modulo,
                    "min" => ArithOp::Min,
                    "max" => ArithOp::Max,
                    "power" => ArithOp::Power,
                    "safepower" => ArithOp::SafePower,
                    _ => ArithOp::Atan2,
                };
                let (a_name, b_name, a_default, b_default) = match op {
                    ArithOp::Atan2 => ("iny", "inx", 0.0_f32, 1.0_f32),
                    ArithOp::Multiply
                    | ArithOp::Divide
                    | ArithOp::Modulo
                    | ArithOp::Power
                    | ArithOp::SafePower => ("in1", "in2", 0.0, 1.0),
                    _ => ("in1", "in2", 0.0, 0.0),
                };
                let a = self
                    .input_value_param(node, a_name, Some(out_color_ty))?
                    .or(if a_name != "in1" {
                        self.input_value_param(node, "in1", Some(out_color_ty))?
                    } else {
                        None
                    })
                    .unwrap_or(ParamRef::Float(a_default));
                let b = self
                    .input_value_param(node, b_name, Some(out_color_ty))?
                    .or(if b_name != "in2" {
                        self.input_value_param(node, "in2", Some(out_color_ty))?
                    } else {
                        None
                    })
                    .unwrap_or(ParamRef::Float(b_default));
                Ok(self.b_arith(op, out_color_ty, &a, &b))
            }
            "invert" => {
                let amount = self
                    .input_value_param(node, "amount", Some(out_color_ty))?
                    .unwrap_or_else(|| one_param(Some(out_color_ty)));
                let v = self
                    .input_value_param(node, "in", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                Ok(self.b_arith(ArithOp::Subtract, out_color_ty, &amount, &v))
            }
            "sin" | "cos" | "tan" | "asin" | "acos" | "sqrt" | "ln" | "exp" | "absval" | "sign"
            | "floor" | "ceil" | "round" | "fract" | "trianglewave" | "normalize" | "rgbtohsv"
            | "hsvtorgb" => {
                let op = match category {
                    "sin" => UnaryOp::Sin,
                    "cos" => UnaryOp::Cos,
                    "tan" => UnaryOp::Tan,
                    "asin" => UnaryOp::Asin,
                    "acos" => UnaryOp::Acos,
                    "sqrt" => UnaryOp::Sqrt,
                    "ln" => UnaryOp::Ln,
                    "exp" => UnaryOp::Exp,
                    "absval" => UnaryOp::Abs,
                    "sign" => UnaryOp::Sign,
                    "floor" => UnaryOp::Floor,
                    "ceil" => UnaryOp::Ceil,
                    "round" => UnaryOp::Round,
                    "fract" => UnaryOp::Fract,
                    "trianglewave" => UnaryOp::Trianglewave,
                    "normalize" => UnaryOp::Normalize,
                    "rgbtohsv" => UnaryOp::RgbToHsv,
                    _ => UnaryOp::HsvToRgb,
                };
                let v = self
                    .input_value_param(node, "in", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                Ok(self.b_unary(op, out_color_ty, &v))
            }
            "magnitude" => {
                let in_ty = Self::input_declared_value_type(node, "in", ValueType::Vector3);
                let v = self
                    .input_value_param(node, "in", Some(in_ty))?
                    .unwrap_or(zero_param(Some(in_ty)));
                Ok(self.b_unary(UnaryOp::Length, in_ty, &v))
            }
            "clamp" => {
                let v = self
                    .input_value_param(node, "in", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let lo = self
                    .input_value_param(node, "low", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let hi = self
                    .input_value_param(node, "high", Some(out_color_ty))?
                    .unwrap_or(one_param(Some(out_color_ty)));
                Ok(self.b_clamp(out_color_ty, &v, &lo, &hi))
            }
            "mix" => {
                let mix_value_ty = out_ty;
                let bg = self
                    .input_value_param(node, "bg", Some(mix_value_ty))?
                    .unwrap_or(zero_param(Some(mix_value_ty)));
                let fg = self
                    .input_value_param(node, "fg", Some(mix_value_ty))?
                    .unwrap_or(zero_param(Some(mix_value_ty)));
                let mix_ty = Self::input_declared_value_type(node, "mix", ValueType::Float);
                let mix = self
                    .input_value_param(node, "mix", Some(mix_ty))?
                    .unwrap_or(ParamRef::Float(0.0));
                Ok(self.b_mix(mix_value_ty, &bg, &fg, &mix))
            }
            "smoothstep" => {
                let v = self
                    .input_value_param(node, "in", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let lo = self
                    .input_value_param(node, "low", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let hi = self
                    .input_value_param(node, "high", Some(out_color_ty))?
                    .unwrap_or(one_param(Some(out_color_ty)));
                let v_op = self.param_to_operand(&v);
                let lo_op = self.param_to_operand(&lo);
                let hi_op = self.param_to_operand(&hi);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Smoothstep {
                    dst: dst as u16,
                    ty: out_color_ty,
                    v: v_op,
                    lo: lo_op,
                    hi: hi_op,
                });
                Ok(dst)
            }
            "convert" | "dot" => {
                let from = match Self::input_binding(node, "in") {
                    Some(FlatInput::Node { node: id, .. }) => {
                        ValueType::from_mtlx(&self.graph.nodes[*id as usize].output_type)
                            .unwrap_or(out_ty)
                    }
                    Some(FlatInput::Value(v)) => mtlx_value_type(v).unwrap_or(out_ty),
                    _ => out_ty,
                };
                if category == "convert" && !Self::convert_node_supported(from, out_ty) {
                    return Err(CompileError::Unsupported(format!(
                        "convert from {:?} to {:?}",
                        from, out_ty
                    )));
                }
                let v = self
                    .input_value_param(node, "in", Some(from))?
                    .unwrap_or(zero_param(Some(from)));
                if category == "dot" {
                    let op = self.param_to_operand(&v);
                    return Ok(self.operand_to_vreg(op));
                }
                Ok(self.b_convert(from, out_ty, &v))
            }
            "ifgreater" | "ifgreatereq" | "ifequal" => {
                let value_ty = Self::input_declared_value_type(node, "value1", ValueType::Float);
                let v1 = self
                    .input_value_param(node, "value1", Some(value_ty))?
                    .unwrap_or(if matches!(category, "ifequal") {
                        zero_param(Some(value_ty))
                    } else {
                        one_param(Some(value_ty))
                    });
                let v2 = self
                    .input_value_param(node, "value2", Some(value_ty))?
                    .unwrap_or(zero_param(Some(value_ty)));
                let in_true = self
                    .input_value_param(node, "in1", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let in_false = self
                    .input_value_param(node, "in2", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let op = match category {
                    "ifgreater" => CompareOp::Greater,
                    "ifgreatereq" => CompareOp::GreaterEq,
                    _ => CompareOp::Equal,
                };
                let v1_op = self.param_to_operand(&v1);
                let v2_op = self.param_to_operand(&v2);
                let t_op = self.param_to_operand(&in_true);
                let f_op = self.param_to_operand(&in_false);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Compare {
                    dst: dst as u16,
                    op,
                    v1: v1_op,
                    v2: v2_op,
                    in_true: t_op,
                    in_false: f_op,
                });
                Ok(dst)
            }
            "combine2" | "combine3" | "combine4" => {
                let kind = self.combine_kind(node, category, out_ty)?;
                let arg_names: &[&str] = match kind {
                    CombineKind::Vector2FromFloats
                    | CombineKind::Color4FromColor3Float
                    | CombineKind::Vector4FromVector3Float
                    | CombineKind::Vector4FromVector2Vector2 => &["in1", "in2"],
                    CombineKind::Color3FromFloats | CombineKind::Vector3FromFloats => {
                        &["in1", "in2", "in3"]
                    }
                    CombineKind::Color4FromFloats | CombineKind::Vector4FromFloats => {
                        &["in1", "in2", "in3", "in4"]
                    }
                };
                let in_types = combine_input_types(kind);
                let mut ops = Vec::with_capacity(arg_names.len());
                for (i, name) in arg_names.iter().enumerate() {
                    let ty = in_types.get(i).copied().unwrap_or(ValueType::Float);
                    let p = self
                        .input_value_param(node, name, Some(ty))?
                        .unwrap_or(zero_param(Some(ty)));
                    ops.push(self.param_to_operand(&p));
                }
                let operands_start = self.push_operands(&ops);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Combine {
                    dst: dst as u16,
                    kind,
                    operands_start,
                });
                Ok(dst)
            }
            "image" | "tiledimage" => {
                Self::warn_animated_image_inputs(node, category)?;
                let file = Self::input_static_string(node, category, "file")?
                    .unwrap_or("")
                    .to_string();
                let cs = Self::image_color_space(node, "file");
                let texcoord = self
                    .input_value_param(node, "texcoord", Some(ValueType::Vector2))?
                    .unwrap_or_else(|| {
                        let idx = self.ensure_geometric_local(&FgKind::Texcoord);
                        ParamRef::Local(idx)
                    });
                let tiling = self
                    .input_value_param(node, "uvtiling", Some(ValueType::Vector2))?
                    .unwrap_or(ParamRef::Vector2(Vec2::ONE));
                let offset = self
                    .input_value_param(node, "uvoffset", Some(ValueType::Vector2))?
                    .unwrap_or(ParamRef::Vector2(Vec2::ZERO));
                let default = self
                    .input_value_param(node, "default", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let uaddress =
                    parse_address_mode(Self::input_static_string(node, category, "uaddressmode")?)?;
                let vaddress =
                    parse_address_mode(Self::input_static_string(node, category, "vaddressmode")?)?;
                let filter =
                    parse_filter_type(Self::input_static_string(node, category, "filtertype")?)?;
                let kind = match category {
                    "tiledimage" => ImageKind::TiledImage,
                    "latlongimage" => ImageKind::LatLongImage,
                    _ => ImageKind::Image,
                };
                let texture = self.lookup_texture(&file, out_color_ty);
                let tc_op = self.param_to_operand(&texcoord);
                let tl_op = self.param_to_operand(&tiling);
                let of_op = self.param_to_operand(&offset);
                let de_op = self.param_to_operand(&default);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Image {
                    dst: dst as u16,
                    texture,
                    kind,
                    output: out_color_ty,
                    color_space: cs,
                    uaddress,
                    vaddress,
                    filter,
                    texcoord: tc_op,
                    tiling: tl_op,
                    offset: of_op,
                    default: de_op,
                });
                Ok(dst)
            }
            "latlongimage" => {
                Self::warn_animated_image_inputs(node, category)?;
                let file = Self::input_static_string(node, category, "file")?
                    .unwrap_or("")
                    .to_string();
                let cs = Self::image_color_space(node, "file");
                let viewdir = self
                    .input_value_param(node, "viewdir", Some(ValueType::Vector3))?
                    .unwrap_or(ParamRef::Vector3(Vec3::Z));
                let rotation = self
                    .input_value_param(node, "rotation", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(0.0));
                let viewdir_op = self.param_to_operand(&viewdir);
                let rotation_op = self.param_to_operand(&rotation);
                let uv_dst = self.alloc_vreg();
                self.instructions.push(Instruction::LatlongUv {
                    dst: uv_dst as u16,
                    viewdir: viewdir_op,
                    rotation: rotation_op,
                });
                let default = self
                    .input_value_param(node, "default", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let tiling_op = self.param_to_operand(&ParamRef::Vector2(Vec2::ONE));
                let offset_op = self.param_to_operand(&ParamRef::Vector2(Vec2::ZERO));
                let default_op = self.param_to_operand(&default);
                let texture = self.lookup_texture(&file, out_color_ty);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Image {
                    dst: dst as u16,
                    texture,
                    kind: ImageKind::LatLongImage,
                    output: out_color_ty,
                    color_space: cs,
                    uaddress: AddressMode::Periodic,
                    vaddress: AddressMode::Mirror,
                    filter: FilterType::Linear,
                    texcoord: Operand::Reg(uv_dst as u16),
                    tiling: tiling_op,
                    offset: offset_op,
                    default: default_op,
                });
                Ok(dst)
            }
            "place2d" => {
                let trs = self
                    .input_value_param(node, "operationorder", Some(ValueType::Integer))?
                    .map(|p| matches!(p, ParamRef::Integer(1)))
                    .unwrap_or(false);
                let texcoord = self
                    .input_value_param(node, "texcoord", Some(ValueType::Vector2))?
                    .unwrap_or_else(|| {
                        let idx = self.ensure_geometric_local(&FgKind::Texcoord);
                        ParamRef::Local(idx)
                    });
                let pivot = self
                    .input_value_param(node, "pivot", Some(ValueType::Vector2))?
                    .unwrap_or(ParamRef::Vector2(Vec2::ZERO));
                let scale = self
                    .input_value_param(node, "scale", Some(ValueType::Vector2))?
                    .unwrap_or(ParamRef::Vector2(Vec2::ONE));
                let rotate = self
                    .input_value_param(node, "rotate", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(0.0));
                let offset = self
                    .input_value_param(node, "offset", Some(ValueType::Vector2))?
                    .unwrap_or(ParamRef::Vector2(Vec2::ZERO));
                let tc_op = self.param_to_operand(&texcoord);
                let pv_op = self.param_to_operand(&pivot);
                let sc_op = self.param_to_operand(&scale);
                let ro_op = self.param_to_operand(&rotate);
                let of_op = self.param_to_operand(&offset);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Place2d {
                    dst: dst as u16,
                    trs,
                    texcoord: tc_op,
                    pivot: pv_op,
                    scale: sc_op,
                    rotate: ro_op,
                    offset: of_op,
                });
                Ok(dst)
            }
            "normalmap" => {
                let raw = self
                    .input_value_param(node, "in", Some(ValueType::Vector3))?
                    .unwrap_or(ParamRef::Vector3(Vec3::new(0.5, 0.5, 1.0)));
                let scale = self
                    .input_value_param(node, "scale", Some(ValueType::Vector2))?
                    .unwrap_or(ParamRef::Vector2(Vec2::ONE));
                let has_override = Self::input_binding(node, "normal").is_some()
                    || Self::input_binding(node, "tangent").is_some()
                    || Self::input_binding(node, "bitangent").is_some();
                if has_override {
                    let n_override = self
                        .input_value_param(node, "normal", Some(ValueType::Vector3))?
                        .unwrap_or_else(|| {
                            let idx = self.ensure_geometric_kind_local(GeometricKind::Normal(
                                GeomSpace::World,
                            ));
                            ParamRef::Local(idx)
                        });
                    let t_override = self
                        .input_value_param(node, "tangent", Some(ValueType::Vector3))?
                        .unwrap_or_else(|| {
                            let idx = self.ensure_geometric_kind_local(GeometricKind::Tangent(
                                GeomSpace::World,
                            ));
                            ParamRef::Local(idx)
                        });
                    let b_override = self
                        .input_value_param(node, "bitangent", Some(ValueType::Vector3))?
                        .unwrap_or_else(|| {
                            let idx = self.ensure_geometric_kind_local(GeometricKind::Bitangent(
                                GeomSpace::World,
                            ));
                            ParamRef::Local(idx)
                        });
                    let ops = [
                        self.param_to_operand(&raw),
                        self.param_to_operand(&scale),
                        self.param_to_operand(&n_override),
                        self.param_to_operand(&t_override),
                        self.param_to_operand(&b_override),
                    ];
                    let operands_start = self.push_operands(&ops);
                    let dst = self.alloc_vreg();
                    self.instructions.push(Instruction::NormalmapWithFrame {
                        dst: dst as u16,
                        operands_start,
                    });
                    Ok(dst)
                } else {
                    let raw_op = self.param_to_operand(&raw);
                    let scale_op = self.param_to_operand(&scale);
                    let dst = self.alloc_vreg();
                    self.instructions.push(Instruction::Normalmap {
                        dst: dst as u16,
                        raw: raw_op,
                        scale: scale_op,
                    });
                    Ok(dst)
                }
            }
            "bump" => {
                let height = self
                    .input_value_param(node, "height", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(0.0));
                if !matches!(height, ParamRef::Float(_)) {
                    return Err(CompileError::Unsupported(
                        "bump requires heightfield derivatives/sample-grid evaluation; dynamic height inputs are not implemented".into(),
                    ));
                }
                let scale = self
                    .input_value_param(node, "scale", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(1.0));
                let has_override = Self::input_binding(node, "normal").is_some()
                    || Self::input_binding(node, "tangent").is_some()
                    || Self::input_binding(node, "bitangent").is_some();
                let raw = ParamRef::Vector3(Vec3::new(0.5, 0.5, 1.0));
                if has_override {
                    let n_override = self
                        .input_value_param(node, "normal", Some(ValueType::Vector3))?
                        .unwrap_or_else(|| {
                            let idx = self.ensure_geometric_kind_local(GeometricKind::Normal(
                                GeomSpace::World,
                            ));
                            ParamRef::Local(idx)
                        });
                    let t_override = self
                        .input_value_param(node, "tangent", Some(ValueType::Vector3))?
                        .unwrap_or_else(|| {
                            let idx = self.ensure_geometric_kind_local(GeometricKind::Tangent(
                                GeomSpace::World,
                            ));
                            ParamRef::Local(idx)
                        });
                    let b_override = self
                        .input_value_param(node, "bitangent", Some(ValueType::Vector3))?
                        .unwrap_or_else(|| {
                            let idx = self.ensure_geometric_kind_local(GeometricKind::Bitangent(
                                GeomSpace::World,
                            ));
                            ParamRef::Local(idx)
                        });
                    let ops = [
                        self.param_to_operand(&raw),
                        self.param_to_operand(&scale),
                        self.param_to_operand(&n_override),
                        self.param_to_operand(&t_override),
                        self.param_to_operand(&b_override),
                    ];
                    let operands_start = self.push_operands(&ops);
                    let dst = self.alloc_vreg();
                    self.instructions.push(Instruction::NormalmapWithFrame {
                        dst: dst as u16,
                        operands_start,
                    });
                    Ok(dst)
                } else {
                    let raw_op = self.param_to_operand(&raw);
                    let s_op = self.param_to_operand(&scale);
                    let dst = self.alloc_vreg();
                    self.instructions.push(Instruction::Normalmap {
                        dst: dst as u16,
                        raw: raw_op,
                        scale: s_op,
                    });
                    Ok(dst)
                }
            }
            "facingratio" => {
                let invert = Self::input_static_bool(node, category, "invert")?.unwrap_or(false);
                let faceforward =
                    Self::input_static_bool(node, category, "faceforward")?.unwrap_or(true);
                let view = self
                    .input_value_param(node, "viewdirection", Some(ValueType::Vector3))?
                    .unwrap_or_else(|| {
                        let idx = self.ensure_geometric_kind_local(GeometricKind::ViewDirection(
                            GeomSpace::World,
                        ));
                        ParamRef::Local(idx)
                    });
                let normal = self
                    .input_value_param(node, "normal", Some(ValueType::Vector3))?
                    .unwrap_or_else(|| {
                        let idx = self
                            .ensure_geometric_kind_local(GeometricKind::Normal(GeomSpace::World));
                        ParamRef::Local(idx)
                    });
                let v_op = self.param_to_operand(&view);
                let n_op = self.param_to_operand(&normal);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::FacingRatio {
                    dst: dst as u16,
                    view: v_op,
                    normal: n_op,
                    invert,
                    faceforward,
                });
                Ok(dst)
            }
            "extract" => {
                let in_ty = match Self::input_binding(node, "in") {
                    Some(FlatInput::Node { node: id, .. }) => {
                        ValueType::from_mtlx(&self.graph.nodes[*id as usize].output_type)
                            .unwrap_or(ValueType::Color3)
                    }
                    Some(FlatInput::Value(v)) => mtlx_value_type(v).unwrap_or(ValueType::Color3),
                    _ => ValueType::Color3,
                };
                if matches!(in_ty, ValueType::Matrix33 | ValueType::Matrix44) {
                    let dim4 = in_ty == ValueType::Matrix44;
                    let expected_out = if dim4 {
                        ValueType::Vector4
                    } else {
                        ValueType::Vector3
                    };
                    if out_ty != expected_out {
                        return Err(CompileError::Unsupported(format!(
                            "extract output type {:?} is incompatible with {:?}",
                            out_ty, in_ty
                        )));
                    }
                    let matrix =
                        self.input_value_param(node, "in", Some(in_ty))?
                            .unwrap_or(if dim4 {
                                ParamRef::Matrix44(Mat4::IDENTITY)
                            } else {
                                ParamRef::Matrix33(Mat3::IDENTITY)
                            });
                    let index = self
                        .input_value_param(node, "index", Some(ValueType::Integer))?
                        .unwrap_or(ParamRef::Integer(0));
                    let ParamRef::Integer(index) = index else {
                        return Err(CompileError::Unsupported(
                            "extract.index for matrix input must be a static integer".into(),
                        ));
                    };
                    let max_index = if dim4 { 3 } else { 2 };
                    if !(0..=max_index).contains(&index) {
                        return Err(CompileError::Unsupported(format!(
                            "extract.index `{}` out of range 0..={}",
                            index, max_index
                        )));
                    }
                    let src = self.param_to_operand(&matrix);
                    let dst = self.alloc_vreg();
                    self.instructions.push(Instruction::ExtractRowVector {
                        dst: dst as u16,
                        dim4,
                        src,
                        index: index as u8,
                    });
                    return Ok(dst);
                }
                let v = self
                    .input_value_param(node, "in", Some(in_ty))?
                    .unwrap_or(zero_param(Some(in_ty)));
                let idx_p = self
                    .input_value_param(node, "index", Some(ValueType::Integer))?
                    .unwrap_or(ParamRef::Integer(0));
                let v_op = self.param_to_operand(&v);
                let idx_op = self.param_to_operand(&idx_p);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Extract {
                    dst: dst as u16,
                    in_ty,
                    src: v_op,
                    idx: idx_op,
                });
                Ok(dst)
            }
            "luminance" => {
                let v = self
                    .input_value_param(node, "in", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let lumacoeffs = self
                    .input_value_param(node, "lumacoeffs", Some(ValueType::Color3))?
                    .unwrap_or(ParamRef::Color3(Vec3::new(0.2722287, 0.6740818, 0.0536895)));
                let v_op = self.param_to_operand(&v);
                let lc_op = self.param_to_operand(&lumacoeffs);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::LuminanceWithCoeffs {
                    dst: dst as u16,
                    ty: out_color_ty,
                    c: v_op,
                    lumacoeffs: lc_op,
                });
                Ok(dst)
            }
            "roughness_anisotropy" => {
                let r = self
                    .input_value_param(node, "roughness", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(0.0));
                let a = self
                    .input_value_param(node, "anisotropy", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(0.0));
                let r_op = self.param_to_operand(&r);
                let a_op = self.param_to_operand(&a);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::RoughnessAnisotropy {
                    dst: dst as u16,
                    r: r_op,
                    a: a_op,
                });
                Ok(dst)
            }
            "glossiness_anisotropy" => {
                let g = self
                    .input_value_param(node, "glossiness", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(1.0));
                let a = self
                    .input_value_param(node, "anisotropy", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(0.0));
                let g_op = self.param_to_operand(&g);
                let a_op = self.param_to_operand(&a);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::GlossinessAnisotropy {
                    dst: dst as u16,
                    g: g_op,
                    a: a_op,
                });
                Ok(dst)
            }
            "roughness_dual" => {
                let r = self
                    .input_value_param(node, "roughness", Some(ValueType::Vector2))?
                    .unwrap_or(ParamRef::Vector2(Vec2::ZERO));
                let r_op = self.param_to_operand(&r);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::RoughnessDual {
                    dst: dst as u16,
                    src: r_op,
                });
                Ok(dst)
            }
            "artistic_ior" => {
                let which = match _output {
                    Some("ior") => ArtisticIorOutput::Ior,
                    Some("extinction") => ArtisticIorOutput::Extinction,
                    Some(name) => {
                        return Err(CompileError::Unsupported(format!(
                            "artistic_ior output `{}` is not defined",
                            name
                        )));
                    }
                    None => {
                        return Err(CompileError::Unsupported(
                            "artistic_ior requires output `ior` or `extinction`".into(),
                        ));
                    }
                };
                let refl = self
                    .input_value_param(node, "reflectivity", Some(ValueType::Color3))?
                    .unwrap_or(ParamRef::Color3(Vec3::new(0.944, 0.776, 0.373)));
                let edge = self
                    .input_value_param(node, "edge_color", Some(ValueType::Color3))?
                    .unwrap_or(ParamRef::Color3(Vec3::new(0.998, 0.981, 0.751)));
                let r_op = self.param_to_operand(&refl);
                let e_op = self.param_to_operand(&edge);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::ArtisticIor {
                    dst: dst as u16,
                    which,
                    refl: r_op,
                    edge: e_op,
                });
                Ok(dst)
            }
            "blackbody" => {
                let temp = self
                    .input_value_param(node, "temperature", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(5000.0));
                let t_op = self.param_to_operand(&temp);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Blackbody {
                    dst: dst as u16,
                    temp: t_op,
                });
                Ok(dst)
            }
            "premult" => {
                let v = self
                    .input_value_param(node, "in", Some(ValueType::Color4))?
                    .unwrap_or(ParamRef::Color4(Vec4::new(0.0, 0.0, 0.0, 1.0)));
                let v_op = self.param_to_operand(&v);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Premult {
                    dst: dst as u16,
                    src: v_op,
                });
                Ok(dst)
            }
            "unpremult" => {
                let v = self
                    .input_value_param(node, "in", Some(ValueType::Color4))?
                    .unwrap_or(ParamRef::Color4(Vec4::new(0.0, 0.0, 0.0, 1.0)));
                let v_op = self.param_to_operand(&v);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Unpremult {
                    dst: dst as u16,
                    src: v_op,
                });
                Ok(dst)
            }
            "transformpoint" | "transformvector" | "transformnormal" => {
                let from_s = Self::input_static_string(node, category, "fromspace")?;
                let to_s = Self::input_static_string(node, category, "tospace")?;
                let from = parse_geom_space(from_s.filter(|s| !s.is_empty()).or(Some("object")))?;
                let to = parse_geom_space(to_s.filter(|s| !s.is_empty()).or(Some("world")))?;
                let default = if category == "transformnormal" {
                    Vec3::Z
                } else {
                    Vec3::ZERO
                };
                let v = self
                    .input_value_param(node, "in", Some(ValueType::Vector3))?
                    .unwrap_or(ParamRef::Vector3(default));
                let v_op = self.param_to_operand(&v);
                let dst = self.alloc_vreg();
                let instr = match category {
                    "transformpoint" => Instruction::TransformPoint {
                        dst: dst as u16,
                        from,
                        to,
                        v: v_op,
                    },
                    "transformvector" => Instruction::TransformVector {
                        dst: dst as u16,
                        from,
                        to,
                        v: v_op,
                    },
                    _ => Instruction::TransformNormal {
                        dst: dst as u16,
                        from,
                        to,
                        v: v_op,
                    },
                };
                self.instructions.push(instr);
                Ok(dst)
            }
            "noise2d" | "noise3d" | "fractal2d" | "fractal3d" => {
                let kind = match category {
                    "noise2d" => NoiseKind::Perlin2d,
                    "noise3d" => NoiseKind::Perlin3d,
                    "fractal2d" => NoiseKind::Fractal2d,
                    _ => NoiseKind::Fractal3d,
                };
                let is_2d = matches!(kind, NoiseKind::Perlin2d | NoiseKind::Fractal2d);
                let coord_ty = if is_2d {
                    ValueType::Vector2
                } else {
                    ValueType::Vector3
                };
                let texcoord = self
                    .input_value_param(node, "texcoord", Some(coord_ty))?
                    .unwrap_or_else(|| {
                        if is_2d {
                            let idx = self.ensure_geometric_local(&FgKind::Texcoord);
                            ParamRef::Local(idx)
                        } else {
                            let idx = self.ensure_geometric_local(&FgKind::Position);
                            ParamRef::Local(idx)
                        }
                    });
                let amp = self
                    .input_value_param(node, "amplitude", Some(out_color_ty))?
                    .unwrap_or(one_param(Some(out_color_ty)));
                let pivot = self
                    .input_value_param(node, "pivot", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(0.0));
                let octaves = self
                    .input_value_param(node, "octaves", Some(ValueType::Integer))?
                    .unwrap_or(ParamRef::Integer(3));
                let lacunarity = self
                    .input_value_param(node, "lacunarity", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(2.0));
                let diminish = self
                    .input_value_param(node, "diminish", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(0.5));
                let jitter = self
                    .input_value_param(node, "jitter", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(1.0));
                let ops = [
                    self.param_to_operand(&texcoord),
                    self.param_to_operand(&amp),
                    self.param_to_operand(&pivot),
                    self.param_to_operand(&octaves),
                    self.param_to_operand(&lacunarity),
                    self.param_to_operand(&diminish),
                    self.param_to_operand(&jitter),
                ];
                let operands_start = self.push_operands(&ops);
                let output = noise_output_for(out_color_ty);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Noise {
                    dst: dst as u16,
                    kind,
                    output,
                    operands_start,
                });
                Ok(dst)
            }
            "cellnoise2d" | "cellnoise3d" => {
                let dim3 = category == "cellnoise3d";
                let coord_ty = if dim3 {
                    ValueType::Vector3
                } else {
                    ValueType::Vector2
                };
                let coord = self
                    .input_value_param(node, "texcoord", Some(coord_ty))?
                    .or(self.input_value_param(node, "position", Some(coord_ty))?)
                    .unwrap_or_else(|| {
                        let idx = self.ensure_geometric_local(if dim3 {
                            &FgKind::Position
                        } else {
                            &FgKind::Texcoord
                        });
                        ParamRef::Local(idx)
                    });
                let c_op = self.param_to_operand(&coord);
                let output = noise_output_for(out_color_ty);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Cellnoise {
                    dst: dst as u16,
                    dim3,
                    output,
                    coord: c_op,
                });
                Ok(dst)
            }
            "worleynoise2d" | "worleynoise3d" => {
                let dim3 = category == "worleynoise3d";
                let coord_ty = if dim3 {
                    ValueType::Vector3
                } else {
                    ValueType::Vector2
                };
                let coord = self
                    .input_value_param(node, "texcoord", Some(coord_ty))?
                    .or(self.input_value_param(node, "position", Some(coord_ty))?)
                    .unwrap_or_else(|| {
                        let idx = self.ensure_geometric_local(if dim3 {
                            &FgKind::Position
                        } else {
                            &FgKind::Texcoord
                        });
                        ParamRef::Local(idx)
                    });
                let jitter = self
                    .input_value_param(node, "jitter", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(1.0));
                let style = Self::input_worley_style(node, category)?;
                let ops = [
                    self.param_to_operand(&coord),
                    self.param_to_operand(&jitter),
                ];
                let operands_start = self.push_operands(&ops);
                let output = noise_output_for(out_color_ty);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Worley {
                    dst: dst as u16,
                    dim3,
                    output,
                    style,
                    operands_start,
                });
                Ok(dst)
            }
            "flake2d" | "flake3d" => {
                let dim3 = category == "flake3d";
                let output = match _output {
                    Some("id") => FlakeOutput::Id,
                    Some("rand") => FlakeOutput::Rand,
                    Some("presence") => FlakeOutput::Presence,
                    Some("flakenormal") => FlakeOutput::Normal,
                    Some(name) => {
                        return Err(CompileError::Unsupported(format!(
                            "{} output `{}` is not defined",
                            category, name
                        )));
                    }
                    None => {
                        return Err(CompileError::Unsupported(format!(
                            "{} requires an output name",
                            category
                        )));
                    }
                };
                let size = self
                    .input_value_param(node, "size", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(0.01));
                let roughness = self
                    .input_value_param(node, "roughness", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(0.1));
                let coverage = self
                    .input_value_param(node, "coverage", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(0.5));
                let coord = if dim3 {
                    self.input_value_param(node, "position", Some(ValueType::Vector3))?
                        .unwrap_or_else(|| {
                            let idx = self.ensure_geometric_local(&FgKind::Position);
                            ParamRef::Local(idx)
                        })
                } else {
                    self.input_value_param(node, "texcoord", Some(ValueType::Vector2))?
                        .unwrap_or_else(|| {
                            let idx = self.ensure_geometric_local(&FgKind::Texcoord);
                            ParamRef::Local(idx)
                        })
                };
                let normal = self
                    .input_value_param(node, "normal", Some(ValueType::Vector3))?
                    .unwrap_or_else(|| {
                        let idx = self
                            .ensure_geometric_kind_local(GeometricKind::Normal(GeomSpace::World));
                        ParamRef::Local(idx)
                    });
                let tangent = self
                    .input_value_param(node, "tangent", Some(ValueType::Vector3))?
                    .unwrap_or_else(|| {
                        let idx = self
                            .ensure_geometric_kind_local(GeometricKind::Tangent(GeomSpace::World));
                        ParamRef::Local(idx)
                    });
                let bitangent = self
                    .input_value_param(node, "bitangent", Some(ValueType::Vector3))?
                    .unwrap_or_else(|| {
                        let idx = self.ensure_geometric_kind_local(GeometricKind::Bitangent(
                            GeomSpace::World,
                        ));
                        ParamRef::Local(idx)
                    });
                let ops = [
                    self.param_to_operand(&size),
                    self.param_to_operand(&roughness),
                    self.param_to_operand(&coverage),
                    self.param_to_operand(&coord),
                    self.param_to_operand(&normal),
                    self.param_to_operand(&tangent),
                    self.param_to_operand(&bitangent),
                ];
                let operands_start = self.push_operands(&ops);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Flake {
                    dst: dst as u16,
                    dim3,
                    output,
                    operands_start,
                });
                Ok(dst)
            }
            "ramplr" | "ramptb" => {
                let texcoord = self
                    .input_value_param(node, "texcoord", Some(ValueType::Vector2))?
                    .unwrap_or_else(|| {
                        let idx = self.ensure_geometric_local(&FgKind::Texcoord);
                        ParamRef::Local(idx)
                    });
                let l = self
                    .input_value_param(node, "valuel", Some(out_color_ty))?
                    .or(self.input_value_param(node, "valuet", Some(out_color_ty))?)
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let r = self
                    .input_value_param(node, "valuer", Some(out_color_ty))?
                    .or(self.input_value_param(node, "valueb", Some(out_color_ty))?)
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let tc_op = self.param_to_operand(&texcoord);
                let l_op = self.param_to_operand(&l);
                let r_op = self.param_to_operand(&r);
                let dst = self.alloc_vreg();
                let instr = if category == "ramplr" {
                    Instruction::Ramplr {
                        dst: dst as u16,
                        ty: out_color_ty,
                        texcoord: tc_op,
                        l: l_op,
                        r: r_op,
                    }
                } else {
                    Instruction::Ramptb {
                        dst: dst as u16,
                        ty: out_color_ty,
                        texcoord: tc_op,
                        t: l_op,
                        b: r_op,
                    }
                };
                self.instructions.push(instr);
                Ok(dst)
            }
            "ramp4" => {
                let texcoord = self
                    .input_value_param(node, "texcoord", Some(ValueType::Vector2))?
                    .unwrap_or_else(|| {
                        let idx = self.ensure_geometric_local(&FgKind::Texcoord);
                        ParamRef::Local(idx)
                    });
                let tl = self
                    .input_value_param(node, "valuetl", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let tr = self
                    .input_value_param(node, "valuetr", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let bl = self
                    .input_value_param(node, "valuebl", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let br = self
                    .input_value_param(node, "valuebr", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let tc_op = self.param_to_operand(&texcoord);
                let tl_op = self.param_to_operand(&tl);
                let tr_op = self.param_to_operand(&tr);
                let bl_op = self.param_to_operand(&bl);
                let br_op = self.param_to_operand(&br);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Ramp4 {
                    dst: dst as u16,
                    ty: out_color_ty,
                    texcoord: tc_op,
                    tl: tl_op,
                    tr: tr_op,
                    bl: bl_op,
                    br: br_op,
                });
                Ok(dst)
            }
            "splitlr" | "splittb" => {
                let texcoord = self
                    .input_value_param(node, "texcoord", Some(ValueType::Vector2))?
                    .unwrap_or_else(|| {
                        let idx = self.ensure_geometric_local(&FgKind::Texcoord);
                        ParamRef::Local(idx)
                    });
                let center = self
                    .input_value_param(node, "center", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(0.5));
                let l = self
                    .input_value_param(node, "valuel", Some(out_color_ty))?
                    .or(self.input_value_param(node, "valuet", Some(out_color_ty))?)
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let r = self
                    .input_value_param(node, "valuer", Some(out_color_ty))?
                    .or(self.input_value_param(node, "valueb", Some(out_color_ty))?)
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let tc_op = self.param_to_operand(&texcoord);
                let c_op = self.param_to_operand(&center);
                let l_op = self.param_to_operand(&l);
                let r_op = self.param_to_operand(&r);
                let dst = self.alloc_vreg();
                let instr = if category == "splitlr" {
                    Instruction::Splitlr {
                        dst: dst as u16,
                        ty: out_color_ty,
                        texcoord: tc_op,
                        center: c_op,
                        l: l_op,
                        r: r_op,
                    }
                } else {
                    Instruction::Splittb {
                        dst: dst as u16,
                        ty: out_color_ty,
                        texcoord: tc_op,
                        center: c_op,
                        t: l_op,
                        b: r_op,
                    }
                };
                self.instructions.push(instr);
                Ok(dst)
            }
            "range" => {
                let v = self
                    .input_value_param(node, "in", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let inlow = self
                    .input_value_param(node, "inlow", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let inhigh = self
                    .input_value_param(node, "inhigh", Some(out_color_ty))?
                    .unwrap_or(one_param(Some(out_color_ty)));
                let gamma = self
                    .input_value_param(node, "gamma", Some(out_color_ty))?
                    .unwrap_or(one_param(Some(out_color_ty)));
                let outlow = self
                    .input_value_param(node, "outlow", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let outhigh = self
                    .input_value_param(node, "outhigh", Some(out_color_ty))?
                    .unwrap_or(one_param(Some(out_color_ty)));
                let doclamp = Self::input_static_bool(node, category, "doclamp")?.unwrap_or(false);
                let ops = [
                    self.param_to_operand(&v),
                    self.param_to_operand(&inlow),
                    self.param_to_operand(&inhigh),
                    self.param_to_operand(&gamma),
                    self.param_to_operand(&outlow),
                    self.param_to_operand(&outhigh),
                ];
                let operands_start = self.push_operands(&ops);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Range {
                    dst: dst as u16,
                    ty: out_color_ty,
                    doclamp,
                    operands_start,
                });
                Ok(dst)
            }
            "remap" => {
                let v = self
                    .input_value_param(node, "in", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let inlow = self
                    .input_value_param(node, "inlow", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let inhigh = self
                    .input_value_param(node, "inhigh", Some(out_color_ty))?
                    .unwrap_or(one_param(Some(out_color_ty)));
                let outlow = self
                    .input_value_param(node, "outlow", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let outhigh = self
                    .input_value_param(node, "outhigh", Some(out_color_ty))?
                    .unwrap_or(one_param(Some(out_color_ty)));
                let ops = [
                    self.param_to_operand(&v),
                    self.param_to_operand(&inlow),
                    self.param_to_operand(&inhigh),
                    self.param_to_operand(&outlow),
                    self.param_to_operand(&outhigh),
                ];
                let operands_start = self.push_operands(&ops);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Remap {
                    dst: dst as u16,
                    ty: out_color_ty,
                    operands_start,
                });
                Ok(dst)
            }
            "contrast" => {
                let v = self
                    .input_value_param(node, "in", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let amount = self
                    .input_value_param(node, "amount", Some(out_color_ty))?
                    .unwrap_or(one_param(Some(out_color_ty)));
                let pivot = self
                    .input_value_param(node, "pivot", Some(out_color_ty))?
                    .unwrap_or(ParamRef::Float(0.5));
                let v_op = self.param_to_operand(&v);
                let a_op = self.param_to_operand(&amount);
                let p_op = self.param_to_operand(&pivot);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Contrast {
                    dst: dst as u16,
                    ty: out_color_ty,
                    v: v_op,
                    amount: a_op,
                    pivot: p_op,
                });
                Ok(dst)
            }
            "hsvadjust" => {
                let c = self
                    .input_value_param(node, "in", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let amount = self
                    .input_value_param(node, "amount", Some(ValueType::Vector3))?
                    .unwrap_or(ParamRef::Vector3(Vec3::new(0.0, 1.0, 1.0)));
                let c_op = self.param_to_operand(&c);
                let a_op = self.param_to_operand(&amount);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::HsvAdjust {
                    dst: dst as u16,
                    ty: out_color_ty,
                    c: c_op,
                    amount: a_op,
                });
                Ok(dst)
            }
            "saturate" => {
                let c = self
                    .input_value_param(node, "in", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let amount = self
                    .input_value_param(node, "amount", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(1.0));
                let lumacoeffs = self
                    .input_value_param(node, "lumacoeffs", Some(ValueType::Color3))?
                    .unwrap_or(ParamRef::Color3(Vec3::new(0.2722287, 0.6740818, 0.0536895)));
                let c_op = self.param_to_operand(&c);
                let a_op = self.param_to_operand(&amount);
                let l_op = self.param_to_operand(&lumacoeffs);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Saturate {
                    dst: dst as u16,
                    ty: out_color_ty,
                    c: c_op,
                    amount: a_op,
                    lumacoeffs: l_op,
                });
                Ok(dst)
            }
            "srgb_texture" | "linear" | "g22" | "lin_rec709" => {
                let v = self
                    .input_value_param(node, "in", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let op = self.color_xform_to_rendering(category)?;
                let v_op = self.param_to_operand(&v);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::TransformColor {
                    dst: dst as u16,
                    op,
                    ty: out_color_ty,
                    src: v_op,
                });
                Ok(dst)
            }
            "reflect" => {
                let i = self
                    .input_value_param(node, "in", Some(ValueType::Vector3))?
                    .unwrap_or(ParamRef::Vector3(Vec3::X));
                let n = self
                    .input_value_param(node, "normal", Some(ValueType::Vector3))?
                    .unwrap_or_else(|| {
                        let idx = self
                            .ensure_geometric_kind_local(GeometricKind::Normal(GeomSpace::World));
                        ParamRef::Local(idx)
                    });
                let i_op = self.param_to_operand(&i);
                let n_op = self.param_to_operand(&n);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Reflect {
                    dst: dst as u16,
                    i: i_op,
                    n: n_op,
                });
                Ok(dst)
            }
            "refract" => {
                let i = self
                    .input_value_param(node, "in", Some(ValueType::Vector3))?
                    .unwrap_or(ParamRef::Vector3(Vec3::X));
                let n = self
                    .input_value_param(node, "normal", Some(ValueType::Vector3))?
                    .unwrap_or_else(|| {
                        let idx = self
                            .ensure_geometric_kind_local(GeometricKind::Normal(GeomSpace::World));
                        ParamRef::Local(idx)
                    });
                let eta = self
                    .input_value_param(node, "ior", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(1.0));
                let i_op = self.param_to_operand(&i);
                let n_op = self.param_to_operand(&n);
                let e_op = self.param_to_operand(&eta);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Refract {
                    dst: dst as u16,
                    i: i_op,
                    n: n_op,
                    eta: e_op,
                });
                Ok(dst)
            }
            "dotproduct" => {
                let in_ty = Self::input_declared_value_type(node, "in1", ValueType::Vector3);
                let a = self
                    .input_value_param(node, "in1", Some(in_ty))?
                    .unwrap_or(zero_param(Some(in_ty)));
                let b = self
                    .input_value_param(node, "in2", Some(in_ty))?
                    .unwrap_or(zero_param(Some(in_ty)));
                let a_op = self.param_to_operand(&a);
                let b_op = self.param_to_operand(&b);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::DotProduct {
                    dst: dst as u16,
                    ty: in_ty,
                    a: a_op,
                    b: b_op,
                });
                Ok(dst)
            }
            "crossproduct" => {
                let a = self
                    .input_value_param(node, "in1", Some(ValueType::Vector3))?
                    .unwrap_or(ParamRef::Vector3(Vec3::ZERO));
                let b = self
                    .input_value_param(node, "in2", Some(ValueType::Vector3))?
                    .unwrap_or(ParamRef::Vector3(Vec3::ZERO));
                let a_op = self.param_to_operand(&a);
                let b_op = self.param_to_operand(&b);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::CrossProduct {
                    dst: dst as u16,
                    a: a_op,
                    b: b_op,
                });
                Ok(dst)
            }
            "distance" => {
                let in_ty = Self::input_declared_value_type(node, "in1", ValueType::Vector3);
                let a = self
                    .input_value_param(node, "in1", Some(in_ty))?
                    .unwrap_or(zero_param(Some(in_ty)));
                let b = self
                    .input_value_param(node, "in2", Some(in_ty))?
                    .unwrap_or(zero_param(Some(in_ty)));
                let a_op = self.param_to_operand(&a);
                let b_op = self.param_to_operand(&b);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Distance {
                    dst: dst as u16,
                    ty: in_ty,
                    a: a_op,
                    b: b_op,
                });
                Ok(dst)
            }
            "rotate2d" => {
                let v = self
                    .input_value_param(node, "in", Some(ValueType::Vector2))?
                    .unwrap_or(ParamRef::Vector2(Vec2::ZERO));
                let amount = self
                    .input_value_param(node, "amount", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(0.0));
                let v_op = self.param_to_operand(&v);
                let a_op = self.param_to_operand(&amount);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Rotate2d {
                    dst: dst as u16,
                    v: v_op,
                    amount: a_op,
                });
                Ok(dst)
            }
            "rotate3d" => {
                let v = self
                    .input_value_param(node, "in", Some(ValueType::Vector3))?
                    .unwrap_or(ParamRef::Vector3(Vec3::ZERO));
                let axis = self
                    .input_value_param(node, "axis", Some(ValueType::Vector3))?
                    .unwrap_or(ParamRef::Vector3(Vec3::Y));
                let amount = self
                    .input_value_param(node, "amount", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(0.0));
                let v_op = self.param_to_operand(&v);
                let x_op = self.param_to_operand(&axis);
                let a_op = self.param_to_operand(&amount);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Rotate3d {
                    dst: dst as u16,
                    v: v_op,
                    axis: x_op,
                    amount: a_op,
                });
                Ok(dst)
            }
            "hextiledimage" => {
                let file = Self::input_static_string(node, category, "file")?
                    .unwrap_or("")
                    .to_string();
                let cs = Self::image_color_space(node, "file");
                let default_color =
                    match self.input_value_param(node, "default", Some(out_color_ty))? {
                        Some(ParamRef::Color3(c)) | Some(ParamRef::Vector3(c)) => {
                            Vec4::new(c.x, c.y, c.z, 0.0)
                        }
                        Some(ParamRef::Color4(c)) | Some(ParamRef::Vector4(c)) => c,
                        _ => Vec4::ZERO,
                    };
                let texcoord = self
                    .input_value_param(node, "texcoord", Some(ValueType::Vector2))?
                    .unwrap_or_else(|| {
                        let idx = self.ensure_geometric_local(&FgKind::Texcoord);
                        ParamRef::Local(idx)
                    });
                let tiling = self
                    .input_value_param(node, "tiling", Some(ValueType::Vector2))?
                    .unwrap_or(ParamRef::Vector2(Vec2::ONE));
                let rotation = self
                    .input_value_param(node, "rotation", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(1.0));
                let rotation_range = self
                    .input_value_param(node, "rotationrange", Some(ValueType::Vector2))?
                    .unwrap_or(ParamRef::Vector2(Vec2::new(0.0, 360.0)));
                let scale = self
                    .input_value_param(node, "scale", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(1.0));
                let scale_range = self
                    .input_value_param(node, "scalerange", Some(ValueType::Vector2))?
                    .unwrap_or(ParamRef::Vector2(Vec2::new(0.5, 2.0)));
                let offset = self
                    .input_value_param(node, "offset", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(1.0));
                let offset_range = self
                    .input_value_param(node, "offsetrange", Some(ValueType::Vector2))?
                    .unwrap_or(ParamRef::Vector2(Vec2::new(0.0, 1.0)));
                let falloff = self
                    .input_value_param(node, "falloff", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(0.5));
                let falloff_contrast = self
                    .input_value_param(node, "falloffcontrast", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(0.5));
                let lumacoeffs = self
                    .input_value_param(node, "lumacoeffs", Some(ValueType::Color3))?
                    .unwrap_or(ParamRef::Color3(Vec3::new(0.2722287, 0.6740818, 0.0536895)));
                let ops = [
                    self.param_to_operand(&texcoord),
                    self.param_to_operand(&tiling),
                    self.param_to_operand(&rotation),
                    self.param_to_operand(&rotation_range),
                    self.param_to_operand(&scale),
                    self.param_to_operand(&scale_range),
                    self.param_to_operand(&offset),
                    self.param_to_operand(&offset_range),
                    self.param_to_operand(&falloff),
                    self.param_to_operand(&falloff_contrast),
                    self.param_to_operand(&lumacoeffs),
                ];
                let operands_start = self.push_operands(&ops);
                let texture = self.lookup_texture(&file, out_color_ty);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::HextiledImage {
                    dst: dst as u16,
                    texture,
                    output: out_color_ty,
                    default_color,
                    color_space: cs,
                    operands_start,
                });
                Ok(dst)
            }
            "hextilednormalmap" => {
                let file = Self::input_static_string(node, category, "file")?
                    .unwrap_or("")
                    .to_string();
                let flip_g = Self::input_static_bool(node, category, "flip_g")?.unwrap_or(false);
                let texcoord = self
                    .input_value_param(node, "texcoord", Some(ValueType::Vector2))?
                    .unwrap_or_else(|| {
                        let idx = self.ensure_geometric_local(&FgKind::Texcoord);
                        ParamRef::Local(idx)
                    });
                let tiling = self
                    .input_value_param(node, "tiling", Some(ValueType::Vector2))?
                    .unwrap_or(ParamRef::Vector2(Vec2::ONE));
                let rotation = self
                    .input_value_param(node, "rotation", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(1.0));
                let rotation_range = self
                    .input_value_param(node, "rotationrange", Some(ValueType::Vector2))?
                    .unwrap_or(ParamRef::Vector2(Vec2::new(0.0, 360.0)));
                let scale = self
                    .input_value_param(node, "scale", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(1.0));
                let scale_range = self
                    .input_value_param(node, "scalerange", Some(ValueType::Vector2))?
                    .unwrap_or(ParamRef::Vector2(Vec2::new(0.5, 2.0)));
                let offset = self
                    .input_value_param(node, "offset", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(1.0));
                let offset_range = self
                    .input_value_param(node, "offsetrange", Some(ValueType::Vector2))?
                    .unwrap_or(ParamRef::Vector2(Vec2::new(0.0, 1.0)));
                let falloff = self
                    .input_value_param(node, "falloff", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(0.5));
                let strength = self
                    .input_value_param(node, "strength", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(1.0));
                let default = self
                    .input_value_param(node, "default", Some(ValueType::Vector3))?
                    .unwrap_or(ParamRef::Vector3(Vec3::new(0.5, 0.5, 1.0)));
                let normal = self
                    .input_value_param(node, "normal", Some(ValueType::Vector3))?
                    .unwrap_or_else(|| {
                        let idx = self
                            .ensure_geometric_kind_local(GeometricKind::Normal(GeomSpace::World));
                        ParamRef::Local(idx)
                    });
                let tangent = self
                    .input_value_param(node, "tangent", Some(ValueType::Vector3))?
                    .unwrap_or_else(|| {
                        let idx = self
                            .ensure_geometric_kind_local(GeometricKind::Tangent(GeomSpace::World));
                        ParamRef::Local(idx)
                    });
                let bitangent = self
                    .input_value_param(node, "bitangent", Some(ValueType::Vector3))?
                    .unwrap_or_else(|| {
                        let idx = self.ensure_geometric_kind_local(GeometricKind::Bitangent(
                            GeomSpace::World,
                        ));
                        ParamRef::Local(idx)
                    });
                let ops = [
                    self.param_to_operand(&texcoord),
                    self.param_to_operand(&tiling),
                    self.param_to_operand(&rotation),
                    self.param_to_operand(&rotation_range),
                    self.param_to_operand(&scale),
                    self.param_to_operand(&scale_range),
                    self.param_to_operand(&offset),
                    self.param_to_operand(&offset_range),
                    self.param_to_operand(&falloff),
                    self.param_to_operand(&strength),
                    self.param_to_operand(&default),
                    self.param_to_operand(&normal),
                    self.param_to_operand(&tangent),
                    self.param_to_operand(&bitangent),
                ];
                let operands_start = self.push_operands(&ops);
                let tex = match self.lookup_texture(&file, ValueType::Color3) {
                    ImageTexture::Color(t) => Some(t),
                    ImageTexture::ColorAlpha { rgb, .. } => Some(rgb),
                    _ => None,
                };
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::HextiledNormalMap {
                    dst: dst as u16,
                    texture: tex,
                    flip_g,
                    operands_start,
                });
                Ok(dst)
            }
            "heighttonormal" => {
                let h = self
                    .input_value_param(node, "in", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(0.0));
                let _scale = self
                    .input_value_param(node, "scale", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(1.0));
                let _texcoord =
                    self.input_value_param(node, "texcoord", Some(ValueType::Vector2))?;
                if !matches!(h, ParamRef::Float(_)) {
                    return Err(CompileError::Unsupported(
                        "heighttonormal requires heightfield derivatives/sample-grid evaluation; dynamic height inputs are not implemented".into(),
                    ));
                }
                let flat = self.param_to_operand(&ParamRef::Vector3(Vec3::new(0.5, 0.5, 1.0)));
                Ok(self.operand_to_vreg(flat))
            }

            "switch" => {
                let which_ty = Self::input_declared_value_type(node, "which", ValueType::Float);
                let which = self
                    .input_value_param(node, "which", Some(which_ty))?
                    .unwrap_or(zero_param(Some(which_ty)));
                let which_op = self.param_to_operand(&which);
                let mut ops = Vec::with_capacity(10);
                for i in 0..10 {
                    let name = format!("in{}", i + 1);
                    let p = self
                        .input_value_param(node, &name, Some(out_color_ty))?
                        .unwrap_or(zero_param(Some(out_color_ty)));
                    ops.push(self.param_to_operand(&p));
                }
                let branches_start = self.push_operands(&ops);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Switch {
                    dst: dst as u16,
                    ty: out_color_ty,
                    which: which_op,
                    branches_start,
                });
                Ok(dst)
            }
            "ifelse" => {
                let in_true = self
                    .input_value_param(node, "in1", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let in_false = self
                    .input_value_param(node, "in2", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let cond = self
                    .input_value_param(node, "cond", Some(ValueType::Boolean))?
                    .or(self.input_value_param(node, "value", Some(ValueType::Boolean))?)
                    .unwrap_or(ParamRef::Bool(false));
                let t_op = self.param_to_operand(&in_true);
                let f_op = self.param_to_operand(&in_false);
                let c_op = self.param_to_operand(&cond);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::IfElse {
                    dst: dst as u16,
                    cond: c_op,
                    in_true: t_op,
                    in_false: f_op,
                });
                Ok(dst)
            }
            "plus" | "minus" | "difference" | "burn" | "dodge" | "screen" | "overlay" => {
                let blend_ty = match out_ty {
                    ValueType::Float | ValueType::Color3 | ValueType::Color4 => out_ty,
                    _ => {
                        return Err(CompileError::Unsupported(format!(
                            "blend node `{}` output type {:?}",
                            category, out_ty
                        )));
                    }
                };
                let op = match category {
                    "plus" => BlendOp::Plus,
                    "minus" => BlendOp::Minus,
                    "difference" => BlendOp::Difference,
                    "burn" => BlendOp::Burn,
                    "dodge" => BlendOp::Dodge,
                    "screen" => BlendOp::Screen,
                    _ => BlendOp::Overlay,
                };
                let bg = self
                    .input_value_param(node, "bg", Some(blend_ty))?
                    .unwrap_or(zero_param(Some(blend_ty)));
                let fg = self
                    .input_value_param(node, "fg", Some(blend_ty))?
                    .unwrap_or(zero_param(Some(blend_ty)));
                let mix = self
                    .input_value_param(node, "mix", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(1.0));
                let bg_op = self.param_to_operand(&bg);
                let fg_op = self.param_to_operand(&fg);
                let mix_op = self.param_to_operand(&mix);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Blend {
                    dst: dst as u16,
                    op,
                    ty: blend_ty,
                    bg: bg_op,
                    fg: fg_op,
                    mix: mix_op,
                });
                Ok(dst)
            }
            "disjointover" | "in" | "mask" | "matte" | "out" | "over" => {
                if !matches!(out_ty, ValueType::Color4) {
                    return Err(CompileError::Unsupported(format!(
                        "merge node `{}` output type {:?}",
                        category, out_ty
                    )));
                }
                let op = match category {
                    "disjointover" => MergeOp::Disjointover,
                    "in" => MergeOp::In,
                    "mask" => MergeOp::Mask,
                    "matte" => MergeOp::Matte,
                    "out" => MergeOp::Out,
                    _ => MergeOp::Over,
                };
                let bg = self
                    .input_value_param(node, "bg", Some(ValueType::Color4))?
                    .unwrap_or(ParamRef::Color4(Vec4::ZERO));
                let fg = self
                    .input_value_param(node, "fg", Some(ValueType::Color4))?
                    .unwrap_or(ParamRef::Color4(Vec4::ZERO));
                let mix = self
                    .input_value_param(node, "mix", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(1.0));
                let bg_op = self.param_to_operand(&bg);
                let fg_op = self.param_to_operand(&fg);
                let mix_op = self.param_to_operand(&mix);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Merge {
                    dst: dst as u16,
                    op,
                    bg: bg_op,
                    fg: fg_op,
                    mix: mix_op,
                });
                Ok(dst)
            }
            "inside" | "outside" => {
                let mask_ty = match out_ty {
                    ValueType::Float | ValueType::Color3 | ValueType::Color4 => out_ty,
                    _ => {
                        return Err(CompileError::Unsupported(format!(
                            "mask node `{}` output type {:?}",
                            category, out_ty
                        )));
                    }
                };
                let op = if category == "inside" {
                    MaskOp::Inside
                } else {
                    MaskOp::Outside
                };
                let default_mask = if category == "inside" { 1.0 } else { 0.0 };
                let v = self
                    .input_value_param(node, "in", Some(mask_ty))?
                    .unwrap_or(zero_param(Some(mask_ty)));
                let mask = self
                    .input_value_param(node, "mask", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(default_mask));
                let v_op = self.param_to_operand(&v);
                let m_op = self.param_to_operand(&mask);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Mask {
                    dst: dst as u16,
                    op,
                    ty: mask_ty,
                    v: v_op,
                    mask: m_op,
                });
                Ok(dst)
            }
            "transformmatrix" => {
                let out_vec_ty = match out_ty {
                    ValueType::Vector2 | ValueType::Vector3 | ValueType::Vector4 => out_ty,
                    _ => {
                        return Err(CompileError::Unsupported(format!(
                            "transformmatrix output type {:?}",
                            out_ty
                        )));
                    }
                };
                let declared_mat_ty =
                    Self::input_declared_value_type(node, "mat", ValueType::Matrix33);
                let mat_is_44 = match Self::input_binding(node, "mat") {
                    Some(FlatInput::Value(MtlxValue::Matrix44(_))) => true,
                    Some(FlatInput::Value(MtlxValue::Matrix33(_))) => false,
                    Some(FlatInput::Node { node: id, .. }) => matches!(
                        ValueType::from_mtlx(&self.graph.nodes[*id as usize].output_type),
                        Some(ValueType::Matrix44)
                    ),
                    _ => matches!(declared_mat_ty, ValueType::Matrix44),
                };
                if (matches!(out_vec_ty, ValueType::Vector2) && mat_is_44)
                    || (matches!(out_vec_ty, ValueType::Vector4) && !mat_is_44)
                {
                    return Err(CompileError::Unsupported(format!(
                        "transformmatrix output {:?} cannot use {}",
                        out_vec_ty,
                        if mat_is_44 { "matrix44" } else { "matrix33" }
                    )));
                }
                let v = self
                    .input_value_param(node, "in", Some(out_vec_ty))?
                    .unwrap_or(zero_param(Some(out_vec_ty)));
                let mat_ty = if mat_is_44 {
                    ValueType::Matrix44
                } else {
                    ValueType::Matrix33
                };
                let mat = self
                    .input_value_param(node, "mat", Some(mat_ty))?
                    .unwrap_or(if mat_is_44 {
                        ParamRef::Matrix44(Mat4::IDENTITY)
                    } else {
                        ParamRef::Matrix33(Mat3::IDENTITY)
                    });
                let v_op = self.param_to_operand(&v);
                let m_op = self.param_to_operand(&mat);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::TransformMatrix {
                    dst: dst as u16,
                    out_ty: out_vec_ty,
                    dim4: mat_is_44,
                    mat: m_op,
                    v: v_op,
                });
                Ok(dst)
            }
            "transformcolor" => {
                let v = self
                    .input_value_param(node, "in", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let from = Self::input_static_string(node, category, "fromspace")?.unwrap_or("");
                let to = Self::input_static_string(node, category, "tospace")?.unwrap_or("");
                let op = self.color_xform_between(from, to)?;
                let v_op = self.param_to_operand(&v);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::TransformColor {
                    dst: dst as u16,
                    op,
                    ty: out_color_ty,
                    src: v_op,
                });
                Ok(dst)
            }
            "transpose" => {
                let dim4 = matches!(out_ty, ValueType::Matrix44);
                let mat_ty = if dim4 {
                    ValueType::Matrix44
                } else {
                    ValueType::Matrix33
                };
                let v = self
                    .input_value_param(node, "in", Some(mat_ty))?
                    .unwrap_or(if dim4 {
                        ParamRef::Matrix44(Mat4::IDENTITY)
                    } else {
                        ParamRef::Matrix33(Mat3::IDENTITY)
                    });
                let v_op = self.param_to_operand(&v);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Transpose {
                    dst: dst as u16,
                    dim4,
                    src: v_op,
                });
                Ok(dst)
            }
            "determinant" => {
                let declared_in_ty =
                    Self::input_declared_value_type(node, "in", ValueType::Matrix33);
                let dim4 = match Self::input_binding(node, "in") {
                    Some(FlatInput::Node { node: id, .. }) => matches!(
                        ValueType::from_mtlx(&self.graph.nodes[*id as usize].output_type),
                        Some(ValueType::Matrix44)
                    ),
                    Some(FlatInput::Value(MtlxValue::Matrix44(_))) => true,
                    _ => matches!(declared_in_ty, ValueType::Matrix44),
                };
                let mat_ty = if dim4 {
                    ValueType::Matrix44
                } else {
                    ValueType::Matrix33
                };
                let v = self
                    .input_value_param(node, "in", Some(mat_ty))?
                    .unwrap_or(if dim4 {
                        ParamRef::Matrix44(Mat4::IDENTITY)
                    } else {
                        ParamRef::Matrix33(Mat3::IDENTITY)
                    });
                let v_op = self.param_to_operand(&v);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Determinant {
                    dst: dst as u16,
                    dim4,
                    src: v_op,
                });
                Ok(dst)
            }
            "invertmatrix" => {
                let dim4 = matches!(out_ty, ValueType::Matrix44);
                let mat_ty = if dim4 {
                    ValueType::Matrix44
                } else {
                    ValueType::Matrix33
                };
                let v = self
                    .input_value_param(node, "in", Some(mat_ty))?
                    .unwrap_or(if dim4 {
                        ParamRef::Matrix44(Mat4::IDENTITY)
                    } else {
                        ParamRef::Matrix33(Mat3::IDENTITY)
                    });
                let v_op = self.param_to_operand(&v);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::InvertMatrix {
                    dst: dst as u16,
                    dim4,
                    src: v_op,
                });
                Ok(dst)
            }
            "creatematrix" => self.emit_creatematrix_pattern(node, out_ty),
            "separate2" => {
                let idx = match _output {
                    Some("outx") => 0,
                    Some("outy") => 1,
                    Some(name) => {
                        return Err(CompileError::Unsupported(format!(
                            "separate2 output `{}` is not defined",
                            name
                        )));
                    }
                    None => {
                        return Err(CompileError::Unsupported(
                            "separate2 requires output `outx` or `outy`".into(),
                        ));
                    }
                };
                let in_v = self
                    .input_value_param(node, "in", Some(ValueType::Vector2))?
                    .unwrap_or(ParamRef::Vector2(Vec2::ZERO));
                let v_op = self.param_to_operand(&in_v);
                let i_op = self.intern_value(Value::Integer(idx));
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Extract {
                    dst: dst as u16,
                    in_ty: ValueType::Vector2,
                    src: v_op,
                    idx: i_op,
                });
                Ok(dst)
            }
            "separate3" => {
                let in_ty = Self::input_declared_value_type(node, "in", ValueType::Color3);
                let idx = match (in_ty, _output) {
                    (ValueType::Color3, Some("outr")) | (ValueType::Vector3, Some("outx")) => 0,
                    (ValueType::Color3, Some("outg")) | (ValueType::Vector3, Some("outy")) => 1,
                    (ValueType::Color3, Some("outb")) | (ValueType::Vector3, Some("outz")) => 2,
                    (_, Some(name)) => {
                        return Err(CompileError::Unsupported(format!(
                            "separate3 output `{}` is not defined",
                            name
                        )));
                    }
                    (_, None) => {
                        return Err(CompileError::Unsupported(
                            "separate3 requires an output channel name".into(),
                        ));
                    }
                };
                let in_v = self
                    .input_value_param(node, "in", Some(in_ty))?
                    .unwrap_or(zero_param(Some(in_ty)));
                let v_op = self.param_to_operand(&in_v);
                let i_op = self.intern_value(Value::Integer(idx));
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Extract {
                    dst: dst as u16,
                    in_ty,
                    src: v_op,
                    idx: i_op,
                });
                Ok(dst)
            }
            "separate4" | "separatecolor4" => {
                let in_ty = Self::input_declared_value_type(node, "in", ValueType::Color4);
                let idx = match (in_ty, _output) {
                    (ValueType::Color4, Some("outr")) | (ValueType::Vector4, Some("outx")) => 0,
                    (ValueType::Color4, Some("outg")) | (ValueType::Vector4, Some("outy")) => 1,
                    (ValueType::Color4, Some("outb")) | (ValueType::Vector4, Some("outz")) => 2,
                    (ValueType::Color4, Some("outa")) | (ValueType::Vector4, Some("outw")) => 3,
                    (_, Some(name)) => {
                        return Err(CompileError::Unsupported(format!(
                            "{} output `{}` is not defined",
                            category, name
                        )));
                    }
                    (_, None) => {
                        return Err(CompileError::Unsupported(format!(
                            "{} requires an output channel name",
                            category
                        )));
                    }
                };
                let in_v = self
                    .input_value_param(node, "in", Some(in_ty))?
                    .unwrap_or(zero_param(Some(in_ty)));
                let v_op = self.param_to_operand(&in_v);
                let i_op = self.intern_value(Value::Integer(idx));
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Extract {
                    dst: dst as u16,
                    in_ty,
                    src: v_op,
                    idx: i_op,
                });
                Ok(dst)
            }
            "checkerboard" => {
                let color1 = self
                    .input_value_param(node, "color1", Some(ValueType::Color3))?
                    .unwrap_or(ParamRef::Color3(Vec3::ONE));
                let color2 = self
                    .input_value_param(node, "color2", Some(ValueType::Color3))?
                    .unwrap_or(ParamRef::Color3(Vec3::ZERO));
                let uvtiling = self
                    .input_value_param(node, "uvtiling", Some(ValueType::Vector2))?
                    .unwrap_or(ParamRef::Vector2(Vec2::splat(8.0)));
                let uvoffset = self
                    .input_value_param(node, "uvoffset", Some(ValueType::Vector2))?
                    .unwrap_or(ParamRef::Vector2(Vec2::ZERO));
                let texcoord = self
                    .input_value_param(node, "texcoord", Some(ValueType::Vector2))?
                    .unwrap_or_else(|| {
                        let idx = self.ensure_geometric_local(&FgKind::Texcoord);
                        ParamRef::Local(idx)
                    });
                let c1_op = self.param_to_operand(&color1);
                let c2_op = self.param_to_operand(&color2);
                let ut_op = self.param_to_operand(&uvtiling);
                let uo_op = self.param_to_operand(&uvoffset);
                let tc_op = self.param_to_operand(&texcoord);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Checkerboard {
                    dst: dst as u16,
                    color1: c1_op,
                    color2: c2_op,
                    uvtiling: ut_op,
                    uvoffset: uo_op,
                    texcoord: tc_op,
                });
                Ok(dst)
            }
            "colorcorrect" => {
                let v = self
                    .input_value_param(node, "in", Some(out_color_ty))?
                    .unwrap_or(if matches!(out_color_ty, ValueType::Color4) {
                        ParamRef::Color4(Vec4::new(1.0, 1.0, 1.0, 0.0))
                    } else {
                        ParamRef::Color3(Vec3::ONE)
                    });
                let names = [
                    "hue",
                    "saturation",
                    "gamma",
                    "lift",
                    "gain",
                    "contrast",
                    "contrastpivot",
                    "exposure",
                ];
                let defaults = [0.0_f32, 1.0, 1.0, 0.0, 1.0, 1.0, 0.5, 0.0];
                let mut ops = Vec::with_capacity(9);
                ops.push(self.param_to_operand(&v));
                for (n, d) in names.iter().zip(defaults.iter()) {
                    let p = self
                        .input_value_param(node, n, Some(ValueType::Float))?
                        .unwrap_or(ParamRef::Float(*d));
                    ops.push(self.param_to_operand(&p));
                }
                let operands_start = self.push_operands(&ops);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::ColorCorrect {
                    dst: dst as u16,
                    ty: out_color_ty,
                    operands_start,
                });
                Ok(dst)
            }
            "randomfloat" => {
                let input_ty = Self::input_declared_value_type(node, "in", ValueType::Float);
                let v = self
                    .input_value_param(node, "in", Some(input_ty))?
                    .unwrap_or(zero_param(Some(input_ty)));
                let lo = self
                    .input_value_param(node, "min", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(0.0));
                let hi = self
                    .input_value_param(node, "max", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(1.0));
                let seed = self
                    .input_value_param(node, "seed", Some(ValueType::Integer))?
                    .unwrap_or(ParamRef::Integer(0));
                let ops = [
                    self.param_to_operand(&v),
                    self.param_to_operand(&seed),
                    self.param_to_operand(&lo),
                    self.param_to_operand(&hi),
                ];
                let operands_start = self.push_operands(&ops);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::RandomFloat {
                    dst: dst as u16,
                    integer_input: input_ty == ValueType::Integer,
                    operands_start,
                });
                Ok(dst)
            }
            "randomcolor" => {
                let v = self
                    .input_value_param(node, "in", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(0.0));
                let seed = self
                    .input_value_param(node, "seed", Some(ValueType::Integer))?
                    .unwrap_or(ParamRef::Integer(0));
                let names = [
                    "huelow",
                    "huehigh",
                    "saturationlow",
                    "saturationhigh",
                    "brightnesslow",
                    "brightnesshigh",
                ];
                let defaults = [0.0_f32, 1.0, 0.825, 1.0, 1.0, 1.0];
                let mut ops = Vec::with_capacity(8);
                ops.push(self.param_to_operand(&v));
                ops.push(self.param_to_operand(&seed));
                for (n, d) in names.iter().zip(defaults.iter()) {
                    let p = self
                        .input_value_param(node, n, Some(ValueType::Float))?
                        .unwrap_or(ParamRef::Float(*d));
                    ops.push(self.param_to_operand(&p));
                }
                let operands_start = self.push_operands(&ops);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::RandomColor {
                    dst: dst as u16,
                    operands_start,
                });
                Ok(dst)
            }
            "cellnoise1d" => {
                let in_v = self
                    .input_value_param(node, "in", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(0.0));
                let in_op = self.param_to_operand(&in_v);
                let zero = self.intern_value(Value::Float(0.0));
                let combine_start = self.push_operands(&[in_op, zero]);
                let texcoord_vreg = self.alloc_vreg();
                self.instructions.push(Instruction::Combine {
                    dst: texcoord_vreg as u16,
                    kind: CombineKind::Vector2FromFloats,
                    operands_start: combine_start,
                });
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Cellnoise {
                    dst: dst as u16,
                    dim3: false,
                    output: NoiseOutput::Float,
                    coord: Operand::Reg(texcoord_vreg as u16),
                });
                Ok(dst)
            }
            "triplanarblend" => {
                let inx = self
                    .input_value_param(node, "inx", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let iny = self
                    .input_value_param(node, "iny", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let inz = self
                    .input_value_param(node, "inz", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let normal = self
                    .input_value_param(node, "normal", Some(ValueType::Vector3))?
                    .unwrap_or_else(|| {
                        let idx = self.ensure_geometric_local(&FgKind::Normal);
                        ParamRef::Local(idx)
                    });
                let blend = self
                    .input_value_param(node, "blend", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(1.0));
                let filter = match Self::input_static_string(node, category, "filtertype")? {
                    None | Some("") | Some("linear") => TriplanarFilter::Linear,
                    Some("cubic") => {
                        tracing::warn!(
                            "warning: triplanarblend.filtertype=`cubic` is not implemented; using linear filtering"
                        );
                        TriplanarFilter::Linear
                    }
                    Some("closest") => TriplanarFilter::Closest,
                    Some(other) => {
                        return Err(CompileError::Unsupported(format!(
                            "triplanarblend.filtertype `{}`",
                            other
                        )));
                    }
                };
                let ops = [
                    self.param_to_operand(&inx),
                    self.param_to_operand(&iny),
                    self.param_to_operand(&inz),
                    self.param_to_operand(&normal),
                    self.param_to_operand(&blend),
                ];
                let operands_start = self.push_operands(&ops);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::TriplanarBlend {
                    dst: dst as u16,
                    ty: out_color_ty,
                    filter,
                    operands_start,
                });
                Ok(dst)
            }
            "curveuniformlinear" | "curveuniformcubic" => {
                let v = self
                    .input_value_param(node, "in", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(0.0));
                let kv = match Self::input_binding(node, "knotvalues") {
                    Some(FlatInput::Value(MtlxValue::FloatArray(arr))) => arr.clone(),
                    _ => {
                        return Err(CompileError::Unsupported(format!(
                            "{}.knotvalues must be an inline floatarray",
                            category
                        )));
                    }
                };
                if kv.len() < 2 {
                    return Err(CompileError::Unsupported(format!(
                        "{}.knotvalues must contain at least 2 values",
                        category
                    )));
                }
                let arc = std::sync::Arc::new(kv);
                let v_op = self.param_to_operand(&v);
                let dst = self.alloc_vreg();
                let instr = if category == "curveuniformlinear" {
                    Instruction::CurveUniformLinear {
                        dst: dst as u16,
                        knotvalues: arc,
                        t: v_op,
                    }
                } else {
                    Instruction::CurveUniformCubic {
                        dst: dst as u16,
                        knotvalues: arc,
                        t: v_op,
                    }
                };
                self.instructions.push(instr);
                Ok(dst)
            }
            "curveinversecubic" => {
                let v = self
                    .input_value_param(node, "in", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(0.0));
                let knots = match Self::input_binding(node, "knots") {
                    Some(FlatInput::Value(MtlxValue::FloatArray(arr))) => arr.clone(),
                    _ => {
                        return Err(CompileError::Unsupported(
                            "curveinversecubic.knots must be inline floatarray".into(),
                        ));
                    }
                };
                if knots.len() < 2 {
                    return Err(CompileError::Unsupported(
                        "curveinversecubic.knots must contain at least 2 values".into(),
                    ));
                }
                let v_op = self.param_to_operand(&v);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::CurveInverseCubic {
                    dst: dst as u16,
                    knots: std::sync::Arc::new(knots),
                    x: v_op,
                });
                Ok(dst)
            }
            "chiang_hair_roughness" => {
                let which = match _output {
                    Some("roughness_R") => ChiangHairRoughnessOutput::R,
                    Some("roughness_TT") => ChiangHairRoughnessOutput::TT,
                    Some("roughness_TRT") => ChiangHairRoughnessOutput::TRT,
                    Some(name) => {
                        return Err(CompileError::Unsupported(format!(
                            "chiang_hair_roughness output `{}` is not defined",
                            name
                        )));
                    }
                    None => {
                        return Err(CompileError::Unsupported(
                            "chiang_hair_roughness requires a roughness output name".into(),
                        ));
                    }
                };
                let longi = self
                    .input_value_param(node, "longitudinal", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(0.1));
                let azim = self
                    .input_value_param(node, "azimuthal", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(0.2));
                let scale_tt = self
                    .input_value_param(node, "scale_TT", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(0.5));
                let scale_trt = self
                    .input_value_param(node, "scale_TRT", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(2.0));
                let l_op = self.param_to_operand(&longi);
                let a_op = self.param_to_operand(&azim);
                let s_tt_op = self.param_to_operand(&scale_tt);
                let s_trt_op = self.param_to_operand(&scale_trt);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::ChiangHairRoughness {
                    dst: dst as u16,
                    which,
                    longitudinal: l_op,
                    azimuthal: a_op,
                    scale_tt: s_tt_op,
                    scale_trt: s_trt_op,
                });
                Ok(dst)
            }
            "chiang_hair_absorption_from_color" => {
                let color = self
                    .input_value_param(node, "color", Some(ValueType::Color3))?
                    .unwrap_or(ParamRef::Color3(Vec3::ONE));
                let beta = self
                    .input_value_param(node, "azimuthal_roughness", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(0.2));
                let c_op = self.param_to_operand(&color);
                let b_op = self.param_to_operand(&beta);
                let dst = self.alloc_vreg();
                self.instructions
                    .push(Instruction::ChiangHairAbsorptionFromColor {
                        dst: dst as u16,
                        color: c_op,
                        beta: b_op,
                    });
                Ok(dst)
            }
            "deon_hair_absorption_from_melanin" => {
                let conc = self
                    .input_value_param(node, "melanin_concentration", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(0.25));
                let redness = self
                    .input_value_param(node, "melanin_redness", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(0.5));
                let eum = self
                    .input_value_param(node, "eumelanin_color", Some(ValueType::Color3))?
                    .unwrap_or(ParamRef::Color3(Vec3::new(0.657704, 0.498077, 0.254107)));
                let phe = self
                    .input_value_param(node, "pheomelanin_color", Some(ValueType::Color3))?
                    .unwrap_or(ParamRef::Color3(Vec3::new(0.829444, 0.67032, 0.349938)));
                let ops = [
                    self.param_to_operand(&conc),
                    self.param_to_operand(&redness),
                    self.param_to_operand(&eum),
                    self.param_to_operand(&phe),
                ];
                let operands_start = self.push_operands(&ops);
                let dst = self.alloc_vreg();
                self.instructions
                    .push(Instruction::DeonHairAbsorptionFromMelanin {
                        dst: dst as u16,
                        operands_start,
                    });
                Ok(dst)
            }
            "tokenvalue" => {
                let p = self
                    .input_value_param(node, "value", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let op = self.param_to_operand(&p);
                Ok(self.operand_to_vreg(op))
            }
            "extractrowvector" => {
                let dim4 = match out_ty {
                    ValueType::Vector3 => false,
                    ValueType::Vector4 => true,
                    _ => {
                        return Err(CompileError::Unsupported(format!(
                            "extractrowvector output type {:?}",
                            out_ty
                        )));
                    }
                };
                let matrix_ty = if dim4 {
                    ValueType::Matrix44
                } else {
                    ValueType::Matrix33
                };
                let matrix_default = if dim4 {
                    ParamRef::Matrix44(Mat4::IDENTITY)
                } else {
                    ParamRef::Matrix33(Mat3::IDENTITY)
                };
                let matrix = self
                    .input_value_param(node, "in", Some(matrix_ty))?
                    .unwrap_or(matrix_default);
                let index = self
                    .input_value_param(node, "index", Some(ValueType::Integer))?
                    .unwrap_or(ParamRef::Integer(0));
                let ParamRef::Integer(index) = index else {
                    return Err(CompileError::Unsupported(
                        "extractrowvector.index must be a static integer".into(),
                    ));
                };
                let max_index = if dim4 { 3 } else { 2 };
                if !(0..=max_index).contains(&index) {
                    return Err(CompileError::Unsupported(format!(
                        "extractrowvector.index `{}` out of range 0..={}",
                        index, max_index
                    )));
                }
                let src = self.param_to_operand(&matrix);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::ExtractRowVector {
                    dst: dst as u16,
                    dim4,
                    src,
                    index: index as u8,
                });
                Ok(dst)
            }
            "unifiednoise2d" | "unifiednoise3d" => {
                let dim3 = category == "unifiednoise3d";
                let type_int = Self::input_static_integer(node, category, "type")?.unwrap_or(0);
                if !(0..=3).contains(&type_int) {
                    return Err(CompileError::Unsupported(format!(
                        "{}.type `{}`: must be 0..3",
                        category, type_int
                    )));
                }
                let style = Self::input_worley_style(node, category)?;
                let coord_ty = if dim3 {
                    ValueType::Vector3
                } else {
                    ValueType::Vector2
                };
                let freq_default = if dim3 {
                    ParamRef::Vector3(Vec3::ONE)
                } else {
                    ParamRef::Vector2(Vec2::ONE)
                };
                let offset_default = if dim3 {
                    ParamRef::Vector3(Vec3::ZERO)
                } else {
                    ParamRef::Vector2(Vec2::ZERO)
                };
                let raw_coord = if dim3 {
                    self.input_value_param(node, "position", Some(ValueType::Vector3))?
                        .unwrap_or_else(|| {
                            let idx = self.ensure_geometric_local(&FgKind::Position);
                            ParamRef::Local(idx)
                        })
                } else {
                    self.input_value_param(node, "texcoord", Some(ValueType::Vector2))?
                        .unwrap_or_else(|| {
                            let idx = self.ensure_geometric_local(&FgKind::Texcoord);
                            ParamRef::Local(idx)
                        })
                };
                let freq = self
                    .input_value_param(node, "freq", Some(coord_ty))?
                    .unwrap_or(freq_default);
                let offset = self
                    .input_value_param(node, "offset", Some(coord_ty))?
                    .unwrap_or(offset_default);
                let jitter = self
                    .input_value_param(node, "jitter", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(1.0));
                let octaves = self
                    .input_value_param(node, "octaves", Some(ValueType::Integer))?
                    .unwrap_or(ParamRef::Integer(3));
                let lacunarity = self
                    .input_value_param(node, "lacunarity", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(2.0));
                let diminish = self
                    .input_value_param(node, "diminish", Some(ValueType::Float))?
                    .unwrap_or(ParamRef::Float(0.5));
                let outmin = self
                    .input_value_param(node, "outmin", Some(out_color_ty))?
                    .unwrap_or(zero_param(Some(out_color_ty)));
                let outmax = self
                    .input_value_param(node, "outmax", Some(out_color_ty))?
                    .unwrap_or(one_param(Some(out_color_ty)));
                let clampoutput =
                    Self::input_static_bool(node, category, "clampoutput")?.unwrap_or(true);

                let raw_op = self.param_to_operand(&raw_coord);
                let freq_op = self.param_to_operand(&freq);
                let coord_mul = self.emit_arith(ArithOp::Multiply, coord_ty, raw_op, freq_op);
                let offset_op = self.param_to_operand(&offset);
                let apply_offset_vreg = self.emit_arith(
                    ArithOp::Add,
                    coord_ty,
                    Operand::Reg(coord_mul as u16),
                    offset_op,
                );

                let j_op = self.param_to_operand(&jitter);
                let one_f = self.intern_value(Value::Float(1.0));
                let jm1 = self.emit_arith(ArithOp::Subtract, ValueType::Float, j_op, one_f);
                let big = self.intern_value(Value::Float(90000.0));
                let cell_jitter_vreg = self.emit_arith(
                    ArithOp::Multiply,
                    ValueType::Float,
                    Operand::Reg(jm1 as u16),
                    big,
                );

                let kind = match (type_int, dim3) {
                    (0, false) => NoiseKind::Perlin2d,
                    (0, true) => NoiseKind::Perlin3d,
                    (1, false) => NoiseKind::Cellnoise2d,
                    (1, true) => NoiseKind::Cellnoise3d,
                    (2, false) => NoiseKind::Worleynoise2d,
                    (2, true) => NoiseKind::Worleynoise3d,
                    (3, false) => NoiseKind::Fractal2d,
                    (3, true) => NoiseKind::Fractal3d,
                    _ => unreachable!("unifiednoise type range is checked above"),
                };
                let output = noise_output_for(out_color_ty);
                let is_cellnoise = matches!(kind, NoiseKind::Cellnoise2d | NoiseKind::Cellnoise3d);
                let is_worley = matches!(kind, NoiseKind::Worleynoise2d | NoiseKind::Worleynoise3d);

                let noise_vreg = if is_cellnoise {
                    let dst = self.alloc_vreg();
                    self.instructions.push(Instruction::Cellnoise {
                        dst: dst as u16,
                        dim3,
                        output,
                        coord: Operand::Reg(apply_offset_vreg as u16),
                    });
                    dst
                } else if is_worley {
                    let ops_s = self.push_operands(&[
                        Operand::Reg(apply_offset_vreg as u16),
                        Operand::Reg(cell_jitter_vreg as u16),
                    ]);
                    let dst = self.alloc_vreg();
                    self.instructions.push(Instruction::Worley {
                        dst: dst as u16,
                        dim3,
                        output,
                        style,
                        operands_start: ops_s,
                    });
                    dst
                } else {
                    let half = self.intern_value(Value::Float(0.5));
                    let oct_op = self.param_to_operand(&octaves);
                    let lac_op = self.param_to_operand(&lacunarity);
                    let dim_op = self.param_to_operand(&diminish);
                    let jit_op = self.param_to_operand(&jitter);
                    let ops_s = self.push_operands(&[
                        Operand::Reg(apply_offset_vreg as u16),
                        half,
                        half,
                        oct_op,
                        lac_op,
                        dim_op,
                        jit_op,
                    ]);
                    let dst = self.alloc_vreg();
                    self.instructions.push(Instruction::Noise {
                        dst: dst as u16,
                        kind,
                        output,
                        operands_start: ops_s,
                    });
                    dst
                };

                let zero_op = self.param_to_operand(&zero_param(Some(out_color_ty)));
                let one_op = self.param_to_operand(&one_param(Some(out_color_ty)));
                let outmin_op = self.param_to_operand(&outmin);
                let outmax_op = self.param_to_operand(&outmax);
                let ops_s = self.push_operands(&[
                    Operand::Reg(noise_vreg as u16),
                    zero_op,
                    one_op,
                    one_op,
                    outmin_op,
                    outmax_op,
                ]);
                let dst = self.alloc_vreg();
                self.instructions.push(Instruction::Range {
                    dst: dst as u16,
                    ty: out_color_ty,
                    doclamp: clampoutput,
                    operands_start: ops_s,
                });
                Ok(dst)
            }
            "volume" => Err(CompileError::Unsupported(
                "volume nodes are not part of surface materials".into(),
            )),
            other => Err(CompileError::Unsupported(format!(
                "SSA emit_pattern: category `{}` not implemented (out_ty={:?})",
                other, out_color_ty
            ))),
        }
    }

    fn lookup_texture(&self, file: &str, output: ValueType) -> ImageTexture {
        if let Some(tiles) = self.udim_textures.get(file) {
            return ImageTexture::Udim {
                tiles: tiles.clone(),
            };
        }
        if matches!(output, ValueType::Float | ValueType::Integer)
            && let Some(t) = self.scalar_textures.get(file)
        {
            return ImageTexture::Scalar(t.clone());
        }
        if let Some(t) = self.color_textures.get(file) {
            if let Some(a) = self.alpha_textures.get(file) {
                return ImageTexture::ColorAlpha {
                    rgb: t.clone(),
                    alpha: a.clone(),
                };
            }
            return ImageTexture::Color(t.clone());
        }
        if let Some(t) = self.scalar_textures.get(file) {
            return ImageTexture::Scalar(t.clone());
        }
        ImageTexture::Missing
    }

    fn combine_kind(
        &self,
        node: &FlatNode,
        category: &str,
        out_ty: ValueType,
    ) -> Result<CombineKind, CompileError> {
        let declared_in1_ty = Self::input_declared_value_type(node, "in1", ValueType::Float);
        let in1_ty = match Self::input_binding(node, "in1") {
            Some(FlatInput::Node { node: id, .. }) => {
                ValueType::from_mtlx(&self.graph.nodes[*id as usize].output_type)
                    .unwrap_or(declared_in1_ty)
            }
            Some(FlatInput::Value(v)) => mtlx_value_type(v).unwrap_or(declared_in1_ty),
            Some(FlatInput::Empty) | Some(FlatInput::String(_)) | Some(FlatInput::GeomProp(_)) => {
                declared_in1_ty
            }
            None => declared_in1_ty,
        };
        Ok(match (category, out_ty, in1_ty) {
            ("combine2", ValueType::Vector2, _) => CombineKind::Vector2FromFloats,
            ("combine2", ValueType::Color4, ValueType::Color3) => {
                CombineKind::Color4FromColor3Float
            }
            ("combine2", ValueType::Vector4, ValueType::Vector3) => {
                CombineKind::Vector4FromVector3Float
            }
            ("combine2", ValueType::Vector4, ValueType::Vector2) => {
                CombineKind::Vector4FromVector2Vector2
            }
            ("combine3", ValueType::Color3, _) => CombineKind::Color3FromFloats,
            ("combine3", ValueType::Vector3, _) => CombineKind::Vector3FromFloats,
            ("combine4", ValueType::Color4, _) => CombineKind::Color4FromFloats,
            ("combine4", ValueType::Vector4, _) => CombineKind::Vector4FromFloats,
            (cat, ty, _) => {
                return Err(CompileError::Unsupported(format!(
                    "combine variant `{}` with output {:?} and in1 {:?}",
                    cat, ty, in1_ty
                )));
            }
        })
    }
}

fn output_index(name: Option<&str>) -> u8 {
    match name {
        None | Some("out") => 0,
        Some("ior") => 1,
        Some("extinction") => 2,
        Some(s) => {
            let mut h: u8 = 0;
            for b in s.bytes() {
                h = h.wrapping_mul(31).wrapping_add(b);
            }
            h.max(3)
        }
    }
}

fn zero_param(ty: Option<ValueType>) -> ParamRef {
    match ty {
        Some(ValueType::Float) | None => ParamRef::Float(0.0),
        Some(ValueType::Integer) => ParamRef::Integer(0),
        Some(ValueType::Boolean) => ParamRef::Bool(false),
        Some(ValueType::Color3) => ParamRef::Color3(Vec3::ZERO),
        Some(ValueType::Color4) => ParamRef::Color4(Vec4::ZERO),
        Some(ValueType::Vector2) => ParamRef::Vector2(Vec2::ZERO),
        Some(ValueType::Vector3) => ParamRef::Vector3(Vec3::ZERO),
        Some(ValueType::Vector4) => ParamRef::Vector4(Vec4::ZERO),
        Some(ValueType::Matrix33) => ParamRef::Matrix33(Mat3::ZERO),
        Some(ValueType::Matrix44) => ParamRef::Matrix44(Mat4::ZERO),
    }
}

fn one_param(ty: Option<ValueType>) -> ParamRef {
    match ty {
        Some(ValueType::Float) | None => ParamRef::Float(1.0),
        Some(ValueType::Integer) => ParamRef::Integer(1),
        Some(ValueType::Color3) => ParamRef::Color3(Vec3::ONE),
        Some(ValueType::Color4) => ParamRef::Color4(Vec4::ONE),
        Some(ValueType::Vector2) => ParamRef::Vector2(Vec2::ONE),
        Some(ValueType::Vector3) => ParamRef::Vector3(Vec3::ONE),
        Some(ValueType::Vector4) => ParamRef::Vector4(Vec4::ONE),
        _ => ParamRef::Float(1.0),
    }
}

fn input_error(name: &str, err: CompileError) -> CompileError {
    match err {
        CompileError::Missing(s) => CompileError::Missing(format!("{}: {}", name, s)),
        CompileError::Unsupported(s) => CompileError::Unsupported(format!("{}: {}", name, s)),
    }
}

fn constant_param(v: &MtlxValue) -> Result<ParamRef, CompileError> {
    Ok(match v {
        MtlxValue::Float(x) => ParamRef::Float(*x),
        MtlxValue::Integer(x) => ParamRef::Integer(*x),
        MtlxValue::Boolean(x) => ParamRef::Bool(*x),
        MtlxValue::Color3(x) => ParamRef::Color3(*x),
        MtlxValue::Color4(x) => ParamRef::Color4(*x),
        MtlxValue::Vector2(x) => ParamRef::Vector2(*x),
        MtlxValue::Vector3(x) => ParamRef::Vector3(*x),
        MtlxValue::Vector4(x) => ParamRef::Vector4(*x),
        MtlxValue::Matrix33(x) => ParamRef::Matrix33(*x),
        MtlxValue::Matrix44(x) => ParamRef::Matrix44(*x),
        other => {
            return Err(CompileError::Unsupported(format!(
                "literal {:?} cannot be used as numeric/vector/matrix parameter",
                other
            )));
        }
    })
}

fn constant_value(v: &MtlxValue) -> Result<Value, CompileError> {
    Ok(match v {
        MtlxValue::Float(x) => Value::Float(*x),
        MtlxValue::Integer(x) => Value::Integer(*x),
        MtlxValue::Boolean(x) => Value::Bool(*x),
        MtlxValue::Color3(x) => Value::Color3(*x),
        MtlxValue::Color4(x) => Value::Color4(*x),
        MtlxValue::Vector2(x) => Value::Vector2(*x),
        MtlxValue::Vector3(x) => Value::Vector3(*x),
        MtlxValue::Vector4(x) => Value::Vector4(*x),
        MtlxValue::Matrix33(_) | MtlxValue::Matrix44(_) => {
            panic!(
                "constant_value: matrix MtlxValue has no inline `Value` form; emit PushMatrix3Const/PushMatrix4Const instead"
            )
        }
        other => {
            return Err(CompileError::Unsupported(format!(
                "literal {:?} cannot be emitted as bytecode value",
                other
            )));
        }
    })
}

fn geometric_kind_from_prop(prop: &str) -> FgKind {
    match prop {
        "position" | "Pworld" | "Pobject" => FgKind::Position,
        "normal" | "Nworld" | "Nobject" => FgKind::Normal,
        "tangent" | "Tworld" | "Tobject" => FgKind::Tangent,
        "bitangent" | "Bworld" | "Bobject" => FgKind::Bitangent,
        "texcoord" | "UV0" => FgKind::Texcoord,
        "geomcolor" => FgKind::Geomcolor,
        "viewdirection" | "Vworld" => FgKind::ViewDirection,
        other => FgKind::Geompropvalue(other.to_string()),
    }
}

fn geometric_kind_value_type(kind: &GeometricKind) -> ValueType {
    match kind {
        GeometricKind::Position(_)
        | GeometricKind::Normal(_)
        | GeometricKind::Tangent(_)
        | GeometricKind::Bitangent(_)
        | GeometricKind::ViewDirection(_) => ValueType::Vector3,
        GeometricKind::Texcoord => ValueType::Vector2,
        GeometricKind::Geomcolor => ValueType::Color3,
        GeometricKind::Frame | GeometricKind::Time => ValueType::Float,
    }
}

fn is_volume_only(graph: &FlatGraph) -> bool {
    let mut has_surface = false;
    let mut has_volume = false;
    for node in &graph.nodes {
        match &node.kind {
            FlatNodeKind::Shading { category } => {
                if matches!(category.as_str(), "absorption_vdf" | "anisotropic_vdf") {
                    has_volume = true;
                } else {
                    has_surface = true;
                }
            }
            FlatNodeKind::Surface | FlatNodeKind::SurfaceUnlit => has_surface = true,
            FlatNodeKind::Pattern { category } if category == "volume" => has_volume = true,
            _ => {}
        }
    }
    has_volume && !has_surface
}

fn closure_max_emission(closures: &[ClosureNode], root: u32) -> f32 {
    #[derive(Clone, Copy)]
    struct Bound {
        shape: f32,
        intensity: f32,
    }

    impl Bound {
        fn zero() -> Self {
            Self {
                shape: 0.0,
                intensity: 0.0,
            }
        }

        fn emission(self) -> f32 {
            self.shape * self.intensity
        }
    }

    fn walk(closures: &[ClosureNode], idx: u32, visited: &mut Vec<bool>) -> Bound {
        if (idx as usize) >= closures.len() {
            return Bound::zero();
        }
        if visited[idx as usize] {
            return Bound::zero();
        }
        visited[idx as usize] = true;
        match &closures[idx as usize] {
            ClosureNode::UniformEdf { color } | ClosureNode::ConicalEdf { color, .. } => Bound {
                shape: 1.0,
                intensity: param_max_color(color),
            },
            ClosureNode::GeneralizedSchlickEdf {
                base,
                color0,
                color90,
                ..
            } => {
                let base = walk(closures, *base, visited);
                Bound {
                    shape: base.shape * param_max_color(color0).max(param_max_color(color90)),
                    intensity: base.intensity,
                }
            }
            ClosureNode::Surface { edf, .. } => walk(closures, *edf, visited),
            ClosureNode::Mix { bg, fg, mix, .. } => {
                let bg = walk(closures, *bg, visited);
                let fg = walk(closures, *fg, visited);
                if let ParamRef::Float(m) = mix {
                    let m = m.clamp(0.0, 1.0);
                    Bound {
                        shape: bg.shape * (1.0 - m) + fg.shape * m,
                        intensity: bg.intensity * (1.0 - m) + fg.intensity * m,
                    }
                } else {
                    Bound {
                        shape: bg.shape.max(fg.shape),
                        intensity: bg.intensity.max(fg.intensity),
                    }
                }
            }
            ClosureNode::Add { a, b, .. } => {
                let a = walk(closures, *a, visited);
                let b = walk(closures, *b, visited);
                Bound {
                    shape: a.shape + b.shape,
                    intensity: a.intensity + b.intensity,
                }
            }
            ClosureNode::Multiply { inner, scale, .. } => {
                let inner = walk(closures, *inner, visited);
                Bound {
                    shape: inner.shape,
                    intensity: inner.intensity * param_max_color(scale).max(0.0),
                }
            }
            ClosureNode::IfGreater {
                then_branch,
                else_branch,
                ..
            }
            | ClosureNode::IfGreaterEq {
                then_branch,
                else_branch,
                ..
            }
            | ClosureNode::IfEqual {
                then_branch,
                else_branch,
                ..
            } => {
                let a = walk(closures, *then_branch, visited);
                let b = walk(closures, *else_branch, visited);
                if a.emission() >= b.emission() { a } else { b }
            }
            ClosureNode::Switch { branches, .. } => branches
                .iter()
                .map(|b| walk(closures, *b, visited))
                .max_by(|a, b| a.emission().total_cmp(&b.emission()))
                .unwrap_or_else(Bound::zero),
            _ => Bound::zero(),
        }
    }
    let mut visited = vec![false; closures.len()];
    walk(closures, root, &mut visited).emission()
}

fn param_max_color(p: &ParamRef) -> f32 {
    match p {
        ParamRef::Color3(c) | ParamRef::Vector3(c) => c.x.max(c.y).max(c.z).max(0.0),
        ParamRef::Color4(c) | ParamRef::Vector4(c) => c.x.max(c.y).max(c.z).max(0.0),
        ParamRef::Float(f) => f.max(0.0),
        _ => 1.0,
    }
}

fn parse_address_mode(s: Option<&str>) -> Result<AddressMode, CompileError> {
    match s {
        None | Some("periodic") => Ok(AddressMode::Periodic),
        Some("constant") => Ok(AddressMode::Constant),
        Some("clamp") => Ok(AddressMode::Clamp),
        Some("mirror") => Ok(AddressMode::Mirror),
        Some(other) => Err(CompileError::Unsupported(format!(
            "unknown address mode `{}` (spec: constant|clamp|periodic|mirror)",
            other
        ))),
    }
}

fn parse_filter_type(s: Option<&str>) -> Result<FilterType, CompileError> {
    match s {
        None | Some("") | Some("linear") => Ok(FilterType::Linear),
        Some("closest") => Ok(FilterType::Closest),
        Some("cubic") => {
            tracing::warn!(
                "warning: image.filtertype=`cubic` is not implemented; using linear filtering"
            );
            Ok(FilterType::Linear)
        }
        Some(other) => Err(CompileError::Unsupported(format!(
            "unknown image filtertype `{}` (spec: closest|linear|cubic)",
            other
        ))),
    }
}

fn parse_geom_space(s: Option<&str>) -> Result<GeomSpace, CompileError> {
    match s {
        None | Some("world") => Ok(GeomSpace::World),
        Some("model") => Ok(GeomSpace::Model),
        Some("object") => Ok(GeomSpace::Object),
        Some(other) => Err(CompileError::Unsupported(format!(
            "unknown geom space `{}` (spec: model|object|world)",
            other
        ))),
    }
}

fn geom_space_id(space: GeomSpace) -> u32 {
    match space {
        GeomSpace::World => 0,
        GeomSpace::Object => 1,
        GeomSpace::Model => 2,
    }
}

fn mtlx_value_type(v: &MtlxValue) -> Option<ValueType> {
    Some(match v {
        MtlxValue::Float(_) => ValueType::Float,
        MtlxValue::Integer(_) => ValueType::Integer,
        MtlxValue::Boolean(_) => ValueType::Boolean,
        MtlxValue::Color3(_) => ValueType::Color3,
        MtlxValue::Color4(_) => ValueType::Color4,
        MtlxValue::Vector2(_) => ValueType::Vector2,
        MtlxValue::Vector3(_) => ValueType::Vector3,
        MtlxValue::Vector4(_) => ValueType::Vector4,
        MtlxValue::Matrix33(_) => ValueType::Matrix33,
        MtlxValue::Matrix44(_) => ValueType::Matrix44,
        _ => return None,
    })
}

fn combine_input_types(kind: CombineKind) -> &'static [ValueType] {
    match kind {
        CombineKind::Vector2FromFloats => &[ValueType::Float, ValueType::Float],
        CombineKind::Color3FromFloats | CombineKind::Vector3FromFloats => {
            &[ValueType::Float, ValueType::Float, ValueType::Float]
        }
        CombineKind::Color4FromFloats | CombineKind::Vector4FromFloats => &[
            ValueType::Float,
            ValueType::Float,
            ValueType::Float,
            ValueType::Float,
        ],
        CombineKind::Color4FromColor3Float => &[ValueType::Color3, ValueType::Float],
        CombineKind::Vector4FromVector3Float => &[ValueType::Vector3, ValueType::Float],
        CombineKind::Vector4FromVector2Vector2 => &[ValueType::Vector2, ValueType::Vector2],
    }
}

fn noise_output_for(out_ty: ValueType) -> NoiseOutput {
    match out_ty {
        ValueType::Float | ValueType::Integer => NoiseOutput::Float,
        ValueType::Vector2 => NoiseOutput::Vector2,
        ValueType::Color4 | ValueType::Vector4 => NoiseOutput::Vector4,
        _ => NoiseOutput::Vector3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_literal_is_not_silently_converted_to_zero_param() {
        let err = constant_param(&MtlxValue::String("not-a-float".to_string())).unwrap_err();
        assert!(matches!(err, CompileError::Unsupported(_)));
    }

    #[test]
    fn filename_literal_is_not_emitted_as_empty_value() {
        let err = constant_value(&MtlxValue::Filename("albedo.png".to_string())).unwrap_err();
        assert!(matches!(err, CompileError::Unsupported(_)));
    }

    #[test]
    fn local_constant_one_opacity_is_not_alpha_test() {
        let instructions = vec![
            Instruction::LoadConst {
                dst: 0,
                value_pool_idx: 0,
            },
            Instruction::LoadConst {
                dst: 1,
                value_pool_idx: 1,
            },
            Instruction::LuminanceWithCoeffs {
                dst: 2,
                ty: ValueType::Color3,
                c: Operand::Reg(0),
                lumacoeffs: Operand::Reg(1),
            },
            Instruction::LoadConst {
                dst: 3,
                value_pool_idx: 2,
            },
            Instruction::Extract {
                dst: 4,
                in_ty: ValueType::Color3,
                src: Operand::Reg(2),
                idx: Operand::Reg(3),
            },
        ];
        let value_pool = vec![
            Value::Color3(Vec3::ONE),
            Value::Color3(Vec3::new(0.2722287, 0.6740818, 0.0536895)),
            Value::Integer(0),
        ];
        let closures = vec![
            ClosureNode::Surface {
                bsdf: 1,
                edf: 0,
                opacity: ParamRef::Local(4),
                thin_walled: false,
            },
            ClosureNode::BurleyDiffuse {
                weight: ParamRef::Float(1.0),
                color: ParamRef::Color3(Vec3::ONE),
                roughness: ParamRef::Float(0.0),
                normal: None,
            },
        ];

        assert!(!closure_has_opacity_test(
            &closures,
            0,
            &instructions,
            &value_pool,
            5,
        ));
    }
}
