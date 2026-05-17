use std::sync::Arc;

use glam::{Mat3, Mat4, Vec2, Vec3, Vec4};

use crate::bsdf::mtlx::{ScatterMode, SheenMode};
use crate::scene::mtlx_loader::MtlxType;
use crate::{
    color::OcioColorProcessor,
    material::{ScalarTexture, Texture, TextureColorSpace},
};

/// 32-byte tagged value used by both the bytecode stack and `mtlx_registers`.
/// Matrix variants store a `u32` index into `MtlxScratch`'s matrix pools to
/// keep the enum small — pushing/popping an 80-byte Value used to dominate the
/// bytecode runtime, so the pool indirection costs nothing for matrix-free
/// materials (the common case) and stays cheap for the rare matrix users.
#[derive(Debug, Clone, Copy)]
pub enum Value {
    Float(f32),
    Integer(i32),
    Bool(bool),
    Color3(Vec3),
    Color4(Vec4),
    Vector2(Vec2),
    Vector3(Vec3),
    Vector4(Vec4),
    Matrix33Ref(u32),
    Matrix44Ref(u32),
    Empty,
}

impl Value {
    pub fn as_float(self) -> f32 {
        const LR: f32 = 0.2722287;
        const LG: f32 = 0.6740818;
        const LB: f32 = 0.0536895;
        match self {
            Value::Float(v) => v,
            Value::Integer(v) => v as f32,
            Value::Bool(b) => {
                if b {
                    1.0
                } else {
                    0.0
                }
            }
            Value::Color3(v) | Value::Vector3(v) => LR * v.x + LG * v.y + LB * v.z,
            Value::Color4(v) | Value::Vector4(v) => LR * v.x + LG * v.y + LB * v.z,
            Value::Vector2(v) => 0.5 * (v.x + v.y),
            Value::Matrix33Ref(_) => panic!("Value::as_float called on Matrix33"),
            Value::Matrix44Ref(_) => panic!("Value::as_float called on Matrix44"),
            Value::Empty => panic!("Value::as_float called on Empty"),
        }
    }

    pub fn as_integer(self) -> i32 {
        match self {
            Value::Integer(v) => v,
            Value::Float(v) => v.trunc() as i32,
            Value::Bool(b) => {
                if b {
                    1
                } else {
                    0
                }
            }
            Value::Color3(_) | Value::Vector3(_) => {
                panic!("Value::as_integer called on Color3/Vector3")
            }
            Value::Color4(_) | Value::Vector4(_) => {
                panic!("Value::as_integer called on Color4/Vector4")
            }
            Value::Vector2(_) => panic!("Value::as_integer called on Vector2"),
            Value::Matrix33Ref(_) => panic!("Value::as_integer called on Matrix33"),
            Value::Matrix44Ref(_) => panic!("Value::as_integer called on Matrix44"),
            Value::Empty => panic!("Value::as_integer called on Empty"),
        }
    }

    pub fn as_bool(self) -> bool {
        match self {
            Value::Bool(b) => b,
            Value::Integer(v) => v != 0,
            Value::Float(v) => v != 0.0,
            Value::Color3(_) | Value::Vector3(_) => {
                panic!("Value::as_bool called on Color3/Vector3")
            }
            Value::Color4(_) | Value::Vector4(_) => {
                panic!("Value::as_bool called on Color4/Vector4")
            }
            Value::Vector2(_) => panic!("Value::as_bool called on Vector2"),
            Value::Matrix33Ref(_) => panic!("Value::as_bool called on Matrix33"),
            Value::Matrix44Ref(_) => panic!("Value::as_bool called on Matrix44"),
            Value::Empty => panic!("Value::as_bool called on Empty"),
        }
    }

    pub fn as_color3(self) -> Vec3 {
        match self {
            Value::Color3(v) | Value::Vector3(v) => v,
            Value::Color4(v) | Value::Vector4(v) => Vec3::new(v.x, v.y, v.z),
            Value::Float(v) => Vec3::splat(v),
            Value::Integer(v) => Vec3::splat(v as f32),
            Value::Vector2(v) => Vec3::new(v.x, v.y, 0.0),
            Value::Bool(b) => Vec3::splat(if b { 1.0 } else { 0.0 }),
            Value::Matrix33Ref(_) => panic!("Value::as_color3 called on Matrix33"),
            Value::Matrix44Ref(_) => panic!("Value::as_color3 called on Matrix44"),
            Value::Empty => panic!("Value::as_color3 called on Empty"),
        }
    }

    pub fn as_color4(self) -> Vec4 {
        match self {
            Value::Color4(v) | Value::Vector4(v) => v,
            Value::Color3(v) | Value::Vector3(v) => Vec4::new(v.x, v.y, v.z, 1.0),
            Value::Vector2(v) => Vec4::new(v.x, v.y, 0.0, 0.0),
            Value::Float(v) => Vec4::splat(v),
            Value::Integer(v) => Vec4::splat(v as f32),
            Value::Bool(b) => Vec4::splat(if b { 1.0 } else { 0.0 }),
            Value::Matrix33Ref(_) => panic!("Value::as_color4 called on Matrix33"),
            Value::Matrix44Ref(_) => panic!("Value::as_color4 called on Matrix44"),
            Value::Empty => panic!("Value::as_color4 called on Empty"),
        }
    }

    pub fn as_vector2(self) -> Vec2 {
        match self {
            Value::Vector2(v) => v,
            Value::Color3(v) | Value::Vector3(v) => Vec2::new(v.x, v.y),
            Value::Color4(v) | Value::Vector4(v) => Vec2::new(v.x, v.y),
            Value::Float(v) => Vec2::splat(v),
            Value::Integer(v) => Vec2::splat(v as f32),
            Value::Bool(b) => Vec2::splat(if b { 1.0 } else { 0.0 }),
            Value::Matrix33Ref(_) => panic!("Value::as_vector2 called on Matrix33"),
            Value::Matrix44Ref(_) => panic!("Value::as_vector2 called on Matrix44"),
            Value::Empty => panic!("Value::as_vector2 called on Empty"),
        }
    }

    pub fn as_vector3(self) -> Vec3 {
        match self {
            Value::Color3(v) | Value::Vector3(v) => v,
            Value::Color4(v) | Value::Vector4(v) => Vec3::new(v.x, v.y, v.z),
            Value::Float(v) => Vec3::splat(v),
            Value::Integer(v) => Vec3::splat(v as f32),
            Value::Vector2(v) => Vec3::new(v.x, v.y, 0.0),
            Value::Bool(b) => Vec3::splat(if b { 1.0 } else { 0.0 }),
            Value::Matrix33Ref(_) => panic!("Value::as_vector3 called on Matrix33"),
            Value::Matrix44Ref(_) => panic!("Value::as_vector3 called on Matrix44"),
            Value::Empty => panic!("Value::as_vector3 called on Empty"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    Float,
    Integer,
    Boolean,
    Color3,
    Color4,
    Vector2,
    Vector3,
    Vector4,
    Matrix33,
    Matrix44,
}

impl ValueType {
    pub fn from_mtlx(t: &MtlxType) -> Option<Self> {
        Some(match t {
            MtlxType::Float => Self::Float,
            MtlxType::Integer => Self::Integer,
            MtlxType::Boolean => Self::Boolean,
            MtlxType::Color3 => Self::Color3,
            MtlxType::Color4 => Self::Color4,
            MtlxType::Vector2 => Self::Vector2,
            MtlxType::Vector3 => Self::Vector3,
            MtlxType::Vector4 => Self::Vector4,
            MtlxType::Matrix33 => Self::Matrix33,
            MtlxType::Matrix44 => Self::Matrix44,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone)]
pub enum ParamRef {
    Float(f32),
    Integer(i32),
    Bool(bool),
    Color3(Vec3),
    Color4(Vec4),
    Vector2(Vec2),
    Vector3(Vec3),
    Vector4(Vec4),
    Matrix33(Mat3),
    Matrix44(Mat4),
    Local(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Min,
    Max,
    Power,
    SafePower,
    Atan2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Sqrt,
    Ln,
    Exp,
    Abs,
    Sign,
    Floor,
    Ceil,
    Round,
    Fract,
    Invert,
    Trianglewave,
    Normalize,
    Magnitude,
    Luminance,
    RgbToHsv,
    HsvToRgb,
    Length,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    Greater,
    GreaterEq,
    Equal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalOp {
    And,
    Or,
    Xor,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseKind {
    Perlin2d,
    Perlin3d,
    Cellnoise2d,
    Cellnoise3d,
    Worleynoise2d,
    Worleynoise3d,
    Fractal2d,
    Fractal3d,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseOutput {
    Float,
    Vector2,
    Vector3,
    Vector4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorleyStyle {
    Distance,
    Solid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriplanarFilter {
    Linear,
    Closest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtisticIorOutput {
    Ior,
    Extinction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChiangHairRoughnessOutput {
    R,
    TT,
    TRT,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeomSpace {
    Model,
    Object,
    World,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometricKind {
    Position(GeomSpace),
    Normal(GeomSpace),
    Tangent(GeomSpace),
    Bitangent(GeomSpace),
    Texcoord,
    Geomcolor,
    Frame,
    Time,
    ViewDirection(GeomSpace),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendOp {
    Plus,
    Minus,
    Difference,
    Burn,
    Dodge,
    Screen,
    Overlay,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeOp {
    Disjointover,
    In,
    Mask,
    Matte,
    Out,
    Over,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskOp {
    Inside,
    Outside,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CombineKind {
    Vector2FromFloats,
    Color3FromFloats,
    Vector3FromFloats,
    Color4FromFloats,
    Vector4FromFloats,
    Color4FromColor3Float,
    Vector4FromVector3Float,
    Vector4FromVector2Vector2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageKind {
    Image,
    TiledImage,
    LatLongImage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterType {
    Closest,
    Linear,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColorXform {
    Identity,
    TextureToRendering,
    RenderingToTexture,
    Ocio { processor: u16 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressMode {
    Constant,
    Clamp,
    Periodic,
    Mirror,
}

#[derive(Debug, Clone)]
pub enum ImageTexture {
    Color(Arc<Texture>),
    /// RGB texture paired with an optional alpha pyramid, used for
    /// `image_color4` so spec-compliant materials get the source alpha
    /// channel instead of a constant 1.0.
    ColorAlpha {
        rgb: Arc<Texture>,
        alpha: Arc<ScalarTexture>,
    },
    Scalar(Arc<ScalarTexture>),
    /// UDIM tile set (key = UDIM id 1001..). Sampled by computing the
    /// tile id from the integer UV portion per spec §Filename Substitutions.
    Udim {
        tiles: Arc<UdimTiles>,
    },
    Missing,
}

/// Map of UDIM id (Mari-style 1001..) to RGB texture, optionally with an
/// alpha pyramid per tile. All tiles share the same color space.
#[derive(Debug, Clone)]
pub struct UdimTiles {
    pub tiles: std::collections::HashMap<u32, UdimTile>,
}

#[derive(Debug, Clone)]
pub struct UdimTile {
    pub rgb: Arc<Texture>,
    pub alpha: Option<Arc<ScalarTexture>>,
    pub scalar: Option<Arc<ScalarTexture>>,
}

/// SSA bytecode operand. `Reg(i)` is a register read (current shading-vertex
/// register file), `Const(i)` is an inline value from `CompiledMaterial::value_pool`.
/// 8 bytes (tag + u32). Keeps `Instruction` enum variants compact so the
/// instruction stream is cache-friendly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operand {
    Reg(u16),
    Const(u32),
}

/// SSA register-machine bytecode. Each instruction explicitly names a
/// destination register and reads its inputs as `Operand`s — there is no
/// implicit stack. Variable-arity ops (Combine, Switch, Image, Noise, etc.)
/// store their operands in `CompiledMaterial::operand_pool` and reference them
/// via `operands_start`.
#[derive(Debug, Clone)]
pub enum Instruction {
    LoadConst {
        dst: u16,
        value_pool_idx: u32,
    },
    LoadGeom {
        dst: u16,
        kind: GeometricKind,
    },
    LoadMat3Const {
        dst: u16,
        value: Mat3,
    },
    LoadMat4Const {
        dst: u16,
        value: Mat4,
    },

    Arith {
        dst: u16,
        op: ArithOp,
        ty: ValueType,
        a: Operand,
        b: Operand,
    },
    Unary {
        dst: u16,
        op: UnaryOp,
        ty: ValueType,
        src: Operand,
    },
    Convert {
        dst: u16,
        from: ValueType,
        to: ValueType,
        src: Operand,
    },
    Logical {
        dst: u16,
        op: LogicalOp,
        a: Operand,
        b: Operand,
    },
    CompareBool {
        dst: u16,
        op: CompareOp,
        v1: Operand,
        v2: Operand,
    },
    Compare {
        dst: u16,
        op: CompareOp,
        v1: Operand,
        v2: Operand,
        in_true: Operand,
        in_false: Operand,
    },
    IfElse {
        dst: u16,
        cond: Operand,
        in_true: Operand,
        in_false: Operand,
    },
    MixValue {
        dst: u16,
        ty: ValueType,
        bg: Operand,
        fg: Operand,
        mix: Operand,
    },
    Clamp {
        dst: u16,
        ty: ValueType,
        v: Operand,
        lo: Operand,
        hi: Operand,
    },
    Smoothstep {
        dst: u16,
        ty: ValueType,
        v: Operand,
        lo: Operand,
        hi: Operand,
    },
    Extract {
        dst: u16,
        in_ty: ValueType,
        src: Operand,
        idx: Operand,
    },
    ExtractRowVector {
        dst: u16,
        dim4: bool,
        src: Operand,
        index: u8,
    },
    Reflect {
        dst: u16,
        i: Operand,
        n: Operand,
    },
    Refract {
        dst: u16,
        i: Operand,
        n: Operand,
        eta: Operand,
    },
    Rotate2d {
        dst: u16,
        v: Operand,
        amount: Operand,
    },
    Rotate3d {
        dst: u16,
        v: Operand,
        axis: Operand,
        amount: Operand,
    },
    DotProduct {
        dst: u16,
        ty: ValueType,
        a: Operand,
        b: Operand,
    },
    CrossProduct {
        dst: u16,
        a: Operand,
        b: Operand,
    },
    Distance {
        dst: u16,
        ty: ValueType,
        a: Operand,
        b: Operand,
    },
    FacingRatio {
        dst: u16,
        view: Operand,
        normal: Operand,
        invert: bool,
        faceforward: bool,
    },
    LuminanceWithCoeffs {
        dst: u16,
        ty: ValueType,
        c: Operand,
        lumacoeffs: Operand,
    },

    /// Combine 2-4 operands into a vector/color. operand count is determined by
    /// `kind` (see `combine_input_count`). Operands are at
    /// `operand_pool[operands_start..operands_start + count]`.
    Combine {
        dst: u16,
        kind: CombineKind,
        operands_start: u32,
    },
    /// Build a Matrix33 from 3 row operands (each Vector3).
    CreateMatrix3 {
        dst: u16,
        rows_start: u32,
    },
    /// Build a Matrix44 from 4 row operands (each Vector4).
    CreateMatrix4 {
        dst: u16,
        rows_start: u32,
    },
    CreateMatrix4FromVec3 {
        dst: u16,
        rows_start: u32,
    },
    /// Switch with 10 branches; branches in `operand_pool[branches_start..+10]`.
    Switch {
        dst: u16,
        ty: ValueType,
        which: Operand,
        branches_start: u32,
    },

    /// Image / TiledImage / LatLongImage. Operand layout:
    /// `[texcoord, tiling, offset, default_color]`.
    Image {
        dst: u16,
        texture: ImageTexture,
        kind: ImageKind,
        output: ValueType,
        color_space: TextureColorSpace,
        uaddress: AddressMode,
        vaddress: AddressMode,
        filter: FilterType,
        texcoord: Operand,
        tiling: Operand,
        offset: Operand,
        default: Operand,
    },
    /// Hex-tiled image operands:
    /// `[texcoord, tiling, rotation, rotationrange, scale, scalerange,
    ///   offset, offsetrange, falloff, falloffcontrast, lumacoeffs]`.
    HextiledImage {
        dst: u16,
        texture: ImageTexture,
        output: ValueType,
        default_color: Vec4,
        color_space: TextureColorSpace,
        operands_start: u32,
    },
    /// Hex-tiled normal map operands:
    /// `[texcoord, tiling, rotation, rotationrange, scale, scalerange,
    ///   offset, offsetrange, falloff, strength, default, normal, tangent, bitangent]`.
    HextiledNormalMap {
        dst: u16,
        texture: Option<Arc<Texture>>,
        flip_g: bool,
        operands_start: u32,
    },

    TransformPoint {
        dst: u16,
        from: GeomSpace,
        to: GeomSpace,
        v: Operand,
    },
    TransformVector {
        dst: u16,
        from: GeomSpace,
        to: GeomSpace,
        v: Operand,
    },
    TransformNormal {
        dst: u16,
        from: GeomSpace,
        to: GeomSpace,
        v: Operand,
    },
    /// `transformpoint`/`transformvector`/`transformnormal` variant with an
    /// explicit input matrix operand (instead of from→to space pair).
    TransformMatrix {
        dst: u16,
        out_ty: ValueType,
        dim4: bool,
        mat: Operand,
        v: Operand,
    },
    Transpose {
        dst: u16,
        dim4: bool,
        src: Operand,
    },
    Determinant {
        dst: u16,
        dim4: bool,
        src: Operand,
    },
    InvertMatrix {
        dst: u16,
        dim4: bool,
        src: Operand,
    },

    Place2d {
        dst: u16,
        trs: bool,
        texcoord: Operand,
        pivot: Operand,
        scale: Operand,
        rotate: Operand,
        offset: Operand,
    },
    LatlongUv {
        dst: u16,
        viewdir: Operand,
        rotation: Operand,
    },

    /// Noise operands: `[coord, amplitude, pivot, octaves, lacunarity, diminish, jitter]`.
    Noise {
        dst: u16,
        kind: NoiseKind,
        output: NoiseOutput,
        operands_start: u32,
    },
    /// Worley noise. Operands: `[coord, jitter]`.
    Worley {
        dst: u16,
        dim3: bool,
        output: NoiseOutput,
        style: WorleyStyle,
        operands_start: u32,
    },
    Cellnoise {
        dst: u16,
        dim3: bool,
        output: NoiseOutput,
        coord: Operand,
    },
    /// `randomfloat`. Operands: `[input, seed, min, max]`.
    RandomFloat {
        dst: u16,
        integer_input: bool,
        operands_start: u32,
    },
    /// `randomcolor`. Operands: `[input, seed, hue_min, hue_max, sat_min,
    /// sat_max, brightness_min, brightness_max]`.
    RandomColor {
        dst: u16,
        operands_start: u32,
    },

    Ramplr {
        dst: u16,
        ty: ValueType,
        texcoord: Operand,
        l: Operand,
        r: Operand,
    },
    Ramptb {
        dst: u16,
        ty: ValueType,
        texcoord: Operand,
        t: Operand,
        b: Operand,
    },
    Ramp4 {
        dst: u16,
        ty: ValueType,
        texcoord: Operand,
        tl: Operand,
        tr: Operand,
        bl: Operand,
        br: Operand,
    },
    Splitlr {
        dst: u16,
        ty: ValueType,
        texcoord: Operand,
        center: Operand,
        l: Operand,
        r: Operand,
    },
    Splittb {
        dst: u16,
        ty: ValueType,
        texcoord: Operand,
        center: Operand,
        t: Operand,
        b: Operand,
    },

    Blackbody {
        dst: u16,
        temp: Operand,
    },
    ArtisticIor {
        dst: u16,
        which: ArtisticIorOutput,
        refl: Operand,
        edge: Operand,
    },
    ChiangHairRoughness {
        dst: u16,
        which: ChiangHairRoughnessOutput,
        longitudinal: Operand,
        azimuthal: Operand,
        scale_tt: Operand,
        scale_trt: Operand,
    },
    /// `deon_hair_absorption_from_melanin`. Operands:
    /// `[melanin_concentration, melanin_redness, eumelanin_color, pheomelanin_color]`.
    DeonHairAbsorptionFromMelanin {
        dst: u16,
        operands_start: u32,
    },
    ChiangHairAbsorptionFromColor {
        dst: u16,
        color: Operand,
        beta: Operand,
    },
    RoughnessAnisotropy {
        dst: u16,
        r: Operand,
        a: Operand,
    },
    GlossinessAnisotropy {
        dst: u16,
        g: Operand,
        a: Operand,
    },
    RoughnessDual {
        dst: u16,
        src: Operand,
    },
    TransformColor {
        dst: u16,
        op: ColorXform,
        ty: ValueType,
        src: Operand,
    },
    /// `triplanarblend`. Operands: `[inx, iny, inz, normal, blend]`.
    TriplanarBlend {
        dst: u16,
        ty: ValueType,
        filter: TriplanarFilter,
        operands_start: u32,
    },
    CurveUniformLinear {
        dst: u16,
        knotvalues: Arc<Vec<f32>>,
        t: Operand,
    },
    CurveUniformCubic {
        dst: u16,
        knotvalues: Arc<Vec<f32>>,
        t: Operand,
    },
    CurveInverseCubic {
        dst: u16,
        knots: Arc<Vec<f32>>,
        x: Operand,
    },
    Normalmap {
        dst: u16,
        raw: Operand,
        scale: Operand,
    },
    /// `normalmap` with explicit frame inputs. Operands:
    /// `[raw, scale, normal, tangent, bitangent]`.
    NormalmapWithFrame {
        dst: u16,
        operands_start: u32,
    },
    Bump {
        dst: u16,
        height: Operand,
        scale: Operand,
    },
    /// Operands: `[height, scale, normal, tangent]`.
    BumpWithFrame {
        dst: u16,
        operands_start: u32,
    },
    HeightToNormal {
        dst: u16,
        height: Operand,
        scale: Operand,
    },

    Blend {
        dst: u16,
        op: BlendOp,
        ty: ValueType,
        bg: Operand,
        fg: Operand,
        mix: Operand,
    },
    Merge {
        dst: u16,
        op: MergeOp,
        bg: Operand,
        fg: Operand,
        mix: Operand,
    },
    Mask {
        dst: u16,
        op: MaskOp,
        ty: ValueType,
        v: Operand,
        mask: Operand,
    },
    Premult {
        dst: u16,
        src: Operand,
    },
    Unpremult {
        dst: u16,
        src: Operand,
    },

    Contrast {
        dst: u16,
        ty: ValueType,
        v: Operand,
        amount: Operand,
        pivot: Operand,
    },
    /// `range`. Operands: `[in, inlow, inhigh, gamma, outlow, outhigh]`.
    Range {
        dst: u16,
        ty: ValueType,
        doclamp: bool,
        operands_start: u32,
    },
    /// `remap`. Operands: `[in, inlow, inhigh, outlow, outhigh]`.
    Remap {
        dst: u16,
        ty: ValueType,
        operands_start: u32,
    },
    HsvAdjust {
        dst: u16,
        ty: ValueType,
        c: Operand,
        amount: Operand,
    },
    Saturate {
        dst: u16,
        ty: ValueType,
        c: Operand,
        amount: Operand,
        lumacoeffs: Operand,
    },
    /// `colorcorrect`. Operands: `[in, hue, saturation, gamma, lift, gain,
    /// contrast, contrastpivot, exposure]` (9 operands).
    ColorCorrect {
        dst: u16,
        ty: ValueType,
        operands_start: u32,
    },

    Checkerboard {
        dst: u16,
        color1: Operand,
        color2: Operand,
        uvtiling: Operand,
        uvoffset: Operand,
        texcoord: Operand,
    },
    Passthrough,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClosureKind {
    Bsdf,
    Edf,
    Surface,
    Vdf,
    None,
}

#[derive(Debug, Clone)]
pub enum ClosureNode {
    Zero,

    OrenNayarDiffuse {
        weight: ParamRef,
        color: ParamRef,
        roughness: ParamRef,
        energy_compensation: bool,
        normal: Option<ParamRef>,
    },
    BurleyDiffuse {
        weight: ParamRef,
        color: ParamRef,
        roughness: ParamRef,
        normal: Option<ParamRef>,
    },
    Translucent {
        weight: ParamRef,
        color: ParamRef,
        normal: Option<ParamRef>,
    },
    Dielectric {
        weight: ParamRef,
        tint: ParamRef,
        ior: ParamRef,
        roughness: ParamRef,
        scatter_mode: ScatterMode,
        thinfilm_thickness: ParamRef,
        thinfilm_ior: ParamRef,
        normal: Option<ParamRef>,
        tangent: Option<ParamRef>,
    },
    Conductor {
        weight: ParamRef,
        ior: ParamRef,
        extinction: ParamRef,
        roughness: ParamRef,
        thinfilm_thickness: ParamRef,
        thinfilm_ior: ParamRef,
        normal: Option<ParamRef>,
        tangent: Option<ParamRef>,
    },
    GeneralizedSchlick {
        weight: ParamRef,
        color0: ParamRef,
        color82: ParamRef,
        color90: ParamRef,
        exponent: ParamRef,
        roughness: ParamRef,
        scatter_mode: ScatterMode,
        thinfilm_thickness: ParamRef,
        thinfilm_ior: ParamRef,
        normal: Option<ParamRef>,
        tangent: Option<ParamRef>,
    },
    Sheen {
        weight: ParamRef,
        color: ParamRef,
        roughness: ParamRef,
        mode: SheenMode,
        normal: Option<ParamRef>,
    },
    ChiangHair {
        tint_r: ParamRef,
        tint_tt: ParamRef,
        tint_trt: ParamRef,
        absorption: ParamRef,
        ior: ParamRef,
        roughness_r: ParamRef,
        roughness_tt: ParamRef,
        roughness_trt: ParamRef,
        cuticle_angle: ParamRef,
        normal: Option<ParamRef>,
        curve_direction: ParamRef,
    },
    ThinFilm {
        thickness: ParamRef,
        ior: ParamRef,
    },

    UniformEdf {
        color: ParamRef,
    },
    ConicalEdf {
        color: ParamRef,
        inner_angle: ParamRef,
        outer_angle: ParamRef,
        normal: Option<ParamRef>,
    },
    GeneralizedSchlickEdf {
        base: u32,
        color0: ParamRef,
        color90: ParamRef,
        exponent: ParamRef,
    },

    Mix {
        bg: u32,
        fg: u32,
        mix: ParamRef,
        kind: ClosureKind,
    },
    Layer {
        top: u32,
        base: u32,
    },
    Add {
        a: u32,
        b: u32,
        kind: ClosureKind,
    },
    Multiply {
        inner: u32,
        scale: ParamRef,
        kind: ClosureKind,
    },

    IfGreater {
        value1: ParamRef,
        value2: ParamRef,
        then_branch: u32,
        else_branch: u32,
        kind: ClosureKind,
    },
    IfGreaterEq {
        value1: ParamRef,
        value2: ParamRef,
        then_branch: u32,
        else_branch: u32,
        kind: ClosureKind,
    },
    IfEqual {
        value1: ParamRef,
        value2: ParamRef,
        then_branch: u32,
        else_branch: u32,
        kind: ClosureKind,
    },
    Switch {
        which: ParamRef,
        branches: [u32; 10],
        kind: ClosureKind,
    },

    Surface {
        bsdf: u32,
        edf: u32,
        opacity: ParamRef,
        thin_walled: bool,
    },
    GoochShade {
        warm: ParamRef,
        cool: ParamRef,
        specular_intensity: ParamRef,
        shininess: ParamRef,
        light_direction: ParamRef,
    },
}

#[derive(Debug, Clone)]
pub struct CompiledMaterial {
    pub instructions: Vec<Instruction>,
    pub operand_pool: Vec<Operand>,
    pub value_pool: Vec<Value>,
    pub color_processors: Vec<Arc<OcioColorProcessor>>,
    pub opacity_instructions: Vec<Instruction>,
    pub opacity_operand_pool: Vec<Operand>,
    pub opacity_closure_nodes: Vec<ClosureNode>,
    pub opacity_num_registers: u32,
    pub num_registers: u32,
    pub closure_nodes: Vec<ClosureNode>,
    pub root: u32,
    pub passthrough: bool,
    pub max_emission: f32,
    pub may_emit: bool,
    pub has_opacity_test: bool,
    pub thin_walled: bool,
    pub(crate) sheen_lut: Option<std::sync::Arc<crate::bsdf::SheenDirectionalAlbedoLut>>,
    pub(crate) mtlx_dielectric_lut:
        Option<std::sync::Arc<crate::bsdf::MtlxDielectricGgxDirectionalAlbedoLut>>,
    pub(crate) mtlx_generalized_schlick_lut:
        Option<std::sync::Arc<crate::bsdf::MtlxGeneralizedSchlickGgxDirectionalAlbedoLut>>,
}

impl CompiledMaterial {
    pub fn closure(&self, idx: u32) -> &ClosureNode {
        &self.closure_nodes[idx as usize]
    }
}
