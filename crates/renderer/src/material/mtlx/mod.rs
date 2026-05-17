pub mod compile;
pub mod compiled;
pub mod runtime;

#[cfg(test)]
mod spec_tests;

use std::cell::Cell;

pub use compile::{compile, compile_with_ocio};
pub use compiled::{
    AddressMode, ArithOp, ArtisticIorOutput, BlendOp, ClosureKind, ClosureNode, CombineKind,
    CompareOp, CompiledMaterial, GeomSpace, GeometricKind, ImageKind, ImageTexture, Instruction,
    LogicalOp, MaskOp, MergeOp, NoiseKind, NoiseOutput, Operand, ParamRef, UnaryOp, Value,
    ValueType, WorleyStyle,
};

/// MaterialX 評価器の thread-local scratch。
///
/// インテグレータが per-thread に 1 つ生成し、 hit/intersection の各時点で
/// 明示的に `&mut` で渡す。 GPU 移植時にも同じパターンを使えるよう、暗黙的な
/// thread-local global は使わず必ず引数で運ぶ。
///
/// 内部は stack-allocator として動作する 3 つのプール:
/// - `regs_pool`: precompute_shading が書き込む register file (vertex ごとに
///   独立した region を bump-allocate して使う)
/// - `matrix3_pool` / `matrix4_pool`: `Value::Matrix33Ref(idx)` /
///   `Value::Matrix44Ref(idx)` が指す matrix の実体
///
/// 呼び出し側は intersection / candidate hit の境界で `checkpoint()` / `restore()`
/// を組み合わせて使うことでネストした precompute をサポートしつつアロケーション
/// を増やさない。
#[derive(Debug, Default, Clone)]
pub struct MtlxScratch {
    pub(crate) regs_pool: Vec<Value>,
    pub(crate) matrix3_pool: Vec<glam::Mat3>,
    pub(crate) matrix4_pool: Vec<glam::Mat4>,
    pub(crate) dalbedo_pool: Vec<Cell<Option<glam::Vec3>>>,
}

/// `MtlxScratch` 内に確保された register 領域 へのハンドル。
/// `offset..offset+len` が `regs_pool` 内の slice を表す。
/// matrix_pool の checkpoint も同時に保持し、 register lifetime と
/// 同じ範囲の matrix entry を有効に保つ。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegsHandle {
    pub(crate) offset: u32,
    pub(crate) len: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DalbedoHandle {
    pub(crate) offset: u32,
    pub(crate) len: u32,
}

/// `MtlxScratch` の状態スナップショット。 nested intersection 用に
/// `checkpoint()` で取得し、 `restore()` で巻き戻す。
#[derive(Debug, Clone, Copy)]
pub struct ScratchCheckpoint {
    regs_top: usize,
    matrix3_top: usize,
    matrix4_top: usize,
    dalbedo_top: usize,
}

impl MtlxScratch {
    #[inline(always)]
    pub fn checkpoint(&self) -> ScratchCheckpoint {
        ScratchCheckpoint {
            regs_top: self.regs_pool.len(),
            matrix3_top: self.matrix3_pool.len(),
            matrix4_top: self.matrix4_pool.len(),
            dalbedo_top: self.dalbedo_pool.len(),
        }
    }

    #[inline(always)]
    pub fn restore(&mut self, cp: ScratchCheckpoint) {
        self.regs_pool.truncate(cp.regs_top);
        self.matrix3_pool.truncate(cp.matrix3_top);
        self.matrix4_pool.truncate(cp.matrix4_top);
        self.dalbedo_pool.truncate(cp.dalbedo_top);
    }

    #[inline]
    pub fn alloc_regs(&mut self, n: usize) -> RegsHandle {
        let offset = self.regs_pool.len();
        self.regs_pool.resize(offset + n, Value::Empty);
        RegsHandle {
            offset: offset as u32,
            len: n as u32,
        }
    }

    #[inline]
    pub fn alloc_dalbedo_cache(&mut self, n: usize) -> DalbedoHandle {
        let offset = self.dalbedo_pool.len();
        self.dalbedo_pool
            .resize_with(offset + n, || Cell::new(None));
        DalbedoHandle {
            offset: offset as u32,
            len: n as u32,
        }
    }

    #[inline(always)]
    pub fn regs_slice(&self, h: RegsHandle) -> &[Value] {
        let s = h.offset as usize;
        &self.regs_pool[s..s + h.len as usize]
    }

    #[inline(always)]
    pub fn regs_slice_mut(&mut self, h: RegsHandle) -> &mut [Value] {
        let s = h.offset as usize;
        &mut self.regs_pool[s..s + h.len as usize]
    }

    #[inline(always)]
    pub fn dalbedo_slice(&self, h: DalbedoHandle) -> &[Cell<Option<glam::Vec3>>] {
        let s = h.offset as usize;
        &self.dalbedo_pool[s..s + h.len as usize]
    }
}
