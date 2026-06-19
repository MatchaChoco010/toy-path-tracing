use glam::{UVec2, Vec2};

use super::{pcg, zsobol};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaterialSampleRandoms {
    pub u_lobe: f32,
    pub u_layer: f32,
    pub u_dir: Vec2,
    pub u_extra0: f32,
    pub u_extra1: f32,
    pub u_extra2: f32,
    pub u_extra3: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightSampleRandoms {
    pub u_category: f32,
    pub u_tree: f32,
    pub u_light_aux: f32,
    pub u_surface: Vec2,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathVertexRandoms {
    pub light: LightSampleRandoms,
    pub material: MaterialSampleRandoms,
    pub u_rr: f32,
    pub aux_rng_seed: u32,
}

impl MaterialSampleRandoms {
    pub fn with_lobe(self, u_lobe: f32) -> Self {
        Self {
            u_lobe: u_lobe.clamp(0.0, 1.0 - f32::EPSILON),
            ..self
        }
    }

    #[cfg(test)]
    pub(crate) fn from_aux_rng(rng: &mut crate::sampler::AuxRng) -> Self {
        Self {
            u_lobe: rng.next_f32(),
            u_layer: rng.next_f32(),
            u_dir: Vec2::new(rng.next_f32(), rng.next_f32()),
            u_extra0: rng.next_f32(),
            u_extra1: rng.next_f32(),
            u_extra2: rng.next_f32(),
            u_extra3: rng.next_f32(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PathSampler {
    zsobol_indexer: zsobol::IndexSampler,
    pixel: UVec2,
    sample_index: u32,
}

const CAMERA_SAMPLE_DIM: u32 = 0;
const INITIAL_AUX_RNG_SEED_DIM: u32 = 0x8000_0000;
const PATH_VERTEX_BASE_DIM: u32 = 2;
const PATH_VERTEX_DIM_STRIDE: u32 = 16;
const LIGHT_CATEGORY_OFFSET: u32 = 0;
const LIGHT_TREE_OFFSET: u32 = 1;
const LIGHT_AUX_OFFSET: u32 = 2;
const LIGHT_SURFACE_OFFSET: u32 = 3;
const MATERIAL_LOBE_OFFSET: u32 = 5;
const MATERIAL_LAYER_OFFSET: u32 = 6;
const MATERIAL_DIR_OFFSET: u32 = 7;
const MATERIAL_EXTRA0_OFFSET: u32 = 9;
const MATERIAL_EXTRA1_OFFSET: u32 = 10;
const MATERIAL_EXTRA2_OFFSET: u32 = 11;
const MATERIAL_EXTRA3_OFFSET: u32 = 12;
const RUSSIAN_ROULETTE_OFFSET: u32 = 13;
const AUX_RNG_SEED_OFFSET: u32 = 14;
const SCRAMBLE_SEED: u32 = 0;

impl PathSampler {
    pub fn new(pixel: UVec2, sample_index: u32, samples_per_pixel: u32, resolution: UVec2) -> Self {
        Self {
            zsobol_indexer: zsobol::IndexSampler::new(samples_per_pixel, resolution),
            pixel,
            sample_index,
        }
    }

    pub fn camera_sample(&self) -> Vec2 {
        self.sample_2d(CAMERA_SAMPLE_DIM)
    }

    pub fn initial_aux_rng_seed(&self) -> u32 {
        self.aux_rng_seed(INITIAL_AUX_RNG_SEED_DIM)
    }

    pub fn path_vertex_randoms(&self, depth: u32) -> PathVertexRandoms {
        let base = PATH_VERTEX_BASE_DIM + depth * PATH_VERTEX_DIM_STRIDE;
        PathVertexRandoms {
            light: LightSampleRandoms {
                u_category: self.sample_1d(base + LIGHT_CATEGORY_OFFSET),
                u_tree: self.sample_1d(base + LIGHT_TREE_OFFSET),
                u_light_aux: self.sample_1d(base + LIGHT_AUX_OFFSET),
                u_surface: self.sample_2d(base + LIGHT_SURFACE_OFFSET),
            },
            material: MaterialSampleRandoms {
                u_lobe: self.sample_1d(base + MATERIAL_LOBE_OFFSET),
                u_layer: self.sample_1d(base + MATERIAL_LAYER_OFFSET),
                u_dir: self.sample_2d(base + MATERIAL_DIR_OFFSET),
                u_extra0: self.sample_1d(base + MATERIAL_EXTRA0_OFFSET),
                u_extra1: self.sample_1d(base + MATERIAL_EXTRA1_OFFSET),
                u_extra2: self.sample_1d(base + MATERIAL_EXTRA2_OFFSET),
                u_extra3: self.sample_1d(base + MATERIAL_EXTRA3_OFFSET),
            },
            u_rr: self.sample_1d(base + RUSSIAN_ROULETTE_OFFSET),
            aux_rng_seed: self.aux_rng_seed(base + AUX_RNG_SEED_OFFSET),
        }
    }

    fn sample_1d(&self, dimension: u32) -> f32 {
        zsobol::shuffled_scrambled_sobol_float::<1>(
            self.sample_index_for_dimension(dimension),
            self.seed_for_dimension(dimension),
        )[0]
    }

    fn sample_2d(&self, dimension: u32) -> Vec2 {
        let sample = zsobol::shuffled_scrambled_sobol_float::<2>(
            self.sample_index_for_dimension(dimension),
            self.seed_for_dimension(dimension),
        );
        Vec2::new(sample[0], sample[1])
    }

    fn aux_rng_seed(&self, dimension: u32) -> u32 {
        let index = self
            .zsobol_indexer
            .sample_index(self.pixel, self.sample_index, dimension);
        let bits = zsobol::mix_bits(index ^ ((dimension as u64) << 32) ^ SCRAMBLE_SEED as u64);
        pcg::hash(fold_u64_to_u32(bits))
    }

    fn sample_index_for_dimension(&self, dimension: u32) -> u32 {
        fold_u64_to_u32(
            self.zsobol_indexer
                .sample_index(self.pixel, self.sample_index, dimension),
        )
    }

    fn seed_for_dimension(&self, dimension: u32) -> u32 {
        let bits = zsobol::mix_bits(((SCRAMBLE_SEED as u64) << 32) | dimension as u64);
        fold_u64_to_u32(bits)
    }
}

fn fold_u64_to_u32(value: u64) -> u32 {
    (value as u32) ^ (value >> 32) as u32
}
