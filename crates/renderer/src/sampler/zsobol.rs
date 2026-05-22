use glam::UVec2;

#[cfg(test)]
use super::pcg;

#[cfg(test)]
const FLOAT_ONE_MINUS_EPSILON: f32 = f32::from_bits(0x3f7f_ffff);

pub(super) const fn laine_karras_permutation(mut value: u32, seed: u32) -> u32 {
    value ^= value.wrapping_mul(0x3d20_adea);
    value = value.wrapping_add(seed);
    value = value.wrapping_mul((seed >> 16) | 1);
    value ^= value.wrapping_mul(0x0552_6c56);
    value ^= value.wrapping_mul(0x53a2_2864);
    value
}

pub(super) const fn reverse_and_shuffle(value: u32, seed: u32) -> u32 {
    laine_karras_permutation(value.reverse_bits(), seed)
}

#[cfg(test)]
const fn shuffle(value: u32, seed: u32) -> u32 {
    reverse_and_shuffle(value, seed).reverse_bits()
}

pub(super) const fn rotate_bytes(value: u32, distance: u32) -> u32 {
    value.rotate_right((distance * 8) & 31)
}

pub(super) const fn scramble_and_reverse(value: u32, seed: u32) -> u32 {
    laine_karras_permutation(value, seed).reverse_bits()
}

pub(super) fn uint_to_float(value: u32) -> f32 {
    let mask = value >> 24;
    let safe = value & !mask;
    (safe as f32) * (1.0 / 4_294_967_296.0)
}

pub(super) const fn mix_bits(mut value: u64) -> u64 {
    value ^= value >> 31;
    value = value.wrapping_mul(0x7fb5_d329_728e_a185);
    value ^= value >> 27;
    value = value.wrapping_mul(0x81da_def4_bc2d_d44d);
    value ^= value >> 33;
    value
}

pub(super) const fn left_shift_2(mut value: u64) -> u64 {
    value &= 0xffff_ffff;
    value = (value ^ (value << 16)) & 0x0000_ffff_0000_ffff;
    value = (value ^ (value << 8)) & 0x00ff_00ff_00ff_00ff;
    value = (value ^ (value << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
    value = (value ^ (value << 2)) & 0x3333_3333_3333_3333;
    value = (value ^ (value << 1)) & 0x5555_5555_5555_5555;
    value
}

pub(super) const fn encode_morton_2(x: u32, y: u32) -> u64 {
    (left_shift_2(y as u64) << 1) | left_shift_2(x as u64)
}

fn log2_int(value: u32) -> u32 {
    debug_assert!(value > 0);
    u32::BITS - 1 - value.leading_zeros()
}

fn round_up_pow2(value: u32) -> u32 {
    value.max(1).next_power_of_two()
}

pub(super) fn sobol_reversed_index(index: u16, dimension: usize) -> u16 {
    assert!(dimension < 4);
    if dimension == 0 {
        return index.reverse_bits();
    }

    const MASKS: [u16; 16] = [
        0b0000000000000001,
        0b0000000000000010,
        0b0000000000000100,
        0b0000000000001000,
        0b0000000000010000,
        0b0000000000100000,
        0b0000000001000000,
        0b0000000010000000,
        0b0000000100000000,
        0b0000001000000000,
        0b0000010000000000,
        0b0000100000000000,
        0b0001000000000000,
        0b0010000000000000,
        0b0100000000000000,
        0b1000000000000000,
    ];

    const DIRECTIONS: [[u16; 16]; 4] = [
        [
            0b1000000000000000,
            0b0100000000000000,
            0b0010000000000000,
            0b0001000000000000,
            0b0000100000000000,
            0b0000010000000000,
            0b0000001000000000,
            0b0000000100000000,
            0b0000000010000000,
            0b0000000001000000,
            0b0000000000100000,
            0b0000000000010000,
            0b0000000000001000,
            0b0000000000000100,
            0b0000000000000010,
            0b0000000000000001,
        ],
        [
            0b1111111111111111,
            0b0101010101010101,
            0b0011001100110011,
            0b0001000100010001,
            0b0000111100001111,
            0b0000010100000101,
            0b0000001100000011,
            0b0000000100000001,
            0b0000000011111111,
            0b0000000001010101,
            0b0000000000110011,
            0b0000000000010001,
            0b0000000000001111,
            0b0000000000000101,
            0b0000000000000011,
            0b0000000000000001,
        ],
        [
            0b1010101000001001,
            0b0111011100000110,
            0b0011100100000011,
            0b0001011000000001,
            0b0000100110101010,
            0b0000011001110111,
            0b0000001100111001,
            0b0000000100010110,
            0b0000000010100011,
            0b0000000001110001,
            0b0000000000111010,
            0b0000000000010111,
            0b0000000000001001,
            0b0000000000000110,
            0b0000000000000011,
            0b0000000000000001,
        ],
        [
            0b1010000011000011,
            0b0100000001000001,
            0b0011000000101101,
            0b0001000000011110,
            0b0000101101100111,
            0b0000011110011010,
            0b0000001010100100,
            0b0000000100011011,
            0b0000000011001001,
            0b0000000001000101,
            0b0000000000101110,
            0b0000000000011111,
            0b0000000000001010,
            0b0000000000000100,
            0b0000000000000011,
            0b0000000000000001,
        ],
    ];

    let matrix = DIRECTIONS[dimension];
    let mut sample = 0;
    for i in 0..16 {
        if (index & MASKS[i]) != 0 {
            sample ^= matrix[i];
        }
    }
    sample
}

pub(super) fn shuffled_scrambled_sobol<const DEPTH: usize>(index: u32, seed: u32) -> [u32; DEPTH] {
    assert!((1..=4).contains(&DEPTH));
    let index = reverse_and_shuffle(index, seed);
    let mut sample = [0; DEPTH];
    for (dimension, value) in sample.iter_mut().enumerate() {
        let sobol = sobol_reversed_index((index >> 16) as u16, dimension);
        *value = scramble_and_reverse(sobol as u32, rotate_bytes(seed, dimension as u32));
    }
    sample
}

pub(super) fn shuffled_scrambled_sobol_float<const DEPTH: usize>(
    index: u32,
    seed: u32,
) -> [f32; DEPTH] {
    let values = shuffled_scrambled_sobol::<DEPTH>(index, seed);
    let mut floats = [0.0; DEPTH];
    for (i, value) in values.into_iter().enumerate() {
        floats[i] = uint_to_float(value);
    }
    floats
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct IndexSampler {
    log2_samples_per_pixel: u32,
    n_base4_digits: u32,
}

impl IndexSampler {
    pub(super) fn new(samples_per_pixel: u32, full_resolution: UVec2) -> Self {
        assert!(samples_per_pixel > 0);
        let log2_samples_per_pixel = log2_int(round_up_pow2(samples_per_pixel));
        let resolution = round_up_pow2(full_resolution.x.max(full_resolution.y));
        let log4_samples_per_pixel = log2_samples_per_pixel.div_ceil(2);
        let n_base4_digits = log2_int(resolution) + log4_samples_per_pixel;

        Self {
            log2_samples_per_pixel,
            n_base4_digits,
        }
    }

    pub(super) fn samples_per_pixel(self) -> u32 {
        1 << self.log2_samples_per_pixel
    }

    pub(super) fn sample_index(self, pixel: UVec2, sample_index: u32, dimension: u32) -> u64 {
        debug_assert!(sample_index < self.samples_per_pixel());
        let morton_index = (encode_morton_2(pixel.x, pixel.y) << self.log2_samples_per_pixel)
            | sample_index as u64;
        self.permuted_index(morton_index, dimension)
    }

    fn permuted_index(self, morton_index: u64, dimension: u32) -> u64 {
        const PERMUTATIONS: [[u8; 4]; 24] = [
            [0, 1, 2, 3],
            [0, 1, 3, 2],
            [0, 2, 1, 3],
            [0, 2, 3, 1],
            [0, 3, 2, 1],
            [0, 3, 1, 2],
            [1, 0, 2, 3],
            [1, 0, 3, 2],
            [1, 2, 0, 3],
            [1, 2, 3, 0],
            [1, 3, 2, 0],
            [1, 3, 0, 2],
            [2, 1, 0, 3],
            [2, 1, 3, 0],
            [2, 0, 1, 3],
            [2, 0, 3, 1],
            [2, 3, 0, 1],
            [2, 3, 1, 0],
            [3, 1, 2, 0],
            [3, 1, 0, 2],
            [3, 2, 1, 0],
            [3, 2, 0, 1],
            [3, 0, 2, 1],
            [3, 0, 1, 2],
        ];

        let mut sample_index = 0;
        let pow2_samples = (self.log2_samples_per_pixel & 1) != 0;
        let last_digit = if pow2_samples { 1 } else { 0 };
        let dimension_hash = 0x5555_5555_u64.wrapping_mul(dimension as u64);

        for i in (last_digit..self.n_base4_digits).rev() {
            let digit_shift = 2 * i - u32::from(pow2_samples);
            let digit = ((morton_index >> digit_shift) & 3) as usize;
            let higher_digits = morton_index >> (digit_shift + 2);
            let permutation = ((mix_bits(higher_digits ^ dimension_hash) >> 24) % 24) as usize;
            let digit = PERMUTATIONS[permutation][digit] as u64;
            sample_index |= digit << digit_shift;
        }

        if pow2_samples {
            let digit = morton_index & 1;
            sample_index |= digit ^ (mix_bits((morton_index >> 1) ^ dimension_hash) & 1);
        }

        sample_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    const PRIMES: [u32; 20] = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    ];

    #[test]
    fn reverse_bits_matches_openqmc_examples() {
        let inputs_32 = [
            0b01010101010101010011001100110011_u32,
            0b11111111000000001111000011110000,
            0b11111111111111110000000011111111,
            0b11111111111111111111111111111111,
            0,
        ];
        let outputs_32 = [
            0b11001100110011001010101010101010_u32,
            0b00001111000011110000000011111111,
            0b11111111000000001111111111111111,
            0b11111111111111111111111111111111,
            0,
        ];
        for (&input, &output) in inputs_32.iter().zip(outputs_32.iter()) {
            assert_eq!(input, output.reverse_bits());
        }

        let inputs_16 = [
            0b0101010100110011_u16,
            0b0000000011110000,
            0b1111111100000000,
            0b1111111111111111,
            0,
        ];
        let outputs_16 = [
            0b1100110010101010_u16,
            0b0000111100000000,
            0b0000000011111111,
            0b1111111111111111,
            0,
        ];
        for (&input, &output) in inputs_16.iter().zip(outputs_16.iter()) {
            assert_eq!(input, output.reverse_bits());
        }
    }

    #[test]
    fn uint_to_float_matches_openqmc_edge_cases() {
        assert_eq!(uint_to_float(0), 0.0);
        assert!(uint_to_float(1) > 0.0);
        assert!(uint_to_float(1) < uint_to_float(2));
        assert_eq!(uint_to_float(u32::MAX), FLOAT_ONE_MINUS_EPSILON);
        assert!(uint_to_float(0xffff_feff) < FLOAT_ONE_MINUS_EPSILON);
        assert_eq!(uint_to_float(0xffff_ff00), FLOAT_ONE_MINUS_EPSILON);
        assert_eq!(uint_to_float(0xffff_ffff), FLOAT_ONE_MINUS_EPSILON);
        assert!(uint_to_float(0x7fff_ffff) < 0.5);
        assert_eq!(uint_to_float(0x8000_0000), 0.5);
        assert_eq!(uint_to_float(0x8000_00ff), 0.5);
        assert!(uint_to_float(0x8000_0100) > 0.5);
    }

    #[test]
    fn encode_morton_2_interleaves_x_and_y_bits() {
        assert_eq!(encode_morton_2(0, 0), 0);
        assert_eq!(encode_morton_2(1, 0), 1);
        assert_eq!(encode_morton_2(0, 1), 2);
        assert_eq!(encode_morton_2(1, 1), 3);
        assert_eq!(encode_morton_2(2, 0), 4);
        assert_eq!(encode_morton_2(0, 2), 8);
        assert_eq!(encode_morton_2(3, 3), 15);
        assert_eq!(encode_morton_2(4, 0), 16);
    }

    #[test]
    fn zsobol_sample_indices_are_unique_and_pixel_aligned() {
        let resolution = UVec2::new(16, 9);

        for log_samples in 0..=10 {
            let samples_per_pixel = 1 << log_samples;
            let sampler = IndexSampler::new(samples_per_pixel, resolution);

            for dimension in (0..7).step_by(3) {
                let mut returned_indices = BTreeSet::new();

                for y in 0..resolution.y {
                    for x in 0..resolution.x {
                        let pixel = UVec2::new(x, y);
                        let mut pow2_base = None;

                        for i in 0..samples_per_pixel {
                            let index = sampler.sample_index(pixel, i, dimension);
                            assert!(returned_indices.insert(index));

                            let base = index / samples_per_pixel as u64;
                            if let Some(pow2_base) = pow2_base {
                                assert_eq!(base, pow2_base);
                            } else {
                                pow2_base = Some(base);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn laine_karras_is_lower_bit_preserving_under_high_bit_flip() {
        let values = [
            0b01010101010101010011001100110011_u32,
            0b11111111000000001111000011110000,
            0b11111111111111110000000011111111,
            0b11111111111111111111111111111111,
            0,
        ];
        let mask = 0x0000_ffff;
        let flip = 0x0001_0000;

        for &value in &values {
            for &prime in &PRIMES {
                let v1 = laine_karras_permutation(value, prime);
                let v2 = laine_karras_permutation(value ^ flip, prime);
                assert_eq!(v1 & mask, v2 & mask);
                assert_ne!(v1 & !mask, v2 & !mask);
            }
        }
    }

    #[test]
    fn shuffle_is_a_permutation_over_low_bits() {
        const SIZE: usize = 1 << 4;
        const MASK: u32 = SIZE as u32 - 1;

        for &prime in &PRIMES {
            let mut seen = [false; SIZE];
            for i in 0..SIZE {
                let shuffled = reverse_and_shuffle(i as u32, prime);
                let permuted = shuffled.reverse_bits();
                assert_eq!(permuted, shuffle(i as u32, prime));

                let index = (permuted & MASK) as usize;
                assert!(!seen[index]);
                seen[index] = true;
            }
            assert!(seen.into_iter().all(|v| v));
        }
    }

    #[test]
    fn shuffled_scrambled_sobol_is_a_0_2_sequence_for_256_samples() {
        let m = 8;
        let n = 1 << m;
        for i in 0..=m {
            let x_resolution = 1 << i;
            let y_resolution = 1 << (m - i);
            assert_eq!(x_resolution * y_resolution, n);

            let x_width = u32::MAX / x_resolution;
            let y_width = u32::MAX / y_resolution;
            let mut strata = vec![false; n as usize];

            for index in 0..n {
                let out = shuffled_scrambled_sobol::<2>(index, pcg::hash(0));
                let x = out[0] / x_width;
                let y = out[1] / y_width;
                let coordinate = (x + y * x_resolution) as usize;

                assert!(!strata[coordinate]);
                strata[coordinate] = true;
            }

            assert!(strata.into_iter().all(|v| v));
        }
    }

    #[test]
    fn shuffled_scrambled_sobol_shirley_remapping_property() {
        const NUM_STRATA: u32 = 8;
        const NUM_SAMPLES: u32 = NUM_STRATA * NUM_STRATA;
        let width = u32::MAX / NUM_STRATA;

        for i in 0..NUM_STRATA {
            let mut strata = [false; NUM_STRATA as usize];
            for index in 0..NUM_SAMPLES {
                let out = shuffled_scrambled_sobol::<2>(index, pcg::hash(0));
                let x = out[0] / width;
                let y = out[1] / width;

                if x != i {
                    continue;
                }

                assert!(!strata[y as usize]);
                strata[y as usize] = true;
            }

            assert!(strata.into_iter().all(|v| v));
        }
    }
}
