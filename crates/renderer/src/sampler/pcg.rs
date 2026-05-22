pub(super) const fn state_transition(state: u32) -> u32 {
    state.wrapping_mul(747_796_405).wrapping_add(2_891_336_453)
}

pub(super) const fn output(mut state: u32) -> u32 {
    state ^= state >> (4 + (state >> 28));
    state = state.wrapping_mul(277_803_737);
    state ^ (state >> 22)
}

pub(super) const fn hash(key: u32) -> u32 {
    output(state_transition(key))
}

#[cfg(test)]
pub(super) const fn init(seed: u32) -> u32 {
    state_transition(0).wrapping_add(seed)
}

#[cfg(test)]
pub(super) fn rng(state: &mut u32) -> u32 {
    *state = state_transition(*state);
    output(*state)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRIMES: [u32; 20] = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    ];

    #[test]
    fn state_and_output_behave_like_openqmc() {
        assert_ne!(state_transition(0), 0);
        assert_eq!(output(0), 0);
        assert_eq!(init(0), state_transition(0));

        for &input in &PRIMES {
            assert_ne!(state_transition(input), input);
            assert_ne!(output(input), input);
            assert_ne!(state_transition(input), output(input));
        }
    }

    #[test]
    fn hash_matches_rng_after_init() {
        for &seed in &PRIMES {
            let mut state = init(seed);
            let hash = hash(state);
            let rnd = rng(&mut state);
            assert_eq!(hash, rnd);
        }
    }
}
