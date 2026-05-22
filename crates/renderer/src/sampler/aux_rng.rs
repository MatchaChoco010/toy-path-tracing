use super::pcg;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuxRng {
    state: u32,
}

impl AuxRng {
    /// Auxiliary PRNG for randomness that should not consume fixed QMC
    /// dimensions: stochastic helper work in eval, variable-length random
    /// walks in sample, any-hit alpha tests, and similar secondary decisions.
    /// Do not use this for the main camera, material, light, or Russian
    /// roulette samples; those should come from explicitly assigned sampler
    /// dimensions.
    pub fn from_seed(seed: u32) -> Self {
        Self {
            state: pcg::state_transition(seed),
        }
    }

    pub fn next_u32(&mut self) -> u32 {
        self.state = pcg::state_transition(self.state);
        pcg::output(self.state)
    }

    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u32() >> 8) as f32) * (1.0 / 16_777_216.0)
    }
}

#[cfg(test)]
impl Default for AuxRng {
    fn default() -> Self {
        Self::from_seed(0)
    }
}
