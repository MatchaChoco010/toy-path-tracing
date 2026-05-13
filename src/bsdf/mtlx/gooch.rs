use glam::Vec3;

#[derive(Debug, Clone, Copy)]
pub struct GoochShadeKernel {
    pub warm: Vec3,
    pub cool: Vec3,
    pub specular_intensity: f32,
    pub shininess: f32,
    pub light_direction: Vec3,
}

impl GoochShadeKernel {
    pub fn eval(&self, normal_world: Vec3, _wo_world: Vec3) -> Vec3 {
        let l = self.light_direction.normalize_or(Vec3::Z);
        let n = normal_world.normalize_or_zero();
        let v = _wo_world.normalize_or_zero();
        let ndotl = n.dot(l);
        let mix_t = (ndotl * 0.5 + 0.5).clamp(0.0, 1.0);
        let diffuse = self.warm.lerp(self.cool, mix_t);
        let r = (v - 2.0 * n * v.dot(n)).normalize_or_zero();
        let specular = self.specular_intensity * (-l).dot(r).max(0.0).powf(self.shininess);
        diffuse + Vec3::splat(specular)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_v3(a: Vec3, b: Vec3) -> bool {
        (a - b).abs().max_element() < 1.0e-6
    }

    #[test]
    fn eval_matches_materialx_nodegraph_diffuse_mix_direction() {
        let kernel = GoochShadeKernel {
            warm: Vec3::new(0.8, 0.8, 0.7),
            cool: Vec3::new(0.3, 0.3, 0.8),
            specular_intensity: 0.0,
            shininess: 64.0,
            light_direction: Vec3::Z,
        };
        assert!(approx_v3(kernel.eval(Vec3::Z, Vec3::X), kernel.cool));
        assert!(approx_v3(kernel.eval(-Vec3::Z, Vec3::X), kernel.warm));
    }
}
