use glam::Vec3;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrthonormalBasis {
    tangent: Vec3,
    bitangent: Vec3,
    normal: Vec3,
}

impl OrthonormalBasis {
    pub fn from_normal(normal: Vec3) -> Self {
        let normal = if normal.length_squared() > 0.0 {
            normal.normalize()
        } else {
            Vec3::Z
        };
        // "Building an Orthonormal Basis, Revisited" (Duff et al.) avoids the
        // cancellation issues that show up with cross-product based frames.
        let sign = 1.0_f32.copysign(normal.z);
        let a = -1.0 / (sign + normal.z);
        let b = normal.x * normal.y * a;
        let tangent = Vec3::new(
            1.0 + sign * normal.x * normal.x * a,
            sign * b,
            -sign * normal.x,
        );
        let bitangent = Vec3::new(b, sign + normal.y * normal.y * a, -normal.y);

        Self {
            tangent,
            bitangent,
            normal,
        }
    }

    pub fn from_normal_and_tangent(normal: Vec3, tangent_hint: Vec3) -> Self {
        let normal = if normal.length_squared() > 0.0 {
            normal.normalize()
        } else {
            Vec3::Z
        };
        let tangent = (tangent_hint - tangent_hint.dot(normal) * normal).normalize_or_zero();

        if tangent.length_squared() == 0.0 {
            return Self::from_normal(normal);
        }

        let bitangent = normal.cross(tangent).normalize_or_zero();
        let tangent = bitangent.cross(normal).normalize_or_zero();

        Self {
            tangent,
            bitangent,
            normal,
        }
    }

    pub fn local_to_world(self, local_direction: Vec3) -> Vec3 {
        (local_direction.x * self.tangent
            + local_direction.y * self.bitangent
            + local_direction.z * self.normal)
            .normalize()
    }

    pub fn world_to_local(self, world_direction: Vec3) -> Vec3 {
        Vec3::new(
            world_direction.dot(self.tangent),
            world_direction.dot(self.bitangent),
            world_direction.dot(self.normal),
        )
    }

    pub fn tangent(self) -> Vec3 {
        self.tangent
    }

    pub fn bitangent(self) -> Vec3 {
        self.bitangent
    }

    pub fn normal(self) -> Vec3 {
        self.normal
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::OrthonormalBasis;

    #[test]
    fn basis_maps_normal_to_local_z() {
        let normal = Vec3::new(0.3, -0.4, 0.8660254).normalize();
        let basis = OrthonormalBasis::from_normal(normal);

        let local = basis.world_to_local(normal);

        assert!(local.abs_diff_eq(Vec3::Z, 1.0e-5));
    }

    #[test]
    fn basis_round_trips_direction() {
        let basis = OrthonormalBasis::from_normal(Vec3::new(-0.2, 0.9, 0.38).normalize());
        let local = Vec3::new(0.4, -0.3, 0.8660254).normalize();

        let world = basis.local_to_world(local);
        let reconstructed = basis.world_to_local(world);

        assert!(reconstructed.abs_diff_eq(local, 1.0e-5));
    }

    #[test]
    fn basis_respects_tangent_hint() {
        let basis = OrthonormalBasis::from_normal_and_tangent(Vec3::Z, Vec3::X);

        assert!(basis.tangent().abs_diff_eq(Vec3::X, 1.0e-6));
        assert!(basis.normal().abs_diff_eq(Vec3::Z, 1.0e-6));
    }
}
