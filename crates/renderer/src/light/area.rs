use glam::Vec2;

use super::{LightLiSample, LightSampleContext, LightType};
use crate::{
    material::{MtlxScratch, ShadingVertex},
    scene::{Scene, TriangleRef},
};

/// Sample direct illumination from a *given* triangle leaf. The triangle is
/// chosen by the SG light tree; this function only handles the 2-D
/// barycentric sampling and the area-to-solid-angle conversion.
pub(super) fn sample_li_for_triangle(
    scene: &Scene,
    triangle: TriangleRef,
    ctx: &LightSampleContext,
    us: Vec2,
    mtlx_scratch: &mut MtlxScratch,
) -> Option<LightLiSample> {
    let point = scene.sample_triangle_point(triangle, us);
    if point.pdf_area <= 0.0 {
        return None;
    }

    let to_light = point.p - ctx.p;
    let distance_squared = to_light.length_squared();
    if distance_squared <= 0.0 {
        return None;
    }
    let distance = distance_squared.sqrt();
    let wi = to_light / distance;

    let light_material = scene.instance_material(triangle.instance_index);
    if !light_material.may_emit() {
        return None;
    }

    let mut lvtx = scene.shading_vertex_from_triangle_sample(triangle, point.barycentric, wi);
    light_material.precompute_shading(&mut lvtx, mtlx_scratch);
    let le = light_material.le(&lvtx, mtlx_scratch)?;

    let cos_light = lvtx.ng.dot(-wi).max(0.0);
    if cos_light <= 0.0 {
        return None;
    }

    let pdf_solid_angle = point.pdf_area * distance_squared / cos_light;

    Some(LightLiSample {
        radiance: le,
        wi,
        pdf: pdf_solid_angle,
        distance,
        light_type: LightType::Area,
        target_triangle: Some(triangle),
    })
}

pub fn area_light_pdf_li(scene: &Scene, vtx: &ShadingVertex, lvtx: &ShadingVertex) -> f32 {
    scene.area_light_pdf_solid_angle(vtx, lvtx).unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use glam::{Vec2, Vec3};

    use super::super::LightSampleContext;
    use super::super::test_helpers::unit_mesh;
    use super::sample_li_for_triangle;
    use crate::{
        material::{EmissiveMaterial, Material, MtlxScratch, NormalizedLambertMaterial},
        scene::{Scene, TriangleRef},
    };

    #[test]
    fn sample_li_for_triangle_matches_solid_angle_pdf() {
        let mut scene = Scene::new();
        let floor_mesh = scene.add_mesh(unit_mesh(0.0));
        let light_mesh = scene.add_mesh(unit_mesh(1.0));
        let floor_material = scene.add_material(Material::NormalizedLambert(
            NormalizedLambertMaterial::new(Vec3::splat(0.8)),
        ));
        let light_material =
            scene.add_material(Material::Emissive(EmissiveMaterial::new(Vec3::ONE, 10.0)));
        scene.add_instance(floor_mesh, floor_material, glam::Mat4::IDENTITY);
        let light_instance = scene.add_instance(light_mesh, light_material, glam::Mat4::IDENTITY);
        scene.build_qbvh();
        scene.build_light_tree();

        let ctx = LightSampleContext {
            p: Vec3::new(0.25, 0.25, 0.0),
            ng: Vec3::Z,
            ns: Vec3::Z,
        };
        let triangle = TriangleRef {
            instance_index: light_instance,
            triangle_index: 0,
        };
        let mut scratch = MtlxScratch::default();
        let li = sample_li_for_triangle(&scene, triangle, &ctx, Vec2::new(0.25, 0.5), &mut scratch)
            .expect("expected a sample");

        assert!((li.pdf - 2.0).abs() < 1.0e-4);
        assert!(li.radiance.abs_diff_eq(Vec3::splat(10.0), 1.0e-5));
        assert!((li.distance - 1.0).abs() < 1.0e-5);
        assert!(li.wi.abs_diff_eq(Vec3::Z, 1.0e-5));
    }
}
