//! Amazon Lumberyard Bistro の Exterior + Interior を読み込み、シーン中の emissive ポリゴンだけで照らす。

use glam::{Mat4, Vec3};
use std::{error::Error, path::Path, sync::Arc};

use crate::{
    camera::PinholeCamera,
    color::srgb_to_linear,
    material::{
        DielectricGgxMaterial, EmissiveMaterial, Material, NormalizedLambertMaterial,
        ScalarTexture, SimplePbrMaterial, Texture,
    },
    scene::Scene,
    scene_loader::gltf_scene::{
        GltfAlphaMode, GltfImage, GltfMaterial, GltfMaterialMesh, GltfScene, load_gltf_scene,
    },
};

const BISTRO_GLTF_FILES: &[&str] = &[
    "assets/bistro/gltf/BistroExterior/BistroExterior.gltf",
    "assets/bistro/gltf/BistroInterior/BistroInterior.gltf",
    "assets/bistro/gltf/BistroInterior_Wine/BistroInterior_Wine.gltf",
];

const EMISSIVE_SCALE: f32 = 512.0;

struct ImageCache {
    color: Vec<Option<Arc<Texture>>>,
    alpha: Vec<Option<Option<Arc<ScalarTexture>>>>,
    metallic: Vec<Option<Arc<ScalarTexture>>>,
    roughness: Vec<Option<Arc<ScalarTexture>>>,
}

pub fn create_scene_24() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();
    let mut total_emissive_meshes = 0usize;
    let mut total_emissive_tris = 0usize;
    let mut total_tris = 0usize;
    let mut total_simple_pbr_meshes = 0usize;
    let mut total_dielectric_meshes = 0usize;
    let mut total_lambert_fallback_meshes = 0usize;
    let mut found_any_glb = false;

    for path in BISTRO_GLTF_FILES {
        let glb_path = Path::new(path);
        if !glb_path.exists() {
            continue;
        }
        found_any_glb = true;
        let gltf_scene = load_gltf_scene(glb_path)?;
        let mut cache = ImageCache::new(gltf_scene.images.len());

        for slot in &gltf_scene.material_meshes {
            let tris = slot.mesh.triangle_count();
            total_tris += tris;
            let material = build_material(slot, &gltf_scene, &mut cache);
            match &material {
                Material::Emissive(_) => {
                    total_emissive_meshes += 1;
                    total_emissive_tris += tris;
                }
                Material::SimplePBR(_) => total_simple_pbr_meshes += 1,
                Material::DielectricGgx(_) => total_dielectric_meshes += 1,
                Material::NormalizedLambert(_) => total_lambert_fallback_meshes += 1,
                _ => {}
            }
            let material_index = scene.add_material(material);
            let mesh_index = scene.add_mesh(slot.mesh.clone());
            scene.add_instance(mesh_index, material_index, Mat4::IDENTITY);
        }
    }

    if !found_any_glb {
        return Err(format!(
            "scene 24 expected at least one of {BISTRO_GLTF_FILES:?}; \
             run `bash assets/download.sh` to fetch and convert the original Bistro asset"
        )
        .into());
    }

    if total_emissive_meshes == 0 {
        return Err("scene 24 (Bistro) contains no emissive polygons; \
                    cannot illuminate the scene without sky and without emissive geometry"
            .into());
    }

    eprintln!(
        "scene 24 (Bistro original): {} triangles, emissive={} ({} tri), \
         simple_pbr={}, dielectric_ggx={}, lambert_fallback={}",
        total_tris,
        total_emissive_meshes,
        total_emissive_tris,
        total_simple_pbr_meshes,
        total_dielectric_meshes,
        total_lambert_fallback_meshes,
    );

    let camera = PinholeCamera::new(
        Vec3::new(-17.0, 4.5, 2.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::Y,
        60.0_f32.to_radians(),
        1.0,
    );

    Ok((scene, camera))
}

impl ImageCache {
    fn new(count: usize) -> Self {
        Self {
            color: vec![None; count],
            alpha: vec![None; count],
            metallic: vec![None; count],
            roughness: vec![None; count],
        }
    }

    fn srgb_color(&mut self, images: &[GltfImage], index: usize) -> Arc<Texture> {
        if let Some(texture) = &self.color[index] {
            return Arc::clone(texture);
        }
        let texture = Arc::new(decode_srgb_color(&images[index]));
        self.color[index] = Some(Arc::clone(&texture));
        texture
    }

    fn alpha(&mut self, images: &[GltfImage], index: usize) -> Option<Arc<ScalarTexture>> {
        if let Some(entry) = &self.alpha[index] {
            return entry.clone();
        }
        let texture = decode_alpha(&images[index]).map(Arc::new);
        self.alpha[index] = Some(texture.clone());
        texture
    }

    fn metallic(&mut self, images: &[GltfImage], index: usize) -> Arc<ScalarTexture> {
        if let Some(texture) = &self.metallic[index] {
            return Arc::clone(texture);
        }
        let image = &images[index];
        let texture = Arc::new(ScalarTexture::from_rgba_channel(
            image.width,
            image.height,
            &image.rgba,
            2,
        ));
        self.metallic[index] = Some(Arc::clone(&texture));
        texture
    }

    fn roughness(&mut self, images: &[GltfImage], index: usize) -> Arc<ScalarTexture> {
        if let Some(texture) = &self.roughness[index] {
            return Arc::clone(texture);
        }
        let image = &images[index];
        let texture = Arc::new(ScalarTexture::from_rgba_channel(
            image.width,
            image.height,
            &image.rgba,
            1,
        ));
        self.roughness[index] = Some(Arc::clone(&texture));
        texture
    }
}

fn decode_srgb_color(image: &GltfImage) -> Texture {
    let mut pixels = Vec::with_capacity(image.width * image.height);
    for chunk in image.rgba.chunks_exact(4) {
        let rgb = Vec3::new(
            chunk[0] as f32 / 255.0,
            chunk[1] as f32 / 255.0,
            chunk[2] as f32 / 255.0,
        );
        pixels.push(srgb_to_linear(rgb));
    }
    Texture::from_pixels(image.width, image.height, pixels)
}

fn decode_alpha(image: &GltfImage) -> Option<ScalarTexture> {
    let mut pixels = Vec::with_capacity(image.width * image.height);
    let mut nontrivial = false;
    for chunk in image.rgba.chunks_exact(4) {
        let alpha = chunk[3] as f32 / 255.0;
        if alpha < 1.0 - 1.0e-3 {
            nontrivial = true;
        }
        pixels.push(alpha);
    }
    if !nontrivial {
        return None;
    }
    Some(ScalarTexture::from_pixels(
        image.width,
        image.height,
        pixels,
    ))
}

fn build_material(
    slot: &GltfMaterialMesh,
    gltf_scene: &GltfScene,
    cache: &mut ImageCache,
) -> Material {
    let Some(material_index) = slot.material_index else {
        return Material::NormalizedLambert(NormalizedLambertMaterial::new(Vec3::splat(0.7)));
    };
    let Some(material) = gltf_scene.materials.get(material_index) else {
        return Material::NormalizedLambert(NormalizedLambertMaterial::new(Vec3::splat(0.7)));
    };

    if let Some(emissive) = build_emissive(material, &gltf_scene.images, cache) {
        return emissive;
    }

    if matches!(material.alpha_mode, GltfAlphaMode::Blend) {
        return build_dielectric(material);
    }

    build_simple_pbr(material, &gltf_scene.images, cache)
}

fn build_emissive(
    material: &GltfMaterial,
    images: &[GltfImage],
    cache: &mut ImageCache,
) -> Option<Material> {
    let factor = material.emissive_factor * material.emissive_strength;
    let factor_max = factor.max_element();
    let texture_index = material.emissive_texture;

    if factor_max <= 0.0 && texture_index.is_none() {
        return None;
    }

    let (color, strength, texture) = if factor_max > 0.0 {
        let strength = factor_max;
        let chroma = (factor / strength).clamp(Vec3::ZERO, Vec3::ONE);
        let texture = texture_index.map(|index| cache.srgb_color(images, index));
        (chroma, strength, texture)
    } else {
        let index = texture_index?;
        (Vec3::ONE, 1.0, Some(cache.srgb_color(images, index)))
    };

    let mut emissive_material = EmissiveMaterial::new(color, strength * EMISSIVE_SCALE);
    emissive_material.color_texture = texture;
    Some(Material::Emissive(emissive_material))
}

fn build_dielectric(material: &GltfMaterial) -> Material {
    let factor: Vec3 = material.base_color_factor.truncate();
    let color =
        srgb_to_linear(factor.clamp(Vec3::ZERO, Vec3::ONE)).clamp(Vec3::splat(0.05), Vec3::ONE);
    Material::DielectricGgx(DielectricGgxMaterial::new(
        color,
        1.5,
        material.roughness_factor.clamp(0.05, 1.0),
        0.0,
        false,
    ))
}

fn build_simple_pbr(
    material: &GltfMaterial,
    images: &[GltfImage],
    cache: &mut ImageCache,
) -> Material {
    let factor: Vec3 = material.base_color_factor.truncate();
    let base_color = srgb_to_linear(factor.clamp(Vec3::ZERO, Vec3::ONE));

    // glTF metallicRoughness: final = factor * texture_channel. Without a
    // texture we floor the roughness to keep specular lobes finite; with a
    // texture we let smooth metals such as chrome trim stay sharp and rely on
    // SimplePbrMaterial's internal alpha clamp.
    let metallic_factor = material.metallic_factor.clamp(0.0, 1.0);
    let roughness_factor = if material.metallic_roughness_texture.is_some() {
        material.roughness_factor.clamp(0.0, 1.0)
    } else {
        material.roughness_factor.clamp(0.05, 1.0)
    };

    let mut simple_pbr =
        SimplePbrMaterial::new(base_color, metallic_factor, roughness_factor, 1.5, 0.0);

    if let Some(index) = material.base_color_texture {
        simple_pbr.base_color_texture = Some(cache.srgb_color(images, index));
        simple_pbr.opacity_texture = cache.alpha(images, index);
    }

    if let Some(index) = material.metallic_roughness_texture {
        simple_pbr.metallic_texture = Some(cache.metallic(images, index));
        simple_pbr.roughness_texture = Some(cache.roughness(images, index));
    }

    Material::SimplePBR(simple_pbr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec4;

    fn opaque_gltf_material() -> GltfMaterial {
        GltfMaterial {
            name: "wall".to_string(),
            base_color_factor: Vec4::new(0.8, 0.7, 0.6, 1.0),
            base_color_texture: None,
            metallic_factor: 0.0,
            roughness_factor: 0.5,
            metallic_roughness_texture: None,
            emissive_factor: Vec3::ZERO,
            emissive_strength: 1.0,
            emissive_texture: None,
            alpha_mode: GltfAlphaMode::Opaque,
            double_sided: false,
            unlit: false,
        }
    }

    fn build(material: GltfMaterial) -> Material {
        build_with_images(material, Vec::new())
    }

    fn build_with_images(material: GltfMaterial, images: Vec<GltfImage>) -> Material {
        let gltf_scene = GltfScene {
            material_meshes: Vec::new(),
            materials: vec![material],
            images: images.clone(),
        };
        let mut cache = ImageCache::new(images.len());
        let slot = GltfMaterialMesh {
            material_index: Some(0),
            mesh: crate::mesh::Mesh::new(
                vec![crate::mesh::Vertex {
                    position: Vec3::ZERO,
                    normal: Vec3::Z,
                    uv: glam::Vec2::ZERO,
                }],
                vec![0, 0, 0],
            ),
        };
        build_material(&slot, &gltf_scene, &mut cache)
    }

    #[test]
    fn opaque_material_becomes_simple_pbr() {
        assert!(matches!(
            build(opaque_gltf_material()),
            Material::SimplePBR(_)
        ));
    }

    #[test]
    fn blend_alpha_mode_becomes_dielectric_ggx() {
        let mut material = opaque_gltf_material();
        material.alpha_mode = GltfAlphaMode::Blend;
        assert!(matches!(build(material), Material::DielectricGgx(_)));
    }

    #[test]
    fn strong_emissive_becomes_emissive_material() {
        let mut material = opaque_gltf_material();
        material.emissive_factor = Vec3::new(2.0, 1.0, 0.5);
        assert!(matches!(build(material), Material::Emissive(_)));
    }

    #[test]
    fn nonzero_emissive_factor_without_texture_becomes_emissive_material() {
        let mut material = opaque_gltf_material();
        material.emissive_factor = Vec3::new(0.2, 0.2, 0.2);
        assert!(matches!(build(material), Material::Emissive(_)));
    }

    #[test]
    fn metallic_roughness_texture_populates_metallic_and_roughness_textures() {
        // Bistro packs occlusion/roughness/metalness into RGB channels of a
        // single image. Verify that scene_24 wires the G channel (roughness)
        // and B channel (metalness) into separate scalar textures and that
        // they read back the expected values.
        let mut material = opaque_gltf_material();
        material.metallic_roughness_texture = Some(0);
        // Solid colour image: R=10, G=128, B=200, A=255.
        let image = GltfImage {
            width: 2,
            height: 2,
            rgba: vec![
                10, 128, 200, 255, 10, 128, 200, 255, 10, 128, 200, 255, 10, 128, 200, 255,
            ],
        };
        let built = build_with_images(material, vec![image]);
        let Material::SimplePBR(simple_pbr) = built else {
            panic!("expected SimplePBR material");
        };
        let roughness = simple_pbr
            .roughness_texture
            .as_ref()
            .expect("roughness texture should be populated");
        let metallic = simple_pbr
            .metallic_texture
            .as_ref()
            .expect("metallic texture should be populated");
        let r = roughness.sample(glam::Vec2::splat(0.5));
        let m = metallic.sample(glam::Vec2::splat(0.5));
        assert!((r - 128.0 / 255.0).abs() < 1.0e-5);
        assert!((m - 200.0 / 255.0).abs() < 1.0e-5);
    }
}
