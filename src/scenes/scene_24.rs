use glam::{Mat4, Vec3};
use std::{
    collections::HashMap,
    error::Error,
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::{
    camera::PinholeCamera,
    material::{
        DielectricGgxMaterial, EmissiveMaterial, Material, NormalizedLambertMaterial,
        SimplePbrMaterial, Texture, TextureColorSpace,
    },
    obj_scene::{ObjMaterial, ObjScene, load_obj_scene},
    scene::Scene,
};

struct DiffuseTextureCacheEntry {
    color: Arc<Texture>,
    alpha: Option<Arc<Texture>>,
}

const BISTRO_EXTERIOR_OBJ: &str = "assets/bistro/Exterior/exterior.obj";
const BISTRO_INTERIOR_OBJ: &str = "assets/bistro/Interior/interior.obj";
const STRONG_EMISSION_THRESHOLD: f32 = 1.0;
const STRONG_EMISSIVE_BOOST: f32 = 500.0;
const WEAK_EMISSIVE_BOOST: f32 = 1.0;

pub fn create_scene_24() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();
    let mut texture_cache: HashMap<PathBuf, DiffuseTextureCacheEntry> = HashMap::new();

    let mut total_emissive_meshes = 0usize;
    let mut total_emissive_tris = 0usize;
    let mut total_tris = 0usize;

    for obj_path in &[BISTRO_EXTERIOR_OBJ, BISTRO_INTERIOR_OBJ] {
        let obj_scene = load_obj_scene(Path::new(obj_path))?;
        let (emissive_meshes, emissive_tris, tris) =
            add_obj_scene_to_scene(&mut scene, &obj_scene, &mut texture_cache);
        total_emissive_meshes += emissive_meshes;
        total_emissive_tris += emissive_tris;
        total_tris += tris;
    }

    if total_emissive_meshes == 0 {
        return Err("scene 24 (Bistro) contains no emissive polygons; \
                    cannot illuminate the scene without sky and without emissive geometry"
            .into());
    }

    eprintln!(
        "scene 24 (Bistro): {} triangles, {} emissive meshes ({} emissive triangles)",
        total_tris, total_emissive_meshes, total_emissive_tris,
    );

    let camera = PinholeCamera::new(
        Vec3::new(-1000.0, 250.0, 140.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::Y,
        60.0_f32.to_radians(),
        1.0,
    );

    Ok((scene, camera))
}

fn add_obj_scene_to_scene(
    scene: &mut Scene,
    obj_scene: &ObjScene,
    texture_cache: &mut HashMap<PathBuf, DiffuseTextureCacheEntry>,
) -> (usize, usize, usize) {
    let mut emissive_meshes = 0usize;
    let mut emissive_tris = 0usize;
    let mut total_tris = 0usize;

    for slot in &obj_scene.material_meshes {
        let obj_material = slot
            .material_name
            .as_deref()
            .and_then(|name| obj_scene.material(name));
        let is_emissive = obj_material
            .map(|m| m.emission.length_squared() > 0.0)
            .unwrap_or(false);
        let tris = slot.mesh.triangle_count();
        total_tris += tris;
        if is_emissive {
            emissive_meshes += 1;
            emissive_tris += tris;
        }
        let material = bistro_material(obj_material, &obj_scene.mtl_dir, texture_cache);
        let material_index = scene.add_material(material);
        let mesh_index = scene.add_mesh(slot.mesh.clone());
        scene.add_instance(mesh_index, material_index, Mat4::IDENTITY);
    }

    (emissive_meshes, emissive_tris, total_tris)
}

fn bistro_material(
    obj_material: Option<&ObjMaterial>,
    mtl_dir: &Path,
    texture_cache: &mut HashMap<PathBuf, DiffuseTextureCacheEntry>,
) -> Material {
    let Some(material) = obj_material else {
        return Material::NormalizedLambert(NormalizedLambertMaterial::new(Vec3::splat(0.7)));
    };

    if material.emission.length_squared() > 0.0 {
        let boost = if material.emission.max_element() >= STRONG_EMISSION_THRESHOLD {
            STRONG_EMISSIVE_BOOST
        } else {
            WEAK_EMISSIVE_BOOST
        };
        let boosted = material.emission * boost;
        let strength = boosted.max_element().max(1.0);
        let color = (boosted / strength).clamp(Vec3::ZERO, Vec3::ONE);
        return Material::Emissive(EmissiveMaterial::new(color, strength));
    }

    if is_transparent(material) {
        let color = if material.transmission_filter.length_squared() > 0.0 {
            material.transmission_filter
        } else {
            material.diffuse
        };
        let color = color.clamp(Vec3::splat(0.05), Vec3::ONE);
        return Material::DielectricGgx(DielectricGgxMaterial::new(
            color,
            1.5,
            roughness_from_phong_exponent(material.specular_exponent),
            0.0,
            false,
        ));
    }

    let textures = material
        .diffuse_texture_path
        .as_deref()
        .and_then(|relative_path| load_diffuse_texture(mtl_dir, relative_path, texture_cache));

    let mut simple_pbr = SimplePbrMaterial::new(
        material.diffuse.clamp(Vec3::ZERO, Vec3::ONE),
        0.0,
        roughness_from_phong_exponent(material.specular_exponent),
        1.5,
        0.0,
    );
    if let Some(textures) = textures {
        simple_pbr.base_color_texture = Some(textures.color);
        simple_pbr.opacity_texture = textures.alpha;
    }
    Material::SimplePBR(simple_pbr)
}

fn load_diffuse_texture(
    mtl_dir: &Path,
    relative_path: &Path,
    cache: &mut HashMap<PathBuf, DiffuseTextureCacheEntry>,
) -> Option<DiffuseTextureCacheEntry> {
    let absolute_path = mtl_dir.join(relative_path);
    if let Some(entry) = cache.get(&absolute_path) {
        return Some(DiffuseTextureCacheEntry {
            color: Arc::clone(&entry.color),
            alpha: entry.alpha.as_ref().map(Arc::clone),
        });
    }
    match Texture::from_file_with_alpha(&absolute_path, TextureColorSpace::Srgb) {
        Ok((color, alpha)) => {
            let entry = DiffuseTextureCacheEntry {
                color: Arc::new(color),
                alpha: alpha.map(Arc::new),
            };
            cache.insert(
                absolute_path,
                DiffuseTextureCacheEntry {
                    color: Arc::clone(&entry.color),
                    alpha: entry.alpha.as_ref().map(Arc::clone),
                },
            );
            Some(entry)
        }
        Err(error) => {
            eprintln!(
                "warning: failed to load Bistro texture {}: {error}",
                absolute_path.display()
            );
            None
        }
    }
}

fn is_transparent(material: &ObjMaterial) -> bool {
    if material.dissolve < 0.999 {
        return true;
    }
    if material.illum == 4 || material.illum == 6 || material.illum == 7 || material.illum == 9 {
        return true;
    }
    false
}

fn roughness_from_phong_exponent(ns: f32) -> f32 {
    let ns = ns.max(1.0);
    let alpha = (2.0 / (ns + 2.0)).sqrt();
    alpha.sqrt().clamp(0.05, 1.0)
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, path::Path};

    use glam::Vec3;

    use crate::{material::Material, obj_scene::ObjMaterial};

    use crate::material::EmissiveMaterial;

    use super::{
        STRONG_EMISSIVE_BOOST, WEAK_EMISSIVE_BOOST, bistro_material, is_transparent,
        roughness_from_phong_exponent,
    };

    fn emissive_intensity(material: &Material) -> Vec3 {
        match material {
            Material::Emissive(EmissiveMaterial {
                color, strength, ..
            }) => *color * *strength,
            _ => panic!("expected emissive material"),
        }
    }

    fn opaque_obj_material() -> ObjMaterial {
        ObjMaterial {
            name: "wall".to_string(),
            diffuse: Vec3::new(0.8, 0.7, 0.6),
            specular_exponent: 16.0,
            dissolve: 1.0,
            illum: 2,
            transmission_filter: Vec3::ZERO,
            emission: Vec3::ZERO,
            diffuse_texture_path: None,
        }
    }

    fn build(material: Option<&ObjMaterial>) -> Material {
        let mut cache = HashMap::new();
        bistro_material(material, Path::new("."), &mut cache)
    }

    #[test]
    fn opaque_material_becomes_simple_pbr() {
        assert!(matches!(
            build(Some(&opaque_obj_material())),
            Material::SimplePBR(_)
        ));
    }

    #[test]
    fn dissolve_below_one_marks_material_transparent() {
        let mut obj = opaque_obj_material();
        obj.dissolve = 0.5;
        assert!(is_transparent(&obj));
        assert!(matches!(build(Some(&obj)), Material::DielectricGgx(_)));
    }

    #[test]
    fn illum_seven_marks_material_transparent() {
        let mut obj = opaque_obj_material();
        obj.illum = 7;
        assert!(is_transparent(&obj));
        assert!(matches!(build(Some(&obj)), Material::DielectricGgx(_)));
    }

    #[test]
    fn ke_marks_material_emissive() {
        let mut obj = opaque_obj_material();
        obj.emission = Vec3::new(2.0, 2.0, 2.0);
        assert!(matches!(build(Some(&obj)), Material::Emissive(_)));
    }

    #[test]
    fn strong_emission_uses_strong_boost() {
        let mut obj = opaque_obj_material();
        obj.emission = Vec3::new(2.0, 1.5, 1.0);
        let intensity = emissive_intensity(&build(Some(&obj)));
        let expected = obj.emission * STRONG_EMISSIVE_BOOST;
        assert!((intensity - expected).length() <= 1.0e-3);
    }

    #[test]
    fn weak_emission_uses_weak_boost() {
        let mut obj = opaque_obj_material();
        obj.emission = Vec3::new(0.2, 0.2, 0.2);
        let intensity = emissive_intensity(&build(Some(&obj)));
        let expected = obj.emission * WEAK_EMISSIVE_BOOST;
        assert!((intensity - expected).length() <= 1.0e-3);
    }

    #[test]
    fn emission_at_threshold_is_strong() {
        let mut obj = opaque_obj_material();
        obj.emission = Vec3::new(1.0, 0.0, 0.0);
        let intensity = emissive_intensity(&build(Some(&obj)));
        let expected = obj.emission * STRONG_EMISSIVE_BOOST;
        assert!((intensity - expected).length() <= 1.0e-3);
    }

    #[test]
    fn missing_obj_material_falls_back_to_lambert() {
        assert!(matches!(build(None), Material::NormalizedLambert(_)));
    }

    #[test]
    fn roughness_decreases_with_increasing_phong_exponent() {
        let r4 = roughness_from_phong_exponent(4.0);
        let r16 = roughness_from_phong_exponent(16.0);
        let r256 = roughness_from_phong_exponent(256.0);

        assert!(r4 > r16);
        assert!(r16 > r256);
        assert!(r256 >= 0.05);
        assert!(r4 <= 1.0);
    }
}
