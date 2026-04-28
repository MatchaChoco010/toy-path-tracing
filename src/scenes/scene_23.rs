use glam::{Mat4, Vec3};
use std::{
    collections::HashMap,
    error::Error,
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::{
    camera::PinholeCamera,
    color::srgb_to_linear,
    light::EnvironmentLight,
    material::{
        DielectricGgxMaterial, EmissiveMaterial, Material, NormalizedLambertMaterial,
        SimplePbrMaterial, Texture, TextureColorSpace,
    },
    scene_loader::obj_scene::{ObjMaterial, ObjScene, load_obj_scene},
    scene::Scene,
};

struct DiffuseTextureCacheEntry {
    color: Arc<Texture>,
    alpha: Option<Arc<Texture>>,
}

const SAN_MIGUEL_OBJ: &str = "assets/san_miguel_2.0/san-miguel.obj";
const SAN_MIGUEL_HDR: &str = "assets/sky/kloofendal_48d_partly_cloudy_puresky_4k.hdr";

pub fn create_scene_23() -> Result<(Scene, PinholeCamera), Box<dyn Error>> {
    let mut scene = Scene::new();
    let obj_scene = load_obj_scene(Path::new(SAN_MIGUEL_OBJ))?;
    add_san_miguel_to_scene(&mut scene, &obj_scene);

    let env = EnvironmentLight::from_hdr_file(SAN_MIGUEL_HDR, 50.0, 0.0)?;
    scene.set_environment_light(env);

    let camera = PinholeCamera::new(
        Vec3::new(6.1, 1.8, 1.1),
        Vec3::new(6.2, 1.78, 1.15),
        Vec3::Y,
        60.0_f32.to_radians(),
        1.0,
    );

    Ok((scene, camera))
}

fn add_san_miguel_to_scene(scene: &mut Scene, obj_scene: &ObjScene) {
    let mut texture_cache: HashMap<PathBuf, DiffuseTextureCacheEntry> = HashMap::new();
    for slot in &obj_scene.material_meshes {
        let obj_material = slot
            .material_name
            .as_deref()
            .and_then(|name| obj_scene.material(name));
        let material = san_miguel_material(obj_material, &obj_scene.mtl_dir, &mut texture_cache);
        let material_index = scene.add_material(material);
        let mesh_index = scene.add_mesh(slot.mesh.clone());
        scene.add_instance(mesh_index, material_index, Mat4::IDENTITY);
    }
}

fn san_miguel_material(
    obj_material: Option<&ObjMaterial>,
    mtl_dir: &Path,
    texture_cache: &mut HashMap<PathBuf, DiffuseTextureCacheEntry>,
) -> Material {
    let Some(material) = obj_material else {
        return Material::NormalizedLambert(NormalizedLambertMaterial::new(Vec3::splat(0.7)));
    };

    if material.emission.length_squared() > 0.0 {
        let ke = material.emission;
        let strength = ke.max_element().max(1.0);
        let chroma = (ke / strength).clamp(Vec3::ZERO, Vec3::ONE);
        let color = srgb_to_linear(chroma);
        return Material::Emissive(EmissiveMaterial::new(color, strength));
    }

    if is_transparent(material) {
        let color = if material.transmission_filter.length_squared() > 0.0 {
            srgb_to_linear(material.transmission_filter)
        } else {
            srgb_to_linear(material.diffuse)
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
        srgb_to_linear(material.diffuse).clamp(Vec3::ZERO, Vec3::ONE),
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
                "warning: failed to load San Miguel texture {}: {error}",
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

    use crate::{material::Material, scene_loader::obj_scene::ObjMaterial};

    use super::{is_transparent, roughness_from_phong_exponent, san_miguel_material};

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
            emission_texture_path: None,
        }
    }

    fn build(material: Option<&ObjMaterial>) -> Material {
        let mut cache = HashMap::new();
        san_miguel_material(material, Path::new("."), &mut cache)
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
    fn illum_four_with_tf_marks_material_transparent() {
        let mut obj = opaque_obj_material();
        obj.illum = 4;
        obj.transmission_filter = Vec3::new(0.1, 0.1, 0.1);
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
