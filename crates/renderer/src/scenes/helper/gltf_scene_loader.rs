use std::{error::Error, fmt, path::Path};

use glam::{Mat3, Mat4, Vec2, Vec3, Vec4};

use crate::scene::{LoadMeshError, Mesh, Vertex, generate_vertex_normals};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GltfAlphaMode {
    Opaque,
    Mask(f32),
    Blend,
}

#[derive(Debug, Clone)]
pub struct GltfMaterial {
    pub base_color_factor: Vec4,
    pub base_color_texture: Option<usize>,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub metallic_roughness_texture: Option<usize>,
    pub emissive_factor: Vec3,
    pub emissive_strength: f32,
    pub emissive_texture: Option<usize>,
    pub alpha_mode: GltfAlphaMode,
}

#[derive(Debug, Clone)]
pub struct GltfImage {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct GltfMaterialMesh {
    pub material_index: Option<usize>,
    pub mesh: Mesh,
}

#[derive(Debug, Clone)]
pub struct GltfScene {
    pub material_meshes: Vec<GltfMaterialMesh>,
    pub materials: Vec<GltfMaterial>,
    pub images: Vec<GltfImage>,
}

#[derive(Debug)]
pub enum LoadGltfSceneError {
    Gltf(gltf::Error),
    Mesh(LoadMeshError),
    UnsupportedImageFormat(gltf::image::Format),
    UnsupportedPrimitiveMode(gltf::mesh::Mode),
    MissingPositions,
    InvalidTriangleIndexCount,
}

impl fmt::Display for LoadGltfSceneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gltf(error) => write!(f, "{error}"),
            Self::Mesh(error) => write!(f, "{error}"),
            Self::UnsupportedImageFormat(format) => {
                write!(f, "unsupported glTF image format: {format:?}")
            }
            Self::UnsupportedPrimitiveMode(mode) => {
                write!(f, "unsupported glTF primitive mode: {mode:?}")
            }
            Self::MissingPositions => write!(f, "glTF primitive is missing POSITION data"),
            Self::InvalidTriangleIndexCount => {
                write!(f, "glTF primitive index count is not a multiple of three")
            }
        }
    }
}

impl Error for LoadGltfSceneError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Gltf(error) => Some(error),
            Self::Mesh(error) => Some(error),
            _ => None,
        }
    }
}

impl From<gltf::Error> for LoadGltfSceneError {
    fn from(error: gltf::Error) -> Self {
        Self::Gltf(error)
    }
}

impl From<LoadMeshError> for LoadGltfSceneError {
    fn from(error: LoadMeshError) -> Self {
        Self::Mesh(error)
    }
}

pub fn load_gltf_scene(path: &Path) -> Result<GltfScene, LoadGltfSceneError> {
    let path = crate::utils::workspace_path(path);
    let (document, buffers, images) = gltf::import(path)?;

    let mut material_meshes = Vec::new();
    if let Some(scene) = document
        .default_scene()
        .or_else(|| document.scenes().next())
    {
        for node in scene.nodes() {
            walk_node(&buffers, node, Mat4::IDENTITY, &mut material_meshes)?;
        }
    } else {
        for mesh in document.meshes() {
            append_primitives(&buffers, mesh, Mat4::IDENTITY, &mut material_meshes)?;
        }
    }

    let materials = document.materials().map(build_material).collect();
    let images = images
        .into_iter()
        .map(normalize_image)
        .collect::<Result<Vec<_>, _>>()?;

    Ok(GltfScene {
        material_meshes,
        materials,
        images,
    })
}

fn walk_node(
    buffers: &[gltf::buffer::Data],
    node: gltf::Node<'_>,
    parent_transform: Mat4,
    output: &mut Vec<GltfMaterialMesh>,
) -> Result<(), LoadGltfSceneError> {
    let local_transform = Mat4::from_cols_array_2d(&node.transform().matrix());
    let node_transform = parent_transform * local_transform;

    if let Some(mesh) = node.mesh() {
        append_primitives(buffers, mesh, node_transform, output)?;
    }
    for child in node.children() {
        walk_node(buffers, child, node_transform, output)?;
    }

    Ok(())
}

fn append_primitives(
    buffers: &[gltf::buffer::Data],
    mesh: gltf::Mesh<'_>,
    transform: Mat4,
    output: &mut Vec<GltfMaterialMesh>,
) -> Result<(), LoadGltfSceneError> {
    let normal_transform = Mat3::from_mat4(transform.inverse().transpose());

    for primitive in mesh.primitives() {
        if primitive.mode() != gltf::mesh::Mode::Triangles {
            return Err(LoadGltfSceneError::UnsupportedPrimitiveMode(
                primitive.mode(),
            ));
        }

        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
        let positions = reader
            .read_positions()
            .ok_or(LoadGltfSceneError::MissingPositions)?
            .map(Vec3::from_array)
            .map(|position| transform.transform_point3(position))
            .collect::<Vec<_>>();

        let indices = reader
            .read_indices()
            .map(|indices| indices.into_u32().collect::<Vec<_>>())
            .unwrap_or_else(|| (0..positions.len() as u32).collect());

        if indices.len() % 3 != 0 {
            return Err(LoadGltfSceneError::InvalidTriangleIndexCount);
        }

        let uvs = reader
            .read_tex_coords(0)
            .map(|uvs| uvs.into_f32().map(Vec2::from_array).collect::<Vec<_>>())
            .unwrap_or_else(|| vec![Vec2::ZERO; positions.len()]);

        let generated_normals = generate_vertex_normals(&positions, &indices);
        let normals = reader
            .read_normals()
            .map(|normals| {
                normals
                    .map(Vec3::from_array)
                    .map(|normal| normal_transform.mul_vec3(normal).normalize_or_zero())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| generated_normals.clone());

        let mut vertices = Vec::with_capacity(positions.len());
        for index in 0..positions.len() {
            let normal = normals[index];
            vertices.push(Vertex {
                position: positions[index],
                normal: if normal.length_squared() > 0.0 {
                    normal
                } else {
                    generated_normals[index]
                },
                uv: uvs[index],
            });
        }

        if vertices.is_empty() || indices.is_empty() {
            continue;
        }

        let primitive_mesh = Mesh::new(vertices, indices);
        output.push(GltfMaterialMesh {
            material_index: primitive.material().index(),
            mesh: primitive_mesh,
        });
    }

    Ok(())
}

fn build_material(material: gltf::Material<'_>) -> GltfMaterial {
    let pbr = material.pbr_metallic_roughness();
    let base_color_factor = Vec4::from_array(pbr.base_color_factor());
    let emissive_factor = Vec3::from_array(material.emissive_factor());
    let emissive_strength = 1.0;

    let alpha_mode = match material.alpha_mode() {
        gltf::material::AlphaMode::Opaque => GltfAlphaMode::Opaque,
        gltf::material::AlphaMode::Mask => {
            GltfAlphaMode::Mask(material.alpha_cutoff().unwrap_or(0.5))
        }
        gltf::material::AlphaMode::Blend => GltfAlphaMode::Blend,
    };

    GltfMaterial {
        base_color_factor,
        base_color_texture: pbr
            .base_color_texture()
            .map(|info| info.texture().source().index()),
        metallic_factor: pbr.metallic_factor(),
        roughness_factor: pbr.roughness_factor(),
        metallic_roughness_texture: pbr
            .metallic_roughness_texture()
            .map(|info| info.texture().source().index()),
        emissive_factor,
        emissive_strength,
        emissive_texture: material
            .emissive_texture()
            .map(|info| info.texture().source().index()),
        alpha_mode,
    }
}

fn normalize_image(data: gltf::image::Data) -> Result<GltfImage, LoadGltfSceneError> {
    let width = data.width as usize;
    let height = data.height as usize;
    let pixel_count = width * height;
    let rgba = match data.format {
        gltf::image::Format::R8 => expand_to_rgba(&data.pixels, pixel_count, |chunk| {
            [chunk[0], chunk[0], chunk[0], 255]
        }),
        gltf::image::Format::R8G8 => expand_to_rgba(&data.pixels, pixel_count, |chunk| {
            [chunk[0], chunk[1], 0, 255]
        }),
        gltf::image::Format::R8G8B8 => expand_to_rgba(&data.pixels, pixel_count, |chunk| {
            [chunk[0], chunk[1], chunk[2], 255]
        }),
        gltf::image::Format::R8G8B8A8 => data.pixels,
        gltf::image::Format::R16 => narrow_16_to_rgba(&data.pixels, pixel_count, |chunk| {
            [chunk[0], chunk[0], chunk[0], u16::MAX]
        }),
        gltf::image::Format::R16G16 => narrow_16_to_rgba(&data.pixels, pixel_count, |chunk| {
            [chunk[0], chunk[1], 0, u16::MAX]
        }),
        gltf::image::Format::R16G16B16 => narrow_16_to_rgba(&data.pixels, pixel_count, |chunk| {
            [chunk[0], chunk[1], chunk[2], u16::MAX]
        }),
        gltf::image::Format::R16G16B16A16 => {
            narrow_16_to_rgba(&data.pixels, pixel_count, |chunk| {
                [chunk[0], chunk[1], chunk[2], chunk[3]]
            })
        }
        format @ (gltf::image::Format::R32G32B32FLOAT | gltf::image::Format::R32G32B32A32FLOAT) => {
            return Err(LoadGltfSceneError::UnsupportedImageFormat(format));
        }
    };

    Ok(GltfImage {
        width,
        height,
        rgba,
    })
}

fn expand_to_rgba<F>(pixels: &[u8], pixel_count: usize, mut emit: F) -> Vec<u8>
where
    F: FnMut(&[u8]) -> [u8; 4],
{
    let stride = pixels.len() / pixel_count.max(1);
    let mut output = Vec::with_capacity(pixel_count * 4);
    for index in 0..pixel_count {
        let chunk = &pixels[index * stride..(index + 1) * stride];
        output.extend_from_slice(&emit(chunk));
    }
    output
}

fn narrow_16_to_rgba<F>(pixels: &[u8], pixel_count: usize, mut emit: F) -> Vec<u8>
where
    F: FnMut(&[u16; 4]) -> [u16; 4],
{
    let stride_bytes = pixels.len() / pixel_count.max(1);
    let stride_components = stride_bytes / 2;
    let mut output = Vec::with_capacity(pixel_count * 4);
    let mut chunk = [0u16; 4];
    for index in 0..pixel_count {
        let base = index * stride_bytes;
        for component in 0..stride_components.min(4) {
            let lo = pixels[base + component * 2] as u16;
            let hi = pixels[base + component * 2 + 1] as u16;
            chunk[component] = lo | (hi << 8);
        }
        let normalized = emit(&chunk);
        for value in normalized {
            output.push((value >> 8) as u8);
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_mode_default_is_opaque() {
        let material = GltfMaterial {
            base_color_factor: Vec4::ONE,
            base_color_texture: None,
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            metallic_roughness_texture: None,
            emissive_factor: Vec3::ZERO,
            emissive_strength: 1.0,
            emissive_texture: None,
            alpha_mode: GltfAlphaMode::Opaque,
        };
        assert!(matches!(material.alpha_mode, GltfAlphaMode::Opaque));
    }

    #[test]
    fn rgb_image_expands_to_rgba_with_full_alpha() {
        let pixels = vec![10, 20, 30, 40, 50, 60];
        let result = expand_to_rgba(&pixels, 2, |chunk| [chunk[0], chunk[1], chunk[2], 255]);
        assert_eq!(result, vec![10, 20, 30, 255, 40, 50, 60, 255]);
    }

    #[test]
    fn r_only_image_expands_to_grayscale_rgba() {
        let pixels = vec![64, 192];
        let result = expand_to_rgba(&pixels, 2, |chunk| [chunk[0], chunk[0], chunk[0], 255]);
        assert_eq!(result, vec![64, 64, 64, 255, 192, 192, 192, 255]);
    }
}
