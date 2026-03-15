use glam::Vec3;
use std::{fmt, path::Path};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    pub min: Vec3,
    pub max: Vec3,
}

impl Bounds {
    pub fn center(&self) -> Vec3 {
        0.5 * (self.min + self.max)
    }

    pub fn extent(&self) -> Vec3 {
        self.max - self.min
    }

    pub fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub bounds: Bounds,
}

impl Mesh {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        let bounds = compute_bounds(&vertices).expect("mesh must contain at least one vertex");

        Self {
            vertices,
            indices,
            bounds,
        }
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn triangle_positions(&self, triangle_index: usize) -> [Vec3; 3] {
        let [i0, i1, i2] = self.triangle_indices(triangle_index);

        [
            self.vertices[i0 as usize].position,
            self.vertices[i1 as usize].position,
            self.vertices[i2 as usize].position,
        ]
    }

    pub fn triangle_normals(&self, triangle_index: usize) -> [Vec3; 3] {
        let [i0, i1, i2] = self.triangle_indices(triangle_index);

        [
            self.vertices[i0 as usize].normal,
            self.vertices[i1 as usize].normal,
            self.vertices[i2 as usize].normal,
        ]
    }

    fn triangle_indices(&self, triangle_index: usize) -> [u32; 3] {
        let base = 3 * triangle_index;
        [
            self.indices[base],
            self.indices[base + 1],
            self.indices[base + 2],
        ]
    }
}

#[derive(Debug)]
pub enum LoadMeshError {
    Gltf(gltf::Error),
    EmptyMesh,
    MissingPositions {
        mesh_index: usize,
        primitive_index: usize,
    },
    VertexCountOverflow,
    InvalidTriangleIndexCount {
        mesh_index: usize,
        primitive_index: usize,
    },
    UnsupportedPrimitiveMode {
        mesh_index: usize,
        primitive_index: usize,
        mode: gltf::mesh::Mode,
    },
}

impl fmt::Display for LoadMeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gltf(error) => write!(f, "{error}"),
            Self::EmptyMesh => write!(f, "the glTF asset did not contain any triangle mesh data"),
            Self::MissingPositions {
                mesh_index,
                primitive_index,
            } => write!(
                f,
                "mesh {mesh_index} primitive {primitive_index} did not have POSITION data"
            ),
            Self::VertexCountOverflow => write!(f, "the mesh had more than u32::MAX vertices"),
            Self::InvalidTriangleIndexCount {
                mesh_index,
                primitive_index,
            } => write!(
                f,
                "mesh {mesh_index} primitive {primitive_index} did not contain indices in groups of three"
            ),
            Self::UnsupportedPrimitiveMode {
                mesh_index,
                primitive_index,
                mode,
            } => write!(
                f,
                "mesh {mesh_index} primitive {primitive_index} used unsupported mode {mode:?}"
            ),
        }
    }
}

impl std::error::Error for LoadMeshError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Gltf(error) => Some(error),
            _ => None,
        }
    }
}

impl From<gltf::Error> for LoadMeshError {
    fn from(error: gltf::Error) -> Self {
        Self::Gltf(error)
    }
}

pub fn load_mesh(path: &Path) -> Result<Mesh, LoadMeshError> {
    let (document, buffers, _) = gltf::import(path)?;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for mesh in document.meshes() {
        append_gltf_mesh(&buffers, mesh, &mut vertices, &mut indices)?;
    }

    if vertices.is_empty() || indices.is_empty() {
        return Err(LoadMeshError::EmptyMesh);
    }

    Ok(Mesh::new(vertices, indices))
}

fn append_gltf_mesh(
    buffers: &[gltf::buffer::Data],
    mesh: gltf::Mesh<'_>,
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
) -> Result<(), LoadMeshError> {
    for primitive in mesh.primitives() {
        if primitive.mode() != gltf::mesh::Mode::Triangles {
            return Err(LoadMeshError::UnsupportedPrimitiveMode {
                mesh_index: mesh.index(),
                primitive_index: primitive.index(),
                mode: primitive.mode(),
            });
        }

        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
        let positions = reader
            .read_positions()
            .ok_or_else(|| LoadMeshError::MissingPositions {
                mesh_index: mesh.index(),
                primitive_index: primitive.index(),
            })?
            .map(Vec3::from_array)
            .collect::<Vec<_>>();

        let local_indices = reader
            .read_indices()
            .map(|indices| indices.into_u32().collect::<Vec<_>>())
            .unwrap_or_else(|| (0..positions.len() as u32).collect());

        if local_indices.len() % 3 != 0 {
            return Err(LoadMeshError::InvalidTriangleIndexCount {
                mesh_index: mesh.index(),
                primitive_index: primitive.index(),
            });
        }

        let generated_normals = generate_vertex_normals(&positions, &local_indices);
        let normals = reader
            .read_normals()
            .map(|normals| {
                normals
                    .map(Vec3::from_array)
                    .map(|normal| normal.normalize_or_zero())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| generated_normals.clone());

        let base_vertex =
            u32::try_from(vertices.len()).map_err(|_| LoadMeshError::VertexCountOverflow)?;

        for (index, position) in positions.into_iter().enumerate() {
            let normal = normals[index];
            vertices.push(Vertex {
                position,
                normal: if normal.length_squared() > 0.0 {
                    normal
                } else {
                    generated_normals[index]
                },
            });
        }

        indices.extend(local_indices.into_iter().map(|index| base_vertex + index));
    }

    Ok(())
}

fn generate_vertex_normals(positions: &[Vec3], indices: &[u32]) -> Vec<Vec3> {
    let mut normals = vec![Vec3::ZERO; positions.len()];

    for triangle in indices.chunks_exact(3) {
        let i0 = triangle[0] as usize;
        let i1 = triangle[1] as usize;
        let i2 = triangle[2] as usize;

        let p0 = positions[i0];
        let p1 = positions[i1];
        let p2 = positions[i2];
        let face_normal = (p1 - p0).cross(p2 - p0);

        if face_normal.length_squared() == 0.0 {
            continue;
        }

        normals[i0] += face_normal;
        normals[i1] += face_normal;
        normals[i2] += face_normal;
    }

    normals
        .into_iter()
        .map(|normal| normal.normalize_or_zero())
        .collect()
}

fn compute_bounds(vertices: &[Vertex]) -> Option<Bounds> {
    let mut vertices = vertices.iter();
    let first = vertices.next()?;
    let mut min = first.position;
    let mut max = first.position;

    for vertex in vertices {
        min = min.min(vertex.position);
        max = max.max(vertex.position);
    }

    Some(Bounds { min, max })
}

#[cfg(test)]
mod tests {
    use super::{Bounds, Mesh, Vertex};
    use glam::Vec3;

    #[test]
    fn triangle_accessors_follow_index_buffer() {
        let mesh = Mesh::new(
            vec![
                Vertex {
                    position: Vec3::new(0.0, 0.0, 0.0),
                    normal: Vec3::X,
                },
                Vertex {
                    position: Vec3::new(1.0, 0.0, 0.0),
                    normal: Vec3::Y,
                },
                Vertex {
                    position: Vec3::new(0.0, 1.0, 0.0),
                    normal: Vec3::Z,
                },
            ],
            vec![0, 1, 2],
        );

        assert_eq!(
            mesh.triangle_positions(0),
            [
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ]
        );
        assert_eq!(mesh.triangle_normals(0), [Vec3::X, Vec3::Y, Vec3::Z]);
    }

    #[test]
    fn mesh_computes_bounds() {
        let mesh = Mesh::new(
            vec![
                Vertex {
                    position: Vec3::new(-2.0, 1.0, 3.0),
                    normal: Vec3::X,
                },
                Vertex {
                    position: Vec3::new(4.0, -1.0, -5.0),
                    normal: Vec3::Y,
                },
            ],
            vec![0, 1, 1],
        );

        assert_eq!(
            mesh.bounds,
            Bounds {
                min: Vec3::new(-2.0, -1.0, -5.0),
                max: Vec3::new(4.0, 1.0, 3.0),
            }
        );
    }
}
