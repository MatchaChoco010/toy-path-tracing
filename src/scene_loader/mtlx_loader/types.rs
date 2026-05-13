use std::path::PathBuf;

use glam::{Mat3, Mat4, Vec2, Vec3, Vec4};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MtlxType {
    Boolean,
    Integer,
    Float,
    Color3,
    Color4,
    Vector2,
    Vector3,
    Vector4,
    Matrix33,
    Matrix44,
    String,
    Filename,
    Geomname,
    IntegerArray,
    FloatArray,
    Color3Array,
    Color4Array,
    Vector2Array,
    Vector3Array,
    Vector4Array,
    StringArray,
    GeomnameArray,
    Surfaceshader,
    Displacementshader,
    Volumeshader,
    Lightshader,
    Material,
    Bsdf,
    Edf,
    Vdf,
    None,
    Custom(String),
}

impl MtlxType {
    pub fn parse(s: &str) -> Self {
        match s {
            "boolean" => Self::Boolean,
            "integer" => Self::Integer,
            "float" => Self::Float,
            "color3" => Self::Color3,
            "color4" => Self::Color4,
            "vector2" => Self::Vector2,
            "vector3" => Self::Vector3,
            "vector4" => Self::Vector4,
            "matrix33" => Self::Matrix33,
            "matrix44" => Self::Matrix44,
            "string" => Self::String,
            "filename" => Self::Filename,
            "geomname" => Self::Geomname,
            "integerarray" => Self::IntegerArray,
            "floatarray" => Self::FloatArray,
            "color3array" => Self::Color3Array,
            "color4array" => Self::Color4Array,
            "vector2array" => Self::Vector2Array,
            "vector3array" => Self::Vector3Array,
            "vector4array" => Self::Vector4Array,
            "stringarray" => Self::StringArray,
            "geomnamearray" => Self::GeomnameArray,
            "surfaceshader" => Self::Surfaceshader,
            "displacementshader" => Self::Displacementshader,
            "volumeshader" => Self::Volumeshader,
            "lightshader" => Self::Lightshader,
            "material" => Self::Material,
            "BSDF" => Self::Bsdf,
            "EDF" => Self::Edf,
            "VDF" => Self::Vdf,
            "none" => Self::None,
            other => Self::Custom(other.to_string()),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Boolean => "boolean",
            Self::Integer => "integer",
            Self::Float => "float",
            Self::Color3 => "color3",
            Self::Color4 => "color4",
            Self::Vector2 => "vector2",
            Self::Vector3 => "vector3",
            Self::Vector4 => "vector4",
            Self::Matrix33 => "matrix33",
            Self::Matrix44 => "matrix44",
            Self::String => "string",
            Self::Filename => "filename",
            Self::Geomname => "geomname",
            Self::IntegerArray => "integerarray",
            Self::FloatArray => "floatarray",
            Self::Color3Array => "color3array",
            Self::Color4Array => "color4array",
            Self::Vector2Array => "vector2array",
            Self::Vector3Array => "vector3array",
            Self::Vector4Array => "vector4array",
            Self::StringArray => "stringarray",
            Self::GeomnameArray => "geomnamearray",
            Self::Surfaceshader => "surfaceshader",
            Self::Displacementshader => "displacementshader",
            Self::Volumeshader => "volumeshader",
            Self::Lightshader => "lightshader",
            Self::Material => "material",
            Self::Bsdf => "BSDF",
            Self::Edf => "EDF",
            Self::Vdf => "VDF",
            Self::None => "none",
            Self::Custom(name) => name.as_str(),
        }
    }

    pub fn is_shader_like(&self) -> bool {
        matches!(
            self,
            Self::Surfaceshader
                | Self::Displacementshader
                | Self::Volumeshader
                | Self::Lightshader
                | Self::Bsdf
                | Self::Edf
                | Self::Vdf
                | Self::Material
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MtlxValue {
    Boolean(bool),
    Integer(i32),
    Float(f32),
    Color3(Vec3),
    Color4(Vec4),
    Vector2(Vec2),
    Vector3(Vec3),
    Vector4(Vec4),
    Matrix33(Mat3),
    Matrix44(Mat4),
    String(String),
    Filename(String),
    IntegerArray(Vec<i32>),
    FloatArray(Vec<f32>),
    Color3Array(Vec<Vec3>),
    Color4Array(Vec<Vec4>),
    Vector2Array(Vec<Vec2>),
    Vector3Array(Vec<Vec3>),
    Vector4Array(Vec<Vec4>),
    StringArray(Vec<String>),
}

impl MtlxValue {
    pub fn ty(&self) -> MtlxType {
        match self {
            Self::Boolean(_) => MtlxType::Boolean,
            Self::Integer(_) => MtlxType::Integer,
            Self::Float(_) => MtlxType::Float,
            Self::Color3(_) => MtlxType::Color3,
            Self::Color4(_) => MtlxType::Color4,
            Self::Vector2(_) => MtlxType::Vector2,
            Self::Vector3(_) => MtlxType::Vector3,
            Self::Vector4(_) => MtlxType::Vector4,
            Self::Matrix33(_) => MtlxType::Matrix33,
            Self::Matrix44(_) => MtlxType::Matrix44,
            Self::String(_) => MtlxType::String,
            Self::Filename(_) => MtlxType::Filename,
            Self::IntegerArray(_) => MtlxType::IntegerArray,
            Self::FloatArray(_) => MtlxType::FloatArray,
            Self::Color3Array(_) => MtlxType::Color3Array,
            Self::Color4Array(_) => MtlxType::Color4Array,
            Self::Vector2Array(_) => MtlxType::Vector2Array,
            Self::Vector3Array(_) => MtlxType::Vector3Array,
            Self::Vector4Array(_) => MtlxType::Vector4Array,
            Self::StringArray(_) => MtlxType::StringArray,
        }
    }
}

pub fn parse_literal(ty: &MtlxType, raw: &str) -> Option<MtlxValue> {
    let trimmed = raw.trim();
    match ty {
        MtlxType::Boolean => match trimmed {
            "true" => Some(MtlxValue::Boolean(true)),
            "false" => Some(MtlxValue::Boolean(false)),
            _ => None,
        },
        MtlxType::Integer => trimmed.parse::<i32>().ok().map(MtlxValue::Integer),
        MtlxType::Float => trimmed.parse::<f32>().ok().map(MtlxValue::Float),
        MtlxType::Color3 | MtlxType::Vector3 => parse_floats(trimmed, 3).map(|v| {
            let p = [v[0], v[1], v[2]];
            if matches!(ty, MtlxType::Color3) {
                MtlxValue::Color3(Vec3::from_array(p))
            } else {
                MtlxValue::Vector3(Vec3::from_array(p))
            }
        }),
        MtlxType::Color4 | MtlxType::Vector4 => parse_floats(trimmed, 4).map(|v| {
            let p = [v[0], v[1], v[2], v[3]];
            if matches!(ty, MtlxType::Color4) {
                MtlxValue::Color4(Vec4::from_array(p))
            } else {
                MtlxValue::Vector4(Vec4::from_array(p))
            }
        }),
        MtlxType::Vector2 => {
            parse_floats(trimmed, 2).map(|v| MtlxValue::Vector2(Vec2::new(v[0], v[1])))
        }
        MtlxType::Matrix33 => parse_floats(trimmed, 9).map(|v| {
            // MaterialX spec stores matrix values in row-major order; glam
            // stores them column-major.
            MtlxValue::Matrix33(Mat3::from_cols(
                Vec3::new(v[0], v[3], v[6]),
                Vec3::new(v[1], v[4], v[7]),
                Vec3::new(v[2], v[5], v[8]),
            ))
        }),
        MtlxType::Matrix44 => parse_floats(trimmed, 16).map(|v| {
            MtlxValue::Matrix44(Mat4::from_cols(
                Vec4::new(v[0], v[4], v[8], v[12]),
                Vec4::new(v[1], v[5], v[9], v[13]),
                Vec4::new(v[2], v[6], v[10], v[14]),
                Vec4::new(v[3], v[7], v[11], v[15]),
            ))
        }),
        MtlxType::String => Some(MtlxValue::String(trimmed.to_string())),
        MtlxType::Filename => Some(MtlxValue::Filename(trimmed.to_string())),
        MtlxType::Geomname => Some(MtlxValue::String(trimmed.to_string())),
        MtlxType::IntegerArray => parse_int_array(trimmed).map(MtlxValue::IntegerArray),
        MtlxType::FloatArray => parse_float_array(trimmed).map(MtlxValue::FloatArray),
        MtlxType::Color3Array | MtlxType::Vector3Array => parse_vec3_array(trimmed).map(|arr| {
            if matches!(ty, MtlxType::Color3Array) {
                MtlxValue::Color3Array(arr)
            } else {
                MtlxValue::Vector3Array(arr)
            }
        }),
        MtlxType::Color4Array | MtlxType::Vector4Array => parse_vec4_array(trimmed).map(|arr| {
            if matches!(ty, MtlxType::Color4Array) {
                MtlxValue::Color4Array(arr)
            } else {
                MtlxValue::Vector4Array(arr)
            }
        }),
        MtlxType::Vector2Array => parse_vec2_array(trimmed).map(MtlxValue::Vector2Array),
        MtlxType::StringArray | MtlxType::GeomnameArray => {
            Some(MtlxValue::StringArray(parse_string_array(trimmed)))
        }
        _ => None,
    }
}

fn parse_floats(s: &str, expected: usize) -> Option<Vec<f32>> {
    let parts: Vec<&str> = s
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .collect();
    if parts.len() != expected {
        return None;
    }
    parts.into_iter().map(|p| p.parse::<f32>().ok()).collect()
}

fn parse_int_array(s: &str) -> Option<Vec<i32>> {
    s.split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(|p| p.parse::<i32>().ok())
        .collect()
}

fn parse_float_array(s: &str) -> Option<Vec<f32>> {
    s.split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(|p| p.parse::<f32>().ok())
        .collect()
}

fn parse_vec2_array(s: &str) -> Option<Vec<Vec2>> {
    let nums = parse_float_array(s)?;
    if !nums.len().is_multiple_of(2) {
        return None;
    }
    Some(
        nums.chunks_exact(2)
            .map(|c| Vec2::new(c[0], c[1]))
            .collect(),
    )
}

fn parse_vec3_array(s: &str) -> Option<Vec<Vec3>> {
    let nums = parse_float_array(s)?;
    if !nums.len().is_multiple_of(3) {
        return None;
    }
    Some(
        nums.chunks_exact(3)
            .map(|c| Vec3::new(c[0], c[1], c[2]))
            .collect(),
    )
}

fn parse_vec4_array(s: &str) -> Option<Vec<Vec4>> {
    let nums = parse_float_array(s)?;
    if !nums.len().is_multiple_of(4) {
        return None;
    }
    Some(
        nums.chunks_exact(4)
            .map(|c| Vec4::new(c[0], c[1], c[2], c[3]))
            .collect(),
    )
}

fn parse_string_array(s: &str) -> Vec<String> {
    if s.is_empty() {
        return Vec::new();
    }
    let mut values = Vec::new();
    let mut current = String::new();
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.peek().copied() {
                Some(',') | Some(';') | Some('\\') => {
                    current.push(chars.next().unwrap());
                }
                Some(next) => {
                    current.push(ch);
                    current.push(next);
                    chars.next();
                }
                None => current.push(ch),
            }
        } else if ch == ',' || ch == ';' {
            values.push(current.trim().to_string());
            current.clear();
        } else {
            current.push(ch);
        }
    }
    values.push(current.trim().to_string());
    values
}

#[derive(Debug, Clone, PartialEq)]
pub enum InputBinding {
    Empty,
    Value(String),
    NodeRef {
        nodename: String,
        output: Option<String>,
    },
    NodeGraphRef {
        nodegraph: String,
        output: Option<String>,
    },
    InterfaceName(String),
    DefaultGeomProp(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct RawInput {
    pub name: String,
    pub ty: MtlxType,
    pub binding: InputBinding,
    pub colorspace: Option<String>,
    pub unit: Option<String>,
    pub unittype: Option<String>,
    pub uniform: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RawOutput {
    pub name: String,
    pub ty: MtlxType,
    pub binding: InputBinding,
    pub default: Option<String>,
    pub default_input: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RawNodeDef {
    pub name: String,
    pub node: String,
    pub inputs: Vec<RawInput>,
    pub tokens: Vec<RawToken>,
    pub outputs: Vec<RawOutput>,
    pub version: Option<String>,
    pub is_default_version: bool,
    pub inherit: Option<String>,
    pub target: Option<String>,
    pub nodegroup: Option<String>,
    pub doc: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RawImplementation {
    pub name: String,
    pub nodedef: String,
    pub nodegraph: Option<String>,
    pub function: Option<String>,
    pub file: Option<String>,
    pub target: Option<String>,
    pub format: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RawTypeDef {
    pub name: String,
    pub semantic: Option<String>,
    pub context: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RawGeomPropDef {
    pub name: String,
    pub ty: MtlxType,
    pub uniform: bool,
    pub geomprop: Option<String>,
    pub space: Option<String>,
    pub index: Option<i32>,
    pub unittype: Option<String>,
    pub unit: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RawNodeUse {
    pub name: String,
    pub category: String,
    pub ty: MtlxType,
    pub inputs: Vec<RawInput>,
    pub tokens: Vec<RawToken>,
    pub outputs: Vec<RawOutput>,
    pub version: Option<String>,
    pub nodedef: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RawNodeGraph {
    pub name: String,
    pub nodedef: Option<String>,
    pub target: Option<String>,
    pub nodes: Vec<RawNodeUse>,
    pub inputs: Vec<RawInput>,
    pub outputs: Vec<RawOutput>,
    /// `<token>` elements declared inside a nodegraph/nodedef interface.
    /// Used for `[interface_token]` filename substitution.
    pub tokens: Vec<RawToken>,
}

#[derive(Debug, Clone)]
pub struct RawToken {
    pub name: String,
    pub ty: MtlxType,
    pub value: Option<String>,
    pub interface: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RawMaterial {
    pub name: String,
    pub category: String,
    pub inputs: Vec<RawInput>,
    pub parent_nodegraph: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct RawMtlxDocument {
    pub version: (u32, u32),
    pub colorspace: Option<String>,
    pub namespace: Option<String>,
    pub nodedefs: Vec<RawNodeDef>,
    pub nodegraphs: Vec<RawNodeGraph>,
    pub implementations: Vec<RawImplementation>,
    pub typedefs: Vec<RawTypeDef>,
    pub geompropdefs: Vec<RawGeomPropDef>,
    pub materials: Vec<RawMaterial>,
    pub root_nodes: Vec<RawNodeUse>,
    pub source_path: PathBuf,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_color3_literal() {
        let v = parse_literal(&MtlxType::Color3, "0.1, 0.2, 0.3").unwrap();
        assert_eq!(v, MtlxValue::Color3(Vec3::new(0.1, 0.2, 0.3)));
    }

    #[test]
    fn parse_float_literal_handles_negative() {
        let v = parse_literal(&MtlxType::Float, "-1.5").unwrap();
        assert_eq!(v, MtlxValue::Float(-1.5));
    }

    #[test]
    fn parse_boolean_literal() {
        assert_eq!(
            parse_literal(&MtlxType::Boolean, "true"),
            Some(MtlxValue::Boolean(true))
        );
        assert_eq!(
            parse_literal(&MtlxType::Boolean, "false"),
            Some(MtlxValue::Boolean(false))
        );
        assert_eq!(parse_literal(&MtlxType::Boolean, "1"), None);
        assert_eq!(parse_literal(&MtlxType::Boolean, "0"), None);
    }

    #[test]
    fn integer_array_rejects_malformed_element() {
        assert_eq!(parse_literal(&MtlxType::IntegerArray, "1, nope, 3"), None);
    }

    #[test]
    fn float_array_rejects_malformed_element() {
        assert_eq!(parse_literal(&MtlxType::FloatArray, "1.0 bad 3.0"), None);
    }

    #[test]
    fn vector_array_rejects_malformed_element() {
        assert_eq!(
            parse_literal(&MtlxType::Vector2Array, "1.0, 2.0, bad, 4.0"),
            None
        );
    }

    #[test]
    fn string_array_supports_materialx_escape_convention() {
        assert_eq!(
            parse_literal(&MtlxType::StringArray, r"hello\,there, a\\b, c\;d"),
            Some(MtlxValue::StringArray(vec![
                "hello,there".to_string(),
                r"a\b".to_string(),
                "c;d".to_string(),
            ]))
        );
        assert_eq!(
            parse_literal(&MtlxType::StringArray, ""),
            Some(MtlxValue::StringArray(Vec::new()))
        );
    }

    #[test]
    fn type_round_trip() {
        let cases = [
            "float",
            "color3",
            "color4",
            "vector2",
            "vector3",
            "matrix33",
            "matrix44",
            "BSDF",
            "EDF",
            "surfaceshader",
            "material",
        ];
        for c in cases {
            let parsed = MtlxType::parse(c);
            assert_eq!(parsed.as_str(), c);
        }
    }

    #[test]
    fn matrix33_parsed_row_major_preserves_element_positions() {
        // MaterialX spec: matrix values are listed in row-major order:
        // "m11,m12,m13, m21,m22,m23, m31,m32,m33". After parsing, accessing
        // `m.row(i).col(j)` (which in glam is `m.col_at(j).at(i)`) must give
        // the i-th row j-th column element.
        let v = parse_literal(&MtlxType::Matrix33, "1,2,3, 4,5,6, 7,8,9").unwrap();
        if let MtlxValue::Matrix33(m) = v {
            assert_eq!(m.x_axis, Vec3::new(1.0, 4.0, 7.0));
            assert_eq!(m.y_axis, Vec3::new(2.0, 5.0, 8.0));
            assert_eq!(m.z_axis, Vec3::new(3.0, 6.0, 9.0));
            // Verify `m * v` matches row-major semantics: (Mv).x = m11*v.x + m12*v.y + m13*v.z
            let r = m.mul_vec3(Vec3::new(1.0, 0.0, 0.0));
            assert_eq!(r, Vec3::new(1.0, 4.0, 7.0));
        } else {
            panic!("expected Matrix33");
        }
    }

    #[test]
    fn unknown_type_becomes_custom() {
        match MtlxType::parse("mycolor") {
            MtlxType::Custom(s) => assert_eq!(s, "mycolor"),
            _ => panic!("expected Custom"),
        }
    }
}
