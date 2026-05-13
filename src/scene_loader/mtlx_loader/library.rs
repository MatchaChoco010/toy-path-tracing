use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::parser::{ParseError, parse_document};
use super::types::{
    MtlxType, RawGeomPropDef, RawImplementation, RawMaterial, RawMtlxDocument, RawNodeDef,
    RawNodeGraph, RawTypeDef,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeSignature {
    pub inputs: Vec<MtlxType>,
    pub output: MtlxType,
}

#[derive(Debug, Clone)]
pub struct LibraryNodeDef {
    pub def: RawNodeDef,
    pub source: PathBuf,
}

#[derive(Debug, Clone)]
pub struct LibraryNodeGraph {
    pub graph: RawNodeGraph,
    pub source: PathBuf,
}

#[derive(Debug, Clone)]
pub struct LibraryMaterial {
    pub material: RawMaterial,
    pub source: PathBuf,
}

#[derive(Debug, Default)]
pub struct MtlxLibrary {
    pub typedefs: Vec<RawTypeDef>,
    pub geompropdefs: Vec<RawGeomPropDef>,
    pub nodedefs: Vec<LibraryNodeDef>,
    pub nodegraphs: Vec<LibraryNodeGraph>,
    pub implementations: Vec<RawImplementation>,
    pub materials: Vec<LibraryMaterial>,
    by_qualified_name: HashMap<String, usize>,
    by_category: HashMap<String, Vec<usize>>,
    nodegraph_by_name: HashMap<String, usize>,
    nodegraph_by_nodedef: HashMap<String, Vec<usize>>,
    impl_by_nodedef: HashMap<String, Vec<usize>>,
}

impl MtlxLibrary {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_document(&mut self, doc: RawMtlxDocument) {
        let source = doc.source_path.clone();
        for td in doc.typedefs {
            self.typedefs.push(td);
        }
        for gpd in doc.geompropdefs {
            self.geompropdefs.push(gpd);
        }
        for nd in doc.nodedefs {
            let idx = self.nodedefs.len();
            self.by_qualified_name.insert(nd.name.clone(), idx);
            self.by_category
                .entry(nd.node.clone())
                .or_default()
                .push(idx);
            self.nodedefs.push(LibraryNodeDef {
                def: nd,
                source: source.clone(),
            });
        }
        for ng in doc.nodegraphs {
            let idx = self.nodegraphs.len();
            self.nodegraph_by_name.insert(ng.name.clone(), idx);
            if let Some(nd) = &ng.nodedef {
                self.nodegraph_by_nodedef
                    .entry(nd.clone())
                    .or_default()
                    .push(idx);
            }
            self.nodegraphs.push(LibraryNodeGraph {
                graph: ng,
                source: source.clone(),
            });
        }
        for im in doc.implementations {
            let idx = self.implementations.len();
            self.impl_by_nodedef
                .entry(im.nodedef.clone())
                .or_default()
                .push(idx);
            self.implementations.push(im);
        }
        for mat in doc.materials {
            self.materials.push(LibraryMaterial {
                material: mat,
                source: source.clone(),
            });
        }
    }

    pub fn nodedef_by_name(&self, name: &str) -> Option<&LibraryNodeDef> {
        self.by_qualified_name
            .get(name)
            .and_then(|i| self.nodedefs.get(*i))
            .filter(|d| d.def.target.is_none())
    }

    pub fn resolve_inheritance(&mut self) -> Result<(), ParseError> {
        let by_name = self.by_qualified_name.clone();
        let snapshot: Vec<RawNodeDef> = self.nodedefs.iter().map(|d| d.def.clone()).collect();
        for i in 0..self.nodedefs.len() {
            let mut chain: Vec<String> = Vec::new();
            let mut cur_inherit = self.nodedefs[i].def.inherit.clone();
            while let Some(parent_name) = cur_inherit {
                if chain.contains(&parent_name) {
                    return Err(ParseError::Structure {
                        message: format!(
                            "nodedef `{}` has cyclic inheritance through `{}`",
                            self.nodedefs[i].def.name, parent_name
                        ),
                        path: self.nodedefs[i].source.clone(),
                    });
                }
                chain.push(parent_name.clone());
                if let Some(parent_idx) = by_name.get(&parent_name).copied() {
                    let parent_def = &snapshot[parent_idx];
                    for parent_input in &parent_def.inputs {
                        if !self.nodedefs[i]
                            .def
                            .inputs
                            .iter()
                            .any(|inp| inp.name == parent_input.name)
                        {
                            self.nodedefs[i].def.inputs.push(parent_input.clone());
                        }
                    }
                    for parent_output in &parent_def.outputs {
                        if !self.nodedefs[i]
                            .def
                            .outputs
                            .iter()
                            .any(|o| o.name == parent_output.name)
                        {
                            self.nodedefs[i].def.outputs.push(parent_output.clone());
                        }
                    }
                    for parent_token in &parent_def.tokens {
                        if !self.nodedefs[i]
                            .def
                            .tokens
                            .iter()
                            .any(|t| t.name == parent_token.name)
                        {
                            self.nodedefs[i].def.tokens.push(parent_token.clone());
                        }
                    }
                    cur_inherit = parent_def.inherit.clone();
                } else {
                    return Err(ParseError::Structure {
                        message: format!(
                            "nodedef `{}` inherits missing nodedef `{}`",
                            self.nodedefs[i].def.name, parent_name
                        ),
                        path: self.nodedefs[i].source.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    pub fn nodedefs_for_category(&self, category: &str) -> &[usize] {
        self.by_category
            .get(category)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub fn find_matching(
        &self,
        category: &str,
        inputs: &[(String, MtlxType)],
        output: &MtlxType,
        version: Option<&str>,
    ) -> Option<&LibraryNodeDef> {
        let candidates = self.nodedefs_for_category(category);
        if candidates.is_empty() {
            return None;
        }
        let mut filtered: Vec<&LibraryNodeDef> = candidates
            .iter()
            .map(|i| &self.nodedefs[*i])
            .filter(|d| d.def.target.is_none())
            .filter(|d| nodedef_matches(d, inputs, output))
            .collect();
        if filtered.is_empty() {
            return None;
        }
        if let Some(v) = version
            && let Some(d) = filtered
                .iter()
                .find(|d| version_matches(d.def.version.as_deref(), v))
        {
            return Some(*d);
        }
        if version.is_some() {
            return None;
        }
        if let Some(d) = filtered.iter().find(|d| d.def.is_default_version) {
            return Some(*d);
        }
        filtered.sort_by(|a, b| a.def.name.cmp(&b.def.name));
        Some(filtered[0])
    }

    pub fn nodegraph_by_name(&self, name: &str) -> Option<&LibraryNodeGraph> {
        self.nodegraph_by_name
            .get(name)
            .and_then(|i| self.nodegraphs.get(*i))
    }

    pub fn nodegraph_for_nodedef(&self, nodedef_name: &str) -> Option<&LibraryNodeGraph> {
        if let Some(indices) = self.impl_by_nodedef.get(nodedef_name) {
            for idx in indices {
                let im = &self.implementations[*idx];
                if im.target.is_some() {
                    continue;
                }
                if let Some(ng_name) = &im.nodegraph
                    && let Some(ng) = self.nodegraph_by_name(ng_name)
                    && ng.graph.target.is_none()
                {
                    return Some(ng);
                }
            }
        }
        if let Some(indices) = self.nodegraph_by_nodedef.get(nodedef_name) {
            for i in indices {
                if let Some(ng) = self.nodegraphs.get(*i)
                    && ng.graph.target.is_none()
                {
                    return Some(ng);
                }
            }
        }
        None
    }
}

pub(crate) fn nodedef_matches(
    def: &LibraryNodeDef,
    inputs: &[(String, MtlxType)],
    output: &MtlxType,
) -> bool {
    if def.def.outputs.len() == 1 && !output_type_compatible(&def.def.outputs[0].ty, output) {
        return false;
    }
    if def.def.outputs.len() > 1
        && *output != MtlxType::None
        && !def
            .def
            .outputs
            .iter()
            .any(|o| output_type_compatible(&o.ty, output))
    {
        return false;
    }
    for (name, ty) in inputs {
        if name == "disable" {
            continue;
        }
        let Some(decl) = def.def.inputs.iter().find(|i| &i.name == name) else {
            return false;
        };
        if decl.ty != *ty && *ty != MtlxType::None {
            return false;
        }
    }
    true
}

fn output_type_compatible(output: &MtlxType, expected: &MtlxType) -> bool {
    output == expected
        || *output == MtlxType::None
        || *expected == MtlxType::None
        || (*output == MtlxType::String && *expected == MtlxType::Filename)
}

fn version_matches(candidate: Option<&str>, requested: &str) -> bool {
    let Some(candidate) = candidate else {
        return false;
    };
    match (parse_version(candidate), parse_version(requested)) {
        (Some(a), Some(b)) => a == b,
        _ => candidate == requested,
    }
}

fn parse_version(s: &str) -> Option<(u32, u32)> {
    let mut parts = s.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().map(|p| p.parse().ok()).unwrap_or(Some(0))?;
    if parts.next().is_some() {
        return None;
    }
    Some((major, minor))
}

pub fn load_standard_library(root: &Path) -> Result<MtlxLibrary, ParseError> {
    let mut lib = MtlxLibrary::new();
    let order = [
        "stdlib/stdlib_defs.mtlx",
        "stdlib/stdlib_ng.mtlx",
        "pbrlib/pbrlib_defs.mtlx",
        "pbrlib/pbrlib_ng.mtlx",
        "bxdf/standard_surface.mtlx",
        "bxdf/disney_principled.mtlx",
        "bxdf/open_pbr_surface.mtlx",
        "bxdf/usd_preview_surface.mtlx",
        "bxdf/gltf_pbr.mtlx",
        "nprlib/nprlib_defs.mtlx",
        "nprlib/nprlib_ng.mtlx",
    ];
    for rel in order {
        let path = root.join(rel);
        if !path.exists() {
            return Err(ParseError::Io(
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!(
                        "required MaterialX library file `{}` is missing",
                        path.display()
                    ),
                ),
                path,
            ));
        }
        let doc = parse_document(&path)?;
        lib.add_document(doc);
    }
    lib.resolve_inheritance()?;
    Ok(lib)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene_loader::mtlx_loader::types::{InputBinding, RawInput, RawOutput};
    use std::path::PathBuf;

    fn lib_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("lib/materialx/libraries")
    }

    #[test]
    fn loads_standard_library_without_error() {
        let lib = load_standard_library(&lib_root()).expect("library");
        assert!(lib.nodedefs.len() > 100, "expected many nodedefs");
        assert!(!lib.nodegraphs.is_empty(), "expected nodegraphs");
        assert!(lib.nodedef_by_name("ND_oren_nayar_diffuse_bsdf").is_some());
        assert!(
            lib.nodedef_by_name("ND_standard_surface_surfaceshader")
                .is_some()
        );
    }

    #[test]
    fn matches_nodedef_for_simple_category() {
        let lib = load_standard_library(&lib_root()).expect("library");
        let inputs = vec![
            ("in1".to_string(), MtlxType::Color3),
            ("in2".to_string(), MtlxType::Color3),
        ];
        let def = lib.find_matching("add", &inputs, &MtlxType::Color3, None);
        assert!(def.is_some());
    }

    #[test]
    fn missing_inherited_nodedef_is_an_error() {
        let mut lib = MtlxLibrary::new();
        lib.add_document(RawMtlxDocument {
            source_path: PathBuf::from("inline.mtlx"),
            nodedefs: vec![RawNodeDef {
                name: "ND_child".to_string(),
                node: "constant".to_string(),
                inputs: Vec::new(),
                tokens: Vec::new(),
                outputs: Vec::new(),
                version: None,
                is_default_version: true,
                inherit: Some("ND_missing".to_string()),
                target: None,
                nodegroup: None,
                doc: None,
            }],
            ..Default::default()
        });

        let err = lib
            .resolve_inheritance()
            .expect_err("missing parent must error");
        assert!(err.to_string().contains("inherits missing nodedef"));
    }

    #[test]
    fn requested_version_does_not_fall_back_to_default() {
        let mut lib = MtlxLibrary::new();
        lib.add_document(RawMtlxDocument {
            source_path: PathBuf::from("inline.mtlx"),
            nodedefs: vec![test_nodedef("ND_const_v1", Some("1.0"), true)],
            ..Default::default()
        });

        assert!(
            lib.find_matching("constant", &[], &MtlxType::Float, Some("2.0"))
                .is_none()
        );
    }

    #[test]
    fn requested_version_treats_missing_minor_as_zero() {
        let mut lib = MtlxLibrary::new();
        lib.add_document(RawMtlxDocument {
            source_path: PathBuf::from("inline.mtlx"),
            nodedefs: vec![test_nodedef("ND_const_v1", Some("1.0"), true)],
            ..Default::default()
        });

        let def = lib
            .find_matching("constant", &[], &MtlxType::Float, Some("1"))
            .expect("minor-zero version should match");
        assert_eq!(def.def.name, "ND_const_v1");
    }

    #[test]
    fn unknown_input_does_not_match_nodedef() {
        let mut lib = MtlxLibrary::new();
        lib.add_document(RawMtlxDocument {
            source_path: PathBuf::from("inline.mtlx"),
            nodedefs: vec![test_nodedef("ND_const_v1", Some("1.0"), true)],
            ..Default::default()
        });

        let inputs = vec![("unknown".to_string(), MtlxType::Float)];
        assert!(
            lib.find_matching("constant", &inputs, &MtlxType::Float, None)
                .is_none()
        );
    }

    #[test]
    fn target_specific_nodedef_is_not_universal_match() {
        let mut target_def = test_nodedef("ND_const_genmdl", None, true);
        target_def.target = Some("genmdl".to_string());
        let mut lib = MtlxLibrary::new();
        lib.add_document(RawMtlxDocument {
            source_path: PathBuf::from("inline.mtlx"),
            nodedefs: vec![target_def],
            ..Default::default()
        });

        assert!(lib.nodedef_by_name("ND_const_genmdl").is_none());
        assert!(
            lib.find_matching("constant", &[], &MtlxType::Float, None)
                .is_none()
        );
    }

    #[test]
    fn nodegraph_for_nodedef_ignores_target_specific_graphs() {
        let mut lib = MtlxLibrary::new();
        lib.add_document(RawMtlxDocument {
            source_path: PathBuf::from("inline.mtlx"),
            nodedefs: vec![test_nodedef("ND_const_v1", None, true)],
            nodegraphs: vec![RawNodeGraph {
                name: "NG_const_genmdl".to_string(),
                nodedef: Some("ND_const_v1".to_string()),
                target: Some("genmdl".to_string()),
                nodes: Vec::new(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                tokens: Vec::new(),
            }],
            ..Default::default()
        });

        assert!(lib.nodegraph_for_nodedef("ND_const_v1").is_none());
    }

    #[test]
    fn nodegraph_implementation_precedes_direct_universal_graph() {
        let mut lib = MtlxLibrary::new();
        lib.add_document(RawMtlxDocument {
            source_path: PathBuf::from("inline.mtlx"),
            nodedefs: vec![test_nodedef("ND_const_v1", None, true)],
            nodegraphs: vec![
                RawNodeGraph {
                    name: "NG_direct".to_string(),
                    nodedef: Some("ND_const_v1".to_string()),
                    target: None,
                    nodes: Vec::new(),
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                    tokens: Vec::new(),
                },
                RawNodeGraph {
                    name: "NG_impl".to_string(),
                    nodedef: None,
                    target: None,
                    nodes: Vec::new(),
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                    tokens: Vec::new(),
                },
            ],
            implementations: vec![RawImplementation {
                name: "IM_const".to_string(),
                nodedef: "ND_const_v1".to_string(),
                nodegraph: Some("NG_impl".to_string()),
                function: None,
                file: None,
                target: None,
                format: None,
            }],
            ..Default::default()
        });

        let graph = lib.nodegraph_for_nodedef("ND_const_v1").expect("nodegraph");
        assert_eq!(graph.graph.name, "NG_impl");
    }

    fn test_nodedef(name: &str, version: Option<&str>, is_default_version: bool) -> RawNodeDef {
        RawNodeDef {
            name: name.to_string(),
            node: "constant".to_string(),
            inputs: vec![RawInput {
                name: "value".to_string(),
                ty: MtlxType::Float,
                binding: InputBinding::Empty,
                colorspace: None,
                unit: None,
                unittype: None,
                uniform: false,
            }],
            tokens: Vec::new(),
            outputs: vec![RawOutput {
                name: "out".to_string(),
                ty: MtlxType::Float,
                binding: InputBinding::Empty,
                default: None,
                default_input: None,
            }],
            version: version.map(str::to_string),
            is_default_version,
            inherit: None,
            target: None,
            nodegroup: None,
            doc: None,
        }
    }
}
