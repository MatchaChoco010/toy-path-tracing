use std::fmt;

use super::library::MtlxLibrary;
use super::types::{MtlxType, RawInput, RawNodeUse};

#[derive(Debug)]
pub enum ResolveError {
    UnknownNode { category: String, context: String },
    UnknownNodeDef { name: String },
    MismatchedNodeDef { name: String, reason: String },
    UnknownInput { node: String, input: String },
}

impl fmt::Display for ResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownNode { category, context } => {
                write!(
                    f,
                    "no nodedef found for category `{}` in {}",
                    category, context
                )
            }
            Self::UnknownNodeDef { name } => {
                write!(f, "unknown nodedef `{}`", name)
            }
            Self::MismatchedNodeDef { name, reason } => {
                write!(f, "nodedef `{}` does not match node use: {}", name, reason)
            }
            Self::UnknownInput { node, input } => {
                write!(f, "unknown input `{}` on node `{}`", input, node)
            }
        }
    }
}

impl std::error::Error for ResolveError {}

pub fn resolve_node_use<'a>(
    lib: &'a MtlxLibrary,
    use_node: &RawNodeUse,
    input_types: &[(String, MtlxType)],
) -> Result<&'a super::library::LibraryNodeDef, ResolveError> {
    if let Some(name) = &use_node.nodedef {
        if let Some(def) = lib.nodedef_by_name(name) {
            if def.def.node != use_node.category {
                return Err(ResolveError::MismatchedNodeDef {
                    name: name.clone(),
                    reason: format!(
                        "declares node `{}`, but instance category is `{}`",
                        def.def.node, use_node.category
                    ),
                });
            }
            if !super::library::nodedef_matches(def, input_types, &use_node.ty) {
                return Err(ResolveError::MismatchedNodeDef {
                    name: name.clone(),
                    reason: format!(
                        "inputs or output type do not match category `{}` type `{}`",
                        use_node.category,
                        use_node.ty.as_str()
                    ),
                });
            }
            return Ok(def);
        }
        return Err(ResolveError::UnknownNodeDef { name: name.clone() });
    }
    let def = lib.find_matching(
        &use_node.category,
        input_types,
        &use_node.ty,
        use_node.version.as_deref(),
    );
    def.ok_or_else(|| ResolveError::UnknownNode {
        category: use_node.category.clone(),
        context: format!("type `{}`", use_node.ty.as_str()),
    })
}

pub fn collect_input_types(use_node: &RawNodeUse) -> Vec<(String, MtlxType)> {
    use_node
        .inputs
        .iter()
        .map(|i: &RawInput| (i.name.clone(), i.ty.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene_loader::mtlx_loader::library::load_standard_library;
    use crate::scene_loader::mtlx_loader::types::{InputBinding, RawInput, RawNodeDef, RawOutput};
    use std::path::PathBuf;

    fn lib_root() -> PathBuf {
        crate::paths::workspace_path("lib/materialx/libraries")
    }

    #[test]
    fn resolves_known_category() {
        let lib = load_standard_library(&lib_root()).expect("library");
        let use_node = RawNodeUse {
            name: "n".into(),
            category: "add".into(),
            ty: MtlxType::Color3,
            inputs: vec![],
            tokens: vec![],
            outputs: vec![],
            version: None,
            nodedef: None,
        };
        let inputs = vec![
            ("in1".into(), MtlxType::Color3),
            ("in2".into(), MtlxType::Color3),
        ];
        let def = resolve_node_use(&lib, &use_node, &inputs).expect("resolve");
        assert_eq!(def.def.node, "add");
    }

    #[test]
    fn explicit_nodedef_must_match_node_category_and_signature() {
        let mut lib = MtlxLibrary::new();
        lib.add_document(super::super::types::RawMtlxDocument {
            source_path: PathBuf::from("inline.mtlx"),
            nodedefs: vec![RawNodeDef {
                name: "ND_constant_float".to_string(),
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
                version: None,
                is_default_version: true,
                inherit: None,
                target: None,
                nodegroup: None,
                doc: None,
            }],
            ..Default::default()
        });

        let wrong_category = RawNodeUse {
            name: "n".into(),
            category: "add".into(),
            ty: MtlxType::Float,
            inputs: vec![],
            tokens: vec![],
            outputs: vec![],
            version: None,
            nodedef: Some("ND_constant_float".into()),
        };
        let err = resolve_node_use(&lib, &wrong_category, &[]).expect_err("category mismatch");
        assert!(err.to_string().contains("declares node `constant`"));

        let wrong_input = RawNodeUse {
            name: "n".into(),
            category: "constant".into(),
            ty: MtlxType::Float,
            inputs: vec![RawInput {
                name: "unknown".to_string(),
                ty: MtlxType::Float,
                binding: InputBinding::Empty,
                colorspace: None,
                unit: None,
                unittype: None,
                uniform: false,
            }],
            tokens: vec![],
            outputs: vec![],
            version: None,
            nodedef: Some("ND_constant_float".into()),
        };
        let inputs = collect_input_types(&wrong_input);
        let err = resolve_node_use(&lib, &wrong_input, &inputs).expect_err("signature mismatch");
        assert!(err.to_string().contains("inputs or output type"));
    }
}
