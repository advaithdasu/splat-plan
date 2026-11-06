//! The planner never grades itself.
//!
//! `gs-verify` may depend on `splat-types` and on external math/serde crates,
//! but never (directly or transitively) on any other crate in this workspace,
//! in particular not on `planner` or `gs-map`. This test asks
//! `cargo metadata` for the resolved dependency graph and fails if that rule
//! is ever broken.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::{BTreeMap, BTreeSet};
use std::process::Command;

/// Workspace crates `gs-verify` is allowed to reach.
const ALLOWED_WORKSPACE_DEPS: &[&str] = &["splat-types"];

#[test]
fn gs_verify_depends_only_on_splat_types_within_workspace() {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let out = Command::new(cargo)
        .args(["metadata", "--format-version", "1"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("failed to run cargo metadata");
    assert!(out.status.success(), "cargo metadata failed");
    let meta: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();

    // Package id -> name, and the set of workspace member ids.
    let mut names: BTreeMap<String, String> = BTreeMap::new();
    for p in meta["packages"].as_array().unwrap() {
        names.insert(
            p["id"].as_str().unwrap().to_string(),
            p["name"].as_str().unwrap().to_string(),
        );
    }
    let members: BTreeSet<String> = meta["workspace_members"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();

    // Adjacency over *normal* (non-dev, non-build) dependencies.
    let mut adj: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for node in meta["resolve"]["nodes"].as_array().unwrap() {
        let id = node["id"].as_str().unwrap().to_string();
        let mut deps = Vec::new();
        for d in node["deps"].as_array().unwrap() {
            let normal = d["dep_kinds"]
                .as_array()
                .unwrap()
                .iter()
                .any(|k| k["kind"].is_null());
            if normal {
                deps.push(d["pkg"].as_str().unwrap().to_string());
            }
        }
        adj.insert(id, deps);
    }

    let root = names
        .iter()
        .find(|(id, n)| n.as_str() == "gs-verify" && members.contains(*id))
        .map(|(id, _)| id.clone())
        .expect("gs-verify not found in workspace");

    // Depth-first walk of everything gs-verify can reach.
    let mut seen = BTreeSet::new();
    let mut stack = vec![root.clone()];
    while let Some(id) = stack.pop() {
        if !seen.insert(id.clone()) {
            continue;
        }
        if let Some(ds) = adj.get(&id) {
            stack.extend(ds.iter().cloned());
        }
    }

    let forbidden: Vec<&String> = seen
        .iter()
        .filter(|id| **id != root && members.contains(*id))
        .map(|id| &names[id])
        .filter(|n| !ALLOWED_WORKSPACE_DEPS.contains(&n.as_str()))
        .collect();
    assert!(
        forbidden.is_empty(),
        "independence violated: gs-verify reaches workspace crates {forbidden:?}; only {ALLOWED_WORKSPACE_DEPS:?} allowed"
    );
}
