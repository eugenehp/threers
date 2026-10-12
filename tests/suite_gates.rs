//! That every test suite in this crate can actually run.
//!
//! A test file gated with a file-level `#![cfg(…)]` that the build does not
//! satisfy compiles to an **empty test binary**, and an empty test binary
//! reports `ok. 0 passed`. Nothing in that output distinguishes "this suite
//! ran and was happy" from "this suite did not exist", so a CI job reads both
//! as green. On this crate, `cargo test --features "full,fluid,webgl"` prints
//! exactly that for `tests/webgl.rs` and `tests/fluid_webgl.rs`, which are
//! wasm-only.
//!
//! Two failure modes follow, and this suite covers both.
//!
//! **A gate the manifest does not satisfy.** If a file needs a feature that
//! its `[[test]]` entry does not require, the suite silently disappears from
//! every run — including the ones meant to cover it. That is a mistake rather
//! than a trade-off, and [`every_feature_gate_is_declared_in_the_manifest`]
//! fails on it.
//!
//! **A gate nothing can satisfy here.** A wasm-only or macOS-only suite is
//! legitimately inert on another machine; what is wrong is only that it says
//! nothing. [`report_which_suites_are_inert_here`] names them and says how to
//! run each, so a build log carries the list rather than five silent zeroes.
//!
//! This file has no gate of its own, deliberately: it is the one suite that
//! must run everywhere.

use std::collections::{BTreeMap, BTreeSet};

/// `feature = "x"` implies these, per `Cargo.toml`. A suite that declares the
/// outer one has the inner ones too, so a gate on `captions` is satisfied by a
/// `required-features = ["native-codec"]`.
const IMPLIES: &[(&str, &[&str])] = &[
    ("native-codec", &["captions"]),
    ("video", &["captions"]),
    ("videotoolbox", &["native-codec", "captions"]),
    ("bvh-csg", &["mesh-bvh"]),
    ("openscad", &["bvh-csg", "mesh-bvh"]),
    ("manifold", &["openscad", "bvh-csg", "mesh-bvh"]),
    ("brep", &["nurbs"]),
    (
        "brep-csg",
        &["brep", "nurbs", "openscad", "bvh-csg", "mesh-bvh"],
    ),
    (
        "brep-kernel",
        &[
            "brep-csg", "brep", "nurbs", "openscad", "bvh-csg", "mesh-bvh",
        ],
    ),
    (
        "step",
        &[
            "brep-kernel",
            "brep-csg",
            "brep",
            "nurbs",
            "openscad",
            "bvh-csg",
            "mesh-bvh",
        ],
    ),
    ("visionos", &["metal"]),
    ("learned-denoise", &["raytrace"]),
];

/// Features that are on unless `--no-default-features` is passed, so a suite
/// gated on one needs no `required-features` entry to run by default.
const DEFAULT_FEATURES: &[&str] = &["captions"];

fn closure(declared: &BTreeSet<String>) -> BTreeSet<String> {
    let mut out = declared.clone();
    // One pass is enough: `IMPLIES` is written transitively closed already,
    // which is checked by `the_implication_table_is_closed`.
    for (outer, inner) in IMPLIES {
        if out.contains(*outer) {
            out.extend(inner.iter().map(|s| s.to_string()));
        }
    }
    out.extend(DEFAULT_FEATURES.iter().map(|s| s.to_string()));
    out
}

/// `[[test]] name = … required-features = […]` from the manifest.
fn declared_targets() -> BTreeMap<String, BTreeSet<String>> {
    let manifest = include_str!("../Cargo.toml");
    let mut out = BTreeMap::new();
    let mut name: Option<String> = None;
    let mut in_test = false;
    for line in manifest.lines() {
        let t = line.trim();
        if t == "[[test]]" {
            in_test = true;
            name = None;
            continue;
        }
        if t.starts_with('[') {
            in_test = false;
            continue;
        }
        if !in_test {
            continue;
        }
        if let Some(rest) = t.strip_prefix("name = ") {
            name = Some(rest.trim_matches('"').to_string());
        } else if let Some(rest) = t.strip_prefix("required-features = ") {
            let feats: BTreeSet<String> = rest
                .trim_matches(|c| c == '[' || c == ']')
                .split(',')
                .map(|s| s.trim().trim_matches('"').to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if let Some(n) = name.clone() {
                out.insert(n, feats);
            }
        }
    }
    out
}

/// Every `tests/*.rs`, with the `feature = "…"` names in its file-level
/// `#![cfg(…)]` and the raw condition text.
fn gated_suites() -> Vec<(String, BTreeSet<String>, String)> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        // The file-level attribute, if any: the first `#![cfg(` before any item.
        let head: String = text.lines().take(40).collect::<Vec<_>>().join("\n");
        let Some(at) = head.find("#![cfg(") else {
            continue;
        };
        let rest = &head[at + "#![cfg(".len()..];
        let mut depth = 1usize;
        let mut end = 0usize;
        for (i, c) in rest.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = i;
                        break;
                    }
                }
                _ => {}
            }
        }
        let cond = rest[..end].split_whitespace().collect::<Vec<_>>().join(" ");
        let feats: BTreeSet<String> = cond
            .split("feature = \"")
            .skip(1)
            .filter_map(|s| s.split('"').next())
            .map(|s| s.to_string())
            .collect();
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        out.push((name, feats, cond));
    }
    out.sort();
    out
}

/// A suite whose file-level gate asks for a feature its manifest entry does
/// not require can never run — not in CI, not locally, not in the job written
/// to cover it. It compiles to nothing and reports success.
#[test]
fn every_feature_gate_is_declared_in_the_manifest() {
    let declared = declared_targets();
    let mut problems = Vec::new();
    for (name, needs, cond) in gated_suites() {
        if needs.is_empty() {
            continue;
        }
        let have = match declared.get(&name) {
            Some(f) => closure(f),
            None => closure(&BTreeSet::new()),
        };
        let missing: Vec<_> = needs.difference(&have).cloned().collect();
        if !missing.is_empty() {
            problems.push(format!(
                "  tests/{name}.rs is gated on `{cond}` but its [[test]] entry \
                 supplies {:?} — missing {missing:?}",
                declared.get(&name)
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "these suites can never run:\n{}\n\nAdd the feature to the \
         `[[test]]` entry in Cargo.toml, or relax the file's `#![cfg]`.",
        problems.join("\n")
    );
}

/// The implication table is used one level deep, so it has to be written
/// transitively closed. Checking that here is cheaper than a fixed point.
#[test]
fn the_implication_table_is_closed() {
    let map: BTreeMap<&str, &[&str]> = IMPLIES.iter().copied().collect();
    for (outer, inner) in IMPLIES {
        for i in *inner {
            if let Some(deeper) = map.get(i) {
                for d in *deeper {
                    assert!(
                        inner.contains(d),
                        "`{outer}` implies `{i}` which implies `{d}`, but \
                         `{outer}`'s list does not mention `{d}`"
                    );
                }
            }
        }
    }
}

/// Name the suites that cannot run in this configuration, and say how to run
/// each. Always passes: a wasm-only suite being inert on a desktop is correct.
/// What is not correct is it being *silent*, which is what this fixes.
#[test]
fn report_which_suites_are_inert_here() {
    let have_wasm = cfg!(target_arch = "wasm32");
    let apple = cfg!(any(target_os = "macos", target_os = "ios"));
    let mut inert = Vec::new();
    for (name, _, cond) in gated_suites() {
        let arch_ok = !cond.contains("target_arch = \"wasm32\"") || have_wasm;
        let os_ok = !cond.contains("target_os = \"macos\"") || apple;
        if !arch_ok || !os_ok {
            inert.push((name, cond));
        }
    }
    if inert.is_empty() {
        eprintln!("every gated suite can run on this target");
        return;
    }
    eprintln!(
        "{} suite(s) are inert on this target and will report `ok. 0 passed`:",
        inert.len()
    );
    for (name, cond) in &inert {
        let how = if cond.contains("wasm32") {
            "scripts/wasm-browser-test.sh (see the file's own docs for FEATURES/TEST)"
        } else {
            "a macOS or iOS host"
        };
        eprintln!("  tests/{name}.rs — needs `{cond}`; run with {how}");
    }
}

/// A test file with a feature gate and no `[[test]]` entry vanishes under
/// `--no-default-features` with no sign that it did.
#[test]
fn every_gated_suite_has_a_manifest_entry() {
    let declared = declared_targets();
    let orphans: Vec<_> = gated_suites()
        .into_iter()
        .filter(|(name, feats, _)| !feats.is_empty() && !declared.contains_key(name))
        .map(|(name, feats, _)| format!("  tests/{name}.rs (gated on {feats:?})"))
        .collect();
    assert!(
        orphans.is_empty(),
        "these suites have a feature gate but no [[test]] entry, so nothing \
         records which features they need:\n{}",
        orphans.join("\n")
    );
}
