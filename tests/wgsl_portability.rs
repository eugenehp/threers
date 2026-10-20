//! WGSL that only wgpu's own parser accepts.
//!
//! The fluid rendered a black canvas at sixty frames a second in Safari and
//! in Chrome's WebGPU, with nothing in wgpu's log, because one line of the
//! frame-rate read-out's glyph code mixed `&&` and `||` without
//! parentheses. The WGSL specification forbids that; **naga accepts it
//! anyway**, so every native test and every WebGL2 browser test passed —
//! naga compiles the module itself there. In a real WebGPU browser the
//! module is compiled by Tint, which rejects it, and a rejected *module*
//! takes every pipeline in it down at once. The error arrives in the
//! browser's console rather than through wgpu, so the page looks like it is
//! running and draws nothing.
//!
//! Two of these have bitten, both the same shape: naga is the lenient one,
//! Tint is the specification, and the fluid's browser tests ran on WebGL2
//! where naga compiles everything itself.
//!
//! This walks the WGSL in the tree rather than asking each module to
//! register its source, so a shader added later is covered without anyone
//! remembering to come back here.

use std::path::{Path, PathBuf};

/// Drop `//` and `/* */` so an operator inside a comment is not an operator.
fn strip_comments(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < b.len() {
        if b[i..].starts_with(b"//") {
            // Keep the newline: the line numbers below are what a reader greps for.
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if b[i..].starts_with(b"/*") {
            i += 2;
            while i < b.len() && !b[i..].starts_with(b"*/") {
                if b[i] == b'\n' {
                    out.push('\n');
                }
                i += 1;
            }
            i = (i + 2).min(b.len());
        } else {
            out.push(b[i] as char);
            i += 1;
        }
    }
    out
}

/// `(file, line, the offending line)` for every parenthesised group — or
/// statement — that holds both `&&` and `||`.
fn mixed_logical_ops(source: &str) -> Vec<(usize, String)> {
    let text = strip_comments(source);
    let b = text.as_bytes();
    let mut found = Vec::new();
    let mut depth: Vec<[bool; 2]> = vec![[false, false]];
    let mut line = 1usize;
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'\n' => line += 1,
            b'(' => depth.push([false, false]),
            b')' => {
                if depth.len() > 1 {
                    depth.pop();
                }
            }
            // A statement boundary resets it: two statements on one line are
            // not one expression. `{` and `}` likewise.
            b';' | b'{' | b'}' => depth = vec![[false, false]],
            _ => {}
        }
        let here = depth.last_mut().expect("never empty");
        if b[i..].starts_with(b"&&") {
            here[0] = true;
            i += 1;
        } else if b[i..].starts_with(b"||") {
            here[1] = true;
            i += 1;
        }
        if here[0] && here[1] {
            let whole = text.lines().nth(line - 1).unwrap_or("").trim().to_string();
            found.push((line, whole));
            *here = [false, false];
        }
        i += 1;
    }
    found
}

/// Raw-string literals in a Rust file that look like WGSL.
fn wgsl_literals(rust: &str) -> Vec<(usize, &str)> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(start) = rust[from..].find("r#\"") {
        let open = from + start + 3;
        let Some(len) = rust[open..].find("\"#") else {
            break;
        };
        let body = &rust[open..open + len];
        // WGSL, and not the OpenSCAD, GLSL or JSON that also live in raw
        // strings here: these three attributes exist in no other language.
        if body.contains("@group(")
            || body.contains("@fragment")
            || body.contains("@vertex")
            || body.contains("vec4<f32>")
        {
            out.push((rust[..open].matches('\n').count() + 1, body));
        }
        from = open + len + 2;
    }
    out
}

fn rust_files(dir: &Path, into: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_files(&path, into);
        } else if path.extension().is_some_and(|e| e == "rs") {
            into.push(path);
        }
    }
}

/// Mixing `&&` and `||` without parentheses: legal in naga, rejected by
/// Tint, so it passes every test here and fails in the browser.
#[test]
fn no_wgsl_mixes_and_with_or() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_files(&root, &mut files);
    assert!(files.len() > 50, "only found {} rust files", files.len());

    let (mut shaders, mut problems) = (0usize, Vec::new());
    for file in &files {
        let Ok(text) = std::fs::read_to_string(file) else {
            continue;
        };
        for (offset, body) in wgsl_literals(&text) {
            shaders += 1;
            for (line, source) in mixed_logical_ops(body) {
                problems.push(format!(
                    "{}:{}: {source}",
                    file.strip_prefix(&root).unwrap_or(file).display(),
                    offset + line - 1,
                ));
            }
        }
    }
    eprintln!(
        "  checked {shaders} WGSL literals across {} files",
        files.len()
    );
    assert!(shaders > 10, "only found {shaders} WGSL literals to check");
    assert!(
        problems.is_empty(),
        "WGSL that Tint will reject, taking the whole module with it \u{2014} \
         parenthesise the operands:\n  {}",
        problems.join("\n  ")
    );
}

/// The checker itself, since a lint that cannot fail is not a lint.
#[test]
fn the_checker_finds_what_it_is_looking_for() {
    assert!(!mixed_logical_ops("let a = x && y || z && w;").is_empty());
    assert!(mixed_logical_ops("let a = (x && y) || (z && w);").is_empty());
    assert!(mixed_logical_ops("let a = x || y || z;").is_empty());
    assert!(mixed_logical_ops("let a = x && y && z;").is_empty());
    // Separate statements are separate expressions.
    assert!(mixed_logical_ops("let a = x && y; let b = p || q;").is_empty());
    // And an operator in a comment is not an operator.
    assert!(mixed_logical_ops("// x && y || z\nlet a = p || q;").is_empty());
    // The real line, before and after the fix.
    assert!(!mixed_logical_ops(
        "let leading = column == 0 && shown < 100 || column == 1 && shown < 10;"
    )
    .is_empty());
    assert!(mixed_logical_ops(
        "let leading = (column == 0 && shown < 100) || (column == 1 && shown < 10);"
    )
    .is_empty());
}

/// The volume samples with an explicit level, because it samples inside a
/// loop.
///
/// `textureSample` needs implicit derivatives, so WGSL permits it only in
/// uniform control flow — and the volume's fetches happen inside the ray
/// march, under its conditionals and its loop. naga does not enforce that;
/// Tint rejects the module, and with it every pipeline, so the volume drew
/// nothing in any real WebGPU browser while WebGL2 was perfect.
///
/// The plane's shader is left alone deliberately: its seventy-odd
/// `textureSample` calls are all at the top level of a fragment function,
/// which *is* uniform control flow, and they want the plain form.
#[test]
fn the_volume_shader_samples_with_an_explicit_level() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/fluid/volume_shader.rs");
    let src = std::fs::read_to_string(&path).expect("the volume's shader is there");
    // `textureSampleLevel(` does not contain `textureSample(`, so this
    // counts only the derivative-taking form.
    let offenders: Vec<&str> = src
        .lines()
        .filter(|l| l.contains("textureSample(") && !l.trim_start().starts_with("//"))
        .collect();
    assert!(
        offenders.is_empty(),
        "the volume shader must sample with `textureSampleLevel`, which is \
         legal outside uniform control flow:\n  {}",
        offenders.join("\n  ")
    );
    assert!(
        src.contains("textureSampleLevel("),
        "no sampling at all in the volume shader — has it moved?"
    );
}
