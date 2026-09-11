//! Flatten both connectome text trees into the tables the viewer fetches.
//!
//! ```text
//! cargo run --release -p threers-connectome --bin connectome-build
//! cargo run --release -p threers-connectome --bin connectome-build -- --tree /Volumes/C4TB/connectome-fs
//! ```

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use threers_connectome::{tables, Release, Tree};

const DEFAULT_TREE: &str = "/Volumes/C4TB/connectome-fs";

fn main() -> ExitCode {
    let mut tree_root = PathBuf::from(
        std::env::var("CONNECTOME_FS").unwrap_or_else(|_| DEFAULT_TREE.to_string()),
    );
    let mut out = default_data_dir();

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--tree" => tree_root = PathBuf::from(args.next().unwrap_or_default()),
            "--out" => out = PathBuf::from(args.next().unwrap_or_default()),
            "-h" | "--help" => {
                println!("usage: connectome-build [--tree <connectome-fs>] [--out <data dir>]");
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown argument: {other}");
                return ExitCode::FAILURE;
            }
        }
    }

    if !tree_root.is_dir() {
        eprintln!("connectome tree not found at {}", tree_root.display());
        eprintln!("pass --tree <dir> or set CONNECTOME_FS");
        return ExitCode::FAILURE;
    }

    let started = Instant::now();
    let mut trees = Vec::new();
    for rel in Release::ALL.iter() {
        println!("==> {}", rel.key);
        let t = match Tree::read(&tree_root, rel) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("    {}: {e}", rel.key);
                return ExitCode::FAILURE;
            }
        };
        println!(
            "    {} positioned from {}, {} recovered from skeleton.swc, {} unplaceable",
            group(t.from_index),
            rel.pos_source,
            group(t.from_skeleton),
            group(t.unplaceable.len())
        );
        trees.push(t);
    }

    match tables::write_tables(&out, &tree_root, &trees) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("writing {}: {e}", out.display());
            return ExitCode::FAILURE;
        }
    }

    let total: usize = trees.iter().map(|t| t.nodes.len()).sum();
    let indexed: usize = trees.iter().map(|t| t.indexed).sum();
    println!(
        "==> {} of {} neurons -> {} in {:.1}s",
        group(total),
        group(indexed),
        out.display(),
        started.elapsed().as_secs_f64()
    );
    ExitCode::SUCCESS
}

/// `crates/threers-connectome/data`, resolved from the crate rather than from
/// the shell's cwd so `cargo run` from anywhere in the workspace lands here.
fn default_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data")
}

fn group(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}
