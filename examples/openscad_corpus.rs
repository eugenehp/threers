//! M3 corpus generator: **parse a `.scad` file with our own front end** and
//! write the resulting STL. The harness (`scripts/ci-openscad.sh`) then compares
//! this against real OpenSCAD/CGAL rendering the *same* source — an end-to-end
//! test of parser + evaluator + kernel vs the reference implementation.
//!
//! Run: `cargo run --example openscad_corpus --features openscad -- <model> <out.stl>`

use threers::parse_scad_file;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: openscad_corpus <model> <out.stl>");
        std::process::exit(2);
    }
    let (model, out) = (&args[1], &args[2]);
    let path = format!("tests/openscad-corpus/{model}.scad");
    match parse_scad_file(&path) {
        Ok(solid) => {
            std::fs::write(out, solid.to_stl()).expect("write stl");
            println!("parsed {model}.scad → {out}");
        }
        Err(e) => {
            eprintln!("parse error in {model}.scad: {e}");
            std::process::exit(2);
        }
    }
}
