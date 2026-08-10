//! M3 oracle-comparison tool: compare two STL meshes with the four-metric gate
//! (watertight · volume · Euler characteristic · Hausdorff). Used by
//! `scripts/ci-openscad.sh` to check our kernel's output against an
//! OpenSCAD/CGAL reference STL.
//!
//! Run: `cargo run --example openscad_compare --features openscad -- ours.stl ref.stl [vol_tol] [haus_tol]`
//! Exit code 0 = PASS, 1 = FAIL.

use threers::exact_csg::metrics::compare;
use threers::StlLoader;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: openscad_compare <ours.stl> <reference.stl> [vol_tol] [haus_tol]");
        std::process::exit(2);
    }
    let vol_tol = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1e-3);
    let haus_tol = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1e-3);

    let ours = StlLoader::parse(&std::fs::read(&args[1]).expect("read ours.stl"));
    let refm = StlLoader::parse(&std::fs::read(&args[2]).expect("read reference.stl"));

    let r = compare(&ours, &refm, vol_tol, haus_tol);
    println!(
        "watertight={}  vol_rel_err={:.6}  euler_match={}  hausdorff={:.6}  =>  {}",
        r.watertight,
        r.vol_rel_err,
        r.euler_match,
        r.hausdorff,
        if r.pass { "PASS" } else { "FAIL" }
    );
    std::process::exit(if r.pass { 0 } else { 1 });
}
