//! Parse an OpenSCAD `.scad` file with the pure-Rust front end and write a mesh.
//! `include`/`use`/`import` resolve relative to the file's folder.
//!
//! The **output format is chosen by the extension** of the output path:
//! `.stl` (binary), `.obj`, `.off`, `.3mf`, or `.glb` (binary glTF 2.0).
//!
//! Run: `cargo run --example scad2stl --features openscad -- <in.scad> <out.{stl,obj,off,3mf,glb}>`
//!
//! By default the watertight exact CSG kernel is used. Pass `--float` for the
//! faster float kernel — the right choice for multi-part *assemblies* (many
//! disjoint/lightly-overlapping parts) where exact booleans are unnecessary.

use threers::{
    geometry_to_3mf, geometry_to_glb, geometry_to_obj, geometry_to_off, geometry_to_stl,
    parse_scad_file,
};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let float = args.iter().any(|a| a == "--float");
    let pos: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    if pos.len() < 2 {
        eprintln!("usage: scad2stl [--float] <in.scad> <out.{{stl,obj,off,3mf,glb}}>");
        std::process::exit(2);
    }
    let (input, output) = (pos[0], pos[1]);
    let ext = output.rsplit('.').next().unwrap_or("").to_ascii_lowercase();

    let solid = match parse_scad_file(input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error in {input}: {e}");
            std::process::exit(2);
        }
    };
    // Evaluate once with the requested kernel, then encode.
    let geom = if float { solid.to_geometry() } else { solid.to_geometry_exact() };
    let bytes: Vec<u8> = match ext.as_str() {
        "stl" => geometry_to_stl(&geom),
        "obj" => geometry_to_obj(&geom).into_bytes(),
        "off" => geometry_to_off(&geom).into_bytes(),
        "3mf" => geometry_to_3mf(&geom),
        "glb" => geometry_to_glb(&geom),
        other => {
            eprintln!("unknown output format '.{other}' (use stl/obj/off/3mf/glb)");
            std::process::exit(2);
        }
    };

    let tris = geom.index.as_ref().map(|i| i.len() / 3).unwrap_or_else(|| {
        geom.get_attribute("position").map(|a| a.count() / 3).unwrap_or(0)
    });
    std::fs::write(output, &bytes).expect("write output");
    println!(
        "parsed {input} → {output} · {tris} triangles · {} bytes · {} kernel",
        bytes.len(),
        if float { "float" } else { "exact" }
    );
}
