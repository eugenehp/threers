//! Project a `.scad` model to technical-drawing SVG — the drawing counterpart
//! to `scad2stl` (`--features openscad`).
//!
//! Run:
//!   cargo run --release --example scad_schematic --features openscad -- \
//!       part.scad out/part.svg --view front --hidden
//!
//! `--view all` writes one file per standard view, suffixing the output stem
//! (`out/part_front.svg`, `out/part_top.svg`, …).
//!
//! Flags:
//!   --view <front|back|left|right|top|bottom|iso|all>   default front
//!   --hidden          resolve occlusion and draw hidden edges dashed
//!   --crease <deg>    keep interior edges sharper than this (default 22)
//!   --stroke <mm>     line width (default 0.35)
//!   --raster <n>      depth-buffer resolution for hidden-line removal
//!   --px <n>          pixel size when the output path ends in .png
//!   --float           use the float kernel instead of the exact one

use threers::{
    encode_png, parse_scad_file, schematic_project_view, schematic_to_rgba, schematic_to_svg,
    SchematicOptions, View,
};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |k: &str| args.iter().any(|a| a == k);
    let val = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let pos: Vec<&String> = args
        .iter()
        .enumerate()
        .filter(|(i, a)| {
            !a.starts_with("--")
                && !matches!(
                    args.get(i.wrapping_sub(1)).map(|s| s.as_str()),
                    Some("--view")
                        | Some("--crease")
                        | Some("--stroke")
                        | Some("--raster")
                        | Some("--px")
                )
        })
        .map(|(_, a)| a)
        .collect();
    if pos.len() < 2 {
        eprintln!(
            "usage: scad_schematic <in.scad> <out.svg> [--view V|all] [--hidden] \
             [--crease deg] [--stroke mm] [--raster n] [--float]"
        );
        std::process::exit(2);
    }
    let (input, output) = (pos[0], pos[1]);

    let opts = SchematicOptions {
        crease_deg: val("--crease").and_then(|v| v.parse().ok()).unwrap_or(22.0),
        hidden: flag("--hidden"),
        raster: val("--raster").and_then(|v| v.parse().ok()).unwrap_or(2048),
        ..Default::default()
    };
    let stroke: f32 = val("--stroke").and_then(|v| v.parse().ok()).unwrap_or(0.35);

    let solid = match parse_scad_file(input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error in {input}: {e}");
            std::process::exit(2);
        }
    };
    let geom = if flag("--float") {
        solid.to_geometry()
    } else {
        solid.to_geometry_exact()
    };

    let which = val("--view").unwrap_or_else(|| "front".into());
    let views: Vec<(&str, View)> = if which.eq_ignore_ascii_case("all") {
        vec![
            ("front", View::Front),
            ("back", View::Back),
            ("left", View::Left),
            ("right", View::Right),
            ("top", View::Top),
            ("bottom", View::Bottom),
            ("iso", View::Iso),
        ]
    } else {
        match View::parse(&which) {
            Some(v) => vec![("", v)],
            None => {
                eprintln!("unknown view '{which}'");
                std::process::exit(2);
            }
        }
    };

    for (name, view) in views {
        let drawing = schematic_project_view(&geom, view, &opts);
        let path = if name.is_empty() {
            output.clone()
        } else {
            let (stem, ext) = output.rsplit_once('.').unwrap_or((output.as_str(), "svg"));
            format!("{stem}_{name}.{ext}")
        };
        if path.ends_with(".png") {
            let px = val("--px").and_then(|v| v.parse().ok()).unwrap_or(1400);
            let (w, h, rgba) = schematic_to_rgba(&drawing, px, stroke * 4.0);
            std::fs::write(&path, encode_png(w, h, &rgba)).expect("write png");
        } else {
            std::fs::write(&path, schematic_to_svg(&drawing, stroke)).expect("write svg");
        }
        let [w, h] = drawing.size();
        println!(
            "{path} · {w:.1} x {h:.1} mm · {} visible, {} hidden segments",
            drawing.visible.len(),
            drawing.hidden.len()
        );
    }
}
