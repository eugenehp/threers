//! Parametric compound spur gearbox → FreeCAD `.FCStd`.
//!
//! Ports the geometry pattern from `threers-mechanism-tour`'s `gear_train.scad`:
//! shared module `M`, trapezoid tooth profiles, phased meshes, compound layshaft.
//! Writes a multi-mesh FreeCAD document (one `Mesh::Feature` per colored part).
//!
//! ```text
//! cargo run --example parametric_gearbox --features openscad
//! cargo run --example parametric_gearbox --features openscad -- \
//!     --m 1.5 --n-in 12 --n-big 36 --n-small 16 --n-out 40 \
//!     --out out/parametric_gearbox.FCStd
//! ```

use std::f32::consts::{FRAC_PI_2, PI};
use threers::{
    cube, cylinder, linear_extrude, FcstdDocumentMeta, FcstdMeshSpec, FcstdWriteOptions,
    FcstdWriter, Solid,
};

struct Params {
    module: f32,
    n_in: u32,
    n_big: u32,
    n_small: u32,
    n_out: u32,
    face: f32,
    shaft: f32,
    deck: f32,
    out: String,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            module: 1.5,
            n_in: 12,
            n_big: 36,
            n_small: 16,
            n_out: 40,
            face: 8.0,
            shaft: 4.0,
            deck: 8.0,
            out: "out/parametric_gearbox.FCStd".into(),
        }
    }
}

fn arg_f(args: &[String], flag: &str) -> Option<f32> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .and_then(|w| w[1].parse().ok())
}

fn arg_u(args: &[String], flag: &str) -> Option<u32> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .and_then(|w| w[1].parse().ok())
}

fn arg_s(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].clone())
}

fn parse_params() -> Params {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut p = Params::default();
    if let Some(v) = arg_f(&args, "--m") {
        p.module = v;
    }
    if let Some(v) = arg_u(&args, "--n-in") {
        p.n_in = v;
    }
    if let Some(v) = arg_u(&args, "--n-big") {
        p.n_big = v;
    }
    if let Some(v) = arg_u(&args, "--n-small") {
        p.n_small = v;
    }
    if let Some(v) = arg_u(&args, "--n-out") {
        p.n_out = v;
    }
    if let Some(v) = arg_f(&args, "--face") {
        p.face = v;
    }
    if let Some(v) = arg_f(&args, "--shaft") {
        p.shaft = v;
    }
    if let Some(v) = arg_f(&args, "--deck") {
        p.deck = v;
    }
    if let Some(v) = arg_s(&args, "--out") {
        p.out = v;
    }
    p
}

fn pitch_r(module: f32, n: u32) -> f32 {
    module * n as f32 / 2.0
}

/// Trapezoid tooth ring in the XY plane (OpenSCAD `gear_2d`).
fn gear_2d(module: f32, n: u32) -> Vec<[f32; 2]> {
    let p = 360.0 / n as f32;
    let r = pitch_r(module, n);
    let tip = r + module;
    let root = r - 1.25 * module;
    let mut pts = Vec::with_capacity(n as usize * 4);
    for i in 0..n {
        let i = i as f32;
        for (rad, ang) in [
            (root, (i - 0.25) * p),
            (tip, (i - 0.07) * p),
            (tip, (i + 0.07) * p),
            (root, (i + 0.25) * p),
        ] {
            let a = ang * PI / 180.0;
            pts.push([rad * a.cos(), rad * a.sin()]);
        }
    }
    pts
}

fn wheel(module: f32, n: u32, height: f32, phase_deg: f32) -> Solid {
    let outline = gear_2d(module, n);
    linear_extrude(height, &outline).rotate_z(phase_deg * PI / 180.0)
}

/// OpenSCAD-style axis-aligned box with a corner at the origin.
fn box_from_origin(size: [f32; 3]) -> Solid {
    cube(size).translate([size[0] / 2.0, size[1] / 2.0, size[2] / 2.0])
}

fn build(p: &Params) -> Solid {
    let m = p.module;
    let a = [
        26.0,
        40.0,
        p.deck,
    ];
    let b = [
        a[0] + pitch_r(m, p.n_in) + pitch_r(m, p.n_big),
        40.0,
        p.deck,
    ];
    let c = [
        b[0] + pitch_r(m, p.n_small) + pitch_r(m, p.n_out),
        40.0,
        p.deck,
    ];

    let ph_c = 180.0 / p.n_big as f32;
    let ph_out = 180.0 - ph_c * p.n_small as f32 / p.n_out as f32 - 180.0 / p.n_out as f32;
    let overlap = 0.4;

    let deck_w = (c[0] + pitch_r(m, p.n_out) + 16.0).max(145.0);
    let deck_d = 80.0;

    let frame = box_from_origin([deck_w, deck_d, p.deck])
        .color_named("slategray")
        .union(
            cylinder(20.0, p.shaft)
                .rotate_x(FRAC_PI_2)
                .translate([a[0], a[1], 10.0])
                .color_named("dimgray"),
        )
        .union(
            cylinder(27.0, p.shaft)
                .rotate_x(FRAC_PI_2)
                .translate([b[0], b[1], 13.5])
                .color_named("dimgray"),
        )
        .union(
            cylinder(29.0, p.shaft)
                .rotate_x(FRAC_PI_2)
                .translate([c[0], c[1], 14.5])
                .color_named("dimgray"),
        );

    // Simple housing walls (clearance around the gear stack).
    let wall = 4.0;
    let house_h = p.deck + p.face * 2.0 + 12.0;
    let housing = box_from_origin([deck_w, wall, house_h])
        .color_named("gray")
        .union(
            box_from_origin([deck_w, wall, house_h]).translate([0.0, deck_d - wall, 0.0]),
        )
        .union(box_from_origin([wall, deck_d, house_h]))
        .union(
            box_from_origin([wall, deck_d, house_h]).translate([deck_w - wall, 0.0, 0.0]),
        );

    let input = wheel(m, p.n_in, p.face, 0.0)
        .translate([a[0], a[1], a[2]])
        .color_named("indianred");

    let compound = wheel(m, p.n_big, p.face, ph_c)
        .union(
            wheel(m, p.n_small, p.face + overlap, ph_c)
                .translate([0.0, 0.0, p.face - overlap]),
        )
        .translate([b[0], b[1], b[2]])
        .color_named("steelblue");

    let output = wheel(m, p.n_out, p.face, ph_out)
        .translate([c[0], c[1], c[2] + p.face])
        .color_named("olivedrab");

    frame.union(housing).union(input).union(compound).union(output)
}

fn main() {
    let p = parse_params();
    let ratio = (p.n_big as f32 / p.n_in as f32) * (p.n_out as f32 / p.n_small as f32);

    println!("parametric gearbox");
    println!(
        "  module={:.3}  teeth=[{}/{}/{}/{}]  face={:.1}  shaft={:.1}",
        p.module, p.n_in, p.n_big, p.n_small, p.n_out, p.face, p.shaft
    );
    println!("  overall ratio = {ratio:.3} : 1  (input turns per output turn)");

    let solid = build(&p);
    let parts = solid.parts();
    println!("  colored parts: {}", parts.len());

    let specs = FcstdMeshSpec::from_parts(&parts);
    let opts = FcstdWriteOptions {
        neighbours: true,
        weld: true,
        meta: FcstdDocumentMeta {
            label: "ParametricGearbox".into(),
            comment: format!(
                "module={} teeth={}/{}/{}/{} ratio={:.3}",
                p.module, p.n_in, p.n_big, p.n_small, p.n_out, ratio
            ),
            created_by: "threers parametric_gearbox".into(),
            company: String::new(),
            uid: "auto".into(),
        },
    };
    let report = FcstdWriter::save(&p.out, &specs, &opts).expect("write FCStd");
    println!(
        "  wrote {}: {} objects, {} facets",
        p.out, report.objects, report.facets
    );
}
