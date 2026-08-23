//! OpenSCAD feature **gallery** — a single runnable example that exercises the
//! full breadth of the pure-Rust OpenSCAD front end. Every snippet below is real
//! `.scad` source; each is parsed, turned into geometry with the hybrid exact
//! kernel, written out as a binary STL, and summarised (triangles, bounding box,
//! watertight?). Pass `--png` to also render a thumbnail per demo (needs a GPU).
//!
//! Run:
//!   cargo run --example openscad_gallery --features openscad
//!   cargo run --example openscad_gallery --features openscad -- out/gallery --png
//!
//! It is intentionally a catalogue: primitives (2D/3D), booleans, transforms,
//! extrusions, projection, hull/minkowski/offset, control flow, modules with
//! `children()`, functions + recursion, list comprehensions, special variables,
//! and `surface()` from a height field — so you can *see* each render end-to-end.

use std::path::{Path, PathBuf};
use threers::{parse_scad, BufferGeometry};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let want_png = args.iter().any(|a| a == "--png");
    let out = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "out/openscad-gallery".into());
    let out = PathBuf::from(out);
    std::fs::create_dir_all(&out).expect("create out dir");

    let demos = demos(&out);
    println!(
        "OpenSCAD gallery — {} demos → {}\n",
        demos.len(),
        out.display()
    );

    let mut rows: Vec<Row> = Vec::new();
    for d in &demos {
        use std::io::Write;
        print!("  · {:<24} ", d.name);
        std::io::stdout().flush().ok();
        let t = std::time::Instant::now();
        let row = build_one(d, &out);
        println!(
            "{:>7} tris  {:>5}ms  {}",
            row.tris,
            t.elapsed().as_millis(),
            row.err.as_deref().unwrap_or("ok")
        );
        rows.push(row);
    }
    println!();

    report(&rows);

    if want_png {
        match render_pngs(&demos, &rows, &out) {
            Ok(n) => println!("\nrendered {n} PNG thumbnails → {}/png/", out.display()),
            Err(e) => println!("\n(PNG rendering skipped: {e})"),
        }
    } else {
        println!("\n(STL only — re-run with `--png` for thumbnails)");
    }

    let ok = rows.iter().filter(|r| r.err.is_none()).count();
    let wt = rows.iter().filter(|r| r.watertight).count();
    println!(
        "\n{ok}/{} demos rendered · {wt} watertight · STLs in {}",
        rows.len(),
        out.display()
    );
}

/// One catalogue entry: category, name, and its `.scad` source.
struct Demo {
    cat: &'static str,
    name: &'static str,
    src: String,
}
fn d(cat: &'static str, name: &'static str, src: &str) -> Demo {
    Demo {
        cat,
        name,
        src: src.to_string(),
    }
}

/// The catalogue. Grouped by category; every snippet uses only implemented
/// features so the whole gallery renders clean.
fn demos(out: &Path) -> Vec<Demo> {
    // A small height field for surface(): a Gaussian bump on a 9×9 grid.
    let dat = out.join("heightmap.dat");
    let mut hf = String::new();
    for j in 0..9 {
        for i in 0..9 {
            let (x, y) = (i as f64 - 4.0, j as f64 - 4.0);
            let z = 8.0 * (-(x * x + y * y) / 8.0).exp();
            hf.push_str(&format!("{z:.4} "));
        }
        hf.push('\n');
    }
    std::fs::write(&dat, hf).expect("write heightmap");
    let dat = dat.display().to_string();

    vec![
        // --- 3D primitives ---
        d("3D primitives", "cube", "cube([24, 16, 8], center=true);"),
        d("3D primitives", "sphere", "sphere(r=10, $fn=48);"),
        d("3D primitives", "cone", "cylinder(h=18, r1=10, r2=2, $fn=48);"),
        d(
            "3D primitives",
            "polyhedron",
            "polyhedron(
               points=[[0,0,10],[8,8,-4],[8,-8,-4],[-8,-8,-4],[-8,8,-4]],
               faces=[[0,1,2],[0,2,3],[0,3,4],[0,4,1],[1,4,3],[1,3,2]]);",
        ),
        // --- 2D primitives (extruded so they have volume to show) ---
        d("2D primitives", "circle", "linear_extrude(3) circle(r=10, $fn=64);"),
        d("2D primitives", "square", "linear_extrude(3) square([18, 11], center=true);"),
        d("2D primitives", "polygon", "linear_extrude(4) polygon([[0,0],[22,0],[16,14],[4,14]]);"),
        d(
            "2D primitives",
            "text",
            r#"linear_extrude(4) text("SCAD", size=11, halign="center", valign="center");"#,
        ),
        // --- Booleans (curved → exercises the exact arrangement kernel) ---
        d("Booleans", "difference", "difference(){ cube(18,center=true); sphere(r=11,$fn=44); }"),
        d("Booleans", "union", "union(){ cube(15,center=true); sphere(r=9.5,$fn=44); }"),
        d("Booleans", "intersection", "intersection(){ cube(18,center=true); sphere(r=11,$fn=44); }"),
        d(
            "Booleans",
            "through-hole",
            // Offset hole → the exact arrangement kernel closes it watertight
            // (a perfectly-centred hole hits a symmetric degeneracy → float fallback).
            "difference(){ cube(20,center=true); translate([3,2,0]) cylinder(h=30,r=6,$fn=48,center=true); }",
        ),
        // --- Transforms ---
        d(
            "Transforms",
            "translate-rotate-scale",
            "translate([0,0,4]) rotate([0,30,0]) scale([1,1,2]) cube(8, center=true);",
        ),
        d("Transforms", "mirror", "mirror([1,0,0]) translate([6,0,0]) cylinder(h=12,r1=6,r2=0,$fn=40);"),
        d(
            "Transforms",
            "multmatrix-shear",
            "multmatrix([[1,0,0.6,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) cube([10,10,14], center=true);",
        ),
        d("Transforms", "resize", "resize([34,10,10]) sphere(r=10, $fn=40);"),
        d("Transforms", "color", r#"color("orange") cube([16,16,4], center=true);"#),
        // --- Extrusion ---
        d("Extrusion", "linear_extrude", "linear_extrude(height=14) polygon([[0,0],[16,0],[8,12]]);"),
        d("Extrusion", "linear_extrude-twist", "linear_extrude(height=26, twist=200, scale=0.4, slices=48) square([16,16], center=true);"),
        d("Extrusion", "rotate_extrude-torus", "rotate_extrude($fn=64) translate([10,0]) circle(r=3, $fn=32);"),
        d(
            "Extrusion",
            "rotate_extrude-partial",
            "rotate_extrude(angle=270, $fn=64) translate([9,0]) square([4,10]);",
        ),
        // --- 3D → 2D ---
        d(
            "Projection",
            "projection-cut",
            "linear_extrude(3) projection(cut=true) translate([0,0,4]) sphere(r=9, $fn=44);",
        ),
        // --- Hull / Minkowski / Offset ---
        d(
            "Hull/Mink/Offset",
            "hull-2d",
            "linear_extrude(4) hull(){ translate([-11,0]) circle(4,$fn=40); translate([11,0]) circle(6,$fn=40); }",
        ),
        d(
            "Hull/Mink/Offset",
            "hull-3d",
            "hull(){ translate([-9,0,0]) sphere(4,$fn=28); translate([9,0,6]) cube(5, center=true); }",
        ),
        d(
            "Hull/Mink/Offset",
            "minkowski-rounded",
            "minkowski(){ cube([18,12,6], center=true); sphere(r=2.5, $fn=20); }",
        ),
        d(
            "Hull/Mink/Offset",
            "offset-round",
            "linear_extrude(4) offset(r=4, $fn=40) square([18,10], center=true);",
        ),
        d(
            "Hull/Mink/Offset",
            "fill-holes",
            // fill() closes the ring's hole back into a solid disk.
            "linear_extrude(4) fill() difference(){ circle(11,$fn=56); circle(6,$fn=56); }",
        ),
        // --- Control flow / modules / functions ---
        d(
            "Control flow",
            "for-array",
            "for(i=[0:6]) translate([i*6-18, 0, 0]) cube([4, 4, 2 + i*2], center=false);",
        ),
        d(
            "Control flow",
            "for-radial",
            "for(a=[0:30:330]) rotate([0,0,a]) translate([13,0,0]) cylinder(h=10, r=1.6, $fn=20);",
        ),
        d(
            "Control flow",
            "intersection_for",
            "intersection_for(a=[0:45:135]) rotate([0,0,a]) cube([26,7,7], center=true);",
        ),
        d(
            "Modules",
            "children-ring",
            "module ring(n){ for(i=[0:n-1]) rotate([0,0,i*360/n]) translate([12,0,0]) children(); }
             ring(9) sphere(r=2.4, $fn=20);",
        ),
        d(
            "Modules",
            "recursion-tree",
            // Branches are boxes (planar union → exact & watertight); a cylinder
            // tree would exercise the curved-curved union that still falls back.
            "module tree(n, len){
               if (n > 0){
                 translate([-len/12, -len/12, 0]) cube([len/6, len/6, len]);
                 translate([0,0,len])
                   for(a=[32,-32]) rotate([a,0,0]) tree(n-1, len*0.7);
               }
             }
             tree(4, 20);",
        ),
        d(
            "Functions",
            "function+comprehension",
            "function star(n, ro, ri) =
               [ for(i=[0:2*n-1]) let(a = i*180/n, r = (i%2==0)?ro:ri) [r*cos(a), r*sin(a)] ];
             linear_extrude(4) polygon(star(6, 15, 6));",
        ),
        d(
            "Functions",
            "rands-scatter",
            // Seeded rands() → a reproducible scatter of pillars of varying height.
            "h = rands(3, 14, 36, 5);
             for(i=[0:35]) translate([(i%6)*5-13, floor(i/6)*5-13, 0])
               cube([3.4, 3.4, h[i]]);",
        ),
        // --- Special variables / math ---
        d(
            "Special vars",
            "fn-vs-fa-fs",
            "translate([-12,0,0]) sphere(r=9, $fn=10);
             translate([ 12,0,0]) sphere(r=9, $fa=6, $fs=1);",
        ),
        d(
            "Special vars",
            "sin-field",
            "for(x=[0:1:22]) translate([x-11, 0, 4*sin(x*22)]) cube([0.9,6,0.9], center=true);",
        ),
        // --- Import / height field ---
        d("Surface", "surface-heightfield", &format!(r#"surface(file="{dat}", center=true);"#)),
    ]
}

struct Row {
    cat: &'static str,
    name: &'static str,
    tris: usize,
    size: [f32; 3],
    watertight: bool,
    err: Option<String>,
    geom: Option<BufferGeometry>,
}

fn build_one(demo: &Demo, out: &Path) -> Row {
    let mk = |tris, size, watertight, err, geom| Row {
        cat: demo.cat,
        name: demo.name,
        tris,
        size,
        watertight,
        err,
        geom,
    };
    match parse_scad(&demo.src) {
        Ok(solid) => {
            let g = solid.to_geometry_exact();
            let pts = positions(&g);
            let tris = pts.len() / 3;
            if tris == 0 {
                return mk(0, [0.0; 3], false, Some("empty geometry".into()), None);
            }
            let size = bbox_size(&pts);
            let wt = watertight(&pts);
            std::fs::write(
                out.join(format!("{}.stl", demo.name)),
                threers::geometry_to_stl(&g),
            )
            .ok();
            mk(tris, size, wt, None, Some(g))
        }
        Err(e) => mk(0, [0.0; 3], false, Some(e), None),
    }
}

/// The geometry as a flat triangle soup (3 verts per triangle), expanding the
/// index buffer when present — primitives ship indexed, the CSG kernel ships soup.
fn positions(g: &BufferGeometry) -> Vec<[f32; 3]> {
    let verts: Vec<[f32; 3]> = match g.positions() {
        Some(it) => it.map(|v| [v.x, v.y, v.z]).collect(),
        None => return Vec::new(),
    };
    match &g.index {
        Some(idx) => idx.iter().map(|&i| verts[i as usize]).collect(),
        None => verts,
    }
}

fn bbox_size(pts: &[[f32; 3]]) -> [f32; 3] {
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for p in pts {
        for k in 0..3 {
            mn[k] = mn[k].min(p[k]);
            mx[k] = mx[k].max(p[k]);
        }
    }
    [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]]
}

/// Watertight = every welded edge is shared by exactly two triangles.
fn watertight(pts: &[[f32; 3]]) -> bool {
    use std::collections::HashMap;
    let key = |p: [f32; 3]| {
        (
            (p[0] * 1e4).round() as i64,
            (p[1] * 1e4).round() as i64,
            (p[2] * 1e4).round() as i64,
        )
    };
    type PointKey = (i64, i64, i64);
    let mut edges: HashMap<(PointKey, PointKey), i32> = HashMap::new();
    for t in pts.chunks_exact(3) {
        for k in 0..3 {
            let (mut a, mut b) = (key(t[k]), key(t[(k + 1) % 3]));
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
            *edges.entry((a, b)).or_insert(0) += 1;
        }
    }
    !edges.is_empty() && edges.values().all(|&c| c == 2)
}

fn report(rows: &[Row]) {
    println!(
        "{:<18} {:<24} {:>8}  {:<18} {:>3}",
        "category", "demo", "tris", "size (x×y×z)", "wt"
    );
    println!("{}", "─".repeat(80));
    let mut last = "";
    for r in rows {
        if r.cat != last {
            last = r.cat;
        }
        match &r.err {
            Some(e) => println!(
                "{:<18} {:<24} {:>8}  {}",
                r.cat,
                r.name,
                "ERR",
                truncate(e, 34)
            ),
            None => println!(
                "{:<18} {:<24} {:>8}  {:>5.1}×{:>5.1}×{:>5.1}   {:>3}",
                r.cat,
                r.name,
                r.tris,
                r.size[0],
                r.size[1],
                r.size[2],
                if r.watertight { "yes" } else { "—" }
            ),
        }
    }
}

fn truncate(s: &str, n: usize) -> String {
    let s = s.replace('\n', " ");
    if s.len() > n {
        format!("{}…", &s[..n])
    } else {
        s
    }
}

// ---------------------------------------------------------------------------
// Optional: render a PNG thumbnail per demo (auto-framed camera). Needs a GPU.
// ---------------------------------------------------------------------------

fn render_pngs(demos: &[Demo], rows: &[Row], out: &Path) -> Result<usize, String> {
    use threers::{
        AmbientLight, Color, DirectionalLight, HeadlessRenderer, Mesh, Object3D, PerspectiveCamera,
        Scene, StandardMaterial, Vector3,
    };

    let dir = out.join("png");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let (w, h) = (480u32, 480u32);
    let mut hr = HeadlessRenderer::builder()
        .size(w, h)
        .supersample(2)
        .build()
        .map_err(|e| format!("no GPU adapter ({e})"))?;
    let (rw, rh) = hr.render_size();

    let mut n = 0;
    for (demo, row) in demos.iter().zip(rows) {
        let Some(geom) = &row.geom else { continue };

        let pts = positions(geom);
        let (center, radius) = bounds_sphere(&pts);

        let mut scene = Scene::new();
        scene.background = Color::new(0.09, 0.10, 0.13);
        scene.add_light(AmbientLight::new(Color::WHITE, 0.4));
        scene.add_light(
            DirectionalLight::new(Color::WHITE, 2.6)
                .with_direction(Vector3::new(-0.5, -0.8, -0.4).normalize()),
        );
        let mut mat = StandardMaterial::new(Color::new(0.26, 0.64, 0.96));
        mat.metalness = 0.15;
        mat.roughness = 0.45;
        scene.add(Object3D::mesh(Mesh::new(geom.clone(), mat.into())));

        // Frame the object: look at its center from an iso-ish direction.
        let dist = (radius / (22.0f32.to_radians() * 0.5).tan()).max(radius * 2.2);
        let eye = Vector3::new(
            center.x + dist * 0.62,
            center.y - dist * 0.62,
            center.z + dist * 0.5,
        );
        let mut cam = PerspectiveCamera::new(45.0, w as f32 / h as f32, 0.05, dist * 8.0 + 100.0);
        cam.position = eye;
        cam.look_at(center);

        let rgba = hr.render_to_rgba(&mut scene, &cam);
        let path = dir.join(format!("{}.png", demo.name));
        write_png(path.to_str().unwrap(), rw, rh, &rgba).map_err(|e| e.to_string())?;
        n += 1;
    }
    Ok(n)
}

fn bounds_sphere(pts: &[[f32; 3]]) -> (threers::Vector3, f32) {
    use threers::Vector3;
    if pts.is_empty() {
        return (Vector3::ZERO, 1.0);
    }
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for p in pts {
        for k in 0..3 {
            mn[k] = mn[k].min(p[k]);
            mx[k] = mx[k].max(p[k]);
        }
    }
    let c = Vector3::new(
        (mn[0] + mx[0]) / 2.0,
        (mn[1] + mx[1]) / 2.0,
        (mn[2] + mx[2]) / 2.0,
    );
    let mut r2 = 0.0f32;
    for p in pts {
        let d = [p[0] - c.x, p[1] - c.y, p[2] - c.z];
        r2 = r2.max(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    }
    (c, r2.sqrt().max(1e-3))
}

// --- Minimal PNG writer (RGBA8, stored/zlib — no external deps) ---
fn write_png(path: &str, w: u32, h: u32, rgba: &[u8]) -> std::io::Result<()> {
    let mut raw = Vec::with_capacity((w * h * 4 + h) as usize);
    for y in 0..h as usize {
        raw.push(0);
        let row = &rgba[y * w as usize * 4..(y + 1) * w as usize * 4];
        raw.extend_from_slice(row);
    }
    let mut png = Vec::new();
    png.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
    chunk(&mut png, b"IHDR", &ihdr);
    chunk(&mut png, b"IDAT", &zlib_store(&raw));
    chunk(&mut png, b"IEND", &[]);
    std::fs::write(path, png)
}
fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc_in = kind.to_vec();
    crc_in.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_in).to_be_bytes());
}
fn zlib_store(data: &[u8]) -> Vec<u8> {
    let mut z = vec![0x78, 0x01];
    for (i, block) in data.chunks(65535).enumerate() {
        let last = (i + 1) * 65535 >= data.len();
        z.push(if last { 1 } else { 0 });
        let len = block.len() as u16;
        z.extend_from_slice(&len.to_le_bytes());
        z.extend_from_slice(&(!len).to_le_bytes());
        z.extend_from_slice(block);
    }
    z.extend_from_slice(&adler32(data).to_be_bytes());
    z
}
fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}
fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}
