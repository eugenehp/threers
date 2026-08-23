//! Compare the two CSG kernels on a corpus of models.
//!
//! ```text
//! cargo run --release --features openscad          --example kernel_compare -- --out /tmp/a.txt
//! cargo run --release --features openscad,manifold --example kernel_compare -- --out /tmp/b.txt
//! diff /tmp/a.txt /tmp/b.txt
//! ```
//!
//! Two kernels will not agree triangle for triangle — a solid can be cut into
//! triangles many ways, and a better kernel produces fewer. So this reports what
//! does not depend on the tessellation: whether the result is watertight, what it
//! encloses, and what its topology is. Those must agree; triangle count is
//! printed alongside as information, not as a criterion.
//!
//! Run it before changing which kernel is the default, and before trusting the
//! `manifold` feature on your own models.
use std::time::Instant;
use threers::core::BufferGeometry;
use threers::exact_csg::metrics;
use threers::openscad::scad::parse_scad;

fn triangles(g: &BufferGeometry) -> Vec<[[f64; 3]; 3]> {
    let pos = match g.get_attribute("position") {
        Some(p) => p.array.clone(),
        None => return Vec::new(),
    };
    let idx: Vec<usize> = match &g.index {
        Some(i) => i.iter().map(|v| *v as usize).collect(),
        None => (0..pos.len() / 3).collect(),
    };
    idx.chunks_exact(3)
        .map(|c| {
            let v = |i: usize| {
                [
                    pos[i * 3] as f64,
                    pos[i * 3 + 1] as f64,
                    pos[i * 3 + 2] as f64,
                ]
            };
            [v(c[0]), v(c[1]), v(c[2])]
        })
        .collect()
}

fn watertight(tris: &[[[f64; 3]; 3]]) -> bool {
    use std::collections::HashMap;
    let key = |p: [f64; 3]| {
        (
            (p[0] * 1e4).round() as i64,
            (p[1] * 1e4).round() as i64,
            (p[2] * 1e4).round() as i64,
        )
    };
    let mut e: HashMap<_, u32> = HashMap::new();
    for t in tris {
        for k in 0..3 {
            let (mut u, mut v) = (key(t[k]), key(t[(k + 1) % 3]));
            if u > v {
                std::mem::swap(&mut u, &mut v);
            }
            *e.entry((u, v)).or_insert(0) += 1;
        }
    }
    !e.is_empty() && e.values().all(|&c| c == 2)
}

fn main() {
    let out = std::env::args()
        .position(|a| a == "--out")
        .and_then(|i| std::env::args().nth(i + 1));

    let kernel = if cfg!(feature = "manifold") {
        "manifold"
    } else {
        "in-house"
    };
    let mut report = String::new();
    report.push_str(&format!("kernel: {kernel}\n"));

    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir("tests/openscad-corpus")
        .map(|d| {
            d.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "scad"))
                .collect()
        })
        .unwrap_or_default();
    files.push("examples/scad_animate.scad".into());
    files.sort();

    let (mut closed, mut total) = (0usize, 0usize);
    for path in &files {
        let Ok(src) = std::fs::read_to_string(path) else {
            continue;
        };
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let t0 = Instant::now();
        let solid = match parse_scad(&src) {
            Ok(s) => s,
            Err(e) => {
                report.push_str(&format!("{name:<24} PARSE ERROR {e}\n"));
                continue;
            }
        };
        let g = solid.to_geometry_exact();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let tris = triangles(&g);
        let (vol, chi, wt) = (
            metrics::volume(&tris),
            metrics::euler_characteristic(&tris),
            watertight(&tris),
        );
        total += 1;
        if wt {
            closed += 1;
        }
        report.push_str(&format!(
            "{name:<24} closed {:<5} volume {:>14.4}  chi {:>5}   [tris {:>6}  {:>8.1} ms]\n",
            wt,
            vol,
            chi,
            tris.len(),
            ms
        ));
    }
    report.push_str(&format!("watertight: {closed}/{total}\n"));
    #[cfg(feature = "manifold")]
    report.push_str(&format!(
        "manifold results rejected by the watertightness gate: {}\n",
        threers::exact_csg::manifold_backend::rejected_results()
    ));

    print!("{report}");
    if let Some(p) = out {
        std::fs::write(&p, &report).expect("write report");
        eprintln!("wrote {p}");
    }
}
