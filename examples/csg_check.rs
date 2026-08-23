//! Fingerprint the evaluated solids of a SCAD animation.
//!
//! ```text
//! cargo run --release --features openscad --example csg_check
//! ```
//!
//! A change to the triangulator may legitimately produce a *different* mesh of
//! the *same* solid — a face can be cut into triangles many ways. Comparing
//! renders or triangle lists flags those as regressions when they are not.
//! Volume, watertightness and Euler characteristic don't care how a face was
//! cut, so they are what to compare across such a change: the volume says the
//! solid is the same size, watertightness says no cracks were opened, and the
//! Euler characteristic says the topology (how many holes) is unchanged.
use threers::core::BufferGeometry;
use threers::exact_csg::metrics;

/// Triangles of a geometry, as `metrics` wants them.
fn triangles(g: &BufferGeometry) -> Vec<[[f64; 3]; 3]> {
    let pos = g.get_attribute("position").expect("position").array.clone();
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

/// Watertight = every edge is used by exactly two triangles. Welded on the same
/// 1e-4 grid `metrics` uses, so an f32 round-trip does not read as a crack.
fn is_closed_manifold(tris: &[[[f64; 3]; 3]]) -> bool {
    use std::collections::HashMap;
    let key = |p: [f64; 3]| {
        (
            (p[0] * 1e4).round() as i64,
            (p[1] * 1e4).round() as i64,
            (p[2] * 1e4).round() as i64,
        )
    };
    type PointKey = (i64, i64, i64);
    let mut edges: HashMap<(PointKey, PointKey), u32> = HashMap::new();
    for t in tris {
        for k in 0..3 {
            let (mut u, mut v) = (key(t[k]), key(t[(k + 1) % 3]));
            if u > v {
                std::mem::swap(&mut u, &mut v);
            }
            *edges.entry((u, v)).or_insert(0) += 1;
        }
    }
    !edges.is_empty() && edges.values().all(|&c| c == 2)
}
use threers::openscad::animate::ScadAnimation;

fn main() {
    // Sanity-check the checker itself before trusting what it says about the
    // model: a box is closed, and a box with a corner missing is not.
    {
        use threers::BoxGeometry;
        let b: threers::core::BufferGeometry = BoxGeometry::new(2.0, 3.0, 4.0);
        let t = triangles(&b);
        let v = metrics::volume(&t);
        println!(
            "self-test: box 2x3x4 -> volume {v:.4} (want 24), closed {} (want true), chi {} (want 2)",
            is_closed_manifold(&t),
            metrics::euler_characteristic(&t)
        );
        let mut cut = t.clone();
        cut.pop();
        println!(
            "self-test: same box minus one triangle -> closed {} (want false)",
            is_closed_manifold(&cut)
        );
    }
    let src = std::fs::read_to_string("examples/scad_animate.scad").unwrap();
    let frames: usize = std::env::args()
        .position(|a| a == "--frames")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let mut anim = ScadAnimation::from_source(&src).frames(frames);
    let evaluated = anim.evaluate().unwrap();

    let (mut vol, mut tris, mut open) = (0.0f64, 0usize, 0usize);
    for (fi, f) in evaluated.iter().enumerate() {
        for (pi, part) in f.parts.iter().enumerate() {
            let t = triangles(&part.geometry);
            let v = metrics::volume(&t);
            let closed = is_closed_manifold(&t);
            let chi = metrics::euler_characteristic(&t);
            if !closed {
                open += 1;
            }
            println!(
                "frame {fi} part {pi}: volume {v:>14.6}  tris {:>6}  closed {closed}  chi {chi}",
                t.len()
            );
            vol += v;
            tris += t.len();
        }
    }
    println!("TOTAL volume {vol:.6}  triangles {tris}  not-closed {open}");
}
