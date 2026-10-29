//! `threers::softbody` against the JavaScript it was ported from.
//!
//! `tests/support/softbody/gen-fixture.mjs` runs the original solver
//! (Axolotl Jelly's `soft-sim.js` and `soft-lattice.js`, kept beside it) on a
//! small creature — an ellipsoid body, two capsule arms, two chains — and
//! records what it builds and what it computes over 90 frames: idle, a hand
//! pulling one arm until it tears, letting go, a startle. Here the same
//! fields go through the Rust port and the record is compared.
//!
//! The lattice, glue and landmarks must match exactly. The particles are
//! held to the original bit for bit as well: the port computes in f64 and
//! stores to f32 where the JavaScript stores into a `Float32Array`, uses
//! fdlibm for `Math.sin`/`exp`/`pow` and V8's `Math.hypot`, and draws the
//! goo's randomness from the same seeded generator.
//!
//! Regenerate the fixture after changing the reference:
//! `node tests/support/softbody/gen-fixture.mjs`.

use std::collections::HashMap;

use threers::softbody::{Bounds, ChainSpec, Creature, Event, Field, Marks};

struct Fixture {
    arrays: HashMap<String, Vec<f64>>,
    events: Vec<(usize, String)>,
}

fn load() -> Fixture {
    let text = include_str!("support/softbody/fixture.txt");
    let mut arrays = HashMap::new();
    let mut events = Vec::new();
    for line in text.lines() {
        let mut it = line.split(' ');
        let name = it.next().unwrap().to_string();
        if name == "event" {
            let frame = it.next().unwrap().parse().unwrap();
            events.push((frame, it.filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" ")));
            continue;
        }
        let n: usize = it.next().unwrap().parse().unwrap();
        let toks: Vec<&str> = it.filter(|s| !s.is_empty()).collect();
        // a trailing `f32`: the values are the hex of f32 bits
        let v: Vec<f64> = if toks.last() == Some(&"f32") {
            toks[..toks.len() - 1].iter().map(|s| f32::from_bits(u32::from_str_radix(s, 16).unwrap()) as f64).collect()
        } else {
            toks.iter().map(|s| s.parse().unwrap()).collect()
        };
        assert_eq!(v.len(), n, "{name}");
        arrays.insert(name, v);
    }
    Fixture { arrays, events }
}

impl Fixture {
    fn get(&self, k: &str) -> &[f64] {
        self.arrays.get(k).unwrap_or_else(|| panic!("fixture has no {k}"))
    }
    fn f32s(&self, k: &str) -> Vec<f32> {
        self.get(k).iter().map(|&v| v as f32).collect()
    }
    fn bounds(&self, k: &str) -> Bounds {
        let b = self.get(k);
        Bounds { min: [b[0], b[1], b[2]], max: [b[3], b[4], b[5]] }
    }
}

// The fields, as gen-fixture.mjs writes them.
fn body(x: f64, y: f64, z: f64) -> f64 {
    let (xx, yy, zz) = (x / 0.35, (y - 0.17) / 0.16, z / 0.3);
    ((xx * xx + yy * yy + zz * zz).sqrt() - 1.0) * 0.16
}
// `t < 0 ? 0 : t > 1 ? 1 : t`, as the JavaScript writes it: `clamp` would
// treat NaN differently, and the point is the same arithmetic.
#[allow(clippy::manual_clamp)]
fn capsule(x: f64, y: f64, z: f64, a: [f64; 3], b: [f64; 3], r: f64) -> f64 {
    let (bx, by, bz) = (b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    let (qx, qy, qz) = (x - a[0], y - a[1], z - a[2]);
    let mut t = (qx * bx + qy * by + qz * bz) / (bx * bx + by * by + bz * bz);
    t = if t < 0.0 { 0.0 } else if t > 1.0 { 1.0 } else { t };
    let (dx, dy, dz) = (qx - bx * t, qy - by * t, qz - bz * t);
    (dx * dx + dy * dy + dz * dz).sqrt() - r
}
const ARMS: [([f64; 3], [f64; 3], f64); 2] =
    [([0.2, 0.12, 0.05], [0.72, 0.07, 0.16], 0.065), ([-0.2, 0.12, -0.05], [-0.7, 0.07, -0.2], 0.06)];
fn arm(k: usize, x: f64, y: f64, z: f64) -> f64 {
    let (a, b, r) = ARMS[k];
    capsule(x, y, z, a, b, r).max(-(body(x, y, z) + 0.03))
}

fn creature(fx: &Fixture) -> Creature {
    let arm0 = |x, y, z| arm(0, x, y, z);
    let arm1 = |x, y, z| arm(1, x, y, z);
    let chains = [
        ChainSpec { points: vec![[0.1, 0.3, 0.1], [0.12, 0.38, 0.14], [0.14, 0.45, 0.18], [0.16, 0.5, 0.22]], r0: 0.02, r1: 0.008, stiff: 0.2 },
        ChainSpec { points: vec![[-0.1, 0.3, -0.1], [-0.12, 0.38, -0.14], [-0.14, 0.45, -0.18]], r0: 0.02, r1: 0.008, stiff: 0.16 },
    ];
    let marks = Marks { head: [[0.0, 0.3, 0.0], [0.2, 0.25, 0.0], [0.0, 0.25, 0.2]], span: [[0.33, 0.17, 0.0], [-0.33, 0.17, 0.0]] };
    Creature::from_fields(
        Field { f: &body, bounds: fx.bounds("body.bounds"), h: 0.08, skin: fx.f32s("body.skin") },
        vec![
            ("arm 0".into(), Field { f: &arm0, bounds: fx.bounds("arm0.bounds"), h: 0.045, skin: fx.f32s("arm0.skin") }),
            ("arm 1".into(), Field { f: &arm1, bounds: fx.bounds("arm1.bounds"), h: 0.045, skin: fx.f32s("arm1.skin") }),
        ],
        &chains,
        marks,
    )
    .expect("the arms reach the body")
}

fn same<T: Copy + Into<f64>>(what: &str, ours: &[T], theirs: &[f64]) {
    assert_eq!(ours.len(), theirs.len(), "{what}: length");
    for (i, (&a, &b)) in ours.iter().zip(theirs).enumerate() {
        let a: f64 = a.into();
        assert!(a == b || (a.is_nan() && b.is_nan()), "{what}[{i}]: {a} against the original's {b}");
    }
}

#[test]
fn the_lattice_and_the_glue_are_the_originals() {
    let fx = load();
    let c = creature(&fx);
    for (tag, p) in [("body", &c.body), ("arm0", &c.limbs[0].part), ("arm1", &c.limbs[1].part)] {
        same(&format!("{tag}.nodes"), &p.lattice.nodes, fx.get(&format!("{tag}.nodes")));
        same(&format!("{tag}.tets"), &p.lattice.tets, fx.get(&format!("{tag}.tets")));
        same(&format!("{tag}.edges"), &p.edges, fx.get(&format!("{tag}.edges")));
        same(&format!("{tag}.restVol"), &p.rest_vol, fx.get(&format!("{tag}.restVol")));
        same(&format!("{tag}.dmInv"), &p.dm_inv, fx.get(&format!("{tag}.dmInv")));
        same(&format!("{tag}.boundary"), &p.boundary, fx.get(&format!("{tag}.boundary")));
        same(&format!("{tag}.radius"), &p.radius, fx.get(&format!("{tag}.radius")));
        same(&format!("{tag}.tetOf"), &p.skin.tet_of, fx.get(&format!("{tag}.tetOf")));
        same(&format!("{tag}.bary"), &p.skin.bary, fx.get(&format!("{tag}.bary")));
    }
    for (k, l) in c.limbs.iter().enumerate() {
        same(&format!("arm{k}.glueLeg"), &l.glue_part, fx.get(&format!("arm{k}.glueLeg")));
        same(&format!("arm{k}.glueBody"), &l.glue_body, fx.get(&format!("arm{k}.glueBody")));
        same(&format!("arm{k}.pivot"), &l.pivot, fx.get(&format!("arm{k}.pivot")));
        same(&format!("arm{k}.anchor"), &[l.anchor as f64], fx.get(&format!("arm{k}.anchor")));
    }
    same("chains.rest", &c.chains.rest, fx.get("chains.rest"));
    same("chains.stiff", &c.chains.stiff, fx.get("chains.stiff"));
    same("chains.radius", &c.chains.radius, fx.get("chains.radius"));
    same("chains.segRadius", &c.chains.seg_radius, fx.get("chains.segRadius"));
    same("chains.tetOf", &c.chains.root_emb.tet_of, fx.get("chains.tetOf"));
    same("chains.bary", &c.chains.root_emb.bary, fx.get("chains.bary"));
    let lm = c.landmarks;
    same("landmarks", &[lm.head[0], lm.head[1], lm.head[2], lm.snout, lm.tail].map(|v| v as f64), fx.get("landmarks"));
}

fn event_line(e: &Event) -> String {
    match e {
        Event::Tear { leg } => format!("tear {leg}"),
        Event::Regrown { leg } => format!("regrown {leg}"),
        Event::Tick { speed } => format!("tick {speed}"),
        Event::Plop { r, speed } => format!("plop {r} {speed}"),
        Event::Plip { r, speed } => format!("plip {r} {speed}"),
    }
}

#[test]
fn the_particles_are_the_originals_through_a_grab_and_a_tear() {
    let fx = load();
    let mut sim = creature(&fx).into_soft_body(7);
    sim.walls = Some([1.0, 0.0, 0.0, 0.9, -1.0, 0.0, 0.0, 0.9, 0.0, 0.0, 1.0, 0.8, 0.0, 0.0, -1.0, 0.8]);
    let tip = ARMS[0].1;
    let mut events = Vec::new();
    for frame in 1..=90usize {
        if frame == 11 {
            let slot = sim.leg_slots[0][0];
            sim.start_grab(tip, slot);
            sim.grab.as_mut().unwrap().axis = [0.3, -0.5, -0.8];
        }
        if frame > 11 {
            if let Some(g) = sim.grab.as_mut() {
                let k = (frame - 11) as f64;
                g.target = [tip[0] + 0.03 * k, tip[1] + 0.01 * k, tip[2] + 0.015 * k];
                g.angle = 0.02 * k;
            }
        }
        if frame == 45 {
            sim.grab = None;
        }
        if frame == 50 {
            sim.flick();
        }
        sim.step(1.0 / 60.0);
        events.extend(sim.events.drain(..).map(|e| (frame, event_line(&e))));
        if [1, 10, 20, 30, 60, 90].contains(&frame) {
            same(&format!("f{frame}.x"), &sim.x, fx.get(&format!("f{frame}.x")));
            same(&format!("f{frame}.gx"), &sim.gx, fx.get(&format!("f{frame}.gx")));
            same(&format!("f{frame}.dp"), &sim.dp[..3 * sim.d_count], fx.get(&format!("f{frame}.dp")));
            if frame == 20 || frame == 90 {
                same(&format!("f{frame}.nodeF"), &sim.node_f, fx.get(&format!("f{frame}.nodeF")));
            }
            let blocks: Vec<f64> =
                sim.blocks.iter().flat_map(|b| [b.active as u8 as f64, b.scale, b.grow, b.twitch, b.strain]).collect();
            same(&format!("f{frame}.blocks"), &blocks, fx.get(&format!("f{frame}.blocks")));
            let m = sim.measure();
            same(&format!("f{frame}.measure"), &[m.volume, m.length, m.attached as f64, m.loose as f64], fx.get(&format!("f{frame}.measure")));
        }
    }
    let theirs: Vec<(usize, String)> = fx.events.clone();
    assert_eq!(events.len(), theirs.len(), "events: ours {events:?}\ntheirs {theirs:?}");
    for (a, b) in events.iter().zip(&theirs) {
        assert_eq!(a, b);
    }
    let d = {
        let d = [0.05f64, -1.0, -0.2];
        let l = threers::softbody::js_math::hypot(&d);
        d.map(|v| v / l)
    };
    let hit = sim.raycast([0.0, 1.5, 0.3], d);
    let ours = match hit {
        Some(threers::softbody::Hit::Part { block, t, .. }) => vec![0.0, block as f64, t],
        Some(threers::softbody::Hit::Drop { index, t, .. }) => vec![1.0, index as f64, t],
        None => vec![-1.0],
    };
    same("raycast", &ours, fx.get("raycast"));
}

/// Not a check: how long a step takes here, beside the JavaScript's
/// (`node tests/support/softbody/gen-fixture.mjs` prints nothing of the sort;
/// run with `--ignored --nocapture`).
#[test]
#[ignore]
fn step_time() {
    let fx = load();
    let mut sim = creature(&fx).into_soft_body(7);
    let t = std::time::Instant::now();
    for _ in 0..300 {
        sim.step(1.0 / 60.0);
    }
    eprintln!("{:.3} ms a step over {} particles", t.elapsed().as_secs_f64() * 1000.0 / 300.0, sim.n);
}
