//! The boolean measured by property rather than by case — stage A of
//! `docs/brep-boolean-v2-plan.md`.
//!
//! Random pairs, each operation, and one question asked of every answer: is
//! it right, declined, or wrong? A decline is allowed. A result that is not a
//! valid solid, or a valid solid of the wrong volume, is not.
//!
//! Boxes are placed on an integer grid, so faces and edges coincide far more
//! often than chance would have them: that is where the boolean breaks, and an
//! axis-aligned box pair has its volumes in closed form. Box and cylinder pairs
//! are checked by the identities between the three operations instead.
//!
//! `BREP_PROPERTY_CASES` sets how many pairs (default 24), `BREP_PROPERTY_SEED`
//! the seed, and `BREP_PROPERTY_VERBOSE` prints every case that is not right.
//! The counts are a baseline: the test fails only on a *wrong volume* from a
//! result that passed validity, which is the one outcome a caller cannot see.

#![cfg(feature = "brep-kernel")]

use threers::brep::{Body, BooleanOp};

const TOLERANCE: f64 = 1e-3;

/// xorshift64*, so a failing case is a seed and a number to replay.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn env(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Volume of a tessellation, or `None` when it does not close.
fn volume(body: &Body) -> Option<f64> {
    let mut b = body.clone();
    b.refine_edges(TOLERANCE);
    let (mesh, report) = b.tessellate(TOLERANCE);
    if !report.is_closed() || report.carried_through != 0 {
        return None;
    }
    let pos = &mesh.get_attribute("position")?.array;
    let idx = mesh.index.as_ref()?;
    let v = |i: u32| -> [f64; 3] {
        let o = i as usize * 3;
        [pos[o] as f64, pos[o + 1] as f64, pos[o + 2] as f64]
    };
    Some(
        idx.as_chunks::<3>()
            .0
            .iter()
            .map(|t| {
                let (a, b, c) = (v(t[0]), v(t[1]), v(t[2]));
                (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                    + a[2] * (b[0] * c[1] - b[1] * c[0]))
                    / 6.0
            })
            .sum::<f64>()
            .abs(),
    )
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Outcome {
    Right,
    Declined,
    /// Returned, but not a valid solid.
    Invalid,
    /// A valid solid of the wrong volume.
    Wrong,
}

/// What one operation gave: the outcome, and the volume when there is one.
fn run(a: &Body, b: &Body, op: BooleanOp, expected: Option<f64>) -> (Outcome, Option<f64>) {
    let Ok(r) = a.boolean(b, op, TOLERANCE) else {
        return (Outcome::Declined, None);
    };
    if !r.is_valid_solid(TOLERANCE) && !r.faces().is_empty() {
        return (Outcome::Invalid, None);
    }
    let got = if r.faces().is_empty() {
        Some(0.0)
    } else {
        volume(&r)
    };
    let Some(got) = got else {
        return (Outcome::Invalid, None);
    };
    match expected {
        Some(e) if (got - e).abs() > 1e-6 * e.max(1.0) => (Outcome::Wrong, Some(got)),
        _ => (Outcome::Right, Some(got)),
    }
}

#[derive(Default)]
struct Tally {
    right: usize,
    declined: usize,
    invalid: usize,
    wrong: usize,
}

impl Tally {
    fn add(&mut self, o: Outcome) {
        match o {
            Outcome::Right => self.right += 1,
            Outcome::Declined => self.declined += 1,
            Outcome::Invalid => self.invalid += 1,
            Outcome::Wrong => self.wrong += 1,
        }
    }
}

const OPS: [BooleanOp; 3] = [
    BooleanOp::Union,
    BooleanOp::Intersection,
    BooleanOp::Difference,
];

#[test]
fn boxes_on_a_grid() {
    let cases = env("BREP_PROPERTY_CASES", 24);
    let mut rng = Rng(env("BREP_PROPERTY_SEED", 0x9e37_79b9_7f4a_7c15) | 1);
    let verbose = std::env::var_os("BREP_PROPERTY_VERBOSE").is_some();
    let mut tally = Tally::default();
    let mut wrong = Vec::new();
    for case in 0..cases {
        let mut size = || [1 + rng.below(4), 1 + rng.below(4), 1 + rng.below(4)].map(|x| x as f64);
        let (sa, sb) = (size(), size());
        // Half-unit steps, so odd and even sizes both land faces on faces.
        let at = [rng.below(9), rng.below(9), rng.below(9)].map(|x| x as f64 * 0.5 - 2.0);
        // `cuboid` is centred on the origin; B's centre is moved to `at`.
        let lo = |s: [f64; 3], c: [f64; 3], k: usize| c[k] - s[k] / 2.0;
        let hi = |s: [f64; 3], c: [f64; 3], k: usize| c[k] + s[k] / 2.0;
        let o = [0.0; 3];
        // The two must meet: overlap or touch along every axis.
        if (0..3).any(|k| lo(sb, at, k) > hi(sa, o, k) || hi(sb, at, k) < lo(sa, o, k)) {
            continue;
        }
        let overlap: f64 = (0..3)
            .map(|k| (hi(sa, o, k).min(hi(sb, at, k)) - lo(sa, o, k).max(lo(sb, at, k))).max(0.0))
            .product();
        let (va, vb) = (sa.iter().product::<f64>(), sb.iter().product::<f64>());
        let a = Body::cuboid(sa);
        let Some(b) = Body::cuboid(sb).translated(at) else {
            continue;
        };
        for op in OPS {
            let expected = match op {
                BooleanOp::Union => va + vb - overlap,
                BooleanOp::Intersection => overlap,
                BooleanOp::Difference => va - overlap,
            };
            let (o, got) = run(&a, &b, op, Some(expected));
            tally.add(o);
            if o != Outcome::Right && verbose {
                eprintln!("case {case}: {sa:?} with {sb:?} at {at:?} {op:?}: {o:?} {got:?}, expected {expected}");
            }
            if o == Outcome::Wrong {
                wrong.push(format!("case {case}: {sa:?} with {sb:?} at {at:?} {op:?}: {got:?}, expected {expected}"));
            }
        }
    }
    eprintln!(
        "boxes on a grid: {} right, {} declined, {} invalid, {} wrong",
        tally.right, tally.declined, tally.invalid, tally.wrong
    );
    assert!(
        wrong.is_empty(),
        "valid results of the wrong volume:\n{}",
        wrong.join("\n")
    );
}

#[test]
fn a_box_and_a_cylinder_keep_the_identities() {
    let cases = env("BREP_PROPERTY_CASES", 24);
    let mut rng = Rng(env("BREP_PROPERTY_SEED", 0x2545_f491_4f6c_dd1d) | 1);
    let verbose = std::env::var_os("BREP_PROPERTY_VERBOSE").is_some();
    let mut tally = Tally::default();
    let mut broken = Vec::new();
    for case in 0..cases {
        let sa = [2 + rng.below(3), 2 + rng.below(3), 2 + rng.below(3)].map(|x| x as f64);
        // Half-unit steps: an axis on a face, a rim on a face, or neither.
        let base = [rng.below(9), rng.below(9), rng.below(7)].map(|x| x as f64 * 0.5 - 1.0);
        let radius = 0.5 * (1 + rng.below(3)) as f64;
        let height = (1 + rng.below(5)) as f64;
        let axis = match rng.below(3) {
            0 => [1.0, 0.0, 0.0],
            1 => [0.0, 1.0, 0.0],
            _ => [0.0, 0.0, 1.0],
        };
        let a = Body::cuboid(sa);
        let b = Body::cylinder(base, axis, radius, height);
        let results: Vec<(Outcome, Option<f64>)> =
            OPS.iter().map(|&op| run(&a, &b, op, None)).collect();
        for (op, (o, got)) in OPS.iter().zip(&results) {
            tally.add(*o);
            if *o != Outcome::Right && verbose {
                eprintln!("case {case}: {sa:?} with r {radius} h {height} at {base:?} along {axis:?} {op:?}: {o:?} {got:?}");
            }
        }
        // Where all three came back, they must agree with each other and
        // with the operands: V(A∪B) + V(A∩B) = V(A) + V(B), V(A−B) = V(A) − V(A∩B).
        if let [(Outcome::Right, Some(u)), (Outcome::Right, Some(i)), (Outcome::Right, Some(d))] =
            results[..]
        {
            let (Some(va), Some(vb)) = (volume(&a), volume(&b)) else {
                continue;
            };
            let scale = va.max(vb);
            let tol = 1e-2 * scale;
            if (u + i - va - vb).abs() > tol || (d - (va - i)).abs() > tol {
                broken.push(format!(
                    "case {case}: {sa:?} with r {radius} h {height} at {base:?} along {axis:?}: \
                     union {u}, common {i}, cut {d}, operands {va} and {vb}"
                ));
            }
        }
    }
    eprintln!(
        "box and cylinder: {} right, {} declined, {} invalid, {} wrong; {} identities broken",
        tally.right,
        tally.declined,
        tally.invalid,
        tally.wrong,
        broken.len()
    );
    assert!(
        broken.is_empty(),
        "results that disagree with each other:\n{}",
        broken.join("\n")
    );
}
