//! Road tunnels through the mountain ring.
//!
//! The lanes out of the city run to thirty half-widths and the mountains stand
//! at about sixteen, so every lane crosses the ring — and did so by driving
//! straight into a cone and disappearing. Nothing was wrong with the geometry:
//! the hillside is solid and opaque, so the buried road is correctly hidden.
//! It just read as a road that stops dead in a bank of earth, which on seed 3
//! alone happened seventeen times.
//!
//! # The hole that cannot be cut
//!
//! A tunnel wants a hole through the mountain, and there is no hole to be had.
//! The cones are ordinary solid geometry with no CSG behind them, so a bore
//! sunk into one is simply *inside* it, and the cone's near face hides the bore
//! exactly as effectively as it hides the road.
//!
//! So the portal projects. The headwall stands proud of the slope by a dozen
//! metres, clear of the cone's footprint, and the bore is a recess in that
//! block rather than a passage through the hill. Looking at it you see an arch
//! and darkness receding into it, which is all a tunnel mouth ever shows from
//! outside; the road between the two portals stays buried, which is what a
//! tunnel is. Portals that project from a slope are also just what gets built,
//! so nothing has been traded away to get round the limitation.
//!
//! # One tunnel per range, not one per cone
//!
//! A range is a chain of overlapping cones. Testing each cone separately puts
//! portals at every internal boundary — that is, in the middle of the
//! hillside, facing the inside of the next cone along. The runs of buried road
//! are merged along the lane before any portal is placed, so a lane crossing
//! six overlapping peaks gets one tunnel with two ends.

use std::f32::consts::PI;

use threers::{Color, Vector3};

use crate::batches::Batches;
use crate::country::Summit;
use crate::mesh::{MeshBuilder, Uv};
use crate::rng::Rng;
use crate::roads::Route;

/// Below this a "tunnel" is a road passing behind a hummock, and two portals
/// nose to nose look like a folly rather than a crossing.
const MIN_BORE: f32 = 70.0;
/// How far outside the hillside the mouth stands.
///
/// The mouth goes at the toe — where the cone meets the ground — and then a
/// little further out, so the headwall is clear of the slope rather than
/// sunk into it. Both other options were built first and both are wrong. Site
/// it by *distance* into the footprint and it lands in an open field, because
/// the cone has no height at its foot. Site it by depth of cover and it is
/// buried: the slope runs about 1.4 vertical to 1 horizontal, so a mouth six
/// metres inside the toe already has eight metres of hillside across the front
/// of it and only its coping shows.
const MOUTH_STANDOFF: f32 = 12.0;
/// Rock over the middle of the bore before this counts as a tunnel at all.
/// Less than this is a hummock the road should be going over.
const MIN_MID_COVER: f32 = 25.0;
/// Two buried stretches closer together than this are treated as one tunnel.
///
/// A range is a chain of cones and the road can clip out into daylight for a
/// few metres between two of them. That is a saddle, not the far end of a
/// tunnel and the near end of the next: four portals in forty metres, and — as
/// the mouths stand off from the toe by a dozen metres — some of them sited
/// inside the neighbouring hill. Measured on manhattan seed 29 an exit mouth
/// ended up under 12.6 m of the next cone along.
const MERGE_GAP: f32 = 46.0;
/// Clear width and height of the opening. A country lane is 7 m wide, and the
/// bore has to look like it could take a lorry.
const BORE_W: f32 = 9.0;
const BORE_H: f32 = 7.4;
/// Segments in the arch. Nine reads as a curve at the distance these are seen
/// from and costs 27 triangles a portal.
const ARCH_SEGS: usize = 9;

/// What one tunnel came out as. Reported so the siting can be tested: getting
/// a mouth into the right place took several goes, and both failures — a
/// portal standing in an open field, and a portal buried under the hillside —
/// are invisible from any camera that does not happen to look at that lane.
#[derive(Clone)]
pub(crate) struct Tunnel {
    /// Rock over each mouth. Must be zero: a mouth stands clear of the slope.
    pub(crate) mouth_cover: [f32; 2],
    /// Rock over the middle of the bore. This is what makes it a tunnel.
    pub(crate) mid_cover: f32,
    pub(crate) length: f32,
}

/// Drive every lane through the mountains it meets.
pub(crate) fn add_road_tunnels(
    b: &mut Batches,
    routes: &[Route],
    summits: &[Summit],
    rng: &mut Rng,
) -> Vec<Tunnel> {
    let mut built = Vec::new();
    for route in routes {
        for (enter, exit) in buried_runs(route, summits) {
            // Only where there is a mountain over the middle, not a mound.
            let mid = point(route, (enter + exit) * 0.5);
            if cover_at(summits, mid.0, mid.1) < MIN_MID_COVER {
                continue;
            }
            // Back each end out of the hill to the toe and a little beyond.
            let enter = mouth(route, summits, enter, -1.0);
            let exit = mouth(route, summits, exit, 1.0);
            if exit - enter < MIN_BORE {
                continue;
            }
            // Face each mouth along the road, outward: the entry looks back the
            // way the traffic came, the exit looks on.
            let into = heading(route, enter);
            let out = heading(route, exit);
            // Face each portal along the road, outward. The entry looks back
            // the way the traffic came, the exit looks on.

            let (a, c) = (point(route, enter), point(route, exit));
            portal(b, a, into + PI, rng);
            portal(b, c, out, rng);
            let m = point(route, (enter + exit) * 0.5);
            built.push(Tunnel {
                mouth_cover: [
                    cover_at(summits, a.0, a.1),
                    cover_at(summits, c.0, c.1),
                ],
                mid_cover: cover_at(summits, m.0, m.1),
                length: exit - enter,
            });
        }
    }
    built
}

/// Depth of rock over a point: the tallest cone covering it, zero if none.
fn cover_at(summits: &[Summit], x: f32, z: f32) -> f32 {
    summits
        .iter()
        .map(|m| m.cover_at(x, z))
        .fold(0.0f32, f32::max)
}

/// Arclength of the tunnel mouth at one end of a buried run.
///
/// `out_dir` is -1 at the entry and +1 at the exit: the direction that leads
/// out of the hill. Walks to the toe, where the cone's height falls to zero,
/// then stands off a little further so the headwall is in front of the slope
/// rather than under it.
fn mouth(route: &Route, summits: &[Summit], from: f32, out_dir: f32) -> f32 {
    let mut s = from;
    // Half a metre: the run was found on a six-metre sweep, so `from` can be
    // most of a sample inside the hill, and the slope turns every metre of
    // that into a metre and a half of headwall buried.
    for _ in 0..200 {
        let (x, z) = point(route, s);
        if cover_at(summits, x, z) <= 0.0 {
            break;
        }
        s += 0.5 * out_dir;
    }
    s += MOUTH_STANDOFF * out_dir;
    // And if the standoff itself walked into the next hill, keep going. Runs
    // this close are merged above, so this only fires at the far end of a
    // lane where the next cone is a separate crossing entirely.
    for _ in 0..400 {
        let (x, z) = point(route, s);
        if cover_at(summits, x, z) <= 0.0 {
            break;
        }
        s += 0.5 * out_dir;
    }
    s
}

/// Arclength positions `(enter, exit)` of each stretch of `route` that runs
/// inside the mountains, merged across overlapping cones.
fn buried_runs(route: &Route, summits: &[Summit]) -> Vec<(f32, f32)> {
    let total = length(route);
    if total <= 0.0 {
        return Vec::new();
    }
    // Six metres is finer than any cone this matters for and keeps a long
    // lane under a couple of thousand samples.
    let step = 6.0f32;
    let n = (total / step).ceil() as usize;
    let mut runs: Vec<(f32, f32)> = Vec::new();
    let mut open: Option<f32> = None;
    for i in 0..=n {
        let s = (i as f32 * step).min(total);
        let (x, z) = point(route, s);
        // Inside the cone's footprint. The full foot radius, not a fraction of
        // it: at the foot the cone has no height, so a portal placed there sits
        // against ground rising away from it rather than part way up a slope.
        let inside = cover_at(summits, x, z) > 0.0;
        match (inside, open) {
            (true, None) => open = Some(s),
            (false, Some(start)) => {
                if s - start >= MIN_BORE {
                    runs.push((start, s));
                }
                open = None;
            }
            _ => {}
        }
    }
    // A run still open when the lane ends never comes out, so it gets no
    // second mouth and is not a tunnel. `open` is dropped deliberately.
    let _ = open;
    // Join stretches separated by a saddle.
    let mut merged: Vec<(f32, f32)> = Vec::new();
    for r in runs {
        match merged.last_mut() {
            Some(prev) if r.0 - prev.1 < MERGE_GAP => prev.1 = r.1,
            _ => merged.push(r),
        }
    }
    merged
}

/// One tunnel mouth: a projecting portal block with an arched recess in it,
/// wing walls splaying back into the slope, and a coping across the top.
///
/// `bearing` points the way the opening faces — out of the hill.
fn portal(b: &mut Batches, at: (f32, f32), bearing: f32, rng: &mut Rng) {
    let (sa, ca) = bearing.sin_cos();
    // `f` is metres out along the facing direction, `t` across it, `y` up.
    let p = |f: f32, t: f32, y: f32| Vector3::new(at.0 + ca * f - sa * t, y, at.1 + sa * f + ca * t);
    let v = |f: f32, t: f32, y: f32| {
        let q = p(f, t, y);
        [q.x, q.y, q.z]
    };
    let out = [ca, 0.0, sa];
    let up = [0.0, 1.0, 0.0];
    // Across the opening, pointing to increasing `t`.
    let across = [-sa, 0.0, ca];

    let concrete = scale(Color::from_hex(0x8d8b84), rng.range(0.94, 1.06));
    let stain = scale(Color::from_hex(0x6f6d67), rng.range(0.94, 1.06));
    let coping = Color::from_hex(0x5f5d58);
    let inner = Color::from_hex(0x24262a);
    let dark = Color::from_hex(0x0d0f12);

    let bh = BORE_W * 0.5;
    // Springing line: vertical walls to here, arch above.
    let spring = BORE_H - bh;
    let hw = bh + 3.4;
    let top = BORE_H + 3.2;
    // The face leans back into the hill. A vertical slab of concrete against a
    // slope this steep reads as a wall someone left there; every retaining
    // structure of the size gets a batter.
    let batter = 1.8f32;
    let lean = |y: f32| -batter * (y / top);

    // --- Headwall: the two piers either side of the opening, and the spandrel
    // over it. Not one slab with a hole in it, which a triangle list cannot
    // express.
    for (t0, t1) in [(-hw, -bh), (bh, hw)] {
        panel(
            &mut b.trim,
            [
                v(lean(0.0), t0, 0.0),
                v(lean(0.0), t1, 0.0),
                v(lean(top), t1, top),
                v(lean(top), t0, top),
            ],
            out,
            concrete,
        );
    }
    // Spandrel: the wall above the arch, following its curve underneath.
    for k in 0..ARCH_SEGS {
        let a0 = PI * k as f32 / ARCH_SEGS as f32;
        let a1 = PI * (k + 1) as f32 / ARCH_SEGS as f32;
        let (t0, y0) = (-bh * a0.cos(), spring + bh * a0.sin());
        let (t1, y1) = (-bh * a1.cos(), spring + bh * a1.sin());
        panel(
            &mut b.trim,
            [
                v(lean(y0), t0, y0),
                v(lean(y1), t1, y1),
                v(lean(top), t1, top),
                v(lean(top), t0, top),
            ],
            out,
            concrete,
        );
    }

    // --- The bore. A recess, not a passage: see the note at the top of the
    // file.
    //
    // Its depth is tied to the standoff, and has to be. Anything deeper than
    // the block projects is behind the hillside, which occludes it perfectly
    // well and is a pale grey — so a bore driven past the slope shows the
    // mountain through the arch instead of darkness, and the mouth reads as a
    // hole punched in a wall with the sky behind it. That is what a 25 m bore
    // on a 4 m standoff looked like.
    let depth = MOUTH_STANDOFF - 1.5;
    for k in 0..ARCH_SEGS {
        let a0 = PI * k as f32 / ARCH_SEGS as f32;
        let a1 = PI * (k + 1) as f32 / ARCH_SEGS as f32;
        let (t0, y0) = (-bh * a0.cos(), spring + bh * a0.sin());
        let (t1, y1) = (-bh * a1.cos(), spring + bh * a1.sin());
        // Soffit normals point down and inwards, towards the bore's axis.
        let (mt, my) = ((t0 + t1) * 0.5, (y0 + y1) * 0.5 - spring);
        let ln = (mt * mt + my * my).sqrt().max(1e-3);
        let n = [
            across[0] * (-mt / ln),
            -my / ln,
            across[2] * (-mt / ln),
        ];
        panel(
            &mut b.trim,
            [
                v(0.0, t0, y0),
                v(0.0, t1, y1),
                v(-depth, t1, y1),
                v(-depth, t0, y0),
            ],
            n,
            inner,
        );
    }
    for side in [-1.0f32, 1.0] {
        panel(
            &mut b.trim,
            [
                v(0.0, side * bh, 0.0),
                v(0.0, side * bh, spring),
                v(-depth, side * bh, spring),
                v(-depth, side * bh, 0.0),
            ],
            [-across[0] * side, 0.0, -across[2] * side],
            inner,
        );
    }
    // Carriageway through the mouth, and the dark end wall.
    panel(
        &mut b.road,
        [
            v(0.0, -bh, 0.07),
            v(0.0, bh, 0.07),
            v(-depth, bh, 0.07),
            v(-depth, -bh, 0.07),
        ],
        up,
        Color::from_hex(0x2f3237),
    );
    panel(
        &mut b.trim,
        [
            v(-depth, -bh, 0.0),
            v(-depth, bh, 0.0),
            v(-depth, bh, BORE_H),
            v(-depth, -bh, BORE_H),
        ],
        out,
        dark,
    );

    // --- Coping across the top of the headwall, proud of the face.
    panel(
        &mut b.trim,
        [
            v(lean(top) - 0.6, -hw - 0.5, top),
            v(lean(top) - 0.6, hw + 0.5, top),
            v(1.2, hw + 0.5, top),
            v(1.2, -hw - 0.5, top),
        ],
        up,
        coping,
    );
    panel(
        &mut b.trim,
        [
            v(1.2, -hw - 0.5, top),
            v(1.2, hw + 0.5, top),
            v(1.2, hw + 0.5, top - 0.8),
            v(1.2, -hw - 0.5, top - 0.8),
        ],
        out,
        coping,
    );

    // --- Wing walls, running back into the slope and dropping as they go.
    // Without these the portal reads as a doorway standing in a field.
    for side in [-1.0f32, 1.0] {
        let (a, c) = (side * hw, side * (hw + 7.5));
        let (h0, h1) = (top - 1.4, 2.4);
        let n = [-across[0] * side, 0.0, -across[2] * side];
        panel(
            &mut b.trim,
            [
                v(0.0, a, 0.0),
                v(-16.0, c, 0.0),
                v(-16.0, c, h1),
                v(0.0, a, h0),
            ],
            n,
            stain,
        );
        panel(
            &mut b.trim,
            [
                v(0.0, a, h0),
                v(-16.0, c, h1),
                v(-16.0, c + side * 1.0, h1),
                v(0.0, a + side * 1.0, h0),
            ],
            up,
            coping,
        );
    }

    // --- A light either side of the mouth, on `glow`: that batch is hidden by
    // day, where `neon` is merely scaled to black and left drawn.
    for side in [-1.0f32, 1.0] {
        let t = side * (bh + 1.2);
        // The lens sits PROUD of its housing. Tucked inside it — which is
        // where the first version put it, a millimetre in on every face — the
        // housing encloses it completely and the lamp is invisible at every
        // hour, lit or not.
        b.trim.add_box(
            p(-1.0, t - 0.35, BORE_H - 0.95),
            p(0.15, t + 0.35, BORE_H - 0.25),
            Color::from_hex(0x46484c),
            Uv::Unit,
        );
        b.glow.add_box(
            p(0.1, t - 0.26, BORE_H - 0.85),
            p(0.55, t + 0.26, BORE_H - 0.35),
            Color::from_hex(0xffd9a0),
            Uv::Unit,
        );
    }
}

/// Emit a quad wound to match `n`, instead of trusting the caller to have got
/// it right.
///
/// Front faces are counter-clockwise and a quad wound the other way is simply
/// not drawn. A portal is a dozen flat panels at an arbitrary bearing, some
/// facing out of the hill and some facing into a bore, and hand-deriving the
/// winding for each is a coin toss per panel — the first version got the
/// headwall wrong and rendered as an arch floating over an empty field.
fn panel(m: &mut MeshBuilder, v: [[f32; 3]; 4], n: [f32; 3], c: Color) {
    let e1 = [v[1][0] - v[0][0], v[1][1] - v[0][1], v[1][2] - v[0][2]];
    let e2 = [v[2][0] - v[0][0], v[2][1] - v[0][1], v[2][2] - v[0][2]];
    let cr = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    if cr[0] * n[0] + cr[1] * n[1] + cr[2] * n[2] >= 0.0 {
        m.quad(v, n, Uv::Unit, c);
    } else {
        m.quad([v[3], v[2], v[1], v[0]], n, Uv::Unit, c);
    }
}

fn scale(c: Color, k: f32) -> Color {
    Color::new(
        (c.r * k).clamp(0.0, 1.0),
        (c.g * k).clamp(0.0, 1.0),
        (c.b * k).clamp(0.0, 1.0),
    )
}

fn length(route: &Route) -> f32 {
    route
        .pts
        .windows(2)
        .map(|w| (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1))
        .sum()
}

/// Position at arclength `s` along the route.
fn point(route: &Route, s: f32) -> (f32, f32) {
    let mut acc = 0.0;
    for w in route.pts.windows(2) {
        let seg = (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1);
        if acc + seg >= s && seg > 1e-4 {
            let t = (s - acc) / seg;
            return (w[0].0 + (w[1].0 - w[0].0) * t, w[0].1 + (w[1].1 - w[0].1) * t);
        }
        acc += seg;
    }
    *route.pts.last().unwrap_or(&(0.0, 0.0))
}

/// Direction of travel at arclength `s`, as a bearing.
fn heading(route: &Route, s: f32) -> f32 {
    let a = point(route, (s - 4.0).max(0.0));
    let b = point(route, s + 4.0);
    (b.1 - a.1).atan2(b.0 - a.0)
}
