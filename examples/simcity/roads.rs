//! Part of the `simcity` example; see `mod.rs`.
//!
//! The road network outside the street grid.
//!
//! Everything this generator put in the countryside — the power station, the
//! container terminal, the retail parks, the stadium, the courts, the farms —
//! arrived with no way of getting to it. A car park in a field with no road to
//! it is not a car park, and once you have noticed it you cannot stop noticing
//! it: the buildings stop looking like buildings and start looking like models
//! set down on a mat.
//!
//! So there is a hierarchy here, and it is the hierarchy real places have.
//! Streets carry local traffic. A **ring** takes the traffic that wants to get
//! past rather than in. **Lanes** run out from the ring across the fields to
//! the outlying works. **Spurs** connect individual sites to whichever of
//! those passes nearest. And a **highway** crosses the lot in a cutting, with
//! slip roads at the one junction, because a bypass with no way on or off it
//! is scenery rather than infrastructure.
#![allow(dead_code)]

use super::*;

/// Surface height of a country road. Above the countryside's parcels (0.05)
/// and below the city's kerbs, so it beds into the fields rather than floating.
pub(crate) const RURAL_Y: f32 = 0.09;

const LANE_W: f32 = 7.0;
const RING_W: f32 = 11.0;

/// A road, as a centreline plus what it is for.
#[derive(Clone)]
pub(crate) struct Route {
    pub(crate) pts: Vec<(f32, f32)>,
    pub(crate) width: f32,
}

impl Route {
    /// Nearest point on the centreline, and how far away it is.
    pub(crate) fn nearest(&self, x: f32, z: f32) -> ((f32, f32), f32) {
        let mut best = ((0.0, 0.0), f32::MAX);
        for w in self.pts.windows(2) {
            let (a, c) = (w[0], w[1]);
            let (dx, dz) = (c.0 - a.0, c.1 - a.1);
            let l2 = dx * dx + dz * dz;
            let t = if l2 > 1e-6 {
                (((x - a.0) * dx + (z - a.1) * dz) / l2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let p = (a.0 + dx * t, a.1 + dz * t);
            let d = (p.0 - x).hypot(p.1 - z);
            if d < best.1 {
                best = (p, d);
            }
        }
        best
    }
}

/// Lay a carriageway with verges, and claim the corridor.
///
/// The verge is what stops a country road reading as a strip of tape laid on
/// grass: a real one has a metre of scuffed ground either side of it before
/// the field starts.
pub(crate) fn pave(b: &mut Batches, pts: &[(f32, f32)], width: f32, centre_line: bool, rng: &mut Rng) {
    b.grass.add_ground_path(
        pts,
        width + 4.0,
        RURAL_Y - 0.02,
        scale_color(Color::from_hex(0x7d8a4e), rng.range(0.9, 1.1)),
        Uv::Unit,
    );
    b.road.add_ground_path(
        pts,
        width,
        RURAL_Y,
        scale_color(Color::from_hex(0x43434a), rng.range(0.94, 1.06)),
        Uv::Unit,
    );
    if centre_line {
        // Dashes, drawn segment by segment so they follow the bends.
        for w in pts.windows(2) {
            let (a, c) = (w[0], w[1]);
            let len = (c.0 - a.0).hypot(c.1 - a.1);
            let n = (len / 9.0).round().max(1.0) as usize;
            for k in 0..n {
                let (t0, t1) = (
                    (k as f32 + 0.15) / n as f32,
                    (k as f32 + 0.65) / n as f32,
                );
                b.paint.add_ground_path(
                    &[
                        (mix(a.0, c.0, t0), mix(a.1, c.1, t0)),
                        (mix(a.0, c.0, t1), mix(a.1, c.1, t1)),
                    ],
                    0.24,
                    RURAL_Y + 0.02,
                    Color::from_hex(0xd8d4c4),
                    Uv::Unit,
                );
            }
        }
    }
    // Claim a corridor so the countryside and the fringe both keep off it.
    //
    // Walked in short steps rather than one box per segment. `Occupancy` takes
    // axis-aligned rectangles, so the bounding box of a single long diagonal
    // is a vast square of ground — a six-kilometre lane in nine segments would
    // claim, and so erase, most of the countryside it runs through.
    let h = width * 0.5 + 2.0;
    for w in pts.windows(2) {
        let (a, c) = (w[0], w[1]);
        let len = (c.0 - a.0).hypot(c.1 - a.1);
        let steps = (len / 24.0).ceil().max(1.0) as usize;
        for k in 0..steps {
            let (t0, t1) = (k as f32 / steps as f32, (k + 1) as f32 / steps as f32);
            let (x0, z0) = (mix(a.0, c.0, t0), mix(a.1, c.1, t0));
            let (x1, z1) = (mix(a.0, c.0, t1), mix(a.1, c.1, t1));
            b.occ.claim([
                x0.min(x1) - h,
                z0.min(z1) - h,
                x0.max(x1) + h,
                z0.max(z1) + h,
            ]);
        }
    }
}

/// The sloping sides of an embankment carrying a road segment.
///
/// Drawn as two battered quads rather than as a box per segment. A box from
/// ground level to the deck at every step gives a staircase — which is exactly
/// what the first version of the slip roads looked like, a flight of green
/// stairs climbing out of the fields.
fn embankment(
    b: &mut Batches,
    p0: (f32, f32),
    p1: (f32, f32),
    hw: f32,
    y0: f32,
    y1: f32,
    c: Color,
) {
    if y0.max(y1) < 0.35 {
        return;
    }
    let (dx, dz) = (p1.0 - p0.0, p1.1 - p0.1);
    let l = dx.hypot(dz).max(0.001);
    let (nx, nz) = (-dz / l, dx / l);
    // Roughly one in one and a half, which is what an earth batter stands at.
    let toe = |y: f32| hw + y * 1.5;
    for s in [-1.0f32, 1.0] {
        let a = [p0.0 + nx * hw * s, y0, p0.1 + nz * hw * s];
        let d = [p1.0 + nx * hw * s, y1, p1.1 + nz * hw * s];
        let e = [p1.0 + nx * toe(y1) * s, 0.0, p1.1 + nz * toe(y1) * s];
        let f = [p0.0 + nx * toe(y0) * s, 0.0, p0.1 + nz * toe(y0) * s];
        let quad = if s > 0.0 { [a, d, e, f] } else { [f, e, d, a] };
        b.grass.quad(quad, [nx * s, 0.6, nz * s], Uv::Unit, c);
    }
}

/// The ring road: a rounded rectangle outside the built area.
///
/// Rounded, not square, and that is the point — the corners are the only large
/// curves in the whole plan, and they are what stops the outskirts reading as
/// more grid. Wide, with a centre line and lighting.
pub(crate) fn add_ring(b: &mut Batches, hx: f32, hz: f32, rng: &mut Rng) -> Route {
    let r = (hx.min(hz) * 0.34).min(90.0);
    let mut pts: Vec<(f32, f32)> = Vec::new();
    // Four corners, each an arc, joined by the straights between them.
    let corners = [
        (hx - r, hz - r, 0.0f32),
        (-hx + r, hz - r, PI * 0.5),
        (-hx + r, -hz + r, PI),
        (hx - r, -hz + r, PI * 1.5),
    ];
    for (cx, cz, a0) in corners {
        for k in 0..=6 {
            let a = a0 + k as f32 / 6.0 * PI * 0.5;
            pts.push((cx + r * a.cos(), cz + r * a.sin()));
        }
    }
    pts.push(pts[0]);
    pave(b, &pts, RING_W, true, rng);
    // A segregated cycle track outside the verge, following the ring all the
    // way round. Out here it is a track rather than a painted lane, because
    // there is room for one and nothing to give way to.
    let track: Vec<(f32, f32)> = pts
        .iter()
        .map(|p| {
            let l = p.0.hypot(p.1).max(1e-4);
            let d = RING_W * 0.5 + 3.4;
            (p.0 + p.0 / l * d, p.1 + p.1 / l * d)
        })
        .collect();
    b.paint.add_ground_path(
        &track,
        2.4,
        RURAL_Y + 0.01,
        scale_color(Color::from_hex(0x2f6b4f), rng.range(0.94, 1.06)),
        Uv::Unit,
    );
    // Lighting down the outside of it.
    for (i, p) in pts.iter().enumerate() {
        if i % 3 != 0 {
            continue;
        }
        let (nx, nz) = (p.0, p.1);
        let l = nx.hypot(nz).max(0.001);
        let (px, pz) = (p.0 + nx / l * (RING_W * 0.5 + 1.4), p.1 + nz / l * (RING_W * 0.5 + 1.4));
        b.trim.add_limb(
            Vector3::new(px, RURAL_Y, pz),
            Vector3::new(px, 9.0, pz),
            0.16,
            0.11,
            5,
            Color::from_hex(0x4a4f55),
            false,
        );
        b.glow.add_box(
            Vector3::new(px - 0.6, 8.7, pz - 0.4),
            Vector3::new(px + 0.6, 9.0, pz + 0.4),
            Color::from_hex(0xffeec4),
            Uv::Unit,
        );
    }
    Route { pts, width: RING_W }
}

/// A country lane running outward from the ring.
///
/// Lanes wander. A dead straight line across a field is a Roman road or a
/// runway, and neither is what this is: the bends come from a low-frequency
/// wobble whose amplitude grows with distance from the ring, because the
/// further from town the less anybody straightened it.
pub(crate) fn add_lane(
    b: &mut Batches,
    from: (f32, f32),
    bearing: f32,
    len: f32,
    rng: &mut Rng,
) -> Route {
    let (sa, ca) = bearing.sin_cos();
    // One vertex per hundred metres or so. Nine was fine for a four-hundred
    // metre lane and would have made a six-kilometre one a polygon.
    let segs = ((len / 105.0).round() as usize).clamp(8, 64);
    let phase = rng.range(0.0, TAU);
    let swing = rng.range(0.05, 0.16);
    let pts: Vec<(f32, f32)> = (0..=segs)
        .map(|i| {
            let t = i as f32 / segs as f32;
            let d = len * t;
            // Wander scaled to a fixed wavelength rather than to the whole
            // run, so a long lane bends repeatedly instead of once.
            let off = (d / 420.0 + phase).sin() * len.min(700.0) * swing;
            (from.0 + ca * d - sa * off, from.1 + sa * d + ca * off)
        })
        .collect();
    pave(b, &pts, LANE_W, false, rng);
    Route { pts, width: LANE_W }
}

/// Connect a site to the nearest road.
///
/// An L rather than a straight line: access roads meet the road they leave at
/// something close to a right angle, and run square to whatever they serve.
pub(crate) fn add_spur(
    b: &mut Batches,
    site: (f32, f32),
    routes: &[Route],
    rng: &mut Rng,
) -> bool {
    let mut best: Option<((f32, f32), f32)> = None;
    for r in routes {
        let (p, d) = r.nearest(site.0, site.1);
        if best.map(|(_, bd)| d < bd).unwrap_or(true) {
            best = Some((p, d));
        }
    }
    let Some((join, dist)) = best else {
        return false;
    };
    if dist < 6.0 {
        // Already on the road.
        return true;
    }
    // Turn out of the road along whichever axis the site is further away on,
    // then run in square to it.
    let (dx, dz) = (site.0 - join.0, site.1 - join.1);
    let elbow = if dx.abs() > dz.abs() {
        (site.0, join.1)
    } else {
        (join.0, site.1)
    };
    pave(b, &[join, elbow, site], 6.0, false, rng);
    true
}

/// A dual carriageway in a shallow cutting, with one junction.
///
/// The cutting is doing the work that would otherwise need a viaduct: it puts
/// the highway below everything else, so a lane can cross it on a flat
/// overbridge and the slip roads have somewhere to climb to. It is also what a
/// bypass built through farmland usually is.
pub(crate) struct Highway {
    pub(crate) along_x: bool,
    pub(crate) fixed: f32,
    pub(crate) lo: f32,
    pub(crate) hi: f32,
    /// Deck height. Negative: the carriageways are below grade.
    pub(crate) deck: f32,
    /// Centre offsets of the two carriageways from `fixed`.
    pub(crate) carriageways: [f32; 2],
    /// Cross-offset of every running lane, and which way it runs.
    pub(crate) lanes: Vec<(f32, f32)>,
}

pub(crate) fn add_highway(
    b: &mut Batches,
    along_x: bool,
    fixed: f32,
    lo: f32,
    hi: f32,
    junction: f32,
    cross: Option<&Route>,
    lanes: usize,
    rng: &mut Rng,
) -> Highway {
    // At grade, not in a cutting. A cutting is what a bypass through farmland
    // usually is and it was the first thing I built — but the terrain here is
    // a handful of very large slabs and nothing cuts them, so three and a half
    // metres down put the whole carriageway underneath the fields. Grade
    // separation comes from lifting the crossing lane instead.
    // At grade, not in a cutting. A cutting is what a bypass through farmland
    // usually is and it was the first thing I built — but the terrain here is
    // a handful of very large slabs and nothing cuts them, so three and a half
    // metres down put the whole carriageway underneath the fields. Grade
    // separation comes from lifting the crossing lane instead.
    let deck = RURAL_Y;
    // A real motorway cross-section, built from the lane out rather than from
    // one made-up carriageway width. The first version was a single 11.5 m
    // ribbon with two dashes painted on it at guessed offsets, which is a road
    // that looks like a motorway from a long way up and like nothing at all
    // from anywhere else.
    const LANE_W: f32 = 3.65;
    const SHOULDER: f32 = 3.3;
    let lanes = lanes.clamp(2, 5);
    let cw = lanes as f32 * LANE_W + SHOULDER;
    let median = 6.0f32;
    let off = median * 0.5 + cw * 0.5;
    let batter = 9.0f32;

    // Where a point on the highway is, in world terms.
    let at = |s: f32, t: f32| -> (f32, f32) {
        if along_x {
            (s, fixed + t)
        } else {
            (fixed + t, s)
        }
    };
    // A ribbon along the highway between two cross-offsets at a fixed height.
    let ribbon = |b: &mut Batches, t0: f32, t1: f32, y: f32, c: Color, batch: u8| {
        let (a0, a1) = at(lo, t0);
        let (c0, c1) = at(hi, t1);
        let (x0, z0) = (a0.min(c0), a1.min(c1));
        let (x1, z1) = (a0.max(c0), a1.max(c1));
        let mb = match batch {
            0 => &mut b.road,
            1 => &mut b.grass,
            _ => &mut b.paint,
        };
        mb.add_slab(x0, z0, x1, z1, y, c, Uv::Unit);
    };
    // A painted line, solid or dashed, at a cross-offset.
    let line = |b: &mut Batches, t: f32, w: f32, dashed: bool, c: Color| {
        if !dashed {
            let (a0, a1) = at(lo, t - w * 0.5);
            let (c0, c1) = at(hi, t + w * 0.5);
            b.paint.add_slab(
                a0.min(c0),
                a1.min(c1),
                a0.max(c0),
                a1.max(c1),
                deck + 0.02,
                c,
                Uv::Unit,
            );
            return;
        }
        // Nine metres of paint, three of gap, which is close enough to the
        // real thing that the eye reads speed off it.
        let n = ((hi - lo) / 12.0).round().max(1.0) as usize;
        for k in 0..n {
            let s0 = mix(lo, hi, k as f32 / n as f32);
            let s1 = s0 + 9.0;
            let (a0, a1) = at(s0, t - w * 0.5);
            let (c0, c1) = at(s1, t + w * 0.5);
            b.paint.add_slab(
                a0.min(c0),
                a1.min(c1),
                a0.max(c0),
                a1.max(c1),
                deck + 0.02,
                c,
                Uv::Unit,
            );
        }
    };

    let white = Color::from_hex(0xdedacb);
    for s in [-1.0f32, 1.0] {
        // Verge outside the hard shoulder.
        let outer = s * (off + cw * 0.5 + batter);
        let inner = s * (off + cw * 0.5);
        ribbon(
            b,
            inner.min(outer),
            inner.max(outer),
            deck - 0.02,
            scale_color(Color::from_hex(0x74814a), rng.range(0.92, 1.08)),
            1,
        );
        // The carriageway: `lanes` running lanes plus a hard shoulder, laid
        // from the median outwards.
        let m0 = s * median * 0.5;
        let m1 = s * (median * 0.5 + cw);
        ribbon(
            b,
            m0.min(m1),
            m0.max(m1),
            deck,
            scale_color(Color::from_hex(0x3f4046), rng.range(0.96, 1.04)),
            0,
        );
        // Median edge line, solid.
        line(b, s * (median * 0.5 + 0.25), 0.20, false, white);
        // Lane divisions, dashed.
        for k in 1..lanes {
            line(
                b,
                s * (median * 0.5 + k as f32 * LANE_W),
                0.16,
                true,
                white,
            );
        }
        // Hard shoulder edge line, solid and wider — the one line on a
        // motorway that is not the same as the others.
        line(
            b,
            s * (median * 0.5 + lanes as f32 * LANE_W),
            0.30,
            false,
            white,
        );
    }
    // Median, with a barrier down it.
    ribbon(b, -median * 0.5, median * 0.5, deck + 0.05, Color::from_hex(0x6f7a44), 1);
    let n = ((hi - lo) / 6.0).round().max(2.0) as usize;
    for k in 0..=n {
        let s = mix(lo, hi, k as f32 / n as f32);
        let (px, pz) = at(s, 0.0);
        b.trim.add_box(
            Vector3::new(px - 0.12, deck + 0.05, pz - 0.12),
            Vector3::new(px + 0.12, deck + 0.85, pz + 0.12),
            Color::from_hex(0x9aa0a6),
            Uv::Unit,
        );
    }
    // Lighting down the median.
    for k in (0..=n).step_by(8) {
        let s = mix(lo, hi, k as f32 / n as f32);
        let (px, pz) = at(s, 0.0);
        b.trim.add_limb(
            Vector3::new(px, deck + 0.85, pz),
            Vector3::new(px, deck + 13.0, pz),
            0.2,
            0.13,
            5,
            Color::from_hex(0x4a4f55),
            false,
        );
        b.glow.add_box(
            Vector3::new(px - 2.2, deck + 12.6, pz - 0.4),
            Vector3::new(px + 2.2, deck + 13.0, pz + 0.4),
            Color::from_hex(0xffeec4),
            Uv::Unit,
        );
    }

    // --- The junction: a half-diamond. The lane crosses on an overbridge and
    // two slip roads leave the highway and climb to meet it — which is the
    // whole point of a bypass, and without them it is scenery.
    let span = off + cw * 0.5 + batter;
    if let Some(route) = cross {
        let _ = route;
    }
    let br = 7.0f32; // half-width of the crossing lane
    let bh = 6.6f32; // deck height over the highway
    let ramp = 46.0f32; // how far back the approach embankment starts

    // Approach embankments, one each side, stepped so they read as a slope.
    for side in [-1.0f32, 1.0] {
        let steps = 6;
        for k in 0..steps {
            let (u0, u1) = (k as f32 / steps as f32, (k + 1) as f32 / steps as f32);
            let d0 = span + 6.0 + ramp * (1.0 - u0);
            let d1 = span + 6.0 + ramp * (1.0 - u1);
            let (y0, y1) = (bh * u0, bh * u1);
            let (p0x, p0z) = at(junction - br, side * d0);
            let (p1x, p1z) = at(junction + br, side * d1);
            // Carriageway, and the earth it stands on.
            let (c0, c1) = (at(junction, side * d0), at(junction, side * d1));
            embankment(
                b,
                c0,
                c1,
                br,
                y0,
                y1,
                scale_color(Color::from_hex(0x6f8f4a), rng.range(0.92, 1.08)),
            );
            b.road.quad(
                [
                    [p0x.min(p1x), y0, p0z.min(p1z)],
                    [p0x.min(p1x), y1, p0z.max(p1z)],
                    [p0x.max(p1x), y1, p0z.max(p1z)],
                    [p0x.max(p1x), y0, p0z.min(p1z)],
                ],
                [0.0, 1.0, 0.0],
                Uv::Unit,
                Color::from_hex(0x43434a),
            );
        }
    }
    // --- The overbridge.
    //
    // The first version was a flat slab on two thin sticks, which is not a
    // bridge — it is a plank. What makes a road bridge read is the *soffit*:
    // the edge beam you see from below, the girders behind it, and the pier
    // cap those girders land on. None of that is expensive and all of it is
    // what your eye checks.
    let deck_t = 1.35f32; // structural depth of the deck
    let (bx0, bz0) = at(junction - br, -(span + 6.0));
    let (bx1, bz1) = at(junction + br, span + 6.0);
    let (dx0, dz0) = (bx0.min(bx1), bz0.min(bz1));
    let (dx1, dz1) = (bx0.max(bx1), bz0.max(bz1));
    // Running surface.
    b.road.add_slab(dx0, dz0, dx1, dz1, bh, Color::from_hex(0x43434a), Uv::Unit);
    // Edge beams down both sides — the deep fascia that gives a bridge its
    // line.
    for s2 in [-1.0f32, 1.0] {
        let (e0x, e0z) = at(junction + s2 * br - 0.5, -(span + 6.0));
        let (e1x, e1z) = at(junction + s2 * br, span + 6.0);
        b.trim.add_box(
            Vector3::new(e0x.min(e1x), bh - deck_t, e0z.min(e1z)),
            Vector3::new(e0x.max(e1x), bh, e0z.max(e1z)),
            Color::from_hex(0xa8a49c),
            Uv::Unit,
        );
    }
    // Girders under the deck, set back from the fascia so they read as
    // structure rather than as a thicker slab.
    for k in 0..4 {
        let t = mix(-br + 1.6, br - 1.6, k as f32 / 3.0);
        let (g0x, g0z) = at(junction + t - 0.28, -(span + 5.0));
        let (g1x, g1z) = at(junction + t + 0.28, span + 5.0);
        b.trim.add_box(
            Vector3::new(g0x.min(g1x), bh - deck_t + 0.15, g0z.min(g1z)),
            Vector3::new(g0x.max(g1x), bh - 0.1, g0z.max(g1z)),
            Color::from_hex(0x8f8b83),
            Uv::Unit,
        );
    }
    // Footways either side of the carriageway, raised, with railings. A
    // crossing with no way to walk over it is a crossing half the population
    // cannot use.
    for s2 in [-1.0f32, 1.0] {
        let (f0x, f0z) = at(junction + s2 * (br - 1.9), -(span + 6.0));
        let (f1x, f1z) = at(junction + s2 * br, span + 6.0);
        b.pads.add_box(
            Vector3::new(f0x.min(f1x), bh, f0z.min(f1z)),
            Vector3::new(f0x.max(f1x), bh + 0.16, f0z.max(f1z)),
            Color::from_hex(0xb4b0a6),
            Uv::Unit,
        );
        // Parapet: posts and two rails, not a solid wall — you can see through
        // a parapet and that is most of why a bridge looks light.
        let n = 26;
        for i in 0..=n {
            let u = i as f32 / n as f32;
            let (px, pz) = at(junction + s2 * (br - 0.35), mix(-(span + 6.0), span + 6.0, u));
            b.trim.add_limb(
                Vector3::new(px, bh + 0.16, pz),
                Vector3::new(px, bh + 1.35, pz),
                0.055,
                0.05,
                4,
                Color::from_hex(0x9aa0a6),
                false,
            );
        }
        for rail in [0.75f32, 1.28] {
            let (r0x, r0z) = at(junction + s2 * (br - 0.42), -(span + 6.0));
            let (r1x, r1z) = at(junction + s2 * (br - 0.28), span + 6.0);
            b.trim.add_box(
                Vector3::new(r0x.min(r1x), bh + rail, r0z.min(r1z)),
                Vector3::new(r0x.max(r1x), bh + rail + 0.09, r0z.max(r1z)),
                Color::from_hex(0x9aa0a6),
                Uv::Unit,
            );
        }
    }
    // Piers in the median, with a cap the girders land on.
    {
        let (px, pz) = at(junction, 0.0);
        for s2 in [-1.0f32, 1.0] {
            let (cx2, cz2) = at(junction + s2 * (br - 2.2), 0.0);
            b.trim.add_limb(
                Vector3::new(cx2, deck + 0.05, cz2),
                Vector3::new(cx2, bh - deck_t - 0.55, cz2),
                1.05,
                0.85,
                8,
                Color::from_hex(0x9a968e),
                false,
            );
        }
        // Pier cap: a beam across the top of both columns.
        let (c0x, c0z) = at(junction - br + 0.6, -1.5);
        let (c1x, c1z) = at(junction + br - 0.6, 1.5);
        b.trim.add_box(
            Vector3::new(c0x.min(c1x), bh - deck_t - 0.55, c0z.min(c1z)),
            Vector3::new(c0x.max(c1x), bh - deck_t + 0.02, c0z.max(c1z)),
            Color::from_hex(0x9a968e),
            Uv::Unit,
        );
        let _ = (px, pz);
    }
    // Abutments with wing walls, splayed back into the embankment.
    for s2 in [-1.0f32, 1.0] {
        let face = s2 * (span + 6.0);
        let (a0x, a0z) = at(junction - br, face);
        let (a1x, a1z) = at(junction + br, face + s2 * 1.6);
        b.trim.add_box(
            Vector3::new(a0x.min(a1x), 0.0, a0z.min(a1z)),
            Vector3::new(a0x.max(a1x), bh, a0z.max(a1z)),
            Color::from_hex(0x9a968e),
            Uv::Unit,
        );
        for s3 in [-1.0f32, 1.0] {
            let (w0x, w0z) = at(junction + s3 * br, face);
            let (w1x, w1z) = at(junction + s3 * (br + 3.4), face + s2 * 5.0);
            b.trim.add_box(
                Vector3::new(w0x.min(w1x), 0.0, w0z.min(w1z)),
                Vector3::new(w0x.max(w1x), bh * 0.72, w0z.max(w1z)),
                Color::from_hex(0x928e86),
                Uv::Unit,
            );
        }
    }

    // Two slip roads, one off each carriageway. Each is a quadratic curve that
    // leaves the hard shoulder heading along the highway, swings out past the
    // toe of the embankment, and ends running *parallel to the lane* beside
    // it — which is what makes it a junction. The first version diverged in a
    // straight line and stopped in a field: recognisably a slip road, joined
    // to nothing.
    for cs in [-1.0f32, 1.0] {
        let dirn = cs;
        // Start on the hard shoulder, control point further along the
        // highway, end alongside the lane part-way up its embankment.
        let p0 = (junction + dirn * (br + 6.0), cs * (off + cw * 0.5));
        let p1 = (junction + dirn * 78.0, cs * (off + cw * 0.5));
        let end_d = span + 6.0 + ramp * 0.45;
        let p2 = (junction + dirn * (br + 5.0), cs * end_d);
        // Height follows the lane's own ramp, so the two meet level.
        let y_end = bh * (1.0 - (end_d - span - 6.0) / ramp);
        let steps = 12;
        for k in 0..steps {
            let bez = |u: f32| {
                let v = 1.0 - u;
                (
                    v * v * p0.0 + 2.0 * v * u * p1.0 + u * u * p2.0,
                    v * v * p0.1 + 2.0 * v * u * p1.1 + u * u * p2.1,
                )
            };
            let (u0, u1) = (k as f32 / steps as f32, (k + 1) as f32 / steps as f32);
            let (a, c) = (bez(u0), bez(u1));
            // Flat for the first third, then climbing: a slip road runs level
            // while it is still beside the carriageway.
            let climb = |u: f32| ((u - 0.3) / 0.7).clamp(0.0, 1.0).powf(1.4);
            let (y0, y1) = (mix(deck, y_end, climb(u0)), mix(deck, y_end, climb(u1)));
            let (ax, az) = at(a.0, a.1);
            let (bx2, bz2) = at(c.0, c.1);
            let hwid = 3.8f32;
            let (dx, dz) = (bx2 - ax, bz2 - az);
            let l = dx.hypot(dz).max(0.001);
            let (nx, nz) = (-dz / l * hwid, dx / l * hwid);
            b.road.quad(
                [
                    [ax - nx, y0, az - nz],
                    [bx2 - nx, y1, bz2 - nz],
                    [bx2 + nx, y1, bz2 + nz],
                    [ax + nx, y0, az + nz],
                ],
                [0.0, 1.0, 0.0],
                Uv::Unit,
                Color::from_hex(0x43434a),
            );
            embankment(
                b,
                (ax, az),
                (bx2, bz2),
                hwid,
                y0,
                y1,
                scale_color(Color::from_hex(0x6f8f4a), rng.range(0.92, 1.08)),
            );
        }
        // The give-way stub joining it to the lane.
        let (jx, jz) = at(p2.0, p2.1);
        let (lx, lz) = at(junction, cs * end_d);
        b.road.quad(
            [
                [jx, y_end, jz],
                [lx, y_end, lz],
                [lx, y_end, lz + 0.01],
                [jx, y_end, jz + 0.01],
            ],
            [0.0, 1.0, 0.0],
            Uv::Unit,
            Color::from_hex(0x43434a),
        );
    }
    // Claim the whole corridor.
    let (c0x, c0z) = at(lo, -span);
    let (c1x, c1z) = at(hi, span);
    b.occ.claim([
        c0x.min(c1x),
        c0z.min(c1z),
        c0x.max(c1x),
        c0z.max(c1z),
    ]);
    // One entry per running lane: where it sits across the road, and which
    // way traffic on it goes. Drive on the left, so the carriageway at
    // negative offset runs one way and the other runs back.
    let mut lane_list = Vec::with_capacity(lanes * 2);
    for s in [-1.0f32, 1.0] {
        for k in 0..lanes {
            let t = s * (median * 0.5 + (k as f32 + 0.5) * LANE_W);
            lane_list.push((t, s));
        }
    }
    Highway {
        along_x,
        fixed,
        lo,
        hi,
        deck,
        carriageways: [-off, off],
        lanes: lane_list,
    }
}

/// A green bridge: a wildlife crossing over the motorway.
///
/// Wide, planted, and with earth banks along both edges so an animal on it
/// cannot see the traffic below — which is the entire design principle, and
/// the reason these are built forty or fifty metres across instead of the six
/// a footbridge needs. Fencing funnels along the verge lead onto it.
///
/// It is also the single most legible thing you can put over a motorway: a
/// strip of woodland crossing six lanes reads instantly, from any distance.
pub(crate) fn add_green_bridge(
    b: &mut Batches,
    along_x: bool,
    fixed: f32,
    at_s: f32,
    span: f32,
    deck: f32,
    rng: &mut Rng,
) {
    let bh = 7.2f32;
    let half = 22.0f32;
    let at = |s: f32, t: f32| -> (f32, f32) {
        if along_x {
            (s, fixed + t)
        } else {
            (fixed + t, s)
        }
    };
    // Approach ramps, one each side, climbing on earth banks.
    for side in [-1.0f32, 1.0] {
        let steps = 7;
        let ramp = 64.0f32;
        for k in 0..steps {
            let (u0, u1) = (k as f32 / steps as f32, (k + 1) as f32 / steps as f32);
            let d0 = span + 4.0 + ramp * (1.0 - u0);
            let d1 = span + 4.0 + ramp * (1.0 - u1);
            let (y0, y1) = (bh * u0, bh * u1);
            let (p0, p1) = (at(at_s, side * d0), at(at_s, side * d1));
            // Grass over the top of the bank.
            let (q0x, q0z) = at(at_s - half, side * d0);
            let (q1x, q1z) = at(at_s + half, side * d1);
            b.grass.quad(
                [
                    [q0x.min(q1x), y0, q0z.min(q1z)],
                    [q0x.min(q1x), y1, q0z.max(q1z)],
                    [q0x.max(q1x), y1, q0z.max(q1z)],
                    [q0x.max(q1x), y0, q0z.min(q1z)],
                ],
                [0.0, 1.0, 0.0],
                Uv::Unit,
                scale_color(Color::from_hex(0x63963f), rng.range(0.9, 1.1)),
            );
            // Battered sides.
            let (dx, dz) = (p1.0 - p0.0, p1.1 - p0.1);
            let l = dx.hypot(dz).max(0.001);
            let (nx, nz) = (-dz / l, dx / l);
            for sg in [-1.0f32, 1.0] {
                let toe = |y: f32| half + y * 1.5;
                let a = [p0.0 + nx * half * sg, y0, p0.1 + nz * half * sg];
                let d = [p1.0 + nx * half * sg, y1, p1.1 + nz * half * sg];
                let e = [p1.0 + nx * toe(y1) * sg, 0.0, p1.1 + nz * toe(y1) * sg];
                let f = [p0.0 + nx * toe(y0) * sg, 0.0, p0.1 + nz * toe(y0) * sg];
                let q = if sg > 0.0 { [a, d, e, f] } else { [f, e, d, a] };
                b.grass.quad(
                    q,
                    [nx * sg, 0.6, nz * sg],
                    Uv::Unit,
                    scale_color(Color::from_hex(0x6f8f4a), rng.range(0.92, 1.08)),
                );
            }
        }
    }
    // The span itself: a planted deck between two solid earth screens.
    let (s0x, s0z) = at(at_s - half, -(span + 4.0));
    let (s1x, s1z) = at(at_s + half, span + 4.0);
    let (mnx, mxx) = (s0x.min(s1x), s0x.max(s1x));
    let (mnz, mxz) = (s0z.min(s1z), s0z.max(s1z));
    // Soffit, so it is a structure and not a floating lawn.
    b.trim.add_box(
        Vector3::new(mnx, bh - 1.6, mnz),
        Vector3::new(mxx, bh - 0.25, mxz),
        Color::from_hex(0xa09c94),
        Uv::Unit,
    );
    b.grass.add_slab(
        mnx,
        mnz,
        mxx,
        mxz,
        bh,
        scale_color(Color::from_hex(0x63963f), rng.range(0.95, 1.05)),
        Uv::Unit,
    );
    // The screening banks along both edges: the thing that makes it work.
    for s2 in [-1.0f32, 1.0] {
        let (e0x, e0z) = at(at_s + s2 * (half - 4.5), -(span + 4.0));
        let (e1x, e1z) = at(at_s + s2 * half, span + 4.0);
        b.grass.add_box(
            Vector3::new(e0x.min(e1x), bh, e0z.min(e1z)),
            Vector3::new(e0x.max(e1x), bh + 2.4, e0z.max(e1z)),
            scale_color(Color::from_hex(0x5c8a3c), rng.range(0.9, 1.1)),
            Uv::Unit,
        );
    }
    // Planting: scrub down the middle and trees along the banks, continuous
    // with the woodland either side.
    for _ in 0..18 {
        let u = rng.range(-1.0, 1.0);
        let v = rng.range(-1.0, 1.0);
        let (tx, tz) = at(at_s + u * (half - 7.0), v * (span + 2.0));
        if rng.chance(0.55) {
            add_tree_as(
                b,
                tx,
                tz,
                bh,
                false,
                if rng.chance(0.5) { Species::Conifer } else { Species::Broadleaf },
                None,
                rng.range(0.45, 0.8),
                rng,
            );
        } else {
            let r = rng.range(0.9, 2.0);
            b.foliage.add_blob(
                Vector3::new(tx, bh + r * 0.5, tz),
                r,
                r * 0.6,
                r,
                3,
                6,
                0.3,
                rng.next_u32() as i32 & 0xffff,
                0.5,
                scale_color(Color::from_hex(0x3f5a2b), rng.range(0.85, 1.15)),
            );
        }
    }
    // Funnel fencing along the verge, leading animals to the crossing.
    for side in [-1.0f32, 1.0] {
        for s2 in [-1.0f32, 1.0] {
            let n = 20;
            for k in 0..n {
                let u = k as f32 / n as f32;
                let s3 = at_s + s2 * (half + 6.0 + u * 150.0);
                let (px, pz) = at(s3, side * (span - 1.0));
                b.trim.add_limb(
                    Vector3::new(px, deck, pz),
                    Vector3::new(px, deck + 2.2, pz),
                    0.05,
                    0.045,
                    4,
                    Color::from_hex(0x6f747a),
                    false,
                );
            }
        }
    }
}

/// A pedestrian and cycle bridge over the motorway.
///
/// Slender: a shallow deck on a single central pier, with a mesh parapet and
/// switchback ramps at both ends rather than steps, because a footbridge with
/// steps is one nobody with a bike or a pushchair can use.
pub(crate) fn add_foot_bridge(
    b: &mut Batches,
    along_x: bool,
    fixed: f32,
    at_s: f32,
    span: f32,
    deck: f32,
    rng: &mut Rng,
) {
    let bh = 7.6f32;
    let half = 1.9f32;
    let at = |s: f32, t: f32| -> (f32, f32) {
        if along_x {
            (s, fixed + t)
        } else {
            (fixed + t, s)
        }
    };
    // The span.
    let (s0x, s0z) = at(at_s - half, -(span + 4.0));
    let (s1x, s1z) = at(at_s + half, span + 4.0);
    b.pads.add_box(
        Vector3::new(s0x.min(s1x), bh - 0.55, s0z.min(s1z)),
        Vector3::new(s0x.max(s1x), bh, s0z.max(s1z)),
        Color::from_hex(0xb4b0a6),
        Uv::Unit,
    );
    // Mesh parapets: posts and a top rail, tall, because a footbridge over a
    // motorway is fenced to head height and then some.
    for s2 in [-1.0f32, 1.0] {
        let n = 30;
        for k in 0..=n {
            let u = k as f32 / n as f32;
            let (px, pz) = at(at_s + s2 * half, mix(-(span + 4.0), span + 4.0, u));
            b.trim.add_limb(
                Vector3::new(px, bh, pz),
                Vector3::new(px, bh + 2.3, pz),
                0.05,
                0.045,
                4,
                Color::from_hex(0x7d8288),
                false,
            );
        }
        let (r0x, r0z) = at(at_s + s2 * half - 0.08, -(span + 4.0));
        let (r1x, r1z) = at(at_s + s2 * half + 0.08, span + 4.0);
        b.trim.add_box(
            Vector3::new(r0x.min(r1x), bh + 2.25, r0z.min(r1z)),
            Vector3::new(r0x.max(r1x), bh + 2.4, r0z.max(r1z)),
            Color::from_hex(0x7d8288),
            Uv::Unit,
        );
    }
    // One slim pier in the median.
    let (px, pz) = at(at_s, 0.0);
    b.trim.add_limb(
        Vector3::new(px, deck, pz),
        Vector3::new(px, bh - 0.55, pz),
        0.55,
        0.42,
        8,
        Color::from_hex(0xa09c94),
        false,
    );
    // Switchback ramps at both ends: two flights doubling back, on legs.
    for side in [-1.0f32, 1.0] {
        for leg in 0..2 {
            let steps = 6;
            for k in 0..steps {
                let (u0, u1) = (k as f32 / steps as f32, (k + 1) as f32 / steps as f32);
                // Each leg covers half the climb; the second doubles back.
                let (h0, h1) = (
                    bh * (leg as f32 + u0) * 0.5,
                    bh * (leg as f32 + u1) * 0.5,
                );
                let off = if leg == 0 { -6.0f32 } else { 6.0 };
                let d0 = span + 4.0 + if leg == 0 { 28.0 * u0 } else { 28.0 * (1.0 - u0) };
                let d1 = span + 4.0 + if leg == 0 { 28.0 * u1 } else { 28.0 * (1.0 - u1) };
                let (q0x, q0z) = at(at_s + off - half, side * d0);
                let (q1x, q1z) = at(at_s + off + half, side * d1);
                b.pads.quad(
                    [
                        [q0x.min(q1x), h0, q0z.min(q1z)],
                        [q0x.min(q1x), h1, q0z.max(q1z)],
                        [q0x.max(q1x), h1, q0z.max(q1z)],
                        [q0x.max(q1x), h0, q0z.min(q1z)],
                    ],
                    [0.0, 1.0, 0.0],
                    Uv::Unit,
                    Color::from_hex(0xb4b0a6),
                );
                // A leg under every other step.
                if k % 2 == 0 {
                    let (lx, lz) = at(at_s + off, side * d0);
                    b.trim.add_limb(
                        Vector3::new(lx, 0.0, lz),
                        Vector3::new(lx, h0, lz),
                        0.16,
                        0.13,
                        5,
                        Color::from_hex(0x7d8288),
                        false,
                    );
                }
            }
        }
    }
    let _ = rng;
}
