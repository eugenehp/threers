//! Part of the `simcity` example; see `mod.rs`.
#![allow(dead_code)]

use super::*;

// ---------------------------------------------------------------------------
// Street grid.
// ---------------------------------------------------------------------------

/// Whether crossing number `road` is an avenue rather than a side street.
pub(crate) fn is_avenue(plan: &LayoutPlan, road: usize) -> bool {
    road.is_multiple_of(plan.avenue_every.max(1))
}

/// Where the water is.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Water {
    /// A channel down one block column, crossed by the avenues.
    River,
    /// A bay filling everything past one edge of the grid.
    Bay,
    /// Landlocked.
    Dry,
}

/// Which part of the city the tall buildings cluster around.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Core {
    /// A downtown in the middle.
    Centre,
    /// A downtown crowding the water.
    Waterfront,
    /// A ring of towers around a central park.
    Ring,
}

/// Where the green goes.
///
/// A probability per block gives you green confetti, which is what most of
/// these blocks used to be. Real cities put their parks somewhere on purpose:
/// one big one in the middle, a belt at a fixed radius, wedges running out
/// from the centre. The pattern is a property of the plan, not of the dice.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ParkPattern {
    /// Wherever the dice say, and nowhere else.
    Scatter,
    /// One large square of green in the middle, with the towers ringed round.
    Central,
    /// A continuous ring of green at a fixed distance from the middle.
    Belt,
    /// Four wedges radiating from the middle out past the edge.
    Wedges,
}

/// A named street pattern. The whole city — grid metrics, where the height
/// piles up, how finely blocks are platted, and where the water goes —
/// follows from this one choice.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Layout {
    /// Uniform grid, avenues every third crossing, a river through it and a
    /// downtown in the middle.
    Manhattan,
    /// Coarse grid, very wide avenues, big blocks and big footprints.
    Boulevard,
    /// Narrow lanes, small blocks, low-rise everywhere. No skyline at all,
    /// which is the point.
    OldTown,
    /// The water is a bay along one edge, and the towers crowd it.
    Waterfront,
    /// A large park in the middle with the towers ringed around it.
    Parkway,
    /// Low-rise on big blocks with four green wedges running out from the
    /// middle: the garden-city diagram, more or less.
    GardenCity,
    /// An ordinary grid interrupted by a continuous ring of parkland part way
    /// out, which the built-up area then resumes past.
    Greenbelt,
}

pub(crate) struct LayoutPlan {
    /// Every `avenue_every`-th crossing is an avenue.
    pub(crate) avenue_every: usize,
    pub(crate) avenue_w: f32,
    pub(crate) street_w: f32,
    pub(crate) block_min: f32,
    pub(crate) block_max: f32,
    /// Storeys in the tallest ordinary building downtown.
    pub(crate) peak_floors: f32,
    /// Zone value at which height has fallen to nothing. Larger spreads the
    /// tall buildings further out.
    pub(crate) core_reach: f32,
    pub(crate) park_base: f32,
    pub(crate) park_slope: f32,
    /// Target parcel area downtown and at the edge, in square metres.
    pub(crate) lot_core: f32,
    pub(crate) lot_edge: f32,
    pub(crate) water: Water,
    pub(crate) core: Core,
    pub(crate) parks: ParkPattern,
}

impl Layout {
    pub(crate) const ALL: [Layout; 7] = [
        Layout::Manhattan,
        Layout::Boulevard,
        Layout::OldTown,
        Layout::Waterfront,
        Layout::Parkway,
        Layout::GardenCity,
        Layout::Greenbelt,
    ];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Layout::Manhattan => "manhattan",
            Layout::Boulevard => "boulevard",
            Layout::OldTown => "oldtown",
            Layout::Waterfront => "waterfront",
            Layout::Parkway => "parkway",
            Layout::GardenCity => "gardencity",
            Layout::Greenbelt => "greenbelt",
        }
    }

    pub(crate) fn from_name(s: &str) -> Option<Self> {
        Layout::ALL.into_iter().find(|l| l.name() == s)
    }

    /// Cycles the layouts. Used by the interactive viewer's `l` key and by
    /// the contact sheet; the Metal example takes a layout and stops there.
    #[allow(dead_code)]
    pub(crate) fn next(self) -> Self {
        let i = Layout::ALL.iter().position(|l| *l == self).unwrap_or(0);
        Layout::ALL[(i + 1) % Layout::ALL.len()]
    }

    pub(crate) fn plan(self) -> LayoutPlan {
        match self {
            Layout::Manhattan => LayoutPlan {
                avenue_every: 3,
                avenue_w: 14.0,
                street_w: 9.5,
                block_min: 20.0,
                block_max: 33.0,
                peak_floors: 25.0,
                core_reach: 0.70,
                park_base: 0.05,
                park_slope: 0.12,
                lot_core: 430.0,
                lot_edge: 85.0,
                water: Water::River,
                core: Core::Centre,
                parks: ParkPattern::Scatter,
            },
            Layout::Boulevard => LayoutPlan {
                avenue_every: 2,
                avenue_w: 21.0,
                street_w: 11.0,
                block_min: 30.0,
                block_max: 47.0,
                peak_floors: 19.0,
                core_reach: 0.85,
                park_base: 0.07,
                park_slope: 0.14,
                lot_core: 760.0,
                lot_edge: 190.0,
                water: Water::River,
                core: Core::Centre,
                parks: ParkPattern::Scatter,
            },
            Layout::OldTown => LayoutPlan {
                avenue_every: 4,
                avenue_w: 11.0,
                street_w: 6.5,
                block_min: 14.0,
                block_max: 25.0,
                peak_floors: 5.5,
                core_reach: 1.10,
                park_base: 0.04,
                park_slope: 0.09,
                lot_core: 130.0,
                lot_edge: 45.0,
                water: Water::River,
                core: Core::Centre,
                parks: ParkPattern::Scatter,
            },
            Layout::Waterfront => LayoutPlan {
                avenue_every: 3,
                avenue_w: 15.0,
                street_w: 9.5,
                block_min: 20.0,
                block_max: 34.0,
                peak_floors: 28.0,
                core_reach: 0.62,
                park_base: 0.05,
                park_slope: 0.13,
                lot_core: 470.0,
                lot_edge: 95.0,
                water: Water::Bay,
                core: Core::Waterfront,
                parks: ParkPattern::Scatter,
            },
            Layout::Parkway => LayoutPlan {
                avenue_every: 3,
                avenue_w: 16.0,
                street_w: 10.0,
                block_min: 22.0,
                block_max: 35.0,
                peak_floors: 24.0,
                core_reach: 0.50,
                park_base: 0.05,
                park_slope: 0.10,
                lot_core: 430.0,
                lot_edge: 90.0,
                water: Water::Dry,
                core: Core::Ring,
                parks: ParkPattern::Central,
            },
            Layout::GardenCity => LayoutPlan {
                avenue_every: 2,
                avenue_w: 17.0,
                street_w: 10.5,
                block_min: 27.0,
                block_max: 42.0,
                peak_floors: 8.0,
                core_reach: 0.95,
                park_base: 0.09,
                park_slope: 0.15,
                lot_core: 620.0,
                lot_edge: 240.0,
                water: Water::River,
                core: Core::Centre,
                parks: ParkPattern::Wedges,
            },
            Layout::Greenbelt => LayoutPlan {
                avenue_every: 3,
                avenue_w: 14.0,
                street_w: 9.0,
                block_min: 21.0,
                block_max: 33.0,
                peak_floors: 21.0,
                core_reach: 0.58,
                park_base: 0.035,
                park_slope: 0.07,
                lot_core: 400.0,
                lot_edge: 95.0,
                water: Water::Dry,
                core: Core::Centre,
                parks: ParkPattern::Belt,
            },
        }
    }
}

/// Is this block designated green, and if so where does it sit in the design?
///
/// Returns `None` for an ordinary block, or `Some(t)` where `t` runs 0 at the
/// heart of the green area to 1 at its edge. That grading is what stops a
/// large park being the same design repeated across every block it covers:
/// water and rough grass in the middle, gardens and squares where it meets
/// the street.
pub(crate) fn park_role(
    plan: &LayoutPlan,
    cx: f32,
    cz: f32,
    extent: f32,
    park_r: f32,
) -> Option<f32> {
    let r = cx.hypot(cz);
    match plan.parks {
        ParkPattern::Scatter => None,
        ParkPattern::Central => (park_r > 0.0 && r < park_r).then(|| (r / park_r).min(1.0)),
        ParkPattern::Belt => {
            // A ring two blocks deep. `t` is 0 at its middle, 1 at either lip.
            let mid = extent * 0.56;
            let half = extent * 0.13;
            let d = (r - mid).abs();
            (d < half).then(|| (d / half).min(1.0))
        }
        ParkPattern::Wedges => {
            // Four wedges on the diagonals, kept clear of the middle so the
            // centre still reads as a town and not as a crossroads in a field.
            if r < extent * 0.20 {
                return None;
            }
            let a = cz.atan2(cx);
            let off = (0..4)
                .map(|k| {
                    let c = -PI * 0.75 + k as f32 * PI * 0.5;
                    let d = (a - c + PI).rem_euclid(TAU) - PI;
                    d.abs()
                })
                .fold(f32::INFINITY, f32::min);
            // The wedge narrows in angle as it runs out, so it stays a wedge
            // in area rather than fanning into a quadrant.
            let width = 0.30 * (extent * 0.30 / r.max(1.0)).clamp(0.55, 1.6);
            (off < width).then(|| (off / width).min(1.0))
        }
    }
}

/// How far a block sits from wherever this layout puts its tall buildings.
/// `0` is the middle of the core, `1` the edge of the city.
pub(crate) fn core_distance(
    plan: &LayoutPlan,
    cx: f32,
    cz: f32,
    extent: f32,
    shore: f32,
    land_span: f32,
    park_r: f32,
) -> f32 {
    match plan.core {
        Core::Centre => ((cx / extent).powi(2) + (cz / extent).powi(2)).sqrt(),
        Core::Waterfront => {
            // Inland from the quay, with a gentler pull along the shore so the
            // whole waterfront is not one uniform wall of towers.
            let inland = ((cx - shore).abs() / land_span.max(1.0)).clamp(0.0, 1.0);
            (inland * inland * 2.3 + (cz / extent).powi(2) * 0.5).sqrt()
        }
        Core::Ring => {
            let r = (cx * cx + cz * cz).sqrt();
            ((r - park_r * 1.22).abs() / extent * 2.6).min(1.4)
        }
    }
}

/// Alternating road / block spans along one axis: `road, block, road, ...`.
/// Index `k` is a road when `k` is even, a block when `k` is odd.
pub(crate) struct Axis {
    pub(crate) spans: Vec<(f32, f32)>,
}

impl Axis {
    pub(crate) fn build(
        blocks: usize,
        plan: &LayoutPlan,
        rng: &mut Rng,
        wide_block: Option<usize>,
    ) -> Self {
        let mut spans = Vec::with_capacity(2 * blocks + 1);
        let mut cursor = 0.0f32;
        for i in 0..=blocks {
            let w = if is_avenue(plan, i) {
                plan.avenue_w
            } else {
                plan.street_w
            };
            spans.push((cursor, cursor + w));
            cursor += w;
            if i < blocks {
                // Blocks grow towards the edge. A city with one block size
                // everywhere is the single strongest reason a generated grid
                // reads as a grid: downtown blocks are small because the land
                // is worth subdividing and the streets carry traffic, and
                // suburban blocks are two or three times the size because
                // nobody is walking across them and the roads inside are
                // somebody's driveway. It also makes room for the layouts that
                // are not grids at all — a close needs a block a turning head
                // fits into, and at the old sizes almost none did.
                let t = if blocks > 1 {
                    ((i as f32 + 0.5) / blocks as f32 * 2.0 - 1.0).abs()
                } else {
                    0.0
                };
                let grow = 1.0 + 1.15 * t.powf(1.6);
                let b = if wide_block == Some(i) {
                    38.0
                } else {
                    // Snap to the bay grid so building fronts land on it.
                    (rng.range(plan.block_min, plan.block_max) * grow / BAY).round() * BAY
                };
                spans.push((cursor, cursor + b));
                cursor += b;
            }
        }
        let shift = cursor * 0.5;
        for s in spans.iter_mut() {
            s.0 -= shift;
            s.1 -= shift;
        }
        Axis { spans }
    }

    /// Slide the whole axis. Used to re-centre a layout whose built area is
    /// off to one side, so the shadow camera and the default framing — both
    /// of which sit on the origin — still cover it.
    pub(crate) fn shift(&mut self, d: f32) {
        for s in self.spans.iter_mut() {
            s.0 += d;
            s.1 += d;
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.spans.len()
    }

    pub(crate) fn is_road(k: usize) -> bool {
        k.is_multiple_of(2)
    }

    pub(crate) fn span(&self, k: usize) -> (f32, f32) {
        self.spans[k]
    }

    pub(crate) fn center(&self, k: usize) -> f32 {
        let (a, b) = self.spans[k];
        (a + b) * 0.5
    }

    pub(crate) fn half(&self, k: usize) -> f32 {
        let (a, b) = self.spans[k];
        (b - a) * 0.5
    }

    pub(crate) fn min(&self) -> f32 {
        self.spans[0].0
    }

    pub(crate) fn max(&self) -> f32 {
        self.spans[self.spans.len() - 1].1
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Rect {
    pub(crate) x0: f32,
    pub(crate) z0: f32,
    pub(crate) x1: f32,
    pub(crate) z1: f32,
}

impl Rect {
    pub(crate) fn w(&self) -> f32 {
        self.x1 - self.x0
    }
    pub(crate) fn d(&self) -> f32 {
        self.z1 - self.z0
    }
    pub(crate) fn area(&self) -> f32 {
        self.w() * self.d()
    }
    pub(crate) fn cx(&self) -> f32 {
        (self.x0 + self.x1) * 0.5
    }
    pub(crate) fn cz(&self) -> f32 {
        (self.z0 + self.z1) * 0.5
    }
    pub(crate) fn inset(&self, m: f32) -> Rect {
        Rect {
            x0: self.x0 + m,
            z0: self.z0 + m,
            x1: self.x1 - m,
            z1: self.z1 - m,
        }
    }
}

/// Recursive binary subdivision — the standard way a block turns into parcels,
/// and the reason the footprints look platted rather than gridded.
/// The narrowest parcel worth cutting: two window bays plus the setback on
/// each side. Anything thinner fails `snap_span` and is built on by nothing,
/// so splitting down to it turns a block into a car park.
pub(crate) const MIN_LOT_SIDE: f32 = BAY * 2.0 + 1.6;

/// The most lopsided a split can be. The guard below has to use this and not
/// a half, or a 38/62 cut still produces a sliver the check thought it had
/// ruled out — which it did, until a test said otherwise.
pub(crate) const SPLIT_MIN: f32 = 0.38;

pub(crate) fn subdivide(r: Rect, target: f32, depth: u32, rng: &mut Rng, out: &mut Vec<Rect>) {
    let long = r.w().max(r.d());
    if depth == 0 || (r.area() < target * rng.range(0.9, 1.5) && long < 34.0) {
        out.push(r);
        return;
    }
    // Only cut along an axis that leaves both halves buildable. Without this
    // a small block splits into slivers, every one of them fails to take a
    // building, and the whole block comes back as surface parking.
    // A cut is possible whenever both halves CAN be buildable — then the cut
    // point is clamped so they both are. Testing `side * SPLIT_MIN` instead
    // refuses any block under 20 m outright, which is correct but blunt: it
    // halved the number of buildings in the tighter layouts.
    let can_x = r.w() >= 2.0 * MIN_LOT_SIDE;
    let can_z = r.d() >= 2.0 * MIN_LOT_SIDE;
    if !can_x && !can_z {
        out.push(r);
        return;
    }
    let split_x = if can_x && can_z { r.w() >= r.d() } else { can_x };
    let side = if split_x { r.w() } else { r.d() };
    // Never nearer either edge than one buildable lot.
    let lo = (MIN_LOT_SIDE / side).max(SPLIT_MIN);
    let t = rng.range(lo, 1.0 - lo);
    let (a, b) = if split_x {
        let m = r.x0 + r.w() * t;
        (Rect { x1: m, ..r }, Rect { x0: m, ..r })
    } else {
        let m = r.z0 + r.d() * t;
        (Rect { z1: m, ..r }, Rect { z0: m, ..r })
    };
    subdivide(a, target, depth - 1, rng, out);
    subdivide(b, target, depth - 1, rng, out);
}

/// Shrink `a..b` to a whole number of `step`s, centred. `None` if too small.
pub(crate) fn snap_span(a: f32, b: f32, step: f32, min_steps: f32) -> Option<(f32, f32)> {
    let c = (a + b) * 0.5;
    let n = ((b - a) / step).floor();
    if n < min_steps {
        return None;
    }
    let half = n * step * 0.5;
    Some((c - half, c + half))
}
