//! Part of the `simcity` example; see `mod.rs`.
//!
//! Who is standing where.
//!
//! Everything in this city is placed by scatter: a tree at a random point in
//! the setback strip, a bin at a random point in the pavement, a bench at a
//! random point on a path. Each of those is reasonable on its own and none of
//! them knows about the others, so a lamp post grows through a tree, a bin
//! stands inside a bench and a parked car occupies the same two metres as a
//! fire hydrant. At street level that is the difference between a city and a
//! pile of props.
//!
//! The fix is a claim: before anything is drawn, it asks for the ground it
//! needs, and it is only drawn if that ground is free. A uniform grid of
//! buckets holding axis-aligned rectangles is enough — the objects are small,
//! the city is flat, and the query is "does this box touch any other box",
//! which at four metres a cell touches a handful of candidates.
#![allow(dead_code)]

// Self-contained: nothing from the shared prelude is needed here.
use std::collections::HashMap;

/// A claimed rectangle: `[x0, z0, x1, z1]`.
type Rect4 = [f32; 4];

pub(crate) struct Occupancy {
    cell: f32,
    grid: HashMap<(i32, i32), Vec<Rect4>>,
    claims: usize,
    refusals: usize,
}

impl Default for Occupancy {
    fn default() -> Self {
        // Four metres: bigger than almost everything that claims, so most
        // rectangles land in one or two buckets and the candidate list stays
        // short.
        Occupancy::new(4.0)
    }
}

impl Occupancy {
    pub(crate) fn new(cell: f32) -> Self {
        Self {
            cell: cell.max(0.5),
            grid: HashMap::new(),
            claims: 0,
            refusals: 0,
        }
    }

    fn cells(&self, r: Rect4) -> impl Iterator<Item = (i32, i32)> {
        let c = self.cell;
        let (x0, z0) = ((r[0] / c).floor() as i32, (r[1] / c).floor() as i32);
        let (x1, z1) = ((r[2] / c).floor() as i32, (r[3] / c).floor() as i32);
        (x0..=x1).flat_map(move |x| (z0..=z1).map(move |z| (x, z)))
    }

    /// Is this ground clear?
    pub(crate) fn free(&self, r: Rect4) -> bool {
        for key in self.cells(r) {
            let Some(list) = self.grid.get(&key) else {
                continue;
            };
            for o in list {
                if r[0] < o[2] && r[2] > o[0] && r[1] < o[3] && r[3] > o[1] {
                    return false;
                }
            }
        }
        true
    }

    /// Take the ground whether or not it was free. For things that are placed
    /// by the plan rather than by the dice — a building, a road — and that
    /// everything else has to work around.
    pub(crate) fn claim(&mut self, r: Rect4) {
        self.claims += 1;
        for key in self.cells(r) {
            self.grid.entry(key).or_default().push(r);
        }
    }

    /// Take the ground if it is free. `false` means "do not draw this".
    pub(crate) fn try_claim(&mut self, r: Rect4) -> bool {
        if !self.free(r) {
            self.refusals += 1;
            return false;
        }
        self.claim(r);
        true
    }

    /// As [`try_claim`](Self::try_claim), for something centred at `(x, z)`
    /// with a radius. Almost everything that scatters is round enough for
    /// this, and a square is a safe over-estimate of a circle.
    pub(crate) fn try_spot(&mut self, x: f32, z: f32, r: f32) -> bool {
        self.try_claim([x - r, z - r, x + r, z + r])
    }

    /// Claim without a test, centred.
    pub(crate) fn take_spot(&mut self, x: f32, z: f32, r: f32) {
        self.claim([x - r, z - r, x + r, z + r]);
    }

    pub(crate) fn claimed(&self) -> usize {
        self.claims
    }

    /// How many placements were refused. Zero means nothing is consulting it.
    pub(crate) fn refused(&self) -> usize {
        self.refusals
    }
}
