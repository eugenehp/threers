//! Screen-space boxes several layers place their labels and icons against.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Boxes already placed on screen this frame, in pixels, `[x0, y0, x1, y1]`
/// from the top left — what keeps labels and icons from overlapping.
///
/// One index shared by every label and sprite layer (a [`SharedCollision`])
/// makes them collide across layers and tiles: each layer places in draw
/// order, so an earlier layer's boxes win over a later one's. Start each
/// frame with [`begin_frame`](Self::begin_frame), before drawing:
///
/// ```
/// use threers::marks::CollisionIndex;
/// let index = CollisionIndex::new().shared();
/// // Every frame: `index.lock().unwrap().begin_frame(&view);`
/// // ... then draw the layers that hold `index.clone()`.
/// assert!(index.lock().unwrap().try_insert([0.0, 0.0, 10.0, 10.0]));
/// assert!(!index.lock().unwrap().try_insert([5.0, 5.0, 20.0, 20.0]));
/// ```
///
/// `begin_frame` keeps last frame's boxes while the view holds still and no
/// layer changed, and the layers, seeing the same view and generation, skip
/// placement altogether — a still map costs nothing to place.
/// [`clear`](Self::clear) every frame also works; it just places every frame.
#[derive(Debug, Default)]
pub struct CollisionIndex {
    /// Each box with the layer that placed it (0: none in particular).
    grid: HashMap<(i32, i32), Vec<OwnedBox>>,
    generation: u64,
    /// The view the boxes were placed for.
    frame: Option<FrameKey>,
    /// A layer re-placed in place this frame: rebuild from scratch next one,
    /// so the draw order decides again.
    stale: bool,
}

/// A box and the layer that placed it.
type OwnedBox = ([f32; 4], u64);

type FrameKey = ([f32; 16], (u32, u32), [f64; 3]);

/// A fresh owner for [`CollisionIndex::insert_owned`]: each label or sprite
/// layer takes one when made.
pub(crate) fn next_owner() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// A collision index several layers share.
pub type SharedCollision = Arc<Mutex<CollisionIndex>>;

const CELL: f32 = 64.0;

impl CollisionIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn shared(self) -> SharedCollision {
        Arc::new(Mutex::new(self))
    }

    /// Forget every box: the start of a frame.
    pub fn clear(&mut self) {
        self.grid.clear();
        self.generation += 1;
        self.frame = None;
        self.stale = false;
    }

    /// The start of a frame drawn with `view`: clears when the view moved
    /// or a layer changed since the boxes were placed, else keeps them.
    /// Whether it cleared.
    pub fn begin_frame(&mut self, view: &super::MarkView) -> bool {
        let key = (view.view_proj, view.viewport, view.origin.position);
        if self.frame == Some(key) && !self.stale {
            return false;
        }
        self.clear();
        self.frame = Some(key);
        true
    }

    /// Boxes placed so far, each counted once.
    pub fn len(&self) -> usize {
        let mut seen: Vec<([u32; 4], u64)> = self
            .grid
            .values()
            .flatten()
            .map(|(r, o)| (r.map(f32::to_bits), *o))
            .collect();
        seen.sort_unstable();
        seen.dedup();
        seen.len()
    }

    pub fn is_empty(&self) -> bool {
        self.grid.values().all(Vec::is_empty)
    }

    /// Have the next [`begin_frame`](Self::begin_frame) clear whatever the
    /// view — after removing layers that placed boxes, or changing the
    /// order they draw in, neither of which the index can see.
    pub fn invalidate(&mut self) {
        self.stale = true;
    }

    /// Take back the boxes `owner` placed, for it to place again.
    pub fn remove_owner(&mut self, owner: u64) {
        for boxes in self.grid.values_mut() {
            boxes.retain(|b| b.1 != owner);
        }
    }

    /// Changes on every [`clear`](Self::clear): a layer that placed into an
    /// older generation must place again.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    fn cells(r: [f32; 4]) -> impl Iterator<Item = (i32, i32)> {
        let (cx0, cy0) = ((r[0] / CELL).floor() as i32, (r[1] / CELL).floor() as i32);
        let (cx1, cy1) = ((r[2] / CELL).floor() as i32, (r[3] / CELL).floor() as i32);
        (cy0..=cy1).flat_map(move |y| (cx0..=cx1).map(move |x| (x, y)))
    }

    /// Whether `r` overlaps a box already placed.
    pub fn collides(&self, r: [f32; 4]) -> bool {
        Self::cells(r).any(|c| {
            self.grid.get(&c).is_some_and(|boxes| {
                boxes
                    .iter()
                    .any(|(o, _)| r[0] < o[2] && o[0] < r[2] && r[1] < o[3] && o[1] < r[3])
            })
        })
    }

    /// Place `r` whatever it overlaps.
    pub fn insert(&mut self, r: [f32; 4]) {
        self.insert_owned(0, r);
    }

    /// Place `r` if it overlaps nothing; whether it was placed.
    pub fn try_insert(&mut self, r: [f32; 4]) -> bool {
        self.try_insert_owned(0, r)
    }

    /// [`insert`](Self::insert), remembering who placed it.
    pub fn insert_owned(&mut self, owner: u64, r: [f32; 4]) {
        for c in Self::cells(r) {
            self.grid.entry(c).or_default().push((r, owner));
        }
    }

    /// [`try_insert`](Self::try_insert), remembering who placed it.
    pub fn try_insert_owned(&mut self, owner: u64, r: [f32; 4]) -> bool {
        if self.collides(r) {
            return false;
        }
        self.insert_owned(owner, r);
        true
    }

    /// Before a layer places into a shared index: when the index has not been
    /// cleared since the layer last placed, its old boxes are still in it —
    /// take them out, and have the next frame rebuild in draw order.
    pub(crate) fn reclaim(&mut self, owner: u64, placed_generation: Option<u64>) {
        if placed_generation == Some(self.generation) {
            self.remove_owner(owner);
            self.stale = true;
        }
    }
}
