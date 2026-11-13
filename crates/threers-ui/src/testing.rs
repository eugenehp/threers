//! Random trees for testing: reproducible from a seed, varied in everything layout, painting
//! and hit-testing depend on, for checking that two ways of computing
//! something agree.
//!
//! ```
//! use threers_ui::testing::{random_tree, Rng};
//! use threers_ui::FixedWidthText;
//!
//! let (mut ui, nodes) = random_tree(&mut Rng::new(7), 80);
//! ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
//! // Same seed, same tree.
//! let (again, _) = random_tree(&mut Rng::new(7), 80);
//! assert_eq!(nodes.len(), 81);
//! assert_eq!(again.live_nodes(), ui.live_nodes());
//! ```

use super::style::{AlignItems, Display, Edges, FlexDirection, GridPlacement, Length, Position, Style, Track, TrackSize};
use super::tree::{NodeId, Ui};

/// xorshift64: small, fast and the same everywhere, which is what a
/// reproducible test wants.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    /// From any seed (0 is moved off the one state xorshift cannot leave).
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).max(1))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// In `0..n` (`n` above 0).
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }

    /// In `lo..hi`.
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (self.next_u64() % 1_000_000) as f32 / 1_000_000.0 * (hi - lo)
    }

    /// True one time in `n`.
    pub fn one_in(&mut self, n: usize) -> bool {
        self.below(n) == 0
    }
}

/// Words of different widths, for text that wraps differently by where it is.
pub const WORDS: [&str; 6] = ["mill", "wide words", "i", "lattice", "m w m", "a longer run of text"];

/// A style with some of everything layout reads: flex or block, sizes,
/// margins and padding, gaps, wrapping and ellipsis, clipping, absolute
/// positions, scrolling, and now and then `display: none`.
pub fn random_style(rng: &mut Rng) -> Style {
    let mut s = match rng.below(4) {
        0 => Style::flex(FlexDirection::Row),
        1 => Style::flex(FlexDirection::Column),
        _ => Style::default(),
    };
    if rng.one_in(3) {
        s.width = Length::Px(rng.range(20.0, 160.0));
    }
    if rng.one_in(4) {
        s.height = Length::Px(rng.range(10.0, 80.0));
    }
    if rng.one_in(3) {
        s.flex_grow = rng.range(0.0, 2.0);
    }
    s.padding = Edges::all(rng.range(0.0, 3.0));
    s.margin = Edges::symmetric(rng.range(0.0, 2.0), rng.range(0.0, 2.0));
    s.gap = rng.range(0.0, 4.0);
    s.wrap = rng.one_in(2);
    s.ellipsis = rng.one_in(6);
    s.clip = s.ellipsis || rng.one_in(5);
    s.font_size = [11.0, 13.0, 16.0][rng.below(3)];
    s.align_items = [AlignItems::Stretch, AlignItems::Center, AlignItems::FlexStart][rng.below(3)];
    s.background = Some([rng.range(0.0, 1.0), 0.2, 0.3, 1.0]);
    if rng.one_in(8) {
        s.position = Position::Absolute;
        s.left = Length::Px(rng.range(-20.0, 200.0));
        s.top = Length::Px(rng.range(-20.0, 150.0));
    }
    if rng.one_in(7) {
        s.scroll = true;
        s.clip = true;
        s.height = Length::Px(rng.range(20.0, 60.0));
        s.scroll_top = rng.range(0.0, 40.0);
    }
    if rng.one_in(8) {
        s.display = Display::Grid;
        s.grid_columns = (0..1 + rng.below(4))
            .map(|_| match rng.below(4) {
                0 => Track::px(rng.range(10.0, 80.0)),
                1 => Track::fr(rng.range(0.5, 3.0)),
                2 => Track::auto(),
                _ => Track { min: TrackSize::Px(rng.range(10.0, 40.0)), max: TrackSize::Fr(1.0) },
            })
            .collect();
    }
    if rng.one_in(6) {
        s.grid_column = GridPlacement { start: None, end: None, span: 1 + rng.below(2) as u32 };
    }
    if rng.one_in(15) {
        s.pointer_events = false;
    }
    if rng.one_in(12) {
        s.display = Display::None;
    }
    s
}

/// A tree of `n` nodes under the root: boxes with [`random_style`], and
/// text a third of the time. Returns the tree and every node, root first.
pub fn random_tree(rng: &mut Rng, n: usize) -> (Ui, Vec<NodeId>) {
    let mut ui = Ui::new();
    let mut nodes = vec![ui.root()];
    let mut parents = vec![ui.root()];
    for _ in 0..n {
        let parent = parents[rng.below(parents.len())];
        if rng.one_in(3) {
            nodes.push(ui.add_text(parent, WORDS[rng.below(WORDS.len())]));
        } else {
            let node = ui.add_box(parent, random_style(rng));
            nodes.push(node);
            parents.push(node);
        }
    }
    (ui, nodes)
}
