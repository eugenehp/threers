//! A cached layout is the layout: trees with `cache_layout` boundaries,
//! edited at random, lay out and paint exactly as the same trees without
//! them, frame after frame.

use threers_ui::{
    AlignItems, Display, Edges, FlexDirection, Length, NodeId, Position, Style, TextMeasure, Ui,
};

/// Characters of different widths, so that wrapping and flexing depend on
/// which text is where.
struct Proportional;

impl TextMeasure for Proportional {
    fn measure(&mut self, text: &str, style: &Style) -> (f32, f32) {
        let em = style.font_size / 13.0;
        let w: f32 = text
            .chars()
            .map(|c| match c {
                'i' | 'l' | '.' => 3.0,
                'm' | 'w' => 9.5,
                ' ' => 3.5,
                _ => 6.25,
            })
            .sum();
        (w * em, style.line_height.unwrap_or(style.font_size * 1.2))
    }
}

/// xorshift: the same edits on every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn f(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (self.next() % 10_000) as f32 / 10_000.0 * (hi - lo)
    }
}

const WORDS: [&str; 6] = ["mill", "wide words", "i", "lattice", "m w m", "a longer run of text"];

fn random_style(rng: &mut Rng) -> Style {
    let mut s = Style::default();
    match rng.below(4) {
        0 => s = Style::flex(FlexDirection::Row),
        1 => s = Style::flex(FlexDirection::Column),
        _ => {}
    }
    if rng.below(3) == 0 {
        s.width = Length::Px(rng.f(20.0, 160.0));
    }
    if rng.below(4) == 0 {
        s.height = Length::Px(rng.f(10.0, 80.0));
    }
    if rng.below(3) == 0 {
        s.flex_grow = rng.f(0.0, 2.0);
    }
    s.padding = Edges::all(rng.f(0.0, 3.0));
    s.margin = Edges::symmetric(rng.f(0.0, 2.0), rng.f(0.0, 2.0));
    s.gap = rng.f(0.0, 4.0);
    s.wrap = rng.below(2) == 0;
    s.ellipsis = rng.below(6) == 0;
    s.clip = s.ellipsis || rng.below(5) == 0;
    s.font_size = [11.0, 13.0, 16.0][rng.below(3)];
    s.align_items = [AlignItems::Stretch, AlignItems::Center, AlignItems::FlexStart][rng.below(3)];
    s.background = Some([rng.f(0.0, 1.0), 0.2, 0.3, 1.0]);
    if rng.below(8) == 0 {
        s.position = Position::Absolute;
        s.left = Length::Px(rng.f(0.0, 30.0));
        s.top = Length::Px(rng.f(0.0, 30.0));
    }
    if rng.below(7) == 0 {
        s.scroll = true;
        s.clip = true;
        s.height = Length::Px(rng.f(20.0, 60.0));
        s.scroll_top = rng.f(0.0, 40.0);
    }
    if rng.below(8) == 0 {
        s.display = Display::Grid;
        s.grid_columns = (0..1 + rng.below(3))
            .map(|k| match k % 3 {
                0 => threers_ui::Track::px(rng.f(10.0, 60.0)),
                1 => threers_ui::Track::fr(rng.f(0.5, 2.0)),
                _ => threers_ui::Track::auto(),
            })
            .collect();
    }
    if rng.below(12) == 0 {
        s.display = Display::None;
    }
    s
}

/// The same edit on both trees; `boundary` decides which nodes of the
/// cached one are boundaries.
struct Pair {
    plain: Ui,
    cached: Ui,
    nodes: Vec<NodeId>,
}

impl Pair {
    fn both(&mut self, f: impl Fn(&mut Ui) -> Option<NodeId>) {
        let a = f(&mut self.plain);
        let b = f(&mut self.cached);
        assert_eq!(a, b, "the trees diverged");
        if let Some(id) = a {
            self.nodes.push(id);
        }
    }

    fn live(&self, rng: &mut Rng) -> Option<NodeId> {
        let live: Vec<NodeId> = self.nodes.iter().copied().filter(|&n| self.plain.contains(n)).collect();
        (!live.is_empty()).then(|| live[rng.below(live.len())])
    }

    fn add(&mut self, rng: &mut Rng) {
        let parent = match self.live(rng) {
            Some(p) if !matches!(self.plain.kind(p), threers_ui::NodeKind::Text(_)) => p,
            _ => self.plain.root(),
        };
        if rng.below(3) == 0 {
            let text = WORDS[rng.below(WORDS.len())];
            self.both(|ui| Some(ui.add_text(parent, text)));
        } else {
            let style = random_style(rng);
            self.both(|ui| Some(ui.add_box(parent, style.clone())));
        }
    }

    fn edit(&mut self, rng: &mut Rng) {
        let Some(id) = self.live(rng) else {
            return self.add(rng);
        };
        match rng.below(8) {
            0 | 1 => self.add(rng),
            2 => {
                if id != self.plain.root() {
                    self.both(|ui| {
                        ui.remove(id);
                        None
                    });
                }
            }
            3 => {
                if let threers_ui::NodeKind::Text(_) = self.plain.kind(id) {
                    let text = WORDS[rng.below(WORDS.len())];
                    self.both(|ui| {
                        ui.set_text(id, text);
                        None
                    });
                }
            }
            4 => {
                let style = random_style(rng);
                self.both(|ui| {
                    ui.set_style(id, style.clone());
                    None
                });
            }
            5 => {
                let top = rng.f(0.0, 50.0);
                self.both(|ui| {
                    ui.style_mut(id).scroll_top = top;
                    None
                });
            }
            6 => {
                let w = Length::Px(rng.f(10.0, 200.0));
                self.both(|ui| {
                    ui.style_mut(id).width = w;
                    None
                });
            }
            _ => {
                let to = rng.below(4);
                if let Some(parent) = self.plain.parent(id) {
                    self.both(|ui| {
                        ui.move_child(parent, id, to);
                        None
                    });
                }
            }
        }
    }
}

fn assert_same(pair: &Pair, frame: usize) {
    assert_eq!(pair.plain.scene(), pair.cached.scene(), "frame {frame}: scenes differ");
    for &n in &pair.nodes {
        if pair.plain.contains(n) {
            let (a, b) = (pair.plain.rect(n), pair.cached.rect(n));
            assert_eq!(
                [a.x, a.y, a.width, a.height].map(f32::to_bits),
                [b.x, b.y, b.width, b.height].map(f32::to_bits),
                "frame {frame}: {n:?} placed differently"
            );
            if pair.plain.style(n).scroll {
                assert_eq!(pair.plain.scroll_extent(n), pair.cached.scroll_extent(n));
            }
        }
    }
}

fn run(seed: u64, layout_units: bool) {
    let mut rng = Rng(seed);
    let mut pair = Pair {
        plain: Ui::new(),
        cached: Ui::new(),
        nodes: vec![],
    };
    let root = pair.plain.root();
    let style = Style::flex(FlexDirection::Column);
    pair.both(|ui| {
        ui.set_layout_units(layout_units);
        ui.set_style(root, style.clone());
        None
    });
    for _ in 0..40 {
        pair.add(&mut rng);
    }
    let mut size = (320.0, 240.0);
    for frame in 0..400 {
        for _ in 0..rng.below(3) {
            pair.edit(&mut rng);
        }
        // Boundaries come and go, as views are cached or not.
        if let Some(n) = pair.live(&mut rng) {
            let on = rng.below(3) != 0;
            pair.cached.cache_layout(n, on);
        }
        if rng.below(20) == 0 {
            size = (rng.f(150.0, 400.0), rng.f(100.0, 300.0));
        }
        pair.plain.layout(size.0, size.1, &mut Proportional);
        pair.cached.layout(size.0, size.1, &mut Proportional);
        assert_same(&pair, frame);
        // An app moving something after layout, which the next layout undoes.
        if rng.below(10) == 0
            && let Some(n) = pair.live(&mut rng)
        {
            pair.both(|ui| {
                ui.offset(n, 0.5, 1.25);
                None
            });
        }
    }
}

#[test]
fn cached_layouts_match_uncached_ones() {
    for seed in 1u64..=12 {
        run(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15), false);
    }
}

#[test]
fn cached_layouts_match_on_the_layout_unit_grid() {
    for seed in 1u64..=6 {
        run(seed.wrapping_mul(0xD1B5_4A32_D192_ED03), true);
    }
}

#[test]
fn an_unchanged_cached_panel_is_not_laid_out_again() {
    let mut ui = Ui::new();
    let root = ui.root();
    ui.set_style(root, Style::flex(FlexDirection::Row));
    let panel = ui.add_box(root, Style::flex(FlexDirection::Column));
    for i in 0..50 {
        let row = ui.add_box(panel, Style::flex(FlexDirection::Row));
        ui.add_text(row, format!("row {i}"));
    }
    let counter = ui.add_text(root, "0");
    ui.cache_layout(panel, true);
    ui.layout(400.0, 300.0, &mut Proportional);
    let first = ui.laid_out_last();
    let before = ui.scene();

    // Only the counter changes: the panel's 101 nodes come from the cache.
    ui.set_text(counter, "1");
    ui.layout(400.0, 300.0, &mut Proportional);
    assert!(ui.laid_out_last() < 5, "{} of {first} laid out again", ui.laid_out_last());
    assert_ne!(ui.scene(), before);

    // Changing a row inside it invalidates the panel and nothing else.
    let row = ui.children(panel)[7];
    ui.set_text(ui.children(row)[0], "changed");
    ui.layout(400.0, 300.0, &mut Proportional);
    assert!(ui.laid_out_last() > 100);
}

#[test]
fn removed_ids_go_stale_and_their_slots_are_reused() {
    let mut ui = Ui::new();
    let root = ui.root();
    let a = ui.add_box(root, Style::default());
    let inner = ui.add_text(a, "x");
    ui.remove(a);
    assert!(!ui.contains(a) && !ui.contains(inner));
    // The removed slots are taken again, under new generations.
    let b = ui.add_box(root, Style::default());
    let c = ui.add_box(root, Style::default());
    assert!(ui.contains(b) && ui.contains(c));
    assert_ne!(a, b);
    assert_ne!(inner, c);
    ui.remove(a); // already gone: nothing happens
    assert!(ui.contains(b));
    assert_eq!(ui.children(root), &[b, c]);
}

#[test]
#[should_panic(expected = "is not in this Ui")]
fn a_stale_id_is_refused() {
    let mut ui = Ui::new();
    let root = ui.root();
    let a = ui.add_box(root, Style::default());
    ui.remove(a);
    ui.add_box(root, Style::default());
    ui.style_mut(a).opacity = 0.5;
}

#[test]
fn rebuilding_every_frame_does_not_grow_the_tree() {
    let mut ui = Ui::new();
    let root = ui.root();
    for frame in 0..100 {
        ui.clear_children(root);
        for i in 0..20 {
            let b = ui.add_box(root, Style::default());
            ui.add_text(b, format!("{frame}.{i}"));
        }
    }
    // 1 root + 40 live; nothing kept for the 3,960 removed.
    let mut count = 0;
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        count += 1;
        stack.extend(ui.children(n));
    }
    assert_eq!(count, 41);
    // The slots of one frame's worth of nodes, reused ever since.
    assert!(ui.slots() <= 1 + 2 * 40, "{} slots", ui.slots());
}
