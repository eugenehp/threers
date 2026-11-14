//! Input, layers, images, animation and the rest, each through the
//! public API.

use threers_ui::debug_overlay::{FrameSample, FrameStats};
use threers_ui::image::{Image, ImageCache, ImageState};
use threers_ui::input::{Event, Gesture, GesturePhase, Input, Phase, TouchPhase};
use threers_ui::layers::{Anchor, AnchorTo, Corner, Fit};
use threers_ui::scene::Item;
use threers_ui::text_cache::TextCache;
use threers_ui::virtual_list::{RowHeight, VirtualList};
use threers_ui::{
    style, Cursor, Display, FixedWidthText, Length, NodeId, ObjectFit, Position, Role, Semantics,
    Style, Ui,
};

fn layout(ui: &mut Ui) {
    ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
}

#[test]
fn tab_index_orders_like_the_dom() {
    let mut ui = Ui::new();
    let root = ui.root();
    let mut add = |sem: Semantics| {
        let n = ui.add_box(root, style! { height: 10px; });
        ui.set_semantics(n, sem);
        n
    };
    let a = add(Semantics::new(Role::Button));
    let b = add(Semantics::new(Role::Button).tab_index(2));
    let c = add(Semantics::new(Role::Button).tab_index(1));
    let d = add(Semantics::new(Role::Button).tab_index(-1));
    let e = add(Semantics::new(Role::Generic).tab_index(0));
    assert_eq!(ui.focus_order(), [c, b, a, e]);
    // Out of the sequence, but a press still focuses it.
    layout(&mut ui);
    let mut input = Input::new();
    input.click(&mut ui, &mut (), 5.0, 35.0);
    assert_eq!(input.focused(), Some(d));
}

#[test]
fn cursors_inherit_and_follow_a_captured_pointer() {
    let mut ui = Ui::new();
    let root = ui.root();
    let link = ui.add_box(root, style! { height: 20px; cursor: pointer; });
    let inner = ui.add_box(link, style! { width: 10px; height: 10px; });
    let field = ui.add_box(root, style! { height: 20px; cursor: text; });
    layout(&mut ui);
    assert_eq!(
        ui.cursor_at(5.0, 5.0),
        Cursor::Pointer,
        "inherited by {inner:?}"
    );
    assert_eq!(ui.cursor_at(5.0, 25.0), Cursor::Text);
    assert_eq!(ui.cursor_at(5.0, 200.0), Cursor::Default);
    assert_eq!(Cursor::parse("col-resize"), Some(Cursor::ResizeHorizontal));
    assert_eq!(Cursor::Text.css(), "text");

    let mut input = Input::new();
    input.on(field, Phase::Bubble, |_: &mut (), _, e, cx| {
        if let Event::PointerDown { .. } = e {
            cx.capture_pointer();
        }
    });
    input.dispatch(
        &mut ui,
        &mut (),
        Event::PointerDown {
            x: 5.0,
            y: 25.0,
            button: Default::default(),
        },
    );
    input.dispatch(&mut ui, &mut (), Event::PointerMove { x: 5.0, y: 5.0 });
    assert_eq!(
        input.cursor(&ui),
        Cursor::Text,
        "held by the field while dragging"
    );
}

#[test]
fn the_text_cache_measures_each_string_once_while_it_shows() {
    let mut ui = Ui::new();
    let root = ui.root();
    for i in 0..30 {
        ui.add_text(root, format!("line {i}"));
    }
    let mut cache = TextCache::new();
    let mut inner = FixedWidthText(7.0);
    for _ in 0..5 {
        // A layout that measures again (as after a font loads): the cache
        // answers. (An unchanged tree would not be laid out at all.)
        ui.invalidate_layout();
        ui.layout(400.0, 300.0, &mut cache.measure_with(&mut inner));
        cache.finish_frame();
    }
    let (hits, misses) = cache.stats();
    assert_eq!(misses, 30, "each string measured once");
    assert!(hits >= 4 * 30);
}

#[test]
fn a_uniform_list_builds_only_what_shows() {
    let mut ui = Ui::new();
    let root = ui.root();
    let mut list = VirtualList::new(
        &mut ui,
        root,
        style! { height: 200px; },
        100_000,
        RowHeight::Uniform(20.0),
    );
    let sync = |ui: &mut Ui, list: &mut VirtualList| {
        layout(ui);
        let mut rounds = 0;
        while list.sync(ui, |ui, row, i| {
            ui.add_text(row, format!("row {i}"));
        }) {
            layout(ui);
            rounds += 1;
            assert!(rounds < 5, "never settled");
        }
    };
    sync(&mut ui, &mut list);
    assert_eq!(list.visible(&ui), 0..10);
    assert!(list.built().len() <= 13);
    assert!(ui.live_nodes() < 50, "{} nodes", ui.live_nodes());

    list.scroll_to(&mut ui, 54_321);
    sync(&mut ui, &mut list);
    let row = list.row(54_321).unwrap();
    let view = ui.rect(list.container());
    let r = ui.rect(row);
    assert!(
        r.y >= view.y && r.y + r.height <= view.y + view.height + 0.01,
        "{r:?} in {view:?}"
    );
    let threers_ui::NodeKind::Text(t) = ui.kind(ui.children(row)[0]) else {
        panic!()
    };
    assert_eq!(t, "row 54321");
    assert_eq!(ui.hit_test(10.0, r.y + 5.0), Some(ui.children(row)[0]));
    assert!(ui.live_nodes() < 50);
}

#[test]
fn a_measured_list_keeps_the_row_in_view_still_as_heights_arrive() {
    let mut ui = Ui::new();
    let root = ui.root();
    let height = |i: usize| 10.0 + (i % 5) as f32 * 10.0;
    let mut list = VirtualList::new(
        &mut ui,
        root,
        style! { height: 120px; },
        2_000,
        RowHeight::Measured { estimate: 20.0 },
    );
    let sync = |ui: &mut Ui, list: &mut VirtualList| {
        layout(ui);
        let mut rounds = 0;
        while list.sync(ui, |ui, row, i| {
            ui.add_box(
                row,
                Style {
                    height: Length::Px(height(i)),
                    ..Style::default()
                },
            );
        }) {
            layout(ui);
            rounds += 1;
            assert!(rounds < 10, "never settled");
        }
    };
    sync(&mut ui, &mut list);
    list.scroll_to(&mut ui, 1_000);
    sync(&mut ui, &mut list);
    let view = ui.rect(list.container());
    let row = list.row(1_000).expect("built");
    let top = ui.rect(row).y;
    assert!(
        top >= view.y - 0.01 && top < view.y + view.height,
        "row 1000 at {top}, view {view:?}"
    );
    assert_eq!(ui.rect(row).height, height(1_000));

    // Scrolling up a little measures rows above: the row stays where it is
    // on screen while the content above it changes height.
    let before = ui.rect(row).y;
    let up = ui.style(list.container()).scroll_top - 5.0;
    ui.style_mut(list.container()).scroll_top = up;
    sync(&mut ui, &mut list);
    let after = ui.rect(list.row(1_000).unwrap()).y;
    assert!(
        (after - (before + 5.0)).abs() < 0.01,
        "moved from {before} to {after}"
    );
    // Offsets are the measured heights where measured.
    let (start, end) = (list.offset_of(1_000), list.offset_of(1_001));
    assert_eq!(end - start, height(1_000));
}

/// xorshift.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn f(&mut self, hi: f32) -> f32 {
        (self.next() % 10_000) as f32 / 10_000.0 * hi
    }
}

#[test]
fn the_hit_index_agrees_with_walking_the_tree() {
    for seed in 1u64..=20 {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut ui = Ui::new();
        let mut nodes = vec![ui.root()];
        for _ in 0..120 {
            let parent = nodes[rng.below(nodes.len() as u64) as usize];
            let mut s = Style {
                width: Length::Px(rng.f(200.0)),
                height: Length::Px(rng.f(120.0)),
                margin: threers_ui::Edges::all(rng.f(6.0)),
                ..Style::default()
            };
            if rng.below(4) == 0 {
                s.position = Position::Absolute;
                s.left = Length::Px(rng.f(300.0) - 50.0);
                s.top = Length::Px(rng.f(200.0) - 50.0);
            }
            if rng.below(4) == 0 {
                s.clip = true;
            }
            if rng.below(15) == 0 {
                s.display = Display::None;
            }
            let n = ui.add_box(parent, s);
            if rng.below(12) == 0 {
                ui.defer(n, rng.below(3) as i32);
            }
            nodes.push(n);
        }
        ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
        if rng.below(2) == 0 {
            let n = nodes[1 + rng.below(nodes.len() as u64 - 1) as usize];
            ui.offset(n, 3.5, -2.25);
        }
        for _ in 0..800 {
            let (x, y) = (rng.f(460.0) - 30.0, rng.f(360.0) - 30.0);
            assert_eq!(
                ui.hit_test(x, y),
                ui.hit_test_walk(x, y),
                "seed {seed} at ({x}, {y})"
            );
        }
    }
}

#[test]
fn deferred_nodes_paint_last_and_escape_their_clip() {
    let mut ui = Ui::new();
    let root = ui.root();
    let clip = ui.add_box(
        root,
        style! { width: 50px; height: 20px; overflow: hidden; },
    );
    let popup = ui.add_box(
        clip,
        style! { position: absolute; top: 15px; width: 100px; height: 40px; background: #f00; },
    );
    let after = ui.add_box(root, style! { height: 30px; background: #00f; });
    ui.defer(popup, 0);
    layout(&mut ui);
    let scene = ui.scene();
    let last = scene
        .items
        .iter()
        .rposition(|i| matches!(i, Item::Rect { color, .. } if color[0] == 1.0))
        .unwrap();
    let blue = scene
        .items
        .iter()
        .position(|i| matches!(i, Item::Rect { color, .. } if color[2] == 1.0))
        .unwrap();
    assert!(last > blue, "the popup is painted after what follows it");
    assert!(
        !matches!(scene.items[last - 1], Item::PushClip { .. }),
        "and not inside its parent's clip"
    );
    // Over the blue box, outside the parent's clip: the popup is hit.
    assert_eq!(ui.hit_test(80.0, 40.0), Some(popup));
    ui.undefer(popup);
    assert_eq!(ui.hit_test(80.0, 40.0), Some(after));
}

#[test]
fn anchored_nodes_flip_then_snap_into_the_window() {
    let mut ui = Ui::new();
    let root = ui.root();
    let at = ui.add_box(
        root,
        style! { position: absolute; left: 350px; top: 270px; width: 40px; height: 20px; },
    );
    let menu = ui.add_box(
        root,
        style! { position: absolute; width: 120px; height: 80px; },
    );
    ui.anchor(
        menu,
        Anchor::new(AnchorTo::Node(at, Corner::BottomLeft), Corner::TopLeft).offset(0.0, 4.0),
    );
    layout(&mut ui);
    // No room right or below: it hangs up and to the left of the button.
    let r = ui.rect(menu);
    assert_eq!((r.x, r.y), (390.0 - 120.0, 270.0 - 4.0 - 80.0));
    // Snap only: pushed inside, on its own side.
    ui.anchor(
        menu,
        Anchor::new(AnchorTo::Point(380.0, 290.0), Corner::TopLeft).fit(Fit::Snap),
    );
    layout(&mut ui);
    let r = ui.rect(menu);
    assert_eq!((r.x, r.y), (280.0, 220.0));
}

#[test]
fn the_inspector_names_the_line_that_made_a_node() {
    let mut ui = Ui::new();
    let root = ui.root();
    let line = line!() + 1;
    let node = ui.add_box(
        root,
        style! { width: 30px; height: 30px; padding: 2px; cursor: grab; },
    );
    layout(&mut ui);
    let found = ui.inspect(10.0, 10.0).unwrap();
    assert_eq!(found.node, node);
    assert_eq!(found.origin.unwrap().line(), line);
    assert!(found.origin.unwrap().file().ends_with("features.rs"));
    assert_eq!(found.cursor, Cursor::Grab);
    let text = found.to_string();
    assert!(
        text.contains("padding: Edges { top: 2.0, right: 2.0, bottom: 2.0, left: 2.0 }"),
        "{text}"
    );
    assert!(!text.contains("font_size"), "defaults are left out: {text}");
    let mut scene = ui.scene();
    let before = scene.items.len();
    ui.paint_highlight(node, &mut scene.recorder());
    assert!(scene.items.len() > before);
}

#[test]
fn the_debug_overlay_paints_its_numbers() {
    let mut ui = Ui::new();
    layout(&mut ui);
    let mut stats = FrameStats::default();
    for i in 0..200 {
        stats.record(FrameSample::of(&ui, 10.0 + (i % 30) as f32, 0.25));
    }
    assert_eq!(stats.samples().count(), FrameStats::CAPACITY);
    let mut scene = threers_ui::scene::Scene::default();
    stats.paint(&mut scene.recorder(), 8.0, 8.0);
    let texts: Vec<&str> = scene
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert!(texts[0].starts_with("frame"), "{texts:?}");
    assert!(texts[2].contains("of 1 nodes"));
    let bars = scene
        .items
        .iter()
        .filter(|i| matches!(i, Item::Rect { .. }))
        .count();
    assert!(bars > FrameStats::CAPACITY);
}

#[allow(clippy::too_many_arguments)]
fn touch(
    input: &mut Input<Vec<String>>,
    ui: &mut Ui,
    log: &mut Vec<String>,
    id: u64,
    phase: TouchPhase,
    x: f32,
    y: f32,
    time: f64,
) {
    input.dispatch(
        ui,
        log,
        Event::Touch {
            id,
            phase,
            x,
            y,
            time,
        },
    );
}

#[test]
fn fingers_tap_pan_press_and_pinch() {
    let mut ui = Ui::new();
    let root = ui.root();
    let list = ui.add_box(root, style! { height: 100px; overflow: auto; });
    let button = ui.add_box(list, style! { height: 40px; });
    ui.set_semantics(button, Semantics::new(Role::Button));
    ui.add_box(list, style! { height: 400px; });
    layout(&mut ui);
    let mut input: Input<Vec<String>> = Input::new();
    input.on_click(button, |log, _, _| log.push("click".into()));
    input.on(list, Phase::Bubble, |log, _, e, _| {
        if let Event::Gesture(g) = e {
            log.push(match g {
                Gesture::Tap { .. } => "tap".into(),
                Gesture::LongPress { .. } => "long".into(),
                Gesture::Pan { phase, .. } => format!("pan {phase:?}"),
                Gesture::Pinch { scale, phase, .. } => format!("pinch {phase:?} {scale:.2}"),
            });
        }
    });
    let mut log = vec![];
    // A tap: a click and a tap.
    touch(
        &mut input,
        &mut ui,
        &mut log,
        1,
        TouchPhase::Start,
        10.0,
        10.0,
        0.0,
    );
    touch(
        &mut input,
        &mut ui,
        &mut log,
        1,
        TouchPhase::End,
        11.0,
        11.0,
        0.1,
    );
    assert_eq!(log, ["click", "tap"]);
    log.clear();

    // A drag up: a pan that scrolls the list with the finger, and no click.
    touch(
        &mut input,
        &mut ui,
        &mut log,
        2,
        TouchPhase::Start,
        10.0,
        60.0,
        1.0,
    );
    touch(
        &mut input,
        &mut ui,
        &mut log,
        2,
        TouchPhase::Move,
        10.0,
        40.0,
        1.05,
    );
    touch(
        &mut input,
        &mut ui,
        &mut log,
        2,
        TouchPhase::Move,
        10.0,
        30.0,
        1.1,
    );
    touch(
        &mut input,
        &mut ui,
        &mut log,
        2,
        TouchPhase::End,
        10.0,
        30.0,
        1.15,
    );
    assert_eq!(log, ["pan Began", "pan Changed", "pan Ended"]);
    assert_eq!(ui.style(list).scroll_top, 30.0);
    log.clear();

    // Held still: a long press, then no click on lifting.
    touch(
        &mut input,
        &mut ui,
        &mut log,
        3,
        TouchPhase::Start,
        10.0,
        10.0,
        2.0,
    );
    input.tick(&mut ui, &mut log, 2.2);
    assert!(log.is_empty());
    input.tick(&mut ui, &mut log, 2.6);
    touch(
        &mut input,
        &mut ui,
        &mut log,
        3,
        TouchPhase::End,
        10.0,
        10.0,
        2.7,
    );
    assert_eq!(log, ["long"]);
    log.clear();

    // Two fingers apart to twice the distance.
    touch(
        &mut input,
        &mut ui,
        &mut log,
        4,
        TouchPhase::Start,
        100.0,
        50.0,
        3.0,
    );
    touch(
        &mut input,
        &mut ui,
        &mut log,
        5,
        TouchPhase::Start,
        140.0,
        50.0,
        3.0,
    );
    touch(
        &mut input,
        &mut ui,
        &mut log,
        5,
        TouchPhase::Move,
        180.0,
        50.0,
        3.1,
    );
    touch(
        &mut input,
        &mut ui,
        &mut log,
        5,
        TouchPhase::End,
        180.0,
        50.0,
        3.2,
    );
    touch(
        &mut input,
        &mut ui,
        &mut log,
        4,
        TouchPhase::End,
        100.0,
        50.0,
        3.2,
    );
    assert_eq!(
        log,
        ["pinch Began 1.00", "pinch Changed 2.00", "pinch Ended 2.00"]
    );
    assert!(!input.touching());
    let _ = GesturePhase::Began;
}

#[test]
fn images_size_by_aspect_fit_their_box_and_paint() {
    let mut ui = Ui::new();
    let root = ui.root();
    let px: Vec<u8> = (0..40 * 20)
        .flat_map(|i| {
            if i % 40 < 20 {
                [255, 0, 0, 255]
            } else {
                [0, 0, 255, 255]
            }
        })
        .collect();
    let img = Image::new(40, 20, px).unwrap();
    let natural = ui.add_image(root, Style::default(), img.clone());
    let tall = ui.add_image(root, style! { height: 10px; }, img.clone());
    let cover = ui.add_image(
        root,
        style! { width: 20px; height: 20px; object-fit: cover; },
        img.clone(),
    );
    layout(&mut ui);
    assert_eq!(
        (ui.rect(natural).width, ui.rect(natural).height),
        (40.0, 20.0)
    );
    assert_eq!((ui.rect(tall).width, ui.rect(tall).height), (20.0, 10.0));
    assert_eq!(ui.style(cover).object_fit, ObjectFit::Cover);
    let scene = ui.scene();
    let images: Vec<_> = scene
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Image { rect, image, .. } => Some((*rect, image.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(images.len(), 3);
    assert_eq!(images[0].1, img);
    // Cover: 40 wide in a 20 box, centred, so the box clips it.
    assert_eq!((images[2].0.x, images[2].0.width), (-10.0, 40.0));
    let i = scene
        .items
        .iter()
        .position(|i| matches!(i, Item::Image { rect, .. } if rect.width == 40.0 && rect.x < 0.0))
        .unwrap();
    assert!(matches!(scene.items[i - 1], Item::PushClip { .. }));

    // The cache: ready after polling.
    let mut cache = ImageCache::new();
    let made = img.clone();
    cache.get_or_load("photo", move || Ok(made));
    let start = std::time::Instant::now();
    while cache.loading() {
        cache.poll();
        assert!(start.elapsed().as_secs() < 5);
        std::thread::yield_now();
    }
    assert!(matches!(cache.get("photo"), Some(ImageState::Ready(i)) if *i == img));
    let _: Option<NodeId> = None;
}

#[test]
fn an_unchanged_tree_is_not_laid_out_again() {
    struct Counting(usize);
    impl threers_ui::TextMeasure for Counting {
        fn measure(&mut self, t: &str, s: &Style) -> (f32, f32) {
            self.0 += 1;
            FixedWidthText(7.0).measure(t, s)
        }
    }
    let mut ui = Ui::new();
    let root = ui.root();
    let t = ui.add_text(root, "hello");
    let mut m = Counting(0);
    ui.layout(400.0, 300.0, &mut m);
    let first = m.0;
    assert!(first > 0);
    ui.layout(400.0, 300.0, &mut m);
    ui.layout(400.0, 300.0, &mut m);
    assert_eq!(m.0, first, "nothing changed: nothing measured");
    ui.set_text(t, "hello, world");
    ui.layout(400.0, 300.0, &mut m);
    assert!(m.0 > first);
    let n = m.0;
    ui.layout(500.0, 300.0, &mut m);
    assert!(m.0 > n, "a new size lays out again");
    assert_eq!(ui.rect(t).width, 500.0);
}

/// A browser can lose a finger's end (iOS Safari does, mid-flick). The next
/// touch under that id is a new finger, not a second one: it scrolls.
#[test]
fn a_touch_whose_end_was_lost_does_not_turn_the_next_into_a_pinch() {
    let mut ui = Ui::new();
    let root = ui.root();
    let list = ui.add_box(root, style! { height: 100px; overflow: auto; });
    ui.add_box(list, style! { height: 2000px; });
    layout(&mut ui);
    let mut input: Input<Vec<String>> = Input::new();
    let mut log = vec![];
    touch(
        &mut input,
        &mut ui,
        &mut log,
        1,
        TouchPhase::Start,
        10.0,
        80.0,
        0.0,
    );
    touch(
        &mut input,
        &mut ui,
        &mut log,
        1,
        TouchPhase::Move,
        10.0,
        60.0,
        0.05,
    );
    // No end arrives.
    touch(
        &mut input,
        &mut ui,
        &mut log,
        1,
        TouchPhase::Start,
        10.0,
        80.0,
        1.0,
    );
    let before = ui.style(list).scroll_top;
    touch(
        &mut input,
        &mut ui,
        &mut log,
        1,
        TouchPhase::Move,
        10.0,
        50.0,
        1.05,
    );
    touch(
        &mut input,
        &mut ui,
        &mut log,
        1,
        TouchPhase::Move,
        10.0,
        40.0,
        1.1,
    );
    assert!(
        ui.style(list).scroll_top > before + 20.0,
        "{} → {}",
        before,
        ui.style(list).scroll_top
    );
}

/// Safari reports a lift some time after the finger's last move: the flick
/// still glides, at the speed it was moving.
#[test]
fn a_flick_whose_lift_arrives_late_still_glides() {
    let glide = |lift: f64| {
        let mut ui = Ui::new();
        let root = ui.root();
        let list = ui.add_box(root, style! { height: 100px; overflow: auto; });
        ui.add_box(list, style! { height: 20000px; });
        layout(&mut ui);
        let mut input: Input<Vec<String>> = Input::new();
        let mut log = vec![];
        touch(
            &mut input,
            &mut ui,
            &mut log,
            1,
            TouchPhase::Start,
            10.0,
            90.0,
            0.0,
        );
        for i in 1..=6 {
            touch(
                &mut input,
                &mut ui,
                &mut log,
                1,
                TouchPhase::Move,
                10.0,
                90.0 - 15.0 * i as f32,
                i as f64 * 0.016,
            );
        }
        touch(
            &mut input,
            &mut ui,
            &mut log,
            1,
            TouchPhase::End,
            10.0,
            0.0,
            0.096 + lift,
        );
        let released = ui.style(list).scroll_top;
        for i in 1..=60 {
            input.tick(&mut ui, &mut log, 0.096 + lift + i as f64 / 60.0);
        }
        ui.style(list).scroll_top - released
    };
    assert!(glide(0.0) > 200.0, "{}", glide(0.0));
    assert!(glide(0.07) > 200.0, "late lift: {}", glide(0.07));
    // Held still before lifting: no glide.
    assert_eq!(glide(0.3), 0.0);
}

/// A finger on a scrollbar's thumb drags the thumb, as the mouse does: down
/// the track scrolls down, rather than panning the content up with it. A
/// fingertip a little left of the thin thumb still takes it.
#[test]
fn a_finger_drags_a_scrollbar_thumb() {
    for reach in [0.0, 12.0] {
        let mut ui = Ui::new();
        let root = ui.root();
        let list = ui.add_box(
            root,
            style! { width: 200px; height: 100px; overflow: auto; scrollbar-width: auto; },
        );
        ui.add_box(list, style! { height: 1000px; });
        layout(&mut ui);
        let bar = ui.scrollbar(list).expect("a scrollbar");
        let thumb = bar.thumb;
        // The thumb's travel down the track is the whole 900 of scrolling.
        let want = 45.0 / (bar.track.height - thumb.height) * 900.0;
        let (x, y) = (
            thumb.x + thumb.width / 2.0 - reach,
            thumb.y + thumb.height / 2.0,
        );
        let mut input: Input<Vec<String>> = Input::new();
        let mut log = vec![];
        touch(
            &mut input,
            &mut ui,
            &mut log,
            1,
            TouchPhase::Start,
            x,
            y,
            0.0,
        );
        touch(
            &mut input,
            &mut ui,
            &mut log,
            1,
            TouchPhase::Move,
            x,
            y + 20.0,
            0.05,
        );
        touch(
            &mut input,
            &mut ui,
            &mut log,
            1,
            TouchPhase::Move,
            x,
            y + 45.0,
            0.1,
        );
        touch(
            &mut input,
            &mut ui,
            &mut log,
            1,
            TouchPhase::End,
            x,
            y + 45.0,
            0.15,
        );
        assert!(
            (ui.style(list).scroll_top - want).abs() < 1.0,
            "reach {reach}: {} not {want}",
            ui.style(list).scroll_top
        );
        for i in 1..=30 {
            input.tick(&mut ui, &mut log, 0.15 + i as f64 / 60.0);
        }
        assert!(
            (ui.style(list).scroll_top - want).abs() < 1.0,
            "no glide after: {}",
            ui.style(list).scroll_top
        );
        assert!(!input.touching());
    }
}

/// The node a finger pressed on can go while the content moves (a virtual
/// list drops rows that scroll out of sight): the pan and the glide after it
/// move the box that scrolled, and carry on.
#[test]
fn a_fling_outlives_the_row_it_started_on() {
    let mut ui = Ui::new();
    let root = ui.root();
    let list = ui.add_box(root, style! { height: 100px; overflow: auto; });
    let row = ui.add_box(list, style! { height: 20px; });
    ui.add_box(list, style! { height: 20000px; });
    layout(&mut ui);
    let mut input: Input<Vec<String>> = Input::new();
    let mut log = vec![];
    touch(
        &mut input,
        &mut ui,
        &mut log,
        1,
        TouchPhase::Start,
        10.0,
        10.0,
        0.0,
    );
    for i in 1..=6 {
        touch(
            &mut input,
            &mut ui,
            &mut log,
            1,
            TouchPhase::Move,
            10.0,
            10.0 - 15.0 * i as f32,
            i as f64 * 0.016,
        );
    }
    ui.remove(row);
    touch(
        &mut input,
        &mut ui,
        &mut log,
        1,
        TouchPhase::End,
        10.0,
        -80.0,
        0.105,
    );
    let released = ui.style(list).scroll_top;
    for i in 1..=30 {
        input.tick(&mut ui, &mut log, 0.105 + i as f64 / 60.0);
    }
    assert!(
        ui.style(list).scroll_top > released + 200.0,
        "{released} → {}",
        ui.style(list).scroll_top
    );
}

/// `scrollbar-visibility: auto`: the bar is hidden until the box scrolls,
/// stays for the fade's delay after the last scroll, then fades out over its
/// duration; the pointer over it brings it back, to be grabbed.
#[test]
fn auto_scrollbars_show_while_used_and_fade() {
    let mut ui = Ui::new();
    let root = ui.root();
    let list = ui.add_box(
        root,
        style! {
            height: 100px; overflow: auto; scrollbar-width: auto;
            scrollbar-visibility: auto; scrollbar-fade: 500ms 0.2s;
        },
    );
    ui.add_box(list, style! { height: 1000px; });
    layout(&mut ui);
    assert_eq!(ui.style(list).scrollbar_fade, (0.5, 0.2));
    ui.step(0.0);
    assert_eq!(ui.scrollbar_opacity(list), 0.0, "hidden until used");
    let painted = |ui: &Ui| {
        ui.scene()
            .items
            .iter()
            .filter(|i| matches!(i, threers_ui::scene::Item::Rect { .. }))
            .count()
    };
    let hidden = painted(&ui);

    ui.scroll_to(list, 50.0);
    assert!(ui.step(1.0 / 60.0), "frames wanted while it shows");
    assert_eq!(ui.scrollbar_opacity(list), 1.0);
    assert!(painted(&ui) > hidden, "the thumb paints");
    for _ in 0..29 {
        ui.step(1.0 / 60.0);
    }
    assert_eq!(ui.scrollbar_opacity(list), 1.0, "through the delay");
    ui.step(0.1 + 1.0 / 60.0);
    assert!(
        (ui.scrollbar_opacity(list) - 0.5).abs() < 0.1,
        "half faded: {}",
        ui.scrollbar_opacity(list)
    );
    ui.step(0.2);
    assert_eq!(ui.scrollbar_opacity(list), 0.0);
    layout(&mut ui);
    assert!(
        !ui.step(1.0 / 60.0) && !ui.needs_frame(),
        "no frames once faded"
    );
    assert_eq!(painted(&ui), hidden);

    // The pointer over where the bar is shows it.
    let bar = ui.scrollbar(list).expect("geometry, faded or not").thumb;
    let mut input: Input<Vec<String>> = Input::new();
    input.dispatch(
        &mut ui,
        &mut vec![],
        Event::PointerMove {
            x: bar.x + 2.0,
            y: bar.y + 2.0,
        },
    );
    assert_eq!(ui.scrollbar_opacity(list), 1.0);

    // `always`, the default, is not affected.
    let other = ui.add_box(
        root,
        style! { height: 100px; overflow: auto; scrollbar-width: auto; },
    );
    ui.add_box(other, style! { height: 1000px; });
    layout(&mut ui);
    assert_eq!(ui.scrollbar_opacity(other), 1.0);
}

/// Flicks in a row build speed, so a list of any length can be crossed:
/// 2.6 million pixels (100,000 rows) in a handful of flicks. One flick alone
/// still glides only so far, and flicking back starts over.
#[test]
fn flicks_in_a_row_build_speed_to_cross_a_long_list() {
    let setup = || {
        let mut ui = Ui::new();
        let root = ui.root();
        let list = ui.add_box(root, style! { height: 600px; overflow: auto; });
        ui.add_box(list, style! { height: 2600000px; });
        layout(&mut ui);
        (ui, list)
    };
    // One flick of 3,000 px/s, starting at `t`; returns when it lifted.
    let flick = |input: &mut Input<Vec<String>>, ui: &mut Ui, id: u64, t: f64, up: bool| {
        let mut log = vec![];
        let (y0, dy) = if up { (500.0, -48.0) } else { (50.0, 48.0) };
        touch(input, ui, &mut log, id, TouchPhase::Start, 10.0, y0, t);
        for i in 1..=6 {
            touch(
                input,
                ui,
                &mut log,
                id,
                TouchPhase::Move,
                10.0,
                y0 + dy * i as f32,
                t + i as f64 * 0.016,
            );
        }
        let end = t + 0.106;
        touch(
            input,
            ui,
            &mut log,
            id,
            TouchPhase::End,
            10.0,
            y0 + dy * 6.0,
            end,
        );
        end
    };
    let glide = |input: &mut Input<Vec<String>>, ui: &mut Ui, from: f64, seconds: f64| {
        let mut log = vec![];
        let frames = (seconds * 60.0) as usize;
        for i in 1..=frames {
            input.tick(ui, &mut log, from + i as f64 / 60.0);
        }
        from + frames as f64 / 60.0
    };

    // One flick: a few thousand pixels, as before.
    let (mut ui, list) = setup();
    let mut input: Input<Vec<String>> = Input::new();
    let end = flick(&mut input, &mut ui, 1, 0.0, true);
    glide(&mut input, &mut ui, end, 6.0);
    let one = ui.style(list).scroll_top;
    assert!(one > 1000.0 && one < 4500.0, "one flick: {one}");

    // Flick after flick: the end of the list.
    let (mut ui, list) = setup();
    let mut input: Input<Vec<String>> = Input::new();
    let mut t = 0.0;
    let mut flicks = 0;
    while ui.style(list).scroll_top < 2600000.0 - 600.0 && flicks < 20 {
        let end = flick(&mut input, &mut ui, flicks + 1, t, true);
        t = glide(&mut input, &mut ui, end, 0.25);
        flicks += 1;
    }
    t = glide(&mut input, &mut ui, t, 8.0);
    let _ = t;
    assert!(
        ui.style(list).scroll_top >= 2600000.0 - 601.0,
        "reached {} after {flicks}",
        ui.style(list).scroll_top
    );
    assert!(flicks <= 10, "{flicks} flicks");

    // Back the other way: no build-up carried over.
    let (mut ui, list) = setup();
    let mut input: Input<Vec<String>> = Input::new();
    ui.scroll_to(list, 1_000_000.0);
    let mut t = 0.0;
    for k in 0..4 {
        let end = flick(&mut input, &mut ui, k + 1, t, true);
        t = glide(&mut input, &mut ui, end, 0.25);
    }
    let before = ui.style(list).scroll_top;
    let end = flick(&mut input, &mut ui, 9, t, false);
    glide(&mut input, &mut ui, end, 6.0);
    let back = before - ui.style(list).scroll_top;
    assert!(
        back > 0.0 && back < 4500.0,
        "one flick's worth back: {back}"
    );
}

/// Pull to refresh: dragging down on a list at its top opens a pull instead
/// of scrolling; let go short of the threshold it springs back, past it the
/// handler runs and the pull holds open until the refresh is finished. A
/// list scrolled down scrolls back up first.
#[test]
fn pulling_a_list_down_from_its_top_refreshes_it() {
    use threers_ui::input::{PullPhase, PULL_THRESHOLD};
    let mut ui = Ui::new();
    let root = ui.root();
    let list = ui.add_box(root, style! { height: 100px; overflow: auto; });
    ui.add_box(list, style! { height: 1000px; });
    layout(&mut ui);
    let mut input: Input<Vec<String>> = Input::new();
    input.on_refresh(list, |log, _| log.push("refresh".into()));
    let mut log = vec![];
    let drag = |input: &mut Input<Vec<String>>,
                ui: &mut Ui,
                log: &mut Vec<String>,
                id: u64,
                to: f32,
                t: f64| {
        touch(input, ui, log, id, TouchPhase::Start, 10.0, 10.0, t);
        for i in 1..=10 {
            touch(
                input,
                ui,
                log,
                id,
                TouchPhase::Move,
                10.0,
                10.0 + to * i as f32 / 10.0,
                t + i as f64 * 0.03,
            );
        }
        touch(
            input,
            ui,
            log,
            id,
            TouchPhase::End,
            10.0,
            10.0 + to,
            t + 0.5,
        );
    };
    let settle = |input: &mut Input<Vec<String>>, ui: &mut Ui, log: &mut Vec<String>, from: f64| {
        for i in 1..=60 {
            input.tick(ui, log, from + i as f64 / 60.0);
        }
    };

    // A short pull: open while pulled, less than the finger moved, and back.
    touch(
        &mut input,
        &mut ui,
        &mut log,
        1,
        TouchPhase::Start,
        10.0,
        10.0,
        0.0,
    );
    touch(
        &mut input,
        &mut ui,
        &mut log,
        1,
        TouchPhase::Move,
        10.0,
        70.0,
        0.1,
    );
    let p = input.pull(list).expect("pulled");
    assert!(
        p.distance > 10.0 && p.distance < 60.0 && p.phase == PullPhase::Pulling,
        "{p:?}"
    );
    assert_eq!(ui.style(list).scroll_top, 0.0);
    touch(
        &mut input,
        &mut ui,
        &mut log,
        1,
        TouchPhase::End,
        10.0,
        70.0,
        0.4,
    );
    settle(&mut input, &mut ui, &mut log, 0.4);
    assert_eq!(input.pull(list), None, "sprung back");
    assert!(log.is_empty());

    // Far enough: armed, then refreshing on release, held at the threshold.
    drag(&mut input, &mut ui, &mut log, 2, 300.0, 2.0);
    assert_eq!(log, ["refresh"]);
    settle(&mut input, &mut ui, &mut log, 2.5);
    let p = input.pull(list).expect("held open");
    assert_eq!(
        (p.phase, p.distance),
        (PullPhase::Refreshing, PULL_THRESHOLD)
    );
    assert!(!input.needs_frame(), "nothing moves while it waits");
    input.finish_refresh(list);
    assert!(input.needs_frame());
    settle(&mut input, &mut ui, &mut log, 3.5);
    assert_eq!(input.pull(list), None);

    // Scrolled down: the same drag scrolls back up, and pulls no further.
    ui.scroll_to(list, 50.0);
    log.clear();
    drag(&mut input, &mut ui, &mut log, 3, 40.0, 5.0);
    assert_eq!(ui.style(list).scroll_top, 10.0);
    assert_eq!(input.pull(list), None);
    assert!(log.is_empty());
}

/// A box restyled from an `auto` scrollbar to an `always` one (a container
/// query can do that) has nothing left to fade: frames stop.
#[test]
fn a_scrollbar_restyled_to_always_asks_for_no_frames() {
    let mut ui = Ui::new();
    let root = ui.root();
    let list = ui.add_box(root, style! { height: 100px; overflow: auto; scrollbar-width: auto; scrollbar-visibility: auto; });
    ui.add_box(list, style! { height: 1000px; });
    layout(&mut ui);
    ui.step(0.0);
    ui.scroll_to(list, 10.0);
    ui.step(0.016);
    assert_eq!(ui.scrollbar_opacity(list), 1.0);
    ui.style_mut(list).scrollbar_visibility = threers_ui::ScrollbarVisibility::Always;
    layout(&mut ui);
    ui.step(0.016);
    assert!(!ui.needs_frame());
}

/// Scrolling a virtual list re-measures no row already in view: each row
/// keeps its layout, so wrapped text is not wrapped again every frame.
#[test]
fn scrolling_a_list_measures_only_the_rows_that_arrive() {
    use threers_ui::virtual_list::{RowHeight, VirtualList};
    struct Counting(usize);
    impl threers_ui::TextMeasure for Counting {
        fn measure(&mut self, text: &str, style: &threers_ui::Style) -> (f32, f32) {
            self.0 += 1;
            FixedWidthText(7.0).measure(text, style)
        }
    }
    let mut ui = Ui::new();
    let root = ui.root();
    let mut list = VirtualList::new(
        &mut ui,
        root,
        style! { height: 300px; width: 200px; },
        1000,
        RowHeight::Measured { estimate: 40.0 },
    );
    let build = |ui: &mut Ui, row: NodeId, i: usize| {
        let b = ui.add_box(row, style! { white-space: normal; padding: 4px; });
        ui.add_text(
            b,
            "words that wrap over a few lines in a narrow list ".repeat(1 + i % 3),
        );
    };
    let mut m = Counting(0);
    ui.layout(400.0, 400.0, &mut m);
    while list.sync(&mut ui, build) {
        ui.layout(400.0, 400.0, &mut m);
    }
    // A pixel at a time: in a step where no row arrives, nothing is measured.
    let c = list.container();
    let mut quiet = 0;
    for _ in 0..40 {
        let before: Vec<usize> = list.built().iter().map(|b| b.0).collect();
        m.0 = 0;
        let top = ui.style(c).scroll_top;
        ui.scroll_to(c, top + 1.0);
        ui.layout(400.0, 400.0, &mut m);
        list.sync(&mut ui, build);
        ui.layout(400.0, 400.0, &mut m);
        let after: Vec<usize> = list.built().iter().map(|b| b.0).collect();
        if before == after {
            assert_eq!(m.0, 0, "rows measured again while scrolling");
            quiet += 1;
        }
    }
    assert!(quiet > 20, "{quiet} steps with no row arriving");
}

/// Swipe to delete: a finger moving sideways on a row slides it; a short
/// slide springs back, a longer one holds the row open on its action, a tap
/// elsewhere only closes it, and a slide past half the row deletes it. A
/// finger moving up or down still scrolls.
#[test]
fn swiping_a_row_left_opens_its_action_or_deletes_it() {
    use threers_ui::input::{SwipePhase, SWIPE_ACTION_WIDTH};
    let mut ui = Ui::new();
    let root = ui.root();
    let list = ui.add_box(
        root,
        style! { width: 300px; height: 200px; overflow: auto; },
    );
    let rows: Vec<NodeId> = (0..20)
        .map(|_| ui.add_box(list, style! { height: 40px; }))
        .collect();
    let button = ui.add_box(rows[3], style! { width: 10px; height: 10px; });
    layout(&mut ui);
    let mut input: Input<Vec<String>> = Input::new();
    input.on_swipe_delete(list, |log, _, row| log.push(format!("delete {row:?}")));
    input.on_click(button, |log, _, _| log.push("button".into()));
    input.on_click(rows[1], |log, _, _| log.push("row 1 clicked".into()));
    let mut log = vec![];
    let slide = |input: &mut Input<Vec<String>>,
                 ui: &mut Ui,
                 log: &mut Vec<String>,
                 id: u64,
                 y: f32,
                 by: f32,
                 t: f64| {
        touch(input, ui, log, id, TouchPhase::Start, 250.0, y, t);
        for i in 1..=10 {
            touch(
                input,
                ui,
                log,
                id,
                TouchPhase::Move,
                250.0 + by * i as f32 / 10.0,
                y + 1.0,
                t + i as f64 * 0.02,
            );
        }
        touch(
            input,
            ui,
            log,
            id,
            TouchPhase::End,
            250.0 + by,
            y + 1.0,
            t + 0.25,
        );
    };
    let settle = |input: &mut Input<Vec<String>>, ui: &mut Ui, log: &mut Vec<String>, from: f64| {
        for i in 1..=40 {
            input.tick(ui, log, from + i as f64 / 60.0);
        }
    };

    // Under the finger it follows: row 1 (y 40..80).
    touch(
        &mut input,
        &mut ui,
        &mut log,
        1,
        TouchPhase::Start,
        250.0,
        60.0,
        0.0,
    );
    touch(
        &mut input,
        &mut ui,
        &mut log,
        1,
        TouchPhase::Move,
        220.0,
        61.0,
        0.05,
    );
    let s = input.swipe(list).expect("sliding");
    assert_eq!(
        (s.row, s.offset, s.phase),
        (rows[1], -30.0, SwipePhase::Dragging)
    );
    // Short: back to closed.
    touch(
        &mut input,
        &mut ui,
        &mut log,
        1,
        TouchPhase::End,
        220.0,
        61.0,
        0.1,
    );
    settle(&mut input, &mut ui, &mut log, 0.1);
    assert_eq!(input.swipe(list), None);

    // Further: held open on the action.
    slide(&mut input, &mut ui, &mut log, 2, 60.0, -70.0, 1.0);
    settle(&mut input, &mut ui, &mut log, 1.3);
    let s = input.swipe(list).unwrap();
    assert_eq!((s.phase, s.offset), (SwipePhase::Open, -SWIPE_ACTION_WIDTH));
    // A tap on another row closes it, and clicks nothing.
    touch(
        &mut input,
        &mut ui,
        &mut log,
        3,
        TouchPhase::Start,
        100.0,
        140.0,
        2.0,
    );
    touch(
        &mut input,
        &mut ui,
        &mut log,
        3,
        TouchPhase::End,
        100.0,
        140.0,
        2.05,
    );
    settle(&mut input, &mut ui, &mut log, 2.05);
    assert_eq!(input.swipe(list), None);
    assert!(log.is_empty(), "{log:?}");

    // Open again; a tap on the action goes through as a click.
    slide(&mut input, &mut ui, &mut log, 4, 60.0, -70.0, 3.0);
    settle(&mut input, &mut ui, &mut log, 3.3);
    touch(
        &mut input,
        &mut ui,
        &mut log,
        5,
        TouchPhase::Start,
        280.0,
        60.0,
        4.0,
    );
    touch(
        &mut input,
        &mut ui,
        &mut log,
        5,
        TouchPhase::End,
        280.0,
        60.0,
        4.05,
    );
    assert_eq!(log, ["row 1 clicked"]);
    log.clear();
    input.close_swipe();
    settle(&mut input, &mut ui, &mut log, 4.05);

    // Past half the row: deleted.
    slide(&mut input, &mut ui, &mut log, 6, 140.0, -200.0, 5.0);
    assert_eq!(log, [format!("delete {:?}", rows[3])]);
    assert_eq!(input.swipe(list), None);

    // Up and down still scrolls, and slides nothing.
    log.clear();
    touch(
        &mut input,
        &mut ui,
        &mut log,
        7,
        TouchPhase::Start,
        250.0,
        150.0,
        6.0,
    );
    for i in 1..=5 {
        touch(
            &mut input,
            &mut ui,
            &mut log,
            7,
            TouchPhase::Move,
            248.0,
            150.0 - 20.0 * i as f32,
            6.0 + i as f64 * 0.02,
        );
    }
    touch(
        &mut input,
        &mut ui,
        &mut log,
        7,
        TouchPhase::End,
        248.0,
        50.0,
        6.5,
    );
    assert!(ui.style(list).scroll_top > 50.0);
    assert_eq!(input.swipe(list), None);
}

/// Removing a row from the middle of a virtual list moves the rows after it
/// up one, keeping their nodes and measured heights.
#[test]
fn removing_a_list_row_keeps_the_rest() {
    use threers_ui::virtual_list::{RowHeight, VirtualList};
    let mut ui = Ui::new();
    let root = ui.root();
    let mut list = VirtualList::new(
        &mut ui,
        root,
        style! { height: 200px; width: 200px; },
        100,
        RowHeight::Measured { estimate: 30.0 },
    );
    // Row i is 20 + i px tall.
    let build = |ui: &mut Ui, row: NodeId, i: usize| {
        let b = ui.add_box(row, Style::default());
        ui.style_mut(b).height = threers_ui::Length::Px(20.0 + i as f32);
    };
    layout(&mut ui);
    while list.sync(&mut ui, build) {
        layout(&mut ui);
    }
    let before: Vec<(usize, NodeId)> = list.built().to_vec();
    let total = list.content_height();
    list.remove(&mut ui, 2);
    assert_eq!(list.len(), 99);
    assert_eq!(
        list.row(2),
        Some(before[3].1),
        "row 3 is row 2 now, the same node"
    );
    assert!(!ui.contains(before[2].1));
    assert_eq!(
        list.content_height(),
        total - 22.0,
        "row 2 was 22 px; the rest keep theirs"
    );
    assert_eq!(list.offset_of(3), 20.0 + 21.0 + 23.0);
    layout(&mut ui);
    let y = |n: NodeId| ui.rect(n).y;
    assert_eq!(
        y(before[3].1) - y(before[1].1),
        21.0,
        "moved up into the gap"
    );
}
