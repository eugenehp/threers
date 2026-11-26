//! The rest of the kit on a piece's glass, in every theme a piece takes and
//! under each setting a person may have on: that it draws, reads, and keeps
//! its controls on the piece.
#![cfg(any(target_os = "macos", target_os = "ios", target_os = "visionos"))]

use std::rc::Rc;

use threers::ui::binding;
use threers::ui::components;
use threers::ui::environment::{LayoutDirection, Token, Tokens};
use threers::ui::image::Image;
use threers::ui::input::Input;
use threers::ui::native;
use threers::ui::notifications::{Notice, Notifications, Tone};
use threers::ui::palette::{open_palette, Command};
use threers::ui::sheet::{sheet, Detent};
use threers::ui::table::{data_table, TableColumn};
use threers::ui::view::*;
use threers::ui::Ui;
use threers::overlay::Overlay;
use threers::ui::OverlayPainter;

#[derive(Default)]
struct S {
    level: f32,
    at: (f32, f32),
    n: u32,
    on: bool,
    pick: Option<usize>,
    tab: usize,
    open: bool,
    text: String,
    path: Vec<String>,
    rows: Rc<Vec<Vec<String>>>,
    sort: Option<(usize, bool)>,
}

const PLUS: &str = r#"<path d="M12 5v14M5 12h14"/>"#;
const TINTS: [Tint; 5] = [Tint::Neutral, Tint::Accent, Tint::Success, Tint::Warning, Tint::Destructive];

type Make = fn(&S) -> Vec<El<S>>;
/// Each, and how many controls it has at the least.
const ALL: &[(&str, usize, Make)] = &[
    ("form", 2, |s| vec![form([section("About", [toggle("Show it", s.on, |s: &mut S, v| s.on = v), text_field("Name", binding!(|s: S| s.text)), labeled_content("Version", "1.2.3")])])]),
    ("group_box", 1, |_| vec![group_box("Box", [text("Inside it"), button("Go", |s: &mut S| s.n += 1)])]),
    ("inset_group", 1, |s| vec![inset_group_with_footer("Group", [setting_row("Wi-Fi", "Joins by itself", toggle("Wi-Fi", s.on, |s: &mut S, v| s.on = v))], "What it is for")]),
    ("list", 0, |_| vec![list((0..5).map(|i| text(format!("Row {i}"))))]),
    ("tree", 1, |s| {
        let mut all = TreeItem::new("a", "Folder");
        all.expanded = true;
        all.children = vec![TreeItem::new("b", "One"), TreeItem::new("c", "Two")];
        vec![tree("Files", &[all], Some(if s.on { "c" } else { "b" }), |s: &mut S, _| s.on = true, |s: &mut S, _, _| s.n += 1)]
    }),
    ("callouts", 0, |_| TINTS.map(|t| callout("Mind this, it matters", t)).into()),
    ("pills", 0, |_| TINTS.map(|t| pill("Beta", t)).into()),
    ("empty_state", 1, |_| vec![empty_state(PLUS, "Nothing here", "Add one to begin", Some(("Add".into(), Box::new(|s: &mut S| s.n += 1))))]),
    ("unavailable", 0, |_| vec![content_unavailable("?", "No results", "Try other words")]),
    ("confirmation", 2, |_| vec![confirmation("Delete it?", "Keep", "Delete", |s: &mut S| s.on = false, |s: &mut S| s.n += 1)]),
    ("skeleton", 0, |_| vec![skeleton("120px", 16.0), redacted(3, "Loading")]),
    ("stats", 0, |_| vec![stat("42", "Answers"), stat_tile("7", "Days"), hero(PLUS, "Title", "Under it", Some(("9".into(), "lives".into())))]),
    ("icon_button", 1, |_| vec![icon_button(PLUS, "Add", |s: &mut S| s.n += 1)]),
    ("toolbar", 2, |s| vec![control_group([toolbar_button(PLUS, "Bold", s.on, |s: &mut S| s.on = !s.on), toolbar_button(PLUS, "Italic", !s.on, |s: &mut S| s.on = !s.on)])]),
    ("link", 1, |_| vec![link("Read more", |s: &mut S| s.n += 1)]),
    ("faces", 0, |_| vec![avatar("AB", 40.0), avatar_group(["Ann Bee", "Cy Dee", "Ed Eff"], 2, 32.0), icon_label(PLUS, "Label"), key_caps(["⌘", "K"])]),
    ("image", 0, |_| vec![image("i", Image::new(2, 2, [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 255, 255, 255]].concat()).unwrap(), "width: 80px; height: 80px; border-radius: 12px")]),
    ("title_bar", 1, |_| vec![title_bar("Title", "Under it", [button("Done", |s: &mut S| s.n += 1)]), divider()]),
    ("grid", 3, |_| vec![adaptive_grid(80.0, 8.0, (0..3).map(|i| button(format!("B{i}"), |s: &mut S| s.n += 1)))]),
    ("disabled", 2, |s| vec![
        button("Off", |s: &mut S| s.n += 1).disabled(true),
        toggle("Off too", s.on, |s: &mut S, v| s.on = v).disabled(true),
        segmented(["One", "Two"], s.pick, |s: &mut S, i| s.pick = Some(i)),
    ]),
    ("destructive", 1, |_| vec![text("It failed").foreground(Token::Destructive), text("It worked").foreground(Token::Success), text("Careful").foreground(Token::Warning), text("Link").foreground(Token::AccentText), button("Prominent", |s: &mut S| s.n += 1).background(Token::Prominent).foreground(Token::OnProminent), button("Delete", |s: &mut S| s.n += 1).background(Token::DestructiveFill).foreground(Token::OnAccent)]),
    ("tabs", 3, |_| vec![El::host("tabs", |ui, input, s, at| {
        let t = components::tabs(ui, input, s, at, &["One", "Two", "Three"], binding!(|s: S| s.tab));
        for (i, p) in t.panels.iter().enumerate() {
            ui.add_text(*p, format!("Panel {i}"));
        }
    })]),
    ("table", 2, |_| vec![El::host("table", |ui, input, s, at| {
        data_table(ui, input, s, at, "Table", vec![TableColumn::new("Name", "1fr"), TableColumn::new("Size", "80px").numeric()], binding!(|s: S| s.rows), binding!(|s: S| s.sort), binding!(|s: S| s.pick), 28.0);
    }).css("height: 200px; display: flex; flex-direction: column")]),
    ("nav", 1, |_| vec![El::host("nav", |ui, input, s, at| {
        threers::ui::nav_stack::nav_stack(ui, input, s, at, "Home", binding!(|s: S| s.path), |ui, _, _, entry, screen| {
            ui.add_text(screen, format!("At {entry}"));
        }, |ui, _, _, screen| {
            let b = components::button(ui, screen, "Go in", components::ButtonKind::default());
            let _ = b;
        });
    }).css("height: 240px; display: flex; flex-direction: column")]),
    ("sheet", 0, |_| vec![text("Under it"), El::host("sheet", |_, input, _, _| {
        sheet(input, binding!(|s: S| s.open), vec![Detent::Medium], "Share", |ui, _, _, panel| {
            ui.add_text(panel, "Share to…");
            components::button(ui, panel, "Send", components::ButtonKind::default());
        });
    })]),
    ("notices", 1, |_| vec![El::host("notices", |ui, input, _, _| {
        let mut n = Notifications::new(ui);
        n.push(ui, input, Notice::new("Saved", "All of it").action("Undo").tone(Tone::Success).timeout(None), None);
        n.push(ui, input, Notice::new("Not sent", "Try again").tone(Tone::Error).timeout(None), None);
    })]),
    ("knob", 1, |_| vec![knob("Volume", binding!(|s: S| s.level), 0.0..=10.0, 1.0, 64.0)]),
    ("pad", 1, |_| vec![xy_pad("Aim", binding!(|s: S| s.at), 140.0)]),
    ("popover", 1, |_| vec![El::host("popover", |ui, _, _, at| {
        components::button(ui, at, "Opens", components::ButtonKind::default());
    })]),
];

const W: f32 = 320.0;
const H: f32 = 480.0;

struct One {
    ui: Ui,
    input: Input<S>,
    view: View<S>,
    s: S,
    overlay: Overlay,
    time: f64,
}

impl One {
    fn new(make: Make, tokens: Tokens, set: impl Fn(&mut Ui)) -> Self {
        let (mut ui, input) = (Ui::new(), Input::new());
        ui.set_tokens(tokens);
        set(&mut ui);
        let root = ui.root();
        let view = View::new(&mut ui, root, "display: flex; flex-direction: column; gap: 10px; padding: 16px; width: 320px; height: 480px; box-sizing: border-box");
        ui.token_style(view.root(), &[("color", Token::Text)]);
        let rows = Rc::new((0..20).map(|i| vec![format!("File {i}"), (i * 37 % 11).to_string()]).collect());
        let s = S { rows, ..S::default() };
        let mut o = One { ui, input, view, s, overlay: Overlay::new(W as u32, H as u32), time: 0.0 };
        o.ui.layout(W, H, &mut OverlayPainter::with_scale(&mut o.overlay, 1.0));
        o.show(make);
        o
    }
    fn show(&mut self, make: Make) {
        for _ in 0..4 {
            let body = make(&self.s);
            self.view.update_with(&mut self.ui, &mut self.input, &self.s, body);
            self.time += 1.0;
            self.input.tick(&mut self.ui, &mut self.s, self.time);
            self.ui.step(1.0);
            self.ui.layout(W, H, &mut OverlayPainter::with_scale(&mut self.overlay, 1.0));
        }
    }
    /// What reads badly on any of `grounds`, and what is off the piece.
    fn faults(&self, grounds: &[[f32; 4]], out: &mut Vec<String>, name: &str) {
        for g in grounds {
            for i in self.ui.contrast_issues(*g, false) {
                out.push(format!("{name} on {:.2}: {:?} {:.2} of {:.1}", g[0], i.text, i.ratio, i.required));
            }
        }
        for c in native::controls(&self.ui) {
            let r = c.rect;
            if r.x < -0.5 || r.y < -0.5 || r.x + r.width > W + 0.5 || r.y + r.height > H + 0.5 {
                out.push(format!("{name}: {:?} is off the piece {r:?}", c.label));
            }
            if r.width < 24.0 && r.height < 24.0 {
                out.push(format!("{name}: {:?} is small to look at {}x{}", c.label, r.width, r.height));
            }
        }
    }
}

const DARK: [[f32; 4]; 3] = [Tokens::GLASS_GROUND, [0.2, 0.2, 0.2, 1.0], [0.0, 0.0, 0.0, 1.0]];
const PALE: [[f32; 4]; 3] = [Tokens::PALE_GLASS_GROUND, [0.8, 0.8, 0.8, 1.0], [1.0; 4]];

fn themes() -> [(&'static str, Tokens, [[f32; 4]; 3]); 2] {
    [("glass", Tokens::glass(), DARK), ("pale", Tokens::pale_glass(), PALE)]
}

fn settings() -> Vec<(&'static str, fn(&mut Ui))> {
    vec![
        ("plain", |_| {}),
        ("large text", |ui| ui.set_text_scale(1.6)),
        ("bold", |ui| ui.set_bold_text(true)),
        ("contrast", |ui| ui.set_increase_contrast(true)),
        ("right to left", |ui| ui.set_layout_direction(LayoutDirection::RightToLeft)),
        ("less motion", |ui| ui.set_reduce_motion(true)),
    ]
}

/// Opens what opens, for the ones that do.
fn open(name: &str, o: &mut One, make: Make) {
    match name {
        "sheet" => o.s.open = true,
        "table" => {
            assert!(native::controls(&o.ui).iter().any(|c| c.label == "Size"), "a heading to sort by");
            o.s.pick = Some(1);
            o.show(make);
            o.s.pick = Some(2);
        }
        "tabs" => o.s.tab = 1,
        "tree" | "toolbar" | "form" | "inset_group" => o.s.on = true,
        "disabled" => o.s.pick = Some(1),
        "nav" => o.s.path = vec!["There".into()],
        "popover" => {
            let at = native::controls(&o.ui).iter().find(|c| c.label == "Opens").unwrap().node;
            components::popover(&mut o.ui, &mut o.input, &mut o.s, at, "More", |ui, panel| {
                ui.add_text(panel, "In the popover");
            });
        }
        _ => return,
    }
    o.show(make);
}

#[test]
fn the_rest_of_the_kit_reads_on_glass() {
    let mut bad = vec![];
    for (theme, tokens, grounds) in themes() {
        for (setting, set) in settings() {
            for (name, least, make) in ALL {
                let mut o = One::new(*make, tokens, set);
                let name_in = format!("{name} ({theme}, {setting})");
                let had = native::controls(&o.ui).len();
                if setting == "plain" && had < *least {
                    bad.push(format!("{name_in}: {had} controls"));
                }
                o.faults(&grounds, &mut bad, &name_in);
                open(name, &mut o, *make);
                o.faults(&grounds, &mut bad, &format!("{name_in} open"));
            }
        }
    }
    bad.dedup();
    assert!(bad.is_empty(), "{} faults\n{}", bad.len(), bad.join("\n"));
}

/// A palette opens inside a piece no wider than most.
#[test]
fn a_palette_fits_a_piece() {
    let make: Make = |_| vec![text("Under it")];
    let mut bad = vec![];
    for (theme, tokens, grounds) in themes() {
        let mut o = One::new(make, tokens, |_| {});
        open_palette(&mut o.ui, &mut o.input, &mut o.s, vec![Command::new("a", "Open file"), Command::new("b", "Close file")], |_, _, _| {});
        o.show(make);
        o.faults(&grounds, &mut bad, theme);
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// Scrolled, a control shows where it is seen and no more: what stands out
/// of the piece does not stand past the list's edge.
#[test]
fn scrolled_controls_are_cut_at_their_list() {
    let make: Make = |_| vec![scroll_view((0..30).map(|i| button(format!("B{i}"), |s: &mut S| s.n += 1))).css("height: 200px")];
    let mut o = One::new(make, Tokens::glass(), |_| {});
    let list = o.ui.children(o.view.root())[0];
    for step in 0..12 {
        o.ui.scroll_to(list, step as f32 * 13.0);
        o.ui.layout(W, H, &mut OverlayPainter::with_scale(&mut o.overlay, 1.0));
        let l = o.ui.rect(list);
        for c in native::controls(&o.ui) {
            for r in [c.rect, c.solid] {
                assert!(r.y >= l.y - 0.5 && r.y + r.height <= l.y + l.height + 0.5, "{step}: {} {r:?} past {l:?}", c.label);
            }
        }
    }
}

/// Pressed where the shell presses (a control's middle): a table's heading
/// sorts, its rows and a tree's are chosen, and each is the one pressed.
#[test]
fn rows_and_headings_are_pressed() {
    use threers::ui::input::{Event, PointerButton};
    let press = |o: &mut One, make: Make, label: &str| {
        let all = native::controls(&o.ui);
        let c = all.iter().find(|c| c.label.starts_with(label)).unwrap_or_else(|| panic!("no {label} in {:?}", all.iter().map(|c| &c.label).collect::<Vec<_>>()));
        let (x, y) = c.centre();
        for e in [Event::PointerMove { x, y }, Event::PointerDown { x, y, button: PointerButton::Primary }, Event::PointerUp { x, y, button: PointerButton::Primary }] {
            o.input.dispatch(&mut o.ui, &mut o.s, e);
        }
        o.show(make);
    };
    let make_of = |name: &str| ALL.iter().find(|a| a.0 == name).unwrap().2;
    let make = make_of("table");
    let mut o = One::new(make, Tokens::glass(), |_| {});
    press(&mut o, make, "File 2");
    assert_eq!(o.s.pick, Some(2));
    press(&mut o, make, "Size");
    assert_eq!(o.s.sort, Some((1, true)));
    let make = make_of("tree");
    let mut o = One::new(make, Tokens::glass(), |_| {});
    press(&mut o, make, "Two");
    assert!(o.s.on);
    let make = make_of("tabs");
    let mut o = One::new(make, Tokens::glass(), |_| {});
    press(&mut o, make, "Three");
    assert_eq!(o.s.tab, 2);
    let make = make_of("disabled");
    let mut o = One::new(make, Tokens::glass(), |_| {});
    assert!(native::controls(&o.ui).iter().filter(|c| c.label.starts_with("Off")).all(|c| c.disabled));
    press(&mut o, make, "Off");
    assert_eq!(o.s.n, 0);
}

/// The system's settings reach a piece's UI and a piece's glass, and what
/// the app set itself in between stays.
#[test]
fn the_systems_settings_reach_the_pieces() {
    use threers::{Scene, Vector3};
    use threers_ios::volume::{Volume, PIECES, PIECE_FLOATS};
    use threers_ios::{App, Glass, Piece, PieceIn, Place, Surface};
    struct A(Ui);
    impl App for A {
        fn setup(_: &mut Scene) -> Self {
            let mut ui = Ui::new();
            let t = ui.add_box(ui.root(), threers::ui::style! { font-size: 20px; });
            ui.add_text(t, "Words");
            A(ui)
        }
        fn pieces(&mut self) -> Vec<Piece> {
            let mut p = Piece::new(0, PieceIn::Volume, 200.0, 100.0, Vector3::ZERO);
            p.glass = Glass::liquid(20.0);
            p.see_through = 0.5;
            vec![p]
        }
        fn ui(&mut self, s: &Surface) -> Option<&mut Ui> {
            matches!(s.place, Place::Piece(0)).then_some(&mut self.0)
        }
    }
    let mut v: Volume<A> = Volume::new();
    let frame = |v: &mut Volume<A>| {
        v.frame();
        v.ui(PIECES, 200, 100, 1.0);
    };
    frame(&mut v);
    assert_eq!(v.pieces()[23], 0.5);
    threers_ios::threers_ios_legibility(true, true, true, true, 1.5);
    frame(&mut v);
    let ui = &v.app().0;
    assert!(ui.bold_text() && ui.increase_contrast() && ui.reduce_motion() && ui.text_scale() == 1.5);
    assert_eq!(v.pieces()[PIECE_FLOATS - 1], 0.0, "no glass thinned");
    // the app's own, until the system's change again
    v.app().0.set_text_scale(1.0);
    frame(&mut v);
    assert_eq!(v.app().0.text_scale(), 1.0);
    threers_ios::threers_ios_legibility(false, false, false, false, 1.0);
    frame(&mut v);
    let ui = &v.app().0;
    assert!(!ui.bold_text() && !ui.increase_contrast() && !ui.reduce_motion());
    assert_eq!(v.pieces()[23], 0.5);
}

/// At a large text size, nothing in a piece of a set width is wider than
/// it, and discs of initials hold them.
#[test]
fn large_text_stays_in_a_piece() {
    use threers_ios::{PieceSet, Place, Surface};
    let mut set = PieceSet::new(S::default());
    set.add(|s: &S| vec![El::host("tabs", |ui, input, s, at| {
        components::tabs(ui, input, s, at, &["One", "Two", "Three", "Four"], binding!(|s: S| s.tab));
    }), avatar_group(["Ann Bee", "Cy Dee"], 2, 28.0), avatar("AB", 32.0), icon(PLUS, 20.0, 2.0), text(format!("{}", s.n))]).width(200.0);
    set.get(0).unwrap().ui().set_text_scale(2.2);
    set.pieces();
    let ui = set.ui(&Surface { place: Place::Piece(0), width: 200.0, height: 400.0, scale: 1.0, insets: [0.0; 4], see_through: true }).unwrap();
    ui.layout(200.0, 400.0, &mut OverlayPainter::with_scale(&mut Overlay::new(200, 400), 1.0));
    let piece = ui.rect(ui.root());
    let mut boxes = vec![ui.root()];
    while let Some(b) = boxes.pop() {
        let r = ui.rect(b);
        if !matches!(ui.kind(b), threers::ui::NodeKind::Text(_)) && !ui.ancestors(b).skip(1).any(|a| ui.style(a).scroll_x || ui.style(a).scroll) {
            assert!(r.x >= piece.x - 0.5 && r.x + r.width <= piece.x + 200.5, "{b:?} {r:?}");
        }
        if let threers::ui::NodeKind::Text(t) = ui.kind(b) && t.len() == 2 {
            let disc = ui.rect(ui.parent(b).unwrap());
            assert!(r.width <= disc.width + 0.5, "{t}: {r:?} out of {disc:?}");
            assert!((disc.width - disc.height).abs() < 1.0 && disc.height > r.height && disc.width > 32.0, "{t}: round, round its letters {disc:?}");
        }
        if ui.style(b).flips_rtl || matches!(ui.kind(b), threers::ui::NodeKind::Path { .. }) {
            assert!(r.height > 20.0 * 1.8, "an icon grows with the text: {r:?}");
        }
        boxes.extend(ui.children(b));
    }
}

/// A bare piece has nothing behind its UI, takes the glass's colours, and
/// says so to the shell.
#[test]
fn bare_pieces_have_no_pane() {
    use threers::Scene;
    use threers_ios::volume::{Volume, PIECES};
    use threers_ios::{App, Glass, Piece, PieceSet, Place, Surface};
    struct A(PieceSet<S>);
    impl App for A {
        fn setup(_: &mut Scene) -> Self {
            let mut set = PieceSet::new(S::default());
            set.glass(Glass::Bare).raised(0.01);
            set.add(|s: &S| vec![button(format!("Go {}", s.n), |s: &mut S| s.n += 1)]);
            set.add(|s: &S| vec![toggle("On", s.on, |s: &mut S, v| s.on = v)]).glass(Glass::None);
            A(set)
        }
        fn pieces(&mut self) -> Vec<Piece> {
            self.0.pieces()
        }
        fn ui(&mut self, s: &Surface) -> Option<&mut Ui> {
            self.0.ui(s)
        }
    }
    let mut v: Volume<A> = Volume::new();
    v.frame();
    v.ui(PIECES, 100, 60, 1.0);
    v.ui(PIECES + 1, 100, 60, 1.0);
    assert_eq!(v.pieces()[13], 8.0, "bare");
    let set = &mut v.app().0;
    let bare = set.ui(&Surface { place: Place::Piece(0), width: 100.0, height: 60.0, scale: 1.0, insets: [0.0; 4], see_through: true }).unwrap();
    assert_eq!(*bare.tokens(), Tokens::glass());
    let root = bare.root();
    let frame = bare.children(root)[0];
    assert!(bare.style(frame).background.is_none_or(|c| c[3] == 0.0), "no pane");
    let plain = set.ui(&Surface { place: Place::Piece(1), width: 100.0, height: 60.0, scale: 1.0, insets: [0.0; 4], see_through: true }).unwrap();
    let frame = plain.children(plain.root())[0];
    assert!(plain.style(frame).background.is_some_and(|c| c[3] > 0.5), "a pane to read on");
}

/// Pressed from its native view, a control is not left looking hovered.
#[test]
fn a_press_leaves_no_hover() {
    use threers::Scene;
    use threers_ios::volume::{Volume, PIECES};
    use threers_ios::{App, Glass, Piece, PieceSet, Surface};
    struct A(PieceSet<S>);
    impl App for A {
        fn setup(_: &mut Scene) -> Self {
            let mut set = PieceSet::new(S::default());
            set.glass(Glass::Bare);
            set.add(|s: &S| vec![button(format!("Go {}", s.n), |s: &mut S| s.n += 1)]);
            A(set)
        }
        fn pieces(&mut self) -> Vec<Piece> {
            self.0.pieces()
        }
        fn ui(&mut self, s: &Surface) -> Option<&mut Ui> {
            self.0.ui(s)
        }
        fn ui_event(&mut self, place: threers_ios::Place, e: threers::ui::input::Event, time: f64) -> bool {
            self.0.ui_event(place, e, time)
        }
    }
    let mut v: Volume<A> = Volume::new();
    let mut draw = |v: &mut Volume<A>| {
        v.frame();
        v.ui(PIECES, 120, 60, 1.0).map(<[u8]>::to_vec)
    };
    draw(&mut v);
    let ui = v.app().0.get(0).unwrap().ui();
    let c = native::controls(ui)[0].clone();
    let look = |ui: &Ui| (ui.style(c.node).background, ui.style(c.node).opacity);
    let before = look(ui);
    threers_ios::threers_ios_press_on(PIECES as u32, c.key);
    draw(&mut v);
    draw(&mut v);
    assert_eq!(v.app().0.state.n, 1);
    assert_eq!(look(v.app().0.get(0).unwrap().ui()), before, "it looks as it did, not hovered");
}

/// Typed into on glass, a field reads (its text, its selection, its caret).
#[test]
fn a_typed_field_reads_on_glass() {
    let make: Make = |_| vec![text_field("Name", binding!(|s: S| s.text)), search_field("Search", binding!(|s: S| s.text))];
    let mut bad = vec![];
    for (theme, tokens, grounds) in themes() {
        let mut o = One::new(make, tokens, |_| {});
        for label in ["Name", "Search"] {
            let c = native::controls(&o.ui).into_iter().find(|c| c.label == label).unwrap();
            let (x, y) = c.centre();
            use threers::ui::input::{Event, PointerButton};
            for e in [Event::PointerDown { x, y, button: PointerButton::Primary }, Event::PointerUp { x, y, button: PointerButton::Primary }] {
                o.input.dispatch(&mut o.ui, &mut o.s, e);
            }
            o.input.set_focused_text(&mut o.ui, &mut o.s, "typed".into(), 5, 5);
            o.show(make);
            o.faults(&grounds, &mut bad, &format!("{label} ({theme})"));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// Typed into through the shell, a bare field is drawn as the UI has it:
/// its shade, not its focus or selection colour all over.
#[test]
fn a_field_typed_through_the_shell() {
    use threers::Scene;
    use threers_ios::volume::{Volume, PIECES};
    use threers_ios::{App, Glass, Piece, PieceSet, Place, Surface};
    struct A(PieceSet<S>);
    impl App for A {
        fn setup(_: &mut Scene) -> Self {
            let mut set = PieceSet::new(S::default());
            set.glass(Glass::Bare);
            set.add(|_| vec![text_field("Name", binding!(|s: S| s.text)).css("width: 200px")]);
            A(set)
        }
        fn pieces(&mut self) -> Vec<Piece> {
            self.0.pieces()
        }
        fn ui(&mut self, s: &Surface) -> Option<&mut Ui> {
            self.0.ui(s)
        }
        fn ui_event(&mut self, place: Place, e: threers::ui::input::Event, time: f64) -> bool {
            self.0.ui_event(place, e, time)
        }
        fn ui_text(&mut self, place: Place, text: String) -> bool {
            self.0.ui_text(place, text)
        }
    }
    let mut v: Volume<A> = Volume::new();
    v.frame();
    let (w, h) = { let p = v.pieces(); (p[2] as u32, p[3] as u32) };
    v.ui(PIECES, w, h, 1.0);
    let c = native::controls(v.app().0.get(0).unwrap().ui())[0].clone();
    unsafe { threers_ios::threers_ios_text_on(PIECES as u32, c.key, b"typed".as_ptr(), 5) };
    let mut px = vec![];
    for _ in 0..3 {
        v.frame();
        if let Some(p) = v.ui(PIECES, w, h, 1.0) {
            px = p.to_vec();
        }
    }
    assert_eq!(v.app().0.state.text, "typed");
    // the field's middle, right of the words: its shade, nothing bright
    let (x, y) = ((c.rect.x + c.rect.width * 0.8) as u32, (c.rect.y + c.rect.height / 2.0) as u32);
    let at = ((y * w + x) * 4) as usize;
    let p = &px[at..at + 4];
    assert!(p[3] < 120 && p[2] < 120, "{p:?}: lit all over");
}

/// What a shell makes solid of each control on a bare piece: a block for
/// what is pressed, a ball and a rail for what is slid (a range's rail
/// once), a rim for a field; all of it on the piece, where its control is.
#[test]
fn every_control_has_its_solid_shape() {
    #[derive(Default)]
    struct T { n: u32, on: bool, level: f32, span: (f32, f32), stars: u32, text: String }
    type Make = fn(&T) -> Vec<El<T>>;
    let all: [(&str, Make); 9] = [
        ("button", |s| vec![button(format!("Go {}", s.n), |s: &mut T| s.n += 1)]),
        ("toggle", |s| vec![toggle("On", s.on, |s: &mut T, v| s.on = v)]),
        ("checkbox", |s| vec![checkbox("Tick", s.on, |s: &mut T, v| s.on = v)]),
        ("slider", |_| vec![slider("Level", binding!(|s: T| s.level), 0.0..=1.0, 0.01).css("width: 200px")]),
        ("range", |_| vec![range_slider("Span", ("From", "To"), binding!(|s: T| s.span), 0.0..=100.0, 1.0).css("width: 200px")]),
        ("stepper", |s| vec![stepper(s.n.to_string(), ("Less", "More"), Some(Box::new(|s: &mut T| s.n = s.n.saturating_sub(1))), Some(Box::new(|s: &mut T| s.n += 1)))]),
        ("rating", |s| vec![rating("Stars", s.stars, 5, Some(Box::new(|s: &mut T, n| s.stars = n)))]),
        ("field", |_| vec![text_field("Name", binding!(|s: T| s.text)).css("width: 200px")]),
        ("search", |_| vec![search_field("Find", binding!(|s: T| s.text)).css("width: 200px")]),
    ];
    for (name, make) in all {
        let (mut ui, mut input) = (Ui::new(), Input::<T>::new());
        ui.set_tokens(Tokens::glass());
        let root = ui.root();
        let mut view = View::new(&mut ui, root, "display: flex; padding: 16px");
        let s = T { level: 0.3, span: (20.0, 70.0), stars: 2, ..T::default() };
        view.update_with(&mut ui, &mut input, &s, make(&s));
        ui.layout(320.0, 120.0, &mut OverlayPainter::with_scale(&mut Overlay::new(320, 120), 1.0));
        let piece = ui.rect(root);
        let inside = |r: threers::ui::Rect| r.x >= piece.x - 0.5 && r.y >= piece.y - 0.5 && r.x + r.width <= piece.x + piece.width + 0.5 && r.y + r.height <= piece.y + piece.height + 0.5;
        let cs = native::controls(&ui);
        assert!(!cs.is_empty(), "{name}");
        let mut rails = 0;
        for c in &cs {
            let what = format!("{name}: {}", c.label);
            assert!(inside(c.rect) && inside(c.solid), "{what}: off the piece");
            if c.slider {
                if let Some(t) = c.thumb {
                    assert!(inside(t) && (t.width - t.height).abs() < 1.0, "{what}: a round thumb {t:?}");
                    let (tx, ty) = (t.x + t.width / 2.0, t.y + t.height / 2.0);
                    assert!(c.rect.x - 1.0 <= tx && tx <= c.rect.x + c.rect.width + 1.0 && (ty - (c.rect.y + c.rect.height / 2.0)).abs() < c.rect.height, "{what}: its thumb on it");
                }
                for (r, k) in &c.rails {
                    assert!(inside(*r) && r.height <= 10.0 && k[3] > 0.3, "{what}: a rail {r:?} {k:?}");
                }
                rails += c.rails.len();
            } else if c.editable {
                assert!(c.rect.width > 100.0 && c.rect.height >= 24.0, "{what}: a field to sink {:?}", c.rect);
            } else {
                assert!(c.solid.width >= 10.0 && c.solid.height >= 10.0 && c.solid_radius >= 0.0, "{what}: a block {:?}", c.solid);
            }
        }
        match name {
            "slider" | "range" => assert_eq!(rails, 2, "{name}: a track and its fill, once"),
            // stars are slid, with neither ball nor rail
            "rating" => assert!(cs.iter().all(|c| c.thumb.is_none() && c.rails.is_empty()), "{name}"),
            "toggle" | "checkbox" => assert!(cs[0].solid.width < cs[0].rect.width, "{name}: its mark alone"),
            _ => {}
        }
    }
}

/// A line longer than its field scrolls to keep the caret, drawn, in view.
#[test]
fn a_long_line_keeps_its_caret_in_view() {
    use threers::ui::input::{Event, PointerButton};
    let make: Make = |_| vec![text_field("Message", binding!(|s: S| s.text)).css("width: 200px")];
    let mut o = One::new(make, Tokens::glass(), |_| {});
    let c = native::controls(&o.ui).into_iter().find(|c| c.label == "Message").unwrap();
    let (x, y) = c.centre();
    for e in [Event::PointerDown { x, y, button: PointerButton::Primary }, Event::PointerUp { x, y, button: PointerButton::Primary }] {
        o.input.dispatch(&mut o.ui, &mut o.s, e);
    }
    let long = "a line much longer than the field it is typed in, and longer still";
    let n = long.chars().count();
    o.input.set_focused_text(&mut o.ui, &mut o.s, long.into(), n, n);
    o.show(make);
    let field = o.ui.rect(c.node);
    let mut st = vec![c.node];
    let mut carets = vec![];
    while let Some(b) = st.pop() {
        let r = o.ui.rect(b);
        if o.ui.style(b).display != threers::ui::Display::None && r.width == 1.0 && r.height >= 12.0 && o.ui.style(b).background.is_some() {
            carets.push(r);
        }
        st.extend(o.ui.children(b));
    }
    assert_eq!(carets.len(), 1, "a caret, drawn");
    // where it shows: scrolled with its line
    let row = o.ui.children(c.node)[0];
    let (content, view, scrolled) = o.ui.scroll_extent_x(row);
    assert!(content > view && scrolled > 0.0, "scrolled along: {content} {view} {scrolled}");
    let at = carets[0].x;
    assert!(at >= field.x && at <= field.x + field.width, "the caret at {at}, the field {field:?}");
    assert_eq!(o.s.text, long);
}

/// Pressed by looking: the head pointed at a room piece's button presses it
/// once it has been pointed at long enough, and not again till it looks
/// away and back; where it points is the shell's to show.
#[test]
fn the_head_presses_by_dwelling() {
    use threers::math::Vector3;
    use threers::Scene;
    use threers_ios::volume::{Volume, PIECES};
    use threers_ios::{App, Glass, Piece, PieceSet, Place, Surface};
    struct A(PieceSet<S>);
    impl App for A {
        fn setup(_: &mut Scene) -> Self {
            let mut set = PieceSet::new(S::default());
            set.glass(Glass::Bare).dwell(0.3);
            set.add(|s: &S| vec![button(format!("Go {}", s.n), |s: &mut S| s.n += 1).css("width: 160px; height: 60px")])
                .position(Vector3::new(0.0, 1.5, -1.0));
            A(set)
        }
        fn pieces(&mut self) -> Vec<Piece> {
            self.0.pieces()
        }
        fn ui(&mut self, s: &Surface) -> Option<&mut Ui> {
            self.0.ui(s)
        }
        fn ui_event(&mut self, place: Place, e: threers::ui::input::Event, time: f64) -> bool {
            self.0.ui_event(place, e, time)
        }
    }
    let mut v: Volume<A> = Volume::new();
    let mut frame = |v: &mut Volume<A>, look: bool| {
        // straight ahead from 1 m in front of it, or up at the ceiling
        let q = if look { [0.0, 0.0, 0.0, 1.0] } else { [0.7071, 0.0, 0.0, 0.7071] };
        threers_ios::threers_ios_head(0.0, 1.5, 0.0, q[0], q[1], q[2], q[3]);
        v.frame();
        let (w, h) = (v.pieces()[2] as u32, v.pieces()[3] as u32);
        v.ui(PIECES, w, h, 1.0);
        v.gaze()
    };
    frame(&mut v, true);
    let g = frame(&mut v, true);
    assert_eq!(g[0], 1.0, "at its button: {g:?}");
    assert!((g[3] + 1.0).abs() < 1e-3 && (g[6] - 1.0).abs() < 1e-3, "on it, facing the head: {g:?}");
    assert_eq!(v.app().0.state.n, 0, "not yet");
    std::thread::sleep(std::time::Duration::from_millis(350));
    frame(&mut v, true);
    frame(&mut v, true);
    assert_eq!(v.app().0.state.n, 1, "pressed");
    std::thread::sleep(std::time::Duration::from_millis(350));
    frame(&mut v, true);
    frame(&mut v, true);
    assert_eq!(v.app().0.state.n, 1, "once");
    assert_eq!(frame(&mut v, false)[0], 0.0, "looking away");
    frame(&mut v, true);
    std::thread::sleep(std::time::Duration::from_millis(350));
    frame(&mut v, true);
    frame(&mut v, true);
    assert_eq!(v.app().0.state.n, 2, "and back");
    // moved by its bar a metre right: pointed at where it is shown
    threers_ios::threers_ios_piece_pose(0, 1.0, 1.5, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0);
    assert_eq!(frame(&mut v, true)[0], 0.0, "not where the app put it");
    let right = [0.0, (-std::f32::consts::FRAC_PI_8).sin(), 0.0, (-std::f32::consts::FRAC_PI_8).cos()];
    threers_ios::threers_ios_piece_pose(0, 1.0, 1.5, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0);
    threers_ios::threers_ios_head(0.0, 1.5, 0.0, right[0], right[1], right[2], right[3]);
    v.frame();
    assert!(v.gaze()[0] == 1.0 && (v.gaze()[1] - 1.0).abs() < 0.01, "where it was moved: {:?}", v.gaze());
}

/// The spatial controls: a knob turned by a drag and the keys, its dial's
/// turn for a shell to show; a pad that holds a point, its dot a thumb.
#[test]
fn a_knob_turns_and_a_pad_holds_a_point() {
    use threers::ui::input::{Event, PointerButton};
    let drag = |o: &mut One, make: Make, from: (f32, f32), to: (f32, f32)| {
        o.input.dispatch(&mut o.ui, &mut o.s, Event::PointerDown { x: from.0, y: from.1, button: PointerButton::Primary });
        o.input.dispatch(&mut o.ui, &mut o.s, Event::PointerMove { x: to.0, y: to.1 });
        o.input.dispatch(&mut o.ui, &mut o.s, Event::PointerUp { x: to.0, y: to.1, button: PointerButton::Primary });
        o.show(make);
    };
    let make: Make = |_| vec![knob("Volume", binding!(|s: S| s.level), 0.0..=10.0, 1.0, 64.0)];
    let mut o = One::new(make, Tokens::glass(), |_| {});
    let k = native::controls(&o.ui).into_iter().find(|c| c.label == "Volume").unwrap();
    assert!(k.slider && k.thumb.is_none(), "a slider, its mark no ball");
    let low = k.dial.expect("a dial");
    assert!((low + 135f32.to_radians()).abs() < 1e-4, "at the low end, 135° left: {low}");
    // dragged up 100 px: half a turn's worth
    let (x, y) = k.centre();
    drag(&mut o, make, (x, y), (x, y - 100.0));
    assert_eq!(o.s.level, 5.0);
    let mid = native::controls(&o.ui)[0].dial.unwrap();
    assert!(mid.abs() < 1e-4, "half way: straight up, {mid}");
    for (key, want) in [("ArrowUp", 6.0), ("PageDown", 0.0), ("End", 10.0)] {
        o.input.set_focus(&mut o.ui, &mut o.s, Some(k.node));
        o.input.dispatch(&mut o.ui, &mut o.s, Event::Key { key: key.into(), modifiers: Default::default() });
        assert_eq!(o.s.level, want, "{key}");
    }
    o.show(make);
    assert!((native::controls(&o.ui)[0].dial.unwrap() - 135f32.to_radians()).abs() < 1e-4);
    let make: Make = |_| vec![xy_pad("Aim", binding!(|s: S| s.at), 140.0)];
    let mut o = One::new(make, Tokens::glass(), |_| {});
    let p = native::controls(&o.ui).into_iter().find(|c| c.label == "Aim").unwrap();
    assert!(p.slider && p.dial.is_none());
    let t = p.thumb.expect("its dot, a ball to a shell");
    assert!((t.width - 20.0).abs() < 0.5);
    let r = p.rect;
    drag(&mut o, make, (r.x + 30.0, r.y + 30.0), (r.x + 10.0 + 0.75 * 120.0, r.y + 10.0 + 0.25 * 120.0));
    assert!((o.s.at.0 - 0.75).abs() < 0.01 && (o.s.at.1 - 0.25).abs() < 0.01, "{:?}", o.s.at);
    let t = native::controls(&o.ui)[0].thumb.unwrap();
    let (tx, ty) = (t.x + t.width / 2.0 - r.x, t.y + t.height / 2.0 - r.y);
    assert!((tx - (10.0 + 0.75 * 120.0)).abs() < 3.0 && (ty - (10.0 + 0.25 * 120.0)).abs() < 3.0, "the dot where it was put: {tx}, {ty}");
    o.input.set_focus(&mut o.ui, &mut o.s, Some(p.node));
    o.input.dispatch(&mut o.ui, &mut o.s, Event::Key { key: "ArrowDown".into(), modifiers: Default::default() });
    o.show(make);
    assert!((o.s.at.1 - 0.30).abs() < 0.01);
    let said = o.ui.semantics(p.node).unwrap().value.clone().unwrap();
    assert_eq!(said, "75%, 30%");
}

/// German, Japanese and Arabic, as long as they come.
#[test]
fn long_words_stay_on_the_piece() {
    let make: Make = |_| vec![
        button("Geschwindigkeitsbegrenzungsanzeige einschalten und ausschalten", |s: &mut S| s.n += 1),
        toggle("Donaudampfschifffahrtsgesellschaftskapitän benachrichtigen", true, |s: &mut S, v| s.on = v),
        segmented(["Einstellungen", "Benachrichtigungen", "Datenschutz"], Some(0), |s: &mut S, i| s.pick = Some(i)),
        picker_menu("Wählen", vec!["日本語の長い選択肢 🎌".into(), "العربية".into()], Some(0), |s: &mut S, i| s.pick = Some(i)),
        pill("Datenschutzgrundverordnung", Tint::Accent),
    ];
    let mut bad = vec![];
    for (theme, tokens, grounds) in themes() {
        for (setting, set) in settings() {
            let o = One::new(make, tokens, set);
            o.faults(&grounds, &mut bad, &format!("long ({theme}, {setting})"));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}
