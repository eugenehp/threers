//! The kit's components, each a piece by itself, worked as the visionOS
//! shell works them: pressed by key, slid and scrolled by touch, typed into
//! through a native field.
#![cfg(any(target_os = "macos", target_os = "ios", target_os = "visionos"))]

use std::collections::HashMap;

use threers::ui::binding;
use threers::ui::environment::{Token, Tokens};
use threers::ui::input::{Event, Input};
use threers::ui::view::*;
use threers::ui::Ui;
use threers::{Scene, Vector3};
use threers_ios::volume::{Volume, PIECES, PIECE_FLOATS};
use threers_ios::{App, Glass, Piece, PieceIn, Place, Surface};

#[derive(Default)]
struct S {
    n: u32,
    on: bool,
    pick: Option<usize>,
    open: Option<usize>,
    page: usize,
    level: f32,
    span: (f32, f32),
    stars: u32,
    text: String,
    tags: Vec<String>,
    day: Option<Day>,
    month: Option<Day>,
}

/// Every one, by its id: a name, and its UI from the state.
type Make = fn(&S) -> Vec<El<S>>;
const ALL: &[(&str, Make)] = &[
    ("button", |s| vec![button(format!("Go {}", s.n), |s: &mut S| s.n += 1)]),
    ("toggle", |s| vec![toggle("On", s.on, |s: &mut S, v| s.on = v)]),
    ("checkbox", |s| vec![checkbox("On", s.on, |s: &mut S, v| s.on = v)]),
    ("radio", |s| vec![radio_group("Pick", [("One", "1"), ("Two", "2"), ("Three", "3")], s.pick, |s: &mut S, i| s.pick = Some(i))]),
    ("segmented", |s| vec![segmented(["One", "Two", "Three"], s.pick, |s: &mut S, i| s.pick = Some(i))]),
    ("picker", |s| vec![picker(["One", "Two", "Three"], s.pick, |s: &mut S, i| s.pick = Some(i))]),
    ("picker menu", |s| vec![picker_menu("Pick", vec!["One".into(), "Two".into(), "Three".into()], s.pick, |s: &mut S, i| s.pick = Some(i))]),
    ("menu", |_| vec![menu("More", None, vec!["One".into(), "Two".into(), "Three".into()], |s: &mut S, i| s.pick = Some(i))]),
    ("stepper", |s| vec![stepper(s.n.to_string(), ("Less", "More"), Some(Box::new(|s: &mut S| s.n = s.n.saturating_sub(1))), Some(Box::new(|s: &mut S| s.n += 1)))]),
    ("slider", |_| vec![slider("Level", binding!(|s: S| s.level), 0.0..=1.0, 0.01)]),
    ("range", |_| vec![range_slider("Span", ("From", "To"), binding!(|s: S| s.span), 0.0..=100.0, 1.0)]),
    ("rating", |s| vec![rating("Stars", s.stars, 5, Some(Box::new(|s: &mut S, n| s.stars = n)))]),
    ("pagination", |s| vec![pagination("Pages", s.page, 9, ("Back", "On"), |s: &mut S, p| s.page = p)]),
    ("accordion", |s| vec![accordion([("First", vec![text("one")]), ("Second", vec![text("two")])], s.open, |s: &mut S, o| s.open = o)]),
    ("breadcrumbs", |_| vec![breadcrumbs("Path", ["Home", "Work", "Here"], |s: &mut S, i| s.pick = Some(i))]),
    ("disclosure", |s| vec![disclosure_group("More", s.on, |s: &mut S, v| s.on = v, [text("inside")])]),
    ("chip", |_| vec![chip("Red", "Remove Red", |s: &mut S| s.n += 1)]),
    ("search", |_| vec![search_field("Search", binding!(|s: S| s.text))]),
    ("text", |_| vec![text_field("Name", binding!(|s: S| s.text))]),
    ("editor", |_| vec![text_editor("Notes", binding!(|s: S| s.text), 3)]),
    ("password", |_| vec![password_field("Password", ("Show", "Hide"), binding!(|s: S| s.text))]),
    ("tags", |_| vec![tag_input("Tags", "Remove {tag}", binding!(|s: S| s.tags))]),
    ("date", |_| vec![date_picker("Due", "No date", binding!(|s: S| s.day), Day::new(2026, 11, 24).unwrap(), CalendarWords::for_language("en-GB"))]),
    ("calendar", |s| {
        let shown = s.month.unwrap_or(Day::new(2026, 11, 24).unwrap());
        vec![calendar(shown, s.day, None, &CalendarWords::for_language("en-GB"), |s: &mut S, d| s.day = Some(d), |s: &mut S, d| s.month = Some(d))]
    }),
    ("scroll", |_| vec![scroll_view((0..40).map(|i| text(format!("Row {i}")))).css("height: 200px")]),
    ("gauge", |_| vec![gauge(0.6, "Load", "60%", 90.0)]),
    ("progress", |_| vec![progress_view(Some(0.4))]),
    ("chart", |_| vec![line_chart("Line", vec![1.0, 3.0, 2.0, 5.0], (200.0, 80.0))]),
    ("steps", |_| vec![step_indicator("Steps", ["One", "Two", "Three"], 1)]),
    ("timeline", |_| vec![timeline("Log", [("Made", "today", "9:00"), ("Sent", "today", "9:30")])]),
];

struct One {
    ui: Ui,
    input: Input<S>,
    view: View<S>,
    s: S,
    make: Make,
}

impl One {
    fn show(&mut self) {
        let body = (self.make)(&self.s);
        self.view.update_with(&mut self.ui, &mut self.input, &self.s, body);
    }
}

struct Kit(HashMap<u32, One>, Glass);

/// Wide enough for any; tall enough for what opens.
const W: u32 = 320;
const H: u32 = 420;

fn id_of(name: &str) -> u32 {
    ALL.iter().position(|a| a.0 == name).unwrap() as u32
}

impl App for Kit {
    fn setup(_: &mut Scene) -> Self {
        let one = |make: Make| {
            let (mut ui, input) = (Ui::new(), Input::new());
            let root = ui.root();
            let view = View::new(&mut ui, root, "display: flex; flex-direction: column; padding: 12px; width: 320px; height: 420px; box-sizing: border-box");
            ui.token_style(view.root(), &[("background", Token::Surface), ("color", Token::Text)]);
            let s = S { span: (20.0, 80.0), ..S::default() };
            let mut o = One { ui, input, view, s, make };
            o.show();
            o
        };
        Kit(ALL.iter().enumerate().map(|(i, a)| (i as u32, one(a.1))).collect(), Glass::None)
    }
    fn pieces(&mut self) -> Vec<Piece> {
        (0..ALL.len() as u32)
            .map(|i| {
                let at = if i % 2 == 0 { PieceIn::Room } else { PieceIn::Volume };
                let mut p = Piece::new(i, at, W as f32, H as f32, Vector3::ZERO);
                p.glass = self.1;
                p
            })
            .collect()
    }
    fn ui(&mut self, s: &Surface) -> Option<&mut Ui> {
        let Place::Piece(id) = s.place else { return None };
        let o = self.0.get_mut(&id)?;
        o.show();
        o.ui.step(1.0);
        Some(&mut o.ui)
    }
    fn ui_event(&mut self, place: Place, e: Event, time: f64) -> bool {
        let Place::Piece(id) = place else { return false };
        let o = self.0.get_mut(&id).unwrap();
        o.input.tick(&mut o.ui, &mut o.s, time);
        o.input.dispatch(&mut o.ui, &mut o.s, e);
        o.show();
        true
    }
    fn ui_text(&mut self, place: Place, text: String) -> bool {
        let Place::Piece(id) = place else { return false };
        let o = self.0.get_mut(&id).unwrap();
        let n = text.chars().count();
        let did = o.input.set_focused_text(&mut o.ui, &mut o.s, text, n, n);
        o.show();
        did
    }
}

/// A control as the shell reads it.
#[derive(Debug, Clone)]
struct Ctl {
    key: u64,
    role: String,
    label: String,
    rect: [f32; 4],
    json: String,
}

struct Shell {
    v: Volume<Kit>,
    controls: HashMap<u32, Vec<Ctl>>,
    pixels: HashMap<u32, Vec<u8>>,
}

const K: f32 = 2.0;

impl Shell {
    /// A frame of the shell's: every piece drawn and its controls read.
    fn frame(&mut self) {
        // (what opens is laid out at the next frame)
        for _ in 0..3 {
            self.v.frame();
            for i in 0..ALL.len() as u32 {
                let which = PIECES + i as usize;
                if let Some(p) = self.v.ui(which, (W as f32 * K) as u32, (H as f32 * K) as u32, K) {
                    self.pixels.insert(i, p.to_vec());
                }
                if let Some(j) = self.v.controls(which) {
                    self.controls.insert(i, parse(j));
                }
            }
        }
    }
    fn s(&mut self, name: &str) -> &mut S {
        &mut self.v.app().0.get_mut(&id_of(name)).unwrap().s
    }
    fn ctl(&self, name: &str, label: &str) -> Ctl {
        let all = &self.controls[&id_of(name)];
        all.iter().find(|c| c.label == label).unwrap_or_else(|| panic!("{name}: no {label:?} in {:?}", all.iter().map(|c| (&c.role, &c.label)).collect::<Vec<_>>())).clone()
    }
    fn has(&self, name: &str, label: &str) -> bool {
        self.controls[&id_of(name)].iter().any(|c| c.label == label)
    }
    fn press(&mut self, name: &str, label: &str) {
        let c = self.ctl(name, label);
        assert!(c.rect[0] >= 0.0 && c.rect[0] + c.rect[2] <= W as f32 + 0.5 && c.rect[1] >= 0.0 && c.rect[1] + c.rect[3] <= H as f32 + 0.5, "{name}: {label} is off the piece: {:?}", c.rect);
        threers_ios::threers_ios_press_on(PIECES as u32 + id_of(name), c.key);
        self.frame();
    }
    fn type_in(&mut self, name: &str, label: &str, text: &str) {
        let c = self.ctl(name, label);
        assert!(c.json.contains("\"edit\":true"), "{}", c.json);
        unsafe { threers_ios::threers_ios_text_on(PIECES as u32 + id_of(name), c.key, text.as_ptr(), text.len()) };
        self.frame();
    }
    fn drag(&mut self, name: &str, from: (f32, f32), to: (f32, f32)) {
        let which = PIECES as u32 + id_of(name);
        threers_ios::threers_ios_touch(which, 0, from.0 * K, from.1 * K);
        for i in 1..=10 {
            let t = i as f32 / 10.0;
            threers_ios::threers_ios_touch(which, 1, (from.0 + (to.0 - from.0) * t) * K, (from.1 + (to.1 - from.1) * t) * K);
        }
        threers_ios::threers_ios_touch(which, 2, to.0 * K, to.1 * K);
        self.frame();
    }
}

fn parse(j: &str) -> Vec<Ctl> {
    let field = |o: &str, f: &str| -> String {
        let at = o.find(&format!("\"{f}\":")).unwrap() + f.len() + 3;
        let v = &o[at..];
        match v.strip_prefix('"') {
            Some(t) => t[..t.find('"').unwrap()].to_owned(),
            None => v[..v.find([',', '}']).unwrap()].to_owned(),
        }
    };
    j.split("{\"key\"").skip(1).map(|o| format!("{{\"key\"{o}")).map(|o| Ctl {
        key: field(&o, "key").parse().unwrap(),
        role: field(&o, "role"),
        label: field(&o, "label"),
        rect: ["x", "y", "w", "h"].map(|f| field(&o, f).parse().unwrap()),
        json: o,
    }).collect()
}

#[test]
fn the_kits_components_work_as_pieces() {
    let mut h = Shell { v: Volume::new(), controls: HashMap::new(), pixels: HashMap::new() };
    h.frame();
    assert_eq!(h.v.pieces().len(), ALL.len() * PIECE_FLOATS);
    // Each draws something of its own.
    for (i, (name, _)) in ALL.iter().enumerate() {
        let px = &h.pixels[&(i as u32)];
        let first = &px[..4];
        assert!(px.chunks(4).any(|p| p != first), "{name} draws nothing");
    }
    h.press("button", "Go 0");
    assert_eq!(h.s("button").n, 1);
    assert!(h.has("button", "Go 1"));
    for name in ["toggle", "checkbox"] {
        h.press(name, "On");
        assert!(h.s(name).on, "{name}");
        assert!(h.ctl(name, "On").json.contains("\"on\":true"), "{name}");
        h.press(name, "On");
        assert!(!h.s(name).on, "{name}");
    }
    h.press("segmented", "Two");
    assert_eq!(h.s("segmented").pick, Some(1));
    h.press("picker", "Three");
    assert_eq!(h.s("picker").pick, Some(2));
    h.press("picker", "One");
    assert_eq!(h.s("picker").pick, Some(0));
    for (name, opener) in [("picker menu", "Pick"), ("menu", "More")] {
        h.press(name, opener);
        h.press(name, "Two");
        assert_eq!(h.s(name).pick, Some(1), "{name}");
        assert!(!h.has(name, "Three"), "{name} closes");
    }
    h.press("stepper", "More");
    h.press("stepper", "More");
    h.press("stepper", "Less");
    assert_eq!(h.s("stepper").n, 1);
    h.press("pagination", "On");
    assert_eq!(h.s("pagination").page, 1);
    h.press("accordion", "Second");
    assert_eq!(h.s("accordion").open, Some(1));
    h.press("breadcrumbs", "Work");
    assert_eq!(h.s("breadcrumbs").pick, Some(1));
    h.press("disclosure", "More");
    assert!(h.s("disclosure").on);
    h.press("chip", "Remove Red");
    assert_eq!(h.s("chip").n, 1);
    // slid
    let r = h.ctl("slider", "Level").rect;
    h.drag("slider", (r[0] + 4.0, r[1] + r[3] / 2.0), (r[0] + r[2] * 0.75, r[1] + r[3] / 2.0));
    let level = h.s("slider").level;
    assert!((level - 0.75).abs() < 0.08, "{level}");
    // what a shell shows solid: a slider's thumb where it is, none for stars
    let thumb = h.ctl("slider", "Level").json;
    let t: Vec<f32> = thumb.split("\"thumb\":[").nth(1).expect("a thumb").split(']').next().unwrap().split(',').map(|v| v.parse().unwrap()).collect();
    let at = (t[0] + t[2] / 2.0 - r[0]) / r[2];
    assert!((at - level).abs() < 0.06 && (t[2] - t[3]).abs() < 1.0, "{t:?} for {level}");
    assert!(h.ctl("range", "From").json.contains("\"thumb\":["));
    // a rail: the track, then its fill up to the thumb; a range's once, filled between its thumbs
    let rails = |h: &Shell, name: &str, label: &str| -> Vec<Vec<f32>> {
        let j = h.ctl(name, label).json;
        let all = j.split("\"rails\":[").nth(1).unwrap();
        let all = &all[..all.find("]}").map_or(0, |i| i + 1)];
        all.split("],[").map(|b| b.trim_matches(|c| c == '[' || c == ']').split(',').filter(|v| !v.is_empty()).map(|v| v.parse().unwrap()).collect()).filter(|b: &Vec<f32>| !b.is_empty()).collect()
    };
    let r = rails(&h, "slider", "Level");
    assert_eq!(r.len(), 2, "{r:?}");
    let (track, fill) = (&r[0], &r[1]);
    assert!(track[2] > fill[2] && (fill[0] + fill[2] - (t[0] + t[2] / 2.0)).abs() < t[2], "filled to the thumb: {r:?} {t:?}");
    let (from, to) = (rails(&h, "range", "From"), rails(&h, "range", "To"));
    assert_eq!((from.len(), to.len()), (2, 0), "{from:?} {to:?}");
    let (a, b) = (h.ctl("range", "From").rect, h.ctl("range", "To").rect);
    let f = &from[1];
    assert!((f[0] - (a[0] + a[2] / 2.0)).abs() < a[2] && (f[0] + f[2] - (b[0] + b[2] / 2.0)).abs() < b[2], "filled between: {f:?} {a:?} {b:?}");
    assert!(h.ctl("rating", "Stars").json.contains("\"thumb\":null"));
    // of a row that is on or off, its mark; of a button, all of it
    let solid = |h: &Shell, name: &str, label: &str| -> Vec<f32> {
        let j = h.ctl(name, label).json;
        j.split("\"solid\":[").nth(1).unwrap().split(']').next().unwrap().split(',').map(|v| v.parse().unwrap()).collect()
    };
    let s = solid(&h, "toggle", "On");
    assert_eq!((s[2], s[3], s[4]), (38.0, 22.0, 11.0), "the switch alone: {s:?}");
    for (name, label) in [("checkbox", "On"), ("radio", "Two")] {
        let s = solid(&h, name, label);
        assert!(s[2] < 30.0 && s[3] < 30.0, "{name}: {s:?}");
    }
    let (s, c) = (solid(&h, "accordion", "First"), h.ctl("accordion", "First").rect);
    assert_eq!(s[..4], c);
    let (s, c) = (solid(&h, "segmented", "One"), h.ctl("segmented", "One").rect);
    assert_eq!(s[..4], c);
    let mid = |r: [f32; 4]| (r[0] + r[2] / 2.0, r[1] + r[3] / 2.0);
    let (a, b) = (mid(h.ctl("range", "From").rect), mid(h.ctl("range", "To").rect));
    h.drag("range", a, (a.0 + 60.0, a.1));
    let (from, to) = h.s("range").span;
    assert!(from > 35.0 && from < 50.0 && to == 80.0, "{from} to {to}");
    h.drag("range", b, (b.0 - 30.0, b.1));
    let (from, to) = h.s("range").span;
    assert!(to > 60.0 && to < 75.0 && from < 50.0, "{from} to {to}");
    // a tap on the fourth star
    let r = h.ctl("rating", "Stars").rect;
    h.drag("rating", (r[0] + 3.5 * 26.0, r[1] + r[3] / 2.0), (r[0] + 3.5 * 26.0, r[1] + r[3] / 2.0));
    assert_eq!(h.s("rating").stars, 4);
    h.press("radio", "Three");
    assert_eq!(h.s("radio").pick, Some(2));
    // scrolled
    let before = h.pixels[&id_of("scroll")].clone();
    h.drag("scroll", (60.0, 180.0), (60.0, 40.0));
    assert_ne!(before, h.pixels[&id_of("scroll")], "it scrolls");
    // typed
    for (name, label) in [("search", "Search"), ("text", "Name"), ("editor", "Notes"), ("password", "Password")] {
        h.type_in(name, label, "hunter2");
        assert_eq!(h.s(name).text, "hunter2", "{name}");
    }
    let c = h.ctl("password", "Password");
    assert!(c.json.contains("\"secret\":true") && !c.json.contains("hunter2"), "{}", c.json);
    h.press("password", "Show");
    assert!(h.has("password", "Hide"));
    h.type_in("tags", "Tags", "red, green,");
    assert_eq!(h.s("tags").tags, ["red", "green"]);
    h.press("tags", "Remove red");
    assert_eq!(h.s("tags").tags, ["green"]);
    // picked from what opens
    h.press("date", "Due");
    h.press("date", "Next month");
    h.press("date", "December 25, 2026");
    assert_eq!(h.s("date").day, Day::new(2026, 12, 25));
    assert!(!h.has("date", "Next month"), "it closes");
    h.press("calendar", "Previous month");
    h.press("calendar", "October 5, 2026");
    assert_eq!(h.s("calendar").day, Day::new(2026, 10, 5));
    // Nothing done to one was done to another.
    assert_eq!((h.s("gauge").n, h.s("gauge").pick, h.s("scroll").on), (0, None, false));

    // On glass, all of it can be read, what opens too, over the lightest
    // glass and darker.
    h.v.app().1 = Glass::liquid(20.0);
    for (name, opener) in [("picker menu", "Pick"), ("menu", "More"), ("date", "Due")] {
        h.frame();
        h.press(name, opener);
    }
    h.frame();
    let mut bad = vec![];
    for (i, (name, _)) in ALL.iter().enumerate() {
        let o = h.v.app().0.get_mut(&(i as u32)).unwrap();
        assert_eq!(*o.ui.tokens(), Tokens::glass(), "{name}");
        let root = o.view.root();
        o.ui.token_style(root, &[("background", Token::Background)]);
        for ground in [Tokens::GLASS_GROUND, [0.2, 0.2, 0.2, 1.0], [0.0, 0.0, 0.0, 1.0]] {
            for issue in o.ui.contrast_issues(ground, false) {
                bad.push(format!("{name} on {}: {issue:?}", ground[0]));
            }
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
    // on pale glass, dark text, over the darkest and lighter
    h.v.app().1 = Glass::liquid_tinted(20.0, 0.95, 0.9, 0.7);
    h.frame();
    for (i, (name, _)) in ALL.iter().enumerate() {
        assert_eq!(h.v.pieces()[i * PIECE_FLOATS + 22], 1.0, "{name}");
        let o = h.v.app().0.get_mut(&(i as u32)).unwrap();
        assert_eq!(*o.ui.tokens(), Tokens::pale_glass(), "{name}");
        for ground in [Tokens::PALE_GLASS_GROUND, [0.8, 0.8, 0.8, 1.0], [1.0; 4]] {
            for issue in o.ui.contrast_issues(ground, false) {
                bad.push(format!("{name} on pale {}: {issue:?}", ground[0]));
            }
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
    // a colour of some depth, or a trace of a light one, is not pale
    assert!(!Glass::liquid_tinted(20.0, 0.05, 0.5, 0.15).is_pale() && !Glass::liquid(20.0).is_pale());
    h.v.app().1 = Glass::None;
    h.frame();
    assert_eq!(*h.v.app().0[&0].ui.tokens(), Tokens::default(), "off it, as it was");
}
