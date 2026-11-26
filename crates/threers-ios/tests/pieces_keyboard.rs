//! The UI's own on-screen keyboard on a piece: a text field pressed (not a
//! platform field over it) brings it up, its keys are controls a shell
//! presses and raises, and what they type lands in the field, readable on
//! every glass.
#![cfg(any(target_os = "macos", target_os = "ios", target_os = "visionos"))]

use threers::ui::binding;
use threers::ui::environment::{Token, Tokens};
use threers::ui::input::{Event, Input};
use threers::ui::native;
use threers::ui::view::*;
use threers::ui::virtual_keyboard::{KeyboardConfig, VirtualKeyboard};
use threers::ui::Ui;
use threers::{Scene, Vector3};
use threers_ios::volume::{Volume, PIECES};
use threers_ios::{App, Glass, Piece, PieceIn, Place, Surface};

#[derive(Default)]
struct S {
    text: String,
}

const W: f32 = 380.0;
const H: f32 = 400.0;

struct A {
    ui: Ui,
    input: Input<S>,
    view: View<S>,
    s: S,
    kb: VirtualKeyboard,
    glass: Glass,
    dwell: f32,
}

impl A {
    fn show(&mut self) {
        let body = vec![text_field("Message", binding!(|s: S| s.text)).css("width: 300px")];
        self.view.update_with(&mut self.ui, &mut self.input, &self.s, body);
    }
}

impl App for A {
    fn setup(_: &mut Scene) -> Self {
        let (mut ui, mut input) = (Ui::new(), Input::new());
        let root = ui.root();
        let view = View::new(&mut ui, root, "display: flex; flex-direction: column; padding: 16px; width: 380px; height: 400px; box-sizing: border-box");
        ui.token_style(view.root(), &[("color", Token::Text)]);
        let kb = VirtualKeyboard::with_config(&mut ui, &mut input, KeyboardConfig::iphone());
        kb.set_enabled(true);
        let mut a = A { ui, input, view, s: S::default(), kb, glass: Glass::liquid(20.0), dwell: 0.0 };
        a.show();
        a
    }
    fn pieces(&mut self) -> Vec<Piece> {
        let mut p = Piece::new(0, PieceIn::Volume, W, H, Vector3::ZERO);
        p.glass = self.glass;
        if self.dwell > 0.0 {
            // in the room, a metre ahead, typed on by looking
            (p.at, p.position, p.metres_per_pixel, p.dwell) = (PieceIn::Room, Vector3::new(0.0, 1.5, -1.0), 0.001, self.dwell);
        }
        vec![p]
    }
    fn ui(&mut self, s: &Surface) -> Option<&mut Ui> {
        matches!(s.place, Place::Piece(0)).then(|| {
            self.show();
            // (its keyboard rises: a frame's worth)
            self.ui.step(0.1);
            &mut self.ui
        })
    }
    fn ui_event(&mut self, _: Place, e: Event, time: f64) -> bool {
        self.input.tick(&mut self.ui, &mut self.s, time);
        self.input.dispatch(&mut self.ui, &mut self.s, e);
        self.show();
        true
    }
    fn ui_text(&mut self, _: Place, text: String) -> bool {
        let n = text.chars().count();
        self.input.set_focused_text(&mut self.ui, &mut self.s, text, n, n)
    }
}

struct Shell(Volume<A>);

impl Shell {
    fn frame(&mut self) {
        for _ in 0..3 {
            self.0.frame();
            self.0.ui(PIECES, W as u32, H as u32, 1.0);
        }
    }
    fn controls(&mut self) -> Vec<native::Control> {
        native::controls(&self.0.app().ui)
    }
    fn showing(&mut self) -> bool {
        let a = self.0.app();
        a.kb.showing(&a.ui)
    }
    fn press(&mut self, label: &str) {
        let all = self.controls();
        let c = all.iter().find(|c| c.label == label).unwrap_or_else(|| panic!("no {label:?} in {:?}", all.iter().map(|c| &c.label).collect::<Vec<_>>()));
        threers_ios::threers_ios_press_on(PIECES as u32, c.key);
        self.frame();
    }
}

#[test]
fn the_uis_keyboard_types_on_a_piece() {
    for (glass, tokens, grounds) in [
        (Glass::liquid(20.0), Tokens::glass(), [Tokens::GLASS_GROUND, [0.0, 0.0, 0.0, 1.0]]),
        (Glass::liquid_tinted(20.0, 0.95, 0.9, 0.7), Tokens::pale_glass(), [Tokens::PALE_GLASS_GROUND, [1.0; 4]]),
        (Glass::Bare, Tokens::glass(), [Tokens::GLASS_GROUND, [0.0, 0.0, 0.0, 1.0]]),
    ] {
        let mut sh = Shell(Volume::new());
        sh.0.app().glass = glass;
        sh.frame();
        assert_eq!(*sh.0.app().ui.tokens(), tokens, "{glass:?}");
        // the field: pressed, not a platform field over it, while the keyboard is on
        let field = sh.controls().into_iter().find(|c| c.label == "Message").unwrap();
        assert!(!field.editable, "{glass:?}: no platform keyboard's field");
        assert!(!sh.showing());
        sh.press("Message");
        assert!(sh.showing(), "{glass:?}: it comes up");
        // typed on, through the shell, as a look and a pinch would
        for k in ["h", "i", "space", "t", "h", "e", "r", "e"] {
            let label = if k == "space" { "space".to_owned() } else { k.to_owned() };
            let all = sh.controls();
            let c = all.iter().find(|c| c.label.eq_ignore_ascii_case(&label)).unwrap_or_else(|| panic!("{glass:?}: no key {label:?} in {:?}", all.iter().map(|c| &c.label).collect::<Vec<_>>()));
            threers_ios::threers_ios_press_on(PIECES as u32, c.key);
            sh.frame();
        }
        assert_eq!(sh.0.app().s.text.to_lowercase(), "hi there", "{glass:?}");
        // its keys: blocks a shell raises, on the piece, not too small to look at
        let keys: Vec<_> = sh.controls().into_iter().filter(|c| c.label.chars().count() == 1).collect();
        assert!(keys.len() >= 26, "{glass:?}: {} keys", keys.len());
        for k in &keys {
            let r = k.rect;
            assert!(r.x >= -0.5 && r.y >= -0.5 && r.x + r.width <= W + 0.5 && r.y + r.height <= H + 0.5, "{glass:?}: {:?} off the piece {r:?}", k.label);
            assert!(r.width >= 24.0 && r.height >= 24.0, "{glass:?}: {:?} is {}x{}", k.label, r.width, r.height);
            assert!(!k.editable && k.solid.width > 0.0);
        }
        // every word on it reads, on the glass's lightest and darkest
        let ui = &sh.0.app().ui;
        for g in grounds {
            let bad = ui.contrast_issues(g, false);
            assert!(bad.is_empty(), "{glass:?} on {:?}: {:?}", g, bad.iter().map(|i| (&i.text, i.ratio)).collect::<Vec<_>>());
        }
        // off, the field is a platform's again
        sh.0.app().kb.set_enabled(false);
        sh.frame();
        assert!(sh.controls().into_iter().find(|c| c.label == "Message").unwrap().editable);
    }
}

/// Typed on by looking: the head pointed at each key in turn, held there,
/// then away, types the word.
#[test]
fn the_keyboard_types_by_dwelling() {
    let mut sh = Shell(Volume::new());
    sh.0.app().dwell = 0.15;
    sh.frame();
    // the head a metre in front of the piece, looking at `(x, y)` of it
    let look = |x: f32, y: f32| {
        let at = Vector3::new((x - W / 2.0) * 0.001, 1.5 + (H / 2.0 - y) * 0.001, -1.0);
        let d = at - Vector3::new(0.0, 1.5, 0.0);
        let yaw = (-d.x).atan2(-d.z);
        let pitch = d.y.atan2((d.x * d.x + d.z * d.z).sqrt());
        let q = threers::math::Quaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), yaw)
            * threers::math::Quaternion::from_axis_angle(Vector3::new(1.0, 0.0, 0.0), pitch);
        threers_ios::threers_ios_head(0.0, 1.5, 0.0, q.x, q.y, q.z, q.w);
    };
    let dwell_on = |sh: &mut Shell, label: &str| {
        let all = sh.controls();
        let c = all.iter().find(|c| c.label == label).unwrap_or_else(|| panic!("no {label:?}"));
        let (x, y) = c.centre();
        look(x, y);
        sh.frame();
        std::thread::sleep(std::time::Duration::from_millis(200));
        look(x, y);
        sh.frame();
        sh.frame();
        // and away, to the ceiling
        threers_ios::threers_ios_head(0.0, 1.5, 0.0, 0.7071, 0.0, 0.0, 0.7071);
        sh.frame();
    };
    dwell_on(&mut sh, "Message");
    assert!(sh.showing(), "the field, looked at: the keyboard comes up");
    for k in ["h", "e", "l", "l", "o"] {
        dwell_on(&mut sh, k);
    }
    assert_eq!(sh.0.app().s.text, "hello", "a letter twice, looked at twice");
}
