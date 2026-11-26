//! [`PieceSet`]: pieces described from state, as a [`View`] is. It keeps
//! their trees, sizes each to what is in it, gives the events on one to its
//! controls, and lays them out; an app hands it what it is asked.
//!
//! ```no_run
//! use threers::ui::binding;
//! use threers::ui::view::{button, slider, text};
//! use threers::{Scene, Vector3};
//! use threers_ios::{Align, App, Glass, Piece, PieceIn, PieceSet};
//!
//! #[derive(Default)]
//! struct State { level: f32, n: u32 }
//!
//! struct Mine { pieces: PieceSet<State> }
//!
//! impl App for Mine {
//!     fn setup(_: &mut Scene) -> Self {
//!         let mut pieces = PieceSet::new(State::default());
//!         pieces.glass(Glass::liquid(22.0));
//!         pieces.add(|s: &State| vec![text(format!("Pressed {}", s.n)), button("Press", |s: &mut State| s.n += 1)]);
//!         pieces.add(|_| vec![slider("Level", binding!(|s: State| s.level), 0.0..=1.0, 0.01)]).width(240.0);
//!         pieces.arrange(PieceIn::Room, |p| Piece::row(p, Vector3::new(0.0, 1.4, -1.2), Default::default(), 0.03, Align::Center));
//!         Mine { pieces }
//!     }
//!     threers_ios::piece_set!(pieces);
//! }
//! ```

use threers::overlay::Overlay;
use threers::ui::environment::Token;
use threers::ui::input::{Event, Input};
use threers::ui::view::{El, View};
use threers::ui::{style, AlignItems, Length, NodeId, OverlayPainter, Ui};
use threers::Vector3;

use crate::{Glass, Piece, PieceIn, Place, Surface};

type Body<S> = Box<dyn Fn(&S) -> Vec<El<S>>>;
type Arrange = Box<dyn Fn(&mut [Piece])>;

/// What is in a piece and how it is made, by [`PieceSet::add`].
pub struct PieceOf<S> {
    body: Body<S>,
    ui: Ui,
    input: Input<S>,
    view: View<S>,
    /// The box its content is in, which the piece is as large as.
    frame: NodeId,
    piece: Piece,
    width: Option<f32>,
    height: Option<f32>,
    padding: f32,
    gap: f32,
    align: crate::Align,
    glass: Option<Glass>,
    see_through: Option<f32>,
    dwell: Option<f32>,
    raised: Option<f32>,
    movable: Option<bool>,
    shown: bool,
    /// What it is laid out by is as it was last.
    fresh: bool,
}

impl<S: 'static> PieceOf<S> {
    /// As wide as this (UI pixels), not as its content is.
    pub fn width(&mut self, width: f32) -> &mut Self {
        self.width = Some(width);
        self.fresh = false;
        self
    }

    /// As tall as this: room for a menu or a calendar that opens in it.
    pub fn height(&mut self, height: f32) -> &mut Self {
        self.height = Some(height);
        self.fresh = false;
        self
    }

    /// Room round its content (16 UI pixels, unless said).
    pub fn padding(&mut self, padding: f32) -> &mut Self {
        self.padding = padding;
        self.fresh = false;
        self
    }

    /// Between one thing in it and the next (10 UI pixels, unless said).
    pub fn gap(&mut self, gap: f32) -> &mut Self {
        self.gap = gap;
        self.fresh = false;
        self
    }

    /// How what is in it lines up sideways: in its middle, unless said.
    pub fn align(&mut self, align: crate::Align) -> &mut Self {
        self.align = align;
        self.fresh = false;
        self
    }

    /// What it is made of, not what the rest are.
    pub fn glass(&mut self, glass: Glass) -> &mut Self {
        self.glass = Some(glass);
        self
    }

    /// Pressed by looking, after this many seconds ([`Piece::dwell`]), not
    /// as the rest are.
    pub fn dwell(&mut self, seconds: f32) -> &mut Self {
        self.dwell = Some(seconds);
        self
    }

    /// How see-through its glass is made, not as the rest's.
    pub fn see_through(&mut self, much: f32) -> &mut Self {
        self.see_through = Some(much);
        self
    }

    /// How far its controls stand out (metres), not as far as the rest's.
    pub fn raised(&mut self, metres: f32) -> &mut Self {
        self.raised = Some(metres);
        self
    }

    /// Whether it can be taken and moved, whatever the rest can.
    pub fn movable(&mut self, on: bool) -> &mut Self {
        self.movable = Some(on);
        self
    }

    /// Where it stands (metres), if no [`arrange`](PieceSet::arrange) says.
    pub fn position(&mut self, position: Vector3) -> &mut Self {
        self.piece.position = position;
        self
    }

    /// In the volume, or in the room (the room, unless said).
    pub fn at(&mut self, at: PieceIn) -> &mut Self {
        self.piece.at = at;
        self
    }

    /// Metres a UI pixel (0.8 mm, unless said).
    pub fn metres_per_pixel(&mut self, metres: f32) -> &mut Self {
        self.piece.metres_per_pixel = metres;
        self
    }

    /// It turns to face you.
    pub fn faces_you(&mut self, on: bool) -> &mut Self {
        self.piece.faces_you = on;
        self
    }

    /// Whether it is there.
    pub fn shown(&mut self, on: bool) -> &mut Self {
        self.shown = on;
        self
    }

    /// Its id, as [`Place::Piece`] says it.
    pub fn id(&self) -> u32 {
        self.piece.id
    }

    /// Its tree, for what a description cannot say.
    pub fn ui(&mut self) -> &mut Ui {
        &mut self.ui
    }

    fn restyle(&mut self) {
        if std::mem::replace(&mut self.fresh, true) {
            return;
        }
        let (w, h, p, g, a) = (self.width, self.height, self.padding, self.gap, self.align);
        self.ui.update_style(self.frame, |s| {
            s.width = w.map_or(Length::Auto, Length::Px);
            s.height = h.map_or(Length::Auto, Length::Px);
            s.padding = threers::ui::Edges::all(p);
        });
        let root = self.view.root();
        self.ui.update_style(root, |s| {
            s.gap = g;
            s.align_items = match a {
                crate::Align::Start => AlignItems::FlexStart,
                crate::Align::Center => AlignItems::Center,
                crate::Align::End => AlignItems::FlexEnd,
            };
        });
    }

    fn describe(&mut self, state: &S) {
        let body = (self.body)(state);
        self.view.update_with(&mut self.ui, &mut self.input, state, body);
        // none wider than the piece (a row of tabs at a large text size):
        // centred, it would be cut off at both edges
        let root = self.view.root();
        for c in self.ui.children(root).to_vec() {
            if self.ui.style(c).max_width == Length::Auto {
                self.ui.update_style(c, |s| s.max_width = Length::Percent(100.0));
            }
        }
    }
}

/// Pieces described from a state they share; see the [module docs](self).
pub struct PieceSet<S> {
    /// The state they show and change.
    pub state: S,
    all: Vec<PieceOf<S>>,
    arrange: Vec<(PieceIn, Arrange)>,
    glass: Glass,
    see_through: f32,
    dwell: f32,
    raised: f32,
    movable: bool,
    /// Text measured with the fonts it is drawn in.
    measure: Overlay,
    /// Its own clock, for an app that does not [`step`](Self::step) it:
    /// stepped by it since the last frame, and when it last stepped.
    clock: std::time::Instant,
    stepped: bool,
    last: f64,
}

impl<S: 'static> PieceSet<S> {
    pub fn new(state: S) -> Self {
        PieceSet { state, all: vec![], arrange: vec![], glass: Glass::None, see_through: 0.0, dwell: 0.0, raised: 0.0, movable: false, measure: Overlay::new(1, 1), clock: std::time::Instant::now(), stepped: false, last: 0.0 }
    }

    /// What they are made of (nothing, unless said here or of one by itself).
    pub fn glass(&mut self, glass: Glass) -> &mut Self {
        self.glass = glass;
        self
    }

    /// Pressed by looking, after this many seconds ([`Piece::dwell`]).
    pub fn dwell(&mut self, seconds: f32) -> &mut Self {
        self.dwell = seconds;
        self
    }

    /// How see-through their glass is made ([`Piece::see_through`]).
    pub fn see_through(&mut self, much: f32) -> &mut Self {
        self.see_through = much;
        self
    }

    /// How far their controls stand out, in metres ([`Piece::raised`]).
    pub fn raised(&mut self, metres: f32) -> &mut Self {
        self.raised = metres;
        self
    }

    /// Whether they can be taken and moved ([`Piece::movable`]).
    pub fn movable(&mut self, on: bool) -> &mut Self {
        self.movable = on;
        self
    }

    /// A piece showing what `body` says, as large as that is, in the room.
    pub fn add(&mut self, body: impl Fn(&S) -> Vec<El<S>> + 'static) -> &mut PieceOf<S> {
        let (mut ui, input) = (Ui::new(), Input::new());
        let root = ui.root();
        ui.update_style(root, |s| *s = style! { display: flex; flex-direction: column; align-items: flex-start; });
        // as large as what is in it, or as it is told
        let frame = ui.add_box(
            root,
            style! { display: flex; flex-direction: column; justify-content: center; align-items: stretch; box-sizing: border-box; flex-shrink: 0; },
        );
        let view = View::new(&mut ui, frame, "display: flex; flex-direction: column");
        ui.token_style(frame, &[("color", Token::Text)]);
        let id = self.all.len() as u32;
        self.all.push(PieceOf {
            body: Box::new(body),
            ui,
            input,
            view,
            frame,
            piece: Piece::new(id, PieceIn::Room, 1.0, 1.0, Vector3::ZERO),
            width: None,
            height: None,
            padding: 16.0,
            gap: 10.0,
            align: crate::Align::Center,
            glass: None,
            see_through: None,
            dwell: None,
            raised: None,
            movable: None,
            shown: true,
            fresh: false,
        });
        self.all.last_mut().expect("just added")
    }

    /// The piece of this id.
    pub fn get(&mut self, id: u32) -> Option<&mut PieceOf<S>> {
        self.all.get_mut(id as usize)
    }

    /// Lays out those in `at`, in the order they were added, each time they
    /// are asked for (so a size that changes moves the rest):
    /// [`Piece::row`], [`Piece::column`], [`Piece::around`], or your own.
    pub fn arrange(&mut self, at: PieceIn, how: impl Fn(&mut [Piece]) + 'static) -> &mut Self {
        self.arrange.retain(|a| a.0 != at);
        self.arrange.push((at, Box::new(how)));
        self
    }

    /// For [`App::pieces`](crate::App::pieces), once a frame: their
    /// animations moved on by its own clock unless the app
    /// [`step`](Self::step)ped them since the last.
    pub fn pieces(&mut self) -> Vec<Piece> {
        if !std::mem::take(&mut self.stepped) {
            let now = self.clock.elapsed().as_secs_f64();
            let dt = (now - self.last).clamp(0.0, 0.25) as f32;
            self.step(now, dt);
            self.stepped = false;
        }
        let mut out = vec![];
        for p in self.all.iter_mut().filter(|p| p.shown) {
            p.restyle();
            p.describe(&self.state);
            // its own size, then (a surface is whole pixels) a size it fits
            p.ui.layout(4096.0, 4096.0, &mut OverlayPainter::with_scale(&mut self.measure, 3.0));
            let r = p.ui.rect(p.frame);
            (p.piece.width, p.piece.height) = (r.width.ceil().max(1.0), r.height.ceil().max(1.0));
            p.piece.glass = p.glass.unwrap_or(self.glass);
            p.piece.see_through = p.see_through.unwrap_or(self.see_through);
            p.piece.dwell = p.dwell.unwrap_or(self.dwell);
            p.piece.raised = p.raised.unwrap_or(self.raised);
            p.piece.movable = p.movable.unwrap_or(self.movable);
            // something behind it where there is no glass
            if p.piece.glass == Glass::None {
                p.ui.token_style(p.frame, &[("background", Token::Surface)]);
                p.ui.update_style(p.frame, |s| s.border_radius = Length::Px(20.0));
            } else {
                p.ui.clear_token_style(p.frame, "background");
            }
            out.push(p.piece);
        }
        for (at, how) in &self.arrange {
            let mut some: Vec<Piece> = out.iter().filter(|p| p.at == *at).copied().collect();
            how(&mut some);
            for p in some {
                if let Some(o) = out.iter_mut().find(|o| o.id == p.id) {
                    *o = p;
                }
            }
        }
        out
    }

    /// For [`App::ui`](crate::App::ui): a piece's; `None` for any other
    /// place.
    pub fn ui(&mut self, surface: &Surface) -> Option<&mut Ui> {
        let Place::Piece(id) = surface.place else { return None };
        let p = self.all.get_mut(id as usize).filter(|p| p.shown)?;
        p.restyle();
        p.describe(&self.state);
        Some(&mut p.ui)
    }

    /// For [`App::ui_event`](crate::App::ui_event).
    pub fn ui_event(&mut self, place: Place, event: Event, time: f64) -> bool {
        let Place::Piece(id) = place else { return false };
        let Some(p) = self.all.get_mut(id as usize) else { return false };
        p.input.tick(&mut p.ui, &mut self.state, time);
        p.input.dispatch(&mut p.ui, &mut self.state, event);
        p.describe(&self.state);
        true
    }

    /// For [`App::ui_text`](crate::App::ui_text).
    pub fn ui_text(&mut self, place: Place, text: String) -> bool {
        let Place::Piece(id) = place else { return false };
        let Some(p) = self.all.get_mut(id as usize) else { return false };
        let n = text.chars().count();
        let did = p.input.set_focused_text(&mut p.ui, &mut self.state, text, n, n);
        p.describe(&self.state);
        did
    }

    /// Moves their animations on by `dt` seconds: in
    /// [`App::update`](crate::App::update), with the frame's (else the set
    /// steps itself, each frame, by its own clock).
    pub fn step(&mut self, time: f64, dt: f32) {
        self.stepped = true;
        self.last = self.clock.elapsed().as_secs_f64();
        for p in &mut self.all {
            p.input.tick(&mut p.ui, &mut self.state, time);
            p.ui.step(dt);
        }
    }
}

/// An [`App`](crate::App)'s `pieces`, `ui`, `ui_event` and `ui_text`, by the
/// [`PieceSet`] in its field: in the `impl App`, `threers_ios::piece_set!(field);`.
/// An app with other UI too writes the four and asks the set first. The set
/// animates itself; an app may [`step`](PieceSet::step) it with its frame.
///
/// ```compile_fail
/// struct A;
/// impl A { threers_ios::piece_set!(); }
/// ```
#[macro_export]
macro_rules! piece_set {
    ($field:ident $(,)?) => {
        fn pieces(&mut self) -> ::std::vec::Vec<$crate::Piece> {
            self.$field.pieces()
        }
        fn ui(&mut self, surface: &$crate::Surface) -> ::core::option::Option<&mut $crate::threers::ui::Ui> {
            self.$field.ui(surface)
        }
        fn ui_event(&mut self, place: $crate::Place, event: $crate::threers::ui::input::Event, time: f64) -> bool {
            self.$field.ui_event(place, event, time)
        }
        fn ui_text(&mut self, place: $crate::Place, text: ::std::string::String) -> bool {
            self.$field.ui_text(place, text)
        }
    };
    ($($other:tt)*) => {
        ::core::compile_error!(::core::concat!(
            "piece_set! takes the name of the app's `PieceSet` field, `piece_set!(pieces)`, not `",
            ::core::stringify!($($other)*),
            "`"
        ));
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::volume::{Volume, PIECES, PIECE_FLOATS};
    use crate::{Align, App};
    use threers::ui::view::{button, text};
    use threers::Scene;

    #[derive(Default)]
    struct N(u32, String);

    struct Set {
        pieces: PieceSet<N>,
    }

    impl App for Set {
        fn setup(_: &mut Scene) -> Self {
            let mut pieces = PieceSet::new(N::default());
            pieces.glass(Glass::liquid(20.0)).see_through(0.5).raised(0.01).movable(true);
            pieces.add(|s: &N| vec![text(format!("{} {}", s.0, "wide ".repeat(s.0 as usize))), button("Go", |s: &mut N| s.0 += 1)]);
            pieces.add(|s: &N| vec![text(format!("n {}", s.0))]).width(200.0).height(90.0).raised(0.0);
            pieces.add(|_| vec![text("off")]).shown(false);
            pieces.arrange(PieceIn::Room, |p| Piece::row(p, Vector3::new(0.0, 1.5, -1.0), Default::default(), 0.05, Align::Center));
            Set { pieces }
        }
        crate::piece_set!(pieces);
    }

    /// A set sizes its pieces, centres what is in them, lays them out, and
    /// one state serves them all.
    #[test]
    fn a_set_keeps_its_pieces() {
        let mut v = Volume::<Set>::new();
        v.frame();
        let p = v.pieces().to_vec();
        assert_eq!(p.len(), 2 * PIECE_FLOATS, "the hidden one is not there");
        let (w, h) = (p[2], p[3]);
        assert_eq!(p[20..22], [0.01, 1.0], "raised, and can be moved");
        assert_eq!(p[PIECE_FLOATS + 20..][..2], [0.0, 1.0], "or as it says itself");
        assert_eq!(p[23], 0.5, "as see-through as the set is");
        assert!(w > 40.0 && w < 200.0 && h > 50.0 && h < 140.0, "as large as it needs: {w}×{h}");
        assert_eq!((p[PIECE_FLOATS + 2], p[PIECE_FLOATS + 3]), (200.0, 90.0), "or as told");
        let gap = |p: &[f32]| (p[PIECE_FLOATS + 5] - p[PIECE_FLOATS + 2] * p[PIECE_FLOATS + 4] / 2.0) - (p[5] + p[2] * p[4] / 2.0);
        assert!((gap(&p) - 0.05).abs() < 1e-4);
        assert!(v.ui(PIECES, (w * 2.0) as u32, (h * 2.0) as u32, 2.0).is_some());
        assert!(v.ui(PIECES + 1, 400, 180, 2.0).is_some());
        // in the middle, both ways
        let set = &mut v.app().pieces;
        let one = &mut set.all[1];
        let t = one.ui.children(one.view.root())[0];
        let r = one.ui.rect(t);
        assert!((r.x + r.width / 2.0 - 100.0).abs() < 1.0 && (r.y + r.height / 2.0 - 45.0).abs() < 1.0, "{r:?}");
        // pressed, all of them follow, and the first grows and the rest move
        let json = v.controls(PIECES).unwrap().to_owned();
        let key: u64 = json.split("\"key\":\"").nth(1).unwrap().split('"').next().unwrap().parse().unwrap();
        for _ in 0..3 {
            crate::threers_ios_press_on(PIECES as u32, key);
            v.ui(PIECES, (w * 2.0) as u32, (h * 2.0) as u32, 2.0);
        }
        assert_eq!(v.app().pieces.state.0, 3);
        v.frame();
        let q = v.pieces().to_vec();
        assert!(q[2] > w + 20.0, "{} from {w}", q[2]);
        assert!((gap(&q) - 0.05).abs() < 1e-4);
        assert!(v.app().pieces.get(1).is_some() && v.app().pieces.get(9).is_none());
    }
}
