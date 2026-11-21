//! Gesture typing and next-word predictions on the threers on-screen
//! keyboard, in the browser: a message field, the keyboard docked under
//! it, a trained swipe model reading each swipe after the words before it,
//! and an n-gram model offering the next word in the bar.
//!
//! ```text
//! web/swipe-demo/build.sh            # then SERVE=1 to serve web/ on :8787
//! ```
//!
//! The page fetches the models from `web/swipe-demo/models/` (the swipe
//! model and its words, then each next-word model with its share of the
//! mix) and hands them over ([`Demo::load_swipe`], [`Demo::add_ngram`],
//! [`Demo::add_neural`], [`Demo::start`]); until then the keyboard types
//! taps only.

use std::cell::RefCell;
use std::rc::Rc;

use threers_swipe::prelude::*;
use threers_ui::prelude::*;
use threers_ui::virtual_keyboard::{KeyboardConfig, KeyboardLayout, VirtualKeyboard};

/// The share of the next-word mix that is what the person typed: chosen
/// on text of a style the language's models had not seen.
pub const USER_WEIGHT: f32 = 0.2;

/// The page, apart from the browser: built, fed events and laid out the
/// same natively (for tests) as in a page.
pub struct Demo {
    pub ui: Ui,
    pub input: Input<TextField>,
    pub field: TextField,
    pub keyboard: VirtualKeyboard,
    target: NodeId,
    text: NodeId,
    status: NodeId,
    /// What the models said they are, for the status line.
    loaded: Rc<RefCell<Option<String>>>,
    /// The swipe model and its words, until [`start`](Self::start).
    swipe: Option<(Model, Lexicon)>,
    /// The next-word models given so far, their weights and what they are.
    next: Vec<(Box<dyn WordModel>, f32, String)>,
    /// What this person has typed, learnt; the page keeps it.
    pub user: Rc<UserModel>,
    shown: (String, String),
    last: Option<f64>,
}

impl Demo {
    pub fn new() -> Self {
        let mut ui = Ui::new();
        let mut input: Input<TextField> = Input::new();
        let root = ui.root();
        ui.set_style(
            root,
            style! { display: flex; flex-direction: column; padding: 18px 16px; gap: 10px; font-family: system-ui; font-size: 15px; background: #f2f2f7; color: #1c1c1e; },
        );
        let title = ui.add_box(root, style! { font-size: 22px; font-weight: bold; });
        ui.add_text(title, "Swipe and next word");
        let hint = ui.add_box(root, style! { font-size: 13px; color: #6c6c70; white-space: normal; line-height: 18px; });
        ui.add_text(
            hint,
            "Draw a word across the letters, or tap. The bar offers the next word from the words before it, as an iPhone's does.",
        );
        let target = ui.add_box(
            root,
            style! { min-height: 44px; border-radius: 18px; padding: 11px 14px; box-sizing: border-box; background: #ffffff; border: 1px solid #d1d1d6; font-size: 17px; white-space: normal; line-height: 22px; },
        );
        let text = ui.add_text(target, "▏");
        ui.set_semantics(target, Semantics::new(Role::TextBox).label("Message"));
        input.on_key(target, |f: &mut TextField, _, key, m, _| {
            !matches!(f.key(key, m, |_| true), FieldKey::Ignored)
        });
        let status_box = ui.add_box(root, style! { font-size: 12px; color: #8e8e93; white-space: normal; });
        let status = ui.add_text(status_box, "Loading the models…");
        let keyboard = VirtualKeyboard::with_config(&mut ui, &mut input, KeyboardConfig::iphone());
        keyboard.set_enabled(true);
        keyboard.set_layouts(&mut ui, vec![KeyboardLayout::english()]);
        keyboard.set_context(|f: &TextField, _| Some(f.before_caret()));
        let mut field = TextField::new("");
        input.set_focus(&mut ui, &mut field, Some(target));
        Self {
            ui,
            input,
            field,
            keyboard,
            target,
            text,
            status,
            loaded: Rc::default(),
            swipe: None,
            next: vec![],
            user: Rc::default(),
            shown: (String::new(), String::new()),
            last: None,
        }
    }

    /// The swipe model's safetensors and its word list (`word count`
    /// lines).
    pub fn load_swipe(&mut self, swipe: &[u8], words: &str) -> Result<(), String> {
        self.swipe = Some((Model::from_safetensors(swipe)?, Lexicon::parse(words)));
        Ok(())
    }

    /// An n-gram model's file (binary or text) for the next words, its
    /// share of the mix `weight`.
    pub fn add_ngram(&mut self, bytes: &[u8], weight: f32) -> Result<(), String> {
        let m = NgramModel::from_bytes(bytes)?;
        let about = format!("a {}-gram of {} words", m.order(), m.vocab_len());
        self.next.push((Box::new(m), weight, about));
        Ok(())
    }

    /// A neural model's safetensors for the next words, its share `weight`.
    pub fn add_neural(&mut self, bytes: &[u8], weight: f32) -> Result<(), String> {
        let m = NeuralLm::from_safetensors(bytes)?;
        let about = format!("a neural model of {} words", m.vocab_len());
        self.next.push((Box::new(m), weight, about));
        Ok(())
    }

    /// The keyboard decoding swipes, and with next-word models given, the
    /// bar predicting from them mixed; what it runs, for the status line.
    pub fn start(&mut self) -> Result<String, String> {
        let (model, lexicon) = self.swipe.take().ok_or("no swipe model: load_swipe first")?;
        let mut about = format!("{} words to swipe", lexicon.len());
        let mut typing = SwipeTyping::new(model).language("en", lexicon);
        if !self.next.is_empty() {
            let parts: Vec<String> = self
                .next
                .iter()
                .map(|(_, w, a)| format!("{a} ({:.0}%)", 100.0 * w))
                .collect();
            about += &format!(" · next words from {}", parts.join(", "));
            let mix = self
                .next
                .drain(..)
                .fold(Interpolated::new(), |m, (lm, w, _)| m.with(lm, w));
            typing = typing.next_words("en", mix);
        }
        typing = typing.user("en", self.user.clone(), USER_WEIGHT);
        typing.attach(&self.keyboard);
        // Read the field again, so the bar offers its first words now.
        self.keyboard.set_context(|f: &TextField, _| Some(f.before_caret()));
        *self.loaded.borrow_mut() = Some(about.clone());
        Ok(about)
    }

    /// A frame at `time` (seconds) in a `width` × `height` window.
    pub fn frame(&mut self, time: f64, width: f32, height: f32, measure: &mut dyn TextMeasure) {
        let dt = self.last.map_or(0.0, |l| (time - l).max(0.0));
        self.last = Some(time);
        self.input.tick(&mut self.ui, &mut self.field, time);
        self.ui.step(dt as f32);
        let text = format!("{}▏{}", self.field.before_caret(), &self.field.text[self.field.before_caret().len()..]);
        if text != self.shown.0 {
            self.ui.set_text(self.text, text.clone());
            self.shown.0 = text;
        }
        let status = self.loaded.borrow().clone().map_or_else(
            || "Loading the models…".into(),
            |s| format!("{s} · {} words learnt from you", self.user.len()),
        );
        if status != self.shown.1 {
            self.ui.set_text(self.status, status.clone());
            self.shown.1 = status;
        }
        self.ui.layout(width, height, measure);
    }

    pub fn event(&mut self, e: Event) {
        // The keyboard keeps the message field's focus whatever is tapped.
        self.input.dispatch(&mut self.ui, &mut self.field, e);
        if self.input.focused() != Some(self.target) {
            self.input
                .set_focus(&mut self.ui, &mut self.field, Some(self.target));
        }
    }

    pub fn pointer(&mut self, kind: &str, x: f32, y: f32) {
        let button = PointerButton::Primary;
        match kind {
            "down" => {
                self.event(Event::PointerMove { x, y });
                self.event(Event::PointerDown { x, y, button });
            }
            "up" => self.event(Event::PointerUp { x, y, button }),
            _ => self.event(Event::PointerMove { x, y }),
        }
    }

    pub fn touch(&mut self, id: u64, phase: &str, x: f32, y: f32, time: f64) {
        let phase = match phase {
            "start" => TouchPhase::Start,
            "move" => TouchPhase::Move,
            "end" => TouchPhase::End,
            _ => TouchPhase::Cancel,
        };
        self.event(Event::Touch { id, phase, x, y, time });
    }

    pub fn needs_frame(&self) -> bool {
        self.ui.needs_frame() || self.input.needs_frame() || self.input.next_timer().is_some()
    }
}

impl Default for Demo {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    use threers_ui::canvas::CanvasPainter;
    use wasm_bindgen::prelude::*;
    use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

    /// The demo on a 2D canvas, driven by the page.
    #[wasm_bindgen]
    pub struct SwipeDemo {
        demo: super::Demo,
        painter: CanvasPainter,
    }

    #[wasm_bindgen]
    impl SwipeDemo {
        #[wasm_bindgen(constructor)]
        pub fn new(canvas: HtmlCanvasElement) -> Result<SwipeDemo, JsValue> {
            let ctx: CanvasRenderingContext2d = canvas
                .get_context("2d")?
                .ok_or("no 2D context")?
                .dyn_into()?;
            Ok(SwipeDemo {
                demo: super::Demo::new(),
                painter: CanvasPainter::new(ctx, 1.0),
            })
        }

        pub fn load_swipe(&mut self, swipe: &[u8], words: &str) -> Result<(), JsValue> {
            self.demo.load_swipe(swipe, words).map_err(|e| JsValue::from_str(&e))
        }

        pub fn add_ngram(&mut self, bytes: &[u8], weight: f32) -> Result<(), JsValue> {
            self.demo.add_ngram(bytes, weight).map_err(|e| JsValue::from_str(&e))
        }

        pub fn add_neural(&mut self, bytes: &[u8], weight: f32) -> Result<(), JsValue> {
            self.demo.add_neural(bytes, weight).map_err(|e| JsValue::from_str(&e))
        }

        pub fn start(&mut self) -> Result<String, JsValue> {
            self.demo.start().map_err(|e| JsValue::from_str(&e))
        }

        /// Draws a frame; whether another is wanted soon.
        pub fn frame(&mut self, time: f64, width: f32, height: f32, dpr: f64) -> bool {
            self.demo.ui.set_pixel_ratio(dpr as f32);
            self.demo.frame(time / 1000.0, width, height, &mut self.painter);
            let (w, h) = (
                (width as f64 * dpr).round() as u32,
                (height as f64 * dpr).round() as u32,
            );
            self.painter.begin(w, h, dpr);
            self.demo.ui.paint(&mut self.painter);
            // The finger's trail over the keys, fading once it lifts.
            let trail = self
                .demo
                .keyboard
                .paint_swipe_trail(&mut self.painter, time / 1000.0);
            self.painter.finish();
            self.demo.needs_frame() || trail
        }

        pub fn pointer(&mut self, kind: &str, x: f32, y: f32) {
            self.demo.pointer(kind, x, y);
        }

        pub fn touch(&mut self, id: f64, phase: &str, x: f32, y: f32, time: f64) {
            self.demo.touch(id as u64, phase, x, y, time / 1000.0);
        }

        /// What the person has typed, learnt, as text to keep
        /// (`localStorage`); and how many words that is.
        pub fn user_text(&self) -> String {
            self.demo.user.to_text()
        }

        pub fn user_words(&self) -> f64 {
            self.demo.user.len() as f64
        }

        /// A kept user model back, before [`start`](Self::start).
        pub fn load_user(&mut self, text: &str) -> Result<(), JsValue> {
            let m = threers_swipe::user::UserModel::from_text(text).map_err(|e| JsValue::from_str(&e))?;
            self.demo.user = std::rc::Rc::new(m);
            Ok(())
        }

        /// Forgets what was learnt.
        pub fn forget_user(&mut self) {
            self.demo.user.clear();
        }

        /// The message typed, for the page (and its tests).
        pub fn text(&self) -> String {
            self.demo.field.text.clone()
        }

        /// The bar's words now, for the page's tests.
        pub fn bar(&self) -> Vec<String> {
            super::bar_words(&self.demo.ui)
        }

        /// Where the button named `name` is (x, y, width, height in the
        /// canvas's pixels), for the page's tests; empty if none.
        pub fn rect(&self, name: &str) -> Vec<f32> {
            self.demo
                .ui
                .accessibility_tree()
                .into_iter()
                .find(|a| a.name == name && a.role == threers_ui::Role::Button)
                .map_or(vec![], |a| {
                    let r = self.demo.ui.rect(a.id);
                    vec![r.x, r.y, r.width, r.height]
                })
        }
    }
}

/// The suggestion chips showing, by the names the keyboard gives them.
pub fn bar_words(ui: &Ui) -> Vec<String> {
    ui.accessibility_tree()
        .into_iter()
        .filter_map(|a| a.name.strip_prefix("Suggestion: ").map(str::to_owned))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use threers_ui::FixedWidthText;

    fn settle(d: &mut Demo, t: &mut f64) {
        for _ in 0..3 {
            *t += 0.3;
            d.frame(*t, 390.0, 844.0, &mut FixedWidthText(7.0));
        }
    }

    fn tap(d: &mut Demo, t: &mut f64, name: &str) {
        let n = d
            .ui
            .accessibility_tree()
            .into_iter()
            .find(|a| a.name == name && a.role == Role::Button)
            .unwrap_or_else(|| panic!("no key {name}"))
            .id;
        let r = d.ui.rect(n);
        let (x, y) = (r.x + r.width / 2.0, r.y + r.height / 2.0);
        d.pointer("down", x, y);
        d.pointer("up", x, y);
        settle(d, t);
    }

    #[test]
    fn the_bar_offers_the_next_word_once_loaded() {
        let mut d = Demo::new();
        let mut t = 0.0;
        settle(&mut d, &mut t);
        let lm = NgramModel::train("I am going to the shop. I am going to bed.", &Default::default());
        let swipe = Model::new(threers_swipe::ModelConfig::tiny(), 1).to_safetensors().unwrap();
        d.load_swipe(&swipe, "i 10\nam 5\ngoing 5\nto 9\n").unwrap();
        d.add_ngram(&lm.to_bytes(), 0.7).unwrap();
        // A second model that knows other words: the mix offers both.
        let chat = NgramModel::train("I am going home. Going home now.", &Default::default());
        d.add_ngram(&chat.to_text().into_bytes(), 0.3).unwrap();
        let about = d.start().unwrap();
        assert!(about.contains("3-gram") && about.contains("(70%)"), "{about}");
        settle(&mut d, &mut t);
        assert_eq!(bar_words(&d.ui).first().map(String::as_str), Some("I"), "a sentence starts with a capital");
        for k in ["I", "Space", "a", "m", "Space", "g", "o", "i", "n", "g", "Space"] {
            tap(&mut d, &mut t, k);
        }
        assert_eq!(d.field.text, "I am going ");
        assert_eq!(bar_words(&d.ui).first().map(String::as_str), Some("to"));
        assert!(bar_words(&d.ui).contains(&"home".to_owned()), "the second model's word too: {:?}", bar_words(&d.ui));
        tap(&mut d, &mut t, "Suggestion: to");
        assert_eq!(d.field.text, "I am going to ");
        assert_eq!(d.user.len(), 4, "I, am, going, to learnt as they were finished");
        assert!(bar_words(&d.ui).contains(&"the".to_owned()), "{:?}", bar_words(&d.ui));
    }
}
