//! Gesture typing on the threers keyboard: a trained model and the
//! languages' words, as the keyboard's swipe decoder.
//!
//! ```no_run
//! use threers_swipe::keyboard::SwipeTyping;
//! use threers_swipe::{Lexicon, Model};
//! # fn kb() -> threers_ui::virtual_keyboard::VirtualKeyboard { unimplemented!() }
//!
//! let typing = SwipeTyping::new(Model::load("swipe.safetensors").unwrap())
//!     .language("en", Lexicon::load("words_en.txt").unwrap())
//!     .language("de", Lexicon::load("words_de.txt").unwrap());
//! typing.attach(&kb());
//! ```
//!
//! With a next-word model for a language
//! ([`next_words`](SwipeTyping::next_words): an n-gram, the neural one or
//! both [`Interpolated`](crate::Interpolated)), a swipe is read after the
//! words before it, and the bar offers the next word as the
//! iPhone's does.
//!
//! The keys come with each swipe as they are drawn, so a keyboard
//! resized, split or floating decodes as it shows; the word trie is made
//! once per language and set of keys and kept.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use threers_ui::virtual_keyboard::{Swipe, VirtualKeyboard};

use crate::decode::{DecodeConfig, Decoder, Lexicon};
use crate::geometry::KeyboardGeometry;
use crate::model::Model;
use crate::next_word::{self, WordModel};
use crate::user::{finished_words, UserModel, WithUser};
use crate::trace::TouchPoint;

/// Words offered for a swipe: the one typed and the bar's.
const WORDS: usize = 4;
/// Words in the bar.
const BAR: usize = 3;

/// A model and a lexicon per language, decoding the keyboard's swipes.
pub struct SwipeTyping {
    model: Model,
    lexicons: HashMap<String, Lexicon>,
    next_words: HashMap<String, Box<dyn WordModel>>,
    /// Each language's learner of what is typed, and its share.
    users: HashMap<String, (Rc<UserModel>, f32)>,
    /// The text before the caret last seen, by language, to tell which
    /// words were finished since.
    seen: RefCell<HashMap<String, String>>,
    config: DecodeConfig,
    /// Decoders by language and the keys' letters in order.
    decoders: RefCell<HashMap<String, Rc<Decoder>>>,
}

impl SwipeTyping {
    pub fn new(model: Model) -> Self {
        Self {
            model,
            lexicons: HashMap::new(),
            next_words: HashMap::new(),
            users: HashMap::new(),
            seen: RefCell::default(),
            config: DecodeConfig::default(),
            decoders: RefCell::default(),
        }
    }

    /// The words of the layout tagged `language` (`en`, `de`, …).
    pub fn language(mut self, language: &str, lexicon: Lexicon) -> Self {
        self.lexicons.insert(language.to_owned(), lexicon);
        self.decoders.borrow_mut().clear();
        self
    }

    /// What follows what in `language`: swipes are read after the words
    /// before them, and the bar offers the next word.
    pub fn next_words(mut self, language: &str, model: impl WordModel + 'static) -> Self {
        self.next_words.insert(language.to_owned(), Box::new(model));
        self
    }

    /// Learns what is typed in `language` into `model` (each word as it is
    /// finished), which gets `weight` of the next-word mix: the bar and the
    /// swipes come to know the person's own words. Keep the `Rc` to save
    /// it ([`UserModel::to_text`]).
    pub fn user(mut self, language: &str, model: Rc<UserModel>, weight: f32) -> Self {
        self.users.insert(language.to_owned(), (model, weight));
        self
    }

    /// The language's next-word model with the user's mixed in, for `f`.
    fn with_words<T>(&self, language: &str, f: impl FnOnce(Option<&dyn WordModel>) -> T) -> T {
        match (self.next_words.get(language), self.users.get(language)) {
            (Some(base), Some((user, weight))) => f(Some(&WithUser {
                base: base.as_ref(),
                user,
                weight: *weight,
            })),
            (Some(base), None) => f(Some(base.as_ref())),
            (None, Some((user, _))) => f(Some(user.as_ref())),
            (None, None) => f(None),
        }
    }

    /// The words finished since the text before the caret was last seen,
    /// learnt.
    fn learn(&self, language: &str, before: &str) {
        let Some((user, _)) = self.users.get(language) else {
            return;
        };
        let mut seen = self.seen.borrow_mut();
        let was = seen.entry(language.to_owned()).or_default();
        for (ctx, word) in finished_words(was, before) {
            let ctx: Vec<&str> = ctx.iter().map(String::as_str).collect();
            user.learn(&ctx, &word);
        }
        was.clear();
        was.push_str(before);
    }

    /// The bar's words for the text `before` the caret in `language`:
    /// completions of the word being typed, or the next one after a space;
    /// none without a next-word model for it. Words just finished are
    /// learnt first, with a user model.
    pub fn suggest(&self, language: &str, before: &str) -> Vec<String> {
        self.learn(language, before);
        self.with_words(language, |m| {
            m.map_or_else(Vec::new, |m| next_word::suggest(m, before, BAR))
        })
    }

    pub fn decode_config(mut self, config: DecodeConfig) -> Self {
        self.config = config;
        self.decoders.borrow_mut().clear();
        self
    }

    /// The words `swipe` may be, best first; none for a language without
    /// words.
    pub fn decode(&self, swipe: &Swipe) -> Vec<String> {
        let Some(lexicon) = self.lexicons.get(&swipe.language) else {
            return vec![];
        };
        let keyboard = KeyboardGeometry::from_rects(
            &swipe.language,
            swipe
                .keys
                .iter()
                .map(|k| (k.text.as_str(), k.x, k.y, k.width, k.height)),
        );
        let id = format!(
            "{}:{}",
            swipe.language,
            keyboard
                .keys
                .iter()
                .map(|k| k.text.as_str())
                .collect::<String>()
        );
        let decoder = self
            .decoders
            .borrow_mut()
            .entry(id)
            .or_insert_with(|| {
                Rc::new(Decoder::new(&keyboard, lexicon).with_config(self.config.clone()))
            })
            .clone();
        let t0 = swipe.points.first().map_or(0.0, |p| p.t);
        let points: Vec<TouchPoint> = swipe
            .points
            .iter()
            .map(|p| {
                let (x, y) = keyboard.to_units(p.x, p.y);
                TouchPoint::new(x, y, (p.t - t0) as f32)
            })
            .collect();
        let found = self.with_words(&swipe.language, |lm| match lm {
            Some(lm) => {
                // The keyboard puts a space after a word before the swiped
                // one, so a word half-typed is a word before it.
                let before = format!("{} ", swipe.before);
                let (context, _) = next_word::split_context(&before, lm.context_len());
                let context: Vec<&str> = context.iter().map(String::as_str).collect();
                decoder.decode_on_after(&self.model, &keyboard, &points, lm, &context, WORDS)
            }
            None => decoder.decode_on(&self.model, &keyboard, &points, WORDS),
        });
        // "i" is "I" wherever it comes; the keyboard's Shift does the rest.
        found
            .unwrap_or_default()
            .into_iter()
            .map(|c| next_word::cased(&c.word, "", false))
            .collect()
    }

    /// Makes this `keyboard`'s swipe decoder, and with next-word models
    /// its bar's predictions.
    pub fn attach(self, keyboard: &VirtualKeyboard) {
        let me = Rc::new(self);
        if !me.next_words.is_empty() || !me.users.is_empty() {
            let m = me.clone();
            keyboard.set_predictions(move |language, before| m.suggest(language, before));
        }
        keyboard.set_swipe_decoder(move |s| me.decode(s));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::{self, Rng, SynthConfig};
    use crate::{ModelConfig, TrainConfig, Trainer};
    use threers_ui::virtual_keyboard::{KeyboardConfig, KeyboardLayout, SwipeKey, SwipePoint};

    /// A made-up swipe as the keyboard would hand it over: pixels, keys as
    /// drawn (here at 40 px a key, 54 a row, from (10, 500)).
    fn as_drawn(word: &str, rng: &mut Rng) -> (Swipe, KeyboardGeometry) {
        let g = KeyboardGeometry::from_config(
            &KeyboardConfig::iphone(),
            &KeyboardLayout::english(),
            390.0,
        );
        let ((x0, y0), _) = g.bounds();
        let px = |x: f32, y: f32| (10.0 + (x - x0) * 40.0, 500.0 + (y - y0) * 40.0);
        let keys = g
            .keys
            .iter()
            .map(|k| {
                let (x, y) = px(k.x, k.y);
                SwipeKey {
                    text: k.text.clone(),
                    x: x - 18.0,
                    y: y - 21.0,
                    width: 36.0,
                    height: 42.0,
                }
            })
            .collect();
        let pts = synth::swipe(&g, word, &SynthConfig::default(), rng).unwrap();
        let points = pts
            .iter()
            .map(|p| {
                let (x, y) = px(p.x, p.y);
                SwipePoint {
                    x,
                    y,
                    t: 3.0 + p.t as f64,
                }
            })
            .collect();
        (
            Swipe {
                points,
                keys,
                language: "en".into(),
                before: String::new(),
            },
            g,
        )
    }

    #[test]
    fn decodes_swipes_as_the_keyboard_hands_them_over() {
        let words = [
            "hello", "world", "keyboard", "swipe", "typing", "quick", "brown", "fox",
        ];
        let g = KeyboardGeometry::from_config(
            &KeyboardConfig::iphone(),
            &KeyboardLayout::english(),
            390.0,
        );
        let mut t = Trainer::new(TrainConfig {
            model: ModelConfig::tiny(),
            steps: 300,
            batch: 16,
            lr: 5e-3,
            ..TrainConfig::default()
        });
        t.add_synthetic(g, words.map(String::from).to_vec());
        t.run(|_, _| {});
        let typing =
            SwipeTyping::new(t.model).language("en", Lexicon::from_counts(words.map(|w| (w, 1.0))));
        let mut rng = Rng::new(42);
        let mut right = 0;
        for w in words {
            let (s, _) = as_drawn(w, &mut rng);
            let got = typing.decode(&s);
            assert!(got.len() <= WORDS);
            right += (got.first().map(String::as_str) == Some(w)) as usize;
        }
        assert!(right >= 6, "{right} of {} right", words.len());
        assert_eq!(typing.decoders.borrow().len(), 1, "one trie, kept");
        let (mut s, _) = as_drawn("hello", &mut rng);
        s.language = "fr".into();
        assert!(typing.decode(&s).is_empty(), "no French words given");
    }
}
