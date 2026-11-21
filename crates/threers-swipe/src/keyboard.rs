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
use crate::trace::TouchPoint;

/// Words offered for a swipe: the one typed and the bar's.
const WORDS: usize = 4;

/// A model and a lexicon per language, decoding the keyboard's swipes.
pub struct SwipeTyping {
    model: Model,
    lexicons: HashMap<String, Lexicon>,
    config: DecodeConfig,
    /// Decoders by language and the keys' letters in order.
    decoders: RefCell<HashMap<String, Rc<Decoder>>>,
}

impl SwipeTyping {
    pub fn new(model: Model) -> Self {
        Self {
            model,
            lexicons: HashMap::new(),
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
        decoder
            .decode_on(&self.model, &keyboard, &points, WORDS)
            .unwrap_or_default()
            .into_iter()
            .map(|c| c.word)
            .collect()
    }

    /// Makes this `keyboard`'s swipe decoder.
    pub fn attach(self, keyboard: &VirtualKeyboard) {
        keyboard.set_swipe_decoder(move |s| self.decode(s));
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
