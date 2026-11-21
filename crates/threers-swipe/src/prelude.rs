//! One import for gesture typing and next words: the swipe model and its
//! decoder over a word list, the keyboard's geometry and a swipe's points,
//! the next-word models (each with its feature on) and their mix, and with
//! `keyboard`, the on-screen keyboard's decoder.
//!
//! ```
//! use threers_swipe::prelude::*;
//! use threers_ui::virtual_keyboard::{KeyboardConfig, KeyboardLayout};
//!
//! let keyboard = KeyboardGeometry::from_config(&KeyboardConfig::iphone(), &KeyboardLayout::english(), 390.0);
//! let decoder = Decoder::new(&keyboard, &Lexicon::from_counts([("to", 9.0), ("too", 3.0)]));
//! let model = Model::new(ModelConfig::tiny(), 7);
//! # #[cfg(feature = "ngram")] {
//! let next = NgramModel::train("I am going to bed. Me too.", &NgramConfig::default());
//! let swipe = vec![TouchPoint::new(4.5, 0.5, 0.0), TouchPoint::new(8.5, 0.5, 0.2)];
//! let words = decoder.decode_on_after(&model, &keyboard, &swipe, &next, &["going"], 2);
//! assert!(words.is_some());
//! # }
//! ```
//!
//! Training (`train`, `synth`, `augment`, `data`) stays at its module's
//! path.

pub use crate::decode::{Candidate, DecodeConfig, Decoder, Lexicon};
pub use crate::geometry::KeyboardGeometry;
pub use crate::model::{Model, ModelConfig};
pub use crate::next_word::{Interpolated, WordModel};
pub use crate::trace::TouchPoint;
pub use crate::user::UserModel;

#[cfg(feature = "keyboard")]
pub use crate::keyboard::SwipeTyping;
#[cfg(feature = "rlx")]
pub use crate::neural::{NeuralConfig, NeuralLm, NeuralTrainConfig};
#[cfg(feature = "ngram")]
pub use crate::ngram::{NgramConfig, NgramModel};
