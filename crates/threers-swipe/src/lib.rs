//! Gesture typing for the threers on-screen keyboard: draw a word across
//! the keys without lifting the finger, get the word.
//!
//! One small model serves every keyboard. It never learns where the keys
//! are: at each point of the swipe it says whether the finger meant a key
//! there and where it was aiming, and the keyboard's own key centres turn
//! that into a score per key ([`model`]). So the seven built-in layouts, a
//! phone's keyboard and a tablet's, one-handed and floating ones all share
//! it, and a layout added later needs no retraining to work, only to work
//! better.
//!
//! - [`geometry`]: key centres from a
//!   [`KeyboardConfig`](threers_ui::virtual_keyboard::KeyboardConfig) and a
//!   [`KeyboardLayout`](threers_ui::virtual_keyboard::KeyboardLayout), or
//!   from the rectangles on screen.
//! - [`trace`]: a swipe resampled to [`FRAMES`] points and its features.
//! - [`synth`]: swipes made up from a word list, as a hand moves
//!   (minimum-jerk strokes, aim scattered around each key), for layouts no
//!   one has swiped on yet.
//! - [`augment`]: the keyboard and the swipe stretched, sheared, turned and
//!   mirrored together, so the model learns shapes, not one layout.
//! - [`data`]: recorded swipes, FUTO's MIT-licensed corpus included.
//! - [`model`], [`ctc`], [`train`]: the network, its loss and its training.
//! - [`decode`]: the words a swipe could be, through a trie of the
//!   language's words weighted by how common they are.
//! - `keyboard` (feature `keyboard`): all of it as the on-screen
//!   keyboard's swipe decoder.
//!
//! ```
//! use threers_swipe::{Decoder, KeyboardGeometry, Lexicon, Model, ModelConfig, synth};
//! use threers_ui::virtual_keyboard::{KeyboardConfig, KeyboardLayout};
//!
//! let keyboard = KeyboardGeometry::from_config(&KeyboardConfig::iphone(), &KeyboardLayout::english(), 390.0);
//! let model = Model::new(ModelConfig::tiny(), 7);
//! let lexicon = Lexicon::from_counts([("hello", 100.0), ("world", 80.0)]);
//! let decoder = Decoder::new(&keyboard, &lexicon);
//! let mut rng = synth::Rng::new(1);
//! let swipe = synth::swipe(&keyboard, "hello", &synth::SynthConfig::default(), &mut rng).unwrap();
//! let words = decoder.decode(&model, &swipe, 3);
//! assert!(words.len() <= 2, "only words of the lexicon");
//! ```

pub mod augment;
pub mod ctc;
pub mod data;
pub mod decode;
pub mod geometry;
#[cfg(feature = "keyboard")]
pub mod keyboard;
pub mod model;
pub mod synth;
pub mod trace;
pub mod train;

pub use decode::{Candidate, Decoder, Lexicon};
pub use geometry::{Key, KeyboardGeometry};
pub use model::{Model, ModelConfig};
pub use trace::{TouchPoint, FEATURES, FRAMES};
pub use train::{TrainConfig, Trainer};
