//! Gesture typing: a word drawn across the letter keys without lifting the
//! finger.
//!
//! The keyboard records the path and hands it, with the letter keys as
//! they are drawn, to the app's decoder
//! ([`VirtualKeyboard::set_swipe_decoder`](super::VirtualKeyboard::set_swipe_decoder));
//! the `threers-swipe` crate makes one from a trained model. The word types
//! with a space after it; punctuation typed next takes that space's place. A press turns
//! into a swipe once the finger has gone [`SWIPE_START`] key widths, before
//! a hold offers accents; a tap or a slide to a neighbouring key types as
//! it did.

/// A point of a swipe: where, in the window's pixels, and when, in seconds
/// on the input's clock.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SwipePoint {
    pub x: f32,
    pub y: f32,
    pub t: f64,
}

/// A letter key as drawn when the swipe ended, in the window's pixels.
#[derive(Debug, Clone, PartialEq)]
pub struct SwipeKey {
    /// What it types, lower case.
    pub text: String,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// A finished swipe, for a decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct Swipe {
    pub points: Vec<SwipePoint>,
    /// The letter keys showing, as drawn (a split keyboard's halves apart).
    pub keys: Vec<SwipeKey>,
    /// The layout's language tag (`en`, `de`, …).
    pub language: String,
    /// The field's text before the caret when the swipe began (what the
    /// keyboard typed lately without
    /// [`set_context`](super::VirtualKeyboard::set_context)), for the
    /// words before the swiped one.
    pub before: String,
}

/// How far, in widths of the key it began on, a press must travel to be a
/// swipe.
pub const SWIPE_START: f32 = 0.9;
