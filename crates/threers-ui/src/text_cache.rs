//! Text measurements kept from one frame to the next.
//!
//! Measuring a run means shaping it, which is most of what laying out text
//! costs, and a UI shows mostly the same strings frame after frame. A
//! [`TextCache`] wraps the measurer passed to [`Ui::layout`](super::Ui::layout)
//! and answers from the last frame where it can. What one frame measures is
//! kept for the next. What the next frame does not use again is dropped at
//! [`TextCache::finish_frame`], so the cache holds what is on screen and
//! does not grow.
//!
//! ```
//! use threers_ui::text_cache::TextCache;
//! use threers_ui::{FixedWidthText, Style, TextMeasure};
//!
//! let mut cache = TextCache::new();
//! let mut fixed = FixedWidthText(7.0);
//! let style = Style::default();
//! for _ in 0..3 {
//!     cache.measure_with(&mut fixed).measure("hello", &style);
//!     cache.finish_frame();
//! }
//! assert_eq!(cache.stats(), (2, 1)); // hits, misses
//! ```

use std::collections::HashMap;

use super::layout::TextMeasure;
use super::style::Style;

/// What a measurement depends on: the text and the font.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Key {
    text: Box<str>,
    family: Box<str>,
    size: u32,
    line_height: Option<u32>,
    weight: u16,
    italic: bool,
    vertical: bool,
}

impl Key {
    fn of(text: &str, s: &Style) -> Self {
        Key {
            text: text.into(),
            family: s.font_family.as_str().into(),
            size: s.font_size.to_bits(),
            line_height: s.line_height.map(f32::to_bits),
            weight: super::paint::TextStyle::of(s).weight(),
            italic: s.italic,
            vertical: s.vertical,
        }
    }
}

/// Measurements from this frame and the last. See the [module docs](self).
#[derive(Debug, Default, Clone)]
pub struct TextCache {
    previous: HashMap<Key, (f32, f32)>,
    current: HashMap<Key, (f32, f32)>,
    hits: u64,
    misses: u64,
}

impl TextCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// A measurer answering from the cache, and asking `inner` (and
    /// remembering what it says) otherwise. `inner` must measure the same
    /// way every frame; call [`TextCache::clear`] when it changes (a font
    /// loaded, a new scale).
    pub fn measure_with<'a>(&'a mut self, inner: &'a mut dyn TextMeasure) -> CachedMeasure<'a> {
        CachedMeasure { cache: self, inner }
    }

    /// Ends a frame: what it measured is kept for the next, and what it did
    /// not use is let go.
    pub fn finish_frame(&mut self) {
        self.previous = std::mem::take(&mut self.current);
    }

    /// Forgets everything.
    pub fn clear(&mut self) {
        self.previous.clear();
        self.current.clear();
    }

    /// How many measurements were answered from the cache, and how many
    /// had to be made, since it was made.
    pub fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }

    /// Strings held.
    pub fn len(&self) -> usize {
        self.previous.len() + self.current.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A [`TextMeasure`] in front of another, through a [`TextCache`].
pub struct CachedMeasure<'a> {
    cache: &'a mut TextCache,
    inner: &'a mut dyn TextMeasure,
}

impl TextMeasure for CachedMeasure<'_> {
    fn measure(&mut self, text: &str, style: &Style) -> (f32, f32) {
        let key = Key::of(text, style);
        let c = &mut *self.cache;
        if let Some(&size) = c.current.get(&key) {
            c.hits += 1;
            return size;
        }
        if let Some((key, size)) = c.previous.remove_entry(&key) {
            c.hits += 1;
            c.current.insert(key, size);
            return size;
        }
        c.misses += 1;
        let size = self.inner.measure(text, style);
        c.current.insert(key, size);
        size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FixedWidthText;

    #[test]
    fn keeps_what_is_used_and_drops_what_is_not() {
        let mut cache = TextCache::new();
        let mut fixed = FixedWidthText(7.0);
        let s = Style::default();
        let big = Style {
            font_size: 30.0,
            ..Style::default()
        };
        {
            let mut m = cache.measure_with(&mut fixed);
            m.measure("a", &s);
            m.measure("b", &s);
            m.measure("a", &big);
        }
        cache.finish_frame();
        assert_eq!(cache.stats(), (0, 3), "different sizes are different keys");
        // Frame two uses "a" only.
        cache.measure_with(&mut fixed).measure("a", &s);
        cache.finish_frame();
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.stats(), (1, 3));
        // "b" is gone: measured again.
        cache.measure_with(&mut fixed).measure("b", &s);
        assert_eq!(cache.stats(), (1, 4));
    }
}
