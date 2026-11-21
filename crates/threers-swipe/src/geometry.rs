//! Where the keys are, in the units the model reads.
//!
//! One unit is a letter key's width, gaps included, so a phone's keyboard
//! and a tablet's are the same size to the model. The origin is the centre
//! of the letter keys, so where on the screen the keyboard sits does not
//! matter either. [`KeyboardGeometry::to_units`] takes a touch from the
//! keyboard's pixels into these units.

use std::collections::HashMap;

use threers_ui::virtual_keyboard::{
    alternates_for, KeyAction, KeyboardConfig, KeyboardLayout, Page,
};

/// A letter key: what it types and its centre, in key units.
#[derive(Debug, Clone, PartialEq)]
pub struct Key {
    /// Lower case.
    pub text: String,
    pub x: f32,
    pub y: f32,
}

/// The letter keys of a keyboard, centred on their middle, in key units.
#[derive(Debug, Clone)]
pub struct KeyboardGeometry {
    /// The language tag (`en`, `ru`, …), or empty.
    pub language: String,
    pub keys: Vec<Key>,
    /// Pixels to units: `(p - origin) / unit`.
    origin: (f32, f32),
    unit: f32,
    by_char: HashMap<char, usize>,
}

impl KeyboardGeometry {
    /// From key centres already in key units; they are re-centred.
    pub fn new(language: &str, keys: Vec<Key>) -> Self {
        Self::with_transform(language, keys, (0.0, 0.0), 1.0)
    }

    fn with_transform(language: &str, mut keys: Vec<Key>, origin: (f32, f32), unit: f32) -> Self {
        let (cx, cy) = middle(&keys);
        for k in &mut keys {
            k.x -= cx;
            k.y -= cy;
        }
        let mut g = Self {
            language: language.into(),
            keys,
            origin: (origin.0 + cx * unit, origin.1 + cy * unit),
            unit,
            by_char: HashMap::new(),
        };
        g.index();
        g
    }

    /// The letters page of `config` in `layout`, for a keyboard `width`
    /// pixels across: the letter keys where the keyboard draws them
    /// (unparted). [`to_units`](Self::to_units) then takes pixels from the
    /// keyboard panel's top-left corner.
    pub fn from_config(config: &KeyboardConfig, layout: &KeyboardLayout, width: f32) -> Self {
        let boxes = config.key_boxes(Page::Letters, layout, 1);
        let total = boxes.iter().map(|b| b.x + b.width).fold(1.0f32, f32::max);
        let [top, right, _, left] = config.padding;
        let unit = ((width - left - right) / total).max(1.0);
        let pitch = (config.key_height + config.row_gap) / unit;
        let half_key = config.key_height / unit / 2.0;
        let keys = boxes
            .iter()
            .filter_map(|b| match &b.key.action {
                KeyAction::Text(t) if is_letter(t) => Some(Key {
                    text: t.to_lowercase(),
                    x: b.x + b.width / 2.0,
                    y: b.row as f32 * pitch + half_key,
                }),
                _ => None,
            })
            .collect();
        Self::with_transform(&layout.language, keys, (left, top), unit)
    }

    /// From the keys' rectangles on screen, `(text, x, y, width, height)`
    /// in any one unit (pixels): the unit becomes the median key width.
    /// For a keyboard drawn some other way, or a split one as it shows.
    pub fn from_rects<'a>(
        language: &str,
        rects: impl IntoIterator<Item = (&'a str, f32, f32, f32, f32)>,
    ) -> Self {
        let rects: Vec<_> = rects.into_iter().filter(|r| is_letter(r.0)).collect();
        let mut widths: Vec<f32> = rects.iter().map(|r| r.3).collect();
        widths.sort_by(f32::total_cmp);
        let unit = widths
            .get(widths.len() / 2)
            .copied()
            .unwrap_or(1.0)
            .max(1e-6);
        let keys = rects
            .iter()
            .map(|&(t, x, y, w, h)| Key {
                text: t.to_lowercase(),
                x: (x + w / 2.0) / unit,
                y: (y + h / 2.0) / unit,
            })
            .collect();
        Self::with_transform(language, keys, (0.0, 0.0), unit)
    }

    /// The seven built-in layouts on an iPhone's keyboard 390 pixels wide.
    pub fn built_in() -> Vec<Self> {
        KeyboardLayout::all()
            .iter()
            .map(|l| Self::from_config(&KeyboardConfig::iphone(), l, 390.0))
            .collect()
    }

    /// A point in the frame [`from_config`](Self::from_config) or
    /// [`from_rects`](Self::from_rects) was given, in key units.
    pub fn to_units(&self, x: f32, y: f32) -> (f32, f32) {
        (
            (x - self.origin.0) / self.unit,
            (y - self.origin.1) / self.unit,
        )
    }

    /// The same keyboard with every key moved by `f` (augmentation).
    pub fn map(&self, f: impl Fn(f32, f32) -> (f32, f32)) -> Self {
        let mut g = self.clone();
        for k in &mut g.keys {
            (k.x, k.y) = f(k.x, k.y);
        }
        g
    }

    /// The key typing `c`: the one labelled with it, else the one holding
    /// it as an accented form (é on e's key, ё on е's).
    pub fn key_for(&self, c: char) -> Option<usize> {
        let lower = c.to_lowercase().next().unwrap_or(c);
        self.by_char.get(&lower).copied()
    }

    /// The keys a word crosses, one per letter, repeats kept; `None` when a
    /// letter has no key here. Apostrophes are passed over, as a swipe
    /// of "don't" crosses d, o, n, t.
    pub fn word_keys(&self, word: &str) -> Option<Vec<usize>> {
        word.chars()
            .filter(|&c| c != '\'' && c != '’')
            .map(|c| self.key_for(c))
            .collect()
    }

    /// [`word_keys`](Self::word_keys) with a key typed twice running
    /// counted once: a swipe cannot show "ll", so neither the model's
    /// target nor the lexicon's path does.
    pub fn word_path(&self, word: &str) -> Option<Vec<usize>> {
        let mut keys = self.word_keys(word)?;
        keys.dedup();
        Some(keys)
    }

    /// The middle of the keys and how far they reach, `(min, max)`.
    pub fn bounds(&self) -> ((f32, f32), (f32, f32)) {
        let mut lo = (f32::MAX, f32::MAX);
        let mut hi = (f32::MIN, f32::MIN);
        for k in &self.keys {
            lo = (lo.0.min(k.x), lo.1.min(k.y));
            hi = (hi.0.max(k.x), hi.1.max(k.y));
        }
        (lo, hi)
    }

    fn index(&mut self) {
        self.by_char.clear();
        for (i, k) in self.keys.iter().enumerate() {
            if let Some(c) = single(&k.text) {
                self.by_char.insert(c, i);
            }
        }
        // Accented forms after, so a key of their own wins.
        for (i, k) in self.keys.iter().enumerate() {
            for a in alternates_for(&k.text) {
                if let Some(c) = single(&a) {
                    self.by_char.entry(c).or_insert(i);
                }
            }
        }
    }
}

fn single(s: &str) -> Option<char> {
    let mut it = s.chars();
    let c = it.next()?;
    it.next().is_none().then_some(c)
}

fn is_letter(t: &str) -> bool {
    single(t).is_some_and(char::is_alphabetic)
}

fn middle(keys: &[Key]) -> (f32, f32) {
    if keys.is_empty() {
        return (0.0, 0.0);
    }
    let (mut lo, mut hi) = ((f32::MAX, f32::MAX), (f32::MIN, f32::MIN));
    for k in keys {
        lo = (lo.0.min(k.x), lo.1.min(k.y));
        hi = (hi.0.max(k.x), hi.1.max(k.y));
    }
    ((lo.0 + hi.0) / 2.0, (lo.1 + hi.1) / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iphone_english_is_qwerty_in_units() {
        let g = KeyboardGeometry::from_config(
            &KeyboardConfig::iphone(),
            &KeyboardLayout::english(),
            390.0,
        );
        assert_eq!(g.keys.len(), 26);
        let at = |c| &g.keys[g.key_for(c).unwrap()];
        assert!((at('w').x - at('q').x - 1.0).abs() < 1e-4, "a unit apart");
        assert!((at('a').x - at('q').x - 0.5).abs() < 1e-4);
        assert!(at('a').y > at('q').y && at('z').y > at('a').y);
        let ((x0, _), (x1, _)) = g.bounds();
        assert!((x0 + x1).abs() < 1e-4, "centred");
    }

    #[test]
    fn accents_and_pixels() {
        let g = KeyboardGeometry::from_config(
            &KeyboardConfig::iphone(),
            &KeyboardLayout::english(),
            390.0,
        );
        assert_eq!(g.key_for('é'), g.key_for('e'));
        assert_eq!(g.key_for('E'), g.key_for('e'));
        assert_eq!(g.word_path("hello").unwrap().len(), 4, "ll once");
        assert_eq!(g.word_path("don't"), g.word_path("dont"));
        let fr = KeyboardGeometry::from_config(
            &KeyboardConfig::iphone(),
            &KeyboardLayout::french(),
            390.0,
        );
        assert_ne!(fr.key_for('é'), fr.key_for('e'), "French has é's own key");
        // The middle of the panel's letters maps to the origin.
        let q = &g.keys[g.key_for('q').unwrap()];
        let p = g.to_units(3.0 + 0.5 * g.unit, 8.0 + 43.0 / 2.0);
        assert!(
            (p.0 - q.x).abs() < 1e-3 && (p.1 - q.y).abs() < 1e-3,
            "{p:?} vs q"
        );
    }

    #[test]
    fn rects_take_the_median_width() {
        let g = KeyboardGeometry::from_rects(
            "en",
            [
                ("a", 0.0, 0.0, 40.0, 50.0),
                ("b", 40.0, 0.0, 40.0, 50.0),
                ("c", 0.0, 60.0, 40.0, 50.0),
            ],
        );
        let (a, b) = (&g.keys[0], &g.keys[1]);
        assert!((b.x - a.x - 1.0).abs() < 1e-5);
        assert!((g.keys[2].y - a.y - 1.5).abs() < 1e-5);
    }
}
