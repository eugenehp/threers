//! Where a word may be broken with a hyphen: Liang's algorithm (TeX's)
//! over the patterns of hyph-utf8 and their lists of exceptions, in
//! `data/hyph-*`, each with its notice (`hyph-*.LICENSE.txt`: MIT, BSD,
//! LPPL or free to copy). Languages: English (US and British), German,
//! French, Spanish, Italian, Portuguese, Dutch, Swedish, Danish, Norwegian
//! Bokmål, Finnish, Polish, Russian and Ukrainian; [`for_language`] picks
//! one by its tag.
//!
//! ```
//! let h = threers_spell::hyphenate::english();
//! assert_eq!(h.split("hyphenation"), ["hy", "phen", "ation"]);
//! let de = threers_spell::hyphenate::for_language("de-DE").unwrap();
//! assert_eq!(de.split("Silbentrennung"), ["Sil", "ben", "tren", "nung"]);
//! ```

use std::collections::HashMap;
use std::sync::OnceLock;

/// Each language: the tags it answers to (their first part, or whole),
/// its patterns and exceptions, and the letters a break leaves at least
/// before it and after it.
macro_rules! languages {
    ($($name:literal: [$($tag:literal),*] $l:literal $r:literal;)*) => {
        const LANGUAGES: &[(&[&str], &str, &str, usize, usize)] = &[
            $((&[$($tag),*], include_str!(concat!("../data/hyph-", $name, ".pat.txt")), include_str!(concat!("../data/hyph-", $name, ".hyp.txt")), $l, $r),)*
        ];
    };
}

languages! {
    "en-us": ["en-us", "en"] 2 3;
    "en-gb": ["en-gb", "en-au", "en-nz", "en-ie", "en-za", "en-in"] 2 3;
    "de-1996": ["de"] 2 2;
    "fr": ["fr"] 2 2;
    "es": ["es"] 2 2;
    "it": ["it"] 2 2;
    "pt": ["pt"] 2 3;
    "nl": ["nl"] 2 2;
    "sv": ["sv"] 2 2;
    "da": ["da"] 2 2;
    "nb": ["nb", "no", "nn"] 2 2;
    "fi": ["fi"] 2 2;
    "pl": ["pl"] 2 2;
    "ru": ["ru"] 2 2;
    "uk": ["uk"] 2 2;
}

impl std::fmt::Debug for Hyphenator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hyphenator").field("patterns", &self.patterns.len()).field("left_min", &self.left_min).field("right_min", &self.right_min).finish()
    }
}

pub struct Hyphenator {
    /// A pattern's letters → the values between them (one more than the letters).
    patterns: HashMap<String, Vec<u8>>,
    longest: usize,
    /// Whole words with their breaks given (character counts before each).
    exceptions: HashMap<String, Vec<usize>>,
    /// Letters a break leaves before it and after it at the least.
    pub left_min: usize,
    pub right_min: usize,
}

impl Hyphenator {
    /// From patterns (`.ach4`, one a line or whitespace apart) and
    /// exceptions (`as-so-ciate`).
    pub fn new(patterns: &str, exceptions: &str, left_min: usize, right_min: usize) -> Hyphenator {
        let mut map = HashMap::new();
        let mut longest = 0;
        for p in patterns.split_whitespace() {
            let mut letters = String::new();
            let mut values = vec![0u8];
            for c in p.chars() {
                match c.to_digit(10) {
                    Some(d) => *values.last_mut().unwrap() = d as u8,
                    None => {
                        letters.push(c);
                        values.push(0);
                    }
                }
            }
            longest = longest.max(letters.chars().count());
            map.insert(letters, values);
        }
        let mut ex = HashMap::new();
        for w in exceptions.split_whitespace() {
            let mut breaks = Vec::new();
            let mut n = 0;
            for c in w.chars() {
                if c == '-' {
                    breaks.push(n);
                } else {
                    n += 1;
                }
            }
            ex.insert(w.replace('-', ""), breaks);
        }
        Hyphenator { patterns: map, longest, exceptions: ex, left_min, right_min }
    }

    /// Where `word` may break: byte offsets into it, ascending. A word with
    /// anything but letters (and inner apostrophes) is not broken.
    pub fn points(&self, word: &str) -> Vec<usize> {
        let chars: Vec<(usize, char)> = word.char_indices().collect();
        let n = chars.len();
        if n < self.left_min + self.right_min || !chars.iter().all(|(_, c)| c.is_alphabetic() || *c == '\'') {
            return Vec::new();
        }
        let lower: String = word.to_lowercase();
        let counts: Vec<usize> = if let Some(b) = self.exceptions.get(&lower) {
            b.clone()
        } else {
            let dotted: Vec<char> = std::iter::once('.').chain(lower.chars()).chain(std::iter::once('.')).collect();
            // values between characters of the dotted word
            let mut values = vec![0u8; dotted.len() + 1];
            let mut key = String::new();
            for i in 0..dotted.len() {
                key.clear();
                for (j, &c) in dotted[i..].iter().enumerate().take(self.longest) {
                    key.push(c);
                    if let Some(v) = self.patterns.get(&key) {
                        for (k, &val) in v.iter().enumerate() {
                            let at = i + k;
                            if at <= i + j + 1 && val > values[at] {
                                values[at] = val;
                            }
                        }
                    }
                }
            }
            // between the word's characters k-1 and k is dotted position k+1
            (1..n).filter(|&k| values[k + 1] % 2 == 1).collect()
        };
        counts
            .into_iter()
            .filter(|&k| k >= self.left_min && n - k >= self.right_min && k < n)
            .map(|k| chars[k].0)
            .collect()
    }

    /// The word cut at every place it may break.
    pub fn split<'a>(&self, word: &'a str) -> Vec<&'a str> {
        let mut out = Vec::new();
        let mut last = 0;
        for p in self.points(word) {
            out.push(&word[last..p]);
            last = p;
        }
        out.push(&word[last..]);
        out
    }
}

/// US English, as TeX hyphenates it (two letters at least before a break,
/// three after).
pub fn english() -> &'static Hyphenator {
    language(0)
}

fn language(i: usize) -> &'static Hyphenator {
    static H: [OnceLock<Hyphenator>; LANGUAGES.len()] = [const { OnceLock::new() }; LANGUAGES.len()];
    H[i].get_or_init(|| {
        let (_, pat, hyp, l, r) = LANGUAGES[i];
        Hyphenator::new(pat, hyp, l, r)
    })
}

/// The hyphenation of a language by its tag (`en-GB`, `de_CH`, `fr`): the
/// whole tag's, else its language's; `None` for a language without patterns.
pub fn for_language(tag: &str) -> Option<&'static Hyphenator> {
    let tag = tag.trim().to_ascii_lowercase().replace('_', "-");
    let first = tag.split('-').next().unwrap_or("");
    let find = |t: &str| LANGUAGES.iter().position(|(tags, ..)| tags.contains(&t));
    find(&tag).or_else(|| find(first)).map(language)
}

/// The tags [`for_language`] knows.
pub fn languages() -> impl Iterator<Item = &'static str> {
    LANGUAGES.iter().flat_map(|(tags, ..)| tags.iter().copied())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_break_as_tex_breaks_them() {
        let h = english();
        assert_eq!(h.split("computer"), ["com", "puter"]);
        assert_eq!(h.split("algorithm"), ["al", "go", "rithm"]);
        assert_eq!(h.split("Hyphenation"), ["Hy", "phen", "ation"]);
        // an exception
        assert_eq!(h.split("associate"), ["as", "so", "ciate"]);
        // too short, or not a word
        assert!(h.points("cat").is_empty());
        assert!(h.points("x2y4z6w8").is_empty());
    }

    #[test]
    fn languages_are_found_by_their_tags() {
        assert!(std::ptr::eq(for_language("en-US").unwrap(), english()));
        assert!(std::ptr::eq(for_language("en").unwrap(), english()));
        assert!(!std::ptr::eq(for_language("en-GB").unwrap(), english()));
        assert!(for_language("xx-YY").is_none());
        let de = for_language("de_AT").unwrap();
        assert_eq!(de.split("Wörterbuch"), ["Wör", "ter", "buch"]);
        let fr = for_language("fr-CA").unwrap();
        assert_eq!(fr.split("ordinateur"), ["or", "di", "na", "teur"]);
        let ru = for_language("ru").unwrap();
        assert_eq!(ru.split("программирование").len() > 3, true);
        let es = for_language("es-MX").unwrap();
        assert_eq!(es.split("universidad"), ["uni", "ver", "si", "dad"]);
        for t in languages() {
            assert!(for_language(t).is_some());
        }
    }
}
