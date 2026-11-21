//! Spellcheck and autocorrect for text typed into a threers app.
//!
//! A [`Speller`] knows words from up to three places, each switched on or
//! off by the app: English (Webster's Second International, public domain,
//! with its inflections understood: "walked", "computers", "happier" are
//! words because "walk", "computer", "happy" are; plus everyday words newer
//! than it), words of the app's own with how much they count (what the user
//! has typed before), and the words on screen now.
//!
//! A word it doesn't know gets fixes within two edits, ranked as a phone
//! keyboard ranks them: a swapped pair ("teh") or a neighbouring key
//! ("tge") costs less than any other slip, a common word or one the user
//! types often wins a tie. [`Speller::autocorrect`] is the fix to make on
//! space, when there's one sure enough.
//!
//! ```
//! use threers_spell::Speller;
//! let mut s = Speller::new(true);
//! assert!(s.is_correct("walked") && !s.is_correct("teh"));
//! assert_eq!(s.autocorrect("teh").as_deref(), Some("the"));
//! s.add_word("kubectl", 3.0);
//! assert!(s.is_correct("kubectl"));
//! ```

use std::collections::HashMap;
use std::sync::OnceLock;

mod english;

/// The English word list, front-coded (see [`front_code`]).
static EN: &[u8] = include_bytes!("../data/en.words");

/// A sorted list of words, binary-searched.
struct Words {
    text: String,
    starts: Vec<u32>,
}

impl Words {
    fn decode(data: &[u8]) -> Words {
        let mut text = String::with_capacity(data.len() * 3);
        let mut starts = Vec::new();
        let mut prev: Vec<u8> = Vec::new();
        let mut i = 0;
        while i < data.len() {
            let shared = data[i] as usize;
            let end = data[i + 1..].iter().position(|&b| b == b'\n').map_or(data.len(), |p| i + 1 + p);
            prev.truncate(shared);
            prev.extend_from_slice(&data[i + 1..end]);
            starts.push(text.len() as u32);
            text.push_str(std::str::from_utf8(&prev).unwrap_or(""));
            i = end + 1;
        }
        starts.push(text.len() as u32);
        Words { text, starts }
    }

    fn get(&self, i: usize) -> &str {
        &self.text[self.starts[i] as usize..self.starts[i + 1] as usize]
    }

    fn len(&self) -> usize {
        self.starts.len() - 1
    }

    fn contains(&self, w: &str) -> bool {
        let (mut lo, mut hi) = (0, self.len());
        while lo < hi {
            let mid = (lo + hi) / 2;
            match self.get(mid).cmp(w) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return true,
            }
        }
        false
    }
}

fn english_words() -> &'static Words {
    static W: OnceLock<Words> = OnceLock::new();
    W.get_or_init(|| Words::decode(EN))
}

/// Front-codes sorted words: for each, a byte of how much it shares with
/// the word before (at most 255), the rest of it, a newline.
pub fn front_code(words: &[String]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut prev = "";
    for w in words {
        let shared = prev.bytes().zip(w.bytes()).take_while(|(a, b)| a == b).count().min(255);
        out.push(shared as u8);
        out.extend_from_slice(&w.as_bytes()[shared..]);
        out.push(b'\n');
        prev = w;
    }
    out
}

/// Whether `w` (lowercase) is English: listed, an inflection of a listed
/// word, a prefix on one, or two in one.
fn english_has(w: &str) -> bool {
    english_inflected(w) || english_derived(w)
}

/// Listed as it is: the word list, the common and newer words, irregular
/// forms, British spellings.
fn english_listed(w: &str) -> bool {
    english::common_rank(w).is_some()
        || english::NEWER.contains(&w)
        || english::IRREGULAR.contains(&w)
        || english::BRITISH.contains(&w)
        || english_words().contains(w)
}

/// Listed, or an ending on a listed word (walked, computers, happier),
/// two endings deep (workers).
fn english_inflected(w: &str) -> bool {
    suffixed(w, 2)
}

fn suffixed(w: &str, depth: u8) -> bool {
    if w.len() < 2 && w != "a" && w != "i" {
        return false;
    }
    if english_listed(w) {
        return true;
    }
    if depth == 0 {
        return false;
    }
    let base = |b: &str| b.len() >= 2 && suffixed(b, depth - 1);
    let strip = |suffix: &str| w.strip_suffix(suffix);
    // a doubled last consonant before -ed, -ing, -er, -est (stopped, running, bigger)
    let undouble = |stem: &str| {
        let b = stem.as_bytes();
        b.len() >= 3 && b[b.len() - 1] == b[b.len() - 2] && base(&stem[..stem.len() - 1])
    };
    if let Some(s) = strip("'s").or_else(|| strip("s'")) {
        return base(s) || base(&format!("{s}s"));
    }
    if let Some(s) = strip("ies").or_else(|| strip("ied")) {
        if base(&format!("{s}y")) {
            return true;
        }
    }
    // comparatives of -y adjectives, from bases of four letters or more
    // (easy, busy, happy): else "thier" would be "thy" compared
    if let Some(s) = strip("ier").or_else(|| strip("iest")).or_else(|| strip("ily")) {
        if s.len() >= 3 && base(&format!("{s}y")) {
            return true;
        }
    }
    for suffix in ["es", "s", "ed", "d", "ing", "er", "ers", "est", "ly", "ness", "ment", "ments", "able", "ings", "ful"] {
        if let Some(s) = strip(suffix) {
            // -d only after an e (baked), -ed not after one ("wierd" isn't "wier" + d)
            let fits = match suffix {
                "d" => s.ends_with('e'),
                "ed" => !s.ends_with('e'),
                _ => true,
            };
            if fits && (base(s) || undouble(s)) {
                return true;
            }
            // dropped e: baking, larger, usable
            if matches!(suffix, "ing" | "er" | "ers" | "est" | "ed" | "able" | "ings") && base(&format!("{s}e")) {
                return true;
            }
        }
    }
    false
}

/// A prefix on a word (unencrypted, reconnect, nonzero), or two words in
/// one (desktop, clipboard, sidebar): words the list can't have them all of.
fn english_derived(w: &str) -> bool {
    for prefix in ["un", "re", "non", "pre", "sub", "over", "under", "multi", "auto", "self", "out", "mis", "dis", "de", "co", "inter"] {
        if let Some(rest) = w.strip_prefix(prefix) {
            if rest.len() >= 3 && english_inflected(rest) {
                return true;
            }
        }
    }
    if w.len() >= 6 {
        for i in 3..=w.len() - 3 {
            if english_listed(&w[..i]) && english_inflected(&w[i..]) {
                return true;
            }
        }
    }
    false
}

/// The words in `text`: runs of letters (and apostrophes inside them).
pub fn words_in(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !(c.is_alphabetic() || c == '\''))
        .map(|w| w.trim_matches('\''))
        .filter(|w| !w.is_empty())
}

/// Where each letter is on a QWERTY keyboard (its rows staggered).
fn key_at(c: u8) -> Option<(f32, f32)> {
    const ROWS: [(&[u8], f32); 3] = [(b"qwertyuiop", 0.0), (b"asdfghjkl", 0.25), (b"zxcvbnm", 0.75)];
    ROWS.iter().enumerate().find_map(|(y, (row, dx))| row.iter().position(|&k| k == c).map(|x| (x as f32 + dx, y as f32)))
}

/// Whether `a` and `b` are keys next to each other.
fn neighbours(a: u8, b: u8) -> bool {
    match (key_at(a), key_at(b)) {
        (Some((ax, ay)), Some((bx, by))) => (ax - bx).abs() <= 1.0 && (ay - by).abs() <= 1.0 && (a != b),
        _ => false,
    }
}

/// The words a [`Speller`] knows, and fixes for the rest.
pub struct Speller {
    english: bool,
    /// The app's own words (lowercase) and how much each counts.
    own: HashMap<String, f32>,
    /// The words on screen now (lowercase).
    screen: HashMap<String, f32>,
}

const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz";

impl Speller {
    /// A speller knowing English (if `english`) and nothing else yet.
    pub fn new(english: bool) -> Self {
        Speller { english, own: HashMap::new(), screen: HashMap::new() }
    }

    pub fn set_english(&mut self, on: bool) {
        self.english = on;
    }

    /// The word list, read now rather than at the first check (it takes a
    /// few milliseconds).
    pub fn warm_up() {
        english_words();
    }

    /// A word of the app's own, counting `weight` more (typed more often).
    pub fn add_word(&mut self, word: &str, weight: f32) {
        *self.own.entry(word.to_lowercase()).or_default() += weight;
    }

    /// Every word in `text`, as [`add_word`](Self::add_word).
    pub fn add_words_in(&mut self, text: &str, weight: f32) {
        for w in words_in(text) {
            self.add_word(w, weight);
        }
    }

    /// Forget the app's own words.
    pub fn clear_words(&mut self) {
        self.own.clear();
    }

    /// The words on screen now: these, until the next call.
    pub fn set_screen(&mut self, text: &str) {
        self.screen.clear();
        for w in words_in(text) {
            *self.screen.entry(w.to_lowercase()).or_default() += 1.0;
        }
    }

    /// As `knows`, without working out prefixes and compounds (for the
    /// many words two edits away).
    fn knows_listed(&self, w: &str) -> bool {
        self.own.contains_key(w) || self.screen.contains_key(w) || (self.english && english_inflected(w))
    }

    /// Whether `w` (lowercase) is a word this speller knows.
    fn knows(&self, w: &str) -> bool {
        self.own.contains_key(w) || self.screen.contains_key(w) || (self.english && english_has(w))
    }

    /// Whether `word` is spelled right. What isn't a word of letters (a
    /// path, a flag, a number) always is.
    pub fn is_correct(&self, word: &str) -> bool {
        if !word.chars().all(|c| c.is_alphabetic() || c == '\'') || word.is_empty() {
            return true;
        }
        let w = word.to_lowercase();
        self.knows(&w) || (self.english && english::contraction(&w).is_some())
    }

    /// How much more likely a word is than another the same edits away
    /// (lower is likelier).
    fn prior(&self, w: &str) -> f32 {
        let mut p = 0.0;
        if self.english {
            if let Some(rank) = english::common_rank(w) {
                p -= 0.6 - 0.3 * (rank as f32 / english::COMMON.len() as f32);
            }
        }
        if let Some(c) = self.own.get(w) {
            p -= (0.15 * (1.0 + c).ln()).min(0.6);
        } else if self.english && !self.screen.contains_key(w) && !english_inflected(w) {
            // known only as a compound or a prefix on a word: a guess, not a listing
            p += 0.4;
        }
        if self.screen.contains_key(w) {
            p -= 0.2;
        }
        p
    }

    /// Words one edit from `w`, each with what the edit costs.
    fn edits(w: &[u8], out: &mut Vec<(Vec<u8>, f32)>) {
        let n = w.len();
        for i in 0..n {
            // a letter too many (a doubled one is the likeliest)
            let mut d = w.to_vec();
            d.remove(i);
            let doubled = (i > 0 && w[i - 1] == w[i]) || (i + 1 < n && w[i + 1] == w[i]);
            out.push((d, if doubled { 0.5 } else { 0.8 }));
            // two letters swapped
            if i + 1 < n && w[i] != w[i + 1] {
                let mut t = w.to_vec();
                t.swap(i, i + 1);
                out.push((t, 0.6));
            }
            // a wrong key (a neighbour, likelier)
            for &c in ALPHABET {
                if c != w[i] {
                    let mut r = w.to_vec();
                    r[i] = c;
                    out.push((r, if neighbours(w[i], c) { 0.7 } else { 1.0 }));
                }
            }
        }
        // a letter missing (a double typed once, likelier)
        for i in 0..=n {
            for &c in ALPHABET {
                let mut a = w.to_vec();
                a.insert(i, c);
                let doubles = (i > 0 && w[i - 1] == c) || (i < n && w[i] == c);
                out.push((a, if doubles { 0.6 } else { 0.85 }));
            }
        }
    }

    /// Fixes for `word`, best first, with what each costs (the edits and
    /// the word's likelihood), and what the edits alone cost.
    fn ranked(&self, word: &str) -> Vec<(String, f32, f32)> {
        let w = word.to_lowercase();
        if !w.bytes().all(|b| b.is_ascii_lowercase()) {
            return Vec::new();
        }
        let mut best: HashMap<Vec<u8>, (f32, f32)> = HashMap::new();
        let mut one = Vec::new();
        Self::edits(w.as_bytes(), &mut one);
        for (e, cost) in &one {
            if let Ok(s) = std::str::from_utf8(e) {
                if self.knows(s) {
                    let c = cost + self.prior(s);
                    let slot = best.entry(e.clone()).or_insert((f32::MAX, f32::MAX));
                    if c < slot.0 {
                        *slot = (c, *cost);
                    }
                }
            }
        }
        // two edits away, only for words long enough not to be anything:
        // the second a likely slip (swapped, a neighbour, a double), and
        // only listed words, so it stays quick
        if best.is_empty() && (5..=14).contains(&w.len()) {
            let mut two = Vec::new();
            let mut tried: std::collections::HashSet<&[u8]> = std::collections::HashSet::new();
            for (e, cost) in &one {
                if !tried.insert(e.as_slice()) {
                    continue;
                }
                two.clear();
                Self::edits(e, &mut two);
                for (e2, cost2) in &two {
                    if *cost2 > 0.7 {
                        continue;
                    }
                    if let Ok(s) = std::str::from_utf8(e2) {
                        if s != w && self.knows_listed(s) {
                            let c = cost + cost2 + self.prior(s);
                            let slot = best.entry(e2.clone()).or_insert((f32::MAX, f32::MAX));
                            if c < slot.0 {
                                *slot = (c, cost + cost2);
                            }
                        }
                    }
                }
            }
        }
        let mut v: Vec<(String, f32, f32)> =
            best.into_iter().filter_map(|(k, (c, raw))| String::from_utf8(k).ok().map(|s| (s, c, raw))).collect();
        v.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        v
    }

    /// Up to `n` fixes for a misspelled `word`, best first, in its case
    /// ("Teh" → "The"). None for a word spelled right.
    pub fn suggestions(&self, word: &str, n: usize) -> Vec<String> {
        if self.is_correct(word) {
            return Vec::new();
        }
        self.ranked(word).into_iter().take(n).map(|(s, ..)| in_case_of(word, &s)).collect()
    }

    /// The fix to make as `word` is ended (by a space), if one is sure
    /// enough: a word of three letters or more that isn't known, isn't in
    /// capitals (an acronym), with a best fix close enough. Contractions
    /// typed without their apostrophe ("dont") get it.
    pub fn autocorrect(&self, word: &str) -> Option<String> {
        let w = word.to_lowercase();
        if self.english {
            if let Some(c) = english::contraction(&w) {
                if !self.own.contains_key(&w) && !self.screen.contains_key(&w) {
                    return Some(in_case_of(word, c));
                }
            }
        }
        let letters = word.chars().count();
        // a capital past the first letter: an identifier or acronym (NSView, userId, NASA)
        if letters < 3 || word.chars().skip(1).any(|c| c.is_uppercase()) || self.is_correct(word) {
            return None;
        }
        let ranked = self.ranked(word);
        let (fix, cost, raw) = ranked.first()?;
        // a word everyone types, or one of the app's own
        let familiar = english::common_rank(fix).is_some() || english::NEWER.contains(&fix.as_str()) || self.own.contains_key(fix);
        let capital = word.chars().next().is_some_and(|c| c.is_uppercase());
        // short words and Names are jargon and names as often as slips: only
        // a slip a phone fixes (swapped, a neighbour, a double) to a common word
        if letters <= 4 || capital {
            // three letters: only a swap ("teh"); "dir", "src", "enc" are words here
            let most = if letters == 3 { 0.6 } else { 0.7 };
            return (familiar && *raw <= most).then(|| in_case_of(word, fix));
        }
        // one edit for shorter words; two only for long ones
        let most = if letters >= 6 { 1.7 } else { 1.05 };
        (*cost <= most).then(|| in_case_of(word, fix))
    }
}

/// `fix` in the case `typed` was in: "Teh" → "The"; "I"-words keep theirs.
fn in_case_of(typed: &str, fix: &str) -> String {
    if fix.starts_with("I'") || fix == "I" {
        return fix.to_owned();
    }
    let mut c = typed.chars();
    match c.next() {
        Some(f) if f.is_uppercase() => {
            let mut out: String = fix.chars().next().map(|f| f.to_uppercase().collect()).unwrap_or_default();
            out.push_str(&fix[fix.chars().next().map_or(0, |f| f.len_utf8())..]);
            out
        }
        _ => fix.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn front_coding_round_trips() {
        let words: Vec<String> = ["a", "ab", "abc", "abd", "b", "ba"].iter().map(|s| s.to_string()).collect();
        let w = Words::decode(&front_code(&words));
        assert_eq!((0..w.len()).map(|i| w.get(i)).collect::<Vec<_>>(), ["a", "ab", "abc", "abd", "b", "ba"]);
        assert!(w.contains("abd") && !w.contains("abe") && !w.contains(""));
    }

    #[test]
    fn english_with_its_inflections() {
        let s = Speller::new(true);
        for w in [
            "the", "cats", "walked", "running", "houses", "computers", "typed", "typing", "tried", "happier",
            "bigger", "stopped", "baking", "quickly", "email", "website", "dog's", "workers", "Walked", "don't",
        ] {
            assert!(s.is_correct(w), "{w}");
        }
        for w in ["teh", "recieve", "thier", "wierd", "becuase", "adn", "taht"] {
            assert!(!s.is_correct(w), "{w}");
        }
        // not words of letters: left alone
        for w in ["/usr/bin", "--force", "x86_64", "42", ""] {
            assert!(s.is_correct(w), "{w}");
        }
    }

    #[test]
    fn fixes_as_a_phone_keyboard_makes_them() {
        let s = Speller::new(true);
        for (typo, fix) in [
            ("teh", "the"),
            ("recieve", "receive"),
            ("thier", "their"),
            ("becuase", "because"),
            ("adn", "and"),
            ("taht", "that"),
            ("Teh", "The"),
            ("woudl", "would"),
            ("helo", "hello"),
            ("dont", "don't"),
            ("im", "I'm"),
            ("definately", "definitely"),
            ("seperate", "separate"),
        ] {
            assert_eq!(s.autocorrect(typo).as_deref(), Some(fix), "{typo}");
        }
        // left as they are: right, too short, acronyms and identifiers, not
        // words of letters, three letters a neighbouring key from a word
        // (terminals are full of those: dir, src, enc)
        for w in ["the", "ok", "NASA", "ls", "git", "/etc/hosts", "walked", "NSView", "userId", "dir", "src", "tge"] {
            assert_eq!(s.autocorrect(w), None, "{w}");
        }
        // though offered
        assert_eq!(s.suggestions("tge", 3).first().map(String::as_str), Some("the"));
        // words the list lacks, and aren't slips
        for w in ["desktop", "clipboard", "sidebar", "unencrypted", "held", "colour", "dialog", "passphrase"] {
            assert_eq!(s.autocorrect(w), None, "{w}");
        }
    }

    #[test]
    fn the_apps_own_words_and_the_screen() {
        let mut s = Speller::new(true);
        assert!(!s.is_correct("kubectl"));
        s.add_word("kubectl", 5.0);
        assert!(s.is_correct("kubectl"));
        assert_eq!(s.autocorrect("kubectll").as_deref(), Some("kubectl"));
        s.set_screen("drwxr-xr-x  2 me staff  64 Nov 20 deployment.yaml");
        assert!(s.is_correct("deployment") && s.is_correct("yaml"));
        // a word of its own isn't "fixed" to English
        s.add_word("teh", 1.0);
        assert_eq!(s.autocorrect("teh"), None);
        // without English, only its own words
        let mut s = Speller::new(false);
        assert!(!s.is_correct("the"));
        s.add_words_in("git commit -m 'the fix'", 1.0);
        assert!(s.is_correct("commit") && s.is_correct("the"));
    }

    #[test]
    fn checks_are_quick() {
        let s = Speller::new(true);
        Speller::warm_up();
        let t = std::time::Instant::now();
        for w in ["definately", "seperate", "becuase", "recieve", "accomodate"] {
            s.autocorrect(w);
        }
        // two-edit searches, debug build included
        assert!(t.elapsed().as_secs_f64() < 5.0, "{:?}", t.elapsed());
    }
}
