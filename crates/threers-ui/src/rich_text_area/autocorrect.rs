//! Correcting text as it is typed, as Word and a phone's keyboard do:
//!
//! - a misspelt word fixed as it ends (a space, punctuation, a new
//!   paragraph), when threers-spell is sure enough of the fix
//!   ([`Speller::autocorrect`](threers_spell::Speller::autocorrect)), in the
//!   word's own case;
//! - a sentence's first word capitalised, and "i" made "I";
//! - straight quotes curled (“ ” ‘ ’, an apostrophe inside a word ’);
//! - two hyphens an em dash;
//! - two spaces after a word a full stop and a space (a phone's habit: off
//!   unless asked for);
//!
//! Backspace straight after a correction puts the word back as it was
//! typed, and leaves it alone after.

use std::collections::HashSet;

use threers_spell::Speller;

use super::{Document, Pos, Selection};

/// A correction just made, to undo with Backspace.
#[derive(Clone, Debug)]
struct Applied {
    para: usize,
    start: usize,
    /// Where the correction ends, and the character typed after it.
    end: usize,
    term: usize,
    original: String,
}

pub struct Autocorrect {
    /// Misspellings fixed as words end.
    pub spelling: bool,
    pub capitalize: bool,
    pub smart_quotes: bool,
    pub smart_dashes: bool,
    pub double_space_period: bool,
    speller: Option<Speller>,
    ignored: HashSet<String>,
    last: Option<Applied>,
}

impl Default for Autocorrect {
    fn default() -> Self {
        Self::new()
    }
}

fn ends_word(c: char) -> bool {
    c.is_whitespace() || matches!(c, '.' | ',' | ';' | ':' | '!' | '?' | ')' | ']' | '}' | '"' | '”')
}

fn is_word(c: char) -> bool {
    c.is_alphabetic() || c == '\'' || c == '’'
}

/// `word` with `like`'s case: all capitals, a capital first, or as it is.
fn same_case(word: &str, like: &str) -> String {
    if like.chars().count() > 1 && like.chars().all(|c| !c.is_lowercase()) {
        word.to_uppercase()
    } else if like.chars().next().is_some_and(char::is_uppercase) {
        capitalized(word)
    } else {
        word.to_string()
    }
}

fn capitalized(w: &str) -> String {
    let mut c = w.chars();
    c.next().map_or(String::new(), |f| f.to_uppercase().chain(c).collect())
}

impl Autocorrect {
    pub fn new() -> Autocorrect {
        Autocorrect {
            spelling: true,
            capitalize: true,
            smart_quotes: true,
            smart_dashes: true,
            double_space_period: false,
            speller: None,
            ignored: HashSet::new(),
            last: None,
        }
    }

    /// Every correction off (a code editor's).
    pub fn off() -> Autocorrect {
        Autocorrect { spelling: false, capitalize: false, smart_quotes: false, smart_dashes: false, double_space_period: false, ..Autocorrect::new() }
    }

    fn speller(&mut self) -> &Speller {
        self.speller.get_or_insert_with(|| Speller::new(true))
    }

    pub fn is_correct(&mut self, word: &str) -> bool {
        let w = word.replace('’', "'");
        if self.ignored.contains(&w.to_lowercase()) {
            return true;
        }
        let sp = self.speller();
        sp.is_correct(&w) || sp.is_correct(&w.to_lowercase())
    }

    pub fn suggestions(&mut self, word: &str, n: usize) -> Vec<String> {
        let w = word.replace('’', "'");
        self.speller().suggestions(&w.to_lowercase(), n).into_iter().map(|s| same_case(&s, word)).collect()
    }

    /// A word no longer marked or corrected.
    pub fn learn(&mut self, word: &str) {
        self.ignored.insert(word.to_lowercase());
        self.speller.get_or_insert_with(|| Speller::new(true)).add_word(word, 5.0);
    }

    /// Nothing to put back: the caret moved on.
    pub fn forget(&mut self) {
        self.last = None;
    }

    /// The character before the caret, and the one before that.
    fn before(doc: &dyn Document, p: Pos) -> (Option<char>, Option<char>) {
        let t = doc.text(p.para);
        let mut it = t[..p.off.min(t.len())].chars().rev();
        (it.next(), it.next())
    }

    /// A character typed at the caret: what to put in (maybe nothing, maybe
    /// other than it), the text before it corrected as it ends a word.
    pub fn typed(&mut self, doc: &mut dyn Document, ch: &str) -> String {
        let s = doc.selection();
        if !s.is_caret() {
            self.last = None;
            return ch.to_string();
        }
        let c = ch.chars().next().unwrap_or(' ');
        let p = s.head;
        let (b1, b2) = Self::before(doc, p);
        // two hyphens, a dash
        if self.smart_dashes && c == '-' && b1 == Some('-') && b2 != Some('-') {
            let end = doc.replace(Pos::new(p.para, p.off - 1), p, "—", true);
            doc.set_selection(Selection::caret(end));
            self.last = None;
            return String::new();
        }
        // two spaces after a word, a full stop
        if self.double_space_period && c == ' ' && b1 == Some(' ') && b2.is_some_and(char::is_alphanumeric) {
            let end = doc.replace(Pos::new(p.para, p.off - 1), p, ". ", true);
            doc.set_selection(Selection::caret(end));
            self.last = None;
            return String::new();
        }
        if ends_word(c) {
            self.correct_word(doc, ch.len());
        } else {
            self.last = None;
        }
        // quotes curled by what is before them
        if self.smart_quotes && (c == '"' || c == '\'') {
            let p = doc.selection().head;
            let (b1, _) = Self::before(doc, p);
            let opening = b1.is_none_or(|b| b.is_whitespace() || matches!(b, '(' | '[' | '{' | '—' | '–' | '“' | '‘'));
            return match (c, opening) {
                ('"', true) => "“",
                ('"', false) => "”",
                (_, true) => "‘",
                (_, false) => "’",
            }
            .to_string();
        }
        ch.to_string()
    }

    /// A word ends without a character after it (a new paragraph).
    pub fn word_ends(&mut self, doc: &mut dyn Document) {
        if doc.selection().is_caret() {
            self.correct_word(doc, 0);
        }
    }

    /// The word before the caret corrected; `term` bytes will follow it.
    fn correct_word(&mut self, doc: &mut dyn Document, term: usize) {
        self.last = None;
        let p = doc.selection().head;
        let text = doc.text(p.para);
        let off = p.off.min(text.len());
        let mut start = off;
        for (i, c) in text[..off].char_indices().rev() {
            if !is_word(c) {
                break;
            }
            start = i;
        }
        let word = text[start..off].trim_end_matches(['\'', '’']).to_string();
        if word.is_empty() || self.ignored.contains(&word.to_lowercase()) {
            return;
        }
        let before = text[..start].chars().next_back();
        // not a word of something else (a path, a number, an address)
        if before.is_some_and(|c| c.is_ascii_digit() || matches!(c, '_' | '@' | '/' | '\\' | '.' | '#' | '-')) && before != Some('.') {
            return;
        }
        if before == Some('.') && text[..start.saturating_sub(1)].chars().next_back().is_some_and(char::is_alphanumeric) {
            return;
        }
        let mut fix = word.clone();
        if word == "i" && self.capitalize {
            fix = "I".into();
        } else if self.spelling && word.chars().count() > 1 && !word.chars().skip(1).any(char::is_uppercase) && !self.is_correct(&word) {
            let lower = word.to_lowercase().replace('’', "'");
            if let Some(f) = self.speller().autocorrect(&lower) {
                fix = same_case(&f, &word);
            }
        }
        // a sentence's first word
        if self.capitalize && fix.chars().next().is_some_and(char::is_lowercase) {
            let prev = text[..start].trim_end_matches([' ', '\t', '“', '"', '‘', '(']).chars().next_back();
            let starts_sentence = prev.is_none_or(|c| matches!(c, '.' | '!' | '?'));
            // after a full stop, not an abbreviation's ("e.g. this")
            let abbreviation = prev == Some('.') && text[..start].trim_end().split_whitespace().last().is_some_and(|w| w.matches('.').count() > 1);
            if starts_sentence && !abbreviation {
                fix = capitalized(&fix);
            }
        }
        if fix != word {
            let end = doc.replace(Pos::new(p.para, start), Pos::new(p.para, start + word.len()), &fix, true);
            let after = Pos::new(p.para, end.off + (off - start - word.len()));
            doc.set_selection(Selection::caret(after));
            self.last = Some(Applied { para: p.para, start, end: after.off, term, original: text[start..off].to_string() });
        }
    }

    /// Backspace straight after a correction: the word as typed, the
    /// character after it gone. Whether there was one to put back.
    pub fn revert(&mut self, doc: &mut dyn Document) -> bool {
        let Some(a) = self.last.take() else { return false };
        let p = doc.selection().head;
        if p != Pos::new(a.para, a.end + a.term) {
            return false;
        }
        doc.checkpoint();
        let end = doc.replace(Pos::new(a.para, a.start), p, &a.original, true);
        doc.set_selection(Selection::caret(end));
        self.ignored.insert(a.original.to_lowercase());
        true
    }

    /// What the word being typed would be corrected to, and where it is.
    pub fn pending(&mut self, doc: &dyn Document) -> Option<(Pos, Pos, String)> {
        let s = doc.selection();
        if !s.is_caret() || !self.spelling {
            return None;
        }
        let p = s.head;
        let text = doc.text(p.para);
        let mut start = p.off.min(text.len());
        for (i, c) in text[..p.off.min(text.len())].char_indices().rev() {
            if !is_word(c) {
                break;
            }
            start = i;
        }
        let word = &text[start..p.off.min(text.len())];
        if word.chars().count() < 2 || self.is_correct(word) {
            return None;
        }
        let fix = self.speller().autocorrect(&word.to_lowercase())?;
        Some((Pos::new(p.para, start), p, same_case(&fix, word)))
    }

    /// The misspelt words in a paragraph's text: byte ranges. Names in
    /// capitals, words beside digits, addresses and paths are left alone.
    pub fn misspelled(&mut self, text: &str) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < text.len() {
            let c = text[i..].chars().next().unwrap();
            if !c.is_alphabetic() {
                i += c.len_utf8();
                continue;
            }
            let start = i;
            let mut end = i;
            for (j, ch) in text[i..].char_indices() {
                if ch.is_alphabetic() || (ch == '\'' || ch == '’') && text[i + j + ch.len_utf8()..].chars().next().is_some_and(char::is_alphabetic) {
                    end = i + j + ch.len_utf8();
                } else {
                    break;
                }
            }
            i = end.max(start + c.len_utf8());
            let word = &text[start..end];
            let after = text[end..].chars().next();
            let before = text[..start].chars().next_back();
            if word.chars().count() < 2
                || word.chars().all(char::is_uppercase)
                || word.chars().skip(1).any(char::is_uppercase)
                || after.is_some_and(|c| c.is_ascii_digit() || c == '_' || c == '@' || c == '.' && text[end + 1..].chars().next().is_some_and(char::is_alphanumeric))
                || before.is_some_and(|c| c.is_ascii_digit() || matches!(c, '_' | '@' | '/' | '\\' | '.'))
                || !word.is_ascii() && !word.contains('’')
            {
                continue;
            }
            if !self.is_correct(word) {
                out.push((start, end));
            }
        }
        out
    }
}
