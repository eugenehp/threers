//! The keyboard shortcuts every text input answers to, as each platform has
//! them: ⌘ on macOS, Ctrl on Windows and Linux.
//!
//! [`text_command`] reads a key (DOM `KeyboardEvent.key` with its modifiers)
//! as a [`TextCommand`]: select all, copy, cut, paste, undo and redo, moving
//! and deleting by word and by line, and the formatting toggles (bold, italic,
//! underline) a rich-text editor takes. [`TextField::key`](crate::TextField::key)
//! and [`TextArea::key`](crate::text_area::TextArea::key) run them, so a
//! field edits as one does on the platform with no wiring.
//!
//! | | macOS | Windows, Linux |
//! |---|---|---|
//! | select all, copy, cut, paste | ⌘A ⌘C ⌘X ⌘V | Ctrl+A/C/X/V, Ctrl+Insert, Shift+Delete, Shift+Insert |
//! | undo, redo | ⌘Z, ⇧⌘Z | Ctrl+Z, Ctrl+Y or Ctrl+Shift+Z |
//! | by word | ⌥← ⌥→, ⌥⌫ ⌥⌦ | Ctrl+← Ctrl+→, Ctrl+Backspace Ctrl+Delete |
//! | to the line's ends | ⌘← ⌘→ (and ⌃A ⌃E), ⌘⌫, ⌃K | Home End |
//! | to the text's ends | ⌘↑ ⌘↓ | Ctrl+Home Ctrl+End |
//! | bold, italic, underline | ⌘B ⌘I ⌘U | Ctrl+B/I/U |
//!
//! Shift with a move extends the selection.
//!
//! Copy and paste go through the thread's [`Clipboard`]: by default one held
//! in memory, so text moves between the app's own fields; install the
//! system's with [`set_clipboard`] (threers' `clipboard` feature has one). A
//! clipboard that can't be read at once (the browser's) returns `None` from
//! [`Clipboard::read`]: the field leaves the key alone, and the platform's
//! own paste arrives as text ([`Event::Text`](crate::input::Event::Text)).
//!
//! The platform is the one the code is built for; in a browser, where one
//! build runs on every platform, set it from the user agent with
//! [`set_platform`].

use std::cell::RefCell;

use crate::text_field::KeyModifiers;

/// Whose shortcuts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    /// macOS and iOS: ⌘ for commands, ⌥ for words.
    Mac,
    /// Windows, Linux and the rest: Ctrl for commands and words.
    Other,
}

thread_local! {
    static PLATFORM: std::cell::Cell<Option<Platform>> = const { std::cell::Cell::new(None) };
    static CLIPBOARD: RefCell<Box<dyn Clipboard>> = RefCell::new(Box::new(MemoryClipboard::default()));
}

impl Platform {
    /// The platform shortcuts follow on this thread: [`set_platform`]'s, or
    /// the one the code is built for.
    pub fn current() -> Self {
        PLATFORM.with(|p| p.get()).unwrap_or(if cfg!(any(target_os = "macos", target_os = "ios")) {
            Platform::Mac
        } else {
            Platform::Other
        })
    }
}

/// Follow `platform`'s shortcuts on this thread (in a browser, from
/// `navigator.platform`).
pub fn set_platform(platform: Platform) {
    PLATFORM.with(|p| p.set(Some(platform)));
}

/// What a text input does for a shortcut. Moves carry whether Shift extends
/// the selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextCommand {
    SelectAll,
    Copy,
    Cut,
    Paste,
    Undo,
    Redo,
    WordLeft { extend: bool },
    WordRight { extend: bool },
    LineStart { extend: bool },
    LineEnd { extend: bool },
    TextStart { extend: bool },
    TextEnd { extend: bool },
    DeleteWordBack,
    DeleteWordForward,
    DeleteToLineStart,
    DeleteToLineEnd,
    /// Formatting, for a rich-text editor (a plain field leaves these to the
    /// app).
    Bold,
    Italic,
    Underline,
}

impl TextCommand {
    /// Bold, italic or underline.
    pub fn is_format(self) -> bool {
        matches!(self, Self::Bold | Self::Italic | Self::Underline)
    }
}

/// The shortcut `key` with `m` held is, on the [current](Platform::current)
/// platform.
pub fn text_command(key: &str, m: KeyModifiers) -> Option<TextCommand> {
    text_command_on(Platform::current(), key, m)
}

/// The shortcut `key` with `m` held is, on `platform`.
pub fn text_command_on(platform: Platform, key: &str, m: KeyModifiers) -> Option<TextCommand> {
    use TextCommand as T;
    let extend = m.shift;
    // A letter as typed: `A` with Shift, `a` without.
    let letter = {
        let mut cs = key.chars();
        match (cs.next(), cs.next()) {
            (Some(c), None) if c.is_ascii_alphabetic() => Some(c.to_ascii_lowercase()),
            _ => None,
        }
    };
    match platform {
        Platform::Mac => {
            let cmd = m.meta && !m.control && !m.alt;
            let opt = m.alt && !m.meta && !m.control;
            let ctrl = m.control && !m.meta && !m.alt;
            match (letter, key) {
                (Some('a'), _) if cmd => Some(T::SelectAll),
                (Some('c'), _) if cmd => Some(T::Copy),
                (Some('x'), _) if cmd => Some(T::Cut),
                (Some('v'), _) if cmd => Some(T::Paste),
                (Some('z'), _) if cmd && m.shift => Some(T::Redo),
                (Some('z'), _) if cmd => Some(T::Undo),
                (Some('b'), _) if cmd => Some(T::Bold),
                (Some('i'), _) if cmd => Some(T::Italic),
                (Some('u'), _) if cmd => Some(T::Underline),
                // The Emacs keys every Cocoa text field has.
                (Some('a'), _) if ctrl => Some(T::LineStart { extend }),
                (Some('e'), _) if ctrl => Some(T::LineEnd { extend }),
                (Some('k'), _) if ctrl => Some(T::DeleteToLineEnd),
                (_, "ArrowLeft") if opt => Some(T::WordLeft { extend }),
                (_, "ArrowRight") if opt => Some(T::WordRight { extend }),
                (_, "ArrowLeft") if cmd => Some(T::LineStart { extend }),
                (_, "ArrowRight") if cmd => Some(T::LineEnd { extend }),
                (_, "ArrowUp") if cmd => Some(T::TextStart { extend }),
                (_, "ArrowDown") if cmd => Some(T::TextEnd { extend }),
                (_, "Backspace") if opt => Some(T::DeleteWordBack),
                (_, "Delete") if opt => Some(T::DeleteWordForward),
                (_, "Backspace") if cmd => Some(T::DeleteToLineStart),
                (_, "Delete") if cmd => Some(T::DeleteToLineEnd),
                _ => None,
            }
        }
        Platform::Other => {
            let ctrl = m.control && !m.meta && !m.alt;
            let plain_shift = m.shift && !m.control && !m.meta && !m.alt;
            match (letter, key) {
                (Some('a'), _) if ctrl => Some(T::SelectAll),
                (Some('c'), _) if ctrl => Some(T::Copy),
                (Some('x'), _) if ctrl => Some(T::Cut),
                (Some('v'), _) if ctrl => Some(T::Paste),
                (Some('z'), _) if ctrl && m.shift => Some(T::Redo),
                (Some('z'), _) if ctrl => Some(T::Undo),
                (Some('y'), _) if ctrl => Some(T::Redo),
                (Some('b'), _) if ctrl => Some(T::Bold),
                (Some('i'), _) if ctrl => Some(T::Italic),
                (Some('u'), _) if ctrl => Some(T::Underline),
                (_, "Insert") if ctrl => Some(T::Copy),
                (_, "Insert") if plain_shift => Some(T::Paste),
                (_, "Delete") if plain_shift => Some(T::Cut),
                (_, "ArrowLeft") if ctrl => Some(T::WordLeft { extend }),
                (_, "ArrowRight") if ctrl => Some(T::WordRight { extend }),
                (_, "Home") if ctrl => Some(T::TextStart { extend }),
                (_, "End") if ctrl => Some(T::TextEnd { extend }),
                (_, "Backspace") if ctrl => Some(T::DeleteWordBack),
                (_, "Delete") if ctrl => Some(T::DeleteWordForward),
                _ => None,
            }
        }
    }
}

/// Whether a focused text input should hear `key` before the app's keymap:
/// its editing shortcuts (formatting aside), and the keys that type or move
/// the caret. Enter, Escape, Tab and other commands go to the keymap first,
/// as a page's shortcuts do.
pub fn text_input_owns(key: &str, m: KeyModifiers) -> bool {
    if let Some(c) = text_command(key, m) {
        return !c.is_format();
    }
    let command = m.control || m.meta;
    !command
        && (key.chars().count() == 1
            || matches!(
                key,
                "Backspace" | "Delete" | "ArrowLeft" | "ArrowRight" | "ArrowUp" | "ArrowDown" | "Home" | "End"
            ))
}

/// Where copied text goes and pasted text comes from.
pub trait Clipboard {
    /// The clipboard's text, or `None` when there is none or it can't be
    /// read at once (the field then leaves paste to the platform).
    fn read(&mut self) -> Option<String>;
    fn write(&mut self, text: &str);
}

/// A clipboard in memory: copy and paste between the app's own fields.
#[derive(Debug, Clone, Default)]
pub struct MemoryClipboard(pub Option<String>);

impl Clipboard for MemoryClipboard {
    fn read(&mut self) -> Option<String> {
        self.0.clone()
    }
    fn write(&mut self, text: &str) {
        self.0 = Some(text.to_owned());
    }
}

/// Text inputs on this thread copy to and paste from `clipboard` from now on.
pub fn set_clipboard(clipboard: impl Clipboard + 'static) {
    CLIPBOARD.with(|c| *c.borrow_mut() = Box::new(clipboard));
}

/// Runs `f` with this thread's clipboard.
pub fn with_clipboard<R>(f: impl FnOnce(&mut dyn Clipboard) -> R) -> R {
    CLIPBOARD.with(|c| f(&mut **c.borrow_mut()))
}

/// What kind of edit, for merging quick runs into one undo step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EditKind {
    Typing,
    Deleting,
    Other,
}

#[derive(Debug, Clone)]
struct Snapshot {
    text: String,
    anchor: usize,
    caret: usize,
}

/// A text input's own undo and redo: the text and selection before each
/// edit, typing and deleting merged a word at a time. Two inputs compare
/// equal whatever their histories.
#[derive(Debug, Clone, Default)]
pub struct EditHistory {
    undo: Vec<Snapshot>,
    redo: Vec<Snapshot>,
    last: Option<EditKind>,
}

/// Steps kept.
const HISTORY: usize = 200;

impl PartialEq for EditHistory {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl EditHistory {
    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Forgets everything (the text was replaced from outside).
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// Records the state `before` an edit of `kind`. Typing merges with the
    /// typing before it until a space; deleting with the deleting before it.
    pub(crate) fn record(&mut self, before: (&str, usize, usize), kind: EditKind, typed: Option<char>) {
        let merge = self.last == Some(kind) && kind != EditKind::Other && !typed.is_some_and(char::is_whitespace);
        if !merge {
            self.undo.push(Snapshot {
                text: before.0.to_owned(),
                anchor: before.1,
                caret: before.2,
            });
            if self.undo.len() > HISTORY {
                self.undo.remove(0);
            }
        }
        self.redo.clear();
        self.last = Some(kind);
    }

    /// Takes a step back from `now`: what to restore.
    pub(crate) fn undo(&mut self, now: (&str, usize, usize)) -> Option<(String, usize, usize)> {
        let s = self.undo.pop()?;
        self.redo.push(Snapshot {
            text: now.0.to_owned(),
            anchor: now.1,
            caret: now.2,
        });
        self.last = None;
        Some((s.text, s.anchor, s.caret))
    }

    pub(crate) fn redo(&mut self, now: (&str, usize, usize)) -> Option<(String, usize, usize)> {
        let s = self.redo.pop()?;
        self.undo.push(Snapshot {
            text: now.0.to_owned(),
            anchor: now.1,
            caret: now.2,
        });
        self.last = None;
        Some((s.text, s.anchor, s.caret))
    }
}

/// A text input the commands run on: [`TextField`](crate::TextField) and
/// [`TextArea`](crate::text_area::TextArea).
pub(crate) trait Editable {
    fn text(&self) -> &str;
    fn selection_ends(&self) -> (usize, usize);
    fn set_selection(&mut self, anchor: usize, caret: usize);
    fn replace_all(&mut self, text: String, anchor: usize, caret: usize);
    fn insert_text(&mut self, s: &str);
    fn history(&mut self) -> &mut EditHistory;
    /// The start and end (in characters) of the line holding `at`.
    fn line_bounds(&self, at: usize) -> (usize, usize);
}

/// What running a command did.
pub(crate) enum Ran {
    Edited,
    Moved,
    Copied,
    /// Nothing to do (an empty selection to copy, nothing to undo, an
    /// unreadable clipboard, formatting in a plain input): the key is the
    /// app's.
    Unused,
}

#[derive(PartialEq, Clone, Copy)]
enum Class {
    Space,
    Word,
    Punct,
}

fn class(c: char) -> Class {
    if c.is_whitespace() {
        Class::Space
    } else if c.is_alphanumeric() || c == '_' {
        Class::Word
    } else {
        Class::Punct
    }
}

/// Where a word move left of `from` lands: back over spaces, then over a
/// run of one kind (letters, or punctuation).
pub(crate) fn word_left(chars: &[char], from: usize) -> usize {
    let mut i = from.min(chars.len());
    while i > 0 && class(chars[i - 1]) == Class::Space {
        i -= 1;
    }
    if i > 0 {
        let k = class(chars[i - 1]);
        while i > 0 && class(chars[i - 1]) == k {
            i -= 1;
        }
    }
    i
}

/// Where a word move right of `from` lands: on macOS the end of the next
/// word, elsewhere the start of the one after (as each platform's fields do).
pub(crate) fn word_right(chars: &[char], from: usize, platform: Platform) -> usize {
    let n = chars.len();
    let mut i = from.min(n);
    match platform {
        Platform::Mac => {
            while i < n && class(chars[i]) == Class::Space {
                i += 1;
            }
            if i < n {
                let k = class(chars[i]);
                while i < n && class(chars[i]) == k {
                    i += 1;
                }
            }
        }
        Platform::Other => {
            if i < n {
                let k = class(chars[i]);
                while i < n && class(chars[i]) == k && k != Class::Space {
                    i += 1;
                }
            }
            while i < n && class(chars[i]) == Class::Space {
                i += 1;
            }
        }
    }
    i
}

/// Runs `cmd` on `e`; `accept` filters pasted characters as typing does.
pub(crate) fn run(e: &mut impl Editable, cmd: TextCommand, accept: &dyn Fn(char) -> bool) -> Ran {
    use TextCommand as T;
    let chars: Vec<char> = e.text().chars().collect();
    let n = chars.len();
    let (anchor, caret) = e.selection_ends();
    let (a, z) = (anchor.min(caret), anchor.max(caret));
    let selected: String = chars[a..z].iter().collect();
    let before = (e.text().to_owned(), anchor, caret);
    let record = |e: &mut dyn EditableDyn, kind| {
        e.history_dyn().record((&before.0, before.1, before.2), kind, None)
    };
    let mv = |e: &mut dyn EditableDyn, to: usize, extend: bool| {
        let to = to.min(n);
        e.set_selection_dyn(if extend { anchor } else { to }, to);
        Ran::Moved
    };
    // Deletes from `from` to `to` (either order), or the selection if there
    // is one.
    let delete = |e: &mut dyn EditableDyn, from: usize, to: usize| {
        if a == z {
            if from == to {
                return Ran::Unused;
            }
            e.set_selection_dyn(from.min(to), from.max(to));
        }
        record(e, EditKind::Deleting);
        e.insert_dyn("");
        Ran::Edited
    };
    let e: &mut dyn EditableDyn = e;
    match cmd {
        T::SelectAll => {
            e.set_selection_dyn(0, n);
            Ran::Moved
        }
        T::Copy | T::Cut if a == z => Ran::Unused,
        T::Copy => {
            with_clipboard(|c| c.write(&selected));
            Ran::Copied
        }
        T::Cut => {
            with_clipboard(|c| c.write(&selected));
            record(e, EditKind::Other);
            e.insert_dyn("");
            Ran::Edited
        }
        T::Paste => match with_clipboard(|c| c.read()) {
            Some(text) => {
                let text: String = text.chars().filter(|&c| accept(c)).collect();
                record(e, EditKind::Other);
                e.insert_dyn(&text);
                Ran::Edited
            }
            None => Ran::Unused,
        },
        T::Undo | T::Redo => {
            let now = (before.0.as_str(), before.1, before.2);
            let step = if cmd == T::Undo {
                e.history_dyn().undo(now)
            } else {
                e.history_dyn().redo(now)
            };
            match step {
                Some((text, anchor, caret)) => {
                    e.replace_all_dyn(text, anchor, caret);
                    Ran::Edited
                }
                None => Ran::Unused,
            }
        }
        T::WordLeft { extend } => {
            let from = if a != z && !extend { a } else { caret };
            mv(e, if a != z && !extend { a } else { word_left(&chars, from) }, extend)
        }
        T::WordRight { extend } => {
            let to = if a != z && !extend {
                z
            } else {
                word_right(&chars, caret, Platform::current())
            };
            mv(e, to, extend)
        }
        T::LineStart { extend } => {
            let start = e.line_bounds_dyn(caret).0;
            mv(e, start, extend)
        }
        T::LineEnd { extend } => {
            let end = e.line_bounds_dyn(caret).1;
            mv(e, end, extend)
        }
        T::TextStart { extend } => mv(e, 0, extend),
        T::TextEnd { extend } => mv(e, n, extend),
        T::DeleteWordBack => delete(e, word_left(&chars, caret), caret),
        T::DeleteWordForward => delete(e, caret, word_right(&chars, caret, Platform::Mac)),
        T::DeleteToLineStart => {
            let start = e.line_bounds_dyn(caret).0;
            delete(e, start, caret)
        }
        T::DeleteToLineEnd => {
            let end = e.line_bounds_dyn(caret).1;
            delete(e, caret, end)
        }
        T::Bold | T::Italic | T::Underline => Ran::Unused,
    }
}

impl Ran {
    pub(crate) fn field_key(self) -> crate::FieldKey {
        match self {
            Ran::Edited => crate::FieldKey::Edited,
            Ran::Moved => crate::FieldKey::Moved,
            Ran::Copied => crate::FieldKey::Copied,
            Ran::Unused => crate::FieldKey::Ignored,
        }
    }
}

/// Inserts `text` over the selection as one undo step (a host's paste).
pub(crate) fn run_insert(e: &mut impl Editable, text: &str) {
    let (anchor, caret) = e.selection_ends();
    let before = e.text().to_owned();
    e.history().record((&before, anchor, caret), EditKind::Other, None);
    e.insert_text(text);
}

/// Records the state `before` a plain key edited the text: typing merges a
/// word at a time, Backspace and Delete with each other.
pub(crate) fn record_key(history: &mut EditHistory, before: (String, usize, usize), key: &str) {
    let mut cs = key.chars();
    let (kind, typed) = match (cs.next(), cs.next()) {
        (Some(c), None) => (EditKind::Typing, Some(c)),
        _ if matches!(key, "Backspace" | "Delete") => (EditKind::Deleting, None),
        _ => (EditKind::Other, None),
    };
    history.record((&before.0, before.1, before.2), kind, typed);
}

/// [`Editable`] as a trait object, for the closures above.
trait EditableDyn {
    fn set_selection_dyn(&mut self, anchor: usize, caret: usize);
    fn replace_all_dyn(&mut self, text: String, anchor: usize, caret: usize);
    fn insert_dyn(&mut self, s: &str);
    fn history_dyn(&mut self) -> &mut EditHistory;
    fn line_bounds_dyn(&self, at: usize) -> (usize, usize);
}

impl<T: Editable> EditableDyn for T {
    fn set_selection_dyn(&mut self, anchor: usize, caret: usize) {
        self.set_selection(anchor, caret)
    }
    fn replace_all_dyn(&mut self, text: String, anchor: usize, caret: usize) {
        self.replace_all(text, anchor, caret)
    }
    fn insert_dyn(&mut self, s: &str) {
        self.insert_text(s)
    }
    fn history_dyn(&mut self) -> &mut EditHistory {
        self.history()
    }
    fn line_bounds_dyn(&self, at: usize) -> (usize, usize) {
        self.line_bounds(at)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(spec: &str) -> KeyModifiers {
        KeyModifiers {
            shift: spec.contains("shift"),
            control: spec.contains("ctrl"),
            alt: spec.contains("alt"),
            meta: spec.contains("cmd"),
        }
    }

    #[test]
    fn each_platform_has_its_own_shortcuts() {
        use Platform::{Mac, Other};
        use TextCommand as T;
        let on = |p, k, s| text_command_on(p, k, m(s));
        assert_eq!(on(Mac, "a", "cmd"), Some(T::SelectAll));
        assert_eq!(on(Other, "a", "ctrl"), Some(T::SelectAll));
        // ⌃A on a Mac is the line's start; ⌘A elsewhere is nothing.
        assert_eq!(on(Mac, "a", "ctrl"), Some(T::LineStart { extend: false }));
        assert_eq!(on(Other, "a", "cmd"), None);
        assert_eq!(on(Mac, "Z", "cmd shift"), Some(T::Redo));
        assert_eq!(on(Other, "y", "ctrl"), Some(T::Redo));
        assert_eq!(on(Mac, "y", "cmd"), None);
        assert_eq!(on(Mac, "ArrowLeft", "alt shift"), Some(T::WordLeft { extend: true }));
        assert_eq!(on(Other, "ArrowLeft", "ctrl"), Some(T::WordLeft { extend: false }));
        assert_eq!(on(Mac, "ArrowLeft", "cmd"), Some(T::LineStart { extend: false }));
        assert_eq!(on(Other, "Delete", "shift"), Some(T::Cut));
        assert_eq!(on(Other, "Insert", "shift"), Some(T::Paste));
        assert_eq!(on(Mac, "b", "cmd"), Some(T::Bold));
        assert_eq!(on(Other, "b", "ctrl"), Some(T::Bold));
        // Plain typing is no command.
        assert_eq!(on(Mac, "a", ""), None);
        assert_eq!(on(Other, "c", "shift"), None);
    }

    #[test]
    fn words_are_runs_of_one_kind() {
        let s: Vec<char> = "let x = foo.bar(1);".chars().collect();
        assert_eq!(word_left(&s, 11), 8);
        assert_eq!(word_left(&s, 8), 6);
        assert_eq!(word_right(&s, 0, Platform::Mac), 3);
        assert_eq!(word_right(&s, 0, Platform::Other), 4);
        assert_eq!(word_right(&s, 8, Platform::Mac), 11);
    }

    #[test]
    fn typing_undoes_a_word_at_a_time() {
        let mut h = EditHistory::default();
        h.record(("", 0, 0), EditKind::Typing, Some('h'));
        h.record(("h", 1, 1), EditKind::Typing, Some('i'));
        h.record(("hi", 2, 2), EditKind::Typing, Some(' '));
        h.record(("hi ", 3, 3), EditKind::Typing, Some('y'));
        assert_eq!(h.undo(("hi y", 4, 4)).map(|s| s.0), Some("hi".into()));
        assert_eq!(h.undo(("hi", 2, 2)).map(|s| s.0), Some("".into()));
        assert_eq!(h.redo(("", 0, 0)).map(|s| s.0), Some("hi".into()));
    }
}
