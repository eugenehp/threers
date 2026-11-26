//! Keys to editing: each platform's shortcuts for a rich text area, as
//! bindings an app can change.
//!
//! | | macOS | Windows, Linux |
//! |---|---|---|
//! | by character, word | ← →, ⌥← ⌥→ (and ⌃B ⌃F) | ← →, Ctrl+← Ctrl+→ |
//! | by line | ↑ ↓ (⌃P ⌃N), ⌘← ⌘→ (⌃A ⌃E) to its ends | ↑ ↓, Home End |
//! | by paragraph, page, text | ⌥↑ ⌥↓, PgUp PgDn, ⌘↑ ⌘↓ (Home End) | Ctrl+↑ Ctrl+↓, PgUp PgDn, Ctrl+Home Ctrl+End |
//! | deleting | ⌫ ⌦ (⌃H ⌃D), ⌥⌫ ⌥⌦ a word, ⌘⌫ to the line's start, ⌃K to the paragraph's end | Backspace Delete, Ctrl+Backspace Ctrl+Delete a word |
//! | clipboard, undo | ⌘X ⌘C ⌘V ⌥⇧⌘V, ⌘Z ⇧⌘Z, ⌘A | Ctrl+X C V, Shift+Delete, Ctrl+Insert, Shift+Insert, Ctrl+Z Ctrl+Y, Ctrl+A |
//! | paragraphs | ↩ a new one, ⇧↩ a line break, ⌘↩ a page break, ⇥ ⇧⇥ | the same with Ctrl |
//! | formatting | ⌘B ⌘I ⌘U, ⇧⌘X struck, ⌘= ⇧⌘= sub- and superscript, ⇧⌘. ⇧⌘, bigger and smaller, ⌃Space plain | Ctrl and the same keys |
//! | paragraph look | ⌘L ⌘E ⌘R ⌘J, ⌘[ ⌘], ⌥⌘B ⌥⌘N lists, ⌥⌘1–3 headings, ⇧⌘N Normal | Ctrl and the same keys |
//! | find | ⌘F, ⌘G ⇧⌘G | Ctrl+F, F3 Shift+F3 |
//!
//! Shift with a move extends the selection.

use super::Motion;

pub use crate::text_shortcuts::Platform;

/// Modifier keys held (⌘ is `meta`).
pub use crate::KeyModifiers as Mods;

/// The clipboard's commands: the app reaches the clipboard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Clip {
    Copy,
    Cut,
    Paste,
    /// Paste without the formatting.
    PastePlain,
}

/// A command that is the app's to carry out (the text area knows nothing
/// of formatting or paragraphs' kinds).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    Enter,
    LineBreak,
    PageBreak,
    Tab,
    BackTab,
    Undo,
    Redo,
    Bold,
    Italic,
    Underline,
    Strike,
    Superscript,
    Subscript,
    Bigger,
    Smaller,
    ClearFormatting,
    AlignLeft,
    AlignCenter,
    AlignRight,
    Justify,
    Indent,
    Outdent,
    Bullets,
    Numbering,
    /// A paragraph style by its id (`Heading1`, `Normal`).
    Style(String),
    Find,
    FindNext,
    FindPrevious,
    Escape,
    /// One of the app's own, by name.
    Other(String),
}

/// What a key does.
#[derive(Clone, Debug, PartialEq)]
pub enum KeyAction {
    /// Move the caret (extending the selection with `true`).
    Move(Motion, bool),
    /// Delete by a motion from the caret (or the selection).
    Delete(Motion),
    SelectAll,
    Clipboard(Clip),
    Command(Command),
    /// Type this.
    Text(String),
}

/// A key and its modifiers, as `"cmd-shift-z"`, `"alt-ArrowLeft"`,
/// `"ctrl-Home"` (`cmd`/`meta`, `ctrl`, `alt`/`opt`, `shift`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Combo {
    pub key: String,
    pub mods: Mods,
}

impl Combo {
    pub fn parse(s: &str) -> Option<Combo> {
        let mut mods = Mods::default();
        let parts: Vec<&str> = s.split('-').collect();
        let (key, rest) = match parts.split_last()? {
            // "cmd--": the minus key
            (k, rest) if k.is_empty() && !rest.is_empty() => ("-", &rest[..rest.len() - 1]),
            (k, rest) => (*k, rest),
        };
        for m in rest {
            match *m {
                "cmd" | "meta" | "super" => mods.meta = true,
                "ctrl" => mods.control = true,
                "alt" | "opt" | "option" => mods.alt = true,
                "shift" => mods.shift = true,
                _ => return None,
            }
        }
        Some(Combo { key: normal_key(key), mods })
    }
}

/// A key's name as bindings use it: letters lower case, `Space` for the space bar.
fn normal_key(k: &str) -> String {
    match k {
        " " => "Space".into(),
        "Esc" => "Escape".into(),
        "Del" => "Delete".into(),
        "Left" => "ArrowLeft".into(),
        "Right" => "ArrowRight".into(),
        "Up" => "ArrowUp".into(),
        "Down" => "ArrowDown".into(),
        k if k.chars().count() == 1 => k.to_lowercase(),
        k => k.into(),
    }
}

/// The keys bound, and what each does.
#[derive(Clone, Debug)]
pub struct Keymap {
    bindings: Vec<(Combo, KeyAction)>,
    platform: Platform,
}

impl Keymap {
    pub fn empty(platform: Platform) -> Keymap {
        Keymap { bindings: Vec::new(), platform }
    }

    /// Binds (or rebinds) a key; `false` for a key that doesn't parse.
    pub fn bind(&mut self, combo: &str, action: KeyAction) -> bool {
        let Some(c) = Combo::parse(combo) else { return false };
        self.bindings.retain(|(b, _)| *b != c);
        self.bindings.push((c, action));
        true
    }

    pub fn unbind(&mut self, combo: &str) {
        if let Some(c) = Combo::parse(combo) {
            self.bindings.retain(|(b, _)| *b != c);
        }
    }

    /// What a key does (a printable key with no command modifier types itself).
    pub fn action(&self, key: &str, mods: Mods) -> Option<KeyAction> {
        let c = Combo { key: normal_key(key), mods };
        if let Some((_, a)) = self.bindings.iter().find(|(b, _)| *b == c) {
            return Some(a.clone());
        }
        // shifted punctuation comes as its own key (`+`): try it unshifted too
        if mods.shift && key.chars().count() == 1 {
            let c = Combo { key: normal_key(key), mods: Mods { shift: false, ..mods } };
            if let Some((_, a)) = self.bindings.iter().find(|(b, _)| *b == c && !key.chars().all(char::is_alphanumeric)) {
                return Some(a.clone());
            }
        }
        let command_mod = if self.platform == Platform::Mac { mods.meta || mods.control } else { mods.control || mods.meta };
        if key.chars().count() == 1 && !command_mod {
            return Some(KeyAction::Text(key.to_string()));
        }
        None
    }

    /// The shortcut bound to an action, to show in a menu (`⇧⌘Z`, `Ctrl+Shift+Z`).
    pub fn label(&self, action: &KeyAction) -> Option<String> {
        let (c, _) = self.bindings.iter().find(|(_, a)| a == action)?;
        let key = match c.key.as_str() {
            "ArrowLeft" => "←".to_string(),
            "ArrowRight" => "→".to_string(),
            "ArrowUp" => "↑".to_string(),
            "ArrowDown" => "↓".to_string(),
            "Enter" => "↩".to_string(),
            "Backspace" => "⌫".to_string(),
            "Space" => "Space".to_string(),
            k => k.to_uppercase(),
        };
        Some(if self.platform == Platform::Mac {
            let mut s = String::new();
            if c.mods.control {
                s.push('⌃');
            }
            if c.mods.alt {
                s.push('⌥');
            }
            if c.mods.shift {
                s.push('⇧');
            }
            if c.mods.meta {
                s.push('⌘');
            }
            s + &key
        } else {
            let mut parts = Vec::new();
            if c.mods.control {
                parts.push("Ctrl");
            }
            if c.mods.alt {
                parts.push("Alt");
            }
            if c.mods.shift {
                parts.push("Shift");
            }
            parts.join("+") + if parts.is_empty() { "" } else { "+" } + &key
        })
    }

    /// The platform's bindings.
    pub fn defaults(platform: Platform) -> Keymap {
        let mut k = Keymap::empty(platform);
        let mac = platform == Platform::Mac;
        // the command key: ⌘ on the Mac, Ctrl elsewhere
        let c = if mac { "cmd" } else { "ctrl" };
        let m = |x: Motion, e: bool| KeyAction::Move(x, e);
        let moves: Vec<(&str, Motion)> = if mac {
            vec![
                ("ArrowLeft", Motion::Left),
                ("ArrowRight", Motion::Right),
                ("ArrowUp", Motion::Up),
                ("ArrowDown", Motion::Down),
                ("alt-ArrowLeft", Motion::WordLeft),
                ("alt-ArrowRight", Motion::WordRight),
                ("alt-ArrowUp", Motion::ParaStart),
                ("alt-ArrowDown", Motion::ParaEnd),
                ("cmd-ArrowLeft", Motion::LineStart),
                ("cmd-ArrowRight", Motion::LineEnd),
                ("cmd-ArrowUp", Motion::DocStart),
                ("cmd-ArrowDown", Motion::DocEnd),
                ("Home", Motion::DocStart),
                ("End", Motion::DocEnd),
                ("PageUp", Motion::PageUp),
                ("PageDown", Motion::PageDown),
                ("ctrl-b", Motion::Left),
                ("ctrl-f", Motion::Right),
                ("ctrl-p", Motion::Up),
                ("ctrl-n", Motion::Down),
                ("ctrl-a", Motion::ParaStart),
                ("ctrl-e", Motion::ParaEnd),
            ]
        } else {
            vec![
                ("ArrowLeft", Motion::Left),
                ("ArrowRight", Motion::Right),
                ("ArrowUp", Motion::Up),
                ("ArrowDown", Motion::Down),
                ("ctrl-ArrowLeft", Motion::WordLeft),
                ("ctrl-ArrowRight", Motion::WordRight),
                ("ctrl-ArrowUp", Motion::ParaStart),
                ("ctrl-ArrowDown", Motion::ParaEnd),
                ("Home", Motion::LineStart),
                ("End", Motion::LineEnd),
                ("ctrl-Home", Motion::DocStart),
                ("ctrl-End", Motion::DocEnd),
                ("PageUp", Motion::PageUp),
                ("PageDown", Motion::PageDown),
            ]
        };
        for (key, mo) in moves {
            k.bind(key, m(mo, false));
            // shift extends (Emacs keys too)
            k.bind(&format!("shift-{key}"), m(mo, true));
        }
        let deletes: Vec<(&str, Motion)> = if mac {
            vec![
                ("Backspace", Motion::Left),
                ("shift-Backspace", Motion::Left),
                ("Delete", Motion::Right),
                ("alt-Backspace", Motion::WordLeft),
                ("alt-Delete", Motion::WordRight),
                ("cmd-Backspace", Motion::LineStart),
                ("cmd-Delete", Motion::LineEnd),
                ("ctrl-h", Motion::Left),
                ("ctrl-d", Motion::Right),
                ("ctrl-k", Motion::ParaEnd),
            ]
        } else {
            vec![
                ("Backspace", Motion::Left),
                ("shift-Backspace", Motion::Left),
                ("Delete", Motion::Right),
                ("ctrl-Backspace", Motion::WordLeft),
                ("ctrl-Delete", Motion::WordRight),
            ]
        };
        for (key, mo) in deletes {
            k.bind(key, KeyAction::Delete(mo));
        }
        let cmd = |x: Command| KeyAction::Command(x);
        let commands: Vec<(String, KeyAction)> = vec![
            (format!("{c}-a"), KeyAction::SelectAll),
            (format!("{c}-c"), KeyAction::Clipboard(Clip::Copy)),
            (format!("{c}-x"), KeyAction::Clipboard(Clip::Cut)),
            (format!("{c}-v"), KeyAction::Clipboard(Clip::Paste)),
            (format!("{c}-alt-shift-v"), KeyAction::Clipboard(Clip::PastePlain)),
            (format!("{c}-z"), cmd(Command::Undo)),
            (format!("{c}-shift-z"), cmd(Command::Redo)),
            ("Enter".into(), cmd(Command::Enter)),
            ("shift-Enter".into(), cmd(Command::LineBreak)),
            (format!("{c}-Enter"), cmd(Command::PageBreak)),
            ("Tab".into(), cmd(Command::Tab)),
            ("shift-Tab".into(), cmd(Command::BackTab)),
            ("Escape".into(), cmd(Command::Escape)),
            (format!("{c}-b"), cmd(Command::Bold)),
            (format!("{c}-i"), cmd(Command::Italic)),
            (format!("{c}-u"), cmd(Command::Underline)),
            (format!("{c}-shift-x"), cmd(Command::Strike)),
            (format!("{c}-="), cmd(Command::Subscript)),
            (format!("{c}-shift-="), cmd(Command::Superscript)),
            (format!("{c}-shift-+"), cmd(Command::Superscript)),
            (format!("{c}-shift-."), cmd(Command::Bigger)),
            (format!("{c}-shift->"), cmd(Command::Bigger)),
            (format!("{c}-shift-,"), cmd(Command::Smaller)),
            (format!("{c}-shift-<"), cmd(Command::Smaller)),
            ("ctrl-Space".into(), cmd(Command::ClearFormatting)),
            (format!("{c}-l"), cmd(Command::AlignLeft)),
            (format!("{c}-e"), cmd(Command::AlignCenter)),
            (format!("{c}-r"), cmd(Command::AlignRight)),
            (format!("{c}-j"), cmd(Command::Justify)),
            (format!("{c}-]"), cmd(Command::Indent)),
            (format!("{c}-["), cmd(Command::Outdent)),
            (format!("{c}-alt-b"), cmd(Command::Bullets)),
            (format!("{c}-alt-n"), cmd(Command::Numbering)),
            (format!("{c}-alt-1"), cmd(Command::Style("Heading1".into()))),
            (format!("{c}-alt-2"), cmd(Command::Style("Heading2".into()))),
            (format!("{c}-alt-3"), cmd(Command::Style("Heading3".into()))),
            (format!("{c}-shift-n"), cmd(Command::Style("Normal".into()))),
            (format!("{c}-f"), cmd(Command::Find)),
            (format!("{c}-g"), cmd(Command::FindNext)),
            (format!("{c}-shift-g"), cmd(Command::FindPrevious)),
        ];
        for (key, a) in commands {
            k.bind(&key, a);
        }
        if !mac {
            k.bind("ctrl-y", cmd(Command::Redo));
            k.bind("shift-Delete", KeyAction::Clipboard(Clip::Cut));
            k.bind("ctrl-Insert", KeyAction::Clipboard(Clip::Copy));
            k.bind("shift-Insert", KeyAction::Clipboard(Clip::Paste));
            k.bind("F3", cmd(Command::FindNext));
            k.bind("shift-F3", cmd(Command::FindPrevious));
        }
        k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mods(s: &str) -> Mods {
        Combo::parse(&format!("{s}-x")).unwrap().mods
    }

    #[test]
    fn each_platform_has_its_keys() {
        let mac = Keymap::defaults(Platform::Mac);
        let pc = Keymap::defaults(Platform::Other);
        assert_eq!(mac.action("ArrowLeft", mods("alt")), Some(KeyAction::Move(Motion::WordLeft, false)));
        assert_eq!(pc.action("ArrowLeft", mods("ctrl")), Some(KeyAction::Move(Motion::WordLeft, false)));
        assert_eq!(mac.action("ArrowRight", mods("cmd-shift")), Some(KeyAction::Move(Motion::LineEnd, true)));
        assert_eq!(mac.action("z", mods("cmd-shift")), Some(KeyAction::Command(Command::Redo)));
        assert_eq!(pc.action("y", mods("ctrl")), Some(KeyAction::Command(Command::Redo)));
        assert_eq!(mac.action("k", mods("ctrl")), Some(KeyAction::Delete(Motion::ParaEnd)));
        assert_eq!(mac.action("B", mods("cmd")), Some(KeyAction::Command(Command::Bold)));
        assert_eq!(mac.action("é", Mods::default()), Some(KeyAction::Text("é".into())));
        assert_eq!(mac.action("q", mods("cmd")), None);
        assert_eq!(mac.label(&KeyAction::Command(Command::Redo)).as_deref(), Some("⇧⌘Z"));
        assert_eq!(pc.label(&KeyAction::Command(Command::Bold)).as_deref(), Some("Ctrl+B"));
    }

    #[test]
    fn bindings_change() {
        let mut k = Keymap::defaults(Platform::Mac);
        assert!(k.bind("cmd-shift-l", KeyAction::Command(Command::Bullets)));
        assert_eq!(k.action("l", mods("cmd-shift")), Some(KeyAction::Command(Command::Bullets)));
        k.unbind("cmd-b");
        assert_eq!(k.action("b", mods("cmd")), None);
        assert!(!k.bind("hyper-x", KeyAction::SelectAll));
    }
}
