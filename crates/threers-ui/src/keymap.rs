//! Keys to actions: what a keystroke *means*, decided by where focus is.
//!
//! An action is a value of a type of your own — `struct Save;`, or one with
//! fields — that names an operation. A [`Keymap`] binds keystrokes to
//! actions, optionally within a *key context*: a name given to a node (with
//! [`Input::set_key_context`](super::input::Input::set_key_context)) that is
//! in force while focus is inside it. The same key can then mean one thing
//! in an editor and another in a list, and a binding in a deeper context
//! wins over one in a shallower one.
//!
//! Handlers are on nodes ([`Input::on_action`](super::input::Input::on_action)),
//! found from the focused node outward, so the nearest handler runs.
//!
//! ```
//! use threers_ui::keymap::{Keymap, Keystroke};
//!
//! #[derive(Debug, Clone, PartialEq)]
//! struct Save;
//! #[derive(Debug, Clone, PartialEq)]
//! struct Delete;
//!
//! let mut keymap = Keymap::new();
//! keymap.bind("ctrl-s", Save, None).unwrap();
//! keymap.bind("ctrl-k ctrl-d", Delete, Some("Editor")).unwrap();
//!
//! let ctrl_s = Keystroke::parse("ctrl-s").unwrap();
//! let action = keymap.match_stroke(&ctrl_s, &[]).action().unwrap();
//! assert!(action.as_any().downcast_ref::<Save>().is_some());
//!
//! // Two strokes, and only inside an editor.
//! let k = Keystroke::parse("ctrl-k").unwrap();
//! let d = Keystroke::parse("ctrl-d").unwrap();
//! assert!(keymap.match_stroke(&k, &["Editor"]).is_pending());
//! assert!(keymap.match_stroke(&d, &["Editor"]).action().is_some());
//! assert!(keymap.match_stroke(&k, &[]).action().is_none());
//! ```

use std::any::Any;
use std::fmt;

use super::text_field::KeyModifiers;

/// A value naming an operation. Implemented for every `Clone + Debug` type,
/// so an action is just a type of your own.
pub trait Action: Any + fmt::Debug {
    fn boxed_clone(&self) -> Box<dyn Action>;
    fn as_any(&self) -> &dyn Any;
    /// The type's name, for logs and key-binding listings.
    fn name(&self) -> &'static str;
}

impl<T: Any + Clone + fmt::Debug> Action for T {
    fn boxed_clone(&self) -> Box<dyn Action> {
        Box::new(self.clone())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &'static str {
        std::any::type_name::<T>()
    }
}


/// One key with the modifiers held, as bindings spell it: `ctrl-shift-z`,
/// `cmd-s`, `alt-left`, `shift-tab`, `f5`, `space`.
///
/// Key names follow the DOM's `KeyboardEvent.key`, lower-cased, with
/// `Arrow` dropped and a few spelled out: `left`, `enter`, `escape`,
/// `backspace`, `space`. Modifiers are `ctrl`, `alt` (or `option`), `shift`
/// and `cmd` (or `meta`, `super`, `win`); `secondary` is `cmd` on macOS and
/// `ctrl` elsewhere, for the shortcuts that differ only by that.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Keystroke {
    pub key: String,
    pub modifiers: KeyModifiers,
}

/// A DOM `KeyboardEvent.key` (or a binding's spelling) as a binding names it.
pub fn key_name(key: &str) -> String {
    let lower = key.to_lowercase();
    let lower = lower.strip_prefix("arrow").unwrap_or(&lower);
    match lower {
        " " | "spacebar" => "space".into(),
        "esc" => "escape".into(),
        "return" => "enter".into(),
        "del" => "delete".into(),
        other => other.into(),
    }
}

impl Keystroke {
    /// The keystroke as a menu or tooltip shows it: `⌘⇧S` on macOS
    /// (`mac`), `Ctrl+Shift+S` elsewhere.
    pub fn label(&self, mac: bool) -> String {
        let m = &self.modifiers;
        let key = match self.key.as_str() {
            "enter" => if mac { "↩" } else { "Enter" }.to_owned(),
            "escape" => if mac { "⎋" } else { "Esc" }.to_owned(),
            "backspace" => if mac { "⌫" } else { "Backspace" }.to_owned(),
            "delete" => if mac { "⌦" } else { "Delete" }.to_owned(),
            "tab" => if mac { "⇥" } else { "Tab" }.to_owned(),
            "space" => "Space".to_owned(),
            "up" => "↑".to_owned(),
            "down" => "↓".to_owned(),
            "left" => "←".to_owned(),
            "right" => "→".to_owned(),
            "pageup" => "PgUp".to_owned(),
            "pagedown" => "PgDn".to_owned(),
            "home" => "Home".to_owned(),
            "end" => "End".to_owned(),
            "minus" => "-".to_owned(),
            k if k.chars().count() == 1 => k.to_uppercase(),
            k => {
                let mut c = k.chars();
                c.next().map_or(String::new(), |f| f.to_uppercase().chain(c).collect())
            }
        };
        if mac {
            // Apple's order: control, option, shift, command.
            let mut out = String::new();
            for (on, sign) in [(m.control, "⌃"), (m.alt, "⌥"), (m.shift, "⇧"), (m.meta, "⌘")] {
                if on {
                    out.push_str(sign);
                }
            }
            out + &key
        } else {
            let mut parts: Vec<&str> = vec![];
            for (on, name) in [(m.control, "Ctrl"), (m.alt, "Alt"), (m.shift, "Shift"), (m.meta, "Win")] {
                if on {
                    parts.push(name);
                }
            }
            parts.push(&key);
            parts.join("+")
        }
    }

    /// A key as a keyboard event gives it: the DOM's `key`, and the
    /// modifiers held. A shifted letter (`"A"`) is the letter with `shift`.
    pub fn from_event(key: &str, modifiers: KeyModifiers) -> Self {
        let mut modifiers = modifiers;
        let mut chars = key.chars();
        if let (Some(c), None) = (chars.next(), chars.next())
            && c.is_uppercase()
        {
            modifiers.shift = true;
        }
        Keystroke {
            key: key_name(key),
            modifiers,
        }
    }

    /// Parses one keystroke: modifiers and a key joined by `-` (`ctrl-s`;
    /// `-` itself is `minus`, or a trailing `-` as in `ctrl--`).
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();
        if s.is_empty() {
            return Err("an empty keystroke".into());
        }
        let (mods, key) = match s.strip_suffix("--") {
            Some(head) => (head, "-"),
            None if s == "-" => ("", "-"),
            None => match s.rsplit_once('-') {
                Some((head, key)) => (head, key),
                None => ("", s),
            },
        };
        let mut modifiers = KeyModifiers::default();
        for m in mods.split('-').filter(|m| !m.is_empty()) {
            match m.to_lowercase().as_str() {
                "ctrl" | "control" => modifiers.control = true,
                "alt" | "option" | "opt" => modifiers.alt = true,
                "shift" => modifiers.shift = true,
                "cmd" | "meta" | "super" | "win" | "command" => modifiers.meta = true,
                "secondary" => {
                    if cfg!(target_os = "macos") {
                        modifiers.meta = true
                    } else {
                        modifiers.control = true
                    }
                }
                other => return Err(format!("unknown modifier `{other}` in `{s}`")),
            }
        }
        let key = match key {
            "minus" => "-".to_string(),
            k => key_name(k),
        };
        if key.is_empty() {
            return Err(format!("no key in `{s}`"));
        }
        Ok(Keystroke { key, modifiers })
    }
}

impl fmt::Display for Keystroke {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let m = self.modifiers;
        for (on, name) in [
            (m.control, "ctrl-"),
            (m.alt, "alt-"),
            (m.shift, "shift-"),
            (m.meta, "cmd-"),
        ] {
            if on {
                f.write_str(name)?;
            }
        }
        f.write_str(&self.key)
    }
}

/// A sequence of keystrokes bound to an action.
#[derive(Debug)]
pub struct Binding {
    pub keys: Vec<Keystroke>,
    pub action: Box<dyn Action>,
    /// The context expression it applies in, as written; `None` everywhere.
    pub context: Option<String>,
    predicate: Option<Predicate>,
}

// Not `Clone for Box<dyn Action>`: that would make a boxed action an
// `Action` itself, and `as_any` on the box would answer for the box.
impl Clone for Binding {
    fn clone(&self) -> Self {
        Binding {
            keys: self.keys.clone(),
            action: self.action.boxed_clone(),
            context: self.context.clone(),
            predicate: self.predicate.clone(),
        }
    }
}

/// What a keystroke matched.
#[derive(Debug)]
pub enum Match {
    /// Nothing: the key goes on as a key event.
    None,
    /// The start of a longer binding; the key is taken, and the next decides.
    Pending,
    Action(Box<dyn Action>),
}

impl Match {
    pub fn action(self) -> Option<Box<dyn Action>> {
        match self {
            Match::Action(a) => Some(a),
            _ => None,
        }
    }
    pub fn is_pending(&self) -> bool {
        matches!(self, Match::Pending)
    }
}

/// Bindings, and the keys typed so far of a binding not yet finished.
#[derive(Debug, Clone, Default)]
pub struct Keymap {
    bindings: Vec<Binding>,
    pending: Vec<Keystroke>,
}

impl Keymap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Binds `keys` — keystrokes separated by spaces, `ctrl-k ctrl-s` for a
    /// chord of two — to `action` in `context` (`None` for everywhere). A
    /// later binding of the same keys in the same context replaces an
    /// earlier one, as a user's keymap overrides the defaults.
    pub fn bind(
        &mut self,
        keys: &str,
        action: impl Action,
        context: Option<&str>,
    ) -> Result<(), String> {
        let keys = keys
            .split_whitespace()
            .map(Keystroke::parse)
            .collect::<Result<Vec<_>, _>>()?;
        if keys.is_empty() {
            return Err("no keys to bind".into());
        }
        let predicate = context.map(Predicate::parse).transpose()?;
        self.bindings.push(Binding {
            keys,
            action: Box::new(action),
            context: context.map(str::to_owned),
            predicate,
        });
        Ok(())
    }

    /// The label of the first binding of the action named `action` (its
    /// type's name, with or without its path), for a menu item or a
    /// tooltip: `⌘S`, `Ctrl+S`, or a chord `⌘K ⌘S`.
    pub fn label_for(&self, action: &str, mac: bool) -> Option<String> {
        let named = |n: &str| n == action || n.rsplit("::").next() == Some(action);
        let b = self.bindings().iter().find(|b| named(b.action.name()))?;
        Some(b.keys.iter().map(|k| k.label(mac)).collect::<Vec<_>>().join(" "))
    }

    pub fn bindings(&self) -> &[Binding] {
        &self.bindings
    }

    /// The keys of the binding that would run an action of type `A` in
    /// `contexts` (innermost first) — for showing a shortcut in a menu.
    pub fn keys_for<A: Action>(&self, contexts: &[&str]) -> Option<&[Keystroke]> {
        self.bindings
            .iter()
            .rev()
            .filter(|b| b.action.as_any().is::<A>())
            .filter_map(|b| Some((Self::depth(b, contexts)?, b)))
            .min_by_key(|(d, _)| *d)
            .map(|(_, b)| b.keys.as_slice())
    }

    /// How deep in the focus path a binding's context is (0 innermost), or
    /// `None` when it does not apply there. A binding for everywhere is the
    /// shallowest of all.
    fn depth(b: &Binding, contexts: &[&str]) -> Option<usize> {
        match &b.predicate {
            None => Some(contexts.len()),
            Some(p) => {
                let stack: Vec<KeyContext> = contexts.iter().map(|c| KeyContext::parse(c)).collect();
                (0..stack.len()).find(|&i| p.eval(&stack, i))
            }
        }
    }

    /// Feeds one keystroke typed with focus inside `contexts` (the key
    /// contexts on the focus path, innermost first).
    pub fn match_stroke(&mut self, stroke: &Keystroke, contexts: &[&str]) -> Match {
        let mut typed = std::mem::take(&mut self.pending);
        typed.push(stroke.clone());
        match self.lookup(&typed, contexts) {
            Match::None if typed.len() > 1 => {
                // The chord broke off: the key is judged on its own.
                self.match_stroke(stroke, contexts)
            }
            Match::Pending => {
                self.pending = typed;
                Match::Pending
            }
            m => m,
        }
    }

    /// Whether a chord has been started and not finished.
    pub fn is_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Abandons a started chord.
    pub fn cancel_pending(&mut self) {
        self.pending.clear();
    }

    /// What the keys do with focus inside `contexts` (innermost first):
    /// each key sequence once, bound to what it would run there (the
    /// deepest context's binding, the later at equal depth), in the order
    /// they were bound. For a list of shortcuts, a cheat sheet, or a key
    /// hint panel.
    pub fn available(&self, contexts: &[&str]) -> Vec<&Binding> {
        let mut best: Vec<(usize, usize)> = vec![];
        for (i, b) in self.bindings.iter().enumerate() {
            let Some(depth) = Self::depth(b, contexts) else { continue };
            match best.iter_mut().find(|(j, _)| self.bindings[*j].keys == b.keys) {
                Some(slot) if depth <= slot.1 => *slot = (i, depth),
                Some(_) => {}
                None => best.push((i, depth)),
            }
        }
        best.sort_by_key(|&(i, _)| i);
        best.into_iter().map(|(i, _)| &self.bindings[i]).collect()
    }

    /// While a chord is started: the bindings it can still become in
    /// `contexts`, with the keys already typed (what to show as the next
    /// key's choices). Empty when none is started.
    pub fn continuations(&self, contexts: &[&str]) -> Vec<&Binding> {
        if self.pending.is_empty() {
            return vec![];
        }
        let n = self.pending.len();
        self.available(contexts).into_iter().filter(|b| b.keys.len() > n && b.keys[..n] == self.pending[..]).collect()
    }

    fn lookup(&self, typed: &[Keystroke], contexts: &[&str]) -> Match {
        let mut best: Option<(usize, &Binding)> = None;
        let mut longer = false;
        // Later bindings first, so that at equal depth the later one wins.
        for b in self.bindings.iter().rev() {
            let Some(depth) = Self::depth(b, contexts) else {
                continue;
            };
            if b.keys.len() > typed.len() && b.keys[..typed.len()] == *typed {
                longer = true;
            }
            if b.keys == typed && best.is_none_or(|(d, _)| depth < d) {
                best = Some((depth, b));
            }
        }
        // A complete binding wins over waiting for a longer one, so `ctrl-k`
        // bound alone still works when `ctrl-k ctrl-s` is bound too.
        match (best, longer) {
            (Some((_, b)), _) => Match::Action(b.action.boxed_clone()),
            (None, true) => Match::Pending,
            (None, false) => Match::None,
        }
    }
}

/// Actions by name, for keymaps written as data: what `"editor::Save"` in a
/// keymap file makes.
#[derive(Default)]
pub struct ActionRegistry {
    #[allow(clippy::type_complexity)]
    makers: std::collections::HashMap<String, Box<dyn Fn(Option<&ActionArgs>) -> Result<Box<dyn Action>, String>>>,
}

/// An action's arguments in a keymap file.
#[cfg(feature = "json")]
pub type ActionArgs = serde_json::Value;
#[cfg(not(feature = "json"))]
pub type ActionArgs = ();

impl ActionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// `name` makes a copy of `action`, which takes no arguments.
    pub fn register<A: Action + Clone>(&mut self, name: &str, action: A) {
        self.makers
            .insert(name.to_owned(), Box::new(move |_| Ok(Box::new(action.clone()) as Box<dyn Action>)));
    }

    /// `name` makes an action from its arguments with `make`.
    pub fn register_with<A: Action>(
        &mut self,
        name: &str,
        make: impl Fn(&ActionArgs) -> Result<A, String> + 'static,
    ) {
        self.makers.insert(
            name.to_owned(),
            Box::new(move |args| match args {
                Some(a) => make(a).map(|a| Box::new(a) as Box<dyn Action>),
                None => Err("this action needs arguments".into()),
            }),
        );
    }

    /// The action `name` names, made with `args`.
    pub fn make(&self, name: &str, args: Option<&ActionArgs>) -> Result<Box<dyn Action>, String> {
        let maker = self.makers.get(name).ok_or_else(|| format!("no action named `{name}`"))?;
        maker(args)
    }
}

impl Keymap {
    /// Binds an action already boxed (one made by name).
    pub fn bind_boxed(&mut self, keys: &str, action: Box<dyn Action>, context: Option<&str>) -> Result<(), String> {
        let keys = keys
            .split_whitespace()
            .map(Keystroke::parse)
            .collect::<Result<Vec<_>, _>>()?;
        let predicate = context.map(Predicate::parse).transpose()?;
        self.bindings.push(Binding {
            keys,
            action,
            context: context.map(str::to_owned),
            predicate,
        });
        Ok(())
    }

    /// Removes the bindings of exactly `keys` in exactly `context`.
    pub fn unbind(&mut self, keys: &str, context: Option<&str>) -> Result<usize, String> {
        let keys = keys
            .split_whitespace()
            .map(Keystroke::parse)
            .collect::<Result<Vec<_>, _>>()?;
        let before = self.bindings.len();
        self.bindings
            .retain(|b| !(b.keys == keys && b.context.as_deref() == context));
        Ok(before - self.bindings.len())
    }

    /// Adds the bindings of a JSON keymap file: a list of
    /// `{"context": "...", "bindings": {"ctrl-s": "app::Save", ...}}`, where
    /// an action is a name, `[name, arguments]`, or `null` to remove what
    /// was bound to those keys in that context before (a user's keymap
    /// switching a default off). Later files override earlier ones, as later
    /// bindings do. Returns how many bindings were added.
    #[cfg(feature = "json")]
    pub fn load_json(&mut self, json: &str, actions: &ActionRegistry) -> Result<usize, String> {
        let blocks: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
        let blocks = blocks.as_array().ok_or("a keymap is a list of blocks")?;
        let mut added = 0;
        for block in blocks {
            let context = block.get("context").and_then(|c| c.as_str());
            let Some(bindings) = block.get("bindings").and_then(|b| b.as_object()) else {
                continue;
            };
            for (keys, action) in bindings {
                let made = match action {
                    serde_json::Value::Null => {
                        self.unbind(keys, context)?;
                        continue;
                    }
                    serde_json::Value::String(name) => actions.make(name, None),
                    serde_json::Value::Array(a) => match a.as_slice() {
                        [serde_json::Value::String(name)] => actions.make(name, None),
                        [serde_json::Value::String(name), args] => actions.make(name, Some(args)),
                        _ => Err(format!("`{keys}`: an action is a name or [name, arguments]")),
                    },
                    _ => Err(format!("`{keys}`: an action is a name or [name, arguments]")),
                };
                let action = made.map_err(|e| format!("`{keys}`: {e}"))?;
                self.bind_boxed(keys, action, context).map_err(|e| format!("`{keys}`: {e}"))?;
                added += 1;
            }
        }
        Ok(added)
    }
}

/// A node's key context: names (`Editor`) and `key=value` pairs
/// (`mode=insert`), written `"Editor mode=insert"`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KeyContext {
    pub names: Vec<String>,
    pub pairs: Vec<(String, String)>,
}

impl KeyContext {
    pub fn parse(s: &str) -> Self {
        let mut c = KeyContext::default();
        for word in s.split_whitespace() {
            match word.split_once('=') {
                Some((k, v)) => c.pairs.push((k.to_owned(), v.to_owned())),
                None => c.names.push(word.to_owned()),
            }
        }
        c
    }

    fn has(&self, name: &str) -> bool {
        self.names.iter().any(|n| n == name) || self.pairs.iter().any(|(k, _)| k == name)
    }

    fn get(&self, key: &str) -> Option<&str> {
        self.pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }
}

/// A binding's context expression: `Editor`, `mode == insert`, `mode != normal`, `!Terminal`,
/// `Editor && !vim`, `Picker || Menu`, and `Workspace > Pane` for a `Pane`
/// context inside a `Workspace` one, with parentheses to group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Predicate {
    Name(String),
    Equal(String, String),
    NotEqual(String, String),
    Not(Box<Predicate>),
    And(Box<Predicate>, Box<Predicate>),
    Or(Box<Predicate>, Box<Predicate>),
    /// The second matching a context inside one the first matches.
    Descendant(Box<Predicate>, Box<Predicate>),
}

impl Predicate {
    pub fn parse(source: &str) -> Result<Self, String> {
        let tokens = tokenize(source)?;
        let mut at = 0;
        let p = parse_or(&tokens, &mut at)?;
        if at != tokens.len() {
            return Err(format!("unexpected `{}` in context `{source}`", tokens[at]));
        }
        Ok(p)
    }

    /// Whether it holds at `stack[i]`, the stack being the focus path's
    /// contexts, innermost first.
    pub fn eval(&self, stack: &[KeyContext], i: usize) -> bool {
        let Some(c) = stack.get(i) else { return false };
        match self {
            Predicate::Name(n) => c.has(n),
            Predicate::Equal(k, v) => c.get(k) == Some(v.as_str()),
            Predicate::NotEqual(k, v) => c.get(k) != Some(v.as_str()),
            Predicate::Not(p) => !p.eval(stack, i),
            Predicate::And(a, b) => a.eval(stack, i) && b.eval(stack, i),
            Predicate::Or(a, b) => a.eval(stack, i) || b.eval(stack, i),
            Predicate::Descendant(outer, inner) => {
                inner.eval(stack, i) && (i + 1..stack.len()).any(|j| outer.eval(stack, j))
            }
        }
    }
}

fn tokenize(s: &str) -> Result<Vec<String>, String> {
    let mut out = vec![];
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
        } else if let Some(two) = ["&&", "||", "==", "!="].iter().find(|t| s[s.char_indices().nth(i).unwrap().0..].starts_with(**t)) {
            out.push(two.to_string());
            i += 2;
        } else if "()!>".contains(c) {
            out.push(c.to_string());
            i += 1;
        } else if c.is_alphanumeric() || "_-:.".contains(c) {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || "_-:.".contains(chars[i])) {
                i += 1;
            }
            out.push(chars[start..i].iter().collect());
        } else {
            return Err(format!("`{c}` in context `{s}`"));
        }
    }
    Ok(out)
}

fn parse_or(t: &[String], at: &mut usize) -> Result<Predicate, String> {
    let mut left = parse_and(t, at)?;
    while t.get(*at).map(String::as_str) == Some("||") {
        *at += 1;
        left = Predicate::Or(Box::new(left), Box::new(parse_and(t, at)?));
    }
    Ok(left)
}

fn parse_and(t: &[String], at: &mut usize) -> Result<Predicate, String> {
    let mut left = parse_child(t, at)?;
    while t.get(*at).map(String::as_str) == Some("&&") {
        *at += 1;
        left = Predicate::And(Box::new(left), Box::new(parse_child(t, at)?));
    }
    Ok(left)
}

fn parse_child(t: &[String], at: &mut usize) -> Result<Predicate, String> {
    let mut left = parse_unary(t, at)?;
    while t.get(*at).map(String::as_str) == Some(">") {
        *at += 1;
        left = Predicate::Descendant(Box::new(left), Box::new(parse_unary(t, at)?));
    }
    Ok(left)
}

fn parse_unary(t: &[String], at: &mut usize) -> Result<Predicate, String> {
    match t.get(*at).map(String::as_str) {
        Some("!") => {
            *at += 1;
            Ok(Predicate::Not(Box::new(parse_unary(t, at)?)))
        }
        Some("(") => {
            *at += 1;
            let p = parse_or(t, at)?;
            if t.get(*at).map(String::as_str) != Some(")") {
                return Err("a `(` without its `)`".into());
            }
            *at += 1;
            Ok(p)
        }
        Some(word) if !["&&", "||", "==", "!=", ")", ">"].contains(&word) => {
            let name = word.to_owned();
            *at += 1;
            match t.get(*at).map(String::as_str) {
                Some(op @ ("==" | "!=")) => {
                    let op = op.to_owned();
                    *at += 1;
                    let value = t.get(*at).ok_or("a comparison without a value")?.clone();
                    *at += 1;
                    Ok(if op == "==" {
                        Predicate::Equal(name, value)
                    } else {
                        Predicate::NotEqual(name, value)
                    })
                }
                _ => Ok(Predicate::Name(name)),
            }
        }
        other => Err(format!("expected a name, found {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    struct A(u32);

    fn stroke(s: &str) -> Keystroke {
        Keystroke::parse(s).unwrap()
    }

    fn fired(m: Match) -> Option<u32> {
        m.action().map(|a| a.as_any().downcast_ref::<A>().unwrap().0)
    }

    #[test]
    fn parses_modifiers_and_keys() {
        let k = stroke("ctrl-shift-ArrowLeft");
        assert_eq!(k.key, "left");
        assert!(k.modifiers.control && k.modifiers.shift && !k.modifiers.alt);
        assert_eq!(stroke("cmd-=").key, "=");
        assert_eq!(stroke("ctrl--").key, "-");
        assert_eq!(stroke("ctrl-minus").key, "-");
        assert_eq!(stroke("space"), Keystroke::from_event(" ", KeyModifiers::default()));
        assert_eq!(stroke("ctrl-alt-delete").to_string(), "ctrl-alt-delete");
        assert!(Keystroke::parse("hyper-x").is_err());
        assert!(Keystroke::parse("").is_err());
    }

    #[test]
    fn a_shifted_letter_is_the_letter_with_shift() {
        let k = Keystroke::from_event("Z", KeyModifiers { control: true, ..Default::default() });
        assert_eq!(k, stroke("ctrl-shift-z"));
    }

    #[test]
    fn the_deepest_context_wins_and_later_bindings_override() {
        let mut km = Keymap::new();
        km.bind("enter", A(1), None).unwrap();
        km.bind("enter", A(2), Some("List")).unwrap();
        km.bind("enter", A(3), Some("Editor")).unwrap();
        km.bind("enter", A(4), Some("Editor")).unwrap();
        let enter = stroke("enter");
        assert_eq!(fired(km.match_stroke(&enter, &[])), Some(1));
        assert_eq!(fired(km.match_stroke(&enter, &["List", "Workspace"])), Some(2));
        assert_eq!(fired(km.match_stroke(&enter, &["Editor", "List"])), Some(4));
        assert_eq!(km.keys_for::<A>(&["List"]).unwrap(), &[enter]);
    }

    #[test]
    fn chords_wait_for_their_second_key_and_break_off_cleanly() {
        let mut km = Keymap::new();
        km.bind("ctrl-k ctrl-s", A(1), None).unwrap();
        km.bind("x", A(2), None).unwrap();
        assert!(km.match_stroke(&stroke("ctrl-k"), &[]).is_pending());
        assert_eq!(fired(km.match_stroke(&stroke("ctrl-s"), &[])), Some(1));
        assert!(!km.is_pending());
        // A key that does not continue the chord is judged on its own.
        assert!(km.match_stroke(&stroke("ctrl-k"), &[]).is_pending());
        assert_eq!(fired(km.match_stroke(&stroke("x"), &[])), Some(2));
        assert!(!km.is_pending());
    }

    #[test]
    fn context_expressions_match() {
        let stack: Vec<KeyContext> = ["Editor mode=insert", "Pane", "Workspace"]
            .iter()
            .map(|c| KeyContext::parse(c))
            .collect();
        let holds = |p: &str| {
            let p = Predicate::parse(p).unwrap();
            (0..stack.len()).any(|i| p.eval(&stack, i))
        };
        assert!(holds("Editor"));
        assert!(holds("mode == insert && Editor"));
        assert!(!holds("Editor && mode == normal"));
        assert!(holds("Editor && mode != normal"));
        assert!(holds("Workspace > Editor"));
        assert!(!holds("Editor > Workspace"));
        assert!(holds("!Terminal && (Picker || Pane)"));
        assert!(Predicate::parse("Editor &&").is_err());
        assert!(Predicate::parse("(Editor").is_err());

        let mut km = Keymap::new();
        km.bind("escape", A(1), Some("Editor && mode == insert")).unwrap();
        km.bind("escape", A(2), Some("Workspace")).unwrap();
        let esc = stroke("escape");
        assert_eq!(fired(km.match_stroke(&esc, &["Editor mode=insert", "Workspace"])), Some(1));
        assert_eq!(fired(km.match_stroke(&esc, &["Editor mode=normal", "Workspace"])), Some(2));
    }

    #[cfg(feature = "json")]
    #[test]
    fn keymaps_load_from_json() {
        #[derive(Debug, Clone, PartialEq)]
        struct Save;
        #[derive(Debug, Clone, PartialEq)]
        struct Go(u32);
        let mut reg = ActionRegistry::new();
        reg.register("app::Save", Save);
        reg.register_with("app::Go", |a| {
            a["line"].as_u64().map(|l| Go(l as u32)).ok_or_else(|| "`line` is a number".to_string())
        });
        let mut km = Keymap::new();
        let defaults = r#"[
            { "bindings": { "ctrl-s": "app::Save", "ctrl-g": ["app::Go", { "line": 1 }] } },
            { "context": "Editor && mode == insert", "bindings": { "ctrl-g": ["app::Go", { "line": 9 }] } }
        ]"#;
        assert_eq!(km.load_json(defaults, &reg), Ok(3));
        let g = stroke("ctrl-g");
        let go = |m: Match| m.action().map(|a| a.as_any().downcast_ref::<Go>().unwrap().0);
        assert_eq!(go(km.match_stroke(&g, &["Editor mode=insert"])), Some(9));
        assert_eq!(go(km.match_stroke(&g, &[])), Some(1));
        // A user file switches one off.
        assert_eq!(km.load_json(r#"[{ "bindings": { "ctrl-s": null } }]"#, &reg), Ok(0));
        assert!(km.match_stroke(&stroke("ctrl-s"), &[]).action().is_none());
        assert!(km.load_json(r#"[{ "bindings": { "ctrl-x": "app::Nothing" } }]"#, &reg).unwrap_err().contains("no action"));
        assert!(km.load_json(r#"[{ "bindings": { "ctrl-y": ["app::Go", { "line": "x" }] } }]"#, &reg).is_err());
    }
}

#[cfg(test)]
mod available_tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct Save;
    #[derive(Debug, Clone)]
    struct Find;
    #[derive(Debug, Clone)]
    struct FindInList;
    #[derive(Debug, Clone)]
    struct Close;

    fn names(b: &[&Binding]) -> Vec<String> {
        b.iter().map(|b| format!("{}={}", b.keys.iter().map(|k| k.label(false)).collect::<Vec<_>>().join(" "), b.action.name().rsplit("::").next().unwrap())).collect()
    }

    #[test]
    fn what_the_keys_do_here_innermost_first_and_what_finishes_a_chord() {
        let mut k = Keymap::new();
        k.bind("ctrl-s", Save, None).unwrap();
        k.bind("ctrl-f", Find, None).unwrap();
        k.bind("ctrl-f", FindInList, Some("List")).unwrap();
        k.bind("ctrl-k ctrl-w", Close, Some("Editor")).unwrap();
        assert_eq!(names(&k.available(&[])), ["Ctrl+S=Save", "Ctrl+F=Find"]);
        assert_eq!(names(&k.available(&["List"])), ["Ctrl+S=Save", "Ctrl+F=FindInList"]);
        assert_eq!(names(&k.available(&["Editor"])), ["Ctrl+S=Save", "Ctrl+F=Find", "Ctrl+K Ctrl+W=Close"]);
        assert!(k.continuations(&["Editor"]).is_empty());
        assert!(k.match_stroke(&Keystroke::parse("ctrl-k").unwrap(), &["Editor"]).is_pending());
        assert_eq!(names(&k.continuations(&["Editor"])), ["Ctrl+K Ctrl+W=Close"]);
    }
}
