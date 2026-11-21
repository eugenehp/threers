//! Keyboards written as text, drawn as text, and checked.
//!
//! The row syntax is on [`row`].

use super::config::{keys, Key, KeyAction, KeyLook, KeySize, KeyboardConfig, Page};
use super::layouts::{alternates_for, KeyboardLayout};

/// A key and where it sits on its page ([`KeyboardConfig::key_boxes`]).
#[derive(Debug, Clone, PartialEq)]
pub struct KeyBox {
    pub key: Key,
    /// Its row, from the top of the page.
    pub row: usize,
    /// Its left edge, in letter-key units from the left of the widest row.
    pub x: f32,
    /// In the same units, gaps between keys included.
    pub width: f32,
}

/// The keys `spec` writes.
///
/// Reads a row from a line: each word a key typing itself, `a|b`
/// a key typing `a` (`b` under Shift), `{name}` one of the special keys,
/// and `@` after any of them its width — units, or `*` for what the row
/// leaves:
///
/// | Written | Key |
/// |---|---|
/// | `q`, `.com` | types it |
/// | `1\|!` | types `1`, `!` under Shift, both shown |
/// | `{shift}` `{caps}` `{backspace}` `{return}` `{space}` `{tab}` | those keys |
/// | `{123}` `{abc}` `{#+=}` | the pages: symbols, letters, more symbols |
/// | `{globe}` `{dismiss}` `{gap}` | next language, hide, an empty unit |
/// | `{row:1}` | the language's letters of row 1 |
/// | `{key:ArrowLeft:←}` | sends a named key, labelled `←` |
/// | `{action:emoji:☺}` | calls [`on_action`](super::VirtualKeyboard::on_action) with `emoji` |
/// | `{shift@1.3}`, `{space@*}`, `{return@*2}` | 1.3 units; fill; fill at least 2 |
///
/// `\` before `{`, `|`, `@` or `\` types it; so does `@` with no width
/// after it (`2|@`).
///
/// ```
/// use threers_ui::virtual_keyboard::{KeyboardConfig, KeyboardLayout, Page};
///
/// let pad = KeyboardConfig::from_rows(&[
///     "7 8 9 {backspace}",
///     "4 5 6 {key:Tab:next}",
///     "1 2 3 {return@*}",
///     "{gap} 0 . {dismiss}",
/// ]);
/// assert!(pad.validate().is_empty());
/// let drawn = pad.sketch(Page::Letters, &KeyboardLayout::english(), 1);
/// assert!(drawn.lines().next().unwrap().contains("[ 7 ]"));
/// ```
///
/// # Panics
///
/// On a key it cannot read, saying which; [`try_row`] returns the error.
pub fn row(spec: &str) -> Vec<Key> {
    try_row(spec).unwrap_or_else(|e| panic!("keyboard row {spec:?}: {e}"))
}

/// [`row`], the error said rather than panicked.
pub fn try_row(spec: &str) -> Result<Vec<Key>, String> {
    let mut out = vec![];
    let mut chars = spec.chars().peekable();
    loop {
        while chars.peek().is_some_and(|c| c.is_whitespace()) {
            chars.next();
        }
        let Some(&first) = chars.peek() else {
            break;
        };
        // The key: a `{…}` or a run of characters, escapes kept apart.
        let (mut key, mut special) = (String::new(), false);
        if first == '{' {
            chars.next();
            special = true;
            loop {
                match chars.next() {
                    Some('}') => break,
                    Some(c) => key.push(c),
                    None => return Err(format!("{{{key} is not closed")),
                }
            }
        }
        let mut width = String::new();
        let (mut dual, mut in_width) = (None::<String>, false);
        while let Some(&c) = chars.peek() {
            if c.is_whitespace() {
                break;
            }
            chars.next();
            let c = if c == '\\' {
                match chars.next() {
                    Some(e) => {
                        push(&mut key, &mut dual, &mut width, in_width, e);
                        continue;
                    }
                    None => return Err("\\ at the end".into()),
                }
            } else {
                c
            };
            match c {
                // A width when a number or `*` follows; else the character.
                '@' if !in_width
                    && chars
                        .peek()
                        .is_some_and(|n| n.is_ascii_digit() || *n == '*' || *n == '.') =>
                {
                    in_width = true
                }
                '|' if !special && !in_width && dual.is_none() => dual = Some(String::new()),
                _ => push(&mut key, &mut dual, &mut width, in_width, c),
            }
        }
        // `{shift@1.3}`: the width inside the braces, when what follows
        // the last `@` reads as one (`{key:At:@}` keeps its `@`).
        if special
            && !in_width
            && let Some(at) = key.rfind('@')
            && parse_width(&key[at + 1..]).is_ok()
        {
            width = key[at + 1..].to_owned();
            key.truncate(at);
            in_width = true;
        }
        let mut k = if special {
            special_key(&key)?
        } else if let Some(shifted) = dual {
            if key.is_empty() || shifted.is_empty() {
                return Err(format!("{key}|{shifted}: both sides of | need a character"));
            }
            Key::dual(&key, &shifted)
        } else {
            Key::text(&key)
        };
        if in_width {
            k.size = parse_width(&width)?;
        }
        out.push(k);
    }
    Ok(out)
}

fn push(key: &mut String, dual: &mut Option<String>, width: &mut String, in_width: bool, c: char) {
    if in_width {
        width.push(c);
    } else if let Some(d) = dual {
        d.push(c);
    } else {
        key.push(c);
    }
}

fn parse_width(w: &str) -> Result<KeySize, String> {
    let number = |s: &str| {
        s.parse::<f32>()
            .ok()
            .filter(|v| v.is_finite() && *v >= 0.0)
            .ok_or_else(|| format!("@{w}: a width is units (@1.5) or * for the rest (@*, @*2)"))
    };
    match w.strip_prefix('*') {
        Some("") => Ok(KeySize::Fill(1.0)),
        Some(min) => number(min).map(KeySize::Fill),
        None => number(w).map(KeySize::Units),
    }
}

fn special_key(spec: &str) -> Result<Key, String> {
    let mut parts = spec.splitn(3, ':');
    let name = parts.next().unwrap_or("").trim();
    let arg = parts.next();
    let label = parts.next();
    Ok(match name.to_lowercase().as_str() {
        "shift" => Key::shift(),
        "caps" | "capslock" => Key::caps_lock(),
        "backspace" | "delete" => Key::backspace(),
        "return" | "enter" => Key::enter(),
        "space" => Key::space(),
        "tab" => Key::tab(),
        "123" | "symbols" => Key::symbols(),
        "abc" | "letters" => Key::letters_page(),
        "#+=" | "more" => Key::more_symbols(),
        "globe" => Key::globe(),
        "dismiss" | "hide" => Key::dismiss(),
        "gap" => Key::gap(1.0),
        "row" => {
            let n = arg
                .and_then(|a| a.trim().parse::<usize>().ok())
                .ok_or_else(|| format!("{{{spec}}}: say which row, {{row:0}} to {{row:2}}"))?;
            Key::letters(n)
        }
        "key" => {
            let n = arg.ok_or_else(|| format!("{{{spec}}}: name the key, {{key:ArrowLeft:←}}"))?;
            Key::named(n, label.unwrap_or(n))
        }
        "action" => {
            let id =
                arg.ok_or_else(|| format!("{{{spec}}}: name the action, {{action:emoji:☺}}"))?;
            Key::action(id, label.unwrap_or(id))
        }
        other => {
            return Err(format!(
                "{{{other}}} is not a key: shift, caps, backspace, return, space, tab, 123, abc, #+=, globe, dismiss, gap, row:N, key:Name:label, action:id:label"
            ));
        }
    })
}

/// The rows showing on `page`: those over every page, the language's
/// accents, the page's, those under every page; its letters put in,
/// 🌐 left out for one language, and alternates given where asked.
pub(crate) fn page_rows(
    cfg: &KeyboardConfig,
    page: Page,
    layout: &KeyboardLayout,
    languages: usize,
) -> Vec<Vec<Key>> {
    let expand = |row: &Vec<Key>| -> Vec<Key> {
        let mut out = vec![];
        for k in row {
            match &k.action {
                KeyAction::Letters(r) => {
                    if page == Page::Letters {
                        for c in layout.rows.get(*r).map_or("", |s| s.as_str()).chars() {
                            out.push(Key::text(&c.to_string()));
                        }
                    }
                }
                KeyAction::NextLayout if languages < 2 => {}
                _ => out.push(k.clone()),
            }
        }
        if cfg.alternates {
            for k in &mut out {
                if let KeyAction::Text(t) = &k.action
                    && k.alternates.is_empty()
                    && k.look == KeyLook::Character
                {
                    k.alternates = alternates_for(t);
                }
            }
        }
        out
    };
    let page_rows = match page {
        Page::Letters => &cfg.letters,
        Page::Symbols => &cfg.symbols,
        Page::More => &cfg.more,
    };
    let mut rows: Vec<Vec<Key>> = cfg.top_rows.iter().map(expand).collect();
    if page == Page::Letters && !layout.accents.is_empty() {
        rows.push(expand(&keys(&layout.accents)));
    }
    // A page's row typing just what a row over or under every page types
    // (its digits, under a number row) is left out, not shown twice.
    let types = |row: &[Key]| -> Vec<String> {
        row.iter()
            .filter_map(|k| match &k.action {
                KeyAction::Text(t) => Some(t.clone()),
                _ => None,
            })
            .collect()
    };
    let everywhere: Vec<Vec<String>> = cfg
        .top_rows
        .iter()
        .chain(&cfg.bottom_rows)
        .map(|r| types(r))
        .filter(|t| !t.is_empty())
        .collect();
    rows.extend(
        page_rows
            .iter()
            .map(expand)
            .filter(|r| !everywhere.contains(&types(r))),
    );
    rows.extend(cfg.bottom_rows.iter().map(expand));
    rows.retain(|r| !r.is_empty());
    rows
}

/// The widest row's units.
pub(crate) fn total_units(rows: &[Vec<Key>]) -> f32 {
    rows.iter()
        .map(|r| r.iter().map(|k| k.units_and_fill().0).sum::<f32>())
        .fold(1.0f32, f32::max)
}

impl KeyboardConfig {
    /// An iPhone's keyboard with these letter rows (see [`row`]); its
    /// symbol pages as they are.
    pub fn from_rows(rows: &[&str]) -> Self {
        Self::iphone().page_rows(Page::Letters, rows)
    }

    /// A keyboard written as a text file: settings, then pages of rows in
    /// the [row syntax](row). What a file leaves out keeps the `base`
    /// preset's. Errors say the line.
    ///
    /// ```text
    /// # A kiosk's keyboard.
    /// base: iphone            # iphone, ipad, ipad-pro, ipad-floating, number-pad,
    ///                         # decimal-pad, phone-pad, email, url
    /// theme: system           # apple, system
    /// placement: centered 640 # docked, one-handed left|right, centered W,
    ///                         # anchored left|right W, floating W; then split G
    /// return: go              # return, go, search, send, next, done, join, continue, or words
    /// key-height: 48
    /// sound: off              # sound, haptics, preview, alternates: on|off
    ///
    /// [letters]
    /// {row:0}
    /// {row:1}
    /// {shift@1.3} {row:2} {backspace@1.3}
    /// {123@1.25} {space} {return@2.5}
    ///
    /// [bottom]                # also [symbols], [more], [top]
    /// {key:ArrowLeft:←} {key:ArrowRight:→} .com@2
    /// ```
    ///
    /// ```
    /// use threers_ui::virtual_keyboard::{KeyboardConfig, ReturnKey};
    /// let c = KeyboardConfig::parse("base: ipad\nreturn: search\n[top]\n1 2 3").unwrap();
    /// assert_eq!(c.return_key, ReturnKey::Search);
    /// assert_eq!(c.top_rows.len(), 1);
    /// let e = KeyboardConfig::parse("base: ipad\nplacement: sideways").unwrap_err();
    /// assert!(e.starts_with("line 2:"), "{e}");
    /// ```
    pub fn parse(text: &str) -> Result<Self, String> {
        use super::config::{Align, Placement, ReturnKey};
        let mut c = Self::iphone();
        let mut section: Option<&str> = None;
        let mut rows: Vec<(String, Vec<Vec<Key>>)> = vec![];
        for (n, raw) in text.lines().enumerate() {
            let at = |e: String| format!("line {}: {e}", n + 1);
            // A comment from `#` after a space (or at the start): `{#+=}` is a key.
            let line = match raw.find(" #") {
                Some(i) => &raw[..i],
                None if raw.trim_start().starts_with('#') => "",
                None => raw,
            }
            .trim();
            if line.is_empty() {
                continue;
            }
            if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
                let name = name.trim();
                if !["letters", "symbols", "more", "top", "bottom"].contains(&name) {
                    return Err(at(format!(
                        "[{name}] is not a section: [letters], [symbols], [more], [top], [bottom]"
                    )));
                }
                section = Some(match name {
                    "letters" => "letters",
                    "symbols" => "symbols",
                    "more" => "more",
                    "top" => "top",
                    _ => "bottom",
                });
                rows.push((name.to_owned(), vec![]));
                continue;
            }
            if section.is_some() {
                let keys = try_row(line).map_err(at)?;
                rows.last_mut().unwrap().1.push(keys);
                continue;
            }
            let Some((key, value)) = line.split_once(':') else {
                return Err(at(format!(
                    "“{line}”: a setting is `name: value`, or start a [section]"
                )));
            };
            let (key, value) = (key.trim(), value.trim());
            let words: Vec<&str> = value.split_whitespace().collect();
            let number = |w: Option<&&str>| -> Result<f32, String> {
                w.and_then(|w| w.parse::<f32>().ok())
                    .filter(|v| v.is_finite() && *v >= 0.0)
                    .ok_or_else(|| {
                        at(format!(
                            "{key}: a number of pixels is wanted after “{value}”"
                        ))
                    })
            };
            let on = || match value {
                "on" | "yes" | "true" => Ok(true),
                "off" | "no" | "false" => Ok(false),
                _ => Err(at(format!("{key}: on or off, not “{value}”"))),
            };
            match key {
                "base" => {
                    // A base resets everything: it comes first.
                    let tweaks_kept = c.clone();
                    c = match value {
                        "iphone" | "phone" => Self::iphone(),
                        "ipad" | "tablet" => Self::ipad(),
                        "ipad-pro" => Self::ipad_pro(),
                        "ipad-floating" => Self::ipad_floating(),
                        "number-pad" => Self::number_pad(),
                        "decimal-pad" => Self::decimal_pad(),
                        "phone-pad" => Self::phone_pad(),
                        "email" => Self::email(),
                        "url" => Self::url(),
                        _ => {
                            return Err(at(format!(
                                "base: “{value}” is not one: iphone, ipad, ipad-pro, ipad-floating, number-pad, decimal-pad, phone-pad, email, url"
                            )));
                        }
                    };
                    if tweaks_kept != Self::iphone() {
                        return Err(at("base: put it first, before other settings".into()));
                    }
                }
                "theme" => {
                    c.theme = match value {
                        "apple" => super::config::KeyboardTheme::apple(),
                        "system" => super::config::KeyboardTheme::system(),
                        _ => return Err(at(format!("theme: apple or system, not “{value}”"))),
                    }
                }
                "placement" => {
                    let side = |w: Option<&&str>| match w.copied() {
                        Some("left") => Ok(Align::Left),
                        Some("right") => Ok(Align::Right),
                        Some("center" | "centre") => Ok(Align::Center),
                        _ => Err(at("placement: left, right or center".into())),
                    };
                    let (mut p, rest) = match words.first().copied() {
                        Some("docked") => (Placement::docked(), 1),
                        Some("one-handed") => (Placement::one_handed(side(words.get(1))?), 2),
                        Some("centered" | "centred") => {
                            (Placement::centered(number(words.get(1))?), 2)
                        }
                        Some("anchored") => (
                            Placement::anchored(side(words.get(1))?, number(words.get(2))?),
                            3,
                        ),
                        Some("floating") => (Placement::floating(number(words.get(1))?), 2),
                        _ => {
                            return Err(at(format!(
                                "placement: “{value}” — docked, one-handed left|right, centered W, anchored left|right W, floating W, then split G"
                            )));
                        }
                    };
                    match words.get(rest).copied() {
                        Some("split") => p = p.split(number(words.get(rest + 1))?),
                        None => {}
                        Some(w) => {
                            return Err(at(format!("placement: “{w}”? only split G may follow")))
                        }
                    }
                    c.placement = p;
                }
                "return" => {
                    c.return_key = match value {
                        "return" => ReturnKey::Return,
                        "go" => ReturnKey::Go,
                        "search" => ReturnKey::Search,
                        "send" => ReturnKey::Send,
                        "next" => ReturnKey::Next,
                        "done" => ReturnKey::Done,
                        "join" => ReturnKey::Join,
                        "continue" => ReturnKey::Continue,
                        other => ReturnKey::Label(other.to_owned()),
                    }
                }
                "key-height" => c.key_height = number(words.first())?,
                "gap" => c.gap = number(words.first())?,
                "row-gap" => c.row_gap = number(words.first())?,
                "sound" => c.feedback.sound = on()?,
                "haptics" => c.feedback.haptics = on()?,
                "preview" => c.feedback.preview = on()?,
                "alternates" => c.alternates = on()?,
                "bar" => c.bar = on()?,
                _ => {
                    return Err(at(format!(
                        "“{key}” is not a setting: base, theme, placement, return, key-height, gap, row-gap, sound, haptics, preview, alternates, bar"
                    )));
                }
            }
        }
        for (name, r) in rows {
            match name.as_str() {
                "letters" => c.letters = r,
                "symbols" => c.symbols = r,
                "more" => c.more = r,
                "top" => c.top_rows = r,
                _ => c.bottom_rows = r,
            }
        }
        Ok(c)
    }

    /// `page` written as rows of text (see [`row`]).
    ///
    /// # Panics
    ///
    /// On a key it cannot read, saying which.
    pub fn page_rows(mut self, page: Page, rows: &[&str]) -> Self {
        let rows: Vec<Vec<Key>> = rows.iter().map(|r| row(r)).collect();
        match page {
            Page::Letters => self.letters = rows,
            Page::Symbols => self.symbols = rows,
            Page::More => self.more = rows,
        }
        self
    }

    /// `page` drawn as text, a key a `[label]` five characters a unit
    /// wide, short rows centred: to see a keyboard in a test, a doc
    /// or a log.
    ///
    /// ```
    /// use threers_ui::virtual_keyboard::{KeyboardConfig, KeyboardLayout, Page};
    /// let s = KeyboardConfig::iphone().sketch(Page::Letters, &KeyboardLayout::english(), 1);
    /// let lines: Vec<&str> = s.lines().collect();
    /// assert!(lines[0].starts_with("[ q ][ w ]"));
    /// assert!(lines[1].starts_with("   [ a ]"), "half a key in");
    /// assert!(lines[3].contains("space") && lines[3].contains("return"));
    /// ```
    pub fn sketch(&self, page: Page, layout: &KeyboardLayout, languages: usize) -> String {
        let rows = page_rows(self, page, layout, languages);
        let total = total_units(&rows);
        let mut out = String::new();
        for row in &rows {
            let sum: f32 = row.iter().map(|k| k.units_and_fill().0).sum();
            let fills = row.iter().filter(|k| k.units_and_fill().1).count();
            let spare = (total - sum).max(0.0);
            let lead = if fills == 0 { spare / 2.0 } else { 0.0 };
            let mut line = " ".repeat((lead * 5.0).round() as usize);
            for k in row {
                let (u, fill) = k.units_and_fill();
                let u = if fill { u + spare / fills as f32 } else { u };
                if k.action == KeyAction::None {
                    line.push_str(&" ".repeat((u * 5.0).round() as usize));
                    continue;
                }
                let cells = ((u * 5.0).round() as usize).max(3);
                let label = match &k.action {
                    KeyAction::Named(n) if n == "Enter" => self.return_key.label(),
                    _ => k.label.clone(),
                };
                let inner = cells.saturating_sub(2).max(label.chars().count());
                let pad = inner - label.chars().count();
                line.push('[');
                line.push_str(&" ".repeat(pad / 2));
                line.push_str(&label);
                line.push_str(&" ".repeat(pad - pad / 2));
                line.push(']');
            }
            out.push_str(line.trim_end());
            out.push('\n');
        }
        out
    }

    /// Where each key of `page` sits, as [`sketch`](Self::sketch) draws
    /// them: in letter-key units from the left of the widest row, rows
    /// counted from the top, gaps left out. Unparted: a
    /// [`split`](super::Placement::split) keyboard moves its halves apart. What a
    /// gesture decoder or a hit map wants without laying the keyboard out.
    ///
    /// ```
    /// use threers_ui::virtual_keyboard::{KeyAction, KeyboardConfig, KeyboardLayout, Page};
    ///
    /// let boxes = KeyboardConfig::iphone().key_boxes(Page::Letters, &KeyboardLayout::english(), 1);
    /// let at = |t: &str| boxes.iter().find(|b| b.key.action == KeyAction::Text(t.into())).unwrap();
    /// assert_eq!((at("q").row, at("q").x, at("q").width), (0, 0.0, 1.0));
    /// assert_eq!((at("a").row, at("a").x), (1, 0.5), "half a key in");
    /// assert_eq!(at("z").x, 1.5, "after Shift and its gap");
    /// ```
    pub fn key_boxes(&self, page: Page, layout: &KeyboardLayout, languages: usize) -> Vec<KeyBox> {
        let rows = page_rows(self, page, layout, languages);
        let total = total_units(&rows);
        let mut out = vec![];
        for (r, row) in rows.iter().enumerate() {
            let sum: f32 = row.iter().map(|k| k.units_and_fill().0).sum();
            let fills = row.iter().filter(|k| k.units_and_fill().1).count();
            let spare = (total - sum).max(0.0);
            let mut x = if fills == 0 { spare / 2.0 } else { 0.0 };
            for k in row {
                let (u, fill) = k.units_and_fill();
                let u = if fill { u + spare / fills as f32 } else { u };
                if k.action != KeyAction::None {
                    out.push(KeyBox {
                        key: k.clone(),
                        row: r,
                        x,
                        width: u,
                    });
                }
                x += u;
            }
        }
        out
    }

    /// What is wrong with it, each said with where: a page no key reaches
    /// or none leaves, letters of a row no layout has, an empty text key, a
    /// key or a gap of no size. Empty when nothing is. Shown on the console
    /// in debug builds when the keyboard first shows it.
    pub fn validate(&self) -> Vec<String> {
        let mut issues = vec![];
        if self.key_height.is_nan() || self.key_height <= 0.0 {
            issues.push(format!(
                "key_height is {}: keys need some height",
                self.key_height
            ));
        }
        if self.gap < 0.0 || self.row_gap < 0.0 {
            issues.push("gaps cannot be negative".into());
        }
        if self.letters.is_empty() {
            issues.push("the letters page has no rows: nothing would show".into());
        }
        let pages = [
            (Page::Letters, "letters", &self.letters),
            (Page::Symbols, "symbols", &self.symbols),
            (Page::More, "more", &self.more),
        ];
        let shared: Vec<&Key> = self
            .top_rows
            .iter()
            .chain(&self.bottom_rows)
            .flatten()
            .collect();
        let leads_to = |rows: &Vec<Vec<Key>>, to: Page| {
            rows.iter()
                .flatten()
                .chain(shared.iter().copied())
                .any(|k| k.action == KeyAction::Page(to))
        };
        for (page, name, rows) in pages {
            for (r, row) in rows.iter().enumerate() {
                for (i, k) in row.iter().enumerate() {
                    let at = format!("{name} row {r} key {i}");
                    match &k.action {
                        KeyAction::Letters(n) if *n > 2 => issues.push(format!(
                            "{at}: Key::letters({n}) — a layout has rows 0 to 2"
                        )),
                        KeyAction::Letters(_) if page != Page::Letters => issues.push(format!(
                            "{at}: Key::letters on the {name} page shows nothing; it is for the letters page"
                        )),
                        KeyAction::Text(t) if t.is_empty() => {
                            issues.push(format!("{at}: a text key that types nothing"))
                        }
                        KeyAction::Page(to) => {
                            let target = match to {
                                Page::Letters => &self.letters,
                                Page::Symbols => &self.symbols,
                                Page::More => &self.more,
                            };
                            if target.is_empty() {
                                issues.push(format!("{at}: shows the {to:?} page, which has no rows"));
                            }
                        }
                        _ => {}
                    }
                    let (u, fill) = k.units_and_fill();
                    if u <= 0.0 && !fill && k.action != KeyAction::None {
                        issues.push(format!("{at}: “{}” is 0 units wide", k.label));
                    }
                }
            }
            if page != Page::Letters
                && !rows.is_empty()
                && leads_to(&self.letters, page)
                && !leads_to(rows, Page::Letters)
            {
                issues.push(format!(
                    "no key on the {name} page goes back to the letters"
                ));
            }
        }
        issues
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_read_keys_widths_duals_and_escapes() {
        assert_eq!(row("2|@")[0], Key::dual("2", "@"));
        assert_eq!(row("@")[0], Key::text("@"));
        let r = row("q 1|! {shift@1.3} {space@*} {return@*2} .com \\{ a\\|b {key:Escape:esc} {row:1} {action:emoji:☺}");
        assert_eq!(r[0], Key::text("q"));
        assert_eq!(r[1], Key::dual("1", "!"));
        assert_eq!(r[2].action, KeyAction::Shift);
        assert_eq!(r[2].size, KeySize::Units(1.3));
        assert_eq!(r[3].size, KeySize::Fill(1.0));
        assert_eq!(r[4].size, KeySize::Fill(2.0));
        assert_eq!(r[5].action, KeyAction::Text(".com".into()));
        assert_eq!(r[6].action, KeyAction::Text("{".into()));
        assert_eq!(r[7].action, KeyAction::Text("a|b".into()));
        assert_eq!(r[8].action, KeyAction::Named("Escape".into()));
        assert_eq!(r[8].label, "esc");
        assert_eq!(r[9].action, KeyAction::Letters(1));
        assert_eq!(r[10].action, KeyAction::Action("emoji".into()));
    }

    #[test]
    fn a_bad_row_says_what_is_wrong() {
        for (spec, says) in [
            ("{shfit}", "{shfit} is not a key"),
            ("{shift", "is not closed"),
            ("q@*wide", "a width is units"),
            ("{row:x}", "say which row"),
            ("|a", "both sides"),
        ] {
            let e = try_row(spec).unwrap_err();
            assert!(e.contains(says), "{spec}: {e}");
        }
    }

    #[test]
    fn validation_finds_dead_ends_and_bad_keys() {
        assert!(KeyboardConfig::iphone().validate().is_empty());
        assert!(KeyboardConfig::ipad().validate().is_empty());
        assert!(KeyboardConfig::ipad_pro().validate().is_empty());
        assert!(KeyboardConfig::phone_pad().validate().is_empty());
        let mut c = KeyboardConfig::iphone();
        c.symbols[3].retain(|k| k.action != KeyAction::Page(Page::Letters));
        c.letters[0].push(Key::letters(4));
        c.letters[1].push(Key::text("x").units(0.0));
        let issues = c.validate();
        assert!(
            issues.iter().any(|i| i.contains("symbols page goes back")),
            "{issues:?}"
        );
        assert!(
            issues.iter().any(|i| i.contains("letters(4)")),
            "{issues:?}"
        );
        assert!(issues.iter().any(|i| i.contains("0 units")), "{issues:?}");
    }
}
