//! Choosing a face for a family name as fontconfig does: its configuration
//! files read (the font directories, the `<match target="pattern">` rules
//! and `<alias>` lists), the pattern substituted and given its defaults,
//! the installed faces scanned for what matching compares (families,
//! language coverage, slant, weight, width, version), and the face with
//! the best score chosen (`FcFontMatch`).
//!
//! The behaviour follows fontconfig's; the code is written afresh.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::sfnt::{Face, Reader};

// --- a small XML reader ----------------------------------------------------

/// An XML element: its name, attributes, children and text.
#[derive(Clone, Debug, Default)]
struct Node {
    name: String,
    attrs: Vec<(String, String)>,
    children: Vec<Node>,
    text: String,
}

impl Node {
    fn attr(&self, name: &str) -> Option<&str> {
        self.attrs.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }
}

fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;

    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        rest = &rest[i..];

        let Some(end) = rest.find(';') else { break };
        let entity = &rest[1..end];

        match entity {
            "amp" => out.push('&'),
            "lt" => out.push('<'),
            "gt" => out.push('>'),
            "quot" => out.push('"'),
            "apos" => out.push('\''),
            _ if entity.starts_with("#x") => {
                if let Some(c) = u32::from_str_radix(&entity[2..], 16).ok().and_then(char::from_u32) {
                    out.push(c);
                }
            }
            _ if entity.starts_with('#') => {
                if let Some(c) = entity[1..].parse().ok().and_then(char::from_u32) {
                    out.push(c);
                }
            }
            _ => {}
        }

        rest = &rest[end + 1..];
    }

    out.push_str(rest);
    out
}

/// Parses an XML document into its root element.
fn parse_xml(text: &str) -> Option<Node> {
    let mut stack: Vec<Node> = vec![Node::default()];
    let mut rest = text;

    while !rest.is_empty() {
        if let Some(r) = rest.strip_prefix("<!--") {
            rest = &r[r.find("-->")? + 3..];
        } else if rest.starts_with("<?") {
            rest = &rest[rest.find("?>")? + 2..];
        } else if rest.starts_with("<!") {
            rest = &rest[rest.find('>')? + 1..];
        } else if let Some(r) = rest.strip_prefix("</") {
            let end = r.find('>')?;
            rest = &r[end + 1..];
            let node = stack.pop()?;
            stack.last_mut()?.children.push(node);
        } else if let Some(r) = rest.strip_prefix('<') {
            let end = r.find('>')?;
            let tag = &r[..end];
            rest = &r[end + 1..];
            let self_closing = tag.ends_with('/');
            let tag = tag.trim_end_matches('/');
            let name_end = tag.find(char::is_whitespace).unwrap_or(tag.len());
            let mut node = Node { name: tag[..name_end].to_string(), ..Node::default() };
            let mut attrs = &tag[name_end..];

            while let Some(eq) = attrs.find('=') {
                let key = attrs[..eq].trim().to_string();
                let after = attrs[eq + 1..].trim_start();
                let quote = after.chars().next()?;
                let close = after[1..].find(quote)?;
                node.attrs.push((key, unescape(&after[1..1 + close])));
                attrs = &after[close + 2..];
            }

            if self_closing {
                stack.last_mut()?.children.push(node);
            } else {
                stack.push(node);
            }
        } else {
            let end = rest.find('<').unwrap_or(rest.len());
            let t = unescape(&rest[..end]);
            stack.last_mut()?.text.push_str(&t);
            rest = &rest[end..];
        }
    }

    stack.pop()?.children.into_iter().next()
}

// --- patterns ------------------------------------------------------------

/// A pattern value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Str(String),
    Int(i64),
    Double(f64),
    Bool(bool),
}

impl Value {
    fn number(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::Double(d) => Some(*d),
            Value::Bool(b) => Some(f64::from(u8::from(*b))),
            Value::Str(_) => None,
        }
    }
}

/// How strongly a value binds in matching.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binding {
    Weak,
    Strong,
    Same,
}

/// A pattern: each element's values, in order.
#[derive(Clone, Debug, Default)]
pub struct Pattern {
    pub elements: BTreeMap<String, Vec<(Value, Binding)>>,
}

impl Pattern {
    fn get(&self, name: &str) -> &[(Value, Binding)] {
        self.elements.get(name).map_or(&[], Vec::as_slice)
    }

    /// The family names, in order.
    pub fn families(&self) -> Vec<String> {
        self.get("family")
            .iter()
            .filter_map(|(v, _)| match v {
                Value::Str(s) => Some(s.clone()),
                _ => None,
            })
            .collect()
    }
}

/// fontconfig's named constants: the element they belong to and value.
fn constant(name: &str) -> Option<(&'static str, i64)> {
    Some(match name {
        "thin" => ("weight", 0),
        "extralight" | "ultralight" => ("weight", 40),
        "light" => ("weight", 50),
        "demilight" | "semilight" => ("weight", 55),
        "book" => ("weight", 75),
        "regular" | "normal" => ("weight", 80),
        "medium" => ("weight", 100),
        "demibold" | "semibold" => ("weight", 180),
        "bold" => ("weight", 200),
        "extrabold" | "ultrabold" => ("weight", 205),
        "black" | "heavy" => ("weight", 210),
        "extrablack" | "ultrablack" => ("weight", 215),
        "roman" => ("slant", 0),
        "italic" => ("slant", 100),
        "oblique" => ("slant", 110),
        "ultracondensed" => ("width", 50),
        "extracondensed" => ("width", 63),
        "condensed" => ("width", 75),
        "semicondensed" => ("width", 87),
        "semiexpanded" => ("width", 113),
        "expanded" => ("width", 125),
        "extraexpanded" => ("width", 150),
        "ultraexpanded" => ("width", 200),
        "proportional" => ("spacing", 0),
        "dual" => ("spacing", 90),
        "mono" => ("spacing", 100),
        "charcell" => ("spacing", 110),
        "unknown" => ("rgba", 0),
        "rgb" => ("rgba", 1),
        "bgr" => ("rgba", 2),
        "vrgb" => ("rgba", 3),
        "vbgr" => ("rgba", 4),
        "none" => ("rgba", 5),
        "lcddefault" => ("lcdfilter", 1),
        "lcdlight" => ("lcdfilter", 2),
        "lcdlegacy" => ("lcdfilter", 3),
        "hintnone" => ("hintstyle", 0),
        "hintslight" => ("hintstyle", 1),
        "hintmedium" => ("hintstyle", 2),
        "hintfull" => ("hintstyle", 3),
        "serif" => ("genericfamily", 1),
        "sans-serif" => ("genericfamily", 2),
        "monospace" => ("genericfamily", 3),
        "cursive" => ("genericfamily", 4),
        "fantasy" => ("genericfamily", 5),
        "emoji" => ("genericfamily", 6),
        "math" => ("genericfamily", 7),
        _ => return None,
    })
}

// --- configuration -----------------------------------------------------------

/// A test of a rule.
#[derive(Clone, Debug)]
struct Test {
    qual: String,
    name: String,
    compare: String,
    /// None for a `<const xsi:nil="true"/>`: no value.
    value: Option<Value>,
}

/// An edit of a rule.
#[derive(Clone, Debug)]
struct Edit {
    name: String,
    mode: String,
    binding: Binding,
    values: Vec<Value>,
}

/// A `<match target="pattern">` rule (aliases become these too).
#[derive(Clone, Debug)]
struct Rule {
    tests: Vec<Test>,
    edits: Vec<Edit>,
}

/// fontconfig's configuration: the font directories and the pattern rules.
pub struct Config {
    pub dirs: Vec<PathBuf>,
    rules: Vec<Rule>,
}

fn home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default()
}

fn expand(path: &str, prefix: Option<&str>, base: &Path) -> PathBuf {
    if prefix == Some("xdg") {
        let data = std::env::var_os("XDG_DATA_HOME").map(PathBuf::from).unwrap_or_else(|| home().join(".local/share"));
        return data.join(path);
    }

    if let Some(rest) = path.strip_prefix("~/") {
        return home().join(rest);
    }

    let p = PathBuf::from(path);
    if p.is_absolute() { p } else { base.join(p) }
}

/// A value of an edit or test element.
fn node_value(node: &Node) -> Option<Option<Value>> {
    let text = node.text.trim();

    Some(match node.name.as_str() {
        "string" => Some(Value::Str(text.to_string())),
        "int" => Some(Value::Int(text.parse().ok()?)),
        "double" => Some(Value::Double(text.parse().ok()?)),
        "bool" => Some(Value::Bool(text == "true")),
        "const" => {
            if node.attr("xsi:nil") == Some("true") {
                None
            } else {
                Some(Value::Int(constant(text)?.1))
            }
        }
        _ => return None,
    })
}

fn binding(attr: Option<&str>) -> Binding {
    match attr {
        Some("strong") => Binding::Strong,
        Some("same") => Binding::Same,
        _ => Binding::Weak,
    }
}

impl Config {
    /// Reads a configuration file and what it includes.
    pub fn load(path: &Path) -> Config {
        let mut config = Config { dirs: Vec::new(), rules: Vec::new() };
        config.read(path);
        config
    }

    fn read(&mut self, path: &Path) {
        let Ok(text) = std::fs::read_to_string(path) else { return };
        let Some(root) = parse_xml(&text) else { return };
        let base = path.parent().unwrap_or(Path::new("/")).to_path_buf();

        for node in &root.children {
            match node.name.as_str() {
                "dir" => self.dirs.push(expand(node.text.trim(), node.attr("prefix"), &base)),
                "include" => {
                    let target = expand(node.text.trim(), node.attr("prefix"), &base);

                    if target.is_dir() {
                        let mut files: Vec<PathBuf> = std::fs::read_dir(&target)
                            .map(|d| d.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "conf")).collect())
                            .unwrap_or_default();
                        files.sort();

                        for f in files {
                            self.read(&f);
                        }
                    } else {
                        self.read(&target);
                    }
                }
                "match" if node.attr("target").unwrap_or("pattern") == "pattern" => {
                    if let Some(rule) = Self::rule(node) {
                        self.rules.push(rule);
                    }
                }
                "alias" => self.alias(node),
                _ => {}
            }
        }
    }

    fn rule(node: &Node) -> Option<Rule> {
        let mut rule = Rule { tests: Vec::new(), edits: Vec::new() };

        for child in &node.children {
            match child.name.as_str() {
                "test" => {
                    // only what the pattern has is tested here
                    if child.attr("target").is_some_and(|t| t == "font") {
                        return None;
                    }

                    let value = child.children.first().and_then(node_value)?;
                    rule.tests.push(Test {
                        qual: child.attr("qual").unwrap_or("any").to_string(),
                        name: child.attr("name")?.to_string(),
                        compare: child.attr("compare").unwrap_or("eq").to_string(),
                        value,
                    });
                }
                "edit" => {
                    let mut values = Vec::new();

                    for v in &child.children {
                        match node_value(v) {
                            Some(Some(v)) => values.push(v),
                            Some(None) => {}
                            // expressions: not what the pattern rules here need
                            None => return None,
                        }
                    }

                    rule.edits.push(Edit {
                        name: child.attr("name")?.to_string(),
                        mode: child.attr("mode").unwrap_or("assign").to_string(),
                        binding: binding(child.attr("binding")),
                        values,
                    });
                }
                _ => {}
            }
        }

        Some(rule)
    }

    /// An `<alias>`: a family test, its `prefer` list before the family,
    /// `accept` after it and `default` at the end.
    fn alias(&mut self, node: &Node) {
        let b = binding(node.attr("binding"));
        let families = |name: &str| -> Vec<Value> {
            node.children
                .iter()
                .filter(|c| c.name == name)
                .flat_map(|c| c.children.iter().filter(|f| f.name == "family").map(|f| Value::Str(f.text.trim().to_string())))
                .collect()
        };

        let tests: Vec<Value> = node.children.iter().filter(|c| c.name == "family").map(|f| Value::Str(f.text.trim().to_string())).collect();

        for family in tests {
            let mut edits = Vec::new();

            for (list, mode) in [("prefer", "prepend"), ("accept", "append"), ("default", "append_last")] {
                let values = families(list);

                if !values.is_empty() {
                    edits.push(Edit { name: "family".into(), mode: mode.into(), binding: b, values });
                }
            }

            self.rules.push(Rule {
                tests: vec![Test { qual: "any".into(), name: "family".into(), compare: "eq".into(), value: Some(family) }],
                edits,
            });
        }
    }
}

/// Strings compared without case or blanks (family names) or just without
/// case.
fn fold(s: &str, blanks: bool) -> String {
    s.chars().filter(|c| !(blanks && *c == ' ')).flat_map(char::to_lowercase).collect()
}

/// `FcConfigCompareValue`
fn compare_value(name: &str, left: &Value, op: &str, right: &Value) -> bool {
    match (left, right) {
        (Value::Str(a), Value::Str(b)) => {
            let blanks = name == "family";
            let (a, b) = (fold(a, blanks), fold(b, blanks));

            match op {
                "eq" => a == b,
                "not_eq" => a != b,
                "contains" => a.contains(&b),
                "not_contains" => !a.contains(&b),
                _ => false,
            }
        }
        _ => {
            let (Some(a), Some(b)) = (left.number(), right.number()) else { return false };

            match op {
                "eq" | "contains" => a == b,
                "not_eq" | "not_contains" => a != b,
                "less" => a < b,
                "less_eq" => a <= b,
                "more" => a > b,
                "more_eq" => a >= b,
                _ => false,
            }
        }
    }
}

impl Config {
    /// `FcConfigSubstitute( FcMatchPattern )`
    pub fn substitute(&self, pattern: &mut Pattern) {
        for rule in &self.rules {
            // which value of the tested element each test matched
            let mut matched: BTreeMap<String, usize> = BTreeMap::new();
            let mut ok = true;

            for t in &rule.tests {
                let values = pattern.get(&t.name);

                let Some(want) = &t.value else {
                    // a test against nothing: whether the element is absent
                    let absent = values.is_empty();
                    ok = match t.compare.as_str() {
                        "eq" | "contains" => absent,
                        _ => !absent,
                    };

                    if !ok {
                        break;
                    }

                    continue;
                };

                let hits: Vec<bool> = values.iter().map(|(v, _)| compare_value(&t.name, v, &t.compare, want)).collect();

                let pass = match t.qual.as_str() {
                    "all" => hits.iter().all(|h| *h),
                    "first" => hits.first().copied().unwrap_or(false),
                    "not_first" => hits.iter().skip(1).any(|h| *h),
                    _ => hits.iter().any(|h| *h),
                };

                if !pass {
                    ok = false;
                    break;
                }

                if let Some(i) = hits.iter().position(|h| *h) {
                    let i = if t.qual == "not_first" { hits.iter().skip(1).position(|h| *h).map_or(i, |k| k + 1) } else { i };
                    matched.entry(t.name.clone()).or_insert(i);
                }
            }

            if !ok {
                continue;
            }

            for e in &rule.edits {
                let at = matched.get(&e.name).copied();
                let list = pattern.elements.entry(e.name.clone()).or_default();
                let same = at.and_then(|i| list.get(i)).map_or(Binding::Weak, |v| v.1);
                let b = if e.binding == Binding::Same { same } else { e.binding };
                let new: Vec<(Value, Binding)> = e.values.iter().map(|v| (v.clone(), b)).collect();

                match (e.mode.as_str(), at) {
                    ("assign", Some(i)) => {
                        list.splice(i..=i, new);
                    }
                    ("assign", None) | ("assign_replace", _) => *list = new,
                    ("prepend", Some(i)) => {
                        list.splice(i..i, new);
                    }
                    ("prepend", None) | ("prepend_first", _) => {
                        list.splice(0..0, new);
                    }
                    ("append", Some(i)) => {
                        list.splice(i + 1..i + 1, new);
                    }
                    ("append", None) | ("append_last", _) => list.extend(new),
                    ("delete", Some(i)) => {
                        list.remove(i);
                    }
                    ("delete_all", _) => list.clear(),
                    _ => {}
                }

                if list.is_empty() {
                    pattern.elements.remove(&e.name);
                }
            }
        }
    }
}

/// `FcDefaultSubstitute`, as far as matching sees it.
pub fn default_substitute(pattern: &mut Pattern, lang: &str) {
    let mut add = |name: &str, v: Value, b: Binding| {
        pattern.elements.entry(name.to_string()).or_insert_with(|| vec![(v, b)]);
    };

    add("weight", Value::Int(80), Binding::Strong);
    add("slant", Value::Int(0), Binding::Strong);
    add("width", Value::Int(100), Binding::Strong);
    add("fontversion", Value::Int(0x7FFF_FFFF), Binding::Strong);
    add("decorative", Value::Bool(false), Binding::Strong);
    add("symbol", Value::Bool(false), Binding::Strong);
    add("variable", Value::Bool(false), Binding::Strong);
    add("order", Value::Int(0), Binding::Strong);
    add("lang", Value::Str(lang.to_string()), Binding::Weak);
}

// --- the installed faces ---------------------------------------------------------

/// A scanned face: what matching compares.
#[derive(Clone, Debug)]
pub struct FontEntry {
    pub path: PathBuf,
    /// The face's index, a named instance's number in the high bits.
    pub index: u32,
    pub families: Vec<String>,
    pub style: String,
    pub weight: f64,
    pub slant: i64,
    pub width: f64,
    /// Whether it covers English (`FcLangSetHasLang( "en" ) == FcLangEqual`).
    pub english: bool,
    pub version: i64,
    pub symbol: bool,
    pub variable: bool,
    pub decorative: bool,
    /// A variable font's own entry: the weights and widths its axes span.
    pub weight_range: Option<(f64, f64)>,
    pub width_range: Option<(f64, f64)>,
}

/// `FcWeightFromOpenTypeDouble`
fn weight_from_opentype(ot: f64) -> f64 {
    let ot = if (1.0..=9.0).contains(&ot) { ot * 100.0 } else { ot };
    const MAP: [(f64, f64); 13] = [
        (0.0, 0.0),
        (100.0, 0.0),
        (200.0, 40.0),
        (300.0, 50.0),
        (350.0, 55.0),
        (380.0, 75.0),
        (400.0, 80.0),
        (500.0, 100.0),
        (600.0, 180.0),
        (700.0, 200.0),
        (800.0, 205.0),
        (900.0, 210.0),
        (1000.0, 215.0),
    ];

    let ot = ot.clamp(0.0, 1000.0);

    for w in MAP.windows(2) {
        let ((a, fa), (b, fb)) = (w[0], w[1]);

        if ot <= b {
            return if b == a { fa } else { fa + (fb - fa) * (ot - a) / (b - a) };
        }
    }

    215.0
}

/// The names with one of `ids`, in the order fontconfig takes them: by
/// platform (Microsoft, Unicode, Macintosh), then by the order of `ids`,
/// the English ones first and the rest as the table has them; repeats
/// once.
fn names(face: &Face, ids: &[u16]) -> Vec<String> {
    let Some(t) = face.table(b"name") else { return Vec::new() };
    let r = Reader::new(t);
    let (Ok(count), Ok(storage)) = (r.u16(2), r.u16(4)) else { return Vec::new() };
    let mut out: Vec<String> = Vec::new();

    for platform in [3u16, 0, 1] {
        for &id in ids {
            let mut group: Vec<(bool, String)> = Vec::new();

            for i in 0..count as usize {
                let at = 6 + 12 * i;
                let (Ok(pl), Ok(encoding), Ok(language), Ok(nid), Ok(len), Ok(off)) =
                    (r.u16(at), r.u16(at + 2), r.u16(at + 4), r.u16(at + 6), r.u16(at + 8), r.u16(at + 10))
                else {
                    continue;
                };

                if pl != platform || nid != id {
                    continue;
                }

                let Ok(bytes) = r.slice(storage as usize + off as usize, len as usize) else { continue };

                let (english, text) = match (pl, encoding) {
                    (3, 0) | (3, 1) | (3, 10) | (0, _) => {
                        let units: Vec<u16> = bytes.as_chunks::<2>().0.iter().map(|c| u16::from_be_bytes([c[0], c[1]])).collect();
                        (pl == 0 || language & 0x3FF == 0x09, String::from_utf16_lossy(&units))
                    }
                    (1, 0) => (language == 0, bytes.iter().map(|&b| mac_roman(b)).collect()),
                    _ => continue,
                };

                if !text.is_empty() {
                    group.push((english, text));
                }
            }

            // stable: the English names ahead of the others
            group.sort_by_key(|(english, _)| !english);

            for (_, text) in group {
                if !out.contains(&text) {
                    out.push(text);
                }
            }
        }
    }

    out
}

/// The first of a list of keywords a style name holds, as fontconfig
/// looks for them (`FcContainsWeight` and the others).
fn style_keyword(style: &str, table: &[(&str, f64)]) -> Option<f64> {
    let s = fold(style, true);
    table.iter().find(|(k, _)| s.contains(k)).map(|(_, v)| *v)
}

const WEIGHT_WORDS: [(&str, f64); 22] = [
    ("thin", 0.0),
    ("extralight", 40.0),
    ("ultralight", 40.0),
    ("demilight", 55.0),
    ("semilight", 55.0),
    ("light", 50.0),
    ("book", 75.0),
    ("regular", 80.0),
    ("normal", 80.0),
    ("medium", 100.0),
    ("demibold", 180.0),
    ("demi", 180.0),
    ("semibold", 180.0),
    ("extrabold", 205.0),
    ("superbold", 205.0),
    ("ultrabold", 205.0),
    ("bold", 200.0),
    ("ultrablack", 215.0),
    ("superblack", 215.0),
    ("extrablack", 215.0),
    ("black", 210.0),
    ("heavy", 210.0),
];

const WIDTH_WORDS: [(&str, f64); 10] = [
    ("ultracondensed", 50.0),
    ("extracondensed", 63.0),
    ("semicondensed", 87.0),
    ("condensed", 75.0),
    ("normal", 100.0),
    ("semiexpanded", 113.0),
    ("extraexpanded", 150.0),
    ("ultraexpanded", 200.0),
    ("expanded", 125.0),
    ("extended", 125.0),
];

const SLANT_WORDS: [(&str, f64); 3] = [("italic", 100.0), ("kursiv", 100.0), ("oblique", 110.0)];

/// A Mac OS Roman byte as a character.
fn mac_roman(b: u8) -> char {
    const HIGH: &str = "ÄÅÇÉÑÖÜáàâäãåçéèêëíìîïñóòôöõúùûü†°¢£§•¶ß®©™´¨≠ÆØ∞±≤≥¥µ∂∑∏π∫ªºΩæø¿¡¬√ƒ≈∆«»… ÀÃÕŒœ–—“”‘’÷◊ÿŸ⁄€‹›ﬁﬂ‡·‚„‰ÂÊÁËÈÍÎÏÌÓÔ\u{F8FF}ÒÚÛÙıˆ˜¯˘˙˚¸˝˛ˇ";

    if b < 0x80 {
        char::from(b)
    } else {
        HIGH.chars().nth(usize::from(b - 0x80)).unwrap_or('?')
    }
}

/// fontconfig's English orthography (`en.orth`).
const ENGLISH: [(u32, u32); 16] = [
    (0x41, 0x5A),
    (0x61, 0x7A),
    (0xC0, 0xC0),
    (0xC2, 0xC2),
    (0xC7, 0xCB),
    (0xCE, 0xCF),
    (0xD1, 0xD1),
    (0xD4, 0xD4),
    (0xD6, 0xD6),
    (0xE0, 0xE0),
    (0xE2, 0xE2),
    (0xE7, 0xEB),
    (0xEE, 0xEF),
    (0xF1, 0xF1),
    (0xF4, 0xF4),
    (0xF6, 0xF6),
];

/// A variable font's axes (tag, min, default, max) and its named instances
/// (subfamily name ID, coordinates).
type Fvar = (Vec<([u8; 4], f64, f64, f64)>, Vec<(u16, Vec<f64>)>);

/// A variable font's axes (tag, min, default, max) and named instances
/// (subfamily name ID, coordinates), from `fvar`.
fn fvar(face: &Face) -> Option<Fvar> {
    let t = face.table(b"fvar")?;
    let r = Reader::new(t);
    let axes_at = r.u16(4).ok()? as usize;
    let (n_axes, axis_size) = (r.u16(8).ok()? as usize, r.u16(10).ok()? as usize);
    let (n_inst, inst_size) = (r.u16(12).ok()? as usize, r.u16(14).ok()? as usize);
    let fixed = |at: usize| r.u32(at).ok().map(|v| f64::from(v as i32) / 65536.0);
    let mut axes = Vec::with_capacity(n_axes);

    for i in 0..n_axes {
        let at = axes_at + i * axis_size;
        let tag: [u8; 4] = r.slice(at, 4).ok()?.try_into().ok()?;
        axes.push((tag, fixed(at + 4)?, fixed(at + 8)?, fixed(at + 12)?));
    }

    let mut instances = Vec::with_capacity(n_inst);
    let inst_at = axes_at + n_axes * axis_size;

    for i in 0..n_inst {
        let at = inst_at + i * inst_size;
        let name = r.u16(at).ok()?;
        let coords = (0..n_axes).map(|k| fixed(at + 4 + 4 * k)).collect::<Option<Vec<f64>>>()?;
        instances.push((name, coords));
    }

    Some((axes, instances))
}

impl FontEntry {
    /// The face's entries: itself, then, when variable, its named instances
    /// (but the one at the default) and its variable entry.
    fn scan(path: &Path, data: &[u8], index: u32) -> Vec<FontEntry> {
        let Ok(face) = Face::new(data.to_vec(), index) else { return Vec::new() };
        let mut families: Vec<String> = Vec::new();

        // duplicates that differ in blanks or case are one family
        for f in names(&face, &[21, 16, 1]) {
            if !families.iter().any(|g| fold(g, true) == fold(&f, true)) {
                families.push(f);
            }
        }

        let styles = names(&face, &[22, 17, 2]);
        let style = styles.first().cloned().unwrap_or_default();
        let os2 = face.os2;

        // FreeType's style flags: OS/2's, else head's macStyle
        let mac_style = face.table(b"head").and_then(|h| Reader::new(h).u16(44).ok()).unwrap_or(0);
        let (flag_bold, flag_italic) = match os2 {
            Some(o) => (o.fs_selection & 0x20 != 0, o.fs_selection & 1 != 0),
            None => (mac_style & 1 != 0, mac_style & 2 != 0),
        };

        let mut weight = os2.map(|o| weight_from_opentype(f64::from(o.weight_class)));
        let mut width = os2.and_then(|o| match o.width_class {
            1 => Some(50.0),
            2 => Some(62.5),
            3 => Some(75.0),
            4 => Some(87.5),
            5 => Some(100.0),
            6 => Some(112.5),
            7 => Some(125.0),
            8 => Some(150.0),
            9 => Some(200.0),
            _ => None,
        });
        let mut slant = None;
        let mut decorative = false;

        // what OS/2 does not say, from the style names
        for st in &styles {
            weight = weight.or_else(|| style_keyword(st, &WEIGHT_WORDS));
            width = width.or_else(|| style_keyword(st, &WIDTH_WORDS));
            slant = slant.or_else(|| style_keyword(st, &SLANT_WORDS));
            decorative |= fold(st, true).contains("decorative");
        }

        let weight = weight.unwrap_or(if flag_bold { 200.0 } else { 80.0 });
        let width = width.unwrap_or(100.0);
        let slant = slant.map_or(if flag_italic { 100 } else { 0 }, |s| s as i64);
        let version = face.table(b"head").map_or(0, |h| i64::from(Reader::new(h).u32(4).unwrap_or(0)));

        // English: every letter of fontconfig's orthography
        let english = ENGLISH.iter().all(|&(a, b)| (a..=b).all(|c| face.glyph_index(c) != 0));
        let symbol = face.has_symbol_cmap();

        let base = FontEntry {
            path: path.to_path_buf(),
            index,
            families,
            style,
            weight,
            slant,
            width,
            english,
            version,
            symbol,
            variable: false,
            decorative,
            weight_range: None,
            width_range: None,
        };

        let Some((axes, instances)) = fvar(&face) else { return vec![base] };
        let mut out = vec![base.clone()];
        let defaults: Vec<f64> = axes.iter().map(|a| a.2).collect();

        for (i, (name, coords)) in instances.iter().enumerate() {
            if *coords == defaults {
                continue;
            }

            let mut e = base.clone();
            e.index = index | ((i as u32 + 1) << 16);
            e.style = names(&face, &[*name]).into_iter().next().unwrap_or_default();

            for ((tag, ..), v) in axes.iter().zip(coords) {
                match tag {
                    b"wght" => e.weight = weight_from_opentype(*v),
                    b"wdth" => e.width = *v,
                    b"ital" if *v > 0.0 => e.slant = 100,
                    b"slnt" if *v != 0.0 => e.slant = 110,
                    _ => {}
                }
            }

            out.push(e);
        }

        let mut master = base;
        master.variable = true;
        master.style = String::new();

        for (tag, min, _, max) in &axes {
            match tag {
                b"wght" => master.weight_range = Some((weight_from_opentype(*min), weight_from_opentype(*max))),
                b"wdth" => master.width_range = Some((*min, *max)),
                _ => {}
            }
        }

        out.push(master);
        out
    }
}

/// Every face in the directories, in fontconfig's order: the directories
/// as given, then their subdirectories, the files of each sorted by name.
pub fn scan(dirs: &[PathBuf]) -> Vec<FontEntry> {
    let mut queue: std::collections::VecDeque<PathBuf> = dirs.iter().cloned().collect();
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();

    while let Some(dir) = queue.pop_front() {
        if !seen.insert(dir.clone()) {
            continue;
        }

        let Ok(read) = std::fs::read_dir(&dir) else { continue };
        let mut entries: Vec<PathBuf> = read.flatten().map(|e| e.path()).collect();
        entries.sort();

        for path in entries {
            if path.is_dir() {
                queue.push_back(path);
                continue;
            }

            let Ok(data) = std::fs::read(&path) else { continue };

            let count = if data.starts_with(b"ttcf") {
                Reader::new(&data).u32(8).unwrap_or(0)
            } else {
                1
            };

            for i in 0..count {
                out.extend(FontEntry::scan(&path, &data, i));
            }
        }
    }

    out
}

// --- matching ----------------------------------------------------------------------

/// `FcFontMatch`: the face whose scores, element by element in
/// fontconfig's priority order, are least; the first such.
pub fn font_match<'a>(fonts: &'a [FontEntry], pattern: &Pattern) -> Option<&'a FontEntry> {
    let families = pattern.get("family");
    let number = |name: &str| pattern.get(name).iter().filter_map(|(v, _)| v.number()).collect::<Vec<f64>>();
    let (weights, slants, widths, versions) = (number("weight"), number("slant"), number("width"), number("fontversion"));

    // the least of value distance * 1000 + the value's place
    let best = |values: &[f64], font: f64| -> f64 {
        values.iter().enumerate().map(|(j, v)| (v - font).abs() * 1000.0 + j as f64).fold(f64::INFINITY, f64::min)
    };

    // a range: no distance inside it
    let best_range = |values: &[f64], (lo, hi): (f64, f64)| -> f64 {
        values
            .iter()
            .enumerate()
            .map(|(j, v)| (if *v < lo { lo - v } else if *v > hi { v - hi } else { 0.0 }) * 1000.0 + j as f64)
            .fold(f64::INFINITY, f64::min)
    };

    let score = |f: &FontEntry| -> Vec<f64> {
        let (mut strong, mut weak) = (1e99f64, 1e99f64);

        for (j, (v, b)) in families.iter().enumerate() {
            let Value::Str(name) = v else { continue };
            let name = fold(name, true);
            let d = if f.families.iter().any(|fam| fold(fam, true) == name) { 0.0 } else { 1.0 };
            let s = d * 1000.0 + j as f64;

            if *b == Binding::Strong {
                strong = strong.min(s);
            } else {
                weak = weak.min(s);
            }
        }

        vec![
            if f.variable { 1000.0 } else { 0.0 },
            strong,
            if f.english { 0.0 } else { 2000.0 },
            weak,
            if f.symbol { 1000.0 } else { 0.0 },
            best(&slants, f.slant as f64),
            f.weight_range.map_or_else(|| best(&weights, f.weight), |r| best_range(&weights, r)),
            f.width_range.map_or_else(|| best(&widths, f.width), |r| best_range(&widths, r)),
            if f.decorative { 1000.0 } else { 0.0 },
            best(&versions, f.version as f64),
        ]
    };

    let mut chosen: Option<(&FontEntry, Vec<f64>)> = None;

    for f in fonts {
        let s = score(f);

        if chosen.as_ref().is_none_or(|(_, b)| s.partial_cmp(b) == Some(std::cmp::Ordering::Less)) {
            chosen = Some((f, s));
        }
    }

    chosen.map(|(f, _)| f)
}

/// A pattern for a family, bold or italic as `FONTCONFIG::FindFont` asks.
pub fn find_pattern(family: &str, bold: bool, italic: bool) -> Pattern {
    let mut p = Pattern::default();

    if bold {
        p.elements.insert("weight".into(), vec![(Value::Int(200), Binding::Strong)]);
    }

    if italic {
        p.elements.insert("slant".into(), vec![(Value::Int(100), Binding::Strong)]);
    }

    p.elements.insert("family".into(), vec![(Value::Str(family.to_string()), Binding::Strong)]);
    p
}
