//! ISO 10303-21 — the STEP physical file.
//!
//! The exchange syntax only, with no opinion about what the entities mean: a
//! file is a header and a list of `#id = NAME(args);` instances, and an argument
//! is a reference, a number, a string, an enum, a list, or one of the two
//! placeholders. What those entities *say* is AP203/AP214, and that lives in
//! [`mod@super::export`] and [`mod@super::import`].
//!
//! Keeping the split sharp is worth it: nearly every real-world STEP problem is
//! a lexical one — a real written without its decimal point, a name with a
//! quote in it, a `-0.` where a reader expected an integer — and those are
//! findable here, against a round-trip, without constructing any geometry.

use std::collections::HashMap;
use std::fmt::Write as _;

/// One argument of an entity instance.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// `#42` — a reference to another instance.
    Ref(u64),
    Int(i64),
    Real(f64),
    /// `'text'`, already unescaped.
    Text(String),
    /// `.T.`, `.CARTESIAN.` — held without the dots.
    Enum(String),
    /// `"0F3A"` — a binary literal, held as its hex digits.
    Binary(String),
    List(Vec<Value>),
    /// `NAME(args)` appearing as an argument, which is how a SELECT type that
    /// needs naming, or a nested constructor, is written.
    Typed(String, Vec<Value>),
    /// `$` — unset.
    Omitted,
    /// `*` — inherited from a supertype and not restated.
    Derived,
}

impl Value {
    /// The referenced id, for the common case of an argument that must be one.
    pub fn as_ref(&self) -> Option<u64> {
        match self {
            Value::Ref(id) => Some(*id),
            _ => None,
        }
    }

    /// A number, accepting an integer where a real is expected.
    ///
    /// STEP requires reals to carry a decimal point, so `1` and `1.` are
    /// formally different tokens. Readers that enforce that reject a great many
    /// files in the wild; this accepts both and writes the correct one.
    pub fn as_real(&self) -> Option<f64> {
        match self {
            Value::Real(x) => Some(*x),
            Value::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            Value::Text(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_list(&self) -> Option<&[Value]> {
        match self {
            Value::List(v) => Some(v),
            _ => None,
        }
    }

    /// `.T.` / `.F.`, the STEP spelling of a boolean.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Enum(e) if e == "T" => Some(true),
            Value::Enum(e) if e == "F" => Some(false),
            _ => None,
        }
    }
}

/// One `#id = NAME(args);` instance.
///
/// A *complex* instance — `#1 = (A(..) B(..));`, which AP203 uses for the
/// rational B-spline surfaces and for units — has an empty `name` and carries
/// its parts as [`Value::Typed`] in `args`.
#[derive(Debug, Clone, PartialEq)]
pub struct Entity {
    pub id: u64,
    pub name: String,
    pub args: Vec<Value>,
}

impl Entity {
    /// The part of a complex instance with this name, or the whole entity if it
    /// is simple and named that.
    pub fn part(&self, name: &str) -> Option<&[Value]> {
        if self.name == name {
            return Some(&self.args);
        }
        self.args.iter().find_map(|a| match a {
            Value::Typed(n, args) if n == name => Some(&args[..]),
            _ => None,
        })
    }

    /// Whether a simple or complex instance carries this name at all.
    pub fn is(&self, name: &str) -> bool {
        self.part(name).is_some()
    }
}

/// A parsed exchange file.
#[derive(Debug, Clone, Default)]
pub struct StepFile {
    /// Header instances (`FILE_DESCRIPTION`, `FILE_NAME`, `FILE_SCHEMA`), which
    /// are unnumbered in the file and given sequential ids here.
    pub header: Vec<Entity>,
    pub data: Vec<Entity>,
}

impl StepFile {
    /// Index the data section by id.
    pub fn index(&self) -> HashMap<u64, &Entity> {
        self.data.iter().map(|e| (e.id, e)).collect()
    }

    /// Every data instance carrying `name`, in file order.
    pub fn all<'s>(&'s self, name: &'s str) -> impl Iterator<Item = &'s Entity> + 's {
        self.data.iter().filter(move |e| e.is(name))
    }
}

/// Why a file could not be read.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
    /// Byte offset into the input, for pointing at the problem.
    pub at: usize,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at byte {}", self.message, self.at)
    }
}

impl std::error::Error for ParseError {}

// --------------------------------------------------------------------------
// Reading
// --------------------------------------------------------------------------

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn err<T>(&self, message: impl Into<String>) -> Result<T, ParseError> {
        Err(ParseError {
            message: message.into(),
            at: self.i,
        })
    }

    /// Whitespace and `/* */` comments.
    fn space(&mut self) {
        loop {
            while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
                self.i += 1;
            }
            if self.s[self.i..].starts_with(b"/*") {
                match find(&self.s[self.i + 2..], b"*/") {
                    Some(n) => self.i += 2 + n + 2,
                    None => {
                        self.i = self.s.len();
                        return;
                    }
                }
            } else {
                return;
            }
        }
    }

    fn eat(&mut self, word: &[u8]) -> bool {
        self.space();
        // Keywords are case-insensitive in practice, and files written by hand
        // are not consistent about it.
        if self.s.len() >= self.i + word.len()
            && self.s[self.i..self.i + word.len()].eq_ignore_ascii_case(word)
        {
            self.i += word.len();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, word: &[u8]) -> Result<(), ParseError> {
        if self.eat(word) {
            Ok(())
        } else {
            self.err(format!("expected `{}`", String::from_utf8_lossy(word)))
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.space();
        self.s.get(self.i).copied()
    }

    fn name(&mut self) -> Result<String, ParseError> {
        self.space();
        let start = self.i;
        while self.i < self.s.len()
            && (self.s[self.i].is_ascii_alphanumeric() || self.s[self.i] == b'_')
        {
            self.i += 1;
        }
        if self.i == start {
            return self.err("expected an entity name");
        }
        Ok(String::from_utf8_lossy(&self.s[start..self.i]).to_ascii_uppercase())
    }

    fn value(&mut self) -> Result<Value, ParseError> {
        match self.peek() {
            None => self.err("unexpected end of file"),
            Some(b'$') => {
                self.i += 1;
                Ok(Value::Omitted)
            }
            Some(b'*') => {
                self.i += 1;
                Ok(Value::Derived)
            }
            Some(b'#') => {
                self.i += 1;
                let start = self.i;
                while self.i < self.s.len() && self.s[self.i].is_ascii_digit() {
                    self.i += 1;
                }
                if self.i == start {
                    return self.err("expected a digit after `#`");
                }
                let text = std::str::from_utf8(&self.s[start..self.i]).unwrap_or("");
                match text.parse() {
                    Ok(id) => Ok(Value::Ref(id)),
                    Err(_) => self.err("instance id out of range"),
                }
            }
            Some(b'\'') => Ok(Value::Text(self.text()?)),
            Some(b'"') => {
                self.i += 1;
                let start = self.i;
                while self.i < self.s.len() && self.s[self.i] != b'"' {
                    self.i += 1;
                }
                let hex = String::from_utf8_lossy(&self.s[start..self.i]).to_string();
                if self.i >= self.s.len() {
                    return self.err("unterminated binary literal");
                }
                self.i += 1;
                Ok(Value::Binary(hex))
            }
            Some(b'.') => {
                self.i += 1;
                let start = self.i;
                while self.i < self.s.len() && self.s[self.i] != b'.' {
                    self.i += 1;
                }
                let e = String::from_utf8_lossy(&self.s[start..self.i]).to_string();
                if self.i >= self.s.len() {
                    return self.err("unterminated enumeration");
                }
                self.i += 1;
                Ok(Value::Enum(e))
            }
            Some(b'(') => {
                self.i += 1;
                Ok(Value::List(self.args(b')')?))
            }
            Some(c) if c.is_ascii_digit() || c == b'-' || c == b'+' => self.number(),
            Some(c) if c.is_ascii_alphabetic() || c == b'_' || c == b'!' => {
                if self.s[self.i] == b'!' {
                    self.i += 1; // a user-defined entity name
                }
                let name = self.name()?;
                self.expect(b"(")?;
                Ok(Value::Typed(name, self.args(b')')?))
            }
            Some(c) => self.err(format!("unexpected `{}`", c as char)),
        }
    }

    /// A string literal, with `''` unescaped and `\X2\..\X0\` decoded.
    fn text(&mut self) -> Result<String, ParseError> {
        self.i += 1; // opening quote
        let mut raw = String::new();
        loop {
            let Some(&c) = self.s.get(self.i) else {
                return self.err("unterminated string");
            };
            self.i += 1;
            if c == b'\'' {
                if self.s.get(self.i) == Some(&b'\'') {
                    self.i += 1;
                    raw.push('\'');
                    continue;
                }
                break;
            }
            raw.push(c as char);
        }
        Ok(decode_control_directives(&raw))
    }

    fn number(&mut self) -> Result<Value, ParseError> {
        let start = self.i;
        if matches!(self.s.get(self.i), Some(b'-') | Some(b'+')) {
            self.i += 1;
        }
        let mut real = false;
        while let Some(&c) = self.s.get(self.i) {
            if c.is_ascii_digit() {
                self.i += 1;
            } else if c == b'.' {
                real = true;
                self.i += 1;
            } else if c == b'e' || c == b'E' {
                real = true;
                self.i += 1;
                if matches!(self.s.get(self.i), Some(b'-') | Some(b'+')) {
                    self.i += 1;
                }
            } else {
                break;
            }
        }
        let text = String::from_utf8_lossy(&self.s[start..self.i]).to_string();
        if real {
            // `1.E-3` and `1.` are legal STEP but not legal Rust; patch the
            // missing mantissa digit before parsing.
            let patched = text.replace(".E", ".0E").replace(".e", ".0e");
            let patched = if patched.ends_with('.') {
                format!("{patched}0")
            } else {
                patched
            };
            match patched.parse() {
                Ok(x) => Ok(Value::Real(x)),
                Err(_) => self.err(format!("`{text}` is not a real")),
            }
        } else {
            match text.parse() {
                Ok(i) => Ok(Value::Int(i)),
                Err(_) => self.err(format!("`{text}` is not an integer")),
            }
        }
    }

    /// Comma-separated values up to `close`, which is consumed.
    fn args(&mut self, close: u8) -> Result<Vec<Value>, ParseError> {
        let mut out = Vec::new();
        if self.peek() == Some(close) {
            self.i += 1;
            return Ok(out);
        }
        loop {
            out.push(self.value()?);
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(c) if c == close => {
                    self.i += 1;
                    return Ok(out);
                }
                None => return self.err("unterminated argument list"),
                Some(c) => return self.err(format!("expected `,` or `)`, found `{}`", c as char)),
            }
        }
    }

    /// One instance, or `None` at `ENDSEC;`.
    fn instance(&mut self, next_header_id: &mut u64) -> Result<Option<Entity>, ParseError> {
        if self.peek().is_none() || self.eat(b"ENDSEC") {
            let _ = self.eat(b";");
            return Ok(None);
        }
        let id = if self.peek() == Some(b'#') {
            self.i += 1;
            let start = self.i;
            while self.i < self.s.len() && self.s[self.i].is_ascii_digit() {
                self.i += 1;
            }
            let text = std::str::from_utf8(&self.s[start..self.i]).unwrap_or("");
            let Ok(id) = text.parse::<u64>() else {
                return self.err("instance id out of range");
            };
            self.expect(b"=")?;
            id
        } else {
            // A header instance, which has no id of its own.
            *next_header_id += 1;
            *next_header_id
        };

        let entity = if self.peek() == Some(b'(') {
            // Complex instance: `(A(..) B(..))`, whitespace-separated.
            self.i += 1;
            let mut parts = Vec::new();
            loop {
                match self.peek() {
                    Some(b')') => {
                        self.i += 1;
                        break;
                    }
                    Some(b',') => self.i += 1,
                    None => return self.err("unterminated complex instance"),
                    _ => {
                        let name = self.name()?;
                        self.expect(b"(")?;
                        parts.push(Value::Typed(name, self.args(b')')?));
                    }
                }
            }
            Entity {
                id,
                name: String::new(),
                args: parts,
            }
        } else {
            let name = self.name()?;
            self.expect(b"(")?;
            let args = self.args(b')')?;
            Entity { id, name, args }
        };
        self.expect(b";")?;
        Ok(Some(entity))
    }
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Decode the `\X2\....\X0\` and `\X\hh` control directives of ISO 10303-21.
///
/// Only the ones that carry text: `\X2\` is UTF-16BE, which is how every writer
/// spells a non-ASCII character, and `\X\` is a single Latin-1 byte.
fn decode_control_directives(raw: &str) -> String {
    if !raw.contains('\\') {
        return raw.to_string();
    }
    let b: Vec<char> = raw.chars().collect();
    let mut out = String::with_capacity(raw.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] != '\\' {
            out.push(b[i]);
            i += 1;
            continue;
        }
        let rest: String = b[i..].iter().collect();
        if rest.starts_with("\\X2\\") {
            let mut j = i + 4;
            let mut units: Vec<u16> = Vec::new();
            while j + 3 < b.len() {
                let quad: String = b[j..j + 4].iter().collect();
                let Ok(u) = u16::from_str_radix(&quad, 16) else {
                    break;
                };
                units.push(u);
                j += 4;
            }
            out.push_str(&String::from_utf16_lossy(&units));
            let tail: String = b[j..].iter().collect();
            i = if tail.starts_with("\\X0\\") { j + 4 } else { j };
        } else if rest.starts_with("\\X\\") && i + 5 <= b.len() {
            let pair: String = b[i + 3..i + 5].iter().collect();
            match u8::from_str_radix(&pair, 16) {
                Ok(byte) => out.push(byte as char),
                Err(_) => out.push('\\'),
            }
            i += 5;
        } else {
            out.push('\\');
            i += 1;
        }
    }
    out
}

/// Parse an exchange file.
pub fn parse(input: &str) -> Result<StepFile, ParseError> {
    let mut p = Parser {
        s: input.as_bytes(),
        i: 0,
    };
    // The magic line is expected but not required: files trimmed of it are
    // common enough, and rejecting them buys nothing.
    let _ = p.eat(b"ISO-10303-21") && p.eat(b";");

    let mut file = StepFile::default();
    let mut header_id = 0;
    if p.eat(b"HEADER") {
        p.expect(b";")?;
        while let Some(e) = p.instance(&mut header_id)? {
            file.header.push(e);
        }
    }
    if p.eat(b"DATA") {
        p.expect(b";")?;
        // `DATA` may carry a parameter list, `DATA(#1)`, which is legal and
        // which nothing here needs.
        while let Some(e) = p.instance(&mut header_id)? {
            file.data.push(e);
        }
    }
    let _ = p.eat(b"END-ISO-10303-21") && p.eat(b";");
    Ok(file)
}

// --------------------------------------------------------------------------
// Writing
// --------------------------------------------------------------------------

/// Format a real the way STEP requires: always with a decimal point, never in a
/// form that reads back as an integer.
///
/// `{}` on an f64 gives `1` for 1.0, which is a *different token* — an integer
/// where the schema wants a real. Readers that enforce the distinction reject
/// the file, so this is not cosmetic.
pub fn real(x: f64) -> String {
    if !x.is_finite() {
        // No spelling for these exists in the exchange syntax. Zero is the
        // least destructive stand-in, and callers should not be producing them.
        return "0.".to_string();
    }
    let mut s = format!("{x:?}"); // shortest round-tripping form
    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
        s.push('.');
    }
    // `{:?}` gives `1.0`, which is legal but not what a STEP writer emits; the
    // canonical spelling drops the trailing zero.
    if let Some(head) = s.strip_suffix(".0") {
        s = format!("{head}.");
    }
    // `1e-7` round-trips but has no decimal point in its mantissa.
    if let Some(pos) = s.find(['e', 'E']) {
        if !s[..pos].contains('.') {
            s.insert(pos, '.');
        }
    }
    s
}

/// Escape a string for the exchange syntax.
fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for c in s.chars() {
        match c {
            '\'' => out.push_str("''"),
            '\\' => out.push_str("\\\\"),
            c if c.is_ascii() && !c.is_control() => out.push(c),
            c => {
                // Anything outside ASCII goes through the UTF-16 directive,
                // which is the only portable spelling.
                out.push_str("\\X2\\");
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    let _ = write!(out, "{unit:04X}");
                }
                out.push_str("\\X0\\");
            }
        }
    }
    out.push('\'');
    out
}

impl Value {
    fn write(&self, out: &mut String) {
        match self {
            Value::Ref(id) => {
                let _ = write!(out, "#{id}");
            }
            Value::Int(i) => {
                let _ = write!(out, "{i}");
            }
            Value::Real(x) => out.push_str(&real(*x)),
            Value::Text(s) => out.push_str(&quote(s)),
            Value::Enum(e) => {
                let _ = write!(out, ".{e}.");
            }
            Value::Binary(h) => {
                let _ = write!(out, "\"{h}\"");
            }
            Value::Omitted => out.push('$'),
            Value::Derived => out.push('*'),
            Value::List(v) => {
                out.push('(');
                for (i, x) in v.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    x.write(out);
                }
                out.push(')');
            }
            Value::Typed(name, args) => {
                out.push_str(name);
                out.push('(');
                for (i, x) in args.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    x.write(out);
                }
                out.push(')');
            }
        }
    }
}

impl Entity {
    fn write(&self, out: &mut String, with_id: bool) {
        if with_id {
            let _ = write!(out, "#{}=", self.id);
        }
        if self.name.is_empty() {
            // Complex instance: the parts are juxtaposed, not comma-separated.
            out.push('(');
            for a in &self.args {
                a.write(out);
            }
            out.push(')');
        } else {
            out.push_str(&self.name);
            out.push('(');
            for (i, a) in self.args.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                a.write(out);
            }
            out.push(')');
        }
        out.push_str(";\n");
    }
}

impl StepFile {
    /// Serialise back to an exchange file.
    ///
    /// Named for what it produces rather than as an inherent `to_string`, which
    /// would shadow the `Display` impl and silently diverge from it.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str("ISO-10303-21;\nHEADER;\n");
        for e in &self.header {
            e.write(&mut out, false);
        }
        out.push_str("ENDSEC;\nDATA;\n");
        for e in &self.data {
            e.write(&mut out, true);
        }
        out.push_str("ENDSEC;\nEND-ISO-10303-21;\n");
        out
    }
}

impl std::fmt::Display for StepFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.render())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('a part'),'2;1');
FILE_NAME('box.step','2026-08-14T00:00:00',('me'),(''),'threers','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=DIRECTION('',(0.,0.,1.));
#3=AXIS2_PLACEMENT_3D('',#1,#2,$);
#4=PLANE('',#3);
ENDSEC;
END-ISO-10303-21;
";

    #[test]
    fn a_file_parses_into_its_two_sections() {
        let f = parse(SAMPLE).unwrap();
        assert_eq!(f.header.len(), 3);
        assert_eq!(f.data.len(), 4);
        assert_eq!(f.data[3].name, "PLANE");
        assert_eq!(f.data[3].args[1], Value::Ref(3));
        assert_eq!(f.data[2].args[3], Value::Omitted);
    }

    #[test]
    fn the_written_form_parses_back_the_same() {
        let a = parse(SAMPLE).unwrap();
        let b = parse(&a.to_string()).unwrap();
        assert_eq!(a.data, b.data);
        assert_eq!(a.header, b.header);
    }

    #[test]
    fn a_real_always_keeps_its_decimal_point() {
        // `{}` on 1.0f64 gives "1", which is an *integer* token in STEP and a
        // schema violation where a real is required.
        assert_eq!(real(1.0), "1.");
        assert_eq!(real(-0.5), "-0.5");
        assert_eq!(real(0.0), "0.");
        for x in [1.0, -2.5, 1e-7, 1e21, 6.02e23, f64::MIN_POSITIVE] {
            let s = real(x);
            assert!(s.contains('.'), "{x} wrote as `{s}`");
            let Value::Real(back) = parse_value(&s) else {
                panic!("`{s}` did not read back as a real");
            };
            assert_eq!(back, x, "{x} round-tripped as {back}");
        }
    }

    /// Parse a bare value, for the tests.
    fn parse_value(s: &str) -> Value {
        let mut p = Parser {
            s: s.as_bytes(),
            i: 0,
        };
        p.value().unwrap()
    }

    #[test]
    fn reals_are_accepted_in_every_spelling_that_occurs() {
        assert_eq!(parse_value("1."), Value::Real(1.0));
        assert_eq!(parse_value("1.E-3"), Value::Real(1e-3));
        assert_eq!(parse_value("-2.5E+2"), Value::Real(-250.0));
        assert_eq!(parse_value("+3"), Value::Int(3));
        // An integer where a real belongs is common and readable.
        assert_eq!(parse_value("7").as_real(), Some(7.0));
    }

    #[test]
    fn a_quote_inside_a_string_survives_the_round_trip() {
        let f = parse("DATA;#1=PERSON('O''Brien');ENDSEC;").unwrap();
        assert_eq!(f.data[0].args[0].as_text(), Some("O'Brien"));
        let again = parse(&f.to_string()).unwrap();
        assert_eq!(again.data[0].args[0].as_text(), Some("O'Brien"));
    }

    #[test]
    fn non_ascii_goes_through_the_utf16_directive() {
        let f = parse(r"DATA;#1=P('caf\X2\00E9\X0\ \X2\03A9\X0\');ENDSEC;").unwrap();
        assert_eq!(f.data[0].args[0].as_text(), Some("café Ω"));

        // And back out again, so a name survives a load/save cycle.
        let text = f.to_string();
        assert!(text.contains(r"\X2\00E9\X0\"), "{text}");
        assert_eq!(
            parse(&text).unwrap().data[0].args[0].as_text(),
            Some("café Ω")
        );
    }

    #[test]
    fn a_complex_instance_keeps_its_parts() {
        // How AP203 spells a rational B-spline surface: several entity names
        // juxtaposed, not comma-separated.
        let f = parse(
            "DATA;#1=(BOUNDED_SURFACE()B_SPLINE_SURFACE(3,3,((#2)),.UNSPECIFIED.,.F.,.F.,.F.)\
             RATIONAL_B_SPLINE_SURFACE(((1.,1.))));ENDSEC;",
        )
        .unwrap();
        let e = &f.data[0];
        assert!(e.name.is_empty());
        assert!(e.is("B_SPLINE_SURFACE"));
        assert!(e.is("RATIONAL_B_SPLINE_SURFACE"));
        assert!(!e.is("PLANE"));
        assert_eq!(e.part("B_SPLINE_SURFACE").unwrap()[0], Value::Int(3));
        assert_eq!(parse(&f.to_string()).unwrap().data, f.data);
    }

    #[test]
    fn comments_and_layout_do_not_change_the_result() {
        let terse = parse("DATA;#1=A(1,2);ENDSEC;").unwrap();
        let spaced = parse(
            "ISO-10303-21;\nDATA; /* a comment\n spanning lines */\n\
             #1 = A ( 1 , 2 ) ;\nENDSEC;\nEND-ISO-10303-21;",
        )
        .unwrap();
        assert_eq!(terse.data, spaced.data);
    }

    #[test]
    fn a_malformed_file_says_where_it_broke() {
        let e = parse("DATA;#1=A(1,;ENDSEC;").unwrap_err();
        assert!(e.at > 0, "{e}");
        assert!(e.to_string().contains("byte"), "{e}");
    }

    #[test]
    fn header_instances_are_numbered_so_they_can_be_referred_to() {
        // They carry no `#id` in the file, but are entities all the same.
        let f = parse(SAMPLE).unwrap();
        assert_eq!(f.header[0].name, "FILE_DESCRIPTION");
        assert_eq!(f.header[1].name, "FILE_NAME");
        assert!(f.header.iter().all(|h| h.id > 0));
        // ...and they come back out without ids.
        assert!(!f.to_string().contains("#1=FILE_DESCRIPTION"));
    }

    #[test]
    fn lookup_by_name_and_by_id() {
        let f = parse(SAMPLE).unwrap();
        assert_eq!(f.all("CARTESIAN_POINT").count(), 1);
        assert_eq!(f.index()[&4].name, "PLANE");
    }

    #[test]
    fn nested_lists_keep_their_shape() {
        let f = parse("DATA;#1=A(((1.,2.),(3.,4.)));ENDSEC;").unwrap();
        let outer = f.data[0].args[0].as_list().unwrap();
        assert_eq!(outer.len(), 2);
        assert_eq!(outer[1].as_list().unwrap()[0].as_real(), Some(3.0));
    }
}
