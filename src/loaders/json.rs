//! Tiny hand-rolled JSON parser. Handles enough of the spec to read GLTF JSON
//! chunks: objects, arrays, strings (no `\uXXXX`), numbers, true/false/null.
//! Errors are kept terse (`&'static str`) — bring in `serde_json` for anything
//! production-grade.

use std::collections::HashMap;
use std::fmt::Write;

#[derive(Debug, Clone)]
pub enum Value {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Value>),
    Object(HashMap<String, Value>),
}

impl Value {
    pub fn as_object(&self) -> Option<&HashMap<String, Value>> {
        if let Value::Object(o) = self {
            Some(o)
        } else {
            None
        }
    }
    pub fn as_array(&self) -> Option<&Vec<Value>> {
        if let Value::Array(a) = self {
            Some(a)
        } else {
            None
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        if let Value::String(s) = self {
            Some(s.as_str())
        } else {
            None
        }
    }
    pub fn as_number(&self) -> Option<f64> {
        if let Value::Number(n) = self {
            Some(*n)
        } else {
            None
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        self.as_number().map(|n| n as u64)
    }
    pub fn as_bool(&self) -> Option<bool> {
        if let Value::Bool(b) = self {
            Some(*b)
        } else {
            None
        }
    }

    /// The named member of an object, for reaching into a nested block.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.as_object()?.get(key)
    }

    pub fn as_object_mut(&mut self) -> Option<&mut HashMap<String, Value>> {
        if let Value::Object(o) = self {
            Some(o)
        } else {
            None
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        self.as_number().map(|n| n as f32)
    }
}

impl Value {
    /// Build an object from pairs, which reads better than inserting one by one.
    pub fn object<const N: usize>(pairs: [(&str, Value); N]) -> Value {
        Value::Object(
            pairs
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
        )
    }

    pub fn number(v: impl Into<f64>) -> Value {
        Value::Number(v.into())
    }

    pub fn string(v: impl Into<String>) -> Value {
        Value::String(v.into())
    }

    /// Serialise back to JSON.
    ///
    /// Object keys are written in sorted order. A `HashMap` has no order of its
    /// own, so without this the same scene would serialise differently each
    /// run, and a format whose output changes between runs cannot be diffed or
    /// checksummed.
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            Value::Null => out.push_str("null"),
            Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Value::Number(n) => {
                // An integral float is written without its `.0`: glTF's schema
                // types many of these as integers, and a validator rejects
                // `"count": 36.0`.
                if n.fract() == 0.0 && n.abs() < 9e15 {
                    let _ = write!(out, "{}", *n as i64);
                } else {
                    let _ = write!(out, "{n}");
                }
            }
            Value::String(s) => escape(s, out),
            Value::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write(out);
                }
                out.push(']');
            }
            Value::Object(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                out.push('{');
                for (i, key) in keys.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    escape(key, out);
                    out.push(':');
                    map[*key].write(out);
                }
                out.push('}');
            }
        }
    }
}

fn escape(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

pub fn parse(src: &str) -> Result<Value, &'static str> {
    let mut p = Parser {
        bytes: src.as_bytes(),
        pos: 0,
    };
    p.skip_ws();
    let v = p.value()?;
    p.skip_ws();
    if p.pos != p.bytes.len() {
        return Err("trailing data after JSON value");
    }
    Ok(v)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }
    fn bump(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.pos += 1;
        Some(b)
    }
    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }
    fn expect(&mut self, b: u8) -> Result<(), &'static str> {
        if self.bump() == Some(b) {
            Ok(())
        } else {
            Err("unexpected byte")
        }
    }
    fn value(&mut self) -> Result<Value, &'static str> {
        self.skip_ws();
        let Some(c) = self.peek() else {
            return Err("unexpected end");
        };
        match c {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => self.string().map(Value::String),
            b't' | b'f' => self.boolean(),
            b'n' => self.null(),
            b'-' | b'0'..=b'9' => self.number(),
            _ => Err("unexpected token"),
        }
    }
    fn object(&mut self) -> Result<Value, &'static str> {
        self.expect(b'{')?;
        let mut map = HashMap::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Value::Object(map));
        }
        loop {
            self.skip_ws();
            let key = self.string()?;
            self.skip_ws();
            self.expect(b':')?;
            let v = self.value()?;
            map.insert(key, v);
            self.skip_ws();
            match self.bump() {
                Some(b',') => continue,
                Some(b'}') => break,
                _ => return Err("expected , or } in object"),
            }
        }
        Ok(Value::Object(map))
    }
    fn array(&mut self) -> Result<Value, &'static str> {
        self.expect(b'[')?;
        let mut out = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Value::Array(out));
        }
        loop {
            let v = self.value()?;
            out.push(v);
            self.skip_ws();
            match self.bump() {
                Some(b',') => continue,
                Some(b']') => break,
                _ => return Err("expected , or ] in array"),
            }
        }
        Ok(Value::Array(out))
    }
    fn string(&mut self) -> Result<String, &'static str> {
        self.expect(b'"')?;
        // Raw bytes first: the input is UTF-8, and pushing byte by byte as
        // `char`s would turn every non-ASCII character into Latin-1 mojibake.
        let mut s: Vec<u8> = Vec::new();
        let push = |s: &mut Vec<u8>, c: char| {
            let mut buf = [0u8; 4];
            s.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
        };
        loop {
            let Some(b) = self.bump() else {
                return Err("unterminated string");
            };
            match b {
                b'"' => return String::from_utf8(s).map_err(|_| "invalid UTF-8 in string"),
                b'\\' => {
                    let Some(esc) = self.bump() else {
                        return Err("bad escape");
                    };
                    match esc {
                        b'"' => s.push(b'"'),
                        b'\\' => s.push(b'\\'),
                        b'/' => s.push(b'/'),
                        b'b' => s.push(0x08),
                        b'f' => s.push(0x0c),
                        b'n' => s.push(b'\n'),
                        b'r' => s.push(b'\r'),
                        b't' => s.push(b'\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let code = if (0xD800..0xDC00).contains(&hi) {
                                // A surrogate pair: `\uD83D\uDE00`.
                                if self.bump() != Some(b'\\') || self.bump() != Some(b'u') {
                                    return Err("lone surrogate");
                                }
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err("lone surrogate");
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else {
                                hi
                            };
                            push(&mut s, char::from_u32(code).unwrap_or('\u{FFFD}'));
                        }
                        _ => return Err("bad escape"),
                    }
                }
                _ => s.push(b),
            }
        }
    }
    fn hex4(&mut self) -> Result<u32, &'static str> {
        let mut v = 0u32;
        for _ in 0..4 {
            let d = self.bump().and_then(|b| (b as char).to_digit(16)).ok_or("bad \\u escape")?;
            v = v * 16 + d;
        }
        Ok(v)
    }
    fn boolean(&mut self) -> Result<Value, &'static str> {
        if self.bytes[self.pos..].starts_with(b"true") {
            self.pos += 4;
            Ok(Value::Bool(true))
        } else if self.bytes[self.pos..].starts_with(b"false") {
            self.pos += 5;
            Ok(Value::Bool(false))
        } else {
            Err("bad bool")
        }
    }
    fn null(&mut self) -> Result<Value, &'static str> {
        if self.bytes[self.pos..].starts_with(b"null") {
            self.pos += 4;
            Ok(Value::Null)
        } else {
            Err("bad null")
        }
    }
    fn number(&mut self) -> Result<Value, &'static str> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while let Some(b) = self.peek() {
            if b.is_ascii_digit() || b == b'.' || b == b'e' || b == b'E' || b == b'+' || b == b'-' {
                self.pos += 1;
            } else {
                break;
            }
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos]).map_err(|_| "utf8")?;
        let n: f64 = s.parse().map_err(|_| "bad number")?;
        Ok(Value::Number(n))
    }
}
