//! The text form: the same node tree written with braces.
//!
//! ```text
//! Objects:  {
//!     Geometry: 140126, "Geometry::Cube", "Mesh" {
//!         Vertices: *24 {
//!             a: -1,-1,-1,-1,-1,1,...
//!         }
//!         LayerElementNormal: 0 {
//!             MappingInformationType: "ByPolygonVertex"
//!         }
//!     }
//! }
//! ```
//!
//! A name, a comma-separated property list, and optionally a braced body. An
//! array is `*count { a: … }`, which is a property rather than a child however
//! much it looks like one.

use super::tree::{Node, Property};
use super::FbxError;

pub fn parse(text: &str) -> Result<Node, FbxError> {
    let mut cursor = Cursor {
        bytes: text.as_bytes(),
        at: 0,
    };
    let children = cursor.nodes(0)?;
    Ok(Node {
        name: String::new(),
        properties: Vec::new(),
        children,
    })
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

const MAX_DEPTH: usize = 64;

impl<'a> Cursor<'a> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.at).copied()
    }

    /// Whitespace and comments. A `;` starts a comment only outside a string,
    /// and this is never called mid-string, so the check is safe here.
    fn skip(&mut self) {
        while let Some(c) = self.peek() {
            if c == b';' {
                while let Some(c) = self.peek() {
                    self.at += 1;
                    if c == b'\n' {
                        break;
                    }
                }
            } else if c.is_ascii_whitespace() {
                self.at += 1;
            } else {
                break;
            }
        }
    }

    fn nodes(&mut self, depth: usize) -> Result<Vec<Node>, FbxError> {
        if depth > MAX_DEPTH {
            return Err(FbxError::Corrupt);
        }
        let mut out = Vec::new();
        loop {
            self.skip();
            match self.peek() {
                None => return Ok(out),
                // The end of the body this list belongs to.
                Some(b'}') => {
                    self.at += 1;
                    return Ok(out);
                }
                _ => {}
            }
            let Some(node) = self.node(depth)? else {
                return Ok(out);
            };
            out.push(node);
        }
    }

    fn node(&mut self, depth: usize) -> Result<Option<Node>, FbxError> {
        let start = self.at;
        // The name runs to the colon.
        while let Some(c) = self.peek() {
            if c == b':' || c == b'\n' || c == b'{' || c == b'}' {
                break;
            }
            self.at += 1;
        }
        if self.peek() != Some(b':') {
            // Not a node — skip the line rather than spin on it.
            if self.at == start {
                self.at += 1;
            }
            return Ok(None);
        }
        let name = String::from_utf8_lossy(&self.bytes[start..self.at])
            .trim()
            .to_string();
        self.at += 1;

        let mut properties = Vec::new();
        let mut children = Vec::new();
        loop {
            // Only spaces and tabs: a newline ends the property list.
            while matches!(self.peek(), Some(b' ') | Some(b'\t') | Some(b'\r')) {
                self.at += 1;
            }
            match self.peek() {
                Some(b'{') => {
                    self.at += 1;
                    children = self.nodes(depth + 1)?;
                    break;
                }
                Some(b'\n') | None => break,
                Some(b',') => {
                    self.at += 1;
                    continue;
                }
                Some(b'*') => {
                    properties.push(self.array()?);
                    continue;
                }
                Some(b'"') => {
                    properties.push(Property::String(self.string()));
                    continue;
                }
                _ => {
                    if let Some(scalar) = self.scalar() {
                        properties.push(scalar);
                        continue;
                    }
                    break;
                }
            }
        }
        Ok(Some(Node {
            name,
            properties,
            children,
        }))
    }

    fn string(&mut self) -> String {
        self.at += 1; // the opening quote
        let start = self.at;
        while let Some(c) = self.peek() {
            if c == b'"' {
                break;
            }
            self.at += 1;
        }
        let s = String::from_utf8_lossy(&self.bytes[start..self.at]).into_owned();
        if self.peek() == Some(b'"') {
            self.at += 1;
        }
        s
    }

    /// `*count { a: 1,2,3 }` — the count is advisory and the values are taken
    /// as they come, because a writer that disagrees with its own count should
    /// not cost you the array.
    fn array(&mut self) -> Result<Property, FbxError> {
        self.at += 1; // the `*`
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.at += 1;
        }
        self.skip();
        if self.peek() != Some(b'{') {
            return Ok(Property::Doubles(Vec::new()));
        }
        self.at += 1;
        self.skip();
        // The body is `a: …`; tolerate its absence for an empty array.
        if self.peek() == Some(b'a') {
            self.at += 1;
            if self.peek() == Some(b':') {
                self.at += 1;
            }
        }
        let start = self.at;
        let mut depth = 1usize;
        while let Some(c) = self.peek() {
            if c == b'{' {
                depth += 1;
            } else if c == b'}' {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            self.at += 1;
        }
        let body = String::from_utf8_lossy(&self.bytes[start..self.at]);
        if self.peek() == Some(b'}') {
            self.at += 1;
        }
        // Integers and reals are not distinguished in the text form, so this
        // keeps both: whether an array is indices or positions is decided by
        // the node it sits under, not by how the numbers look.
        let mut doubles = Vec::new();
        let mut ints = Vec::new();
        let mut all_integral = true;
        for token in body.split([',', '\n', '\t', '\r', ' ']) {
            let token = token.trim();
            if token.is_empty() {
                continue;
            }
            match token.parse::<f64>() {
                Ok(v) => {
                    doubles.push(v);
                    if v.fract() == 0.0 && v.abs() < i32::MAX as f64 {
                        ints.push(v as i32);
                    } else {
                        all_integral = false;
                    }
                }
                Err(_) => all_integral = false,
            }
        }
        Ok(if all_integral && !ints.is_empty() {
            Property::Ints(ints)
        } else {
            Property::Doubles(doubles)
        })
    }

    fn scalar(&mut self) -> Option<Property> {
        let start = self.at;
        while let Some(c) = self.peek() {
            if c == b',' || c == b'\n' || c == b'{' || c == b'}' {
                break;
            }
            self.at += 1;
        }
        let token = String::from_utf8_lossy(&self.bytes[start..self.at]);
        let token = token.trim();
        if token.is_empty() {
            return None;
        }
        if let Ok(v) = token.parse::<i64>() {
            return Some(Property::Long(v));
        }
        if let Ok(v) = token.parse::<f64>() {
            return Some(Property::Double(v));
        }
        match token {
            "T" | "Y" | "true" => Some(Property::Bool(true)),
            "F" | "N" | "false" => Some(Property::Bool(false)),
            other => Some(Property::String(other.to_string())),
        }
    }
}
