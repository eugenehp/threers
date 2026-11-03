//! The value model a USD layer is made of, and the tokenizer that reads it.
//!
//! USD's text syntax is small but not regular: a value can be a scalar, a tuple
//! written `(1, 2, 3)`, an array written `[...]`, a dictionary, a path written
//! `</World/Mesh>`, or an asset reference written `@file.png@`. Types are
//! declared rather than inferred — `point3f[] points = [...]` — so the parser
//! never has to guess what a bracketed list of numbers is for.
//!
//! Nothing here knows what a `Mesh` is. This layer reads the *document*; the
//! mapping onto geometry lives in [`scene`](super::scene), which keeps the
//! parser honest about round-tripping things it does not understand.

use std::fmt;

/// A value as it appears in a layer.
#[derive(Debug, Clone, PartialEq)]
pub enum UsdValue {
    /// An attribute declared with no value — `float3 foo` on its own.
    None,
    /// An attribute explicitly *blocked* — `float3 foo = None`.
    ///
    /// Not the same as having no value, and the difference is the whole point:
    /// a declaration with nothing said takes whatever a weaker layer says,
    /// while a block stops the weaker opinion from coming through. Collapsing
    /// the two turns "I do not want this" into "I have no opinion".
    Block,
    Bool(bool),
    /// Held as `i128` so that both `int64` and `uint64` fit without loss —
    /// `uint64`'s upper half does not fit an `i64`, and saturating there turns
    /// a large number into a different large number with no complaint.
    Int(i128),
    Float(f64),
    /// A quoted string. Kept apart from [`Token`](Self::Token) because the two
    /// are written differently and mean different things to USD.
    String(String),
    /// A bare identifier used as a value, and the quoted form of the same.
    Token(String),
    /// `@path/to/file.png@` — a reference to another file.
    Asset(String),
    /// `</World/Mesh>` — a prim path.
    Path(String),
    /// `(1, 2, 3)`, `(1, 0, 0, 0)`, or a matrix's row.
    Tuple(Vec<UsdValue>),
    /// `[...]`.
    Array(Vec<UsdValue>),
    /// `{ string foo = "bar" }`.
    Dict(Vec<(String, UsdValue)>),
    /// A composition arc's target: `@shot.usda@</World/Set>`.
    ///
    /// Kept whole rather than as an asset beside a path, because the two halves
    /// only mean anything together — and either may be absent. No asset is an
    /// *internal* reference to elsewhere in the same layer; no prim path means
    /// the referenced layer's `defaultPrim`.
    Reference(UsdReference),
    /// An animated value: `{ 0: (0, 0, 0), 24: (10, 0, 0) }`, keyed by time
    /// code. Times are in the layer's time codes, not seconds — dividing by
    /// `timeCodesPerSecond` is the caller's job, because a layer that never
    /// states a rate is played at 24 by convention rather than by rule.
    TimeSamples(Vec<(f64, UsdValue)>),
}

/// Where a reference, payload, inherit or specialize points.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct UsdReference {
    /// The layer to pull from, empty for an internal arc.
    pub asset: String,
    /// The prim inside it, empty to mean that layer's `defaultPrim`.
    pub prim_path: String,
    /// `(offset, scale)` applied to the referenced layer's time codes, so a
    /// clip authored at its own frame 0 can start at frame 100 here.
    pub offset: f64,
    pub scale: f64,
}

impl UsdReference {
    pub fn new(asset: impl Into<String>, prim_path: impl Into<String>) -> Self {
        Self {
            asset: asset.into(),
            prim_path: prim_path.into(),
            offset: 0.0,
            scale: 1.0,
        }
    }

    /// Whether the arc points inside the layer that authored it.
    pub fn is_internal(&self) -> bool {
        self.asset.is_empty()
    }
}

impl UsdValue {
    /// The arcs in a composition field, however many were authored.
    ///
    /// A field may hold one arc or a list of them, and this flattens both to
    /// the same shape so a caller never has to ask which it got.
    pub fn references(&self) -> Vec<UsdReference> {
        match self {
            UsdValue::Reference(r) => vec![r.clone()],
            // `inherits` and `specializes` are stored as bare paths, and an
            // arc to a path with no layer is an arc into this one.
            UsdValue::Path(p) => vec![UsdReference::new("", p.clone())],
            // A layer with no prim named: its `defaultPrim`.
            UsdValue::Asset(a) => vec![UsdReference::new(a.clone(), "")],
            UsdValue::Array(items) => items.iter().flat_map(UsdValue::references).collect(),
            _ => Vec::new(),
        }
    }

    /// The samples of an animated value, if it is one.
    pub fn samples(&self) -> Option<&[(f64, UsdValue)]> {
        match self {
            UsdValue::TimeSamples(s) => Some(s),
            _ => None,
        }
    }

    /// How a value behaves between two samples.
    ///
    /// USD interpolates linearly by default and can be told to hold instead —
    /// a stage-wide choice rather than anything in the file, which is why it
    /// is asked for here rather than read. Held is what a mocap or simulation
    /// cache usually wants: every frame is a measurement, and a value halfway
    /// between two of them is an invention.
    pub fn at_time_held(&self, time: f64) -> UsdValue {
        let Some(samples) = self.samples() else {
            return self.clone();
        };
        match samples {
            [] => UsdValue::None,
            _ => {
                // The last sample at or before the time asked about.
                let mut held = &samples[0].1;
                for (at, value) in samples {
                    if *at <= time {
                        held = value;
                    } else {
                        break;
                    }
                }
                held.clone()
            }
        }
    }

    /// The value an animated attribute holds at a given time code, with
    /// neighbouring samples blended and the ends held flat.
    ///
    /// Quaternions are spherically interpolated rather than blended component
    /// by component: the componentwise result of two rotations is not a
    /// rotation, and normalising it afterwards still swings through the wrong
    /// arc at the wrong speed.
    ///
    /// A value that is not animated is simply itself, at every time.
    pub fn at_time(&self, time: f64) -> UsdValue {
        let Some(samples) = self.samples() else {
            return self.clone();
        };
        match samples {
            [] => UsdValue::None,
            [(_, only)] => only.clone(),
            _ => {
                if time <= samples[0].0 {
                    return samples[0].1.clone();
                }
                if time >= samples[samples.len() - 1].0 {
                    return samples[samples.len() - 1].1.clone();
                }
                let next = samples.iter().position(|(t, _)| *t >= time).unwrap_or(1);
                let (t0, a) = &samples[next - 1];
                let (t1, b) = &samples[next];
                let span = t1 - t0;
                // Two samples at the same time code: the later one wins, which
                // is how USD resolves a held value.
                if span <= 0.0 {
                    return b.clone();
                }
                lerp(a, b, (time - t0) / span)
            }
        }
    }

    /// The value as a number, if it is one. Ints widen; nothing else converts.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            UsdValue::Float(v) => Some(*v),
            UsdValue::Int(v) => Some(*v as f64),
            UsdValue::Bool(v) => Some(if *v { 1.0 } else { 0.0 }),
            _ => None,
        }
    }

    /// The value as text, whether it was written as a string, a token, an asset
    /// reference or a path.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            UsdValue::String(s)
            | UsdValue::Token(s)
            | UsdValue::Asset(s)
            | UsdValue::Path(s) => Some(s),
            _ => None,
        }
    }

    /// The elements of an array or a tuple. A scalar is a sequence of one, which
    /// is what lets `points = (0, 0, 0)` and `points = [(0, 0, 0)]` be read by
    /// the same code.
    pub fn items(&self) -> &[UsdValue] {
        match self {
            UsdValue::Array(v) | UsdValue::Tuple(v) => v,
            _ => std::slice::from_ref(self),
        }
    }

    /// Flatten to numbers, whatever the nesting. `[(1, 2), (3, 4)]` gives
    /// `[1, 2, 3, 4]` — which is the layout every geometry attribute wants.
    pub fn flat_f32(&self) -> Vec<f32> {
        let mut out = Vec::new();
        self.push_flat(&mut out);
        out
    }

    fn push_flat(&self, out: &mut Vec<f32>) {
        match self {
            UsdValue::Array(v) | UsdValue::Tuple(v) => {
                for item in v {
                    item.push_flat(out);
                }
            }
            other => {
                if let Some(n) = other.as_f64() {
                    out.push(n as f32);
                }
            }
        }
    }

    /// Flatten to the strings inside, for the name lists USD uses to order
    /// transform operations and to enumerate children.
    pub fn flat_tokens(&self) -> Vec<&str> {
        match self {
            UsdValue::Array(v) | UsdValue::Tuple(v) => {
                v.iter().flat_map(|i| i.flat_tokens()).collect()
            }
            other => other.as_str().into_iter().collect(),
        }
    }

    /// Flatten to integers, for index and count arrays.
    pub fn flat_u32(&self) -> Vec<u32> {
        match self {
            UsdValue::Array(v) | UsdValue::Tuple(v) => {
                v.iter().flat_map(|i| i.flat_u32()).collect()
            }
            other => other
                .as_f64()
                .filter(|n| *n >= 0.0)
                .map(|n| vec![n as u32])
                .unwrap_or_default(),
        }
    }
}

/// One lexical unit of a `.usda` document.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    /// `def`, `over`, `class`, `uniform`, `rel`, a type name, an attribute
    /// name — anything bare. Namespaced names like `primvars:st` arrive whole.
    Ident(String),
    Number(f64),
    /// Whether the number was written without a decimal point, so that
    /// `faceVertexCounts = [3, 3]` round-trips as integers rather than `3.0`.
    Integer(i128),
    String(String),
    Asset(String),
    Path(String),
    Punct(char),
}

/// Split a `.usda` document into tokens.
///
/// Errors carry a byte offset rather than a line: the parser turns that into a
/// line and column once, at the point where it has a reason to.
pub fn tokenize(src: &str) -> Result<Vec<Token>, (usize, &'static str)> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < b.len() {
        let c = b[i];
        match c {
            // Whitespace, and the commas USD allows but does not require.
            b' ' | b'\t' | b'\r' | b'\n' => i += 1,
            b'#' => {
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
            }
            // A colon only stands alone between a time code and its sample;
            // inside a name like `xformOp:translate` it is an identifier
            // character, and that branch is taken first.
            // A comma is a token rather than whitespace, and it has to be:
            // `[@a@, </P>]` is two arcs and `[@a@</P>]` is one, and nothing
            // else tells them apart.
            b'{' | b'}' | b'[' | b']' | b'(' | b')' | b'=' | b';' | b':' | b',' => {
                out.push(Token::Punct(c as char));
                i += 1;
            }
            b'<' => {
                let start = i + 1;
                let end = find(b, start, b'>').ok_or((i, "unterminated path"))?;
                out.push(Token::Path(src[start..end].to_string()));
                i = end + 1;
            }
            b'@' => {
                // `@@@...@@@` quotes an asset path containing an `@`.
                if b[i..].starts_with(b"@@@") {
                    let start = i + 3;
                    let end = find_seq(b, start, b"@@@").ok_or((i, "unterminated asset"))?;
                    out.push(Token::Asset(src[start..end].to_string()));
                    i = end + 3;
                } else {
                    let start = i + 1;
                    let end = find(b, start, b'@').ok_or((i, "unterminated asset"))?;
                    out.push(Token::Asset(src[start..end].to_string()));
                    i = end + 1;
                }
            }
            b'"' | b'\'' => {
                let (text, next) = read_string(src, i)?;
                out.push(Token::String(text));
                i = next;
            }
            _ => {
                if c == b'-' || c == b'+' || c.is_ascii_digit() || (c == b'.' && i + 1 < b.len() && b[i + 1].is_ascii_digit())
                {
                    let (tok, next) = read_number(src, i).ok_or((i, "bad number"))?;
                    out.push(tok);
                    i = next;
                } else if is_ident_start(c) {
                    let start = i;
                    while i < b.len() && is_ident(b[i]) {
                        i += 1;
                    }
                    out.push(Token::Ident(src[start..i].to_string()));
                } else {
                    return Err((i, "unexpected character"));
                }
            }
        }
    }
    Ok(out)
}

fn is_ident_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_'
}

/// Namespaced properties (`primvars:st`), dotted ones (`xformOp:transform`) and
/// the `.connect` / `.timeSamples` suffixes are all one identifier here, so the
/// parser never has to reassemble a name from pieces.
fn is_ident(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_' || c == b':' || c == b'.' || c == b'|'
}

fn find(b: &[u8], from: usize, needle: u8) -> Option<usize> {
    (from..b.len()).find(|&i| b[i] == needle)
}

fn find_seq(b: &[u8], from: usize, needle: &[u8]) -> Option<usize> {
    (from..b.len().saturating_sub(needle.len() - 1)).find(|&i| b[i..].starts_with(needle))
}

/// Read a quoted string, handling `"""` blocks and backslash escapes.
fn read_string(src: &str, at: usize) -> Result<(String, usize), (usize, &'static str)> {
    let b = src.as_bytes();
    let quote = b[at];
    let triple = [quote; 3];
    if b[at..].starts_with(&triple) {
        let start = at + 3;
        let end = find_seq(b, start, &triple).ok_or((at, "unterminated string"))?;
        return Ok((src[start..end].to_string(), end + 3));
    }
    let mut out = String::new();
    let mut i = at + 1;
    while i < b.len() {
        match b[i] {
            b'\\' if i + 1 < b.len() => {
                out.push(match b[i + 1] {
                    b'n' => '\n',
                    b't' => '\t',
                    b'r' => '\r',
                    other => other as char,
                });
                i += 2;
            }
            c if c == quote => return Ok((out, i + 1)),
            c => {
                out.push(c as char);
                i += 1;
            }
        }
    }
    Err((at, "unterminated string"))
}

/// Read a number, keeping integers integral.
///
/// `inf`, `-inf` and `nan` are spelled out in USD and are ordinary values in a
/// layer — an empty mesh's extent is written with them.
fn read_number(src: &str, at: usize) -> Option<(Token, usize)> {
    let b = src.as_bytes();
    let mut i = at;
    if b[i] == b'-' || b[i] == b'+' {
        i += 1;
    }
    let sign = if b[at] == b'-' { -1.0 } else { 1.0 };
    for word in ["inf", "nan"] {
        if src[i..].starts_with(word) {
            let v = if word == "inf" { f64::INFINITY } else { f64::NAN };
            return Some((Token::Number(sign * v), i + word.len()));
        }
    }
    let digits = i;
    let mut float = false;
    while i < b.len() {
        match b[i] {
            b'0'..=b'9' => i += 1,
            b'.' => {
                float = true;
                i += 1;
            }
            b'e' | b'E' => {
                float = true;
                i += 1;
                if i < b.len() && (b[i] == b'-' || b[i] == b'+') {
                    i += 1;
                }
            }
            _ => break,
        }
    }
    if i == digits {
        return None;
    }
    let text = &src[at..i];
    if float {
        text.parse::<f64>().ok().map(|v| (Token::Number(v), i))
    } else {
        match text.parse::<i128>() {
            Ok(v) => Some((Token::Integer(v), i)),
            // Out of even `i128`: keep it as a float rather than failing the
            // whole document over one number nothing will index with.
            Err(_) => text.parse::<f64>().ok().map(|v| (Token::Number(v), i)),
        }
    }
}

impl fmt::Display for UsdValue {
    /// Write the value back in `.usda` syntax.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UsdValue::None => Ok(()),
            UsdValue::Block => write!(f, "None"),
            UsdValue::Bool(v) => write!(f, "{}", if *v { "true" } else { "false" }),
            UsdValue::Int(v) => write!(f, "{v}"),
            UsdValue::Float(v) => write!(f, "{}", super::write::number(*v)),
            UsdValue::String(s) => write!(f, "\"{}\"", escape(s)),
            UsdValue::Token(s) => write!(f, "\"{}\"", escape(s)),
            UsdValue::Asset(s) => write!(f, "@{s}@"),
            UsdValue::Path(s) => write!(f, "<{s}>"),
            UsdValue::Reference(r) => {
                if !r.asset.is_empty() {
                    write!(f, "@{}@", r.asset)?;
                }
                if !r.prim_path.is_empty() {
                    write!(f, "<{}>", r.prim_path)?;
                }
                if r.offset != 0.0 || r.scale != 1.0 {
                    write!(
                        f,
                        " (offset = {}; scale = {})",
                        super::write::number(r.offset),
                        super::write::number(r.scale)
                    )?;
                }
                Ok(())
            }
            // The layer writer renders these with the surrounding
            // indentation; this is the fallback for a value printed alone.
            UsdValue::TimeSamples(samples) => {
                write!(f, "{{")?;
                for (time, value) in samples {
                    write!(f, " {}: {value},", super::write::number(*time))?;
                }
                write!(f, " }}")
            }
            UsdValue::Tuple(items) => {
                write!(f, "(")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{item}")?;
                }
                write!(f, ")")
            }
            UsdValue::Array(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{item}")?;
                }
                write!(f, "]")
            }
            UsdValue::Dict(entries) => {
                write!(f, "{{")?;
                for (i, (k, v)) in entries.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{k} = {v}")?;
                }
                write!(f, "}}")
            }
        }
    }
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

/// Blend two values of the same shape. Structure is taken from `a`, so a pair
/// that disagrees in length falls back to the earlier sample rather than
/// producing a value of neither shape.
///
/// A four-element tuple is treated as a quaternion and interpolated
/// spherically. That is what USD does — `GfLerp` specialises for the quaternion
/// types — and it matters: blending `(1,0,0,0)` and `(0,1,0,0)` componentwise
/// gives a half-length vector that is not a rotation at all, and a rotation
/// blended that way slows down in the middle of its arc.
fn lerp(a: &UsdValue, b: &UsdValue, t: f64) -> UsdValue {
    if let (UsdValue::Tuple(x), UsdValue::Tuple(y)) = (a, b) {
        if x.len() == 4 && y.len() == 4 {
            if let (Some(p), Some(q)) = (four(x), four(y)) {
                return UsdValue::Tuple(
                    slerp(p, q, t).iter().map(|v| UsdValue::Float(*v)).collect(),
                );
            }
        }
    }
    lerp_componentwise(a, b, t)
}

fn four(items: &[UsdValue]) -> Option<[f64; 4]> {
    let mut out = [0.0; 4];
    for (i, item) in items.iter().enumerate().take(4) {
        out[i] = item.as_f64()?;
    }
    Some(out)
}

/// Spherical interpolation, with the real part first as USD writes it.
fn slerp(a: [f64; 4], b: [f64; 4], t: f64) -> [f64; 4] {
    let mut dot = a.iter().zip(&b).map(|(x, y)| x * y).sum::<f64>();
    let mut b = b;
    // Take the shorter arc: `q` and `-q` are the same rotation, and without
    // this a pair that happens to be written with opposite signs spins most of
    // the way round rather than a little way back.
    if dot < 0.0 {
        dot = -dot;
        b = [-b[0], -b[1], -b[2], -b[3]];
    }
    // Nearly parallel: the sine below goes to zero, and a straight blend is
    // both correct to within rounding and free of the division.
    if dot > 0.9995 {
        let mut out = [0.0; 4];
        for i in 0..4 {
            out[i] = a[i] + (b[i] - a[i]) * t;
        }
        return normalise(out);
    }
    let angle = dot.clamp(-1.0, 1.0).acos();
    let (sin_angle, sin_t) = (angle.sin(), (angle * t).sin());
    let wa = ((1.0 - t) * angle).sin() / sin_angle;
    let wb = sin_t / sin_angle;
    let mut out = [0.0; 4];
    for i in 0..4 {
        out[i] = a[i] * wa + b[i] * wb;
    }
    out
}

fn normalise(q: [f64; 4]) -> [f64; 4] {
    let length = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    if length == 0.0 {
        return q;
    }
    [q[0] / length, q[1] / length, q[2] / length, q[3] / length]
}

fn lerp_componentwise(a: &UsdValue, b: &UsdValue, t: f64) -> UsdValue {
    match (a, b) {
        (UsdValue::Float(x), UsdValue::Float(y)) => UsdValue::Float(x + (y - x) * t),
        (UsdValue::Int(x), UsdValue::Int(y)) => {
            UsdValue::Float(*x as f64 + (*y as f64 - *x as f64) * t)
        }
        (UsdValue::Tuple(x), UsdValue::Tuple(y)) if x.len() == y.len() => {
            UsdValue::Tuple(x.iter().zip(y).map(|(p, q)| lerp_componentwise(p, q, t)).collect())
        }
        // An array of quaternions is interpolated element by element, and each
        // element is a quaternion — so this recurses through `lerp`.
        (UsdValue::Array(x), UsdValue::Array(y)) if x.len() == y.len() => {
            UsdValue::Array(x.iter().zip(y).map(|(p, q)| lerp(p, q, t)).collect())
        }
        // Tokens, strings, paths and mismatched shapes hold rather than blend.
        _ => a.clone(),
    }
}

#[cfg(test)]
mod interpolation {
    use super::*;

    fn samples(pairs: &[(f64, [f64; 4])]) -> UsdValue {
        UsdValue::TimeSamples(
            pairs
                .iter()
                .map(|(t, q)| {
                    (
                        *t,
                        UsdValue::Tuple(q.iter().map(|v| UsdValue::Float(*v)).collect()),
                    )
                })
                .collect(),
        )
    }

    /// A rotation halfway between two others is not the average of their
    /// components. Blending `(1,0,0,0)` and `(0,1,0,0)` that way gives a
    /// quaternion of length 0.707, which is not a rotation at all.
    #[test]
    fn quaternions_are_interpolated_spherically() {
        let value = samples(&[(0.0, [1.0, 0.0, 0.0, 0.0]), (1.0, [0.0, 1.0, 0.0, 0.0])]);
        let half = value.at_time(0.5).flat_f32();

        let length = half.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((length - 1.0).abs() < 1e-5, "not a unit quaternion: {length}");
        // Exactly halfway along the arc: both components equal.
        assert!((half[0] - half[1]).abs() < 1e-5, "{half:?}");
        assert!((half[0] - 0.5f32.sqrt()).abs() < 1e-5, "{half:?}");
    }

    /// `q` and `-q` are the same rotation, so a pair written with opposite
    /// signs must take the short way round rather than most of the way about.
    #[test]
    fn the_shorter_arc_is_taken() {
        let value = samples(&[(0.0, [1.0, 0.0, 0.0, 0.0]), (1.0, [-1.0, 0.0, 0.0, 0.0])]);
        let half = value.at_time(0.5).flat_f32();
        // The two ends are the same rotation, so every point between them is
        // too — not a tumble through the far side.
        assert!((half[0].abs() - 1.0).abs() < 1e-4, "{half:?}");
    }

    /// Nearly-equal rotations must not divide by a sine that has gone to zero.
    #[test]
    fn nearly_equal_rotations_are_finite() {
        let value = samples(&[
            (0.0, [1.0, 0.0, 0.0, 0.0]),
            (1.0, [0.999999, 0.001, 0.0, 0.0]),
        ]);
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let out = value.at_time(t).flat_f32();
            assert!(out.iter().all(|v| v.is_finite()), "at {t}: {out:?}");
        }
    }

    /// Held interpolation steps rather than blends. A cache is a set of
    /// measurements, and a value halfway between two of them is an invention.
    #[test]
    fn held_interpolation_steps() {
        let value = UsdValue::TimeSamples(vec![
            (0.0, UsdValue::Float(0.0)),
            (10.0, UsdValue::Float(100.0)),
        ]);
        assert_eq!(value.at_time(5.0).as_f64(), Some(50.0), "linear blends");
        assert_eq!(value.at_time_held(5.0).as_f64(), Some(0.0), "held steps");
        assert_eq!(value.at_time_held(9.999).as_f64(), Some(0.0));
        assert_eq!(value.at_time_held(10.0).as_f64(), Some(100.0));
        // Before the first sample, the first value — as with linear.
        assert_eq!(value.at_time_held(-1.0).as_f64(), Some(0.0));
        // A value that is not animated is itself, either way.
        assert_eq!(UsdValue::Float(7.0).at_time_held(3.0).as_f64(), Some(7.0));
    }

    /// Everything that is not a quaternion still blends component by
    /// component, including an array of them.
    #[test]
    fn other_shapes_blend_as_before() {
        let value = UsdValue::TimeSamples(vec![
            (
                0.0,
                UsdValue::Array(vec![UsdValue::Tuple(vec![
                    UsdValue::Float(0.0),
                    UsdValue::Float(0.0),
                    UsdValue::Float(0.0),
                ])]),
            ),
            (
                1.0,
                UsdValue::Array(vec![UsdValue::Tuple(vec![
                    UsdValue::Float(2.0),
                    UsdValue::Float(4.0),
                    UsdValue::Float(6.0),
                ])]),
            ),
        ]);
        assert_eq!(value.at_time(0.5).flat_f32(), vec![1.0, 2.0, 3.0]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizes_the_pieces_of_a_layer() {
        let toks = tokenize(
            r#"def Mesh "m" { point3f[] points = [(0, 0, 0)] rel mat = </a/b> asset f = @x.png@ }"#,
        )
        .unwrap();
        assert!(toks.contains(&Token::Ident("def".into())));
        assert!(toks.contains(&Token::String("m".into())));
        assert!(toks.contains(&Token::Path("/a/b".into())));
        assert!(toks.contains(&Token::Asset("x.png".into())));
        // An integer stays an integer, so index arrays round-trip.
        assert!(toks.contains(&Token::Integer(0)));
    }

    #[test]
    fn comments_are_skipped_and_commas_are_kept() {
        let toks = tokenize("# a comment\nfoo = [1, 2]\n").unwrap();
        assert_eq!(
            toks,
            vec![
                Token::Ident("foo".into()),
                Token::Punct('='),
                Token::Punct('['),
                Token::Integer(1),
                Token::Punct(','),
                Token::Integer(2),
                Token::Punct(']'),
            ]
        );
    }

    /// A comma cannot be whitespace, however tempting it looks: an asset
    /// followed by a path is one composition arc, and an asset followed by a
    /// comma and a path is two. Dropping the comma erases the difference.
    #[test]
    fn a_comma_separates_an_arc_from_the_next_one() {
        let one = tokenize("[@a.usda@</P>]").unwrap();
        let two = tokenize("[@a.usda@, </P>]").unwrap();
        assert_ne!(one, two);
        // `[`, the asset, the path, `]` — and one more for the comma.
        assert_eq!(one.len(), 4, "{one:?}");
        assert_eq!(two.len(), 5, "{two:?}");
    }

    #[test]
    fn strings_triple_quoted_and_escaped() {
        let toks = tokenize("a = \"\"\"two\nlines\"\"\" b = \"say \\\"hi\\\"\"").unwrap();
        assert!(toks.contains(&Token::String("two\nlines".into())));
        assert!(toks.contains(&Token::String("say \"hi\"".into())));
    }

    #[test]
    fn numbers_keep_their_kind() {
        let toks = tokenize("a = 3 b = 3.0 c = -1e-4 d = -inf").unwrap();
        assert!(toks.contains(&Token::Integer(3)));
        assert!(toks.contains(&Token::Number(3.0)));
        assert!(toks.contains(&Token::Number(-1e-4)));
        assert!(toks.iter().any(|t| matches!(t, Token::Number(v) if v.is_infinite() && *v < 0.0)));
    }

    #[test]
    fn namespaced_names_arrive_whole() {
        let toks = tokenize("texCoord2f[] primvars:st = []").unwrap();
        assert!(toks.contains(&Token::Ident("primvars:st".into())));
    }

    #[test]
    fn flattening_ignores_the_nesting() {
        let v = UsdValue::Array(vec![
            UsdValue::Tuple(vec![UsdValue::Float(1.0), UsdValue::Float(2.0)]),
            UsdValue::Tuple(vec![UsdValue::Float(3.0), UsdValue::Int(4)]),
        ]);
        assert_eq!(v.flat_f32(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(v.items().len(), 2);
    }

    #[test]
    fn an_asset_with_an_at_in_it_uses_the_long_quote() {
        let toks = tokenize("f = @@@odd@name.png@@@").unwrap();
        assert!(toks.contains(&Token::Asset("odd@name.png".into())));
    }
}
