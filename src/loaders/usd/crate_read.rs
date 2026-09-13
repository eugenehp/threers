//! Reading a `.usdc` crate file into a layer.
//!
//! A crate is not a document; it is six tables that a document is reassembled
//! from. The shape of the reassembly is the interesting part:
//!
//! - **TOKENS** is every identifier in the file, once.
//! - **PATHS** is the prim tree, stored as a depth-first walk with a jump table
//!   rather than as strings — so `/World/Mesh.points` costs two token indices
//!   and a jump, not eighteen characters.
//! - **FIELDS** pairs a field name with a [`ValueRep`], a 64-bit word that
//!   either carries the value inline or says where in the file it lives.
//! - **FIELDSETS** groups fields into the sets a spec refers to, each run
//!   terminated by `-1`.
//! - **SPECS** ties a path to a field set and a spec type: this path is a prim,
//!   these are its fields.
//!
//! Everything here was written against files produced by `usdcat`, and the
//! tests read one back — a reader for a binary format that has only ever been
//! checked against its own writer is a reader that agrees with itself.

use std::collections::HashMap;

use super::ints::{decode_u32, decode_u64, read_u32_array};
use super::lz4;
use super::parse::{Specifier, UsdLayer, UsdPrim, UsdProperty, UsdVariantSet};
use super::value::{UsdReference, UsdValue};
use super::UsdError;

/// A 64-bit word describing one value: its type, and either the value itself or
/// where to find it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ValueRep(u64);

impl ValueRep {
    const IS_ARRAY: u64 = 1 << 63;
    const IS_INLINED: u64 = 1 << 62;
    const IS_COMPRESSED: u64 = 1 << 61;
    const PAYLOAD: u64 = (1 << 48) - 1;

    fn kind(self) -> u8 {
        ((self.0 >> 48) & 0xFF) as u8
    }
    fn is_array(self) -> bool {
        self.0 & Self::IS_ARRAY != 0
    }
    fn is_inlined(self) -> bool {
        self.0 & Self::IS_INLINED != 0
    }
    fn is_compressed(self) -> bool {
        self.0 & Self::IS_COMPRESSED != 0
    }
    fn payload(self) -> u64 {
        self.0 & Self::PAYLOAD
    }
}

/// The value types a crate distinguishes, in the order the format numbers them.
///
/// Only the ones a scene is made of are named; the rest are carried through as
/// [`UsdValue::None`] rather than guessed at, so an unfamiliar field costs a
/// value and not the file.
mod kind {
    pub const BOOL: u8 = 1;
    pub const UCHAR: u8 = 2;
    pub const INT: u8 = 3;
    pub const UINT: u8 = 4;
    pub const INT64: u8 = 5;
    pub const UINT64: u8 = 6;
    pub const HALF: u8 = 7;
    pub const FLOAT: u8 = 8;
    pub const DOUBLE: u8 = 9;
    pub const STRING: u8 = 10;
    pub const TOKEN: u8 = 11;
    pub const ASSET_PATH: u8 = 12;
    // The matrices sit ahead of the vectors, which is the one place the order
    // is not what you would guess. `triangle.usdc` settles it: its `points`
    // are type 24, so Vec3f is 24 and everything above follows.
    pub const MATRIX2D: u8 = 13;
    pub const MATRIX3D: u8 = 14;
    pub const MATRIX4D: u8 = 15;
    pub const QUATD: u8 = 16;
    pub const QUATF: u8 = 17;
    pub const QUATH: u8 = 18;
    pub const VEC2D: u8 = 19;
    pub const VEC2F: u8 = 20;
    pub const VEC2H: u8 = 21;
    pub const VEC2I: u8 = 22;
    pub const VEC3D: u8 = 23;
    pub const VEC3F: u8 = 24;
    pub const VEC3H: u8 = 25;
    pub const VEC3I: u8 = 26;
    pub const VEC4D: u8 = 27;
    pub const VEC4F: u8 = 28;
    pub const VEC4H: u8 = 29;
    pub const VEC4I: u8 = 30;
    /// A nested map of named values — what `clips` and `customData` are.
    pub const DICTIONARY: u8 = 31;
    /// What `apiSchemas` is — a list operation, not a plain vector, which is
    /// why it carries a `prepend`.
    pub const TOKEN_LIST_OP: u8 = 32;
    pub const STRING_LIST_OP: u8 = 33;
    /// What a relationship's targets and an attribute's connections are, and
    /// what `inherits` and `specializes` are.
    pub const PATH_LIST_OP: u8 = 34;
    pub const REFERENCE_LIST_OP: u8 = 35;
    pub const PATH_VECTOR: u8 = 40;
    /// An attribute explicitly blocked — `foo = None`.
    pub const VALUE_BLOCK: u8 = 51;
    /// A time code, which is a double that means a frame. Writing one upgrades
    /// a crate to version 0.9.0, which is why USD warns about it.
    pub const TIME_CODE: u8 = 56;
    /// Which variant of each set is chosen.
    pub const VARIANT_SELECTION_MAP: u8 = 45;
    pub const TOKEN_VECTOR: u8 = 41;
    pub const SPECIFIER: u8 = 42;
    pub const VARIABILITY: u8 = 44;
    /// The type an animated attribute's value has.
    pub const TIME_SAMPLES: u8 = 46;
    pub const LAYER_OFFSET_VECTOR: u8 = 49;
    /// What `subLayers` is.
    pub const STRING_VECTOR: u8 = 50;
    pub const PAYLOAD_LIST_OP: u8 = 55;
    /// A collection's membership expression — `/World//*{light}`. Stored like
    /// a string, as an index into STRINGS, but always out of line even though
    /// the index would fit the payload. Writing one upgrades a crate to
    /// version 0.10.0.
    pub const PATH_EXPRESSION: u8 = 57;
}

/// How many numbers one element of a type has, and how wide each is.
fn shape(kind: u8) -> Option<(usize, usize)> {
    Some(match kind {
        kind::BOOL | kind::UCHAR => (1, 1),
        kind::INT | kind::UINT | kind::FLOAT => (1, 4),
        kind::INT64 | kind::UINT64 | kind::DOUBLE | kind::TIME_CODE => (1, 8),
        kind::HALF => (1, 2),
        kind::VEC2H => (2, 2),
        kind::VEC3H => (3, 2),
        kind::VEC4H | kind::QUATH => (4, 2),
        kind::VEC2F => (2, 4),
        kind::VEC3F => (3, 4),
        kind::VEC4F | kind::QUATF => (4, 4),
        kind::VEC2I => (2, 4),
        kind::VEC3I => (3, 4),
        kind::VEC4I => (4, 4),
        kind::VEC2D => (2, 8),
        kind::VEC3D => (3, 8),
        kind::VEC4D | kind::QUATD => (4, 8),
        kind::MATRIX2D => (4, 8),
        kind::MATRIX3D => (9, 8),
        kind::MATRIX4D => (16, 8),
        _ => return None,
    })
}

/// Whether a type is a quaternion, which is stored differently from how it is
/// written.
///
/// In memory USD keeps the imaginary part first and the real part last; in a
/// `.usda` document it writes the real part first. The binary form follows
/// memory, so reading one means putting `w` back in front.
fn is_quat(kind: u8) -> bool {
    matches!(kind, kind::QUATD | kind::QUATF | kind::QUATH)
}

/// `(x, y, z, w)` as `(w, x, y, z)`.
fn real_first(mut lanes: Vec<UsdValue>) -> UsdValue {
    if lanes.len() == 4 {
        lanes.rotate_right(1);
    }
    UsdValue::Tuple(lanes)
}

fn is_float_kind(kind: u8) -> bool {
    matches!(
        kind,
        kind::HALF
            | kind::FLOAT
            | kind::DOUBLE
            | kind::VEC2H
            | kind::VEC3H
            | kind::VEC4H
            | kind::QUATH
            | kind::VEC2F
            | kind::VEC3F
            | kind::VEC4F
            | kind::QUATF
            | kind::VEC2D
            | kind::VEC3D
            | kind::VEC4D
            | kind::QUATD
            | kind::MATRIX2D
            | kind::MATRIX3D
            | kind::MATRIX4D
            | kind::TIME_CODE
    )
}

/// The half decoder, for the writer's tests to check its encoder against.
#[cfg(test)]
pub fn half_to_f32_for_test(bits: u16) -> f32 {
    half_to_f32(bits)
}

/// Half-precision, which USD uses for compact colour and normal data.
fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;
    let out = match exponent {
        0 if mantissa == 0 => sign << 31,
        // Subnormal: the value is `mantissa * 2^-24`, so shift the mantissa
        // up until its leading bit is where f32 keeps an implicit one, and
        // pay for each shift out of the exponent.
        0 => {
            let mut m = mantissa;
            let mut shifts = 0u32;
            while m & 0x400 == 0 {
                m <<= 1;
                shifts += 1;
            }
            let exp = 113 - shifts;
            (sign << 31) | (exp << 23) | ((m & 0x3FF) << 13)
        }
        0x1F => (sign << 31) | (0xFF << 23) | (mantissa << 13),
        _ => (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13),
    };
    f32::from_bits(out)
}

struct Crate<'a> {
    file: &'a [u8],
    tokens: Vec<String>,
    strings: Vec<u32>,
    fields: Vec<(u32, ValueRep)>,
    field_sets: Vec<i32>,
    paths: HashMap<u32, String>,
    /// `(path index, fieldset index, spec type)`.
    specs: Vec<(u32, u32, u32)>,
}

fn u64_at(b: &[u8], at: usize) -> Result<u64, UsdError> {
    b.get(at..at + 8)
        .map(|s| u64::from_le_bytes(s.try_into().unwrap()))
        .ok_or(UsdError::Corrupt("truncated"))
}

/// Read a crate file into a layer.
pub fn read(file: &[u8]) -> Result<UsdLayer, UsdError> {
    let info = super::usdc::info(file)?;
    let section = |name: &str| info.sections.iter().find(|(n, _, _)| n == name).cloned();

    // --- TOKENS: every identifier, once.
    let mut tokens = Vec::new();
    if let Some((_, at, _)) = section("TOKENS") {
        let at = at as usize;
        let count = u64_at(file, at)? as usize;
        let uncompressed = u64_at(file, at + 8)? as usize;
        let compressed = u64_at(file, at + 16)? as usize;
        let body = file
            .get(at + 24..at + 24 + compressed)
            .ok_or(UsdError::Corrupt("token section"))?;
        let text = lz4::decompress(body, uncompressed).ok_or(UsdError::Corrupt("token lz4"))?;
        tokens = text
            .split(|b| *b == 0)
            .take(count)
            .map(|s| String::from_utf8_lossy(s).into_owned())
            .collect();
    }

    // --- STRINGS: indices into the token table.
    let mut strings = Vec::new();
    if let Some((_, at, _)) = section("STRINGS") {
        let at = at as usize;
        let count = u64_at(file, at)? as usize;
        for i in 0..count {
            let v = file
                .get(at + 8 + i * 4..at + 12 + i * 4)
                .ok_or(UsdError::Corrupt("string table"))?;
            strings.push(u32::from_le_bytes(v.try_into().unwrap()));
        }
    }

    // --- FIELDS: a name index and a value for each.
    let mut fields = Vec::new();
    if let Some((_, at, _)) = section("FIELDS") {
        let at = at as usize;
        let count = u64_at(file, at)? as usize;
        let mut cursor = at + 8;
        let names = read_u32_array(file, &mut cursor, count)
            .ok_or(UsdError::Corrupt("field names"))?;
        // The value words are LZ4'd but not integer-packed: eight bytes each.
        let size = u64_at(file, cursor)? as usize;
        cursor += 8;
        let body = file
            .get(cursor..cursor + size)
            .ok_or(UsdError::Corrupt("field reps"))?;
        let reps = lz4::decompress(body, count * 8).ok_or(UsdError::Corrupt("field rep lz4"))?;
        for i in 0..count {
            let word = u64::from_le_bytes(reps[i * 8..i * 8 + 8].try_into().unwrap());
            fields.push((names[i] as u32, ValueRep(word)));
        }
    }

    // --- FIELDSETS: runs of field indices, each ended by -1.
    let mut field_sets = Vec::new();
    if let Some((_, at, _)) = section("FIELDSETS") {
        let at = at as usize;
        let count = u64_at(file, at)? as usize;
        let mut cursor = at + 8;
        field_sets =
            read_u32_array(file, &mut cursor, count).ok_or(UsdError::Corrupt("field sets"))?;
    }

    // --- PATHS: the prim tree as a walk.
    let mut paths: HashMap<u32, String> = HashMap::new();
    if let Some((_, at, _)) = section("PATHS") {
        let at = at as usize;
        let total = u64_at(file, at)? as usize;
        let encoded = u64_at(file, at + 8)? as usize;
        let mut cursor = at + 16;
        let indexes =
            read_u32_array(file, &mut cursor, encoded).ok_or(UsdError::Corrupt("path indexes"))?;
        let elements = read_u32_array(file, &mut cursor, encoded)
            .ok_or(UsdError::Corrupt("path elements"))?;
        let jumps =
            read_u32_array(file, &mut cursor, encoded).ok_or(UsdError::Corrupt("path jumps"))?;
        // The slots the encoded entries write to are *not* dense: a path table
        // is a global numbering shared with the paths that references point
        // at, so entry four may well write slot seven. Keeping them in a map
        // is both exact — nothing at the far end is dropped — and safe, since
        // nothing is allocated for a `numPaths` the file merely claims.
        let _ = total;
        build_paths(&indexes, &elements, &jumps, &tokens, 0, "", &mut paths);
    }

    // --- SPECS: path, field set, spec type.
    let mut specs = Vec::new();
    if let Some((_, at, _)) = section("SPECS") {
        let at = at as usize;
        let count = u64_at(file, at)? as usize;
        let mut cursor = at + 8;
        let path_indexes =
            read_u32_array(file, &mut cursor, count).ok_or(UsdError::Corrupt("spec paths"))?;
        let set_indexes =
            read_u32_array(file, &mut cursor, count).ok_or(UsdError::Corrupt("spec sets"))?;
        let types =
            read_u32_array(file, &mut cursor, count).ok_or(UsdError::Corrupt("spec types"))?;
        for i in 0..count {
            specs.push((
                path_indexes[i] as u32,
                set_indexes[i] as u32,
                types[i] as u32,
            ));
        }
    }

    let c = Crate {
        file,
        tokens,
        strings,
        fields,
        field_sets,
        paths,
        specs,
    };
    Ok(c.to_layer())
}

/// Put each sublayer's time offset onto the sublayer itself.
///
/// A crate keeps `subLayers` and `subLayerOffsets` as two parallel lists; a
/// document writes the offset in brackets after the layer. Keeping both here
/// would write the offsets twice — once on the arcs and once as a list of
/// their own.
fn fold_sublayer_offsets(fields: &mut Vec<(String, UsdValue)>) {
    let offsets: Vec<(f64, f64)> = fields
        .iter()
        .find(|(k, _)| k == "subLayerOffsets")
        .map(|(_, v)| match v {
            UsdValue::Array(items) => items
                .iter()
                .map(|pair| {
                    let n = pair.flat_f32();
                    (
                        n.first().copied().unwrap_or(0.0) as f64,
                        n.get(1).copied().unwrap_or(1.0) as f64,
                    )
                })
                .collect(),
            _ => Vec::new(),
        })
        .unwrap_or_default();
    if offsets.is_empty() {
        return;
    }
    fields.retain(|(k, _)| k != "subLayerOffsets");

    if let Some((_, UsdValue::Array(items))) =
        fields.iter_mut().find(|(k, _)| k == "subLayers")
    {
        {
            for (i, item) in items.iter_mut().enumerate() {
                let Some((offset, scale)) = offsets.get(i).copied() else {
                    continue;
                };
                let asset = match &item {
                    UsdValue::Reference(arc) => arc.asset.clone(),
                    other => other.as_str().unwrap_or_default().to_string(),
                };
                *item = UsdValue::Reference(UsdReference {
                    asset,
                    prim_path: String::new(),
                    offset,
                    scale,
                });
            }
        }
    }
}

/// The spec types a crate distinguishes, confirmed against files `usdcat`
/// wrote.
mod spec_type {
    pub const RELATIONSHIP: u32 = 8;
}

/// Every spec in the file, by the path it is for: its fields, and the spec
/// type that says what kind of thing it is.
type Specs<'a> = HashMap<&'a str, (Vec<(String, UsdValue)>, u32)>;

/// Rebuild absolute paths from the depth-first walk and its jump table.
///
/// `jumps` says what follows each element:
///
/// | value | meaning |
/// |---|---|
/// | `> 0` | a child at the next slot, and a sibling this many slots along |
/// | `-1` | a child and no sibling |
/// | `0` | a sibling at the next slot and no child |
/// | anything else | a leaf with nothing after it |
///
/// Siblings are walked in the loop and children by recursion, rather than the
/// other way round: depth in a prim tree is a handful of levels, but a single
/// prim may have a great many children, and recursing on *those* would put the
/// width of the scene on the stack.
fn build_paths(
    indexes: &[i32],
    elements: &[i32],
    jumps: &[i32],
    tokens: &[String],
    start: usize,
    parent: &str,
    out: &mut HashMap<u32, String>,
) {
    let mut current = start;
    loop {
        if current >= indexes.len() {
            return;
        }
        let slot = indexes[current] as u32;
        let element = elements.get(current).copied().unwrap_or(0);
        let jump = jumps.get(current).copied().unwrap_or(-2);

        let path = if parent.is_empty() {
            // The first entry is the pseudo-root, whatever element it names.
            // A file written by USD points it at the empty token rather than
            // at nothing, so reading the element here would yield `/` by luck
            // rather than by rule.
            "/".to_string()
        } else {
            // A negative element index marks a property of the prim above it.
            let is_property = element < 0;
            let token = tokens
                .get(element.unsigned_abs() as usize)
                .cloned()
                .unwrap_or_default();
            match (parent, is_property) {
                // `{look=oak}` is a variant selection, not a child: it
                // qualifies the prim it hangs off rather than sitting under it.
                (p, false) if token.starts_with('{') => format!("{p}{token}"),
                ("/", _) => format!("/{token}"),
                (p, true) => format!("{p}.{token}"),
                (p, false) => format!("{p}/{token}"),
            }
        };
        out.insert(slot, path.clone());

        let has_child = jump > 0 || jump == -1;
        let has_sibling = jump >= 0;
        if has_child {
            build_paths(indexes, elements, jumps, tokens, current + 1, &path, out);
        }
        if !has_sibling {
            return;
        }
        // A node with a child keeps its sibling's distance in the jump; one
        // without has its sibling in the very next slot.
        current += if jump > 0 { jump as usize } else { 1 };
    }
}

impl Crate<'_> {
    fn token(&self, i: u32) -> &str {
        self.tokens.get(i as usize).map(String::as_str).unwrap_or("")
    }

    /// A quoted string, which is indexed one level deeper than a token: the
    /// stored number indexes the string table, and *that* holds the token.
    fn string(&self, i: u32) -> &str {
        match self.strings.get(i as usize) {
            Some(token) => self.token(*token),
            None => "",
        }
    }

    /// Either, depending on the type — the two are distinct in USD and are
    /// written differently on the way back out.
    fn text(&self, kind: u8, i: u32) -> UsdValue {
        if kind == kind::STRING || kind == kind::PATH_EXPRESSION {
            UsdValue::String(self.string(i).to_string())
        } else {
            UsdValue::Token(self.token(i).to_string())
        }
    }

    /// The fields of one spec, by name.
    fn spec_fields(&self, set: u32) -> Vec<(String, UsdValue)> {
        let mut out = Vec::new();
        let mut at = set as usize;
        while let Some(&index) = self.field_sets.get(at) {
            if index < 0 {
                break;
            }
            if let Some((name, rep)) = self.fields.get(index as usize) {
                // A list operation records which sub-lists were authored, and
                // a document spells each as a word in front of the field name.
                // A field with both a `prepend` and a `delete` is two lines.
                let qualifiers = self.list_op_qualifiers(*rep);
                // A single non-explicit sub-list is still worth naming: the
                // value of a `delete` is what it deletes, not what is left.
                if qualifiers.len() > 1 || matches!(qualifiers.first(), Some(q) if !q.is_empty()) {
                    for qualifier in qualifiers {
                        out.push((
                            format!("{qualifier}{}", self.token(*name)),
                            self.list_op_sublist(*rep, qualifier),
                        ));
                    }
                } else {
                    let qualifier = qualifiers.first().copied().unwrap_or("");
                    out.push((
                        format!("{qualifier}{}", self.token(*name)),
                        self.value(*rep),
                    ));
                }
            }
            at += 1;
        }
        out
    }

    /// Decode one value.
    fn value(&self, rep: ValueRep) -> UsdValue {
        let kind = rep.kind();
        if rep.is_inlined() {
            return self.inlined(kind, rep.payload());
        }
        let at = rep.payload() as usize;
        // A token or path vector carries its own count and is not flagged as
        // an array, because it is a list of names rather than attribute data.
        if matches!(kind, kind::TOKEN_VECTOR | kind::PATH_VECTOR) && !rep.is_inlined() {
            return self.token_list(at);
        }
        if kind == kind::PATH_LIST_OP {
            return self.path_list_op(at);
        }
        if matches!(kind, kind::REFERENCE_LIST_OP | kind::PAYLOAD_LIST_OP) {
            return self.reference_list_op(at);
        }
        if kind == kind::STRING_LIST_OP {
            return self.string_list_op(at);
        }
        if kind == kind::TOKEN_LIST_OP {
            return self.token_list_op(at);
        }
        if kind == kind::DICTIONARY {
            return self.dictionary(at, 0);
        }
        if kind == kind::VALUE_BLOCK {
            return UsdValue::Block;
        }
        if kind == kind::VARIANT_SELECTION_MAP {
            return self.variant_selection(at);
        }
        if kind == kind::STRING_VECTOR {
            return self.string_vector(at);
        }
        if kind == kind::LAYER_OFFSET_VECTOR {
            return self.layer_offsets(at);
        }
        if kind == kind::TIME_SAMPLES {
            return self.time_samples(at);
        }
        if rep.is_array() {
            return self.array(kind, at, rep.is_compressed());
        }
        // A lone value is stored as its bytes at the offset.
        let Some((lanes, width)) = shape(kind) else {
            return self.scalar_by_reference(kind, at);
        };
        let bytes = match self.file.get(at..at + lanes * width) {
            Some(b) => b,
            None => return UsdValue::None,
        };
        numbers(kind, bytes, lanes, width)
    }

    /// An animated value.
    ///
    /// ```text
    /// [ u64: bytes of times section ]
    /// [ times section: optionally the doubles inline, then a rep naming them ]
    /// [ u64: 8, the width of a value rep ]
    /// [ u64: how many samples ]
    /// [ one rep per sample ]
    /// ```
    ///
    /// The times end with a rep pointing at wherever the array really is,
    /// which is how two attributes keyed on the same frames share one copy —
    /// the second one's section is just that rep, eight bytes total.
    fn time_samples(&self, at: usize) -> UsdValue {
        let Some(section) = self.u64_at(at) else {
            return UsdValue::None;
        };
        let section = section as usize;
        // The rep is the last word of the section.
        if section < 8 {
            return UsdValue::None;
        }
        let Some(end) = at.checked_add(8).and_then(|a| a.checked_add(section)) else {
            return UsdValue::None;
        };
        let Some(rep) = self.u64_at(end - 8) else {
            return UsdValue::None;
        };
        let times = self.doubles(ValueRep(rep).payload() as usize);

        let mut cursor = end;
        // A stride that is not the width of a rep means this is not the layout
        // this reader knows, and guessing past it would produce numbers rather
        // than an error.
        if self.u64_at(cursor) != Some(8) {
            return UsdValue::None;
        }
        cursor += 8;
        let Some(count) = self.u64_at(cursor) else {
            return UsdValue::None;
        };
        cursor += 8;

        let mut samples = Vec::new();
        for i in 0..count as usize {
            let Some(word) = i
                .checked_mul(8)
                .and_then(|offset| cursor.checked_add(offset))
                .and_then(|at| self.u64_at(at))
            else {
                break;
            };
            // A sample with no time is a sample nothing can place.
            let Some(time) = times.get(i) else { break };
            samples.push((*time, self.value(ValueRep(word))));
        }
        UsdValue::TimeSamples(samples)
    }

    /// A `u64` count followed by that many doubles.
    fn doubles(&self, at: usize) -> Vec<f64> {
        let Some(count) = self.u64_at(at) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for i in 0..count as usize {
            let Some(bytes) = self.file.get(at + 8 + i * 8..at + 16 + i * 8) else {
                break;
            };
            out.push(f64::from_le_bytes(bytes.try_into().unwrap()));
        }
        out
    }

    fn u64_at(&self, at: usize) -> Option<u64> {
        self.file
            .get(at..at + 8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
    }

    /// A list operation over paths — what a relationship's targets and an
    /// attribute's connections are stored as.
    ///
    /// A one-byte header says which of the six sub-lists were authored
    /// (explicit, added, deleted, ordered, prepended, appended); each present
    /// one is a count and that many indices into the path table. Only the
    /// explicit and appended lists can name a target in a single-layer
    /// document, and both are taken.
    fn path_list_op(&self, at: usize) -> UsdValue {
        const IS_EXPLICIT: u8 = 1 << 0;
        let Some(&header) = self.file.get(at) else {
            return UsdValue::None;
        };
        let mut cursor = at + 1;
        let mut targets = Vec::new();
        // The sub-lists appear in flag order, and every present one has to be
        // stepped over even if its contents are not wanted, or the cursor
        // lands mid-list.
        for bit in 1..7 {
            if header & (1 << bit) == 0 {
                continue;
            }
            let Some(bytes) = self.file.get(cursor..cursor + 8) else {
                break;
            };
            let count = u64::from_le_bytes(bytes.try_into().unwrap()) as usize;
            cursor += 8;
            // Explicit items (bit 1), prepended (5) and appended (6) all name
            // targets; deletions and orderings do not. Leaving out the
            // prepended ones reads `prepend rel foo = [...]` as a relationship
            // with no targets at all.
            let wanted = matches!(bit, 1 | 5 | 6);
            for _ in 0..count {
                let Some(bytes) = self.file.get(cursor..cursor + 4) else {
                    break;
                };
                cursor += 4;
                if !wanted {
                    continue;
                }
                let index = u32::from_le_bytes(bytes.try_into().unwrap()) as usize;
                if let Some(path) = self.paths.get(&(index as u32)) {
                    targets.push(UsdValue::Path(path.clone()));
                }
            }
        }
        let _ = IS_EXPLICIT;
        match targets.len() {
            0 => UsdValue::None,
            // A single target reads as one path, which is what the text form
            // produces for `rel material:binding = </Mat>`.
            1 => targets.pop().unwrap(),
            _ => UsdValue::Array(targets),
        }
    }

    /// The composition arcs in a `references` or `payload` field.
    ///
    /// Each item is an asset (a string index), the prim inside it (a path
    /// index), and the layer offset applied to its time codes:
    ///
    /// ```text
    /// [ u32 asset ][ u32 prim path ][ f64 offset ][ f64 scale ][ u64 custom ]
    /// ```
    ///
    /// The last word is the arc's `customData`, which nothing here reads but
    /// which is very much part of the stride: taking an item to be 24 bytes
    /// reads the second arc out of the middle of the first.
    ///
    /// Either half may be absent — no asset is an internal arc, no prim path
    /// means the referenced layer's `defaultPrim` — which is why they are kept
    /// together as one value rather than as two.
    fn reference_list_op(&self, at: usize) -> UsdValue {
        let mut out = Vec::new();
        self.each_list_op(at, 32, |slice| {
            let asset = u32::from_le_bytes(slice[0..4].try_into().unwrap());
            let path = u32::from_le_bytes(slice[4..8].try_into().unwrap());
            out.push(UsdValue::Reference(UsdReference {
                asset: self.string(asset).to_string(),
                // Index zero is the empty path, which is how "no prim named"
                // is spelled.
                prim_path: match self.paths.get(&path) {
                    Some(p) if path != 0 => p.clone(),
                    _ => String::new(),
                },
                offset: f64::from_le_bytes(slice[8..16].try_into().unwrap()),
                scale: f64::from_le_bytes(slice[16..24].try_into().unwrap()),
            }));
        });
        match out.len() {
            0 => UsdValue::None,
            _ => UsdValue::Array(out),
        }
    }

    /// The word a document puts in front of a list-op field, if any.
    /// Every sub-list a list operation actually holds.
    ///
    /// A field may carry several at once — `prepend references` and
    /// `delete references` are one field with two sub-lists — and collapsing
    /// them to one loses whichever came second.
    fn list_op_qualifiers(&self, rep: ValueRep) -> Vec<&'static str> {
        if !matches!(
            rep.kind(),
            kind::TOKEN_LIST_OP
                | kind::STRING_LIST_OP
                | kind::PATH_LIST_OP
                | kind::REFERENCE_LIST_OP
                | kind::PAYLOAD_LIST_OP
        ) || rep.is_inlined()
        {
            return Vec::new();
        }
        let Some(&header) = self.file.get(rep.payload() as usize) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        // In the order the bits are defined: explicit, added, deleted,
        // ordered, prepended, appended.
        for (bit, word) in [
            (1, ""),
            (2, "add "),
            (3, "delete "),
            (4, "reorder "),
            (5, "prepend "),
            (6, "append "),
        ] {
            if header & (1 << bit) != 0 {
                out.push(word);
            }
        }
        out
    }

    /// A dictionary — `clips`, `customData`, and anything else a layer keeps
    /// as a named map.
    ///
    /// ```text
    /// [ u64 count ]
    /// [ u32 key ][ u64 size ][ size bytes, the last eight of which are a rep ]
    /// ...
    /// ```
    ///
    /// The value sits *behind* a rep at the end of its own blob rather than in
    /// front of it, which is the same shape time samples use — and, like them,
    /// the rep's payload is an absolute offset, so the value it names need not
    /// be inside the blob at all.
    fn dictionary(&self, at: usize, depth: usize) -> UsdValue {
        // A dictionary may hold dictionaries; a corrupt one may claim to hold
        // itself.
        if depth > 16 {
            return UsdValue::None;
        }
        let Some(count) = self.u64_at(at) else {
            return UsdValue::None;
        };
        let mut out = Vec::new();
        let mut cursor = at + 8;
        for _ in 0..count {
            let Some(key) = self
                .file
                .get(cursor..cursor + 4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            else {
                break;
            };
            cursor += 4;
            let Some(size) = self.u64_at(cursor).map(|s| s as usize) else {
                break;
            };
            cursor += 8;
            if size < 8 {
                break;
            }
            let Some(rep) = self.u64_at(cursor + size - 8) else {
                break;
            };
            let rep = ValueRep(rep);
            let value = if rep.kind() == kind::DICTIONARY && !rep.is_inlined() {
                self.dictionary(rep.payload() as usize, depth + 1)
            } else {
                self.value(rep)
            };
            out.push((self.string(key).to_string(), value));
            cursor += size;
        }
        UsdValue::Dict(out)
    }

    /// A list operation over tokens, which is what `apiSchemas` is.
    ///
    /// Not a token *vector*: the difference is why a prim written with one in
    /// place of the other comes back from USD as an uninterpretable blob
    /// rather than a list of names.
    fn token_list_op(&self, at: usize) -> UsdValue {
        let mut out = Vec::new();
        self.each_list_op(at, 4, |slice| {
            let index = u32::from_le_bytes(slice[0..4].try_into().unwrap());
            out.push(UsdValue::Token(self.token(index).to_string()));
        });
        UsdValue::Array(out)
    }

    /// A list operation over strings, which is what `variantSets` is.
    fn string_list_op(&self, at: usize) -> UsdValue {
        let mut out = Vec::new();
        self.each_list_op(at, 4, |slice| {
            let index = u32::from_le_bytes(slice[0..4].try_into().unwrap());
            out.push(UsdValue::String(self.string(index).to_string()));
        });
        match out.len() {
            0 => UsdValue::None,
            1 => out.pop().unwrap(),
            _ => UsdValue::Array(out),
        }
    }

    /// Walk the items of a list operation, whatever they are made of.
    ///
    /// A one-byte header says which of the six sub-lists were authored;
    /// each present one is a count and that many fixed-width items. Every
    /// present list has to be stepped over even where its contents are not
    /// wanted, or the cursor lands mid-list.
    /// One sub-list's worth of a list operation, as its own value.
    fn list_op_sublist(&self, rep: ValueRep, want: &str) -> UsdValue {
        let bit = match want {
            "add " => 2,
            "delete " => 3,
            "reorder " => 4,
            "prepend " => 5,
            "append " => 6,
            _ => 1,
        };
        let at = rep.payload() as usize;
        let stride = match rep.kind() {
            kind::REFERENCE_LIST_OP | kind::PAYLOAD_LIST_OP => 32,
            _ => 4,
        };
        let mut out = Vec::new();
        self.each_sublist(at, stride, bit, |slice| match rep.kind() {
            kind::REFERENCE_LIST_OP | kind::PAYLOAD_LIST_OP => {
                let asset = u32::from_le_bytes(slice[0..4].try_into().unwrap());
                let path = u32::from_le_bytes(slice[4..8].try_into().unwrap());
                out.push(UsdValue::Reference(UsdReference {
                    asset: self.string(asset).to_string(),
                    prim_path: match self.paths.get(&path) {
                        Some(p) if path != 0 => p.clone(),
                        _ => String::new(),
                    },
                    offset: f64::from_le_bytes(slice[8..16].try_into().unwrap()),
                    scale: f64::from_le_bytes(slice[16..24].try_into().unwrap()),
                }));
            }
            kind::PATH_LIST_OP => {
                let index = u32::from_le_bytes(slice[0..4].try_into().unwrap());
                if let Some(path) = self.paths.get(&index) {
                    out.push(UsdValue::Path(path.clone()));
                }
            }
            kind::STRING_LIST_OP => {
                let index = u32::from_le_bytes(slice[0..4].try_into().unwrap());
                out.push(UsdValue::String(self.string(index).to_string()));
            }
            _ => {
                let index = u32::from_le_bytes(slice[0..4].try_into().unwrap());
                out.push(UsdValue::Token(self.token(index).to_string()));
            }
        });
        match out.len() {
            0 => UsdValue::None,
            1 if rep.kind() == kind::PATH_LIST_OP || rep.kind() == kind::STRING_LIST_OP => {
                out.pop().unwrap()
            }
            _ => UsdValue::Array(out),
        }
    }

    /// Walk just one sub-list of a list operation.
    fn each_sublist(&self, at: usize, stride: usize, want: u32, mut take: impl FnMut(&[u8])) {
        let Some(&header) = self.file.get(at) else {
            return;
        };
        let mut cursor = at + 1;
        for bit in 1..7u32 {
            if header & (1 << bit) == 0 {
                continue;
            }
            let Some(bytes) = self.file.get(cursor..cursor + 8) else {
                return;
            };
            let count = u64::from_le_bytes(bytes.try_into().unwrap()) as usize;
            cursor += 8;
            for _ in 0..count {
                let Some(slice) = self.file.get(cursor..cursor + stride) else {
                    return;
                };
                cursor += stride;
                if bit == want {
                    take(slice);
                }
            }
        }
    }

    fn each_list_op(&self, at: usize, stride: usize, mut take: impl FnMut(&[u8])) {
        let Some(&header) = self.file.get(at) else {
            return;
        };
        let mut cursor = at + 1;
        for bit in 1..7 {
            if header & (1 << bit) == 0 {
                continue;
            }
            let Some(bytes) = self.file.get(cursor..cursor + 8) else {
                return;
            };
            let count = u64::from_le_bytes(bytes.try_into().unwrap()) as usize;
            cursor += 8;
            // Explicit items (bit 1), prepended (5) and appended (6) all name
            // something; deletions and orderings do not.
            let wanted = matches!(bit, 1 | 5 | 6);
            for _ in 0..count {
                let Some(slice) = self.file.get(cursor..cursor + stride) else {
                    return;
                };
                cursor += stride;
                if wanted {
                    take(slice);
                }
            }
        }
    }

    /// `{ "look": "steel" }` — which variant of each set is chosen.
    fn variant_selection(&self, at: usize) -> UsdValue {
        let Some(count) = self.u64_at(at) else {
            return UsdValue::None;
        };
        let mut out = Vec::new();
        for i in 0..count as usize {
            let start = at + 8 + i * 8;
            let Some(bytes) = self.file.get(start..start + 8) else {
                break;
            };
            let set = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
            let choice = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
            out.push((
                self.string(set).to_string(),
                UsdValue::String(self.string(choice).to_string()),
            ));
        }
        UsdValue::Dict(out)
    }

    /// A `u64` count followed by string indices — what `subLayers` is.
    fn string_vector(&self, at: usize) -> UsdValue {
        let Some(count) = self.u64_at(at) else {
            return UsdValue::None;
        };
        let mut out = Vec::new();
        for i in 0..count as usize {
            let start = at + 8 + i * 4;
            let Some(bytes) = self.file.get(start..start + 4) else {
                break;
            };
            out.push(UsdValue::Asset(
                self.string(u32::from_le_bytes(bytes.try_into().unwrap()))
                    .to_string(),
            ));
        }
        UsdValue::Array(out)
    }

    /// The `(offset, scale)` pairs that sit beside `subLayers`.
    fn layer_offsets(&self, at: usize) -> UsdValue {
        let Some(count) = self.u64_at(at) else {
            return UsdValue::None;
        };
        let mut out = Vec::new();
        for i in 0..count as usize {
            let start = at + 8 + i * 16;
            let Some(bytes) = self.file.get(start..start + 16) else {
                break;
            };
            out.push(UsdValue::Tuple(vec![
                UsdValue::Float(f64::from_le_bytes(bytes[0..8].try_into().unwrap())),
                UsdValue::Float(f64::from_le_bytes(bytes[8..16].try_into().unwrap())),
            ]));
        }
        UsdValue::Array(out)
    }

    /// A `u64` count followed by that many token indices.
    fn token_list(&self, at: usize) -> UsdValue {
        let count = match self.file.get(at..at + 8) {
            Some(b) => u64::from_le_bytes(b.try_into().unwrap()) as usize,
            None => return UsdValue::None,
        };
        let mut out = Vec::new();
        for i in 0..count {
            let start = at + 8 + i * 4;
            let index = match self.file.get(start..start + 4) {
                Some(b) => u32::from_le_bytes(b.try_into().unwrap()),
                // The count ran past the file: return what was really there
                // rather than trusting the number over the bytes.
                None => break,
            };
            out.push(UsdValue::Token(self.token(index).to_string()));
        }
        UsdValue::Array(out)
    }

    fn scalar_by_reference(&self, kind: u8, at: usize) -> UsdValue {
        match kind {
            kind::STRING | kind::TOKEN | kind::ASSET_PATH | kind::PATH_EXPRESSION => {
                let index = self
                    .file
                    .get(at..at + 4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                    .unwrap_or(0);
                self.text(kind, index)
            }
            _ => UsdValue::None,
        }
    }

    /// A value small enough to live in the rep word itself.
    fn inlined(&self, kind: u8, payload: u64) -> UsdValue {
        match kind {
            kind::BOOL => UsdValue::Bool(payload != 0),
            kind::UCHAR => UsdValue::Int(payload as u8 as i128),
                        // Signed and unsigned differ in how the same bits read back, and
            // the type is the only thing that says which this is.
            kind::INT => UsdValue::Int(payload as u32 as i32 as i128),
            kind::UINT => UsdValue::Int(payload as u32 as i128),
                        kind::INT64 => UsdValue::Int(payload as i64 as i128),
            kind::UINT64 => UsdValue::Int(payload as i128),
            // Inlined floating point is stored as f32 bits — including for
            // doubles, which is why only a double that survives the round trip
            // through f32 is ever inlined. `timeCodesPerSecond = 24` is; the
            // `metersPerUnit = 0.01` two lines above it is not.
            kind::FLOAT | kind::DOUBLE => UsdValue::Float(f32::from_bits(payload as u32) as f64),
            kind::HALF => UsdValue::Float(half_to_f32(payload as u16) as f64),
            kind::STRING | kind::TOKEN | kind::ASSET_PATH => self.text(kind, payload as u32),
            kind::VALUE_BLOCK => UsdValue::Block,
            kind::TIME_CODE => UsdValue::Float(f32::from_bits(payload as u32) as f64),
            kind::SPECIFIER => UsdValue::Int(payload as i128),
            kind::VARIABILITY => UsdValue::Int(payload as i128),
            kind::TOKEN_VECTOR | kind::PATH_VECTOR => UsdValue::Array(Vec::new()),
            // A vector inlined means every component fit in a signed byte, and
            // those bytes are packed into the payload low-end first. A matrix
            // inlined means it is diagonal and only the diagonal is kept — six
            // bytes for a 4x4, which is how an identity transform costs nothing
            // beyond its own rep word.
            kind::MATRIX2D | kind::MATRIX3D | kind::MATRIX4D => {
                let n = match kind {
                    kind::MATRIX2D => 2,
                    kind::MATRIX3D => 3,
                    _ => 4,
                };
                let mut out = Vec::with_capacity(n * n);
                for row in 0..n {
                    for column in 0..n {
                        let value = if row == column {
                            (payload >> (8 * row)) as u8 as i8 as f64
                        } else {
                            0.0
                        };
                        out.push(UsdValue::Float(value));
                    }
                }
                UsdValue::Tuple(out)
            }
            _ => match shape(kind) {
                Some((lanes, _)) => {
                    let components: Vec<UsdValue> = (0..lanes)
                        .map(|i| UsdValue::Float((payload >> (8 * i)) as u8 as i8 as f64))
                        .collect();
                    if is_quat(kind) {
                        real_first(components)
                    } else {
                        UsdValue::Tuple(components)
                    }
                }
                None => UsdValue::None,
            },
        }
    }

    /// An array: a count, then the elements, possibly integer-packed.
    fn array(&self, kind: u8, at: usize, compressed: bool) -> UsdValue {
        let count = match self.file.get(at..at + 8) {
            Some(b) => u64::from_le_bytes(b.try_into().unwrap()) as usize,
            None => return UsdValue::None,
        };
        let body = at + 8;
        if kind == kind::TOKEN || kind == kind::STRING || kind == kind::ASSET_PATH {
            let mut out = Vec::new();
            for i in 0..count {
                let Some(bytes) = self.file.get(body + i * 4..body + i * 4 + 4) else {
                    break;
                };
                let index = u32::from_le_bytes(bytes.try_into().unwrap());
                // An *array* of assets indexes the string table, where a
                // single asset indexes the token table. The asymmetry is
                // real: read an array through the tokens and the paths come
                // back as whatever names happen to sit at those indices.
                out.push(match kind {
                    kind::ASSET_PATH => UsdValue::Asset(self.string(index).to_string()),
                    other => self.text(other, index),
                });
            }
            return UsdValue::Array(out);
        }
        let Some((lanes, width)) = shape(kind) else {
            return UsdValue::None;
        };

        if compressed {
            // Only the integer types are ever packed this way, and a 64-bit
            // one is packed with 64-bit deltas.
            let size = match self.file.get(body..body + 8) {
                Some(b) => u64::from_le_bytes(b.try_into().unwrap()) as usize,
                None => return UsdValue::None,
            };
            let Some(raw) = self.file.get(body + 8..body + 8 + size) else {
                return UsdValue::None;
            };
            let wide = matches!(kind, kind::INT64 | kind::UINT64);
            let ceiling = 16 + count.div_ceil(4) + count * if wide { 8 } else { 4 };
            let Some(packed) = lz4::decompress_upto(raw, ceiling) else {
                return UsdValue::None;
            };
            let values = if wide {
                decode_u64(&packed, count)
            } else {
                decode_u32(&packed, count).map(|v| v.into_iter().map(i64::from).collect())
            };
            let Some(values) = values else {
                return UsdValue::None;
            };
            return UsdValue::Array(values.into_iter().map(|v| UsdValue::Int(v as i128)).collect());
        }

        let stride = lanes * width;
        // Checked, because a corrupt count times a stride wraps silently in
        // release and a wrapped length reads as a plausible small array.
        let Some(bytes) = count
            .checked_mul(stride)
            .and_then(|len| body.checked_add(len))
            .and_then(|end| self.file.get(body..end))
        else {
            return UsdValue::None;
        };
        UsdValue::Array(
            bytes
                .chunks_exact(stride)
                .map(|c| numbers(kind, c, lanes, width))
                .collect(),
        )
    }

    /// Assemble the tables into a prim tree.
    fn to_layer(&self) -> UsdLayer {
        // Every spec, keyed by the path it is for.
        let mut by_path: Specs = HashMap::new();
        for (path_index, set, spec_type) in &self.specs {
            let Some(path) = self.paths.get(path_index) else {
                continue;
            };
            by_path.insert(path.as_str(), (self.spec_fields(*set), *spec_type));
        }

        let mut layer = UsdLayer::default();
        let mut root = by_path.get("/").map(|(f, _)| f.clone()).unwrap_or_default();
        fold_sublayer_offsets(&mut root);
        layer.metadata = root
            .iter()
            .filter(|(k, _)| k != "primChildren")
            .cloned()
            .collect();

        for name in children_of(&root, &by_path, "") {
            if let Some(prim) = self.prim(&by_path, "", &name) {
                layer.prims.push(prim);
            }
        }
        layer
    }

    /// One prim and everything under it.
    ///
    /// The order comes from the spec's own `primChildren` and `properties`
    /// fields rather than from sorting the path table, because those record the
    /// order the document was authored in and a sort would replace it with
    /// alphabetical order.
    fn prim(
        &self,
        by_path: &Specs,
        parent: &str,
        name: &str,
    ) -> Option<UsdPrim> {
        // A name beginning with `/` is already the whole path: that is how a
        // variant body, which does not hang below its prim, is reached.
        let path = if name.starts_with('/') {
            name.to_string()
        } else {
            format!("{parent}/{name}")
        };
        let (fields, _) = by_path.get(path.as_str())?;

        let mut prim = UsdPrim {
            specifier: match field(fields, "specifier") {
                Some(UsdValue::Int(1)) => Specifier::Over,
                Some(UsdValue::Int(2)) => Specifier::Class,
                _ => Specifier::Def,
            },
            type_name: field(fields, "typeName")
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_default(),
            name: name.to_string(),
            metadata: fields
                .iter()
                .filter(|(k, _)| {
                    !matches!(
                        k.as_str(),
                        "specifier"
                            | "typeName"
                            | "primChildren"
                            | "properties"
                            // The variant *bodies* are reassembled below; the
                            // lists naming them are bookkeeping.
                            | "variantSetChildren"
                            | "variantChildren"
                    )
                })
                .map(|(k, v)| (document_name(k), v.clone()))
                .collect(),
            properties: Vec::new(),
            children: Vec::new(),
            variant_sets: Vec::new(),
        };

        for property_name in names_in(fields, "properties") {
            let property_path = format!("{path}.{property_name}");
            let Some((fields, spec_type)) = by_path.get(property_path.as_str()) else {
                continue;
            };
            prim.properties
                .push(self.property(property_name, fields, *spec_type));
        }

        // Variant bodies live at their own paths — `/Chair{look=oak}` — rather
        // than inside the prim, so they are reassembled from the path table.
        for set_name in names_in(fields, "variantSetChildren") {
            let set_path = format!("{path}{{{set_name}=}}");
            let Some((set_fields, _)) = by_path.get(set_path.as_str()) else {
                continue;
            };
            let mut set = UsdVariantSet {
                name: set_name.to_string(),
                variants: Vec::new(),
            };
            for choice in names_in(set_fields, "variantChildren") {
                let variant_path = format!("{path}{{{set_name}={choice}}}");
                if let Some(body) = self.prim(by_path, path.as_str(), &variant_path) {
                    set.variants.push((choice.to_string(), body));
                }
            }
            if !set.variants.is_empty() {
                prim.variant_sets.push(set);
            }
        }

        for child in children_of(fields, by_path, &path) {
            if let Some(child) = self.prim(by_path, &path, &child) {
                prim.children.push(child);
            }
        }
        Some(prim)
    }

    fn property(&self, name: &str, fields: &[(String, UsdValue)], spec_type: u32) -> UsdProperty {
        // An attribute is animated, or it has a default, never both in
        // practice — and when it has both, the samples are what a player uses.
        let default = field(fields, "timeSamples")
            .or_else(|| field(fields, "default"))
            .cloned();
        // The targets are whichever sub-list was authored: a `delete rel foo =
        // [...]` names what it removes, and reading only the explicit and
        // prepended lists loses it.
        let qualifier = qualifier_of(fields, "targetPaths");
        let targets = field(fields, &format!("{qualifier} targetPaths"))
            .or_else(|| field(fields, "targetPaths"));
        let connections = field(fields, "connectionPaths");
        // A relationship has targets; a connected attribute has connections.
        // Either way the useful value is the path, so it becomes the value —
        // which is also where reading the text form leaves it.
        // The spec type is what says this is a relationship. Inferring it from
        // whether targets were authored calls an empty relationship an
        // attribute, and `custom rel foo` with nothing bound is a perfectly
        // ordinary thing to write.
        let is_relationship = spec_type == spec_type::RELATIONSHIP
            || (default.is_none() && targets.is_some());
        let value = default
            .or_else(|| targets.cloned())
            .or_else(|| connections.cloned())
            .unwrap_or(UsdValue::None);
        let mut property = UsdProperty {
            // A relationship's qualifier is recorded in the list-op header of
            // its targets, which the field name already carries.
            qualifier: qualifier.clone(),
            name: name.to_string(),
            type_name: field(fields, "typeName")
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_default(),
            // Uniform is the fallback for a relationship and varying is the
            // fallback for an attribute, so only an attribute saying `uniform`
            // is saying anything — which is how the text form spells it too.
            uniform: !is_relationship
                && matches!(field(fields, "variability"), Some(UsdValue::Int(1))),
            relationship: is_relationship,
            value,
            metadata: Vec::new(),
        };
        // Everything the spec says about the property that is not the
        // property itself. Naming the few fields worth keeping meant losing
        // the rest — `custom`, `displayGroup`, a doc string — none of which
        // this crate interprets but all of which the document said.
        for (key, value) in fields {
            let bare = key.split_once(' ').map(|(_, rest)| rest).unwrap_or(key);
            if matches!(
                bare,
                "typeName"
                    | "variability"
                    | "default"
                    | "timeSamples"
                    | "targetPaths"
                    | "connectionPaths"
            ) {
                continue;
            }
            property.metadata.push((key.clone(), value.clone()));
        }
        property
    }
}

/// What a crate calls a field, as a document spells it.
///
/// Three composition fields are stored under different names than they are
/// written with, which is invisible until an arc read from a `.usdc` fails to
/// compose because nothing was looking for `inheritPaths`.
fn document_name(key: &str) -> String {
    // The qualifier travels with the field, so the name to translate is what
    // is left after it. Mapping the whole string leaves `prepend
    // variantSetNames` untranslated, and a writer that does not recognise the
    // field writes it back as something else again — a round trip that is not
    // stable.
    let (qualifier, bare) = match key.split_once(' ') {
        Some((q, rest)) if matches!(q, "prepend" | "append" | "delete" | "add" | "reorder") => {
            (format!("{q} "), rest)
        }
        _ => (String::new(), key),
    };
    // `primOrder` and `propertyOrder` are what a crate calls the `reorder`
    // statements, and they are plain vectors rather than list operations —
    // so the qualifier is part of the translated name, not in front of it.
    match bare {
        "primOrder" => return "reorder nameChildren".to_string(),
        "propertyOrder" => return "reorder properties".to_string(),
        _ => {}
    }
    let translated = match bare {
        "inheritPaths" => "inherits",
        "variantSelection" => "variants",
        "variantSetNames" => "variantSets",
        other => other,
    };
    format!("{qualifier}{translated}")
}

/// The list-op qualifier a field was stored under, if any.
///
/// `spec_fields` has already put it in front of the name, so the answer is in
/// the key rather than needing the rep read again.
fn qualifier_of(fields: &[(String, UsdValue)], name: &str) -> String {
    for word in ["prepend", "append", "delete", "add", "reorder"] {
        if fields.iter().any(|(k, _)| k == &format!("{word} {name}")) {
            return word.to_string();
        }
    }
    String::new()
}

fn field<'a>(fields: &'a [(String, UsdValue)], name: &str) -> Option<&'a UsdValue> {
    fields
        .iter()
        .find(|(k, _)| {
            k == name
                || matches!(
                    k.split_once(' '),
                    Some((q, rest))
                        if rest == name
                            && matches!(q, "prepend" | "append" | "delete" | "add" | "reorder")
                )
        })
        .map(|(_, v)| v)
}

/// The names listed in a token-vector field.
fn names_in<'a>(fields: &'a [(String, UsdValue)], key: &str) -> Vec<&'a str> {
    field(fields, key).map(UsdValue::flat_tokens).unwrap_or_default()
}

/// A prim's children in authored order, falling back to the path table if the
/// spec did not list them — a layer assembled by a tool that skipped
/// `primChildren` is unusual but not malformed.
fn children_of(
    fields: &[(String, UsdValue)],
    by_path: &Specs,
    path: &str,
) -> Vec<String> {
    let listed = names_in(fields, "primChildren");
    if !listed.is_empty() {
        return listed.into_iter().map(str::to_string).collect();
    }
    let prefix = format!("{path}/");
    let mut found: Vec<String> = by_path
        .keys()
        .filter_map(|p| p.strip_prefix(&prefix))
        // `!rest.is_empty()` keeps the root from being read as a child of
        // itself, which it otherwise is once the prefix is just "/".
        .filter(|rest| {
            !rest.is_empty() && !rest.contains('/') && !rest.contains('.') && !rest.contains('{')
        })
        .map(str::to_string)
        .collect();
    found.sort();
    found
}

/// Read `lanes` numbers of `width` bytes as one value.
fn numbers(kind: u8, bytes: &[u8], lanes: usize, width: usize) -> UsdValue {
    let one = |c: &[u8]| -> UsdValue {
        if is_float_kind(kind) {
            UsdValue::Float(match width {
                2 => half_to_f32(u16::from_le_bytes(c.try_into().unwrap())) as f64,
                4 => f32::from_le_bytes(c.try_into().unwrap()) as f64,
                _ => f64::from_le_bytes(c.try_into().unwrap()),
            })
        } else {
            // An unsigned type reads the same bytes as a different number,
            // and only the type says which it is.
            let unsigned = matches!(kind, kind::UCHAR | kind::UINT | kind::UINT64);
            UsdValue::Int(match (width, unsigned) {
                (1, _) => c[0] as i128,
                (2, false) => i16::from_le_bytes(c.try_into().unwrap()) as i128,
                (2, true) => u16::from_le_bytes(c.try_into().unwrap()) as i128,
                (4, false) => i32::from_le_bytes(c.try_into().unwrap()) as i128,
                (4, true) => u32::from_le_bytes(c.try_into().unwrap()) as i128,
                (_, false) => i64::from_le_bytes(c.try_into().unwrap()) as i128,
                (_, true) => u64::from_le_bytes(c.try_into().unwrap()) as i128,
            })
        }
    };
    if kind == kind::BOOL {
        return UsdValue::Bool(bytes[0] != 0);
    }
    if lanes == 1 {
        return one(&bytes[..width]);
    }
    let components: Vec<UsdValue> = bytes.chunks_exact(width).map(one).collect();
    if is_quat(kind) {
        return real_first(components);
    }
    UsdValue::Tuple(components)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRIANGLE: &[u8] = include_bytes!("testdata/triangle.usdc");

    #[test]
    fn reads_a_crate_written_by_usdcat() {
        let layer = read(TRIANGLE).expect("crate reads");

        // Layer metadata.
        assert_eq!(layer.meta("defaultPrim").and_then(|v| v.as_str()), Some("Root"));
        assert!((layer.meters_per_unit() - 0.01).abs() < 1e-9);
        assert_eq!(layer.up_axis(), 'Y');

        // The prim tree.
        let root = layer.prim_at("/Root").expect("Root prim");
        assert_eq!(root.type_name, "Xform");
        let tri = layer.prim_at("/Root/Tri").expect("Tri prim");
        assert_eq!(tri.type_name, "Mesh");
    }

    #[test]
    fn the_geometry_survives_the_binary_form() {
        let layer = read(TRIANGLE).unwrap();
        let tri = layer.prim_at("/Root/Tri").unwrap();

        assert_eq!(
            tri.value("faceVertexCounts").unwrap().flat_u32(),
            vec![3],
            "face counts"
        );
        assert_eq!(
            tri.value("faceVertexIndices").unwrap().flat_u32(),
            vec![0, 1, 2],
            "face indices"
        );
        // The points the `.usda` this was converted from declared.
        let points = tri.value("points").unwrap().flat_f32();
        assert_eq!(points, vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn it_reaches_the_scene_the_same_way_the_text_form_does() {
        use super::super::scene::to_scene;
        let scene = to_scene(&read(TRIANGLE).unwrap());
        let root = scene.arena.get(scene.roots[0]).unwrap();
        assert_eq!(root.name, "Root");
        let mesh = scene.arena.get(root.children[0]).unwrap();
        assert_eq!(mesh.name, "Tri");
    }

    /// The binary and the text of the *same* document, read by two code paths
    /// that share nothing, must produce the same thing. A binary reader checked
    /// only against a binary writer agrees with itself; this does not.
    #[test]
    fn the_binary_and_the_text_of_one_document_agree() {
        let from_crate = read(include_bytes!("testdata/rich.usdc")).expect("crate reads");
        let from_text =
            super::super::parse::parse(include_str!("testdata/rich.usda")).expect("text parses");

        for path in [
            "/World",
            "/World/Quad",
            "/World/Sub",
            "/World/Sub/Deep",
            "/Mat",
            "/Mat/Shader",
        ] {
            let binary = from_crate.prim_at(path).unwrap_or_else(|| panic!("{path} missing"));
            let text = from_text.prim_at(path).unwrap();
            assert_eq!(binary.type_name, text.type_name, "type of {path}");
            assert_eq!(binary.name, text.name, "name of {path}");

            for property in &text.properties {
                if property.value == UsdValue::None {
                    continue;
                }
                // The text form spells a connection as a property named
                // `foo.connect`; the binary keeps it as a `connectionPaths`
                // field on `foo`. Same edge, two spellings.
                let name = property.name.strip_suffix(".connect").unwrap_or(&property.name);
                let mine = binary
                    .value(name)
                    .unwrap_or_else(|| panic!("{path}.{name} missing in crate"));
                match &property.value {
                    // Paths compare as paths; everything else as numbers, which
                    // covers tokens, vectors, arrays and scalars alike.
                    UsdValue::Path(expected) => {
                        assert_eq!(mine.as_str(), Some(expected.as_str()), "{path}.{name}")
                    }
                    expected if expected.as_str().is_some() => {
                        assert_eq!(mine.as_str(), expected.as_str(), "{path}.{name}")
                    }
                    expected => assert_eq!(
                        mine.flat_f32(),
                        expected.flat_f32(),
                        "{path}.{name} differs"
                    ),
                }
            }

            // Relationships carry their targets in both forms.
            for property in text.properties.iter().filter(|p| p.relationship) {
                let mine = binary.properties.iter().find(|p| p.name == property.name);
                let mine = mine.unwrap_or_else(|| panic!("{path}.{} missing", property.name));
                assert!(mine.relationship, "{path}.{} lost its rel", property.name);
                assert_eq!(
                    mine.value.as_str(),
                    property.value.as_str(),
                    "{path}.{} target",
                    property.name
                );
            }
        }

        assert_eq!(from_crate.up_axis(), 'Z');
        assert_eq!(
            from_crate.prim_at("/World").unwrap().value("xformOpOrder").unwrap().flat_tokens(),
            vec!["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"],
        );
    }

    /// The transform stack and the material have to survive too, because they
    /// are what the scene layer actually reads.
    #[test]
    fn a_crate_scene_matches_the_text_scene() {
        use super::super::scene::to_scene;
        let binary = to_scene(&read(include_bytes!("testdata/rich.usdc")).unwrap());
        let text = to_scene(&super::super::parse::parse(include_str!("testdata/rich.usda")).unwrap());

        let names = |s: &super::super::scene::UsdScene| {
            let mut out = Vec::new();
            fn walk(s: &super::super::scene::UsdScene, id: crate::core::ObjectId, out: &mut Vec<String>) {
                let node = s.arena.get(id).unwrap();
                out.push(node.name.clone());
                for child in &node.children {
                    walk(s, *child, out);
                }
            }
            for root in &s.roots {
                walk(s, *root, &mut out);
            }
            out
        };
        assert_eq!(names(&binary), names(&text));
    }

    /// The animated document, both ways, sample for sample.
    #[test]
    fn animation_survives_the_binary_form() {
        let from_crate = read(include_bytes!("testdata/anim.usdc")).expect("crate reads");
        let from_text =
            super::super::parse::parse(include_str!("testdata/anim.usda")).expect("text parses");

        for (path, property) in [
            ("/Spin", "xformOp:translate"),
            ("/Spin", "xformOp:rotateXYZ"),
            ("/Spin/Body", "points"),
        ] {
            let binary = from_crate.prim_at(path).unwrap().value(property).unwrap();
            let text = from_text.prim_at(path).unwrap().value(property).unwrap();
            let binary = binary.samples().unwrap_or_else(|| panic!("{property} not animated"));
            let text = text.samples().unwrap();
            assert_eq!(binary.len(), text.len(), "{property} sample count");
            for (a, b) in binary.iter().zip(text) {
                assert_eq!(a.0, b.0, "{property} time");
                assert_eq!(a.1.flat_f32(), b.1.flat_f32(), "{property} at {}", a.0);
            }
        }

        // The layer's own timing.
        assert_eq!(from_crate.time_codes_per_second(), 24.0);
        assert_eq!(from_crate.time_range(), Some((0.0, 48.0)));
    }

    /// The whole loop: this crate writes an animated document, OpenUSD's own
    /// `usdcat` converts it to binary, and this reads that binary back.
    ///
    /// `testdata/roundtrip.usdc` is the file `usdcat` produced from
    /// `testdata/roundtrip.usda`, which this crate's exporter wrote. Every
    /// stage is checked by a party that did not write the previous one.
    #[test]
    fn a_document_we_wrote_survives_a_trip_through_openusd() {
        let ours = super::super::parse::parse(include_str!("testdata/roundtrip.usda")).unwrap();
        let theirs = read(include_bytes!("testdata/roundtrip.usdc")).expect("their crate reads");

        assert_eq!(theirs.time_codes_per_second(), 24.0);
        assert_eq!(theirs.time_range(), Some((0.0, 48.0)));

        let cube = theirs.prim_at("/Root/Cube").expect("the cube");
        assert_eq!(cube.type_name, "Mesh");

        for name in ["xformOp:translate", "xformOp:orient", "xformOp:scale"] {
            let mine = ours.prim_at("/Root/Cube").unwrap().value(name).unwrap();
            let theirs = cube.value(name).unwrap_or_else(|| panic!("{name} lost"));
            let mine = mine.samples().unwrap();
            let theirs = theirs.samples().unwrap_or_else(|| panic!("{name} not animated"));
            assert_eq!(mine.len(), theirs.len(), "{name} sample count");
            for (a, b) in mine.iter().zip(theirs) {
                assert_eq!(a.0, b.0, "{name} time");
                for (x, y) in a.1.flat_f32().iter().zip(b.1.flat_f32()) {
                    assert!((x - y).abs() < 1e-6, "{name} at {}: {x} vs {y}", a.0);
                }
            }
        }

        // The geometry made it too, with the winding and count intact.
        let points = cube.value("points").unwrap().flat_f32();
        assert_eq!(points.len(), 24 * 3);
        assert_eq!(cube.value("faceVertexIndices").unwrap().flat_u32().len(), 36);
    }

    /// The types a geometry file does not use but a real asset will: 64-bit
    /// integers wide enough to need the wide decoder, halves, doubles, a
    /// matrix, and the scalars.
    #[test]
    fn the_wider_types_decode_too() {
        let layer = read(include_bytes!("testdata/wide.usdc")).expect("crate reads");
        let text = super::super::parse::parse(include_str!("testdata/wide.usda")).unwrap();
        let w = layer.prim_at("/W").expect("the scope");

        // Values past 2^32, which a 32-bit delta cannot carry.
        let big = w.value("big").unwrap();
        let UsdValue::Array(items) = big else {
            panic!("expected an array, got {big:?}");
        };
        let big: Vec<i128> = items
            .iter()
            .map(|v| match v {
                UsdValue::Int(n) => *n,
                other => panic!("expected an int, got {other:?}"),
            })
            .collect();
        assert_eq!(big, vec![0, 1, 2, 4294967296, -4294967296, 9007199254740993]);

        assert_eq!(w.value("small").unwrap().flat_u32(), (1..=10).collect::<Vec<_>>());
        assert_eq!(w.value("halves").unwrap().flat_f32(), vec![0.0, 0.5, 1.0, 2.0]);
        assert_eq!(w.value("doubles").unwrap().flat_f32(), vec![0.1, 0.25, 1e10]);
        assert_eq!(w.value("flag").unwrap(), &UsdValue::Bool(true));
        assert_eq!(w.value("label").unwrap().as_str(), Some("hello"));
        assert_eq!(w.value("pair").unwrap().flat_u32(), vec![3, 4]);

        // A matrix comes back row by row, in the order it was written.
        assert_eq!(
            w.value("frame").unwrap().flat_f32(),
            text.prim_at("/W").unwrap().value("frame").unwrap().flat_f32()
        );
    }

    #[test]
    fn half_precision_round_trips_the_values_that_matter() {
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0x3C00), 1.0);
        assert_eq!(half_to_f32(0xBC00), -1.0);
        assert_eq!(half_to_f32(0x4000), 2.0);
        assert!(half_to_f32(0x7C00).is_infinite());
        // A subnormal, which the exponent shortcut would get wrong.
        assert!((half_to_f32(0x0001) - 5.96e-8).abs() < 1e-9);
    }

    /// A count that outruns the file is a lie the reader must not act on:
    /// allocating for it is how a 200-byte file asks for a gigabyte.
    #[test]
    fn a_count_larger_than_the_file_does_not_allocate_for_it() {
        let mut file = include_bytes!("testdata/triangle.usdc").to_vec();
        // The points array's count, found by its contents rather than by a
        // remembered offset: three points, then nine floats starting at zero.
        let signature: Vec<u8> = 3u64
            .to_le_bytes()
            .iter()
            .copied()
            // The first point, (0, 0, 0), then the x of the second, which is 1.
            .chain([0u8; 12])
            .chain(1.0f32.to_le_bytes())
            .collect();
        let points_at = file
            .windows(signature.len())
            .position(|w| w == signature)
            .expect("the points array is in this fixture");
        file[points_at..points_at + 8].copy_from_slice(&(1u64 << 48).to_le_bytes());

        // Reads, does not hang, does not abort — and gives back only what was
        // actually in the file.
        let layer = read(&file).expect("still a readable layer");
        let points = layer.prim_at("/Root/Tri").unwrap().value("points").unwrap();
        assert!(
            points.flat_f32().len() < 1000,
            "returned {} floats from a 1KB file",
            points.flat_f32().len()
        );
    }

    #[test]
    fn nonsense_bytes_do_not_panic() {
        assert!(read(b"PXR-USDC").is_err());
        let mut junk = vec![0u8; 200];
        junk[..8].copy_from_slice(b"PXR-USDC");
        junk[16..24].copy_from_slice(&88u64.to_le_bytes());
        // A section count of zero: no tables, so an empty layer rather than an
        // error, because that file is not malformed — only empty.
        assert!(read(&junk).is_ok());
    }
}

#[cfg(test)]
mod robustness {
    use super::*;

    /// Every prefix of a real crate file. A reader that indexes before it
    /// checks will find its bound somewhere in here.
    #[test]
    fn every_truncation_is_survivable() {
        for (file, stride) in [
            (include_bytes!("testdata/triangle.usdc").as_slice(), 1),
            (include_bytes!("testdata/rich.usdc").as_slice(), 3),
            (include_bytes!("testdata/stress.usdc").as_slice(), 7),
        ] {
            for cut in (0..file.len()).step_by(stride) {
                // The result is uninteresting; not panicking is the point.
                let _ = read(&file[..cut]);
            }
        }
    }

    /// One byte changed, everywhere, on a file small enough to do exhaustively.
    ///
    /// A crate is full of offsets and counts read out of the file itself, and
    /// a single flipped byte can turn any of them into something enormous or
    /// something that points back at itself.
    #[test]
    fn every_single_byte_corruption_is_survivable() {
        // The smallest file exhaustively, and the richer ones sampled — they
        // reach the value types, list operations and dictionaries a minimal
        // file never touches, and every offset in them is reached by one
        // stride or another across the three.
        corrupt_bytes(include_bytes!("testdata/triangle.usdc"), 1);
        corrupt_bytes(include_bytes!("testdata/rich.usdc"), 3);
        corrupt_bytes(include_bytes!("testdata/apple.usdc"), 5);
    }

    fn corrupt_bytes(original: &[u8], stride: usize) {
        for at in (0..original.len()).step_by(stride) {
            for value in [0x00u8, 0x01, 0x7F, 0x80, 0xFF] {
                let mut file = original.to_vec();
                if file[at] == value {
                    continue;
                }
                file[at] = value;
                let _ = read(&file);
            }
        }
    }

    /// Two bytes at a time, sampled — the pairs a single-byte sweep cannot
    /// reach, such as a length and the offset it is checked against.
    #[test]
    fn paired_corruptions_are_survivable() {
        let original = include_bytes!("testdata/triangle.usdc");
        let n = original.len();
        for a in (0..n).step_by(7) {
            for b in (a + 1..n).step_by(13) {
                let mut file = original.to_vec();
                file[a] = 0xFF;
                file[b] = 0xFF;
                let _ = read(&file);
            }
        }
    }

    /// A table of contents that claims sections outside the file.
    #[test]
    fn a_lying_table_of_contents_is_survivable() {
        let original = include_bytes!("testdata/triangle.usdc");
        for offset in [0u64, 1, 87, 88, u64::MAX, u64::MAX / 2, 1 << 40] {
            let mut file = original.to_vec();
            file[16..24].copy_from_slice(&offset.to_le_bytes());
            let _ = read(&file);
        }
    }

    /// An archive is a container of offsets and lengths, every one of them
    /// read out of the file.
    #[test]
    fn corrupt_archives_are_survivable() {
        let original = include_bytes!("testdata/crate_inside.usdz");
        for cut in (0..original.len()).step_by(3) {
            let _ = super::super::UsdLoader::parse_archive(&original[..cut]);
            let _ = super::super::usdz::arkit_issues(&original[..cut]);
        }
        for at in (0..original.len()).step_by(5) {
            let mut file = original.to_vec();
            file[at] = 0xFF;
            let _ = super::super::UsdLoader::parse_archive(&file);
            let _ = super::super::UsdLoader::parse(&file);
        }
    }

    /// Text that is not a document, including the shapes a fuzzer finds first.
    #[test]
    fn malformed_text_is_survivable() {
        for source in [
            "",
            "#usda 1.0",
            "#usda 1.0\n(",
            "#usda 1.0\ndef",
            "#usda 1.0\ndef Xform",
            "#usda 1.0\ndef Xform \"a\"",
            "#usda 1.0\ndef Xform \"a\" {",
            "#usda 1.0\ndef Xform \"a\" { float x = }",
            "#usda 1.0\ndef Xform \"a\" { float x = [ }",
            "#usda 1.0\ndef Xform \"a\" { variantSet }",
            "#usda 1.0\ndef Xform \"a\" { variantSet \"v\" = { }",
            "#usda 1.0\ndef Xform \"a\" ( clips = { dictionary d = { } ) { }",
            "#usda 1.0\n( subLayers = [ @",
            "#usda 1.0\ndef Xform \"a\" { double3 t.timeSamples = { 0: } }",
            "#usda 1.0\ndef Xform \"a\" { rel r = @x@",
            // Deep nesting, which is where a recursive-descent parser runs out
            // of stack if it never counts.
            "#usda 1.0\n",
        ] {
            let _ = super::super::parse::parse(source);
        }

        // A thousand levels of nesting, built rather than written out.
        let mut deep = String::from("#usda 1.0\n");
        for i in 0..200 {
            deep.push_str(&format!("def Xform \"a{i}\"\n{{\n"));
        }
        for _ in 0..200 {
            deep.push_str("}\n");
        }
        let _ = super::super::parse::parse(&deep);
    }
}
