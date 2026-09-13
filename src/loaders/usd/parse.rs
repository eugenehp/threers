//! The `.usda` grammar: a layer is metadata and a forest of prims.
//!
//! Deliberately structural rather than schema-aware. A prim keeps its type name
//! as a string and its properties as declared, so a document survives a
//! round trip through this crate whether or not anything here knows what a
//! `UsdPhysicsScene` is — and the geometry mapping in [`scene`](super::scene)
//! reads that structure instead of the text.

use super::value::{tokenize, Token, UsdReference, UsdValue};
use super::UsdError;

/// How a prim relates to one it may be overriding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Specifier {
    /// `def` — this layer defines it.
    Def,
    /// `over` — this layer changes one defined elsewhere.
    Over,
    /// `class` — an inheritable template, not itself rendered.
    Class,
}

impl Specifier {
    fn keyword(self) -> &'static str {
        match self {
            Specifier::Def => "def",
            Specifier::Over => "over",
            Specifier::Class => "class",
        }
    }
}

/// One property of a prim: an attribute with a declared type, or a relationship.
#[derive(Debug, Clone)]
pub struct UsdProperty {
    /// The list operation a relationship's targets were authored with —
    /// `prepend`, `append`, or empty for an explicit list.
    ///
    /// It is part of what was said: `prepend rel prototypes` and
    /// `rel prototypes` compose differently, and dropping the word turns one
    /// into the other.
    pub qualifier: String,
    pub name: String,
    /// `point3f[]`, `token`, `matrix4d` — empty for a relationship.
    pub type_name: String,
    /// `uniform` or `varying`; USD's default is varying.
    pub uniform: bool,
    /// True for `rel`, where the value is one or more paths.
    pub relationship: bool,
    pub value: UsdValue,
    /// `(interpolation = "vertex")` and friends.
    pub metadata: Vec<(String, UsdValue)>,
}

impl UsdProperty {
    /// The value of one of this property's metadata fields.
    pub fn meta(&self, key: &str) -> Option<&UsdValue> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
}

/// A prim: a named node with a type, properties and children.
#[derive(Debug, Clone)]
pub struct UsdPrim {
    pub specifier: Specifier,
    /// `Mesh`, `Xform`, `Material` — empty when the prim is untyped.
    pub type_name: String,
    pub name: String,
    pub metadata: Vec<(String, UsdValue)>,
    pub properties: Vec<UsdProperty>,
    pub children: Vec<UsdPrim>,
    /// The variant sets authored on this prim. Which one is *selected* is a
    /// separate question, answered by the `variants` metadata — possibly on a
    /// different layer entirely, which is the point of them.
    pub variant_sets: Vec<UsdVariantSet>,
}

/// A named set of alternatives for one prim.
#[derive(Debug, Clone, Default)]
pub struct UsdVariantSet {
    pub name: String,
    /// Each choice and the prim body it contributes when chosen.
    pub variants: Vec<(String, UsdPrim)>,
}

impl UsdVariantSet {
    pub fn get(&self, choice: &str) -> Option<&UsdPrim> {
        self.variants.iter().find(|(n, _)| n == choice).map(|(_, p)| p)
    }
}

impl UsdPrim {
    /// A property by name, whatever its type.
    pub fn property(&self, name: &str) -> Option<&UsdProperty> {
        self.properties.iter().find(|p| p.name == name)
    }

    /// The value of a property, if it has one.
    pub fn value(&self, name: &str) -> Option<&UsdValue> {
        self.property(name).map(|p| &p.value)
    }

    /// The value of a prim metadata field.
    ///
    /// A list operation is stored under the name it was written with,
    /// qualifier and all — `prepend apiSchemas` — because that qualifier is
    /// part of what was said. Asking for `apiSchemas` finds it anyway: a
    /// caller after the schemas should not have to guess which of three
    /// spellings the author used. [`meta_exact`](Self::meta_exact) is there
    /// for the cases that do care.
    pub fn meta(&self, key: &str) -> Option<&UsdValue> {
        self.meta_exact(key)
            .or_else(|| self.meta_exact(&format!("prepend {key}")))
            .or_else(|| self.meta_exact(&format!("append {key}")))
    }

    /// The value of a metadata field under exactly that name.
    pub fn meta_exact(&self, key: &str) -> Option<&UsdValue> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// A child prim by name.
    pub fn child(&self, name: &str) -> Option<&UsdPrim> {
        self.children.iter().find(|c| c.name == name)
    }

    /// Every prim in this subtree, with its absolute path, depth first.
    pub fn walk<'a>(&'a self, prefix: &str, out: &mut Vec<(String, &'a UsdPrim)>) {
        let path = format!("{prefix}/{}", self.name);
        out.push((path.clone(), self));
        for child in &self.children {
            child.walk(&path, out);
        }
    }
}

impl UsdProperty {
    /// Fold a second declaration of the same property into this one, keeping
    /// whatever each of them actually stated.
    fn merge(&mut self, other: UsdProperty) {
        if self.type_name.is_empty() {
            self.type_name = other.type_name;
        }
        self.uniform |= other.uniform;
        self.relationship |= other.relationship;
        if !matches!(other.value, UsdValue::None) {
            self.value = other.value;
        }
        for (key, value) in other.metadata {
            if !self.metadata.iter().any(|(k, _)| *k == key) {
                self.metadata.push((key, value));
            }
        }
    }
}

/// A quoted value as the named thing its declaration says it is.
fn as_named(value: UsdValue, asset: bool) -> UsdValue {
    match value {
        UsdValue::String(s) | UsdValue::Token(s) => {
            if asset {
                UsdValue::Asset(s)
            } else {
                UsdValue::Token(s)
            }
        }
        UsdValue::Array(items) => {
            UsdValue::Array(items.into_iter().map(|v| as_named(v, asset)).collect())
        }
        other => other,
    }
}

/// Add a metadata field, replacing any earlier one of the same name.
///
/// A layer may say the same thing twice — a bare doc comment *and* an explicit
/// `documentation` — and what it means is the later, explicit one. Keeping
/// both writes the field out twice, which is not a document USD would produce.
fn push_meta(out: &mut Vec<(String, UsdValue)>, key: &str, value: UsdValue) {
    match out.iter_mut().find(|(k, _)| k == key) {
        Some(slot) => slot.1 = value,
        None => out.push((key.to_string(), value)),
    }
}

/// Every integer inside a value as a float, for a dictionary entry whose
/// declared type is a floating one.
fn as_floating(value: UsdValue) -> UsdValue {
    match value {
        UsdValue::Int(v) => UsdValue::Float(v as f64),
        UsdValue::Array(items) => UsdValue::Array(items.into_iter().map(as_floating).collect()),
        UsdValue::Tuple(items) => UsdValue::Tuple(items.into_iter().map(as_floating).collect()),
        other => other,
    }
}

/// A whole `.usda` document.
#[derive(Debug, Clone, Default)]
pub struct UsdLayer {
    pub metadata: Vec<(String, UsdValue)>,
    pub prims: Vec<UsdPrim>,
}

/// An integer written where a boolean was declared, as the boolean it is.
///
/// Applied by the declared type, so a genuine integer elsewhere is untouched.
fn as_bools(value: UsdValue) -> UsdValue {
    match value {
        UsdValue::Int(n) => UsdValue::Bool(n != 0),
        UsdValue::Array(items) => UsdValue::Array(items.into_iter().map(as_bools).collect()),
        other => other,
    }
}

impl UsdLayer {
    /// The value of a layer metadata field.
    pub fn meta(&self, key: &str) -> Option<&UsdValue> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Every prim in the layer, keyed by absolute path.
    pub fn prims_by_path(&self) -> Vec<(String, &UsdPrim)> {
        let mut out = Vec::new();
        for prim in &self.prims {
            prim.walk("", &mut out);
        }
        out
    }

    /// The prim at an absolute path.
    pub fn prim_at(&self, path: &str) -> Option<&UsdPrim> {
        let mut current: Option<&UsdPrim> = None;
        for part in path.trim_start_matches('/').split('/').filter(|s| !s.is_empty()) {
            current = match current {
                None => self.prims.iter().find(|p| p.name == part),
                Some(prim) => prim.child(part),
            };
            current?;
        }
        current
    }

    /// `metersPerUnit`, or USD's default of 1 (metres).
    /// How many time codes make a second. USD's stated default is 24, which
    /// is also what a layer that never says means in practice.
    pub fn time_codes_per_second(&self) -> f64 {
        self.meta("timeCodesPerSecond")
            .and_then(|v| v.as_f64())
            .filter(|v| *v > 0.0)
            .unwrap_or(24.0)
    }

    /// The layer's authored time range, if it declared one.
    ///
    /// A layer may animate without stating a range, in which case the range
    /// has to be taken from the samples themselves — see
    /// [`sampled_time_range`](Self::sampled_time_range).
    pub fn time_range(&self) -> Option<(f64, f64)> {
        let start = self.meta("startTimeCode")?.as_f64()?;
        let end = self.meta("endTimeCode")?.as_f64()?;
        Some((start, end))
    }

    /// The time range covered by every sample in the layer, which is what a
    /// layer that declared no range actually spans.
    pub fn sampled_time_range(&self) -> Option<(f64, f64)> {
        let mut range: Option<(f64, f64)> = None;
        for (_, prim) in self.prims_by_path() {
            for property in &prim.properties {
                let Some(samples) = property.value.samples() else {
                    continue;
                };
                for (time, _) in samples {
                    range = Some(match range {
                        None => (*time, *time),
                        Some((lo, hi)) => (lo.min(*time), hi.max(*time)),
                    });
                }
            }
        }
        range
    }

    /// The layer as it stands at one instant, with every animated value
    /// replaced by what it holds then.
    ///
    /// This is what USD means by evaluating at a time code, and doing it once
    /// up front means nothing downstream has to know that animation exists.
    pub fn at_time(&self, time: f64) -> UsdLayer {
        fn resolve(prim: &UsdPrim, time: f64) -> UsdPrim {
            let mut out = prim.clone();
            for property in &mut out.properties {
                if property.value.samples().is_some() {
                    property.value = property.value.at_time(time);
                }
            }
            out.children = prim.children.iter().map(|c| resolve(c, time)).collect();
            out
        }
        UsdLayer {
            metadata: self.metadata.clone(),
            prims: self.prims.iter().map(|p| resolve(p, time)).collect(),
        }
    }

    /// Whether anything in the layer is animated.
    pub fn is_animated(&self) -> bool {
        self.prims_by_path()
            .iter()
            .any(|(_, prim)| prim.properties.iter().any(|p| p.value.samples().is_some()))
    }

    pub fn meters_per_unit(&self) -> f32 {
        self.meta("metersPerUnit")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .filter(|v| *v > 0.0)
            .unwrap_or(1.0)
    }

    /// `upAxis`, or USD's default of `Y`.
    pub fn up_axis(&self) -> char {
        self.meta("upAxis")
            .and_then(|v| v.as_str())
            .and_then(|s| s.chars().next())
            .unwrap_or('Y')
    }
}

/// Parse a `.usda` document.
pub fn parse(src: &str) -> Result<UsdLayer, UsdError> {
    let tokens = tokenize(src).map_err(|(at, why)| UsdError::Syntax {
        line: line_of(src, at),
        what: why,
    })?;
    let mut p = Parser { tokens, at: 0, src };
    p.layer()
}

fn line_of(src: &str, byte: usize) -> usize {
    src.as_bytes()[..byte.min(src.len())]
        .iter()
        .filter(|&&c| c == b'\n')
        .count()
        + 1
}

struct Parser<'a> {
    tokens: Vec<Token>,
    at: usize,
    src: &'a str,
}

impl Parser<'_> {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.at)
    }

    fn next(&mut self) -> Option<Token> {
        let t = self.tokens.get(self.at).cloned();
        if t.is_some() {
            self.at += 1;
        }
        t
    }

    fn eat_punct(&mut self, c: char) -> bool {
        if matches!(self.peek(), Some(Token::Punct(p)) if *p == c) {
            self.at += 1;
            true
        } else {
            false
        }
    }

    fn expect_punct(&mut self, c: char) -> Result<(), UsdError> {
        if self.eat_punct(c) {
            Ok(())
        } else {
            Err(self.err("expected punctuation"))
        }
    }

    fn err(&self, what: &'static str) -> UsdError {
        UsdError::Syntax {
            // The token stream has no offsets; report how far in we are, which
            // is enough to find the spot in a file you are already looking at.
            line: self.at,
            what,
        }
    }

    fn layer(&mut self) -> Result<UsdLayer, UsdError> {
        let mut layer = UsdLayer::default();
        // Layer metadata, when present, is the first parenthesised block.
        if matches!(self.peek(), Some(Token::Punct('('))) {
            layer.metadata = self.metadata_block()?;
        }
        while self.peek().is_some() {
            layer.prims.push(self.prim()?);
        }
        let _ = self.src;
        Ok(layer)
    }

    /// `( key = value ... )`, also used for prim and property metadata.
    fn metadata_block(&mut self) -> Result<Vec<(String, UsdValue)>, UsdError> {
        self.expect_punct('(')?;
        let mut out = Vec::new();
        loop {
            match self.peek() {
                None => return Err(self.err("unterminated metadata")),
                Some(Token::Punct(')')) => {
                    self.at += 1;
                    return Ok(out);
                }
                // A bare string is a doc comment shorthand: `( "notes" )`.
                // A bare string is the `comment` field, which is a different
                // thing from `documentation` — a layer may carry both, and USD
                // writes the comment bare and the documentation as `doc`.
                Some(Token::String(_)) => {
                    if let Some(Token::String(s)) = self.next() {
                        push_meta(&mut out, "comment", UsdValue::String(s));
                    }
                }
                // `( offset = 10; scale = 2 )` separates its entries with
                // semicolons where a layer's metadata uses newlines, and a
                // list written inline uses commas.
                Some(Token::Punct(';')) | Some(Token::Punct(',')) => self.at += 1,
                Some(Token::Ident(_)) => {
                    let mut key = match self.next() {
                        Some(Token::Ident(k)) => k,
                        _ => unreachable!(),
                    };
                    // `prepend apiSchemas = [...]` is one field with a list
                    // operation in front of it, not a bare word followed by a
                    // field. Keeping them together is what lets the writer put
                    // the line back as it found it.
                    if matches!(key.as_str(), "prepend" | "append" | "delete" | "add") {
                        if let Some(Token::Ident(next)) = self.peek().cloned() {
                            self.at += 1;
                            key = format!("{key} {next}");
                        }
                    }
                    if self.eat_punct('=') {
                        let v = self.value()?;
                        push_meta(&mut out, &key, v);
                    } else {
                        // A qualifier with nothing after it: keep the word so a
                        // writer can put it back.
                        out.push((key, UsdValue::None));
                    }
                }
                _ => {
                    // Anything else inside metadata is kept positionally rather
                    // than dropped, so an unfamiliar field cannot silently
                    // vanish on a round trip.
                    let v = self.value()?;
                    out.push((String::new(), v));
                }
            }
        }
    }

    fn prim(&mut self) -> Result<UsdPrim, UsdError> {
        let specifier = match self.next() {
            Some(Token::Ident(k)) if k == "def" => Specifier::Def,
            Some(Token::Ident(k)) if k == "over" => Specifier::Over,
            Some(Token::Ident(k)) if k == "class" => Specifier::Class,
            _ => return Err(self.err("expected def, over or class")),
        };
        // An optional type name, then the prim's own name in quotes.
        let mut type_name = String::new();
        if let Some(Token::Ident(t)) = self.peek().cloned() {
            type_name = t;
            self.at += 1;
        }
        let name = match self.next() {
            Some(Token::String(n)) => n,
            _ => return Err(self.err("expected prim name")),
        };
        let metadata = if matches!(self.peek(), Some(Token::Punct('('))) {
            self.metadata_block()?
        } else {
            Vec::new()
        };

        let mut prim = UsdPrim {
            specifier,
            type_name,
            name,
            metadata,
            properties: Vec::new(),
            children: Vec::new(),
            variant_sets: Vec::new(),
        };
        self.prim_body(&mut prim)?;
        Ok(prim)
    }

    /// The `{ ... }` of a prim: properties, children and variant sets.
    fn prim_body(&mut self, prim: &mut UsdPrim) -> Result<(), UsdError> {
        self.expect_punct('{')?;
        loop {
            match self.peek() {
                None => return Err(self.err("unterminated prim")),
                Some(Token::Punct('}')) => {
                    self.at += 1;
                    return Ok(());
                }
                Some(Token::Ident(k)) if k == "def" || k == "over" || k == "class" => {
                    prim.children.push(self.prim()?);
                }
                // `reorder nameChildren = [...]` says what order a prim's
                // children compose in. It sits in the body like a property
                // and is nothing of the sort.
                Some(Token::Ident(k))
                    if k == "reorder"
                        && matches!(
                            self.tokens.get(self.at + 1),
                            Some(Token::Ident(w)) if w == "nameChildren" || w == "properties"
                        ) =>
                {
                    self.at += 1;
                    let Some(Token::Ident(what)) = self.next() else {
                        unreachable!()
                    };
                    if !self.eat_punct('=') {
                        return Err(self.err("expected = after reorder"));
                    }
                    let value = self.value()?;
                    prim.metadata.push((format!("reorder {what}"), value));
                }
                Some(Token::Ident(k)) if k == "variantSet" => {
                    let set = self.variant_set()?;
                    match prim.variant_sets.iter_mut().find(|s| s.name == set.name) {
                        Some(existing) => existing.variants.extend(set.variants),
                        None => prim.variant_sets.push(set),
                    }
                }
                _ => {
                    let property = self.property()?;
                    // `token foo` and `foo.connect = <path>` on one prim are
                    // the same property declared twice; the second carries
                    // the target the first is missing.
                    match prim.properties.iter_mut().find(|p| p.name == property.name) {
                        Some(existing) => existing.merge(property),
                        None => prim.properties.push(property),
                    }
                }
            }
        }
    }

    /// `variantSet "look" = { "red" { ... } "blue" { ... } }`.
    ///
    /// Each variant's body is a prim body — properties, children, metadata —
    /// which is exactly what gets grafted on when that variant is selected.
    fn variant_set(&mut self) -> Result<UsdVariantSet, UsdError> {
        self.at += 1; // `variantSet`
        let name = match self.next() {
            Some(Token::String(n)) => n,
            _ => return Err(self.err("expected a variant set name")),
        };
        if !self.eat_punct('=') {
            return Err(self.err("expected = after a variant set name"));
        }
        self.expect_punct('{')?;

        let mut variants = Vec::new();
        while !self.eat_punct('}') {
            let choice = match self.next() {
                Some(Token::String(c)) => c,
                _ => return Err(self.err("expected a variant name")),
            };
            let metadata = if matches!(self.peek(), Some(Token::Punct('('))) {
                self.metadata_block()?
            } else {
                Vec::new()
            };
            // The body is parsed as a prim named for the variant; only its
            // contents are used.
            let mut body = UsdPrim {
                specifier: Specifier::Over,
                type_name: String::new(),
                name: choice.clone(),
                metadata,
                properties: Vec::new(),
                children: Vec::new(),
                variant_sets: Vec::new(),
            };
            self.prim_body(&mut body)?;
            variants.push((choice, body));
            if self.peek().is_none() {
                return Err(self.err("unterminated variant set"));
            }
        }
        Ok(UsdVariantSet { name, variants })
    }

    /// `[uniform] type name [= value] [( meta )]`, or `rel name = <path>`.
    fn property(&mut self) -> Result<UsdProperty, UsdError> {
        let mut uniform = false;
        let mut relationship = false;
        let mut qualifier = String::new();
        let mut custom = false;
        let mut words: Vec<String> = Vec::new();
        // Qualifiers come before the type; collect words until the name.
        while let Some(Token::Ident(w)) = self.peek().cloned() {
            self.at += 1;
            match w.as_str() {
                "uniform" => uniform = true,
                "varying" => {}
                "custom" => custom = true,
                "prepend" | "append" | "delete" | "add" | "reorder" => {
                    qualifier = w.clone()
                }
                "rel" => relationship = true,
                _ => {
                    words.push(w);
                    // A type may be followed by `[]`, which the tokenizer gives
                    // as two punctuation marks.
                    if self.eat_punct('[') {
                        self.expect_punct(']')?;
                        if let Some(last) = words.last_mut() {
                            last.push_str("[]");
                        }
                    }
                    if words.len() == 2 || relationship {
                        break;
                    }
                }
            }
        }
        let (type_name, name) = match words.len() {
            2 => (words[0].clone(), words[1].clone()),
            1 => (String::new(), words[0].clone()),
            _ => return Err(self.err("expected a property")),
        };

        // `foo.timeSamples = {...}` is an animated `foo`, and `foo.connect =
        // <path>` is a connected `foo` — neither is a property in its own
        // right. Both readers land on the same shape: one property named `foo`
        // carrying the samples or the target. That matters beyond tidiness: a
        // dot is not legal *inside* a property name, so a crate file that
        // stored one would give USD an unusable path.
        let animated = name.ends_with(".timeSamples");
        let name = match name.strip_suffix(".timeSamples") {
            Some(base) => base.to_string(),
            None => match name.strip_suffix(".connect") {
                Some(base) => base.to_string(),
                None => name,
            },
        };
        let value = if self.eat_punct('=') {
            if animated {
                self.time_samples()?
            } else {
                self.value()?
            }
        } else {
            UsdValue::None
        };
        let metadata = if matches!(self.peek(), Some(Token::Punct('('))) {
            self.metadata_block()?
        } else {
            Vec::new()
        };
        // A stray `;` separates properties written on one line.
        self.eat_punct(';');
        let mut metadata = metadata;
        if custom {
            // USD keeps this as a field rather than as syntax, which is how it
            // survives into a crate.
            metadata.insert(0, ("custom".to_string(), UsdValue::Bool(true)));
        }
        // USD writes a boolean as `1` or `0` as readily as `true` or `false` —
        // `usdcat` always writes the digits — and the digits are indis-
        // tinguishable from an integer until the declared type is consulted.
        // Without this, `uniform bool doubleSided = 1` read back as the number
        // one and every boolean from an OpenUSD-written file was silently not
        // a boolean.
        let value = if type_name == "bool" || type_name == "bool[]" {
            as_bools(value)
        } else {
            value
        };
        Ok(UsdProperty {
            qualifier,
            name,
            type_name,
            uniform,
            relationship,
            value,
            metadata,
        })
    }

    /// Whether what follows is a layer offset rather than a metadata block.
    ///
    /// `offset` and `scale` are the only two fields one can hold, and neither
    /// is a name that attribute metadata uses, so the first word inside settles
    /// it without needing to know the context.
    fn at_layer_offset(&self) -> bool {
        if !matches!(self.peek(), Some(Token::Punct('('))) {
            return false;
        }
        matches!(
            self.tokens.get(self.at + 1),
            Some(Token::Ident(word)) if word == "offset" || word == "scale"
        )
    }

    /// The optional `( offset = 10; scale = 2 )` that may follow an arc.
    fn layer_offset(&mut self, mut arc: UsdReference) -> UsdReference {
        if !self.at_layer_offset() {
            return arc;
        }
        // Anything unrecognised inside is skipped rather than refused: an arc
        // may carry metadata this crate has no use for.
        if let Ok(entries) = self.metadata_block() {
            for (key, value) in entries {
                match (key.as_str(), value.as_f64()) {
                    ("offset", Some(v)) => arc.offset = v,
                    ("scale", Some(v)) => arc.scale = v,
                    _ => {}
                }
            }
        }
        arc
    }

    /// `{ 0: (0, 0, 0), 24: (10, 0, 0) }` — times paired with values.
    fn time_samples(&mut self) -> Result<UsdValue, UsdError> {
        if !self.eat_punct('{') {
            return Err(self.err("expected time samples"));
        }
        let mut samples = Vec::new();
        while !self.eat_punct('}') {
            if self.eat_punct(',') {
                continue;
            }
            let time = match self.next() {
                Some(Token::Integer(v)) => v as f64,
                Some(Token::Number(v)) => v,
                _ => return Err(self.err("expected a time code")),
            };
            self.expect_punct(':')?;
            samples.push((time, self.value()?));
            if self.peek().is_none() {
                return Err(self.err("unterminated time samples"));
            }
        }
        // USD does not require the samples be authored in order, but every
        // consumer assumes they are.
        samples.sort_by(|a, b| a.0.total_cmp(&b.0));
        Ok(UsdValue::TimeSamples(samples))
    }

    fn value(&mut self) -> Result<UsdValue, UsdError> {
        match self.next() {
            None => Err(self.err("expected a value")),
            Some(Token::Integer(v)) => Ok(UsdValue::Int(v)),
            Some(Token::Number(v)) => Ok(UsdValue::Float(v)),
            Some(Token::String(v)) => Ok(UsdValue::String(v)),
            // `@layer.usda@</Prim>` is one composition arc. An asset that is
            // *not* followed by a path stays a plain asset, because that is
            // what a texture attribute holds.
            Some(Token::Asset(v)) => match self.peek() {
                Some(Token::Path(_)) => {
                    let Some(Token::Path(prim_path)) = self.next() else {
                        unreachable!()
                    };
                    Ok(UsdValue::Reference(self.layer_offset(UsdReference::new(
                        v, prim_path,
                    ))))
                }
                // `@clip.usda@ (offset = 100)` is an arc with a time shift and
                // no prim named. The parentheses cannot be consumed on sight —
                // an `asset` attribute is followed by its own metadata block in
                // exactly the same position — so the decision is made on what
                // is inside them.
                _ if self.at_layer_offset() => {
                    Ok(UsdValue::Reference(self.layer_offset(UsdReference::new(v, ""))))
                }
                _ => Ok(UsdValue::Asset(v)),
            },
            Some(Token::Path(v)) => Ok(UsdValue::Path(v)),
            Some(Token::Ident(v)) => Ok(match v.as_str() {
                "true" | "True" => UsdValue::Bool(true),
                "false" | "False" => UsdValue::Bool(false),
                // An explicit block, which is not the same as saying nothing.
                "None" => UsdValue::Block,
                _ => UsdValue::Token(v),
            }),
            Some(Token::Punct('[')) => {
                let mut items = Vec::new();
                while !self.eat_punct(']') {
                    if self.peek().is_none() {
                        return Err(self.err("unterminated array"));
                    }
                    if self.eat_punct(',') {
                        continue;
                    }
                    items.push(self.value()?);
                }
                Ok(UsdValue::Array(items))
            }
            Some(Token::Punct('(')) => {
                let mut items = Vec::new();
                while !self.eat_punct(')') {
                    if self.peek().is_none() {
                        return Err(self.err("unterminated tuple"));
                    }
                    if self.eat_punct(',') {
                        continue;
                    }
                    items.push(self.value()?);
                }
                Ok(UsdValue::Tuple(items))
            }
            Some(Token::Punct('{')) => {
                let mut entries = Vec::new();
                while !self.eat_punct('}') {
                    if self.peek().is_none() {
                        return Err(self.err("unterminated dictionary"));
                    }
                    if self.eat_punct(',') {
                        continue;
                    }
                    // `relocates` is a map of path to path — `<a>: <b>` — which
                    // is a different shape from the `type name = value` every
                    // other dictionary uses.
                    if let Some(Token::Path(from)) = self.peek().cloned() {
                        self.at += 1;
                        self.expect_punct(':')?;
                        let to = self.value()?;
                        entries.push((from, to));
                        continue;
                    }
                    // Entries are `[type] name = value`; the type is optional
                    // and carries no meaning we need.
                    let mut key = String::new();
                    let mut declared = String::new();
                    while let Some(Token::Ident(w)) = self.peek().cloned() {
                        self.at += 1;
                        if !key.is_empty() {
                            declared = key.clone();
                        }
                        key = w;
                        // An array type is written `asset[] assetPaths`, and
                        // the brackets are two separate tokens — without
                        // stepping over them the type is taken for the name
                        // and the `=` is never reached.
                        if self.eat_punct('[') {
                            self.eat_punct(']');
                        }
                        if matches!(self.peek(), Some(Token::Punct('='))) {
                            break;
                        }
                    }
                    if !self.eat_punct('=') {
                        return Err(self.err("expected = in dictionary"));
                    }
                    // A dictionary entry states its type and the value does
                    // not carry one, so `double[] n = [1, 2, 3]` would be
                    // taken for integers on the way out. Coercing here keeps
                    // the declaration without widening what a value is.
                    let mut value = self.value()?;
                    if declared.starts_with("double")
                        || declared.starts_with("float")
                        || declared.starts_with("half")
                        || declared.starts_with("timecode")
                    {
                        value = as_floating(value);
                    } else if declared.starts_with("token") || declared.starts_with("asset") {
                        // A quoted value under a `token` declaration is a
                        // token, not a string — and the two are written
                        // differently on the way back out.
                        value = as_named(value, declared.starts_with("asset"));
                    }
                    entries.push((key, value));
                }
                Ok(UsdValue::Dict(entries))
            }
            Some(_) => Err(self.err("unexpected token in value")),
        }
    }
}

impl Specifier {
    pub(super) fn as_keyword(self) -> &'static str {
        self.keyword()
    }
}

#[cfg(test)]
mod tests {
    /// A boolean written as a digit is a boolean.
    ///
    /// `usdcat` always writes `1` and `0`, never `true` and `false`, so every
    /// boolean attribute in a file OpenUSD wrote arrives as an integer unless
    /// the declared type is consulted. It read back as the number one, and
    /// `doubleSided` on a mesh Apple's tools had touched quietly stopped
    /// meaning anything.
    #[test]
    fn a_boolean_written_as_a_digit_is_a_boolean() {
        let layer = parse(
            r#"#usda 1.0

def Mesh "M"
{
    uniform bool doubleSided = 1
    bool off = 0
    bool spelt = true
    bool[] many = [0, 1, 1]
    int notABool = 1
}
"#,
        )
        .unwrap();
        let prim = layer.prim_at("/M").unwrap();
        assert_eq!(prim.value("doubleSided"), Some(&UsdValue::Bool(true)));
        assert_eq!(prim.value("off"), Some(&UsdValue::Bool(false)));
        assert_eq!(prim.value("spelt"), Some(&UsdValue::Bool(true)));
        assert_eq!(
            prim.value("many"),
            Some(&UsdValue::Array(vec![
                UsdValue::Bool(false),
                UsdValue::Bool(true),
                UsdValue::Bool(true),
            ]))
        );
        // And an integer that is an integer stays one.
        assert_eq!(prim.value("notABool"), Some(&UsdValue::Int(1)));
    }

    use super::*;

    const CUBE: &str = r#"#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 0.01
    upAxis = "Y"
)

def Xform "World"
{
    def Mesh "Cube"
    {
        float3[] extent = [(-1, -1, -1), (1, 1, 1)]
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
            interpolation = "vertex"
        )
        uniform token subdivisionScheme = "none"
        rel material:binding = </World/Mat>
    }

    def Material "Mat"
    {
        token outputs:surface.connect = </World/Mat/Shader.outputs:surface>
    }
}
"#;

    #[test]
    fn parses_layer_metadata() {
        let layer = parse(CUBE).unwrap();
        assert_eq!(layer.meta("defaultPrim").unwrap().as_str(), Some("World"));
        assert!((layer.meters_per_unit() - 0.01).abs() < 1e-9);
        assert_eq!(layer.up_axis(), 'Y');
    }

    #[test]
    fn parses_the_prim_tree() {
        let layer = parse(CUBE).unwrap();
        assert_eq!(layer.prims.len(), 1);
        let world = &layer.prims[0];
        assert_eq!(world.type_name, "Xform");
        assert_eq!(world.children.len(), 2);
        let cube = world.child("Cube").unwrap();
        assert_eq!(cube.type_name, "Mesh");
        assert_eq!(cube.specifier, Specifier::Def);
    }

    #[test]
    fn reads_typed_arrays_and_keeps_integers_integral() {
        let layer = parse(CUBE).unwrap();
        let cube = layer.prim_at("/World/Cube").unwrap();
        let points = cube.value("points").unwrap();
        assert_eq!(points.items().len(), 4);
        assert_eq!(points.flat_f32().len(), 12);
        assert_eq!(
            cube.value("faceVertexIndices").unwrap().flat_u32(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(cube.property("points").unwrap().type_name, "point3f[]");
    }

    #[test]
    fn time_samples_become_one_animated_property() {
        let layer = parse(include_str!("testdata/anim.usda")).expect("parses");
        let spin = layer.prim_at("/Spin").unwrap();

        // `xformOp:translate.timeSamples` is the property `xformOp:translate`,
        // animated — not a second property with a longer name.
        assert!(spin.property("xformOp:translate.timeSamples").is_none());
        let translate = spin.value("xformOp:translate").expect("translate");
        let samples = translate.samples().expect("animated");
        assert_eq!(samples.len(), 3);
        assert_eq!(samples[0].0, 0.0);
        assert_eq!(samples[2].0, 48.0);
        assert_eq!(samples[2].1.flat_f32(), vec![10.0, 10.0, 0.0]);

        // Arrays animate the same way.
        let points = layer.prim_at("/Spin/Body").unwrap().value("points").unwrap();
        assert_eq!(points.samples().unwrap().len(), 2);
        assert_eq!(
            points.samples().unwrap()[1].1.flat_f32(),
            vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0]
        );
    }

    #[test]
    fn an_animated_value_reads_between_its_samples() {
        let layer = parse(include_str!("testdata/anim.usda")).unwrap();
        let translate = layer.prim_at("/Spin").unwrap().value("xformOp:translate").unwrap();

        // Held before the first sample and after the last.
        assert_eq!(translate.at_time(-5.0).flat_f32(), vec![0.0, 0.0, 0.0]);
        assert_eq!(translate.at_time(100.0).flat_f32(), vec![10.0, 10.0, 0.0]);
        // On a sample, and between two.
        assert_eq!(translate.at_time(24.0).flat_f32(), vec![10.0, 0.0, 0.0]);
        assert_eq!(translate.at_time(36.0).flat_f32(), vec![10.0, 5.0, 0.0]);
    }

    /// A list operation belongs to the field it qualifies, not to itself.
    #[test]
    fn a_list_op_qualifier_stays_with_its_field() {
        let layer = parse(
            r#"#usda 1.0
def Mesh "M" (
    prepend apiSchemas = ["MaterialBindingAPI"]
)
{
}
"#,
        )
        .unwrap();
        let m = layer.prim_at("/M").unwrap();
        assert_eq!(
            m.meta("prepend apiSchemas").unwrap().flat_tokens(),
            vec!["MaterialBindingAPI"]
        );

        // And it comes back out on one line, the way it went in.
        let text = super::super::write::layer_to_usda(&layer);
        assert!(
            text.contains(r#"prepend apiSchemas = ["MaterialBindingAPI"]"#),
            "{text}"
        );
    }

    #[test]
    fn property_metadata_and_uniform_survive() {
        let layer = parse(CUBE).unwrap();
        let cube = layer.prim_at("/World/Cube").unwrap();
        let st = cube.property("primvars:st").unwrap();
        assert_eq!(st.meta("interpolation").unwrap().as_str(), Some("vertex"));
        assert!(cube.property("subdivisionScheme").unwrap().uniform);
    }

    #[test]
    fn relationships_are_paths_not_strings() {
        let layer = parse(CUBE).unwrap();
        let cube = layer.prim_at("/World/Cube").unwrap();
        let binding = cube.property("material:binding").unwrap();
        assert!(binding.relationship);
        assert_eq!(binding.value, UsdValue::Path("/World/Mat".into()));
    }

    #[test]
    fn paths_resolve_through_the_tree() {
        let layer = parse(CUBE).unwrap();
        assert!(layer.prim_at("/World/Mat").is_some());
        assert!(layer.prim_at("/World/Nope").is_none());
        assert_eq!(layer.prims_by_path().len(), 3);
    }

    #[test]
    fn a_connection_is_the_attribute_it_connects() {
        let layer = parse(CUBE).unwrap();
        let mat = layer.prim_at("/World/Mat").unwrap();
        // Not a property called `outputs:surface.connect` — a dot inside a
        // property name is not something USD can hold.
        assert!(mat.property("outputs:surface.connect").is_none());
        let surface = mat.property("outputs:surface").expect("the connected attribute");
        assert!(!surface.relationship, "a connection is not a relationship");
        assert!(matches!(surface.value, UsdValue::Path(_)), "{:?}", surface.value);
    }

    /// A declaration and a connection for one name are one property.
    #[test]
    fn a_declaration_and_its_connection_are_one_property() {
        let layer = parse(
            r#"#usda 1.0
def Shader "S"
{
    token outputs:surface
    token outputs:surface.connect = </Other.outputs:surface>
}
"#,
        )
        .unwrap();
        let s = layer.prim_at("/S").unwrap();
        assert_eq!(s.properties.len(), 1, "{:?}", s.properties);
        assert_eq!(s.properties[0].type_name, "token");
        assert_eq!(s.properties[0].value.as_str(), Some("/Other.outputs:surface"));
    }

    #[test]
    fn an_unterminated_prim_is_an_error_not_a_panic() {
        assert!(parse("def Xform \"a\" {").is_err());
        assert!(parse("def Xform \"a\" { float x = }").is_err());
    }
}
