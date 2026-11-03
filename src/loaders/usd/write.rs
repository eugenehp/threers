//! Writing a layer back out as `.usda`.
//!
//! The output is meant to be read by a person as well as by USD: four-space
//! indentation, one property per line, arrays broken across lines once they are
//! long enough that a single line stops being legible.

use super::parse::{UsdLayer, UsdPrim, UsdProperty};
use super::value::UsdValue;
use std::fmt::Write as _;

/// How many array elements fit on one line before it is broken up.
const WRAP_AT: usize = 8;

/// Format a float the way a `.usda` file does: short, and never in a form USD
/// cannot read back.
///
/// `{}` on an `f64` gives `1` for 1.0, which USD would read as an integer — so
/// whole numbers keep a trailing `.0` wherever the declared type is a float.
/// Infinities are spelled out, because that is what an empty mesh's extent is.
pub fn number(v: f64) -> String {
    if v.is_nan() {
        return "nan".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if v == v.trunc() && v.abs() < 1e15 {
        format!("{v:.1}")
    } else {
        let mut s = format!("{v}");
        if !s.contains('.') && !s.contains('e') && !s.contains('E') {
            s.push_str(".0");
        }
        s
    }
}

/// Serialise a layer as a `.usda` document.
pub fn layer_to_usda(layer: &UsdLayer) -> String {
    let mut out = String::from("#usda 1.0\n");
    if !layer.metadata.is_empty() {
        out.push_str("(\n");
        for (k, v) in &layer.metadata {
            if k.is_empty() {
                continue;
            }
            match v {
                UsdValue::None => {
                    let _ = writeln!(out, "    {k}");
                }
                _ => {
                    let _ = writeln!(out, "    {k} = {}", render(v, 1));
                }
            }
        }
        out.push_str(")\n");
    }
    out.push('\n');
    for prim in &layer.prims {
        write_prim(&mut out, prim, 0);
    }
    out
}

fn indent(out: &mut String, depth: usize) {
    for _ in 0..depth {
        out.push_str("    ");
    }
}

fn write_prim(out: &mut String, prim: &UsdPrim, depth: usize) {
    indent(out, depth);
    let _ = write!(out, "{} ", prim.specifier.as_keyword());
    if !prim.type_name.is_empty() {
        let _ = write!(out, "{} ", prim.type_name);
    }
    let _ = write!(out, "\"{}\"", prim.name);
    if prim.metadata.iter().any(|(k, _)| !k.is_empty() && !k.starts_with("reorder ")) {
        out.push('\n');
        indent(out, depth);
        out.push_str("(\n");
        for (k, v) in &prim.metadata {
            // A reorder statement lives in the body, not the header.
            if k.is_empty() || k.starts_with("reorder ") {
                continue;
            }
            indent(out, depth + 1);
            // The comment field is written as a bare string, which is the
            // form it is read from.
            if k == "comment" {
                let _ = writeln!(out, "{v}");
                continue;
            }
            match v {
                UsdValue::None => {
                    let _ = writeln!(out, "{k}");
                }
                _ => {
                    let _ = writeln!(out, "{k} = {}", render(v, depth + 1));
                }
            }
        }
        indent(out, depth);
        out.push(')');
    }
    out.push('\n');
    indent(out, depth);
    out.push_str("{\n");

    // `reorder` comes first in the body, which is where USD writes it and
    // where it reads naturally: the order before the things being ordered.
    for (key, value) in &prim.metadata {
        if let Some(what) = key.strip_prefix("reorder ") {
            indent(out, depth + 1);
            let _ = writeln!(out, "reorder {what} = {}", render(value, depth + 1));
        }
    }
    for property in &prim.properties {
        write_property(out, property, depth + 1);
    }
    if !prim.properties.is_empty() && !prim.children.is_empty() {
        out.push('\n');
    }
    for (i, child) in prim.children.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        write_prim(out, child, depth + 1);
    }

    indent(out, depth);
    out.push_str("}\n");
}

fn write_property(out: &mut String, p: &UsdProperty, depth: usize) {
    indent(out, depth);
    // An animated attribute is written under its own name with a `.timeSamples`
    // suffix, and a connected one with `.connect` — the spellings the parser
    // undoes on the way back in. A relationship keeps its path as a plain
    // value, which is why it is excluded here.
    let animated = if p.value.samples().is_some() {
        ".timeSamples"
    } else if !p.relationship && matches!(p.value, UsdValue::Path(_)) {
        ".connect"
    } else {
        ""
    };
    // `custom` says the property belongs to no schema. It is written in front
    // of everything else and kept as a field rather than as syntax.
    if p.metadata.iter().any(|(k, v)| k == "custom" && *v == UsdValue::Bool(true)) {
        out.push_str("custom ");
    }
    if p.relationship {
        // `prepend rel foo` and `rel foo` compose differently, so the word is
        // written back where it was said.
        if !p.qualifier.is_empty() {
            let _ = write!(out, "{} ", p.qualifier);
        }
        let _ = write!(out, "rel {}", p.name);
    } else {
        if p.uniform {
            out.push_str("uniform ");
        }
        if p.type_name.is_empty() {
            let _ = write!(out, "{}{animated}", p.name);
        } else {
            let _ = write!(out, "{} {}{animated}", p.type_name, p.name);
        }
    }
    if !matches!(p.value, UsdValue::None) {
        let _ = write!(out, " = {}", render(&p.value, depth));
    }
    if p.metadata.iter().any(|(k, _)| !k.is_empty() && k != "custom") {
        out.push_str(" (\n");
        for (k, v) in &p.metadata {
            // `custom` is written in front of the property, not inside its
            // metadata block.
            if k.is_empty() || k == "custom" {
                continue;
            }
            indent(out, depth + 1);
            // The comment field is written as a bare string, which is the
            // form it is read from.
            if k == "comment" {
                let _ = writeln!(out, "{v}");
                continue;
            }
            match v {
                UsdValue::None => {
                    let _ = writeln!(out, "{k}");
                }
                _ => {
                    let _ = writeln!(out, "{k} = {}", render(v, depth + 1));
                }
            }
        }
        indent(out, depth);
        out.push(')');
    }
    out.push('\n');
}

/// Render a value, breaking long arrays over several lines.
fn render(v: &UsdValue, depth: usize) -> String {
    match v {
        UsdValue::Array(items) if items.len() > WRAP_AT => {
            let mut s = String::from("[\n");
            for (i, item) in items.iter().enumerate() {
                for _ in 0..depth + 1 {
                    s.push_str("    ");
                }
                s.push_str(&item.to_string());
                if i + 1 < items.len() {
                    s.push(',');
                }
                s.push('\n');
            }
            for _ in 0..depth {
                s.push_str("    ");
            }
            s.push(']');
            s
        }
        // Time samples are always one per line, however few there are: a
        // block of frames is read down the page, not across it.
        // A path-keyed dictionary is `relocates`, whose entries are written
        // `<from>: <to>` rather than as assignments.
        UsdValue::Dict(entries)
            if !entries.is_empty() && entries.iter().all(|(k, _)| k.starts_with('/')) =>
        {
            let mut out = String::from("{\n");
            for (from, to) in entries {
                for _ in 0..depth + 1 {
                    out.push_str("    ");
                }
                let _ = writeln!(out, "<{from}>: {to},");
            }
            for _ in 0..depth {
                out.push_str("    ");
            }
            out.push('}');
            out
        }
        UsdValue::TimeSamples(samples) => {
            let mut s = String::from("{\n");
            for (time, value) in samples {
                for _ in 0..depth + 1 {
                    s.push_str("    ");
                }
                let _ = writeln!(s, "{}: {},", number(*time), render(value, depth + 1));
            }
            for _ in 0..depth {
                s.push_str("    ");
            }
            s.push('}');
            s
        }
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::super::parse::parse;
    use super::*;

    #[test]
    fn whole_floats_keep_a_decimal_point() {
        // `1` would read back as an integer, which USD's typed attributes care
        // about even when the arithmetic does not.
        assert_eq!(number(1.0), "1.0");
        assert_eq!(number(-0.5), "-0.5");
        assert_eq!(number(f64::INFINITY), "inf");
        assert_eq!(number(f64::NEG_INFINITY), "-inf");
    }

    #[test]
    fn a_layer_round_trips_through_text() {
        let src = r#"#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
)

def Xform "World"
{
    def Mesh "M"
    {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        uniform token subdivisionScheme = "none"
        rel material:binding = </World/Mat>
    }
}
"#;
        let first = parse(src).unwrap();
        let text = layer_to_usda(&first);
        let second = parse(&text).unwrap();

        assert_eq!(second.meta("defaultPrim").unwrap().as_str(), Some("World"));
        let m = second.prim_at("/World/M").unwrap();
        assert_eq!(m.type_name, "Mesh");
        assert_eq!(m.value("faceVertexIndices").unwrap().flat_u32(), vec![0, 1, 2]);
        assert_eq!(m.value("points").unwrap().flat_f32().len(), 9);
        assert!(m.property("subdivisionScheme").unwrap().uniform);
        assert!(m.property("material:binding").unwrap().relationship);

        // And writing the second one gives the same text as the first: the
        // representation is a fixed point, not merely readable.
        assert_eq!(text, layer_to_usda(&second));
    }

    #[test]
    fn long_arrays_are_broken_over_lines() {
        let src = format!(
            "def Mesh \"M\" {{\n    int[] a = [{}]\n}}\n",
            (0..20).map(|i| i.to_string()).collect::<Vec<_>>().join(", ")
        );
        let text = layer_to_usda(&parse(&src).unwrap());
        assert!(text.contains("[\n"), "long array stayed on one line");
        // …and still reads back.
        assert_eq!(
            parse(&text).unwrap().prim_at("/M").unwrap().value("a").unwrap().flat_u32().len(),
            20
        );
    }
}
