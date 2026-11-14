//! Icons from SVG markup: the inside of an `<svg>` element — `path`, `line`,
//! `circle`, `ellipse`, `rect` (rounded too), `polyline` and `polygon` —
//! turned into path nodes, so an icon set written for the web (Feather,
//! Lucide, an app's own) is used as it is.
//!
//! ```
//! use threers_ui::{style, svg, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let zoom = r#"<circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>"#;
//! let icon = svg::add_icon(&mut ui, root, style! { width: 20px; height: 20px; }, zoom, 2.0);
//! assert_eq!(ui.children(icon).len(), 1);
//! ```
//!
//! Shapes are stroked in the node's colour (`currentColor`) with the stroke
//! width given, as `stroke="currentColor" fill="none"` on the `<svg>` would
//! draw them; a shape with `fill="currentColor"` (or that in its `style`)
//! is filled instead. Other attributes are ignored.

use crate::{style, NodeId, Style, Ui};

/// One shape of an icon, as path data.
#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    pub d: String,
    /// Filled rather than stroked.
    pub fill: bool,
}

/// The shapes in `markup`, in order.
pub fn parse(markup: &str) -> Vec<Shape> {
    let mut out = Vec::new();
    let mut rest = markup;
    while let Some(open) = rest.find('<') {
        rest = &rest[open + 1..];
        let Some(close) = rest.find('>') else { break };
        let tag = rest[..close].trim_end_matches('/').trim();
        rest = &rest[close + 1..];
        if tag.starts_with('/') || tag.starts_with('!') || tag.starts_with('?') {
            continue;
        }
        let (name, attrs) = tag.split_once(char::is_whitespace).unwrap_or((tag, ""));
        let attrs = attributes(attrs);
        let get = |k: &str| attrs.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str());
        let num = |k: &str| {
            get(k)
                .and_then(|v| v.trim().trim_end_matches("px").parse::<f32>().ok())
                .unwrap_or(0.0)
        };
        let d = match name {
            "path" => get("d").map(str::to_string),
            "line" => Some(format!(
                "M{} {}L{} {}",
                num("x1"),
                num("y1"),
                num("x2"),
                num("y2")
            )),
            "circle" => Some(crate::paint::circle_path(num("cx"), num("cy"), num("r"))),
            "ellipse" => {
                let (cx, cy, rx, ry) = (num("cx"), num("cy"), num("rx"), num("ry"));
                Some(format!(
                    "M{} {cy}A{rx} {ry} 0 1 1 {} {cy}A{rx} {ry} 0 1 1 {} {cy}Z",
                    cx + rx,
                    cx - rx,
                    cx + rx
                ))
            }
            "rect" => {
                let (x, y, w, h) = (num("x"), num("y"), num("width"), num("height"));
                let rx = get("rx")
                    .map(|_| num("rx"))
                    .or(get("ry").map(|_| num("ry")))
                    .unwrap_or(0.0);
                let ry = get("ry").map(|_| num("ry")).unwrap_or(rx);
                let (rx, ry) = (rx.min(w / 2.0), ry.min(h / 2.0));
                Some(if rx > 0.0 && ry > 0.0 {
                    format!(
                        "M{} {y}H{}A{rx} {ry} 0 0 1 {} {}V{}A{rx} {ry} 0 0 1 {} {}H{}A{rx} {ry} 0 0 1 {x} {}V{}A{rx} {ry} 0 0 1 {} {y}Z",
                        x + rx,
                        x + w - rx,
                        x + w,
                        y + ry,
                        y + h - ry,
                        x + w - rx,
                        y + h,
                        x + rx,
                        y + h - ry,
                        y + ry,
                        x + rx,
                    )
                } else {
                    format!("M{x} {y}H{}V{}H{x}Z", x + w, y + h)
                })
            }
            "polyline" | "polygon" => {
                let n: Vec<f32> = get("points")
                    .unwrap_or("")
                    .split(|c: char| c == ',' || c.is_whitespace())
                    .filter_map(|v| v.parse().ok())
                    .collect();
                (n.len() >= 4).then(|| {
                    let mut d = String::new();
                    for (i, p) in n.as_chunks::<2>().0.iter().enumerate() {
                        d.push_str(&format!(
                            "{}{} {}",
                            if i == 0 { 'M' } else { 'L' },
                            p[0],
                            p[1]
                        ));
                    }
                    if name == "polygon" {
                        d.push('Z');
                    }
                    d
                })
            }
            _ => None,
        };
        let Some(d) = d else { continue };
        let filled = |v: &str| {
            let v = v.trim();
            !v.is_empty() && v != "none"
        };
        let style_fill = get("style").and_then(|s| {
            s.split(';').find_map(|decl| {
                let (k, v) = decl.split_once(':')?;
                (k.trim() == "fill").then(|| filled(v))
            })
        });
        let fill = style_fill.or(get("fill").map(filled)).unwrap_or(false);
        out.push(Shape { d, fill });
    }
    out
}

/// `name="value"` pairs (either quote).
fn attributes(s: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut rest = s.trim();
    while let Some(eq) = rest.find('=') {
        let name = rest[..eq].trim().to_string();
        let after = rest[eq + 1..].trim_start();
        let Some(q) = after.chars().next().filter(|c| *c == '"' || *c == '\'') else {
            break;
        };
        let body = &after[1..];
        let Some(end) = body.find(q) else { break };
        out.push((name, body[..end].to_string()));
        rest = body[end + 1..].trim_start();
    }
    out
}

/// A box styled `style` (give it a size) showing `markup` drawn in a 24 × 24
/// view box — the web's icon size — its strokes `stroke` view-box units
/// wide. See [`add_icon_in`] for another view box.
pub fn add_icon(ui: &mut Ui, parent: NodeId, style: Style, markup: &str, stroke: f32) -> NodeId {
    add_icon_in(ui, parent, style, markup, stroke, [0.0, 0.0, 24.0, 24.0])
}

/// As [`add_icon`], in `view_box` (`[x, y, width, height]`).
pub fn add_icon_in(
    ui: &mut Ui,
    parent: NodeId,
    mut style: Style,
    markup: &str,
    stroke: f32,
    view_box: [f32; 4],
) -> NodeId {
    let shapes = parse(markup);
    style.position = style! { position: relative; }.position;
    let icon = ui.add_box(parent, style);
    // the stroked shapes in one node, the filled in another, over them
    for fill in [false, true] {
        let d: Vec<String> = shapes
            .iter()
            .filter(|s| s.fill == fill)
            .map(|s| s.d.clone())
            .collect();
        if d.is_empty() {
            continue;
        }
        let layer = style! { position: absolute; left: 0px; top: 0px; width: 100%; height: 100%; };
        ui.add_path(icon, layer, d, view_box, (!fill).then_some(stroke), fill);
    }
    icon
}

/// Replace the drawing of an icon [`add_icon`] made.
pub fn set_icon(ui: &mut Ui, icon: NodeId, markup: &str, stroke: f32) {
    let shapes = parse(markup);
    for child in ui.children(icon).to_vec() {
        ui.remove(child);
    }
    for fill in [false, true] {
        let d: Vec<String> = shapes
            .iter()
            .filter(|s| s.fill == fill)
            .map(|s| s.d.clone())
            .collect();
        if d.is_empty() {
            continue;
        }
        let layer = style! { position: absolute; left: 0px; top: 0px; width: 100%; height: 100%; };
        ui.add_path(
            icon,
            layer,
            d,
            [0.0, 0.0, 24.0, 24.0],
            (!fill).then_some(stroke),
            fill,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn web_icon_markup_becomes_path_data() {
        let s = parse(
            r#"<rect x="3" y="4" width="18" height="16" rx="2"/><line x1="9" y1="4" x2="9" y2="20"/>
               <polygon points="10 8.2 16 12 10 15.8 10 8.2" style="fill:currentColor"/>
               <path d="M12 3a9 9 0 0 1 0 18z" fill="currentColor"/><polyline points="20 6 9 17 4 12"/>"#,
        );
        assert_eq!(s.len(), 5);
        assert!(s[0].d.starts_with("M5 4H19A2 2 0 0 1 21 6"), "{}", s[0].d);
        assert_eq!(
            s[1],
            Shape {
                d: "M9 4L9 20".into(),
                fill: false
            }
        );
        assert_eq!(
            s[2],
            Shape {
                d: "M10 8.2L16 12L10 15.8L10 8.2Z".into(),
                fill: true
            }
        );
        assert!(s[3].fill);
        assert_eq!(
            s[4],
            Shape {
                d: "M20 6L9 17L4 12".into(),
                fill: false
            }
        );
    }

    #[test]
    fn an_icon_is_a_stroked_and_a_filled_layer() {
        let mut ui = Ui::new();
        let root = ui.root();
        let icon = add_icon(
            &mut ui,
            root,
            style! { width: 24px; height: 24px; },
            r#"<circle cx="12" cy="12" r="9"/><polygon points="10 8 16 12 10 16" fill="currentColor"/>"#,
            2.0,
        );
        assert_eq!(ui.children(icon).len(), 2);
    }
}
