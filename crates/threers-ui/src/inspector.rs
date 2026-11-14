//! What is this, and where did it come from: point at a box and get its
//! node, its place in the tree, its style and the line of source that added
//! it, as a browser's inspector gives them.
//!
//! ```
//! use threers_ui::{style, FixedWidthText, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let panel = ui.add_box(root, style! { width: 120px; height: 40px; padding: 4px; background: #222; });
//! ui.layout(300.0, 200.0, &mut FixedWidthText(7.0));
//! let found = ui.inspect(10.0, 10.0).unwrap();
//! assert_eq!(found.node, panel);
//! assert!(found.origin.unwrap().file().ends_with(".rs"));
//! assert!(found.to_string().contains("padding"));
//! ```

use std::fmt;
use std::panic::Location;

use super::a11y::Semantics;
use super::paint::Painter;
use super::style::{Cursor, Rgba, Style};
use super::tree::{NodeId, NodeKind, Rect, Ui};

/// A node as the inspector shows it.
#[derive(Debug, Clone)]
pub struct Inspection {
    pub node: NodeId,
    /// From the root down to the node.
    pub path: Vec<NodeId>,
    /// The border box, as last laid out.
    pub rect: Rect,
    /// The content box: inside the border and padding.
    pub content: Rect,
    pub style: Style,
    pub kind: NodeKind,
    pub semantics: Option<Semantics>,
    pub cursor: Cursor,
    /// The `add_box` / `add_text` / `add_path` call (or the `ui!` macro)
    /// that made it.
    pub origin: Option<&'static Location<'static>>,
    /// Painted after the rest (see [`Ui::defer`]).
    pub deferred: Option<i32>,
}

impl Ui {
    /// The node at `(x, y)`, described. See the [module docs](self).
    pub fn inspect(&self, x: f32, y: f32) -> Option<Inspection> {
        self.hit_test(x, y).map(|n| self.inspect_node(n))
    }

    /// `id`, described.
    pub fn inspect_node(&self, id: NodeId) -> Inspection {
        let n = self.node(id);
        let s = &n.style;
        let r = n.rect;
        let mut path: Vec<NodeId> = self.ancestors(id).collect();
        path.reverse();
        Inspection {
            node: id,
            path,
            rect: r,
            content: Rect {
                x: r.x + s.border.left + s.padding.left,
                y: r.y + s.border.top + s.padding.top,
                width: (r.width - s.border.horizontal() - s.padding.horizontal()).max(0.0),
                height: (r.height - s.border.vertical() - s.padding.vertical()).max(0.0),
            },
            style: s.clone(),
            kind: n.kind.clone(),
            semantics: n.semantics.clone(),
            cursor: self.cursor_of(id),
            origin: n.origin,
            deferred: n.deferred,
        }
    }

    /// Paints a browser-style highlight of `id` over a frame: the margin
    /// box in orange, the padding in green and the content in blue,
    /// translucent.
    pub fn paint_highlight(&self, id: NodeId, painter: &mut dyn Painter) {
        let i = self.inspect_node(id);
        let m = i.style.margin;
        let r = i.rect;
        let margin = Rect {
            x: r.x - m.left,
            y: r.y - m.top,
            width: r.width + m.horizontal(),
            height: r.height + m.vertical(),
        };
        const MARGIN: Rgba = [0.98, 0.62, 0.24, 0.35];
        const PADDING: Rgba = [0.55, 0.82, 0.45, 0.35];
        const CONTENT: Rgba = [0.36, 0.6, 0.94, 0.45];
        // Each band, then the box inside it, so they do not stack.
        bands(painter, margin, r, MARGIN);
        let b = i.style.border;
        let padding_box = Rect {
            x: r.x + b.left,
            y: r.y + b.top,
            width: (r.width - b.horizontal()).max(0.0),
            height: (r.height - b.vertical()).max(0.0),
        };
        bands(painter, padding_box, i.content, PADDING);
        painter.fill_rect(i.content, CONTENT);
    }
}

/// Fills `outer` outside `inner`.
fn bands(painter: &mut dyn Painter, outer: Rect, inner: Rect, color: Rgba) {
    let top = inner.y - outer.y;
    let bottom = outer.y + outer.height - inner.y - inner.height;
    let left = inner.x - outer.x;
    let right = outer.x + outer.width - inner.x - inner.width;
    for r in [
        Rect {
            height: top,
            ..outer
        },
        Rect {
            y: inner.y + inner.height,
            height: bottom,
            ..outer
        },
        Rect {
            y: inner.y,
            width: left,
            height: inner.height,
            ..outer
        },
        Rect {
            x: inner.x + inner.width,
            y: inner.y,
            width: right,
            height: inner.height,
        },
    ] {
        if r.width > 0.0 && r.height > 0.0 {
            painter.fill_rect(r, color);
        }
    }
}

impl fmt::Display for Inspection {
    /// The node, where it came from, its box, and the style properties that
    /// differ from the default.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match &self.kind {
            NodeKind::Box => "box".to_string(),
            NodeKind::Text(t) => format!("text {t:?}"),
            NodeKind::Path { .. } => "path".to_string(),
            NodeKind::Image { image, .. } => format!("image {}×{}", image.width, image.height),
            NodeKind::Surface { id, width, height } => format!("surface {id} {width}×{height}"),
        };
        write!(f, "{kind} {:?}", self.node)?;
        if let Some(o) = self.origin {
            write!(f, " at {}:{}", o.file(), o.line())?;
        }
        let r = self.rect;
        writeln!(
            f,
            "\n  box {} × {} at ({}, {})",
            r.width, r.height, r.x, r.y
        )?;
        if let Some(s) = &self.semantics {
            writeln!(
                f,
                "  role {:?}{}",
                s.role,
                s.label
                    .as_ref()
                    .map(|l| format!(" {l:?}"))
                    .unwrap_or_default()
            )?;
        }
        if let Some(p) = self.deferred {
            writeln!(f, "  deferred, priority {p}")?;
        }
        // The style: each field whose value differs from a default style's.
        let defaults = fields(&Style::default());
        for field in fields(&self.style) {
            if !defaults.contains(&field) {
                writeln!(f, "  {field}")?;
            }
        }
        Ok(())
    }
}

/// A style's fields, one `name: value` string each, from its debug form.
fn fields(style: &Style) -> Vec<String> {
    let pretty = format!("{style:#?}");
    let mut out: Vec<String> = vec![];
    for line in pretty.lines().skip(1) {
        // Top-level fields are indented by four; deeper lines continue one.
        let top = line.starts_with("    ") && !line.starts_with("     ");
        let t = line.trim();
        let closes = t.starts_with(['}', ')', ']']);
        if top && !closes {
            out.push(t.to_string());
        } else if let Some(last) = out.last_mut() {
            last.push(' ');
            last.push_str(t);
        }
    }
    // `padding: Edges { top: 2.0, …, left: 2.0, },` on one line, tidied.
    out.into_iter()
        .map(|f| {
            f.trim_end_matches(',')
                .replace(", }", " }")
                .replace(", ]", "]")
                .replace(", )", ")")
                .replace("( ", "(")
                .replace(" )", ")")
                .replace("[ ", "[")
        })
        .collect()
}
