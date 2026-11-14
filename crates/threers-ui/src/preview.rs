//! A component built on its own, in several variants, and written down:
//! in the light and the dark appearance, at a few window sizes and text
//! sizes. [`Preview::render`] gives a text
//! snapshot of each — every node that means something to assistive
//! technology, with its role, name, state, box and colours — and
//! [`assert_snapshot`] checks it against one kept in a file, so a change
//! that moves a dialog off-centre or leaves a panel dark in the light
//! appearance shows up as a diff.
//!
//! ```
//! use threers_ui::preview::{snapshot, Preview};
//! use threers_ui::{style, Role, Semantics};
//!
//! let out = Preview::new().size(300.0, 100.0).render(|ui, _input, _state: &mut ()| {
//!     let root = ui.root();
//!     let b = ui.add_box(root, style! { width: 80px; height: 20px; background: rgb(37, 99, 235); });
//!     ui.set_semantics(b, Semantics::new(Role::Button).label("OK"));
//! });
//! assert!(out.contains("button \"OK\" @0,0 80x20 bg #2563eb"));
//! ```

use std::fmt::Write;
use std::path::Path;

use crate::input::Input;
use crate::states::Appearance;
use crate::tree::{NodeId, Ui};
use crate::{FixedWidthText, Rgba};

/// Variants to build a component in.
#[derive(Debug, Clone)]
pub struct Preview {
    pub sizes: Vec<(f32, f32)>,
    pub appearances: Vec<Appearance>,
    pub text_scales: Vec<f32>,
}

impl Default for Preview {
    fn default() -> Self {
        Self::new()
    }
}

impl Preview {
    /// Dark and light, at 600×400, text at its size.
    pub fn new() -> Self {
        Self {
            sizes: vec![],
            appearances: vec![Appearance::Dark, Appearance::Light],
            text_scales: vec![1.0],
        }
    }

    /// A window size (the first replaces the default).
    pub fn size(mut self, width: f32, height: f32) -> Self {
        self.sizes.push((width, height));
        self
    }

    /// Text sizes besides its own.
    pub fn text_scales(mut self, scales: &[f32]) -> Self {
        self.text_scales = scales.to_vec();
        self
    }

    pub fn appearances(mut self, appearances: &[Appearance]) -> Self {
        self.appearances = appearances.to_vec();
        self
    }

    /// Builds the component with `build` in a fresh UI for each variant,
    /// lays it out (text 7 px a character) and writes each down under a
    /// heading naming the variant.
    pub fn render<S: Default + 'static>(
        &self,
        build: impl Fn(&mut Ui, &mut Input<S>, &mut S),
    ) -> String {
        let sizes = if self.sizes.is_empty() {
            vec![(600.0, 400.0)]
        } else {
            self.sizes.clone()
        };
        let mut out = String::new();
        for &(w, h) in &sizes {
            for &scale in &self.text_scales {
                for &a in &self.appearances {
                    let mut ui = Ui::new();
                    ui.set_system_appearance(a);
                    let mut input: Input<S> = Input::new();
                    let mut state = S::default();
                    ui.layout(w, h, &mut FixedWidthText(7.0));
                    build(&mut ui, &mut input, &mut state);
                    ui.set_text_scale(scale);
                    input.tick(&mut ui, &mut state, 0.0);
                    ui.layout(w, h, &mut FixedWidthText(7.0));
                    let _ = writeln!(out, "== {w}x{h} {a:?} text {scale}");
                    out.push_str(&snapshot(&ui));
                }
            }
        }
        out
    }
}

fn hex(c: Rgba) -> String {
    let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    if c[3] >= 1.0 {
        format!("#{:02x}{:02x}{:02x}", b(c[0]), b(c[1]), b(c[2]))
    } else {
        format!(
            "#{:02x}{:02x}{:02x}{:02x}",
            b(c[0]),
            b(c[1]),
            b(c[2]),
            b(c[3])
        )
    }
}

/// The UI as assistive technology sees it, a line a node, indented by
/// depth: role, name, states, box, and the colours it shows (its
/// background, or the nearest one behind it, and its text colour).
pub fn snapshot(ui: &Ui) -> String {
    let tree = ui.accessibility_tree();
    let mut out = String::new();
    let depth = |id: NodeId| {
        ui.ancestors(id)
            .skip(1)
            .filter(|&a| tree.iter().any(|n| n.id == a))
            .count()
    };
    for n in &tree {
        let _ = write!(
            out,
            "{}{}",
            "  ".repeat(depth(n.id)),
            format!("{:?}", n.role).to_lowercase()
        );
        if !n.name.is_empty() {
            let _ = write!(out, " {:?}", n.name);
        }
        for (flag, on) in [
            ("checked", n.checked),
            ("selected", n.selected),
            ("expanded", n.expanded),
        ] {
            if let Some(v) = on {
                let _ = write!(out, " {flag}={v}");
            }
        }
        if let Some(v) = &n.value {
            let _ = write!(out, " value={v:?}");
        }
        if n.disabled {
            out.push_str(" disabled");
        }
        let b = n.bounds;
        let _ = write!(
            out,
            " @{},{} {}x{}",
            b.x.round(),
            b.y.round(),
            b.width.round(),
            b.height.round()
        );
        let bg = ui
            .ancestors(n.id)
            .find_map(|a| ui.style(a).background.filter(|c| c[3] > 0.0));
        if let Some(bg) = bg {
            let _ = write!(out, " bg {}", hex(bg));
        }
        let _ = writeln!(out, " fg {}", hex(ui.color_of(n.id)));
    }
    out
}

/// Checks `actual` against the snapshot in `path`; with the environment
/// variable `UPDATE_SNAPSHOTS` set (or no file yet), writes it instead.
/// Panics with both on a difference.
pub fn assert_snapshot(path: impl AsRef<Path>, actual: &str) {
    let path = path.as_ref();
    if std::env::var_os("UPDATE_SNAPSHOTS").is_some() || !path.exists() {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        std::fs::write(path, actual).unwrap_or_else(|e| panic!("writing {}: {e}", path.display()));
        return;
    }
    let want =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    if want != actual {
        let first = want
            .lines()
            .zip(actual.lines())
            .position(|(a, b)| a != b)
            .unwrap_or(want.lines().count().min(actual.lines().count()));
        panic!(
            "snapshot {} differs from line {} (UPDATE_SNAPSHOTS=1 to accept):\n--- kept\n{}\n--- now\n{}",
            path.display(),
            first + 1,
            want.lines().skip(first).take(6).collect::<Vec<_>>().join("\n"),
            actual.lines().skip(first).take(6).collect::<Vec<_>>().join("\n"),
        );
    }
}
