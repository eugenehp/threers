//! A container that shows the first of its children that fits its width:
//! the full toolbar where there is room,
//! a shorter one where there is less, an icon menu where there is none.
//! Each child is measured at its min-content width (its text unwrapped
//! where it does not wrap, its row's items side by side); the last shows
//! when none fits. Chosen after each layout, and laid out again when the
//! choice changes.
//!
//! ```
//! use threers_ui::{style, FixedWidthText, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let bar = ui.add_box(root, style! { display: flex; });
//! let long = ui.add_box(bar, style! {});
//! ui.add_text(long, "Save, Share and Export");
//! let short = ui.add_box(bar, style! {});
//! ui.add_text(short, "Save");
//! ui.view_that_fits(bar);
//! ui.layout(400.0, 100.0, &mut FixedWidthText(7.0));
//! assert!(ui.rect(long).width > 0.0 && ui.rect(short).width == 0.0);
//! ui.layout(100.0, 100.0, &mut FixedWidthText(7.0));
//! assert!(ui.rect(long).width == 0.0 && ui.rect(short).width > 0.0);
//! ```

use crate::tree::{NodeId, Ui};
use crate::{Display, TextMeasure};

impl Ui {
    /// Makes `container` show only the first of its children (as they are
    /// now) that fits its content width. See the [module docs](self).
    pub fn view_that_fits(&mut self, container: NodeId) {
        let shown: Vec<Display> = self.children(container).iter().map(|&c| self.style(c).display).collect();
        self.fits.retain(|(c, _)| *c != container);
        self.fits.push((container, shown));
        self.invalidate_layout();
    }

    /// Picks each fitting container's child by what layout found; whether
    /// any changed.
    pub(crate) fn apply_fits(&mut self, text: &mut dyn TextMeasure) -> bool {
        if self.fits.is_empty() {
            return false;
        }
        let fits = std::mem::take(&mut self.fits);
        let mut changed = false;
        let mut keep = vec![];
        for (c, shown) in fits {
            if !self.contains(c) {
                continue;
            }
            let kids: Vec<NodeId> = self.children(c).to_vec();
            let s = self.style(c);
            let room = self.rect(c).width - s.padding.horizontal() - s.border.horizontal();
            // Measured as it shows, hidden or not.
            let mut pick = kids.len().saturating_sub(1);
            for (i, &k) in kids.iter().enumerate() {
                let own = shown.get(i).copied().unwrap_or(Display::Block);
                let now = self.nodes[k.0].style.display;
                self.nodes[k.0].style.display = own;
                let w = crate::layout::min_content_width_of(self, k, text) + self.style(k).margin.horizontal();
                self.nodes[k.0].style.display = now;
                if w <= room + 0.01 {
                    pick = i;
                    break;
                }
            }
            for (i, &k) in kids.iter().enumerate() {
                let want = if i == pick { shown.get(i).copied().unwrap_or(Display::Block) } else { Display::None };
                if self.style(k).display != want {
                    self.update_style(k, |s| s.display = want);
                    changed = true;
                }
            }
            keep.push((c, shown));
        }
        self.fits = keep;
        changed
    }
}
