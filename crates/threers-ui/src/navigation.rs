//! A list worked from the keyboard: a
//! highlighted row the arrows, Home, End, Page Up and Page Down move (and
//! scroll into view), and Enter or Space activates. The app draws the
//! highlight and gives the list's box focus (role `listbox`, its rows
//! `option`s with `selected`, so a screen reader follows along).
//!
//! ```
//! use threers_ui::navigation::{ListNavigation, Nav};
//!
//! let mut nav = ListNavigation::new(100);
//! assert_eq!(nav.key("ArrowDown", 10), Some(Nav::Moved(0)));
//! assert_eq!(nav.key("End", 10), Some(Nav::Moved(99)));
//! assert_eq!(nav.key("PageUp", 10), Some(Nav::Moved(89)));
//! assert_eq!(nav.key("Enter", 10), Some(Nav::Activated(89)));
//! ```

use crate::scrolling::ScrollStrategy;
use crate::tree::Ui;
use crate::virtual_list::VirtualList;

/// What a key did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nav {
    /// The highlight moved to this row.
    Moved(usize),
    /// This row was activated (Enter, Space).
    Activated(usize),
}

/// The highlighted row of a list of `len` rows, moved by keys.
#[derive(Debug, Clone, Default)]
pub struct ListNavigation {
    pub len: usize,
    pub highlighted: Option<usize>,
}

impl ListNavigation {
    pub fn new(len: usize) -> Self {
        Self { len, highlighted: None }
    }

    /// A key (DOM `KeyboardEvent.key`), with `page` rows to a page. `None`
    /// for a key it does not use.
    pub fn key(&mut self, key: &str, page: usize) -> Option<Nav> {
        if self.len == 0 {
            return None;
        }
        let last = self.len - 1;
        let page = page.max(1);
        let to = match (key, self.highlighted) {
            ("Enter" | " ", Some(i)) => return Some(Nav::Activated(i)),
            ("ArrowDown", None) | ("Home", _) => 0,
            ("ArrowUp", None) | ("End", _) => last,
            ("ArrowDown", Some(i)) => (i + 1).min(last),
            ("ArrowUp", Some(i)) => i.saturating_sub(1),
            ("PageDown", i) => i.map_or(0, |i| (i + page).min(last)),
            ("PageUp", i) => i.map_or(0, |i| i.saturating_sub(page)),
            _ => return None,
        };
        self.highlighted = Some(to);
        Some(Nav::Moved(to))
    }

    /// [`key`](Self::key) on a virtual list: its page is the rows its view
    /// holds, and the highlighted row is scrolled into view.
    pub fn key_list(&mut self, ui: &mut Ui, list: &mut VirtualList, key: &str) -> Option<Nav> {
        self.len = list.len();
        let (_, view, _) = ui.scroll_extent(list.container());
        let row = (list.content_height() / list.len().max(1) as f32).max(1.0);
        let nav = self.key(key, (view / row).floor() as usize)?;
        if let Nav::Moved(i) = nav {
            list.scroll_to_with(ui, i, ScrollStrategy::Nearest);
        }
        Some(nav)
    }
}
