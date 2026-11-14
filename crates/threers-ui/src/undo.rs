//! Undo and redo: an [`UndoManager`] in the app's state keeps, for each
//! change made, how to take it back and how to make it again. Changes can
//! be grouped into one step (a drag that moved several things), and quick
//! repeats of the same kind of change merged (typing undoes a burst at a
//! time, not a letter). A new change after an undo drops what could have
//! been redone. The [`Undo`] and [`Redo`] actions, bound by
//! [`bind_undo_keys`], are for an app's handlers to run it; a menu shows
//! [`UndoManager::undo_name`].
//!
//! ```
//! use threers_ui::undo::UndoManager;
//!
//! struct Doc { text: String, undo: UndoManager<Doc> }
//! let mut doc = Doc { text: String::new(), undo: UndoManager::new() };
//! // A change: made, then recorded with its way back and forward.
//! doc.text.push_str("Hello");
//! doc.undo.record("Typing", |d: &mut Doc| d.text.truncate(0), |d: &mut Doc| d.text.push_str("Hello"));
//! assert_eq!(doc.undo.undo_name(), Some("Typing"));
//! let mut undo = std::mem::take(&mut doc.undo);
//! undo.undo(&mut doc);
//! assert_eq!(doc.text, "");
//! undo.redo(&mut doc);
//! assert_eq!(doc.text, "Hello");
//! ```

use std::rc::Rc;

use crate::keymap::Keymap;

type Change<S> = Rc<dyn Fn(&mut S)>;

/// One change: how to take it back, and how to make it again.
struct Step<S> {
    undo: Change<S>,
    redo: Change<S>,
}

/// What one undo takes back: a change, or several made as one.
struct Entry<S> {
    name: String,
    /// Changes of this kind made soon after merge into it.
    merge: Option<(String, f64)>,
    steps: Vec<Step<S>>,
}

/// Undo and redo for state `S`. See the [module docs](self).
pub struct UndoManager<S> {
    undo: Vec<Entry<S>>,
    redo: Vec<Entry<S>>,
    group: Option<Entry<S>>,
    /// Steps kept; the oldest go past it.
    pub limit: usize,
    /// Seconds within which changes of one kind merge.
    pub merge_window: f64,
}

impl<S> std::fmt::Debug for UndoManager<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UndoManager")
            .field("undo", &self.undo.len())
            .field("redo", &self.redo.len())
            .finish()
    }
}

impl<S> Default for UndoManager<S> {
    fn default() -> Self {
        UndoManager {
            undo: vec![],
            redo: vec![],
            group: None,
            limit: 200,
            merge_window: 1.0,
        }
    }
}

impl<S> UndoManager<S> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a change already made, named for a menu ("Typing", "Move"):
    /// `undo` takes it back, `redo` makes it again.
    pub fn record(
        &mut self,
        name: &str,
        undo: impl Fn(&mut S) + 'static,
        redo: impl Fn(&mut S) + 'static,
    ) {
        self.push(
            name,
            None,
            Step {
                undo: Rc::new(undo),
                redo: Rc::new(redo),
            },
        );
    }

    /// [`record`](Self::record), merged into the last step if that was a
    /// change of the same `kind` made less than
    /// [`merge_window`](Self::merge_window) seconds before `time` (and
    /// nothing has been undone since): one undo takes back the burst.
    pub fn record_merging(
        &mut self,
        name: &str,
        kind: &str,
        time: f64,
        undo: impl Fn(&mut S) + 'static,
        redo: impl Fn(&mut S) + 'static,
    ) {
        self.push(
            name,
            Some((kind.to_owned(), time)),
            Step {
                undo: Rc::new(undo),
                redo: Rc::new(redo),
            },
        );
    }

    fn push(&mut self, name: &str, merge: Option<(String, f64)>, step: Step<S>) {
        self.redo.clear();
        if let Some(g) = &mut self.group {
            g.steps.push(step);
            return;
        }
        if let (Some((kind, time)), Some(last)) = (&merge, self.undo.last_mut())
            && let Some((k, t)) = &last.merge
            && k == kind
            && time - t < self.merge_window
        {
            last.steps.push(step);
            last.merge = merge;
            return;
        }
        self.undo.push(Entry {
            name: name.to_owned(),
            merge,
            steps: vec![step],
        });
        if self.undo.len() > self.limit {
            self.undo.remove(0);
        }
    }

    /// Starts a group: changes recorded until [`end_group`](Self::end_group)
    /// are one step named `name`.
    pub fn begin_group(&mut self, name: &str) {
        if self.group.is_none() {
            self.group = Some(Entry {
                name: name.to_owned(),
                merge: None,
                steps: vec![],
            });
        }
    }

    /// Ends the group (an empty one leaves nothing).
    pub fn end_group(&mut self) {
        if let Some(g) = self.group.take()
            && !g.steps.is_empty()
        {
            self.undo.push(g);
            if self.undo.len() > self.limit {
                self.undo.remove(0);
            }
        }
    }

    /// Takes back the last step (its changes, last first). Returns whether
    /// there was one.
    pub fn undo(&mut self, state: &mut S) -> bool {
        self.end_group();
        let Some(e) = self.undo.pop() else {
            return false;
        };
        for s in e.steps.iter().rev() {
            (s.undo)(state);
        }
        self.redo.push(Entry { merge: None, ..e });
        true
    }

    /// Makes again the last step undone. Returns whether there was one.
    pub fn redo(&mut self, state: &mut S) -> bool {
        let Some(e) = self.redo.pop() else {
            return false;
        };
        for s in &e.steps {
            (s.redo)(state);
        }
        self.undo.push(e);
        true
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty() || self.group.as_ref().is_some_and(|g| !g.steps.is_empty())
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// The name of what [`undo`](Self::undo) would take back, for a menu's
    /// "Undo …".
    pub fn undo_name(&self) -> Option<&str> {
        self.undo.last().map(|e| e.name.as_str())
    }

    /// The name of what [`redo`](Self::redo) would make again.
    pub fn redo_name(&self) -> Option<&str> {
        self.redo.last().map(|e| e.name.as_str())
    }

    /// Forgets everything (a document opened or saved for good).
    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
        self.group = None;
    }
}

/// The action that undoes (see [`bind_undo_keys`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Undo;

/// The action that redoes.
#[derive(Debug, Clone, PartialEq)]
pub struct Redo;

/// Binds the usual keys: ⌘Z and Ctrl+Z undo; ⇧⌘Z, Ctrl+Shift+Z and Ctrl+Y
/// redo.
pub fn bind_undo_keys(keymap: &mut Keymap) {
    for k in ["cmd-z", "ctrl-z"] {
        keymap.bind(k, Undo, None).expect("a keystroke");
    }
    for k in ["shift-cmd-z", "shift-ctrl-z", "ctrl-y"] {
        keymap.bind(k, Redo, None).expect("a keystroke");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Doc {
        items: Vec<i32>,
    }

    fn add(m: &mut UndoManager<Doc>, d: &mut Doc, v: i32) {
        d.items.push(v);
        m.record(
            "Add",
            |d: &mut Doc| {
                d.items.pop();
            },
            move |d: &mut Doc| d.items.push(v),
        );
    }

    #[test]
    fn undo_redo_groups_and_a_new_change_drops_the_redo() {
        let (mut m, mut d) = (UndoManager::new(), Doc::default());
        add(&mut m, &mut d, 1);
        m.begin_group("Add two");
        add(&mut m, &mut d, 2);
        add(&mut m, &mut d, 3);
        m.end_group();
        assert_eq!(m.undo_name(), Some("Add two"));
        assert!(m.undo(&mut d));
        assert_eq!(d.items, [1], "the group at once");
        assert_eq!(m.redo_name(), Some("Add two"));
        assert!(m.redo(&mut d));
        assert_eq!(d.items, [1, 2, 3]);
        m.undo(&mut d);
        add(&mut m, &mut d, 9);
        assert!(!m.can_redo(), "a new change drops what was undone");
        m.undo(&mut d);
        m.undo(&mut d);
        assert_eq!(d.items, Vec::<i32>::new());
        assert!(!m.undo(&mut d));
    }

    #[test]
    fn quick_changes_of_a_kind_merge() {
        let (mut m, mut d) = (UndoManager::<Doc>::new(), Doc::default());
        for (i, t) in [0.0, 0.3, 0.6, 2.0].into_iter().enumerate() {
            d.items.push(i as i32);
            m.record_merging(
                "Typing",
                "type",
                t,
                |d: &mut Doc| {
                    d.items.pop();
                },
                move |d: &mut Doc| d.items.push(i as i32),
            );
        }
        m.undo(&mut d);
        assert_eq!(d.items, [0, 1, 2], "the late one alone");
        m.undo(&mut d);
        assert!(d.items.is_empty(), "the burst at once");
        m.redo(&mut d);
        assert_eq!(d.items, [0, 1, 2]);
    }

    #[test]
    fn the_keys_are_bound() {
        use crate::keymap::Keystroke;
        let mut k = Keymap::new();
        bind_undo_keys(&mut k);
        let a = k
            .match_stroke(&Keystroke::parse("ctrl-z").unwrap(), &[])
            .action()
            .unwrap();
        assert!(a.as_any().is::<Undo>());
        let a = k
            .match_stroke(&Keystroke::parse("ctrl-y").unwrap(), &[])
            .action()
            .unwrap();
        assert!(a.as_any().is::<Redo>());
    }
}
