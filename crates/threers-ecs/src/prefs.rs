//! What a game remembers between runs that is not a save: settings, a best
//! score, a tutorial already done. Text keys and values, kept where
//! [`SaveSlots`](crate::saves::SaveSlots) keeps saves (a file natively, in
//! the app's own Library on iOS; `localStorage` in a browser), written
//! whenever one changes.
//!
//! ```no_run
//! use threers_ecs::prefs::Prefs;
//! let mut prefs = Prefs::open("my-game");
//! if !prefs.get_bool("tutorial-done") {
//!     // show it, then:
//!     prefs.set("tutorial-done", true);
//! }
//! ```

use crate::saves::SaveSlots;
use std::collections::BTreeMap;

/// The slot the values live in.
const SLOT: &str = "prefs";

/// A game's remembered values. Make it a resource.
#[derive(Debug, Clone, Default)]
pub struct Prefs {
    /// Where they are kept; `None` keeps them in memory only.
    slots: Option<SaveSlots>,
    values: BTreeMap<String, String>,
}

impl Prefs {
    /// The values `game` remembered last time, kept as they change.
    pub fn open(game: &str) -> Self {
        let slots = SaveSlots::new(game);
        let values = slots
            .get(SLOT)
            .ok()
            .flatten()
            .map(|text| parse(&text))
            .unwrap_or_default();
        Prefs {
            slots: Some(slots),
            values,
        }
    }

    /// Values kept in memory only: for a test, or a game that may not write.
    pub fn memory() -> Self {
        Prefs::default()
    }

    /// The value under `key`, if there is one.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }

    /// A number, if there is one under `key`.
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.get(key)?.parse().ok()
    }

    /// Whether `key` is `true`.
    pub fn get_bool(&self, key: &str) -> bool {
        self.get(key) == Some("true")
    }

    /// Every key starting with `prefix`, and its value, in key order.
    pub fn with_prefix<'a>(&'a self, prefix: &'a str) -> impl Iterator<Item = (&'a str, &'a str)> + 'a {
        self.values
            .range(prefix.to_owned()..)
            .take_while(move |(k, _)| k.starts_with(prefix))
            .map(|(k, v)| (k.as_str(), v.as_str()))
    }

    /// Remember `value` under `key`, and keep it. A value that cannot be
    /// kept (no storage) is still remembered for this run.
    pub fn set(&mut self, key: &str, value: impl ToString) {
        let value = value.to_string();
        if self.values.get(key) == Some(&value) {
            return;
        }
        self.values.insert(key.to_owned(), value);
        self.keep();
    }

    /// Forget `key`.
    pub fn remove(&mut self, key: &str) {
        if self.values.remove(key).is_some() {
            self.keep();
        }
    }

    fn keep(&self) {
        if let Some(slots) = &self.slots
            && let Err(e) = slots.put(SLOT, &write(&self.values))
        {
            eprintln!("prefs not kept: {e}");
        }
    }
}

/// One `key<TAB>value` a line, with tabs, newlines and backslashes escaped.
fn write(values: &BTreeMap<String, String>) -> String {
    let escape = |s: &str| {
        s.replace('\\', "\\\\")
            .replace('\t', "\\t")
            .replace('\n', "\\n")
    };
    values
        .iter()
        .map(|(k, v)| format!("{}\t{}\n", escape(k), escape(v)))
        .collect()
}

fn parse(text: &str) -> BTreeMap<String, String> {
    let unescape = |s: &str| {
        let mut out = String::new();
        let mut chars = s.chars();
        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some('t') => out.push('\t'),
                    Some('n') => out.push('\n'),
                    Some(other) => out.push(other),
                    None => {}
                }
            } else {
                out.push(c);
            }
        }
        out
    };
    text.lines()
        .filter_map(|line| line.split_once('\t'))
        .map(|(k, v)| (unescape(k), unescape(v)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_survive_the_trip_through_text() {
        let mut values = BTreeMap::new();
        values.insert("best.htv · grapple".to_owned(), "8.25".to_owned());
        values.insert("odd\tkey".to_owned(), "two\nlines \\ here".to_owned());
        assert_eq!(parse(&write(&values)), values);
    }

    #[test]
    fn numbers_flags_and_prefixes() {
        let mut p = Prefs::memory();
        p.set("best.a", 3.5);
        p.set("best.b", 1.0);
        p.set("done", true);
        p.set("other", "x");
        assert_eq!(p.get_f32("best.a"), Some(3.5));
        assert!(p.get_bool("done"));
        assert!(!p.get_bool("missing"));
        let bests: Vec<&str> = p.with_prefix("best.").map(|(k, _)| k).collect();
        assert_eq!(bests, ["best.a", "best.b"]);
        p.remove("done");
        assert!(!p.get_bool("done"));
    }
}
