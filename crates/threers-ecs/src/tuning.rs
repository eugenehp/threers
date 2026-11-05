//! Numbers to tune while the game runs: grip, speeds, camera distance.
//!
//! [`App::tune`] keeps a [`Reflect`] type as a resource and in a JSON file
//! beside the game. Edit the file and the resource changes on the next frame, with
//! no rebuild. A missing file is written with the type's defaults, so there is
//! always one to edit.
//!
//! ```no_run
//! # use threers_ecs::prelude::*;
//! #[derive(Default)]
//! struct Handling { grip: f32, top_speed: f32 }
//! reflect!(Handling { grip, top_speed });
//!
//! let mut app = App::new();
//! app.tune::<Handling>("tuning/handling.json");
//! // then read it as any resource: w.resource::<Handling>().grip
//! ```
//!
//! A field the file leaves out keeps its value, and one it gets wrong is reported
//! and left as it was, so a typo mid-edit does not stop the game.

use crate::app::{App, Time};
use crate::registry::{FieldValue, Reflect};
use crate::world::World;
use std::path::PathBuf;
use threers::loaders::json::{self, Value};

/// A tuned type's file was read again.
#[derive(Debug, Clone, PartialEq)]
pub struct Retuned {
    /// The type, as [`Reflect::type_name`] gives it.
    pub name: &'static str,
    /// Why some or all of the file was not used, or `None` if all of it was.
    pub problem: Option<String>,
}

struct Watch {
    path: PathBuf,
    modified: Option<std::time::SystemTime>,
    apply: fn(&mut World, &str) -> Retuned,
}

/// The files being watched, as a resource.
#[derive(Default)]
pub struct TuneFiles {
    watches: Vec<Watch>,
    checked: f64,
}

/// How often files are looked at, in seconds.
const POLL: f64 = 0.25;

impl App {
    /// Keep `T` as a resource, read from `path` and read again whenever it
    /// changes. See the [module docs](crate::tuning).
    pub fn tune<T: Reflect>(&mut self, path: impl Into<PathBuf>) -> &mut Self {
        let path = path.into();
        if self.world.get_resource::<T>().is_none() {
            self.world.insert_resource(T::default());
        }
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                let report = apply_text::<T>(&mut self.world, &text);
                if let Some(problem) = &report.problem {
                    eprintln!("{}: {problem}", path.display());
                }
            }
            Err(_) => {
                let text = to_text(&self.world.resource::<T>().fields());
                if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
                    let _ = std::fs::create_dir_all(dir);
                }
                let _ = std::fs::write(&path, text);
            }
        }
        let modified = modified(&path);
        self.world.resource_mut::<TuneFiles>().watches.push(Watch {
            path,
            modified,
            apply: apply_text::<T>,
        });
        self
    }
}

fn modified(path: &std::path::Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(path)
        .and_then(|meta| meta.modified())
        .ok()
}

/// Fields as JSON, one to a line in declaration order, for a person to edit.
pub fn to_text(fields: &[(&'static str, FieldValue)]) -> String {
    let value = |v: &FieldValue| match v {
        FieldValue::Bool(b) => b.to_string(),
        FieldValue::Number(n) => Value::Number(*n).to_json(),
        FieldValue::Text(t) => Value::String(t.clone()).to_json(),
        FieldValue::Vector([x, y, z]) => format!("[{x}, {y}, {z}]"),
        FieldValue::Entity(_) => "null".to_owned(),
    };
    let lines: Vec<String> = fields
        .iter()
        .map(|(name, v)| format!("  \"{name}\": {}", value(v)))
        .collect();
    format!("{{\n{}\n}}\n", lines.join(",\n"))
}

/// A JSON value as a field value, where it can be one.
pub(crate) fn json_to_field(value: &Value) -> Option<FieldValue> {
    Some(match value {
        Value::Bool(v) => FieldValue::Bool(*v),
        Value::Number(v) => FieldValue::Number(*v),
        Value::String(v) => FieldValue::Text(v.clone()),
        Value::Array(items) if items.len() == 3 => {
            let n = |i: usize| items[i].as_number().unwrap_or(0.0) as f32;
            FieldValue::Vector([n(0), n(1), n(2)])
        }
        _ => return None,
    })
}

fn apply_text<T: Reflect>(w: &mut World, text: &str) -> Retuned {
    let name = T::type_name();
    let parsed = match json::parse(text) {
        Ok(parsed) => parsed,
        Err(error) => {
            return Retuned {
                name,
                problem: Some(format!("not JSON: {error}")),
            }
        }
    };
    let Some(object) = parsed.as_object() else {
        return Retuned {
            name,
            problem: Some("expected an object of fields".to_owned()),
        };
    };
    let known: Vec<&str> = T::default().fields().iter().map(|(n, _)| *n).collect();
    let mut problems = Vec::new();
    let target = w.resource_mut::<T>();
    let mut keys: Vec<&String> = object.keys().collect();
    keys.sort();
    for field in keys {
        let value = &object[field];
        let ok = json_to_field(value).is_some_and(|v| target.set_field(field, &v));
        if !ok {
            problems.push(if known.contains(&field.as_str()) {
                format!("{field} has the wrong kind of value")
            } else {
                format!("{name} has no field {field} (it has {})", known.join(", "))
            });
        }
    }
    Retuned {
        name,
        problem: (!problems.is_empty()).then(|| problems.join("; ")),
    }
}

pub(crate) fn watch_tuning(w: &mut World) {
    let now = w.resource::<Time>().real_elapsed;
    let files = w.resource_mut::<TuneFiles>();
    if files.watches.is_empty() || now - files.checked < POLL {
        return;
    }
    files.checked = now;
    let mut changed = Vec::new();
    for watch in &mut files.watches {
        let stamp = modified(&watch.path);
        if stamp.is_some() && stamp != watch.modified {
            watch.modified = stamp;
            changed.push((watch.path.clone(), watch.apply));
        }
    }
    for (path, apply) in changed {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let report = apply(w, &text);
        if let Some(problem) = &report.problem {
            eprintln!("{}: {problem}", path.display());
        }
        w.send(report);
    }
}
