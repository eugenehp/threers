//! A console in the running game: type a command, see what it did.
//!
//! Backquote (the key left of 1) opens it, with [`DevTools`](crate::dev::DevTools)
//! or [`Console`] as a plugin. While it is open the game's actions read as idle,
//! so typing does not drive. `help` lists what there is.
//!
//! | Command | Does |
//! |---|---|
//! | `help` | list the commands |
//! | `pause`, `resume`, `step` | stop the clock, start it, run one fixed step |
//! | `speed 0.5` | slow motion, or fast |
//! | `select car` | select the first entity with that name for the inspector |
//! | `get` | the selected entity's fields |
//! | `set Health.hp 10` | set a field of the selected entity's registered component |
//! | `tp 0 5 0` | move the selected entity, and its body if it has one, there |
//! | `systems` | the systems by cost |
//!
//! A game adds its own with [`World::console_command`]:
//!
//! ```
//! # use threers_ecs::prelude::*;
//! let mut app = App::new();
//! app.plugin(threers_ecs::console::Console);
//! app.world.console_command("laps", "laps <n>: set the race length", |w, args| {
//!     let n: u32 = args.first().and_then(|a| a.parse().ok()).ok_or("laps needs a number")?;
//!     Ok(format!("{n} laps"))
//! });
//! assert_eq!(threers_ecs::console::run(&mut app.world, "laps 5"), Ok("5 laps".to_owned()));
//! ```

use crate::app::{App, Plugin, Time};
use crate::hud::with_hud;
use crate::input::{Input, Key};
use crate::registry::FieldValue;
use crate::world::World;
use std::collections::BTreeMap;
use threers::math::Color;

type Handler = fn(&mut World, &[&str]) -> Result<String, String>;
type Custom = Box<dyn Fn(&mut World, &[&str]) -> Result<String, String>>;

/// Adds the console. [`DevTools`](crate::dev::DevTools) adds it too.
#[derive(Default)]
pub struct Console;

/// What the console holds: whether it is open, the line being typed, and what
/// has been said.
#[derive(Default)]
pub struct ConsoleState {
    /// Whether it is showing and taking the keys.
    pub open: bool,
    /// What is typed so far.
    pub line: String,
    /// Lines run, oldest first; up recalls them.
    pub history: Vec<String>,
    output: Vec<(String, bool)>,
    recall: Option<usize>,
    commands: BTreeMap<String, (String, std::rc::Rc<Custom>)>,
}

impl Plugin for Console {
    fn build(self, app: &mut App) {
        if app.world.get_resource::<ConsoleState>().is_none() {
            app.world.insert_resource(ConsoleState::default());
            app.system(console_step);
        }
    }
}

impl World {
    /// Add a console command: `name`, a line of help, and what it does with
    /// its arguments. What it returns is printed; an `Err` in red.
    pub fn console_command(
        &mut self,
        name: &str,
        help: &str,
        run: impl Fn(&mut World, &[&str]) -> Result<String, String> + 'static,
    ) {
        if self.get_resource::<ConsoleState>().is_none() {
            self.insert_resource(ConsoleState::default());
        }
        let custom: Custom = Box::new(run);
        self.resource_mut::<ConsoleState>()
            .commands
            .insert(name.to_owned(), (help.to_owned(), std::rc::Rc::new(custom)));
    }
}

const BUILT_IN: &[(&str, &str, Handler)] = &[
    ("help", "help: list the commands", help),
    ("pause", "pause: stop the clock", |w, _| {
        w.resource_mut::<Time>().paused = true;
        Ok("paused".into())
    }),
    ("resume", "resume: start the clock", |w, _| {
        w.resource_mut::<Time>().paused = false;
        Ok("running".into())
    }),
    ("step", "step: run one fixed step while paused", |w, _| {
        let time = w.resource_mut::<Time>();
        time.paused = true;
        time.step_once += 1;
        Ok("stepped".into())
    }),
    (
        "speed",
        "speed <x>: game time runs at x, 1 normal",
        |w, args| {
            let x: f64 = args
                .first()
                .and_then(|a| a.parse().ok())
                .ok_or("speed needs a number, such as 0.5")?;
            w.resource_mut::<Time>().scale = x.max(0.0);
            Ok(format!("speed {x}"))
        },
    ),
    (
        "select",
        "select <name>: select an entity by its node's name",
        |w, args| {
            let name = args.join(" ");
            let found = w
                .iter::<(crate::world::Entity, crate::query::NodeRef)>()
                .find(|(_, n)| n.name == name)
                .map(|(id, _)| id);
            let id = found.ok_or_else(|| format!("nothing is named {name:?}"))?;
            select(w, id);
            Ok(format!("selected {name} ({id:?})"))
        },
    ),
    ("get", "get: the selected entity's fields", |w, _| {
        let id = selected(w)?;
        let mut out = Vec::new();
        for (component, fields) in w.inspect(id) {
            for (field, value) in fields {
                out.push(format!("{component}.{field} = {value}"));
            }
        }
        if out.is_empty() {
            return Ok("no registered components on it".into());
        }
        Ok(out.join("\n"))
    }),
    (
        "set",
        "set <Component.field> <value>: change the selected entity",
        |w, args| {
            let id = selected(w)?;
            let (target, value) = match args {
                [target, rest @ ..] if !rest.is_empty() => (*target, rest.join(" ")),
                _ => return Err("set Component.field value".into()),
            };
            let (component, field) = target.split_once('.').ok_or("name it as Component.field")?;
            let value = parse_value(&value);
            if w.set_field(id, component, field, &value) {
                Ok(format!("{target} = {value}"))
            } else {
                Err(format!("no {component}.{field} on it that takes {value}"))
            }
        },
    ),
    (
        "tp",
        "tp <x> <y> <z>: move the selected entity there",
        |w, args| {
            let id = selected(w)?;
            let n: Vec<f32> = args.iter().filter_map(|a| a.parse().ok()).collect();
            let [x, y, z] = n[..] else {
                return Err("tp x y z".into());
            };
            let at = threers::math::Vector3::new(x, y, z);
            #[cfg(feature = "physics")]
            {
                use crate::physics::AttachBody;
                let turned = w.node(id).map(|n| n.quaternion).unwrap_or_default();
                if w.teleport_body(id, at, turned) {
                    return Ok(format!("moved to {x} {y} {z}"));
                }
            }
            w.node_mut(id).ok_or("it has no node")?.position = at;
            Ok(format!("moved to {x} {y} {z}"))
        },
    ),
    ("systems", "systems: what each system costs", |w, _| {
        let profile = w.resource::<crate::profile::Profile>();
        let lines: Vec<String> = profile
            .slowest()
            .into_iter()
            .take(12)
            .map(|(name, t)| format!("{:>7.3} ms  {}", t.ms, crate::profile::short(name)))
            .collect();
        Ok(lines.join("\n"))
    }),
];

fn select(w: &mut World, id: crate::world::Entity) {
    if let Some(state) = w.get_resource_mut::<crate::dev::DevState>() {
        state.selected = Some(id);
        state.field = 0;
    } else {
        w.insert_resource(Selected(id));
    }
}

/// The selection, for a console with no dev panel to hold it.
struct Selected(crate::world::Entity);

fn selected(w: &World) -> Result<crate::world::Entity, String> {
    let from_panel = w
        .get_resource::<crate::dev::DevState>()
        .and_then(|s| s.selected);
    from_panel
        .or_else(|| w.get_resource::<Selected>().map(|s| s.0))
        .filter(|&id| w.is_alive(id))
        .ok_or_else(|| {
            "select something first: click it with the panel up, or `select <name>`".into()
        })
}

fn parse_value(text: &str) -> FieldValue {
    if let Ok(n) = text.parse::<f64>() {
        return FieldValue::Number(n);
    }
    match text {
        "true" | "yes" | "on" => return FieldValue::Bool(true),
        "false" | "no" | "off" => return FieldValue::Bool(false),
        _ => {}
    }
    let parts: Vec<f32> = text
        .split([' ', ','])
        .filter(|p| !p.is_empty())
        .filter_map(|p| p.parse().ok())
        .collect();
    if let [x, y, z] = parts[..] {
        return FieldValue::Vector([x, y, z]);
    }
    FieldValue::Text(text.trim_matches('"').to_owned())
}

fn help(w: &mut World, _: &[&str]) -> Result<String, String> {
    let mut lines: Vec<String> = BUILT_IN
        .iter()
        .map(|(_, help, _)| help.to_string())
        .collect();
    if let Some(state) = w.get_resource::<ConsoleState>() {
        lines.extend(state.commands.values().map(|(help, _)| help.clone()));
    }
    Ok(lines.join("\n"))
}

/// Run one line as the console would, returning what it printed.
pub fn run(w: &mut World, line: &str) -> Result<String, String> {
    let words: Vec<&str> = line.split_whitespace().collect();
    let Some((&name, args)) = words.split_first() else {
        return Ok(String::new());
    };
    if let Some((_, _, handler)) = BUILT_IN.iter().find(|(n, _, _)| *n == name) {
        return handler(w, args);
    }
    let custom = w
        .get_resource::<ConsoleState>()
        .and_then(|s| s.commands.get(name).map(|(_, f)| f.clone()));
    match custom {
        Some(f) => f(w, args),
        None => {
            let known: Vec<String> = BUILT_IN
                .iter()
                .map(|(n, _, _)| n.to_string())
                .chain(
                    w.get_resource::<ConsoleState>()
                        .into_iter()
                        .flat_map(|s| s.commands.keys().cloned()),
                )
                .collect();
            Err(format!(
                "no command {name:?}; there are {}",
                known.join(", ")
            ))
        }
    }
}

/// Open, type, run, and draw.
pub fn console_step(w: &mut World) {
    let input = w.resource::<Input>();
    let toggle = input.was_pressed(Key::Backquote);
    let (enter, back, up, escape) = (
        input.was_pressed(Key::Enter),
        input.was_pressed(Key::Backspace),
        input.was_pressed(Key::ArrowUp),
        input.was_pressed(Key::Escape),
    );
    let typed: String = input.text().chars().filter(|&c| c != '`').collect();
    let state = w.resource_mut::<ConsoleState>();
    let was_open = state.open;
    if toggle {
        state.open = !state.open;
    } else if state.open && escape {
        state.open = false;
    }
    let open = state.open;
    let input = w.resource_mut::<Input>();
    input.captured = open;
    // The key that closed the console is not also the game's: Escape would
    // otherwise open a pause menu the moment the console shut.
    if was_open && !open {
        input.consume(Key::Escape);
        input.consume(Key::Backquote);
    }
    if !open {
        return;
    }
    let state = w.resource_mut::<ConsoleState>();
    if !toggle {
        state.line.push_str(&typed);
    }
    if back {
        state.line.pop();
    }
    if up && !state.history.is_empty() {
        let at = state
            .recall
            .map_or(state.history.len() - 1, |i| i.saturating_sub(1));
        state.recall = Some(at);
        state.line = state.history[at].clone();
    }
    if enter {
        let line = std::mem::take(&mut state.line);
        state.recall = None;
        if !line.trim().is_empty() {
            state.history.push(line.clone());
            state.output.push((format!("> {line}"), true));
            let result = run(w, &line);
            let state = w.resource_mut::<ConsoleState>();
            let (text, ok) = match result {
                Ok(text) => (text, true),
                Err(text) => (text, false),
            };
            for row in text.lines() {
                state.output.push((row.to_owned(), ok));
            }
            let excess = state.output.len().saturating_sub(12);
            state.output.drain(..excess);
        }
    }
    let state = w.resource::<ConsoleState>();
    let (line, output) = (state.line.clone(), state.output.clone());
    with_hud(w, |hud| {
        let (width, height) = hud.size();
        let (width, height) = (width as f32, height as f32);
        let rows = output.len() as f32 + 1.0;
        let top = height - 20.0 - rows * 20.0;
        hud.no_stroke();
        hud.fill_alpha(Color::from_hex(0x000000), 0.75);
        hud.rect(0.0, top - 10.0, width, height - top + 10.0);
        hud.text_size(15.0);
        for (i, (text, ok)) in output.iter().enumerate() {
            hud.fill(if *ok {
                Color::from_hex(0xdddddd)
            } else {
                Color::from_hex(0xff6b6b)
            });
            hud.text(12.0, top + 14.0 + i as f32 * 20.0, text);
        }
        hud.fill(Color::from_hex(0xffcc33));
        hud.text(12.0, height - 16.0, &format!("> {line}_"));
    });
}
