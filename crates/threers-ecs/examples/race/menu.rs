//! Escape pauses and opens a menu: resume, start again, sound levels, quit.

use super::effects::clear_marks;
use super::ghost::reset_ghost;
use super::rules::start_again;
use threers_ecs::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Item {
    Resume,
    Restart,
    Music,
    Effects,
    Quit,
}

const ITEMS: [Item; 5] = [
    Item::Resume,
    Item::Restart,
    Item::Music,
    Item::Effects,
    Item::Quit,
];

/// Whether the menu is up, which item is picked, and the sound levels.
pub struct Menu {
    pub open: bool,
    pub item: usize,
    pub music: f32,
    pub effects: f32,
}

impl Default for Menu {
    fn default() -> Self {
        Self {
            open: false,
            item: 0,
            music: 0.8,
            effects: 1.0,
        }
    }
}

impl Menu {
    pub fn selected(&self) -> Item {
        ITEMS[self.item % ITEMS.len()]
    }
}

fn set_open(w: &mut World, open: bool) {
    w.resource_mut::<Menu>().open = open;
    w.resource_mut::<Time>().paused = open;
}

fn apply_volumes(w: &mut World) {
    let (music, effects) = {
        let m = w.resource::<Menu>();
        (m.music, m.effects)
    };
    if let Some(out) = w.get_resource_mut::<AudioOut>() {
        out.set_bus_volume(Bus::Music, music);
        out.set_bus_volume(Bus::Effects, effects);
    }
}

/// Open and close, move, change and choose.
pub fn menu(w: &mut World) {
    let input = w.resource::<Input>();
    let toggle = input.just_pressed("menu");
    let open = w.resource::<Menu>().open;
    if toggle {
        set_open(w, !open);
        w.resource_mut::<Menu>().item = 0;
        return;
    }
    if !open {
        return;
    }
    let input = w.resource::<Input>();
    let pressed = |keys: &[Button]| keys.iter().any(|&k| input.was_pressed(k));
    let up = pressed(&[
        Key::ArrowUp.into(),
        Key::KeyW.into(),
        PadButton::DPadUp.into(),
    ]);
    let down = pressed(&[
        Key::ArrowDown.into(),
        Key::KeyS.into(),
        PadButton::DPadDown.into(),
    ]);
    let less = pressed(&[
        Key::ArrowLeft.into(),
        Key::KeyA.into(),
        PadButton::DPadLeft.into(),
    ]);
    let more = pressed(&[
        Key::ArrowRight.into(),
        Key::KeyD.into(),
        PadButton::DPadRight.into(),
    ]);
    let choose = pressed(&[
        Key::Enter.into(),
        Key::Space.into(),
        PadButton::South.into(),
    ]);

    let menu = w.resource_mut::<Menu>();
    let count = ITEMS.len();
    if up {
        menu.item = (menu.item + count - 1) % count;
    }
    if down {
        menu.item = (menu.item + 1) % count;
    }
    let step = if more {
        0.1
    } else if less {
        -0.1
    } else {
        0.0
    };
    match menu.selected() {
        Item::Music if step != 0.0 => menu.music = (menu.music + step).clamp(0.0, 1.0),
        Item::Effects if step != 0.0 => menu.effects = (menu.effects + step).clamp(0.0, 1.0),
        _ => {}
    }
    let item = menu.selected();
    if step != 0.0 {
        apply_volumes(w);
    }
    if choose {
        match item {
            Item::Resume => set_open(w, false),
            Item::Restart => {
                set_open(w, false);
                clear_marks(w);
                reset_ghost(w);
                start_again(w);
            }
            Item::Quit => {
                w.insert_resource(Exit);
            }
            Item::Music | Item::Effects => {}
        }
    }
}

pub fn draw_menu(w: &mut World) {
    let menu = w.resource::<Menu>();
    if !menu.open {
        return;
    }
    let lines: Vec<(String, bool)> = ITEMS
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let bar = |v: f32| format!("{:<10} {}", "", "#".repeat((v * 10.0).round() as usize));
            let text = match item {
                Item::Resume => "Resume".to_owned(),
                Item::Restart => "Start again".to_owned(),
                Item::Music => format!(
                    "Music   {:>3.0}% {}",
                    menu.music * 100.0,
                    bar(menu.music).trim()
                ),
                Item::Effects => format!(
                    "Effects {:>3.0}% {}",
                    menu.effects * 100.0,
                    bar(menu.effects).trim()
                ),
                Item::Quit => "Quit".to_owned(),
            };
            (text, i == menu.item)
        })
        .collect();
    with_hud(w, |hud| {
        let (width, height) = hud.size();
        let (width, height) = (width as f32, height as f32);
        hud.no_stroke();
        hud.fill_alpha(Color::from_hex(0x000000), 0.55);
        hud.rect(0.0, 0.0, width, height);
        let (left, top) = (width / 2.0 - 200.0, height / 2.0 - 150.0);
        hud.fill_alpha(Color::from_hex(0x1c1c22), 0.95);
        hud.rounded_rect(left, top, 400.0, 300.0, 16.0);
        hud.fill(Color::WHITE);
        hud.text_size(34.0);
        hud.text(left + 30.0, top + 50.0, "PAUSED");
        hud.text_size(24.0);
        for (i, (text, picked)) in lines.iter().enumerate() {
            let y = top + 100.0 + i as f32 * 38.0;
            if *picked {
                hud.fill(Color::from_hex(0xffcc33));
                hud.rounded_rect(left + 20.0, y - 26.0, 360.0, 34.0, 8.0);
                hud.fill(Color::from_hex(0x111111));
            } else {
                hud.fill(Color::WHITE);
            }
            hud.text(left + 34.0, y, text);
        }
        hud.fill_alpha(Color::WHITE, 0.6);
        hud.text_size(15.0);
        hud.text(
            left + 30.0,
            top + 288.0,
            "up/down choose  left/right change  enter select",
        );
    });
}
