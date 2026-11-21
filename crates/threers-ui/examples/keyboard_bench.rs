//! What the on-screen keyboard costs a frame: each operation — a letter
//! tapped, Shift, a page switched, a key held for its accents, a pointer
//! moved elsewhere — as one frame of dispatch, layout, damage and paint,
//! in this thread's CPU time (other work on the machine aside), and how
//! much of the window it damaged.
//!
//! ```text
//! cargo run -p threers-ui --release --example keyboard_bench
//! ```

use threers_ui::damage::{self, Damage};
use threers_ui::input::{Event, Input, PointerButton};
use threers_ui::virtual_keyboard::VirtualKeyboard;
use threers_ui::{
    style, FieldKey, FixedWidthText, NodeId, Painter, Rect, Rgba, Role, Semantics, Style,
    TextField, TextMeasure, TextStyle, Ui,
};

/// This thread's CPU time, in µs.
fn cpu_us() -> f64 {
    #[repr(C)]
    struct Timespec {
        sec: i64,
        nsec: i64,
    }
    unsafe extern "C" {
        fn clock_gettime(clock: i32, out: *mut Timespec) -> i32;
    }
    #[cfg(target_os = "macos")]
    const THREAD: i32 = 16;
    #[cfg(not(target_os = "macos"))]
    const THREAD: i32 = 3;
    let mut t = Timespec { sec: 0, nsec: 0 };
    unsafe { clock_gettime(THREAD, &mut t) };
    t.sec as f64 * 1e6 + t.nsec as f64 / 1e3
}

/// Counts what it is asked to draw.
#[derive(Default)]
struct Count(usize);

impl TextMeasure for Count {
    fn measure(&mut self, text: &str, style: &Style) -> (f32, f32) {
        FixedWidthText(7.0).measure(text, style)
    }
}

impl Painter for Count {
    fn fill_rect(&mut self, _: Rect, _: Rgba) {
        self.0 += 1;
    }
    fn text(&mut self, _: Rect, _: &str, _: &TextStyle) {
        self.0 += 1;
    }
    fn path(&mut self, _: Rect, _: &[String], _: [f32; 4], _: Option<f32>, _: bool, _: Rgba) {
        self.0 += 1;
    }
}

struct Bench {
    ui: Ui,
    input: Input<TextField>,
    field: TextField,
    _kb: VirtualKeyboard,
    last: Vec<damage::Node>,
    t: f64,
    size: (f32, f32),
}

impl Bench {
    fn new(w: f32, h: f32) -> Self {
        let mut ui = Ui::new();
        let mut input: Input<TextField> = Input::new();
        let root = ui.root();
        ui.set_style(root, style! { display: flex; flex-direction: column; });
        // A page behind it: a field and a long list.
        let target = ui.add_box(root, style! { height: 36px; });
        ui.set_semantics(target, Semantics::new(Role::TextBox).label("Message"));
        input.on_key(target, |f: &mut TextField, _, key, m, _| {
            !matches!(f.key(key, m, |_| true), FieldKey::Ignored)
        });
        let list = ui.add_box(
            root,
            style! { display: flex; flex-direction: column; overflow: auto; flex-grow: 1; },
        );
        for i in 0..300 {
            let row = ui.add_box(list, style! { padding: 6px; border: 1px solid #ddd; });
            ui.add_text(row, format!("Row {i}: some text to lay out"));
        }
        let kb = VirtualKeyboard::new(&mut ui, &mut input);
        kb.set_enabled(true);
        kb.set_context(|f: &TextField, _| Some(f.before_caret()));
        kb.set_suggestions(|w| {
            ["hello", "help", "helmet", "world", "word"]
                .iter()
                .filter(|s| s.starts_with(w))
                .map(|s| s.to_string())
                .collect()
        });
        let mut b = Self {
            ui,
            input,
            field: TextField::new(""),
            _kb: kb,
            last: vec![],
            t: 0.0,
            size: (w, h),
        };
        b.input.set_focus(&mut b.ui, &mut b.field, Some(target));
        // Past its slide in.
        for _ in 0..40 {
            b.frame(|_| {});
        }
        b
    }

    fn node(&self, name: &str) -> NodeId {
        self.ui
            .accessibility_tree()
            .into_iter()
            .find(|a| a.name.to_lowercase() == name.to_lowercase())
            .unwrap_or_else(|| panic!("no {name}"))
            .id
    }

    fn center(&self, name: &str) -> (f32, f32) {
        let r = self.ui.rect(self.node(name));
        (r.x + r.width / 2.0, r.y + r.height / 2.0)
    }

    /// One frame: `events`, a tick, layout, the damage, a paint. Its CPU
    /// time in µs, whether all of it was damaged, and the damaged share.
    fn frame(&mut self, events: impl FnOnce(&mut Self)) -> (f64, bool, f32, bool) {
        let passes = self.ui.layout_passes();
        let start = cpu_us();
        events(self);
        self.t += 1.0 / 60.0;
        self.input.tick(&mut self.ui, &mut self.field, self.t);
        self.ui.step(1.0 / 60.0);
        let (w, h) = self.size;
        let mut count = Count::default();
        self.ui.layout(w, h, &mut count);
        let now = damage::collect(&self.ui);
        let d = damage::diff(&self.last, &now, w, h);
        if std::env::var("SHOW").is_ok()
            && self.t > 0.7
            && let Damage::Regions(r) = &d
        {
            let changed: Vec<usize> = (0..now.len().min(self.last.len()))
                .filter(|&i| now[i] != self.last[i])
                .collect();
            eprintln!("regions {r:?}");
            // What a key's own change would grow into.
            for i in 0..now.len().min(self.last.len()) {
                if now[i] == self.last[i] {
                    continue;
                }
                let k = now[i].rect;
                for (j, n) in now.iter().enumerate() {
                    let whole = n.text.is_some()
                        || n.path.is_some()
                        || n.image.is_some()
                        || (n.style.background.is_some()
                            && !matches!(
                                n.style.border_radius,
                                threers_ui::Length::Auto | threers_ui::Length::Px(0.0)
                            ));
                    let r = n.rect;
                    let meets = r.x < k.x + k.width + 1.0
                        && k.x - 1.0 < r.x + r.width
                        && r.y < k.y + k.height + 1.0
                        && k.y - 1.0 < r.y + r.height;
                    let inside = r.x >= k.x - 1.0
                        && r.y >= k.y - 1.0
                        && r.x + r.width <= k.x + k.width + 1.0
                        && r.y + r.height <= k.y + k.height + 1.0;
                    if whole && meets && !inside && r.width > 0.0 {
                        eprintln!(
                            "    grows into node {j}: {:?} radius {:?} bg {:?} text {:?}",
                            r, n.style.border_radius, n.style.background, n.text
                        );
                    }
                }
                break;
            }
            for i in changed.iter().take(8) {
                let (a, b) = (&self.last[*i], &now[*i]);
                eprintln!("  node {i}: rect {:?}->{:?} text {:?}->{:?} bg {:?}->{:?} op {}->{} disp {:?}->{:?}", a.rect, b.rect, a.text, b.text, a.style.background, b.style.background, a.style.opacity, b.style.opacity, a.style.display, b.style.display);
            }
        }
        self.last = now;
        self.ui.paint(&mut count);
        let us = cpu_us() - start;
        let (full, share) = match d {
            Damage::None => (false, 0.0),
            Damage::Full => (true, 1.0),
            Damage::Regions(r) => (
                false,
                r.iter().map(|r| r.width * r.height).sum::<f32>() / (w * h),
            ),
        };
        (us, full, share, self.ui.layout_passes() != passes)
    }

    fn send(&mut self, e: Event) {
        self.input.dispatch(&mut self.ui, &mut self.field, e);
    }
}

fn report(name: &str, mut samples: Vec<(f64, bool, f32, bool)>) {
    samples.sort_by(|a, b| a.0.total_cmp(&b.0));
    let n = samples.len();
    let q = |p: f64| samples[((n - 1) as f64 * p) as usize].0;
    let full = samples.iter().filter(|s| s.1).count();
    let share = samples.iter().map(|s| s.2).sum::<f32>() / n as f32;
    let laid = samples.iter().filter(|s| s.3).count();
    println!(
        "{name:<28} p10 {:>7.0} µs  p50 {:>7.0} µs  p90 {:>7.0} µs   laid out {laid:>3}/{n}  full repaints {full:>3}/{n}  damaged {:>5.1}%",
        q(0.1),
        q(0.5),
        q(0.9),
        share * 100.0
    );
}

fn main() {
    if std::env::var("SHOW").is_ok() {
        let mut b = Bench::new(820.0, 1180.0);
        let bt = PointerButton::Primary;
        for k in ["q", "w"] {
            let at = b.center(k);
            eprintln!("--- down {k}");
            b.frame(|b| {
                b.send(Event::PointerDown {
                    x: at.0,
                    y: at.1,
                    button: bt,
                })
            });
            eprintln!("--- up {k}");
            b.frame(|b| {
                b.send(Event::PointerUp {
                    x: at.0,
                    y: at.1,
                    button: bt,
                })
            });
        }
        return;
    }
    for (device, w, h) in [("phone", 402.0, 874.0), ("tablet", 820.0, 1180.0)] {
        println!("{device} {w}×{h}");
        let mut b = Bench::new(w, h);
        let b_ = PointerButton::Primary;
        let letters = ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"];
        let mut down = vec![];
        let mut up = vec![];
        for i in 0..60 {
            let at = b.center(letters[i % letters.len()]);
            down.push(b.frame(|b| {
                b.send(Event::PointerMove { x: at.0, y: at.1 });
                b.send(Event::PointerDown {
                    x: at.0,
                    y: at.1,
                    button: b_,
                });
            }));
            up.push(b.frame(|b| {
                b.send(Event::PointerUp {
                    x: at.0,
                    y: at.1,
                    button: b_,
                })
            }));
            if i % 12 == 11 {
                // Back to an empty field now and then.
                b.field = TextField::new("");
            }
        }
        report("letter: finger down", down);
        report("letter: finger up (types)", up);

        let mut shift = vec![];
        for _ in 0..30 {
            let at = b.center("Shift");
            shift.push(b.frame(|b| {
                b.send(Event::PointerDown {
                    x: at.0,
                    y: at.1,
                    button: b_,
                });
                b.send(Event::PointerUp {
                    x: at.0,
                    y: at.1,
                    button: b_,
                });
            }));
            b.frame(|_| {});
        }
        report("shift", shift);

        let mut pages = vec![];
        for i in 0..30 {
            let name = if i % 2 == 0 {
                "Numbers and symbols"
            } else {
                "Letters"
            };
            let at = b.center(name);
            pages.push(b.frame(|b| {
                b.send(Event::PointerDown {
                    x: at.0,
                    y: at.1,
                    button: b_,
                });
                b.send(Event::PointerUp {
                    x: at.0,
                    y: at.1,
                    button: b_,
                });
            }));
        }
        report("page switch", pages);

        let mut hold = vec![];
        for _ in 0..20 {
            let at = b.center("e");
            b.frame(|b| {
                b.send(Event::PointerDown {
                    x: at.0,
                    y: at.1,
                    button: b_,
                })
            });
            for _ in 0..30 {
                b.frame(|_| {});
            }
            hold.push(b.frame(|b| {
                b.send(Event::PointerMove {
                    x: at.0 + 30.0,
                    y: at.1 - 40.0,
                })
            }));
            b.frame(|b| {
                b.send(Event::PointerUp {
                    x: at.0 + 30.0,
                    y: at.1 - 40.0,
                    button: b_,
                })
            });
        }
        report("accents: slide to one", hold);

        let mut moves = vec![];
        for i in 0..100 {
            moves.push(b.frame(|b| {
                b.send(Event::PointerMove {
                    x: 50.0 + i as f32,
                    y: 200.0,
                });
            }));
        }
        report("pointer over the page", moves);
        let mut idle = vec![];
        for _ in 0..100 {
            idle.push(b.frame(|_| {}));
        }
        report("idle frame", idle);
        println!();
    }
}
