//! Scripted pinches, for running the app where nobody can pinch (the
//! simulator, a test): `THREERS_XR_SCRIPT` lists taps and drags at points
//! on the panel, in the UI's pixels, at times in seconds from launch:
//!
//! ```text
//! THREERS_XR_SCRIPT="2:tap:97,129; 4:drag:640,600,640,250; 6:poke:97,96"
//! ```
//!
//! A `poke` is a fingertip instead: brought from 6 cm in front of the
//! point through the panel's surface and back out, as hand tracking would
//! report it (through `ImmersiveSession::set_scripted_fingertip`).
//!
//! Each becomes what visionOS would send: a look-and-pinch whose selection
//! ray runs from the head to that point on the panel, as active events
//! while it is held (a drag moves the point over half a second) and an
//! ended one when it lets go. They go in through
//! `threers_xr_spatial_event`, the way the app's Swift forwards real
//! pinches, so everything after it is what a real pinch meets.

use threers::xr::PanelPlane;
use threers::Vector3;

/// A pinch in the script.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Gesture {
    Tap {
        x: f32,
        y: f32,
    },
    Drag {
        from: (f32, f32),
        to: (f32, f32),
    },
    /// A fingertip pressing the panel at a point (hand tracking's poke).
    Poke {
        x: f32,
        y: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Step {
    pub at: f64,
    pub gesture: Gesture,
}

/// The steps in `script` (see the module docs), in time order; what does
/// not parse is reported and skipped.
pub fn parse(script: &str) -> Vec<Step> {
    let mut steps: Vec<Step> = script
        .split(';')
        .filter(|s| !s.trim().is_empty())
        .filter_map(|s| {
            let parts: Vec<&str> = s.trim().splitn(3, ':').collect();
            let nums = |p: &str| {
                p.split(',')
                    .map(|n| n.trim().parse::<f32>())
                    .collect::<Result<Vec<_>, _>>()
                    .ok()
            };
            let step = match parts.as_slice() {
                [at, "tap", p] => nums(p)
                    .filter(|n| n.len() == 2)
                    .map(|n| (at, Gesture::Tap { x: n[0], y: n[1] })),
                [at, "drag", p] => nums(p).filter(|n| n.len() == 4).map(|n| {
                    (
                        at,
                        Gesture::Drag {
                            from: (n[0], n[1]),
                            to: (n[2], n[3]),
                        },
                    )
                }),
                [at, "poke", p] => nums(p)
                    .filter(|n| n.len() == 2)
                    .map(|n| (at, Gesture::Poke { x: n[0], y: n[1] })),
                _ => None,
            }
            .and_then(|(at, gesture)| at.trim().parse::<f64>().ok().map(|at| Step { at, gesture }));
            if step.is_none() {
                eprintln!("threers script: can't read {s:?}");
            }
            step
        })
        .collect();
    steps.sort_by(|a, b| a.at.total_cmp(&b.at));
    steps
}

/// How long a drag takes.
const DRAG: f64 = 0.5;
/// How long a poke takes, in and out.
const POKE: f64 = 0.6;

/// Plays a script: call [`step`](Self::step) each frame.
pub struct Player {
    steps: Vec<Step>,
    next: usize,
    /// The step being held: its index and pinch id.
    held: Option<(usize, u64)>,
}

impl Player {
    pub fn new(steps: Vec<Step>) -> Self {
        Player {
            steps,
            next: 0,
            held: None,
        }
    }

    /// The script from `THREERS_XR_SCRIPT`, if set.
    pub fn from_env() -> Option<Self> {
        std::env::var("THREERS_XR_SCRIPT")
            .ok()
            .map(|s| Player::new(parse(&s)))
    }

    pub fn done(&self) -> bool {
        self.next >= self.steps.len() && self.held.is_none()
    }

    /// The pinches due at `time`, sent as visionOS would from a head at
    /// `head` (a pose) towards the panel at `plane` showing a UI of `size`
    /// pixels. Returns the steps that finished, to report.
    pub fn step(
        &mut self,
        time: f64,
        head: &[f32; 16],
        plane: &PanelPlane,
        size: (f32, f32),
        fingertip: &mut dyn FnMut(Option<[f32; 3]>),
    ) -> Vec<Step> {
        let eye = Vector3::new(head[12], head[13], head[14]);
        let on_panel = |(x, y): (f32, f32)| {
            plane.center
                + plane.right * ((x / size.0 - 0.5) * 2.0 * plane.half.0)
                + plane.up * ((0.5 - y / size.1) * 2.0 * plane.half.1)
        };
        let send = |id: u64, phase: u32, (x, y): (f32, f32)| {
            let p = on_panel((x, y));
            let d = (p - eye).normalize();
            threers::xr::visionos::threers_xr_spatial_event(
                id, phase, eye.x, eye.y, eye.z, d.x, d.y, d.z,
            );
        };
        let mut finished = vec![];
        if let Some((i, id)) = self.held {
            let step = self.steps[i];
            match step.gesture {
                Gesture::Tap { x, y } => {
                    send(id, 2, (x, y));
                    self.held = None;
                    finished.push(step);
                }
                Gesture::Poke { x, y } => {
                    // In from 6 cm to 1 cm through, and back out.
                    let k = ((time - step.at) / POKE).clamp(0.0, 1.0) as f32;
                    let depth = 0.06 - 0.07 * (1.0 - (2.0 * k - 1.0).abs());
                    let normal = plane.right.cross(plane.up);
                    let tip = on_panel((x, y)) + normal * depth;
                    if k < 1.0 {
                        fingertip(Some([tip.x, tip.y, tip.z]));
                    } else {
                        fingertip(None);
                        self.held = None;
                        finished.push(step);
                    }
                }
                Gesture::Drag { from, to } => {
                    let k = ((time - step.at) / DRAG).clamp(0.0, 1.0) as f32;
                    let at = (from.0 + (to.0 - from.0) * k, from.1 + (to.1 - from.1) * k);
                    send(id, if k < 1.0 { 1 } else { 2 }, at);
                    if k >= 1.0 {
                        self.held = None;
                        finished.push(step);
                    }
                }
            }
        } else if let Some(step) = self.steps.get(self.next).copied()
            && time >= step.at
        {
            let id = 0x7e57_0000 + self.next as u64;
            match step.gesture {
                Gesture::Tap { x, y } => send(id, 1, (x, y)),
                Gesture::Drag { from, .. } => send(id, 1, from),
                Gesture::Poke { x, y } => {
                    let tip = on_panel((x, y)) + plane.right.cross(plane.up) * 0.06;
                    fingertip(Some([tip.x, tip.y, tip.z]));
                }
            }
            self.held = Some((self.next, id));
            self.next += 1;
        }
        finished
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_script_reads_in_time_order_and_skips_what_it_cannot() {
        let s = parse("4:drag:640,600,640,250; 2:tap:97,129; nonsense; 6:tap:1");
        assert_eq!(
            s,
            [
                Step {
                    at: 2.0,
                    gesture: Gesture::Tap { x: 97.0, y: 129.0 }
                },
                Step {
                    at: 4.0,
                    gesture: Gesture::Drag {
                        from: (640.0, 600.0),
                        to: (640.0, 250.0)
                    }
                },
            ]
        );
    }
}
