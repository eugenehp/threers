//! Playing a game from a test.
//!
//! The clock and the input are both fed in by hand, so a run is the same every
//! time, and no window or GPU is needed unless the test compares pixels.
//!
//! ```
//! use threers_ecs::prelude::*;
//! use threers_ecs::testing::TestGame;
//!
//! #[derive(Default)]
//! struct Jumps(u32);
//!
//! fn jump(w: &mut World) {
//!     if w.resource::<Input>().just_pressed("jump") {
//!         w.resource_mut::<Jumps>().0 += 1;
//!     }
//! }
//!
//! let mut game = TestGame::new(|app| {
//!     app.resource(Jumps::default());
//!     app.world.resource_mut::<Input>().bind("jump", Key::Space);
//!     app.fixed_system(jump);
//! });
//! game.tap(Key::Space);
//! assert_eq!(game.resource::<Jumps>().0, 1);
//! ```

use crate::app::App;
use crate::input::{Button, Input};
use crate::world::World;

/// A game under test, running at a steady 60 frames a second.
pub struct TestGame {
    pub app: App,
    frame_time: f64,
}

impl TestGame {
    pub fn new(build: impl FnOnce(&mut App)) -> Self {
        let mut app = App::new();
        build(&mut app);
        Self {
            app,
            frame_time: 1.0 / 60.0,
        }
    }

    /// Change the frame rate, to check that a game does not depend on it.
    pub fn at_fps(mut self, fps: f64) -> Self {
        self.frame_time = 1.0 / fps;
        self
    }

    pub fn world(&mut self) -> &mut World {
        &mut self.app.world
    }

    pub fn resource<R: 'static>(&self) -> &R {
        self.app.world.resource()
    }

    pub fn input(&mut self) -> &mut Input {
        self.app.world.resource_mut()
    }

    /// Run this many frames.
    pub fn frames(&mut self, frames: u32) -> &mut Self {
        for _ in 0..frames {
            self.app.frame(self.frame_time);
        }
        self
    }

    /// Run for this long.
    pub fn seconds(&mut self, seconds: f64) -> &mut Self {
        self.frames((seconds / self.frame_time).round() as u32)
    }

    pub fn press(&mut self, button: impl Into<Button>) -> &mut Self {
        self.input().press(button);
        self
    }

    pub fn release(&mut self, button: impl Into<Button>) -> &mut Self {
        self.input().release(button);
        self
    }

    /// Press, wait, release.
    pub fn hold(&mut self, button: impl Into<Button> + Copy, frames: u32) -> &mut Self {
        self.press(button).frames(frames).release(button)
    }

    /// Press and release within one frame, then run it.
    pub fn tap(&mut self, button: impl Into<Button> + Copy) -> &mut Self {
        self.press(button).release(button).frames(1)
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod pixels {
    use super::TestGame;
    use crate::app::MainCamera;
    use crate::hud::Hud;
    use std::path::Path;
    use threers::{decode_png, encode_png, HeadlessRenderer};

    /// Lay straight-alpha `top` over `picture`, as a host's compositor does.
    fn over(picture: &mut [u8], top: &[u8]) {
        for (under, above) in picture.chunks_exact_mut(4).zip(top.chunks_exact(4)) {
            let alpha = above[3] as u32;
            for channel in 0..3 {
                let mixed = under[channel] as u32 * (255 - alpha) + above[channel] as u32 * alpha;
                under[channel] = ((mixed + 127) / 255) as u8;
            }
        }
    }

    /// Write what was drawn and where it differs, beside the reference: the
    /// actual frame as `<name>.actual.png`, and as `<name>.diff.png` the
    /// reference faded to grey with every changed pixel in red, brighter the
    /// more it changed.
    fn write_failure(
        path: &Path,
        width: u32,
        height: u32,
        actual: &[u8],
        expected: &[u8],
    ) -> (std::path::PathBuf, std::path::PathBuf) {
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("frame");
        let dir = path.parent().unwrap_or(Path::new("."));
        let (actual_path, diff_path) = (
            dir.join(format!("{stem}.actual.png")),
            dir.join(format!("{stem}.diff.png")),
        );
        let mut diff = Vec::with_capacity(expected.len());
        for (a, e) in actual.chunks_exact(4).zip(expected.chunks_exact(4)) {
            let change = (0..3).map(|c| a[c].abs_diff(e[c])).max().unwrap_or(0);
            if change > 2 {
                let strength = 128 + (change as u32 * 127 / 255) as u8;
                diff.extend([strength, 0, 0, 255]);
            } else {
                let grey = ((e[0] as u32 + e[1] as u32 + e[2] as u32) / 3 / 3 + 40) as u8;
                diff.extend([grey, grey, grey, 255]);
            }
        }
        let _ = std::fs::write(&actual_path, encode_png(width, height, actual));
        let _ = std::fs::write(&diff_path, encode_png(width, height, &diff));
        (actual_path, diff_path)
    }

    impl TestGame {
        /// The picture this frame, through the real renderer, as RGBA, with the
        /// [`Hud`] over it if it is this size.
        ///
        /// # Panics
        ///
        /// If there is no [`MainCamera`] or no GPU.
        pub fn render(&mut self, width: u32, height: u32) -> Vec<u8> {
            let mut renderer = HeadlessRenderer::builder()
                .size(width, height)
                .build()
                .expect("a GPU to render with");
            let mut rgba = self.app.world.with_resource::<MainCamera, _>(|w, camera| {
                renderer.render_to_rgba(&mut w.scene, &camera.0)
            });
            if let Some(hud) = self.app.world.get_resource::<Hud>() {
                if hud.size() == (width, height) {
                    over(&mut rgba, hud.pixels());
                }
            }
            rgba
        }

        /// Compare the picture with a PNG, allowing `tolerance` of mean difference
        /// per channel out of 255, since GPUs round differently. With
        /// `THREERS_UPDATE_FRAMES` set, the PNG is written instead.
        ///
        /// # Panics
        ///
        /// If the picture differs, or the reference is missing. A missing file
        /// fails rather than passing, or a test could quietly check nothing.
        pub fn assert_frame_matches(&mut self, path: impl AsRef<Path>, tolerance: f64) {
            let path = path.as_ref();
            let (width, height) = (320, 180);
            let rgba = self.render(width, height);
            if std::env::var_os("THREERS_UPDATE_FRAMES").is_some() {
                std::fs::write(path, encode_png(width, height, &rgba)).expect("write reference");
                return;
            }
            let bytes = std::fs::read(path).unwrap_or_else(|e| {
                panic!(
                    "no reference at {}: {e}. Set THREERS_UPDATE_FRAMES to make one",
                    path.display()
                )
            });
            let reference = decode_png(&bytes).expect("reference is a PNG");
            assert_eq!(
                (reference.width, reference.height),
                (width, height),
                "reference is a different size"
            );
            let total: u64 = rgba
                .iter()
                .zip(&reference.rgba)
                .map(|(a, b)| a.abs_diff(*b) as u64)
                .sum();
            let mean = total as f64 / rgba.len() as f64;
            if mean <= tolerance {
                return;
            }
            let (actual, diff) = write_failure(path, width, height, &rgba, &reference.rgba);
            panic!(
                "frame differs by {mean:.3} of 255 (allowed {tolerance}).\n  \
                 expected {}\n  actual   {}\n  diff     {}\n\
                 If the new picture is right, set THREERS_UPDATE_FRAMES to keep it.",
                path.display(),
                actual.display(),
                diff.display()
            );
        }
    }
}
