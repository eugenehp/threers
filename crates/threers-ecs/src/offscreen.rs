//! Pictures drawn before each frame: a camera into a texture for a monitor in
//! the scene, a minimap, a rear-view mirror. Add a pass to the [`Offscreen`]
//! resource; the window, the browser and [`TestGame::render`] run every pass
//! with their renderer just before they draw the frame.
//!
//! ```no_run
//! use threers::views::CameraTexture;
//! use threers_ecs::offscreen::Offscreen;
//! use threers_ecs::prelude::*;
//!
//! let mut app = App::new();
//! let mut screen: Option<CameraTexture> = None;
//! app.world.insert_resource(Offscreen::default());
//! app.world.resource_mut::<Offscreen>().add(move |renderer, w| {
//!     let screen = screen.get_or_insert_with(|| CameraTexture::new(renderer, 512, 288));
//!     let camera = PerspectiveCamera::new(50.0, 16.0 / 9.0, 0.1, 100.0);
//!     screen.render(renderer, &mut w.scene, &camera);
//! });
//! ```
//!
//! A [`TestGame`] makes a renderer for each picture it takes, so a pass that
//! keeps GPU things should check [`Renderer::device_arc`] and remake them on
//! a new device.
//!
//! [`TestGame`]: crate::testing::TestGame
//! [`TestGame::render`]: crate::testing::TestGame::render

use crate::World;
use threers::Renderer;

/// One pass: draws what it likes with the renderer, before the frame.
pub type Pass = Box<dyn FnMut(&mut Renderer, &mut World)>;

/// The passes run before each frame, in the order added.
#[derive(Default)]
pub struct Offscreen {
    passes: Vec<Pass>,
}

impl Offscreen {
    /// Run `pass` before every frame from now on.
    pub fn add(&mut self, pass: impl FnMut(&mut Renderer, &mut World) + 'static) {
        self.passes.push(Box::new(pass));
    }

    /// Stop running every pass.
    pub fn clear(&mut self) {
        self.passes.clear();
    }

    /// How many passes there are.
    pub fn len(&self) -> usize {
        self.passes.len()
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
    }

    /// Run the world's passes with `renderer`, if it has an [`Offscreen`].
    /// The resource is out of the world while they run, so a pass may add
    /// another only through what it captured.
    pub fn run(world: &mut World, renderer: &mut Renderer) {
        let Some(mut offscreen) = world.remove_resource::<Offscreen>() else {
            return;
        };
        for pass in &mut offscreen.passes {
            pass(renderer, world);
        }
        world.insert_resource(offscreen);
    }
}
