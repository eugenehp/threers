//! AR: the game over the world, seen through the device's camera.
//!
//! Insert [`Ar`] and set [`Ar::wanted`]; on iOS, with the `ar` feature, the
//! window starts ARKit's world tracking, shows its camera feed behind the
//! game, and makes the game's surface see-through wherever the scene's
//! `background_alpha` leaves it clear. Each frame, before the systems run,
//! [`Ar::camera`] holds where the device is. The game puts its own camera
//! there, in whatever scale and place it wants the world to be.
//!
//! Everywhere else (the Mac, a browser, the simulator, a device with no world
//! tracking) [`Ar::supported`] stays `false` and nothing happens: a game
//! offers AR only where it is.
//!
//! The ARKit session is threers-ios' own (its `ar` module), the same
//! source compiled in here, so a game on threers-ecs needs nothing of
//! threers-ios' shell.

use threers::math::{Quaternion, Vector3};

/// The device's camera in the world ARKit tracks: metres, +Y against
/// gravity, the origin where the session started.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArCamera {
    /// Where the camera is.
    pub position: Vector3,
    /// Which way it faces: down its -Z, its +Y up, as a scene camera.
    pub rotation: Quaternion,
    /// Its vertical field of view, radians, for the window's shape.
    pub fov: f32,
}

/// The app's wish for AR, and what the device does about it. A resource.
#[derive(Debug, Clone, Default)]
pub struct Ar {
    /// Show the camera and track the device. The window starts and stops
    /// ARKit to match.
    pub wanted: bool,
    /// Whether ARKit is running now.
    pub running: bool,
    /// Whether this device can track the world. Known once the window is
    /// open; `false` until then, and off iOS.
    pub supported: bool,
    /// The camera this frame. `None` while ARKit has no frame yet, or is
    /// not running.
    pub camera: Option<ArCamera>,
    /// Where a horizontal surface (a table, the floor) is under the middle
    /// of the screen this frame, in the same world as the camera: where to
    /// put something down. `None` while none is found.
    pub surface: Option<Vector3>,
}

impl Ar {
    /// Whether the camera's pose is known this frame.
    pub fn tracking(&self) -> bool {
        self.running && self.camera.is_some()
    }
}

#[cfg(all(target_os = "ios", feature = "ar", feature = "window"))]
#[path = "../../threers-ios/src/ar.rs"]
#[allow(missing_docs, dead_code)]
mod arkit;

/// Start or stop ARKit for what [`Ar`] wants, and read the camera: the
/// window's part, once a frame before the systems run.
#[cfg(all(target_os = "ios", feature = "ar", feature = "window"))]
pub(crate) struct Driver {
    session: Option<arkit::ArSession>,
    checked: bool,
    /// When AR was last wanted, while the session lingers unwanted.
    unwanted_since: Option<std::time::Instant>,
}

/// How long a session no longer wanted keeps tracking, so AR switched off
/// and on again comes back at once, still knowing the room.
#[cfg(all(target_os = "ios", feature = "ar", feature = "window"))]
const LINGER: std::time::Duration = std::time::Duration::from_secs(30);

#[cfg(all(target_os = "ios", feature = "ar", feature = "window"))]
impl Driver {
    pub(crate) fn new() -> Self {
        Driver {
            session: None,
            checked: false,
            unwanted_since: None,
        }
    }

    pub(crate) fn drive(
        &mut self,
        window: &winit::window::Window,
        view: &mut threers::renderer::WindowRenderer,
        world: &mut crate::World,
    ) {
        use objc2::msg_send;
        use objc2::runtime::AnyObject;
        use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};

        let Some(ar) = world.get_resource_mut::<Ar>() else {
            return;
        };
        if !self.checked {
            self.checked = true;
            ar.supported = arkit::supported();
        }
        if ar.wanted && self.session.is_none() && ar.supported {
            let layer = window.window_handle().ok().and_then(|h| match h.as_raw() {
                // SAFETY: winit's view, alive with the window; the main thread.
                RawWindowHandle::UiKit(h) => unsafe {
                    let view = &*(h.ui_view.as_ptr() as *const AnyObject);
                    let layer: *mut AnyObject = msg_send![view, layer];
                    Some(layer.cast())
                },
                _ => None,
            });
            // SAFETY: the view's own layer, which has the view as its delegate.
            self.session = layer.and_then(|layer| unsafe { arkit::ArSession::start(layer) });
            if self.session.is_some() {
                view.set_transparent(true);
            } else {
                // Asked and refused: do not ask every frame.
                ar.supported = false;
            }
        } else if ar.wanted && self.unwanted_since.take().is_some() {
            // Back before it stopped: see through again.
            view.set_transparent(true);
        } else if !ar.wanted && self.session.is_some() {
            let since = *self.unwanted_since.get_or_insert_with(|| {
                view.set_transparent(false);
                std::time::Instant::now()
            });
            if since.elapsed() > LINGER {
                self.session = None;
                self.unwanted_since = None;
            }
        }
        ar.running = self.session.is_some() && ar.wanted;
        ar.surface = self
            .session
            .as_ref()
            .filter(|_| ar.wanted)
            .and_then(|s| s.surface())
            .map(|[x, y, z]| Vector3::new(x, y, z));
        ar.camera = self.session.as_ref().filter(|_| ar.wanted).and_then(|session| {
            let (world, projection) = session.camera(view.size(), 0.01, 100.0)?;
            let world = threers::math::Matrix4 { elements: world };
            let (position, rotation, _) = world.decompose();
            Some(ArCamera {
                position,
                rotation,
                fov: 2.0 * (1.0 / projection[5]).atan(),
            })
        });
    }
}
