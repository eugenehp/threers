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
//! Two devices can share one room. Set [`Ar::collaborate`] on both, carry
//! each one's [`Ar::outgoing`] to the other's [`Ar::incoming`] over any link
//! (`share`, `nearby`), and their maps of the room merge: an anchor one
//! [pins](Ar::pin) then shows in the other's [`Ar::pins`], where it is in the
//! other's own world, so both can put the game in the same spot.
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

/// A named anchor in the world ARKit tracks.
#[derive(Debug, Clone, PartialEq)]
pub struct ArPin {
    /// What the app called it.
    pub name: String,
    /// Where it is, metres.
    pub position: Vector3,
    /// Which way it faces.
    pub rotation: Quaternion,
    /// Whether this device pinned it (else another sharing the room did).
    pub mine: bool,
}

/// What one device tells another about the room, to share it: opaque bytes,
/// and whether they must arrive (else newer will do, and they may be sent
/// lossily).
#[derive(Debug, Clone, PartialEq)]
pub struct ArData {
    /// ARKit's, archived.
    pub bytes: Vec<u8>,
    /// Whether it must arrive.
    pub critical: bool,
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
    /// Share the room with other devices (ARKit's collaborative session).
    pub collaborate: bool,
    /// What this device has to tell the others about the room: the app sends
    /// it on, and empties it.
    pub outgoing: Vec<ArData>,
    /// What the others have told it: the app fills it, the window empties it.
    pub incoming: Vec<Vec<u8>>,
    /// An anchor to pin, in place of this device's last of its name: the
    /// window takes it.
    pub pin: Option<ArPin>,
    /// The named anchors now, this device's and (sharing) the others'.
    pub pins: Vec<ArPin>,
    /// How many other devices share the room now.
    pub others: usize,
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
        if let Some(session) = self.session.as_mut().filter(|_| ar.wanted) {
            session.collaborate(ar.collaborate);
            ar.outgoing.extend(
                session
                    .take_collaboration()
                    .into_iter()
                    .map(|(bytes, critical)| ArData { bytes, critical }),
            );
            // Unsent, it must not pile up: drop what newer data replaces.
            if ar.outgoing.len() > 128 {
                ar.outgoing.retain(|d| d.critical);
                let over = ar.outgoing.len().saturating_sub(128);
                ar.outgoing.drain(..over);
            }
            for bytes in ar.incoming.drain(..) {
                session.give_collaboration(&bytes);
            }
            if let Some(pin) = ar.pin.take() {
                let m = threers::math::Matrix4::compose(pin.position, pin.rotation, Vector3::new(1.0, 1.0, 1.0));
                session.pin(&pin.name, m.elements);
            }
            let (pins, others) = session.anchors();
            ar.others = others;
            ar.pins = pins
                .into_iter()
                .map(|p| {
                    let (position, rotation, _) = threers::math::Matrix4 { elements: p.transform }.decompose();
                    ArPin {
                        name: p.name,
                        position,
                        rotation,
                        mine: p.mine,
                    }
                })
                .collect();
        } else {
            ar.incoming.clear();
            ar.pin = None;
            ar.pins.clear();
            ar.others = 0;
        }
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
