//! One table: the room, the candles on it, the sweep, the camera — and the
//! frame that draws them. `lib.rs` wraps each method in the C ABI.

use std::collections::BTreeMap;
use std::ffi::c_void;

use threers::controls::{OrbitControls, PointerEvent};
use threers::extras::candle::{Candle, CandleHandle, FlameStyle};
use threers::materials::Blending;
use threers::metal::objc::{Id, Owned};
use threers::metal::{MetalDevice, MetalRenderer, MetalSurface};
use threers::prelude::*;

use crate::fire::Fire;

/// The table's radius, in metres: candles are kept on it.
pub(crate) const TABLE_RADIUS: f32 = 0.6;
/// How near a touch must land to a candle's foot to pick it, in metres.
const PICK_RADIUS: f32 = 0.05;
/// How fast the flames flicker, against a candle's own pace: a calm burn.
pub(crate) const CALM: f32 = 0.5;

pub(crate) struct Placed {
    pub(crate) handle: CandleHandle,
    pub(crate) at: (f32, f32),
    /// How hard the flame is flaring (just played), decaying to 0.
    pub(crate) flare: f32,
    /// The colour last asked for, kept for a rebuild.
    pub(crate) color: Option<(Color, f32)>,
}

/// One table. Opaque to the caller.
pub struct ThreersCandles {
    pub(crate) renderer: MetalRenderer,
    surface: MetalSurface,
    size: (u32, u32),
    scene: Scene,
    pub(crate) camera: PerspectiveCamera,
    controls: OrbitControls,
    style: FlameStyle,
    /// Whether the flames stand on candles or burn on their own.
    bodies: bool,
    /// The sweep on the table — the hand that plays the flames it passes —
    /// and its material.
    sweep: (ObjectId, ShaderMaterial),
    /// The flames as burning gas, when on: the volume replaces the flames'
    /// meshes, their lights stay.
    pub(crate) fire: Option<Fire>,
    /// How much fire, 0–1; kept while the fire is off.
    fire_amount: f32,
    candles: BTreeMap<u32, Placed>,
    next_id: u32,
    time: f32,
}

impl ThreersCandles {
    /// A table drawing into `layer`, a `CAMetalLayer`.
    ///
    /// # Safety
    /// `layer` is a live `CAMetalLayer` that outlives the table.
    pub(crate) unsafe fn new(
        layer: *mut c_void,
        width: u32,
        height: u32,
        realistic: bool,
    ) -> Option<Self> {
        let (width, height) = (width.max(1), height.max(1));
        let device = MetalDevice::new().ok()?;
        let mut renderer = MetalRenderer::with_device(device.clone()).ok()?;
        renderer.set_tone_mapping(ToneMapping::AcesFilmic, 1.0);
        let surface = unsafe { MetalSurface::from_layer(&device, layer, width, height, 4) }.ok()?;
        let mut scene = Scene::new();
        room(&mut scene);
        let sweep = sweep(&mut scene);
        let mut camera = PerspectiveCamera::new(38.0, width as f32 / height as f32, 0.02, 20.0);
        camera.position = Vector3::new(0.0, 0.62, 0.95);
        camera.look_at(Vector3::new(0.0, 0.04, 0.0));
        let mut controls = OrbitControls::new(&camera);
        controls.min_distance = 0.35;
        controls.max_distance = 3.0;
        Some(ThreersCandles {
            renderer,
            surface,
            size: (width, height),
            scene,
            camera,
            controls,
            style: if realistic {
                FlameStyle::Realistic
            } else {
                FlameStyle::Simple
            },
            bodies: true,
            sweep,
            fire: None,
            fire_amount: 1.0,
            candles: BTreeMap::new(),
            next_id: 1,
            time: 0.0,
        })
    }

    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        self.size = (width.max(1), height.max(1));
        let _ = self.surface.resize(self.size.0, self.size.1);
        self.camera.aspect = self.size.0 as f32 / self.size.1 as f32;
    }

    pub(crate) fn count(&self) -> usize {
        self.candles.len()
    }

    // -- candles ----------------------------------------------------------

    /// A candle at `x`, `z`: its id, or `None` off the table.
    pub(crate) fn add(&mut self, x: f32, z: f32) -> Option<u32> {
        if x * x + z * z > TABLE_RADIUS * TABLE_RADIUS {
            return None;
        }
        let id = self.next_id;
        self.next_id += 1;
        let placed = self.place(id, (x, z), None);
        self.candles.insert(id, placed);
        Some(id)
    }

    /// Candle `id` to `x`, `z`, kept on the table.
    pub(crate) fn move_to(&mut self, id: u32, x: f32, z: f32) {
        let r = (x * x + z * z).sqrt();
        let k = if r > TABLE_RADIUS {
            TABLE_RADIUS / r
        } else {
            1.0
        };
        let (x, z) = (x * k, z * k);
        if let Some(p) = self.candles.get_mut(&id) {
            p.at = (x, z);
            if let Some(o) = self.scene.get_mut(p.handle.root) {
                o.position = Vector3::new(x, 0.0, z);
            }
        }
    }

    pub(crate) fn remove(&mut self, id: u32) {
        if let Some(p) = self.candles.remove(&id) {
            self.scene.arena.remove_subtree(p.handle.root);
        }
    }

    pub(crate) fn set_color(&mut self, id: u32, color: Color, amount: f32) {
        if let Some(p) = self.candles.get_mut(&id) {
            p.handle.set_flame_color(&mut self.scene, color, amount);
            p.color = Some((color, amount));
        }
    }

    pub(crate) fn flare(&mut self, id: u32, amount: f32) {
        if let Some(p) = self.candles.get_mut(&id) {
            p.flare = p.flare.max(amount.clamp(0.0, 1.0));
        }
    }

    /// With or without candles under the flames: every candle rebuilt,
    /// keeping its id and colour.
    pub(crate) fn set_bodies(&mut self, bodies: bool) {
        if self.bodies == bodies {
            return;
        }
        self.bodies = bodies;
        let ids: Vec<u32> = self.candles.keys().copied().collect();
        for id in ids {
            let old = self.candles.remove(&id).expect("listed above");
            self.scene.arena.remove_subtree(old.handle.root);
            let placed = self.place(id, old.at, old.color);
            self.candles.insert(id, placed);
        }
    }

    fn candle(&self, id: u32) -> Candle {
        // A different flicker each, so a row of candles does not breathe as one.
        let mut candle = Candle::new(self.style)
            .with_seed(id as f32 * 1.618)
            .with_body(self.bodies);
        // Shadows from every candle would be six views each.
        candle.cast_shadow = false;
        candle
    }

    /// Candle `id` into the scene at `at`, in `color` if it has one, its mesh
    /// flame hidden when the fire burns instead.
    fn place(&mut self, id: u32, at: (f32, f32), color: Option<(Color, f32)>) -> Placed {
        let handle = self
            .candle(id)
            .add_to(&mut self.scene, Vector3::new(at.0, 0.0, at.1));
        if let Some((c, amount)) = color {
            handle.set_flame_color(&mut self.scene, c, amount);
        }
        if let Some(o) = self.scene.get_mut(handle.flame) {
            o.visible = self.fire.is_none();
        }
        Placed {
            handle,
            at,
            color,
            flare: 0.0,
        }
    }

    // -- the fire and the sweep ---------------------------------------------

    /// The fire on or off; whether it burns (it may not be makeable).
    pub(crate) fn set_fire(&mut self, on: bool) -> bool {
        if on != self.fire.is_some() {
            self.fire = if on {
                Fire::new(self.renderer.device())
            } else {
                None
            };
            let burning = self.fire.is_some();
            for p in self.candles.values() {
                if let Some(o) = self.scene.get_mut(p.handle.flame) {
                    o.visible = !burning;
                }
            }
        }
        self.fire.is_some()
    }

    pub(crate) fn set_fire_amount(&mut self, amount: f32) {
        self.fire_amount = amount.clamp(0.0, 1.0);
    }

    pub(crate) fn set_sweep(&mut self, turns: f32, strength: f32) {
        let strength = strength.clamp(0.0, 1.0);
        self.sweep
            .1
            .set_uniform(0, [turns.rem_euclid(1.0), strength, 0.0, 0.0]);
        if let Some(o) = self.scene.get_mut(self.sweep.0) {
            o.visible = strength > 0.0;
        }
    }

    // -- the screen and the table -------------------------------------------

    /// Where the screen point `u`, `v` meets the table, if it does.
    pub(crate) fn table_point(&self, u: f32, v: f32) -> Option<(f32, f32)> {
        let inv = self
            .camera
            .projection_matrix()
            .multiply(&self.camera.view_matrix())
            .invert();
        let at = |z: f32| Vector3::new(u * 2.0 - 1.0, 1.0 - v * 2.0, z).apply_matrix4(&inv);
        let (near, far) = (at(0.0), at(1.0));
        let dir = far - near;
        if dir.y.abs() < 1e-6 {
            return None;
        }
        // The table top is the plane y = 0.
        let t = -near.y / dir.y;
        if t <= 0.0 {
            return None;
        }
        let hit = near + dir * t;
        (hit.x * hit.x + hit.z * hit.z <= TABLE_RADIUS * TABLE_RADIUS).then_some((hit.x, hit.z))
    }

    /// The candle whose foot is nearest where `u`, `v` meets the table,
    /// within a few centimetres.
    pub(crate) fn pick(&self, u: f32, v: f32) -> Option<u32> {
        let (x, z) = self.table_point(u, v)?;
        self.candles
            .iter()
            .map(|(id, p)| (*id, (p.at.0 - x).powi(2) + (p.at.1 - z).powi(2)))
            .filter(|(_, d)| *d <= PICK_RADIUS * PICK_RADIUS)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(id, _)| id)
    }

    /// Where a point is on screen, in uv (origin top left).
    pub(crate) fn screen_point(&self, x: f32, y: f32, z: f32) -> (f32, f32) {
        let p = Vector3::new(x, y, z)
            .project(&self.camera.view_matrix(), &self.camera.projection_matrix());
        ((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5)
    }

    /// As [`screen_point`](Self::screen_point), `None` behind the camera.
    pub(crate) fn project(&self, x: f32, y: f32, z: f32) -> Option<(f32, f32)> {
        let eye = Vector3::new(x, y, z).apply_matrix4(&self.camera.view_matrix());
        (eye.z < 0.0).then(|| self.screen_point(x, y, z))
    }

    /// Where candle `id`'s flame is on screen.
    pub(crate) fn flame_on_screen(&self, id: u32) -> Option<(f32, f32)> {
        let p = self.candles.get(&id)?;
        Some(self.screen_point(p.at.0, self.flame_y(), p.at.1))
    }

    /// How high above the table a flame's middle is.
    fn flame_y(&self) -> f32 {
        if self.fire.is_some() {
            // The bright part of the plume.
            return 0.06;
        }
        let c = self.candle(0);
        c.flame_base() + c.flame_height * 0.4
    }

    /// Turn the view by a drag, or zoom by `wheel`.
    pub(crate) fn orbit(&mut self, dx: f32, dy: f32, wheel: f32) {
        let size = (self.size.0 as f32, self.size.1 as f32);
        let event = PointerEvent {
            dx,
            dy,
            wheel,
            rotating: wheel == 0.0,
            panning: false,
            cursor: (0.0, 0.0),
        };
        self.controls.update(event, &mut self.camera, size);
    }

    // -- the frame ------------------------------------------------------------

    /// Flicker, burn, draw — and, given `record`, copy the frame into it
    /// before it goes on screen. `None` when nothing was drawn (off screen);
    /// else, recording, the copy's uncommitted command buffer.
    pub(crate) fn frame(&mut self, dt: f32, record: Option<Id>) -> Option<Option<Owned>> {
        let dt = dt.clamp(0.0, 0.1);
        self.time += dt;
        let fade = (-dt * 3.5).exp();
        for p in self.candles.values_mut() {
            // The meshes and lights flicker at the same calm pace.
            p.handle.update(&mut self.scene, self.time * CALM);
            p.flare *= fade;
        }
        if let Some(fire) = self.fire.as_mut()
            && !fire.burn(&self.candles, self.time, dt, self.fire_amount)
        {
            return None;
        }
        let frame = self.surface.next_frame()?;
        let pass = frame.attachments();
        // The table, then the fire over it, then (recording) the copy, then
        // on screen.
        let off = pass.without_drawable();
        self.renderer
            .render(&mut self.scene, &self.camera, &off)
            .ok()?;
        if let Some(fire) = self.fire.as_mut() {
            fire.sim
                .render_over(&mut self.renderer, &self.camera, &off)
                .ok()?;
        }
        let copy = match record {
            // `dst` is the caller's texture, the view's size (see `threers_candles_frame_record`).
            Some(dst) => Some(unsafe { self.renderer.copy_frame(&pass, dst) }.ok()?),
            None => None,
        };
        self.renderer.present(&pass).ok()?;
        Some(copy)
    }
}

/// A dark room and a round wooden table.
pub(crate) fn room(scene: &mut Scene) {
    scene.background = Color::from_hex(0x050403);
    scene.add_light(AmbientLight::new(Color::from_hex(0x2a3550), 0.06));
    let wood = StandardMaterial::new(Color::from_hex(0x4a2f1c)).with_roughness(0.6);
    let mut table = Object3D::mesh(Mesh::new(
        CylinderGeometry::new(
            TABLE_RADIUS,
            TABLE_RADIUS,
            0.04,
            96,
            1,
            false,
            0.0,
            std::f32::consts::TAU,
        ),
        wood.into(),
    ));
    table.position = Vector3::new(0.0, -0.02, 0.0);
    table.receive_shadow = true;
    scene.add(table);
}

/// The sweep: a disc on the table, drawn additively — a line at the hand's
/// angle and a fading trail behind it, faint at the middle and the rim.
/// data[0]: the hand's angle in turns, strength; data[1]: the radius.
const SWEEP: &str = r#"
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = in.world_pos.xz;
    let r = length(p) / u_data.data[1].x;
    // Turns from the far side (-z), clockwise seen from above.
    let a = fract(atan2(p.x, -p.y) / 6.2831853 + 1.0);
    let behind = fract(u_data.data[0].x - a + 1.0);
    let hand = exp(-behind * 220.0) + exp(-(1.0 - behind) * 600.0);
    let trail = exp(-behind * 7.0) * 0.18;
    let fade = smoothstep(0.04, 0.2, r) * (1.0 - smoothstep(0.9, 1.0, r));
    let c = vec3<f32>(1.0, 0.82, 0.6) * (hand * 0.9 + trail) * fade * u_data.data[0].y;
    return vec4<f32>(framebuffer_encode(apply_tone_mapping(c)), 1.0);
}
"#;

pub(crate) fn sweep(scene: &mut Scene) -> (ObjectId, ShaderMaterial) {
    let mut data = vec![[0.0f32; 4]; 16];
    data[1] = [TABLE_RADIUS, 0.0, 0.0, 0.0];
    let mut material = ShaderMaterial::new(SWEEP)
        .with_data(data)
        .with_transparent(true)
        .with_depth_write(false);
    material.render_state.blending = Blending::Additive;
    let mut disc = Object3D::mesh(Mesh::new(
        CircleGeometry::new(TABLE_RADIUS, 96, 0.0, std::f32::consts::TAU),
        Material::Shader(material.clone()),
    ));
    // Flat on the table, a hair above it.
    disc.quaternion = Euler::new(-std::f32::consts::FRAC_PI_2, 0.0, 0.0).to_quaternion();
    disc.position = Vector3::new(0.0, 0.0008, 0.0);
    disc.visible = false;
    (scene.add(disc), material)
}
