//! The demo in a headset: the whole UI on a [`Panel`] standing in a room
//! (a threers scene), the headset's pointers given to the app as a
//! screen's mouse and touches. The pieces are platform-free:
//!
//! - [`Room`]: the scene around the panel, and its animation;
//! - [`UiPanel`]: the panel's input into the app, and the UI drawn on it,
//!   standing in the room, depth-sorted;
//!
//! The browser build (WebXR, `lib.rs` and `gpu.rs`) drives them from its
//! page's frames; the native Vision Pro app (`threers-ui-demo-visionos`)
//! is an [`xr::App`](threers::xr::App) made of them.

use threers::extras::candle::{Candle, CandleHandle, FlameStyle};
use threers::renderer::UiPass;
use threers::xr::{Frame, Gpu, HoverArea, Panel, PanelInput, PanelPass, Pointer};
use threers_ui::input::{PointerButton, TouchPhase};
use threers_ui::TextMeasure;

use threers::{
    AmbientLight, BoxGeometry, Color, DirectionalLight, Euler, HemisphereLight,
    IcosahedronGeometry, Matrix4, Mesh, Object3D, ObjectId, PerspectiveCamera, PlaneGeometry,
    Scene, SphereGeometry, StandardMaterial, TorusGeometry, TorusKnotGeometry, Vector3,
};

use crate::app::App;

/// The UI's size on the panel, in its pixels, and the device pixels each
/// in the panel's texture to start with: about what the panel covers of a
/// headset's view ([`UiPanel::update_scale`] follows the eyes from there).
pub const PANEL_PX: (f32, f32) = (1280.0, 800.0);
pub const PANEL_SCALE: f32 = 1.25;

/// The lens the eyes see through: its near and far planes (a headset's
/// projections use them; visionOS takes no near plane under 0.1 m).
pub fn lens() -> PerspectiveCamera {
    PerspectiveCamera::new(60.0, 1.0, 0.1, 100.0)
}

/// The room around the panel: a floor, shapes turning in a ring, candles
/// on a plinth.
pub struct Room {
    floor: ObjectId,
    shapes: Vec<ObjectId>,
    candles: Vec<CandleHandle>,
}

impl Room {
    /// Furnishes `scene` with the room.
    pub fn furnish(scene: &mut Scene) -> Room {
        let (floor, shapes, candles) = furnish(scene);
        Room {
            floor,
            shapes,
            candles,
        }
    }

    /// The room at `time` seconds: the shapes turn, the candles flicker;
    /// see-through (AR), no sky and no floor — the real room shows.
    pub fn animate(&self, scene: &mut Scene, time: f64, see_through: bool) {
        let t = time as f32;
        for c in &self.candles {
            c.update(scene, t);
        }
        for (i, id) in self.shapes.iter().enumerate() {
            if let Some(o) = scene.get_mut(*id) {
                o.quaternion = Euler::new(
                    t * (0.2 + 0.05 * i as f32),
                    t * (0.35 + 0.07 * i as f32),
                    0.0,
                )
                .to_quaternion();
            }
        }
        scene.background_alpha = if see_through { 0.0 } else { 1.0 };
        if let Some(f) = scene.get_mut(self.floor) {
            f.visible = !see_through;
        }
    }
}

/// The demo's UI on a panel: the frame's pointers into the app, and the UI
/// drawn on the panel, standing in the scene.
pub struct UiPanel {
    pub panel: Panel,
    pointers: Vec<Pointer>,
    /// Made on the first draw, for the target's format.
    pass: Option<PanelPass>,
    /// Device pixels a UI pixel in the panel's texture, now.
    pub scale: f32,
}

impl Default for UiPanel {
    fn default() -> Self {
        UiPanel {
            panel: Panel::new(PANEL_PX.0, PANEL_PX.1),
            pointers: vec![],
            pass: None,
            scale: PANEL_SCALE,
        }
    }
}

impl UiPanel {
    /// `frame`'s pointers on the panel given to `app`, as a screen's mouse
    /// and touches.
    pub fn input(&mut self, app: &mut App, frame: &Frame, measure: &mut dyn TextMeasure) {
        let (inputs, pointers) = self.panel.step(frame);
        self.pointers = pointers;
        let (b, time) = (PointerButton::Primary, frame.time);
        for i in inputs {
            match i {
                PanelInput::Move(x, y) => app.pointer_move(x, y, measure),
                PanelInput::Down(x, y) => app.pointer_down(x, y, b, false, measure),
                PanelInput::Up(x, y) => app.pointer_up(x, y, b),
                PanelInput::Wheel(x, y, dx, dy) => app.wheel(x, y, dx, dy),
                PanelInput::TouchStart(id, x, y) => {
                    app.touch(id, TouchPhase::Start, x, y, time, measure)
                }
                PanelInput::TouchMove(id, x, y) => {
                    app.touch(id, TouchPhase::Move, x, y, time, measure)
                }
                PanelInput::TouchEnd(id, x, y) => {
                    app.touch(id, TouchPhase::End, x, y, time, measure)
                }
                PanelInput::TouchCancel(id, x, y) => {
                    app.touch(id, TouchPhase::Cancel, x, y, time, measure)
                }
            }
        }
    }

    /// The texture's scale for what the eyes of `gpu` can see of the panel,
    /// a quarter more (its smaller copies smooth the rest), in steps of a
    /// quarter between 1 and 3: returns whether it changed (the UI's pixel
    /// ratio has to follow).
    pub fn update_scale(&mut self, gpu: &Gpu<'_>) -> bool {
        let Some(need) = PanelPass::ui_scale_for(gpu.source, &self.panel) else {
            return false;
        };
        let want = ((need * 1.25 * 4.0).ceil() / 4.0).clamp(1.0, 3.0);
        // Not back and forth at a step's edge: two steps or more.
        if (want - self.scale).abs() < 0.375 {
            return false;
        }
        self.scale = want;
        true
    }

    /// The app's controls marked, in `gpu`'s eyes, for the system to light
    /// up when they are looked at (Apple Vision Pro); elsewhere nothing.
    /// After [`draw`](Self::draw). Returns how many.
    pub fn mark_hover(
        &mut self,
        gpu: &Gpu<'_>,
        regions: &[threers_ui::spatial::HoverRegion],
    ) -> usize {
        let Some(pass) = &mut self.pass else { return 0 };
        let areas: Vec<HoverArea> = regions
            .iter()
            .map(|r| HoverArea {
                id: r.id,
                rect: [r.rect.x, r.rect.y, r.rect.width, r.rect.height],
                radius: r.radius,
            })
            .collect();
        pass.draw_hover_regions(gpu.device, gpu.queue, gpu.source, &self.panel, &areas)
    }

    /// The UI `ui_scene` (drawn by `ui` at `scale` device pixels a UI pixel,
    /// over `clear`) on the panel, standing in `scene`, in each eye of
    /// `gpu`: the scene through `lens`, the panel depth-sorted with it.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        gpu: &mut Gpu<'_>,
        ui: &mut UiPass,
        ui_scene: &threers::ui::scene::Scene,
        scale: f32,
        clear: [f32; 4],
        scene: &mut Scene,
        lens: &PerspectiveCamera,
    ) {
        let pass = self
            .pass
            .get_or_insert_with(|| PanelPass::new(gpu.device, gpu.format));
        let size = (
            (PANEL_PX.0 * scale).round() as u32,
            (PANEL_PX.1 * scale).round() as u32,
        );
        let format = pass.ui_format();
        let view = pass.ui_target(gpu.device, size).clone();
        let damage = ui.draw_retained(
            gpu.device, gpu.queue, ui_scene, scale, &view, size, format, clear,
        );
        // Nothing changed: the texture's smaller copies still stand.
        if damage == threers::ui::scene::SceneDamage::None {
            pass.ui_unchanged();
        }
        pass.draw_in_scene(
            gpu.renderer,
            scene,
            lens,
            gpu.source,
            &Matrix4::identity(),
            gpu.target,
            &self.panel,
            &self.pointers,
        );
    }
}

/// The room's furniture: a sky, a floor, eight shapes in a ring around the
/// viewer at arm's reach and beyond, lit by the sky and a sun, and three
/// candles on a plinth beside the panel.
fn furnish(scene: &mut Scene) -> (ObjectId, Vec<ObjectId>, Vec<CandleHandle>) {
    scene.background = Color::from_hex(0x1f2533);
    scene.add_light(HemisphereLight::new(
        Color::from_hex(0xdfe8ff),
        Color::from_hex(0x2a2520),
        0.9,
    ));
    scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.15));
    let mut sun = Object3D::light(
        DirectionalLight::new(Color::from_hex(0xffffff), 1.6)
            .with_direction(Vector3::new(-0.4, -1.0, -0.3)),
    );
    sun.position = Vector3::new(3.0, 6.0, 2.0);
    scene.add(sun);
    let mut floor = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(30.0, 30.0),
        StandardMaterial::new(Color::from_hex(0x2c3342))
            .with_roughness(0.95)
            .into(),
    ));
    floor.quaternion = Euler::new(-std::f32::consts::FRAC_PI_2, 0.0, 0.0).to_quaternion();
    let floor = scene.add(floor);
    let palette = [
        0xe8664a, 0xf2b33d, 0x5cc98a, 0x4fa3e8, 0xa77de8, 0xe86fa8, 0x3cc6c0, 0xf28c3d,
    ];
    let mut shapes = vec![];
    for (i, hex) in palette.into_iter().enumerate() {
        let geometry = match i % 5 {
            0 => TorusKnotGeometry::new(0.22, 0.07, 128, 16, 2, 3),
            1 => SphereGeometry::new(0.25, 48, 24),
            2 => BoxGeometry::new(0.4, 0.4, 0.4),
            3 => IcosahedronGeometry::new(0.3, 0),
            _ => TorusGeometry::new(0.25, 0.09, 24, 64, std::f32::consts::TAU),
        };
        let material = StandardMaterial::new(Color::from_hex(hex))
            .with_roughness(0.35)
            .with_metalness(0.1);
        let mut o = Object3D::mesh(Mesh::new(geometry, material.into()));
        // A ring, leaving the panel's side (ahead, -Z) open.
        let a = std::f32::consts::FRAC_PI_4 * 1.1
            + i as f32 * (std::f32::consts::TAU - std::f32::consts::FRAC_PI_2 * 1.1) / 7.0;
        let r = if i % 2 == 0 { 2.2 } else { 3.2 };
        o.position = Vector3::new(-a.sin() * r, 1.0 + 0.35 * (i % 3) as f32, -a.cos() * r);
        shapes.push(scene.add(o));
    }
    // A plinth to the panel's right, and candles of three heights on it,
    // their light strong enough to show against the daylight.
    let mut plinth = Object3D::mesh(Mesh::new(
        BoxGeometry::new(0.34, 0.95, 0.34),
        StandardMaterial::new(Color::from_hex(0x3a3f4b))
            .with_roughness(0.7)
            .into(),
    ));
    plinth.position = Vector3::new(0.98, 0.475, -1.15);
    plinth.receive_shadow = true;
    plinth.cast_shadow = true;
    scene.add(plinth);
    let candles = [
        (-0.08, -0.06, 0.16),
        (0.07, -0.04, 0.11),
        (0.0, 0.08, 0.075),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, (x, z, h))| {
        let mut c = Candle::new(FlameStyle::Realistic)
            .with_seed(i as f32 * 3.7)
            .with_size(h, 0.026);
        c.light_intensity = 0.35;
        c.add_to(scene, Vector3::new(0.98 + x, 0.95, -1.15 + z))
    })
    .collect();
    (floor, shapes, candles)
}
