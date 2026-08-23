//! Native Node.js bindings for threers (`npm i threers-node` via napi-rs).
//!
//! Deno and the browser use the wasm ESM package (`threers` / `crates/threers-js`).
//! This crate is the **native** Node addon — Neon is intentionally not used;
//! napi-rs provides `#[napi]` macros and generated TypeScript types.

#![deny(clippy::all)]

use napi::bindgen_prelude::*;
use napi_derive::napi;

use threers::cameras::PerspectiveCamera;
use threers::core::{Mesh, Object3D, ObjectId};
use threers::geometries::{BoxGeometry, SphereGeometry};
use threers::materials::{BasicMaterial, Material, StandardMaterial};
use threers::math::{Color, Vector3};
use threers::scene::Scene;

#[napi]
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

// ---- math ------------------------------------------------------------------

#[napi(js_name = "Vector3")]
#[derive(Clone, Copy)]
pub struct JsVector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[napi]
impl JsVector3 {
    #[napi(constructor)]
    pub fn new(x: Option<f64>, y: Option<f64>, z: Option<f64>) -> Self {
        Self {
            x: x.unwrap_or(0.0),
            y: y.unwrap_or(0.0),
            z: z.unwrap_or(0.0),
        }
    }

    fn to_inner(self) -> Vector3 {
        Vector3::new(self.x as f32, self.y as f32, self.z as f32)
    }

    fn from_inner(v: Vector3) -> Self {
        Self {
            x: v.x as f64,
            y: v.y as f64,
            z: v.z as f64,
        }
    }

    #[napi]
    pub fn length(&self) -> f64 {
        self.to_inner().length() as f64
    }

    #[napi]
    pub fn normalized(&self) -> Self {
        Self::from_inner(self.to_inner().normalize())
    }
}

#[napi(js_name = "Color")]
#[derive(Clone, Copy)]
pub struct JsColor {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

#[napi]
impl JsColor {
    #[napi(constructor)]
    pub fn new(r: Option<f64>, g: Option<f64>, b: Option<f64>) -> Self {
        Self {
            r: r.unwrap_or(1.0),
            g: g.unwrap_or(1.0),
            b: b.unwrap_or(1.0),
        }
    }

    #[napi(factory)]
    pub fn from_hex(hex: u32) -> Self {
        let c = Color::from_hex(hex);
        Self {
            r: c.r as f64,
            g: c.g as f64,
            b: c.b as f64,
        }
    }
}

// ---- scene graph -----------------------------------------------------------

#[napi(js_name = "PerspectiveCamera")]
pub struct JsPerspectiveCamera {
    inner: PerspectiveCamera,
}

#[napi]
impl JsPerspectiveCamera {
    #[napi(constructor)]
    pub fn new(
        fov: Option<f64>,
        aspect: Option<f64>,
        near: Option<f64>,
        far: Option<f64>,
    ) -> Self {
        Self {
            inner: PerspectiveCamera::new(
                fov.unwrap_or(50.0) as f32,
                aspect.unwrap_or(1.0) as f32,
                near.unwrap_or(0.1) as f32,
                far.unwrap_or(2000.0) as f32,
            ),
        }
    }

    #[napi]
    pub fn set_position(&mut self, x: f64, y: f64, z: f64) {
        self.inner.position = Vector3::new(x as f32, y as f32, z as f32);
    }

    #[napi]
    pub fn look_at(&mut self, x: f64, y: f64, z: f64) {
        self.inner.look_at(Vector3::new(x as f32, y as f32, z as f32));
    }

    #[napi]
    pub fn set_aspect(&mut self, aspect: f64) {
        self.inner.aspect = aspect as f32;
    }
}

#[napi(js_name = "Scene")]
pub struct JsScene {
    inner: Scene,
    handles: Vec<ObjectId>,
}

impl Default for JsScene {
    fn default() -> Self {
        Self::new()
    }
}

#[napi]
impl JsScene {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: Scene::new(),
            handles: Vec::new(),
        }
    }

    /// Add an axis-aligned box mesh. Returns a stable handle index.
    // Every dimension and placement value is optional on the JS side, so they
    // arrive as a flat list of `Option`s rather than an options object.
    #[allow(clippy::too_many_arguments)]
    #[napi]
    pub fn add_box(
        &mut self,
        width: Option<f64>,
        height: Option<f64>,
        depth: Option<f64>,
        color: Option<u32>,
        x: Option<f64>,
        y: Option<f64>,
        z: Option<f64>,
    ) -> u32 {
        let geom = BoxGeometry::new(
            width.unwrap_or(1.0) as f32,
            height.unwrap_or(1.0) as f32,
            depth.unwrap_or(1.0) as f32,
        );
        let mat = Material::Standard(StandardMaterial {
            color: Color::from_hex(color.unwrap_or(0xff6633)),
            ..StandardMaterial::default()
        });
        let mut obj = Object3D::mesh(Mesh::new(geom, mat));
        obj.position = Vector3::new(
            x.unwrap_or(0.0) as f32,
            y.unwrap_or(0.0) as f32,
            z.unwrap_or(0.0) as f32,
        );
        let id = self.inner.arena.insert(obj);
        self.inner.arena.add_child(self.inner.root, id);
        self.handles.push(id);
        (self.handles.len() - 1) as u32
    }

    #[napi]
    pub fn add_sphere(
        &mut self,
        radius: Option<f64>,
        color: Option<u32>,
        x: Option<f64>,
        y: Option<f64>,
        z: Option<f64>,
    ) -> u32 {
        let geom = SphereGeometry::new(radius.unwrap_or(0.5) as f32, 24, 16);
        let mat = Material::Basic(BasicMaterial::new(Color::from_hex(
            color.unwrap_or(0x6ea8fe),
        )));
        let mut obj = Object3D::mesh(Mesh::new(geom, mat));
        obj.position = Vector3::new(
            x.unwrap_or(0.0) as f32,
            y.unwrap_or(0.0) as f32,
            z.unwrap_or(0.0) as f32,
        );
        let id = self.inner.arena.insert(obj);
        self.inner.arena.add_child(self.inner.root, id);
        self.handles.push(id);
        (self.handles.len() - 1) as u32
    }

    #[napi]
    pub fn set_background(&mut self, hex: u32) {
        self.inner.background = Color::from_hex(hex);
    }

    #[napi]
    pub fn set_position(&mut self, handle: u32, x: f64, y: f64, z: f64) -> Result<()> {
        let id = *self
            .handles
            .get(handle as usize)
            .ok_or_else(|| Error::from_reason("bad object handle"))?;
        let obj = self
            .inner
            .arena
            .get_mut(id)
            .ok_or_else(|| Error::from_reason("object removed"))?;
        obj.position = Vector3::new(x as f32, y as f32, z as f32);
        Ok(())
    }

    #[napi]
    pub fn object_count(&self) -> u32 {
        self.handles.len() as u32
    }
}

// ---- headless --------------------------------------------------------------

#[cfg(feature = "headless")]
#[napi(js_name = "HeadlessRenderer")]
pub struct JsHeadlessRenderer {
    inner: threers::HeadlessRenderer,
}

#[cfg(feature = "headless")]
#[napi]
impl JsHeadlessRenderer {
    #[napi(constructor)]
    pub fn new(width: Option<u32>, height: Option<u32>, supersample: Option<u32>) -> Result<Self> {
        let inner = threers::HeadlessRenderer::builder()
            .size(width.unwrap_or(640), height.unwrap_or(480))
            .supersample(supersample.unwrap_or(1).max(1))
            .build()
            .map_err(Error::from_reason)?;
        Ok(Self { inner })
    }

    /// Render and return tightly packed RGBA8 bytes (`width * height * 4`).
    #[napi]
    pub fn render_rgba(
        &mut self,
        scene: &mut JsScene,
        camera: &JsPerspectiveCamera,
    ) -> Buffer {
        let rgba = self
            .inner
            .render_to_rgba(&mut scene.inner, &camera.inner);
        Buffer::from(rgba)
    }

    /// Render and return a PNG as bytes.
    #[napi]
    pub fn render_png(
        &mut self,
        scene: &mut JsScene,
        camera: &JsPerspectiveCamera,
    ) -> Buffer {
        let (w, h) = self.inner.render_size();
        let rgba = self
            .inner
            .render_to_rgba(&mut scene.inner, &camera.inner);
        let png = threers::encode_png(w, h, &rgba);
        Buffer::from(png)
    }

    #[napi]
    pub fn size(&self) -> Vec<u32> {
        let (w, h) = self.inner.render_size();
        vec![w, h]
    }
}

// ---- animation -------------------------------------------------------------

#[cfg(feature = "animation")]
#[napi(js_name = "Tween")]
pub struct JsTween {
    inner: threers_animation::tween::Tween<f32>,
}

#[cfg(feature = "animation")]
#[napi]
impl JsTween {
    #[napi(constructor)]
    pub fn new(
        start: Option<f64>,
        end: Option<f64>,
        duration: Option<f64>,
        easing: Option<String>,
    ) -> Result<Self> {
        use threers_animation::tween::Tween;
        let e = parse_easing(easing.as_deref().unwrap_or("cubicOut"))?;
        Ok(Self {
            inner: Tween::new(
                start.unwrap_or(0.0) as f32,
                end.unwrap_or(1.0) as f32,
                duration.unwrap_or(1.0) as f32,
            )
            .easing(e),
        })
    }

    #[napi]
    pub fn update(&mut self, dt: f64) -> f64 {
        self.inner.update(dt as f32) as f64
    }

    #[napi]
    pub fn value(&self) -> f64 {
        self.inner.value() as f64
    }

    #[napi]
    pub fn is_finished(&self) -> bool {
        self.inner.is_finished()
    }
}

#[cfg(feature = "animation")]
fn parse_easing(name: &str) -> Result<threers_animation::easing::Easing> {
    use threers_animation::easing::Easing;
    Ok(match name {
        "linear" => Easing::Linear,
        "quadIn" => Easing::QuadIn,
        "quadOut" => Easing::QuadOut,
        "quadInOut" => Easing::QuadInOut,
        "cubicIn" => Easing::CubicIn,
        "cubicOut" => Easing::CubicOut,
        "cubicInOut" => Easing::CubicInOut,
        other => {
            return Err(Error::from_reason(format!(
                "unknown easing '{other}' (try linear, cubicOut, cubicInOut, …)"
            )))
        }
    })
}

// ---- physics ---------------------------------------------------------------

#[cfg(feature = "physics")]
#[napi(js_name = "PhysicsWorld")]
pub struct JsPhysicsWorld {
    inner: threers_physics::prelude::World,
    bodies: Vec<threers_physics::prelude::BodyId>,
}

#[cfg(feature = "physics")]
impl Default for JsPhysicsWorld {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "physics")]
#[napi]
impl JsPhysicsWorld {
    #[napi(constructor)]
    pub fn new() -> Self {
        use threers_physics::prelude::*;
        let mut world = World::new();
        world.add_body(RigidBody::fixed().shape(Shape::ground()));
        Self {
            inner: world,
            bodies: Vec::new(),
        }
    }

    /// Add a dynamic ball. Returns a stable handle index.
    #[napi]
    pub fn add_ball(
        &mut self,
        radius: Option<f64>,
        x: Option<f64>,
        y: Option<f64>,
        z: Option<f64>,
        restitution: Option<f64>,
    ) -> u32 {
        use threers_physics::prelude::*;
        let id = self.inner.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(radius.unwrap_or(0.5) as f32))
                .translation(Vector3::new(
                    x.unwrap_or(0.0) as f32,
                    y.unwrap_or(5.0) as f32,
                    z.unwrap_or(0.0) as f32,
                ))
                .restitution(restitution.unwrap_or(0.4) as f32),
        );
        self.bodies.push(id);
        (self.bodies.len() - 1) as u32
    }

    #[napi]
    pub fn step(&mut self, dt: f64) {
        self.inner.step(dt as f32);
    }

    #[napi]
    pub fn body_translation(&self, handle: u32) -> Result<JsVector3> {
        let id = *self
            .bodies
            .get(handle as usize)
            .ok_or_else(|| Error::from_reason("bad body handle"))?;
        let body = self
            .inner
            .body(id)
            .ok_or_else(|| Error::from_reason("body removed"))?;
        Ok(JsVector3::from_inner(body.translation()))
    }

    #[napi]
    pub fn body_count(&self) -> u32 {
        self.bodies.len() as u32
    }
}
