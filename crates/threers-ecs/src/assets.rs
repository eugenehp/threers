//! Loading files without stopping the game.
//! [`Assets::load`] returns a [`Handle`] at once; the data arrives later. Natively a
//! background thread both reads and parses, so a large level never costs a frame;
//! in a browser the bytes come through `fetch`. A handle stays
//! valid when the file changes, because the data behind it is what gets swapped,
//! which is what makes hot reload work.
//!
//! ```no_run
//! use threers_ecs::assets::*;
//! use threers_ecs::prelude::*;
//!
//! fn setup(app: &mut App) {
//!     app.plugin(AssetsPlugin::new("assets"));
//!     app.world.spawn_scene("level.glb");
//! }
//! ```
//!
//! # Components from an authoring tool
//!
//! A node's glTF `extras` name its components: a custom property `Health` holding
//! `{"hp": 5}`, or `Coin` holding `true` for a component with nothing to set. Any
//! type passed to [`World::register`] works. A capitalised name that is not
//! registered is reported, since it is most likely a typo; other keys are left
//! alone, as authoring tools add their own.
//!
//! With the `physics` feature three more keys give level geometry collision:
//!
//! | Key | Gives |
//! |---|---|
//! | `solid: true` | a fixed body shaped exactly like the mesh: floors, walls |
//! | `sensor: true` | the same as a trigger, reported in `Collision` events |
//! | `body: "dynamic"` | a moving body, the mesh's convex hull: crates, barrels |

use crate::app::{App, Plugin};
use crate::hud::{with_hud, Hud};
use crate::registry::FieldValue;
use crate::schedule::System;
use crate::states::GameState;
use crate::world::{Component, Entity, World};
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Mutex;
use threers::animation::{AnimationClip, KeyframeTrack};
use threers::captions::CaptionFont;
use threers::core::{Mesh, Object3D};
use threers::geometries::BoxGeometry;
use threers::loaders::json::Value;
use threers::loaders::{add_to_scene_mapped, GltfLoader, GltfScene};
use threers::materials::{BasicMaterial, Material};
use threers::math::Color;

/// Something that can be made from a file's bytes. `Send`, because it is made on a
/// background thread.
pub trait Asset: Sized + Send + Sync + 'static {
    /// # Errors
    ///
    /// With a message for the player's log if the bytes are not this kind of file.
    fn parse(bytes: &[u8]) -> Result<Self, String>;
}

/// The bytes as they are, for formats the game reads itself.
pub struct Bytes(pub Vec<u8>);

impl Asset for Bytes {
    fn parse(bytes: &[u8]) -> Result<Self, String> {
        Ok(Bytes(bytes.to_vec()))
    }
}

/// A loaded or loading asset. Cheap to clone.
pub struct Handle<T> {
    path: Rc<str>,
    kind: PhantomData<T>,
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Self {
            path: self.path.clone(),
            kind: PhantomData,
        }
    }
}

impl<T> Handle<T> {
    pub fn path(&self) -> &str {
        &self.path
    }
}

/// An asset's data has arrived.
#[derive(Debug, Clone, PartialEq)]
pub struct AssetLoaded {
    pub path: String,
    /// False the first time, true when a change on disk replaced it.
    pub reloaded: bool,
}

/// An asset could not be read or parsed.
#[derive(Debug, Clone, PartialEq)]
pub struct AssetFailed {
    pub path: String,
    pub error: String,
}

/// A font for the HUD, from a TrueType or OpenType file.
pub struct FontAsset(Vec<u8>);

impl Asset for FontAsset {
    fn parse(bytes: &[u8]) -> Result<Self, String> {
        CaptionFont::from_ttf_bytes(bytes).map_err(|e| format!("{e:?}"))?;
        Ok(FontAsset(bytes.to_vec()))
    }
}

/// Many files as one, so a level costs one request rather than dozens.
///
/// A pack is a count, then each file's name and length, then the bytes.
pub struct Pack(HashMap<String, Vec<u8>>);

impl Pack {
    /// Join files into a pack's bytes.
    pub fn build<'a>(files: impl IntoIterator<Item = (&'a str, &'a [u8])>) -> Vec<u8> {
        let files: Vec<_> = files.into_iter().collect();
        let mut out = b"THPK".to_vec();
        out.extend_from_slice(&(files.len() as u32).to_le_bytes());
        for (name, bytes) in &files {
            out.extend_from_slice(&(name.len() as u32).to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        }
        for (_, bytes) in &files {
            out.extend_from_slice(bytes);
        }
        out
    }
}

impl Asset for Pack {
    fn parse(bytes: &[u8]) -> Result<Self, String> {
        fn take<'a>(bytes: &mut &'a [u8], count: usize) -> Result<&'a [u8], String> {
            if bytes.len() < count {
                return Err("pack is cut short".to_owned());
            }
            let (head, tail) = bytes.split_at(count);
            *bytes = tail;
            Ok(head)
        }
        fn number(bytes: &mut &[u8]) -> Result<usize, String> {
            let head = take(bytes, 4)?;
            Ok(u32::from_le_bytes([head[0], head[1], head[2], head[3]]) as usize)
        }
        let mut rest = bytes;
        if take(&mut rest, 4)? != b"THPK" {
            return Err("not a pack".to_owned());
        }
        let mut entries = Vec::new();
        for _ in 0..number(&mut rest)? {
            let length = number(&mut rest)?;
            let name = String::from_utf8(take(&mut rest, length)?.to_vec())
                .map_err(|_| "a pack entry's name is not text".to_owned())?;
            entries.push((name, number(&mut rest)?));
        }
        let mut files = HashMap::new();
        for (name, length) in entries {
            files.insert(name, take(&mut rest, length)?.to_vec());
        }
        Ok(Pack(files))
    }
}

type Parsed = Arc<dyn Any + Send + Sync>;
type Parse = fn(&[u8]) -> Result<Parsed, String>;

struct Slot {
    parse: Parse,
    kind: TypeId,
    value: Option<Parsed>,
    error: Option<String>,
    /// How many times data has landed. Lets users notice a reload.
    version: u32,
    #[cfg(not(target_arch = "wasm32"))]
    modified: Option<std::time::SystemTime>,
}

pub(crate) type Arrival = (String, Result<Parsed, String>);

/// Every asset in the game, as a resource.
pub struct Assets {
    root: PathBuf,
    slots: HashMap<String, Slot>,
    /// Files from packs that have arrived, ready for anyone who asks.
    packed: HashMap<String, Vec<u8>>,
    /// Packs still on their way.
    pending_packs: Vec<Handle<Pack>>,
    sender: Sender<Arrival>,
    receiver: Receiver<Arrival>,
    /// Watch loaded files and reload them when they change. On by default in a
    /// debug build. There is nothing to watch in a browser.
    pub hot_reload: bool,
    #[cfg(not(target_arch = "wasm32"))]
    next_check: f64,
}

impl Assets {
    /// Paths are relative to `root`: a directory natively, a URL prefix on the web.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let (sender, receiver) = channel();
        Self {
            root: root.into(),
            slots: HashMap::new(),
            packed: HashMap::new(),
            pending_packs: Vec::new(),
            sender,
            receiver,
            hot_reload: cfg!(debug_assertions),
            #[cfg(not(target_arch = "wasm32"))]
            next_check: 0.0,
        }
    }

    /// Start loading, or find what is already loaded. Loading a path twice gives
    /// two handles to one asset.
    ///
    /// # Panics
    ///
    /// If the path was loaded as a different type, which could never arrive.
    pub fn load<T: Asset>(&mut self, path: &str) -> Handle<T> {
        if let Some(slot) = self.slots.get(path) {
            assert!(
                slot.kind == TypeId::of::<T>(),
                "{path} is already loaded as another type"
            );
        } else {
            self.slots.insert(path.to_owned(), Self::slot::<T>());
            match self.packed.get(path) {
                Some(bytes) => self.deliver(path, bytes.clone()),
                // A pack on its way may hold this, so it waits to see.
                None if !self.pending_packs.is_empty() => {}
                None => self.read(path),
            }
        }
        Handle {
            path: path.into(),
            kind: PhantomData,
        }
    }

    /// Fetch a [`Pack`]. Assets asked for meanwhile wait for it, and come from it
    /// if it has them.
    pub fn load_pack(&mut self, path: &str) {
        let handle = self.load::<Pack>(path);
        self.pending_packs.push(handle);
    }

    /// How many assets have finished, out of how many were asked for.
    pub fn progress(&self) -> (usize, usize) {
        let done = self
            .slots
            .values()
            .filter(|slot| slot.value.is_some() || slot.error.is_some())
            .count();
        (done, self.slots.len())
    }

    /// Supply an asset's bytes directly, as from `include_bytes!`.
    pub fn insert_bytes<T: Asset>(&mut self, path: &str, bytes: Vec<u8>) -> Handle<T> {
        self.slots
            .entry(path.to_owned())
            .or_insert_with(Self::slot::<T>);
        // Through the channel, so it reports the way a load does.
        self.deliver(path, bytes);
        Handle {
            path: path.into(),
            kind: PhantomData,
        }
    }

    fn slot<T: Asset>() -> Slot {
        Slot {
            parse: |bytes| T::parse(bytes).map(|value| Arc::new(value) as Parsed),
            kind: TypeId::of::<T>(),
            value: None,
            error: None,
            version: 0,
            #[cfg(not(target_arch = "wasm32"))]
            modified: None,
        }
    }

    /// The data, once it has arrived. After a reload this is the new data.
    pub fn get<T: Asset>(&self, handle: &Handle<T>) -> Option<Arc<T>> {
        let value = self.slots.get(&*handle.path)?.value.clone()?;
        value.downcast().ok()
    }

    /// Why it failed, if it did.
    pub fn error<T>(&self, handle: &Handle<T>) -> Option<&str> {
        self.slots.get(&*handle.path)?.error.as_deref()
    }

    /// How many times data has arrived for this asset.
    pub fn version<T>(&self, handle: &Handle<T>) -> u32 {
        self.slots.get(&*handle.path).map_or(0, |slot| slot.version)
    }

    /// Whether anything has been asked for and not yet arrived or failed.
    pub fn is_loading(&self) -> bool {
        self.slots
            .values()
            .any(|slot| slot.value.is_none() && slot.error.is_none())
    }

    fn parser(&self, path: &str) -> Option<Parse> {
        self.slots.get(path).map(|slot| slot.parse)
    }

    /// Read and parse off this thread.
    #[cfg(not(target_arch = "wasm32"))]
    fn read(&self, path: &str) {
        let Some(parse) = self.parser(path) else {
            return;
        };
        let (sender, name, full) = (self.sender.clone(), path.to_owned(), self.root.join(path));
        std::thread::spawn(move || {
            let result = std::fs::read(&full)
                .map_err(|e| format!("{}: {e}", full.display()))
                .and_then(|bytes| parse(&bytes));
            let _ = sender.send((name, result));
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn deliver(&self, path: &str, bytes: Vec<u8>) {
        let Some(parse) = self.parser(path) else {
            return;
        };
        let (sender, name) = (self.sender.clone(), path.to_owned());
        std::thread::spawn(move || {
            let _ = sender.send((name, parse(&bytes)));
        });
    }

    #[cfg(all(target_arch = "wasm32", feature = "web"))]
    fn read(&self, path: &str) {
        let Some(parse) = self.parser(path) else {
            return;
        };
        let url = self.root.join(path).to_string_lossy().into_owned();
        crate::web::fetch_bytes(url, path.to_owned(), parse, self.sender.clone());
    }

    #[cfg(all(target_arch = "wasm32", not(feature = "web")))]
    fn read(&self, path: &str) {
        let error = "reading files in a browser needs the `web` feature".to_owned();
        let _ = self.sender.send((path.to_owned(), Err(error)));
    }

    #[cfg(target_arch = "wasm32")]
    fn deliver(&self, path: &str, bytes: Vec<u8>) {
        if let Some(parse) = self.parser(path) {
            let _ = self.sender.send((path.to_owned(), parse(&bytes)));
        }
    }

    /// Look for changed files. Natively only; a stat per asset, a few times a second.
    #[cfg(not(target_arch = "wasm32"))]
    fn check_for_changes(&mut self, now: f64) {
        if !self.hot_reload || now < self.next_check {
            return;
        }
        self.next_check = now + 0.25;
        let mut changed = Vec::new();
        for (path, slot) in &mut self.slots {
            let modified = std::fs::metadata(self.root.join(path))
                .and_then(|meta| meta.modified())
                .ok();
            if modified.is_some() && slot.modified.is_some() && modified != slot.modified {
                changed.push(path.clone());
            }
            if modified.is_some() {
                slot.modified = modified;
            }
        }
        for path in changed {
            self.read(&path);
        }
    }
}

/// Adds [`Assets`] and the systems that bring results in.
pub struct AssetsPlugin {
    root: PathBuf,
}

impl AssetsPlugin {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
}

impl Plugin for AssetsPlugin {
    fn build(self, app: &mut App) {
        app.resource(Assets::new(self.root))
            .system(assets_step)
            .system(System::new(scene_step).after(assets_step))
            .system(System::new(font_step).after(assets_step));
    }
}

/// Bring in whatever has arrived.
pub fn assets_step(w: &mut World) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let now = w.resource::<crate::app::Time>().real_elapsed;
        w.resource_mut::<Assets>().check_for_changes(now);
    }
    let mut events = Vec::new();
    let assets = w.resource_mut::<Assets>();
    while let Ok((path, result)) = assets.receiver.try_recv() {
        let Some(slot) = assets.slots.get_mut(&path) else {
            continue;
        };
        match result {
            Ok(value) => {
                let reloaded = slot.version > 0;
                slot.value = Some(value);
                slot.error = None;
                slot.version += 1;
                events.push(Ok(AssetLoaded { path, reloaded }));
            }
            // A bad save of a file being edited keeps the last good version.
            Err(error) => {
                eprintln!("asset {path}: {error}");
                slot.error = Some(error.clone());
                events.push(Err(AssetFailed { path, error }));
            }
        }
    }
    open_packs(w.resource_mut::<Assets>());
    for event in events {
        match event {
            Ok(event) => w.send(event),
            Err(event) => w.send(event),
        }
    }
}

/// Hand out the contents of packs that have arrived. What a pack turned out not
/// to hold goes back to being an ordinary read.
fn open_packs(assets: &mut Assets) {
    let waiting = std::mem::take(&mut assets.pending_packs);
    let mut opened = false;
    for handle in waiting {
        if let Some(pack) = assets.get(&handle) {
            assets.packed.extend(pack.0.clone());
            opened = true;
        } else if assets.error(&handle).is_some() {
            opened = true;
        } else {
            assets.pending_packs.push(handle);
        }
    }
    if !opened || !assets.pending_packs.is_empty() {
        return;
    }
    let unstarted: Vec<String> = assets
        .slots
        .iter()
        .filter(|(_, slot)| slot.value.is_none() && slot.error.is_none())
        .map(|(path, _)| path.clone())
        .collect();
    for path in unstarted {
        match assets.packed.get(&path) {
            Some(bytes) => assets.deliver(&path, bytes.clone()),
            None => assets.read(&path),
        }
    }
}

/// The font to switch the HUD to once it has loaded.
struct PendingFont(Option<Handle<FontAsset>>);

/// Draw the HUD in this font from when it has loaded. Needs [`AssetsPlugin`].
pub fn use_font(w: &mut World, path: &str) {
    let font = w.resource_mut::<Assets>().load::<FontAsset>(path);
    w.insert_resource(PendingFont(Some(font)));
}

fn font_step(w: &mut World) {
    let Some(font) = w.get_resource::<PendingFont>().and_then(|f| f.0.clone()) else {
        return;
    };
    let Some(loaded) = w.resource::<Assets>().get(&font) else {
        return;
    };
    let Some(hud) = w.get_resource_mut::<Hud>() else {
        return;
    };
    if let Ok(font) = CaptionFont::from_ttf_bytes(&loaded.0) {
        hud.0.set_font(font);
    }
    w.resource_mut::<PendingFont>().0 = None;
}

/// A progress bar in one state, moving to another once everything has loaded.
///
/// ```no_run
/// # use threers_ecs::prelude::*;
/// # use threers_ecs::assets::*;
/// #[derive(Clone, Copy, PartialEq, Eq, Hash)]
/// enum Screen { Loading, Playing }
/// impl GameState for Screen {}
///
/// let mut app = App::new();
/// app.plugin(AssetsPlugin::new("assets"))
///     .state(Screen::Loading)
///     .plugin(LoadingScreen { during: Screen::Loading, then: Screen::Playing });
/// ```
pub struct LoadingScreen<S> {
    pub during: S,
    pub then: S,
}

impl<S: GameState> Plugin for LoadingScreen<S> {
    fn build(self, app: &mut App) {
        let (during, then) = (self.during, self.then);
        app.system(
            System::new(move |w: &mut World| {
                let (done, total) = w.resource::<Assets>().progress();
                let loading = w.resource::<Assets>().is_loading();
                with_hud(w, |hud| {
                    let (width, height) = hud.0.size();
                    let (width, height) = (width as f32, height as f32);
                    let (x, y, length) = (width * 0.2, height * 0.5, width * 0.6);
                    let fraction = if total == 0 {
                        1.0
                    } else {
                        done as f32 / total as f32
                    };
                    hud.no_stroke();
                    hud.fill_alpha(Color::WHITE, 0.25);
                    hud.rect(x, y, length, 12.0);
                    hud.fill(Color::WHITE);
                    hud.rect(x, y, length * fraction, 12.0);
                });
                if !loading {
                    w.set_state(then);
                }
            })
            .after(assets_step)
            .in_state(during),
        );
    }
}

/// A glTF binary (`.glb`). Natively the first spawn takes what the background thread
/// parsed; any more parse the kept bytes, since each copy needs nodes of its own.
pub struct SceneAsset {
    bytes: Vec<u8>,
    // A parsed scene holds GPU handles that cannot leave their thread on wasm, and
    // there is no background thread there to have parsed it on anyway.
    #[cfg(not(target_arch = "wasm32"))]
    parsed: Mutex<Option<GltfScene>>,
}

impl SceneAsset {
    #[cfg(not(target_arch = "wasm32"))]
    fn scene(&self) -> Option<GltfScene> {
        let parsed = self.parsed.lock().ok().and_then(|mut parsed| parsed.take());
        parsed.or_else(|| GltfLoader::parse_glb(&self.bytes).ok())
    }

    #[cfg(target_arch = "wasm32")]
    fn scene(&self) -> Option<GltfScene> {
        GltfLoader::parse_glb(&self.bytes).ok()
    }
}

impl Asset for SceneAsset {
    fn parse(bytes: &[u8]) -> Result<Self, String> {
        let _parsed = GltfLoader::parse_glb(bytes).map_err(|e| format!("{e:?}"))?;
        Ok(Self {
            bytes: bytes.to_vec(),
            #[cfg(not(target_arch = "wasm32"))]
            parsed: Mutex::new(Some(_parsed)),
        })
    }
}

/// An entity that shows a scene file. Everything from the file hangs under it.
pub struct SceneInstance {
    pub scene: Handle<SceneAsset>,
    /// The file's animations, retargeted at the nodes of this copy.
    pub animations: Vec<AnimationClip>,
    version: u32,
    placeholder: Option<Entity>,
}

impl Component for SceneInstance {}

/// A [`SceneInstance`]'s nodes are in the world, for the first time or again.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneReady(pub Entity);

impl World {
    /// Show a `.glb` under a new entity, as soon as it has loaded. Needs
    /// [`AssetsPlugin`]. If it cannot be loaded, a bright box marks the spot
    /// and [`AssetFailed`] says why, rather than the game stopping.
    pub fn spawn_scene(&mut self, path: &str) -> Entity {
        let scene = self.resource_mut::<Assets>().load::<SceneAsset>(path);
        let anchor = self.spawn(Object3D::group());
        self.insert(
            anchor,
            SceneInstance {
                scene,
                animations: Vec::new(),
                version: 0,
                placeholder: None,
            },
        );
        anchor
    }
}

/// Put loaded scenes into the world, and replace them when their file changes.
pub fn scene_step(w: &mut World) {
    let mut work = Vec::new();
    w.each_ref::<(Entity, &SceneInstance)>(|(id, instance)| {
        work.push((
            id,
            instance.scene.clone(),
            instance.version,
            instance.placeholder,
        ));
    });
    for (id, scene, version, placeholder) in work {
        let assets = w.resource::<Assets>();
        let current = assets.version(&scene);
        if current == version {
            if current == 0 && placeholder.is_none() && assets.error(&scene).is_some() {
                show_placeholder(w, id);
            }
            continue;
        }
        let Some(asset) = assets.get(&scene) else {
            continue;
        };
        let Some(loaded) = asset.scene() else {
            continue;
        };

        // Out with the old copy, or the placeholder.
        let children = w
            .scene
            .get(id)
            .map_or_else(Vec::new, |n| n.children.clone());
        w.despawn_many(&children);

        let (animations, extras) = (loaded.animations.clone(), loaded.extras.clone());
        let (roots, map) = add_to_scene_mapped(&mut w.scene, loaded);
        for root in roots {
            w.scene.arena.add_child(id, root);
        }
        for (old, value) in &extras {
            if let Some(&new) = map.get(old) {
                apply_extras(w, new, value);
            }
        }
        let animations = animations
            .into_iter()
            .map(|clip| retarget(clip, &map))
            .collect();
        if let Some(instance) = w.get_mut::<SceneInstance>(id) {
            instance.version = current;
            instance.animations = animations;
            instance.placeholder = None;
        }
        w.send(SceneReady(id));
    }
}

fn retarget(mut clip: AnimationClip, map: &HashMap<Entity, Entity>) -> AnimationClip {
    clip.tracks = clip
        .tracks
        .into_iter()
        .filter_map(|track: KeyframeTrack| {
            let object = *map.get(&track.object)?;
            Some(KeyframeTrack { object, ..track })
        })
        .collect();
    clip
}

fn show_placeholder(w: &mut World, id: Entity) {
    let mesh = Mesh::new(
        BoxGeometry::new(1.0, 1.0, 1.0),
        Material::Basic(BasicMaterial::new(Color::from_hex(0xff00ff))),
    );
    let marker = w.spawn_under(id, Object3D::mesh(mesh));
    if let Some(instance) = w.get_mut::<SceneInstance>(id) {
        instance.placeholder = Some(marker);
    }
}

fn apply_extras(w: &mut World, id: Entity, extras: &Value) {
    let Some(object) = extras.as_object() else {
        return;
    };
    let mut names: Vec<&String> = object.keys().collect();
    names.sort();
    for name in names {
        let value = &object[name];
        if physics_extra(w, id, name, value) {
            continue;
        }
        let fields: Vec<(String, FieldValue)> = value
            .as_object()
            .map(|fields| {
                fields
                    .iter()
                    .filter_map(|(field, value)| {
                        Some((field.clone(), crate::tuning::json_to_field(value)?))
                    })
                    .collect()
            })
            .unwrap_or_default();
        if !w.insert_registered(id, name, &fields) && looks_like_component(name) {
            let node = w.node(id).map_or_else(String::new, |n| n.name.clone());
            let known = w.registered_names().join(", ");
            eprintln!(
                "{node:?} names {name}, which is not registered: call w.register::<{name}>(). \
                 Registered: {}",
                if known.is_empty() { "none" } else { &known }
            );
        }
    }
}

/// Authoring tools add properties of their own; a component's name is
/// capitalised, which theirs mostly are not.
fn looks_like_component(name: &str) -> bool {
    name.chars().next().is_some_and(char::is_uppercase)
}

/// `solid`, `sensor` and `body`: collision for level geometry, set in the
/// authoring tool. Returns whether `name` was one of them.
#[cfg(feature = "physics")]
fn physics_extra(w: &mut World, id: Entity, name: &str, value: &Value) -> bool {
    use crate::physics::{AttachBody, Collider, RigidBody, Shape};
    let wanted = !matches!(value, Value::Bool(false) | Value::Null);
    let geometry = match w.node(id).map(|node| &node.kind) {
        Some(threers::core::ObjectKind::Mesh(mesh)) => Some(mesh.geometry.clone()),
        _ => None,
    };
    let body = match name {
        "solid" if wanted => geometry
            .and_then(|g| Shape::trimesh_from_geometry(&g))
            .map(|shape| RigidBody::fixed().shape(shape)),
        "sensor" if wanted => geometry
            .and_then(|g| Shape::trimesh_from_geometry(&g))
            .map(|shape| RigidBody::fixed().collider(Collider::new(shape).sensor(true))),
        "body" => {
            let kind = value.as_str().unwrap_or("");
            let shape = geometry.and_then(|g| Shape::convex_hull_from_geometry(&g));
            match kind {
                "dynamic" => shape.map(|shape| RigidBody::dynamic().shape(shape)),
                "fixed" => shape.map(|shape| RigidBody::fixed().shape(shape)),
                _ => {
                    eprintln!("body should be \"dynamic\" or \"fixed\", not {value:?}");
                    None
                }
            }
        }
        "solid" | "sensor" => return true,
        _ => return false,
    };
    match body {
        Some(body) => {
            w.attach_body(id, body);
        }
        None => eprintln!("{name} is set on a node with no mesh to shape it from"),
    }
    true
}

#[cfg(not(feature = "physics"))]
fn physics_extra(_: &mut World, _: Entity, name: &str, _: &Value) -> bool {
    matches!(name, "solid" | "sensor" | "body")
}
