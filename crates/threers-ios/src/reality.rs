//! What RealityKit needs of a scene beyond its meshes, read out for the
//! shell ([`volume`](crate::volume)): the app's lights, an environment to
//! light by, a material's maps, and keys that move when a pose does.
//!
//! RealityKit takes lights in photometric units — a directional light in
//! lux, a point or spot light in lumens — and exposes them as a camera
//! would. threers draws three.js's units with no exposure: a directional
//! light of intensity π lights a white surface facing it white. RealityKit's
//! own default directional light, 2145.7 lux, is that brightness on its
//! default exposure, so a three.js intensity is taken as 683 times its lux
//! (the luminous efficacy three.js's units are defined with), a point or
//! spot light's candela as 4π times that in lumens. The two renderers tone
//! map differently, so the match is close, not exact.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use threers::core::ObjectKind;
use threers::lights::Light;
use threers::materials::Material;
use threers::math::{Color, Matrix4, Vector3};
use threers::scene::Scene;
use threers::textures::{CubeTexture, Texture, TextureFormat, TextureWrap};

/// How RealityKit lights an app's scene ([`App::reality_lighting`](crate::App::reality_lighting)).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RealityLighting {
    /// The room's light where there is a room (a volume, AR over the
    /// camera); the app's own lights and environment where there is none
    /// (RealityKit on an iPhone's or a Mac's screen).
    #[default]
    Auto = 0,
    /// RealityKit's alone: the room's on visionOS and in AR, its default
    /// light elsewhere. The app's lights are left out.
    Room = 1,
    /// The app's alone: its lights, and its environment (or its ambient and
    /// hemisphere lights) as the image-based light in place of the room's.
    Scene = 2,
    /// The room's light, the app's lights added to it.
    Both = 3,
}

/// What draws an app's scene in a visionOS volume ([`App::volume_drawing`](crate::App::volume_drawing)).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VolumeDrawing {
    /// threers' own renderer, into a picture on a quad that turns to face
    /// you: every material and effect as the window draws it, but flat —
    /// both eyes see one picture.
    #[default]
    Picture = 0,
    /// RealityKit, entity by entity: in depth for each eye, lit by the room
    /// (or the app's lights), the materials it can draw (a shader material
    /// on a flat mesh baked to a texture; on any other, left out).
    Entities = 1,
}

/// One light, as the shell reads it: 16 floats.
///
/// `[0]` the kind (1 directional, 2 point, 3 spot), `[1..4]` the colour
/// (linear), `[4]` the intensity in RealityKit's units (lux for a
/// directional light, lumens otherwise; the scene's scale is the shell's),
/// `[5..8]` where it is and `[8..11]` the way it shines (the scene's
/// frame, a unit vector), `[11]` its reach (0: no limit), `[12]` its decay
/// exponent, `[13]` a spot's outer and `[14]` inner half-angle (degrees),
/// `[15]` 1 when it casts shadows.
pub const LIGHT_FLOATS: usize = 16;

/// three.js's lux to RealityKit's: see the module's notes.
const LUX: f32 = 683.0;

/// The way `local` points under `m`, normalised (zero stays zero).
fn direction(m: &Matrix4, local: Vector3) -> Vector3 {
    let e = &m.elements;
    let v = Vector3::new(
        e[0] * local.x + e[4] * local.y + e[8] * local.z,
        e[1] * local.x + e[5] * local.y + e[9] * local.z,
        e[2] * local.x + e[6] * local.y + e[10] * local.z,
    );
    let l = v.length();
    if l > 0.0 { v * (1.0 / l) } else { v }
}

/// The scene's directional, point and spot lights ([`LIGHT_FLOATS`] each),
/// those `visible` says are shown.
pub(crate) fn lights(scene: &Scene, visible: impl Fn(threers::core::ObjectId) -> bool) -> Vec<f32> {
    let mut out = vec![];
    for (id, o) in scene.arena.nodes.iter() {
        let ObjectKind::Light(light) = &o.kind else { continue };
        if !visible(id) {
            continue;
        }
        let e = &o.matrix_world.elements;
        let at = Vector3::new(e[12], e[13], e[14]);
        let mut f = [0.0f32; LIGHT_FLOATS];
        let (kind, color, intensity, toward, reach, decay, cone, shadow) = match light {
            Light::Directional(d) => (1.0, d.color, d.intensity * LUX, d.direction, 0.0, 0.0, None, d.cast_shadow),
            Light::Point(p) => (
                2.0,
                p.color,
                p.intensity * LUX * 4.0 * std::f32::consts::PI,
                Vector3::new(0.0, -1.0, 0.0),
                p.distance,
                p.decay,
                None,
                p.cast_shadow,
            ),
            Light::Spot(s) => (
                3.0,
                s.color,
                s.intensity * LUX * 4.0 * std::f32::consts::PI,
                s.direction,
                s.distance,
                s.decay,
                Some((s.angle, s.penumbra)),
                s.cast_shadow,
            ),
            // ambient and hemisphere light the environment ([`equirect`]);
            // a rectangle has no RealityKit kind
            _ => continue,
        };
        let way = direction(&o.matrix_world, toward);
        f[0] = kind;
        f[1..4].copy_from_slice(&[color.r, color.g, color.b]);
        f[4] = intensity.max(0.0);
        f[5..8].copy_from_slice(&[at.x, at.y, at.z]);
        f[8..11].copy_from_slice(&[way.x, way.y, way.z]);
        f[11] = reach.max(0.0);
        f[12] = decay;
        if let Some((angle, penumbra)) = cone {
            let outer = angle.to_degrees().clamp(0.0, 90.0);
            f[13] = outer;
            f[14] = outer * (1.0 - penumbra.clamp(0.0, 1.0));
        }
        f[15] = shadow as u32 as f32;
        out.extend_from_slice(&f);
    }
    out
}

/// What lights the scene indirectly: its environment map, else its ambient
/// and hemisphere lights. Changes when any of them does.
pub(crate) fn environment_key(scene: &Scene) -> u64 {
    let mut h = DefaultHasher::new();
    if let Some(env) = &scene.environment {
        (Arc::as_ptr(env) as usize).hash(&mut h);
        (Arc::as_ptr(&env.faces[0]) as usize).hash(&mut h);
        scene.environment_intensity.to_bits().hash(&mut h);
        let r = scene.environment_rotation;
        [r.x, r.y, r.z].map(f32::to_bits).hash(&mut h);
    }
    for (_, o) in scene.arena.nodes.iter() {
        let ObjectKind::Light(l) = &o.kind else { continue };
        let colors: [Color; 2] = match l {
            Light::Ambient(a) => [a.color, a.color],
            Light::Hemisphere(s) => [s.sky_color, s.ground_color],
            _ => continue,
        };
        o.visible.hash(&mut h);
        l.intensity().to_bits().hash(&mut h);
        for c in colors {
            [c.r, c.g, c.b].map(f32::to_bits).hash(&mut h);
        }
    }
    h.finish()
}

/// Whether the scene has anything [`equirect`] would light it with.
pub(crate) fn has_environment(scene: &Scene) -> bool {
    scene.environment.is_some()
        || scene
            .arena
            .nodes
            .iter()
            .any(|(_, o)| o.visible && matches!(&o.kind, ObjectKind::Light(Light::Ambient(_) | Light::Hemisphere(_))))
}

/// The light round the scene as an equirectangular image, `width` ×
/// `width / 2`, linear RGBA floats, rows from the top (the sky): its
/// environment map where it has one, plus its ambient light (even all
/// round) and its hemisphere lights (the sky's colour above, the ground's
/// below). Each as the radiance that lights a white surface as three.js's
/// light does: an ambient light of intensity `k` is `k / π` in every
/// direction. Black without any of them.
pub(crate) fn equirect(scene: &Scene, width: u32) -> Vec<f32> {
    let (w, h) = (width.max(4) as usize, (width.max(4) / 2) as usize);
    let mut ambient = [0.0f32; 3];
    let mut hemispheres: Vec<([f32; 3], [f32; 3])> = vec![];
    for (_, o) in scene.arena.nodes.iter() {
        if !o.visible {
            continue;
        }
        let ObjectKind::Light(l) = &o.kind else { continue };
        let k = l.intensity() / std::f32::consts::PI;
        match l {
            Light::Ambient(a) => {
                for (s, c) in ambient.iter_mut().zip([a.color.r, a.color.g, a.color.b]) {
                    *s += c * k;
                }
            }
            Light::Hemisphere(s) => hemispheres.push((
                [s.sky_color.r * k, s.sky_color.g * k, s.sky_color.b * k],
                [s.ground_color.r * k, s.ground_color.g * k, s.ground_color.b * k],
            )),
            _ => {}
        }
    }
    let cube = scene.environment.as_ref().and_then(|c| CubeFaces::new(c));
    // three.js turns the lookup by environmentRotation
    let turn = {
        let r = scene.environment_rotation;
        threers::math::Quaternion::from_euler(threers::math::Euler::new(r.x, r.y, r.z))
    };
    let k = scene.environment_intensity;
    let mut out = vec![0.0f32; w * h * 4];
    for y in 0..h {
        let lat = std::f32::consts::FRAC_PI_2 - (y as f32 + 0.5) / h as f32 * std::f32::consts::PI;
        for x in 0..w {
            // longitude 0 (the middle column) looks along -z
            let lon = (x as f32 + 0.5) / w as f32 * std::f32::consts::TAU - std::f32::consts::PI;
            let d = Vector3::new(lat.cos() * lon.sin(), lat.sin(), -lat.cos() * lon.cos());
            let mut c = ambient;
            let t = d.y * 0.5 + 0.5;
            for (sky, ground) in &hemispheres {
                for i in 0..3 {
                    c[i] += ground[i] + (sky[i] - ground[i]) * t;
                }
            }
            if let Some(cube) = &cube {
                let s = cube.sample(d.apply_quaternion(turn));
                for i in 0..3 {
                    c[i] += s[i] * k;
                }
            }
            let o = (y * w + x) * 4;
            out[o..o + 4].copy_from_slice(&[c[0], c[1], c[2], 1.0]);
        }
    }
    out
}

/// A cube map's faces as linear RGB, for [`equirect`].
struct CubeFaces {
    size: usize,
    faces: Vec<Vec<f32>>,
}

impl CubeFaces {
    fn new(cube: &CubeTexture) -> Option<Self> {
        let size = cube.size as usize;
        if size == 0 {
            return None;
        }
        let faces: Option<Vec<Vec<f32>>> = match &cube.faces_f32 {
            Some(f) => Some(f.iter().map(|v| v.to_vec()).collect()),
            None => cube
                .faces
                .iter()
                .map(|bytes| face_texels(cube.format, bytes, size))
                .collect(),
        };
        let faces = faces?;
        faces.iter().all(|f| f.len() >= size * size * 4).then_some(CubeFaces { size, faces })
    }

    /// Along `d` (the world's frame), as three.js samples a cube map: its
    /// x mirrored (`flipEnvMap`), then the faces as WebGL lays them out.
    fn sample(&self, d: Vector3) -> [f32; 3] {
        let (x, y, z) = (-d.x, d.y, d.z);
        let (ax, ay, az) = (x.abs(), y.abs(), z.abs());
        let (face, sc, tc, ma) = if ax >= ay && ax >= az {
            if x > 0.0 { (0, -z, -y, ax) } else { (1, z, -y, ax) }
        } else if ay >= az {
            if y > 0.0 { (2, x, z, ay) } else { (3, x, -z, ay) }
        } else if z > 0.0 {
            (4, x, -y, az)
        } else {
            (5, -x, -y, az)
        };
        if ma <= 0.0 {
            return [0.0; 3];
        }
        let n = self.size;
        let texel = |t: f32| (((t / ma + 1.0) * 0.5 * n as f32) as usize).min(n - 1);
        let (u, v) = (texel(sc), texel(tc));
        let p = &self.faces[face][(v * n + u) * 4..];
        [p[0], p[1], p[2]]
    }
}

/// A face's texels as linear RGBA floats.
fn face_texels(format: TextureFormat, bytes: &[u8], size: usize) -> Option<Vec<f32>> {
    let count = size * size;
    match format {
        TextureFormat::Rgba8UnormSrgb => Some(
            bytes
                .get(..count * 4)?
                .as_chunks::<4>().0.iter()
                .flat_map(|p| [srgb_to_linear(p[0]), srgb_to_linear(p[1]), srgb_to_linear(p[2]), p[3] as f32 / 255.0])
                .collect(),
        ),
        TextureFormat::Rgba8Unorm => Some(bytes.get(..count * 4)?.iter().map(|&b| b as f32 / 255.0).collect()),
        f if f.is_float() && f.channels() == 4 => f.texels_f32(bytes.get(..f.data_len(size as u32, size as u32))?),
        _ => None,
    }
}

fn srgb_to_linear(b: u8) -> f32 {
    let c = b as f32 / 255.0;
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

/// A material's maps, by slot: the bit the shell reads for it is `1 << slot`.
///
/// 0 colour, 1 normal, 2 roughness, 3 metalness, 4 emissive, 5 ambient
/// occlusion, 6 alpha.
pub const MAP_SLOTS: usize = 7;

/// The bit [`maps`] sets for a normal map (slot 1).
pub(crate) const NORMAL_MAP: u32 = 1 << 1;

/// The maps of a lit material, by slot ([`MAP_SLOTS`]).
fn slots(material: &Material) -> [Option<&Arc<Texture>>; MAP_SLOTS] {
    match material {
        Material::Standard(s) => [
            s.map.as_ref(),
            s.normal_map.as_ref(),
            s.roughness_map.as_ref(),
            s.metalness_map.as_ref(),
            s.emissive_map.as_ref(),
            s.ao_map.as_ref(),
            s.alpha_map.as_ref(),
        ],
        Material::Physical(p) => [
            p.map.as_ref(),
            p.normal_map.as_ref(),
            p.roughness_map.as_ref(),
            p.metalness_map.as_ref(),
            p.emissive_map.as_ref(),
            p.ao_map.as_ref(),
            p.alpha_map.as_ref(),
        ],
        _ => [None; MAP_SLOTS],
    }
}

/// Which maps `material` has that the shell can show: a bit per slot.
pub(crate) fn maps(material: &Material) -> u32 {
    slots(material)
        .iter()
        .enumerate()
        .filter(|(_, t)| t.is_some_and(|t| !t.is_handle() && t.width > 0 && t.height > 0))
        .fold(0, |m, (i, _)| m | 1 << i)
}

/// The maps' identities, for the material's key: a map whose pixels are
/// replaced keeps its texture's id but moves its sequence.
pub(crate) fn maps_key(material: &Material) -> u64 {
    let mut h = DefaultHasher::new();
    for t in slots(material) {
        t.map(|t| (t.id, t.upload_seq, Arc::as_ptr(&t.data) as usize)).hash(&mut h);
    }
    h.finish()
}

/// The map that sets the material's texture coordinates: its colour map's,
/// else the first it has (three.js gives each its own; RealityKit one).
pub(crate) fn uv_map(material: &Material) -> Option<&Arc<Texture>> {
    slots(material).into_iter().flatten().next()
}

/// Whether any of its maps repeats rather than stretching its edge.
pub(crate) fn maps_repeat(material: &Material) -> bool {
    slots(material)
        .into_iter()
        .flatten()
        .any(|t| t.wrap_s != TextureWrap::ClampToEdge || t.wrap_t != TextureWrap::ClampToEdge)
}

/// Map `slot` of `material` as the shell's material takes it: RGBA8, rows
/// as [`realitykit_rows`] lays them; roughness from the green channel,
/// metalness from the blue, occlusion from the red and alpha from the
/// green, each as grey (three.js packs them so; RealityKit reads one).
pub(crate) fn map_pixels(material: &Material, slot: usize) -> Option<(Vec<u8>, u32, u32)> {
    let t = (*slots(material).get(slot)?)?;
    let mut rgba = realitykit_rows(t)?;
    let channel = match slot {
        2 | 6 => Some(1),
        3 => Some(2),
        5 => Some(0),
        _ => None,
    };
    if let Some(c) = channel {
        for p in rgba.as_chunks_mut::<4>().0 {
            *p = [p[c], p[c], p[c], 255];
        }
    }
    Some((rgba, t.width, t.height))
}

/// A texture's pixels as the shell's meshes sample them: RGBA8, the rows
/// that `v = 0` reads first (the shell hands RealityKit `1 - v`).
pub(crate) fn realitykit_rows(t: &Texture) -> Option<Vec<u8>> {
    let top_first = t.to_rgba8()?;
    let row = t.width as usize * 4;
    Some(top_first.chunks_exact(row).rev().flatten().copied().collect())
}

/// A key that moves when a skinned mesh's pose does (its bones' places, its
/// morph weights): the shell then rewrites its vertices in place.
pub(crate) fn pose_key(palette: Option<&[Matrix4]>, influences: &[f32]) -> u64 {
    let mut h = DefaultHasher::new();
    for m in palette.unwrap_or_default() {
        m.elements.map(f32::to_bits).hash(&mut h);
    }
    influences.iter().map(|f| f.to_bits()).for_each(|b| b.hash(&mut h));
    h.finish()
}

/// A key for an instanced mesh's transforms.
pub(crate) fn instances_key(transforms: &[Matrix4], version: Option<u64>) -> u64 {
    let mut h = DefaultHasher::new();
    transforms.len().hash(&mut h);
    match version {
        Some(v) => v.hash(&mut h),
        None => transforms.iter().for_each(|m| m.elements.map(f32::to_bits).hash(&mut h)),
    }
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use threers::lights::{AmbientLight, DirectionalLight, HemisphereLight, SpotLight};

    #[test]
    fn a_directional_light_shines_its_way_in_lux() {
        let mut scene = Scene::new();
        let mut d = DirectionalLight::new(Color::new(1.0, 0.5, 0.25), std::f32::consts::PI);
        d.direction = Vector3::new(0.0, 0.0, -1.0);
        d.cast_shadow = true;
        let id = scene.add_light(d);
        // turned a quarter about y: -z becomes -x
        scene.get_mut(id).unwrap().set_rotation(threers::math::Euler::new(0.0, std::f32::consts::FRAC_PI_2, 0.0));
        scene.get_mut(id).unwrap().position = Vector3::new(1.0, 2.0, 3.0);
        scene.update_world();
        let f = lights(&scene, |_| true);
        assert_eq!(f.len(), LIGHT_FLOATS);
        assert_eq!(f[0], 1.0);
        assert_eq!(&f[1..4], &[1.0, 0.5, 0.25]);
        assert!((f[4] - 2145.7).abs() < 1.0, "π in three.js is RealityKit's default {}", f[4]);
        assert_eq!(&f[5..8], &[1.0, 2.0, 3.0]);
        assert!((f[8] + 1.0).abs() < 1e-5 && f[9].abs() < 1e-5 && f[10].abs() < 1e-5, "{:?}", &f[8..11]);
        assert_eq!(f[15], 1.0);
    }

    #[test]
    fn a_spot_lights_cone_is_in_half_angles() {
        let mut scene = Scene::new();
        let mut s = SpotLight::new(Color::new(1.0, 1.0, 1.0), 1.0);
        s.angle = std::f32::consts::FRAC_PI_6;
        s.penumbra = 0.5;
        scene.add_light(s);
        scene.add_light(AmbientLight::new(Color::new(1.0, 1.0, 1.0), 1.0));
        scene.update_world();
        let f = lights(&scene, |_| true);
        assert_eq!(f.len(), LIGHT_FLOATS, "the ambient light is the environment's");
        assert_eq!(f[0], 3.0);
        assert!((f[13] - 30.0).abs() < 1e-3 && (f[14] - 15.0).abs() < 1e-3);
        assert!((f[4] - 683.0 * 4.0 * std::f32::consts::PI).abs() < 1.0);
    }

    #[test]
    fn hemisphere_and_ambient_make_a_sky_over_a_ground() {
        let mut scene = Scene::new();
        assert!(!has_environment(&scene));
        let empty = environment_key(&scene);
        let k = std::f32::consts::PI;
        scene.add_light(HemisphereLight::new(Color::new(0.0, 0.0, 1.0), Color::new(1.0, 0.0, 0.0), k));
        scene.add_light(AmbientLight::new(Color::new(0.0, 1.0, 0.0), k));
        assert!(has_environment(&scene));
        assert_ne!(environment_key(&scene), empty);
        let img = equirect(&scene, 16);
        assert_eq!(img.len(), 16 * 8 * 4);
        let top = &img[..4];
        let bottom = &img[(7 * 16) * 4..(7 * 16) * 4 + 4];
        // top row: nearly all sky, plus the ambient everywhere
        assert!(top[2] > 0.9 && top[0] < 0.1 && (top[1] - 1.0).abs() < 1e-5, "{top:?}");
        assert!(bottom[0] > 0.9 && bottom[2] < 0.1 && (bottom[1] - 1.0).abs() < 1e-5, "{bottom:?}");
    }

    #[test]
    fn a_cube_environment_is_seen_where_it_faces() {
        // +y face white, the rest black: the top of the image lit, the bottom dark
        let size = 4u32;
        let face = |v: u8| [v, v, v, 255].repeat((size * size) as usize);
        let cube = CubeTexture::new(size, TextureFormat::Rgba8Unorm, [face(0), face(0), face(255), face(0), face(0), face(0)]);
        let mut scene = Scene::new();
        scene.environment = Some(Arc::new(cube));
        assert!(has_environment(&scene));
        let img = equirect(&scene, 32);
        assert_eq!(img[0], 1.0);
        assert_eq!(img[(15 * 32) * 4], 0.0);
    }

    #[test]
    fn packed_maps_come_out_as_grey() {
        let mut m = threers::materials::StandardMaterial::default();
        // 1×2: rows top-first are red then blue, as to_rgba8 gives them
        let mut t = Texture::new(1, 2, TextureFormat::Rgba8Unorm, vec![10, 20, 30, 255, 40, 50, 60, 255]);
        t.flip_y = true;
        m.roughness_map = Some(Arc::new(t.clone()));
        m.map = Some(Arc::new(t));
        let m = Material::Standard(m);
        assert_eq!(maps(&m), 0b101);
        // the green channel, the bottom row first
        let (px, w, h) = map_pixels(&m, 2).unwrap();
        assert_eq!((w, h), (1, 2));
        assert_eq!(px, vec![50, 50, 50, 255, 20, 20, 20, 255]);
        let (px, _, _) = map_pixels(&m, 0).unwrap();
        assert_eq!(px, vec![40, 50, 60, 255, 10, 20, 30, 255]);
        assert!(map_pixels(&m, 1).is_none());
    }
}
