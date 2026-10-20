//! three.js r165 tone mapping, per material, evaluated analytically.
//!
//! `WebGLPrograms` defines `TONE_MAPPING` for a material drawn to the canvas
//! when `material.toneMapped` is true (the default) and the renderer's
//! `toneMapping` is not `NoToneMapping`; the material's fragment shader then
//! runs `tonemapping_fragment` — `gl_FragColor.rgb = toneMapping(rgb)` — before
//! `colorspace_fragment` encodes it. The shaders that include the chunk are
//! `meshbasic`, `meshlambert`, `meshphong`, `meshphysical` (standard and
//! physical), `meshtoon`, `meshmatcap`, `points`, `sprite`, `linedashed` and
//! `shadow`; `meshnormal`, `depth` and `distanceRGBA` do not, so those are
//! never tone mapped. The curves are transcribed from r165's
//! `tonemapping_pars_fragment` in f64, independently of the crate's own.
//!
//! Each case is a flat surface filling an orthographic view with no lights,
//! so its outgoing light is a known colour: a basic / sprite colour, the
//! emissive of a black lit material, half a matcap colour (no matcap texture:
//! `mix(0.2, 0.8, uv.y)` at the centre, where `uv.y` is ½). Shared by
//! `tests/tone_mapping.rs` (wgpu) and `tests/backend_parity.rs` (Metal)
//! through `#[path]`.
#![allow(dead_code)]

use threers::cameras::{Camera, OrthographicCamera};
use threers::core::{Mesh, Object3D, Sprite};
use threers::geometries::PlaneGeometry;
use threers::materials::{
    BasicMaterial, LambertMaterial, Material, MatcapMaterial, NormalMaterial, PhongMaterial,
    PhysicalMaterial, SpriteMaterial, StandardMaterial, ToonMaterial,
};
use threers::math::{Color, Vector3};
use threers::scene::Scene;
use threers::ToneMapping;

pub const N: u32 = 32;

type V3 = [f64; 3];

fn saturate(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}

/// GLSL `mat3 * v` for a matrix given as its three columns.
fn mat3(cols: [V3; 3], v: V3) -> V3 {
    let mut o = [0.0; 3];
    for (k, col) in cols.iter().enumerate() {
        for r in 0..3 {
            o[r] += col[r] * v[k];
        }
    }
    o
}

/// three.js `toneMapping(color)` for `mode` at `exposure`.
pub fn three_tone_map(mode: ToneMapping, exposure: f64, c: V3) -> V3 {
    match mode {
        ToneMapping::None => c,
        ToneMapping::Linear => c.map(|x| saturate(exposure * x)),
        ToneMapping::AcesFilmic => {
            let input = [
                [0.59719, 0.07600, 0.02840],
                [0.35458, 0.90834, 0.13383],
                [0.04823, 0.01566, 0.83777],
            ];
            let output = [
                [1.60475, -0.10208, -0.00327],
                [-0.53108, 1.10813, -0.07276],
                [-0.07367, -0.00605, 1.07602],
            ];
            let v = mat3(input, c.map(|x| x * exposure / 0.6));
            let fit = v.map(|v| {
                (v * (v + 0.0245786) - 0.000090537) / (v * (0.983729 * v + 0.4329510) + 0.238081)
            });
            mat3(output, fit).map(saturate)
        }
        ToneMapping::AgX => {
            let to2020 = [[0.6274, 0.0691, 0.0164], [0.3293, 0.9195, 0.0880], [0.0433, 0.0113, 0.8956]];
            let to709 = [[1.6605, -0.1246, -0.0182], [-0.5876, 1.1329, -0.1006], [-0.0728, -0.0083, 1.1187]];
            let inset = [
                [0.856627153315983, 0.137318972929847, 0.11189821299995],
                [0.0951212405381588, 0.761241990602591, 0.0767994186031903],
                [0.0482516061458583, 0.101439036467562, 0.811302368396859],
            ];
            let outset = [
                [1.1271005818144368, -0.1413297634984383, -0.14132976349843826],
                [-0.11060664309660323, 1.157823702216272, -0.11060664309660294],
                [-0.016493938717834573, -0.016493938717834257, 1.2519364065950405],
            ];
            let (min_ev, max_ev) = (-12.47393, 4.026069);
            let v = mat3(inset, mat3(to2020, c.map(|x| x * exposure)));
            let v = v.map(|x| {
                let x = saturate((x.max(1e-10).log2() - min_ev) / (max_ev - min_ev));
                let (x2, x4) = (x * x, x * x * x * x);
                15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2
                    + 0.1191 * x
                    - 0.00232
            });
            let v = mat3(outset, v).map(|x| x.max(0.0).powf(2.2));
            mat3(to709, v).map(saturate)
        }
        ToneMapping::Neutral => {
            let start = 0.8 - 0.04;
            let desat = 0.15;
            let mut c = c.map(|x| x * exposure);
            let x = c[0].min(c[1]).min(c[2]);
            let offset = if x < 0.08 { x - 6.25 * x * x } else { 0.04 };
            c = c.map(|v| v - offset);
            let peak = c[0].max(c[1]).max(c[2]);
            if peak < start {
                return c;
            }
            let d = 1.0 - start;
            let new_peak = 1.0 - d * d / (peak + d - start);
            c = c.map(|v| v * new_peak / peak);
            let g = 1.0 - 1.0 / (desat * (peak - new_peak) + 1.0);
            c.map(|v| v + (new_peak - v) * g)
        }
    }
}

/// `colorspace_fragment` into the 8-bit canvas.
pub fn encode(c: V3) -> [u8; 3] {
    c.map(|x| {
        let x = saturate(x);
        let s = if x <= 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
        (s * 255.0).round() as u8
    })
}

pub fn camera() -> OrthographicCamera {
    let mut cam = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    cam.position = Vector3::new(0.0, 0.0, 2.0);
    cam.target = Vector3::ZERO;
    cam
}

fn col(c: V3) -> Color {
    Color::new(c[0] as f32, c[1] as f32, c[2] as f32)
}

/// What a case draws: its name, the material, and the outgoing light three.js
/// computes for it before tone mapping (`None`: a data material, written as
/// the given bytes whatever the tone mapping).
pub enum Out {
    Light(V3),
    Data([u8; 3]),
}

/// Every material kind, with `tone_mapped` set as asked, showing `c`.
pub fn cases(c: V3, tone_mapped: bool) -> Vec<(&'static str, Material, Out)> {
    let black = Color::new(0.0, 0.0, 0.0);
    let mut out: Vec<(&'static str, Material, Out)> = Vec::new();
    let mut basic = BasicMaterial::new(col(c));
    basic.tone_mapped = tone_mapped;
    out.push(("basic", basic.into(), Out::Light(c)));
    let mut lambert = LambertMaterial::new(black);
    lambert.emissive = col(c);
    lambert.tone_mapped = tone_mapped;
    out.push(("lambert", lambert.into(), Out::Light(c)));
    let mut phong = PhongMaterial::new(black);
    phong.emissive = col(c);
    phong.specular = black;
    phong.tone_mapped = tone_mapped;
    out.push(("phong", phong.into(), Out::Light(c)));
    let mut toon = ToonMaterial::new(black);
    toon.emissive = col(c);
    toon.tone_mapped = tone_mapped;
    out.push(("toon", toon.into(), Out::Light(c)));
    let mut standard = StandardMaterial::new(black);
    standard.emissive = col(c);
    standard.metalness = 0.0;
    standard.tone_mapped = tone_mapped;
    out.push(("standard", standard.into(), Out::Light(c)));
    let mut physical = PhysicalMaterial::new(black);
    physical.emissive = col(c);
    physical.metalness = 0.0;
    physical.tone_mapped = tone_mapped;
    out.push(("physical", physical.into(), Out::Light(c)));
    let matcap = MatcapMaterial {
        color: col(c),
        tone_mapped,
        ..Default::default()
    };
    out.push(("matcap", matcap.into(), Out::Light(c.map(|x| x * 0.5))));
    let mut sprite = SpriteMaterial::new(col(c));
    sprite.tone_mapped = tone_mapped;
    out.push(("sprite", Material::Sprite(sprite), Out::Light(c)));
    // The Reflector material (no field for the flag: three.js's Reflector
    // shader tone-maps): without a reflection bound it shows its colour.
    if tone_mapped {
        let mirror = threers::materials::MirrorMaterial {
            color: col(c),
            ..Default::default()
        };
        out.push(("mirror", Material::Mirror(mirror), Out::Light(c)));
    }
    // `packNormalToRGB` of the view-space normal (0, 0, 1), written raw.
    out.push(("normal", NormalMaterial::new().into(), Out::Data([128, 128, 255])));
    out
}

/// A plane (a sprite, for the sprite material) filling the view.
pub fn scene(material: Material) -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    if let Material::Sprite(_) = material {
        let mut o = Object3D::sprite(Sprite::new(material));
        o.scale = Vector3::new(3.0, 3.0, 1.0);
        scene.add(o);
    } else {
        scene.add(Object3D::mesh(Mesh::new(PlaneGeometry::new(3.0, 3.0), material)));
    }
    scene
}

pub const MODES: [(ToneMapping, f32); 6] = [
    (ToneMapping::None, 1.0),
    (ToneMapping::Linear, 1.5),
    (ToneMapping::AcesFilmic, 1.0),
    (ToneMapping::AcesFilmic, 0.7),
    (ToneMapping::Neutral, 1.2),
    (ToneMapping::AgX, 1.0),
];

pub const COLOURS: [V3; 2] = [[0.9, 0.35, 0.05], [2.5, 0.8, 0.1]];

/// Every material, colour, mode and `tone_mapped` setting through `render`
/// after `set_mode`; the centre pixel within two levels of three.js's.
pub fn check(
    backend: &str,
    set_mode: &mut dyn FnMut(ToneMapping, f32),
    render: &mut dyn FnMut(&mut Scene, &dyn Camera) -> Vec<u8>,
) -> Vec<String> {
    let cam = camera();
    let mut failures = Vec::new();
    for (mode, exposure) in MODES {
        set_mode(mode, exposure);
        for c in COLOURS {
            for tone_mapped in [true, false] {
                for (name, material, out) in cases(c, tone_mapped) {
                    let rgba = render(&mut scene(material), &cam);
                    let i = ((N / 2 * N + N / 2) * 4) as usize;
                    let got = [rgba[i], rgba[i + 1], rgba[i + 2]];
                    let want = match out {
                        Out::Light(l) if tone_mapped => {
                            encode(three_tone_map(mode, exposure as f64, l))
                        }
                        Out::Light(l) => encode(l),
                        Out::Data(d) => d,
                    };
                    let err = (0..3).map(|k| (got[k] as i32 - want[k] as i32).abs()).max().unwrap();
                    if err > 2 {
                        failures.push(format!(
                            "{backend} {mode:?} x{exposure} {name} (toneMapped {tone_mapped}) \
                             on {c:?}: got {got:?}, three.js {want:?}"
                        ));
                    }
                }
            }
        }
    }
    failures
}
