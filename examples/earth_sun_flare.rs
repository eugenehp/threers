//! Earth lit by the sun, with a lens flare — a full PBR scene from procedural
//! high-resolution textures, no asset downloads.
//!
//! ```text
//! cargo run --release --example earth_sun_flare
//! cargo run --release --example earth_sun_flare -- --texture 8192     # 8K maps
//! cargo run --release --example earth_sun_flare -- --frames 120 --video out/earth.mp4
//! ```
//!
//! | Flag | Effect |
//! |------|--------|
//! | `--texture N` | equirectangular map width (default 4096; height is N/2) |
//! | `--size WxH` | output resolution (default 1600x900) |
//! | `--frames N` | render an orbit of N frames instead of one still |
//! | `--video PATH` | encode the orbit (needs `--features video`) |
//! | `--no-flare` | skip the lens flare |
//! | `--no-clouds` | skip the cloud shell |
//! | `--dump-maps` | also write the generated maps to `out/earth_map_*.png` |
//!
//! Everything is generated here: five equirectangular maps (albedo, normal,
//! roughness, night lights, clouds) built from one shared elevation field, so
//! the coastlines in the colour map, the relief in the normal map, the shine on
//! the oceans, and the city lights all line up. At 4096×2048 that is 8.4 M
//! texels per map; generation is threaded across cores.
//!
//! The flare is composited in screen space after the render, which is where a
//! lens artifact belongs: it is a property of the camera, not the scene. It
//! also lets it be occluded properly — the flare dies when Earth passes in
//! front of the sun.

use std::f32::consts::{PI, TAU};

use threers::{
    encode_png, Color, HeadlessRenderer, Material, Mesh, Object3D, PerspectiveCamera, Scene,
    SphereGeometry, StandardMaterial, Texture, TextureFormat, TextureWrap, Vector3,
};

// ---------------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------------

fn arg(flag: &str) -> Option<String> {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == flag {
            return args.next();
        }
    }
    None
}
fn flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

// ---------------------------------------------------------------------------
// Noise — sampled in 3D *on the sphere*, not in UV space
// ---------------------------------------------------------------------------
//
// An equirectangular map is a bad place to make noise: longitude compresses to
// nothing at the poles, so 2D noise smears into vertical streaks there and has
// to be faked back into shape, and it does not meet itself at the date line
// without special handling. Sampling a 3D field at the actual surface point
// removes both problems at once — the sphere has no seam and no poles.

/// Uniform hash of a lattice cell, in `0..=1`.
///
/// Mixed in `u32`, deliberately: on a signed integer `>>` sign-extends, so
/// `h ^ (h >> 16)` always clears the top bit and the result never exceeds 0.5.
/// A half-range hash here puts the whole planet below sea level.
fn hash3(x: i32, y: i32, z: i32, seed: i32) -> f32 {
    let mut h = (x as u32).wrapping_mul(374_761_393)
        ^ (y as u32).wrapping_mul(668_265_263)
        ^ (z as u32).wrapping_mul(2_246_822_519)
        ^ (seed as u32).wrapping_mul(1_274_126_177);
    h ^= h >> 13;
    h = h.wrapping_mul(1_274_126_177);
    h ^= h >> 16;
    h as f32 / u32::MAX as f32
}

fn smooth(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Trilinearly interpolated value noise at a point in space.
fn value_noise(p: [f32; 3], seed: i32) -> f32 {
    let i = [p[0].floor(), p[1].floor(), p[2].floor()];
    let f = [
        smooth(p[0] - i[0]),
        smooth(p[1] - i[1]),
        smooth(p[2] - i[2]),
    ];
    let (xi, yi, zi) = (i[0] as i32, i[1] as i32, i[2] as i32);
    let corner = |dx, dy, dz| hash3(xi + dx, yi + dy, zi + dz, seed);
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;

    let x00 = lerp(corner(0, 0, 0), corner(1, 0, 0), f[0]);
    let x10 = lerp(corner(0, 1, 0), corner(1, 1, 0), f[0]);
    let x01 = lerp(corner(0, 0, 1), corner(1, 0, 1), f[0]);
    let x11 = lerp(corner(0, 1, 1), corner(1, 1, 1), f[0]);
    lerp(lerp(x00, x10, f[1]), lerp(x01, x11, f[1]), f[2])
}

fn scaled(p: [f32; 3], k: f32) -> [f32; 3] {
    [p[0] * k, p[1] * k, p[2] * k]
}

/// Fractal sum of `octaves` of value noise, in `0..=1`.
fn fbm(p: [f32; 3], frequency: f32, octaves: u32, seed: i32) -> f32 {
    let (mut sum, mut amp, mut norm, mut freq) = (0.0, 0.5, 0.0, frequency);
    for o in 0..octaves {
        sum += value_noise(scaled(p, freq), seed + o as i32 * 977) * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum / norm.max(1e-6)
}

/// Ridged noise — creases rather than blobs, which is what reads as mountains.
fn ridged(p: [f32; 3], frequency: f32, octaves: u32, seed: i32) -> f32 {
    let (mut sum, mut amp, mut norm, mut freq) = (0.0, 0.5, 0.0, frequency);
    for o in 0..octaves {
        let n = value_noise(scaled(p, freq), seed + o as i32 * 313) * 2.0 - 1.0;
        sum += (1.0 - n.abs()) * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum / norm.max(1e-6)
}

/// The surface point for an equirectangular texel.
fn direction(u: f32, v: f32) -> [f32; 3] {
    let lon = (u - 0.5) * TAU;
    let lat = (0.5 - v) * PI;
    let (cl, sl) = (lat.cos(), lat.sin());
    [cl * lon.cos(), sl, cl * lon.sin()]
}

// ---------------------------------------------------------------------------
// The planet's one source of truth: an elevation field
// ---------------------------------------------------------------------------

/// Elevation in `-1..=1` at a point on the sphere. Below 0 is ocean.
///
/// Every map is derived from this, which is why the coastline in the colour
/// map, the relief in the normal map and the shoreline glow of the city lights
/// all agree.
fn elevation(p: [f32; 3]) -> f32 {
    // A few big masses, carried down to bay-and-headland scale — eight octaves
    // from 1.5 reaches ~190 cycles around the globe, which is what gives the
    // coastlines something to be other than smooth blobs.
    let continents = fbm(p, 1.5, 8, 11);
    // …with their outlines warped, so they are not obviously noise-shaped.
    let warp = fbm(p, 4.3, 4, 71) - 0.5;
    let mass = continents + warp * 0.22;

    // Sea level picked from the noise's own distribution to land near Earth's
    // 29% land fraction.
    let land = mass - 0.545;
    if land <= 0.0 {
        // Ocean: deeper away from shore, with some sea-floor texture.
        let floor = fbm(p, 6.0, 4, 401);
        return (land * 2.4 - floor * 0.10).max(-1.0);
    }
    // Land: mountains along the ridges of the mass field, squared so they form
    // chains rather than covering every continent.
    let mountains = ridged(p, 7.5, 5, 907);
    let relief = fbm(p, 18.0, 4, 55);
    ((land * 3.0) + mountains * mountains * 0.6 * (land * 7.0).min(1.0) + relief * 0.07).min(1.0)
}

/// Run `f` over row bands on every core. The maps are large enough that
/// generating them one row at a time is the slowest part of the example.
fn parallel_rows(height: u32, f: impl Fn(u32, &mut [u8]) + Sync, stride: usize, out: &mut [u8]) {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(height as usize)
        .max(1);
    let rows_per = height.div_ceil(threads as u32) as usize;
    let f = &f;
    std::thread::scope(|scope| {
        for (band, chunk) in out.chunks_mut(rows_per * stride).enumerate() {
            scope.spawn(move || {
                let first = (band * rows_per) as u32;
                for (i, row) in chunk.chunks_mut(stride).enumerate() {
                    f(first + i as u32, row);
                }
            });
        }
    });
}

struct EarthMaps {
    albedo: Texture,
    normal: Texture,
    roughness: Texture,
    night: Texture,
    clouds: Texture,
}

/// Build every map at `width × width/2`.
fn build_earth_maps(width: u32) -> EarthMaps {
    let height = width / 2;
    let (w, h) = (width as usize, height as usize);

    // ---- shared elevation field, sampled once and reused by every map ----
    let mut field = vec![0f32; w * h];
    {
        let rows = &mut field;
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(1);
        let rows_per = h.div_ceil(threads);
        std::thread::scope(|scope| {
            for (band, chunk) in rows.chunks_mut(rows_per * w).enumerate() {
                scope.spawn(move || {
                    for (i, row) in chunk.chunks_mut(w).enumerate() {
                        let y = (band * rows_per + i) as u32;
                        let v = (y as f32 + 0.5) / h as f32;
                        for (x, cell) in row.iter_mut().enumerate() {
                            let u = (x as f32 + 0.5) / w as f32;
                            *cell = elevation(direction(u, v));
                        }
                    }
                });
            }
        });
    }
    let at = |x: isize, y: isize| -> f32 {
        // Wrap in longitude, clamp at the poles.
        let xi = x.rem_euclid(w as isize) as usize;
        let yi = y.clamp(0, h as isize - 1) as usize;
        field[yi * w + xi]
    };

    // ---- albedo: oceans by depth, land by climate band ----
    let mut albedo = vec![0u8; w * h * 4];
    parallel_rows(
        height,
        |y, row| {
            let v = (y as f32 + 0.5) / h as f32;
            let lat = (v - 0.5) * PI;
            // Ice grows from the poles, and further on the colder south.
            let polar_base = (lat.abs() - 1.02) / 0.34;
            for x in 0..w {
                let u = (x as f32 + 0.5) / w as f32;
                let e = field[y as usize * w + x];
                // Ragged ice edge rather than a latitude line.
                let polar = (polar_base + (fbm(direction(u, v), 14.0, 3, 4242) - 0.5) * 0.55)
                    .clamp(0.0, 1.0);
                let rgb = if e <= 0.0 {
                    // Deep ocean is dark and blue; shelves are green-teal.
                    let depth = (-e / 0.6).clamp(0.0, 1.0);
                    let shallow = [0.05, 0.30, 0.36];
                    let deep = [0.008, 0.035, 0.13];
                    [
                        shallow[0] + (deep[0] - shallow[0]) * depth,
                        shallow[1] + (deep[1] - shallow[1]) * depth,
                        shallow[2] + (deep[2] - shallow[2]) * depth,
                    ]
                } else {
                    // Climate by latitude, with noise so the bands are not stripes.
                    let jitter = fbm(direction(u, v), 9.0, 3, 1234);
                    let warmth = (1.0 - lat.abs() / 1.4).clamp(0.0, 1.0) + (jitter - 0.5) * 0.28;
                    // Dry belts sit either side of the tropics, but ragged:
                    // the noise decides where they actually bite.
                    let belt = (1.0 - ((lat.abs() - 0.46).abs() / 0.30)).clamp(0.0, 1.0);
                    let dryness = fbm(direction(u, v), 5.5, 4, 6060);
                    let desert = (belt * 1.25 * (dryness - 0.34) * 3.0).clamp(0.0, 1.0);
                    let tundra = [0.34, 0.33, 0.28];
                    let forest = [0.09, 0.20, 0.08];
                    let jungle = [0.06, 0.24, 0.07];
                    let sand = [0.62, 0.51, 0.31];
                    let base = [
                        tundra[0] + (jungle[0] - tundra[0]) * warmth,
                        tundra[1] + (jungle[1] - tundra[1]) * warmth,
                        tundra[2] + (jungle[2] - tundra[2]) * warmth,
                    ];
                    let green = [
                        base[0] * 0.5 + forest[0] * 0.5,
                        base[1] * 0.5 + forest[1] * 0.5,
                        base[2] * 0.5 + forest[2] * 0.5,
                    ];
                    let land = [
                        green[0] + (sand[0] - green[0]) * desert,
                        green[1] + (sand[1] - green[1]) * desert,
                        green[2] + (sand[2] - green[2]) * desert,
                    ];
                    // The snow line falls as you leave the tropics, so the same
                    // peak is bare near the equator and white further north.
                    let snow_line = 0.78 - 0.62 * (lat.abs() / 1.5).powi(2);
                    let alpine = ((e - snow_line) / 0.14).clamp(0.0, 1.0);
                    let snow = polar.max(alpine);
                    [
                        land[0] + (0.92 - land[0]) * snow,
                        land[1] + (0.94 - land[1]) * snow,
                        land[2] + (0.97 - land[2]) * snow,
                    ]
                };
                // Sea ice: the ocean freezes over at the poles too.
                let rgb = if e <= 0.0 && polar > 0.35 {
                    let t = ((polar - 0.35) / 0.4).clamp(0.0, 1.0);
                    [
                        rgb[0] + (0.88 - rgb[0]) * t,
                        rgb[1] + (0.91 - rgb[1]) * t,
                        rgb[2] + (0.95 - rgb[2]) * t,
                    ]
                } else {
                    rgb
                };
                let o = x * 4;
                row[o] = (rgb[0].clamp(0.0, 1.0) * 255.0) as u8;
                row[o + 1] = (rgb[1].clamp(0.0, 1.0) * 255.0) as u8;
                row[o + 2] = (rgb[2].clamp(0.0, 1.0) * 255.0) as u8;
                row[o + 3] = 255;
            }
        },
        w * 4,
        &mut albedo,
    );

    // ---- normal map from the elevation field (Sobel) ----
    let mut normal = vec![0u8; w * h * 4];
    parallel_rows(
        height,
        |y, row| {
            let v = (y as f32 + 0.5) / h as f32;
            // Longitude compresses toward the poles, so the x slope has to be
            // divided by cos(latitude) or the relief smears sideways.
            let scale = 1.0 / ((v - 0.5) * PI).cos().max(0.2);
            for x in 0..w {
                let (xi, yi) = (x as isize, y as isize);
                let e = at(xi, yi);
                // Only land carries relief; the sea surface is flat.
                let strength = if e > 0.0 { 3.2 } else { 0.25 };
                let dx = (at(xi + 1, yi) - at(xi - 1, yi)) * 0.5 * scale * strength;
                let dy = (at(xi, yi + 1) - at(xi, yi - 1)) * 0.5 * strength;
                let len = (dx * dx + dy * dy + 1.0).sqrt();
                let o = x * 4;
                row[o] = ((-dx / len * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
                row[o + 1] = ((-dy / len * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
                row[o + 2] = ((1.0 / len * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
                row[o + 3] = 255;
            }
        },
        w * 4,
        &mut normal,
    );

    // ---- roughness: water is a mirror, land is not ----
    let mut roughness = vec![0u8; w * h];
    parallel_rows(
        height,
        |y, row| {
            let v = (y as f32 + 0.5) / h as f32;
            let polar = (((v - 0.5) * PI).abs() - 1.16).max(0.0) / 0.28;
            for x in 0..w {
                let e = field[y as usize * w + x];
                let r = if e <= 0.0 {
                    // Open water is smooth; sea ice is not.
                    0.12 + 0.7 * polar.clamp(0.0, 1.0)
                } else {
                    0.82 + (e * 0.15).min(0.12)
                };
                row[x] = (r.clamp(0.0, 1.0) * 255.0) as u8;
            }
        },
        w,
        &mut roughness,
    );

    // ---- night lights: population clusters, on land, biased to coasts ----
    let mut night = vec![0u8; w * h * 4];
    parallel_rows(
        height,
        |y, row| {
            let v = (y as f32 + 0.5) / h as f32;
            let lat = (v - 0.5) * PI;
            // Nobody lives on the ice caps.
            let habitable = (1.0 - ((lat.abs() - 0.30) / 0.95).clamp(0.0, 1.0)).clamp(0.0, 1.0);
            for x in 0..w {
                let e = field[y as usize * w + x];
                let o = x * 4;
                if e <= 0.005 {
                    row[o..o + 4].copy_from_slice(&[0, 0, 0, 255]);
                    continue;
                }
                let u = (x as f32 + 0.5) / w as f32;
                // Cities cluster: a low-frequency "where people are" field
                // multiplied by a high-frequency "individual towns" field.
                let here = direction(u, v);
                let region = fbm(here, 7.0, 4, 5150);
                let towns = fbm(here, 44.0, 3, 8191);
                // Coasts and lowlands are settled; mountains are not.
                let lowland = (1.0 - (e / 0.45).clamp(0.0, 1.0)).powf(1.5);
                let mut lit = (region - 0.52).max(0.0) * 4.0 * lowland * habitable;
                lit *= ((towns - 0.45) * 3.0).clamp(0.0, 1.0);
                let lit = lit.clamp(0.0, 1.0).powf(1.6);
                // Sodium-lamp orange, with the brightest cores going white.
                let warm = [1.0, 0.78, 0.43];
                let core = (lit - 0.55).max(0.0) / 0.45;
                row[o] = (lit * (warm[0] + (1.0 - warm[0]) * core) * 255.0) as u8;
                row[o + 1] = (lit * (warm[1] + (1.0 - warm[1]) * core) * 255.0) as u8;
                row[o + 2] = (lit * (warm[2] + (1.0 - warm[2]) * core) * 255.0) as u8;
                row[o + 3] = 255;
            }
        },
        w * 4,
        &mut night,
    );

    // ---- clouds: banded by latitude, the way the real circulation puts them ----
    let mut clouds = vec![0u8; w * h * 4];
    parallel_rows(
        height,
        |y, row| {
            let v = (y as f32 + 0.5) / h as f32;
            let lat = (v - 0.5) * PI;
            // Wet at the equator and the polar fronts, dry over the horse
            // latitudes — which is also where the deserts are.
            let band = 0.55 + 0.45 * (lat * 6.0).cos() * (1.0 - lat.abs() / 1.7).max(0.0);
            for x in 0..w {
                let u = (x as f32 + 0.5) / w as f32;
                // Stretched in longitude: weather comes in streaks, not blobs.
                // Stretched along longitude: weather comes in streaks, not blobs.
                let d = direction(u, v);
                let n = fbm([d[0], d[1] * 2.6, d[2]], 6.0, 6, 31337);
                let a = ((n * band - 0.42) * 3.4).clamp(0.0, 1.0);
                let o = x * 4;
                row[o] = 255;
                row[o + 1] = 255;
                row[o + 2] = 255;
                row[o + 3] = (a * 255.0) as u8;
            }
        },
        w * 4,
        &mut clouds,
    );

    // Equirectangular maps must repeat in longitude or the date line shows.
    let wrap = |mut t: Texture| {
        t.wrap_s = TextureWrap::Repeat;
        t.flip_y = false;
        t
    };
    EarthMaps {
        albedo: wrap(Texture::new(
            width,
            height,
            TextureFormat::Rgba8UnormSrgb,
            albedo,
        )),
        normal: wrap(Texture::new(
            width,
            height,
            TextureFormat::Rgba8Unorm,
            normal,
        )),
        roughness: wrap(Texture::new(
            width,
            height,
            TextureFormat::R8Unorm,
            roughness,
        )),
        night: wrap(Texture::new(
            width,
            height,
            TextureFormat::Rgba8UnormSrgb,
            night,
        )),
        clouds: wrap(Texture::new(
            width,
            height,
            TextureFormat::Rgba8UnormSrgb,
            clouds,
        )),
    }
}

// ---------------------------------------------------------------------------
// Lens flare — composited in screen space, after the render
// ---------------------------------------------------------------------------

/// Additively draw a soft disc.
#[allow(clippy::too_many_arguments)]
fn splat(
    buf: &mut [f32],
    w: u32,
    h: u32,
    cx: f32,
    cy: f32,
    radius: f32,
    color: [f32; 3],
    gain: f32,
) {
    if gain <= 0.0 || radius <= 0.5 {
        return;
    }
    let x0 = ((cx - radius).floor().max(0.0)) as u32;
    let x1 = ((cx + radius).ceil().min(w as f32 - 1.0)).max(0.0) as u32;
    let y0 = ((cy - radius).floor().max(0.0)) as u32;
    let y1 = ((cy + radius).ceil().min(h as f32 - 1.0)).max(0.0) as u32;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let d = (((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt() / radius).min(1.0);
            // Smooth falloff, brightest at the centre.
            let f = (1.0 - d * d).powi(2) * gain;
            if f <= 0.0 {
                continue;
            }
            let i = ((y * w + x) * 3) as usize;
            buf[i] += color[0] * f;
            buf[i + 1] += color[1] * f;
            buf[i + 2] += color[2] * f;
        }
    }
}

/// A ring: the halo that forms around the optical axis.
#[allow(clippy::too_many_arguments)]
fn ring(
    buf: &mut [f32],
    w: u32,
    h: u32,
    cx: f32,
    cy: f32,
    radius: f32,
    thickness: f32,
    color: [f32; 3],
    gain: f32,
) {
    let outer = radius + thickness;
    let x0 = ((cx - outer).floor().max(0.0)) as u32;
    let x1 = ((cx + outer).ceil().min(w as f32 - 1.0)).max(0.0) as u32;
    let y0 = ((cy - outer).floor().max(0.0)) as u32;
    let y1 = ((cy + outer).ceil().min(h as f32 - 1.0)).max(0.0) as u32;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
            // Gaussian rather than a clipped quadratic: a band with a definite
            // outer boundary reads as a drawn circle however soft its centre is.
            let t = (d - radius) / thickness;
            let f = (-t * t * 2.2).exp() * gain;
            if f < 1e-4 {
                continue;
            }
            if f <= 0.0 {
                continue;
            }
            let i = ((y * w + x) * 3) as usize;
            buf[i] += color[0] * f;
            buf[i + 1] += color[1] * f;
            buf[i + 2] += color[2] * f;
        }
    }
}

/// A thin glow just outside the planet's limb, brightest on the sunward side.
///
/// Screen space, like the flare: the real thing is a few kilometres of air seen
/// edge-on, and faking it here costs one pass over the rim instead of a
/// volumetric shell nobody would look inside.
#[allow(clippy::too_many_arguments)]
fn draw_atmosphere(
    rgba: &mut [u8],
    w: u32,
    h: u32,
    cx: f32,
    cy: f32,
    radius: f32,
    sun: (f32, f32),
    strength: f32,
) {
    if radius <= 2.0 || strength <= 0.0 {
        return;
    }
    let thickness = (radius * 0.085).max(2.0);
    let outer = radius + thickness * 2.2;
    let (sx, sy) = (sun.0 - cx, sun.1 - cy);
    let sun_len = (sx * sx + sy * sy).sqrt().max(1e-3);
    let (sdx, sdy) = (sx / sun_len, sy / sun_len);

    let x0 = ((cx - outer).max(0.0)) as u32;
    let x1 = ((cx + outer).min(w as f32 - 1.0)).max(0.0) as u32;
    let y0 = ((cy - outer).max(0.0)) as u32;
    let y1 = ((cy + outer).min(h as f32 - 1.0)).max(0.0) as u32;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let (dx, dy) = (x as f32 - cx, y as f32 - cy);
            let d = (dx * dx + dy * dy).sqrt();
            // A band hugging the limb, falling off outward and a little inward.
            let band = if d >= radius {
                (1.0 - (d - radius) / (thickness * 2.2)).max(0.0).powi(2)
            } else {
                (1.0 - (radius - d) / (thickness * 0.9)).max(0.0).powi(2)
            };
            if band <= 0.001 {
                continue;
            }
            // Forward-scattering: the air glows where the sun is behind it.
            let facing = ((dx / d.max(1e-3)) * sdx + (dy / d.max(1e-3)) * sdy).max(0.0);
            let gain = band * (0.18 + 0.82 * facing.powf(1.6)) * strength;
            let i = ((y * w + x) * 4) as usize;
            // Rayleigh-ish blue, warming toward the sun.
            let tint = [0.35 + 0.55 * facing, 0.62 + 0.30 * facing, 1.0];
            for c in 0..3 {
                let base = rgba[i + c] as f32 / 255.0;
                let out = 1.0 - (1.0 - base) * (1.0 - (tint[c] * gain).clamp(0.0, 1.0));
                rgba[i + c] = (out * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Composite a lens flare for a light at `(sx, sy)` in pixels.
///
/// The ghosts sit on the line from the light through the centre of the frame,
/// which is what a real lens does: each one is an internal reflection between
/// two elements, mirrored about the optical axis.
fn draw_flare(rgba: &mut [u8], w: u32, h: u32, sx: f32, sy: f32, strength: f32) {
    if strength <= 0.001 {
        return;
    }
    let mut acc = vec![0f32; (w * h * 3) as usize];
    let (cx, cy) = (w as f32 * 0.5, h as f32 * 0.5);
    let (dx, dy) = (cx - sx, cy - sy);
    let span = (w as f32).hypot(h as f32);

    // The sun's own glow, and a tight core.
    let warm = [1.0, 0.93, 0.78];
    splat(&mut acc, w, h, sx, sy, span * 0.22, warm, 0.22 * strength);
    splat(
        &mut acc,
        w,
        h,
        sx,
        sy,
        span * 0.055,
        [1.0, 0.97, 0.9],
        0.85 * strength,
    );

    // Anamorphic streak — the horizontal smear of a real lens.
    for i in 0..2 {
        let len = span * if i == 0 { 0.5 } else { 0.16 };
        let steps = (len * 2.0) as i32;
        for s in -steps..=steps {
            let t = s as f32 / steps as f32;
            let fall = (1.0 - t.abs()).powi(3);
            splat(
                &mut acc,
                w,
                h,
                sx + t * len,
                sy,
                if i == 0 { 5.0 } else { 12.0 },
                if i == 0 { [0.55, 0.72, 1.0] } else { warm },
                0.05 * fall * strength,
            );
        }
    }

    // Ghosts: evenly spaced along the axis, alternating warm and cool, with a
    // couple thrown past the centre.
    let ghosts: [(f32, f32, [f32; 3], f32); 8] = [
        (0.30, 0.030, [0.35, 0.55, 1.00], 0.18),
        (0.55, 0.055, [1.00, 0.72, 0.35], 0.16),
        (0.80, 0.022, [0.55, 1.00, 0.70], 0.20),
        (1.00, 0.070, [0.40, 0.60, 1.00], 0.10),
        (1.25, 0.035, [1.00, 0.55, 0.45], 0.14),
        (1.55, 0.090, [0.45, 0.75, 1.00], 0.07),
        (1.85, 0.028, [1.00, 0.85, 0.50], 0.12),
        (2.10, 0.048, [0.70, 0.50, 1.00], 0.08),
    ];
    for (t, size, color, gain) in ghosts {
        splat(
            &mut acc,
            w,
            h,
            sx + dx * 2.0 * t,
            sy + dy * 2.0 * t,
            span * size,
            color,
            gain * strength,
        );
    }

    // The halo around the optical axis.
    ring(
        &mut acc,
        w,
        h,
        cx + dx * 0.15,
        cy + dy * 0.15,
        span * 0.20,
        // Wide enough to read as a wash rather than an outline.
        span * 0.085,
        [0.85, 0.62, 0.45],
        0.055 * strength,
    );

    // Screen blend, so the flare lifts the image without clipping it flat.
    for (px, add) in rgba.chunks_exact_mut(4).zip(acc.chunks_exact(3)) {
        for c in 0..3 {
            let base = px[c] as f32 / 255.0;
            let out = 1.0 - (1.0 - base) * (1.0 - add[c].clamp(0.0, 1.0));
            px[c] = (out * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
}

/// A unit vector perpendicular to the view direction — used to find a point on
/// the planet's limb so its projected radius can be measured.
fn camera_right(camera: &PerspectiveCamera) -> Vector3 {
    let forward = (camera.target - camera.position).normalize();
    let right = forward.cross(Vector3::new(0.0, 1.0, 0.0));
    if right.length() < 1e-4 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        right.normalize()
    }
}

/// Where `world` lands on screen, and whether it is in front of the camera.
fn project(world: Vector3, camera: &PerspectiveCamera, w: u32, h: u32) -> Option<(f32, f32)> {
    use threers::cameras::Camera;
    let m = camera.projection_matrix().multiply(&camera.view_matrix());
    let e = &m.elements;
    let cw = e[3] * world.x + e[7] * world.y + e[11] * world.z + e[15];
    if cw <= 0.0 {
        return None; // behind the camera
    }
    let x = (e[0] * world.x + e[4] * world.y + e[8] * world.z + e[12]) / cw;
    let y = (e[1] * world.x + e[5] * world.y + e[9] * world.z + e[13]) / cw;
    Some((
        (x * 0.5 + 0.5) * w as f32,
        (1.0 - (y * 0.5 + 0.5)) * h as f32,
    ))
}

/// How much of the sun is visible past a sphere of `radius` at the origin.
///
/// `1` clear, `0` fully behind the planet, with a soft edge so the flare fades
/// as the sun sets behind the limb rather than snapping off.
fn sun_visibility(eye: Vector3, sun: Vector3, radius: f32) -> f32 {
    let dir = (sun - eye).normalize();
    let to_center = Vector3::ZERO - eye;
    let along = to_center.dot(dir);
    if along <= 0.0 {
        return 1.0; // planet is behind us
    }
    if along > (sun - eye).length() {
        return 1.0; // planet is beyond the sun
    }
    let miss = (to_center - dir * along).length();
    // Fade across a band one tenth of the radius wide at the limb.
    ((miss - radius) / (radius * 0.10)).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

const EARTH_R: f32 = 1.0;
const SUN_DISTANCE: f32 = 60.0;

fn main() {
    let tex_size: u32 = arg("--texture")
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096)
        .clamp(256, 8192);
    let (w, h) = arg("--size")
        .and_then(|s| {
            let (a, b) = s.split_once('x')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        })
        .unwrap_or((1600u32, 900u32));
    let frames: usize = arg("--frames").and_then(|s| s.parse().ok()).unwrap_or(1);
    let video = arg("--video");
    let want_flare = !flag("--no-flare");
    let want_clouds = !flag("--no-clouds");
    let _ = std::fs::create_dir_all("out");

    println!("generating {tex_size}x{} maps…", tex_size / 2);
    let start = std::time::Instant::now();
    let maps = build_earth_maps(tex_size);
    let megatexels = (tex_size as f64 * (tex_size / 2) as f64) / 1.0e6;
    println!(
        "  5 maps, {megatexels:.1} Mtexel each, in {:.1}s",
        start.elapsed().as_secs_f32()
    );
    if flag("--dump-maps") {
        let half = tex_size / 2;
        let write = |name: &str, t: &Texture| {
            // R8 maps have to be expanded to RGBA before they can be a PNG.
            let rgba: Vec<u8> = if t.format == TextureFormat::R8Unorm {
                t.data.iter().flat_map(|&v| [v, v, v, 255]).collect()
            } else {
                t.data.as_ref().clone()
            };
            let path = format!("out/earth_map_{name}.png");
            std::fs::write(&path, encode_png(tex_size, half, &rgba)).expect("write map");
            println!("  wrote {path}");
        };
        write("albedo", &maps.albedo);
        write("normal", &maps.normal);
        write("roughness", &maps.roughness);
        write("night", &maps.night);
        write("clouds", &maps.clouds);
    }

    let mut renderer = HeadlessRenderer::builder()
        .size(w, h)
        .supersample(2)
        // Rgba8Unorm: the mesh shader already encodes sRGB.
        .color_format(wgpu::TextureFormat::Rgba8Unorm)
        .high_resolution(true)
        .build()
        .expect("headless renderer (is a GPU adapter available?)");
    // The sun is hundreds of times brighter than the Earth it lights; ACES
    // rolls that off instead of clipping the whole disc to flat white.
    renderer
        .renderer()
        .set_tone_mapping(threers::ToneMapping::AcesFilmic, 1.0);

    // ---- Earth ----
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x01020a);

    let mut earth_mat = StandardMaterial::new(Color::WHITE);
    earth_mat.map = Some(std::sync::Arc::new(maps.albedo));
    earth_mat.normal_map = Some(std::sync::Arc::new(maps.normal));
    earth_mat.roughness_map = Some(std::sync::Arc::new(maps.roughness));
    earth_mat.emissive_map = Some(std::sync::Arc::new(maps.night));
    // The night side is lit only by its own cities.
    earth_mat.emissive = Color::WHITE;
    earth_mat.emissive_intensity = 2.3;
    earth_mat.metalness = 0.0;
    earth_mat.roughness = 1.0; // scaled by the roughness map
    let earth = scene.add(Object3D::mesh(Mesh::new(
        SphereGeometry::new(EARTH_R, 256, 128),
        Material::Standard(earth_mat),
    )));

    // ---- clouds: a second, slightly larger shell ----
    if want_clouds {
        // The per-texel alpha in the cloud map does the shaping; the material
        // opacity only has to be below 1 to put the shell on the alpha
        // pipeline, and doubles as a thin-cloud haze control.
        let mut cloud_mat = StandardMaterial::new(Color::WHITE);
        cloud_mat.map = Some(std::sync::Arc::new(maps.clouds));
        cloud_mat.roughness = 0.95;
        cloud_mat.metalness = 0.0;
        cloud_mat.opacity = 0.96;
        let mut shell = Object3D::mesh(Mesh::new(
            SphereGeometry::new(EARTH_R * 1.012, 192, 96),
            Material::Standard(cloud_mat),
        ));
        shell.name = "clouds".into();
        scene.add(shell);
    }

    // ---- the sun: a bright body, and the light it casts ----
    let sun_dir = Vector3::new(1.0, 0.18, 0.55).normalize();
    let sun_pos = sun_dir * SUN_DISTANCE;
    let mut sun_mat = StandardMaterial::new(Color::from_hex(0xfff6e0));
    sun_mat.emissive = Color::from_hex(0xfff4d6);
    // Far brighter than white: tone mapping turns this into a believable disc.
    sun_mat.emissive_intensity = 90.0;
    let mut sun = Object3D::mesh(Mesh::new(
        SphereGeometry::new(1.7, 48, 24),
        Material::Standard(sun_mat),
    ));
    sun.position = sun_pos;
    scene.add(sun);

    scene.add_light(
        threers::DirectionalLight::new(Color::from_hex(0xfff6ea), 3.1)
            .with_direction(sun_dir * -1.0),
    );
    // A whisper of starlight so the night side is not pure black.
    scene.add_light(threers::AmbientLight::new(Color::from_hex(0x0a1428), 0.35));

    // ---- camera ----
    let mut camera = PerspectiveCamera::new(42.0, w as f32 / h as f32, 0.01, 400.0);

    let shot = |i: usize| -> (Vector3, f32) {
        // Swing around so the terminator, the night side and the sun all pass
        // through frame over one loop.
        let t = if frames > 1 {
            i as f32 / frames as f32
        } else {
            0.0
        };
        let yaw = 0.55 + t * TAU;
        let eye = Vector3::new(
            yaw.cos() * 3.3,
            0.45 + (t * TAU).sin() * 0.30,
            yaw.sin() * 3.3,
        );
        (eye, t)
    };

    /// Aim between the planet and the sun, but only when the sun is ahead.
    ///
    /// You cannot have a fully lit Earth *and* the sun in the same shot — the
    /// sun would have to be behind you. So the camera centres the planet while
    /// the sun is at its back, and slides toward the sun as it comes round the
    /// limb, which is where the flare lives.
    fn aim(eye: Vector3, sun: Vector3, max_bias: f32) -> Vector3 {
        let to_earth = (Vector3::ZERO - eye).normalize();
        let to_sun = (sun - eye).normalize();
        // Full bias while the sun is within ~60° of the planet, fading out by
        // ~100°, past which it is behind us and there is nothing to frame.
        let ahead = ((to_earth.dot(to_sun) + 0.17) / 0.67).clamp(0.0, 1.0);
        let bias = max_bias * ahead;
        let blend = (to_earth * (1.0 - bias) + to_sun * bias).normalize();
        eye + blend * eye.length().max(0.1)
    }

    let mut render_frame =
        |i: usize, camera: &mut PerspectiveCamera, scene: &mut Scene| -> Vec<u8> {
            let (eye, t) = shot(i);
            camera.position = eye;
            camera.look_at(aim(eye, sun_pos, 0.42));

            // Spin the planet, and drift the clouds a little faster.
            if let Some(o) = scene.get_mut(earth) {
                o.quaternion = threers::Quaternion::from_euler_xyz(0.0, t * TAU * 0.35, 0.0);
            }

            let mut rgba = renderer.render_to_rgba_resolved(scene, camera);

            if let Some((sx, sy)) = project(sun_pos, camera, w, h) {
                if let (Some(c), Some(edge)) = (
                    project(Vector3::ZERO, camera, w, h),
                    project(camera_right(camera) * EARTH_R, camera, w, h),
                ) {
                    let radius = ((edge.0 - c.0).powi(2) + (edge.1 - c.1).powi(2)).sqrt();
                    draw_atmosphere(&mut rgba, w, h, c.0, c.1, radius, (sx, sy), 0.85);
                }
            }
            if want_flare {
                // Project the sun, and fade the flare out behind the planet.
                if let Some((sx, sy)) = project(sun_pos, camera, w, h) {
                    let visible = sun_visibility(eye, sun_pos, EARTH_R * 1.02);
                    // Also fade as it leaves the frame — a lens that cannot see the
                    // sun does not flare.
                    let margin = 0.35;
                    let inside = |v: f32, size: f32| {
                        let over = (-v).max(v - size) / (size * margin);
                        (1.0 - over).clamp(0.0, 1.0)
                    };
                    let framing = inside(sx, w as f32) * inside(sy, h as f32);
                    draw_flare(&mut rgba, w, h, sx, sy, visible * framing);
                }
            }
            rgba
        };

    if frames <= 1 {
        // Three compositions, because one camera cannot show all of this at
        // once: the sun has to be behind you for full daylight, and in front of
        // you for a flare.
        let mut still = |name: &str, eye: Vector3, target: Vector3, fov: f32, flare: bool| {
            camera.fov = fov.to_radians();
            camera.position = eye;
            camera.look_at(target);
            let mut rgba = renderer.render_to_rgba_resolved(&mut scene, &camera);
            if let Some((sx, sy)) = project(sun_pos, &camera, w, h) {
                // The atmosphere rides the limb whether or not the flare is on.
                if let (Some(c), Some(edge)) = (
                    project(Vector3::ZERO, &camera, w, h),
                    project(camera_right(&camera) * EARTH_R, &camera, w, h),
                ) {
                    let radius = ((edge.0 - c.0).powi(2) + (edge.1 - c.1).powi(2)).sqrt();
                    draw_atmosphere(&mut rgba, w, h, c.0, c.1, radius, (sx, sy), 0.85);
                }
                if flare && want_flare {
                    let visible = sun_visibility(eye, sun_pos, EARTH_R * 1.02);
                    draw_flare(&mut rgba, w, h, sx, sy, visible);
                }
            }
            let path = format!("out/{name}.png");
            std::fs::write(&path, encode_png(w, h, &rgba)).expect("write png");
            println!("wrote {path} ({w}x{h})");
        };

        let back = sun_dir * -1.0;
        let side = back.cross(Vector3::new(0.0, 1.0, 0.0)).normalize();
        let up = side.cross(back).normalize();

        // 1. The headline, and a geometry problem: the further round you swing
        //    from the anti-sun axis the fatter the lit crescent, but the further
        //    the sun slides out of frame. At 3.5 units the disc is 17° across
        //    and a 62° lens spans ~47° to the side, so ~125° off the axis is as
        //    far as it goes with both still in shot.
        let eye = (back * 0.574 + side * 0.819 + up * 0.18).normalize() * 3.5;
        still("earth_sun_flare", eye, aim(eye, sun_pos, 0.45), 62.0, true);

        // 2. Full daylight, sun over our shoulder — this is the textures' shot.
        still(
            "earth_day",
            sun_dir * 3.0 + up * 0.85,
            Vector3::ZERO,
            40.0,
            false,
        );

        // 3. The night side, carried by the city lights alone.
        still(
            "earth_night",
            (back * 1.9 - side * 2.4 + up * 0.6).normalize() * 3.2,
            Vector3::ZERO,
            40.0,
            false,
        );
        return;
    }

    println!("rendering {frames} frames at {w}x{h}…");
    let mut captured = Vec::with_capacity(frames);
    for i in 0..frames {
        captured.push(render_frame(i, &mut camera, &mut scene));
        eprint!("\r  {}/{frames}", i + 1);
        let _ = std::io::Write::flush(&mut std::io::stderr());
    }
    eprintln!();

    if let Some(path) = video {
        #[cfg(feature = "video")]
        {
            use threers::{export_video, VideoCodec, VideoOptions};
            let codec = if path.ends_with(".gif") {
                VideoCodec::Gif
            } else {
                VideoCodec::H264
            };
            let options = VideoOptions::new(path.clone()).fps(30).codec(codec).crf(18);
            match export_video(w, h, captured.len(), &options, |i| captured[i].clone()) {
                Ok(()) => println!("wrote {path}"),
                Err(e) => eprintln!("video export failed: {e}"),
            }
        }
        #[cfg(not(feature = "video"))]
        eprintln!("--video needs `--features video` (path was {path})");
    } else {
        for (i, rgba) in captured.iter().enumerate() {
            let out = format!("out/earth_{i:04}.png");
            std::fs::write(&out, encode_png(w, h, rgba)).expect("write png");
        }
        println!("wrote out/earth_0000.png … ({frames} frames)");
    }
}
