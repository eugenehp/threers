// Shared procedural assets for the spacecraft-material examples.
// Included via `include!("spacecraft_common.inc")`.
//
// Everything here is generated in code so the examples need no downloaded HDRIs
// or texture files — you can run them from a fresh clone.

use std::sync::Arc;

use threers::{Color, CubeTexture, PmremGenerator, Texture, TextureFormat};

// ---------------------------------------------------------------------------
// Value noise — the basis for both the foil crinkle and the star field.
// ---------------------------------------------------------------------------

fn hash2(x: i32, y: i32) -> f32 {
    let mut h = (x.wrapping_mul(374_761_393)).wrapping_add(y.wrapping_mul(668_265_263));
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    ((h ^ (h >> 16)) as u32 as f32) / (u32::MAX as f32)
}

fn smooth(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Bilinearly-interpolated value noise on a unit-cell lattice.
fn value_noise(x: f32, y: f32) -> f32 {
    let xi = x.floor();
    let yi = y.floor();
    let tx = smooth(x - xi);
    let ty = smooth(y - yi);
    let (x0, y0) = (xi as i32, yi as i32);

    let a = hash2(x0, y0);
    let b = hash2(x0 + 1, y0);
    let c = hash2(x0, y0 + 1);
    let d = hash2(x0 + 1, y0 + 1);

    let top = a + (b - a) * tx;
    let bot = c + (d - c) * tx;
    top + (bot - top) * ty
}

/// Ridged fractal noise — the absolute value of signed octaves, which produces
/// creases rather than blobs. That crease structure is what makes the foil read
/// as *crumpled* instead of merely bumpy.
fn ridged_fbm(x: f32, y: f32, octaves: u32) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut freq = 1.0;
    let mut norm = 0.0;
    for _ in 0..octaves {
        let n = value_noise(x * freq, y * freq) * 2.0 - 1.0;
        sum += (1.0 - n.abs()) * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.07; // non-integer, so octaves don't align into visible grids
    }
    sum / norm.max(1e-6)
}

// ---------------------------------------------------------------------------
// Textures
// ---------------------------------------------------------------------------

/// Tangent-space normal map of crumpled foil.
///
/// This is the single most important asset for MLI: a smooth gold sphere reads
/// as a snooker ball, and the same sphere with this map reads as a thermal
/// blanket. `scale` sets the crease frequency, `strength` the relief depth.
fn crinkle_normal_map(size: u32, scale: f32, strength: f32) -> Arc<Texture> {
    let n = size as usize;
    let mut height = vec![0.0f32; n * n];
    for y in 0..n {
        for x in 0..n {
            let u = x as f32 / n as f32 * scale;
            let v = y as f32 / n as f32 * scale;
            height[y * n + x] = ridged_fbm(u, v, 5);
        }
    }

    // Central differences on the height field → tangent-space normal. Wrapping
    // the indices keeps the map tileable.
    let mut data = vec![0u8; n * n * 4];
    for y in 0..n {
        for x in 0..n {
            let xm = (x + n - 1) % n;
            let xp = (x + 1) % n;
            let ym = (y + n - 1) % n;
            let yp = (y + 1) % n;
            let dx = (height[y * n + xp] - height[y * n + xm]) * strength;
            let dy = (height[yp * n + x] - height[ym * n + x]) * strength;

            let len = (dx * dx + dy * dy + 1.0).sqrt();
            let nx = -dx / len;
            let ny = -dy / len;
            let nz = 1.0 / len;

            let i = (y * n + x) * 4;
            data[i] = ((nx * 0.5 + 0.5) * 255.0) as u8;
            data[i + 1] = ((ny * 0.5 + 0.5) * 255.0) as u8;
            data[i + 2] = ((nz * 0.5 + 0.5) * 255.0) as u8;
            data[i + 3] = 255;
        }
    }
    // Normal maps are vector data, not color — must be linear, not sRGB.
    Arc::new(Texture::new(size, size, TextureFormat::Rgba8Unorm, data))
}

/// Solar-cell grid: dark cells separated by lighter busbar / interconnect lines.
fn solar_cell_map(size: u32, cells: u32) -> Arc<Texture> {
    let n = size as usize;
    let mut data = vec![0u8; n * n * 4];
    let cell_px = (n as f32 / cells as f32).max(2.0);
    for y in 0..n {
        for x in 0..n {
            let cx = (x as f32 % cell_px) / cell_px;
            let cy = (y as f32 % cell_px) / cell_px;
            // Cell gap lines, plus three thin busbars running across each cell.
            let gap = cx < 0.03 || cy < 0.03;
            let busbar = (cx - 0.25).abs() < 0.012
                || (cx - 0.5).abs() < 0.012
                || (cx - 0.75).abs() < 0.012;

            let (r, g, b) = if gap {
                (0.42, 0.44, 0.48) // metallic interconnect
            } else if busbar {
                (0.62, 0.63, 0.66) // silver busbar
            } else {
                // Faint per-cell variation so the array isn't perfectly uniform.
                let v = 0.9 + 0.1 * hash2((x as f32 / cell_px) as i32, (y as f32 / cell_px) as i32);
                (0.16 * v, 0.20 * v, 0.34 * v)
            };
            let i = (y * n + x) * 4;
            data[i] = (r * 255.0) as u8;
            data[i + 1] = (g * 255.0) as u8;
            data[i + 2] = (b * 255.0) as u8;
            data[i + 3] = 255;
        }
    }
    Arc::new(Texture::new(size, size, TextureFormat::Rgba8UnormSrgb, data))
}

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

/// Direction vector for texel (x, y) of cube face `face`, in the
/// +X, -X, +Y, -Y, +Z, -Z order `CubeTexture` expects.
fn cube_dir(face: usize, u: f32, v: f32) -> [f32; 3] {
    // u, v in [-1, 1] across the face.
    match face {
        0 => [1.0, -v, -u],
        1 => [-1.0, -v, u],
        2 => [u, 1.0, v],
        3 => [u, -1.0, -v],
        4 => [u, -v, 1.0],
        _ => [-u, -v, -1.0],
    }
}

fn normalize3(d: [f32; 3]) -> [f32; 3] {
    let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
    [d[0] / l, d[1] / l, d[2] / l]
}

/// A low-Earth-orbit environment: black sky, scattered stars, a hard sun disc,
/// and a broad blue Earth glow filling the lower hemisphere.
///
/// This exists because **metals have no diffuse term** — with no environment a
/// `metalness = 1.0` gold sphere renders essentially black no matter how many
/// lights you add. The Earth glow also does real work here: it fills the shadow
/// side of the foils, which is what keeps them from going flat black in orbit.
///
/// `sun_dir` should match the scene's key DirectionalLight so the specular
/// highlight and the cast shadows agree.
fn space_environment(size: u32, sun_dir: [f32; 3]) -> CubeTexture {
    let sun = normalize3(sun_dir);
    let n = size as usize;
    let mut faces: Vec<Vec<f32>> = Vec::with_capacity(6);

    for face in 0..6 {
        let mut data = vec![0.0f32; n * n * 4];
        for y in 0..n {
            for x in 0..n {
                let u = (x as f32 + 0.5) / n as f32 * 2.0 - 1.0;
                let v = (y as f32 + 0.5) / n as f32 * 2.0 - 1.0;
                // NOTE the X flip. The PMREM atlas builder samples the source
                // cube at (-x, y, z) — see `rasterize_mip0_cube_uv_to_atlas_f32`
                // — so a feature authored at cube direction `d` is later seen at
                // world direction (-d.x, d.y, d.z). Without this the sun ends up
                // mirrored across the YZ plane and its reflection no longer
                // lines up with the DirectionalLight aimed along SUN_DIR.
                let dc = normalize3(cube_dir(face, u, v));
                let d = [-dc[0], dc[1], dc[2]];

                // Base: deep space with a faint zodiacal wash.
                let mut r = 0.006;
                let mut g = 0.008;
                let mut b = 0.014;

                // Earth: a blue glow from below (-Y). This is the environment's
                // main job — it's what a metal actually reflects, and it's what
                // separates gold from silver here, since gold absorbs the blue
                // and silver returns it.
                // Kept blue, but not *overwhelmingly* blue. A strongly saturated
                // fill fights warm metals: gold's F0 is (1.0, 0.766, 0.336), so
                // against a 0.23-blue / 0.055-red environment the reflected blue
                // (0.23 × 0.336 = 0.077) actually beats the reflected red
                // (0.055 × 1.0), and the gold turns olive. Balancing the fill
                // lets each metal's own F0 decide its hue.
                let earth = (-d[1]).max(0.0).powf(2.0);
                r += earth * 0.055;
                g += earth * 0.080;
                b += earth * 0.115;
                // Atmospheric limb — a brighter band near the horizon.
                let limb = (1.0 - (d[1] + 0.12).abs() * 4.5).max(0.0);
                r += limb * 0.048;
                g += limb * 0.070;
                b += limb * 0.100;

                // Sun: a tight disc at a genuinely HDR intensity.
                //
                // This is the whole reason the environment is f32. An 8-bit
                // cubemap caps at 1.0, so a sun could only be made to "read"
                // bright by making it physically huge — and a huge disc is
                // exactly wrong, because PMREM then smears it across tens of
                // degrees and every metal reflects the same grey wash. With
                // real range the disc can stay ~0.5° across, as it is in life,
                // and still dominate the prefiltered mips.
                let cos_sun = d[0] * sun[0] + d[1] * sun[1] + d[2] * sun[2];
                if cos_sun > 0.99996 {
                    r = SUN_RADIANCE;
                    g = SUN_RADIANCE * 0.97;
                    b = SUN_RADIANCE * 0.91;
                } else {
                    let bloom = ((cos_sun - 0.9993) / 0.0007).max(0.0).powf(2.0);
                    r += bloom * SUN_RADIANCE * 0.05;
                    g += bloom * SUN_RADIANCE * 0.047;
                    b += bloom * SUN_RADIANCE * 0.042;
                }

                // Stars: sparse bright points from a thresholded hash.
                let sx = (d[0] * 400.0) as i32;
                let sy = (d[1] * 400.0) as i32;
                let sz = (d[2] * 400.0) as i32;
                let s = hash2(sx.wrapping_mul(31).wrapping_add(sz), sy);
                if s > 0.9985 {
                    let mag = (s - 0.9985) / 0.0015;
                    r += mag * 0.8;
                    g += mag * 0.8;
                    b += mag * 0.9;
                }

                // Linear, unclamped — CubeTexture::new_f32 keeps the range.
                let i = (y * n + x) * 4;
                data[i] = r;
                data[i + 1] = g;
                data[i + 2] = b;
                data[i + 3] = 1.0;
            }
        }
        faces.push(data);
    }

    let faces: [Vec<f32>; 6] = faces.try_into().expect("6 cube faces");
    CubeTexture::new_f32(size, faces)
}

/// Build the orbital environment and prefilter it for roughness-based IBL.
///
/// The PMREM step is what lets a rough metal read as rough: it prefilters the
/// cubemap into a mip chain indexed by roughness. Without it every metal
/// samples a mirror-sharp environment regardless of its roughness value.
fn orbital_environment(sun_dir: [f32; 3]) -> Arc<CubeTexture> {
    let cube = space_environment(256, sun_dir);
    Arc::new(PmremGenerator::generate_pmrem(&cube, 256))
}

/// Sun direction shared by the environment and the key light, so the specular
/// highlight in the reflection lines up with the lit side of the geometry.
const SUN_DIR: [f32; 3] = [0.55, 0.42, 0.72];

/// Radiance of the solar disc, relative to the scene's diffuse white.
///
/// Far above 1.0 on purpose — that ratio is what an HDR environment exists to
/// carry, and it only renders sanely with tone mapping enabled.
const SUN_RADIANCE: f32 = 400.0;

/// Colors for the two support lights (Earth bounce and deep-space fill).
fn earth_bounce() -> Color {
    Color::from_hex(0x4a7fc8)
}
