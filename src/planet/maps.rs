//! Equirectangular map generation and derivation.
//!
//! Two jobs: build a plausible planet from noise when you have no imagery, and
//! derive the maps a PBR surface needs (relief, roughness) from a height map
//! when you do — NASA ships elevation for Earth and the Moon, but no normal or
//! specular map, and both fall out of the elevation.
//!
//! Noise is sampled in 3D *on the sphere* rather than in UV space. An
//! equirectangular map has no seam and no poles when you do that; sampling 2D
//! noise in UV gives you both.

use std::sync::Arc;

use crate::textures::{Texture, TextureFormat, TextureWrap};

/// An equirectangular image under construction: `width × height` RGBA8.
pub struct MapBuffer {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl MapBuffer {
    fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![0; (width as usize) * (height as usize) * 4],
        }
    }

    /// Wrap as a texture that repeats in longitude and clamps at the poles.
    pub fn into_texture(self, format: TextureFormat) -> Texture {
        let mut t = Texture::new(self.width, self.height, format, self.data);
        t.wrap_s = TextureWrap::Repeat;
        t.wrap_t = TextureWrap::ClampToEdge;
        t
    }
}

// ---------------------------------------------------------------------------
// Noise
// ---------------------------------------------------------------------------

/// Uniform hash of a lattice cell, in `0..=1`.
///
/// Mixed in `u32` deliberately: on a signed integer `>>` sign-extends, so
/// `h ^ (h >> 16)` always clears the top bit and the result never exceeds 0.5.
/// A half-range hash puts an entire generated planet below sea level.
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
pub fn value_noise(p: [f32; 3], seed: i32) -> f32 {
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
pub fn fbm(p: [f32; 3], frequency: f32, octaves: u32, seed: i32) -> f32 {
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
pub fn ridged(p: [f32; 3], frequency: f32, octaves: u32, seed: i32) -> f32 {
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

/// The surface point for an equirectangular texel at `(u, v)`.
///
/// This is the inverse of how [`SphereGeometry`](crate::SphereGeometry) lays a
/// map out, so a feature written at `(u, v)` appears in that direction on a
/// rendered globe. That is worth stating because it used to be the inverse of
/// something else: the sphere builds `x = -cos(phi) sin(theta)`, which puts
/// texel `u` at `pi - u * 2pi`, while this returned `(u - 0.5) * 2pi` — the
/// same longitudes, running the other way. Every generated map was therefore
/// displayed mirrored east-for-west from how it was authored.
///
/// Nothing looked wrong, because noise mirrors to noise and the one structured
/// thing here — the galactic band — is symmetric enough not to give it away.
/// It surfaced when cube faces resampled from the sphere's convention came out
/// uncorrelated with the sphere, and it would surface again the first time
/// anyone tried to put a known feature at a known longitude.
pub fn direction(u: f32, v: f32) -> [f32; 3] {
    let lon = std::f32::consts::PI - u * std::f32::consts::TAU;
    let lat = (0.5 - v) * std::f32::consts::PI;
    let (cl, sl) = (lat.cos(), lat.sin());
    [cl * lon.cos(), sl, cl * lon.sin()]
}

/// Fill `out` a row at a time, in parallel where threads exist and serially on
/// wasm32. `stride` is the row length in `T`s — 4 per texel for RGBA8, 1 for a
/// scalar field.
fn fill_rows<T: Send>(
    #[cfg_attr(target_arch = "wasm32", allow(unused_variables))] height: u32,
    stride: usize,
    out: &mut [T],
    f: impl Fn(u32, &mut [T]) + Sync,
) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(height.max(1) as usize)
            .max(1);
        let rows_per = (height as usize).div_ceil(threads);
        let f = &f;
        std::thread::scope(|scope| {
            for (band, chunk) in out.chunks_mut(rows_per * stride).enumerate() {
                scope.spawn(move || {
                    for (i, row) in chunk.chunks_mut(stride).enumerate() {
                        f((band * rows_per + i) as u32, row);
                    }
                });
            }
        });
    }
    #[cfg(target_arch = "wasm32")]
    for (y, row) in out.chunks_mut(stride).enumerate() {
        f(y as u32, row);
    }
}

// ---------------------------------------------------------------------------
// Derivation from a height field
// ---------------------------------------------------------------------------

/// A height field sampled over an equirectangular grid, in `0..=1`.
pub struct HeightField {
    pub width: u32,
    pub height: u32,
    pub values: Vec<f32>,
}

impl HeightField {
    /// Read a height field out of a greyscale-ish texture's red channel.
    ///
    /// This is the shape NASA's elevation products arrive in once decoded —
    /// LOLA for the Moon, GEBCO for Earth.
    pub fn from_texture(texture: &Texture) -> Self {
        let (w, h) = (texture.width.max(1), texture.height.max(1));
        let bpp = texture.bytes_per_pixel().max(1);
        let mut values = vec![0.0f32; (w as usize) * (h as usize)];
        for (i, v) in values.iter_mut().enumerate() {
            let o = i * bpp;
            *v = texture.data.get(o).copied().unwrap_or(0) as f32 / 255.0;
        }
        Self {
            width: w,
            height: h,
            values,
        }
    }

    fn at(&self, x: isize, y: isize) -> f32 {
        let w = self.width as isize;
        let h = self.height as isize;
        let xi = x.rem_euclid(w) as usize;
        let yi = y.clamp(0, h - 1) as usize;
        self.values[yi * self.width as usize + xi]
    }

    /// A tangent-space normal map, by Sobel over the height field.
    ///
    /// The x slope is divided by `cos(latitude)`: longitude compresses toward
    /// the poles, and without that correction the relief smears sideways there.
    pub fn to_normal_map(&self, strength: f32) -> Texture {
        let (w, h) = (self.width, self.height);
        let mut buf = MapBuffer::new(w, h);
        let hf = self;
        fill_rows(h, w as usize * 4, &mut buf.data, |y, row| {
            let v = (y as f32 + 0.5) / h as f32;
            let scale = 1.0 / ((v - 0.5) * std::f32::consts::PI).cos().max(0.2);
            for x in 0..w as usize {
                let (xi, yi) = (x as isize, y as isize);
                let dx = (hf.at(xi + 1, yi) - hf.at(xi - 1, yi)) * 0.5 * scale * strength;
                let dy = (hf.at(xi, yi + 1) - hf.at(xi, yi - 1)) * 0.5 * strength;
                let len = (dx * dx + dy * dy + 1.0).sqrt();
                let o = x * 4;
                row[o] = ((-dx / len * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
                row[o + 1] = ((-dy / len * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
                row[o + 2] = ((1.0 / len * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
                row[o + 3] = 255;
            }
        });
        buf.into_texture(TextureFormat::Rgba8Unorm)
    }

    /// The field as a height map, clamped so nothing below `sea_level` dips.
    ///
    /// Bathymetry is real relief, but a displaced globe wants a flat sea and
    /// mountains above it, not trenches carved out of the sphere.
    pub fn to_height_map(&self, sea_level: f32) -> Texture {
        let (w, h) = (self.width, self.height);
        let mut buf = MapBuffer::new(w, h);
        let hf = self;
        let span = (1.0 - sea_level).max(1e-4);
        fill_rows(h, w as usize * 4, &mut buf.data, |y, row| {
            for x in 0..w as usize {
                let e = hf.values[y as usize * w as usize + x];
                let v = (((e - sea_level) / span).clamp(0.0, 1.0) * 255.0) as u8;
                let o = x * 4;
                row[o] = v;
                row[o + 1] = v;
                row[o + 2] = v;
                row[o + 3] = 255;
            }
        });
        buf.into_texture(TextureFormat::Rgba8Unorm)
    }

    /// The lowest and highest samples in the field.
    pub fn range(&self) -> (f32, f32) {
        self.values
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)))
    }

    /// The field as a height map stretched to the full `0..=1` range.
    ///
    /// Products rarely fill their encoding: NASA's LOLA map spans 7..161 of 255
    /// with a mean at 74, so used as-is it wastes two thirds of the precision
    /// *and* adds a constant 29% swell to whatever it displaces. Stretching
    /// recovers both. Pair it with a `displacement_bias` of `-scale / 2` to
    /// keep the body's mean radius where it was.
    pub fn to_height_map_normalized(&self) -> Texture {
        let (lo, hi) = self.range();
        let span = (hi - lo).max(1e-6);
        let (w, h) = (self.width, self.height);
        let mut buf = MapBuffer::new(w, h);
        let hf = self;
        fill_rows(h, w as usize * 4, &mut buf.data, |y, row| {
            for x in 0..w as usize {
                let e = hf.values[y as usize * w as usize + x];
                let v = (((e - lo) / span).clamp(0.0, 1.0) * 255.0) as u8;
                let o = x * 4;
                row[o] = v;
                row[o + 1] = v;
                row[o + 2] = v;
                row[o + 3] = 255;
            }
        });
        buf.into_texture(TextureFormat::Rgba8Unorm)
    }

    /// A roughness map that splits water from land at `sea_level`.
    ///
    /// Blue Marble ships no specular map, but the bathymetry in the elevation
    /// product says where the water is, and water is smooth.
    pub fn to_roughness_map(&self, sea_level: f32, water: f32, land: f32) -> Texture {
        let (w, h) = (self.width, self.height);
        let mut buf = MapBuffer::new(w, h);
        let hf = self;
        fill_rows(h, w as usize * 4, &mut buf.data, |y, row| {
            for x in 0..w as usize {
                let e = hf.values[y as usize * w as usize + x];
                let r = if e <= sea_level { water } else { land };
                let v = (r.clamp(0.0, 1.0) * 255.0) as u8;
                let o = x * 4;
                row[o] = v;
                row[o + 1] = v;
                row[o + 2] = v;
                row[o + 3] = 255;
            }
        });
        buf.into_texture(TextureFormat::Rgba8Unorm)
    }

    /// As [`to_roughness_map`](Self::to_roughness_map), with snow and ice read
    /// off the surface colour and given a roughness of their own.
    ///
    /// A land/sea split is one class too few. Ice is neither: it is far
    /// smoother than rock and far rougher than open water, and a fresh snow
    /// surface has a real sheen at a low sun — which is most of the light the
    /// poles ever get. Left as land, the entire Arctic and Antarctic shade like
    /// dry ground and lose it.
    ///
    /// Detected from the albedo rather than from latitude, so it follows the
    /// actual snow line: Greenland's cap and the Himalaya come out as ice while
    /// the tundra beside them does not. The test is bright *and* neutral, which
    /// is what separates snow from pale desert — the Sahara is as bright but
    /// strongly yellow.
    pub fn to_roughness_map_with_ice(
        &self,
        albedo: &Texture,
        sea_level: f32,
        water: f32,
        land: f32,
        ice: f32,
    ) -> Texture {
        let (w, h) = (self.width, self.height);
        let (aw, ah) = (albedo.width.max(1), albedo.height.max(1));
        let abpp = albedo.bytes_per_pixel().max(1);
        if albedo.data.len() < (aw as usize) * (ah as usize) * abpp {
            return self.to_roughness_map(sea_level, water, land);
        }
        let mut buf = MapBuffer::new(w, h);
        let hf = self;
        fill_rows(h, w as usize * 4, &mut buf.data, |y, row| {
            // Nearest sample: the albedo map need not match the height map's
            // size, and this is a classification, not a filter.
            let ay = ((y as usize) * ah as usize / h.max(1) as usize).min(ah as usize - 1);
            for x in 0..w as usize {
                let ax = (x * aw as usize / w.max(1) as usize).min(aw as usize - 1);
                let ao = (ay * aw as usize + ax) * abpp;
                let (r, g, b) = (
                    albedo.data[ao] as f32 / 255.0,
                    albedo.data[ao + 1] as f32 / 255.0,
                    albedo.data[ao + 2] as f32 / 255.0,
                );
                let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                let chroma = (r - b).abs().max((g - b).abs()).max((r - g).abs());
                let e = hf.values[y as usize * w as usize + x];
                let rough = if e <= sea_level {
                    water
                } else if luma > 0.62 && chroma < 0.10 {
                    ice
                } else {
                    land
                };
                let v = (rough.clamp(0.0, 1.0) * 255.0) as u8;
                let o = x * 4;
                row[o] = v;
                row[o + 1] = v;
                row[o + 2] = v;
                row[o + 3] = 255;
            }
        });
        buf.into_texture(TextureFormat::Rgba8Unorm)
    }
}

// ---------------------------------------------------------------------------
// Procedural Earth
// ---------------------------------------------------------------------------

/// Elevation in `-1..=1` at a point on the sphere. Below 0 is ocean.
fn elevation(p: [f32; 3]) -> f32 {
    // Eight octaves from 1.5 reaches ~190 cycles around the globe, which is
    // what gives coastlines something to be other than smooth blobs.
    let continents = fbm(p, 1.5, 8, 11);
    let warp = fbm(p, 4.3, 4, 71) - 0.5;
    let mass = continents + warp * 0.22;
    // Sea level picked from the noise's own distribution to land near Earth's
    // 29% land fraction.
    let land = mass - 0.545;
    if land <= 0.0 {
        let floor = fbm(p, 6.0, 4, 401);
        return (land * 2.4 - floor * 0.10).max(-1.0);
    }
    let mountains = ridged(p, 7.5, 5, 907);
    let relief = fbm(p, 18.0, 4, 55);
    ((land * 3.0) + mountains * mountains * 0.6 * (land * 7.0).min(1.0) + relief * 0.07).min(1.0)
}

/// Splice another equirectangular map into a latitude cap, feathered.
///
/// This exists because Blue Marble Next Generation has no imagery over the
/// Arctic. It is built from visible-light MODIS passes, and the region inside
/// the Arctic Circle spends months in polar night, so every monthly mosaic pads
/// the top of the map with the same near-black fill — identical bytes in the
/// December and July releases, down to about 82.6°N. Rendered on a globe that
/// is a black cap ringed by a feathered fringe where the fill meets real
/// coastline, which is what the Arctic "pinwheel" actually was. No amount of
/// polar filtering fixes it, because there is nothing there to filter.
///
/// `cap` supplies the missing region — the 2002 Blue Marble's
/// `land_ocean_ice` composite does, at full brightness right over the pole.
/// It is sampled bilinearly, so it need not match `base`'s resolution, and the
/// two are mixed in linear light over `from_lat`..`to_lat` (degrees, positive
/// north) so the join has no step in it. Latitudes beyond `to_lat` are the cap
/// alone; short of `from_lat`, `base` is untouched.
///
/// Negative latitudes address the south cap. Antarctica needs no such help:
/// it is land, it is imaged, and it renders correctly as it is.
pub fn blend_polar_cap(base: &mut Texture, cap: &Texture, from_lat: f32, to_lat: f32) {
    blend_polar_cap_in(base, (-180.0, 180.0, -90.0, 90.0), cap, from_lat, to_lat)
}

/// [`blend_polar_cap`] for a base that covers a WINDOW rather than the globe.
///
/// A planet split into tiles needs the same Arctic fix as a single map, but a
/// tile's rows do not run from 90 N to 90 S — its own bounds decide what
/// latitude each row is, and the cap is still addressed in world coordinates.
/// Without this a tiled globe reproduces exactly the defect the cap exists to
/// remove, because the source tiles it is cut from are the ones with the hole
/// in them.
///
/// `bounds` is `(west, east, south, north)` in degrees.
pub fn blend_polar_cap_in(
    base: &mut Texture,
    bounds: (f32, f32, f32, f32),
    cap: &Texture,
    from_lat: f32,
    to_lat: f32,
) {
    let (bw_deg, be_deg, bs_deg, bn_deg) = bounds;
    let (w, h) = (base.width as usize, base.height as usize);
    let (cw, ch) = (cap.width as usize, cap.height as usize);
    let bpp = base.bytes_per_pixel();
    let cbpp = cap.bytes_per_pixel();
    if w < 2 || h < 2 || cw < 2 || ch < 2 || bpp < 3 || cbpp < 3 {
        return;
    }
    if base.data.len() != w * h * bpp || cap.data.len() != cw * ch * cbpp {
        return;
    }
    // sRGB in, sRGB out; the mix itself has to happen in linear light or a
    // half-and-half blend of white ice and dark sea comes out too dark.
    let to_linear = |b: u8| {
        let v = b as f32 / 255.0;
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    let to_srgb = |v: f32| {
        let v = v.clamp(0.0, 1.0);
        let e = if v <= 0.0031308 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        };
        (e * 255.0).round() as u8
    };
    if (to_lat - from_lat).abs() < 1e-6 {
        return;
    }
    let data = Arc::make_mut(&mut base.data);
    for y in 0..h {
        let lat = bn_deg - (y as f32 + 0.5) / h as f32 * (bn_deg - bs_deg);
        // Signed, so the same expression serves both caps: a south band is
        // simply `from_lat`/`to_lat` both negative and descending.
        let t = ((lat - from_lat) / (to_lat - from_lat)).clamp(0.0, 1.0);
        let mix = t * t * (3.0 - 2.0 * t);
        if mix <= 0.0 {
            continue;
        }
        // Same latitude, same longitude — only the row spacing differs, so the
        // cap is resampled in v and in u independently.
        // The cap is a whole-globe map, so the row is found from the WORLD
        // latitude rather than from this base's own row index.
        let cy = (((90.0 - lat) / 180.0) * ch as f32 - 0.5).clamp(0.0, ch as f32 - 1.0);
        let (y0, fy) = (cy.floor() as usize, cy - cy.floor());
        let y1 = (y0 + 1).min(ch - 1);
        for x in 0..w {
            let lon = bw_deg + (x as f32 + 0.5) / w as f32 * (be_deg - bw_deg);
            let cx = (((lon + 180.0) / 360.0) * cw as f32 - 0.5).rem_euclid(cw as f32);
            let (x0, fx) = (cx.floor() as usize % cw, cx - cx.floor());
            let x1 = (x0 + 1) % cw;
            let o = (y * w + x) * bpp;
            for c in 0..3 {
                let s = |xi: usize, yi: usize| to_linear(cap.data[(yi * cw + xi) * cbpp + c]);
                let top = s(x0, y0) * (1.0 - fx) + s(x1, y0) * fx;
                let bot = s(x0, y1) * (1.0 - fx) + s(x1, y1) * fx;
                let capv = top * (1.0 - fy) + bot * fy;
                let basev = to_linear(data[o + c]);
                data[o + c] = to_srgb(basev + (capv - basev) * mix);
            }
        }
    }
}

/// Flatten the last few rows of an equirectangular map to their zonal mean.
///
/// A UV sphere's pole is a fan of triangles whose apex vertices are all at the
/// same point but carry different `u`. Anything that reads the map *per vertex*
/// — displacement, above all — therefore gets a different answer at each of
/// them, and the pole tears open into a star. Making the top and bottom rows
/// constant along longitude is what closes it, and a constant is the one thing
/// that survives the pole's convergence unchanged.
///
/// This used to also blur along longitude, by up to 24 texels, on the theory
/// that the map is oversampled by `1/cos(latitude)` near the pole and that the
/// surplus gets stretched back out radially as a fan of streaks. It does not:
/// `tests/polar_sampling.rs` renders a map that varies only with latitude
/// straight down the pole and finds it rotationally symmetric to 0.05/255,
/// through both the texture fetch and the derivative-built tangent frame. The
/// sampler handles the convergence on its own.
///
/// What the blur did do was smear features *around* the pole, which is a
/// vortex — visible as a spiral in the cloud deck over the Arctic, and the
/// reason the ice cap came out as a featureless disc: flattening ran from 80°
/// and washed ten degrees of real imagery into its own average. Both are gone
/// with it. The band is now the couple of rows that actually feed the pole
/// vertices, and nothing below them is touched.
pub fn ease_poles(texture: &mut Texture) {
    let (w, h) = (texture.width as usize, texture.height as usize);
    let bpp = texture.bytes_per_pixel();
    if w < 4 || h < 4 || texture.data.len() != w * h * bpp {
        return;
    }
    let data = Arc::make_mut(&mut texture.data);
    for y in 0..h {
        let lat = (0.5 - (y as f32 + 0.5) / h as f32) * std::f32::consts::PI;
        let cos = lat.cos().abs().max(1e-6);
        let over = 1.0 / cos;
        // 30x oversampling is 88.1° of latitude, 90x is 89.4°. Bilinear
        // sampling reaches a row either side of the pole row, so the ramp
        // covers a few rows rather than snapping on at the last one.
        let t = ((over - 30.0) / 60.0).clamp(0.0, 1.0);
        let flat = t * t * (3.0 - 2.0 * t);
        if flat <= 0.001 {
            continue;
        }
        let base = y * w * bpp;
        for c in 0..bpp {
            let mean: u32 = (0..w).map(|x| data[base + x * bpp + c] as u32).sum::<u32>() / w as u32;
            for x in 0..w {
                let v = data[base + x * bpp + c] as f32;
                data[base + x * bpp + c] = (v + (mean as f32 - v) * flat) as u8;
            }
        }
    }
}

/// The maps a [`Planet`](super::Planet) draws with.
///
/// Every field is optional: a planet with only an albedo map still renders.
#[derive(Clone, Default)]
pub struct PlanetMaps {
    /// Surface colour. sRGB.
    pub albedo: Option<Arc<Texture>>,
    /// Tangent-space relief. Linear.
    pub normal: Option<Arc<Texture>>,
    /// Per-texel roughness, red channel. Linear.
    pub roughness: Option<Arc<Texture>>,
    /// Emissive night side — city lights. sRGB.
    pub night: Option<Arc<Texture>>,
    /// Cloud shell: white with per-texel alpha. sRGB.
    pub clouds: Option<Arc<Texture>>,
    /// Height map, red channel, sampled in the vertex stage. Linear.
    ///
    /// Not part of [`is_complete`](Self::is_complete): displacement is opt-in,
    /// and on a whole-globe view Everest is 0.14% of the radius — under a
    /// pixel. It reads only when exaggerated, which is the caller's choice.
    pub displacement: Option<Arc<Texture>>,
}

impl std::fmt::Debug for PlanetMaps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let size = |t: &Option<Arc<Texture>>| {
            t.as_ref()
                .map(|t| format!("{}x{}", t.width, t.height))
                .unwrap_or_else(|| "-".into())
        };
        f.debug_struct("PlanetMaps")
            .field("albedo", &size(&self.albedo))
            .field("normal", &size(&self.normal))
            .field("roughness", &size(&self.roughness))
            .field("night", &size(&self.night))
            .field("clouds", &size(&self.clouds))
            .field("displacement", &size(&self.displacement))
            .finish()
    }
}

impl PlanetMaps {
    /// Empty — the planet falls back to a flat material colour.
    pub fn new() -> Self {
        Self::default()
    }

    /// Surface colour map.
    pub fn albedo(mut self, texture: Texture) -> Self {
        self.albedo = Some(Arc::new(texture));
        self
    }
    /// Tangent-space normal map.
    pub fn normal(mut self, texture: Texture) -> Self {
        self.normal = Some(Arc::new(texture));
        self
    }
    /// Per-texel roughness.
    pub fn roughness(mut self, texture: Texture) -> Self {
        self.roughness = Some(Arc::new(texture));
        self
    }
    /// Emissive night side.
    pub fn night(mut self, texture: Texture) -> Self {
        self.night = Some(Arc::new(texture));
        self
    }
    /// Cloud shell, alpha-shaped.
    pub fn clouds(mut self, texture: Texture) -> Self {
        self.clouds = Some(Arc::new(texture));
        self
    }
    /// Height map for vertex displacement.
    pub fn displacement(mut self, texture: Texture) -> Self {
        self.displacement = Some(Arc::new(texture));
        self
    }

    /// Whether anything is set.
    pub fn is_empty(&self) -> bool {
        self.slots().iter().all(|s| s.is_none())
    }

    /// Whether every slot is filled.
    pub fn is_complete(&self) -> bool {
        self.slots().iter().all(|s| s.is_some())
    }

    fn slots(&self) -> [&Option<Arc<Texture>>; 5] {
        [
            &self.albedo,
            &self.normal,
            &self.roughness,
            &self.night,
            &self.clouds,
        ]
    }
}

/// Build a full procedural Earth map set at `width × width/2`.
///
/// Every map comes from one shared elevation field, which is why the coastlines
/// in the colour map, the relief in the normal map, the shine on the oceans and
/// the shoreline glow of the city lights all agree with each other.
///
/// ```
/// use threers::planet::generate_earth_maps;
/// let maps = generate_earth_maps(256);
/// assert_eq!(maps.albedo.as_ref().unwrap().width, 256);
/// assert_eq!(maps.albedo.as_ref().unwrap().height, 128);
/// assert!(maps.night.is_some() && maps.clouds.is_some());
/// ```
pub fn generate_earth_maps(width: u32) -> PlanetMaps {
    use std::f32::consts::PI;
    let width = width.clamp(16, 8192);
    let height = (width / 2).max(8);
    let (w, h) = (width as usize, height as usize);

    // ---- the shared elevation field ----
    let mut field = vec![0f32; w * h];
    fill_rows(height, w, &mut field, |y, row| {
        let v = (y as f32 + 0.5) / h as f32;
        for (x, cell) in row.iter_mut().enumerate() {
            *cell = elevation(direction((x as f32 + 0.5) / w as f32, v));
        }
    });
    let hf = HeightField {
        width,
        height,
        // Remap to 0..1 so the shared derivation helpers can use it.
        values: field.iter().map(|e| e * 0.5 + 0.5).collect(),
    };

    // ---- albedo ----
    let mut albedo = MapBuffer::new(width, height);
    fill_rows(height, w * 4, &mut albedo.data, |y, row| {
        let v = (y as f32 + 0.5) / h as f32;
        let lat = (v - 0.5) * PI;
        let polar_base = (lat.abs() - 1.02) / 0.34;
        for x in 0..w {
            let u = (x as f32 + 0.5) / w as f32;
            // `direction` takes *data* v — row 0 first — while the sphere's
            // uv is what the sampler sees, and `Texture::flip_y` (on by
            // default, and set by `MapBuffer::into_texture`) turns one into the
            // other on upload. So the comparison has to undo that flip.
            let d = direction(u, 1.0 - v);
            let e = field[y as usize * w + x];
            // Ragged ice edge rather than a latitude line.
            let polar = (polar_base + (fbm(d, 14.0, 3, 4242) - 0.5) * 0.55).clamp(0.0, 1.0);
            let mut rgb = if e <= 0.0 {
                let depth = (-e / 0.6).clamp(0.0, 1.0);
                [
                    0.05 + (0.008 - 0.05) * depth,
                    0.30 + (0.035 - 0.30) * depth,
                    0.36 + (0.13 - 0.36) * depth,
                ]
            } else {
                let jitter = fbm(d, 9.0, 3, 1234);
                let warmth = (1.0 - lat.abs() / 1.4).clamp(0.0, 1.0) + (jitter - 0.5) * 0.28;
                let belt = (1.0 - (lat.abs() - 0.46).abs() / 0.30).clamp(0.0, 1.0);
                let dryness = fbm(d, 5.5, 4, 6060);
                let desert = (belt * 1.25 * (dryness - 0.34) * 3.0).clamp(0.0, 1.0);
                let base = [
                    0.34 + (0.06 - 0.34) * warmth,
                    0.33 + (0.24 - 0.33) * warmth,
                    0.28 + (0.07 - 0.28) * warmth,
                ];
                let green = [
                    base[0] * 0.5 + 0.045,
                    base[1] * 0.5 + 0.100,
                    base[2] * 0.5 + 0.040,
                ];
                let land = [
                    green[0] + (0.62 - green[0]) * desert,
                    green[1] + (0.51 - green[1]) * desert,
                    green[2] + (0.31 - green[2]) * desert,
                ];
                // The snow line falls as you leave the tropics.
                let snow_line = 0.78 - 0.62 * (lat.abs() / 1.5).powi(2);
                let alpine = ((e - snow_line) / 0.14).clamp(0.0, 1.0);
                let snow = polar.max(alpine);
                [
                    land[0] + (0.92 - land[0]) * snow,
                    land[1] + (0.94 - land[1]) * snow,
                    land[2] + (0.97 - land[2]) * snow,
                ]
            };
            if e <= 0.0 && polar > 0.35 {
                let t = ((polar - 0.35) / 0.4).clamp(0.0, 1.0);
                rgb = [
                    rgb[0] + (0.88 - rgb[0]) * t,
                    rgb[1] + (0.91 - rgb[1]) * t,
                    rgb[2] + (0.95 - rgb[2]) * t,
                ];
            }
            let o = x * 4;
            row[o] = (rgb[0].clamp(0.0, 1.0) * 255.0) as u8;
            row[o + 1] = (rgb[1].clamp(0.0, 1.0) * 255.0) as u8;
            row[o + 2] = (rgb[2].clamp(0.0, 1.0) * 255.0) as u8;
            row[o + 3] = 255;
        }
    });

    // ---- night lights ----
    let mut night = MapBuffer::new(width, height);
    fill_rows(height, w * 4, &mut night.data, |y, row| {
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
            let d = direction((x as f32 + 0.5) / w as f32, v);
            // A low-frequency "where people are" field times a high-frequency
            // "individual towns" one.
            let region = fbm(d, 7.0, 4, 5150);
            let towns = fbm(d, 44.0, 3, 8191);
            let lowland = 1.0 - (e / 0.8).clamp(0.0, 1.0);
            // Both thresholds are set from fbm's measured spread — four
            // octaves of averaged value noise lands almost entirely inside
            // 0.35..0.65, not 0..1. Calibrating against the nominal range
            // instead leaves the whole night side at a tenth brightness.
            let mut lit = ((region - 0.55) / 0.15).clamp(0.0, 1.0) * lowland * habitable;
            lit *= ((towns - 0.42) / 0.20).clamp(0.0, 1.0);
            let lit = lit.powf(1.1);
            // Sodium orange, with the brightest cores going white.
            let core = (lit - 0.55).max(0.0) / 0.45;
            row[o] = (lit * 255.0) as u8;
            row[o + 1] = (lit * (0.78 + 0.22 * core) * 255.0) as u8;
            row[o + 2] = (lit * (0.43 + 0.57 * core) * 255.0) as u8;
            row[o + 3] = 255;
        }
    });

    // ---- clouds ----
    let mut clouds = MapBuffer::new(width, height);
    fill_rows(height, w * 4, &mut clouds.data, |y, row| {
        let v = (y as f32 + 0.5) / h as f32;
        let lat = (v - 0.5) * PI;
        // Wet at the equator and the polar fronts, dry over the horse
        // latitudes — which is also where the deserts are.
        let band = 0.55 + 0.45 * (lat * 6.0).cos() * (1.0 - lat.abs() / 1.7).max(0.0);
        for x in 0..w {
            let d = direction((x as f32 + 0.5) / w as f32, v);
            // Stretched along longitude: weather comes in streaks, not blobs.
            let n = fbm([d[0], d[1] * 2.6, d[2]], 6.0, 6, 31337);
            let a = ((n * band - 0.42) * 3.4).clamp(0.0, 1.0);
            let o = x * 4;
            row[o] = 255;
            row[o + 1] = 255;
            row[o + 2] = 255;
            row[o + 3] = (a * 255.0) as u8;
        }
    });

    let ease = |mut t: Texture| {
        ease_poles(&mut t);
        Arc::new(t)
    };
    PlanetMaps {
        albedo: Some(ease(albedo.into_texture(TextureFormat::Rgba8UnormSrgb))),
        normal: Some(ease(hf.to_normal_map(3.2))),
        roughness: Some(ease(hf.to_roughness_map(0.5, 0.14, 0.86))),
        night: Some(ease(night.into_texture(TextureFormat::Rgba8UnormSrgb))),
        clouds: Some(ease(clouds.into_texture(TextureFormat::Rgba8UnormSrgb))),
        // The same field the other maps came from, as a height map. Land only:
        // the ocean floor is not what a displaced globe should show.
        displacement: Some(Arc::new(hf.to_height_map(0.5))),
    }
}

/// The six faces of a cube, in the order [`CUBE_FACES`] lists them.
///
/// A cube has no poles. That is the whole point of resampling onto one: an
/// equirectangular map converges every longitude onto a single texel at each
/// end, and *any* structure in the map near there — a star's footprint, a
/// filtering kernel, an interpolation between rows — gets fanned out radially
/// when it is wrapped onto a sphere. A cube face is a plane. Its texels are
/// near enough the same size everywhere, in every direction, and no point on it
/// is special, so there is nothing for a fan to converge on.
///
/// `face_size` should be about a quarter of the source's width: four faces
/// carry the 360 degrees that the equirectangular map spends its full width on,
/// so a 8192-wide source resamples losslessly at the equator into 2048 faces —
/// and *gains* resolution toward the poles, where the source was wasting texels.
pub fn equirect_to_cube_faces(src: &Texture, face_size: u32) -> Vec<Texture> {
    let n = face_size.clamp(4, 8192);
    let (sw, sh) = (src.width.max(1) as usize, src.height.max(1) as usize);
    let bpp = src.bytes_per_pixel();
    let half = matches!(src.format, TextureFormat::Rgba16Float);
    let flip = src.flip_y;
    if src.data.len() < sw * sh * bpp || bpp < 3 {
        return Vec::new();
    }

    // One channel of one texel, as a float, whatever the storage.
    let texel = |x: usize, y: usize, c: usize| -> f32 {
        let o = (y * sw + x) * bpp;
        if half {
            crate::renderer::gpu_texture::f16_bits_to_f32(u16::from_le_bytes([
                src.data[o + c * 2],
                src.data[o + c * 2 + 1],
            ]))
        } else {
            src.data[o + c] as f32 / 255.0
        }
    };

    // Bilinear, wrapping in longitude and clamping in latitude — the same rule
    // the sampler uses on the sphere, so the two agree where they overlap.
    let sample = |d: [f32; 3]| -> [f32; 4] {
        let lat = d[1].clamp(-1.0, 1.0).asin();
        // `SphereGeometry` lays the map out with `x = -cos(phi) sin(theta)`, so
        // the direction it shows texel `u` at is `pi - u * 2pi`, not
        // `(u - 0.5) * 2pi` — longitude runs the other way round. Inverting the
        // wrong one of those mirrors the whole sky east-for-west, which leaves
        // every statistic about it unchanged and is invisible until you compare
        // the two side by side.
        let u = (std::f32::consts::PI - d[2].atan2(d[0])) / std::f32::consts::TAU;
        let u = u - u.floor();
        // Which latitude row 0 holds is decided by `flip_y`, because that is
        // what the uploader will do to it. With the flip on, the sampler sees
        // the last row at v = 0, and the sphere puts the north pole at v = 1 —
        // so row 0 is north. With it off the rows go up verbatim and row 0 is
        // south, which is where a `DataTexture` the caller already flipped by
        // hand lands. Reading the wrong one turns the sky upside down.
        let v = if flip {
            0.5 - lat / std::f32::consts::PI
        } else {
            0.5 + lat / std::f32::consts::PI
        };
        let fx = (u * sw as f32 - 0.5).rem_euclid(sw as f32);
        let fy = (v * sh as f32 - 0.5).clamp(0.0, sh as f32 - 1.0);
        let (x0, y0) = (fx.floor(), fy.floor());
        let (tx, ty) = (fx - x0, fy - y0);
        let (x0, y0) = (x0 as usize % sw, y0 as usize);
        let x1 = (x0 + 1) % sw;
        let y1 = (y0 + 1).min(sh - 1);
        let mut out = [0.0f32; 4];
        for (c, o) in out.iter_mut().enumerate().take(bpp.min(4)) {
            let top = texel(x0, y0, c) * (1.0 - tx) + texel(x1, y0, c) * tx;
            let bot = texel(x0, y1, c) * (1.0 - tx) + texel(x1, y1, c) * tx;
            *o = top * (1.0 - ty) + bot * ty;
        }
        if bpp < 4 {
            out[3] = 1.0;
        }
        out
    };

    CUBE_FACES
        .iter()
        .map(|(f, r, up)| {
            let mut data = vec![0u8; (n as usize) * (n as usize) * bpp];
            fill_rows(n, n as usize * bpp, &mut data, |j, row| {
                // `flip_y` is off on the result, so row j sits at v = (j+0.5)/n
                // and the face's y axis runs the same way as its v axis.
                let _y = 2.0 * (j as f32 + 0.5) / n as f32 - 1.0;
                for i in 0..n as usize {
                    // SUPERSAMPLED, because one bilinear tap cannot cover this
                    // texel's footprint.
                    //
                    // A cube face does not sample the source evenly. At the
                    // face centre a texel spans 90/n degrees; at the edge the
                    // tangent mapping compresses it to half that. Against a
                    // 16384-wide equirect a 4096 face is downsampling 1.27:1 in
                    // the middle and upsampling 0.64:1 at the rim, and the
                    // sweep through 1:1 is a beat — moire, which on a smooth
                    // gradient like the Milky Way reads as fine regular stripes
                    // across the glow and nowhere else.
                    //
                    // 2x2 over the texel is enough for a ratio this close to
                    // one. It is also 4x the work of a resample that happens
                    // once at startup and never again.
                    const SS: usize = 2;
                    let mut acc = [0.0f32; 4];
                    for sy in 0..SS {
                        for sx in 0..SS {
                            let ox = (sx as f32 + 0.5) / SS as f32 - 0.5;
                            let oy = (sy as f32 + 0.5) / SS as f32 - 0.5;
                            let x = 2.0 * (i as f32 + 0.5 + ox) / n as f32 - 1.0;
                            let yy = 2.0 * (j as f32 + 0.5 + oy) / n as f32 - 1.0;
                            let d = [
                                f[0] + x * r[0] + yy * up[0],
                                f[1] + x * r[1] + yy * up[1],
                                f[2] + x * r[2] + yy * up[2],
                            ];
                            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-8);
                            let t = sample([d[0] / len, d[1] / len, d[2] / len]);
                            for c in 0..4 {
                                acc[c] += t[c];
                            }
                        }
                    }
                    let inv = 1.0 / (SS * SS) as f32;
                    let s = [acc[0] * inv, acc[1] * inv, acc[2] * inv, acc[3] * inv];
                    let o = i * bpp;
                    for c in 0..bpp.min(4) {
                        if half {
                            let h = crate::renderer::gpu_texture::f32_to_f16_bits(s[c]);
                            row[o + c * 2..o + c * 2 + 2].copy_from_slice(&h.to_le_bytes());
                        } else {
                            row[o + c] = (s[c].clamp(0.0, 1.0) * 255.0).round() as u8;
                        }
                    }
                }
            });
            let mut t = Texture::new(n, n, src.format, data);
            // Clamped on both axes: a face's edge is a real edge, and wrapping
            // it would fetch the opposite side of the sky.
            t.wrap_s = TextureWrap::ClampToEdge;
            t.wrap_t = TextureWrap::ClampToEdge;
            // Rows are written in face space, first row at v = 0, which is
            // what the quad's UVs expect — so never flipped on upload,
            // whatever the source was.
            t.flip_y = false;
            t
        })
        .collect()
}

/// Face bases as `(forward, right, up)`, all unit and mutually perpendicular.
///
/// Which way round each face runs does not matter to anything outside this
/// module, because the same basis generates the texture *and* places the quad —
/// they only have to agree with each other.
pub const CUBE_FACES: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
    ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
    ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
    ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
    ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
];

/// Linear sRGB for a blackbody at `kelvin`, normalised to unit luminance.
///
/// Star colour is temperature and nothing else, so it belongs on the hue and
/// not on the brightness — an M dwarf is not a dim white star, it is an orange
/// one, and how bright it looks is the magnitude's business. Normalising here
/// keeps the two separable.
///
/// The Planckian locus comes from Kim et al.'s cubic fit, good from 1667 K to
/// 25000 K, which covers everything from an M dwarf to a hot B star.
// The coefficients are quoted at the precision they were published at, which is
// more than f32 keeps. Rounding them to fit would make the source no longer
// match the paper for no gain, since the compiler does the rounding anyway.
#[allow(clippy::excessive_precision)]
pub fn blackbody_rgb(kelvin: f32) -> [f32; 3] {
    let t = kelvin.clamp(1667.0, 25000.0);
    let (t1, t2, t3) = (1.0 / t, 1.0 / (t * t), 1.0 / (t * t * t));
    let x = if t <= 4000.0 {
        -0.2661239e9 * t3 - 0.2343589e6 * t2 + 0.8776956e3 * t1 + 0.179910
    } else {
        -3.0258469e9 * t3 + 2.1070379e6 * t2 + 0.2226347e3 * t1 + 0.240390
    };
    let (x2, x3) = (x * x, x * x * x);
    let y = if t <= 2222.0 {
        -1.1063814 * x3 - 1.34811020 * x2 + 2.18555832 * x - 0.20219683
    } else if t <= 4000.0 {
        -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867
    } else {
        3.0817580 * x3 - 5.87338670 * x2 + 3.75112997 * x - 0.37001483
    };
    // xyY with Y = 1 into XYZ, then the sRGB primaries.
    let y = y.max(1e-4);
    let (xx, zz) = (x / y, (1.0 - x - y) / y);
    let rgb = [
        3.2404542 * xx - 1.5371385 - 0.4985314 * zz,
        -0.9692660 * xx + 1.8760108 + 0.0415560 * zz,
        0.0556434 * xx - 0.2040259 + 1.0572252 * zz,
    ];
    // Clipping to the gamut costs some luminance, so renormalise after.
    let rgb = [rgb[0].max(0.0), rgb[1].max(0.0), rgb[2].max(0.0)];
    let luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    if luma <= 1e-6 {
        return [1.0, 1.0, 1.0];
    }
    [rgb[0] / luma, rgb[1] / luma, rgb[2] / luma]
}

/// Build a procedural starfield at `width × width/2`, in linear HDR.
///
/// Point sources are what makes a sky hard. A star's flux is fixed, so a camera
/// concentrates it into about one pixel however wide the field of view — which
/// means brightness has to live in the *value*, not in how many texels the star
/// covers. Three things follow from taking that seriously:
///
/// - **Half-float, not 8-bit.** Sirius is 1500 times a magnitude-6 star. Eight
///   bits cannot hold both, so the old map clamped the range to nothing and
///   every star came out the same brightness. Here the range is real, and the
///   tone map decides what clips.
/// - **One fixed point-spread function**, sub-texel, energy-conserving. Bright
///   stars then look bigger *for free*, because the same Gaussian's tail stays
///   above the visible threshold further out — which is exactly why they look
///   bigger through a real lens. The old map faked it by painting a plus sign
///   around anything bright, which read as a cross rather than a star. The
///   kernel is stretched by `1/cos(latitude)` so it stays round on the sphere
///   instead of becoming a radial streak near the poles.
/// - **Counts from `N(<m) ∝ 10^(0.6 m)`**, the distribution a uniform spread of
///   stars in space produces, with flux `10^(-0.4 m)` per magnitude. So the sky
///   has a few obvious stars, a lot of faint ones, and the right ratio between.
///
/// Colour is blackbody, sampled over the spectral classes in their rough
/// naked-eye proportions — see [`blackbody_rgb`].
///
/// ```
/// use threers::planet::generate_starfield;
/// let sky = generate_starfield(512, 4000);
/// assert_eq!((sky.width, sky.height), (512, 256));
/// ```
pub fn generate_starfield(width: u32, count: u32) -> Texture {
    let width = width.clamp(64, 16384);
    let height = (width / 2).max(32);
    let (w, h) = (width as usize, height as usize);
    let mut buf = vec![0f32; w * h * 4];

    galactic_band(w, h, &mut buf);
    scatter_stars(w, h, count, &mut buf);

    let data = buf
        .iter()
        .flat_map(|&v| crate::renderer::gpu_texture::f32_to_f16_bits(v).to_le_bytes())
        .collect::<Vec<u8>>();
    let mut t = Texture::new(width, height, TextureFormat::Rgba16Float, data);
    t.wrap_s = TextureWrap::Repeat;
    t.wrap_t = TextureWrap::ClampToEdge;
    t
}

/// The Milky Way: a glow concentrated on the galactic plane, brightest toward
/// the centre, cut by the dust that makes the Great Rift.
///
/// The band is not a uniform stripe. It thickens and brightens enormously
/// toward the galactic centre — that is the bulge, seen through 8 kpc of disc —
/// and the dark lane running down it is not a gap between stars but cold dust
/// in front of them, so it multiplies rather than subtracts.
fn galactic_band(w: usize, h: usize, buf: &mut [f32]) {
    // Galactic pole and centre, roughly, in the same frame `direction` uses.
    let pole = [-0.868, 0.198, 0.456];
    let centre = [-0.055, -0.874, -0.483];
    fill_rows(h as u32, w * 4, buf, |y, row| {
        let v = (y as f32 + 0.5) / h as f32;
        for x in 0..w {
            let d = direction((x as f32 + 0.5) / w as f32, v);
            // Galactic latitude, and how far round the plane from the centre.
            let sin_b = d[0] * pole[0] + d[1] * pole[1] + d[2] * pole[2];
            let toward = d[0] * centre[0] + d[1] * centre[1] + d[2] * centre[2];
            // The bulge is a fat lens, the outer disc a thin one. Clamped
            // because two unit vectors can dot to a hair past -1 in f32, and
            // `sqrt` of that is a NaN that spreads through the whole map.
            let bulge = (((toward + 1.0) * 0.5).clamp(0.0, 1.0)).powi(3);
            let thickness = 0.055 + 0.16 * bulge;
            let glow = (-(sin_b / thickness).powi(2)).exp() * (0.05 + 0.95 * bulge.sqrt());
            // Opaque everywhere, including where there is no light: empty sky
            // is *black*, not see-through, and a transparent sky sphere shows
            // whatever the scene's background happens to be.
            row[x * 4 + 3] = 1.0;
            if glow <= 1e-4 {
                continue;
            }
            // Unresolved stars, clumped; the disc is not smooth.
            //
            // TWO SCALES, not one. A single fbm at 14 puts the smallest feature
            // at about a fourteenth of the sky, and a camera with a 40 degree
            // field magnifying a 16k map sees those as soft blobs -- fog with
            // stars on it, rather than a band with structure in it. The second
            // term is four times finer and weighted low, so it adds grain
            // without breaking up the shape the first one gives.
            let clumps = 0.45 + 0.40 * fbm(d, 14.0, 5, 771) + 0.15 * fbm(d, 58.0, 3, 1097);
            // Dust sits in the plane, so it only bites where the glow is.
            let lane = (-(sin_b / (thickness * 0.55)).powi(2)).exp();
            // Same argument for the dust, and it matters more here: the Great
            // Rift reads as a lane because it has edges, and edges are exactly
            // what the low frequency alone cannot make.
            let coarse = fbm(d, 26.0, 4, 313);
            let fine = fbm(d, 88.0, 3, 641);
            let dust = ((coarse * 0.72 + fine * 0.28) * 1.7 - 0.45).clamp(0.0, 1.0) * lane;
            // 0.018, down from 0.05. Measured off the opening shot, the old
            // scale put the band at rgb(102, 99, 93) against rgb(21, 22, 21)
            // for empty sky -- a 40 % grey wash across a third of frame, which
            // is not what the Milky Way looks like from anywhere, let alone
            // from orbit. The stars have to carry that frame; the glow is the
            // thing they sit in.
            let m = glow * clumps * (1.0 - 0.85 * dust) * 0.018;
            // Warm, but far less than it was. At 0.35/0.50 the warmth term
            // reached 1.85, which drove blue to 0.51 of green and made the core
            // sepia. The bulge IS reddened -- old stars, seen through 8 kpc of
            // dust -- but the disc either side of it is blue-white, and that
            // contrast is most of what makes the band read as a galaxy.
            let warmth = 1.0 + 0.18 * bulge + 0.25 * dust;
            let o = x * 4;
            row[o] = m * (0.74 * warmth).min(1.15);
            row[o + 1] = m * 0.82;
            row[o + 2] = m * (1.02 / warmth.max(1.0)).max(0.78);
            row[o + 3] = 1.0;
        }
    });
}

/// Scatter `count` stars, uniform on the sphere, and splat each one.
fn scatter_stars(w: usize, h: usize, count: u32, buf: &mut [f32]) {
    // A bare LCG will not do here. Its consecutive outputs lie on a lattice,
    // and five draws go into every star — position, magnitude, class — so that
    // structure turns into a correlation between where a star is and how bright
    // it is. The output mix below breaks it for the cost of a few shifts.
    let mut seed = 12_345u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
        let mut x = seed;
        x ^= x >> 16;
        x = x.wrapping_mul(2_246_822_519);
        x ^= x >> 13;
        x = x.wrapping_mul(3_266_489_917);
        x ^= x >> 16;
        (x >> 8) as f32 / 16_777_216.0
    };
    // Spectral classes in roughly the proportions the naked-eye sky shows —
    // which is far hotter than the true stellar population, because the cool
    // dwarfs that dominate it by number are all too faint to see.
    const CLASSES: [(f32, f32, f32); 6] = [
        (0.08, 12000.0, 18000.0), // B
        (0.22, 7800.0, 10000.0),  // A
        (0.20, 6200.0, 7500.0),   // F
        (0.18, 5300.0, 6000.0),   // G
        (0.22, 4000.0, 5200.0),   // K
        (0.10, 2900.0, 3900.0),   // M
    ];
    // Faintest and brightest rendered. Real limits: the naked eye reaches about
    // 6.5, Sirius is -1.46.
    const M_LIM: f32 = 8.0;
    const M_MIN: f32 = -1.5;
    // Flux scale, chosen so a magnitude-6 star sits just under 1.0 and anything
    // brighter runs into the headroom the half-float format is here for.
    let flux = |m: f32| 10f32.powf(-0.4 * (m - 6.0)) * 0.55;

    for _ in 0..count {
        let z = rnd() * 2.0 - 1.0;
        let lon = rnd() * std::f32::consts::TAU;
        // `N(<m) ∝ 10^(0.6 m)` inverted. Uniform space, uniform sky.
        let u = rnd().max(1e-6);
        let m = (M_LIM + u.log10() / 0.6).max(M_MIN);

        let pick = rnd();
        let mut acc = 0.0;
        let mut temp = 5800.0;
        for (share, lo, hi) in CLASSES {
            acc += share;
            if pick <= acc {
                temp = lo + (hi - lo) * rnd();
                break;
            }
        }
        let tint = blackbody_rgb(temp);
        let f = flux(m);

        let lat = z.asin();
        let sx = lon / std::f32::consts::TAU * w as f32;
        let sy = (0.5 - lat / std::f32::consts::PI) * h as f32;
        splat(
            w,
            h,
            buf,
            sx,
            sy,
            lat,
            [f * tint[0], f * tint[1], f * tint[2]],
        );
    }
}

/// Add one star's flux to the map, spread over a small Gaussian and normalised
/// so the total is `rgb` however the kernel lands.
///
/// Stretched along longitude by `1/cos(latitude)`, because a texel there covers
/// far less sky than one at the equator — without it a star near the pole comes
/// out as a radial streak, which is the pinwheel in miniature.
fn splat(w: usize, h: usize, buf: &mut [f32], sx: f32, sy: f32, lat: f32, rgb: [f32; 3]) {
    // Two thirds of a texel. Tight enough that a star still reads as a point
    // once the map is magnified, wide enough to be sampled properly: at half a
    // texel the grid's own variance is a quarter of the kernel's again, and no
    // correction below fully takes the resulting radial bias out. Measured over
    // a pole view, this lands the star's axis ratio at 1.10 against 1.13 for
    // NASA's own 8k map, which is the noise floor of the measurement.
    const SIGMA: f32 = 0.65;
    // Half the map's width covers every longitude, which is all a star at the
    // pole itself can need; past that the kernel would only wrap onto itself.
    let stretch = (1.0 / lat.cos().abs().max(1e-3)).min(w as f32 / 2.0);
    // Sampling a Gaussian on a unit grid adds 1/12 to its variance. At half a
    // texel that is a quarter of the variance again, and it lands on latitude
    // only: the longitude extent is `stretch` times wider and barely feels it.
    // Scaling sigma by `stretch` therefore leaves the star taller than it is
    // wide — measurably, an axis ratio of 1.24 aligned within 15 degrees of the
    // radial direction, which is a fan of streaks pointing at the pole. Solving
    // for the sigma whose *sampled* width is `stretch` times the sampled height
    // takes it out.
    const VAR_GRID: f32 = 1.0 / 12.0;
    let var_v = SIGMA * SIGMA + VAR_GRID;
    let sigma_u = (stretch * stretch * var_v - VAR_GRID)
        .max(SIGMA * SIGMA)
        .sqrt();
    let (rx, ry) = (((sigma_u * 2.5).ceil() as i32).max(1), 2i32);
    // Two passes: weights first, so the kernel is normalised to exactly one
    // even where it is clipped at the top or bottom row.
    let mut total = 0.0;
    for dy in -ry..=ry {
        let y = sy.floor() as i32 + dy;
        if y < 0 || y >= h as i32 {
            continue;
        }
        let fy = (y as f32 + 0.5 - sy) / SIGMA;
        for dx in -rx..=rx {
            let fx = (sx.floor() as i32 + dx) as f32 + 0.5 - sx;
            let fx = fx / sigma_u;
            total += (-0.5 * (fx * fx + fy * fy)).exp();
        }
    }
    if total <= 1e-9 {
        return;
    }
    for dy in -ry..=ry {
        let y = sy.floor() as i32 + dy;
        if y < 0 || y >= h as i32 {
            continue;
        }
        let fy = (y as f32 + 0.5 - sy) / SIGMA;
        for dx in -rx..=rx {
            let xi = sx.floor() as i32 + dx;
            let fx = (xi as f32 + 0.5 - sx) / sigma_u;
            let weight = (-0.5 * (fx * fx + fy * fy)).exp() / total;
            let x = xi.rem_euclid(w as i32) as usize;
            let o = (y as usize * w + x) * 4;
            for c in 0..3 {
                buf[o + c] += rgb[c] * weight;
            }
            buf[o + 3] = 1.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A solid-colour equirectangular map.
    fn flat(w: u32, h: u32, rgb: [u8; 3]) -> Texture {
        let mut data = vec![255u8; (w * h * 4) as usize];
        for px in data.chunks_exact_mut(4) {
            px[..3].copy_from_slice(&rgb);
        }
        Texture::new(w, h, TextureFormat::Rgba8UnormSrgb, data)
    }

    fn row_red(t: &Texture, y: u32) -> u8 {
        t.data[(y * t.width * 4) as usize]
    }

    #[test]
    fn the_polar_cap_replaces_the_pole_and_leaves_the_tropics_alone() {
        // 180 rows over 180 degrees, so row index is colatitude in degrees.
        let mut base = flat(16, 180, [0, 0, 0]);
        let cap = flat(8, 90, [255, 255, 255]);
        blend_polar_cap(&mut base, &cap, 72.0, 82.0);

        // Row 4 is 86 degrees north: past the band, so the cap alone.
        assert_eq!(row_red(&base, 4), 255, "the pole should be all cap");
        // Row 20 is 70 degrees north: short of the band, untouched.
        assert_eq!(
            row_red(&base, 20),
            0,
            "below the band the base must survive"
        );
        // The equator and the far south are nowhere near it.
        assert_eq!(row_red(&base, 90), 0);
        assert_eq!(
            row_red(&base, 176),
            0,
            "a north band must not touch the south"
        );
    }

    #[test]
    fn the_polar_cap_joins_without_a_step() {
        let mut base = flat(16, 180, [0, 0, 0]);
        let cap = flat(8, 90, [255, 255, 255]);
        blend_polar_cap(&mut base, &cap, 72.0, 82.0);
        // Monotone from the untouched base up to the pure cap, with no jump
        // bigger than the smoothstep's own slope — a hard edge here is exactly
        // the seam the feathering exists to avoid.
        let mut prev = 0u8;
        for y in (0..20).rev() {
            let v = row_red(&base, y);
            assert!(v >= prev, "row {y} went backwards: {v} after {prev}");
            assert!(v - prev < 60, "step of {} at row {y}", v - prev);
            prev = v;
        }
        assert_eq!(prev, 255);
    }

    #[test]
    fn a_southward_band_addresses_the_south_cap() {
        let mut base = flat(16, 180, [0, 0, 0]);
        let cap = flat(8, 90, [255, 255, 255]);
        blend_polar_cap(&mut base, &cap, -72.0, -82.0);
        assert_eq!(row_red(&base, 176), 255, "the south pole should be all cap");
        assert_eq!(
            row_red(&base, 4),
            0,
            "a south band must not touch the north"
        );
    }

    #[test]
    fn the_polar_cap_mixes_in_linear_light() {
        // Mixing sRGB bytes directly would make every partial blend of white
        // ice over dark sea too dark, which is visible as a grey ring around
        // the cap. Comparing against the byte-space answer states that
        // directly, and needs no row to land exactly on the midpoint.
        let mut base = flat(16, 180, [0, 0, 0]);
        let cap = flat(8, 90, [255, 255, 255]);
        blend_polar_cap(&mut base, &cap, 72.0, 82.0);
        let mut checked = 0;
        for y in 0..20u32 {
            let lat = 90.0 - (y as f32 + 0.5);
            let t = ((lat - 72.0) / 10.0).clamp(0.0, 1.0);
            let mix = t * t * (3.0 - 2.0 * t);
            if !(0.1..=0.9).contains(&mix) {
                continue;
            }
            // Black to white in linear light, so the mix *is* the linear value
            // and re-encoding it gives the byte to expect.
            let want = (1.055 * mix.powf(1.0 / 2.4) - 0.055) * 255.0;
            let got = row_red(&base, y) as f32;
            assert!(
                (got - want).abs() <= 2.0,
                "row {y} (mix {mix:.2}) came out at {got}, wanted {want:.0} \
                 (a byte-space lerp would give {:.0})",
                mix * 255.0
            );
            checked += 1;
        }
        assert!(checked >= 3, "only {checked} rows fell inside the band");
    }

    #[test]
    fn direction_inverts_how_the_sphere_lays_a_map_out() {
        // `direction` decides where generated content *goes*; `SphereGeometry`
        // decides where it is *shown*. If the two disagree every generated map
        // is mirrored, which is invisible in noise and wrong the moment anyone
        // places a known feature at a known longitude.
        use crate::geometries::SphereGeometry;
        let g = SphereGeometry::new(1.0, 64, 32);
        let pos = g.get_attribute("position").expect("position");
        let uv = g.get_attribute("uv").expect("uv");
        let mut checked = 0;
        for i in 0..pos.count() {
            let (x, y, z) = (pos.array[i * 3], pos.array[i * 3 + 1], pos.array[i * 3 + 2]);
            let (u, v) = (uv.array[i * 2], uv.array[i * 2 + 1]);
            // Skip the poles, where longitude means nothing.
            if y.abs() > 0.995 {
                continue;
            }
            // `direction` takes *data* v — row 0 first — while the sphere's uv
            // is what the sampler sees, and `Texture::flip_y` (on by default,
            // and what `MapBuffer::into_texture` leaves set) turns one into the
            // other on upload. The comparison has to undo that flip.
            let d = direction(u, 1.0 - v);
            let dot = d[0] * x + d[1] * y + d[2] * z;
            assert!(
                dot > 0.999,
                "uv ({u:.3}, {v:.3}) is at ({x:.3}, {y:.3}, {z:.3}) on the sphere \
                 but `direction` puts it at ({:.3}, {:.3}, {:.3})",
                d[0],
                d[1],
                d[2]
            );
            checked += 1;
        }
        assert!(checked > 1000, "only {checked} vertices compared");
    }

    #[test]
    fn cube_faces_sample_the_directions_they_are_placed_at() {
        // The face basis generates the texture *and* places the quad, so the
        // two only have to agree with each other — but the direction-to-uv
        // inverse has to match how `SphereGeometry` lays the map out, which
        // runs longitude the opposite way to `direction()`. Getting that wrong
        // mirrors the sky east-for-west and changes no statistic about it.
        //
        // A map that is a ramp in longitude catches exactly that.
        let (w, h) = (64u32, 32u32);
        let mut data = vec![255u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let o = ((y * w + x) * 4) as usize;
                let v = (x * 255 / (w - 1)) as u8;
                data[o..o + 3].copy_from_slice(&[v, v, v]);
            }
        }
        let src = Texture::new(w, h, TextureFormat::Rgba8Unorm, data);
        let faces = equirect_to_cube_faces(&src, 16);
        assert_eq!(faces.len(), 6);

        // Centre texel of each face looks straight down its forward axis, so
        // it must carry whatever the sphere would show in that direction.
        for (face, (f, _, _)) in faces.iter().zip(CUBE_FACES.iter()) {
            let mid = (8 * 16 + 8) * 4;
            let got = face.data[mid] as f32 / 255.0;
            // What `SphereGeometry` shows along `f`: u = (pi - atan2(z, x)) / tau.
            let want = if f[1].abs() > 0.9 {
                // Straight up or down the axis, longitude is undefined — the
                // ramp there is whatever the pole row averaged to, so skip it.
                continue;
            } else {
                let u = (std::f32::consts::PI - f[2].atan2(f[0])) / std::f32::consts::TAU;
                u - u.floor()
            };
            // The ramp wraps 1 → 0 at u = 0, and the −X face looks straight at
            // that seam. A sampler that wraps in longitude — which is the
            // correct thing to do — has taps either side of the jump and
            // averages them, so the centre texel there reads about 0.09 rather
            // than 0. That is the map being discontinuous, not the projection
            // being wrong, and it is the same reason the poles are skipped
            // above: the check needs somewhere the source is smooth.
            if !(0.02..=0.98).contains(&want) {
                continue;
            }
            assert!(
                (got - want).abs() < 0.05,
                "face {f:?} centre carries {got:.3}, the sphere shows {want:.3} there"
            );
        }
    }

    #[test]
    fn cube_faces_follow_the_sources_flip() {
        // `flip_y` says what the uploader will do, which is what decides
        // whether row 0 is the north pole or the south. A `DataTexture` the
        // caller flipped by hand has it off and holds south first; getting that
        // backwards turns the sky upside down, and a starfield gives no clue
        // that it happened.
        let (w, h) = (32u32, 16u32);
        // North half white, south half black.
        let mut data = vec![255u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let o = ((y * w + x) * 4) as usize;
                let v = if y < h / 2 { 255 } else { 0 };
                data[o..o + 3].copy_from_slice(&[v, v, v]);
            }
        }
        // flip_y on: row 0 is north, so the +Y face should come out white.
        let mut north_first = Texture::new(w, h, TextureFormat::Rgba8Unorm, data.clone());
        north_first.flip_y = true;
        let up = &equirect_to_cube_faces(&north_first, 8)[2]; // +Y
        assert!(
            up.data[0] > 200,
            "+Y face should be the white half, got {}",
            up.data[0]
        );

        // flip_y off: the same bytes now mean row 0 is south, so +Y is black.
        let mut south_first = Texture::new(w, h, TextureFormat::Rgba8Unorm, data);
        south_first.flip_y = false;
        let up = &equirect_to_cube_faces(&south_first, 8)[2];
        assert!(
            up.data[0] < 55,
            "+Y face should be the black half, got {}",
            up.data[0]
        );
    }

    #[test]
    fn cube_faces_keep_the_maps_range() {
        // Half-float in, half-float out: a starfield's whole point is the part
        // above 1.0, and resampling must not clip it.
        let sky = generate_starfield(256, 3000);
        assert!(matches!(sky.format, TextureFormat::Rgba16Float));
        let faces = equirect_to_cube_faces(&sky, 64);
        assert_eq!(faces.len(), 6);
        let peak = faces
            .iter()
            .flat_map(|f| f.data.chunks_exact(2))
            .map(|b| {
                crate::renderer::gpu_texture::f16_bits_to_f32(u16::from_le_bytes([b[0], b[1]]))
            })
            .fold(0.0f32, f32::max);
        assert!(
            peak > 1.0,
            "cube faces clipped the sky's range: peak {peak}"
        );
        for f in &faces {
            assert!(matches!(f.format, TextureFormat::Rgba16Float));
            assert!(
                !f.flip_y,
                "face rows are written in face order, not flipped"
            );
        }
    }

    #[test]
    fn ice_is_told_from_desert_by_its_colour_not_its_brightness() {
        // A land/sea split has nowhere to put ice, and latitude is the wrong
        // test — Greenland's cap and the Sahara sit at the same brightness.
        // What separates them is chroma: snow is neutral, sand is not.
        let (w, h) = (8u32, 4u32);
        let mut field = HeightField {
            width: w,
            height: h,
            values: vec![1.0; (w * h) as usize],
        };
        // One row of ocean, so the water class is exercised too.
        for x in 0..w as usize {
            field.values[x] = 0.0;
        }
        let mut albedo = vec![0u8; (w * h * 4) as usize];
        for i in 0..(w * h) as usize {
            // Column 0-3 bright neutral (snow), 4-7 bright yellow (desert).
            let px = if i % (w as usize) < 4 {
                [240, 242, 238]
            } else {
                [235, 200, 130]
            };
            albedo[i * 4..i * 4 + 3].copy_from_slice(&px);
            albedo[i * 4 + 3] = 255;
        }
        let albedo = Texture::new(w, h, TextureFormat::Rgba8UnormSrgb, albedo);
        let rough = field.to_roughness_map_with_ice(&albedo, 0.5, 0.45, 0.85, 0.30);
        let at = |x: u32, y: u32| rough.data[((y * w + x) * 4) as usize] as f32 / 255.0;

        assert!(
            (at(1, 1) - 0.30).abs() < 0.01,
            "snow should be ice: {}",
            at(1, 1)
        );
        assert!(
            (at(6, 1) - 0.85).abs() < 0.01,
            "desert should be land: {}",
            at(6, 1)
        );
        assert!(
            (at(1, 0) - 0.45).abs() < 0.01,
            "ocean should stay water: {}",
            at(1, 0)
        );
    }

    #[test]
    fn the_hash_covers_the_whole_range() {
        // The sign-extension bug this guards against caps the hash at 0.5,
        // which drowns a generated planet.
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        for i in 0..4000 {
            let v = hash3(i % 60, i / 60, i % 7, 11);
            lo = lo.min(v);
            hi = hi.max(v);
        }
        assert!(lo < 0.05, "hash never gets near 0: {lo}");
        assert!(hi > 0.95, "hash never gets near 1: {hi}");
    }

    #[test]
    fn elevation_lands_near_earths_land_fraction() {
        let mut land = 0;
        let total = 40 * 80;
        for y in 0..40 {
            for x in 0..80 {
                let d = direction((x as f32 + 0.5) / 80.0, (y as f32 + 0.5) / 40.0);
                if elevation(d) > 0.0 {
                    land += 1;
                }
            }
        }
        let fraction = land as f32 / total as f32;
        assert!(
            (0.15..0.45).contains(&fraction),
            "land fraction {fraction} is nowhere near Earth's 0.29"
        );
    }

    #[test]
    fn generated_maps_are_the_right_size_and_all_present() {
        let maps = generate_earth_maps(128);
        for t in [
            &maps.albedo,
            &maps.normal,
            &maps.roughness,
            &maps.night,
            &maps.clouds,
        ] {
            let t = t.as_ref().expect("every map is generated");
            assert_eq!((t.width, t.height), (128, 64));
            assert_eq!(t.data.len(), 128 * 64 * 4);
            // Longitude must wrap or the date line shows as a seam.
            assert_eq!(t.wrap_s, TextureWrap::Repeat);
        }
        // Colour maps are sRGB; data maps are linear.
        assert_eq!(maps.albedo.unwrap().format, TextureFormat::Rgba8UnormSrgb);
        assert_eq!(maps.normal.unwrap().format, TextureFormat::Rgba8Unorm);
    }

    #[test]
    fn the_night_side_is_lit_about_as_much_as_earths() {
        // Roughly 1% of Earth's surface is bright at night. The gates here are
        // calibrated against fbm's actual spread (~0.35..0.65 for four
        // octaves), not its nominal 0..1 range; against the nominal range they
        // pass almost nothing and the night side comes out uniformly black.
        let night = generate_earth_maps(512).night.unwrap();
        let total = night.data.len() / 4;
        let bright = night.data.chunks_exact(4).filter(|p| p[0] >= 40).count();
        let fraction = bright as f32 / total as f32;
        assert!(
            (0.003..0.04).contains(&fraction),
            "{:.2}% of the map is lit; Earth's is around 1%",
            fraction * 100.0
        );
        // And the brightest cores have to actually reach the top of the range,
        // or the emissive has nothing to work with.
        let peak = night.data.chunks_exact(4).map(|p| p[0]).max().unwrap();
        assert!(peak > 150, "the brightest city is only {peak}/255");

        // Cities are on land: nowhere the elevation map calls ocean is lit.
        for y in 0..night.height {
            for x in 0..night.width {
                let o = ((y * night.width + x) * 4) as usize;
                if night.data[o] > 20 {
                    let d = direction(
                        (x as f32 + 0.5) / night.width as f32,
                        (y as f32 + 0.5) / night.height as f32,
                    );
                    assert!(elevation(d) > 0.0, "a lit texel out at sea at ({x}, {y})");
                }
            }
        }
    }

    #[test]
    fn normalising_a_height_map_recovers_the_wasted_range() {
        // NASA's LOLA map spans 7..161 of 255 — used as-is it throws away two
        // thirds of the precision and adds a constant swell to whatever it
        // displaces.
        let (w, h) = (16u32, 8u32);
        let mut data = vec![255u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let o = ((y * w + x) * 4) as usize;
                // A narrow band, 40..120, like a real elevation product.
                let v = 40 + ((x * 80) / (w - 1)) as u8;
                data[o..o + 3].copy_from_slice(&[v, v, v]);
            }
        }
        let hf = HeightField::from_texture(&Texture::new(w, h, TextureFormat::Rgba8Unorm, data));
        let (lo, hi) = hf.range();
        assert!((lo - 40.0 / 255.0).abs() < 1e-3 && (hi - 120.0 / 255.0).abs() < 1e-3);

        let plain = hf.to_height_map(0.0);
        let stretched = hf.to_height_map_normalized();
        let red = |t: &Texture, x: u32| t.data[((h / 2 * w + x) * 4) as usize];
        // Unstretched, the band stays a band.
        assert_eq!(red(&plain, 0), 40);
        assert_eq!(red(&plain, w - 1), 120);
        // Stretched, it spans the encoding.
        assert_eq!(red(&stretched, 0), 0);
        assert_eq!(red(&stretched, w - 1), 255);
    }

    #[test]
    fn easing_the_poles_flattens_only_the_rows_that_feed_the_pole_vertices() {
        // A map that alternates black and white every texel along longitude.
        let (w, h) = (512u32, 256u32);
        let mut data = vec![255u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let o = ((y * w + x) * 4) as usize;
                let v = if x % 2 == 0 { 0 } else { 255 };
                data[o..o + 3].copy_from_slice(&[v, v, v]);
            }
        }
        let mut t = Texture::new(w, h, TextureFormat::Rgba8Unorm, data);
        ease_poles(&mut t);

        let spread = |y: u32| {
            let base = (y * w * 4) as usize;
            let vals: Vec<u8> = (0..w).map(|x| t.data[base + (x * 4) as usize]).collect();
            vals.iter().copied().max().unwrap() - vals.iter().copied().min().unwrap()
        };
        // The pole row is flat, which is what stops a displaced sphere tearing:
        // the fan's apex vertices all sit at the same point but carry different
        // `u`, so they have to read the same value.
        assert!(spread(0) < 20, "the top row still varies by {}", spread(0));
        assert!(
            spread(h - 1) < 20,
            "the bottom row still varies by {}",
            spread(h - 1)
        );

        // And almost nothing else is. Row 5 of 256 is about 86°N — well inside
        // what the old ten-degree ramp washed into its own average, which is
        // what turned the Arctic into a featureless disc. Real imagery there
        // has to survive.
        assert_eq!(spread(5), 255, "86 degrees north was flattened");
        assert_eq!(spread(h / 2), 255, "the equator should keep its full range");
        assert_eq!(spread(h / 2 - 1), 255);
    }

    #[test]
    fn a_displaced_pole_reads_one_height_all_the_way_round() {
        // The reason the flattening exists at all. Every apex vertex of the
        // pole fan samples the top row at its own `u`; if they disagree, each
        // gets displaced by a different amount and the pole opens into a star.
        let (w, h) = (512u32, 256u32);
        let mut data = vec![255u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let o = ((y * w + x) * 4) as usize;
                // Worst case: a single tall spike at one longitude.
                let v = if x == 3 { 255 } else { 0 };
                data[o..o + 3].copy_from_slice(&[v, v, v]);
            }
        }
        let mut t = Texture::new(w, h, TextureFormat::Rgba8Unorm, data);
        ease_poles(&mut t);
        for row in [0u32, h - 1] {
            let base = (row * w * 4) as usize;
            let vals: Vec<u8> = (0..w).map(|x| t.data[base + (x * 4) as usize]).collect();
            let spread = vals.iter().copied().max().unwrap() - vals.iter().copied().min().unwrap();
            assert!(
                spread <= 1,
                "row {row} varies by {spread}; a displaced pole would tear"
            );
        }
    }

    #[test]
    fn clouds_carry_their_shape_in_the_alpha_channel() {
        let maps = generate_earth_maps(256);
        let clouds = maps.clouds.unwrap();
        let alphas: Vec<u8> = clouds.data.chunks_exact(4).map(|p| p[3]).collect();
        assert!(alphas.iter().any(|&a| a > 200), "no thick cloud anywhere");
        assert!(alphas.iter().any(|&a| a < 20), "no clear sky anywhere");
        // …and the colour is white throughout, so the alpha does the shaping.
        assert!(clouds.data.chunks_exact(4).all(|p| p[0] == 255));
    }

    #[test]
    fn a_height_field_derives_relief_and_water() {
        // A ramp: left half "ocean", right half "land".
        let (w, h) = (32u32, 16u32);
        let mut data = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let o = ((y * w + x) * 4) as usize;
                let v = if x < w / 2 { 40 } else { 200 };
                data[o..o + 4].copy_from_slice(&[v, v, v, 255]);
            }
        }
        let hf = HeightField::from_texture(&Texture::new(w, h, TextureFormat::Rgba8Unorm, data));
        assert_eq!(hf.values.len(), (w * h) as usize);

        let rough = hf.to_roughness_map(0.5, 0.1, 0.9);
        let sample = |x: u32| rough.data[((h / 2 * w + x) * 4) as usize];
        assert!(sample(4) < 40, "water should be smooth");
        assert!(sample(28) > 200, "land should be rough");

        // The normal map is flat except at the shoreline step.
        let normal = hf.to_normal_map(3.0);
        let flat = normal.data[((h / 2 * w + 4) * 4) as usize];
        let edge = normal.data[((h / 2 * w + w / 2) * 4) as usize];
        assert!((flat as i32 - 128).abs() < 6, "open water should be flat");
        assert_ne!(flat, edge, "the shoreline should register as a slope");
    }

    /// The sky as linear RGB triples.
    fn sky_pixels(t: &Texture) -> Vec<[f32; 3]> {
        use crate::renderer::gpu_texture::f16_bits_to_f32;
        t.data
            .chunks_exact(8)
            .map(|p| {
                let h = |i: usize| f16_bits_to_f32(u16::from_le_bytes([p[i * 2], p[i * 2 + 1]]));
                [h(0), h(1), h(2)]
            })
            .collect()
    }

    fn luma(p: [f32; 3]) -> f32 {
        0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]
    }

    #[test]
    fn starfield_density_follows_the_star_count() {
        // Density is the caller's to choose, so the property worth pinning is
        // that `count` is what sets it — not some absolute fraction.
        let lit = |count| {
            let sky = generate_starfield(256, count);
            sky_pixels(&sky).iter().filter(|&&p| luma(p) > 0.1).count()
        };
        let sparse = lit(200);
        let dense = lit(2000);
        assert!(sparse > 0, "no stars at all");
        assert!(
            dense > sparse * 3,
            "ten times the stars barely changed the sky: {sparse} → {dense}"
        );
        // At the density `Starfield::procedural` picks, the sky is mostly space.
        assert!(
            sparse * 20 < 256 * 128,
            "sparse sky is too crowded: {sparse}"
        );
    }

    #[test]
    fn the_galactic_band_is_dim_but_present() {
        // With no point stars at all, what is left is the Milky Way glow: it
        // has to be visible, and it has to stay well below star brightness or
        // it reads as fog rather than a galaxy.
        let sky = sky_pixels(&generate_starfield(256, 0));
        let brightest = sky.iter().map(|&p| luma(p)).fold(0.0, f32::max);
        assert!(brightest > 0.002, "no galactic band at all: {brightest}");
        assert!(
            brightest < 0.2,
            "the band is as bright as a star: {brightest}"
        );
        // Most of the sky is not in the band.
        let in_band = sky.iter().filter(|&&p| luma(p) > 0.002).count();
        assert!(in_band * 2 < sky.len(), "the band covers half the sky");
    }

    #[test]
    fn stars_span_the_magnitude_range_they_claim_to() {
        // Eight bits could not hold this, which is the whole reason the map is
        // half-float: a sky where every star is the same brightness is the
        // giveaway that the range got clamped somewhere.
        let sky = sky_pixels(&generate_starfield(1024, 20_000));
        let mut lumas: Vec<f32> = sky.iter().map(|&p| luma(p)).filter(|&l| l > 0.05).collect();
        lumas.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(lumas.len() > 500, "too few stars to judge: {}", lumas.len());
        let faint = lumas[lumas.len() / 20];
        let bright = lumas[lumas.len() - 1];
        assert!(
            bright / faint > 50.0,
            "the brightest star is only {:.1}x the faintest; the magnitude \
             distribution has been flattened",
            bright / faint
        );
        assert!(bright > 1.0, "nothing exceeds display white: {bright}");
    }

    #[test]
    fn a_star_keeps_its_flux_wherever_it_lands() {
        // The splat is normalised, so moving a star from the equator to near
        // the pole must not change how much light it puts into the sky — only
        // how many texels share it. Without that, stars brighten toward the
        // poles exactly where the map already oversamples.
        let total = |count| -> f32 {
            let sky = sky_pixels(&generate_starfield(256, count));
            sky.iter().map(|&p| luma(p)).sum()
        };
        let band = total(0);
        let with_stars = total(4000);
        let flux = with_stars - band;
        assert!(flux > 0.0, "stars added no light at all");

        // Same stars, twice the map: the flux is the same light spread over
        // four times the texels, so the sum should track within sampling noise.
        let hi = {
            let sky = sky_pixels(&generate_starfield(512, 4000));
            sky.iter().map(|&p| luma(p)).sum::<f32>()
        } - {
            let sky = sky_pixels(&generate_starfield(512, 0));
            sky.iter().map(|&p| luma(p)).sum::<f32>()
        };
        let ratio = hi / flux;
        assert!(
            (0.8..1.25).contains(&ratio),
            "doubling the map's edge changed the stars' total flux by {ratio:.2}x"
        );
    }

    #[test]
    fn star_colour_is_temperature_and_not_brightness() {
        // Normalised to unit luminance, so a cool star is orange rather than
        // dim — brightness is the magnitude's job.
        for t in [3000.0, 5800.0, 15000.0] {
            let rgb = blackbody_rgb(t);
            let l = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
            assert!((l - 1.0).abs() < 0.02, "{t}K has luminance {l}");
        }
        let cool = blackbody_rgb(3000.0);
        let hot = blackbody_rgb(15000.0);
        assert!(cool[0] > cool[2], "3000K should be red-leaning: {cool:?}");
        assert!(hot[2] > hot[0], "15000K should be blue-leaning: {hot:?}");
        // The sun is close to neutral.
        let sun = blackbody_rgb(5800.0);
        assert!(
            (sun[0] - sun[2]).abs() < 0.25,
            "5800K should be near neutral: {sun:?}"
        );
    }
}
