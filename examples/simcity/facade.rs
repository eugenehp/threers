//! Part of the `simcity` example; see `mod.rs`.
#![allow(dead_code)]

use super::*;

// ---------------------------------------------------------------------------
// Facade textures.
// ---------------------------------------------------------------------------

/// One architectural idiom: how its windows are cut, and how they light up.
pub(crate) struct FacadeStyle {
    /// Mullion / spandrel colour (sRGB 0..1).
    pub(crate) frame: [f32; 3],
    /// The two glass tones the windows are drawn from.
    pub(crate) glass_a: [f32; 3],
    pub(crate) glass_b: [f32; 3],
    /// Pixels of solid wall on each side of a window inside its 16px cell.
    pub(crate) side: u32,
    /// Pixels of wall above the window (the head) and below it (the sill).
    pub(crate) head: u32,
    pub(crate) sill: u32,
    /// Fraction of windows lit after dark.
    pub(crate) lit_chance: f32,
    /// Roughness written into the map where a window is; the wall is 1.0.
    pub(crate) glass_roughness: f32,
    /// Course lines every N pixels in the wall, or 0 for none.
    pub(crate) coursing: u32,
    pub(crate) roughness: f32,
    pub(crate) metalness: f32,
    /// How much clear lacquer sits over the base: a lot on a curtain wall,
    /// almost none on brick.
    pub(crate) clearcoat: f32,
    /// The colour this idiom burns after dark, and how hard.
    ///
    /// The gains look low against the old constant of 1.6. That constant was
    /// never reaching the facades — the code setting it matched `Standard` and
    /// they are `Physical` — so the windows had been lit by the material
    /// default all along. Once the intensity actually arrived, 1.6 blew the
    /// whole skyline out.
    ///
    /// Every building in the city used to light up the same warm cream,
    /// because one emissive colour and one intensity were applied to every
    /// facade mesh there was. An office floor is fluorescent and slightly
    /// green-blue; a flat is a table lamp. Nothing else distinguishes them at
    /// night, and at night the windows are all you can see.
    pub(crate) emissive: u32,
    pub(crate) emissive_gain: f32,
    /// How far the per-window mix is pushed toward incandescent: 0 all cool,
    /// 1 all warm. Offices are mostly cool with the odd late desk lamp;
    /// walk-ups are the other way round.
    pub(crate) warm_bias: f32,
}

/// How many architectural idioms the city is built from.
///
/// Three was enough to tell a tower from a walk-up and not enough to tell one
/// tower from another: every building over thirteen storeys in the city had
/// the same window in it. Each style is one merged mesh and therefore one
/// draw, so the count is a budget rather than a taste.
pub(crate) const FACADE_STYLES: usize = 5;

pub(crate) const CELL_PX: u32 = 32;
pub(crate) const TILE_PX: u32 = CELL_PX * 8; // one texture wrap = TILE bays x TILE storeys

pub(crate) fn srgb_byte(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Build the albedo tile, the lit-window mask, and the roughness map.
///
/// All three share one set of UVs, so every lit rectangle lands on the window
/// it belongs to at any building size — and so does the roughness, which is
/// what separates the glass from the wall it is set in. A single roughness
/// for a whole facade is the difference between a curtain wall and a
/// painted-on one.
pub(crate) fn facade_textures(style: &FacadeStyle, seed: u64) -> (Arc<Texture>, Arc<Texture>, Arc<Texture>) {
    let mut rng = Rng::new(seed);
    let cells = 8usize;
    // Decide each cell up front so the pixel loop is a lookup.
    let mut plan = Vec::with_capacity(cells * cells);
    for _ in 0..cells * cells {
        plan.push((
            rng.f(),                      // glass tone
            rng.chance(style.lit_chance), // lit after dark
            // Warm vs cool, pushed toward this idiom's own habit rather than
            // being an even coin flip in every building in the city.
            {
                let u = rng.f();
                (u * u * (3.0 - 2.0 * u) * 0.55 + style.warm_bias * 0.75).clamp(0.0, 1.0)
            },
            rng.range(0.45, 1.0),         // brightness
            // A blind, drawn down this far. Most windows have one and most of
            // them are up; a facade where every pane is identical glass reads
            // as a texture rather than as a building.
            if rng.chance(0.45) {
                rng.range(0.15, 0.85)
            } else {
                0.0
            },
        ));
    }

    let n = TILE_PX as usize;
    let mut albedo = vec![0u8; n * n * 4];
    let mut lit = vec![0u8; n * n * 4];
    let mut rough = vec![0u8; n * n * 4];
    for py in 0..n {
        for px in 0..n {
            let (cx, cy) = (px / CELL_PX as usize, py / CELL_PX as usize);
            let (lx, ly) = (px % CELL_PX as usize, py % CELL_PX as usize);
            let (tone, on, warm, bright, blind) = plan[cy * cells + cx];
            // `flip_y` is on, so row 0 of the source ends up at the TOP of the
            // wall: the sill is the high-`ly` edge of the cell.
            let inside = lx >= style.side as usize
                && lx < CELL_PX as usize - style.side as usize
                && ly >= style.head as usize
                && ly < CELL_PX as usize - style.sill as usize;

            let i = (py * n + px) * 4;
            // Glass is smooth; the wall around it and any blind behind it
            // are not.
            let head0 = style.head as usize;
            let br = (plan[cy * cells + cx].4
                * (CELL_PX as usize - head0 - style.sill as usize) as f32)
                as usize;
            let smooth = inside && ly >= head0 + br;
            rough[i] = srgb_byte(if smooth { style.glass_roughness } else { 0.92 });
            rough[i + 1] = rough[i];
            rough[i + 2] = rough[i];
            rough[i + 3] = 255;
            // Row 0 of the source is the TOP of the wall, so a blind drawn
            // from the head occupies the low `ly` rows.
            let head = style.head as usize;
            let blind_rows = (blind * (CELL_PX as usize - head - style.sill as usize) as f32) as usize;
            let behind_blind = inside && ly < head + blind_rows;

            if inside {
                // A touch of sky gradient down the glass, plus a mullion every
                // few pixels so wide windows do not read as flat panels. At
                // 32 px a cell there is room for a transom too.
                let g = ly as f32 / CELL_PX as f32;
                let mullion = if lx % 9 == 0 { 0.74 } else { 1.0 };
                let transom = if ly == (CELL_PX as usize - style.sill as usize) / 2 {
                    0.80
                } else {
                    1.0
                };
                for ch in 0..3 {
                    let base = if behind_blind {
                        // Fabric, not glass: pale, flat, and slightly warm.
                        [0.62, 0.60, 0.55][ch] * (0.85 + 0.3 * tone)
                    } else {
                        mix(style.glass_a[ch], style.glass_b[ch], tone)
                    };
                    let shade = if behind_blind { 1.0 } else { mix(1.18, 0.78, g) };
                    albedo[i + ch] = srgb_byte(base * shade * mullion * transom);
                }
                let l = if on && behind_blind {
                    // A lit room behind a blind glows, dimly and evenly.
                    let g = bright * 0.42;
                    [g, g * 0.92, g * 0.80]
                } else if on {
                    let warm_rgb = [1.0, 0.80, 0.52];
                    let cool_rgb = [0.82, 0.88, 1.0];
                    let edge = if lx % 9 == 0 { 0.7 } else { 1.0 };
                    [
                        mix(cool_rgb[0], warm_rgb[0], warm) * bright * edge,
                        mix(cool_rgb[1], warm_rgb[1], warm) * bright * edge,
                        mix(cool_rgb[2], warm_rgb[2], warm) * bright * edge,
                    ]
                } else {
                    [0.0, 0.0, 0.0]
                };
                for ch in 0..3 {
                    lit[i + ch] = srgb_byte(l[ch]);
                }
            } else {
                // Wall. Grain so large blank flanks are not perfectly uniform
                // under a low sun, plus a course line every few pixels where
                // the style calls for one — brick and precast both show it.
                let grain = 0.94 + 0.12 * city_hash2(px as i32, py as i32);
                let course = if style.coursing > 0 && py % style.coursing as usize == 0 {
                    0.84
                } else {
                    1.0
                };
                // Weather staining. Rain runs off a sill and down the wall
                // below it, which is why real buildings are dirtiest in
                // vertical stripes under their windows and clean between them.
                let below_sill = ly >= CELL_PX as usize - style.sill as usize;
                let under_pane = lx >= style.side as usize
                    && lx < CELL_PX as usize - style.side as usize;
                let streak = if below_sill && under_pane {
                    let run = (ly - (CELL_PX as usize - style.sill as usize)) as f32
                        / style.sill.max(1) as f32;
                    let width = 0.55 + 0.45 * city_hash2(px as i32 / 3, cy as i32 * 31);
                    1.0 - 0.20 * run * width
                } else {
                    1.0
                };
                for ch in 0..3 {
                    albedo[i + ch] = srgb_byte(style.frame[ch] * grain * course * streak);
                    lit[i + ch] = 0;
                }
            }
            albedo[i + 3] = 255;
            lit[i + 3] = 255;
        }
    }

    let wrap = |data: Vec<u8>, format: TextureFormat| {
        let mut t = Texture::new(TILE_PX, TILE_PX, format, data);
        t.wrap_s = TextureWrap::Repeat;
        t.wrap_t = TextureWrap::Repeat;
        Arc::new(t)
    };
    (
        wrap(albedo, TextureFormat::Rgba8UnormSrgb),
        wrap(lit, TextureFormat::Rgba8UnormSrgb),
        // Roughness is data, not colour: linear, or the sRGB decode bends it.
        wrap(rough, TextureFormat::Rgba8Unorm),
    )
}

/// Tiling ground textures: asphalt and paving slabs.
///
/// Both are read through the ground batches' world-space UV override, so one
/// 128x128 tile covers six metres wherever it lands — the same tile under a
/// carriageway, a kerb and a pavement.
pub(crate) fn ground_textures() -> (Arc<Texture>, Arc<Texture>) {
    const N: u32 = 128;
    let n = N as usize;
    let mut asphalt = vec![0u8; n * n * 4];
    let mut paving = vec![0u8; n * n * 4];
    for y in 0..n {
        for x in 0..n {
            let i = (y * n + x) * 4;

            // Asphalt: fine aggregate, plus a coarser blotch so it does not
            // read as uniform noise at a distance.
            let fine = city_hash2(x as i32 * 7 + 3, y as i32 * 7 + 11);
            let coarse = city_noise(x as f32 * 0.08, y as f32 * 0.08);
            let v = 0.80 + 0.30 * fine + 0.22 * (coarse - 0.5);
            for ch in 0..3 {
                asphalt[i + ch] = srgb_byte(v);
            }
            asphalt[i + 3] = 255;

            // Paving: slabs on a 16 px grid, with a darker joint and a little
            // variation slab to slab.
            let joint = x % 16 == 0 || y % 16 == 0;
            let slab = city_hash2((x / 16) as i32, (y / 16) as i32);
            let grain = 0.97 + 0.06 * city_hash2(x as i32 * 3, y as i32 * 3);
            let p = if joint {
                0.74
            } else {
                (0.92 + 0.14 * slab) * grain
            };
            for ch in 0..3 {
                paving[i + ch] = srgb_byte(p);
            }
            paving[i + 3] = 255;
        }
    }
    let wrap = |data: Vec<u8>| {
        let mut t = Texture::new(N, N, TextureFormat::Rgba8UnormSrgb, data);
        t.wrap_s = TextureWrap::Repeat;
        t.wrap_t = TextureWrap::Repeat;
        Arc::new(t)
    };
    (wrap(asphalt), wrap(paving))
}

pub(crate) fn facade_styles() -> [FacadeStyle; FACADE_STYLES] {
    [
        // Curtain-wall tower: thin dark mullions, blue-green glass.
        FacadeStyle {
            frame: [0.42, 0.45, 0.48],
            glass_a: [0.32, 0.50, 0.58],
            glass_b: [0.48, 0.66, 0.70],
            side: 1,
            head: 1,
            sill: 2,
            lit_chance: 0.42,
            glass_roughness: 0.06,
            coursing: 0,
            roughness: 0.13,
            clearcoat: 0.42,
            // Architectural glass is a dielectric — F0 is about 0.04, and its
            // mirror look comes from low roughness plus Fresnel at grazing
            // angles, not from metalness. Pushing metalness up instead kills
            // the diffuse tint and the towers go a flat olive.
            metalness: 0.08,
            emissive: 0xdfe8ff,
            emissive_gain: 0.52,
            warm_bias: 0.18,
        },
        // Post-war office slab: precast panels with punched windows.
        FacadeStyle {
            frame: [0.72, 0.70, 0.66],
            glass_a: [0.16, 0.20, 0.24],
            glass_b: [0.24, 0.28, 0.31],
            side: 3,
            head: 3,
            sill: 4,
            lit_chance: 0.34,
            glass_roughness: 0.16,
            coursing: 16,
            roughness: 0.72,
            clearcoat: 0.12,
            metalness: 0.02,
            emissive: 0xfff2dc,
            emissive_gain: 0.62,
            warm_bias: 0.55,
        },
        // Walk-up brick: small windows, most of the wall is wall.
        FacadeStyle {
            frame: [0.55, 0.35, 0.28],
            glass_a: [0.13, 0.15, 0.18],
            glass_b: [0.20, 0.21, 0.23],
            side: 4,
            head: 4,
            sill: 5,
            lit_chance: 0.55,
            glass_roughness: 0.22,
            coursing: 8,
            roughness: 0.88,
            clearcoat: 0.04,
            metalness: 0.0,
            emissive: 0xffd49a,
            emissive_gain: 0.72,
            warm_bias: 0.86,
        },
        // Ribbon glazing: horizontal bands of window separated by deep
        // spandrels. The inter-war office block, and the one idiom here whose
        // windows are wider than they are tall.
        FacadeStyle {
            frame: [0.80, 0.78, 0.73],
            glass_a: [0.20, 0.28, 0.34],
            glass_b: [0.30, 0.40, 0.46],
            side: 1,
            head: 7,
            sill: 8,
            lit_chance: 0.38,
            glass_roughness: 0.11,
            coursing: 0,
            roughness: 0.45,
            clearcoat: 0.22,
            metalness: 0.03,
            emissive: 0xe6f2ea,
            emissive_gain: 0.46,
            warm_bias: 0.30,
        },
        // Bronze curtain wall: dark, tight-gridded and reflective. What the
        // eighties put up next to the blue-green glass of the sixties.
        FacadeStyle {
            frame: [0.26, 0.21, 0.16],
            glass_a: [0.30, 0.22, 0.13],
            glass_b: [0.44, 0.34, 0.20],
            side: 2,
            head: 2,
            sill: 2,
            lit_chance: 0.46,
            glass_roughness: 0.08,
            coursing: 0,
            roughness: 0.16,
            clearcoat: 0.36,
            metalness: 0.22,
            emissive: 0xffc78a,
            emissive_gain: 0.58,
            warm_bias: 0.68,
        },
    ]
}
