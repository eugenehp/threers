//! Load the NASA imagery `scripts/fetch-earth-textures.sh` downloads.
//!
//! The script writes into `web/assets/earth/`; point [`EarthTextures`] at that
//! directory and it assembles a [`PlanetMaps`] from whatever is present, filling
//! the gaps procedurally. Nothing here is required — [`generate_earth_maps`]
//! alone produces a complete planet — but real Blue Marble beats any amount of
//! noise, and this is what makes the downloaded files usable from Rust.
//!
//! ```no_run
//! use threers::planet::{EarthTextures, Planet};
//!
//! let maps = EarthTextures::from_dir("web/assets/earth").load();
//! let earth = Planet::earth().maps(maps);
//! ```
//!
//! Only PNG is decoded: the crate has a PNG decoder and no JPEG one. Run the
//! fetch script with `--png` to convert. Native only — this reads the
//! filesystem, which wasm32 has none of; in the browser, decode with
//! `createImageBitmap` and hand the bytes to [`PlanetMaps`] directly.
//!
//! All of it is NASA public-domain imagery; the script writes a CREDITS file
//! alongside the downloads.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::textures::{Texture, TextureFormat, TextureWrap};
use crate::utils::png::decode_png;

use super::{generate_earth_maps, generate_starfield, HeightField, PlanetMaps};

/// What was loaded from disk and what was generated to fill the gap.
///
/// Worth logging: a planet that came out wrong is usually a file that was not
/// where it was expected, and this says so without guessing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MapSources {
    /// Files that decoded, relative to the directory.
    pub loaded: Vec<String>,
    /// Maps that fell back to procedural generation.
    pub generated: Vec<String>,
    /// Files that were found but would not decode, with the reason.
    pub failed: Vec<(String, String)>,
}

impl MapSources {
    /// Whether anything at all came off disk.
    pub fn any_loaded(&self) -> bool {
        !self.loaded.is_empty()
    }
}

/// Where the NASA files are and how to interpret them.
///
/// Defaults match what the fetch script writes. Override a filename when your
/// own imagery is laid out differently.
#[derive(Debug, Clone)]
pub struct EarthTextures {
    dir: PathBuf,
    /// Blue Marble surface colour.
    pub day: String,
    /// Black Marble city lights.
    pub night: String,
    /// MODIS cloud composite — greyscale, brightness becomes opacity.
    pub clouds: String,
    /// GEBCO elevation, for relief and the land/water split.
    pub elevation: String,
    /// Imagery for the Arctic, which Blue Marble does not cover — see
    /// [`blend_polar_cap`](super::blend_polar_cap). Empty disables the splice.
    pub ice_cap: String,
    /// Latitudes over which `ice_cap` fades in, in degrees north.
    pub ice_cap_band: (f32, f32),
    /// Fall back to generated maps for anything missing.
    pub fallback: bool,
    /// Size of the generated fallback maps.
    pub fallback_width: u32,
    /// Relief strength when deriving the normal map from elevation.
    pub relief: f32,
    /// Elevation below which a texel is water, in `0..=1`. GEBCO's bathymetry
    /// puts sea level around a quarter of the range.
    pub sea_level: f32,
    /// Roughness assigned to water. See the note in `load_reporting` on why
    /// this is not near zero.
    pub sea_roughness: f32,
    /// Roughness assigned to land.
    pub land_roughness: f32,
    /// Roughness assigned to snow and ice, which a land/sea split has nowhere
    /// to put. Smoother than rock, rougher than open water — fresh snow has a
    /// real sheen at the low sun the poles spend their year under.
    pub ice_roughness: f32,
    /// What the Moon's colour map is multiplied by. LRO's mosaic is
    /// brightness-stretched; the real geometric albedo is about 0.12.
    pub moon_albedo: f32,
}

impl EarthTextures {
    /// Read from `dir`, filling anything missing procedurally.
    pub fn from_dir(dir: impl AsRef<Path>) -> Self {
        Self {
            dir: dir.as_ref().to_path_buf(),
            day: "day.png".into(),
            night: "night.png".into(),
            clouds: "clouds.png".into(),
            // The full GEBCO map is 21600x10800; the script resamples it,
            // and 4096 is already more relief than the shading will show.
            elevation: "elevation.4096.png".into(),
            ice_cap: "ice_cap.png".into(),
            // Blue Marble's fill starts around 82.6°N, but its imagery is
            // already thin below that — the whole Arctic is in polar night for
            // part of the year. Fading in from 72° lands the join over open
            // pack ice rather than across a coastline.
            ice_cap_band: (72.0, 82.0),
            fallback: true,
            fallback_width: 1024,
            relief: 6.0,
            sea_level: 0.25,
            sea_roughness: 0.45,
            land_roughness: 0.85,
            ice_roughness: 0.35,
            moon_albedo: 0.55,
        }
    }

    /// Use only what is on disk; leave the rest unset.
    pub fn strict(mut self) -> Self {
        self.fallback = false;
        self
    }

    /// Size of the generated fallback maps.
    pub fn fallback_width(mut self, width: u32) -> Self {
        self.fallback_width = width;
        self
    }

    /// Relief strength for the derived normal map.
    pub fn relief(mut self, relief: f32) -> Self {
        self.relief = relief.max(0.0);
        self
    }

    /// Where sea level sits in the elevation map's range.
    pub fn sea_level(mut self, sea_level: f32) -> Self {
        self.sea_level = sea_level.clamp(0.0, 1.0);
        self
    }

    /// Roughness for water and for land.
    pub fn roughness(mut self, water: f32, land: f32) -> Self {
        self.sea_roughness = water.clamp(0.0, 1.0);
        self.land_roughness = land.clamp(0.0, 1.0);
        self
    }

    /// Set the water roughness from a wind speed, via Cox & Munk (1954).
    ///
    /// Cox and Munk measured the sea surface's slope distribution from aerial
    /// photographs of sun glitter and got a mean square slope of
    /// `0.003 + 0.00512 W` for `W` the wind at 10 m in m/s. GGX's `alpha` is
    /// the RMS microfacet slope and the shader takes `alpha = roughness²`, so
    /// `roughness = mss^(1/4)`.
    ///
    /// This is why the sun's reflection is a broad patch and not a point: a
    /// dead-calm sea (0 m/s) gives roughness 0.23 and a 3° lobe, while the
    /// global mean ocean wind of about 7 m/s gives 0.44 and 11°. The default
    /// `sea_roughness` of 0.45 is that mean.
    ///
    /// ```
    /// use threers::planet::EarthTextures;
    /// let calm = EarthTextures::from_dir(".").sea_state(0.0);
    /// let windy = EarthTextures::from_dir(".").sea_state(15.0);
    /// assert!(calm.sea_roughness < windy.sea_roughness);
    /// assert!((calm.sea_roughness - 0.234).abs() < 0.01);
    /// ```
    pub fn sea_state(mut self, wind_m_s: f32) -> Self {
        let mss = 0.003 + 0.00512 * wind_m_s.max(0.0);
        self.sea_roughness = mss.sqrt().sqrt().clamp(0.0, 1.0);
        self
    }

    /// Load the maps, filling gaps as configured.
    pub fn load(&self) -> PlanetMaps {
        self.load_reporting().0
    }

    /// Load the maps and report where each one came from.
    pub fn load_reporting(&self) -> (PlanetMaps, MapSources) {
        let mut src = MapSources::default();
        let mut maps = PlanetMaps::new();

        // Surface colour is spliced before it is eased: the Arctic gap has to
        // be filled with imagery first, or easing just spreads the gap.
        maps.albedo = self
            .read_raw(&self.day, TextureFormat::Rgba8UnormSrgb, &mut src)
            .map(|mut day| {
                if !self.ice_cap.is_empty() {
                    if let Some(cap) =
                        self.read_raw(&self.ice_cap, TextureFormat::Rgba8UnormSrgb, &mut src)
                    {
                        let (from, to) = self.ice_cap_band;
                        super::blend_polar_cap(&mut day, &cap, from, to);
                    }
                }
                super::ease_poles(&mut day);
                Arc::new(day)
            });
        maps.night = self
            .read(&self.night, TextureFormat::Rgba8UnormSrgb, &mut src)
            .map(Arc::new);
        maps.clouds = self
            .read(&self.clouds, TextureFormat::Rgba8UnormSrgb, &mut src)
            .map(|grey| Arc::new(clouds_from_grey(&grey)));

        // Elevation is not itself a map the planet draws with — it is where the
        // normal and roughness maps come from, so the relief and the shine on
        // the oceans line up with the same coastlines.
        if let Some(elev) = self.read(&self.elevation, TextureFormat::Rgba8Unorm, &mut src) {
            let hf = HeightField::from_texture(&elev);
            maps.normal = Some(Arc::new(hf.to_normal_map(self.relief)));
            // Water is rough, and 0.45 is not a fudge — see `sea_state`. The
            // open ocean is not a mirror: wind roughens it, and the slope
            // distribution that produces is what spreads the sun's reflection
            // into the broad glint you see from orbit rather than a point.
            // Ice is read off the surface colour, so the roughness map needs
            // the albedo — which by here has the Arctic spliced in, so the pack
            // ice classifies with the ice sheets rather than as bare ground.
            maps.roughness = Some(Arc::new(match maps.albedo.as_ref() {
                Some(day) => hf.to_roughness_map_with_ice(
                    day,
                    self.sea_level,
                    self.sea_roughness,
                    self.land_roughness,
                    self.ice_roughness,
                ),
                None => {
                    hf.to_roughness_map(self.sea_level, self.sea_roughness, self.land_roughness)
                }
            }));
            // The same elevation, as a height map for vertex displacement.
            // Unused unless the caller asks for it — see `Planet::displacement`.
            maps.displacement = Some(Arc::new(hf.to_height_map(self.sea_level)));
        }

        if self.fallback && !maps.is_complete() {
            let gen = generate_earth_maps(self.fallback_width);
            let mut fill = |slot: &mut Option<Arc<Texture>>, from: Option<Arc<Texture>>, name| {
                if slot.is_none() {
                    *slot = from;
                    src.generated.push(String::from(name));
                }
            };
            fill(&mut maps.albedo, gen.albedo, "albedo");
            fill(&mut maps.normal, gen.normal, "normal");
            fill(&mut maps.roughness, gen.roughness, "roughness");
            fill(&mut maps.night, gen.night, "night");
            fill(&mut maps.clouds, gen.clouds, "clouds");
            fill(&mut maps.displacement, gen.displacement, "displacement");
        }

        (maps, src)
    }

    /// The Moon, from the CGI Moon Kit files the script's `--sky` flag fetches.
    ///
    /// Returns `None` when neither the colour nor the height map is there —
    /// unlike Earth there is no procedural fallback, because a generated moon
    /// looks like a generated moon.
    pub fn load_moon(&self) -> Option<PlanetMaps> {
        self.load_moon_reporting().0
    }

    /// The Moon, and where it came from.
    pub fn load_moon_reporting(&self) -> (Option<PlanetMaps>, MapSources) {
        let mut src = MapSources::default();
        let mut maps = PlanetMaps::new();
        // `moon_color.png` is whatever resolution was fetched (8k by default);
        // the 4k name is what older runs of the script left behind.
        for name in ["sky/moon_color.png", "sky/moon_color_4k.png"] {
            if let Some(t) = self.read(name, TextureFormat::Rgba8UnormSrgb, &mut src) {
                // LRO's mosaic is brightness-stretched for legibility. The Moon
                // is one of the darker bodies in the solar system — geometric
                // albedo about 0.12, the same as worn asphalt — and left
                // unscaled it renders as a white disc under any light strong
                // enough to show Earth properly.
                maps.albedo = Some(Arc::new(scale_albedo(&t, self.moon_albedo)));
                break;
            }
        }
        if let Some(h) = self.read("sky/moon_height.png", TextureFormat::Rgba8Unorm, &mut src) {
            let hf = HeightField::from_texture(&h);
            // LOLA relief is the Moon's whole character — craters are all it
            // has — so it takes a heavier hand than Earth's.
            //
            // Scaled by width, because `to_normal_map` measures slope per
            // *texel*, not per unit of ground: the same terrain at half the
            // resolution has twice the height change between neighbours, so a
            // fixed strength turns a coarse map into a field of facets. 14 is
            // tuned for `ldem_16` at 5760 px.
            maps.normal = Some(Arc::new(hf.to_normal_map(14.0 * hf.width as f32 / 5760.0)));
            // No sea level on the Moon, but LOLA fills only 7..161 of its
            // encoding — stretch it rather than throw the precision away.
            maps.displacement = Some(Arc::new(hf.to_height_map_normalized()));
        }
        ((!maps.is_empty()).then_some(maps), src)
    }

    /// The sky, from NASA's Deep Star Maps.
    ///
    /// The Deep Star Maps ship as ZIP-compressed OpenEXR — linear half-float,
    /// so the dynamic range between Sirius and a Gaia catalogue smudge
    /// survives. That is read directly; a converted PNG is used in preference
    /// if one is there, since it has already been tone-mapped.
    ///
    /// Falls back to a generated starfield of `fallback_width` when neither is
    /// present — unlike the Moon, a procedural sky is perfectly convincing at
    /// the distance a sky box is seen from.
    pub fn load_starfield(&self) -> Texture {
        self.load_starfield_reporting().0
    }

    /// The sky, and where it came from.
    pub fn load_starfield_reporting(&self) -> (Texture, MapSources) {
        let mut src = MapSources::default();
        if let Some(t) = self.read(
            "sky/starmap_4k.png",
            TextureFormat::Rgba8UnormSrgb,
            &mut src,
        ) {
            return (t, src);
        }
        // Whatever resolution was fetched; 8k is 8192x4096, the largest a
        // single texture can be and still fit WebGPU's guaranteed limit.
        let exr = ["sky/starmap_8k.exr", "sky/starmap_4k.exr"]
            .iter()
            .map(|n| self.dir.join(n))
            .find(|p| p.is_file())
            .unwrap_or_else(|| self.dir.join("sky/starmap_4k.exr"));
        if exr.is_file() {
            match std::fs::read(&exr)
                .map_err(|e| e.to_string())
                .and_then(|b| {
                    // Half-float, not 8-bit: over half the Deep Star Map's texels
                    // sit below 1/100th of full scale, and clipping to 8 bits
                    // leaves a sky that is 58% pure black.
                    crate::loaders::ExrLoader::parse_hdr(&b).map_err(|e| format!("{e:?}"))
                }) {
                Ok(mut t) => {
                    // Already linear, so no sRGB tag — the renderer tone-maps it.
                    t.wrap_s = TextureWrap::Repeat;
                    t.wrap_t = TextureWrap::ClampToEdge;
                    src.loaded.push(
                        exr.file_name()
                            .map(|n| n.to_string_lossy().into_owned())
                            .unwrap_or_default(),
                    );
                    return (t, src);
                }
                Err(e) => src.failed.push((exr.display().to_string(), e)),
            }
        }
        src.generated.push("starfield".into());
        let w = self.fallback_width;
        (
            generate_starfield(w, (w * w / 320).clamp(500, 200_000)),
            src,
        )
    }

    /// Decode one file, recording what happened.
    fn read(&self, name: &str, format: TextureFormat, src: &mut MapSources) -> Option<Texture> {
        self.read_map(name, format, src, true)
    }

    /// As `read`, but leaving the poles alone — for maps that get spliced
    /// before they are eased, since easing a gap only smears the gap.
    fn read_raw(&self, name: &str, format: TextureFormat, src: &mut MapSources) -> Option<Texture> {
        self.read_map(name, format, src, false)
    }

    fn read_map(
        &self,
        name: &str,
        format: TextureFormat,
        src: &mut MapSources,
        ease: bool,
    ) -> Option<Texture> {
        let path = self.dir.join(name);
        if !path.is_file() {
            return None;
        }
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                src.failed.push((name.into(), e.to_string()));
                return None;
            }
        };
        match decode_png(&bytes) {
            Ok(img) => {
                src.loaded.push(name.into());
                let mut t = Texture::new(img.width, img.height, format, img.rgba);
                if ease {
                    // Equirectangular products all pinwheel at the poles; take
                    // it out once, here, so everything derived from them
                    // inherits it.
                    super::ease_poles(&mut t);
                }
                // Longitude wraps at the date line; latitude must not, or the
                // north pole samples the south.
                t.wrap_s = TextureWrap::Repeat;
                t.wrap_t = TextureWrap::ClampToEdge;
                Some(t)
            }
            Err(e) => {
                let hint = if e == "not a PNG" {
                    "not a PNG — rerun the fetch script with --png".to_string()
                } else {
                    e
                };
                src.failed.push((name.into(), hint));
                None
            }
        }
    }
}

/// Scale a colour map's brightness, in linear light.
///
/// Multiplying sRGB bytes directly would darken the midtones far more than the
/// ends, because the encoding is a curve — so decode, scale, re-encode.
pub fn scale_albedo(src: &Texture, factor: f32) -> Texture {
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
    let mut data = src.data.as_ref().clone();
    for px in data.chunks_exact_mut(4) {
        for c in px.iter_mut().take(3) {
            *c = to_srgb(to_linear(*c) * factor);
        }
    }
    let mut t = Texture::new(src.width, src.height, src.format, data);
    t.wrap_s = src.wrap_s;
    t.wrap_t = src.wrap_t;
    t
}

/// Turn a greyscale cloud photo into an alpha-shaped cloud map.
///
/// MODIS ships cloud cover as brightness on black. A cloud shell needs the
/// opposite split: white everywhere, with the cover in alpha — otherwise the
/// clear sky renders as a black shell over the planet instead of as nothing.
pub fn clouds_from_grey(grey: &Texture) -> Texture {
    let bpp = grey.bytes_per_pixel().max(1);
    let texels = (grey.width as usize) * (grey.height as usize);
    let mut data = vec![255u8; texels * 4];
    for i in 0..texels {
        let o = i * bpp;
        let l = grey.data.get(o).copied().unwrap_or(0) as f32 / 255.0;
        // Lift the faint haze off the floor so thin cloud still reads, and let
        // thick cloud reach full cover.
        let a = ((l - 0.06) / 0.72).clamp(0.0, 1.0);
        data[i * 4 + 3] = (a * 255.0) as u8;
    }
    let mut t = Texture::new(grey.width, grey.height, TextureFormat::Rgba8UnormSrgb, data);
    t.wrap_s = TextureWrap::Repeat;
    t.wrap_t = TextureWrap::ClampToEdge;
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grey(width: u32, height: u32, level: u8) -> Texture {
        let data = vec![level; (width * height * 4) as usize];
        Texture::new(width, height, TextureFormat::Rgba8Unorm, data)
    }

    #[test]
    fn sea_state_follows_cox_munk() {
        // The default is the global mean ocean wind, about 7 m/s.
        let default_r = EarthTextures::from_dir(".").sea_roughness;
        let mean_wind = EarthTextures::from_dir(".").sea_state(7.0).sea_roughness;
        assert!(
            (default_r - mean_wind).abs() < 0.02,
            "the default {default_r} should be roughly the 7 m/s value {mean_wind}"
        );

        // Monotonic in wind, and bracketed by the calm and storm ends.
        let mut last = 0.0;
        for w in [0.0, 2.0, 7.0, 12.0, 20.0] {
            let r = EarthTextures::from_dir(".").sea_state(w).sea_roughness;
            assert!(
                r > last,
                "roughness should rise with wind: {w} m/s gave {r}"
            );
            last = r;
        }
        // A dead calm is still not a mirror — the sea has swell without wind.
        let calm = EarthTextures::from_dir(".").sea_state(0.0).sea_roughness;
        assert!((calm - 0.2340).abs() < 0.001, "calm sea roughness {calm}");
        // And the relation is alpha = roughness², alpha = RMS slope.
        let windy = EarthTextures::from_dir(".").sea_state(12.0).sea_roughness;
        let mss = (windy * windy) * (windy * windy);
        assert!((mss - (0.003 + 0.00512 * 12.0)).abs() < 1e-4);
    }

    #[test]
    fn grey_clouds_become_white_with_alpha_cover() {
        let clear = clouds_from_grey(&grey(4, 2, 0));
        let thick = clouds_from_grey(&grey(4, 2, 255));
        // White throughout, both ways — the alpha does the shaping.
        assert!(clear
            .data
            .chunks_exact(4)
            .all(|p| p[..3] == [255, 255, 255]));
        assert!(thick
            .data
            .chunks_exact(4)
            .all(|p| p[..3] == [255, 255, 255]));
        // Black sky is fully transparent; bright cloud is fully opaque.
        assert_eq!(clear.data[3], 0);
        assert_eq!(thick.data[3], 255);
        // And longitude wraps, so the date line is not a seam.
        assert_eq!(thick.wrap_s, TextureWrap::Repeat);
        assert_eq!(thick.wrap_t, TextureWrap::ClampToEdge);
    }

    #[test]
    fn a_missing_directory_falls_back_to_a_whole_planet() {
        let (maps, src) = EarthTextures::from_dir("/nonexistent/nasa")
            .fallback_width(64)
            .load_reporting();
        assert!(maps.is_complete(), "the fallback has to cover every slot");
        assert!(!src.any_loaded());
        // Five drawing maps plus the height map.
        assert_eq!(src.generated.len(), 6);
        assert!(maps.displacement.is_some(), "a height map is generated too");
        assert!(src.failed.is_empty(), "absent is not the same as broken");
    }

    #[test]
    fn strict_mode_leaves_the_gaps_open() {
        let (maps, src) = EarthTextures::from_dir("/nonexistent/nasa")
            .strict()
            .load_reporting();
        assert!(maps.is_empty());
        assert!(src.generated.is_empty());
    }

    #[test]
    fn a_file_that_is_not_a_png_is_reported_not_ignored() {
        let dir = std::env::temp_dir().join("threers-planet-nasa-test");
        std::fs::create_dir_all(&dir).unwrap();
        // A JPEG is what the fetch script leaves without `--png`; the failure
        // has to name that, or the planet just silently comes out procedural.
        std::fs::write(dir.join("day.png"), b"\xff\xd8\xff\xe0 not really a png").unwrap();

        let (_, src) = EarthTextures::from_dir(&dir)
            .fallback_width(64)
            .load_reporting();
        let (name, why) = src.failed.first().expect("the bad file is reported");
        assert_eq!(name, "day.png");
        assert!(
            why.contains("--png"),
            "the message should say how to fix it: {why}"
        );
        assert!(src.generated.contains(&"albedo".to_string()));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn real_files_are_loaded_and_derived_from() {
        let dir = std::env::temp_dir().join("threers-planet-nasa-real");
        std::fs::create_dir_all(&dir).unwrap();

        // An elevation map: left half below sea level, right half above.
        let (w, h) = (16u32, 8u32);
        let mut rgba = vec![255u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let v = if x < w / 2 { 20 } else { 200 };
                let o = ((y * w + x) * 4) as usize;
                rgba[o..o + 3].copy_from_slice(&[v, v, v]);
            }
        }
        let png = crate::utils::png::encode_png(w, h, &rgba);
        std::fs::write(dir.join("elevation.4096.png"), png).unwrap();

        let (maps, src) = EarthTextures::from_dir(&dir)
            .fallback_width(64)
            .load_reporting();
        assert_eq!(src.loaded, vec!["elevation.4096.png".to_string()]);
        // Normal and roughness come from the file, so they are its size…
        let rough = maps.roughness.as_ref().unwrap();
        assert_eq!((rough.width, rough.height), (w, h));
        assert_eq!(maps.normal.as_ref().unwrap().width, w);
        // …and the ocean half is smoother than the land half, at whatever the
        // configured pair happens to be.
        let at = |x: u32| rough.data[(((h / 2) * w + x) * 4) as usize];
        let cfg = EarthTextures::from_dir(&dir);
        let expect = |r: f32| (r * 255.0) as u8;
        assert_eq!(at(2), expect(cfg.sea_roughness), "ocean roughness");
        assert_eq!(at(13), expect(cfg.land_roughness), "land roughness");
        assert!(at(2) < at(13), "water has to be smoother than land");

        // …and the pair is configurable.
        let custom = EarthTextures::from_dir(&dir)
            .fallback_width(64)
            .roughness(0.05, 1.0)
            .load();
        let cr = custom.roughness.as_ref().unwrap();
        assert_eq!(cr.data[(((h / 2) * w + 2) * 4) as usize], expect(0.05));
        // The rest is generated, at the fallback size.
        assert_eq!(maps.albedo.as_ref().unwrap().width, 64);
        assert!(!src.generated.contains(&"normal".to_string()));

        std::fs::remove_dir_all(&dir).ok();
    }
}
