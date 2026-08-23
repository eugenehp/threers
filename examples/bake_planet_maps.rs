//! Precompute the planet maps, so the browser does none of it.
//!
//! ```text
//! cargo run --release --features planet --example bake_planet_maps
//! cargo run --release --features planet --example bake_planet_maps -- --size 4096x2048
//! ```
//!
//! | Flag | Effect |
//! |------|--------|
//! | `--assets DIR` | where the NASA imagery is (default `web/assets/earth`) |
//! | `--out DIR` | where to write (default `<assets>/baked`) |
//! | `--size WxH` | cap every output at this size (default 4096x2048) |
//! | `--full` | no cap: write each map at whatever size it loaded |
//! | `--no-moon` | skip the Moon maps |
//!
//! Everything between the NASA downloads and a texture the GPU can take is a
//! pure function of files on disk: splicing the Arctic in, easing the poles,
//! differentiating GEBCO into a normal map, turning its bathymetry into ocean
//! roughness, reshaping the MODIS cloud photo into an alpha mask. The demo was
//! doing all of it in JavaScript on every page load, over maps of 15–30 million
//! texels. None of it depends on anything the page knows.
//!
//! So do it once, here, and write PNGs the browser can hand straight to the
//! GPU. `EarthTextures::load` is the same call the native examples make, which
//! is the point — this is not a second implementation of the pipeline, it is
//! the pipeline with a `write` on the end.
//!
//! The starfield is deliberately left alone: it is HDR, and over half the Deep
//! Star Map's texels sit below a hundredth of full scale, so baking it to 8-bit
//! PNG would black out most of the sky.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use threers::planet::{EarthTextures, PlanetMaps};
use threers::{encode_png, Texture};

fn arg(flag: &str) -> Option<String> {
    std::env::args()
        .position(|a| a == flag)
        .and_then(|i| std::env::args().nth(i + 1))
}
fn flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

/// Box-filter down to `(w, h)`.
///
/// A whole-texel average, not a bilinear tap: minifying an 8192-wide map with
/// four samples throws away three quarters of the coastline and leaves the
/// result aliased, which is exactly the detail the map is being kept for.
fn resample(src: &Texture, w: u32, h: u32) -> Texture {
    let (sw, sh) = (src.width as usize, src.height as usize);
    let (dw, dh) = (w as usize, h as usize);
    if sw == dw && sh == dh {
        return src.clone();
    }
    let bpp = src.bytes_per_pixel();
    let mut out = vec![0u8; dw * dh * 4];
    for y in 0..dh {
        let y0 = y * sh / dh;
        let y1 = ((y + 1) * sh).div_ceil(dh).max(y0 + 1).min(sh);
        for x in 0..dw {
            let x0 = x * sw / dw;
            let x1 = ((x + 1) * sw).div_ceil(dw).max(x0 + 1).min(sw);
            let n = ((y1 - y0) * (x1 - x0)) as u32;
            let mut acc = [0u32; 4];
            for sy in y0..y1 {
                for sx in x0..x1 {
                    let o = (sy * sw + sx) * bpp;
                    for (c, channel) in acc.iter_mut().enumerate() {
                        *channel += if c < bpp { src.data[o + c] as u32 } else { 255 };
                    }
                }
            }
            let o = (y * dw + x) * 4;
            for c in 0..4 {
                out[o + c] = (acc[c] / n) as u8;
            }
        }
    }
    let mut t = Texture::new(w, h, src.format, out);
    t.wrap_s = src.wrap_s;
    t.wrap_t = src.wrap_t;
    t
}

fn write(dir: &Path, name: &str, tex: &Option<Arc<Texture>>, size: Option<(u32, u32)>) -> u64 {
    let Some(tex) = tex else {
        eprintln!("  skip {name}: not loaded");
        return 0;
    };
    let t0 = Instant::now();
    let owned;
    // A cap, not a target: upscaling a 2048-wide cloud map to 4096 would
    // quadruple what the browser downloads and decodes to recover nothing, and
    // scaling both axes by the same factor keeps 2:1 maps 2:1.
    let tex = match size {
        Some((mw, mh)) if tex.width > mw || tex.height > mh => {
            let k = (mw as f32 / tex.width as f32).min(mh as f32 / tex.height as f32);
            let (w, h) = (
                ((tex.width as f32 * k).round() as u32).max(1),
                ((tex.height as f32 * k).round() as u32).max(1),
            );
            owned = resample(tex, w, h);
            &owned
        }
        _ => tex.as_ref(),
    };
    // encode_png wants tight RGBA; every map in PlanetMaps already is, but a
    // 3-byte source would silently misalign, so widen rather than assume.
    let rgba;
    let bytes = if tex.bytes_per_pixel() == 4 {
        tex.data.as_ref().as_slice()
    } else {
        let bpp = tex.bytes_per_pixel();
        rgba = (0..(tex.width as usize * tex.height as usize))
            .flat_map(|i| {
                let o = i * bpp;
                (0..4).map(move |c| if c < bpp { tex.data[o + c] } else { 255 })
            })
            .collect::<Vec<u8>>();
        &rgba
    };
    let png = encode_png(tex.width, tex.height, bytes);
    let path = dir.join(name);
    if let Err(e) = std::fs::write(&path, &png) {
        eprintln!("  FAIL {name}: {e}");
        return 0;
    }
    eprintln!(
        "  {name:<16} {:>5}x{:<5} {:>7.1} MB  {:>5.1}s",
        tex.width,
        tex.height,
        png.len() as f64 / 1e6,
        t0.elapsed().as_secs_f64()
    );
    png.len() as u64
}

fn main() {
    let assets = PathBuf::from(arg("--assets").unwrap_or_else(|| "web/assets/earth".into()));
    let out = PathBuf::from(
        arg("--out").unwrap_or_else(|| assets.join("baked").to_string_lossy().into_owned()),
    );
    // Capped, not native, by default. The browser pays for these twice — once
    // on the wire and again decoding to RGBA — and the sizes the sources happen
    // to come in are not sizes anything asks for. The Moon's colour map is
    // 8192x4096: 64 MB of PNG and 134 MB of RGBA, for a body that is thirty
    // pixels across until you fly to it, and which the demo decodes at 4096
    // anyway. Earth's albedo at 5400 wide is likewise past what a 4K globe
    // shows. `--full` opts out.
    let size = if flag("--full") {
        None
    } else {
        Some(
            arg("--size")
                .and_then(|s| {
                    let (w, h) = s.split_once('x')?;
                    Some((w.parse().ok()?, h.parse().ok()?))
                })
                .unwrap_or((4096, 2048)),
        )
    };
    if let Err(e) = std::fs::create_dir_all(&out) {
        eprintln!("cannot create {}: {e}", out.display());
        std::process::exit(1);
    }

    let started = Instant::now();
    let src = EarthTextures::from_dir(&assets);

    eprintln!("==> Earth, from {}", assets.display());
    let t0 = Instant::now();
    let (maps, report) = src.load_reporting();
    eprintln!(
        "  loaded {:?} in {:.1}s",
        report.loaded,
        t0.elapsed().as_secs_f64()
    );
    for (name, why) in &report.failed {
        eprintln!("  FAILED {name}: {why}");
    }
    if !report.generated.is_empty() {
        eprintln!(
            "  NOTE: {:?} were generated, not loaded — baking a procedural map \
             is rarely what you want",
            report.generated
        );
    }

    let mut total = 0u64;
    total += write(&out, "albedo.png", &maps.albedo, size);
    total += write(&out, "normal.png", &maps.normal, size);
    total += write(&out, "roughness.png", &maps.roughness, size);
    total += write(&out, "night.png", &maps.night, size);
    total += write(&out, "clouds.png", &maps.clouds, size);
    total += write(&out, "height.png", &maps.displacement, size);

    if !flag("--no-moon") {
        eprintln!("==> Moon");
        let (moon, report) = src.load_moon_reporting();
        for (name, why) in &report.failed {
            eprintln!("  FAILED {name}: {why}");
        }
        match moon {
            Some(PlanetMaps {
                albedo,
                normal,
                displacement,
                ..
            }) => {
                total += write(&out, "moon_albedo.png", &albedo, size);
                total += write(&out, "moon_normal.png", &normal, size);
                total += write(&out, "moon_height.png", &displacement, size);
            }
            None => eprintln!("  skip: no CGI Moon Kit imagery in {}", assets.display()),
        }
    }

    eprintln!(
        "==> {:.1} MB in {:.1}s → {}",
        total as f64 / 1e6,
        started.elapsed().as_secs_f64(),
        out.display()
    );
}
