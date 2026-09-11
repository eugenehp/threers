//! The same procedural city, drawn by the Metal backend instead of wgpu.
//!
//! ```text
//! cargo run --release --features metal --example simcity_metal
//! cargo run --release --features metal --example simcity_metal -- --layout waterfront
//! ```
//!
//! `src/metal` talks to Metal through the Objective-C runtime directly — no
//! wgpu, no window, no main-thread requirement. The city itself is the same
//! `Scene` the wgpu examples build, which is the point of the exercise: if the
//! two backends disagree, it shows up here.
//!
//! Two things do not carry across. The water is a flat lattice rather than an
//! animated one, because the compute pass that moves it is WGSL. And the
//! prefiltered sky environment is a wgpu-side concept, so glass falls back to
//! its diffuse tint. Everything else — shadows, emissive window masks,
//! instanced traffic, the level-of-detail split — is backend-agnostic.
//!
//! | Flag | |
//! |------|--|
//! | `--seed N` `--blocks N` `--layout L` | as `simcity_render` |
//! | `--time T` | fraction of a day (default `0.07`) |
//! | `--view V` | `aerial` (default), `skyline`, `close`, `street` |
//! | `--size WxH` · `--out PATH` | output |

use std::path::PathBuf;

use threers::metal::{MetalError, MetalHeadlessRenderer};

// The city itself lives in `examples/simcity/`, shared by all three
// entry points. `#[path]` because a directory example would otherwise
// need its own `main.rs`, and there are three of those.
#[path = "simcity/mod.rs"]
mod city;
use city::*;

fn has_flag(name: &str) -> bool {
    std::env::args().skip(1).any(|a| a == name)
}

fn arg(name: &str) -> Option<String> {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == name {
            return args.next();
        }
    }
    None
}

fn main() -> Result<(), MetalError> {
    let defaults = CityParams::default();
    let params = CityParams {
        bake: !has_flag("--no-bake"),
        seed: arg("--seed")
            .and_then(|s| s.parse().ok())
            .unwrap_or(defaults.seed),
        blocks: arg("--blocks")
            .and_then(|s| s.parse().ok())
            .unwrap_or(defaults.blocks),
        cars: defaults.cars,
        layout: arg("--layout")
            .and_then(|s| Layout::from_name(&s))
            .unwrap_or(defaults.layout),
    };
    let (w, h) = arg("--size")
        .and_then(|s| {
            let (a, b) = s.split_once('x')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        })
        .unwrap_or((1600u32, 1000u32));
    let view = arg("--view").unwrap_or_else(|| "aerial".into());
    let time = arg("--time").and_then(|s| s.parse().ok()).unwrap_or(0.07);
    let out = PathBuf::from(arg("--out").unwrap_or_else(|| "out/simcity-metal.png".into()));

    let mut renderer = MetalHeadlessRenderer::builder()
        .size(w, h)
        .msaa(4)
        .build()?;
    println!(
        "metal: {} ({} memory)",
        renderer.device().name(),
        if renderer.device().has_unified_memory() {
            "unified"
        } else {
            "discrete"
        }
    );

    let mut scene = Scene::new();
    let mut city = generate_city(&mut scene, &params);
    let s = &city.stats;
    println!(
        "{} · seed {} · {} blocks/side · {} buildings · {} cars · {} on foot · {:.0} m across",
        params.layout.name(),
        params.seed,
        s.blocks,
        s.buildings,
        s.cars,
        s.walkers,
        s.extent * 2.0
    );

    let camera = frame_camera(&view, s.extent, w as f32 / h as f32, city.vista);
    let (near, total) = city.drive(&mut scene, 0.0, camera.position, 0.0);
    println!("  {near}/{total} movers at full detail from here");
    let sky = city.apply_sky(&mut scene, time);
    city.place_lights(&mut scene, camera.position, sky.lights);
    city.animate_lights(&mut scene, 0.0);
    scene.update_world();

    let rgba = renderer.render_to_rgba(&mut scene, &camera)?;
    let stats = renderer.stats();
    println!(
        "{} draw calls, {} triangles, {} geometries + {} textures uploaded",
        stats.draw_calls, stats.triangles, stats.geometry_uploads, stats.texture_uploads
    );

    if let Some(dir) = std::path::Path::new(&out).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(&out, threers::encode_png(w, h, &rgba)).expect("write png");
    println!("wrote {} ({w}x{h})", out.display());
    Ok(())
}
