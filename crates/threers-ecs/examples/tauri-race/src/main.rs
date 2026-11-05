//! The race in a Tauri window: the game's web build, in the system webview.
//!
//! The game is the same wasm the browser runs (`scripts/web-game.py`), drawn
//! with WebGPU where the webview has it and WebGL2 where it does not. Tauri adds
//! the native window, the app bundle, and a Rust side to talk to.
//!
//! ```sh
//! python3 scripts/web-game.py -p threers-hop --start start_race --title Race \
//!     --out crates/threers-ecs/examples/tauri-race/dist
//! cargo run --release --manifest-path crates/threers-ecs/examples/tauri-race/Cargo.toml
//! ```
//!
//! `-- --probe` starts the race without the click, measures the frame rate
//! after a few seconds, prints it and quits: a check that the game really runs
//! in this webview, for a script.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::{WebviewUrl, WebviewWindowBuilder};

/// Runs in the page before its own scripts when probing: counts frames and
/// samples the canvas once the race has been going a while, then reports.
const PROBE: &str = r#"
window.__errors = [];
addEventListener("error", e => window.__errors.push(String(e.message)));
addEventListener("unhandledrejection", e => window.__errors.push(String(e.reason)));
addEventListener("load", () => setTimeout(() => {
  let frames = 0;
  const count = () => { frames++; requestAnimationFrame(count); };
  requestAnimationFrame(count);
  setTimeout(() => {
    const canvas = document.getElementById("c");
    // Distinct colours on a coarse grid of the canvas: a blank or black frame has one.
    let colours = -1;
    try {
      const probe = document.createElement("canvas");
      probe.width = 32; probe.height = 18;
      const g = probe.getContext("2d");
      g.drawImage(canvas, 0, 0, 32, 18);
      const px = g.getImageData(0, 0, 32, 18).data;
      const seen = new Set();
      for (let i = 0; i < px.length; i += 4) seen.add((px[i] >> 4) + "," + (px[i + 1] >> 4) + "," + (px[i + 2] >> 4));
      colours = seen.size;
    } catch (e) { window.__errors.push("sample: " + e); }
    window.__TAURI__.core.invoke("report", {
      fps: frames / 3,
      width: canvas.width,
      height: canvas.height,
      webgpu: "gpu" in navigator,
      colours,
      errors: window.__errors,
    });
  }, 3000);
}, 8000));
"#;

#[tauri::command]
fn report(app: tauri::AppHandle, fps: f64, width: u32, height: u32, webgpu: bool, colours: i32, errors: Vec<String>) {
    println!("race in the webview: {fps:.0} fps, canvas {width}x{height}, WebGPU {webgpu}, {colours} colours");
    for error in &errors {
        println!("page error: {error}");
    }
    app.exit(if errors.is_empty() && fps > 20.0 && colours > 8 { 0 } else { 1 });
}

fn main() {
    let probe = std::env::args().any(|a| a == "--probe");
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![report])
        .setup(move |app| {
            // `?auto` skips the click to start; a player clicks, which is also
            // what lets sound play.
            let page = if probe { "index.html?auto" } else { "index.html" };
            let mut window = WebviewWindowBuilder::new(app, "main", WebviewUrl::App(page.into()))
                .title("Race")
                .inner_size(1280.0, 720.0);
            if probe {
                window = window.initialization_script(PROBE);
            }
            window.build()?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("the Tauri app");
}
