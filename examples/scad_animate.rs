//! Animate and render an OpenSCAD model — the `$t` loop OpenSCAD's animate
//! button drives, rendered headlessly to stills and video.
//!
//! ```text
//! cargo run --release --example scad_animate --features "openscad,video,native-codec"
//! cargo run --release --example scad_animate --features "openscad,video,native-codec" -- --frames 120
//! cargo run --release --example scad_animate --features "openscad,video,native-codec" -- --scad my.scad
//! cargo run --release --example scad_animate --features "openscad,video,native-codec" -- --palette blueprint --recolor
//! cargo run --release --example scad_animate --features "openscad,video,native-codec" -- --speed 0.5 --ping-pong
//! cargo run --release --example scad_animate --features "openscad,video,native-codec" -- --quality draft
//! ```
//!
//! | Flag | Effect |
//! |------|--------|
//! | `--palette NAME` | color scheme: `studio`, `cornfield`, `metallic`, `sunset`, `midnight`, `blueprint`, `nature`, `monochrome` |
//! | `--recolor` | ignore the model's own `color()` and use the palette instead |
//! | `--speed X` | playback multiplier — `2` twice as fast, `0.5` half |
//! | `--seconds S` | retime the loop to last `S` seconds |
//! | `--ping-pong` | run `$t` out and back rather than wrapping |
//! | `--ease NAME` | speed profile: `linear`, `in`, `out`, `inout`, `sine` |
//! | `--quality NAME` | `draft`, `balanced`, `fine` — the render speed/fidelity dial |
//! | `--light NAME` | lighting rig: `studio` (camera-relative three-point), `sun`, `flat` |
//! | `--no-shadows` | turn off the key light's shadows |
//!
//! Frames are evaluated several at a time (see `ScadAnimation::concurrency`),
//! which is most of the wall-clock win: on this model 24 frames take ~5s
//! instead of ~17s. `--features parallel` additionally parallelises the CSG
//! inside each frame.
//!
//! Writes into `./out/`:
//!
//! | File | What it shows |
//! |------|---------------|
//! | `scad_animate.mp4` | the `$t` loop, camera on a turntable |
//! | `scad_animate.gif` | the same, encoded in-process (no ffmpeg) |
//! | `scad_animate_still.png` | one frame at the default three-quarter view |
//! | `scad_animate_viewport.png` | framed by the model's own `$vpr`/`$vpt`/`$vpd` |
//! | `scad_animate_captioned.mp4` | with the frame's `$t` burned in as a caption |
//!
//! The model is `examples/scad_animate.scad`: a four-jaw chuck that closes on a
//! workpiece and opens again, with `color()` on every part.

use std::path::PathBuf;
use std::time::Instant;

use threers::openscad::animate::{
    ScadAnimation, ScadCamera, ScadColoring, ScadEasing, ScadLighting, ScadPalette, ScadQuality,
    ScadRender,
};

fn arg_value(flag: &str) -> Option<String> {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == flag {
            return args.next();
        }
    }
    None
}

fn has_flag(flag: &str) -> bool {
    std::env::args().any(|a| a == flag)
}

fn arg_number(flag: &str) -> Option<f64> {
    arg_value(flag).and_then(|s| s.parse().ok())
}

fn main() {
    let scad = arg_value("--scad").unwrap_or_else(|| "examples/scad_animate.scad".into());
    let frames: usize = arg_value("--frames")
        .and_then(|s| s.parse().ok())
        .unwrap_or(72);
    let (w, h) = (960u32, 540u32);
    let fps = 24;

    if !PathBuf::from(&scad).is_file() {
        eprintln!("no such model: {scad}");
        std::process::exit(2);
    }
    let _ = std::fs::create_dir_all("out");

    // `quality` is the speed/fidelity dial: Draft takes the cheap kernel and
    // coarse curves, Fine the robust kernel and fine ones. (Draft's float
    // kernel fails outright on this chuck, so those frames quietly fall back to
    // the robust one rather than rendering blank.)
    let quality = match arg_value("--quality").as_deref() {
        Some("draft") => ScadQuality::Draft,
        Some("fine") => ScadQuality::Fine,
        _ => ScadQuality::Balanced,
    };
    let easing = match arg_value("--ease").as_deref() {
        Some("in") => ScadEasing::EaseIn,
        Some("out") => ScadEasing::EaseOut,
        Some("inout") => ScadEasing::EaseInOut,
        Some("sine") => ScadEasing::Sine,
        _ => ScadEasing::Linear,
    };

    let mut animation = ScadAnimation::from_file(&scad)
        .frames(frames)
        .fps(fps)
        .quality(quality)
        .easing(easing)
        .ping_pong(has_flag("--ping-pong"))
        .speed(arg_number("--speed").unwrap_or(1.0))
        .on_progress(|p| {
            eprint!("\r{:<40}", p.message);
            let _ = std::io::Write::flush(&mut std::io::stderr());
        });
    if let Some(seconds) = arg_number("--seconds") {
        animation = animation.seconds(seconds);
    }
    // The encoder must use the rate the animation actually plays at.
    let fps = animation.frame_rate();

    let lighting = match arg_value("--light").as_deref() {
        Some("sun") => ScadLighting::sun(215.0, 38.0),
        Some("flat") => ScadLighting::flat(),
        _ => ScadLighting::studio(),
    }
    .shadows(!has_flag("--no-shadows"));

    let palette = arg_value("--palette")
        .and_then(|n| ScadPalette::by_name(&n))
        .unwrap_or_else(ScadPalette::studio);
    // Without `--recolor` the palette only dresses parts the model left
    // untagged, so a model that colors itself still looks the way it asked to.
    let coloring = if has_flag("--recolor") {
        ScadColoring::Palette
    } else {
        ScadColoring::ModelThenPalette
    };

    println!("model:    {scad}");
    println!(
        "frames:   {frames} @ {fps} fps ({:.2}s)",
        animation.duration()
    );
    println!(
        "quality:  {quality:?}   easing: {easing:?}   palette: {}   light: {:?}{}",
        palette.name,
        lighting.rig,
        if lighting.shadows { " +shadows" } else { "" }
    );
    println!(
        "animated: {} (a static model is evaluated once and shared)",
        animation.is_animated()
    );
    let probe = animation.frame(0).expect("evaluate frame 0");
    println!(
        "frame 0:  {} colored part(s), {} triangles",
        probe.parts.len(),
        probe.triangle_count()
    );

    // Everything below shares the same look; only the camera differs.
    let dressed = |camera| {
        ScadRender::new(w, h)
            .quality(quality)
            .palette(palette.clone())
            .coloring(coloring)
            .lighting(lighting.clone())
            .camera(camera)
    };

    let render = dressed(ScadCamera::turntable());

    // ---- one still, for a quick look ----
    let still = dressed(ScadCamera::auto());
    still
        .render_png(&animation, 0, "out/scad_animate_still.png")
        .expect("still");
    println!("\nwrote out/scad_animate_still.png");

    // ---- the model's own viewport, the way the OpenSCAD GUI would frame it ----
    let vp = animation.viewport_at(0.0);
    if vp.explicit {
        println!(
            "model sets $vp*: rot={:?} dist={}",
            vp.rotation, vp.distance
        );
    } else {
        println!("model sets no $vp* — Viewport falls back to the auto framing");
    }
    dressed(ScadCamera::Viewport)
        .render_png(&animation, 0, "out/scad_animate_viewport.png")
        .expect("viewport still");
    println!("wrote out/scad_animate_viewport.png");

    // ---- the animation, evaluated once and encoded twice ----
    let start = Instant::now();
    let rgba = render.render_frames(&mut animation).expect("render frames");
    eprintln!();
    println!(
        "rendered {} frames in {:.1}s ({:.2}s/frame)",
        rgba.len(),
        start.elapsed().as_secs_f32(),
        start.elapsed().as_secs_f32() / rgba.len() as f32
    );

    encode(&rgba, w, h, fps, "out/scad_animate.mp4");
    encode(&rgba, w, h, fps, "out/scad_animate.gif");

    // ---- and once more with the frame's `$t` burned in ----
    captioned(&rgba, w, h, fps, frames, "out/scad_animate_captioned.mp4");

    println!("done — see ./out/");
}

/// Encode already-rendered frames, choosing the codec from the extension.
fn encode(frames: &[Vec<u8>], w: u32, h: u32, fps: u32, out: &str) {
    use threers::{export_video, VideoCodec, VideoOptions};
    let codec = if out.ends_with(".gif") {
        VideoCodec::Gif
    } else {
        VideoCodec::H264
    };
    let mut options = VideoOptions::new(out).fps(fps).codec(codec);
    if codec == VideoCodec::Gif {
        options = options.gif_colors(128);
    } else {
        options = options.crf(20);
    }
    match export_video(w, h, frames.len(), &options, |i| frames[i].clone()) {
        Ok(()) => println!("wrote {out}"),
        Err(e) => eprintln!("skipped {out}: {e}"),
    }
}

/// The same frames with a caption track built from the animation clock, so the
/// value of `$t` is visible while the model moves.
fn captioned(frames: &[Vec<u8>], w: u32, h: u32, fps: u32, count: usize, out: &str) {
    use threers::captions::{CaptionAnchor, CaptionStyle};
    use threers::{export_video, CaptionTrack, VideoCodec, VideoOptions};

    // One cue per tenth of the loop.
    let mut track = CaptionTrack::new().language("en").label("Animation clock");
    let steps = 10;
    for i in 0..steps {
        let start = i as f64 * count as f64 / steps as f64 / fps as f64;
        let end = (i + 1) as f64 * count as f64 / steps as f64 / fps as f64;
        track.push(threers::Cue::new(
            start,
            end,
            format!("$t = {:.2}", i as f64 / steps as f64),
        ));
    }

    let options = VideoOptions::new(out)
        .fps(fps)
        .codec(VideoCodec::H264)
        .crf(20)
        .captions(track)
        .caption_style(
            CaptionStyle::default()
                .for_height(h)
                .anchor(CaptionAnchor::Top)
                .background([0, 0, 0, 130]),
        );
    match export_video(w, h, frames.len(), &options, |i| frames[i].clone()) {
        Ok(()) => println!("wrote {out}"),
        Err(e) => eprintln!("skipped {out}: {e}"),
    }
}
