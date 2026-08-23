//! Subtitles and captions on an exported video, in every delivery mode.
//!
//! ```text
//! cargo run --release --example captions_video --features "video,native-codec"
//! cargo run --release --example captions_video --features "video,native-codec" -- --font /path/to/Font.ttf
//! ```
//!
//! Writes into `./out/`:
//!
//! | File | Mode | What you get |
//! |------|------|--------------|
//! | `captions_burned.gif` | [`CaptionMode::Burn`] | text painted into the pixels — no player support needed |
//! | `captions_sidecar.gif` + `.srt` | [`CaptionMode::Sidecar`] | clean video, subtitles in a separate file |
//! | `captions_soft.mp4` | [`CaptionMode::Embed`] | a selectable `mov_text` track the viewer can toggle |
//! | `captions_soft.webm` | [`CaptionMode::Embed`] | the same as a WebVTT track |
//!
//! The MP4/WebM outputs need `ffmpeg` on `PATH`; the GIFs are encoded in-process.

use std::f32::consts::TAU;
use std::path::PathBuf;

use threers::captions::{CaptionAnchor, CaptionStyle};
use threers::{
    AmbientLight, BoxGeometry, CaptionMode, CaptionTrack, Color, DirectionalLight, Euler,
    HeadlessRenderer, Mesh, Object3D, PerspectiveCamera, Scene, StandardMaterial, Vector3,
    VideoCodec, VideoError, VideoExporter,
};

const W: u32 = 480;
const H: u32 = 270;
const FPS: u32 = 15;
const FRAMES: usize = 90; // six seconds

/// Optional `--font PATH` for a TrueType face; without it the built-in bitmap
/// face is used, which needs no assets at all.
fn font_arg() -> Option<PathBuf> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--font" {
            return Some(PathBuf::from(args.next().expect("--font needs a path")));
        }
    }
    None
}

/// Dialogue for the clip, authored as WebVTT so cue placement comes along too.
fn track() -> CaptionTrack {
    CaptionTrack::parse_vtt(
        "WEBVTT - Cube commentary [en]\n\
         \n\
         1\n\
         00:00:00.000 --> 00:00:02.000\n\
         A cube begins to turn.\n\
         \n\
         2\n\
         00:00:02.000 --> 00:00:04.000\n\
         Captions wrap automatically when a line\n\
         runs past the safe area.\n\
         \n\
         3\n\
         00:00:04.000 --> 00:00:06.000 align:left line:6%\n\
         Cue settings can move a caption out of the way.\n",
    )
    .expect("parse dialogue")
    .language("en")
    .label("English")
}

struct Clip {
    renderer: HeadlessRenderer,
    scene: Scene,
    camera: PerspectiveCamera,
    cube: threers::ObjectId,
}

impl Clip {
    fn new() -> Self {
        let renderer = HeadlessRenderer::builder()
            .size(W, H)
            .build()
            .expect("headless renderer (is a GPU adapter available?)");

        let mut scene = Scene::new();
        scene.background = Color::new(0.05, 0.06, 0.09);
        scene.add_light(AmbientLight::new(Color::WHITE, 0.3));
        scene.add_light(
            DirectionalLight::new(Color::WHITE, 2.2)
                .with_direction(Vector3::new(-0.4, -0.8, -0.5).normalize()),
        );
        let mut material = StandardMaterial::new(Color::new(0.20, 0.62, 0.95));
        material.metalness = 0.15;
        material.roughness = 0.35;
        let cube = scene.add(Object3D::mesh(Mesh::new(
            BoxGeometry::new(1.6, 1.6, 1.6),
            material.into(),
        )));

        let mut camera = PerspectiveCamera::new(50.0, W as f32 / H as f32, 0.1, 100.0);
        camera.position = Vector3::new(0.0, 1.4, 4.2);
        camera.look_at(Vector3::ZERO);

        Self {
            renderer,
            scene,
            camera,
            cube,
        }
    }

    /// Render frame `index` of the spin.
    fn frame(&mut self, index: usize) -> Vec<u8> {
        let angle = index as f32 / FRAMES as f32 * TAU;
        if let Some(o) = self.scene.get_mut(self.cube) {
            o.quaternion = Euler::new(0.5, angle, 0.0).to_quaternion();
        }
        self.renderer.render_to_rgba(&mut self.scene, &self.camera)
    }
}

/// Export one variant, reporting ffmpeg-only failures as skips rather than
/// killing the run.
fn export(label: &str, exporter: VideoExporter, clip: &mut Clip) {
    match exporter.export(|i| clip.frame(i)) {
        Ok(()) => println!("  {label}: ok"),
        Err(e @ VideoError::Spawn(_)) => {
            println!("  {label}: skipped — {e}");
        }
        Err(e) => {
            eprintln!("  {label}: FAILED — {e}");
            std::process::exit(1);
        }
    }
}

fn main() {
    let _ = std::fs::create_dir_all("out");
    let font = font_arg().map(|p| std::fs::read(&p).expect("read --font file"));
    if font.is_some() {
        println!("using the supplied TrueType face");
    } else {
        println!("using the built-in bitmap face (pass --font FILE.ttf for real typography)");
    }

    // The default style is authored for 1080p and rescales itself to the export
    // height, so this is only here to show the knobs.
    let style = CaptionStyle::default()
        .for_height(H)
        .anchor(CaptionAnchor::Bottom)
        .background([0, 0, 0, 150])
        .outline([0, 0, 0, 255], 2.0);

    let mut clip = Clip::new();
    let base = |out: &str, codec: VideoCodec| {
        let mut e = VideoExporter::new(format!("out/{out}"))
            .size(W, H)
            .frames(FRAMES)
            .fps(FPS)
            .codec(codec)
            .captions(track())
            .caption_style(style.clone());
        if let Some(bytes) = font.clone() {
            e = e.caption_font(bytes);
        }
        e
    };

    println!("exporting {W}×{H}, {FRAMES} frames @ {FPS} fps…");

    // 1. Burned in — works with every codec, including GIF.
    export(
        "captions_burned.gif",
        base("captions_burned.gif", VideoCodec::Gif)
            .gif_colors(128)
            .caption_mode(CaptionMode::Burn),
        &mut clip,
    );

    // 2. Sidecar — clean pixels, subtitles in `out/captions_sidecar.srt`.
    export(
        "captions_sidecar.gif + .srt",
        base("captions_sidecar.gif", VideoCodec::Gif)
            .gif_colors(128)
            .caption_mode(CaptionMode::Sidecar),
        &mut clip,
    );

    // 3. Embedded soft subtitles — a track the viewer can switch off.
    export(
        "captions_soft.mp4",
        base("captions_soft.mp4", VideoCodec::H264)
            .crf(28)
            .caption_mode(CaptionMode::Embed),
        &mut clip,
    );
    export(
        "captions_soft.webm",
        base("captions_soft.webm", VideoCodec::Vp9)
            .crf(34)
            .caption_mode(CaptionMode::Embed),
        &mut clip,
    );

    println!("done — see ./out/");
}
