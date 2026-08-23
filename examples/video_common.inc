// Shared headless spinning-cube scene for video export examples.
// Included via `include!("video_common.inc")` from each `export_*` example.

use std::f32::consts::TAU;
use std::path::PathBuf;

use threers::{
    AmbientLight, BoxGeometry, Color, DirectionalLight, Euler, HeadlessRenderer, Mesh, Object3D,
    PerspectiveCamera, Scene, StandardMaterial, Vector3, VideoCodec, VideoExportEvent,
    VideoExporter,
};

/// Parse `--out PATH` from argv; otherwise return `default`.
fn out_path(default: &str) -> PathBuf {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--out" {
            return PathBuf::from(args.next().expect("--out needs a path"));
        }
    }
    PathBuf::from(default)
}

/// Ensure parent directories exist for `path`.
fn ensure_parent(path: &std::path::Path) {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
}

/// Render a short spinning-cube clip and encode with `export_video`.
///
/// `width`/`height` are logical size (no supersample). Prefer multiples of 8
/// when the codec is VP9/WebM.
fn run_export(
    codec: VideoCodec,
    default_out: &str,
    transparent: bool,
    gif_colors: Option<u16>,
) {
    let out = out_path(default_out);
    ensure_parent(&out);

    let (w, h) = (320u32, 240u32);
    let mut hr = HeadlessRenderer::builder()
        .size(w, h)
        .supersample(1)
        .build()
        .expect("headless renderer");
    let (rw, rh) = hr.render_size();

    let mut scene = Scene::new();
    if transparent {
        scene.background = Color::new(0.0, 0.0, 0.0);
        // Fully transparent clear — HeadlessRenderer still writes RGBA; zero alpha
        // on background pixels depends on material/clear. Use a dark opaque bg for
        // codecs that ignore alpha, and keep a simple opaque cube for all formats.
    } else {
        scene.background = Color::new(0.05, 0.06, 0.09);
    }
    scene.add_light(AmbientLight::new(Color::WHITE, 0.3));
    scene.add_light(
        DirectionalLight::new(Color::WHITE, 2.2)
            .with_direction(Vector3::new(-0.4, -0.8, -0.5).normalize()),
    );
    let mut mat = StandardMaterial::new(Color::new(0.20, 0.62, 0.95));
    mat.metalness = 0.15;
    mat.roughness = 0.35;
    if transparent {
        mat.opacity = 0.85;
    }
    let cube = scene.add(Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.6, 1.6, 1.6),
        mat.into(),
    )));

    let mut cam = PerspectiveCamera::new(50.0, w as f32 / h as f32, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 1.4, 4.2);
    cam.look_at(Vector3::ZERO);

    let frames = 45usize;
    let fps = 15u32;
    let mut exporter = VideoExporter::new(out.to_string_lossy().into_owned())
        .size(rw, rh)
        .frames(frames)
        .fps(fps)
        .codec(codec)
        .transparent(transparent)
        .on_progress(|p| {
            eprint!("\r{:<60}", p.message);
            let _ = std::io::Write::flush(&mut std::io::stderr());
        })
        .on_event(|ev| {
            if let VideoExportEvent::Complete { output, .. } = ev {
                eprintln!("\rwrote {output}");
            }
        });
    if matches!(codec, VideoCodec::H264 | VideoCodec::Hevc | VideoCodec::Vp9) {
        exporter = exporter.crf(28);
    }
    if let Some(n) = gif_colors {
        exporter = exporter.gif_colors(n);
    }

    println!(
        "exporting {:?} → {} ({}×{}, {} frames @ {} fps, transparent={})…",
        codec,
        out.display(),
        rw,
        rh,
        frames,
        fps,
        transparent
    );

    match exporter.export(|f| {
        let angle = f as f32 / frames as f32 * TAU;
        if let Some(o) = scene.get_mut(cube) {
            o.quaternion = Euler::new(0.5, angle, 0.0).to_quaternion();
        }
        let mut rgba = hr.render_to_rgba(&mut scene, &cam);
        if transparent {
            // Punch a checker of fully transparent texels in the corners so GIF/WebM/APNG
            // alpha paths have something to encode.
            for y in 0..rh.min(40) {
                for x in 0..rw.min(40) {
                    let i = ((y * rw + x) * 4) as usize;
                    rgba[i + 3] = 0;
                }
            }
        }
        rgba
    }) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("export failed: {e}");
            eprintln!("hint: ffmpeg-backed codecs need `ffmpeg` on PATH; GIF/APNG need --features native-codec");
            std::process::exit(1);
        }
    }
}
