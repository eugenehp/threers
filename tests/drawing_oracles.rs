//! Drawings checked against tools that read SVG and PDF independently: the
//! SVG through librsvg (`rsvg-convert`), the PDF through Ghostscript (`gs`),
//! poppler (`pdftoppm`, `pdftotext`) and qpdf (`qpdf --check`), each against
//! what `rasterize_svg` draws of the same document.
//!
//! Where the readers disagree among themselves the limits say so: poppler
//! grows an image's edges out to whole pixels (4% from librsvg, Ghostscript
//! and us alike on the image test), and every reader anti-aliases glyph edges
//! its own way, so text is compared blurred, where a glyph out of place still
//! shows and a different edge ramp does not.
//!
//! One document per feature, so a failure names the feature. Text is compared
//! through the PDF only: the PDF embeds the face the rasterizer draws with
//! (DejaVu Sans), while librsvg would set the SVG's text in whatever the
//! machine has under that name.
//!
//! Each check is skipped, and says so, when its tool is not installed.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use threers::exporters::{
    rasterize_svg, write_pdf, write_svg, PathCommand, RasterOptions, SvgDocument, SvgFillRule,
    SvgGradient, SvgGradientKind, SvgLineCap, SvgLineJoin, SvgNode, SvgPath, SvgStop,
    SvgStyle, SvgTextAnchor, SvgUnit,
};
use threers::utils::png::{decode_png, PngImage};

fn tool(name: &str) -> bool {
    let found = Command::new(name).arg("--version").output().is_ok()
        || Command::new(name).arg("-v").output().is_ok();
    if !found {
        eprintln!("skipping: {name} is not installed");
    }
    found
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("drawing_oracles");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

/// RGBA over white, as 0–1 RGB.
fn over_white(img: &PngImage) -> Vec<[f32; 3]> {
    img.rgba
        .chunks(4)
        .map(|p| {
            let a = p[3] as f32 / 255.0;
            let c = |v: u8| v as f32 / 255.0 * a + (1.0 - a);
            [c(p[0]), c(p[1]), c(p[2])]
        })
        .collect()
}

/// A 3×3 box blur, for comparing past anti-aliasing.
fn blurred(px: &[[f32; 3]], w: usize, h: usize) -> Vec<[f32; 3]> {
    let mut out = vec![[0.0; 3]; px.len()];
    for y in 0..h {
        for x in 0..w {
            let mut sum = [0.0f32; 3];
            let mut n = 0.0;
            for yy in y.saturating_sub(1)..(y + 2).min(h) {
                for xx in x.saturating_sub(1)..(x + 2).min(w) {
                    for i in 0..3 {
                        sum[i] += px[yy * w + xx][i];
                    }
                    n += 1.0;
                }
            }
            out[y * w + x] = sum.map(|v| v / n);
        }
    }
    out
}

/// RMS difference (0–1), and the share of pixels off by more than a quarter
/// in some channel; blurred first when `blur`.
fn compare(ours: &PngImage, theirs: &PngImage, blur: bool) -> (f32, f32) {
    assert_eq!((ours.width, ours.height), (theirs.width, theirs.height), "sizes");
    let (mut a, mut b) = (over_white(ours), over_white(theirs));
    if blur {
        let (w, h) = (ours.width as usize, ours.height as usize);
        (a, b) = (blurred(&a, w, h), blurred(&b, w, h));
    }
    let mut sum = 0.0f64;
    let mut far = 0usize;
    for (p, q) in a.iter().zip(&b) {
        let mut worst = 0.0f32;
        for i in 0..3 {
            let d = p[i] - q[i];
            sum += (d * d) as f64;
            worst = worst.max(d.abs());
        }
        if worst > 0.25 {
            far += 1;
        }
    }
    (((sum / (a.len() * 3) as f64).sqrt()) as f32, far as f32 / a.len() as f32)
}

/// Writes both images and a difference image next to each other for a
/// failure to be looked at.
fn keep(name: &str, ours: &PngImage, theirs: &PngImage) {
    let enc = |img: &PngImage| threers::utils::png::encode_png(img.width, img.height, &img.rgba);
    std::fs::write(scratch(&format!("{name}-ours.png")), enc(ours)).unwrap();
    std::fs::write(scratch(&format!("{name}-theirs.png")), enc(theirs)).unwrap();
}

fn ours(doc: &SvgDocument) -> PngImage {
    rasterize_svg(doc, &RasterOptions::default())
}

fn via_rsvg(name: &str, doc: &SvgDocument) -> Option<PngImage> {
    if !tool("rsvg-convert") {
        return None;
    }
    let svg = scratch(&format!("{name}.svg"));
    let png = scratch(&format!("{name}-rsvg.png"));
    std::fs::write(&svg, write_svg(doc)).unwrap();
    let out = Command::new("rsvg-convert").arg("-o").arg(&png).arg(&svg).output().unwrap();
    assert!(out.status.success(), "rsvg-convert: {}", String::from_utf8_lossy(&out.stderr));
    Some(decode_png(&std::fs::read(&png).unwrap()).unwrap())
}

fn via_poppler(name: &str, doc: &SvgDocument) -> Option<PngImage> {
    if !tool("pdftoppm") {
        return None;
    }
    let pdf = scratch(&format!("{name}.pdf"));
    std::fs::write(&pdf, write_pdf(doc)).unwrap();
    let stem = scratch(&format!("{name}-poppler"));
    let out = Command::new("pdftoppm")
        .args(["-png", "-r", "96", "-singlefile", "-aa", "yes", "-aaVector", "yes"])
        .arg(&pdf)
        .arg(&stem)
        .output()
        .unwrap();
    assert!(out.status.success(), "pdftoppm: {}", String::from_utf8_lossy(&out.stderr));
    let png = stem.with_extension("png");
    Some(decode_png(&std::fs::read(&png).unwrap()).unwrap())
}

fn via_ghostscript(name: &str, doc: &SvgDocument) -> Option<PngImage> {
    if !tool("gs") {
        return None;
    }
    let pdf = scratch(&format!("{name}-gs.pdf"));
    std::fs::write(&pdf, write_pdf(doc)).unwrap();
    let png = scratch(&format!("{name}-gs.png"));
    let out = Command::new("gs")
        // At 4× and averaged down: at 1× its 4-bit anti-aliasing and its
        // fill adjustment (shapes grown by a fraction of a pixel) differ from
        // every other reader at every edge.
        .args(["-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=pngalpha", "-r384"])
        .args(["-dTextAlphaBits=1", "-dGraphicsAlphaBits=1"])
        .arg(format!("-sOutputFile={}", png.display()))
        .arg(&pdf)
        .output()
        .unwrap();
    assert!(out.status.success(), "gs: {}", String::from_utf8_lossy(&out.stderr));
    Some(quarter(&decode_png(&std::fs::read(&png).unwrap()).unwrap()))
}

/// Each 4×4 block averaged, premultiplied, into one pixel.
fn quarter(img: &PngImage) -> PngImage {
    let (w, h) = (img.width / 4, img.height / 4);
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let mut sum = [0.0f32; 4];
            for dy in 0..4 {
                for dx in 0..4 {
                    let p = img.pixel(x * 4 + dx, y * 4 + dy);
                    let a = p[3] as f32 / 255.0;
                    for i in 0..3 {
                        sum[i] += p[i] as f32 * a;
                    }
                    sum[3] += a;
                }
            }
            let a = sum[3] / 16.0;
            let c = |v: f32| if sum[3] > 0.0 { (v / sum[3]).round() as u8 } else { 0 };
            rgba.extend_from_slice(&[c(sum[0]), c(sum[1]), c(sum[2]), (a * 255.0).round() as u8]);
        }
    }
    PngImage { width: w, height: h, rgba }
}

fn qpdf_check(name: &str, doc: &SvgDocument) {
    if !tool("qpdf") {
        return;
    }
    let pdf = scratch(&format!("{name}-check.pdf"));
    std::fs::write(&pdf, write_pdf(doc)).unwrap();
    let out = Command::new("qpdf").arg("--check").arg(&pdf).output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success() && !text.contains("WARNING"), "qpdf --check {name}:\n{text}");
}

/// How close each reader must come: RMS difference and share of pixels far
/// off, `None` to skip a reader.
#[derive(Clone, Copy)]
struct Limits {
    librsvg: Option<(f32, f32)>,
    ghostscript: (f32, f32),
    poppler: (f32, f32),
    /// Compare blurred (text: past each reader's own edge anti-aliasing).
    blur: bool,
}

/// The RMS limits allow for edges: every reader places a shape's edge pixels
/// by its own rules (Ghostscript's are 2% from ours over a drawing of thin
/// strokes, librsvg's 0.3%). The far-off limit is what catches a wrong
/// shape, colour or place: it stays near zero against all three.
const SHAPES: Limits = Limits {
    librsvg: Some((0.006, 0.001)),
    ghostscript: (0.025, 0.002),
    poppler: (0.015, 0.002),
    blur: false,
};

/// Checks `doc` against every tool that can read it.
fn check(name: &str, doc: &SvgDocument, limits: Limits) {
    let mine = ours(doc);
    qpdf_check(name, doc);
    let readers = [
        ("librsvg", limits.librsvg.and_then(|l| via_rsvg(name, doc).map(|t| (l, t)))),
        ("ghostscript", via_ghostscript(name, doc).map(|t| (limits.ghostscript, t))),
        ("poppler", via_poppler(name, doc).map(|t| (limits.poppler, t))),
    ];
    for (reader, got) in readers {
        let Some(((max_rmse, max_far), theirs)) = got else { continue };
        let (rmse, far) = compare(&mine, &theirs, limits.blur);
        eprintln!("{name} vs {reader}: rmse {:.3}%, far {:.3}%", rmse * 100.0, far * 100.0);
        if rmse > max_rmse || far > max_far {
            keep(&format!("{name}-{reader}"), &mine, &theirs);
            panic!("{name}: {reader} differs: rmse {:.3}%, {:.3}% of pixels far off", rmse * 100.0, far * 100.0);
        }
    }
}

fn page() -> SvgDocument {
    let mut doc = SvgDocument::new(320.0, 240.0, SvgUnit::Pixels);
    doc.precision = None;
    doc
}

fn star(c: [f64; 2], r: f64) -> SvgPath {
    let pts: Vec<[f64; 2]> = (0..5)
        .map(|i| {
            let a = (i as f64 * 144.0 - 90.0).to_radians();
            [c[0] + r * a.cos(), c[1] + r * a.sin()]
        })
        .collect();
    SvgPath::polygon(&pts, true)
}

#[test]
fn basics_fills_strokes_dashes_and_opacity() {
    let mut doc = page();
    doc.rect([10.0, 10.0], [80.0, 50.0], 8.0, SvgStyle::fill([200, 60, 40]));
    doc.circle([150.0, 40.0], 30.0, SvgStyle::fill([40, 120, 200]).with_opacity(0.6));
    doc.path(star([250.0, 45.0], 35.0), SvgStyle::fill([30, 160, 80]).with_fill_rule(SvgFillRule::EvenOdd));
    doc.path(star([250.0, 45.0], 35.0), SvgStyle::stroke([20, 20, 20], 2.0));
    for (i, (cap, join)) in [
        (SvgLineCap::Butt, SvgLineJoin::Miter),
        (SvgLineCap::Round, SvgLineJoin::Round),
        (SvgLineCap::Square, SvgLineJoin::Bevel),
    ]
    .into_iter()
    .enumerate()
    {
        let x = 20.0 + i as f64 * 100.0;
        let mut style = SvgStyle::stroke([60, 60, 160], 10.0);
        style.line_cap = Some(cap);
        style.line_join = Some(join);
        doc.polyline(&[[x, 150.0], [x + 30.0, 100.0], [x + 60.0, 150.0]], false, style);
    }
    doc.line([20.0, 200.0], [300.0, 200.0], SvgStyle::stroke([180, 40, 120], 4.0).with_dash(&[12.0, 6.0, 2.0, 6.0]));
    let mut curve = SvgPath::new();
    curve.move_to([20.0, 225.0]).cubic_to([90.0, 180.0], [160.0, 260.0], [230.0, 215.0]).quad_to([270.0, 190.0], [300.0, 230.0]);
    doc.path(curve, SvgStyle::stroke([0, 0, 0], 1.5).with_opacity(0.8));
    check("basics", &doc, SHAPES);
}

#[test]
fn gradients_linear_radial_and_stop_opacity() {
    let mut doc = page();
    let lin = doc.gradient(SvgGradient {
        kind: SvgGradientKind::Linear { start: [10.0, 0.0], end: [150.0, 0.0] },
        stops: vec![
            SvgStop { offset: 0.0, color: [255, 0, 0], opacity: 1.0 },
            SvgStop { offset: 0.5, color: [255, 255, 0], opacity: 0.5 },
            SvgStop { offset: 1.0, color: [0, 0, 255], opacity: 1.0 },
        ],
        transform: None,
    });
    doc.rect([10.0, 10.0], [140.0, 100.0], 0.0, SvgStyle { fill: Some(lin), ..SvgStyle::default() });
    let rad = doc.gradient(SvgGradient {
        kind: SvgGradientKind::Radial { center: [240.0, 60.0], radius: 55.0, focus: Some([220.0, 45.0]) },
        stops: vec![
            SvgStop { offset: 0.0, color: [255, 255, 255], opacity: 1.0 },
            SvgStop { offset: 1.0, color: [20, 80, 40], opacity: 1.0 },
        ],
        transform: None,
    });
    doc.circle([240.0, 60.0], 50.0, SvgStyle { fill: Some(rad), ..SvgStyle::default() });
    let turned = doc.gradient(SvgGradient {
        kind: SvgGradientKind::Linear { start: [0.0, 0.0], end: [60.0, 0.0] },
        stops: vec![
            SvgStop { offset: 0.0, color: [0, 0, 0], opacity: 0.0 },
            SvgStop { offset: 1.0, color: [120, 0, 160], opacity: 1.0 },
        ],
        transform: Some([0.5, 0.866, -0.866, 0.5, 60.0, 130.0]),
    });
    doc.rect([20.0, 130.0], [120.0, 90.0], 10.0, SvgStyle { fill: Some(turned), ..SvgStyle::default() });
    let mut stroked = SvgStyle::stroke([0, 0, 0], 14.0);
    stroked.stroke = Some(lin);
    doc.circle([240.0, 175.0], 40.0, stroked);
    check("gradients", &doc, SHAPES);
}

fn test_image() -> Arc<PngImage> {
    let (w, h) = (24u32, 16u32);
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let a = if (x / 4 + y / 4) % 2 == 0 { 255 } else { 96 };
            rgba.extend_from_slice(&[(x * 10) as u8, (y * 15) as u8, 180, a]);
        }
    }
    Arc::new(PngImage { width: w, height: h, rgba })
}

#[test]
fn images_with_alpha_transform_and_opacity() {
    let mut doc = page();
    doc.rect([0.0, 0.0], [320.0, 240.0], 0.0, SvgStyle::fill([230, 230, 200]));
    let image = test_image();
    // Whole texels: 24×16 drawn at 10× and 5×, so filtering doesn't decide.
    doc.children.push(SvgNode::Image { origin: [10.0, 10.0], size: [240.0, 160.0], image: image.clone(), style: SvgStyle::default() });
    doc.children.push(SvgNode::Image {
        origin: [180.0, 120.0],
        size: [120.0, 80.0],
        image,
        style: SvgStyle { opacity: Some(0.5), ..SvgStyle::default() },
    });
    // poppler grows image edges out to whole pixels; the others agree.
    check("images", &doc, Limits { poppler: (0.05, 0.006), ..SHAPES });
}

#[test]
fn clips_nested_group_opacity_and_transforms() {
    let mut doc = page();
    let clip = doc.clip_path(star([160.0, 120.0], 110.0), SvgFillRule::EvenOdd);
    let g = doc.group(Some("clipped"), None, SvgStyle { clip: Some(clip), ..SvgStyle::default() });
    for i in 0..8 {
        g.push(SvgNode::Rect {
            origin: [0.0, i as f64 * 30.0],
            size: [320.0, 15.0],
            corner_radius: 0.0,
            style: SvgStyle::fill([40, 40 + i * 25, 200]),
        });
    }
    // Overlapping children composited once at the group's opacity: no
    // darker lens where they cross.
    let g = doc.group(Some("faded"), Some([0.866, 0.5, -0.5, 0.866, 60.0, -10.0]), SvgStyle { opacity: Some(0.5), ..SvgStyle::default() });
    g.push(SvgNode::Circle { center: [100.0, 120.0], radius: 45.0, style: SvgStyle::fill([220, 30, 30]) });
    g.push(SvgNode::Circle { center: [150.0, 120.0], radius: 45.0, style: SvgStyle::fill([220, 30, 30]) });
    g.push(SvgNode::Group {
        id: None,
        transform: Some([1.5, 0.0, 0.0, 0.5, 0.0, 150.0]),
        style: SvgStyle { opacity: Some(0.5), ..SvgStyle::default() },
        children: vec![SvgNode::Rect { origin: [40.0, 0.0], size: [80.0, 80.0], corner_radius: 20.0, style: SvgStyle::fill([0, 0, 0]) }],
    });
    check("clips_groups", &doc, SHAPES);
}

#[test]
fn y_up_documents_flip_alike() {
    let mut doc = page();
    doc.y_up = true;
    doc.rect([10.0, 10.0], [100.0, 40.0], 0.0, SvgStyle::fill([200, 0, 0]));
    let mut arc = SvgPath::new();
    arc.move_to([200.0, 50.0]);
    arc.commands.push(PathCommand::ArcTo { radii: [60.0, 30.0], rotation: 30.0, large_arc: true, sweep: false, to: [280.0, 150.0] });
    doc.path(arc, SvgStyle::stroke([0, 0, 200], 3.0));
    check("y_up", &doc, SHAPES);
}

#[test]
fn text_in_the_pdf_is_the_text_rasterized_and_extracts() {
    let mut doc = page();
    let mut put = |pos: [f64; 2], text: &str, size: f64, anchor: SvgTextAnchor, rotation: f64, family: Option<&str>| {
        doc.children.push(SvgNode::Text {
            pos,
            text: text.into(),
            size,
            anchor,
            rotation,
            font_family: family.map(Into::into),
            style: SvgStyle::fill([10, 10, 60]),
        });
    };
    put([10.0, 30.0], "Start AV Wavy Täxt", 22.0, SvgTextAnchor::Start, 0.0, None);
    put([160.0, 70.0], "Middle — ünïcödé ✓", 18.0, SvgTextAnchor::Middle, 0.0, Some("sans-serif"));
    put([310.0, 110.0], "End 0123", 16.0, SvgTextAnchor::End, 0.0, Some("monospace"));
    put([60.0, 220.0], "Turned", 20.0, SvgTextAnchor::Start, -30.0, Some("serif"));
    let text = Limits { librsvg: None, ghostscript: (0.015, 0.002), poppler: (0.015, 0.002), blur: true };
    check("text", &doc, text);
    if tool("pdftotext") {
        let pdf = scratch("text-extract.pdf");
        std::fs::write(&pdf, write_pdf(&doc)).unwrap();
        // `-raw`: in content-stream order. Without it pdftotext cuts text at
        // angles other than right ones into letters, from any producer (cairo too).
        let out = Command::new("pdftotext").arg("-raw").arg(&pdf).arg("-").output().unwrap();
        let text = String::from_utf8_lossy(&out.stdout);
        for want in ["Start AV Wavy Täxt", "Middle — ünïcödé ✓", "End 0123", "Turned"] {
            assert!(text.contains(want), "pdftotext lacks {want:?}:\n{text}");
        }
    }
}

/// Several pages in one file, each its own size.
#[test]
fn multi_page_pdfs_check_clean() {
    if !tool("qpdf") {
        return;
    }
    let mut a = page();
    a.circle([100.0, 100.0], 50.0, SvgStyle::fill([0, 100, 0]));
    let mut b = SvgDocument::new(210.0, 297.0, SvgUnit::Millimeters);
    b.text([20.0, 20.0], "A4 page", 10.0, SvgTextAnchor::Start, SvgStyle::fill([0, 0, 0]));
    let pdf = scratch("pages.pdf");
    std::fs::write(&pdf, threers::exporters::write_pdf_pages(&[a, b])).unwrap();
    let out = Command::new("qpdf").arg("--check").arg(&pdf).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stdout));
    let out = Command::new("qpdf").args(["--show-npages"]).arg(&pdf).output().unwrap();
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "2");
}

/// A 3D scene drawn without a GPU: `SvgRenderer::render_to_document` is the
/// drawing the renderer's own markup describes, so librsvg reading that
/// markup must see what we draw from the document, and the PDF readers must
/// too.
#[test]
fn a_lit_scene_through_every_reader() {
    use threers::prelude::*;
    use threers::renderers::SvgRenderer;
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x202830);
    scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.3));
    let mut key = Object3D::light(DirectionalLight::new(Color::from_hex(0xffffff), 1.0));
    key.position = Vector3::new(3.0, 4.0, 5.0);
    scene.add(key);
    let mut cube = Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.6, 1.6, 1.6),
        StandardMaterial::new(Color::from_hex(0xc04030)).into(),
    ));
    cube.position = Vector3::new(-1.2, 0.0, 0.0);
    cube.quaternion = Quaternion::from_axis_angle(Vector3::new(0.3, 1.0, 0.2).normalize(), 0.7);
    scene.add(cube);
    let mut ball = Object3D::mesh(Mesh::new(
        SphereGeometry::new(0.9, 32, 20),
        StandardMaterial::new(Color::from_hex(0x3080c0)).into(),
    ));
    ball.position = Vector3::new(1.3, 0.0, 0.0);
    scene.add(ball);
    let mut camera = PerspectiveCamera::new(45.0, 4.0 / 3.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 1.5, 6.0);
    camera.look_at(Vector3::new(0.0, 0.0, 0.0));

    let renderer = SvgRenderer::new(320, 240);
    let doc = renderer.render_to_document(&mut scene, &camera);
    let mine = ours(&doc);
    if tool("rsvg-convert") {
        let svg = scratch("scene-own.svg");
        let png = scratch("scene-own-rsvg.png");
        std::fs::write(&svg, renderer.render_to_string(&mut scene, &camera)).unwrap();
        let out = Command::new("rsvg-convert").arg("-o").arg(&png).arg(&svg).output().unwrap();
        assert!(out.status.success());
        let theirs = decode_png(&std::fs::read(&png).unwrap()).unwrap();
        let (rmse, far) = compare(&mine, &theirs, false);
        eprintln!("scene: its own markup through librsvg vs our PNG: rmse {:.3}%, far {:.3}%", rmse * 100.0, far * 100.0);
        if rmse > 0.006 || far > 0.001 {
            keep("scene-own-rsvg", &mine, &theirs);
            panic!("scene: librsvg on the renderer's markup differs from our PNG of its document");
        }
    }
    check("scene", &doc, SHAPES);
}
