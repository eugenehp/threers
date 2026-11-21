//! A UI drawn without a GPU as a vector drawing (`threers::ui::vector`): the
//! boxes are where layout put them, text measured is text drawn, clips clip,
//! and the PDF and SVG show what the PNG does to independent readers
//! (Ghostscript and poppler for the PDF, librsvg for the SVG), its text
//! extracting as text.
//!
//! The reader checks skip, saying so, when a tool is not installed.

use std::path::{Path, PathBuf};
use std::process::Command;

use threers::exporters::{write_pdf, write_svg, DrawingFonts, SvgDocument};
use threers::style;
use threers::ui::image::Image;
use threers::ui::vector::{ui_to_document, ui_to_image, ui_to_pdf};
use threers::ui::{NodeId, Ui};
use threers::utils::png::{decode_png, PngImage};

const W: f32 = 480.0;
const H: f32 = 320.0;

struct Nodes {
    page: [u8; 3],
    title: NodeId,
    card: NodeId,
    clipper: NodeId,
    overflow: NodeId,
    italic: NodeId,
    struck: NodeId,
}

/// A page with most of what a UI paints: a gradient header with bold text,
/// a bordered, rounded, shadowed card, a rounded clip with a child spilling
/// out of it, radial and conic gradients, an icon, an image, italic and
/// struck-through text.
fn dashboard(ui: &mut Ui, with_text: bool) -> Nodes {
    let root = ui.root();
    ui.set_style(root, style! { display: flex; flex-direction: column; gap: 12px; padding: 16px; background: #f0f0ea; color: #1e293b; font-size: 15px; });
    let header = ui.add_box(root, style! { height: 48px; padding: 10px; border-radius: 10px; background: linear-gradient(90deg, #1d4fd8, #9333fa); color: #ffffff; font-size: 20px; font-weight: bold; });
    let title = ui.add_text(header, if with_text { "Quarterly report" } else { "" });
    let row = ui.add_box(root, style! { display: flex; flex-direction: row; gap: 16px; });
    let card = ui.add_box(row, style! { width: 130px; height: 100px; padding: 10px; background: #ffffff; border: 2px solid #1e293b; border-radius: 14px; box-shadow: 0 6px 12px rgba(0,0,0,0.3); });
    let italic = ui.add_text(card, if with_text { "Revenue up" } else { "" });
    ui.update_style(italic, |s| s.italic = true);
    let clipper = ui.add_box(row, style! { width: 100px; height: 100px; border-radius: 30px; overflow: hidden; background: radial-gradient(circle at 30% 30%, #fde68a, #b45309); });
    let overflow = ui.add_box(clipper, style! { width: 200px; height: 40px; margin-top: 70px; background: #dc2626; });
    ui.add_box(row, style! { width: 100px; height: 100px; border-radius: 50px; background: conic-gradient(from 0deg, #ef4444, #eab308, #22c55e, #3b82f6, #ef4444); });
    let lower = ui.add_box(root, style! { display: flex; flex-direction: row; gap: 16px; align-items: center; });
    ui.add_path(lower, style! { width: 32px; height: 32px; color: #0f766e; }, vec!["M4 4 L28 28 M28 4 L4 28".into(), "M16 2 a14 14 0 1 0 0.01 0".into()], [0.0, 0.0, 32.0, 32.0], Some(3.0), false);
    let mut rgba = vec![];
    for y in 0..8u32 {
        for x in 0..8u32 {
            rgba.extend_from_slice(if (x + y) % 2 == 0 { &[20, 20, 20, 255] } else { &[230, 200, 40, 255] });
        }
    }
    ui.add_image(lower, style! { width: 64px; height: 64px; }, Image::new(8, 8, rgba).unwrap());
    let struck = ui.add_text(lower, if with_text { "was 42, now 57" } else { "" });
    ui.update_style(struck, |s| s.line_through = true);
    Nodes { page: [0xf0, 0xf0, 0xea], title, card, clipper, overflow, italic, struck }
}

fn px(img: &PngImage, x: f32, y: f32) -> [u8; 3] {
    let p = img.pixel(x as u32, y as u32);
    [p[0], p[1], p[2]]
}

fn near(a: [u8; 3], b: [u8; 3], tol: u8) -> bool {
    (0..3).all(|i| a[i].abs_diff(b[i]) <= tol)
}

#[test]
fn boxes_land_where_layout_put_them_and_clips_clip() {
    let mut ui = Ui::new();
    let n = dashboard(&mut ui, true);
    let fonts = DrawingFonts::new();
    let img = ui_to_image(&mut ui, W, H, 1.0, &fonts);
    assert_eq!((img.width, img.height), (W as u32, H as u32));
    // Inside the card, away from its text: its white.
    let card = ui.rect(n.card);
    assert_eq!(px(&img, card.x + card.width - 12.0, card.y + card.height - 12.0), [255, 255, 255]);
    // Its border, dark; just outside its rounded corner, no card.
    assert!(near(px(&img, card.x + card.width / 2.0, card.y + 1.0), [0x1e, 0x29, 0x3b], 8));
    let corner = px(&img, card.x + 1.5, card.y + 1.5);
    assert!(!near(corner, [255, 255, 255], 30) && !near(corner, [0x1e, 0x29, 0x3b], 30), "corner {corner:?}");
    // The shadow darkens the page under the card's bottom edge, fading out.
    let below = |d: f32| px(&img, card.x + card.width / 2.0, card.y + card.height + d)[0];
    assert!(below(3.0) < below(9.0) && below(9.0) <= n.page[0], "shadow {} {}", below(3.0), below(9.0));
    // The red child shows inside its clip and not past its edge.
    let clip = ui.rect(n.clipper);
    let child = ui.rect(n.overflow);
    assert!(near(px(&img, clip.x + clip.width / 2.0, child.y + 10.0), [0xdc, 0x26, 0x26], 4));
    assert!(child.x + child.width > clip.x + clip.width + 20.0, "the child spills out");
    let past = px(&img, clip.x + clip.width + 6.0, child.y + 10.0);
    assert!(!near(past, [0xdc, 0x26, 0x26], 40), "drawn past its clip: {past:?}");
}

/// Ink (pixels unlike the background) of `rect`'s surroundings, as columns.
fn ink_columns(img: &PngImage, rect: threers::ui::Rect, background: [u8; 3]) -> Option<(f32, f32)> {
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for y in (rect.y as u32).saturating_sub(4)..(rect.y + rect.height) as u32 + 4 {
        for x in (rect.x as u32).saturating_sub(20)..(rect.x + rect.width) as u32 + 20 {
            if !near(px(img, x as f32, y as f32), background, 60) {
                lo = lo.min(x as f32);
                hi = hi.max(x as f32 + 1.0);
            }
        }
    }
    (lo <= hi).then_some((lo, hi))
}

#[test]
fn text_measured_for_layout_is_the_text_drawn() {
    for (size, text) in [(11.0, "Tiny print, kerned AV"), (16.0, "A wider line of body text"), (28.0, "Headline Wo")] {
        let mut ui = Ui::new();
        let root = ui.root();
        // A flex row: the text keeps its own width rather than the row's.
        ui.set_style(root, style! { display: flex; align-items: flex-start; padding: 20px; background: #ffffff; color: #000000; });
        let t = ui.add_text(root, text);
        ui.update_style(t, |s| s.font_size = size);
        let img = ui_to_image(&mut ui, 600.0, 120.0, 1.0, &DrawingFonts::new());
        let r = ui.rect(t);
        let (lo, hi) = ink_columns(&img, r, [255, 255, 255]).expect("ink");
        // The glyphs start at the box (less a side bearing) and end at its
        // measured width (less the last one's).
        assert!((lo - r.x).abs() <= size * 0.15 + 1.0, "{text}: ink starts at {lo}, box at {}", r.x);
        assert!((hi - (r.x + r.width)).abs() <= size * 0.15 + 1.0, "{text}: ink ends at {hi}, box at {}", r.x + r.width);
    }
}

// ---- independent readers ----

fn tool(name: &str) -> bool {
    let found = Command::new(name).arg("--version").output().is_ok() || Command::new(name).arg("-v").output().is_ok();
    if !found {
        eprintln!("skipping: {name} is not installed");
    }
    found
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("ui_vector");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn over_white(img: &PngImage) -> Vec<[f32; 3]> {
    img.rgba
        .chunks(4)
        .map(|p| {
            let a = p[3] as f32 / 255.0;
            [0, 1, 2].map(|i| p[i] as f32 / 255.0 * a + 1.0 - a)
        })
        .collect()
}

/// RMS difference and share of pixels off by more than a quarter, blurred
/// 3×3 first (past each reader's own anti-aliasing of glyph edges).
fn compare(a: &PngImage, b: &PngImage) -> (f32, f32) {
    assert_eq!((a.width, a.height), (b.width, b.height));
    let (w, h) = (a.width as usize, a.height as usize);
    let blur = |p: Vec<[f32; 3]>| {
        let mut out = p.clone();
        for y in 0..h {
            for x in 0..w {
                let mut s = [0.0f32; 3];
                let mut n = 0.0;
                for yy in y.saturating_sub(1)..(y + 2).min(h) {
                    for xx in x.saturating_sub(1)..(x + 2).min(w) {
                        (0..3).for_each(|i| s[i] += p[yy * w + xx][i]);
                        n += 1.0;
                    }
                }
                out[y * w + x] = s.map(|v| v / n);
            }
        }
        out
    };
    let (p, q) = (blur(over_white(a)), blur(over_white(b)));
    let (mut sum, mut far) = (0.0f64, 0usize);
    for (u, v) in p.iter().zip(&q) {
        let mut worst = 0.0f32;
        for i in 0..3 {
            sum += ((u[i] - v[i]) as f64).powi(2);
            worst = worst.max((u[i] - v[i]).abs());
        }
        far += usize::from(worst > 0.25);
    }
    ((sum / (p.len() * 3) as f64).sqrt() as f32, far as f32 / p.len() as f32)
}

fn quarter(img: &PngImage) -> PngImage {
    let (w, h) = (img.width / 4, img.height / 4);
    let mut rgba = vec![];
    for y in 0..h {
        for x in 0..w {
            let mut s = [0.0f32; 4];
            for d in 0..16 {
                let p = img.pixel(x * 4 + d % 4, y * 4 + d / 4);
                let a = p[3] as f32 / 255.0;
                (0..3).for_each(|i| s[i] += p[i] as f32 * a);
                s[3] += a;
            }
            let c = |v: f32| if s[3] > 0.0 { (v / s[3]).round() as u8 } else { 0 };
            rgba.extend_from_slice(&[c(s[0]), c(s[1]), c(s[2]), (s[3] / 16.0 * 255.0).round() as u8]);
        }
    }
    PngImage { width: w, height: h, rgba }
}

fn read(path: &Path) -> PngImage {
    decode_png(&std::fs::read(path).unwrap()).unwrap()
}

fn ghostscript(name: &str, pdf: &[u8]) -> Option<PngImage> {
    if !tool("gs") {
        return None;
    }
    let (f, png) = (scratch(&format!("{name}.pdf")), scratch(&format!("{name}-gs.png")));
    std::fs::write(&f, pdf).unwrap();
    let out = Command::new("gs")
        .args(["-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=pngalpha", "-r384", "-dTextAlphaBits=1", "-dGraphicsAlphaBits=1"])
        .arg(format!("-sOutputFile={}", png.display()))
        .arg(&f)
        .output()
        .unwrap();
    assert!(out.status.success(), "gs: {}", String::from_utf8_lossy(&out.stderr));
    Some(quarter(&read(&png)))
}

fn poppler(name: &str, pdf: &[u8]) -> Option<PngImage> {
    if !tool("pdftoppm") {
        return None;
    }
    let f = scratch(&format!("{name}.pdf"));
    std::fs::write(&f, pdf).unwrap();
    let stem = scratch(&format!("{name}-poppler"));
    let out = Command::new("pdftoppm").args(["-png", "-r", "96", "-singlefile"]).arg(&f).arg(&stem).output().unwrap();
    assert!(out.status.success(), "pdftoppm: {}", String::from_utf8_lossy(&out.stderr));
    Some(read(&stem.with_extension("png")))
}

fn rsvg(name: &str, doc: &SvgDocument) -> Option<PngImage> {
    if !tool("rsvg-convert") {
        return None;
    }
    let (f, png) = (scratch(&format!("{name}.svg")), scratch(&format!("{name}-rsvg.png")));
    std::fs::write(&f, write_svg(doc)).unwrap();
    let out = Command::new("rsvg-convert").arg("-o").arg(&png).arg(&f).output().unwrap();
    assert!(out.status.success(), "rsvg-convert: {}", String::from_utf8_lossy(&out.stderr));
    Some(read(&png))
}

fn agree(name: &str, reader: &str, ours: &PngImage, theirs: Option<PngImage>, limits: (f32, f32)) {
    let Some(theirs) = theirs else { return };
    let (rmse, far) = compare(ours, &theirs);
    eprintln!("{name} vs {reader}: rmse {:.3}%, far {:.3}%", rmse * 100.0, far * 100.0);
    if rmse > limits.0 || far > limits.1 {
        let enc = |i: &PngImage| threers::utils::png::encode_png(i.width, i.height, &i.rgba);
        std::fs::write(scratch(&format!("{name}-{reader}-ours.png")), enc(ours)).unwrap();
        std::fs::write(scratch(&format!("{name}-{reader}-theirs.png")), enc(&theirs)).unwrap();
        panic!("{name}: {reader} differs: rmse {:.3}%, {:.3}% far off", rmse * 100.0, far * 100.0);
    }
}

#[test]
fn the_pdf_shows_the_png_to_ghostscript_and_poppler_and_extracts() {
    let mut ui = Ui::new();
    dashboard(&mut ui, true);
    let fonts = DrawingFonts::new();
    let ours = ui_to_image(&mut ui, W, H, 1.0, &fonts);
    let pdf = ui_to_pdf(&mut ui, W, H, &fonts);
    agree("dashboard", "ghostscript", &ours, ghostscript("dashboard", &pdf), (0.02, 0.003));
    // poppler grows image edges out to whole pixels (see drawing_oracles).
    agree("dashboard", "poppler", &ours, poppler("dashboard", &pdf), (0.03, 0.006));
    if tool("pdftotext") {
        let f = scratch("dashboard-text.pdf");
        std::fs::write(&f, &pdf).unwrap();
        let out = Command::new("pdftotext").arg("-raw").arg(&f).arg("-").output().unwrap();
        let text = String::from_utf8_lossy(&out.stdout);
        for want in ["Quarterly report", "Revenue up", "was 42, now 57"] {
            assert!(text.contains(want), "pdftotext lacks {want:?}:\n{text}");
        }
    }
}

/// librsvg sets text in the machine's own fonts, so the SVG is compared
/// without any: every box, border, clip, gradient, icon and image.
#[test]
fn the_svg_shows_the_png_to_librsvg() {
    let mut ui = Ui::new();
    dashboard(&mut ui, false);
    let fonts = DrawingFonts::new();
    let doc = ui_to_document(&mut ui, W, H, &fonts);
    let ours = ui_to_image(&mut ui, W, H, 1.0, &fonts);
    agree("dashboard-shapes", "librsvg", &ours, rsvg("dashboard-shapes", &doc), (0.01, 0.002));
    // and the same document's PDF, to Ghostscript, without text
    agree("dashboard-shapes", "ghostscript", &ours, ghostscript("dashboard-shapes", &write_pdf(&doc)), (0.02, 0.003));
}

#[test]
fn bold_italic_and_struck_text_draw_as_such() {
    let mut ui = Ui::new();
    let n = dashboard(&mut ui, true);
    let img = ui_to_image(&mut ui, W, H, 1.0, &DrawingFonts::new());
    // The title is white on the gradient; synthetic bold makes it heavier
    // than the same text regular.
    let ink = |img: &PngImage, r: threers::ui::Rect, bg_luma_above: bool| {
        let mut n = 0;
        for y in r.y as u32..(r.y + r.height) as u32 {
            for x in r.x as u32..(r.x + r.width) as u32 {
                let p = img.pixel(x, y);
                let luma = (p[0] as u32 + p[1] as u32 + p[2] as u32) / 3;
                n += usize::from(if bg_luma_above { luma > 200 } else { luma < 100 });
            }
        }
        n
    };
    let bold_ink = ink(&img, ui.rect(n.title), true);
    let mut plain = Ui::new();
    let p = dashboard(&mut plain, true);
    plain.update_style(p.title, |s| s.font_weight = threers::ui::FontWeight::Normal);
    let regular = ui_to_image(&mut plain, W, H, 1.0, &DrawingFonts::new());
    let regular_ink = ink(&regular, plain.rect(p.title), true);
    assert!(bold_ink as f32 > regular_ink as f32 * 1.15, "bold {bold_ink} vs regular {regular_ink}");
    // The struck text has a line through its middle: a row darker than the
    // page all the way along the run (between letters too), where the same
    // text unstruck has gaps.
    let row_cover = |img: &PngImage, r: threers::ui::Rect| {
        let mid = (r.y + r.height / 2.0) as u32;
        (mid - 1..=mid + 1)
            .map(|y| (r.x as u32 + 2..(r.x + r.width) as u32 - 2).filter(|&x| img.pixel(x, y)[0] < n.page[0] - 40).count())
            .max()
            .unwrap()
    };
    let r = ui.rect(n.struck);
    let struck = row_cover(&img, r);
    assert!(struck as f32 > (r.width - 4.0) * 0.95, "struck through {struck} of {}", r.width);
    let mut unstruck = Ui::new();
    let u = dashboard(&mut unstruck, true);
    unstruck.update_style(u.struck, |s| s.line_through = false);
    let clean = ui_to_image(&mut unstruck, W, H, 1.0, &DrawingFonts::new());
    assert!(row_cover(&clean, unstruck.rect(u.struck)) < struck * 3 / 4, "the line is what covers the row");
    // Italic leans about the baseline: the top of the "R" sits right of
    // where it stands upright (by 12° of its cap height, about 2 pixels).
    let top_start = |img: &PngImage, r: threers::ui::Rect| {
        let start = |y: u32| (r.x as u32..(r.x + r.width) as u32 + 4).find(|&x| img.pixel(x, y)[0] < 120);
        let top = (r.y as u32..(r.y + r.height) as u32).find(|&y| start(y).is_some()).unwrap();
        start(top).unwrap()
    };
    let mut upright = Ui::new();
    let u = dashboard(&mut upright, true);
    upright.update_style(u.italic, |s| s.italic = false);
    let straight = ui_to_image(&mut upright, W, H, 1.0, &DrawingFonts::new());
    let (leaning, standing) = (top_start(&img, ui.rect(n.italic)), top_start(&straight, upright.rect(u.italic)));
    assert!(leaning >= standing + 2, "not slanted: its top starts at {leaning}, upright at {standing}");
}

