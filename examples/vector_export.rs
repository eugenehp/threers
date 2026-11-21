//! A UI and a 3D scene written as SVG, PDF and PNG, with no GPU and no
//! window: `SvgRenderer::render_to_document` for the scene and
//! `threers::ui::vector` for the interface, then a two-page PDF of both.
//!
//! ```text
//! cargo run --features ui --example vector_export     # into out/vector/
//! ```

use threers::exporters::{write_pdf_pages_with, write_svg, DrawingFonts, PdfOptions, PdfText};
use threers::prelude::*;
use threers::style;
use threers::ui::vector::{ui_to_document, ui_to_png};
use threers::ui::Ui;

fn scene() -> (Scene, PerspectiveCamera) {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x1b2230);
    scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.35));
    let mut key = Object3D::light(DirectionalLight::new(Color::from_hex(0xffffff), 1.1));
    key.position = Vector3::new(4.0, 6.0, 5.0);
    scene.add(key);
    let mut knot = Object3D::mesh(Mesh::new(
        TorusKnotGeometry::new(0.9, 0.3, 160, 20, 2, 3),
        StandardMaterial::new(Color::from_hex(0xd0763a)).into(),
    ));
    knot.position = Vector3::new(-1.2, 0.2, 0.0);
    scene.add(knot);
    let mut ball = Object3D::mesh(Mesh::new(
        SphereGeometry::new(0.8, 48, 32),
        StandardMaterial::new(Color::from_hex(0x3a86c8)).into(),
    ));
    ball.position = Vector3::new(1.5, 0.0, 0.3);
    scene.add(ball);
    let mut camera = PerspectiveCamera::new(40.0, 4.0 / 3.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 1.2, 7.0);
    camera.look_at(Vector3::new(0.0, 0.0, 0.0));
    (scene, camera)
}

fn panel(ui: &mut Ui) {
    let root = ui.root();
    ui.set_style(root, style! { display: flex; flex-direction: column; gap: 14px; padding: 24px; background: #f4f4f0; color: #1f2937; font-size: 15px; });
    let head = ui.add_box(root, style! { padding: 14px; border-radius: 12px; background: linear-gradient(90deg, #0f766e, #2563fb); color: #ffffff; font-size: 22px; font-weight: bold; });
    ui.add_text(head, "Render report — knot and sphere");
    let row = ui.add_box(root, style! { display: flex; flex-direction: row; gap: 14px; });
    for (label, value, accent) in [("Triangles", "13 440", "#0f766e"), ("Lights", "2", "#2563fb"), ("Draw calls", "2", "#b45309")] {
        let card = ui.add_box(row, style! { flex-grow: 1; padding: 12px; background: #ffffff; border: 1px solid #d1d5db; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.12); });
        let l = ui.add_text(card, label);
        ui.update_style(l, |s| s.font_size = 12.0);
        let v = ui.add_text(card, value);
        let accent = threers::ui::css::color(accent).unwrap();
        ui.update_style(v, |s| {
            s.font_size = 26.0;
            s.color = accent;
            s.font_weight = threers::ui::FontWeight::Bold;
        });
    }
    let note = ui.add_box(root, style! { padding: 12px; border-radius: 10px; background: #fff7ed; border: 1px solid #fdba74; });
    let t = ui.add_text(note, "Drawn without a GPU: the same layout as SVG, PDF and PNG, its text selectable in the PDF.");
    ui.update_style(t, |s| s.italic = true);
}

fn main() -> std::io::Result<()> {
    let dir = std::path::Path::new("out/vector");
    std::fs::create_dir_all(dir)?;
    let fonts = DrawingFonts::new();

    let (mut scene, camera) = scene();
    let renderer = SvgRenderer::new(800, 600);
    let scene_doc = renderer.render_to_document(&mut scene, &camera);
    std::fs::write(dir.join("scene.svg"), write_svg(&scene_doc))?;
    std::fs::write(dir.join("scene.pdf"), renderer.render_to_pdf(&mut scene, &camera))?;
    std::fs::write(dir.join("scene.png"), renderer.render_to_png(&mut scene, &camera, 1.0))?;

    let mut ui = Ui::new();
    panel(&mut ui);
    let ui_doc = ui_to_document(&mut ui, 800.0, 360.0, &fonts);
    std::fs::write(dir.join("ui.svg"), write_svg(&ui_doc))?;
    std::fs::write(dir.join("ui.png"), ui_to_png(&mut ui, 800.0, 360.0, 2.0, &fonts))?;

    // Both on pages of one PDF.
    let options = PdfOptions { text: PdfText::Embedded(fonts), ..PdfOptions::default() };
    std::fs::write(dir.join("report.pdf"), write_pdf_pages_with(&[ui_doc, scene_doc], &options))?;
    for f in ["scene.svg", "scene.pdf", "scene.png", "ui.svg", "ui.png", "report.pdf"] {
        println!("{}", dir.join(f).display());
    }
    Ok(())
}
