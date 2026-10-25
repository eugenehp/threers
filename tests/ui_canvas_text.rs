//! Canvas text set as the page sets it, in a real browser:
//! `FEATURES=ui TEST=ui_canvas_text scripts/wasm-browser-test.sh`.
//!
//! A page kerns a space with its neighbours and a canvas does not, so
//! [`CanvasPainter`] measures and places the words around such spaces
//! itself. These compare it with the page's own layout of the same text.
#![cfg(all(feature = "ui", target_arch = "wasm32"))]

use threers::ui::canvas::CanvasPainter;
use threers::ui::{Painter, Rect, Style, TextMeasure, TextStyle};
use wasm_bindgen::JsCast;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

wasm_bindgen_test_configure!(run_in_browser);

const LINES: [&str; 3] = [
    "Hamburgefonstiv AVATAR Today 0123",
    "A Tale of Two  Cities, V. W. Yates",
    "To be, or not to be: that is the question.",
];

fn document() -> web_sys::Document {
    web_sys::window().unwrap().document().unwrap()
}

fn canvas(width: u32, height: u32) -> (HtmlCanvasElement, CanvasRenderingContext2d) {
    let c: HtmlCanvasElement = document().create_element("canvas").unwrap().dyn_into().unwrap();
    c.set_width(width);
    c.set_height(height);
    document().body().unwrap().append_child(&c).unwrap();
    let ctx = c.get_context("2d").unwrap().unwrap().dyn_into().unwrap();
    (c, ctx)
}

/// The page's width of `text` in `font`.
fn page_width(font: &str, text: &str) -> f64 {
    let span = document().create_element("span").unwrap();
    span.set_attribute("style", &format!("position:absolute;white-space:pre;font:{font}")).unwrap();
    span.set_text_content(Some(text));
    document().body().unwrap().append_child(&span).unwrap();
    let w = span.get_bounding_client_rect().width();
    span.remove();
    w
}

fn style(family: &str, size: f32) -> Style {
    Style { font_family: family.into(), font_size: size, ..Default::default() }
}

/// Measured widths are the page's — to its 1/64-pixel layout units —
/// whether or not the font kerns its spaces (Arial, Times and Helvetica do;
/// Georgia and Verdana do not), where a canvas alone was 0.9–2.2 pixels wide.
#[wasm_bindgen_test]
fn measured_widths_are_the_pages() {
    let (_, ctx) = canvas(8, 8);
    let mut painter = CanvasPainter::new(ctx.clone(), 1.0);
    for (family, size) in [("Arial", 13.0), ("Times", 14.0), ("Helvetica", 13.0), ("Georgia", 15.0), ("Verdana", 12.0)] {
        let s = style(family, size);
        let font = TextStyle::of(&s).css_font();
        for text in LINES {
            let (w, _) = painter.measure(text, &s);
            let page = page_width(&font, text);
            assert!((w as f64 - page).abs() < 0.05, "{font} {text:?}: painter {w}, page {page}");
        }
        // The canvas alone, on the line with "␣T" and "␣A": what was wrong.
        ctx.set_font(&font);
        let plain = ctx.measure_text(LINES[1]).unwrap().width();
        let page = page_width(&font, LINES[1]);
        if matches!(family, "Arial" | "Times" | "Helvetica") {
            assert!(plain - page > 0.5, "{font}: canvas alone {plain}, page {page}");
        }
    }
}

/// Drawn, the line ends where the page's does: the last inked column of
/// the painter's run is the page's width from its start, not the canvas's.
#[wasm_bindgen_test]
fn drawn_text_ends_where_the_page_puts_it() {
    let (w, h) = (420u32, 40u32);
    for (family, size) in [("Arial", 13.0), ("Times", 14.0), ("Georgia", 15.0)] {
        let s = style(family, size);
        let font = TextStyle::of(&s).css_font();
        let text = LINES[1];
        let last_ink = |ctx: &CanvasRenderingContext2d| -> i64 {
            let data = ctx.get_image_data(0.0, 0.0, w as f64, h as f64).unwrap().data();
            (0..w as usize)
                .rev()
                .find(|&x| (0..h as usize).any(|y| data[(y * w as usize + x) * 4 + 3] > 128))
                .unwrap() as i64
        };
        // The painter's run.
        let (_, ctx) = canvas(w, h);
        let mut painter = CanvasPainter::new(ctx.clone(), 1.0);
        painter.begin(w, h, 1.0);
        let mut ts = TextStyle::of(&s);
        ts.color = [0.0, 0.0, 0.0, 1.0];
        painter.text(Rect { x: 10.0, y: 0.0, width: 400.0, height: h as f32 }, text, &ts);
        let painted = last_ink(&ctx);
        // The canvas's own run at the same place, for the difference.
        let (_, plain) = canvas(w, h);
        plain.set_font(&font);
        plain.set_fill_style_str("#000");
        plain.fill_text(text, 10.0, 26.0).unwrap();
        let canvas_alone = last_ink(&plain);
        let shift = page_width(&font, text) - plain.measure_text(text).unwrap().width();
        assert!(
            (painted - canvas_alone) as f64 - shift.round() <= 1.0 && (painted - canvas_alone) as f64 - shift.round() >= -1.0,
            "{font}: painted ends at {painted}, canvas alone at {canvas_alone}, page shift {shift}"
        );
    }
}
