//! What each overlay shape costs, for finding a slow HUD. Run by hand.
use std::time::Instant;
use threers::overlay::Overlay;
use threers::prelude::*;

fn time(label: &str, mut f: impl FnMut(&mut Overlay)) {
    let mut o = Overlay::new(2880, 1620);
    let mut best = f64::MAX;
    for _ in 0..20 {
        o.clear();
        let t = Instant::now();
        f(&mut o);
        best = best.min(t.elapsed().as_secs_f64() * 1000.0);
    }
    eprintln!("{label:<40} {best:.3} ms");
}

#[test]
fn overlay_cost() {
    if std::env::var_os("OVERLAY_TIMING").is_none() {
        return;
    }
    time("rect 250x200", |o| {
        o.no_stroke();
        o.fill_alpha(Color::from_hex(0), 0.35);
        o.rect(14.0, 14.0, 250.0, 200.0);
    });
    time("rounded rect 250x200", |o| {
        o.no_stroke();
        o.fill_alpha(Color::from_hex(0), 0.35);
        o.rounded_rect(14.0, 14.0, 250.0, 200.0, 10.0);
    });
    let pts: Vec<Vector2> = (0..=150)
        .map(|i| {
            let a = i as f32 / 150.0 * std::f32::consts::TAU;
            Vector2::new(1300.0 + 100.0 * a.cos(), 1300.0 + 100.0 * a.sin())
        })
        .collect();
    time("polyline 150 points, 200 px across", |o| {
        o.no_fill();
        o.stroke(Color::WHITE);
        o.stroke_weight(3.0);
        o.polyline(&pts);
    });
    time("4 circles", |o| {
        o.no_stroke();
        for i in 0..4 {
            o.circle(100.0 + i as f32 * 20.0, 100.0, 6.0);
        }
    });
    time("measure x6 (layout only)", |o| {
        for i in 0..6 {
            o.text_size(20.0 + i as f32 * 5.0);
            o.measure_text("LAP 1/3 0:35.18");
        }
    });
    time("text x6, 20-44 px, no outline", |o| {
        o.fill(Color::WHITE);
        for i in 0..6 {
            o.text_size(20.0 + i as f32 * 5.0);
            o.text(30.0, 50.0 + i as f32 * 30.0, "LAP 1/3 0:35.18");
        }
    });
}
