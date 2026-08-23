//! Rigid-origami birds — cardinal, crane, owl, sparrow, flamingo.
//!
//! Each species is an **assembly** of rigid parts (body twist, crest, wings, tail)
//! posed into a bird silhouette. SVG sheets show the part nets in layout.
//!
//! ```text
//! cargo run --release --example origami_birds [-- out/origami-birds.png]
//! ```

include!("origami_common.inc");

use threers::BirdKind;

fn main() {
    let out = out_arg("out/origami-birds.png");
    let t = 0.42_f64;

    let mut birds = Vec::new();
    for kind in BirdKind::ALL {
        let name = kind.name();
        let bird = kind
            .assemble(t)
            .unwrap_or_else(|| panic!("{name}: assembly failed"));
        let net = kind.layout_sheet();
        write_next_to(
            &out,
            &format!("origami-bird-{name}-net.svg"),
            &net.to_svg(None),
        );
        println!(
            "{name}: {} parts, {} creases on layout sheet",
            bird.parts.len(),
            net.hinge_indices().count()
        );
        birds.push((name, bird));
    }

    render_scene(&out, 1920, 520, [0.0, 1.0, 20.0], Vector3::ZERO, |scene| {
        let n = birds.len() as f32;
        let span = 4.8;
        let origin = -0.5 * (n - 1.0) * span;
        let views = [
            Euler::new(-1.05, 0.42, 0.05),
            Euler::new(-0.95, 0.55, 0.12),
            Euler::new(-1.1, 0.35, -0.08),
            Euler::new(-1.0, 0.48, 0.02),
            Euler::new(-0.88, 0.62, 0.18),
        ];
        for (i, (name, bird)) in birds.iter().enumerate() {
            add_assembled_bird(
                scene,
                bird,
                Vector3::new(origin + i as f32 * span, 0.0, 0.0),
                1.0,
                views[i],
                bird_plumage(name),
            );
        }
    });
}
