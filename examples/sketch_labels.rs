//! Labels, nearest-object regions and a hull — 2D over 3D, from the projected
//! positions of the things in the scene.
//!
//! ```text
//! cargo run --release --example sketch_labels
//! cargo run --release --example sketch_labels -- --frames 120   # run and quit
//! ```
//!
//! Eight objects orbit. Every frame their world positions are projected to
//! pixels, and that handful of 2D points drives everything drawn on top:
//!
//! | | |
//! |---|---|
//! | [`Camera::projector`] | where each object is, in the overlay's own coordinates |
//! | [`voronoi_cells`] | the region each object is the nearest one in — which is what "click to select" actually means, drawn |
//! | [`delaunay`] | who is adjacent to whom, as the dual of those regions |
//! | [`convex_hull`] | the outline of the whole group |
//! | [`Overlay`] | the polygons, the leader lines and the type |
//!
//! The cell under the cursor is highlighted, so moving the mouse shows the
//! picking decision directly rather than as a changing read-out.
//!
//! The overlay is the expensive part here, at about 5 ms a frame: the cells
//! cover the whole viewport, a filled polygon costs a distance measurement per
//! edge near its boundary, and the outlines then blend over the fills. That is
//! this example being deliberately extravagant with 2D — an ordinary read-out
//! is two hundred microseconds.

use threers::math::{Box2, ColorRamp};
use threers::overlay::Overlay;
use threers::prelude::*;
use threers::sketch::{sketch, Frame, SketchOptions};

const NAMES: [&str; 8] = [
    "anchor", "bellows", "cradle", "dowel", "eyelet", "ferrule", "gudgeon", "hasp",
];

fn main() {
    let frame_limit: Option<u64> = std::env::args()
        .collect::<Vec<_>>()
        .windows(2)
        .find(|w| w[0] == "--frames")
        .and_then(|w| w[1].parse().ok());

    sketch(
        SketchOptions::new("threers — labels over 3D")
            .size(1000, 640)
            .background(Color::from_hex(0x0b0d12))
            .msaa(4),
        move |s| {
            s.camera.position = Vector3::new(0.0, 4.0, 12.0);
            s.camera.look_at(Vector3::ZERO);
            s.scene.add_light(AmbientLight::new(Color::WHITE, 0.45));
            let mut key = Object3D::light(
                DirectionalLight::new(Color::from_hex(0xfff2e2), 1.5)
                    .with_direction(Vector3::new(-0.5, -1.0, -0.4)),
            );
            key.position = Vector3::new(5.0, 8.0, 4.0);
            s.scene.add(key);

            // The type size only needs saying once, which is what `Setup` has
            // the overlay for — and it is scaled, because the overlay is in
            // physical pixels and 14 of those is small type on a 2x display.
            s.overlay.text_size(14.0 * s.scale_factor);

            let ramp = ColorRamp::turbo();
            let ids: Vec<_> = (0..NAMES.len())
                .map(|i| {
                    let t = i as f32 / NAMES.len() as f32;
                    s.scene.add_mesh(
                        SphereGeometry::new(0.55, 24, 16),
                        StandardMaterial::new(ramp.at(t))
                            .with_metalness(0.25)
                            .with_roughness(0.4)
                            .into(),
                    )
                })
                .collect();

            let mut frames = 0u64;
            move |f: &mut Frame| {
                frames += 1;
                if let Some(limit) = frame_limit {
                    if frames >= limit {
                        f.exit = true;
                    }
                }

                // Two interleaved rings, so the group is not a flat circle and
                // the regions keep rearranging.
                let mut world = Vec::with_capacity(ids.len());
                for (i, id) in ids.iter().enumerate() {
                    let phase = i as f32 / ids.len() as f32 * std::f32::consts::TAU;
                    let spin = f.elapsed * 0.35 + phase;
                    let radius = if i % 2 == 0 { 4.2 } else { 6.4 };
                    let p = Vector3::new(
                        spin.cos() * radius,
                        (f.elapsed * 0.6 + phase * 1.7).sin() * 1.8,
                        spin.sin() * radius,
                    );
                    world.push(p);
                    if let Some(obj) = f.scene.get_mut(*id) {
                        obj.position = p;
                    }
                }

                // World to pixels. `None` is an object behind the camera, which
                // has no place on screen and no site in the diagram — dropping
                // it is why the index bookkeeping below exists.
                //
                // One projector for the whole loop. `world_to_screen` would
                // rebuild the view and projection matrices per object, which is
                // the right trade for one label and the wrong one for a list.
                let mut sites = Vec::with_capacity(world.len());
                let mut labelled = Vec::with_capacity(world.len());
                {
                    let project = f.camera.projector(f.width, f.height);
                    for (i, p) in world.iter().enumerate() {
                        if let Some(q) = project.screen(*p) {
                            sites.push(q);
                            labelled.push(i);
                        }
                    }
                }
                if sites.len() < 3 {
                    return;
                }

                let bounds = Box2::new(
                    Vector2::new(0.0, 0.0),
                    Vector2::new(f.width as f32, f.height as f32),
                );
                let cells = voronoi_cells(&sites, bounds);
                let edges = delaunay(&sites);
                let hull = convex_hull(&sites);

                // Which cell the cursor is in — by distance to the site, which
                // is the definition of the cell and cheaper than a
                // point-in-polygon test against the polygon that encodes it.
                let pointer = f.pointer;
                let nearest = sites
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        let d = |p: &Vector2| (p.x - pointer.x).powi(2) + (p.y - pointer.y).powi(2);
                        d(a).total_cmp(&d(b))
                    })
                    .map(|(i, _)| i);

                let ramp = ColorRamp::turbo();
                let o: &mut Overlay = f.overlay;

                // The regions, barely there except for the one under the cursor.
                o.no_stroke();
                for (i, cell) in cells.iter().enumerate() {
                    let t = labelled[i] as f32 / NAMES.len() as f32;
                    let hot = nearest == Some(i);
                    o.fill_alpha(ramp.at(t), if hot { 0.3 } else { 0.08 });
                    o.polygon(cell);
                }

                // Their boundaries, over the top so neighbouring fills do not
                // bury the line between them.
                o.no_fill();
                o.stroke_alpha(Color::WHITE, 0.22);
                o.stroke_weight(1.0);
                for cell in &cells {
                    o.polygon(cell);
                }

                // The Delaunay edges are the dual: a line between two objects
                // exactly when their regions share a border.
                o.stroke_alpha(Color::from_hex(0x66ccff), 0.35);
                for t in &edges {
                    for k in 0..3 {
                        let (a, b) = (sites[t[k] as usize], sites[t[(k + 1) % 3] as usize]);
                        o.line(a.x, a.y, b.x, b.y);
                    }
                }

                // And the outline of the group.
                o.stroke_alpha(Color::from_hex(0xffcc55), 0.8);
                o.stroke_weight(2.0);
                let outline: Vec<Vector2> = hull.iter().map(|&i| sites[i as usize]).collect();
                o.polygon(&outline);

                // Then the labels, last, so nothing is drawn across them.
                for (i, p) in sites.iter().enumerate() {
                    let hot = nearest == Some(i);
                    let t = labelled[i] as f32 / NAMES.len() as f32;
                    o.no_fill();
                    o.stroke_alpha(Color::WHITE, if hot { 0.9 } else { 0.45 });
                    o.stroke_weight(if hot { 2.0 } else { 1.0 });
                    o.circle(p.x, p.y, if hot { 9.0 } else { 5.0 });

                    let name = NAMES[labelled[i]];
                    let (tw, th) = o.measure_text(name);
                    // Up and to the right of the marker, nudged back inside
                    // when that would run off the edge.
                    let lx = (p.x + 14.0).min(f.width as f32 - tw - 6.0);
                    let ly = (p.y - 10.0 - th).max(4.0);
                    o.no_stroke();
                    o.fill_alpha(Color::BLACK, 0.55);
                    o.rounded_rect(lx - 5.0, ly - 3.0, tw + 10.0, th + 6.0, 4.0);
                    o.fill(if hot { Color::WHITE } else { ramp.at(t) });
                    o.text(lx, ly, name);
                }

                o.no_stroke();
                o.fill(Color::WHITE);
                o.text_centered(
                    f.width as f32 * 0.5,
                    f.height as f32 - 30.0,
                    "regions are nearest-object — move the cursor",
                );
            }
        },
    );
}
