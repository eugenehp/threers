//! The 2D overlay has to actually reach the frame, and reach only the pixels it
//! drew on.
//!
//! [`Overlay`] is tested on the CPU in its own module; what is untested there is
//! the hand-off — that the bytes it produces are the bytes
//! [`Renderer::draw_overlay_rgba8`] expects, in the layout, the alpha convention
//! and the colour space it expects. Four separate conventions have to agree
//! across that boundary and a mismatch in any of them produces a plausible
//! picture: premultiplied alpha dims the overlay, a flipped row order mirrors
//! it, and a colour-space mismatch is invisible until something is drawn at half
//! opacity.
//!
//! This drives the same call [`threers::sketch`] makes, on a real target, and
//! reads the pixels back.
//!
//! Skipped when no GPU adapter is available.

use threers::overlay::Overlay;
use threers::{Color, HeadlessRenderer};

/// 256 pixels is 1024 bytes a row, so readback needs no row padding.
const W: u32 = 256;
const H: u32 = 64;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// Mid grey, in linear light — a clear value an sRGB attachment encodes.
const CLEAR: f64 = 0.216;

fn host() -> Option<HeadlessRenderer> {
    HeadlessRenderer::builder().size(W, H).build().ok()
}

/// Clear a target, optionally composite `overlay` over it, and read it back.
fn frame(host: &mut HeadlessRenderer, overlay: Option<&Overlay>) -> Vec<u8> {
    let device = host.device().clone();
    let queue = host.queue().clone();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("overlay test target"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = device.create_command_encoder(&Default::default());
    encoder
        .begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("overlay test clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: CLEAR,
                        g: CLEAR,
                        b: CLEAR,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
        .forget_lifetime();
    queue.submit(Some(encoder.finish()));

    // The call `sketch` makes, after `render` and before presenting.
    if let Some(overlay) = overlay {
        let (w, h) = overlay.size();
        host.renderer()
            .draw_overlay_rgba8(overlay.pixels(), w, h, &view, FORMAT);
    }

    let bytes = (W * H * 4) as u64;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("overlay test readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(W * 4),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let out = slice
        .get_mapped_range()
        .expect("buffer range is mapped")
        .to_vec();
    buffer.unmap();
    out
}

/// Clear a fresh target to [`CLEAR`], run `draw` against it, and read it back.
fn cleared_target(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    draw: impl FnOnce(&wgpu::TextureView),
) -> Vec<u8> {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("overlay parity target"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder
        .begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("overlay parity clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: CLEAR,
                        g: CLEAR,
                        b: CLEAR,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
        .forget_lifetime();
    queue.submit(Some(encoder.finish()));

    draw(&view);
    read_back(device, queue, &texture)
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<u8> {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("overlay parity readback"),
        size: (W * H * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(W * 4),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let out = slice
        .get_mapped_range()
        .expect("buffer range is mapped")
        .to_vec();
    buffer.unmap();
    out
}

fn px(frame: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = (y as usize * W as usize + x as usize) * 4;
    [frame[i], frame[i + 1], frame[i + 2], frame[i + 3]]
}

/// An opaque overlay replaces the pixels it covers and leaves the rest alone.
///
/// The band is deliberately off-centre vertically, which is what catches a
/// flipped row order — a centred one would survive it.
#[test]
fn an_opaque_overlay_lands_where_it_was_drawn() {
    let Some(mut host) = host() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let before = frame(&mut host, None);

    let mut overlay = Overlay::new(W, H);
    overlay.no_stroke();
    overlay.fill(Color::WHITE);
    overlay.rect(0.0, 8.0, W as f32, 16.0);
    let after = frame(&mut host, Some(&overlay));

    assert_eq!(px(&after, 128, 16), [255, 255, 255, 255], "inside the band");
    assert_eq!(
        px(&after, 128, 48),
        px(&before, 128, 48),
        "below the band must be untouched"
    );
    assert_eq!(
        px(&after, 128, 2),
        px(&before, 128, 2),
        "above the band must be untouched"
    );

    // Row by row: only rows 8..24 may differ, which pins the vertical placement
    // rather than just the presence of ink.
    for y in 0..H {
        let row = |f: &[u8]| {
            let a = y as usize * W as usize * 4;
            f[a..a + W as usize * 4].to_vec()
        };
        let changed = row(&after) != row(&before);
        assert_eq!(
            changed,
            (8..24).contains(&y),
            "row {y} changed = {changed}, which is the wrong way round"
        );
    }
}

/// Half-opacity is where the conventions show. A premultiplied buffer would come
/// back too dark, and a blend in encoded space would land ~30 bytes off the
/// linear answer.
#[test]
fn a_half_opaque_overlay_blends_in_linear_light() {
    let Some(mut host) = host() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let before = frame(&mut host, None);

    let mut overlay = Overlay::new(W, H);
    overlay.no_stroke();
    overlay.fill_alpha(Color::WHITE, 0.5);
    overlay.rect(0.0, 0.0, W as f32, H as f32);
    let after = frame(&mut host, Some(&overlay));

    let dst = px(&before, 128, 32);
    let got = px(&after, 128, 32);

    let decode = |b: u8| {
        let s = b as f32 / 255.0;
        if s <= 0.040_45 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };
    let encode = |v: f32| {
        let v = v.clamp(0.0, 1.0);
        let s = if v <= 0.003_130_8 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0 + 0.5) as u8
    };

    // The alpha byte the overlay wrote, not the nominal 0.5 — it is quantised
    // before the GPU ever sees it.
    let sa = overlay.pixels()[3] as f32 / 255.0;
    let want = encode(1.0 * sa + decode(dst[0]) * (1.0 - sa));
    assert!(
        (got[0] as i32 - want as i32).abs() <= 2,
        "blended {got:?} over {dst:?}, expected ~{want} in linear light"
    );

    // And distinguish it from the two ways this usually goes wrong.
    let encoded_space = (255.0 * sa + dst[0] as f32 * (1.0 - sa)) as i32;
    assert!(
        (got[0] as i32 - encoded_space).abs() > 8,
        "{got:?} matches an encoded-space blend ({encoded_space}), not a linear one"
    );
    let premultiplied = encode(decode(dst[0]) * (1.0 - sa)) as i32;
    assert!(
        (got[0] as i32 - premultiplied).abs() > 8,
        "{got:?} looks like the overlay was premultiplied ({premultiplied})"
    );
}

/// An untouched overlay is the common case — most frames draw no 2D at all —
/// and compositing one must be a no-op rather than a grey wash.
#[test]
fn an_untouched_overlay_changes_nothing() {
    let Some(mut host) = host() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let before = frame(&mut host, None);
    let overlay = Overlay::new(W, H);
    assert!(!overlay.touched());
    let after = frame(&mut host, Some(&overlay));
    assert_eq!(before, after, "a transparent overlay must not alter the frame");
}

/// One frame of the parity sequence below: a label, and what it draws.
type Step = (&'static str, fn(&mut Overlay));

/// Uploading only the rows that changed has to give the same picture as
/// uploading all of them — on every frame of a sequence, not just the first.
///
/// This is the test the optimisation rests on, and the failures it is looking
/// for are all of the shape "the texture still holds something the buffer does
/// not": a panel that should have been erased when the next frame cleared it, a
/// line of type whose rows were under-reported because the painter re-wrapped
/// it or slid it up off the bottom edge, or a band computed from the wrong
/// frame's contents. A single-frame test cannot see any of them.
#[test]
fn uploading_only_the_changed_rows_matches_uploading_all_of_them() {
    let Some(host) = host() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let device = host.device().clone();
    let queue = host.queue().clone();
    // Two renderers over one device, so each has its own overlay texture and
    // the reference cannot be contaminated by the thing it is checking.
    let mut reference = threers::Renderer::new(device.clone(), queue.clone(), FORMAT, W, H);
    let mut partial = threers::Renderer::new(device.clone(), queue.clone(), FORMAT, W, H);

    // Each step draws a frame from scratch. Between them the rows that change
    // are deliberately disjoint, adjacent, and overlapping in turn.
    let steps: Vec<Step> = vec![
        ("panel at the top", |o| {
            o.no_stroke();
            o.fill_alpha(Color::BLACK, 0.6);
            o.rounded_rect(8.0, 6.0, 120.0, 28.0, 4.0);
            o.fill(Color::WHITE);
            o.text(14.0, 10.0, "60 fps");
        }),
        ("same rows, different text", |o| {
            o.no_stroke();
            o.fill_alpha(Color::BLACK, 0.6);
            o.rounded_rect(8.0, 6.0, 120.0, 28.0, 4.0);
            o.fill(Color::WHITE);
            o.text(14.0, 10.0, "12 fps");
        }),
        // Nothing where the panel was: the old rows have to be erased.
        ("a bar near the bottom instead", |o| {
            o.no_stroke();
            o.fill(Color::from_hex(0x33ff99));
            o.rect(20.0, 48.0, 180.0, 10.0);
        }),
        // Against the bottom edge, where the painter slides the block up.
        ("text pushed off the bottom", |o| {
            o.fill(Color::WHITE);
            o.text_size(18.0);
            o.text(10.0, H as f32 - 4.0, "bottom");
        }),
        // Long enough to wrap, anchored near the right edge, which is where the
        // wrap width — and so the row count — is decided by the anchor.
        ("wrapping text at the right edge", |o| {
            o.fill(Color::WHITE);
            o.text_size(14.0);
            o.text(W as f32 - 90.0, 8.0, "a longer string that has to wrap somewhere");
        }),
        ("right-aligned read-out", |o| {
            o.fill(Color::WHITE);
            o.text_size(14.0);
            o.text_right(W as f32 - 8.0, 20.0, "1234");
        }),
        // And a frame that draws nothing at all, which still has to erase.
        ("nothing", |_o| {}),
        ("the panel again", |o| {
            o.no_stroke();
            o.fill_alpha(Color::BLACK, 0.6);
            o.rounded_rect(8.0, 6.0, 120.0, 28.0, 4.0);
        }),
    ];

    let mut full_overlay = Overlay::new(W, H);
    let mut partial_overlay = Overlay::new(W, H);
    for (label, step) in steps {
        full_overlay.clear();
        partial_overlay.clear();
        full_overlay.text_size(16.0);
        partial_overlay.text_size(16.0);
        step(&mut full_overlay);
        step(&mut partial_overlay);

        let want = cleared_target(&device, &queue, |view| {
            reference.draw_overlay_rgba8(full_overlay.pixels(), W, H, view, FORMAT);
        });
        let got = cleared_target(&device, &queue, |view| {
            partial.draw_overlay(&mut partial_overlay, view, FORMAT);
        });

        assert_eq!(
            want.len(),
            got.len(),
            "{label}: frames differ in size"
        );
        if want != got {
            let bad = want
                .chunks(4)
                .zip(got.chunks(4))
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .collect::<Vec<_>>();
            let (i, (a, b)) = bad[0];
            panic!(
                "{label}: {} of {} pixels differ, first at ({}, {}): full upload {a:?}, \
                 row upload {b:?}. Rows marked dirty: {:?}",
                bad.len(),
                W * H,
                i as u32 % W,
                i as u32 / W,
                partial_overlay.dirty_rows()
            );
        }
    }
}

/// A renderer seeing an overlay for the first time has to upload all of it,
/// even when the overlay says only a few rows are dirty.
///
/// Which happens whenever an overlay outlives, or is shared with, the thing
/// compositing it: another renderer already acknowledged those rows, so the
/// dirty band describes what *that* texture was missing and not this one. The
/// band is only ever a delta against a texture that already holds the previous
/// frame, so a texture that holds nothing cannot use one.
#[test]
fn a_renderer_new_to_an_overlay_uploads_the_whole_thing() {
    let Some(host) = host() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let device = host.device().clone();
    let queue = host.queue().clone();

    let mut overlay = Overlay::new(W, H);
    overlay.no_stroke();
    overlay.fill(Color::WHITE);
    overlay.rect(10.0, 10.0, 60.0, 20.0);

    // The first renderer composites and the overlay is told so, which leaves
    // nothing marked dirty.
    let mut first = threers::Renderer::new(device.clone(), queue.clone(), FORMAT, W, H);
    let _ = cleared_target(&device, &queue, |view| {
        first.draw_overlay(&mut overlay, view, FORMAT);
    });
    assert!(
        overlay.dirty_rows().is_none(),
        "the first composite should have left nothing outstanding"
    );

    // A second renderer now has to produce the same picture from the same
    // overlay, with no dirty rows to go on.
    let mut second = threers::Renderer::new(device.clone(), queue.clone(), FORMAT, W, H);
    let got = cleared_target(&device, &queue, |view| {
        second.draw_overlay(&mut overlay, view, FORMAT);
    });
    let want = cleared_target(&device, &queue, |view| {
        let mut full = threers::Renderer::new(device.clone(), queue.clone(), FORMAT, W, H);
        full.draw_overlay_rgba8(overlay.pixels(), W, H, view, FORMAT);
    });
    assert_eq!(px(&got, 30, 20), [255, 255, 255, 255], "the rect is missing");
    assert_eq!(got, want, "a first-time composite must match a full upload");
}

/// Horizontal placement, so a transposed or mirrored upload cannot pass.
#[test]
fn the_overlay_is_not_mirrored_horizontally() {
    let Some(mut host) = host() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut overlay = Overlay::new(W, H);
    overlay.no_stroke();
    overlay.fill(Color::WHITE);
    // A block in the left quarter only.
    overlay.rect(0.0, 0.0, W as f32 * 0.25, H as f32);
    let after = frame(&mut host, Some(&overlay));

    assert_eq!(px(&after, 16, 32), [255, 255, 255, 255], "left quarter is lit");
    assert_ne!(
        px(&after, W - 16, 32),
        [255, 255, 255, 255],
        "right edge must stay clear"
    );
}
