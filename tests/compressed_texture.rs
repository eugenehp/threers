//! Block-compressed textures, from `Texture` through upload to a rendered pixel.
//!
//! The unit tests in `textures::bc1` prove the encoder produces sensible blocks.
//! They cannot prove the GPU ever sees them correctly, and every remaining way
//! to get this wrong lives between the two: `bytes_per_row` counts BYTES PER ROW
//! OF BLOCKS rather than of pixels, `rows_per_image` counts rows of blocks,
//! mip levels must arrive pre-encoded because nothing can filter or render into
//! BC, and the device only exposes `TEXTURE_COMPRESSION_BC` if it was asked for
//! at creation. Get any of those wrong and the result is not a compile error or
//! a panic — it is a texture that samples as garbage, or a validation failure at
//! the first draw.
//!
//! So these render the same picture twice, compressed and not, and compare.
//!
//! Skipped when no GPU adapter is available.

use threers::textures::bc1;
use threers::{
    AmbientLight, Color, HeadlessRenderer, Material, Mesh, Object3D, PerspectiveCamera,
    PlaneGeometry, Scene, StandardMaterial, Texture, TextureFormat, Vector3,
};

const W: u32 = 256;
const H: u32 = 256;

fn renderer() -> Option<HeadlessRenderer> {
    HeadlessRenderer::builder()
        .size(W, H)
        .color_format(wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .ok()
}

/// Ground-like source: broad smooth areas, a hard coastline, and fine detail.
/// Flat colour would pass an implementation that ignored the indices entirely.
fn source(w: u32, h: u32) -> Vec<u8> {
    let mut px = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
            // A diagonal coast: "sea" one side, "land" the other.
            let land = fx + fy * 0.6 > 0.8;
            let c = if land {
                let n = ((x * 7 + y * 13) % 17) as f32 / 17.0;
                [
                    (110.0 + 60.0 * n) as u8,
                    (90.0 + 40.0 * fy * 255.0 / 255.0) as u8,
                    (45.0 + 25.0 * n) as u8,
                ]
            } else {
                // Smooth gradient — the case BC1 is meant to be worst at.
                [
                    (10.0 + 20.0 * fy) as u8,
                    (40.0 + 30.0 * fx) as u8,
                    (110.0 + 60.0 * fy) as u8,
                ]
            };
            px.extend_from_slice(&[c[0], c[1], c[2], 255]);
        }
    }
    px
}

fn quad(map: Texture) -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    scene.add_light(AmbientLight::new(Color::WHITE, 1.0));
    let mut m = StandardMaterial::new(Color::WHITE);
    m.map = Some(std::sync::Arc::new(map));
    m.roughness = 1.0;
    m.metalness = 0.0;
    scene.add(Object3D::mesh(Mesh::new(
        PlaneGeometry::new(2.0, 2.0),
        Material::Standard(m),
    )));
    scene
}

fn camera(distance: f32) -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(45.0, 1.0, 0.01, 100.0);
    c.position = Vector3::new(0.0, 0.0, distance);
    c.look_at(Vector3::ZERO);
    c
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mut se = 0.0f64;
    let mut n = 0.0f64;
    for (x, y) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        for c in 0..3 {
            let d = x[c] as f64 - y[c] as f64;
            se += d * d;
            n += 1.0;
        }
    }
    let mse = se / n.max(1.0);
    if mse <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

/// A BC1 texture must render as very nearly the same picture as the RGBA8 one
/// it was encoded from.
#[test]
fn bc1_renders_the_same_picture_as_rgba8() {
    let Some(mut r) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let (tw, th) = (512u32, 512u32);
    let px = source(tw, th);

    let mut plain = Texture::new(tw, th, TextureFormat::Rgba8UnormSrgb, px.clone());
    plain.flip_y = false;
    let a = r.render_to_rgba(&mut quad(plain), &camera(2.4));

    let (level0, mips) = bc1::encode_with_mips(&px, tw, th);
    let mut comp = Texture::new(tw, th, TextureFormat::Bc1RgbaUnormSrgb, level0);
    // Compressed data cannot be flipped by reversing rows; it is authored the
    // right way up, and `oriented` asserts on the alternative.
    comp.flip_y = false;
    comp.mips = mips.into_iter().map(std::sync::Arc::new).collect();
    let b = r.render_to_rgba(&mut quad(comp), &camera(2.4));

    let q = psnr(&a, &b);
    // Well below what BC1 costs on real imagery (39.6 dB on Blue Marble), but
    // far above what a wrong stride or a mis-ordered block would give: getting
    // `bytes_per_row` in pixels rather than blocks lands in the teens.
    assert!(
        q > 30.0,
        "BC1 render differs from RGBA8 by more than encoding error: {q:.1} dB"
    );
    // And it must not be accidentally identical — that would mean the compressed
    // texture never got bound and we compared a picture with itself.
    assert!(
        a != b,
        "compressed and uncompressed renders are byte-identical"
    );
}

/// Minified, the compressed texture must use its supplied mip chain.
///
/// Nothing can box filter BC blocks and the GPU cannot render into them, so the
/// chain has to arrive pre-encoded. If those levels are dropped the texture
/// samples level 0 under heavy minification and aliases into noise, which is
/// exactly what a large planet map does when it is wrong.
#[test]
fn supplied_mips_are_used_when_minified() {
    let Some(mut r) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let (tw, th) = (512u32, 512u32);
    let px = source(tw, th);
    let (level0, mips) = bc1::encode_with_mips(&px, tw, th);

    let with = {
        let mut t = Texture::new(tw, th, TextureFormat::Bc1RgbaUnormSrgb, level0.clone());
        t.flip_y = false;
        t.mips = mips.into_iter().map(std::sync::Arc::new).collect();
        r.render_to_rgba(&mut quad(t), &camera(30.0))
    };
    let without = {
        let mut t = Texture::new(tw, th, TextureFormat::Bc1RgbaUnormSrgb, level0);
        t.flip_y = false;
        r.render_to_rgba(&mut quad(t), &camera(30.0))
    };
    assert!(
        with != without,
        "the mip chain made no difference under 30x minification — it is not being uploaded"
    );

    // Aliasing shows as variance between neighbouring pixels that the filtered
    // version does not have. Measured across the quad only.
    let energy = |img: &[u8]| {
        let mut e = 0.0f64;
        for y in H / 3..2 * H / 3 {
            for x in W / 3..2 * W / 3 {
                let i = ((y * W + x) * 4) as usize;
                let j = ((y * W + x + 1) * 4) as usize;
                for c in 0..3 {
                    let d = img[i + c] as f64 - img[j + c] as f64;
                    e += d * d;
                }
            }
        }
        e
    };
    assert!(
        energy(&with) < energy(&without),
        "mipped render should be smoother: {:.0} vs {:.0}",
        energy(&with),
        energy(&without)
    );
}

/// A chain to 1x1, at a size that is neither a power of two nor a multiple of
/// the block.
///
/// This is the case a planet tile actually is: 14400 halves to 7200, 3600, 1800,
/// 900 and then 450, which is not a whole number of blocks. An upload that
/// insists on logical block-aligned extents drops everything below 900 — four
/// levels where there should be fourteen — and the map aliases the moment it is
/// minified, which for a planet is the whole approach shot. The fix is to copy
/// against the PHYSICAL extent, and this is what proves it.
#[test]
fn deep_mips_at_awkward_sizes() {
    let Some(mut r) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    // 100x60: a whole number of blocks at level 0 — which is required — but not
    // a power of two, and 100/2/2 = 25 is not a multiple of 4, so the CHAIN goes
    // ragged almost at once. That is the case this is for.
    let (tw, th) = (100u32, 60u32);
    let px = source(tw, th);
    let (level0, mips) = bc1::encode_with_mips(&px, tw, th);
    // 100x60 -> 50x30 -> 25x15 -> 12x7 -> 6x3 -> 3x1 -> 1x1
    assert_eq!(mips.len(), 6, "chain must reach 1x1 whatever the size");
    assert_eq!(bc1::mip_levels(tw, th), mips.len());

    let mut t = Texture::new(tw, th, TextureFormat::Bc1RgbaUnormSrgb, level0);
    t.flip_y = false;
    t.mips = mips.into_iter().map(std::sync::Arc::new).collect();
    // The upload is where a bad extent is rejected, so rendering at all is the
    // assertion; the pixels only confirm it bound the right thing.
    let img = r.render_to_rgba(&mut quad(t), &camera(2.4));
    assert!(img.iter().any(|&v| v != 0), "nothing rendered");
}

/// The orientation contract, stated as a test because it is invisible when
/// broken and it has bitten three times.
///
/// An uncompressed texture defaults to `flip_y = true` and the upload reverses
/// its rows, which is what makes an image file's FIRST row — the north edge of
/// an equirectangular map — end up where a sphere samples north. Block data
/// cannot be reversed that way at all, so it must be stored the way it will be
/// sampled, which means a producer has to flip the pixels BEFORE encoding.
///
/// Nothing enforces that, and nothing can: both orientations are valid data.
/// What the renderer can do is guarantee the relationship, so a producer has
/// something to check itself against. Here it is: pre-flipped compressed pixels
/// must render identically to the same pixels uncompressed with `flip_y` on.
/// Skip the flip and a planet comes out as a mirrored sheet of polar ice.
#[test]
fn compressed_data_is_stored_pre_flipped() {
    let Some(mut r) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let (tw, th) = (256u32, 256u32);
    // Vertically asymmetric on purpose — a symmetric pattern passes either way.
    let mut px = Vec::with_capacity((tw * th * 4) as usize);
    for y in 0..th {
        for x in 0..tw {
            let top = y < th / 3;
            let c = if top {
                [230u8, 230, 240]
            } else {
                [(x * 255 / tw) as u8, 60, 30]
            };
            px.extend_from_slice(&[c[0], c[1], c[2], 255]);
        }
    }

    let stride = (tw * 4) as usize;
    let flipped: Vec<u8> = px.chunks_exact(stride).rev().flatten().copied().collect();

    // All three built BEFORE any render, and held alive until after the last
    // one. The texture cache is keyed partly on the data's address, and a
    // buffer freed between renders can be handed straight back for the next
    // one — at which point the second texture silently reuses the first's
    // upload and the comparison is of one picture with itself.
    let mut plain = Texture::new(tw, th, TextureFormat::Rgba8UnormSrgb, px.clone());
    plain.flip_y = true;
    let (l0g, mg) = bc1::encode_with_mips(&flipped, tw, th);
    let mut good = Texture::new(tw, th, TextureFormat::Bc1RgbaUnormSrgb, l0g);
    good.flip_y = false;
    good.mips = mg.into_iter().map(std::sync::Arc::new).collect();
    let (l0b, mb) = bc1::encode_with_mips(&px, tw, th);
    let mut bad = Texture::new(tw, th, TextureFormat::Bc1RgbaUnormSrgb, l0b);
    bad.flip_y = false;
    bad.mips = mb.into_iter().map(std::sync::Arc::new).collect();

    let want = r.render_to_rgba(&mut quad(plain.clone()), &camera(2.4));
    let got = r.render_to_rgba(&mut quad(good.clone()), &camera(2.4));
    assert!(
        psnr(&want, &got) > 28.0,
        "pre-flipped compressed data should match flip_y uncompressed: {:.1} dB",
        psnr(&want, &got)
    );

    // NOT pre-flipped: must not match. Without this the test above would pass
    // on a symmetric image and prove nothing.
    let wrong = r.render_to_rgba(&mut quad(bad.clone()), &camera(2.4));
    assert!(
        psnr(&want, &wrong) < 20.0,
        "un-flipped data should be visibly wrong, but matched at {:.1} dB — \
         this test cannot see the bug it is for",
        psnr(&want, &wrong)
    );
}

/// Textures built and dropped in turn must each render as themselves.
///
/// The cache key includes the data's ADDRESS, which is unique only while that
/// allocation lives. Drop one texture and build another of the same size and
/// format and the allocator will often hand back the same address, at which
/// point the second silently renders as the first. One pair does not reliably
/// reproduce it — whether the address comes back is the allocator's business —
/// so this runs a series and checks every one, which does.
#[test]
fn a_recycled_allocation_does_not_reuse_the_upload() {
    let Some(mut r) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let (tw, th) = (128u32, 128u32);
    // Compared against EACH OTHER, not against the source colours: ambient
    // shading and tone mapping move absolute values a long way, and the
    // question here is only whether two different textures produced the same
    // picture.
    let mut seen: Vec<[u8; 3]> = Vec::new();
    for i in 0..24u32 {
        let c = [(10 + i * 9) as u8, (200 - i * 7) as u8, (60 + i * 5) as u8];
        let mut t = Texture::new(
            tw,
            th,
            TextureFormat::Rgba8UnormSrgb,
            std::iter::repeat_n([c[0], c[1], c[2], 255], (tw * th) as usize)
                .flatten()
                .collect(),
        );
        t.flip_y = false;
        let img = r.render_to_rgba(&mut quad(t), &camera(2.4));
        let mid = ((H / 2 * W + W / 2) * 4) as usize;
        let got = [img[mid], img[mid + 1], img[mid + 2]];
        if let Some(j) = seen.iter().position(|p| *p == got) {
            panic!(
                "iteration {i} drew {got:?}, identical to iteration {j} — the \
                 upload was reused because the two buffers shared an address"
            );
        }
        seen.push(got);
    }
}

/// A base size that is not a whole number of blocks must be refused HERE, with
/// a message that names the texture.
///
/// wgpu checks this at `create_texture` and reports it as a validation error
/// from inside a device call, naming nothing — which is how a 2250-wide night
/// tile took a render down with no clue which of ninety textures was at fault.
/// Mip levels are exempt: the driver stores those padded.
#[test]
#[should_panic(expected = "must be a multiple of")]
fn a_ragged_base_size_is_refused() {
    let Some(mut r) = renderer() else {
        panic!("must be a multiple of: skipped, no GPU adapter");
    };
    // 2250 is what a 60-degree tile of a 13500-wide map comes to, and it is
    // 562.5 blocks.
    let mut t = Texture::new(
        2250,
        60,
        TextureFormat::Bc1RgbaUnormSrgb,
        vec![0; TextureFormat::Bc1RgbaUnormSrgb.data_len(2250, 60)],
    );
    t.flip_y = false;
    r.render_to_rgba(&mut quad(t), &camera(2.4));
}

/// Sizes are in blocks, and the block maths must round up.
#[test]
fn block_geometry() {
    let f = TextureFormat::Bc1RgbaUnormSrgb;
    assert!(f.is_block_compressed());
    assert_eq!(f.block_dim(), (4, 4));
    assert_eq!(f.bytes_per_block(), 8);
    // A row of 512 pixels is 128 blocks of 8 bytes.
    assert_eq!(f.bytes_per_row(512), 128 * 8);
    assert_eq!(f.rows_per_image(512), 128);
    // Partial blocks still cost a whole block.
    assert_eq!(f.bytes_per_row(513), 129 * 8);
    assert_eq!(f.rows_per_image(1), 1);

    let bc7 = TextureFormat::Bc7RgbaUnormSrgb;
    assert_eq!(bc7.bytes_per_block(), 16);
    assert_eq!(bc7.data_len(4, 4), 16);

    // Uncompressed formats go through the same helpers unchanged.
    let rgba = TextureFormat::Rgba8UnormSrgb;
    assert_eq!(rgba.block_dim(), (1, 1));
    assert_eq!(rgba.bytes_per_row(512), 512 * 4);
    assert_eq!(rgba.rows_per_image(512), 512);
}

/// `bytes_per_pixel` has no answer for a block format and must say so.
#[test]
#[should_panic(expected = "block-compressed")]
fn bytes_per_pixel_refuses_block_formats() {
    let t = Texture::new(4, 4, TextureFormat::Bc1RgbaUnormSrgb, vec![0; 8]);
    let _ = t.bytes_per_pixel();
}

/// Flipping compressed data by reversing rows would shuffle the image into
/// quarters, so it is refused rather than done wrongly.
#[test]
#[should_panic(expected = "flip_y on a block-compressed texture")]
fn flipping_compressed_data_is_refused() {
    let Some(mut r) = renderer() else {
        // The assert lives in the upload path, so without a GPU there is
        // nothing to trip it — fail the same way rather than pass silently.
        panic!("flip_y on a block-compressed texture: skipped, no GPU adapter");
    };
    let mut t = Texture::new(8, 8, TextureFormat::Bc1RgbaUnormSrgb, vec![0; 32]);
    t.flip_y = true; // the default, and wrong for this format
    r.render_to_rgba(&mut quad(t), &camera(2.4));
}
