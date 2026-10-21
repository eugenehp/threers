//! A volume viewer's panes: axial, coronal and sagittal cross-sections
//! through a 3D Shepp–Logan head phantom (the standard CT test object, ten
//! ellipsoids) drawn on the GPU by `SliceRenderer` — a window/level grey
//! ramp, with a label volume segmenting the inner structures blended over
//! it — and an oblique slice, since the plane can be any plane at all;
//! then the phantom's surfaces by marching cubes: the skull cut open, and
//! the segmented structures in the slices' label colours. Written to
//! `out/volume_slices.png`.
//!
//! ```bash
//! cargo run --release --example volume_slices
//! ```

use std::f32::consts::FRAC_PI_2;
use threers::extras::MarchingCubes;
use threers::math::Quaternion;
use threers::prelude::*;
use threers::renderer::{readback, PixelRect};
use threers::volume::{plane_to_voxel, Data3DTexture, GpuVolume, SliceParams, SliceRenderer};

/// Voxels along each side.
const N: u32 = 128;
/// One pane's side, in pixels.
const P: u32 = 360;
const FORMAT: threers::wgpu::TextureFormat = threers::wgpu::TextureFormat::Rgba8UnormSrgb;

/// Kak and Slaney's 3D Shepp–Logan phantom with the higher-contrast
/// densities of the "modified" 2D one: semi-axes, centre, rotation about z,
/// density added inside.
const ELLIPSOIDS: [([f32; 3], [f32; 3], f32, f32); 10] = [
    ([0.69, 0.92, 0.90], [0.0, 0.0, 0.0], 0.0, 1.0),
    ([0.6624, 0.874, 0.88], [0.0, -0.0184, 0.0], 0.0, -0.8),
    ([0.41, 0.16, 0.21], [-0.22, 0.0, -0.25], 108.0, -0.2),
    ([0.31, 0.11, 0.22], [0.22, 0.0, -0.25], 72.0, -0.2),
    ([0.21, 0.25, 0.50], [0.0, 0.35, -0.25], 0.0, 0.2),
    ([0.046, 0.046, 0.046], [0.0, 0.1, -0.25], 0.0, 0.2),
    ([0.046, 0.023, 0.02], [-0.08, -0.605, -0.25], 0.0, 0.1),
    ([0.046, 0.023, 0.02], [0.06, -0.605, -0.25], 90.0, 0.1),
    ([0.056, 0.04, 0.10], [0.06, -0.105, 0.625], 90.0, 0.2),
    ([0.056, 0.056, 0.10], [0.0, 0.1, 0.625], 0.0, -0.2),
];

/// Whether `p` lies inside ellipsoid `e`.
fn inside(e: &([f32; 3], [f32; 3], f32, f32), p: [f32; 3]) -> bool {
    level(e, p) <= 1.0
}

/// Ellipsoid `e`'s implicit function at `p`: below 1 inside, 1 on its
/// surface.
fn level(e: &([f32; 3], [f32; 3], f32, f32), p: [f32; 3]) -> f32 {
    let (a, c, deg, _) = e;
    let (s, co) = deg.to_radians().sin_cos();
    let (x, y, z) = (p[0] - c[0], p[1] - c[1], p[2] - c[2]);
    let (u, v) = (x * co + y * s, -x * s + y * co);
    (u / a[0]).powi(2) + (v / a[1]).powi(2) + (z / a[2]).powi(2)
}

/// Density at `p` in `[-1, 1]³`.
fn density(p: [f32; 3]) -> f32 {
    ELLIPSOIDS
        .iter()
        .filter(|e| inside(e, p))
        .map(|e| e.3)
        .sum()
}

/// The inner structure `p` is in (ellipsoids 3 to 10, the last listed
/// winning), or 0.
fn label(p: [f32; 3]) -> u64 {
    (2..ELLIPSOIDS.len())
        .rev()
        .find(|&i| inside(&ELLIPSOIDS[i], p))
        .map_or(0, |i| i as u64 + 1)
}

/// The centre of voxel `i` along an axis, in `[-1, 1]`.
fn coord(i: u32) -> f32 {
    (i as f32 + 0.5) / N as f32 * 2.0 - 1.0
}

/// The phantom sampled at `N³`, x fastest.
fn volumes() -> (Vec<f32>, Vec<u64>) {
    let mut values = Vec::with_capacity((N * N * N) as usize);
    let mut labels = Vec::with_capacity(values.capacity());
    for k in 0..N {
        for j in 0..N {
            for i in 0..N {
                // Rows run down the image: +y in the phantom is up.
                let p = [coord(i), -coord(j), coord(k)];
                values.push(density(p));
                labels.push(label(p));
            }
        }
    }
    (values, labels)
}

/// Window/level for the soft tissue: the skull saturates, the ventricles
/// and small structures separate.
const BLACK: f32 = 0.05;
const WHITE: f32 = 0.45;

/// The four planes, as (name, orientation from view axes to voxel axes,
/// centre in voxels).
fn planes() -> [(&'static str, Quaternion, Vector3); 4] {
    let c = N as f32 / 2.0;
    // Axial through the ventricles; coronal and sagittal through the middle.
    let axial_k = ((-0.25 + 1.0) / 2.0 * N as f32).floor();
    [
        ("axial", Quaternion::identity(), Vector3::new(c, c, axial_k)),
        (
            "coronal",
            // View y onto voxel z (image up is the top of the head).
            Quaternion::from_axis_angle(Vector3::new(1.0, 0.0, 0.0), -FRAC_PI_2),
            Vector3::new(c, c + 4.0, c),
        ),
        (
            "sagittal",
            // View x onto voxel y, view y onto voxel −z (image rows run down
            // the volume's z, so the crown is up).
            Quaternion::from_axis_angle(Vector3::new(0.0, 0.0, 1.0), FRAC_PI_2)
                * Quaternion::from_axis_angle(Vector3::new(1.0, 0.0, 0.0), -FRAC_PI_2),
            Vector3::new(c, c, c),
        ),
        (
            "oblique",
            // Any plane at all: the axial one tipped 35° about a diagonal.
            Quaternion::from_axis_angle(
                Vector3::new(1.0, 1.0, 0.0).normalize(),
                35f32.to_radians(),
            ),
            Vector3::new(c, c, c - 6.0),
        ),
    ]
}

/// The slices, drawn on the GPU into panes of their own.
fn slice_panes(hr: &HeadlessRenderer, values: &[f32], labels: &[u64]) -> Vec<Vec<u8>> {
    let (device, queue) = (hr.device(), hr.queue());
    let density = GpuVolume::from_texture(device, queue, &Data3DTexture::from_f32(N, N, N, values));
    let segments =
        GpuVolume::from_texture(device, queue, &Data3DTexture::from_u64(N, N, N, labels));
    let grey = SliceRenderer::window_level(device, FORMAT);
    let tint = SliceRenderer::labels(device, FORMAT);
    let grey_binding = grey.create_binding(device, &density);
    let tint_binding = tint.create_binding(device, &segments);
    let target = RenderTarget::new(device, P, P, FORMAT);
    let rect = PixelRect {
        x: 0,
        y: 0,
        width: P,
        height: P,
    };
    let voxels_per_pixel = N as f32 / (P as f32 - 24.0);
    planes()
        .iter()
        .map(|&(_, orientation, centre)| {
            let to_voxel = plane_to_voxel(
                centre,
                orientation,
                voxels_per_pixel,
                (P as f32, P as f32),
                true,
            );
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let _clear = encoder.begin_render_pass(&threers::wgpu::RenderPassDescriptor {
                    label: Some("pane clear"),
                    color_attachments: &[Some(threers::wgpu::RenderPassColorAttachment {
                        view: &target.color_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: threers::wgpu::Operations {
                            load: threers::wgpu::LoadOp::Clear(threers::wgpu::Color {
                                r: 0.006,
                                g: 0.007,
                                b: 0.01,
                                a: 1.0,
                            }),
                            store: threers::wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
            }
            // Separate uniform writes need separate submissions: each draw
            // reads its binding's uniforms when the queue runs it.
            grey.update(
                queue,
                &grey_binding,
                &SliceParams::window(to_voxel, BLACK, WHITE),
            );
            grey.draw(&mut encoder, &target.color_view, rect, &grey_binding);
            tint.update(
                queue,
                &tint_binding,
                &SliceParams::labels(to_voxel, 7, 0.35),
            );
            tint.draw(&mut encoder, &target.color_view, rect, &tint_binding);
            queue.submit([encoder.finish()]);
            readback::texture_bytes(device, queue, &target.color_texture)
        })
        .collect()
}

/// `LABELS`' colour for a label, on the CPU, so the surfaces are coloured
/// as the slices are: MurmurHash3's block mix twice, then a hue and a
/// saturation from the hash.
fn label_color(seed: u32, label: u64) -> Color {
    let mix = |seed: u32, v: u32| {
        let k = v
            .wrapping_mul(0xcc9e_2d51)
            .rotate_left(15)
            .wrapping_mul(0x1b87_3593);
        (seed ^ k)
            .rotate_left(13)
            .wrapping_mul(5)
            .wrapping_add(0xe654_6b64)
    };
    let h = mix(mix(seed, label as u32), (label >> 32) as u32);
    let hue = (h & 0xff) as f32 / 255.0;
    let s = 0.5 + 0.5 * ((h >> 8) & 0xff) as f32 / 255.0;
    let [r, g, b] = [0.0f32, 2.0, 1.0].map(|o| {
        let c = (((hue + o / 3.0).fract() * 6.0 - 3.0).abs() - 1.0).clamp(0.0, 1.0);
        1.0 + (c - 1.0) * s
    });
    Color::from_srgb([r, g, b])
}

/// The phantom's surfaces, extracted by marching cubes from each
/// ellipsoid's implicit function (continuous, so the surface is smooth
/// rather than the staircase of a voxel mask): the skull with the quarter
/// facing the camera cut away, or the segmented structures in their label
/// colours inside a faint outline of the brain.
fn surfaces(structures: bool) -> Vec<u8> {
    let mut hr = HeadlessRenderer::builder()
        .size(P, P)
        .build()
        .expect("a GPU adapter");
    hr.set_msaa(4);
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x0b0d12);
    scene.add_light(HemisphereLight::new(
        Color::from_hex(0xdfe8ff),
        Color::from_hex(0x30281f),
        1.0,
    ));
    let mut sun = DirectionalLight::new(Color::WHITE, 1.8);
    sun.direction = Vector3::new(-0.4, -0.6, -0.7).normalize();
    scene.add_light(sun);
    // Phantom x right, y toward the face, z toward the crown; the scene has
    // y up, so phantom (x, y, z) is scene (x, z, −y) and back.
    let phantom = |x: f32, y: f32, z: f32| [x, -z, y];
    let cubes = MarchingCubes::new(160, Vector3::new(2.0, 2.0, 2.0));
    let mut add = |geometry: BufferGeometry, color: Color, opacity: f32| {
        let mut m = StandardMaterial::new(color);
        m.roughness = 0.55;
        if opacity < 1.0 {
            // Blended, and (by default, being blended) not writing depth.
            m.transparent = Some(true);
            m.opacity = opacity;
        }
        scene.add(Object3D::mesh(Mesh::new(geometry, m.into())));
    };
    if structures {
        for (i, e) in ELLIPSOIDS.iter().enumerate().skip(2) {
            let surface = cubes.extract(0.0, |x, y, z| 1.0 - level(e, phantom(x, y, z)));
            add(surface, label_color(7, i as u64 + 1), 1.0);
        }
        let brain = cubes.extract(0.0, |x, y, z| 1.0 - level(&ELLIPSOIDS[1], phantom(x, y, z)));
        add(brain, Color::from_hex(0x9aa6bc), 0.12);
    } else {
        let skull = cubes.extract(0.0, |x, y, z| {
            let p = phantom(x, y, z);
            let shell = (1.0 - level(&ELLIPSOIDS[0], p)).min(level(&ELLIPSOIDS[1], p) - 1.0);
            // The quarter toward the camera (scene +x, +z) taken away.
            shell.min((-x).max(-z))
        });
        add(skull, Color::from_hex(0xe8dcc4), 1.0);
    }
    let mut camera = PerspectiveCamera::new(35.0, 1.0, 0.1, 50.0);
    camera.position = Vector3::new(2.0, 1.3, 2.6);
    camera.look_at(Vector3::new(0.0, -0.05, 0.0));
    hr.render_to_rgba(&mut scene, &camera)
}

fn main() {
    let (values, labels) = volumes();
    let hr = HeadlessRenderer::builder()
        .size(P, P)
        .color_format(FORMAT)
        .build()
        .expect("a GPU adapter");
    let mut panes = slice_panes(&hr, &values, &labels);
    panes.push(surfaces(false));
    panes.push(surfaces(true));

    let cols = 3;
    let mut sheet = vec![0u8; (P * cols * P * 2 * 4) as usize];
    for (n, px) in panes.iter().enumerate() {
        let (cx, cy) = (n as u32 % cols, n as u32 / cols);
        for y in 0..P {
            let row = &px[(y * P * 4) as usize..((y + 1) * P * 4) as usize];
            let at = (((cy * P + y) * P * cols + cx * P) * 4) as usize;
            sheet[at..at + row.len()].copy_from_slice(row);
        }
    }
    std::fs::create_dir_all("out").unwrap();
    std::fs::write("out/volume_slices.png", encode_png(P * cols, P * 2, &sheet)).unwrap();
    println!("wrote out/volume_slices.png (axial, coronal, sagittal / oblique, skull, structures)");
}
