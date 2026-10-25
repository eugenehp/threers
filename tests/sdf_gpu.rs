//! `threers::sdf::WGSL` against `threers::sdf` on the CPU: the same functions,
//! evaluated at the same points, on a real adapter.
//!
//! The two are written twice by hand, so the only thing keeping them the same
//! shape is this test. Every primitive and operator is evaluated at a few
//! hundred points that cover inside, outside, near the surface and the
//! degenerate cases (zero-length segments, nested capsule ends, k = 0).

use threers::math::{Vector2, Vector3};
use threers::sdf::{self, d2, d3};
use wgpu::util::DeviceExt;

/// Functions per point; the shader writes them in this order.
const FUNCS: usize = 25;

const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> pts: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&pts)) { return; }
    let q = pts[i];
    let p2 = q.xy;
    let p3 = q.xyz;
    let a = q.w;            // a second free value, for the operators
    let b = q.x - q.y;
    var r: array<f32, 25>;
    r[0] = sdf2_circle(p2, 1.3);
    r[1] = sdf2_rect(p2, vec2<f32>(2.0, 0.7));
    r[2] = sdf2_round_rect(p2, vec2<f32>(2.0, 0.7), 0.5);
    r[3] = sdf2_segment(p2, vec2<f32>(-1.0, 0.5), vec2<f32>(2.0, -1.0));
    r[4] = sdf2_capsule(p2, vec2<f32>(0.3, 0.3), vec2<f32>(0.3, 0.3), 0.4);
    let t1 = sdf2_tapered_capsule(p2, vec2<f32>(-2.0, -1.0), vec2<f32>(2.5, 1.5), 1.2, 0.3);
    r[5] = t1.x;
    r[6] = t1.y;
    r[7] = sdf2_tapered_capsule(p2, vec2<f32>(0.0, 0.0), vec2<f32>(0.4, 0.0), 1.5, 0.5).x;
    r[8] = sdf3_sphere(p3, 1.1);
    r[9] = sdf3_cuboid(p3, vec3<f32>(1.0, 0.5, 2.0));
    r[10] = sdf3_round_cuboid(p3, vec3<f32>(1.0, 0.5, 2.0), 0.3);
    r[11] = sdf3_capsule(p3, vec3<f32>(-1.0, 0.0, 1.0), vec3<f32>(1.0, 2.0, -1.0), 0.5);
    r[12] = sdf3_tapered_capsule(p3, vec3<f32>(-1.0, 0.0, 1.0), vec3<f32>(1.5, 1.0, -1.0), 0.9, 0.2);
    r[13] = sdf3_torus(p3, 1.5, 0.4);
    r[14] = sdf3_cylinder(p3, 1.0, 0.8);
    r[15] = sdf3_plane(p3, vec3<f32>(0.6, 0.8, 0.0), -0.5);
    r[16] = sdf_smooth_union(a, b, 0.7);
    r[17] = sdf_smooth_union_blend(a, b, 0.7).y;
    r[18] = sdf_smooth_intersection(a, b, 0.7);
    r[19] = sdf_smooth_subtraction(a, b, 0.7);
    r[20] = sdf_smooth_union(a, b, 0.0);
    r[21] = sdf_onion(sdf_round(a, 0.2), 0.1) + sdf_coverage(b) + sdf_subtraction(a, b) + sdf_union(a, b) + sdf_intersection(a, b);
    r[22] = sdf2_footprint(p2);
    r[23] = sdf_coverage_edge(a * 0.3, sdf2_footprint(p2));
    r[24] = sdf_coverage_stroke(a * 0.4, abs(b) * 0.15, sdf2_footprint(p2));
    for (var k = 0u; k < 25u; k = k + 1u) {
        out[i * 25u + k] = r[k];
    }
}
"#;

fn cpu(q: [f32; 4]) -> [f32; FUNCS] {
    let p2 = Vector2::new(q[0], q[1]);
    let p3 = Vector3::new(q[0], q[1], q[2]);
    let (a, b) = (q[3], q[0] - q[1]);
    let v2 = Vector2::new;
    let v3 = Vector3::new;
    let t1 = d2::tapered_capsule_radius(p2, v2(-2.0, -1.0), v2(2.5, 1.5), 1.2, 0.3);
    [
        d2::circle(p2, 1.3),
        d2::rect(p2, v2(2.0, 0.7)),
        d2::round_rect(p2, v2(2.0, 0.7), 0.5),
        d2::segment(p2, v2(-1.0, 0.5), v2(2.0, -1.0)),
        d2::capsule(p2, v2(0.3, 0.3), v2(0.3, 0.3), 0.4),
        t1.0,
        t1.1,
        d2::tapered_capsule(p2, v2(0.0, 0.0), v2(0.4, 0.0), 1.5, 0.5),
        d3::sphere(p3, 1.1),
        d3::cuboid(p3, v3(1.0, 0.5, 2.0)),
        d3::round_cuboid(p3, v3(1.0, 0.5, 2.0), 0.3),
        d3::capsule(p3, v3(-1.0, 0.0, 1.0), v3(1.0, 2.0, -1.0), 0.5),
        d3::tapered_capsule(p3, v3(-1.0, 0.0, 1.0), v3(1.5, 1.0, -1.0), 0.9, 0.2),
        d3::torus(p3, 1.5, 0.4),
        d3::cylinder(p3, 1.0, 0.8),
        d3::plane(p3, v3(0.6, 0.8, 0.0), -0.5),
        sdf::smooth_union(a, b, 0.7),
        sdf::smooth_union_blend(a, b, 0.7).1,
        sdf::smooth_intersection(a, b, 0.7),
        sdf::smooth_subtraction(a, b, 0.7),
        sdf::smooth_union(a, b, 0.0),
        sdf::onion(sdf::round(a, 0.2), 0.1)
            + sdf::coverage(b)
            + sdf::subtraction(a, b)
            + sdf::union(a, b)
            + sdf::intersection(a, b),
        sdf::footprint(p2),
        sdf::coverage_edge(a * 0.3, sdf::footprint(p2)),
        sdf::coverage_stroke(a * 0.4, b.abs() * 0.15, sdf::footprint(p2)),
    ]
}

fn points() -> Vec<[f32; 4]> {
    let mut seed = 2024u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / (1u32 << 24) as f32 * 8.0 - 4.0
    };
    let mut pts: Vec<[f32; 4]> = (0..512).map(|_| [rnd(), rnd(), rnd(), rnd()]).collect();
    // Points that land exactly on the special cases: the origin, axis
    // points, the capsule's degenerate end, a tie in the operators.
    pts.extend([
        [0.0, 0.0, 0.0, 0.0],
        [0.3, 0.3, 0.0, 0.0],
        [2.0, 0.0, 0.0, 2.0],
        [-2.0, -1.0, 1.0, -3.0],
        [2.5, 1.5, 0.0, 1.0],
        [1.5, 0.0, 0.0, 1.5],
    ]);
    pts
}

#[test]
fn the_wgsl_library_validates() {
    let module = wgpu::naga::front::wgsl::parse_str(sdf::WGSL)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(sdf::WGSL)));
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("sdf.wgsl validates");
}

#[test]
fn wgsl_matches_the_cpu() {
    let Some(gpu) = threers::gpu::testing::context("sdf parity") else {
        return;
    };
    let (device, queue) = (&gpu.device, &gpu.queue);
    let pts = points();
    let source = format!("{}\n{}", sdf::WGSL, SHADER);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sdf parity"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("sdf parity"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("points"),
        contents: bytemuck::cast_slice(&pts),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let size = (pts.len() * FUNCS * 4) as u64;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("distances"),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((pts.len() as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size);
    queue.submit([encoder.finish()]);
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let gpu_out: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped")).to_vec();

    let mut worst = (0.0f32, 0usize, 0usize);
    for (i, q) in pts.iter().enumerate() {
        let want = cpu(*q);
        for (k, &w) in want.iter().enumerate() {
            let got = gpu_out[i * FUNCS + k];
            // GPU sqrt/pow are not correctly rounded; a few ulps at the
            // magnitudes here (up to ~10) is ~1e-5.
            let err = (got - w).abs() / w.abs().max(1.0);
            if err > worst.0 {
                worst = (err, i, k);
            }
            assert!(err < 1e-4, "function {k} at {q:?}: gpu {got}, cpu {w}");
        }
    }
    eprintln!("worst relative error {:.2e} (function {} at point {})", worst.0, worst.2, worst.1);
}
