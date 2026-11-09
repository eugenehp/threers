//! The post-processing shader's `f16_round` without the f16 capability.
//!
//! It was `unpack2x16float(pack2x16float(x))`, which needs naga's
//! `SHADER_FLOAT16_IN_FLOAT32` — absent on Mesa's Vulkan drivers without
//! shader f16 (llvmpipe on arm64, for one). There the post-processing module
//! failed to build and the renderer panicked on its first frame. It is now
//! plain f32 arithmetic; these hold it to that, and to the exact IEEE f16
//! rounding it replaced.

use threers::renderer::shader::postfx_shader;
use wgpu::naga;
use wgpu::util::DeviceExt;

#[test]
fn postfx_needs_no_f16_capability() {
    let source = postfx_shader();
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("postfx: WGSL parse failed:\n{}", e.emit_to_string(source)));
    let caps = naga::valid::Capabilities::all() - naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32;
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), caps)
        .validate(&module)
        .unwrap_or_else(|e| panic!("postfx needs SHADER_FLOAT16_IN_FLOAT32 again: {e:?}"));
}

/// The `f16_round` function as the post-processing module defines it.
fn f16_round_wgsl() -> &'static str {
    let source = postfx_shader();
    let start = source.find("fn f16_round(x: f32) -> f32 {").expect("f16_round in postfx");
    let end = start + source[start..].find("\n}\n").expect("end of f16_round") + 3;
    &source[start..end]
}

/// Inputs that exercise every branch: zeros, subnormals and the normal
/// boundary, ties both ways, the top of the range and past it, inf and NaN,
/// and a sweep of ordinary colour values, all with both signs.
fn inputs() -> Vec<f32> {
    let mut v = vec![
        0.0,
        f32::MIN_POSITIVE,
        5.960_464_5e-8, // 2^-24, the smallest f16 subnormal
        2.980_232_2e-8, // half of it: a tie that goes to zero
        8.940_697e-8,   // 1.5 * 2^-24: a tie that goes up, to 2^-23
        6.097_555e-5,   // just under 2^-14
        6.103_515_6e-5, // 2^-14, the smallest f16 normal
        1.0,
        1.000_488_3, // 1 + 2^-11: halfway between 1 and the next f16, rounds to 1
        1.001_464_8, // 1 + 3 * 2^-11: halfway, rounds up to 1 + 2^-9
        0.1,
        1.0 / 3.0,
        2048.5,
        65504.0, // the largest f16
        65519.0, // rounds down to it
        65520.0, // halfway to the next binade: rounds to infinity
        1.0e6,
        f32::INFINITY,
    ];
    let mut x = 1.0e-7f32;
    while x < 7.0e4 {
        v.push(x);
        x *= 1.013;
    }
    for i in 0..4096 {
        v.push(i as f32 / 4095.0); // 12-bit colour steps
    }
    let negatives: Vec<f32> = v.iter().map(|x| -x).collect();
    v.extend(negatives);
    v.push(f32::NAN);
    v
}

#[test]
fn f16_round_matches_ieee_f16_on_the_gpu() {
    let Some(gpu) = threers::gpu::testing::context("f16_round") else {
        return;
    };
    let (device, queue) = (&gpu.device, &gpu.queue);
    let xs = inputs();
    let source = format!(
        "{}
@group(0) @binding(0) var<storage, read> xs: array<f32>;
@group(0) @binding(1) var<storage, read_write> ys: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    if id.x < arrayLength(&xs) {{
        ys[id.x] = f16_round(xs[id.x]);
    }}
}}",
        f16_round_wgsl()
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("f16_round"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("f16_round"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xs"),
        contents: bytemuck::cast_slice(&xs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let size = (xs.len() * 4) as u64;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ys"),
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
        pass.dispatch_workgroups((xs.len() as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size);
    queue.submit([encoder.finish()]);
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let ys: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped")).to_vec();

    let mut wrong = Vec::new();
    for (&x, &y) in xs.iter().zip(&ys) {
        let want = half::f16::from_f32(x).to_f32();
        // Bit-exact, signed zeros included; any NaN matches any NaN.
        let same = if want.is_nan() { y.is_nan() } else { y.to_bits() == want.to_bits() };
        if !same {
            wrong.push(format!("{x:e} ({:#010x}): got {y:e}, IEEE f16 gives {want:e}", x.to_bits()));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {} values round differently:\n{}",
        wrong.len(),
        xs.len(),
        wrong[..wrong.len().min(20)].join("\n")
    );
}
