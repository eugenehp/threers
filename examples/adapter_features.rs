//! What the GPU on this machine will actually accept.
//!
//! Texture compression is not a portable yes/no: BC is universal on desktop and
//! absent on most mobile parts, ASTC is the other way round, and Apple silicon
//! supports both while older Macs support neither ASTC nor, before Apple GPUs,
//! anything but BC. Guessing which to author for is how an asset pipeline ends
//! up producing files the target cannot sample.
//!
//!     cargo run --release --example adapter_features

fn main() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .expect("no GPU adapter");

    let info = adapter.get_info();
    println!(
        "adapter  {} ({:?}, {:?})",
        info.name, info.device_type, info.backend
    );

    let f = adapter.features();
    let l = adapter.limits();
    println!("\ncompression");
    for (name, bit) in [
        (
            "BC   (BC1-BC7, 'DXT/S3TC')",
            wgpu::Features::TEXTURE_COMPRESSION_BC,
        ),
        ("ETC2", wgpu::Features::TEXTURE_COMPRESSION_ETC2),
        ("ASTC", wgpu::Features::TEXTURE_COMPRESSION_ASTC),
    ] {
        println!(
            "   {:<28} {}",
            name,
            if f.contains(bit) { "yes" } else { "no" }
        );
    }

    println!("\nlimits that bound a planet texture");
    println!(
        "   max_texture_dimension_2d     {}",
        l.max_texture_dimension_2d
    );
    println!(
        "   max_texture_array_layers     {}",
        l.max_texture_array_layers
    );
    println!(
        "   max_buffer_size              {:.1} GB",
        l.max_buffer_size as f64 / 2f64.powi(30)
    );
    println!(
        "   max_binding_size             {:.1} GB",
        l.max_storage_buffer_binding_size as f64 / 2f64.powi(30)
    );

    // What that means for Blue Marble at its native 500 m, which is the case
    // this was written for.
    let (w, h) = (86_400u64, 43_200u64);
    let px = w * h;
    println!(
        "\nwhole Earth at native 500 m ({w} x {h}, {:.1} Mpx)",
        px as f64 / 1e6
    );
    for (name, bytes_per_px) in [
        ("RGBA8", 4.0),
        ("BC7 / ASTC 4x4", 1.0),
        ("BC1", 0.5),
        ("ASTC 8x8", 0.25),
    ] {
        let base = px as f64 * bytes_per_px;
        // A full mip chain adds a third.
        println!(
            "   {:<16} {:6.2} GB   ({:.2} GB with mips)",
            name,
            base / 2f64.powi(30),
            base * 4.0 / 3.0 / 2f64.powi(30)
        );
    }
    let side = l.max_texture_dimension_2d as u64;
    println!(
        "\n   needs at least {} x {} tiles at the {side} limit",
        w.div_ceil(side),
        h.div_ceil(side)
    );
}
