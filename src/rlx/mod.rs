//! [RLX](https://crates.io/crates/rlx) — an ML compiler and runtime — as a
//! neighbour of the renderer.
//!
//! Nothing here changes how threers draws. The bridge is data conversion in
//! both directions and a thin wrapper around compiling a graph: vertex
//! attributes and pixels become tensors of a declared shape, tensors become
//! attributes and textures, and [`crate::rlx::GraphRunner`] compiles once so a per-frame
//! filter does not pay for the compiler sixty times a second.
//!
//! Two flags, two independent subjects:
//!
//! | Feature | Module | What it adds | Crate |
//! |---------|--------|--------------|-------|
//! | `rlx` | this one | Tensors ↔ geometry and pixels, [`crate::rlx::GraphRunner`], [`crate::rlx::FrameFilter`] | [`rlx`](https://crates.io/crates/rlx) |
//! | `rlx-geo` | [`crate::rlx::geo`] | Exact Delaunay + discrete Voronoi as geometry and textures | [`rlx-geo`](https://crates.io/crates/rlx-geo) |
//!
//! `rlx-geo` does *not* imply `rlx`: it is that crate's pure-geometry layer,
//! which needs no part of the graph runtime to do its job.
//!
//! # What is here
//!
//! The bridge itself is [`crate::rlx::Tensor`] and the conversions around it, plus
//! [`crate::rlx::GraphRunner`] and [`crate::rlx::FrameFilter`] for running a graph you wrote. On top
//! of those sit the things worth having pre-built:
//!
//! | | | |
//! |---|---|---|
//! | [`crate::rlx::ConvFilter`], [`crate::rlx::Kernel3x3`] | blur, sharpen, Sobel, emboss over a frame | `rlx` |
//! | [`crate::rlx::Diffusion`] | iterated `h ← h + κ∇²h` over a height grid | `rlx` |
//! | [`crate::rlx::mesh::laplacian_smooth`], [`crate::rlx::mesh::taubin_smooth`] | smoothing as a graph over `[n, 3]` positions | `rlx` |
//! | [`crate::rlx::ColorGrade`] | a colour transform *fitted* to a reference by autodiff | `rlx` |
//! | [`crate::rlx::Palette`] | k-means over pixels, assignment on the device | `rlx` |
//! | [`crate::rlx::geo::heightfield_geometry`], [`crate::rlx::geo::refine_heightfield`] | scattered samples → mesh, and where to sample next | `rlx-geo` |
//! | [`crate::rlx::geo::voronoi_labels`], [`crate::rlx::geo::voronoi_wall_distance`], [`crate::rlx::geo::normal_map_from_height`] | cell textures and the maps built from them | `rlx-geo` |
//!
//! Every one of these compiles for wasm32 as well as native, and `RLX=1
//! web/build.sh` (plus `RLX_GEO=1`) exports them to JavaScript as
//! `rlxConvolve`, `rlxSmoothMesh`, `rlxFitGrade`, `geoHeightfield` and the
//! rest. Nothing in this module needs a filesystem, a thread or a clock; the
//! examples that drive it do — they render headlessly and write PNGs — which
//! is why they are examples and this is a module.
//!
//! # Running a graph over a frame
//!
//! ```no_run
//! use threers::cameras::PerspectiveCamera;
//! use threers::renderer::HeadlessRenderer;
//! use threers::rlx::{preferred_device, FrameFilter};
//! use threers::Scene;
//! # fn my_graph() -> ::rlx::Graph { unimplemented!() }
//!
//! let mut headless = HeadlessRenderer::builder().size(256, 256).build().unwrap();
//! let mut scene = Scene::new();
//! let camera = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
//!
//! let frame = headless.render_to_rgba(&mut scene, &camera);
//! let mut filter = FrameFilter::new(my_graph(), preferred_device(), "frame");
//! let denoised = filter.apply_to_texture(&frame, 256, 256).unwrap();
//! ```
//!
//! # Building a graph
//!
//! The rlx API needed to build one is re-exported here, so a dependent does
//! not have to name a second copy of rlx in its own manifest and keep the two
//! versions in step:
//!
//! ```
//! use threers::rlx::prelude::*;
//! use threers::rlx::{preferred_device, GraphRunner, Tensor};
//!
//! let mut g = Graph::new("scale");
//! let x = g.input("x", Shape::new(&[1, 4], DType::F32));
//! let two = g.constant(2.0, DType::F32);
//! let y = g.mul(x, two);
//! g.set_outputs(vec![y]);
//!
//! let mut runner = GraphRunner::new(g, preferred_device());
//! let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
//! let out = runner.run(&[("x", &input)]);
//! assert_eq!(out[0].data(), &[2.0, 4.0, 6.0, 8.0]);
//! assert_eq!(out[0].dims(), &[1, 4]);
//! ```
//!
//! # Devices
//!
//! On native the dependency carries rlx's `cpu`, `tensor` and `gpu` backends.
//! rlx's `gpu` is its own wgpu, a different major version than the renderer's:
//! a graph running there has its own device, its own queue and its own copy of
//! the data. Tensors cross between the two through host memory, which is what
//! the conversions in this module do.
//!
//! **[`crate::rlx::preferred_device`] returns the CPU**, and the table in its
//! documentation is why: at the sizes this bridge works on, the GPU loses
//! every operation here — by 60× on the iterative fit, which is four hundred
//! dependent round trips. Convolution crosses over around four megapixels;
//! pass [`crate::rlx::Device::Gpu`] explicitly above that. The GPU that matters for a
//! renderer is the one drawing the frame, and this module deliberately leaves
//! it alone.
//!
//! **On wasm32 the `gpu` backend is left out, and the browser gets rlx's CPU
//! backend.** Not a policy choice — two wgpu majors in one wasm binary both
//! generate `#[wasm_bindgen]` bindings for the same WebGPU interfaces, and
//! `wasm-bindgen` rejects the result ("duplicate string enums:
//! `GpuAutoLayoutMode`"). The Rust compiles; the bindgen step is what fails, so
//! a browser build with both would produce no payload at all. The renderer
//! keeps WebGPU, rlx gets the CPU, and `rlxDevices()` reports what a page
//! actually has rather than what it hoped for.

#[cfg(feature = "rlx")]
mod filters;
#[cfg(feature = "rlx")]
mod grade;
/// Mesh operations as graphs over the positions tensor (`rlx` feature).
#[cfg(feature = "rlx")]
pub mod mesh;
#[cfg(feature = "rlx")]
mod palette;
#[cfg(feature = "rlx")]
mod session;
#[cfg(feature = "rlx")]
mod tensor;

#[cfg(feature = "rlx")]
pub use filters::{ConvFilter, Diffusion, Kernel3x3};
#[cfg(feature = "rlx")]
pub use grade::{ColorGrade, FitOptions, FitReport};
#[cfg(feature = "rlx")]
pub use palette::{Palette, PaletteOptions};
#[cfg(feature = "rlx")]
pub use session::{preferred_device, FrameFilter, GraphRunner};
#[cfg(feature = "rlx")]
pub use tensor::{
    frame_to_tensor, geometry_to_tensor, tensor_to_frame, tensor_to_geometry, tensor_to_texture,
    texture_to_tensor, ColorSpace, Layout, Tensor, TensorError,
};

/// Exact Delaunay triangulation and discrete Voronoi (`rlx-geo` feature).
#[cfg(feature = "rlx-geo")]
pub mod geo;

// ── The rlx API, re-exported ────────────────────────────────────────────
//
// Enough of rlx to build, compile, run and differentiate a graph without a
// second dependency edge. What is not here is what this build of rlx does not
// have: GGUF, ONNX, FPGA export and the rest sit behind rlx features threers
// does not enable, and a caller who wants them wants their own `rlx`
// dependency with those flags on.

/// Star-import for graph building — `Graph`, `Shape`, `DType`, `Session`,
/// `Device`, the tensor DSL, autodiff and the compile options.
#[cfg(feature = "rlx")]
pub use ::rlx::prelude;

/// `grad`, `jvp`, `hvp`, `vmap` and the higher-order helpers.
#[cfg(feature = "rlx")]
pub use ::rlx::autodiff;
/// `CompileOptions`, `Precision`, fusion policy.
#[cfg(feature = "rlx")]
pub use ::rlx::compile;
/// The `Device` enum and the backend registries.
#[cfg(feature = "rlx")]
pub use ::rlx::driver;
/// Tensor IR: ops, shapes, the graph builder.
#[cfg(feature = "rlx")]
pub use ::rlx::ir;
/// The op-builder enums: activations, binary ops, reductions, interpolation.
#[cfg(feature = "rlx")]
pub use ::rlx::ops;
/// Graph rewrites, autodiff, `vmap`.
#[cfg(feature = "rlx")]
pub use ::rlx::opt;
/// Quantisation metadata carried by the IR.
#[cfg(feature = "rlx")]
pub use ::rlx::quant;
/// `Session`, `CompiledGraph`, device discovery.
#[cfg(feature = "rlx")]
pub use ::rlx::runtime;

/// The types most call sites touch, without the star import.
#[cfg(feature = "rlx")]
pub use ::rlx::{
    CompileOptions, CompiledGraph, DType, Device, Dim, Graph, GraphExt, Session, Shape,
};

/// The sRGB transfer function and its inverse, in the piecewise form wgpu's
/// `*UnormSrgb` sampling uses — so a texture that round-trips through this
/// module samples the same as it did before.
///
/// Shared by every submodule here rather than per-file, because a bridge whose
/// two halves disagree about the encoding is a bridge that quietly changes
/// every colour that crosses it.
// Only the tensor conversions decode; `geo` produces colour and never reads it
// back.
#[cfg_attr(not(feature = "rlx"), allow(dead_code))]
fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}
