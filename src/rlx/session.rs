//! Compiling a graph once and running it many times, with tensors on both
//! ends.
//!
//! rlx's own [`CompiledGraph`] speaks flat `&[f32]` in and `Vec<Vec<f32>>` out
//! — the shapes were the graph's business at compile time and it does not
//! repeat itself. That is the right contract for rlx and the wrong one here:
//! a frame that comes back as 3.1 million floats with no dimensions attached
//! has to be reshaped by the caller, from knowledge the graph already had.
//! [`crate::rlx::session::GraphRunner`] reads the output shapes off the graph before handing it to
//! the compiler and puts them back on the results.

use ::rlx::{CompileOptions, CompiledGraph, Device, Dim, Graph, Session};

use crate::textures::{Texture, TextureFormat};

use super::tensor::{
    frame_to_tensor, tensor_to_frame, tensor_to_texture, ColorSpace, Layout, Tensor, TensorError,
};

/// The device this bridge's work actually runs fastest on: the CPU, where one
/// is available.
///
/// Not what rlx's own [`fastest_device`](::rlx::fastest_device) answers — that
/// walks a static priority order, accelerators first, and for *these*
/// workloads it is wrong by up to sixty-fold. Measured on an M-series laptop,
/// milliseconds, best of five, `Device::Cpu` against `Device::Gpu`:
///
/// | | CPU | GPU |
/// |---|---|---|
/// | `ConvFilter` 512×512 | **11.3** | 17.6 |
/// | `Palette::extract` 8 colours | **29.7** | 67.0 |
/// | `Diffusion` 20 passes, 256² | **3.9** | 15.6 |
/// | `taubin_smooth` 8 iterations | **8.9** | 33.3 |
/// | `ColorGrade::fit` 400 steps | **479** | 29 312 |
///
/// The reason is the same in every row and it is not the arithmetic. A 512×512
/// frame is one megabyte; the work per byte is a 3×3 kernel or a matrix
/// multiply, and the GPU spends longer being handed the data than working on
/// it. `fit` is the extreme because it is four hundred *dependent* steps — a
/// round trip each, and no way to overlap them.
///
/// The exception is size. Convolution crosses over around four megapixels: at
/// 3840×2160 the GPU takes 373 ms against the CPU's 920. Pass
/// [`crate::rlx::session::Device::Gpu`] explicitly for work that big — every entry point here takes
/// the device as an argument precisely so that this default is a default and
/// not a decision.
pub fn preferred_device() -> Device {
    if ::rlx::is_available(Device::Cpu) {
        Device::Cpu
    } else {
        ::rlx::fastest_device()
    }
}

/// A compiled graph plus the shapes of what it returns.
///
/// One compile, many runs: compilation is the expensive half, and a filter
/// applied per frame would otherwise pay it sixty times a second.
pub struct GraphRunner {
    compiled: CompiledGraph,
    device: Device,
    /// Static dims per output, or `None` where the graph declared a dimension
    /// it would only know at runtime.
    output_dims: Vec<Option<Vec<usize>>>,
}

impl GraphRunner {
    /// Compile `graph` for `device`.
    pub fn new(graph: Graph, device: Device) -> Self {
        let output_dims = declared_output_dims(&graph);
        Self {
            compiled: Session::new(device).compile(graph),
            device,
            output_dims,
        }
    }

    /// Compile with explicit options — precision, fusion policy, and the rest
    /// of [`CompileOptions`].
    pub fn with_options(graph: Graph, device: Device, options: &CompileOptions) -> Self {
        let output_dims = declared_output_dims(&graph);
        Self {
            compiled: Session::new(device).compile_with(graph, options),
            device,
            output_dims,
        }
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Bind a weight by name, as `CompiledGraph::set_param` does.
    pub fn set_param(&mut self, name: &str, data: &[f32]) {
        self.compiled.set_param(name, data);
    }

    /// Bind a weight from a tensor. The shape is the caller's record of what
    /// the weight is; rlx checks it against the graph.
    pub fn set_param_tensor(&mut self, name: &str, tensor: &Tensor) {
        self.compiled.set_param(name, tensor.data());
    }

    /// Run, and reshape each result to what the graph said it would be.
    ///
    /// An output whose declared shape does not match the number of values that
    /// came back — a dynamic dimension, or a backend that returned something
    /// else — arrives rank-1 rather than wrongly folded, so a surprise is
    /// visible instead of silently reinterpreted.
    pub fn run(&mut self, inputs: &[(&str, &Tensor)]) -> Vec<Tensor> {
        let flat: Vec<(&str, &[f32])> = inputs.iter().map(|(n, t)| (*n, t.data())).collect();
        let outputs = self.compiled.run(&flat);
        outputs
            .into_iter()
            .enumerate()
            .map(
                |(i, values)| match self.output_dims.get(i).and_then(|d| d.as_ref()) {
                    Some(dims) if dims.iter().product::<usize>() == values.len() => {
                        Tensor::new(values, dims).expect("product checked above")
                    }
                    _ => Tensor::from_flat(values),
                },
            )
            .collect()
    }

    /// The unwrapped call, for callers who have their buffers flat already.
    pub fn run_flat(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        self.compiled.run(inputs)
    }

    /// The compiled graph itself, for the parts of rlx's API this does not
    /// wrap (typed parameters, slot running, device introspection).
    pub fn compiled_mut(&mut self) -> &mut CompiledGraph {
        &mut self.compiled
    }
}

/// A [`crate::rlx::session::GraphRunner`] wired to one image-shaped input, for the common case: a
/// rendered frame goes in, a frame comes back.
///
/// ```no_run
/// use threers::rlx::{FrameFilter, preferred_device};
/// use threers::rlx::{ColorSpace, Layout};
/// # fn build_graph() -> ::rlx::Graph { unimplemented!() }
///
/// let mut filter = FrameFilter::new(build_graph(), preferred_device(), "frame")
///     .layout(Layout::Nhwc)
///     .color_space(ColorSpace::Linear);
/// # let rgba = vec![0u8; 4];
/// let (out, width, height) = filter.apply(&rgba, 1, 1).unwrap();
/// ```
pub struct FrameFilter {
    runner: GraphRunner,
    input: String,
    layout: Layout,
    space: ColorSpace,
}

impl FrameFilter {
    /// Compile `graph` and remember which of its inputs the frame goes into.
    pub fn new(graph: Graph, device: Device, input: impl Into<String>) -> Self {
        Self {
            runner: GraphRunner::new(graph, device),
            input: input.into(),
            layout: Layout::Nhwc,
            space: ColorSpace::Linear,
        }
    }

    /// Channel order the graph was written for. `Nhwc` by default — the order
    /// the framebuffer is already in.
    pub fn layout(mut self, layout: Layout) -> Self {
        self.layout = layout;
        self
    }

    /// What the graph's numbers mean. `Linear` by default, because a graph
    /// that filters or blends is doing arithmetic and arithmetic belongs in
    /// linear light.
    pub fn color_space(mut self, space: ColorSpace) -> Self {
        self.space = space;
        self
    }

    /// Run the graph over one RGBA8 frame and return its first output as RGBA8
    /// bytes, with the size it came back as.
    pub fn apply(
        &mut self,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(Vec<u8>, u32, u32), TensorError> {
        let out = self.run(rgba, width, height)?;
        tensor_to_frame(&out, self.layout, self.space)
    }

    /// The same, as a [`Texture`] ready for a material. `Rgba8UnormSrgb`,
    /// which is what a colour map should be.
    pub fn apply_to_texture(
        &mut self,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Texture, TensorError> {
        let out = self.run(rgba, width, height)?;
        tensor_to_texture(&out, TextureFormat::Rgba8UnormSrgb, self.layout, self.space)
    }

    /// The first output as a tensor, for a graph whose result is not an image
    /// — a classifier's logits, a per-tile score, an embedding.
    pub fn run(&mut self, rgba: &[u8], width: u32, height: u32) -> Result<Tensor, TensorError> {
        let input = frame_to_tensor(rgba, width, height, self.layout, self.space)?;
        let mut outputs = self.runner.run(&[(self.input.as_str(), &input)]);
        if outputs.is_empty() {
            return Ok(Tensor::from_flat(Vec::new()));
        }
        Ok(outputs.remove(0))
    }

    pub fn runner_mut(&mut self) -> &mut GraphRunner {
        &mut self.runner
    }
}

/// The dimensions the graph declares for each of its outputs, where they are
/// static.
fn declared_output_dims(graph: &Graph) -> Vec<Option<Vec<usize>>> {
    graph
        .outputs
        .iter()
        .map(|id| {
            graph
                .node(*id)
                .shape
                .dims()
                .iter()
                .map(|d| match d {
                    Dim::Static(n) => Some(*n),
                    Dim::Dynamic(_) => None,
                })
                .collect::<Option<Vec<usize>>>()
        })
        .collect()
}
