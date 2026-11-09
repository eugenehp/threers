//! three.js's per-material fixed-function state — `depthTest`, `depthWrite`,
//! `depthFunc`, `colorWrite` — as render-pipeline variants.
//!
//! WebGPU bakes depth compare, depth write and the colour write mask into the
//! pipeline, and this renderer has a pipeline per draw path (plain, culled or
//! not, wireframe, lines, points, sprites, skinned, instanced, blended,
//! multisampled, half-float, encoded-blend, polygon-offset, shader material).
//! Rather than multiply each of those by every state combination up front,
//! every material pipeline is created through [`StatePipelines::create`],
//! which keeps an owned copy of its descriptor — a [`PipelineRecipe`]. A draw
//! whose material asks for something else gets
//! [`StatePipelines::variant`]: the same recipe with the depth-stencil state
//! and write mask replaced, built on first use and cached. Draws at the
//! defaults never leave the stock pipeline.

use crate::materials::{DepthFunc, RenderState};
use std::collections::HashMap;
use std::num::NonZeroU32;

/// three.js's `depthFunc` as a WebGPU compare function. Pipelines are built
/// for standard depth (cleared to 1, nearer is smaller), so the names carry
/// over; with reversed depth on, [`StatePipelines::variant`] flips them.
pub(crate) fn compare_function(func: DepthFunc) -> wgpu::CompareFunction {
    match func {
        DepthFunc::Never => wgpu::CompareFunction::Never,
        DepthFunc::Always => wgpu::CompareFunction::Always,
        DepthFunc::Less => wgpu::CompareFunction::Less,
        DepthFunc::LessEqual => wgpu::CompareFunction::LessEqual,
        DepthFunc::Equal => wgpu::CompareFunction::Equal,
        DepthFunc::GreaterEqual => wgpu::CompareFunction::GreaterEqual,
        DepthFunc::Greater => wgpu::CompareFunction::Greater,
        DepthFunc::NotEqual => wgpu::CompareFunction::NotEqual,
    }
}

#[derive(Clone)]
struct OwnedStage {
    module: wgpu::ShaderModule,
    entry_point: Option<String>,
    constants: Vec<(String, f64)>,
    zero_initialize_workgroup_memory: bool,
}

impl OwnedStage {
    fn capture(
        module: &wgpu::ShaderModule,
        entry_point: Option<&str>,
        options: &wgpu::PipelineCompilationOptions,
    ) -> Self {
        Self {
            module: module.clone(),
            entry_point: entry_point.map(str::to_owned),
            constants: options
                .constants
                .iter()
                .map(|(k, v)| ((*k).to_owned(), *v))
                .collect(),
            zero_initialize_workgroup_memory: options.zero_initialize_workgroup_memory,
        }
    }
}

#[derive(Clone)]
struct OwnedVertexLayout {
    array_stride: wgpu::BufferAddress,
    step_mode: wgpu::VertexStepMode,
    attributes: Vec<wgpu::VertexAttribute>,
}

/// An owned copy of a `RenderPipelineDescriptor`,
/// so the pipeline can be rebuilt with different fixed-function state.
#[derive(Clone)]
pub(crate) struct PipelineRecipe {
    label: Option<String>,
    layout: Option<wgpu::PipelineLayout>,
    vertex: OwnedStage,
    buffers: Vec<Option<OwnedVertexLayout>>,
    fragment: Option<(OwnedStage, Vec<Option<wgpu::ColorTargetState>>)>,
    primitive: wgpu::PrimitiveState,
    depth_stencil: Option<wgpu::DepthStencilState>,
    multisample: wgpu::MultisampleState,
    multiview_mask: Option<NonZeroU32>,
    cache: Option<wgpu::PipelineCache>,
}

/// What a [`RenderState`] resolves to against one pipeline: the three values
/// that end up in the descriptor. Two states that resolve alike share a
/// variant (an explicit `depth_write: Some(true)` on an opaque material is the
/// stock pipeline).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ResolvedState {
    pub compare: Option<wgpu::CompareFunction>,
    pub depth_write: Option<bool>,
    pub color_write: bool,
    /// Built for reversed depth: the compare already flipped, and the
    /// recipe's depth bias to be negated.
    pub reversed: bool,
    /// three.js AdditiveBlending on every colour target.
    pub additive: bool,
}

/// three.js `AdditiveBlending` with straight (non-premultiplied) alpha:
/// `gl.blendFunc(SRC_ALPHA, ONE)`, on colour and alpha alike.
const ADDITIVE: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::SrcAlpha,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::SrcAlpha,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
};

/// A depth compare for reversed depth (cleared to 0, nearer is larger).
pub(crate) fn reverse_compare(c: wgpu::CompareFunction) -> wgpu::CompareFunction {
    use wgpu::CompareFunction as C;
    match c {
        C::Less => C::Greater,
        C::LessEqual => C::GreaterEqual,
        C::Greater => C::Less,
        C::GreaterEqual => C::LessEqual,
        other => other,
    }
}

impl PipelineRecipe {
    pub(crate) fn capture(desc: &wgpu::RenderPipelineDescriptor) -> Self {
        Self {
            label: desc.label.map(str::to_owned),
            layout: desc.layout.cloned(),
            vertex: OwnedStage::capture(
                desc.vertex.module,
                desc.vertex.entry_point,
                &desc.vertex.compilation_options,
            ),
            buffers: desc
                .vertex
                .buffers
                .iter()
                .map(|b| {
                    b.as_ref().map(|b| OwnedVertexLayout {
                        array_stride: b.array_stride,
                        step_mode: b.step_mode,
                        attributes: b.attributes.to_vec(),
                    })
                })
                .collect(),
            fragment: desc.fragment.as_ref().map(|f| {
                (
                    OwnedStage::capture(f.module, f.entry_point, &f.compilation_options),
                    f.targets.to_vec(),
                )
            }),
            primitive: desc.primitive,
            depth_stencil: desc.depth_stencil.clone(),
            multisample: desc.multisample,
            multiview_mask: desc.multiview_mask,
            cache: desc.cache.cloned(),
        }
    }

    /// The stock pipeline's own values.
    fn stock(&self) -> ResolvedState {
        ResolvedState {
            compare: self.depth_stencil.as_ref().and_then(|d| d.depth_compare),
            depth_write: self
                .depth_stencil
                .as_ref()
                .and_then(|d| d.depth_write_enabled),
            color_write: self.fragment.as_ref().is_none_or(|(_, targets)| {
                targets
                    .iter()
                    .flatten()
                    .any(|t| t.write_mask != wgpu::ColorWrites::empty())
            }),
            reversed: false,
            additive: self.fragment.as_ref().is_some_and(|(_, targets)| {
                targets.iter().flatten().any(|t| t.blend == Some(ADDITIVE))
            }),
        }
    }

    /// This recipe with `s` applied, so it can be recorded and built on.
    fn with_state(&self, s: ResolvedState) -> PipelineRecipe {
        let mut r = self.clone();
        if let Some(ds) = r.depth_stencil.as_mut() {
            ds.depth_compare = s.compare;
            ds.depth_write_enabled = s.depth_write;
            if s.reversed {
                ds.bias.constant = -ds.bias.constant;
                ds.bias.slope_scale = -ds.bias.slope_scale;
            }
        }
        if !s.color_write
            && let Some((_, targets)) = r.fragment.as_mut()
        {
            for t in targets.iter_mut().flatten() {
                t.write_mask = wgpu::ColorWrites::empty();
            }
        }
        if s.additive
            && let Some((_, targets)) = r.fragment.as_mut()
        {
            for t in targets.iter_mut().flatten() {
                t.blend = Some(ADDITIVE);
            }
        }
        r
    }

    /// `state` applied over this pipeline. A pipeline with no depth
    /// attachment keeps having none; one with no colour keeps having none.
    pub(crate) fn resolve(&self, state: &RenderState) -> ResolvedState {
        let stock = self.stock();
        let additive = stock.additive || state.blending == crate::materials::Blending::Additive;
        let Some(ds) = self.depth_stencil.as_ref() else {
            return ResolvedState {
                color_write: stock.color_write && state.color_write,
                additive,
                ..stock
            };
        };
        let default_write = ds.depth_write_enabled.unwrap_or(false);
        ResolvedState {
            compare: Some(compare_function(state.effective_func())),
            depth_write: Some(state.effective_write(default_write)),
            color_write: stock.color_write && state.color_write,
            reversed: false,
            additive,
        }
    }

    /// Reconstruct the original descriptor exactly for first use. A material
    /// state override can intentionally normalize all color targets; initial
    /// construction must retain each target's own blend and write mask.
    fn rebuild(&self, device: &wgpu::Device, state: Option<ResolvedState>) -> wgpu::RenderPipeline {
        let buffers: Vec<Option<wgpu::VertexBufferLayout>> = self
            .buffers
            .iter()
            .map(|b| {
                b.as_ref().map(|b| wgpu::VertexBufferLayout {
                    array_stride: b.array_stride,
                    step_mode: b.step_mode,
                    attributes: &b.attributes,
                })
            })
            .collect();
        let vs_constants: Vec<(&str, f64)> = self
            .vertex
            .constants
            .iter()
            .map(|(k, v)| (k.as_str(), *v))
            .collect();
        let mut depth_stencil = self.depth_stencil.clone();
        if let (Some(ds), Some(s)) = (depth_stencil.as_mut(), state) {
            ds.depth_compare = s.compare;
            ds.depth_write_enabled = s.depth_write;
        }
        let mut targets = self
            .fragment
            .as_ref()
            .map(|(_, t)| t.clone())
            .unwrap_or_default();
        if state.is_some_and(|s| !s.color_write) {
            for t in targets.iter_mut().flatten() {
                t.write_mask = wgpu::ColorWrites::empty();
            }
        }
        if state.is_some_and(|s| s.additive) {
            for t in targets.iter_mut().flatten() {
                t.blend = Some(ADDITIVE);
            }
        }
        let fs_constants: Vec<(&str, f64)> = self
            .fragment
            .as_ref()
            .map(|(f, _)| f.constants.iter().map(|(k, v)| (k.as_str(), *v)).collect())
            .unwrap_or_default();
        let label = state.map(|_| {
            format!(
                "{} (material state)",
                self.label.as_deref().unwrap_or("threers")
            )
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: if state.is_some() {
                label.as_deref()
            } else {
                self.label.as_deref()
            },
            layout: self.layout.as_ref(),
            vertex: wgpu::VertexState {
                module: &self.vertex.module,
                entry_point: self.vertex.entry_point.as_deref(),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &vs_constants,
                    zero_initialize_workgroup_memory: self.vertex.zero_initialize_workgroup_memory,
                },
                buffers: &buffers,
            },
            primitive: self.primitive,
            depth_stencil,
            multisample: self.multisample,
            fragment: self.fragment.as_ref().map(|(f, _)| wgpu::FragmentState {
                module: &f.module,
                entry_point: f.entry_point.as_deref(),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &fs_constants,
                    zero_initialize_workgroup_memory: f.zero_initialize_workgroup_memory,
                },
                targets: &targets,
            }),
            multiview_mask: self.multiview_mask,
            cache: self.cache.as_ref(),
        })
    }
}

/// Shared identity is stable before and after first use. Cloning a handle must
/// neither compile it nor give the state-variant caches a different key.
#[derive(Clone)]
pub(crate) struct Pipeline(std::sync::Arc<PipelineInner>);

/// Built, or a recipe to build from. A deferred pipeline always has its
/// recipe; a built one has no use for it.
enum PipelineInner {
    Ready(wgpu::RenderPipeline),
    Deferred(Box<Deferred>),
}

struct Deferred {
    compiled: std::sync::OnceLock<wgpu::RenderPipeline>,
    device: wgpu::Device,
    recipe: PipelineRecipe,
    /// The material state the build applies; `None` builds the recipe as it is.
    state: Option<ResolvedState>,
    /// What validation said when a warm-up built it. A pipeline that failed
    /// is still a handle (wgpu hands back an invalid object), so without
    /// this a second warm-up would call it built and the error would be gone.
    failure: std::sync::OnceLock<String>,
}

impl Pipeline {
    pub(crate) fn create(
        device: &wgpu::Device,
        desc: &wgpu::RenderPipelineDescriptor,
        deferred: bool,
    ) -> Self {
        if deferred {
            Self::deferred(device, PipelineRecipe::capture(desc), None)
        } else {
            Self::ready(device.create_render_pipeline(desc))
        }
    }

    fn ready(pipeline: wgpu::RenderPipeline) -> Self {
        Self(std::sync::Arc::new(PipelineInner::Ready(pipeline)))
    }

    fn deferred(
        device: &wgpu::Device,
        recipe: PipelineRecipe,
        state: Option<ResolvedState>,
    ) -> Self {
        Self(std::sync::Arc::new(PipelineInner::Deferred(Box::new(
            Deferred {
                compiled: std::sync::OnceLock::new(),
                device: device.clone(),
                recipe,
                state,
                failure: std::sync::OnceLock::new(),
            },
        ))))
    }

    pub(crate) fn is_compiled(&self) -> bool {
        match &*self.0 {
            PipelineInner::Ready(_) => true,
            PipelineInner::Deferred(d) => d.compiled.get().is_some(),
        }
    }

    /// The descriptor's label, for a pipeline that has not been built yet.
    pub(crate) fn label(&self) -> Option<&str> {
        match &*self.0 {
            PipelineInner::Ready(_) => None,
            PipelineInner::Deferred(d) => d.recipe.label.as_deref(),
        }
    }

    /// What validation said when a warm-up built this, if it objected.
    pub(crate) fn failure(&self) -> Option<&str> {
        match &*self.0 {
            PipelineInner::Ready(_) => None,
            PipelineInner::Deferred(d) => d.failure.get().map(String::as_str),
        }
    }

    /// Build it now if nobody has, and say what validation made of it.
    ///
    /// Error scopes are per thread, so this catches this build's error and no
    /// one else's. A pipeline another thread is building in the meantime
    /// blocks here until it is done (`OnceLock`), and its error, if any, is
    /// that thread's to see. **Native only**: a browser cannot block on an
    /// error scope, so there this builds without reporting — use
    /// [`compile_async`](Self::compile_async).
    pub(crate) fn compile(&self) -> Result<(), String> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            pollster::block_on(self.compile_async())
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = &**self;
            self.failure().map_or(Ok(()), |f| Err(f.to_owned()))
        }
    }

    /// [`compile`](Self::compile) that awaits the error scope instead of
    /// blocking on it, so it reports validation errors in a browser too.
    pub(crate) async fn compile_async(&self) -> Result<(), String> {
        let PipelineInner::Deferred(d) = &*self.0 else {
            return Ok(());
        };
        if let Some(f) = d.failure.get() {
            return Err(f.clone());
        }
        if d.compiled.get().is_some() {
            return Ok(());
        }
        let scope = d.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _ = &**self;
        match scope.pop().await {
            Some(e) => {
                let message = e.to_string();
                let _ = d.failure.set(message.clone());
                Err(message)
            }
            None => Ok(()),
        }
    }
}

/// Where the renderer's own pipelines come from: built now or on first use
/// (`Renderer::new_lazy`), and through the application's
/// `wgpu::PipelineCache` or not. Every pipeline it defers is kept, so a
/// warm-up can build them before the first draw needs them
/// (`Renderer::warm_up`).
#[derive(Clone, Default)]
pub(crate) struct PipelineSource {
    deferred: bool,
    cache: Option<wgpu::PipelineCache>,
    /// Weak, so a variant nothing draws with any more is freed, and a built
    /// pipeline is dropped from the list the next time it is read.
    pending: std::sync::Arc<std::sync::Mutex<Vec<std::sync::Weak<PipelineInner>>>>,
}

impl PipelineSource {
    pub(crate) fn new(deferred: bool, cache: Option<wgpu::PipelineCache>) -> Self {
        Self {
            deferred,
            cache,
            ..Default::default()
        }
    }

    pub(crate) fn cache(&self) -> Option<&wgpu::PipelineCache> {
        self.cache.as_ref()
    }

    /// `desc` kept, with the cache filled in when it names none.
    pub(crate) fn recipe(&self, desc: &wgpu::RenderPipelineDescriptor) -> PipelineRecipe {
        let mut recipe = PipelineRecipe::capture(desc);
        if recipe.cache.is_none() {
            recipe.cache = self.cache.clone();
        }
        recipe
    }

    /// `desc` built now, or on first use when this source defers.
    pub(crate) fn create(
        &self,
        device: &wgpu::Device,
        desc: &wgpu::RenderPipelineDescriptor,
    ) -> Pipeline {
        if self.deferred {
            self.defer(device, self.recipe(desc), None)
        } else {
            self.create_now(device, desc)
        }
    }

    /// `recipe` with `state` applied, built now or on first use when this
    /// source defers. Variants of a pipeline come through here, so a lazy
    /// renderer's state variants can be warmed like its initial pipelines.
    pub(crate) fn build(
        &self,
        device: &wgpu::Device,
        recipe: PipelineRecipe,
        state: Option<ResolvedState>,
    ) -> Pipeline {
        if self.deferred {
            self.defer(device, recipe, state)
        } else {
            Pipeline::ready(recipe.rebuild(device, state))
        }
    }

    fn defer(
        &self,
        device: &wgpu::Device,
        recipe: PipelineRecipe,
        state: Option<ResolvedState>,
    ) -> Pipeline {
        let pipeline = Pipeline::deferred(device, recipe, state);
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(std::sync::Arc::downgrade(&pipeline.0));
        pipeline
    }

    /// `desc` built now, through the cache.
    pub(crate) fn create_now(
        &self,
        device: &wgpu::Device,
        desc: &wgpu::RenderPipelineDescriptor,
    ) -> Pipeline {
        match (&self.cache, desc.cache) {
            // The descriptor exactly as given: nothing to add to it.
            (None, _) | (_, Some(_)) => Pipeline::create(device, desc, false),
            (Some(_), None) => Pipeline::ready(self.recipe(desc).rebuild(device, None)),
        }
    }

    /// Every deferred pipeline nothing has built yet, in the order they were
    /// made. Built ones are dropped from the list, except those validation
    /// rejected, which [`failed`](Self::failed) still reports.
    pub(crate) fn uncompiled(&self) -> Vec<Pipeline> {
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = Vec::new();
        pending.retain(|weak| {
            let Some(inner) = weak.upgrade() else {
                return false;
            };
            let pipeline = Pipeline(inner);
            if !pipeline.is_compiled() {
                out.push(pipeline);
                true
            } else {
                pipeline.failure().is_some()
            }
        });
        out
    }

    /// Pipelines a warm-up built and validation rejected: `(label, message)`.
    pub(crate) fn failed(&self) -> Vec<(String, String)> {
        let pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending
            .iter()
            .filter_map(|w| w.upgrade())
            .filter_map(|inner| {
                let p = Pipeline(inner);
                p.failure()
                    .map(|m| (p.label().unwrap_or("").to_owned(), m.to_owned()))
            })
            .collect()
    }
}

impl std::ops::Deref for Pipeline {
    type Target = wgpu::RenderPipeline;

    fn deref(&self) -> &Self::Target {
        match &*self.0 {
            PipelineInner::Ready(pipeline) => pipeline,
            PipelineInner::Deferred(d) => d
                .compiled
                .get_or_init(|| d.recipe.rebuild(&d.device, d.state)),
        }
    }
}

impl PartialEq for Pipeline {
    fn eq(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for Pipeline {}
impl std::hash::Hash for Pipeline {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&std::sync::Arc::as_ptr(&self.0), state);
    }
}

/// Every material pipeline's recipe, and the state variants built from them.
#[derive(Default)]
pub(crate) struct StatePipelines {
    recipes: HashMap<Pipeline, PipelineRecipe>,
    /// The initial catalogue, which [`variants_of_initial`](Self::variants_of_initial)
    /// starts from.
    initial: Vec<Pipeline>,
    variants: HashMap<(Pipeline, ResolvedState), Pipeline>,
    culls: HashMap<(Pipeline, Option<wgpu::Face>), Pipeline>,
    /// Each registered pipeline layout's bind group layouts, so a pipeline can
    /// be rebuilt with one group swapped ([`relayout`](Self::relayout)).
    layouts: HashMap<wgpu::PipelineLayout, Vec<Option<wgpu::BindGroupLayout>>>,
    relayouts: HashMap<(Pipeline, u64), Pipeline>,
    swapped_layouts: HashMap<(wgpu::PipelineLayout, u64), wgpu::PipelineLayout>,
    /// Reversed depth: [`variant`](Self::variant) flips every depth compare
    /// and negates the bias, default state included.
    reversed: bool,
    /// How the renderer's pipelines are built; variants inherit its cache
    /// through their recipes.
    source: PipelineSource,
}

impl StatePipelines {
    /// Create a pipeline and remember how, so [`variant`](Self::variant) can
    /// rebuild it. Always builds now, even for a lazy renderer: this is for a
    /// shader material's pipeline, made when the material is first drawn, so
    /// there is nothing earlier to defer it to. The renderer's initial
    /// catalogue goes through [`create_initial`](Self::create_initial).
    pub(crate) fn create(
        &mut self,
        device: &wgpu::Device,
        desc: &wgpu::RenderPipelineDescriptor,
    ) -> Pipeline {
        let pipeline = self.source.create_now(device, desc);
        self.recipes
            .insert(pipeline.clone(), self.source.recipe(desc));
        pipeline
    }

    /// [`create`](Self::create) for the renderer's initial catalogue, which
    /// the source may defer to first use.
    pub(crate) fn create_initial(
        &mut self,
        device: &wgpu::Device,
        desc: &wgpu::RenderPipelineDescriptor,
    ) -> Pipeline {
        let pipeline = self.source.create(device, desc);
        self.recipes
            .insert(pipeline.clone(), self.source.recipe(desc));
        self.initial.push(pipeline.clone());
        pipeline
    }

    /// [`variant`](Self::variant) of every initial pipeline for each of
    /// `states`, so a lazy renderer's warm-up can build the pipelines those
    /// materials will need, and not only the stock ones. Cheap on a lazy
    /// source (it records recipes); an eager one builds them here.
    pub(crate) fn variants_of_initial(&mut self, device: &wgpu::Device, states: &[RenderState]) {
        for base in self.initial.clone() {
            for state in states {
                self.variant(device, &base, state);
            }
        }
    }

    pub(crate) fn with_source(source: PipelineSource) -> Self {
        Self {
            source,
            ..Default::default()
        }
    }

    pub(crate) fn source(&self) -> &PipelineSource {
        &self.source
    }

    /// Draw the main view with reversed depth (or not). Every draw against
    /// the view's depth goes through [`variant`](Self::variant), which is
    /// where the compares flip; shadow passes do not, and stay standard.
    pub(crate) fn set_reversed(&mut self, on: bool) {
        self.reversed = on;
    }

    pub(crate) fn reversed(&self) -> bool {
        self.reversed
    }

    /// `base` with `state` applied — and with reversed depth on, flipped:
    /// `base` itself when that changes nothing (or when `base` was not made
    /// by [`create`](Self::create)). The variant's recipe is recorded, so
    /// [`cull_variant`](Self::cull_variant) applies to it; it used not to be,
    /// and a variant then silently kept its stock culling.
    pub(crate) fn variant(
        &mut self,
        device: &wgpu::Device,
        base: &Pipeline,
        state: &RenderState,
    ) -> Pipeline {
        if state.is_default() && !self.reversed {
            return base.clone();
        }
        let Some(recipe) = self.recipes.get(base) else {
            return base.clone();
        };
        let mut resolved = if state.is_default() {
            recipe.stock()
        } else {
            recipe.resolve(state)
        };
        if self.reversed && recipe.depth_stencil.is_some() {
            resolved.compare = resolved.compare.map(reverse_compare);
            resolved.reversed = true;
        }
        if resolved == recipe.stock() {
            return base.clone();
        }
        if let Some(p) = self.variants.get(&(base.clone(), resolved)) {
            return p.clone();
        }
        let r = recipe.with_state(resolved);
        let p = self.source.build(device, r.clone(), Some(r.stock()));
        self.recipes.insert(p.clone(), r);
        self.variants.insert((base.clone(), resolved), p.clone());
        p
    }

    /// Remember `layout`'s bind group layouts.
    pub(crate) fn register_layout(
        &mut self,
        layout: &wgpu::PipelineLayout,
        groups: Vec<Option<wgpu::BindGroupLayout>>,
    ) {
        self.layouts.insert(layout.clone(), groups);
    }

    /// `base` rebuilt with group `group` of its layout replaced by `bgl` and
    /// every stage that ran `from` running `module` instead — a material
    /// shader variant (see `material_maps`). Cached under `key`, which must
    /// name the (`bgl`, `module`) pair; the result is recorded, so
    /// [`variant`](Self::variant) and [`cull_variant`](Self::cull_variant)
    /// apply to it as to a stock pipeline. `None` when `base` was not made by
    /// [`create`](Self::create) or its layout was not registered.
    pub(crate) fn relayout(
        &mut self,
        device: &wgpu::Device,
        base: &Pipeline,
        key: u64,
        group: usize,
        bgl: &wgpu::BindGroupLayout,
        from: &wgpu::ShaderModule,
        module: &wgpu::ShaderModule,
    ) -> Option<Pipeline> {
        if let Some(p) = self.relayouts.get(&(base.clone(), key)) {
            return Some(p.clone());
        }
        let recipe = self.recipes.get(base)?;
        let old_layout = recipe.layout.clone()?;
        let groups = self.layouts.get(&old_layout)?.clone();
        let layout = self
            .swapped_layouts
            .entry((old_layout, key))
            .or_insert_with(|| {
                let mut groups = groups.clone();
                if group < groups.len() {
                    groups[group] = Some(bgl.clone());
                }
                let refs: Vec<Option<&wgpu::BindGroupLayout>> =
                    groups.iter().map(|g| g.as_ref()).collect();
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("threers material variant layout"),
                    bind_group_layouts: &refs,
                    immediate_size: 0,
                })
            })
            .clone();
        let mut r = recipe.clone();
        r.layout = Some(layout.clone());
        if &r.vertex.module == from {
            r.vertex.module = module.clone();
        }
        if let Some((f, _)) = r.fragment.as_mut()
            && &f.module == from
        {
            f.module = module.clone();
        }
        let pipeline = self.source.build(device, r.clone(), Some(r.stock()));
        self.layouts.entry(layout).or_insert_with(|| {
            let mut g = groups;
            if group < g.len() {
                g[group] = Some(bgl.clone());
            }
            g
        });
        self.recipes.insert(pipeline.clone(), r);
        self.relayouts.insert((base.clone(), key), pipeline.clone());
        Some(pipeline)
    }

    /// `base` culling `cull` instead: `base` itself when it already does (or
    /// when `base` was not made by [`create`](Self::create)). The shadow
    /// passes take three.js's `shadowSide` this way — a caster's side picks
    /// which faces its shadow map records.
    pub(crate) fn cull_variant(
        &mut self,
        device: &wgpu::Device,
        base: &Pipeline,
        cull: Option<wgpu::Face>,
    ) -> Pipeline {
        let Some(recipe) = self.recipes.get(base) else {
            return base.clone();
        };
        if recipe.primitive.cull_mode == cull {
            return base.clone();
        }
        self.culls
            .entry((base.clone(), cull))
            .or_insert_with(|| {
                let mut r = recipe.clone();
                r.primitive.cull_mode = cull;
                self.source.build(device, r.clone(), Some(r.stock()))
            })
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_constants_round_trip() {
        for v in 0..8 {
            assert_eq!(DepthFunc::from_three(v).unwrap() as u32, v);
        }
        assert_eq!(DepthFunc::from_three(8), None);
        assert_eq!(DepthFunc::default(), DepthFunc::LessEqual);
        // Reverse-Z flips the ordering tests and nothing else.
        assert_eq!(DepthFunc::LessEqual.reversed(), DepthFunc::GreaterEqual);
        assert_eq!(DepthFunc::Greater.reversed(), DepthFunc::Less);
        for f in [
            DepthFunc::Never,
            DepthFunc::Always,
            DepthFunc::Equal,
            DepthFunc::NotEqual,
        ] {
            assert_eq!(f.reversed(), f);
        }
    }

    #[test]
    fn additive_blending_is_its_own_state() {
        use crate::materials::Blending;
        assert_eq!(Blending::from_three(1), Some(Blending::Normal));
        assert_eq!(Blending::from_three(2), Some(Blending::Additive));
        assert_eq!(Blending::from_three(5), None); // MultiplyBlending: not applied
        assert_eq!(RenderState::default().blending, Blending::Normal);
        // a non-default state is what routes a draw to a pipeline variant
        let s = RenderState {
            blending: Blending::Additive,
            ..RenderState::default()
        };
        assert!(!s.is_default());
    }

    #[test]
    fn depth_test_off_also_stops_writes() {
        let s = RenderState {
            depth_test: false,
            depth_write: Some(true),
            ..RenderState::default()
        };
        assert!(!s.effective_write(true));
        assert_eq!(s.effective_func(), DepthFunc::Always);
        let s = RenderState::default();
        assert!(s.effective_write(true));
        assert!(!s.effective_write(false));
    }

    /// A lazy recipe must preserve each target's blend and write mask. A cache key must
    /// remain stable after the underlying pipeline has been compiled.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn deferred_pipeline_preserves_descriptor_and_cache_identity() {
        use std::sync::Arc;

        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(desc);
        let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let device = Arc::new(device);
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mixed blend recipe probe"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let p = array<vec2f, 3>(vec2f(-1., -1.), vec2f(3., -1.), vec2f(-1., 3.));
    return vec4f(p[i], 0., 1.);
}
struct Outputs { @location(0) a: vec4f, @location(1) b: vec4f }
@fragment fn fs() -> Outputs {
    return Outputs(vec4f(0.25, 0.4, 0.6, 0.5), vec4f(0.25, 0.4, 0.6, 0.5));
}
"#
                .into(),
            ),
        });
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let targets = [
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: Some(additive),
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::RED | wgpu::ColorWrites::GREEN,
            }),
        ];
        let desc = wgpu::RenderPipelineDescriptor {
            label: Some("mixed blend recipe"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &targets,
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        };
        let mut images = Vec::new();
        for deferred in [false, true] {
            let pipe = Pipeline::create(&device, &desc, deferred);
            assert_eq!(!pipe.is_compiled(), deferred);
            let cloned = pipe.clone();
            // Cache identity must survive compilation of a previously pending handle.
            #[expect(
                clippy::mutable_key_type,
                reason = "hash and equality use only immutable Arc identity; this test checks that contract"
            )]
            let mut keys = std::collections::HashSet::new();
            keys.insert(cloned);
            assert_eq!(!pipe.is_compiled(), deferred);
            let a = crate::RenderTarget::new(&device, 4, 4, wgpu::TextureFormat::Rgba8Unorm);
            let b = crate::RenderTarget::new(&device, 4, 4, wgpu::TextureFormat::Rgba8Unorm);
            let attachment = |view| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.2,
                            g: 0.3,
                            b: 0.4,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })
            };
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[attachment(&a.color_view), attachment(&b.color_view)],
                    ..Default::default()
                });
                pass.set_pipeline(&pipe);
                pass.draw(0..3, 0..1);
            }
            assert!(pipe.is_compiled());
            assert!(
                keys.contains(&pipe),
                "pipeline identity changed after compilation"
            );
            queue.submit([encoder.finish()]);
            images.push((
                crate::renderer::readback::texture_bytes(&device, &queue, &a.color_texture),
                crate::renderer::readback::texture_bytes(&device, &queue, &b.color_texture),
            ));
        }
        assert!(
            pollster::block_on(validation.pop()).is_none(),
            "wgpu validation error"
        );
        assert_eq!(
            images[0], images[1],
            "lazy recipe changed target-specific blend state"
        );
        assert_ne!(
            images[0].0, images[0].1,
            "fixture must distinguish the two blend states"
        );
        eprintln!("[lazy-recipe] PASS exact mixed-target blends and stable cache identity");
    }

    /// A pipeline validation rejects is reported by every later check, not
    /// just the first, and a variant nothing holds any more is not kept.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn rejected_pipeline_stays_reported_and_dropped_variants_are_freed() {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(desc);
        let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let (device, _queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rejected pipeline probe"),
            source: wgpu::ShaderSource::Wgsl(
                "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(0.); }\n\
                 @fragment fn fs() -> @location(0) vec4f { return vec4f(1.); }"
                    .into(),
            ),
        });
        let targets = [Some(wgpu::ColorTargetState::from(
            wgpu::TextureFormat::Rgba8Unorm,
        ))];
        let desc = |entry: &'static str| wgpu::RenderPipelineDescriptor {
            label: Some("probe"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some(entry),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &targets,
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        };
        let source = PipelineSource::new(true, None);
        let good = source.create(&device, &desc("vs"));
        let bad = source.create(&device, &desc("no_such_entry_point"));
        assert_eq!(source.uncompiled().len(), 2);
        assert_eq!(good.compile(), Ok(()));
        assert!(bad.compile().is_err());
        // Asked again, a pipeline that failed still says so…
        assert!(bad.compile().is_err());
        assert!(pollster::block_on(bad.compile_async()).is_err());
        assert_eq!(source.failed().len(), 1);
        assert_eq!(source.failed()[0].0, "probe");
        // …and neither built pipeline is left to warm.
        assert!(source.uncompiled().is_empty());
        // Handles dropped: the source does not keep them alive.
        drop((good, bad));
        assert!(source.failed().is_empty());
        let dropped = source.create(&device, &desc("vs"));
        drop(dropped);
        assert!(source.uncompiled().is_empty());
    }
}
