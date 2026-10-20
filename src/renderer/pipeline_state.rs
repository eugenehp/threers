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

/// An owned copy of a `RenderPipelineDescriptor` (less the pipeline cache),
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
            additive: self
                .fragment
                .as_ref()
                .is_some_and(|(_, targets)| targets.iter().flatten().any(|t| t.blend == Some(ADDITIVE))),
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
        if !s.color_write {
            if let Some((_, targets)) = r.fragment.as_mut() {
                for t in targets.iter_mut().flatten() {
                    t.write_mask = wgpu::ColorWrites::empty();
                }
            }
        }
        if s.additive {
            if let Some((_, targets)) = r.fragment.as_mut() {
                for t in targets.iter_mut().flatten() {
                    t.blend = Some(ADDITIVE);
                }
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

    pub(crate) fn build(&self, device: &wgpu::Device, s: ResolvedState) -> wgpu::RenderPipeline {
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
        if let Some(ds) = depth_stencil.as_mut() {
            ds.depth_compare = s.compare;
            ds.depth_write_enabled = s.depth_write;
        }
        let mut targets = self
            .fragment
            .as_ref()
            .map(|(_, t)| t.clone())
            .unwrap_or_default();
        if !s.color_write {
            for t in targets.iter_mut().flatten() {
                t.write_mask = wgpu::ColorWrites::empty();
            }
        }
        if s.additive {
            for t in targets.iter_mut().flatten() {
                t.blend = Some(ADDITIVE);
            }
        }
        let fs_constants: Vec<(&str, f64)> = self
            .fragment
            .as_ref()
            .map(|(f, _)| f.constants.iter().map(|(k, v)| (k.as_str(), *v)).collect())
            .unwrap_or_default();
        let label = format!(
            "{} (material state)",
            self.label.as_deref().unwrap_or("threers")
        );
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&label),
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
            cache: None,
        })
    }
}

/// Every material pipeline's recipe, and the state variants built from them.
#[derive(Default)]
pub(crate) struct StatePipelines {
    recipes: HashMap<wgpu::RenderPipeline, PipelineRecipe>,
    variants: HashMap<(wgpu::RenderPipeline, ResolvedState), wgpu::RenderPipeline>,
    culls: HashMap<(wgpu::RenderPipeline, Option<wgpu::Face>), wgpu::RenderPipeline>,
    /// Each registered pipeline layout's bind group layouts, so a pipeline can
    /// be rebuilt with one group swapped ([`relayout`](Self::relayout)).
    layouts: HashMap<wgpu::PipelineLayout, Vec<Option<wgpu::BindGroupLayout>>>,
    relayouts: HashMap<(wgpu::RenderPipeline, u64), wgpu::RenderPipeline>,
    swapped_layouts: HashMap<(wgpu::PipelineLayout, u64), wgpu::PipelineLayout>,
    /// Reversed depth: [`variant`](Self::variant) flips every depth compare
    /// and negates the bias, default state included.
    reversed: bool,
}

impl StatePipelines {
    /// Create a pipeline and remember how, so [`variant`](Self::variant) can
    /// rebuild it.
    pub(crate) fn create(
        &mut self,
        device: &wgpu::Device,
        desc: &wgpu::RenderPipelineDescriptor,
    ) -> wgpu::RenderPipeline {
        let pipeline = device.create_render_pipeline(desc);
        self.recipes
            .insert(pipeline.clone(), PipelineRecipe::capture(desc));
        pipeline
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
        base: &wgpu::RenderPipeline,
        state: &RenderState,
    ) -> wgpu::RenderPipeline {
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
        let p = r.build(device, r.stock());
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
        base: &wgpu::RenderPipeline,
        key: u64,
        group: usize,
        bgl: &wgpu::BindGroupLayout,
        from: &wgpu::ShaderModule,
        module: &wgpu::ShaderModule,
    ) -> Option<wgpu::RenderPipeline> {
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
        if let Some((f, _)) = r.fragment.as_mut() {
            if &f.module == from {
                f.module = module.clone();
            }
        }
        let pipeline = r.build(device, r.stock());
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
        base: &wgpu::RenderPipeline,
        cull: Option<wgpu::Face>,
    ) -> wgpu::RenderPipeline {
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
                r.build(device, r.stock())
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
        let s = RenderState { blending: Blending::Additive, ..RenderState::default() };
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
}
