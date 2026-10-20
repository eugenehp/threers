//! The `Pass` trait: three.js's `Pass`, one unit of an [`EffectComposer`](super::EffectComposer) chain.

use super::context::PassContext;
use super::params::{ParamInfo, ParamValue};
use super::target::Target;

/// What every three.js `Pass` has: `enabled`, `needsSwap`, `clear` and
/// `renderToScreen` (which the composer sets on the last enabled pass).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PassState {
    pub enabled: bool,
    pub needs_swap: bool,
    pub clear: bool,
    pub render_to_screen: bool,
}

impl Default for PassState {
    fn default() -> Self {
        Self {
            enabled: true,
            needs_swap: true,
            clear: false,
            render_to_screen: false,
        }
    }
}

impl PassState {
    /// three.js's defaults with `needsSwap` set as the pass sets it.
    pub fn swapping(needs_swap: bool) -> Self {
        Self {
            needs_swap,
            ..Self::default()
        }
    }
}

/// `Send + Sync` natively, so a composer can live in an `Arc<Mutex<..>>`
/// alongside the rest of a renderer; nothing on wasm, where a pass may hold
/// JavaScript-owned state.
#[cfg(not(target_arch = "wasm32"))]
pub trait PassThreading: Send + Sync {}
#[cfg(not(target_arch = "wasm32"))]
impl<T: Send + Sync + ?Sized> PassThreading for T {}

#[cfg(target_arch = "wasm32")]
pub trait PassThreading {}
#[cfg(target_arch = "wasm32")]
impl<T: ?Sized> PassThreading for T {}

/// How a pass takes part in stencil masking: three.js's `MaskPass` turns it
/// on and `ClearMaskPass` off, and the composer confines every pass between
/// the two to the mask.
#[derive(Clone, Debug)]
pub enum MaskRole {
    None,
    /// A mask is on from here: what `target`'s scene covered (its depth is
    /// written there), inverted when `inverse`.
    Set { target: Target, inverse: bool },
    Clear,
}

/// One pass of an [`EffectComposer`](super::EffectComposer) chain — three.js's
/// `Pass`.
///
/// `render` gets the composer's two buffers: it reads `read` and, if it
/// [`needs_swap`](PassState::needs_swap), writes `write` (the composer then
/// swaps them); a pass that does not swap draws into `read` in place. When
/// [`render_to_screen`](PassState::render_to_screen) is set it draws to the
/// screen instead ([`Dest::Screen`](super::Dest::Screen)).
pub trait Pass: std::fmt::Debug + PassThreading {
    /// The three.js class name.
    fn name(&self) -> &'static str;

    /// The pass's flags, by value — a binding may hold the pass behind a
    /// `RefCell`, which cannot lend a reference out.
    fn state(&self) -> PassState;
    fn set_state(&mut self, state: PassState);

    /// three.js `setSize`, in device pixels — called when the pass joins a
    /// composer and whenever the composer is resized.
    fn set_size(&mut self, _width: u32, _height: u32) {}

    /// Draw this pass.
    fn render(
        &mut self,
        ctx: &mut PassContext<'_, '_>,
        write: &Target,
        read: &Target,
        delta: f32,
        mask_active: bool,
    );

    /// The pass's three.js properties and uniforms. See [`super::params`].
    fn params(&self) -> Vec<ParamInfo> {
        Vec::new()
    }

    /// The current value of a property or uniform named in [`params`](Self::params).
    fn get(&self, _name: &str) -> Option<ParamValue> {
        None
    }

    /// Assign a property or uniform; `false` if the name is not one, or the
    /// value is not its shape.
    fn set(&mut self, _name: &str, _value: ParamValue) -> bool {
        false
    }

    /// The scenes and cameras the pass draws, as indices into what the
    /// composer is given each frame: `(scene, camera)` pairs.
    fn scene_slots(&self) -> Vec<(usize, usize)> {
        Vec::new()
    }

    /// Point the pass's `n`th scene / camera reference at slots `scene` and
    /// `camera`. A binding with objects rather than indices calls this each
    /// frame with the slots it assigned.
    fn set_scene_slot(&mut self, _n: usize, _scene: usize, _camera: usize) {}

    fn mask_role(&self) -> MaskRole {
        MaskRole::None
    }

    /// For downcasting to the concrete pass.
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// The trait's boilerplate for a pass whose state is in `self.state` and
/// which is itself `'static`.
macro_rules! pass_common {
    ($name:literal) => {
        fn name(&self) -> &'static str {
            $name
        }
        fn state(&self) -> $crate::postprocessing::PassState {
            self.state
        }
        fn set_state(&mut self, state: $crate::postprocessing::PassState) {
            self.state = state;
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
    };
}
pub(crate) use pass_common;
