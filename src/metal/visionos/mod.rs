//! visionOS: CompositorServices and ARKit, by hand.
//!
//! An immersive visionOS app does not own a layer or a swap chain. SwiftUI hands
//! it a `cp_layer_renderer_t` — a C handle — and the app pulls frames from it:
//! query a frame, wait for the optimal input time, ask ARKit where the head is,
//! take the drawable's textures and per-eye matrices, encode, present.
//!
//! That whole loop is C, not Objective-C, so unlike the rest of this backend it
//! needs no message sending — only `extern "C"` declarations for
//! `CompositorServices.framework` and `ARKit.framework`, transcribed from the
//! visionOS SDK headers.
//!
//! # The two things visionOS insists on
//!
//! **Reverse-Z.** `drawable.h` is explicit: "It only supports reverse-Z depth,
//! which means the value in the texture should be 1 for near 0 for far." So a
//! visionOS pass is built with
//! [`with_reverse_z(true)`](crate::metal::PassAttachments::with_reverse_z), and the
//! projection comes from `cp_drawable_compute_projection`, which already agrees.
//!
//! **Two eyes.** The compositor's layout decides how they are laid out —
//! `layered` (one texture array, a slice per eye) is the one to configure, and
//! [`MetalRenderer::render_views`](crate::metal::MetalRenderer::render_views) then draws
//! both in a single pass. `dedicated` and `shared` work too, one pass per eye.
//!
//! # Getting the handle
//!
//! The Rust side is a library; the app is a SwiftUI `ImmersiveSpace` with a
//! `CompositorLayer`, and the bridge is one call:
//!
//! ```swift
//! // Swift
//! ImmersiveSpace(id: "scene") {
//!     CompositorLayer(configuration: ThreersConfiguration()) { layerRenderer in
//!         threers_visionos_run(Unmanaged.passUnretained(layerRenderer).toOpaque())
//!     }
//! }
//! ```
//!
//! ```no_run
//! # #[cfg(all(feature = "visionos", target_os = "visionos"))] {
//! use threers::metal::visionos::ImmersiveRenderer;
//! use threers::scene::Scene;
//!
//! #[no_mangle]
//! pub extern "C" fn threers_visionos_run(layer_renderer: *mut std::ffi::c_void) {
//!     let mut renderer = unsafe { ImmersiveRenderer::new(layer_renderer) }.unwrap();
//!     let mut scene = Scene::new();
//!     // …populate the scene…
//!     while renderer.render_frame(&mut scene).unwrap() {}
//! }
//! # }
//! ```

use std::ffi::c_void;

use crate::math::Matrix4;

// --------------------------------------------------------------------- types
//
// The value types are compiled for every Apple target, not just visionOS: they
// are the ABI contract with two C frameworks, and a layout that drifts is a
// silently corrupt argument list. Compiled here, the macOS test run checks them.

/// `simd_float4x4` — four columns of four floats, the same column-major layout
/// as [`Matrix4`].
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct SimdFloat4x4 {
    pub columns: [[f32; 4]; 4],
}

impl From<SimdFloat4x4> for Matrix4 {
    fn from(m: SimdFloat4x4) -> Self {
        let mut elements = [0.0f32; 16];
        for (c, column) in m.columns.iter().enumerate() {
            elements[c * 4..c * 4 + 4].copy_from_slice(column);
        }
        Matrix4 { elements }
    }
}

impl From<Matrix4> for SimdFloat4x4 {
    fn from(m: Matrix4) -> Self {
        let mut columns = [[0.0f32; 4]; 4];
        for (c, column) in columns.iter_mut().enumerate() {
            column.copy_from_slice(&m.elements[c * 4..c * 4 + 4]);
        }
        Self { columns }
    }
}

/// `simd_float2`.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct SimdFloat2 {
    pub x: f32,
    pub y: f32,
}

/// `cp_time_t` — a mach absolute time.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct CpTime {
    pub mach_abs_time: u64,
}

/// Opaque CompositorServices handles. Every one is a pointer in C.
pub type CpLayerRenderer = *mut c_void;
pub type CpFrame = *mut c_void;
pub type CpDrawable = *mut c_void;
pub type CpView = *mut c_void;
pub type CpViewTextureMap = *mut c_void;
pub type CpFrameTiming = *mut c_void;
pub type CpLayerRendererProperties = *mut c_void;
pub type CpLayerRendererConfiguration = *mut c_void;

/// ARKit handles.
pub type ArSession = *mut c_void;
pub type ArWorldTrackingProvider = *mut c_void;
pub type ArWorldTrackingConfiguration = *mut c_void;
pub type ArDataProviders = *mut c_void;
pub type ArDeviceAnchor = *mut c_void;

/// `cp_layer_renderer_state`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum LayerRendererState {
    Paused = 1,
    Running = 2,
    Invalidated = 3,
}

impl LayerRendererState {
    /// Map the raw `cp_layer_renderer_state` value. Anything unrecognised
    /// reads as [`Paused`](Self::Paused), which skips a frame rather than
    /// tearing the loop down.
    pub fn from_raw(raw: u32) -> Self {
        match raw {
            2 => Self::Running,
            3 => Self::Invalidated,
            _ => Self::Paused,
        }
    }
}

/// `cp_axis_direction_convention` — which way the NDC axes point.
///
/// [`RightUpBack`](Self::RightUpBack) is Metal's, and the one to use here: x
/// right, y up, and depth increasing towards the viewer, which is what
/// reverse-Z means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum AxisConvention {
    RightUpBack = 0,
    RightUpForward = 1,
    RightDownBack = 2,
    RightDownForward = 3,
}

/// `ar_device_anchor_query_status`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceAnchorStatus {
    Success,
    Error(isize),
}

// The frame loop itself: CompositorServices and ARKit exist only on visionOS,
// so it is compiled only there. The types above stay available on every Apple
// target, which is what lets a host app be written — and this file's ABI
// assumptions checked — from a Mac.
#[cfg(target_os = "visionos")]
mod frame_loop;

#[cfg(target_os = "visionos")]
pub use frame_loop::{ImmersiveRenderer, WorldTracking};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_matrices_round_trip() {
        let m = Matrix4::perspective(1.0, 1.5, 0.1, 100.0);
        let simd: SimdFloat4x4 = m.into();
        // Column-major both sides: column 0 is elements 0..4.
        assert_eq!(
            simd.columns[0],
            [m.elements[0], m.elements[1], m.elements[2], m.elements[3]]
        );
        let back: Matrix4 = simd.into();
        assert_eq!(back.elements, m.elements);
    }

    #[test]
    fn layer_states_map_from_the_c_enum() {
        assert_eq!(LayerRendererState::from_raw(1), LayerRendererState::Paused);
        assert_eq!(LayerRendererState::from_raw(2), LayerRendererState::Running);
        assert_eq!(
            LayerRendererState::from_raw(3),
            LayerRendererState::Invalidated
        );
        // Anything unexpected is treated as paused, which skips the frame
        // rather than tearing down the loop.
        assert_eq!(LayerRendererState::from_raw(99), LayerRendererState::Paused);
    }

    #[test]
    fn axis_conventions_match_the_header_values() {
        assert_eq!(AxisConvention::RightUpBack as u8, 0);
        assert_eq!(AxisConvention::RightUpForward as u8, 1);
        assert_eq!(AxisConvention::RightDownBack as u8, 2);
        assert_eq!(AxisConvention::RightDownForward as u8, 3);
    }

    #[test]
    fn simd_structs_have_the_c_layout() {
        assert_eq!(std::mem::size_of::<SimdFloat4x4>(), 64);
        assert_eq!(std::mem::size_of::<SimdFloat2>(), 8);
        assert_eq!(std::mem::size_of::<CpTime>(), 8);
    }
}
