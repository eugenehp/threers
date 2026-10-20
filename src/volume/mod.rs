//! Volumes: 3D textures, and cross-sections through them.
//!
//! The scene graph draws surfaces. Volume data — a CT or MRI scan, a
//! microscope stack, a simulation's density field, a segmentation — is a
//! block of samples instead, and the first thing anyone does with one is look
//! at a slice of it.
//!
//! | | |
//! |---|---|
//! | [`Data3DTexture`] | three.js's `Data3DTexture`: texels on the CPU, x fastest |
//! | [`GpuVolume`] | The same in a 3D texture, written whole or a region at a time |
//! | [`SliceRenderer`] | An arbitrary plane through a volume, in a rectangle of the target |
//!
//! Like [`marks`](crate::marks), slices draw *over* a frame the
//! [`Renderer`](crate::Renderer) has already drawn, so a viewer can put a 3D
//! scene in one pane and cross-sections in the others on one target.
//!
//! Integer formats are read back exactly (`texture_3d<u32>`), so 32- and
//! 64-bit labels survive the trip to the GPU; float formats are read without
//! filtering, which works for `R32Float` on every backend including WebGL2.

mod slice;
mod texture;

pub use slice::{plane_to_voxel, SliceBinding, SliceParams, SliceRenderer, SliceTarget, LABELS, WINDOW_LEVEL};
pub use texture::{Data3DTexture, GpuVolume, VolumeFormat, VolumeSample};
