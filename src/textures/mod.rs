//! Texture sources. CPU-side: own the pixel data (or a reference to it) plus
//! sampler/format metadata. The renderer uploads them on demand and caches the
//! resulting `wgpu::Texture` by `Arc` pointer identity.

pub mod bc1;
pub mod blob;
mod cube_texture;
mod data_texture;
mod depth_texture;
mod stream;
mod texture;

pub use cube_texture::{CubeMapping, CubeTexture, CubeUvAtlas};
pub use data_texture::DataTexture;
pub use depth_texture::DepthTexture;
#[cfg(target_arch = "wasm32")]
pub use stream::BrowserImage;
pub use stream::{
    Samples, StreamFrame, TextureStream, VideoTexture, YuvCoefficients, YuvFrame, YuvMatrix,
};
pub use texture::{pack_rgba16f, Texture, TextureFilter, TextureFormat, TextureWrap};
