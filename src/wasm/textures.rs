//! Bindings split out of `wasm.rs` by area (see the `mod` list there): textures.

use super::*;

// ---- textures ----
#[wasm_bindgen]
pub struct WebTexture {
    pub(super) inner: Arc<crate::Texture>,
}
#[wasm_bindgen]
impl WebTexture {
    /// three.js `texture.colorSpace`: `true` for `SRGBColorSpace` (the bytes
    /// are sRGB-encoded and the sampler decodes them), `false` for none or
    /// linear. Applies to the 8-bit RGBA formats; others are left as they are.
    /// A material bound to the texture before the change keeps the old one
    /// until it is bound again.
    #[wasm_bindgen(js_name = setSrgb)]
    pub fn set_srgb(&mut self, srgb: bool) {
        use crate::TextureFormat as F;
        let want = if srgb {
            F::Rgba8UnormSrgb
        } else {
            F::Rgba8Unorm
        };
        if matches!(self.inner.format, F::Rgba8Unorm | F::Rgba8UnormSrgb)
            && self.inner.format != want
        {
            Arc::make_mut(&mut self.inner).format = want;
        }
    }
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> WebTexture {
        WebTexture {
            inner: Arc::new(crate::Texture::new(
                width,
                height,
                crate::TextureFormat::Rgba8UnormSrgb,
                data,
            )),
        }
    }
    #[wasm_bindgen(js_name = solid)]
    pub fn solid(r: u8, g: u8, b: u8, a: u8) -> WebTexture {
        WebTexture {
            inner: Arc::new(crate::Texture::solid(
                [r, g, b, a],
                crate::TextureFormat::Rgba8UnormSrgb,
            )),
        }
    }
    /// Set sampler filter/wrap — see [`set_texture_filters`].
    #[wasm_bindgen(js_name = setFilters)]
    pub fn set_filters(&mut self, mag: u32, min: u32, wrap_s: u32, wrap_t: u32) {
        set_texture_filters(Arc::make_mut(&mut self.inner), mag, min, wrap_s, wrap_t);
    }

    /// Resample this equirectangular map onto the six faces of a cube.
    ///
    /// A cube has no poles. An equirectangular map converges every longitude
    /// onto one texel at each end, and anything with width near there fans out
    /// radially when it is wrapped on a sphere; a cube face is a plane, its
    /// texels are near enough uniform everywhere, and no point on it is
    /// special. Returns the faces in `+X, -X, +Y, -Y, +Z, -Z` order, which is
    /// the order `planet::CUBE_FACES` places them in.
    ///
    /// `face_size` of 0 picks a quarter of the map's width — four faces carry
    /// the 360 degrees the map spends its full width on, so that is lossless at
    /// the equator and a gain toward the poles.
    #[cfg(feature = "planet")]
    #[wasm_bindgen(js_name = cubeFaces)]
    pub fn cube_faces(&self, face_size: u32) -> js_sys::Array {
        let n = if face_size == 0 {
            (self.inner.width / 4).clamp(64, 4096)
        } else {
            face_size
        };
        let out = js_sys::Array::new();
        for face in crate::planet::equirect_to_cube_faces(&self.inner, n) {
            out.push(&JsValue::from(WebTexture {
                inner: Arc::new(face),
            }));
        }
        out
    }

    /// A second handle on the same pixels, carrying its own UV transform.
    ///
    /// `Texture`'s bytes live behind their own `Arc`, so this clones a handle
    /// and not an image — and the renderer keys its GPU cache on those bytes,
    /// so every view of one buffer shares a single upload between them.
    ///
    /// This is what lets a body split into patches for tile streaming still
    /// cost one texture per map at level 0: eight patches over one map, each
    /// addressing its own quarter through `offset`/`repeat`, instead of eight
    /// sliced copies uploaded separately.
    #[wasm_bindgen(js_name = view)]
    pub fn view(&self, ox: f32, oy: f32, rx: f32, ry: f32) -> WebDataTexture {
        let mut inner = (*self.inner).clone();
        inner.offset = crate::math::Vector2::new(ox, oy);
        inner.repeat = crate::math::Vector2::new(rx, ry);
        WebDataTexture {
            inner: Arc::new(inner),
        }
    }

    /// three.js `Texture.offset` / `Texture.repeat` — the UV sub-rectangle this
    /// texture covers. What lets several meshes share one atlas.
    #[wasm_bindgen(js_name = setUvTransform)]
    pub fn set_uv_transform(&mut self, ox: f32, oy: f32, rx: f32, ry: f32) {
        let inner = Arc::make_mut(&mut self.inner);
        inner.offset = crate::math::Vector2::new(ox, oy);
        inner.repeat = crate::math::Vector2::new(rx, ry);
        inner.rotation = 0.0;
        inner.center = crate::math::Vector2::ZERO;
        inner.matrix_auto_update = true;
    }

    /// three.js `Texture.channel`: the uv set the texture is read through —
    /// 0 the geometry's `uv`, 1–3 its `uv1`, `uv2`, `uv3`.
    #[wasm_bindgen(js_name = setChannel)]
    pub fn set_channel(&mut self, channel: u32) {
        Arc::make_mut(&mut self.inner).channel = channel.min(3);
    }

    /// three.js `Texture.matrix` as its top two rows, `(a, b, c)` and
    /// `(d, e, f)`: `u' = a·u + b·v + c`, `v' = d·u + e·v + f`. What
    /// `offset`, `repeat`, `rotation` and `center` compose to
    /// (`Matrix3.setUvTransform`), or a matrix set by hand.
    #[wasm_bindgen(js_name = setUvMatrix)]
    pub fn set_uv_matrix(&mut self, a: f32, b: f32, c: f32, d: f32, e: f32, f: f32) {
        set_uv_matrix(Arc::make_mut(&mut self.inner), [a, b, c, d, e, f]);
    }
    pub fn width(&self) -> u32 {
        self.inner.width
    }
    pub fn height(&self) -> u32 {
        self.inner.height
    }
}

#[wasm_bindgen]
pub struct WebCubeTexture {
    pub(super) inner: Arc<crate::CubeTexture>,
}
#[wasm_bindgen]
impl WebCubeTexture {
    #[wasm_bindgen(constructor)]
    pub fn new(
        size: u32,
        px: Vec<u8>,
        nx: Vec<u8>,
        py: Vec<u8>,
        ny: Vec<u8>,
        pz: Vec<u8>,
        nz: Vec<u8>,
    ) -> WebCubeTexture {
        WebCubeTexture {
            inner: Arc::new(crate::CubeTexture::new(
                size,
                crate::TextureFormat::Rgba8UnormSrgb,
                [px, nx, py, ny, pz, nz],
            )),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn size(&self) -> u32 {
        self.inner.size
    }

    /// three.js `cubeTexture.magFilter` / `minFilter`, in its constants (see
    /// `three_filter`).
    #[wasm_bindgen(js_name = setFilters)]
    pub fn set_filters(&mut self, mag: u32, min: u32) {
        let c = Arc::make_mut(&mut self.inner);
        c.mag_filter = three_filter(mag).0;
        (c.min_filter, c.mipmap_filter) = three_filter(min);
    }

    /// three.js `cubeTexture.colorSpace`: `true` (`SRGBColorSpace`, what
    /// `CubeTextureLoader` sets) decodes the faces on sampling; `false` reads
    /// them as they are.
    #[wasm_bindgen(js_name = setSrgb)]
    pub fn set_srgb(&mut self, srgb: bool) {
        Arc::make_mut(&mut self.inner).format = if srgb {
            crate::TextureFormat::Rgba8UnormSrgb
        } else {
            crate::TextureFormat::Rgba8Unorm
        };
    }

    /// three.js `cubeTexture.mapping`: `true` for `CubeRefractionMapping`,
    /// `false` for the default `CubeReflectionMapping`.
    #[wasm_bindgen(js_name = setRefraction)]
    pub fn set_refraction(&mut self, refraction: bool) {
        Arc::make_mut(&mut self.inner).mapping = if refraction {
            crate::textures::CubeMapping::Refraction
        } else {
            crate::textures::CubeMapping::Reflection
        };
    }

    #[wasm_bindgen(js_name = sampleCubeUvEnv)]
    pub fn sample_cube_uv_env(&self, dx: f32, dy: f32, dz: f32, roughness: f32) -> Vec<f32> {
        let Some(atlas) = self.inner.cube_uv_atlas.as_ref() else {
            return vec![0.0, 0.0, 0.0];
        };
        crate::extras::cube_uv::sample_cube_uv_env(atlas, [dx, dy, dz], roughness).to_vec()
    }
}

#[wasm_bindgen]
pub struct WebPmremGenerator;
#[wasm_bindgen]
impl WebPmremGenerator {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebPmremGenerator {
        WebPmremGenerator
    }

    /// Prefilter `cube` into a PMREM mip chain at `size` and return a CubeTexture wrapper.
    #[wasm_bindgen(js_name = fromCubemap)]
    pub fn from_cubemap(cube: &WebCubeTexture, size: u32) -> WebCubeTexture {
        let pmrem = crate::PmremGenerator::generate_pmrem(&cube.inner, size.max(1));
        WebCubeTexture {
            inner: Arc::new(pmrem),
        }
    }

    /// three.js's `pmrem.fromScene(new RoomEnvironment())`: the room built
    /// exactly (`extras::RoomEnvironment`, HDR) and prefiltered at `size`.
    #[wasm_bindgen(js_name = roomEnvironment)]
    pub fn room_environment(size: u32) -> WebCubeTexture {
        WebCubeTexture {
            inner: crate::extras::RoomEnvironment::pmrem(size.clamp(16, 256)),
        }
    }

    /// Convert an equirectangular RGBA8 texture into a PMREM cubemap.
    #[wasm_bindgen(js_name = fromEquirectangular)]
    pub fn from_equirectangular(
        data: Vec<u8>,
        src_w: u32,
        src_h: u32,
        cube_size: u32,
    ) -> WebCubeTexture {
        let cube = crate::PmremGenerator::from_equirect(&data, src_w, src_h, cube_size.max(1));
        let pmrem = crate::PmremGenerator::generate_pmrem(&cube, cube_size.max(1));
        WebCubeTexture {
            inner: Arc::new(pmrem),
        }
    }
}

#[wasm_bindgen]
pub struct WebDataTexture {
    pub(super) inner: Arc<crate::Texture>,
}

/// The `WebDataTexture` around a texture, for the bindings in other threers
/// crates (`threers-video`'s `WebVideoPlayer`). Not exported to JavaScript.
impl WebDataTexture {
    #[doc(hidden)]
    pub fn from_texture(inner: Arc<crate::Texture>) -> Self {
        Self { inner }
    }
}

#[wasm_bindgen]
impl WebDataTexture {
    /// three.js `texture.colorSpace`: `true` for `SRGBColorSpace` (the bytes
    /// are sRGB-encoded and the sampler decodes them), `false` for none or
    /// linear. Applies to the 8-bit RGBA formats; others are left as they are.
    /// A material bound to the texture before the change keeps the old one
    /// until it is bound again.
    #[wasm_bindgen(js_name = setSrgb)]
    pub fn set_srgb(&mut self, srgb: bool) {
        use crate::TextureFormat as F;
        let want = if srgb {
            F::Rgba8UnormSrgb
        } else {
            F::Rgba8Unorm
        };
        if matches!(self.inner.format, F::Rgba8Unorm | F::Rgba8UnormSrgb)
            && self.inner.format != want
        {
            Arc::make_mut(&mut self.inner).format = want;
        }
    }
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> WebDataTexture {
        WebDataTexture {
            inner: Arc::new(crate::DataTexture::new(
                width,
                height,
                crate::TextureFormat::Rgba8Unorm,
                data,
            )),
        }
    }
    /// A texture whose bytes are sRGB-encoded — what every JPEG and PNG colour
    /// map is.
    ///
    /// The default constructor uploads `Rgba8Unorm`, i.e. the sampler reads the
    /// bytes as linear. Feeding it an ordinary image makes the darks far too
    /// bright; this asks the hardware to decode instead, which is both correct
    /// and free. Use it for colour maps, never for normal/roughness data.
    #[wasm_bindgen(js_name = newSrgb)]
    pub fn new_srgb(width: u32, height: u32, data: Vec<u8>) -> WebDataTexture {
        WebDataTexture {
            inner: Arc::new(crate::DataTexture::new(
                width,
                height,
                crate::TextureFormat::Rgba8UnormSrgb,
                data,
            )),
        }
    }

    /// A texture fed with frames, as three.js's `VideoTexture` is: it stays
    /// the same texture whatever is written, so the renderer copies each frame
    /// into the GPU texture it already has. Oriented and formatted as the
    /// plain constructor's, so the shim's UV handling is the same for both.
    #[wasm_bindgen(js_name = newStreamed)]
    pub fn new_streamed(width: u32, height: u32) -> WebDataTexture {
        let video = crate::textures::VideoTexture::new(width, height);
        let mut texture = (*video.texture()).clone();
        texture.flip_y = false;
        texture.format = crate::TextureFormat::Rgba8Unorm;
        WebDataTexture {
            inner: Arc::new(texture),
        }
    }

    /// Show a new frame in a texture made by `newStreamed`: RGBA8,
    /// `width * height * 4` bytes. False if it is not streamed or the length
    /// is wrong.
    #[wasm_bindgen(js_name = writeFrame)]
    pub fn write_frame(&self, data: &[u8]) -> bool {
        self.inner
            .stream
            .as_ref()
            .is_some_and(|s| s.write_rgba(data))
    }

    /// Show the browser's own decoded picture in a texture made by
    /// `newStreamed`: a `<video>` element (its current frame) or a WebCodecs
    /// `VideoFrame` (owned from here on and closed when the next replaces
    /// it). The renderer copies it GPU to GPU, never through the CPU, and the
    /// browser decodes in hardware where it can: how 4K and 8K video plays on
    /// a mesh. False if the texture is not streamed, the source is neither,
    /// or its size is not the texture's.
    #[wasm_bindgen(js_name = writeVideo)]
    pub fn write_video(&self, source: JsValue) -> bool {
        use crate::textures::BrowserImage;
        use wasm_bindgen::JsCast;
        let Some(stream) = self.inner.stream.as_ref() else {
            return false;
        };
        let image = match source.dyn_into::<web_sys::HtmlVideoElement>() {
            Ok(video) => BrowserImage::Video(video),
            Err(other) => match other.dyn_into::<web_sys::VideoFrame>() {
                Ok(frame) => BrowserImage::Frame(frame),
                Err(_) => return false,
            },
        };
        stream.write_browser(image)
    }

    /// Replace RGBA8 level-0 bytes without allocating a new texture id.
    #[wasm_bindgen(js_name = replaceRgba)]
    pub fn replace_rgba(&mut self, data: Vec<u8>) -> bool {
        let inner = Arc::make_mut(&mut self.inner);
        inner.replace_rgba_bytes(data)
    }

    #[wasm_bindgen(js_name = setFilters)]
    pub fn set_filters(&mut self, mag: u32, min: u32, wrap_s: u32, wrap_t: u32) {
        set_texture_filters(Arc::make_mut(&mut self.inner), mag, min, wrap_s, wrap_t);
    }

    /// Resample this equirectangular map onto the six faces of a cube.
    ///
    /// See [`WebTexture::cube_faces`]. Both sky paths need it — the generated
    /// starfield arrives as a `DataTexture`, and leaving it on a sphere left
    /// the pole exactly as it was.
    #[cfg(feature = "planet")]
    #[wasm_bindgen(js_name = cubeFaces)]
    pub fn cube_faces(&self, face_size: u32) -> js_sys::Array {
        let n = if face_size == 0 {
            (self.inner.width / 4).clamp(64, 4096)
        } else {
            face_size
        };
        let out = js_sys::Array::new();
        for face in crate::planet::equirect_to_cube_faces(&self.inner, n) {
            out.push(&JsValue::from(WebDataTexture {
                inner: Arc::new(face),
            }));
        }
        out
    }

    /// A second handle on the same pixels, carrying its own UV transform.
    ///
    /// `Texture`'s bytes live behind their own `Arc`, so this clones a handle
    /// and not an image — and the renderer keys its GPU cache on those bytes,
    /// so every view of one buffer shares a single upload between them.
    #[wasm_bindgen(js_name = view)]
    pub fn view(&self, ox: f32, oy: f32, rx: f32, ry: f32) -> WebDataTexture {
        let mut inner = (*self.inner).clone();
        inner.offset = crate::math::Vector2::new(ox, oy);
        inner.repeat = crate::math::Vector2::new(rx, ry);
        WebDataTexture {
            inner: Arc::new(inner),
        }
    }

    /// three.js `Texture.offset` / `Texture.repeat` — the UV sub-rectangle this
    /// texture covers. What lets several meshes share one atlas.
    #[wasm_bindgen(js_name = setUvTransform)]
    pub fn set_uv_transform(&mut self, ox: f32, oy: f32, rx: f32, ry: f32) {
        let inner = Arc::make_mut(&mut self.inner);
        inner.offset = crate::math::Vector2::new(ox, oy);
        inner.repeat = crate::math::Vector2::new(rx, ry);
        inner.rotation = 0.0;
        inner.center = crate::math::Vector2::ZERO;
        inner.matrix_auto_update = true;
    }

    /// three.js `Texture.channel`: the uv set the texture is read through —
    /// 0 the geometry's `uv`, 1–3 its `uv1`, `uv2`, `uv3`.
    #[wasm_bindgen(js_name = setChannel)]
    pub fn set_channel(&mut self, channel: u32) {
        Arc::make_mut(&mut self.inner).channel = channel.min(3);
    }

    /// three.js `Texture.matrix` as its top two rows, `(a, b, c)` and
    /// `(d, e, f)`: `u' = a·u + b·v + c`, `v' = d·u + e·v + f`. What
    /// `offset`, `repeat`, `rotation` and `center` compose to
    /// (`Matrix3.setUvTransform`), or a matrix set by hand.
    #[wasm_bindgen(js_name = setUvMatrix)]
    pub fn set_uv_matrix(&mut self, a: f32, b: f32, c: f32, d: f32, e: f32, f: f32) {
        set_uv_matrix(Arc::make_mut(&mut self.inner), [a, b, c, d, e, f]);
    }
}

/// The env-map fields Basic, Lambert and Phong share, borrowed.
pub(super) struct EnvState<'a> {
    pub(super) env_map: &'a mut Option<Arc<crate::CubeTexture>>,
    pub(super) combine: &'a mut crate::materials::Combine,
    pub(super) reflectivity: &'a mut f32,
    pub(super) refraction_ratio: &'a mut f32,
    pub(super) env_map_rotation: &'a mut crate::math::Euler,
}

macro_rules! env_state_of {
    ($($t:ty),*) => {$(
        impl<'a> EnvStateOf<'a> for $t {
            fn env_state(&'a mut self) -> EnvState<'a> {
                EnvState {
                    env_map: &mut self.env_map,
                    combine: &mut self.combine,
                    reflectivity: &mut self.reflectivity,
                    refraction_ratio: &mut self.refraction_ratio,
                    env_map_rotation: &mut self.env_map_rotation,
                }
            }
        }
    )*};
}

pub(super) trait EnvStateOf<'a> {
    fn env_state(&'a mut self) -> EnvState<'a>;
}
env_state_of!(
    crate::BasicMaterial,
    crate::LambertMaterial,
    crate::PhongMaterial
);

impl<'a> EnvState<'a> {
    pub(super) fn of<T: EnvStateOf<'a>>(m: &'a mut T) -> Self {
        m.env_state()
    }
}

/// `setUvMatrix` for both texture handles. Offset and repeat are kept in step
/// where the matrix is a plain scale and translation, so code reading them
/// sees the same transform.
pub(super) fn set_uv_matrix(t: &mut crate::Texture, [a, b, c, d, e, f]: [f32; 6]) {
    t.matrix = [a, d, 0.0, b, e, 0.0, c, f, 1.0];
    t.matrix_auto_update = false;
    if b == 0.0 && d == 0.0 {
        t.offset = crate::math::Vector2::new(c, f);
        t.repeat = crate::math::Vector2::new(a, e);
        t.rotation = 0.0;
        t.center = crate::math::Vector2::ZERO;
    }
}

/// three.js `magFilter` / `minFilter` constants as `(within a level, between
/// levels)`: `NearestFilter` 1003, `NearestMipmapNearestFilter` 1004,
/// `NearestMipmapLinearFilter` 1005, `LinearFilter` 1006,
/// `LinearMipmapNearestFilter` 1007, `LinearMipmapLinearFilter` 1008 (the
/// default for anything else). The second is `None` without mipmaps.
pub(super) fn three_filter(
    f: u32,
) -> (
    crate::textures::TextureFilter,
    Option<crate::textures::TextureFilter>,
) {
    use crate::textures::TextureFilter::{Linear, Nearest};
    match f {
        1003 => (Nearest, None),
        1004 => (Nearest, Some(Nearest)),
        1005 => (Nearest, Some(Linear)),
        1006 => (Linear, None),
        1007 => (Linear, Some(Nearest)),
        _ => (Linear, Some(Linear)),
    }
}

/// `setFilters(magFilter, minFilter, wrapS, wrapT)` for both texture handles,
/// in three.js's constants: the filters as [`three_filter`] reads them, the
/// wraps `RepeatWrapping` 1000, `ClampToEdgeWrapping` 1001 (the default),
/// `MirroredRepeatWrapping` 1002. Each is the texture's own, as three.js's are.
pub(super) fn set_texture_filters(
    t: &mut crate::Texture,
    mag: u32,
    min: u32,
    wrap_s: u32,
    wrap_t: u32,
) {
    use crate::textures::TextureWrap;
    let wrap = |w: u32| match w {
        1000 => TextureWrap::Repeat,
        1002 => TextureWrap::MirroredRepeat,
        _ => TextureWrap::ClampToEdge,
    };
    t.mag_filter = three_filter(mag).0;
    (t.min_filter, t.mipmap_filter) = three_filter(min);
    t.wrap_s = wrap(wrap_s);
    t.wrap_t = wrap(wrap_t);
}
