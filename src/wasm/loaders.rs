//! Bindings split out of `wasm.rs` by area (see the `mod` list there): loaders.

use super::*;

// ---- loaders ----
#[wasm_bindgen]
pub struct WebObjLoader;
#[wasm_bindgen]
impl WebObjLoader {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebObjLoader {
        WebObjLoader
    }
    pub fn parse(&self, src: &str) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::ObjLoader::parse(src)),
        }
    }
}

#[wasm_bindgen]
pub struct WebStlLoader;
#[wasm_bindgen]
impl WebStlLoader {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebStlLoader {
        WebStlLoader
    }
    pub fn parse(&self, bytes: Vec<u8>) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::StlLoader::parse(&bytes)),
        }
    }
}

#[wasm_bindgen]
pub struct WebPlyLoader;
#[wasm_bindgen]
impl WebPlyLoader {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebPlyLoader {
        WebPlyLoader
    }
    pub fn parse(&self, bytes: Vec<u8>) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::PlyLoader::parse(&bytes)),
        }
    }
}

#[wasm_bindgen]
pub struct WebHdrLoader;
#[wasm_bindgen]
impl WebHdrLoader {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebHdrLoader {
        WebHdrLoader
    }
    pub fn parse(&self, bytes: Vec<u8>) -> Result<WebTexture, JsValue> {
        crate::HdrLoader::parse(&bytes)
            .map(|t| WebTexture { inner: Arc::new(t) })
            .map_err(|e| JsValue::from_str(&format!("{e:?}")))
    }
}

#[wasm_bindgen]
pub struct WebFbxLoader;
#[wasm_bindgen]
impl WebFbxLoader {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebFbxLoader {
        WebFbxLoader
    }
    pub fn parse(&self, bytes: Vec<u8>) -> Result<WebGeometry, JsValue> {
        crate::FbxLoader::parse(&bytes)
            .map(|g| WebGeometry { inner: Arc::new(g) })
            .map_err(|e| JsValue::from_str(&format!("{e:?}")))
    }
}

/// KTX2 textures, which is how a browser gets a GPU-compressed one.
///
/// The container is read here; a Basis Universal payload still needs
/// transcoding and is reported as such rather than returned as noise.
#[wasm_bindgen]
pub struct WebKtx2Loader;
#[wasm_bindgen]
impl WebKtx2Loader {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebKtx2Loader {
        WebKtx2Loader
    }

    /// The base mip level as raw bytes, with the format and size reported
    /// separately — a compressed payload is not pixels and must not be handed
    /// back as if it were.
    pub fn parse(&self, bytes: Vec<u8>) -> Result<WebKtx2Texture, JsValue> {
        let ktx =
            crate::loaders::ktx2::parse(&bytes).map_err(|e| JsValue::from_str(&format!("{e}")))?;
        let format = format!("{:?}", ktx.texture_format().ok());
        let levels = ktx
            .level_data()
            .map_err(|e| JsValue::from_str(&format!("{e}")))?;
        Ok(WebKtx2Texture {
            width: ktx.width,
            height: ktx.height,
            format,
            block_compressed: ktx
                .texture_format()
                .map(|f| f.is_block_compressed())
                .unwrap_or(false),
            levels,
        })
    }
}

#[wasm_bindgen]
pub struct WebKtx2Texture {
    #[wasm_bindgen(readonly)]
    pub width: u32,
    #[wasm_bindgen(readonly)]
    pub height: u32,
    /// Whether the bytes are blocks rather than pixels, which decides whether a
    /// caller may index them.
    #[wasm_bindgen(readonly)]
    pub block_compressed: bool,
    pub(super) format: String,
    pub(super) levels: Vec<Vec<u8>>,
}

#[wasm_bindgen]
impl WebKtx2Texture {
    #[wasm_bindgen(getter)]
    pub fn format(&self) -> String {
        self.format.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    /// One mip level's bytes, largest first.
    pub fn level(&self, index: usize) -> Option<Vec<u8>> {
        self.levels.get(index).cloned()
    }
}

/// Alembic geometry caches.
#[wasm_bindgen]
pub struct WebAlembicLoader;
#[wasm_bindgen]
impl WebAlembicLoader {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebAlembicLoader {
        WebAlembicLoader
    }

    /// The first `PolyMesh` in the archive, as a geometry.
    pub fn parse(&self, bytes: Vec<u8>) -> Result<WebGeometry, JsValue> {
        let archive = crate::loaders::alembic::Archive::open(&bytes)
            .map_err(|e| JsValue::from_str(&format!("{e}")))?;
        let mesh = archive
            .poly_meshes()
            .into_iter()
            .next()
            .ok_or_else(|| JsValue::from_str("no PolyMesh in this archive"))?;
        Ok(WebGeometry {
            inner: Arc::new(poly_mesh_geometry(&mesh)),
        })
    }

    /// How many meshes the archive holds, for a caller that wants to choose.
    pub fn mesh_names(&self, bytes: Vec<u8>) -> Result<Vec<JsValue>, JsValue> {
        let archive = crate::loaders::alembic::Archive::open(&bytes)
            .map_err(|e| JsValue::from_str(&format!("{e}")))?;
        Ok(archive
            .poly_meshes()
            .iter()
            .map(|m| JsValue::from_str(&m.name))
            .collect())
    }
}

/// An Alembic `PolyMesh` as a triangulated `BufferGeometry`.
///
/// Faces are fanned and remembered as polygons; see [`PolyMesh::geometry`].
///
/// [`PolyMesh::geometry`]: crate::loaders::PolyMesh::geometry
pub(super) fn poly_mesh_geometry(mesh: &crate::loaders::PolyMesh) -> crate::core::BufferGeometry {
    mesh.geometry()
}

/// USD in the browser.
///
/// Gated on the `usd` feature like the rest of it, so a wasm build that does
/// not want ten thousand lines of scene description does not pay for them.
/// `.usda`, `.usdc` and `.usdz` all arrive as bytes here, and the loader tells
/// them apart by what is in them rather than by a file name — which a browser
/// often does not have.
#[cfg(feature = "usd")]
#[wasm_bindgen]
pub struct WebUsdLoader;

#[cfg(feature = "usd")]
#[wasm_bindgen]
impl WebUsdLoader {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebUsdLoader {
        WebUsdLoader
    }

    /// The first mesh in the layer, as a geometry.
    pub fn parse(&self, bytes: Vec<u8>) -> Result<WebGeometry, JsValue> {
        let scene = crate::loaders::usd::UsdLoader::parse(&bytes)
            .map_err(|e| JsValue::from_str(&format!("{e}")))?;
        let geometry = scene
            .arena
            .nodes
            .iter()
            .find_map(|(_, o)| match &o.kind {
                crate::core::ObjectKind::Mesh(m) => Some(m.geometry.clone()),
                _ => None,
            })
            .ok_or_else(|| JsValue::from_str("no mesh in this USD layer"))?;
        Ok(WebGeometry { inner: geometry })
    }

    /// Which of the three forms the bytes are, for a caller that wants to know
    /// before committing to them.
    pub fn sniff(&self, bytes: Vec<u8>) -> Option<String> {
        crate::loaders::usd::sniff(&bytes).map(|f| format!("{f:?}"))
    }
}

/// Writing glTF from the browser, which is how a scene edited there gets out.
#[wasm_bindgen]
pub struct WebGltfExporter;
#[wasm_bindgen]
impl WebGltfExporter {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebGltfExporter {
        WebGltfExporter
    }

    /// One geometry as a self-contained `.glb`.
    pub fn geometry_to_glb(&self, geometry: &WebGeometry) -> Vec<u8> {
        crate::loaders::gltf_geometry_to_glb(&geometry.inner)
    }
}

#[wasm_bindgen]
pub struct WebColladaLoader;
#[wasm_bindgen]
impl WebColladaLoader {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebColladaLoader {
        WebColladaLoader
    }
    pub fn parse(&self, src: &str) -> Result<WebGeometry, JsValue> {
        crate::ColladaLoader::parse(src)
            .map(|g| WebGeometry { inner: Arc::new(g) })
            .map_err(|e| JsValue::from_str(&format!("{e:?}")))
    }
}

#[wasm_bindgen]
pub struct WebExrLoader;
#[wasm_bindgen]
impl WebExrLoader {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebExrLoader {
        WebExrLoader
    }
    pub fn parse(&self, bytes: Vec<u8>) -> Result<WebTexture, JsValue> {
        crate::ExrLoader::parse(&bytes)
            .map(|t| WebTexture { inner: Arc::new(t) })
            .map_err(|e| JsValue::from_str(&format!("{e:?}")))
    }

    /// Decode to half-float instead of clipping to 8 bits.
    ///
    /// EXR is scene-referred, and a star map is the extreme case: over half of
    /// NASA's Deep Star Maps sits below a hundredth of full scale, so the 8-bit
    /// path leaves the galaxy as a handful of grey dots.
    #[wasm_bindgen(js_name = parseHdr)]
    pub fn parse_hdr(&self, bytes: Vec<u8>) -> Result<WebTexture, JsValue> {
        crate::ExrLoader::parse_hdr(&bytes)
            .map(|t| WebTexture { inner: Arc::new(t) })
            .map_err(|e| JsValue::from_str(&format!("{e:?}")))
    }
}
