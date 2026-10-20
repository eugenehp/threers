//! The glTF file itself: the container, the JSON, the buffers and buffer views,
//! accessors and image sources — three.js r165 `GLTFLoader`'s `GLTFBinaryExtension`,
//! `loadBuffer`, `loadBufferView` (with `EXT_meshopt_compression`) and
//! `loadAccessor`, minus the promises.
//!
//! External resources are not fetched here: a `.gltf` names its `.bin` files
//! and images by URI, and whoever drives the load (the shim's `GLTFLoader`,
//! or a native caller) fetches them and hands the bytes in. `data:` URIs and
//! the GLB body are resolved here.

use std::collections::HashMap;
use std::sync::Arc;

use super::super::json::{self, Value};
use super::GltfError;

/// `glTF` — the first four bytes of a binary file.
const GLB_MAGIC: &[u8; 4] = b"glTF";
const CHUNK_JSON: u32 = 0x4E4F_534A;
const CHUNK_BIN: u32 = 0x004E_4942;

/// A parsed glTF: the JSON, and the buffers as far as they are known.
pub struct Document {
    pub root: Value,
    /// The JSON text (the GLB's JSON chunk, or the whole file).
    pub json_text: String,
    /// Whether the file was a GLB (three.js registers `KHR_binary_glTF` then).
    pub is_glb: bool,
    body: Option<Arc<Vec<u8>>>,
    /// One slot per `buffers[i]`: the bytes once known.
    buffers: Vec<Option<Arc<Vec<u8>>>>,
    /// Why a buffer is missing, when its fetch failed.
    buffer_errors: HashMap<usize, String>,
    /// `EXT_meshopt_compression` views, decoded once all buffers are in.
    decoded_views: HashMap<usize, Arc<Vec<u8>>>,
}

/// Where an image's bytes are.
#[derive(Debug, Clone)]
pub struct ImageSource {
    /// An external URI the caller has to fetch (relative to the resource path).
    pub uri: Option<String>,
    /// The bytes, when they are in the file (`bufferView` or a `data:` URI).
    pub bytes: Option<Vec<u8>>,
    /// `mimeType`, or what three.js's `getImageURIMimeType` guesses.
    pub mime_type: String,
}

/// An accessor as stored: raw component values (exact for every 8- and
/// 16-bit type and for floats), with what is needed to interpret them.
#[derive(Debug, Clone)]
pub struct Accessor {
    pub item_size: usize,
    pub count: usize,
    pub component_type: u32,
    pub normalized: bool,
    pub raw: Vec<f32>,
}

impl Accessor {
    /// The scale a normalized component is multiplied by — three.js's
    /// `getNormalizedComponentScale`.
    pub fn normalized_scale(component_type: u32) -> f32 {
        match component_type {
            5120 => 1.0 / 127.0,
            5121 => 1.0 / 255.0,
            5122 => 1.0 / 32767.0,
            5123 => 1.0 / 65535.0,
            5125 => 1.0 / 4_294_967_295.0,
            _ => 1.0,
        }
    }

    fn signed(component_type: u32) -> bool {
        matches!(component_type, 5120 | 5122)
    }

    /// The values a shader reads: a normalized integer divided by its range,
    /// a signed one clamped at -1 (three.js `BufferAttribute.getX`'s
    /// `denormalize`, and the GPU's).
    pub fn values(&self) -> Vec<f32> {
        if !self.normalized || self.component_type == 5126 {
            return self.raw.clone();
        }
        let s = Self::normalized_scale(self.component_type);
        let signed = Self::signed(self.component_type);
        self.raw
            .iter()
            .map(|&v| if signed { (v * s).max(-1.0) } else { v * s })
            .collect()
    }

    /// three.js `_getArrayFromAccessor`: normalized values scaled, unclamped
    /// (what an animation track's values are).
    pub fn scaled(&self) -> Vec<f32> {
        if !self.normalized || self.component_type == 5126 {
            return self.raw.clone();
        }
        let s = Self::normalized_scale(self.component_type);
        self.raw.iter().map(|&v| v * s).collect()
    }
}

/// `count` × `item_size` components from `bytes`, element `i` at
/// `offset + i * stride`.
fn read_elements(
    bytes: &[u8],
    offset: usize,
    stride: usize,
    count: usize,
    item_size: usize,
    component_type: u32,
) -> Result<Vec<f32>, GltfError> {
    let width = component_width(component_type)
        .ok_or_else(|| GltfError::msg(format!("unsupported componentType {component_type}")))?;
    if count == 0 {
        return Ok(Vec::new());
    }
    let last = offset
        .checked_add((count - 1).checked_mul(stride).ok_or(GltfError::BadAccessor)?)
        .and_then(|v| v.checked_add(item_size * width))
        .ok_or(GltfError::BadAccessor)?;
    if last > bytes.len() {
        return Err(GltfError::msg(format!(
            "accessor reads past the end of its buffer view ({last} > {} bytes)",
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(count * item_size);
    for e in 0..count {
        let base = offset + e * stride;
        for c in 0..item_size {
            let at = base + c * width;
            let b = &bytes[at..at + width];
            out.push(match component_type {
                5126 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                5125 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
                5123 => u16::from_le_bytes([b[0], b[1]]) as f32,
                5122 => i16::from_le_bytes([b[0], b[1]]) as f32,
                5121 => b[0] as f32,
                5120 => b[0] as i8 as f32,
                _ => 0.0,
            });
        }
    }
    Ok(out)
}

pub fn component_width(component_type: u32) -> Option<usize> {
    Some(match component_type {
        5120 | 5121 => 1,
        5122 | 5123 => 2,
        5125 | 5126 => 4,
        _ => return None,
    })
}

pub fn type_size(ty: &str) -> Option<usize> {
    Some(match ty {
        "SCALAR" => 1,
        "VEC2" => 2,
        "VEC3" => 3,
        "VEC4" => 4,
        "MAT2" => 4,
        "MAT3" => 9,
        "MAT4" => 16,
        _ => return None,
    })
}

// ---- small JSON helpers ----

pub fn get<'a>(v: &'a Value, key: &str) -> Option<&'a Value> {
    v.as_object()?.get(key)
}
/// A non-negative integer member (an index).
pub fn index(v: &Value, key: &str) -> Option<usize> {
    let n = get(v, key)?.as_number()?;
    (n >= 0.0 && n.fract() == 0.0 && n < 9.0e15).then_some(n as usize)
}
pub fn num(v: &Value, key: &str) -> Option<f64> {
    get(v, key)?.as_number()
}
pub fn string<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    get(v, key)?.as_str()
}
pub fn array<'a>(v: &'a Value, key: &str) -> &'a [Value] {
    get(v, key).and_then(|a| a.as_array()).map(|a| a.as_slice()).unwrap_or(&[])
}
/// An array member of numbers, as f32 (missing entries read as `fill`).
pub fn floats(v: &Value, key: &str, len: usize, fill: f32) -> Option<Vec<f32>> {
    let a = get(v, key)?.as_array()?;
    Some((0..len).map(|i| a.get(i).and_then(|x| x.as_f32()).unwrap_or(fill)).collect())
}
/// The `extensions.<name>` block of a definition.
pub fn extension<'a>(v: &'a Value, name: &str) -> Option<&'a Value> {
    get(v, "extensions")?.get(name)
}
/// The `i`-th entry of a top-level array (`"nodes"`, `"meshes"`, …).
pub fn def<'a>(root: &'a Value, list: &str, i: usize) -> Option<&'a Value> {
    get(root, list)?.as_array()?.get(i)
}
pub fn count(root: &Value, list: &str) -> usize {
    array(root, list).len()
}

/// Decode a `data:` URI: `data:[<mediatype>][;base64],<data>`.
pub fn decode_data_uri(uri: &str) -> Option<(String, Vec<u8>)> {
    let rest = uri.strip_prefix("data:")?;
    let comma = rest.find(',')?;
    let (meta, payload) = (&rest[..comma], &rest[comma + 1..]);
    let base64 = meta.ends_with(";base64");
    let mime = meta.trim_end_matches(";base64").split(';').next().unwrap_or("").to_string();
    let bytes = if base64 {
        decode_base64(payload).ok()?
    } else {
        percent_decode(payload)
    };
    Some((mime, bytes))
}

fn percent_decode(s: &str) -> Vec<u8> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            let hex = |c: u8| (c as char).to_digit(16);
            if let (Some(h), Some(l)) = (hex(b[i + 1]), hex(b[i + 2])) {
                out.push((h * 16 + l) as u8);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    out
}

pub fn decode_base64(s: &str) -> Result<Vec<u8>, GltfError> {
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let mut buf = 0u32;
    let mut bits = 0u32;
    for &b in s.as_bytes() {
        let v = match b {
            b'A'..=b'Z' => (b - b'A') as u32,
            b'a'..=b'z' => (b - b'a' + 26) as u32,
            b'0'..=b'9' => (b - b'0' + 52) as u32,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' | b'\n' | b'\r' | b' ' | b'\t' => continue,
            _ => return Err(GltfError::BadBase64),
        };
        buf = (buf << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

/// three.js `getImageURIMimeType`.
fn guess_mime(uri: &str) -> String {
    let lower = uri.to_ascii_lowercase();
    let path = lower.split('?').next().unwrap_or("");
    if path.ends_with(".jpg") || path.ends_with(".jpeg") || lower.starts_with("data:image/jpeg") {
        "image/jpeg".into()
    } else if path.ends_with(".webp") || lower.starts_with("data:image/webp") {
        "image/webp".into()
    } else {
        "image/png".into()
    }
}

/// Split a GLB into its JSON text and binary body, as `GLTFBinaryExtension`
/// does; anything not starting with `glTF` is the JSON text itself.
pub fn read_container(bytes: &[u8]) -> Result<(String, Option<Vec<u8>>, bool), GltfError> {
    if bytes.len() < 4 || &bytes[0..4] != GLB_MAGIC {
        let text = std::str::from_utf8(bytes).map_err(|_| GltfError::Json("utf8"))?;
        // A UTF-8 byte order mark is legal JSON-in-a-file but not JSON.
        return Ok((text.trim_start_matches('\u{feff}').to_string(), None, false));
    }
    if bytes.len() < 12 {
        return Err(GltfError::BadMagic);
    }
    let u32_at = |p: usize| u32::from_le_bytes([bytes[p], bytes[p + 1], bytes[p + 2], bytes[p + 3]]);
    let version = u32_at(4);
    if version < 2 {
        return Err(GltfError::UnsupportedVersion);
    }
    let total = u32_at(8) as usize;
    if total > bytes.len() || total < 12 {
        return Err(GltfError::BadMagic);
    }
    let mut pos = 12;
    let (mut json_text, mut body) = (None, None);
    while pos + 8 <= total {
        let len = u32_at(pos) as usize;
        let kind = u32_at(pos + 4);
        let start = pos + 8;
        let end = start.checked_add(len).ok_or(GltfError::BadMagic)?;
        if end > total {
            return Err(GltfError::BadMagic);
        }
        match kind {
            CHUNK_JSON => {
                let s = std::str::from_utf8(&bytes[start..end]).map_err(|_| GltfError::Json("utf8"))?;
                json_text = Some(s.to_string());
            }
            CHUNK_BIN => body = Some(bytes[start..end].to_vec()),
            _ => {}
        }
        pos = end;
    }
    let json_text = json_text.ok_or(GltfError::MissingField("JSON chunk"))?;
    Ok((json_text, body, true))
}

impl Document {
    /// Parse a `.glb` or `.gltf` file's bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Document, GltfError> {
        let (text, body, is_glb) = read_container(bytes)?;
        Self::from_parts(text, body, is_glb)
    }

    /// Parse JSON text (a `.gltf`), with an optional GLB body.
    pub fn from_parts(
        json_text: String,
        body: Option<Vec<u8>>,
        is_glb: bool,
    ) -> Result<Document, GltfError> {
        let root = json::parse(&json_text).map_err(GltfError::Json)?;
        if root.as_object().is_none() {
            return Err(GltfError::Json("root not object"));
        }
        // three.js: `json.asset === undefined || json.asset.version[0] < 2`.
        let version_ok = get(&root, "asset")
            .and_then(|a| string(a, "version"))
            .and_then(|v| v.chars().next())
            .and_then(|c| c.to_digit(10))
            .is_some_and(|d| d >= 2);
        if !version_ok {
            return Err(GltfError::UnsupportedVersion);
        }
        let n = count(&root, "buffers");
        let mut doc = Document {
            root,
            json_text,
            is_glb,
            body: body.map(Arc::new),
            buffers: vec![None; n],
            buffer_errors: HashMap::new(),
            decoded_views: HashMap::new(),
        };
        // `data:` URIs and the GLB body are known now.
        for i in 0..n {
            let Some(b) = def(&doc.root, "buffers", i) else { continue };
            match string(b, "uri") {
                Some(uri) if uri.starts_with("data:") => match decode_data_uri(uri) {
                    Some((_, bytes)) => doc.buffers[i] = Some(Arc::new(bytes)),
                    None => {
                        doc.buffer_errors.insert(i, format!("Failed to load buffer \"{uri}\"."));
                    }
                },
                None if i == 0 => doc.buffers[0] = doc.body.clone(),
                _ => {}
            }
        }
        Ok(doc)
    }

    /// The buffers whose bytes have to be fetched: `(index, uri)`. A meshopt
    /// fallback buffer is left out — nothing reads it when the extension is
    /// decoded, which it always is here.
    pub fn external_buffers(&self) -> Vec<(usize, String)> {
        let mut out = Vec::new();
        for (i, slot) in self.buffers.iter().enumerate() {
            if slot.is_some() || self.buffer_errors.contains_key(&i) {
                continue;
            }
            let Some(b) = def(&self.root, "buffers", i) else { continue };
            let Some(uri) = string(b, "uri") else { continue };
            let fallback = extension(b, "EXT_meshopt_compression")
                .and_then(|e| get(e, "fallback"))
                .and_then(|f| f.as_bool())
                .unwrap_or(false);
            if fallback && !self.buffer_referenced_plainly(i) {
                continue;
            }
            out.push((i, uri.to_string()));
        }
        out
    }

    fn buffer_referenced_plainly(&self, buffer: usize) -> bool {
        array(&self.root, "bufferViews").iter().any(|v| {
            index(v, "buffer") == Some(buffer) && extension(v, "EXT_meshopt_compression").is_none()
        })
    }

    pub fn set_buffer(&mut self, i: usize, bytes: Vec<u8>) {
        if i < self.buffers.len() {
            self.buffers[i] = Some(Arc::new(bytes));
            self.buffer_errors.remove(&i);
        }
    }

    pub fn set_buffer_error(&mut self, i: usize, message: String) {
        self.buffer_errors.insert(i, message);
    }

    pub fn buffer(&self, i: usize) -> Result<&[u8], GltfError> {
        if let Some(Some(b)) = self.buffers.get(i) {
            return Ok(b.as_slice());
        }
        if let Some(e) = self.buffer_errors.get(&i) {
            return Err(GltfError::msg(e.clone()));
        }
        let Some(def) = def(&self.root, "buffers", i) else {
            return Err(GltfError::msg(format!("buffer {i} does not exist")));
        };
        if let Some(t) = string(def, "type") {
            if t != "arraybuffer" {
                return Err(GltfError::msg(format!("{t} buffer type is not supported.")));
            }
        }
        Err(GltfError::MissingField("external buffer not supplied"))
    }

    /// Decode every `EXT_meshopt_compression` buffer view (call once the
    /// buffers are in). A view that will not decode is an error only when it
    /// is read.
    pub fn decode_meshopt_views(&mut self) -> Result<(), GltfError> {
        use super::super::meshopt;
        let n = count(&self.root, "bufferViews");
        for i in 0..n {
            if self.decoded_views.contains_key(&i) {
                continue;
            }
            let Some(view) = def(&self.root, "bufferViews", i) else { continue };
            let Some(ext) = extension(view, "EXT_meshopt_compression") else { continue };
            let buffer = index(ext, "buffer").ok_or(GltfError::BadAccessor)?;
            let offset = index(ext, "byteOffset").unwrap_or(0);
            let length = index(ext, "byteLength").unwrap_or(0);
            let stride = index(ext, "byteStride").ok_or(GltfError::BadAccessor)?;
            let count = index(ext, "count").ok_or(GltfError::BadAccessor)?;
            let mode = string(ext, "mode")
                .and_then(meshopt::Mode::parse)
                .ok_or(GltfError::BadAccessor)?;
            let filter = string(ext, "filter")
                .map(meshopt::Filter::parse)
                .unwrap_or(meshopt::Filter::None);
            let source = self
                .buffer(buffer)?
                .get(offset..offset.saturating_add(length))
                .ok_or(GltfError::BadAccessor)?;
            let decoded = meshopt::decode(count, stride, source, mode, filter)
                .map_err(|e| GltfError::msg(format!("EXT_meshopt_compression: {e:?}")))?;
            self.decoded_views.insert(i, Arc::new(decoded));
        }
        Ok(())
    }

    /// `loadBufferView`: the view's bytes (decoded, for a meshopt view).
    pub fn buffer_view(&self, i: usize) -> Result<&[u8], GltfError> {
        if let Some(d) = self.decoded_views.get(&i) {
            return Ok(d.as_slice());
        }
        let view = def(&self.root, "bufferViews", i)
            .ok_or_else(|| GltfError::msg(format!("bufferView {i} does not exist")))?;
        let buffer = self.buffer(index(view, "buffer").ok_or(GltfError::BadAccessor)?)?;
        let offset = index(view, "byteOffset").unwrap_or(0);
        let length = index(view, "byteLength").unwrap_or(0);
        buffer
            .get(offset..offset.checked_add(length).ok_or(GltfError::BadAccessor)?)
            .ok_or_else(|| GltfError::msg(format!("bufferView {i} lies outside its buffer")))
    }

    /// `loadAccessor`.
    pub fn accessor(&self, i: usize) -> Result<Accessor, GltfError> {
        let a = def(&self.root, "accessors", i)
            .ok_or_else(|| GltfError::msg(format!("accessor {i} does not exist")))?;
        let item_size = string(a, "type").and_then(type_size).ok_or(GltfError::BadAccessor)?;
        let component_type = index(a, "componentType").ok_or(GltfError::BadAccessor)? as u32;
        let width = component_width(component_type).ok_or(GltfError::BadAccessor)?;
        let count = index(a, "count").ok_or(GltfError::BadAccessor)?;
        let normalized = get(a, "normalized").and_then(|v| v.as_bool()) == Some(true);
        let total = count.checked_mul(item_size).ok_or(GltfError::BadAccessor)?;
        let mut raw = match index(a, "bufferView") {
            Some(view_index) => {
                let bytes = self.buffer_view(view_index)?;
                let item_bytes = width * item_size;
                let stride = def(&self.root, "bufferViews", view_index)
                    .and_then(|v| index(v, "byteStride"))
                    .filter(|&s| s > 0)
                    .unwrap_or(item_bytes);
                read_elements(
                    bytes,
                    index(a, "byteOffset").unwrap_or(0),
                    stride,
                    count,
                    item_size,
                    component_type,
                )?
            }
            None => {
                // No view: zeros, which a sparse block may then overwrite.
                if total > 1 << 28 {
                    return Err(GltfError::BadAccessor);
                }
                vec![0.0; total]
            }
        };
        if let Some(sparse) = get(a, "sparse") {
            let n = index(sparse, "count").ok_or(GltfError::BadAccessor)?;
            let ind = get(sparse, "indices").ok_or(GltfError::BadAccessor)?;
            let val = get(sparse, "values").ok_or(GltfError::BadAccessor)?;
            let ict = index(ind, "componentType").ok_or(GltfError::BadAccessor)? as u32;
            let iw = component_width(ict).ok_or(GltfError::BadAccessor)?;
            let indices = read_elements(
                self.buffer_view(index(ind, "bufferView").ok_or(GltfError::BadAccessor)?)?,
                index(ind, "byteOffset").unwrap_or(0),
                iw,
                n,
                1,
                ict,
            )?;
            let values = read_elements(
                self.buffer_view(index(val, "bufferView").ok_or(GltfError::BadAccessor)?)?,
                index(val, "byteOffset").unwrap_or(0),
                width * item_size,
                n,
                item_size,
                component_type,
            )?;
            for (k, &target) in indices.iter().enumerate() {
                let t = target as usize;
                if t < count {
                    raw[t * item_size..(t + 1) * item_size]
                        .copy_from_slice(&values[k * item_size..(k + 1) * item_size]);
                }
            }
        }
        Ok(Accessor { item_size, count, component_type, normalized, raw })
    }

    /// An index accessor as integers (exact past 2^24, which floats are not).
    pub fn accessor_u32(&self, i: usize) -> Result<Vec<u32>, GltfError> {
        let a = def(&self.root, "accessors", i)
            .ok_or_else(|| GltfError::msg(format!("accessor {i} does not exist")))?;
        let component_type = index(a, "componentType").ok_or(GltfError::BadAccessor)? as u32;
        if component_type != 5125 || get(a, "sparse").is_some() {
            return Ok(self.accessor(i)?.raw.iter().map(|&v| v as u32).collect());
        }
        let count = index(a, "count").ok_or(GltfError::BadAccessor)?;
        let Some(view_index) = index(a, "bufferView") else {
            return Ok(vec![0; count]);
        };
        let bytes = self.buffer_view(view_index)?;
        let stride = def(&self.root, "bufferViews", view_index)
            .and_then(|v| index(v, "byteStride"))
            .filter(|&s| s > 0)
            .unwrap_or(4);
        let offset = index(a, "byteOffset").unwrap_or(0);
        let mut out = Vec::with_capacity(count);
        for e in 0..count {
            let at = offset + e * stride;
            let b = bytes.get(at..at + 4).ok_or_else(|| {
                GltfError::msg(format!("accessor {i} reads past the end of its buffer view"))
            })?;
            out.push(u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        }
        Ok(out)
    }

    /// Where `images[i]` is.
    pub fn image_source(&self, i: usize) -> Result<ImageSource, GltfError> {
        let def = def(&self.root, "images", i)
            .ok_or_else(|| GltfError::msg(format!("image {i} does not exist")))?;
        let declared = string(def, "mimeType").map(str::to_string);
        if let Some(view) = index(def, "bufferView") {
            let bytes = self.buffer_view(view)?.to_vec();
            return Ok(ImageSource {
                uri: None,
                bytes: Some(bytes),
                mime_type: declared.unwrap_or_else(|| "image/png".into()),
            });
        }
        let Some(uri) = string(def, "uri") else {
            return Err(GltfError::msg(format!("Image {i} is missing URI and bufferView")));
        };
        let mime_type = declared.unwrap_or_else(|| guess_mime(uri));
        if uri.starts_with("data:") {
            let (_, bytes) = decode_data_uri(uri)
                .ok_or_else(|| GltfError::msg(format!("Couldn't load texture {uri}")))?;
            return Ok(ImageSource { uri: None, bytes: Some(bytes), mime_type });
        }
        Ok(ImageSource { uri: Some(uri.to_string()), bytes: None, mime_type })
    }
}
