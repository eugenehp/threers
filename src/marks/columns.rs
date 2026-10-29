//! Per-instance data, one array per property, uploaded as instanced vertex
//! buffers.
//!
//! A column is what a table column is: one value per row. Marks draw a row per
//! instance and read each property from its own buffer, so changing one
//! property re-uploads that property and nothing else, and changing a few rows
//! of it uploads just those rows ([`Column::update`]).
//!
//! Vertex buffers rather than storage buffers, because WebGL2 has none of the
//! latter and every mark has to draw there too.

use std::ops::Range;

/// What a column holds and how the GPU reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColumnFormat {
    /// `1..=4` floats a row, read as `f32` / `vecN<f32>`.
    Float(u32),
    /// Four bytes a row, read as a `vec4<f32>` in `0..=1` — colours.
    Unorm8x4,
    /// Six floats a row, read as two `vec3<f32>` attributes. Two values that
    /// travel together share a buffer this way, and WebGPU allows a pipeline
    /// only eight.
    Float3x2,
    /// Eight floats a row, read as two `vec4<f32>` attributes.
    Float4x2,
    /// Sixteen floats a row, read as four `vec4<f32>` attributes.
    Float4x4,
    /// An absolute `f64` position stored as `(hi, lo)` `f32` triples — see
    /// [`split_f64`](super::split_f64). Two shader attributes.
    PositionF64,
    /// A position relative to the layer's
    /// [`local_origin`](super::MarkCommon::local_origin), one `f32` triple:
    /// half the bytes of [`PositionF64`](Self::PositionF64), for data that
    /// arrives local — a tile's features relative to its corner. Takes the
    /// same two shader locations, so the columns after it do not move.
    PositionLocal,
}

impl ColumnFormat {
    /// Bytes a row.
    pub fn stride(self) -> u64 {
        match self {
            ColumnFormat::Float(n) => 4 * n.clamp(1, 4) as u64,
            ColumnFormat::Unorm8x4 => 4,
            ColumnFormat::PositionF64 | ColumnFormat::Float3x2 => 24,
            ColumnFormat::PositionLocal => 12,
            ColumnFormat::Float4x2 => 32,
            ColumnFormat::Float4x4 => 64,
        }
    }

    fn attributes(self, first_location: u32) -> Vec<wgpu::VertexAttribute> {
        match self {
            ColumnFormat::Float(n) => vec![wgpu::VertexAttribute {
                format: match n.clamp(1, 4) {
                    1 => wgpu::VertexFormat::Float32,
                    2 => wgpu::VertexFormat::Float32x2,
                    3 => wgpu::VertexFormat::Float32x3,
                    _ => wgpu::VertexFormat::Float32x4,
                },
                offset: 0,
                shader_location: first_location,
            }],
            ColumnFormat::PositionLocal => vec![wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: first_location,
            }],
            ColumnFormat::Float4x2 => vec![
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: first_location,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 16,
                    shader_location: first_location + 1,
                },
            ],
            ColumnFormat::Float4x4 => (0..4)
                .map(|k| wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 16 * k as u64,
                    shader_location: first_location + k,
                })
                .collect(),
            ColumnFormat::Unorm8x4 => vec![wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Unorm8x4,
                offset: 0,
                shader_location: first_location,
            }],
            ColumnFormat::PositionF64 | ColumnFormat::Float3x2 => vec![
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: first_location,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 12,
                    shader_location: first_location + 1,
                },
            ],
        }
    }

    /// Shader locations the column takes.
    pub(crate) fn locations(self) -> u32 {
        match self {
            ColumnFormat::PositionF64
            | ColumnFormat::Float3x2
            | ColumnFormat::Float4x2
            | ColumnFormat::PositionLocal => 2,
            _ => 1,
        }
    }
}

/// One property for every instance.
///
/// Rows past the end of the data take the column's default, so a layer can
/// leave a column empty to give every instance the same value — the same rule
/// as [`InstancedMesh::colors`](crate::InstancedMesh::colors).
#[derive(Debug)]
pub struct Column {
    format: ColumnFormat,
    bytes: Vec<u8>,
    default: Vec<u8>,
    /// Bytes changed since the last upload. `None` with a buffer present means
    /// the GPU copy is current.
    dirty: Option<Range<usize>>,
    /// Rows the GPU buffer was filled for — padding included.
    uploaded_rows: usize,
    pub(crate) buffer: Option<wgpu::Buffer>,
}

impl Column {
    pub fn new(format: ColumnFormat) -> Self {
        let default = match format {
            ColumnFormat::Unorm8x4 => vec![255; 4],
            _ => vec![0; format.stride() as usize],
        };
        Self {
            format,
            bytes: Vec::new(),
            default,
            dirty: None,
            uploaded_rows: 0,
            buffer: None,
        }
    }

    /// `1..=4` floats a row.
    pub fn float(components: u32) -> Self {
        Self::new(ColumnFormat::Float(components))
    }

    pub fn format(&self) -> ColumnFormat {
        self.format
    }

    /// Rows of real data (not counting default padding).
    pub fn len(&self) -> usize {
        self.bytes.len() / self.format.stride() as usize
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// The value rows past the end of the data take.
    pub fn set_default_f32(&mut self, value: &[f32]) {
        self.default = pack_f32(self.format, value);
        self.mark_all();
    }

    /// [`set_default_f32`](Self::set_default_f32) for an RGBA8 column.
    pub fn set_default_rgba(&mut self, rgba: [u8; 4]) {
        assert_eq!(self.format, ColumnFormat::Unorm8x4, "not a colour column");
        self.default = rgba.to_vec();
        self.mark_all();
    }

    /// Replace every row. `values` is row-major, `components` floats a row.
    pub fn set_f32(&mut self, values: &[f32]) {
        self.bytes = pack_f32(self.format, values);
        self.mark_all();
    }

    /// Replace every row of a colour column.
    pub fn set_rgba(&mut self, values: &[[u8; 4]]) {
        assert_eq!(self.format, ColumnFormat::Unorm8x4, "not a colour column");
        self.bytes = values.iter().flatten().copied().collect();
        self.mark_all();
    }

    /// Replace every row of a position column with absolute positions,
    /// making it [`PositionF64`](ColumnFormat::PositionF64) if it was local.
    pub fn set_positions(&mut self, values: &[[f64; 3]]) {
        if self.format == ColumnFormat::PositionLocal {
            self.format = ColumnFormat::PositionF64;
            self.default = vec![0; 24];
            self.buffer = None;
        }
        assert_eq!(
            self.format,
            ColumnFormat::PositionF64,
            "not a position column"
        );
        self.bytes = pack_positions(values);
        self.mark_all();
    }

    /// Overwrite rows `start..start + n` with `values` (floats, row-major),
    /// growing the column if they run past its end. Only those rows are
    /// uploaded next frame.
    pub fn update_f32(&mut self, start: usize, values: &[f32]) {
        let packed = pack_f32(self.format, values);
        self.splice(start, &packed);
    }

    /// [`update_f32`](Self::update_f32) for a colour column.
    pub fn update_rgba(&mut self, start: usize, values: &[[u8; 4]]) {
        assert_eq!(self.format, ColumnFormat::Unorm8x4, "not a colour column");
        let packed: Vec<u8> = values.iter().flatten().copied().collect();
        self.splice(start, &packed);
    }

    /// Replace every row with positions relative to the layer's local origin,
    /// making the column [`PositionLocal`](ColumnFormat::PositionLocal).
    pub fn set_local_positions(&mut self, values: &[[f32; 3]]) {
        assert!(
            matches!(
                self.format,
                ColumnFormat::PositionF64 | ColumnFormat::PositionLocal
            ),
            "not a position column"
        );
        if self.format != ColumnFormat::PositionLocal {
            self.format = ColumnFormat::PositionLocal;
            self.default = vec![0; 12];
            // The layout changed: the old buffer's rows are the wrong size.
            self.buffer = None;
        }
        self.bytes = bytemuck::cast_slice(values).to_vec();
        self.mark_all();
    }

    /// [`set_local_positions`](Self::set_local_positions) for some rows.
    pub fn update_local_positions(&mut self, start: usize, values: &[[f32; 3]]) {
        assert_eq!(
            self.format,
            ColumnFormat::PositionLocal,
            "not a local position column"
        );
        self.splice(start, bytemuck::cast_slice(values));
    }

    /// [`update_f32`](Self::update_f32) for a position column.
    pub fn update_positions(&mut self, start: usize, values: &[[f64; 3]]) {
        assert_eq!(
            self.format,
            ColumnFormat::PositionF64,
            "not a position column"
        );
        self.splice(start, &pack_positions(values));
    }

    /// A position column's rows as absolute `f64`s, `origin` added to local
    /// ones; `None` for any other column.
    pub fn positions_f64(&self, origin: Option<[f64; 3]>) -> Option<Vec<[f64; 3]>> {
        let f: Vec<f32> = bytemuck::pod_collect_to_vec(&self.bytes);
        match self.format {
            ColumnFormat::PositionF64 => Some(
                f.chunks(6)
                    .map(|r| {
                        [
                            r[0] as f64 + r[3] as f64,
                            r[1] as f64 + r[4] as f64,
                            r[2] as f64 + r[5] as f64,
                        ]
                    })
                    .collect(),
            ),
            ColumnFormat::PositionLocal => {
                let o = origin.unwrap_or_default();
                Some(
                    f.chunks(3)
                        .map(|r| [o[0] + r[0] as f64, o[1] + r[1] as f64, o[2] + r[2] as f64])
                        .collect(),
                )
            }
            _ => None,
        }
    }

    /// Row `i` of a float column, the default past the data.
    pub fn row_f32(&self, i: usize) -> Vec<f32> {
        let stride = self.format.stride() as usize;
        let bytes = self
            .bytes
            .get(i * stride..(i + 1) * stride)
            .unwrap_or(&self.default);
        bytemuck::pod_collect_to_vec(bytes)
    }

    /// Drop the GPU copy, keeping the data: the next draw uploads it again.
    pub fn release(&mut self) {
        self.buffer = None;
        self.uploaded_rows = 0;
        self.mark_all();
    }

    /// Bytes the GPU copy holds (its capacity), 0 when released.
    pub fn gpu_bytes(&self) -> u64 {
        self.buffer.as_ref().map_or(0, |b| b.size())
    }

    /// Raw bytes, for tests and for formats a typed setter does not cover.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The byte range the next upload will write, if any. For tests.
    pub fn pending_upload(&self) -> Option<Range<usize>> {
        self.dirty.clone()
    }

    fn splice(&mut self, start: usize, packed: &[u8]) {
        let stride = self.format.stride() as usize;
        let begin = start * stride;
        let end = begin + packed.len();
        if self.bytes.len() < end {
            // Rows between the old end and `start` take the default, exactly
            // as they would have been drawn before the write.
            while self.bytes.len() < begin {
                self.bytes.extend_from_slice(&self.default);
            }
            self.bytes.resize(end, 0);
        }
        self.bytes[begin..end].copy_from_slice(packed);
        self.dirty = Some(match self.dirty.take() {
            Some(r) => r.start.min(begin)..r.end.max(end),
            None => begin..end,
        });
    }

    fn mark_all(&mut self) {
        // A full rewrite: drop the buffer's claim to be current.
        self.dirty = Some(0..usize::MAX);
    }

    /// Bring the GPU buffer up to date for `rows` instances.
    pub(crate) fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, rows: usize) {
        let stride = self.format.stride() as usize;
        let rows = rows.max(1);
        let need = rows * stride;
        let fits = self
            .buffer
            .as_ref()
            .is_some_and(|b| b.size() as usize >= need);
        let full = !fits
            || self.uploaded_rows != rows
            || matches!(&self.dirty, Some(r) if r.end == usize::MAX);
        if full {
            let mut data = Vec::with_capacity(need);
            data.extend_from_slice(&self.bytes[..self.bytes.len().min(need)]);
            while data.len() < need {
                data.extend_from_slice(&self.default);
            }
            if !fits {
                // Grow by half again so a column appended to row by row is not
                // reallocated every frame.
                let cap = (need + need / 2).max(stride * 4);
                let cap = cap.div_ceil(4) * 4;
                self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("threers mark column"),
                    size: cap as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            queue.write_buffer(self.buffer.as_ref().unwrap(), 0, &data);
            self.uploaded_rows = rows;
        } else if let Some(r) = self.dirty.clone() {
            let end = r.end.min(need).min(self.bytes.len());
            if r.start < end {
                // `write_buffer` wants a multiple of four bytes; every format's
                // stride is one, so whole rows always are.
                queue.write_buffer(
                    self.buffer.as_ref().unwrap(),
                    r.start as u64,
                    &self.bytes[r.start..end],
                );
            }
        }
        self.dirty = None;
    }

    /// Pretend the GPU copy is current, so a test sees only later changes.
    #[cfg(test)]
    pub(crate) fn mark_uploaded(&mut self) {
        self.dirty = None;
    }

    pub(crate) fn layout(&self, first_location: u32) -> (u64, Vec<wgpu::VertexAttribute>) {
        (self.format.stride(), self.format.attributes(first_location))
    }
}

fn pack_f32(format: ColumnFormat, values: &[f32]) -> Vec<u8> {
    match format {
        ColumnFormat::Float(n) => {
            let n = n.clamp(1, 4) as usize;
            assert!(
                values.len().is_multiple_of(n),
                "{} floats is not a whole number of {n}-float rows",
                values.len()
            );
            bytemuck::cast_slice(values).to_vec()
        }
        ColumnFormat::PositionLocal => {
            assert!(
                values.len().is_multiple_of(3),
                "{} floats is not a whole number of positions",
                values.len()
            );
            bytemuck::cast_slice(values).to_vec()
        }
        ColumnFormat::Float4x2 => {
            assert!(
                values.len().is_multiple_of(8),
                "{} floats is not a whole number of 8-float rows",
                values.len()
            );
            bytemuck::cast_slice(values).to_vec()
        }
        ColumnFormat::Float4x4 => {
            assert!(
                values.len().is_multiple_of(16),
                "{} floats is not a whole number of 16-float rows",
                values.len()
            );
            bytemuck::cast_slice(values).to_vec()
        }
        ColumnFormat::Float3x2 => {
            assert!(
                values.len().is_multiple_of(6),
                "{} floats is not a whole number of 6-float rows",
                values.len()
            );
            bytemuck::cast_slice(values).to_vec()
        }
        ColumnFormat::Unorm8x4 => values
            .iter()
            .map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
            .collect(),
        ColumnFormat::PositionF64 => {
            let rows: Vec<[f64; 3]> = values
                .chunks(3)
                .map(|c| [c[0] as f64, c[1] as f64, c[2] as f64])
                .collect();
            pack_positions(&rows)
        }
    }
}

fn pack_positions(values: &[[f64; 3]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 24);
    for p in values {
        let (hi, lo) = super::split_f64x3(*p);
        for v in hi.iter().chain(lo.iter()) {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// Vertex buffer layouts for `columns`, instanced, from shader location
/// `first_location` on.
pub(crate) fn instance_layouts(
    columns: &[&Column],
    first_location: u32,
) -> Vec<(u64, Vec<wgpu::VertexAttribute>)> {
    let mut loc = first_location;
    columns
        .iter()
        .map(|c| {
            let l = c.layout(loc);
            loc += c.format.locations();
            l
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bytes are not `f32`-aligned in a `Vec<u8>`, so copy rather than cast.
    fn floats(bytes: &[u8]) -> Vec<f32> {
        bytemuck::pod_collect_to_vec(bytes)
    }

    #[test]
    fn partial_update_marks_only_those_rows() {
        let mut c = Column::float(1);
        c.set_f32(&[1.0, 2.0, 3.0, 4.0]);
        // Pretend it went up.
        c.dirty = None;
        c.update_f32(2, &[9.0]);
        assert_eq!(c.pending_upload(), Some(8..12));
        assert_eq!(floats(c.bytes()), [1.0, 2.0, 9.0, 4.0]);
    }

    #[test]
    fn writing_past_the_end_fills_the_gap_with_the_default() {
        let mut c = Column::float(1);
        c.set_default_f32(&[7.0]);
        c.set_f32(&[1.0]);
        c.update_f32(3, &[5.0]);
        assert_eq!(floats(c.bytes()), [1.0, 7.0, 7.0, 5.0]);
    }

    #[test]
    fn positions_pack_as_hi_then_lo() {
        let mut c = Column::new(ColumnFormat::PositionF64);
        c.set_positions(&[[10_000_000.25, 0.0, -1.5]]);
        let f = floats(c.bytes());
        assert_eq!(f.len(), 6);
        assert_eq!(f[0] as f64 + f[3] as f64, 10_000_000.25);
        assert_eq!(f[2], -1.5);
        assert_eq!(f[5], 0.0);
    }

    #[test]
    fn layouts_number_locations_across_two_attribute_columns() {
        let a = Column::new(ColumnFormat::PositionF64);
        let b = Column::float(1);
        let l = instance_layouts(&[&a, &b], 3);
        assert_eq!(l[0].1[0].shader_location, 3);
        assert_eq!(l[0].1[1].shader_location, 4);
        assert_eq!(l[1].1[0].shader_location, 5);
    }
}
