//! `EXT_meshopt_compression` — meshoptimizer's vertex and index codecs.
//!
//! A large share of the `.glb` on the web is compressed, and a loader that
//! cannot decode it fails on real files while passing every test built from
//! files it wrote itself. This is a port of meshoptimizer's *reference*
//! decoder (`meshopt_decoder_reference.js`, MIT, Arseny Kapoulkine / Jasper
//! St. Pierre) rather than a reading of the specification, because a codec
//! recalled rather than transcribed decodes to plausible numbers instead of to
//! an error — and the numbers here are vertex positions, which look like a
//! modelling mistake when they are wrong.
//!
//! Three streams, named by the `mode` on the buffer view:
//!
//! | mode | what it holds |
//! |---|---|
//! | `ATTRIBUTES` | interleaved vertices, byte-plane delta coded |
//! | `TRIANGLES` | a triangle list, coded against two FIFOs of recent edges and vertices |
//! | `INDICES` | a plain index sequence, LEB128 deltas |
//!
//! `ATTRIBUTES` may carry a *filter* as well, which is a second decode step
//! that turns the stored bits into octahedral normals, quaternions, shared-
//! exponent floats or YCoCg colour.

/// How a stream is coded.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    Attributes,
    Triangles,
    Indices,
}

/// The second decode step an attribute stream may ask for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Filter {
    None,
    Octahedral,
    Quaternion,
    Exponential,
    Color,
}

#[derive(Debug, PartialEq)]
pub enum MeshoptError {
    /// The leading byte is not one this decoder knows.
    BadHeader,
    /// A stride or count the codec does not allow — an octahedral filter on a
    /// 12-byte stride, an index buffer that is not whole triangles.
    BadParameters,
    /// The stream ended, or claimed a length that does not fit it.
    Truncated,
}

impl Mode {
    pub fn parse(name: &str) -> Option<Mode> {
        match name {
            "ATTRIBUTES" => Some(Mode::Attributes),
            "TRIANGLES" => Some(Mode::Triangles),
            "INDICES" => Some(Mode::Indices),
            _ => None,
        }
    }
}

impl Filter {
    pub fn parse(name: &str) -> Filter {
        match name {
            "OCTAHEDRAL" => Filter::Octahedral,
            "QUATERNION" => Filter::Quaternion,
            "EXPONENTIAL" => Filter::Exponential,
            "COLOR" => Filter::Color,
            _ => Filter::None,
        }
    }
}

/// Decode one buffer view, as glTF asks for it.
pub fn decode(
    count: usize,
    stride: usize,
    source: &[u8],
    mode: Mode,
    filter: Filter,
) -> Result<Vec<u8>, MeshoptError> {
    let mut target = vec![0u8; count.checked_mul(stride).ok_or(MeshoptError::BadParameters)?];
    match mode {
        Mode::Attributes => {
            decode_vertex_buffer(&mut target, count, stride, source)?;
            apply_filter(&mut target, count, stride, filter)?;
        }
        Mode::Triangles => decode_index_buffer(&mut target, count, stride, source)?,
        Mode::Indices => decode_index_sequence(&mut target, count, stride, source)?,
    }
    Ok(target)
}

fn dezig(v: u32) -> u32 {
    if v & 1 != 0 {
        !(v >> 1)
    } else {
        v >> 1
    }
}

/// A cursor that reports the end of the stream instead of panicking at it.
struct Bytes<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Bytes<'a> {
    fn byte(&mut self) -> Result<u8, MeshoptError> {
        let v = *self.data.get(self.at).ok_or(MeshoptError::Truncated)?;
        self.at += 1;
        Ok(v)
    }

    fn at(&self, index: usize) -> Result<u8, MeshoptError> {
        self.data.get(index).copied().ok_or(MeshoptError::Truncated)
    }

    fn slice(&mut self, n: usize) -> Result<&'a [u8], MeshoptError> {
        let end = self.at.checked_add(n).ok_or(MeshoptError::Truncated)?;
        let s = self.data.get(self.at..end).ok_or(MeshoptError::Truncated)?;
        self.at = end;
        Ok(s)
    }

    /// LEB128, as the index codecs write their escapes.
    fn leb128(&mut self) -> Result<u32, MeshoptError> {
        let mut n: u32 = 0;
        let mut shift = 0;
        loop {
            let b = self.byte()?;
            n |= ((b & 0x7f) as u32) << shift;
            if b < 0x80 {
                return Ok(n);
            }
            shift += 7;
            if shift > 28 {
                return Err(MeshoptError::Truncated);
            }
        }
    }
}

/// The interleaved vertex stream.
///
/// Deltas are stored one *byte plane* at a time — every vertex's first byte,
/// then every vertex's second byte — because the high byte of a coordinate
/// changes far less often than the low one, and a plane of mostly-equal bytes
/// compresses to almost nothing. Version 1 adds per-4-byte-group "channels",
/// which say whether a group is delta-coded as bytes, as 16-bit values, or as
/// rotated XOR.
fn decode_vertex_buffer(
    target: &mut [u8],
    count: usize,
    stride: usize,
    source: &[u8],
) -> Result<(), MeshoptError> {
    let header = *source.first().ok_or(MeshoptError::Truncated)?;
    if header != 0xa0 && header != 0xa1 {
        return Err(MeshoptError::BadHeader);
    }
    let version = (header & 0x0f) as usize;
    if stride == 0 || stride > 256 || !stride.is_multiple_of(4) {
        return Err(MeshoptError::BadParameters);
    }

    let max_block = ((0x2000 / stride) & !0x0f).clamp(1, 0x100);
    let tail_size = if version == 0 {
        stride
    } else {
        stride + stride / 4
    };
    let tail_at = source
        .len()
        .checked_sub(tail_size)
        .ok_or(MeshoptError::Truncated)?;

    // The tail holds what the first block's deltas are relative to, and for
    // version 1 the channel modes as well.
    let mut last = source
        .get(tail_at..tail_at + stride)
        .ok_or(MeshoptError::Truncated)?
        .to_vec();
    let channels = if version == 0 {
        Vec::new()
    } else {
        source
            .get(tail_at + stride..tail_at + tail_size)
            .ok_or(MeshoptError::Truncated)?
            .to_vec()
    };

    // Bits per delta, by version and control mode.
    const MODES: [[u32; 4]; 3] = [
        [0, 2, 4, 8], // version 0
        [0, 1, 2, 4], // version 1, control 0
        [1, 2, 4, 8], // version 1, control 1
    ];

    let mut reader = Bytes {
        data: source,
        at: 1,
    };
    let mut deltas = vec![0u8; max_block * stride];

    let mut base = 0usize;
    while base < count {
        let block = (count - base).min(max_block);
        let groups = (block + 0x0f) >> 4;
        let header_bytes = (groups + 3) >> 2;

        let control_at = reader.at;
        if version != 0 {
            reader.at += stride / 4;
        }
        deltas[..block * stride].fill(0);

        for byte in 0..stride {
            let delta_base = byte * block;
            let control = if version == 0 {
                0u8
            } else {
                (reader.at(control_at + (byte >> 2))? >> ((byte & 3) << 1)) & 3
            };
            // 2: the whole plane is zero. 3: stored verbatim, no header.
            if control == 2 {
                continue;
            }
            if control == 3 {
                let raw = reader.slice(block)?.to_vec();
                deltas[delta_base..delta_base + block].copy_from_slice(&raw);
                continue;
            }

            let header_at = reader.at;
            reader.at += header_bytes;
            for group in 0..groups {
                let selector = (reader.at(header_at + (group >> 2))? >> ((group & 3) << 1)) & 3;
                let bits = MODES[if version == 0 { 0 } else { control as usize + 1 }]
                    [selector as usize];
                let out = delta_base + (group << 4);
                match bits {
                    // Every delta in the group is zero; nothing is stored.
                    0 => {}
                    // Sentinel encodings: the packed value is the delta unless
                    // it is all ones, which means "read a whole byte instead".
                    1 => {
                        let packed = reader.slice(2)?.to_vec();
                        for m in 0..16 {
                            let mut d = (packed[m >> 3] >> (m & 7)) & 1;
                            if d == 1 {
                                d = reader.byte()?;
                            }
                            set(&mut deltas, out + m, d);
                        }
                    }
                    2 => {
                        let packed = reader.slice(4)?.to_vec();
                        for m in 0..16 {
                            let shift = 6 - ((m & 3) << 1);
                            let mut d = (packed[m >> 2] >> shift) & 3;
                            if d == 3 {
                                d = reader.byte()?;
                            }
                            set(&mut deltas, out + m, d);
                        }
                    }
                    4 => {
                        let packed = reader.slice(8)?.to_vec();
                        for m in 0..16 {
                            let shift = 4 - ((m & 1) << 2);
                            let mut d = (packed[m >> 1] >> shift) & 0x0f;
                            if d == 0x0f {
                                d = reader.byte()?;
                            }
                            set(&mut deltas, out + m, d);
                        }
                    }
                    _ => {
                        let raw = reader.slice(16)?.to_vec();
                        for (m, v) in raw.iter().enumerate() {
                            set(&mut deltas, out + m, *v);
                        }
                    }
                }
            }
        }

        // Deltas are now a plane per byte; fold them into vertices.
        for element in 0..block {
            let vertex = base + element;
            for group in (0..stride).step_by(4) {
                let mode = if version == 0 {
                    0
                } else {
                    channels[group >> 2] & 3
                };
                match mode {
                    // Byte deltas, zigzagged, against the previous vertex.
                    0 => {
                        for byte in group..group + 4 {
                            let d = dezig(deltas[byte * block + element] as u32) as u8;
                            let v = last[byte].wrapping_add(d);
                            last[byte] = v;
                            put(target, vertex * stride + byte, v);
                        }
                    }
                    // 16-bit deltas across each pair of bytes.
                    1 => {
                        for byte in (group..group + 4).step_by(2) {
                            let packed = deltas[byte * block + element] as u32
                                | ((deltas[(byte + 1) * block + element] as u32) << 8);
                            let d = dezig(packed) as u16;
                            let was = last[byte] as u16 | ((last[byte + 1] as u16) << 8);
                            let v = was.wrapping_add(d);
                            last[byte] = v as u8;
                            last[byte + 1] = (v >> 8) as u8;
                            put(target, vertex * stride + byte, v as u8);
                            put(target, vertex * stride + byte + 1, (v >> 8) as u8);
                        }
                    }
                    // 32-bit XOR against a rotation of the delta, which is how
                    // float mantissas compress when the exponent is stable.
                    2 => {
                        let mut d = 0u32;
                        let mut was = 0u32;
                        for i in 0..4 {
                            d |= (deltas[(group + i) * block + element] as u32) << (8 * i);
                            was |= (last[group + i] as u32) << (8 * i);
                        }
                        let rot = (channels[group >> 2] >> 4) as u32;
                        let v = was ^ d.rotate_right(rot % 32);
                        for i in 0..4 {
                            let b = (v >> (8 * i)) as u8;
                            last[group + i] = b;
                            put(target, vertex * stride + group + i, b);
                        }
                    }
                    _ => return Err(MeshoptError::BadParameters),
                }
            }
        }
        base += block;
    }
    Ok(())
}

fn set(buffer: &mut [u8], at: usize, value: u8) {
    if let Some(slot) = buffer.get_mut(at) {
        *slot = value;
    }
}

fn put(buffer: &mut [u8], at: usize, value: u8) {
    if let Some(slot) = buffer.get_mut(at) {
        *slot = value;
    }
}

/// A triangle list.
///
/// Each triangle is a byte: the high nibble names an edge from a FIFO of the
/// last 16 edges, the low nibble names the third vertex — from a FIFO of recent
/// vertices, as the next unused index, or spelled out. A mesh whose triangles
/// share edges with their neighbours therefore costs about a byte a triangle,
/// which is the whole point of the codec.
fn decode_index_buffer(
    target: &mut [u8],
    count: usize,
    stride: usize,
    source: &[u8],
) -> Result<(), MeshoptError> {
    if source.first() != Some(&0xe1) {
        return Err(MeshoptError::BadHeader);
    }
    if !count.is_multiple_of(3) || (stride != 2 && stride != 4) {
        return Err(MeshoptError::BadParameters);
    }
    let triangles = count / 3;
    // Codes are a byte each, straight after the header; the variable-length
    // data follows them, and the last 16 bytes are a shared table of common
    // second/third-vertex pairs.
    let mut data = Bytes {
        data: source,
        at: 1 + triangles,
    };
    // One code byte per triangle, straight after the header.
    let codes = source
        .get(1..1 + triangles)
        .ok_or(MeshoptError::Truncated)?
        .to_vec();
    let codeaux_at = source
        .len()
        .checked_sub(16)
        .ok_or(MeshoptError::Truncated)?;

    let mut next: u32 = 0;
    let mut last: u32 = 0;
    let mut edges = [0u32; 32];
    let mut vertices = [0u32; 16];
    let mut out = 0usize;

    for &code in &codes {
        let (hi, lo) = (code >> 4, code & 0x0f);

        let (a, b, c) = if hi < 0x0f {
            let a = edges[(hi as usize) << 1];
            let b = edges[((hi as usize) << 1) + 1];
            let c = match lo {
                0x00 => {
                    let c = next;
                    next += 1;
                    push(&mut vertices, c);
                    c
                }
                v if v < 0x0d => vertices[v as usize],
                0x0d => {
                    last = last.wrapping_sub(1);
                    push(&mut vertices, last);
                    last
                }
                0x0e => {
                    last = last.wrapping_add(1);
                    push(&mut vertices, last);
                    last
                }
                _ => {
                    last = last.wrapping_add(dezig(data.leb128()?));
                    push(&mut vertices, last);
                    last
                }
            };
            // The pushes go in backwards, so the newest edge ends up first.
            push(&mut edges, b);
            push(&mut edges, c);
            push(&mut edges, c);
            push(&mut edges, a);
            (a, b, c)
        } else if lo < 0x0e {
            // A fresh triangle whose shape comes from the shared table.
            let e = *source
                .get(codeaux_at + lo as usize)
                .ok_or(MeshoptError::Truncated)?;
            let (z, w) = (e >> 4, e & 0x0f);
            let a = next;
            next += 1;
            let b = if z == 0 {
                let v = next;
                next += 1;
                v
            } else {
                vertices[(z - 1) as usize]
            };
            let c = if w == 0 {
                let v = next;
                next += 1;
                v
            } else {
                vertices[(w - 1) as usize]
            };
            push(&mut vertices, a);
            if z == 0 {
                push(&mut vertices, b);
            }
            if w == 0 {
                push(&mut vertices, c);
            }
            push(&mut edges, a);
            push(&mut edges, b);
            push(&mut edges, b);
            push(&mut edges, c);
            push(&mut edges, c);
            push(&mut edges, a);
            (a, b, c)
        } else {
            // Fully spelled out, with the shape byte in the data stream.
            let e = data.byte()?;
            if e == 0 {
                next = 0;
            }
            let (z, w) = (e >> 4, e & 0x0f);
            let a = if lo == 0x0e {
                let v = next;
                next += 1;
                v
            } else {
                last = last.wrapping_add(dezig(data.leb128()?));
                last
            };
            let b = match z {
                0 => {
                    let v = next;
                    next += 1;
                    v
                }
                0x0f => {
                    last = last.wrapping_add(dezig(data.leb128()?));
                    last
                }
                _ => vertices[(z - 1) as usize],
            };
            let c = match w {
                0 => {
                    let v = next;
                    next += 1;
                    v
                }
                0x0f => {
                    last = last.wrapping_add(dezig(data.leb128()?));
                    last
                }
                _ => vertices[(w - 1) as usize],
            };
            push(&mut vertices, a);
            if z == 0 || z == 0x0f {
                push(&mut vertices, b);
            }
            if w == 0 || w == 0x0f {
                push(&mut vertices, c);
            }
            push(&mut edges, a);
            push(&mut edges, b);
            push(&mut edges, b);
            push(&mut edges, c);
            push(&mut edges, c);
            push(&mut edges, a);
            (a, b, c)
        };

        for v in [a, b, c] {
            write_index(target, out, stride, v);
            out += 1;
        }
    }
    Ok(())
}

/// A plain index sequence — two interleaved running values, LEB128 deltas.
fn decode_index_sequence(
    target: &mut [u8],
    count: usize,
    stride: usize,
    source: &[u8],
) -> Result<(), MeshoptError> {
    if source.first() != Some(&0xd1) {
        return Err(MeshoptError::BadHeader);
    }
    if stride != 2 && stride != 4 {
        return Err(MeshoptError::BadParameters);
    }
    let mut data = Bytes {
        data: source,
        at: 1,
    };
    let mut last = [0u32; 2];
    for i in 0..count {
        let v = data.leb128()?;
        let which = (v & 1) as usize;
        last[which] = last[which].wrapping_add(dezig(v >> 1));
        write_index(target, i, stride, last[which]);
    }
    Ok(())
}

fn write_index(target: &mut [u8], slot: usize, stride: usize, value: u32) {
    let at = slot * stride;
    if stride == 2 {
        if let Some(s) = target.get_mut(at..at + 2) {
            s.copy_from_slice(&(value as u16).to_le_bytes());
        }
    } else if let Some(s) = target.get_mut(at..at + 4) {
        s.copy_from_slice(&value.to_le_bytes());
    }
}

/// The FIFOs shift rather than wrap: entry 0 is always the newest.
fn push(fifo: &mut [u32], value: u32) {
    for i in (1..fifo.len()).rev() {
        fifo[i] = fifo[i - 1];
    }
    fifo[0] = value;
}

// ---- Filters ----

fn apply_filter(
    target: &mut [u8],
    count: usize,
    stride: usize,
    filter: Filter,
) -> Result<(), MeshoptError> {
    match filter {
        Filter::None => Ok(()),
        Filter::Octahedral => octahedral(target, count, stride),
        Filter::Quaternion => quaternion(target, count, stride),
        Filter::Exponential => exponential(target, count, stride),
        Filter::Color => colour(target, count, stride),
    }
}

/// Two stored components and a scale become a unit vector.
///
/// The third component is recovered from the other two, which is what makes an
/// octahedral normal two numbers instead of three.
fn octahedral(target: &mut [u8], count: usize, stride: usize) -> Result<(), MeshoptError> {
    let wide = match stride {
        4 => false,
        8 => true,
        _ => return Err(MeshoptError::BadParameters),
    };
    let max = if wide { 32767.0f32 } else { 127.0f32 };
    for i in 0..count {
        let at = i * stride;
        let get = |k: usize| -> f32 {
            if wide {
                i16::from_le_bytes([target[at + k * 2], target[at + k * 2 + 1]]) as f32
            } else {
                target[at + k] as i8 as f32
            }
        };
        let one = get(2);
        if one == 0.0 {
            continue;
        }
        let mut x = get(0) / one;
        let mut y = get(1) / one;
        let z = 1.0 - x.abs() - y.abs();
        let t = (-z).max(0.0);
        x -= if x >= 0.0 { t } else { -t };
        y -= if y >= 0.0 { t } else { -t };
        let h = max / (x * x + y * y + z * z).sqrt();
        for (k, v) in [x, y, z].iter().enumerate() {
            let scaled = (v * h).round();
            if wide {
                let b = (scaled as i16).to_le_bytes();
                target[at + k * 2] = b[0];
                target[at + k * 2 + 1] = b[1];
            } else {
                target[at + k] = (scaled as i8) as u8;
            }
        }
    }
    Ok(())
}

/// Three components and two bits saying which one was dropped.
///
/// A unit quaternion's largest component is recoverable from the other three,
/// and its sign can be chosen, so only three need storing — but *which* three
/// varies per quaternion, and those two bits are what say so.
fn quaternion(target: &mut [u8], count: usize, stride: usize) -> Result<(), MeshoptError> {
    if stride != 8 {
        return Err(MeshoptError::BadParameters);
    }
    for i in 0..count {
        let at = i * 8;
        let get = |k: usize| i16::from_le_bytes([target[at + k * 2], target[at + k * 2 + 1]]);
        let input_w = get(3);
        let largest = (input_w & 3) as usize;
        let s = std::f32::consts::FRAC_1_SQRT_2 / (input_w | 3) as f32;
        let x = get(0) as f32 * s;
        let y = get(1) as f32 * s;
        let z = get(2) as f32 * s;
        let w = (1.0 - x * x - y * y - z * z).max(0.0).sqrt();
        for (offset, value) in [(1, x), (2, y), (3, z), (0, w)] {
            let k = (largest + offset) % 4;
            let b = ((value * 32767.0).round() as i16).to_le_bytes();
            target[at + k * 2] = b[0];
            target[at + k * 2 + 1] = b[1];
        }
    }
    Ok(())
}

/// A shared exponent and a 24-bit mantissa become a float.
fn exponential(target: &mut [u8], count: usize, stride: usize) -> Result<(), MeshoptError> {
    if !stride.is_multiple_of(4) {
        return Err(MeshoptError::BadParameters);
    }
    for i in 0..(stride * count) / 4 {
        let at = i * 4;
        let Some(word) = target.get(at..at + 4) else {
            break;
        };
        let v = i32::from_le_bytes([word[0], word[1], word[2], word[3]]);
        let exponent = v >> 24;
        // Sign-extend the low 24 bits.
        let mantissa = (v << 8) >> 8;
        let value = (exponent as f32).exp2() * mantissa as f32;
        target[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }
    Ok(())
}

/// YCoCg with a scale hidden in the alpha channel's high bit.
fn colour(target: &mut [u8], count: usize, stride: usize) -> Result<(), MeshoptError> {
    let wide = match stride {
        4 => false,
        8 => true,
        _ => return Err(MeshoptError::BadParameters),
    };
    let max = if wide { 65535.0f32 } else { 255.0f32 };
    for i in 0..count {
        let at = i * stride;
        let unsigned = |k: usize| -> i32 {
            if wide {
                u16::from_le_bytes([target[at + k * 2], target[at + k * 2 + 1]]) as i32
            } else {
                target[at + k] as i32
            }
        };
        let signed = |k: usize| -> i32 {
            if wide {
                i16::from_le_bytes([target[at + k * 2], target[at + k * 2 + 1]]) as i32
            } else {
                target[at + k] as i8 as i32
            }
        };
        let y = unsigned(0);
        let co = signed(1);
        let cg = signed(2);
        let alpha_in = unsigned(3);
        // The scale is the highest bit set in alpha.
        let scale = if alpha_in == 0 {
            1
        } else {
            (1i32 << (32 - (alpha_in as u32).leading_zeros())) - 1
        };
        let r = y + co - cg;
        let g = y + cg;
        let b = y - co - cg;
        let mut a = alpha_in & (scale >> 1);
        a = (a << 1) | (a & 1);
        let s = max / scale as f32;
        for (k, v) in [r, g, b, a].iter().enumerate() {
            let scaled = (*v as f32 * s).round();
            if wide {
                let bytes = (scaled.clamp(0.0, 65535.0) as u16).to_le_bytes();
                target[at + k * 2] = bytes[0];
                target[at + k * 2 + 1] = bytes[1];
            } else {
                target[at + k] = scaled.clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(())
}
