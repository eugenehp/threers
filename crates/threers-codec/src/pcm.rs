//! Uncompressed PCM files: WAV (RIFF/RF64, PCM 8–32-bit, float 32/64,
//! `WAVE_FORMAT_EXTENSIBLE`) and AIFF / AIFF-C (`NONE`, `twos`, `sowt`,
//! `fl32`, `fl64`), read to float the way libsndfile (`soundfile.read`)
//! normalises them — signed `int / 2^(bits-1)`, unsigned 8-bit
//! `(v - 128) / 128` — and 16-bit WAV writing.

/// Planar samples and their rate.
#[derive(Debug, Clone, PartialEq)]
pub struct Pcm {
    /// Samples per second.
    pub sample_rate: u32,
    /// One `Vec` per channel.
    pub channels: Vec<Vec<f32>>,
}

impl Pcm {
    /// Samples per channel.
    pub fn frames(&self) -> usize {
        self.channels.first().map_or(0, Vec::len)
    }
    /// Length in seconds.
    pub fn duration(&self) -> f64 {
        self.frames() as f64 / self.sample_rate as f64
    }
}

fn le16(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}
fn le32(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(b[i..i + 4].try_into().unwrap())
}
fn be32(b: &[u8], i: usize) -> u32 {
    u32::from_be_bytes(b[i..i + 4].try_into().unwrap())
}

#[derive(Clone, Copy)]
enum Enc {
    Int { bytes: usize, big: bool },
    U8,
    F32 { big: bool },
    F64 { big: bool },
}

fn decode(pcm: &[u8], n_ch: usize, enc: Enc) -> Vec<Vec<f32>> {
    let width = match enc {
        Enc::Int { bytes, .. } => bytes,
        Enc::U8 => 1,
        Enc::F32 { .. } => 4,
        Enc::F64 { .. } => 8,
    };
    let n = pcm.len() / width / n_ch.max(1) * n_ch;
    let mut ch = vec![Vec::with_capacity(n / n_ch.max(1)); n_ch];
    for i in 0..n {
        let b = &pcm[i * width..(i + 1) * width];
        let v = match enc {
            Enc::U8 => (b[0] as f64 - 128.0) / 128.0,
            Enc::Int { bytes, big } => {
                let mut v: i64 = 0;
                for k in 0..bytes {
                    v = (v << 8) | b[if big { k } else { bytes - 1 - k }] as i64;
                }
                let sh = 64 - 8 * bytes as u32;
                ((v << sh) >> sh) as f64 / (1u64 << (8 * bytes - 1)) as f64
            }
            Enc::F32 { big } => {
                let a: [u8; 4] = b.try_into().unwrap();
                (if big { f32::from_be_bytes(a) } else { f32::from_le_bytes(a) }) as f64
            }
            Enc::F64 { big } => {
                let a: [u8; 8] = b.try_into().unwrap();
                if big { f64::from_be_bytes(a) } else { f64::from_le_bytes(a) }
            }
        };
        ch[i % n_ch].push(v as f32);
    }
    ch
}

/// Read a WAV file.
pub fn read_wav(d: &[u8]) -> Result<Pcm, String> {
    if d.len() < 12 || !matches!(&d[0..4], b"RIFF" | b"RF64") || &d[8..12] != b"WAVE" {
        return Err("not a WAV file".into());
    }
    let (mut fmt, mut data, mut ds64_data) = (None, None, None);
    let mut off = 12;
    while off + 8 <= d.len() {
        let id = &d[off..off + 4];
        let mut len = le32(d, off + 4) as usize;
        if id == b"ds64" && off + 24 <= d.len() {
            ds64_data = Some(u64::from_le_bytes(d[off + 16..off + 24].try_into().unwrap()) as usize);
        }
        if id == b"data" {
            if len == 0xffff_ffff {
                len = ds64_data.unwrap_or(d.len() - off - 8);
            }
            data = Some(&d[off + 8..(off + 8 + len).min(d.len())]);
            break;
        }
        let body = d.get(off + 8..off + 8 + len).ok_or("truncated WAV chunk")?;
        if id == b"fmt " {
            fmt = Some(body);
        }
        off += 8 + len + (len & 1);
    }
    let fmt = fmt.ok_or("WAV without fmt")?;
    let data = data.ok_or("WAV without data")?;
    let mut tag = le16(fmt, 0);
    let n_ch = le16(fmt, 2) as usize;
    let rate = le32(fmt, 4);
    let bits = le16(fmt, 14) as usize;
    if tag == 0xfffe && fmt.len() >= 26 {
        tag = le16(fmt, 24);
    }
    let enc = match (tag, bits) {
        (1, 8) => Enc::U8,
        (1, 16 | 24 | 32) => Enc::Int { bytes: bits / 8, big: false },
        (3, 32) => Enc::F32 { big: false },
        (3, 64) => Enc::F64 { big: false },
        _ => return Err(format!("WAV format tag {tag} with {bits} bits is not supported")),
    };
    if n_ch == 0 {
        return Err("WAV with zero channels".into());
    }
    Ok(Pcm { sample_rate: rate, channels: decode(data, n_ch, enc) })
}

/// Read an AIFF or AIFF-C file.
pub fn read_aiff(d: &[u8]) -> Result<Pcm, String> {
    if d.len() < 12 || &d[0..4] != b"FORM" || !matches!(&d[8..12], b"AIFF" | b"AIFC") {
        return Err("not an AIFF file".into());
    }
    let aifc = &d[8..12] == b"AIFC";
    let (mut comm, mut ssnd) = (None, None);
    let mut off = 12;
    while off + 8 <= d.len() {
        let id = &d[off..off + 4];
        let len = be32(d, off + 4) as usize;
        let body = d.get(off + 8..(off + 8 + len).min(d.len())).ok_or("truncated AIFF chunk")?;
        match id {
            b"COMM" => comm = Some(body),
            b"SSND" => ssnd = Some(body),
            _ => {}
        }
        off += 8 + len + (len & 1);
    }
    let comm = comm.ok_or("AIFF without COMM")?;
    let ssnd = ssnd.ok_or("AIFF without SSND")?;
    if comm.len() < 18 || ssnd.len() < 8 {
        return Err("short AIFF chunk".into());
    }
    let n_ch = u16::from_be_bytes([comm[0], comm[1]]) as usize;
    let frames = be32(comm, 2) as usize;
    let bits = u16::from_be_bytes([comm[6], comm[7]]) as usize;
    // 80-bit IEEE extended sample rate.
    let exp = (u16::from_be_bytes([comm[8], comm[9]]) & 0x7fff) as i32 - 16383;
    let mant = u64::from_be_bytes(comm[10..18].try_into().unwrap());
    let rate = (mant as f64 * 2f64.powi(exp - 63)).round() as u32;
    let codec: &[u8] = if aifc { comm.get(18..22).ok_or("AIFC without compression type")? } else { b"NONE" };
    let enc = match codec {
        b"NONE" | b"twos" => Enc::Int { bytes: bits.div_ceil(8), big: true },
        b"sowt" => Enc::Int { bytes: bits.div_ceil(8), big: false },
        b"fl32" | b"FL32" => Enc::F32 { big: true },
        b"fl64" | b"FL64" => Enc::F64 { big: true },
        other => return Err(format!("AIFF-C compression {:?} is not supported", String::from_utf8_lossy(other))),
    };
    let start = be32(ssnd, 0) as usize + 8;
    let pcm = ssnd.get(start..).ok_or("bad SSND offset")?;
    if n_ch == 0 {
        return Err("AIFF with zero channels".into());
    }
    let mut ch = decode(pcm, n_ch, enc);
    for c in &mut ch {
        c.truncate(frames);
    }
    Ok(Pcm { sample_rate: rate, channels: ch })
}

/// A 16-bit PCM WAV file of interleaved `samples`.
pub fn wav_s16(sample_rate: u32, n_ch: u16, samples: &[i16]) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut o = Vec::with_capacity(44 + data_len as usize);
    o.extend_from_slice(b"RIFF");
    o.extend_from_slice(&(36 + data_len).to_le_bytes());
    o.extend_from_slice(b"WAVEfmt ");
    o.extend_from_slice(&16u32.to_le_bytes());
    o.extend_from_slice(&1u16.to_le_bytes());
    o.extend_from_slice(&n_ch.to_le_bytes());
    o.extend_from_slice(&sample_rate.to_le_bytes());
    o.extend_from_slice(&(sample_rate * n_ch as u32 * 2).to_le_bytes());
    o.extend_from_slice(&(n_ch * 2).to_le_bytes());
    o.extend_from_slice(&16u16.to_le_bytes());
    o.extend_from_slice(b"data");
    o.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        o.extend_from_slice(&s.to_le_bytes());
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_round_trip_s16() {
        let s: Vec<i16> = vec![0, 1, -1, i16::MAX, i16::MIN, 1234];
        let p = read_wav(&wav_s16(16000, 2, &s)).unwrap();
        assert_eq!(p.sample_rate, 16000);
        assert_eq!(p.channels[0], vec![0.0, -1.0 / 32768.0, -1.0]);
        assert_eq!(p.channels[1], vec![1.0 / 32768.0, 32767.0 / 32768.0, 1234.0 / 32768.0]);
    }
}
