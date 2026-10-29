//! Ogg pages and packets as libogg delivers them: packets reassembled
//! across pages, each tagged with the page's granule position when it is
//! the last packet completed on that page (`-1` otherwise), a running packet
//! number, and the end-of-stream flag on the final packet of an EOS page.
//! Only the first logical stream (the first BOS page's serial) is followed.

/// One reassembled Ogg packet.
#[derive(Debug, Clone)]
pub struct Packet {
    /// The packet's bytes.
    pub data: Vec<u8>,
    /// The page's granule position if this packet completed last on it, else `-1`.
    pub granulepos: i64,
    /// Running packet number, from 0.
    pub packetno: i64,
    /// The last packet of the stream.
    pub eos: bool,
    /// Index of the page this packet completed on.
    pub page: usize,
}

/// All packets of the first logical stream, plus the last page's granule.
pub fn packets(data: &[u8]) -> Result<(Vec<Packet>, i64), String> {
    let mut out = Vec::new();
    let mut partial: Vec<u8> = Vec::new();
    let mut serial: Option<u32> = None;
    let mut off = 0;
    let mut page_no = 0;
    let mut last_granule = -1;
    let mut packetno = 0i64;
    while off + 27 <= data.len() {
        if &data[off..off + 4] != b"OggS" {
            // Resync on the next capture pattern.
            match data[off + 1..].windows(4).position(|w| w == b"OggS") {
                Some(p) => {
                    off += 1 + p;
                    continue;
                }
                None => break,
            }
        }
        let flags = data[off + 5];
        let granule = i64::from_le_bytes(data[off + 6..off + 14].try_into().unwrap());
        let ser = u32::from_le_bytes(data[off + 14..off + 18].try_into().unwrap());
        let nseg = data[off + 26] as usize;
        let lacing = data.get(off + 27..off + 27 + nseg).ok_or("truncated Ogg page")?;
        let body_len: usize = lacing.iter().map(|&l| l as usize).sum();
        let body_start = off + 27 + nseg;
        let body = data.get(body_start..body_start + body_len).ok_or("truncated Ogg page")?;
        let next = body_start + body_len;
        if serial.is_none() {
            serial = Some(ser);
        }
        if Some(ser) != serial {
            off = next;
            continue;
        }
        if flags & 1 == 0 {
            // A fresh page drops any unterminated packet only when the
            // continuation flag says the page does not continue it.
            partial.clear();
        }
        let eos_page = flags & 4 != 0;
        // Index of the last segment that ends a packet.
        let last_end = lacing.iter().rposition(|&l| l < 255);
        let mut at = 0;
        for (i, &l) in lacing.iter().enumerate() {
            partial.extend_from_slice(&body[at..at + l as usize]);
            at += l as usize;
            if l < 255 {
                let is_last = Some(i) == last_end;
                out.push(Packet {
                    data: std::mem::take(&mut partial),
                    granulepos: if is_last { granule } else { -1 },
                    packetno,
                    eos: eos_page && is_last,
                    page: page_no,
                });
                packetno += 1;
            }
        }
        if granule != -1 {
            last_granule = granule;
        }
        page_no += 1;
        off = next;
        if eos_page {
            break;
        }
    }
    Ok((out, last_granule))
}

/// FLAC in Ogg (`\x7fFLAC` mapping): STREAMINFO from the first packet, one
/// frame per audio packet.
pub fn decode_flac(data: &[u8]) -> Result<super::flac::Flac, String> {
    let (pk, _) = packets(data)?;
    let first = pk.first().ok_or("empty Ogg stream")?;
    if !first.data.starts_with(b"\x7fFLAC") || first.data.len() < 13 + 4 + 34 {
        return Err("not FLAC in Ogg".into());
    }
    let info = super::flac::parse_streaminfo(&first.data[13 + 4..])?;
    let mut channels = vec![Vec::new(); info.channels as usize];
    // Metadata packets follow (their count may be 0 = unknown); audio
    // packets start with the frame sync.
    for p in pk.iter().skip(1) {
        if p.data.len() >= 2 && p.data[0] == 0xff && p.data[1] & 0xfe == 0xf8 {
            super::flac::decode_frames(&p.data, &info, &mut channels)?;
        }
    }
    Ok(super::flac::Flac { info, channels })
}
