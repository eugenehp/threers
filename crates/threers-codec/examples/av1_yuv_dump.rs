//! Decode an AV1 IVF (or Section 5 `.obu`) file with `codec::decode::av1`
//! and write every shown frame as raw planes (8-bit: one byte per sample;
//! 10/12-bit: little-endian u16), as `ffmpeg -f rawvideo` does.
//!
//!   av1_yuv_dump IN.ivf|IN.obu OUT.yuv
use std::io::Write;

use threers_codec::decode::av1::Av1Decoder;
use threers_codec::mkv_demux::parse_ivf;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).expect("read");
    // Section 5 streams start with a temporal delimiter OBU (0x12 0x00);
    // anything else that is not IVF is taken as Annex B.
    let annexb = !data.starts_with(b"DKIF") && !data.starts_with(&[0x12, 0x00]);
    let packets: Vec<Vec<u8>> = if data.starts_with(b"DKIF") {
        parse_ivf(&data).expect("ivf").4.into_iter().map(|(_, d)| d).collect()
    } else if annexb {
        threers_codec::decode::av1::split_annexb(&data).expect("annexb").into_iter().map(|t| t.to_vec()).collect()
    } else {
        vec![data]
    };
    let mut dec = Av1Decoder::new();
    let mut out = std::io::BufWriter::new(std::fs::File::create(&a[2]).expect("create"));
    let mut n = 0;
    let mut desc = String::new();
    let mut sizes = String::new();
    for p in &packets {
        let frames = match if annexb { dec.decode_annexb(p) } else { dec.decode(p) } {
            Ok(f) => f,
            Err(e) => {
                println!("ERROR {e} (after {n} frames)");
                std::process::exit(2);
            }
        };
        for f in frames {
            let pic = &f.pic;
            let np = if pic.layout == threers_codec::decode::av1::obu::Layout::I400 { 1 } else { 3 };
            for pl in 0..np {
                let (ssh, ssv) = pic.ss(pl);
                let w = ((pic.w + ssh) >> ssh) as usize;
                let h = ((pic.h + ssv) >> ssv) as usize;
                let s = pic.pstride(pl);
                for y in 0..h {
                    let row = &pic.planes[pl][y * s..y * s + w];
                    if pic.bpc == 8 {
                        out.write_all(&row.iter().map(|&v| v as u8).collect::<Vec<_>>()).unwrap();
                    } else {
                        out.write_all(&row.iter().flat_map(|&v| v.to_le_bytes()).collect::<Vec<_>>()).unwrap();
                    }
                }
            }
            sizes.push_str(&format!("{} {}\n", pic.w, pic.h));
            desc = format!("{}x{} {}-bit {:?}", pic.w, pic.h, pic.bpc, pic.layout);
            n += 1;
        }
    }
    std::fs::write(format!("{}.sizes", a[2]), sizes).unwrap();
    println!("{n} frames, {desc}");
}
