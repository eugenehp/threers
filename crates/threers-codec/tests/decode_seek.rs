//! `VideoReader::decode_rgb` starts at a sync sample (one GOP of margin)
//! instead of the stream start: the frames it returns must be byte-identical
//! to a decode from the start. Every multi-GOP MP4 / WebM / MKV under
//! `$THREERS_MEDIA` is checked; unset, a clip ffmpeg makes here is.
use std::collections::BTreeSet;
use threers_codec::ffcompat::reader::VideoReader;

fn files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            files(&p, out);
        } else if matches!(
            p.extension().and_then(|x| x.to_str()),
            Some("mp4" | "mov" | "webm" | "mkv")
        ) {
            out.push(p);
        }
    }
}

/// A short clip with B-frames and several keyframes, so a seek has to decode
/// forward from one and put frames back in display order.
fn generated_clip() -> Option<std::path::PathBuf> {
    let path = std::env::temp_dir().join(format!("threers_seek_{}.mp4", std::process::id()));
    let made = std::process::Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x180:r=30:d=3",
        ])
        .args([
            "-c:v", "libx264", "-g", "24", "-bf", "2", "-pix_fmt", "yuv420p",
        ])
        .arg(&path)
        .status();
    made.is_ok_and(|s| s.success()).then_some(path)
}

#[test]
fn seeking_decode_matches_full_decode() {
    let mut all = Vec::new();
    let mut clip = None;
    match std::env::var("THREERS_MEDIA") {
        Ok(root) => files(std::path::Path::new(&root), &mut all),
        Err(_) => {
            clip = generated_clip();
            all.extend(clip.clone());
        }
    }
    if all.is_empty() {
        eprintln!("SKIP: no videos in THREERS_MEDIA, and ffmpeg could not make one");
        return;
    }
    let (mut checked, mut seeked) = (0, 0);
    for p in &all {
        let Ok(r) = VideoReader::open(std::fs::read(p).unwrap()) else {
            continue;
        };
        let n = r.mov.sample.len();
        if n < 8 {
            continue;
        }
        let late: BTreeSet<usize> = [n * 2 / 3, n * 2 / 3 + 1, n - 2, n - 1]
            .into_iter()
            .collect();
        let mut with_zero = late.clone();
        with_zero.insert(0);
        let mut a = std::collections::BTreeMap::new();
        if r.decode_rgb(&late, |i, rgb| {
            a.insert(i, rgb.to_vec());
            Ok(())
        })
        .is_err()
        {
            continue; // a format this reader rejects either way
        }
        let mut b = std::collections::BTreeMap::new();
        r.decode_rgb(&with_zero, |i, rgb| {
            if late.contains(&i) {
                b.insert(i, rgb.to_vec());
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(
            a.keys().collect::<Vec<_>>(),
            b.keys().collect::<Vec<_>>(),
            "{}",
            p.display()
        );
        for (i, fa) in &a {
            assert!(
                fa == &b[i],
                "{}: frame {i} differs between the seeking and the full decode",
                p.display()
            );
        }
        checked += 1;
        seeked += r.seeks_for(&late) as usize;
    }
    if let Some(clip) = clip {
        let _ = std::fs::remove_file(clip);
    }
    eprintln!("{checked} videos checked, {seeked} decoded from a later sync sample");
    assert!(checked > 0);
}
