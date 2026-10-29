//! `VideoReader::decode_rgb` starts at a sync sample (one GOP of margin)
//! instead of the stream start: the frames it returns must be byte-identical
//! to a decode from the start. Every multi-GOP MP4 / WebM / MKV under
//! `$THREERS_MEDIA` (default `/Users/Shared/tribev2-ref/media`) is checked.
use std::collections::BTreeSet;
use threers_codec::ffcompat::reader::VideoReader;

fn files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            files(&p, out);
        } else if matches!(p.extension().and_then(|x| x.to_str()), Some("mp4" | "mov" | "webm" | "mkv")) {
            out.push(p);
        }
    }
}

#[test]
fn seeking_decode_matches_full_decode() {
    let root = std::env::var("THREERS_MEDIA").unwrap_or_else(|_| "/Users/Shared/tribev2-ref/media".into());
    let mut all = Vec::new();
    files(std::path::Path::new(&root), &mut all);
    if all.is_empty() {
        eprintln!("SKIP: no videos under {root}");
        return;
    }
    let (mut checked, mut seeked) = (0, 0);
    for p in &all {
        let Ok(r) = VideoReader::open(std::fs::read(p).unwrap()) else { continue };
        let n = r.mov.sample.len();
        if n < 8 {
            continue;
        }
        let late: BTreeSet<usize> = [n * 2 / 3, n * 2 / 3 + 1, n - 2, n - 1].into_iter().collect();
        let mut with_zero = late.clone();
        with_zero.insert(0);
        let mut a = std::collections::BTreeMap::new();
        if r.decode_rgb(&late, |i, rgb| { a.insert(i, rgb.to_vec()); Ok(()) }).is_err() {
            continue; // a format this reader rejects either way
        }
        let mut b = std::collections::BTreeMap::new();
        r.decode_rgb(&with_zero, |i, rgb| { if late.contains(&i) { b.insert(i, rgb.to_vec()); } Ok(()) }).unwrap();
        assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>(), "{}", p.display());
        for (i, fa) in &a {
            assert!(fa == &b[i], "{}: frame {i} differs between the seeking and the full decode", p.display());
        }
        checked += 1;
        seeked += r.seeks_for(&late) as usize;
    }
    eprintln!("{checked} videos checked, {seeked} decoded from a later sync sample");
    assert!(checked > 0);
}
