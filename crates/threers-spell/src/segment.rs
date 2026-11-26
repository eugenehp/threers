//! Words in text written without spaces between them: Thai, Lao, Khmer
//! and Burmese, found with ICU's word lists (`data/*.words`, under the
//! Unicode licence: `words-sea.LICENSE.txt`) by matching the fewest
//! words, unknown letters taken one cluster at a time. For breaking lines
//! and moving by words.
//!
//! ```
//! let w = threers_spell::segment::words("ภาษาไทยง่ายนิดเดียว");
//! assert_eq!(w, ["ภาษา", "ไทย", "ง่าย", "นิด", "เดียว"]);
//! ```

use std::sync::OnceLock;

use crate::Words;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Script {
    Thai,
    Lao,
    Khmer,
    Myanmar,
}

/// The script of a character that needs its words found, if it does.
pub fn script_of(c: char) -> Option<Script> {
    Some(match c as u32 {
        0xE01..=0xE7F => Script::Thai,
        0xE80..=0xEFF => Script::Lao,
        0x1780..=0x17FF | 0x19E0..=0x19FF => Script::Khmer,
        0x1000..=0x109F | 0xA9E0..=0xA9FF | 0xAA60..=0xAA7F => Script::Myanmar,
        _ => return None,
    })
}

fn list(s: Script) -> &'static Words {
    static L: [OnceLock<Words>; 4] = [const { OnceLock::new() }; 4];
    let (i, data): (usize, &[u8]) = match s {
        Script::Thai => (0, include_bytes!("../data/thai.words")),
        Script::Lao => (1, include_bytes!("../data/lao.words")),
        Script::Khmer => (2, include_bytes!("../data/khmer.words")),
        Script::Myanmar => (3, include_bytes!("../data/burmese.words")),
    };
    L[i].get_or_init(|| Words::decode(data))
}

/// Whether a character goes on in the cluster before it (a vowel sign, a
/// tone mark, a subscript's sign): words do not end before it.
fn continues(c: char) -> bool {
    matches!(c as u32,
        0xE31 | 0xE33..=0xE3A | 0xE47..=0xE4E | 0xEB1 | 0xEB3..=0xEBC | 0xEC8..=0xECE
        | 0x17B4..=0x17D3 | 0x17DD | 0x102B..=0x103E | 0x1056..=0x1059 | 0x105E..=0x1060 | 0x1062..=0x1064
        | 0x1067..=0x106D | 0x1071..=0x1074 | 0x1082..=0x108D | 0x108F | 0x109A..=0x109D | 0x200D)
}

/// Thai and Lao vowels written before their consonant: a word does not end after one.
/// (And Khmer's COENG, Myanmar's stacking virama: the consonant after goes under.)
fn leads(c: char) -> bool {
    matches!(c as u32, 0xE40..=0xE44 | 0xEC0..=0xEC4 | 0x17D2 | 0x1039)
}

/// Byte offsets in `text` where words end, for the runs of these scripts
/// in it (and where each run starts); elsewhere, nothing.
pub fn boundaries(text: &str) -> Vec<usize> {
    let mut out = Vec::new();
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut i = 0;
    while i < chars.len() {
        let Some(script) = script_of(chars[i].1) else {
            i += 1;
            continue;
        };
        let mut j = i;
        while j < chars.len() && script_of(chars[j].1) == Some(script) {
            j += 1;
        }
        let start = chars[i].0;
        let end = chars.get(j).map_or(text.len(), |c| c.0);
        // what came before the run ends there too
        if start > 0 && out.last() != Some(&start) {
            out.push(start);
        }
        out.extend(segment(&text[start..end], script).into_iter().map(|b| start + b));
        i = j;
    }
    out
}

/// The words of `text` (the runs of these scripts split, the rest as it is between them).
pub fn words(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut last = 0;
    for b in boundaries(text) {
        if b > last {
            out.push(&text[last..b]);
            last = b;
        }
    }
    if last < text.len() {
        out.push(&text[last..]);
    }
    out
}

/// Where the words of a run in one script end: the split with the fewest
/// unknown clusters, then the fewest words.
fn segment(run: &str, script: Script) -> Vec<usize> {
    let words = list(script);
    // where a word may end: between clusters
    let mut cuts: Vec<usize> = Vec::new();
    let chars: Vec<(usize, char)> = run.char_indices().collect();
    for (k, &(o, c)) in chars.iter().enumerate() {
        if k > 0 && !continues(c) && !leads(chars[k - 1].1) {
            cuts.push(o);
        }
    }
    cuts.push(run.len());
    let at: Vec<usize> = std::iter::once(0).chain(cuts.iter().copied()).collect();
    let n = at.len();
    const LONGEST: usize = 24;
    // best[k]: (unknown clusters, words) to reach at[k], and where from
    let mut best: Vec<(u32, u32, usize)> = vec![(u32::MAX, u32::MAX, 0); n];
    best[0] = (0, 0, 0);
    for k in 0..n - 1 {
        let (unknown, count, _) = best[k];
        if unknown == u32::MAX {
            continue;
        }
        // a cluster alone, unknown
        let step = (unknown + 1, count + 1, k);
        if (step.0, step.1) < (best[k + 1].0, best[k + 1].1) {
            best[k + 1] = step;
        }
        for m in k + 2..n.min(k + 1 + LONGEST) {
            if words.contains(&run[at[k]..at[m]]) {
                let step = (unknown, count + 1, k);
                if (step.0, step.1) < (best[m].0, best[m].1) {
                    best[m] = step;
                }
            }
        }
        if words.contains(&run[at[k]..at[k + 1]]) && (unknown, count + 1) < (best[k + 1].0, best[k + 1].1) {
            best[k + 1] = (unknown, count + 1, k);
        }
    }
    // unknown clusters running together are one word
    let mut ends = Vec::new();
    let mut k = n - 1;
    while k > 0 {
        ends.push(k);
        k = best[k].2;
    }
    ends.reverse();
    let mut out: Vec<usize> = Vec::new();
    let mut prev = 0;
    for &e in &ends {
        let known = words.contains(&run[at[prev]..at[e]]);
        let prev_known = out.last().map_or(true, |&o: &usize| {
            let s = out.len().checked_sub(2).map_or(0, |i| out[i]);
            words.contains(&run[s..o])
        });
        if !known && !prev_known && !out.is_empty() {
            *out.last_mut().unwrap() = at[e];
        } else {
            out.push(at[e]);
        }
        prev = e;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_are_found_in_each_script() {
        assert_eq!(words("ภาษาไทยง่ายนิดเดียว"), ["ภาษา", "ไทย", "ง่าย", "นิด", "เดียว"]);
        assert_eq!(words("สวัสดีครับ"), ["สวัสดี", "ครับ"]);
        // other text between: as it is
        assert_eq!(words("abc ภาษาไทย."), ["abc ", "ภาษา", "ไทย", "."]);
        assert_eq!(words("ພາສາລາວ").len(), 2);
        assert_eq!(words("ខ្ញុំស្រលាញ់អ្នក"), ["ខ្ញុំ", "ស្រលាញ់", "អ្នក"]);
        assert!(words("မြန်မာစာ").len() >= 2);
        // a word never ends inside a cluster
        for b in boundaries("ที่นี่น้ำ") {
            assert!(!"ที่นี่น้ำ"[b..].chars().next().is_some_and(continues));
        }
    }
}
