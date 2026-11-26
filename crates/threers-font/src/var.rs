//! Variable fonts: an instance's axis values made the normalized
//! coordinates the tables' deltas are for (`fvar`, `avar`), and the deltas
//! of an item variation store (`HVAR`'s advances, GDEF's for GPOS).

use crate::sfnt::Reader;

/// Axis values (`wght` 650…) as the normalized coordinates of the face's
/// axes in `fvar`'s order, in -1..=1 at 2.14's precision, as HarfBuzz has
/// them. Axes not given are at their defaults (0).
pub fn normalize(fvar: &[u8], avar: Option<&[u8]>, values: &[([u8; 4], f32)]) -> Vec<f32> {
    let r = Reader::new(fvar);
    let (Ok(axes_at), Ok(count), Ok(size)) = (r.u16(4), r.u16(8), r.u16(10)) else { return Vec::new() };
    let fixed = |at: usize| r.u32(at).map(|v| v as i32 as f32 / 65536.0).unwrap_or(0.0);
    // each in 2.14 units, rounded before `avar` maps it (as HarfBuzz)
    let mut coords: Vec<i32> = (0..count as usize)
        .map(|k| {
            let at = axes_at as usize + k * size as usize;
            let Ok(tag) = r.slice(at, 4) else { return 0 };
            let Some(&(_, v)) = values.iter().find(|(t, _)| t == tag) else { return 0 };
            let (min, def, max) = (fixed(at + 4), fixed(at + 8), fixed(at + 12));
            let v = v.clamp(min.min(def), max.max(def));
            let n = if v == def {
                0.0
            } else if v < def {
                (v - def) / (def - min)
            } else {
                (v - def) / (max - def)
            };
            hb_round(n * 16384.0) as i32
        })
        .collect();

    // `avar`: each axis's coordinate mapped piece by piece
    if let Some(avar) = avar {
        let a = Reader::new(avar);
        let mut at = 8;
        for c in coords.iter_mut().take(a.u16(6).unwrap_or(0) as usize) {
            let n = a.u16(at).unwrap_or(0) as usize;
            let from = |k: usize| i32::from(a.i16(at + 2 + 4 * k).unwrap_or(0));
            let to = |k: usize| i32::from(a.i16(at + 4 + 4 * k).unwrap_or(0));
            let v = *c;
            *c = if n == 0 {
                v
            } else if n == 1 || v <= from(0) {
                v - from(0) + to(0)
            } else {
                let mut i = 1;
                while i < n - 1 && v > from(i) {
                    i += 1;
                }
                if v >= from(i) {
                    v - from(i) + to(i)
                } else if from(i - 1) == from(i) {
                    to(i - 1)
                } else {
                    hb_round(to(i - 1) as f32 + (to(i) - to(i - 1)) as f32 * (v - from(i - 1)) as f32 / (from(i) - from(i - 1)) as f32) as i32
                }
            };
            at += 2 + 4 * n;
        }
    }

    coords.into_iter().map(|c| c.clamp(-16384, 16384) as f32 / 16384.0).collect()
}

/// Rounded as HarfBuzz rounds (halves up, not away from zero).
pub fn hb_round(x: f32) -> f32 {
    (x + 0.5).floor()
}

/// The delta of item (`outer`, `inner`) of the item variation store at
/// `store` in `r`, at `coords`.
pub(crate) fn delta(r: Reader, store: usize, outer: u16, inner: u16, coords: &[f32]) -> f32 {
    if coords.iter().all(|c| *c == 0.0) {
        return 0.0;
    }
    let Ok(regions) = r.u32(store + 2) else { return 0.0 };
    let regions = store + regions as usize;
    if outer >= r.u16(store + 6).unwrap_or(0) {
        return 0.0;
    }
    let Ok(data) = r.u32(store + 8 + 4 * outer as usize) else { return 0.0 };
    let data = store + data as usize;
    let (Ok(items), Ok(words), Ok(n)) = (r.u16(data), r.u16(data + 2), r.u16(data + 4)) else { return 0.0 };
    if inner >= items {
        return 0.0;
    }
    let long = words & 0x8000 != 0;
    let words = (words & 0x7FFF) as usize;
    let n = n as usize;
    let (big, small) = if long { (4, 2) } else { (2, 1) };
    let row = data + 6 + 2 * n + inner as usize * (words * big + (n - words.min(n)) * small);

    let axes = r.u16(regions).unwrap_or(0) as usize;
    let mut total = 0.0;
    let mut at = row;
    for k in 0..n {
        let d = match (k < words, long) {
            (true, true) => r.u32(at).map(|v| v as i32).unwrap_or(0),
            (true, false) | (false, true) => i32::from(r.i16(at).unwrap_or(0)),
            (false, false) => i32::from(r.slice(at, 1).map(|b| b[0] as i8).unwrap_or(0)),
        };
        at += if k < words { big } else { small };
        if d == 0 {
            continue;
        }
        let region = r.u16(data + 6 + 2 * k).unwrap_or(0) as usize;
        let reg = regions + 4 + region * axes * 6;
        // in 2.14 units, as HarfBuzz reckons (its float rounding with it)
        let mut scalar = 1.0f32;
        for (i, &c) in coords.iter().enumerate().take(axes) {
            let c = (c * 16384.0).round() as i32;
            let at = reg + 6 * i;
            let (s, p, e) = (i32::from(r.i16(at).unwrap_or(0)), i32::from(r.i16(at + 2).unwrap_or(0)), i32::from(r.i16(at + 4).unwrap_or(0)));
            if p == 0 || c == p || s > p || p > e || (s < 0 && e > 0) {
                continue;
            }
            if c <= s || c >= e {
                scalar = 0.0;
                break;
            }
            scalar *= if c < p { (c - s) as f32 / (p - s) as f32 } else { (e - c) as f32 / (e - p) as f32 };
        }
        total += scalar * d as f32;
    }
    total
}

/// `HVAR`: how much an instance's advance of a glyph differs from the
/// default instance's.
pub fn advance_delta(hvar: &[u8], gid: u16, coords: &[f32]) -> f32 {
    let r = Reader::new(hvar);
    let Ok(store) = r.u32(4) else { return 0.0 };
    let map = r.u32(8).unwrap_or(0) as usize;
    let (outer, inner) = if map == 0 {
        (0, gid)
    } else {
        let format = r.slice(map, 1).map(|b| b[0]).unwrap_or(0);
        let entry = r.slice(map + 1, 1).map(|b| b[0]).unwrap_or(0);
        let (count, first) = if format == 0 { (r.u16(map + 2).unwrap_or(0) as usize, map + 4) } else { (r.u32(map + 2).unwrap_or(0) as usize, map + 6) };
        if count == 0 {
            return 0.0;
        }
        let size = ((entry >> 4) & 3) as usize + 1;
        let bits = (entry & 0xF) as u32 + 1;
        let at = first + size * (gid as usize).min(count - 1);
        let v = (0..size).fold(0u32, |v, k| v << 8 | u32::from(r.slice(at + k, 1).map(|b| b[0]).unwrap_or(0)));
        ((v >> bits) as u16, (v & ((1 << bits) - 1)) as u16)
    };
    delta(r, store as usize, outer, inner, coords)
}
