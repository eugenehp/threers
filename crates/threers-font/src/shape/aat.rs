//! Apple's own tables, as HarfBuzz applies them. `morx`, whenever a face
//! has it (its GSUB then unused): each chain's subtables enabled by its
//! default flags, in order, each in logical or layout order as it says,
//! driven by their state machines — rearrangement, contextual, ligature,
//! noncontextual and insertion subtables; deleted glyphs kept, unseen, for
//! kerning to read, and taken out after positioning; a ligature's
//! components' clusters become one. `kerx`'s pairs (formats 0 and 2) and
//! attachments (format 4, by `ankr`'s anchors or given points); `kern`
//! version 1's contextual kerning (format 1, along the line and across it).

use super::*;

const DELETED: u16 = 0xFFFF;

/// An AAT lookup table (`morx`'s class tables and substitutions): the
/// value for a glyph, if it has one.
fn lookup(r: Reader, at: usize, glyph: u16) -> Option<u16> {
    let u = |o: usize| r.u16(o).ok();
    let g = glyph;
    match u(at)? {
        0 => u(at + 2 + 2 * g as usize),
        2 | 4 => {
            let format = u(at)?;
            let unit = u(at + 2)? as usize;
            let n = u(at + 4)? as usize;
            for k in 0..n {
                let seg = at + 12 + k * unit;
                let (last, first) = (u(seg)?, u(seg + 2)?);
                if last == 0xFFFF && first == 0xFFFF {
                    break;
                }
                if (first..=last).contains(&g) {
                    return if format == 2 { u(seg + 4) } else { u(at + u(seg + 4)? as usize + 2 * (g - first) as usize) };
                }
            }
            None
        }
        6 => {
            let unit = u(at + 2)? as usize;
            let n = u(at + 4)? as usize;
            (0..n).map(|k| at + 12 + k * unit).find(|&e| u(e) == Some(g)).and_then(|e| u(e + 2))
        }
        8 => {
            let first = u(at + 2)?;
            let n = u(at + 4)?;
            (g >= first && g - first < n).then(|| u(at + 6 + 2 * (g - first) as usize)).flatten()
        }
        10 => {
            let size = u(at + 2)? as usize;
            let first = u(at + 4)?;
            let n = u(at + 6)?;
            if g < first || g - first >= n {
                return None;
            }
            let v = at + 8 + size * (g - first) as usize;
            match size {
                1 => r.slice(v, 1).ok().map(|b| u16::from(b[0])),
                2 => u(v),
                _ => r.u32(v).ok().map(|x| x as u16),
            }
        }
        _ => None,
    }
}

/// An extended state table's parts: classes, the state array, the entries.
struct States {
    classes: u32,
    class_table: usize,
    states: usize,
    entries: usize,
}

impl States {
    fn read(r: Reader, at: usize) -> Option<States> {
        Some(States {
            classes: r.u32(at).ok()?,
            class_table: at + r.u32(at + 4).ok()? as usize,
            states: at + r.u32(at + 8).ok()? as usize,
            entries: at + r.u32(at + 12).ok()? as usize,
        })
    }

    /// A glyph's class: 0 end of text, 1 out of bounds, 2 deleted.
    fn class(&self, r: Reader, glyph: Option<u16>) -> u32 {
        match glyph {
            None => 0,
            Some(DELETED) => 2,
            Some(g) => lookup(r, self.class_table, g).map_or(1, u32::from),
        }
    }

    /// The entry for a state and class: its index.
    fn entry(&self, r: Reader, state: u16, class: u32) -> u16 {
        let class = if class >= self.classes { 1 } else { class };
        r.u16(self.states + 2 * (state as usize * self.classes as usize + class as usize)).unwrap_or(0)
    }
}

/// The face's `morx` applied to the glyphs, in logical order (deleted ones
/// left hidden); `rtl` for right-to-left text.
pub(super) const DELETED_GLYPH: u16 = DELETED;

pub(super) fn morx(face: &Face, buf: &mut Vec<Info>, rtl: bool) {
    morx_with(face, buf, rtl, false, &[]);
}

/// [`morx`] for text set horizontally or vertically (each subtable only
/// where it says it applies), with features asked for (type, setting) as
/// HarfBuzz maps them onto each chain's flags: vertical forms (4, 0) for
/// text set vertically.
pub(super) fn morx_with(face: &Face, buf: &mut Vec<Info>, rtl: bool, vertical: bool, features: &[(u16, u16)]) {
    let Some(table) = face.table(b"morx") else { return };
    let r = Reader::new(table);
    let chains = r.u32(4).unwrap_or(0) as usize;
    let mut at = 8;
    for _ in 0..chains {
        let Ok(mut flags) = r.u32(at) else { break };
        let len = r.u32(at + 4).unwrap_or(0) as usize;
        let entries = r.u32(at + 8).unwrap_or(0) as usize;
        let subtables = r.u32(at + 12).unwrap_or(0) as usize;
        // the features asked for: their flags turned off, then on
        for k in 0..entries {
            let e = at + 16 + 12 * k;
            let (t, setting) = (r.u16(e).unwrap_or(0), r.u16(e + 2).unwrap_or(0));
            if features.contains(&(t, setting)) {
                flags &= r.u32(e + 8).unwrap_or(u32::MAX);
                flags |= r.u32(e + 4).unwrap_or(0);
            }
        }
        let mut st = at + 16 + 12 * entries;
        for _ in 0..subtables {
            let Ok(slen) = r.u32(st) else { break };
            let coverage = r.u32(st + 4).unwrap_or(0);
            let sub_flags = r.u32(st + 8).unwrap_or(0);
            // the subtables for the text's orientation (or for any)
            let fits = coverage & 0x2000_0000 != 0 || (coverage & 0x8000_0000 != 0) == vertical;
            if sub_flags & flags != 0 && fits {
                let body = st + 12;
                // bit 28: in logical order (else in layout order, which
                // right-to-left text has backwards); bit 30: reversed
                let reversed = coverage & 0x4000_0000 != 0;
                let backwards = if coverage & 0x1000_0000 != 0 { reversed } else { reversed != rtl };
                if backwards {
                    buf.reverse();
                }
                match coverage & 0xFF {
                    0 => rearrangement(r, body, buf),
                    1 => contextual(r, body, buf),
                    2 => ligature(r, body, buf),
                    4 => noncontextual(r, body, buf),
                    5 => insertion(r, body, buf),
                    _ => {}
                }
                if backwards {
                    buf.reverse();
                }
            }
            st += slen as usize;
        }
        at += len;
    }
    // deleted glyphs stay (unseen, no advance) for kerning to read, as
    // HarfBuzz keeps them; positioning takes them out
    for g in buf.iter_mut() {
        if g.glyph == DELETED {
            g.hidden = true;
        }
    }
}

/// The state machine over the glyphs (`size`-byte entries): `act` at each,
/// with the entry's index, its flags and the place, saying how many glyphs
/// it put in before the place; it moves on unless the entry says not to
/// (but never forever).
fn drive(r: Reader, s: &States, size: usize, buf: &mut Vec<Info>, mut act: impl FnMut(&mut Vec<Info>, usize, u16, usize) -> usize) {
    let mut state = 0u16;
    let mut i = 0;
    let mut stuck = 0;
    loop {
        let glyph = buf.get(i).map(|g| g.glyph);
        let class = s.class(r, glyph);
        let e = s.entry(r, state, class) as usize;
        let new_state = r.u16(s.entries + size * e).unwrap_or(0);
        let flags = r.u16(s.entries + size * e + 2).unwrap_or(0);
        i += act(buf, e, flags, i);
        state = new_state;
        if glyph.is_none() {
            break;
        }
        if flags & 0x4000 != 0 && stuck < 64 {
            stuck += 1;
        } else {
            stuck = 0;
            i += 1;
        }
    }
}

fn rearrangement(r: Reader, at: usize, buf: &mut Vec<Info>) {
    let Some(s) = States::read(r, at) else { return };
    let (mut first, mut last) = (0usize, 0usize);
    {
        drive(r, &s, 4, buf, |buf, _e, flags, i| {
            if flags & 0x8000 != 0 {
                first = i;
            }
            if flags & 0x2000 != 0 {
                last = (i + 1).min(buf.len());
            }
            let verb = flags & 0xF;
            if verb != 0 && first < last && last <= buf.len() {
                rearrange(&mut buf[first..last], verb);
                merge(buf, first, last);
            }
            0
        })
    }
}

/// The sixteen rearrangements: how many glyphs move from the start and
/// the end, and whether each pair is reversed.
fn rearrange(g: &mut [Info], verb: u16) {
    // (from the start, reversed?, from the end, reversed?)
    const V: [(usize, bool, usize, bool); 16] = [
        (0, false, 0, false),
        (1, false, 0, false),
        (0, false, 1, false),
        (1, false, 1, false),
        (2, false, 0, false),
        (2, true, 0, false),
        (0, false, 2, false),
        (0, false, 2, true),
        (1, false, 2, false),
        (1, false, 2, true),
        (2, false, 1, false),
        (2, true, 1, false),
        (2, false, 2, false),
        (2, true, 2, false),
        (2, false, 2, true),
        (2, true, 2, true),
    ];
    let (l, rl, rr, rrv) = V[verb as usize];
    let n = g.len();
    if l + rr > n {
        return;
    }
    let mut a: Vec<Info> = g[..l].to_vec();
    let mut b: Vec<Info> = g[n - rr..].to_vec();
    let mid: Vec<Info> = g[l..n - rr].to_vec();
    if rl {
        a.reverse();
    }
    if rrv {
        b.reverse();
    }
    // the end's glyphs to the start, the start's to the end
    let out: Vec<Info> = b.into_iter().chain(mid).chain(a).collect();
    g.copy_from_slice(&out);
}

fn contextual(r: Reader, at: usize, buf: &mut Vec<Info>) {
    let Some(s) = States::read(r, at) else { return };
    let Ok(subs) = r.u32(at + 16) else { return };
    let subs = at + subs as usize;
    let mut mark: Option<usize> = None;
    let entries = s.entries;
    {
        drive(r, &s, 8, buf, |buf, e, flags, i| {
            let rec = entries + 8 * e;
            let mark_index = r.u16(rec + 4).unwrap_or(0xFFFF);
            let current_index = r.u16(rec + 6).unwrap_or(0xFFFF);
            let sub = |index: u16, glyph: u16| -> Option<u16> {
                let off = r.u32(subs + 4 * index as usize).ok()? as usize;
                lookup(r, subs + off, glyph)
            };
            if mark_index != 0xFFFF
                && let Some(m) = mark
                && m < buf.len()
                && let Some(g) = sub(mark_index, buf[m].glyph)
            {
                buf[m].glyph = g;
            }
            if current_index != 0xFFFF {
                // at the text's end: the last glyph
                let c = if i < buf.len() { Some(i) } else { buf.len().checked_sub(1) };
                if let Some(c) = c
                    && let Some(g) = sub(current_index, buf[c].glyph)
                {
                    buf[c].glyph = g;
                }
            }
            if flags & 0x8000 != 0 {
                mark = Some(i.min(buf.len().saturating_sub(1)));
            }
            0
        })
    }
}

fn ligature(r: Reader, at: usize, buf: &mut Vec<Info>) {
    let Some(s) = States::read(r, at) else { return };
    let (Ok(actions), Ok(components), Ok(ligatures)) = (r.u32(at + 16), r.u32(at + 20), r.u32(at + 24)) else { return };
    let (actions, components, ligatures) = (at + actions as usize, at + components as usize, at + ligatures as usize);
    let entries = s.entries;
    let mut stack: Vec<usize> = Vec::new();
    {
        drive(r, &s, 6, buf, |buf, e, flags, i| {
            if flags & 0x8000 != 0 && i < buf.len() {
                if stack.last() == Some(&i) {
                    stack.pop();
                }
                stack.push(i);
                if stack.len() > 64 {
                    stack.remove(0);
                }
            }
            if flags & 0x2000 == 0 || stack.is_empty() || i >= buf.len() {
                return 0;
            }
            let mut action_at = actions + 4 * r.u16(entries + 6 * e + 4).unwrap_or(0) as usize;
            let mut cursor = stack.len();
            let mut index: u32 = 0;
            loop {
                if cursor == 0 {
                    stack.clear();
                    break;
                }
                cursor -= 1;
                let pos = stack[cursor];
                let Ok(action) = r.u32(action_at) else { break };
                let mut offset = action & 0x3FFF_FFFF;
                if offset & 0x2000_0000 != 0 {
                    offset |= 0xC000_0000;
                }
                let component = (u32::from(buf[pos].glyph)).wrapping_add(offset) as usize;
                index = index.wrapping_add(u32::from(r.u16(components + 2 * component).unwrap_or(0)));
                if action & 0xC000_0000 != 0 {
                    let lig = r.u16(ligatures + 2 * index as usize).unwrap_or(buf[pos].glyph);
                    buf[pos].glyph = lig;
                    buf[pos].ligated = true;
                    // the components after it go, their clusters with it
                    let end = stack.last().copied().unwrap_or(pos);
                    merge(buf, pos, end + 1);
                    while stack.len() - 1 > cursor {
                        let p = stack.pop().unwrap();
                        buf[p].glyph = DELETED;
                    }
                    index = 0;
                }
                action_at += 4;
                if action & 0x8000_0000 != 0 {
                    break;
                }
            }
            0
        })
    }
}

fn noncontextual(r: Reader, at: usize, buf: &mut [Info]) {
    for g in buf.iter_mut() {
        if g.glyph != DELETED
            && let Some(n) = lookup(r, at, g.glyph)
        {
            g.glyph = n;
        }
    }
}

fn insertion(r: Reader, at: usize, buf: &mut Vec<Info>) {
    let Some(s) = States::read(r, at) else { return };
    let Ok(glyphs) = r.u32(at + 16) else { return };
    let glyphs = at + glyphs as usize;
    let entries = s.entries;
    let mut mark: Option<usize> = None;
    drive(r, &s, 8, buf, |buf, e, flags, i| {
        let rec = entries + 8 * e;
        let current = r.u16(rec + 4).unwrap_or(0xFFFF);
        let marked = r.u16(rec + 6).unwrap_or(0xFFFF);
        // put in before the glyph at `at`, or after the one before it: copies
        // of that glyph (its cluster), as HarfBuzz makes them
        let insert = |buf: &mut Vec<Info>, index: u16, count: usize, at: usize, before: bool| -> usize {
            let at = at.min(buf.len());
            let from = if before { at } else { at.saturating_sub(1) };
            let template = buf.get(from.min(buf.len().saturating_sub(1))).copied().unwrap_or_default();
            let new: Vec<Info> = (0..count).map(|k| Info { glyph: r.u16(glyphs + 2 * (index as usize + k)).unwrap_or(0), ..template }).collect();
            buf.splice(at..at, new);
            count
        };
        let mut moved = 0;
        if marked != 0xFFFF
            && let Some(m) = mark
        {
            let count = (flags & 0x1F) as usize;
            let before = flags & 0x0400 != 0;
            let place = if before { m } else { m + 1 };
            insert(buf, marked, count, place, before);
            if place <= i {
                moved += count;
            }
        }
        if current != 0xFFFF {
            let count = ((flags >> 5) & 0x1F) as usize;
            let before = flags & 0x0800 != 0;
            let i = i + moved;
            insert(buf, current, count, if before { i } else { i + 1 }, before);
            // on past what was put in (unless told to stay)
            if flags & 0x4000 == 0 || before {
                moved += count;
            }
        }
        if flags & 0x8000 != 0 {
            mark = Some(i + moved);
        }
        moved
    });
}

/// Apple's `kerx` table's kerning for a pair: its horizontal subtables of
/// format 0 (pairs) and 2 (classes), not cross-stream or variation ones.
/// `kerx`'s format 4 subtables (attachment, as HarfBuzz applies them):
/// for each glyph a state machine attaches to the glyph it marked before,
/// the two points that meet (from `ankr`'s anchors, or given in the
/// table): (glyph, the one it attaches to, x and y from its point to that
/// one's, in font units).
pub(super) fn kerx_attachments(face: &Face, buf: &[Info], rtl: bool) -> Vec<(usize, usize, i32, i32)> {
    let Some(table) = face.table(b"kerx") else { return Vec::new() };
    let r = Reader::new(table);
    let ankr = face.table(b"ankr").map(Reader::new);
    let n = r.u32(4).unwrap_or(0) as usize;
    let mut out = Vec::new();
    let mut at = 8;
    for _ in 0..n {
        let Ok(len) = r.u32(at) else { break };
        let coverage = r.u32(at + 4).unwrap_or(0);
        if coverage & 0xE000_0000 == 0 && coverage & 0xFF == 4 {
            let machine = at + 12;
            let flags = r.u32(machine + 16).unwrap_or(0);
            let action = flags >> 30;
            let data = machine + (flags & 0x00FF_FFFF) as usize;
            if let Some(states) = States::read(r, machine) {
                // in logical order, or reversed (its backwards bit against the text's direction)
                let reverse = (coverage & 0x1000_0000 != 0) != rtl;
                let order: Vec<usize> = if reverse { (0..buf.len()).rev().collect() } else { (0..buf.len()).collect() };
                let mut glyphs: Vec<Info> = order.iter().map(|&k| buf[k]).collect();
                let mut mark: Option<usize> = None;
                drive(r, &states, 6, &mut glyphs, |g, e, fl, i| {
                    let index = r.u16(states.entries + 6 * e + 4).unwrap_or(0xFFFF);
                    if let Some(m) = mark
                        && index != 0xFFFF
                        && i < g.len()
                    {
                        let (base, cur) = (g[m].glyph, g[i].glyph);
                        let point = |glyph: u16, k: u16| -> Option<(i32, i32)> {
                            let a = ankr?;
                            let lt = r_u32(a, 4)? as usize;
                            let dt = r_u32(a, 8)? as usize;
                            let off = lookup(a, lt, glyph)? as usize;
                            let anchors = dt + off;
                            if u32::from(k) >= a.u32(anchors).ok()? {
                                return None;
                            }
                            let p = anchors + 4 + 4 * k as usize;
                            Some((i32::from(a.i16(p).ok()?), i32::from(a.i16(p + 2).ok()?)))
                        };
                        let found = match action {
                            1 => {
                                let k = data + 4 * index as usize;
                                match (point(base, r.u16(k).unwrap_or(0)), point(cur, r.u16(k + 2).unwrap_or(0))) {
                                    (Some(b), Some(c)) => Some((b.0 - c.0, b.1 - c.1)),
                                    _ => None,
                                }
                            }
                            2 => {
                                let k = data + 8 * index as usize;
                                let v = |o: usize| i32::from(r.i16(k + o).unwrap_or(0));
                                Some((v(0) - v(4), v(2) - v(6)))
                            }
                            // control points of the outlines: not read
                            _ => None,
                        };
                        if let Some((dx, dy)) = found {
                            out.push((order[i], order[m], dx, dy));
                        }
                    }
                    if fl & 0x8000 != 0 {
                        mark = Some(i);
                    }
                    0
                });
            }
        }
        at += (len as usize).max(1);
    }
    out
}

/// Apple's `kern` table (version 1): its format 1 subtables, contextual
/// kerning by a state machine, as HarfBuzz applies them. Each action pops
/// the glyphs pushed on its stack and gives each a value: along the line
/// (to its advance and offset), or across it (cross-stream: up or down,
/// each glyph carrying the shift of the one before it, as a chain, until
/// a reset). Values in font units, scaled by `em`.
pub(super) fn kern_states(face: &Face, buf: &[Info], pos: &mut [GlyphPosition], rtl: bool, em: &dyn Fn(i32) -> i32) {
    let Some(table) = face.table(b"kern") else { return };
    let r = Reader::new(table);
    if r.u32(0).ok() != Some(0x0001_0000) {
        return;
    }
    let len = buf.len();
    let n = r.u32(4).unwrap_or(0) as usize;
    let mut at = 8;
    // cross-stream: every glyph chained to the one before it (in the text's direction)
    let mut chained: Option<Vec<bool>> = None;
    let mut across = false;
    for sub in 0..n {
        let Ok(sub_len) = r.u32(at) else { break };
        let coverage = r.u16(at + 4).unwrap_or(0);
        let cross = coverage & 0x4000 != 0;
        if coverage & 0xA000 == 0 && coverage & 0xFF == 1 {
            if cross && chained.is_none() {
                chained = Some(vec![true; len]);
            }
            let m = at + 8;
            let (classes, class_table, states, entries, values) = (
                r.u16(m).unwrap_or(0) as usize,
                m + r.u16(m + 2).unwrap_or(0) as usize,
                r.u16(m + 4).unwrap_or(0) as usize,
                m + r.u16(m + 6).unwrap_or(0) as usize,
                r.u16(m + 8).unwrap_or(0) as usize,
            );
            let _ = values;
            let first = r.u16(class_table).unwrap_or(0);
            let count = r.u16(class_table + 2).unwrap_or(0);
            let class_of = |g: Option<u16>| -> usize {
                match g {
                    None => 0,
                    Some(DELETED) => 2,
                    Some(g) if g >= first && g - first < count => r.slice(class_table + 4 + (g - first) as usize, 1).map_or(1, |b| b[0] as usize),
                    Some(_) => 1,
                }
            };
            // the glyphs in the order the subtable reads them (right-to-left text backwards)
            let order: Vec<usize> = if rtl { (0..len).rev().collect() } else { (0..len).collect() };
            let mut stack: Vec<usize> = Vec::new();
            let mut state = 0usize;
            let mut i = 0usize;
            let mut stuck = 0;
            loop {
                let glyph = order.get(i).map(|&k| buf[k].glyph);
                let class = class_of(glyph);
                let class = if class >= classes { 1 } else { class };
                let e = r.slice(m + states + state * classes + class, 1).map_or(0, |b| b[0] as usize);
                let entry = entries + 4 * e;
                let new_state = r.u16(entry).unwrap_or(0) as usize;
                let flags = r.u16(entry + 2).unwrap_or(0);
                if flags & 0x8000 != 0 {
                    if stack.len() < 8 {
                        stack.push(i);
                    } else {
                        stack.clear();
                    }
                }
                let action = (flags & 0x3FFF) as usize;
                // a value for each glyph on the stack must be in the
                // subtable (the table, for the last), else none is applied
                // (as HarfBuzz checks)
                let limit = if sub + 1 < n { at + sub_len as usize } else { r.data.len() };
                if action != 0 && !stack.is_empty() && m + action + 2 * stack.len() > limit {
                    stack.clear();
                }
                if action != 0 && !stack.is_empty() {
                    let mut a = m + action;
                    let mut last = false;
                    while !last {
                        let Some(idx) = stack.pop() else { break };
                        let Ok(v) = r.i16(a) else { stack.clear(); break };
                        a += 2;
                        let Some(&k) = order.get(idx) else { continue };
                        last = v & 1 != 0;
                        let v = i32::from(v) & !1;
                        if cross {
                            let ch = chained.as_mut().unwrap();
                            if v == -0x8000 {
                                ch[k] = false;
                                pos[k].y_offset = 0;
                            } else if ch[k] {
                                pos[k].y_offset += em(v);
                                across |= v != 0;
                            }
                        } else {
                            pos[k].x_advance += em(v);
                            pos[k].x_offset += em(v);
                        }
                    }
                }
                state = new_state.saturating_sub(states) / classes.max(1);
                if glyph.is_none() {
                    break;
                }
                if flags & 0x4000 != 0 && stuck < 64 {
                    stuck += 1;
                } else {
                    stuck = 0;
                    i += 1;
                }
            }
        }
        at += (sub_len as usize).max(1);
    }

    if let Some(ch) = chained
        && across
    {
        carry_chain(&ch, rtl, pos);
    }
}

/// Cross-stream shifts carried along the chain every glyph is on (each
/// from the one before it in the text's direction, unless reset), as
/// HarfBuzz propagates attachments: at most 64 deep.
fn carry_chain(ch: &[bool], rtl: bool, pos: &mut [GlyphPosition]) {
    let len = ch.len().min(pos.len());
    let mut link: Vec<Option<usize>> = (0..len)
        .map(|i| {
            if !ch[i] {
                return None;
            }
            if rtl { (i + 1 < len).then_some(i + 1) } else { i.checked_sub(1) }
        })
        .collect();
    fn settle(i: usize, link: &mut [Option<usize>], pos: &mut [GlyphPosition], level: u32) {
        let Some(j) = link[i].take() else { return };
        if level == 0 {
            return;
        }
        settle(j, link, pos, level - 1);
        pos[i].y_offset += pos[j].y_offset;
    }
    for i in 0..len {
        settle(i, &mut link, pos, 64);
    }
}

/// `kerx`'s format 1 subtables: contextual kerning by a state machine, as
/// HarfBuzz applies them (unlike `kern`'s: the glyphs kept in eight places
/// round; an action's values, one per tuple, given to them from the latest
/// back until a -1).
pub(super) fn kerx_states(face: &Face, buf: &[Info], pos: &mut [GlyphPosition], rtl: bool, em: &dyn Fn(i32) -> i32) {
    let Some(table) = face.table(b"kerx") else { return };
    let r = Reader::new(table);
    let len = buf.len();
    let n = r.u32(4).unwrap_or(0) as usize;
    let mut at = 8;
    let mut chained: Option<Vec<bool>> = None;
    let mut across = false;
    for sub in 0..n {
        let Ok(sub_len) = r.u32(at) else { break };
        let coverage = r.u32(at + 4).unwrap_or(0);
        let tuples = (r.u32(at + 8).unwrap_or(0) as usize).max(1);
        let cross = coverage & 0x4000_0000 != 0;
        if coverage & 0xA000_0000 == 0 && coverage & 0xFF == 1 {
            if cross && chained.is_none() {
                chained = Some(vec![true; len]);
            }
            let machine = at + 12;
            if let Some(states) = States::read(r, machine) {
                let values = machine + r.u32(machine + 16).unwrap_or(0) as usize;
                let limit = if sub + 1 < n { at + sub_len as usize } else { r.data.len() };
                let reverse = (coverage & 0x1000_0000 != 0) != rtl;
                let order: Vec<usize> = if reverse { (0..len).rev().collect() } else { (0..len).collect() };
                let mut glyphs: Vec<Info> = order.iter().map(|&k| buf[k]).collect();
                // HarfBuzz's: eight places round, not emptied by an action
                // (CoreText ignores the reset flag)
                let mut slots: [Option<usize>; 8] = [None; 8];
                let mut top = 0usize;
                let mut depth = 0usize;
                drive(r, &states, 6, &mut glyphs, |_, e, flags, i| {
                    if flags & 0x8000 != 0 {
                        slots[top] = Some(i);
                        top = (top + 1) % 8;
                        depth = (depth + 1).min(8);
                    }
                    let index = r.u16(states.entries + 6 * e + 4).unwrap_or(0xFFFF);
                    if index != 0xFFFF && depth > 0 {
                        // values from the latest glyph back, till a -1
                        let mut a = values + 2 * index as usize;
                        let mut cursor = top;
                        for _ in 0..64 {
                            if a + 2 * tuples > limit {
                                depth = 0;
                                break;
                            }
                            let v = r.i16(a).unwrap_or(-1);
                            if v == -1 {
                                break;
                            }
                            a += 2 * tuples;
                            cursor = (cursor + 7) % 8;
                            let Some(k) = slots[cursor].and_then(|idx| order.get(idx).copied()) else { continue };
                            let v = i32::from(v);
                            if cross {
                                let ch = chained.as_mut().unwrap();
                                if v == -0x8000 {
                                    ch[k] = false;
                                    pos[k].y_offset = 0;
                                } else if ch[k] {
                                    pos[k].y_offset += em(v);
                                    across |= v != 0;
                                }
                            } else {
                                pos[k].x_advance += em(v);
                                pos[k].x_offset += em(v);
                            }
                        }
                    }
                    0
                });
            }
        }
        at += (sub_len as usize).max(1);
    }
    if let Some(ch) = chained
        && across
    {
        carry_chain(&ch, rtl, pos);
    }
}

/// Whether `kerx`'s pair subtables read the glyphs backwards.
pub(super) fn kerx_backwards(face: &Face) -> bool {
    let Some(t) = face.table(b"kerx") else { return false };
    let r = Reader::new(t);
    let n = r.u32(4).unwrap_or(0) as usize;
    let mut at = 8;
    for _ in 0..n {
        let (Ok(len), Ok(coverage)) = (r.u32(at), r.u32(at + 4)) else { break };
        if matches!(coverage & 0xFF, 0 | 2) && coverage & 0xE000_0000 == 0 {
            return coverage & 0x1000_0000 != 0;
        }
        at += (len as usize).max(1);
    }
    false
}

/// Whether Apple's `kern` table kerns by state machines (format 1).
pub(super) fn machine_kern(face: &Face) -> bool {
    let Some(t) = face.table(b"kern") else { return false };
    let r = Reader::new(t);
    if r.u32(0).ok() != Some(0x0001_0000) {
        return false;
    }
    let n = r.u32(4).unwrap_or(0) as usize;
    let mut at = 8;
    for _ in 0..n {
        let (Ok(len), Ok(coverage)) = (r.u32(at), r.u16(at + 4)) else { break };
        if coverage & 0xFF == 1 {
            return true;
        }
        at += (len as usize).max(1);
    }
    false
}

/// Whether the `kern` table moves glyphs across the line (a cross-stream subtable).
pub(super) fn cross_kern(face: &Face) -> bool {
    let Some(t) = face.table(b"kern") else { return false };
    let r = Reader::new(t);
    if r.u32(0).ok() == Some(0x0001_0000) {
        let n = r.u32(4).unwrap_or(0) as usize;
        let mut at = 8;
        for _ in 0..n {
            let (Ok(len), Ok(coverage)) = (r.u32(at), r.u16(at + 4)) else { break };
            if coverage & 0x4000 != 0 {
                return true;
            }
            at += (len as usize).max(1);
        }
        false
    } else {
        let n = r.u16(2).unwrap_or(0) as usize;
        let mut at = 4;
        for _ in 0..n {
            let (Ok(len), Ok(coverage)) = (r.u16(at + 2), r.u16(at + 4)) else { break };
            if coverage & 0x04 != 0 {
                return true;
            }
            at += (len as usize).max(1);
        }
        false
    }
}

fn r_u32(r: Reader, at: usize) -> Option<u32> {
    r.u32(at).ok()
}

pub(super) fn kerx_pair(face: &Face, left: u16, right: u16) -> Option<i32> {
    let r = Reader::new(face.table(b"kerx")?);
    let n = r.u32(4).ok()? as usize;
    let mut at = 8;
    let mut total = None;
    for _ in 0..n {
        let len = r.u32(at).ok()? as usize;
        let coverage = r.u32(at + 4).ok()?;
        if coverage & 0xE000_0000 == 0 {
            let v = match coverage & 0xFF {
                0 => {
                    let pairs = r.u32(at + 12).ok()? as usize;
                    let key = (u32::from(left) << 16) | u32::from(right);
                    let (mut lo, mut hi) = (0, pairs);
                    let mut found = None;
                    while lo < hi {
                        let mid = (lo + hi) / 2;
                        let rec = at + 28 + 6 * mid;
                        match r.u32(rec).ok()?.cmp(&key) {
                            std::cmp::Ordering::Equal => {
                                found = Some(i32::from(r.i16(rec + 4).ok()?));
                                break;
                            }
                            std::cmp::Ordering::Less => lo = mid + 1,
                            std::cmp::Ordering::Greater => hi = mid,
                        }
                    }
                    found
                }
                2 => {
                    let lt = at + r.u32(at + 16).ok()? as usize;
                    let rt = at + r.u32(at + 20).ok()? as usize;
                    let arr = at + r.u32(at + 24).ok()? as usize;
                    // only glyphs the class tables have kern (as HarfBuzz)
                    match (lookup(r, lt, left), lookup(r, rt, right)) {
                        (Some(l), Some(rc)) => r.i16(arr + l as usize + rc as usize).ok().map(i32::from).filter(|&v| v != 0),
                        _ => None,
                    }
                }
                _ => None,
            };
            if let Some(v) = v {
                total = Some(total.unwrap_or(0) + v);
            }
        }
        at += len.max(1);
    }
    total
}
