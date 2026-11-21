//! Recorded swipes, one JSON object a line.
//!
//! Two shapes are read:
//!
//! - **This crate's**: `{"word": "hello", "points": [{"x": …, "y": …, "t": …}, …]}`,
//!   points in key units of the keyboard given to [`Recordings::load`], or
//!   of `"keys": [{"text": "q", "x": …, "y": …}, …]` on the line (any
//!   origin, one unit a key width), `t` in seconds, `"session"` optional.
//!   What [`Recording::to_json`] writes, so a keyboard can log its own.
//! - **FUTO's** (`futo-org/swipe.futo.org`, MIT, about 1 M English swipes):
//!   `{"word", "session", "canvas_width", "canvas_height", "data": [{"x", "y", "t"}], …}`,
//!   `x` and `y` fractions of the canvas, `t` in milliseconds, on the
//!   keyboard of their `layouts/qwerty.json` ([`FutoKeyboard`]). Swipe-5's
//!   two-finger lines are left out. [`alignment`] says how far swipes start
//!   and end from their first and last letters (the CLI's `check`): about
//!   half a key when the keyboard is right.
//!
//! Files are read a line at a time; a swipe keeps at most [`MAX_POINTS`]
//! points.

use std::collections::HashMap;
use std::io::BufRead;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::geometry::{Key, KeyboardGeometry};
use crate::trace::TouchPoint;

/// One recorded swipe.
#[derive(Debug, Clone, PartialEq)]
pub struct Recording {
    pub word: String,
    /// In the keyboard's key units.
    pub points: Vec<TouchPoint>,
    /// Index into [`Recordings::keyboards`].
    pub keyboard: usize,
    /// Who swiped it, so evaluation can hold out whole sessions.
    pub session: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct Line {
    word: String,
    points: Vec<TouchPoint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    session: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    keys: Option<Vec<LineKey>>,
}

#[derive(Serialize, Deserialize)]
struct LineKey {
    text: String,
    x: f32,
    y: f32,
}

impl Recording {
    /// A line of this crate's shape with `keyboard`'s keys on it, so the
    /// line stands alone.
    pub fn to_json(&self, keyboard: &KeyboardGeometry) -> String {
        let line = Line {
            word: self.word.clone(),
            points: self.points.clone(),
            session: self.session.clone(),
            keys: Some(
                keyboard
                    .keys
                    .iter()
                    .map(|k| LineKey {
                        text: k.text.clone(),
                        x: k.x,
                        y: k.y,
                    })
                    .collect(),
            ),
        };
        serde_json::to_string(&line).unwrap_or_default()
    }
}

/// Points kept of a recorded swipe: longer ones are resampled along their
/// path to this many (the model reads [`FRAMES`](crate::FRAMES)), which
/// keeps a million swipes near a gigabyte.
pub const MAX_POINTS: usize = 128;

/// The keyboard FUTO's swipes were taken on: key centres as fractions of
/// the canvas ([`from_layout_json`](Self::from_layout_json) reads their
/// `layouts/qwerty.json`), ten keys across it. How far apart the rows are
/// in key widths depends on the canvas's shape, `10 × height / width`,
/// which each line gives (`canvas_width`, `canvas_height`).
#[derive(Debug, Clone)]
pub struct FutoKeyboard {
    /// Letter and centre, as canvas fractions.
    pub keys: Vec<(String, f32, f32)>,
    /// Keys across the canvas.
    pub across: f32,
    /// `10 × height / width` for a line without a canvas size (a phone's
    /// portrait keyboard, 422 × 170).
    pub default_aspect: f32,
    /// Overrides every line's canvas shape.
    pub aspect: Option<f32>,
}

impl Default for FutoKeyboard {
    /// QWERTY as FUTO's layout file has it (rows at 1/6, 1/2, 5/6 of the
    /// height; the second row half a key in, the third one and a half).
    fn default() -> Self {
        let rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"];
        let lead = [0.0, 0.5, 1.5];
        let keys = rows
            .iter()
            .enumerate()
            .flat_map(|(r, row)| {
                row.chars().enumerate().map(move |(i, c)| {
                    (
                        c.to_string(),
                        (lead[r] + i as f32 + 0.5) / 10.0,
                        (r as f32 + 0.5) / 3.0,
                    )
                })
            })
            .collect();
        Self {
            keys,
            across: 10.0,
            default_aspect: 4.03,
            aspect: None,
        }
    }
}

#[derive(Deserialize)]
struct LayoutFile {
    keys: Vec<LayoutKey>,
}

#[derive(Deserialize)]
struct LayoutKey {
    letter: String,
    cx: f32,
    cy: f32,
    rx: f32,
}

impl FutoKeyboard {
    /// From one of FUTO's layout files (`swipe-5/layouts/*.json`).
    pub fn from_layout_json(text: &str) -> Result<Self, String> {
        let f: LayoutFile = serde_json::from_str(text).map_err(|e| format!("layout: {e}"))?;
        let rx = f.keys.first().map_or(0.05, |k| k.rx).max(1e-3);
        Ok(Self {
            keys: f.keys.into_iter().map(|k| (k.letter, k.cx, k.cy)).collect(),
            across: 1.0 / (2.0 * rx),
            ..Self::default()
        })
    }

    /// Its keys for a canvas of `aspect` (`10 × height / width`), in key
    /// widths; [`to_units`](KeyboardGeometry::to_units) takes
    /// [`raw`](Self::raw) points to them.
    pub fn geometry(&self, aspect: f32) -> KeyboardGeometry {
        let keys = self
            .keys
            .iter()
            .map(|(t, cx, cy)| {
                let (x, y) = self.raw(*cx, *cy, aspect);
                Key {
                    text: t.clone(),
                    x,
                    y,
                }
            })
            .collect();
        KeyboardGeometry::new("en", keys)
    }

    /// A canvas fraction in key widths, before centring.
    pub fn raw(&self, x: f32, y: f32, aspect: f32) -> (f32, f32) {
        (x * self.across, y * aspect * self.across / 10.0)
    }

    /// The canvas shape a line's size gives, rounded so that a handful of
    /// keyboards serve a million lines.
    fn aspect_of(&self, width: Option<f64>, height: Option<f64>) -> f32 {
        let a = match (self.aspect, width, height) {
            (Some(a), _, _) => a,
            (None, Some(w), Some(h)) if w > 1.0 && h > 1.0 => (10.0 * h / w) as f32,
            _ => self.default_aspect,
        };
        (a * 20.0).round() / 20.0
    }
}

/// A point as FUTO's lines have them.
#[derive(Deserialize)]
struct FutoPoint {
    x: f64,
    y: f64,
    t: f64,
}

/// `data`: one finger's path, or the two-finger shape of swipe-5 (left
/// out).
#[derive(Deserialize)]
#[serde(untagged)]
enum FutoData {
    Path(Vec<FutoPoint>),
    Other(serde::de::IgnoredAny),
}

/// Any line either shape may be.
#[derive(Deserialize)]
struct AnyLine {
    #[serde(default)]
    word: String,
    #[serde(default)]
    session: Option<String>,
    #[serde(default)]
    canvas_width: Option<f64>,
    #[serde(default)]
    canvas_height: Option<f64>,
    #[serde(default)]
    data: Option<FutoData>,
    #[serde(default)]
    points: Option<Vec<TouchPoint>>,
    #[serde(default)]
    keys: Option<Vec<LineKey>>,
}

/// Recorded swipes and the keyboards they were taken on.
#[derive(Debug, Clone, Default)]
pub struct Recordings {
    pub keyboards: Vec<KeyboardGeometry>,
    pub items: Vec<Recording>,
    /// Lines not read, and why (the first few).
    pub errors: Vec<String>,
    /// Lines left out: unreadable, empty, two-finger, or a letter with no
    /// key.
    pub skipped: usize,
}

impl Recordings {
    /// Reads `path` a line at a time, either shape per line. Lines of this
    /// crate's shape without keys are on `keyboard`.
    pub fn load(
        path: impl AsRef<Path>,
        keyboard: &KeyboardGeometry,
        futo: &FutoKeyboard,
    ) -> Result<Self, String> {
        let p = path.as_ref();
        let f = std::fs::File::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
        Self::read(
            std::io::BufReader::with_capacity(1 << 20, f),
            keyboard,
            futo,
        )
        .map_err(|e| format!("{}: {e}", p.display()))
    }

    pub fn parse(text: &str, keyboard: &KeyboardGeometry, futo: &FutoKeyboard) -> Self {
        Self::read(text.as_bytes(), keyboard, futo).unwrap_or_default()
    }

    /// Reads lines from `input` until it ends.
    pub fn read(
        input: impl BufRead,
        keyboard: &KeyboardGeometry,
        futo: &FutoKeyboard,
    ) -> Result<Self, String> {
        let mut out = Self {
            keyboards: vec![keyboard.clone()],
            ..Self::default()
        };
        // FUTO keyboards by rounded canvas shape.
        let mut shapes: HashMap<i32, usize> = HashMap::new();
        for (n, line) in input.lines().enumerate() {
            let line = line.map_err(|e| e.to_string())?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let l: AnyLine = match serde_json::from_str(line) {
                Ok(l) => l,
                Err(e) => {
                    out.fail(n, e.to_string());
                    continue;
                }
            };
            // FUTO's words keep their sentence's punctuation ("lava."); the
            // swipe is the letters.
            let word = l
                .word
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_owned();
            if word.is_empty() || word.chars().any(char::is_whitespace) {
                out.skipped += 1;
                continue;
            }
            match (l.data, l.points) {
                (Some(FutoData::Path(data)), _) => {
                    let aspect = futo.aspect_of(l.canvas_width, l.canvas_height);
                    let idx = *shapes
                        .entry((aspect * 20.0).round() as i32)
                        .or_insert_with(|| {
                            out.keyboards.push(futo.geometry(aspect));
                            out.keyboards.len() - 1
                        });
                    let g = &out.keyboards[idx];
                    let t0 = data.first().map_or(0.0, |p| p.t);
                    let points = data
                        .iter()
                        .map(|p| {
                            let (rx, ry) = futo.raw(p.x as f32, p.y as f32, aspect);
                            let (ux, uy) = g.to_units(rx, ry);
                            TouchPoint::new(ux, uy, ((p.t - t0) / 1000.0) as f32)
                        })
                        .collect();
                    out.push(word, points, idx, l.session);
                }
                (Some(FutoData::Other(_)), _) => out.skipped += 1,
                (None, Some(points)) => match l.keys {
                    Some(keys) => {
                        let g = KeyboardGeometry::new(
                            &keyboard.language,
                            keys.into_iter()
                                .map(|k| Key {
                                    text: k.text,
                                    x: k.x,
                                    y: k.y,
                                })
                                .collect(),
                        );
                        let points = points
                            .iter()
                            .map(|p| {
                                let (x, y) = g.to_units(p.x, p.y);
                                TouchPoint::new(x, y, p.t)
                            })
                            .collect();
                        out.keyboards.push(g);
                        let idx = out.keyboards.len() - 1;
                        out.push(word, points, idx, l.session);
                    }
                    None => out.push(word, points, 0, l.session),
                },
                (None, None) => out.fail(n, "neither \"points\" nor \"data\"".into()),
            }
        }
        Ok(out)
    }

    fn push(
        &mut self,
        word: String,
        points: Vec<TouchPoint>,
        keyboard: usize,
        session: Option<String>,
    ) {
        if points.is_empty() || self.keyboards[keyboard].word_keys(&word).is_none() {
            self.skipped += 1;
            return;
        }
        let points = if points.len() > MAX_POINTS {
            crate::trace::resample(&points, MAX_POINTS)
        } else {
            points
        };
        self.items.push(Recording {
            word,
            points,
            keyboard,
            session,
        });
    }

    fn fail(&mut self, line: usize, why: String) {
        self.skipped += 1;
        if self.errors.len() < 8 {
            self.errors.push(format!("line {}: {why}", line + 1));
        }
    }
}

/// Word counts from the `sentence` of each line of FUTO's files (each
/// sentence counted once, however many of its words were swiped), or
/// from a text file of one sentence a line: a lexicon. Only words of letters
/// (apostrophes inside allowed); lower case.
pub fn sentence_counts(path: impl AsRef<Path>) -> Result<Vec<(String, f32)>, String> {
    #[derive(Deserialize)]
    struct S {
        #[serde(default)]
        sentence: Option<String>,
    }
    let p = path.as_ref();
    let f = std::fs::File::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let mut seen = std::collections::HashSet::new();
    let mut counts: HashMap<String, f32> = HashMap::new();
    for line in std::io::BufReader::with_capacity(1 << 20, f).lines() {
        let line = line.map_err(|e| e.to_string())?;
        // A JSON line's `sentence`, or a plain line as it is.
        let s = match serde_json::from_str::<S>(&line) {
            Ok(S { sentence: Some(s) }) => s,
            Ok(_) => continue,
            Err(_) => line.trim().to_owned(),
        };
        if s.is_empty() {
            continue;
        }
        if !seen.insert(s.clone()) {
            continue;
        }
        for w in s.split(|c: char| c.is_whitespace() || c == '-' || c == '—') {
            let w = w.trim_matches(|c: char| !c.is_alphabetic()).to_lowercase();
            if !w.is_empty()
                && w.chars()
                    .all(|c| c.is_alphabetic() || c == '\'' || c == '’')
            {
                *counts.entry(w).or_default() += 1.0;
            }
        }
    }
    let mut out: Vec<(String, f32)> = counts.into_iter().collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    Ok(out)
}

/// How far swipes start from their word's first key and end from its
/// last, mean over `recs`, in key widths: around half a key when the
/// keyboard is where it is assumed to be, a key or more when it is not.
pub fn alignment(recs: &Recordings) -> (f32, f32) {
    let (mut s, mut e, mut n) = (0.0f32, 0.0f32, 0usize);
    for r in &recs.items {
        let g = &recs.keyboards[r.keyboard];
        let Some(keys) = g.word_keys(&r.word) else {
            continue;
        };
        let (Some(&k0), Some(&k1), Some(p0), Some(p1)) =
            (keys.first(), keys.last(), r.points.first(), r.points.last())
        else {
            continue;
        };
        let (a, b) = (&g.keys[k0], &g.keys[k1]);
        s += (p0.x - a.x).hypot(p0.y - a.y);
        e += (p1.x - b.x).hypot(p1.y - b.y);
        n += 1;
    }
    let n = n.max(1) as f32;
    (s / n, e / n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_words_of_json_and_plain_sentences() {
        let dir = std::env::temp_dir().join(format!("threers-swipe-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("s.txt");
        std::fs::write(
            &p,
            "{\"sentence\":\"The cat sat.\"}\n{\"sentence\":\"The cat sat.\"}\nThe dog's well-known 42 tricks!\n",
        )
        .unwrap();
        let c: HashMap<String, f32> = sentence_counts(&p).unwrap().into_iter().collect();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(c["the"], 2.0, "a repeated sentence counts once");
        assert_eq!(c["dog's"], 1.0);
        assert!(c.contains_key("well") && c.contains_key("known"));
        assert!(!c.contains_key("42") && !c.contains_key("sat."));
    }

    #[test]
    fn reads_both_shapes() {
        let kb = FutoKeyboard::default();
        let g = kb.geometry(4.0);
        // "qp": from q's centre to p's, as canvas fractions.
        let futo = r#"{"word":"qp,","session":"s1","canvas_width":400,"canvas_height":160,"data":[{"x":0.05,"y":0.1666,"t":1000},{"x":0.95,"y":0.1666,"t":1250}]}"#;
        let ours =
            r#"{"word":"ok","points":[{"x":0.0,"y":0.0,"t":0.0},{"x":1.0,"y":0.5,"t":0.2}]}"#;
        let keyed = r#"{"word":"ab","points":[{"x":10.0,"y":5.0,"t":0.0},{"x":11.0,"y":5.0,"t":0.1}],"keys":[{"text":"a","x":10.0,"y":5.0},{"text":"b","x":11.0,"y":5.0}]}"#;
        let text =
            format!("{futo}\n{ours}\n{keyed}\nnot json\n{{\"word\":\"x y\",\"points\":[]}}\n");
        let r = Recordings::parse(&text, &g, &kb);
        assert_eq!(r.items.len(), 3, "{:?}", r.errors);
        assert_eq!(r.skipped, 2);
        let qp = &r.items[0];
        assert_eq!(qp.session.as_deref(), Some("s1"));
        assert_eq!(qp.word, "qp", "sentence punctuation dropped");
        assert!((qp.points[1].t - 0.25).abs() < 1e-6);
        let q = &g.keys[g.key_for('q').unwrap()];
        assert!((qp.points[0].x - q.x).abs() < 1e-3 && (qp.points[0].y - q.y).abs() < 1e-2);
        // Keys on the line: centred, points with them.
        let ab = &r.items[2];
        let kg = &r.keyboards[ab.keyboard];
        assert!((ab.points[0].x - kg.keys[0].x).abs() < 1e-6);
        let (s, e) = alignment(&Recordings {
            items: vec![qp.clone()],
            ..r.clone()
        });
        assert!(s < 0.02 && e < 0.02, "{s} {e}");
    }
}
