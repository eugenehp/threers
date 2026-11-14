//! Rotating, scaling and moving a vector shape (a [`NodeKind::Path`]): about the middle of its view
//! box, before it is fitted to the node's box. A spinner turns by
//! animating [`Property::Rotate`](crate::animation::Property::Rotate).
//!
//! The path data is rewritten with every point moved, in absolute
//! commands. Arcs stay exact under rotation, moving and even scaling; under
//! uneven scaling, their radii scale by each axis, an approximation.
//!
//! [`NodeKind::Path`]: crate::NodeKind::Path

use crate::tree::{NodeId, Ui};

/// A rotation (radians, clockwise on screen), a scale and a move, applied
/// in that order about a shape's middle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transformation {
    pub rotate: f32,
    pub scale: (f32, f32),
    pub translate: (f32, f32),
}

impl Default for Transformation {
    fn default() -> Self {
        Self {
            rotate: 0.0,
            scale: (1.0, 1.0),
            translate: (0.0, 0.0),
        }
    }
}

impl Transformation {
    pub fn rotate(radians: f32) -> Self {
        Self {
            rotate: radians,
            ..Self::default()
        }
    }

    pub fn scale(sx: f32, sy: f32) -> Self {
        Self {
            scale: (sx, sy),
            ..Self::default()
        }
    }

    pub fn with_translate(mut self, dx: f32, dy: f32) -> Self {
        self.translate = (dx, dy);
        self
    }

    pub fn is_identity(&self) -> bool {
        *self == Self::default()
    }

    /// Where `p` goes, turned and scaled about `centre`.
    pub fn apply(&self, (x, y): (f32, f32), (cx, cy): (f32, f32)) -> (f32, f32) {
        let (s, c) = self.rotate.sin_cos();
        let (dx, dy) = ((x - cx) * self.scale.0, (y - cy) * self.scale.1);
        (
            cx + dx * c - dy * s + self.translate.0,
            cy + dx * s + dy * c + self.translate.1,
        )
    }
}

impl Ui {
    /// Turns, scales or moves path node `id`'s shape (see the
    /// [module docs](self)); the identity takes it off.
    pub fn set_path_transform(&mut self, id: NodeId, t: Transformation) {
        let t = (!t.is_identity()).then_some(t);
        if self.node(id).transform != t {
            self.node_mut(id).transform = t;
            // Drawn moved; it takes the same room.
            self.touch_paint();
        }
    }

    pub fn path_transform(&self, id: NodeId) -> Transformation {
        self.node(id).transform.unwrap_or_default()
    }
}

/// SVG path data `d`, every point moved by `t` about `centre`, in absolute
/// commands. Data it cannot read is left as it was.
pub fn transform_path(d: &str, t: &Transformation, centre: (f32, f32)) -> String {
    let toks = tokens(d);
    let mut out = String::new();
    let (mut i, mut cmd) = (0, ' ');
    let (mut cur, mut start) = ((0.0f32, 0.0f32), (0.0f32, 0.0f32));
    let num = |i: &mut usize| -> Option<f32> {
        match toks.get(*i) {
            Some(Tok::Num(n)) => {
                *i += 1;
                Some(*n)
            }
            _ => None,
        }
    };
    let pt = |p: (f32, f32)| {
        let (x, y) = t.apply(p, centre);
        format!("{} {} ", fmt(x), fmt(y))
    };
    while i < toks.len() {
        if let Tok::Cmd(c) = toks[i] {
            cmd = c;
            i += 1;
        } else if cmd == ' ' {
            return d.to_owned();
        }
        let rel = cmd.is_ascii_lowercase();
        let base = if rel { cur } else { (0.0, 0.0) };
        let p =
            |i: &mut usize| -> Option<(f32, f32)> { Some((num(i)? + base.0, num(i)? + base.1)) };
        match cmd.to_ascii_uppercase() {
            'M' => {
                let Some(q) = p(&mut i) else {
                    return d.to_owned();
                };
                out += &format!("M{}", pt(q));
                cur = q;
                start = q;
                // More pairs after a move are lines.
                cmd = if rel { 'l' } else { 'L' };
            }
            'L' | 'T' => {
                let Some(q) = p(&mut i) else {
                    return d.to_owned();
                };
                out += &format!("{}{}", cmd.to_ascii_uppercase(), pt(q));
                cur = q;
            }
            'H' => {
                let Some(x) = num(&mut i) else {
                    return d.to_owned();
                };
                cur = (if rel { cur.0 + x } else { x }, cur.1);
                out += &format!("L{}", pt(cur));
            }
            'V' => {
                let Some(y) = num(&mut i) else {
                    return d.to_owned();
                };
                cur = (cur.0, if rel { cur.1 + y } else { y });
                out += &format!("L{}", pt(cur));
            }
            'C' => {
                let (Some(a), Some(b), Some(q)) = (p(&mut i), p(&mut i), p(&mut i)) else {
                    return d.to_owned();
                };
                out += &format!("C{}{}{}", pt(a), pt(b), pt(q));
                cur = q;
            }
            'S' | 'Q' => {
                let (Some(a), Some(q)) = (p(&mut i), p(&mut i)) else {
                    return d.to_owned();
                };
                out += &format!("{}{}{}", cmd.to_ascii_uppercase(), pt(a), pt(q));
                cur = q;
            }
            'A' => {
                let (Some(rx), Some(ry), Some(rot), Some(large), Some(sweep)) = (
                    num(&mut i),
                    num(&mut i),
                    num(&mut i),
                    num(&mut i),
                    num(&mut i),
                ) else {
                    return d.to_owned();
                };
                let Some(q) = p(&mut i) else {
                    return d.to_owned();
                };
                // A mirror (one scale negative) turns the sweep the other way.
                let mirrored = (t.scale.0 < 0.0) != (t.scale.1 < 0.0);
                let sweep = if mirrored { 1.0 - sweep } else { sweep };
                out += &format!(
                    "A{} {} {} {} {} {}",
                    fmt(rx * t.scale.0.abs()),
                    fmt(ry * t.scale.1.abs()),
                    fmt(rot + t.rotate.to_degrees()),
                    large as i32,
                    sweep as i32,
                    pt(q)
                );
                cur = q;
            }
            'Z' => {
                out += "Z ";
                cur = start;
            }
            _ => return d.to_owned(),
        }
    }
    out.trim_end().to_owned()
}

fn fmt(v: f32) -> String {
    let r = (v * 1000.0).round() / 1000.0;
    if r == r.trunc() {
        format!("{}", r as i64)
    } else {
        format!("{r}")
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Tok {
    Cmd(char),
    Num(f32),
}

/// Path data as commands and numbers (`1.5.5` is two numbers, `-` starts
/// one, as SVG reads them).
fn tokens(d: &str) -> Vec<Tok> {
    let mut out = vec![];
    let b = d.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let c = b[i] as char;
        if c.is_ascii_alphabetic() && c != 'e' && c != 'E' {
            out.push(Tok::Cmd(c));
            i += 1;
        } else if c.is_ascii_digit() || c == '-' || c == '+' || c == '.' {
            let s = i;
            i += 1;
            let mut dot = c == '.';
            while i < b.len() {
                let ch = b[i] as char;
                let prev = b[i - 1] as char;
                if ch.is_ascii_digit()
                    || ((ch == '-' || ch == '+') && (prev == 'e' || prev == 'E'))
                    || ch == 'e'
                    || ch == 'E'
                {
                    i += 1;
                } else if ch == '.' && !dot {
                    dot = true;
                    i += 1;
                } else {
                    break;
                }
            }
            if let Ok(n) = d[s..i].parse() {
                out.push(Tok::Num(n));
            }
        } else {
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn points_turn_about_the_middle_and_relative_commands_become_absolute() {
        let quarter = Transformation::rotate(std::f32::consts::FRAC_PI_2);
        // A line from the middle to the right turns to point down.
        assert_eq!(
            transform_path("M12 12 L20 12", &quarter, (12.0, 12.0)),
            "M12 12 L12 20"
        );
        // Relative, horizontal and closing commands.
        assert_eq!(
            transform_path(
                "m2 2h4v4h-4z",
                &Transformation::default().with_translate(1.0, 0.0),
                (0.0, 0.0)
            ),
            "M3 2 L7 2 L7 6 L3 6 Z"
        );
        // An arc: radii scaled, its axis turned, its end point moved.
        assert_eq!(
            transform_path(
                "M0 0 A5 5 0 0 1 10 0",
                &Transformation::scale(2.0, 2.0),
                (0.0, 0.0)
            ),
            "M0 0 A10 10 0 0 1 20 0"
        );
        // Packed numbers: `.5.5` is two.
        assert_eq!(
            transform_path("M.5.5L1-1", &Transformation::default(), (0.0, 0.0)),
            "M0.5 0.5 L1 -1"
        );
    }

    #[test]
    fn a_node_keeps_its_transform_and_the_identity_takes_it_off() {
        let mut ui = Ui::new();
        let root = ui.root();
        let p = ui.add_path(
            root,
            crate::Style::default(),
            vec!["M0 0L1 1".into()],
            [0.0, 0.0, 1.0, 1.0],
            Some(1.0),
            false,
        );
        ui.set_path_transform(p, Transformation::rotate(1.0));
        assert_eq!(ui.path_transform(p).rotate, 1.0);
        ui.set_path_transform(p, Transformation::default());
        assert!(ui.path_transform(p).is_identity());
    }
}
