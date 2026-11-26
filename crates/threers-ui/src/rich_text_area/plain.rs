//! Plain text as a [`Document`]: a paragraph a line, laid out in a
//! fixed-width font (`char_width` by `line_height`), with undo. For simple
//! editors, and for trying the text area out.

use super::{rect, Document, Pos, Rect, Selection};

pub struct PlainText {
    paras: Vec<String>,
    sel: Selection,
    undo: Vec<(Vec<String>, Selection)>,
    redo: Vec<(Vec<String>, Selection)>,
    pub char_width: f32,
    pub line_height: f32,
}

impl PlainText {
    pub fn new(text: &str) -> PlainText {
        PlainText {
            paras: text.split('\n').map(str::to_string).collect(),
            sel: Selection::default(),
            undo: Vec::new(),
            redo: Vec::new(),
            char_width: 8.0,
            line_height: 16.0,
        }
    }

    pub fn text_of(&self, para: usize) -> &str {
        &self.paras[para]
    }

    /// The whole text, paragraphs a newline apart.
    pub fn string(&self) -> String {
        self.paras.join("\n")
    }

    fn x_of(&self, p: Pos) -> f32 {
        self.paras[p.para][..p.off.min(self.paras[p.para].len())].chars().count() as f32 * self.char_width
    }
}

impl Document for PlainText {
    fn paragraphs(&self) -> usize {
        self.paras.len()
    }

    fn text(&self, para: usize) -> String {
        self.paras.get(para).cloned().unwrap_or_default()
    }

    fn selection(&self) -> Selection {
        self.sel
    }

    fn set_selection(&mut self, s: Selection) {
        let clamp = |p: Pos, paras: &[String]| {
            let para = p.para.min(paras.len() - 1);
            let t = &paras[para];
            let mut off = p.off.min(t.len());
            while !t.is_char_boundary(off) {
                off -= 1;
            }
            Pos::new(para, off)
        };
        self.sel = Selection { anchor: clamp(s.anchor, &self.paras), head: clamp(s.head, &self.paras), upstream: s.upstream };
    }

    fn replace(&mut self, a: Pos, b: Pos, text: &str, _typing: bool) -> Pos {
        let (a, b) = if a <= b { (a, b) } else { (b, a) };
        let prefix = self.paras[a.para][..a.off].to_string();
        let suffix = self.paras[b.para][b.off..].to_string();
        let mut lines: Vec<String> = text.split('\n').map(str::to_string).collect();
        let last = lines.len() - 1;
        let end = Pos::new(a.para + last, if last == 0 { prefix.len() + lines[0].len() } else { lines[last].len() });
        lines[0] = prefix + &lines[0];
        lines[last].push_str(&suffix);
        self.paras.splice(a.para..=b.para, lines);
        self.redo.clear();
        end
    }

    fn hit(&self, x: f32, y: f32) -> (Pos, bool) {
        let para = ((y / self.line_height).floor().max(0.0) as usize).min(self.paras.len() - 1);
        let n = (x / self.char_width).round().max(0.0) as usize;
        let t = &self.paras[para];
        let off = t.char_indices().nth(n).map_or(t.len(), |(i, _)| i);
        (Pos::new(para, off), false)
    }

    fn caret_rect(&self, p: Pos, _upstream: bool) -> Rect {
        rect(self.x_of(p), p.para as f32 * self.line_height, 1.0, self.line_height)
    }

    fn selection_rects(&self, a: Pos, b: Pos) -> Vec<Rect> {
        (a.para..=b.para)
            .map(|p| {
                let x0 = if p == a.para { self.x_of(a) } else { 0.0 };
                let x1 = if p == b.para { self.x_of(b) } else { self.x_of(Pos::new(p, self.paras[p].len())) + self.char_width };
                rect(x0, p as f32 * self.line_height, (x1 - x0).max(0.0), self.line_height)
            })
            .collect()
    }

    fn line_ends(&self, p: Pos, _upstream: bool) -> (Pos, Pos) {
        (Pos::new(p.para, 0), Pos::new(p.para, self.paras[p.para].len()))
    }

    fn line_step(&self, p: Pos, _upstream: bool, down: bool, x: f32) -> Option<(Pos, bool)> {
        let para = if down { p.para + 1 } else { p.para.checked_sub(1)? };
        if para >= self.paras.len() {
            return None;
        }
        Some(self.hit(x, (para as f32 + 0.5) * self.line_height))
    }

    fn checkpoint(&mut self) {
        self.undo.push((self.paras.clone(), self.sel));
    }

    fn undo(&mut self) {
        if let Some((p, s)) = self.undo.pop() {
            self.redo.push((std::mem::replace(&mut self.paras, p), self.sel));
            self.sel = s;
        }
    }

    fn redo(&mut self) {
        if let Some((p, s)) = self.redo.pop() {
            self.undo.push((std::mem::replace(&mut self.paras, p), self.sel));
            self.sel = s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Command, Effect, Mods, Phase, Pointer, PointerKind, RichTextArea};
    use super::*;
    use crate::text_shortcuts::Platform;

    fn mods(cmd: bool, shift: bool, alt: bool) -> Mods {
        Mods { meta: cmd, shift, alt, control: false }
    }

    fn area() -> RichTextArea {
        RichTextArea::with_platform(Platform::Mac)
    }

    fn type_(a: &mut RichTextArea, d: &mut PlainText, s: &str) {
        for c in s.chars() {
            if c == '\n' {
                a.key(d, "Enter", Mods::default(), 0.0);
                let at = d.selection().head;
                let end = d.replace(at, at, "\n", true);
                d.set_selection(Selection::caret(end));
            } else {
                a.text(d, &c.to_string(), 0.0);
            }
        }
    }

    #[test]
    fn typing_corrects_capitalises_and_curls() {
        let mut d = PlainText::new("");
        let mut a = area();
        type_(&mut a, &mut d, "i think teh \"best\" -- isn't it? yes. ok ");
        assert_eq!(d.text_of(0), "I think the “best” — isn’t it? Yes. Ok ");
    }

    #[test]
    fn backspace_puts_a_correction_back() {
        let mut d = PlainText::new("");
        let mut a = area();
        type_(&mut a, &mut d, "We sat at teh ");
        assert_eq!(d.text_of(0), "We sat at the ");
        a.key(&mut d, "Backspace", Mods::default(), 0.0);
        assert_eq!(d.text_of(0), "We sat at teh");
        // and it stays as typed
        type_(&mut a, &mut d, " ");
        assert_eq!(d.text_of(0), "We sat at teh ");
    }

    #[test]
    fn keys_move_select_and_delete_by_word_and_line() {
        let mut d = PlainText::new("one two three\nfour");
        let mut a = area();
        a.select(&mut d, Pos::new(0, 13), Pos::new(0, 13));
        a.key(&mut d, "ArrowLeft", mods(false, true, true), 0.0);
        assert_eq!(a.selected_text(&d), "three");
        a.key(&mut d, "Backspace", Mods::default(), 0.0);
        assert_eq!(d.text_of(0), "one two ");
        a.key(&mut d, "Backspace", mods(false, false, true), 0.0);
        assert_eq!(d.text_of(0), "one ");
        a.key(&mut d, "ArrowDown", Mods::default(), 0.0);
        assert_eq!(d.selection().head, Pos::new(1, 4));
        a.key(&mut d, "ArrowLeft", mods(true, true, false), 0.0);
        assert_eq!(a.selected_text(&d), "four");
        let (taken, fx) = a.key(&mut d, "b", mods(true, false, false), 0.0);
        assert!(taken && fx.contains(&Effect::Command(Command::Bold)));
        // each deletion its own undo step
        a.key(&mut d, "z", mods(true, false, false), 0.0);
        assert_eq!(d.string(), "one two \nfour");
        a.key(&mut d, "z", mods(true, false, false), 0.0);
        assert_eq!(d.string(), "one two three\nfour");
    }

    #[test]
    fn a_word_typed_is_one_undo_step() {
        let mut d = PlainText::new("");
        let mut a = area();
        a.autocorrect = super::super::autocorrect::Autocorrect::off();
        type_(&mut a, &mut d, "alpha beta");
        a.key(&mut d, "z", mods(true, false, false), 0.0);
        assert_eq!(d.text_of(0), "alpha ");
        a.key(&mut d, "z", mods(true, false, false), 0.0);
        assert_eq!(d.text_of(0), "");
    }

    #[test]
    fn clicks_select_by_word_and_paragraph_and_drag_by_word() {
        let mut d = PlainText::new("alpha beta gamma");
        let mut a = area();
        let click = |a: &mut RichTextArea, d: &mut PlainText, x: f32, t: f64| {
            a.pointer(d, Pointer::new(PointerKind::Mouse, Phase::Down, x, 4.0, t));
            a.pointer(d, Pointer::new(PointerKind::Mouse, Phase::Up, x, 4.0, t));
        };
        click(&mut a, &mut d, 60.0, 0.0);
        assert!(d.selection().is_caret());
        click(&mut a, &mut d, 60.0, 0.1);
        assert_eq!(a.selected_text(&d), "beta");
        // the second press held and dragged: by words
        a.pointer(&mut d, Pointer::new(PointerKind::Mouse, Phase::Down, 60.0, 4.0, 2.0));
        a.pointer(&mut d, Pointer::new(PointerKind::Mouse, Phase::Up, 60.0, 4.0, 2.0));
        a.pointer(&mut d, Pointer::new(PointerKind::Mouse, Phase::Down, 60.0, 4.0, 2.1));
        a.pointer(&mut d, Pointer::new(PointerKind::Mouse, Phase::Move, 100.0, 4.0, 2.2));
        assert_eq!(a.selected_text(&d), "beta gamma");
        a.pointer(&mut d, Pointer::new(PointerKind::Mouse, Phase::Up, 100.0, 4.0, 2.3));
        click(&mut a, &mut d, 60.0, 5.0);
        click(&mut a, &mut d, 60.0, 5.1);
        click(&mut a, &mut d, 60.0, 5.2);
        assert_eq!(a.selected_text(&d), "alpha beta gamma");
    }

    #[test]
    fn the_selection_dragged_moves_and_a_click_in_it_places_the_caret() {
        let mut d = PlainText::new("alpha beta gamma");
        let mut a = area();
        for t in [0.0, 0.1] {
            a.pointer(&mut d, Pointer::new(PointerKind::Mouse, Phase::Down, 60.0, 4.0, t));
            a.pointer(&mut d, Pointer::new(PointerKind::Mouse, Phase::Up, 60.0, 4.0, t));
        }
        assert_eq!(a.selected_text(&d), "beta");
        // pressed in it and dragged to the end: moved there, still selected
        a.pointer(&mut d, Pointer::new(PointerKind::Mouse, Phase::Down, 64.0, 4.0, 2.0));
        a.pointer(&mut d, Pointer::new(PointerKind::Mouse, Phase::Move, 128.0, 4.0, 2.1));
        assert!(a.overlay(&d, 2.1, 0..1).caret.is_some(), "where it would go shown");
        a.pointer(&mut d, Pointer::new(PointerKind::Mouse, Phase::Up, 128.0, 4.0, 2.2));
        assert_eq!(d.text(0), "alpha  gammabeta");
        assert_eq!(a.selected_text(&d), "beta");
        // pressed in it and let go: just the caret there
        a.pointer(&mut d, Pointer::new(PointerKind::Mouse, Phase::Down, 108.0, 4.0, 5.0));
        a.pointer(&mut d, Pointer::new(PointerKind::Mouse, Phase::Up, 108.0, 4.0, 5.0));
        assert!(d.selection().is_caret());
    }

    #[test]
    fn a_finger_selects_with_handles_and_a_menu() {
        let mut d = PlainText::new("alpha beta gamma");
        let mut a = area();
        let tap = |a: &mut RichTextArea, d: &mut PlainText, x: f32, t: f64| {
            a.pointer(d, Pointer::new(PointerKind::Touch, Phase::Down, x, 8.0, t));
            a.pointer(d, Pointer::new(PointerKind::Touch, Phase::Up, x, 8.0, t + 0.05))
        };
        tap(&mut a, &mut d, 60.0, 0.0);
        tap(&mut a, &mut d, 60.0, 0.2);
        assert_eq!(a.selected_text(&d), "beta");
        let o = a.overlay(&d, 0.3, 0..1);
        assert_eq!(o.handles.len(), 2);
        let menu = o.menu.expect("a menu");
        let labels: Vec<&str> = menu.items.iter().map(|i| i.0.as_str()).collect();
        assert_eq!(labels, ["Cut", "Copy", "Paste", "Look Up"]);
        // drag the end handle to the end
        let (kx, ky) = o.handles[1].1;
        a.pointer(&mut d, Pointer::new(PointerKind::Touch, Phase::Down, kx, ky, 1.0));
        assert!(a.touch_captured());
        a.pointer(&mut d, Pointer::new(PointerKind::Touch, Phase::Move, 200.0, ky, 1.1));
        a.pointer(&mut d, Pointer::new(PointerKind::Touch, Phase::Up, 200.0, ky, 1.2));
        assert_eq!(a.selected_text(&d), "beta gamma");
        // Copy, from the menu
        let o = a.overlay(&d, 1.3, 0..1);
        let copy = o.menu.unwrap().items.iter().find(|i| i.0 == "Copy").unwrap().1;
        let (cx, cy) = (copy.x + copy.width / 2.0, copy.y + copy.height / 2.0);
        a.pointer(&mut d, Pointer::new(PointerKind::Touch, Phase::Down, cx, cy, 2.0));
        let fx = a.pointer(&mut d, Pointer::new(PointerKind::Touch, Phase::Up, cx, cy, 2.05));
        assert!(fx.contains(&Effect::Clipboard(super::super::Clip::Copy)));
        // a long press: the caret under a loupe
        a.pointer(&mut d, Pointer::new(PointerKind::Touch, Phase::Down, 20.0, 8.0, 3.0));
        a.tick(&mut d, 3.6);
        assert!(a.overlay(&d, 3.6, 0..1).loupe.is_some());
        a.pointer(&mut d, Pointer::new(PointerKind::Touch, Phase::Up, 20.0, 8.0, 3.7));
        assert!(d.selection().is_caret());
    }

    #[test]
    fn composing_text_is_replaced_until_kept() {
        let mut d = PlainText::new("ab");
        let mut a = area();
        a.select(&mut d, Pos::new(0, 2), Pos::new(0, 2));
        a.set_marked(&mut d, "n");
        a.set_marked(&mut d, "ni");
        assert_eq!(a.marked(), Some((Pos::new(0, 2), Pos::new(0, 4))));
        a.text(&mut d, "你", 0.0);
        assert_eq!(d.text_of(0), "ab你");
        assert_eq!(a.marked(), None);
    }

    #[test]
    fn find_and_replace() {
        let mut d = PlainText::new("cat dog\ncat");
        let mut a = area();
        assert!(a.find(&mut d, "CAT", false, false));
        assert!(a.find(&mut d, "cat", false, false));
        assert_eq!(d.selection().range().0, Pos::new(1, 0));
        assert_eq!(a.replace_all(&mut d, "cat", "fox", true), 2);
        assert_eq!(d.string(), "fox dog\nfox");
    }

    #[test]
    fn misspellings_are_marked() {
        let mut d = PlainText::new("a sentance with NASA and teh x2y");
        let mut a = area();
        a.select(&mut d, Pos::new(0, 0), Pos::new(0, 0));
        let o = a.overlay(&d, 0.0, 0..1);
        assert_eq!(o.squiggles.len(), 2);
    }
}
