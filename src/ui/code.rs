//! Code folding as CodeMirror's `brace-fold` does it: a line's last `{` or
//! `[` outside strings opens a range that its match, on a later line,
//! closes. Folded, the line shows up to the bracket, a marker, and the rest
//! of the closing line.

/// A foldable range: the bracket's line and column, and the matching
/// bracket's line and column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fold {
    pub start_line: usize,
    pub start_col: usize,
    pub end_line: usize,
    pub end_col: usize,
}

/// The brackets of `text` outside strings: (line, column, char).
fn brackets(text: &str) -> Vec<(usize, usize, char)> {
    let mut out = vec![];
    for (line, l) in text.split('\n').enumerate() {
        let mut quote: Option<char> = None;
        let mut escaped = false;
        for (col, c) in l.chars().enumerate() {
            match quote {
                Some(q) => {
                    if escaped {
                        escaped = false;
                    } else if c == '\\' {
                        escaped = true;
                    } else if c == q {
                        quote = None;
                    }
                }
                None if c == '"' || c == '\'' => quote = Some(c),
                None if "{}[]".contains(c) => out.push((line, col, c)),
                None => {}
            }
        }
    }
    out
}

/// Every range `brace-fold` finds (`braceRangeFinder`): for each line, its
/// last opening bracket whose match is on a later line.
pub fn brace_folds(text: &str) -> Vec<Fold> {
    let all = brackets(text);
    // Matches by a stack.
    let mut matched = vec![None; all.len()];
    let mut stack: Vec<usize> = vec![];
    for (k, &(_, _, c)) in all.iter().enumerate() {
        match c {
            '{' | '[' => stack.push(k),
            _ => {
                if let Some(&open) = stack.last() {
                    if (all[open].2 == '{') == (c == '}') {
                        stack.pop();
                        matched[open] = Some(k);
                    }
                }
            }
        }
    }
    let mut out: Vec<Fold> = vec![];
    for (k, &(line, col, c)) in all.iter().enumerate() {
        if c != '{' && c != '[' {
            continue;
        }
        let Some(m) = matched[k] else { continue };
        let (end_line, end_col, _) = all[m];
        if end_line <= line {
            continue;
        }
        // The line's last such bracket wins.
        if let Some(last) = out.last_mut().filter(|f| f.start_line == line) {
            *last = Fold { start_line: line, start_col: col, end_line, end_col };
        } else {
            out.push(Fold { start_line: line, start_col: col, end_line, end_col });
        }
    }
    out
}

/// Which ranges are folded.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Folds {
    pub folded: Vec<Fold>,
}

impl Folds {
    /// `foldAll`: every range of `text`.
    pub fn fold_all(text: &str) -> Self {
        Self { folded: brace_folds(text) }
    }

    /// `unfold` at `line`: the range starting there opens.
    pub fn unfold(&mut self, line: usize) {
        self.folded.retain(|f| f.start_line != line);
    }

    /// Folds or unfolds the range starting at `line` (a gutter marker
    /// clicked), if `text` has one.
    pub fn toggle(&mut self, text: &str, line: usize) {
        if self.folded.iter().any(|f| f.start_line == line) {
            self.unfold(line);
        } else if let Some(f) = brace_folds(text).into_iter().find(|f| f.start_line == line) {
            self.folded.push(f);
        }
    }

    /// The outermost folded ranges, in order (those inside another are
    /// hidden with it).
    pub fn outermost(&self) -> Vec<Fold> {
        let mut v: Vec<Fold> = self.folded.clone();
        v.sort_by_key(|f| (f.start_line, f.start_col));
        let mut out: Vec<Fold> = vec![];
        for f in v {
            if out.last().is_some_and(|o| (f.start_line, f.start_col) < (o.end_line, o.end_col)) {
                continue;
            }
            out.push(f);
        }
        out
    }
}

/// One line as shown: the source line, and, if a range folds into it, the
/// fold (the line shows `[..=start_col]`, the marker, then the end line from
/// `end_col`).
#[derive(Debug, Clone, PartialEq)]
pub struct ShownLine {
    pub line: usize,
    pub fold: Option<Fold>,
}

/// The lines shown with `folds` applied.
pub fn shown_lines(text: &str, folds: &Folds) -> Vec<ShownLine> {
    let n = text.split('\n').count();
    let outer = folds.outermost();
    let mut out = vec![];
    let mut line = 0;
    while line < n {
        match outer.iter().find(|f| f.start_line == line) {
            Some(f) => {
                out.push(ShownLine { line, fold: Some(*f) });
                line = f.end_line + 1;
            }
            None => {
                out.push(ShownLine { line, fold: None });
                line += 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_as_brace_fold_does() {
        let text = "{\n  \"a\": {\n    \"x\": \"}\"\n  },\n  \"b\": [1, 2],\n  \"c\": [\n    3\n  ]\n}";
        let folds = brace_folds(text);
        assert_eq!(folds.len(), 3);
        assert_eq!(folds[1], Fold { start_line: 1, start_col: 7, end_line: 3, end_col: 2 });
        let mut f = Folds::fold_all(text);
        assert_eq!(shown_lines(text, &f), vec![ShownLine { line: 0, fold: Some(folds[0]) }]);
        f.unfold(0);
        let shown: Vec<usize> = shown_lines(text, &f).iter().map(|s| s.line).collect();
        assert_eq!(shown, vec![0, 1, 4, 5, 8]);
        f.toggle(text, 5);
        assert_eq!(shown_lines(text, &f).len(), 7);
    }
}
