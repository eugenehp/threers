//! IGES — the Initial Graphics Exchange Specification (5.3) file.
//!
//! The exchange *syntax* only, as [`crate::step::part21`] is for STEP: a file
//! is a start section of free text, a global section of parameters, and a
//! list of entities, each a two-line directory entry and a run of
//! parameters. What a type 144 or a type 126 *means* is the business of
//! whatever maps it to geometry; nearly every interchange problem with IGES
//! is lexical (a Hollerith count one short, a real written `1.D-3`, a
//! parameter line continued past column 64), and those are found here,
//! against a round trip, without building anything.
//!
//! Only the fixed-length ASCII form is read and written; the compressed
//! (`C`) and binary forms are refused.
//!
//! Numbers are written by a formatter the caller passes in
//! ([`WriteOptions`]): systems disagree about how many digits a real gets
//! and whether trailing zeros stay, and a writer that must match another
//! one byte for byte needs to say so. The default writes the shortest text
//! that reads back as the same number.

use std::fmt;

/// One parameter of the global section or of an entity.
#[derive(Debug, Clone, PartialEq)]
pub enum Param {
    /// Nothing between two delimiters: the parameter takes its default.
    Void,
    Int(i64),
    Real(f64),
    /// A Hollerith string (`5HHello`), without its count.
    Text(String),
}

impl Param {
    /// An integer, accepting a real with no fraction (some writers put
    /// `2.` where an integer belongs).
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Param::Int(i) => Some(*i),
            Param::Real(r) if r.fract() == 0.0 && r.abs() < 1e15 => Some(*r as i64),
            _ => None,
        }
    }

    /// A number, accepting an integer where a real is expected.
    pub fn as_real(&self) -> Option<f64> {
        match self {
            Param::Real(r) => Some(*r),
            Param::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            Param::Text(t) => Some(t),
            _ => None,
        }
    }

    /// The entity a pointer parameter (a directory sequence number) names,
    /// as an index into [`IgesFile::entities`]; none for 0, a void or a
    /// pointer that is not an odd positive number.
    pub fn as_entity(&self) -> Option<usize> {
        de_index(self.as_int()?)
    }
}

/// The entity a directory pointer `2i + 1` names: index `i`.
pub fn de_index(pointer: i64) -> Option<usize> {
    (pointer > 0 && pointer % 2 == 1).then(|| (pointer as usize - 1) / 2)
}

/// The directory pointer of entity `index`.
pub fn de_pointer(index: usize) -> i64 {
    2 * index as i64 + 1
}

/// An entity's directory entry: its twenty fields over two lines.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Directory {
    pub entity_type: i64,
    pub structure: i64,
    pub line_font: i64,
    pub level: i64,
    pub view: i64,
    /// A transformation matrix (type 124) by directory pointer, or 0.
    pub transform: i64,
    pub label_display: i64,
    /// Blank, subordinate, use and hierarchy statuses.
    pub status: [u8; 4],
    pub line_weight: i64,
    pub color: i64,
    pub form: i64,
    pub label: String,
    pub subscript: Option<i64>,
}

impl Directory {
    pub fn new(entity_type: i64, form: i64) -> Self {
        Directory {
            entity_type,
            form,
            ..Default::default()
        }
    }

    /// Blanked (not to be shown)?
    pub fn blank(&self) -> bool {
        self.status[0] == 1
    }
}

/// One entity: its directory entry and its parameters, the type number
/// that starts the parameter data left out.
#[derive(Debug, Clone, PartialEq)]
pub struct Entity {
    pub directory: Directory,
    pub params: Vec<Param>,
}

/// A whole file.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct IgesFile {
    /// The start section's lines, free text.
    pub start: Vec<String>,
    /// The global section's parameters, the two delimiters first.
    pub global: Vec<Param>,
    pub entities: Vec<Entity>,
}

impl IgesFile {
    /// The parameter delimiter (global parameter 1), `,` unless set.
    pub fn delimiter(&self) -> char {
        delimiter_of(self.global.first(), ',')
    }

    /// The record delimiter (global parameter 2), `;` unless set.
    pub fn record_delimiter(&self) -> char {
        delimiter_of(self.global.get(1), ';')
    }

    /// Millimetres in one model unit, from the unit flag (global 14), or
    /// none for a flag this does not know.
    pub fn unit_mm(&self) -> Option<f64> {
        let flag = self.global.get(13).and_then(Param::as_int).unwrap_or(1);
        Some(match flag {
            1 => 25.4,
            2 | 3 => 1.0,
            4 => 304.8,
            5 => 1_609_344.0,
            6 => 1000.0,
            7 => 1_000_000.0,
            8 => 0.0254,
            9 => 0.001,
            10 => 10.0,
            11 => 0.000_025_4,
            _ => return None,
        })
    }

    /// The entity a pointer names.
    pub fn entity(&self, pointer: i64) -> Option<&Entity> {
        self.entities.get(de_index(pointer)?)
    }
}

fn delimiter_of(p: Option<&Param>, default: char) -> char {
    match p {
        Some(Param::Text(t)) if t.chars().count() == 1 => t.chars().next().unwrap(),
        _ => default,
    }
}

/// What made a file unreadable.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    /// A section letter other than S, G, D, P, T in column 73, or a line
    /// too short to carry one.
    Section { line: usize },
    /// The compressed or binary form.
    NotFixedFormat,
    /// A directory field that is not a number.
    Directory { entity: usize, field: &'static str },
    /// A parameter that does not lex: a bad number, a Hollerith string
    /// running past the end.
    Parameter { entity: Option<usize>, text: String },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::Section { line } => write!(f, "line {line}: not an IGES section line"),
            ParseError::NotFixedFormat => write!(f, "only fixed-format ASCII IGES is read"),
            ParseError::Directory { entity, field } => {
                write!(f, "directory entry {}: bad {field}", de_pointer(*entity))
            }
            ParseError::Parameter {
                entity: Some(e),
                text,
            } => {
                write!(f, "entity {}: bad parameter {text:?}", de_pointer(*e))
            }
            ParseError::Parameter { entity: None, text } => {
                write!(f, "global section: bad parameter {text:?}")
            }
        }
    }
}

impl std::error::Error for ParseError {}

/// A number as IGES writes it: an integer, or a real with an `E` or `D`
/// exponent, a point optional.
fn number(t: &str) -> Option<Param> {
    let t = t.trim();
    if t.is_empty() {
        return Some(Param::Void);
    }
    if let Ok(i) = t.parse::<i64>() {
        return Some(Param::Int(i));
    }
    let r = t.replace(['D', 'd'], "E");
    // Rust accepts "1.", ".5", "1.E3" and "1E3" alike.
    r.parse::<f64>().ok().map(Param::Real)
}

/// The parameters of `data` up to the record delimiter: Holleriths by
/// their count, everything else between delimiters.
fn params(
    data: &str,
    delim: char,
    record: char,
    entity: Option<usize>,
) -> Result<Vec<Param>, ParseError> {
    let chars: Vec<char> = data.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    let bad = |t: &str| ParseError::Parameter {
        entity,
        text: t.to_string(),
    };
    loop {
        // Skip leading blanks.
        while i < chars.len() && chars[i] == ' ' {
            i += 1;
        }
        let start = i;
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }
        if i > start && i < chars.len() && chars[i] == 'H' {
            let n: usize = chars[start..i]
                .iter()
                .collect::<String>()
                .parse()
                .map_err(|_| bad("H"))?;
            let from = i + 1;
            let to = from + n;
            if to > chars.len() {
                return Err(bad(&chars[start..].iter().collect::<String>()));
            }
            out.push(Param::Text(chars[from..to].iter().collect()));
            i = to;
            while i < chars.len() && chars[i] == ' ' {
                i += 1;
            }
        } else {
            while i < chars.len() && chars[i] != delim && chars[i] != record {
                i += 1;
            }
            let t: String = chars[start..i].iter().collect();
            out.push(number(&t).ok_or_else(|| bad(&t))?);
        }
        if i >= chars.len() || chars[i] == record {
            break;
        }
        i += 1;
    }
    Ok(out)
}

fn field(line: &str, k: usize) -> &str {
    line.get(8 * k..8 * k + 8).unwrap_or("").trim()
}

/// Read a fixed-format IGES file.
pub fn parse(text: &str) -> Result<IgesFile, ParseError> {
    let (mut s, mut g, mut d, mut p) = (Vec::new(), String::new(), Vec::new(), Vec::new());
    for (n, line) in text.lines().enumerate() {
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() {
            continue;
        }
        let section = line
            .get(72..73)
            .ok_or(ParseError::Section { line: n + 1 })?;
        let body = &line[..72];
        match section {
            "S" => s.push(body.trim_end().to_string()),
            "G" => g.push_str(body),
            "D" => d.push(body.to_string()),
            "P" => p.push(line.to_string()),
            "T" => {}
            "C" | "B" => return Err(ParseError::NotFixedFormat),
            _ => return Err(ParseError::Section { line: n + 1 }),
        }
    }
    // The global section names its delimiters in its first parameters.
    let (mut delim, mut record) = (',', ';');
    let gb: Vec<char> = g.chars().collect();
    if gb.len() >= 3 && gb[0] == '1' && gb[1] == 'H' {
        delim = gb[2];
        if gb.len() >= 7 && gb[3] == delim && gb[4] == '1' && gb[5] == 'H' {
            record = gb[6];
        }
    } else if gb.len() >= 4 && gb[0] == delim && gb[1] == '1' && gb[2] == 'H' {
        record = gb[3];
    }
    let global = params(&g, delim, record, None)?;
    // Parameter lines by the directory entry they belong to.
    let mut by_entity: std::collections::BTreeMap<usize, String> = Default::default();
    for line in &p {
        let pointer: i64 = line.get(64..72).unwrap_or("").trim().parse().unwrap_or(0);
        if let Some(i) = de_index(pointer) {
            by_entity
                .entry(i)
                .or_default()
                .push_str(line.get(..64).unwrap_or(line));
        }
    }
    let mut entities = Vec::with_capacity(d.len() / 2);
    for (i, pair) in d.chunks(2).enumerate() {
        let (a, b) = (&pair[0], pair.get(1).map(String::as_str).unwrap_or(""));
        let num = |s: &str, f: &'static str| -> Result<i64, ParseError> {
            if s.is_empty() {
                return Ok(0);
            }
            s.parse().map_err(|_| ParseError::Directory {
                entity: i,
                field: f,
            })
        };
        let status_text = field(a, 8);
        let status_text = format!("{status_text:0>8}");
        let st = |k: usize| {
            status_text[2 * k..2 * k + 2]
                .trim()
                .parse::<u8>()
                .unwrap_or(0)
        };
        let directory = Directory {
            entity_type: num(field(a, 0), "entity type")?,
            structure: num(field(a, 2), "structure")?,
            line_font: num(field(a, 3), "line font")?,
            level: num(field(a, 4), "level")?,
            view: num(field(a, 5), "view")?,
            transform: num(field(a, 6), "transformation")?,
            label_display: num(field(a, 7), "label display")?,
            status: [st(0), st(1), st(2), st(3)],
            line_weight: num(field(b, 1), "line weight")?,
            color: num(field(b, 2), "colour")?,
            form: num(field(b, 4), "form")?,
            label: field(b, 7).to_string(),
            subscript: field(b, 8).parse().ok(),
        };
        let data = by_entity.remove(&i).unwrap_or_default();
        let mut ps = params(&data, delim, record, Some(i))?;
        if !ps.is_empty() {
            ps.remove(0);
        }
        entities.push(Entity {
            directory,
            params: ps,
        });
    }
    Ok(IgesFile {
        start: s,
        global,
        entities,
    })
}

/// How numbers are written.
pub struct WriteOptions<'a> {
    /// A real in the parameter section.
    pub real: &'a dyn Fn(f64) -> String,
    /// A real in the global section, by its 0-based position there.
    pub global_real: &'a dyn Fn(usize, f64) -> String,
}

/// The shortest text that reads back as `v`, with a point.
pub fn shortest_real(v: f64) -> String {
    let s = format!("{v:?}");
    if s.contains(['.', 'e', 'E']) || !v.is_finite() {
        s.replace('e', "E")
    } else {
        format!("{s}.")
    }
}

impl Default for WriteOptions<'_> {
    fn default() -> Self {
        WriteOptions {
            real: &shortest_real,
            global_real: &|_, v| shortest_real(v),
        }
    }
}

/// Lines being filled to a width, a field never split from the delimiter
/// before it (`IGESData_IGESWriter`'s `AddString` / `AddChar`).
struct Lines {
    cur: String,
    max: usize,
    out: Vec<String>,
}

impl Lines {
    fn new(max: usize) -> Self {
        Lines {
            cur: String::new(),
            max,
            out: Vec::new(),
        }
    }
    fn flush(&mut self) {
        self.out.push(std::mem::take(&mut self.cur));
    }
    fn push(&mut self, s: &str) {
        let room = self.max - self.cur.len();
        self.cur.push_str(&s[..s.len().min(room)]);
    }
    fn string(&mut self, val: &str) {
        let mut ln = val.len();
        if self.cur.len() + ln + 1 > self.max {
            self.flush();
        }
        let mut n2 = 0;
        while ln > self.max {
            self.push(&val[n2..n2 + ln]);
            self.flush();
            n2 += self.max;
            ln -= self.max;
        }
        self.push(&val[n2..n2 + ln]);
    }
    fn char(&mut self, c: char) {
        if self.cur.len() + 1 > self.max {
            self.flush();
        }
        let mut b = [0u8; 4];
        self.push(c.encode_utf8(&mut b));
    }
    fn done(mut self) -> Vec<String> {
        if !self.cur.is_empty() {
            self.flush();
        }
        self.out
    }
}

fn hollerith(s: &str) -> String {
    if s.is_empty() {
        String::new()
    } else {
        format!("{}H{}", s.len(), s)
    }
}

/// Write a fixed-format IGES file. Entities are numbered in the order they
/// are listed; each one's parameter lines are counted into its directory
/// entry.
pub fn write(file: &IgesFile, options: &WriteOptions) -> String {
    let (delim, record) = (file.delimiter(), file.record_delimiter());
    let text = |p: &Param, real: &dyn Fn(f64) -> String| match p {
        Param::Void => String::new(),
        Param::Int(i) => i.to_string(),
        Param::Real(r) => real(*r),
        Param::Text(t) => hollerith(t),
    };
    let mut g = Lines::new(72);
    for (k, p) in file.global.iter().enumerate() {
        g.string(&text(p, &|v| (options.global_real)(k, v)));
        g.char(if k + 1 < file.global.len() {
            delim
        } else {
            record
        });
    }
    let g = g.done();
    let mut pstart = Vec::with_capacity(file.entities.len() + 1);
    let mut plines: Vec<String> = Vec::new();
    for e in &file.entities {
        let mut l = Lines::new(64);
        pstart.push(plines.len() + 1);
        l.string(&e.directory.entity_type.to_string());
        for p in &e.params {
            l.char(delim);
            l.string(&text(p, options.real));
        }
        l.char(record);
        plines.extend(l.done());
    }
    pstart.push(plines.len() + 1);

    let mut out = String::new();
    if file.start.is_empty() {
        out += &format!("{:72}S{:07}\n", "", 1);
    }
    for (k, line) in file.start.iter().enumerate() {
        out += &format!("{:72.72}S{:07}\n", line, k + 1);
    }
    for (k, line) in g.iter().enumerate() {
        out += &format!("{line:72}G{:07}\n", k + 1);
    }
    for (k, e) in file.entities.iter().enumerate() {
        let d = &e.directory;
        let s = d.status;
        out += &format!(
            "{:8}{:8}{:8}{:8}{:8}{:8}{:8}{:8}{:02}{:02}{:02}{:02}D{:07}\n",
            d.entity_type,
            pstart[k],
            d.structure,
            d.line_font,
            d.level,
            d.view,
            d.transform,
            d.label_display,
            s[0],
            s[1],
            s[2],
            s[3],
            2 * k + 1
        );
        let sub = d.subscript.map(|n| n.to_string()).unwrap_or_default();
        out += &format!(
            "{:8}{:8}{:8}{:8}{:8}{:8}{:8}{:>8.8}{:>8}D{:07}\n",
            d.entity_type,
            d.line_weight,
            d.color,
            pstart[k + 1] - pstart[k],
            d.form,
            "",
            "",
            d.label,
            sub,
            2 * k + 2
        );
    }
    for (k, _) in file.entities.iter().enumerate() {
        for j in pstart[k]..pstart[k + 1] {
            out += &format!("{:64} {:07}P{:07}\n", plines[j - 1], 2 * k + 1, j);
        }
    }
    out += &format!(
        "S{:7}G{:7}D{:7}P{:7}{:40}T0000001\n",
        file.start.len().max(1),
        g.len(),
        file.entities.len() * 2,
        plines.len(),
        ""
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> IgesFile {
        let mut line = Entity {
            directory: Directory::new(110, 0),
            params: [0.0, 0.0, 0.0, 1.0, 2.5, -3e-5].map(Param::Real).to_vec(),
        };
        line.directory.status = [0, 1, 0, 0];
        line.directory.subscript = Some(0);
        let group = Entity {
            directory: Directory::new(402, 1),
            params: vec![Param::Int(1), Param::Int(de_pointer(1))],
        };
        IgesFile {
            start: vec!["a test".into()],
            global: vec![
                Param::Void,
                Param::Void,
                Param::Text("threers".into()),
                Param::Text("x.igs".into()),
                Param::Real(1.0),
                Param::Int(2),
                Param::Text("MM".into()),
            ],
            entities: vec![group, line],
        }
    }

    #[test]
    fn a_written_file_reads_back_the_same() {
        let f = sample();
        let text = write(&f, &WriteOptions::default());
        assert!(text.lines().all(|l| l.len() == 80), "{text}");
        let back = parse(&text).unwrap();
        assert_eq!(back, f);
        assert_eq!(back.entity(3).unwrap().directory.entity_type, 110);
        assert_eq!(back.entities[0].params[1].as_entity(), Some(1));
    }

    #[test]
    fn long_parameter_data_runs_over_lines_and_back() {
        let mut f = sample();
        f.entities[1].params = (0..40)
            .map(|i| Param::Real(i as f64 * 0.123456789))
            .collect();
        let text = write(&f, &WriteOptions::default());
        assert!(
            text.lines()
                .filter(|l| l.ends_with(&format!("P{:07}", 3)))
                .count()
                > 0
        );
        assert_eq!(parse(&text).unwrap(), f);
    }

    #[test]
    fn numbers_in_the_forms_writers_use() {
        assert_eq!(number("1.D-3"), Some(Param::Real(1e-3)));
        assert_eq!(number("2."), Some(Param::Real(2.0)));
        assert_eq!(number(" 7 "), Some(Param::Int(7)));
        assert_eq!(number(".5E+01"), Some(Param::Real(5.0)));
        assert_eq!(number(""), Some(Param::Void));
        assert_eq!(Param::Real(2.0).as_int(), Some(2));
    }

    #[test]
    fn holleriths_may_hold_delimiters() {
        let p = params("3Ha,b,1.5,,4H;;;;;", ',', ';', None).unwrap();
        assert_eq!(
            p,
            vec![
                Param::Text("a,b".into()),
                Param::Real(1.5),
                Param::Void,
                Param::Text(";;;;".into())
            ]
        );
    }

    #[test]
    fn other_delimiters_are_read_from_the_global_section() {
        let mut f = sample();
        f.global[0] = Param::Text("/".into());
        f.global[1] = Param::Text("|".into());
        let text = write(&f, &WriteOptions::default());
        let back = parse(&text).unwrap();
        assert_eq!((back.delimiter(), back.record_delimiter()), ('/', '|'));
        assert_eq!(back.entities, f.entities);
    }
}
