//! OpenSCAD **language** front end — tokenizer → parser → evaluator that turns
//! `.scad` source into a [`Solid`]. Covers the core language:
//!
//! - **3D primitives**: `cube`, `sphere`, `cylinder`, `cone`, `polyhedron`,
//!   `surface` (`.dat` heightmap), `import` (STL/OBJ/OFF for 3D)
//! - **2D primitives**: `square`, `circle`, `polygon` (with `paths` holes),
//!   `text` (via a bundled or supplied TrueType font)
//! - **Extrusions**: `linear_extrude`, `rotate_extrude` (both honour holes and
//!   build closed manifolds directly — no float-CSG differencing)
//! - **Transforms**: `translate`, `rotate`, `scale`, `mirror`, `resize`,
//!   `multmatrix`, `color`
//! - **Booleans**: `union`, `difference`, `intersection`, `hull`, `minkowski`.
//!   3D via the CSG kernel; **2D via a polygon-arrangement kernel** (non-convex,
//!   holes, coincident edges, self-intersection). `minkowski` is **exact in 2D
//!   for non-convex operands** (triangulate → convex-sum → union); in 3D it's the
//!   convex sum (exact for convex operands).
//! - **2D ops**: `offset` (round `r` / mitred·chamfered `delta`), `projection`
//!   (`cut=true` slice at z=0; `cut=false` silhouette)
//! - **Control flow**: `for` (multi-binding), `intersection_for`, `if`/`else`,
//!   `let`/`assign`; `module`/`function` defs (recursive; guarded depth limit);
//!   `children()`/`children(i)`/`children([…])` (resolve anywhere, incl. in loops)
//! - **Files**: `include <…>` (textual), `use <…>` (definitions only) — resolved
//!   by `parse_scad_file`; `import` (STL/OBJ/OFF → solid, DXF/SVG → 2D shape)
//! - **Resolution**: circle/sphere/cylinder use OpenSCAD's exact tessellation —
//!   the fragment count *and* vertex phase from the `$fn`/`$fa`/`$fs` rule — so
//!   default-resolution curves reach vertex parity with OpenSCAD.
//! - **Expressions**: arithmetic, vectors, ranges, indexing (incl. strings),
//!   comparisons, ternary, list comprehensions, **first-class functions**
//!   (`function(x) …` literals, closures, higher-order), `echo`/`assert`, `PI`,
//!   `true`/`false`, special vars (`$fn`/`$fa`/`$fs`/`$t`/`$preview`/…), and the
//!   math/list/string builtins (`lookup`/`search`/`chr`/`ord`/`version`/…)
//!
//! OpenSCAD conventions are honoured (`cube` corner-at-origin unless
//! `center=true`; `rotate` in degrees; `$fn`/`$fa`/`$fs` dynamically scoped). The
//! `#`/`%`/`*`/`!` modifiers parse and are ignored. Caveats: `text` uses a bundled
//! outline font (DejaVu Sans — smooth, watertight glyphs, but not OpenSCAD's
//! Liberation Sans; pass `font="…ttf"` to match); non-convex **3D** `minkowski`
//! is rejected with a clear error (2D is exact); `surface` reads `.dat` (not
//! image) heightmaps. The **biggest gap is curved 3D booleans** (`cube − cylinder`
//! etc.): the exact kernel is watertight for planar/axis-aligned cases but the
//! float fallback leaves seam cracks on curved co-refinement — a full exact
//! result needs exact-arithmetic *construction* (the remaining kernel work).
//! Everything else returns a clear error rather than silently mis-rendering.

use super::{cube, polyhedron, Solid};
use crate::math::Matrix4;
use std::collections::HashMap;

/// File-format importers (OFF/DXF/SVG/3MF/AMF) and `surface()` heightmaps.
mod import;
use import::{
    decode_png_luma, parse_3mf, parse_amf, parse_dxf, parse_off, parse_svg, surface_dat,
    surface_grid,
};

// --- Virtual filesystem -----------------------------------------------------
// A host without a real filesystem (the browser, chiefly) can register file
// contents in memory so `surface`/`import`/`include`/`use` resolve them. Checked
// before the real fs; lookups match either the full resolved path or the bare
// file name, so `surface(file="heightmap.dat")` finds a `"heightmap.dat"` entry
// regardless of the base directory.
static VFS: std::sync::OnceLock<std::sync::Mutex<HashMap<String, Vec<u8>>>> = std::sync::OnceLock::new();
fn vfs() -> &'static std::sync::Mutex<HashMap<String, Vec<u8>>> {
    VFS.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

/// Register an in-memory file for `surface`/`import`/`include`/`use` to read —
/// essential on wasm, where there is no filesystem.
pub fn register_file(name: &str, bytes: Vec<u8>) {
    if let Ok(mut m) = vfs().lock() {
        m.insert(name.to_string(), bytes);
    }
}
/// Drop every registered in-memory file.
pub fn clear_files() {
    if let Ok(mut m) = vfs().lock() {
        m.clear();
    }
}

fn vfs_lookup(path: &std::path::Path) -> Option<Vec<u8>> {
    let m = vfs().lock().ok()?;
    if m.is_empty() {
        return None;
    }
    if let Some(b) = m.get(&path.to_string_lossy().to_string()) {
        return Some(b.clone());
    }
    m.get(&path.file_name()?.to_string_lossy().to_string()).cloned()
}

/// Read a file's bytes: the VFS first, then the real filesystem (native only).
fn read_file_bytes(path: &std::path::Path) -> std::io::Result<Vec<u8>> {
    if let Some(b) = vfs_lookup(path) {
        return Ok(b);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::fs::read(path)
    }
    #[cfg(target_arch = "wasm32")]
    {
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "no filesystem here — register the file via register_file()",
        ))
    }
}
/// Read a file as UTF-8 text: the VFS first, then the real filesystem (native only).
fn read_file_string(path: &std::path::Path) -> std::io::Result<String> {
    let bytes = read_file_bytes(path)?;
    String::from_utf8(bytes).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// Bundled default font for `text()` — DejaVu Sans (Bitstream Vera / Arev
/// license, freely redistributable; see `FONT-LICENSE.txt`). A real outline
/// font, so glyphs are smooth and extrude to watertight solids. OpenSCAD's own
/// default is Liberation Sans, so `text()` output won't match OpenSCAD's glyphs
/// exactly; pass `font="…ttf"` to use a specific typeface.
const DEFAULT_FONT: &[u8] = include_bytes!("DejaVuSans.ttf");

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(f64),
    Str(String),
    Ident(String),
    Op(String), // + - * / % < > = ! & | ? : and multi-char == <= >= != && ||
    Punct(char), // ( ) [ ] { } , ; .
    Eof,
}

fn lex(src: &str) -> Result<Vec<Tok>, String> {
    let b = src.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i] as char;
        if c.is_whitespace() {
            i += 1;
        } else if c == '/' && i + 1 < b.len() && b[i + 1] == b'/' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if c == '/' && i + 1 < b.len() && b[i + 1] == b'*' {
            i += 2;
            while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                i += 1;
            }
            i += 2;
        } else if c == '"' {
            i += 1;
            let mut s = String::new();
            while i < b.len() && b[i] != b'"' {
                if b[i] == b'\\' && i + 1 < b.len() {
                    i += 1;
                    s.push(b[i] as char);
                } else {
                    s.push(b[i] as char);
                }
                i += 1;
            }
            i += 1;
            out.push(Tok::Str(s));
        } else if c.is_ascii_digit() || (c == '.' && i + 1 < b.len() && (b[i + 1] as char).is_ascii_digit()) {
            let start = i;
            while i < b.len() && {
                let ch = b[i] as char;
                ch.is_ascii_digit() || ch == '.' || ch == 'e' || ch == 'E'
                    || ((ch == '+' || ch == '-') && i > start && (b[i - 1] == b'e' || b[i - 1] == b'E'))
            } {
                i += 1;
            }
            let ns = &src[start..i];
            out.push(Tok::Num(ns.parse().map_err(|_| format!("bad number '{ns}'"))?));
        } else if c == '$' || c.is_alphabetic() || c == '_' {
            let start = i;
            i += 1;
            while i < b.len() && {
                let ch = b[i] as char;
                ch.is_alphanumeric() || ch == '_'
            } {
                i += 1;
            }
            out.push(Tok::Ident(src[start..i].to_string()));
        } else if "()[]{},;.".contains(c) {
            out.push(Tok::Punct(c));
            i += 1;
        } else {
            // operators, possibly multi-char
            let two = if i + 1 < b.len() { &src[i..i + 2] } else { "" };
            if ["==", "!=", "<=", ">=", "&&", "||"].contains(&two) {
                out.push(Tok::Op(two.to_string()));
                i += 2;
            } else if "+-*/%<>=!?:#".contains(c) {
                out.push(Tok::Op(c.to_string()));
                i += 1;
            } else {
                return Err(format!("unexpected character '{c}'"));
            }
        }
    }
    out.push(Tok::Eof);
    Ok(out)
}

// ---------------------------------------------------------------------------
// AST
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Expr {
    Num(f64),
    Str(String),
    Ident(String),
    List(Vec<ListElem>),
    Range(Box<Expr>, Option<Box<Expr>>, Box<Expr>),
    Unary(String, Box<Expr>),
    Binary(String, Box<Expr>, Box<Expr>),
    Ternary(Box<Expr>, Box<Expr>, Box<Expr>),
    Index(Box<Expr>, Box<Expr>),
    Call(String, Vec<Arg>),
    Let(Vec<(String, Expr)>, Box<Expr>),
    /// First-class function literal `function(params) body`.
    FuncLit(Vec<Param>, Box<Expr>),
    /// Call the value of an expression, e.g. `fs[1](3)` or `(function(x)x)(4)`.
    CallExpr(Box<Expr>, Vec<Arg>),
}

/// An element inside `[ … ]` — supports list comprehensions.
#[derive(Clone, Debug)]
enum ListElem {
    Item(Expr),
    Each(Expr),
    For(Vec<(String, Expr)>, Box<ListElem>),
    If(Expr, Box<ListElem>, Option<Box<ListElem>>),
    Let(Vec<(String, Expr)>, Box<ListElem>),
}

#[derive(Clone, Debug)]
struct Arg {
    name: Option<String>,
    value: Expr,
}

#[derive(Clone, Debug)]
struct Param {
    name: String,
    default: Option<Expr>,
}

#[derive(Clone, Debug)]
enum Stmt {
    Assign(String, Expr),
    ModuleDef(String, Vec<Param>, Vec<Stmt>),
    FunctionDef(String, Vec<Param>, Expr),
    Call(String, Vec<Arg>, Vec<Stmt>),
    For(String, Expr, Vec<Stmt>),
    If(Expr, Vec<Stmt>, Vec<Stmt>),
    Block(Vec<Stmt>),
    /// A statement with one or more leading modifier characters (`*`/`!`/`#`/`%`).
    Modified(String, Box<Stmt>),
}

// ---------------------------------------------------------------------------
// Parser (recursive descent)
// ---------------------------------------------------------------------------

struct Parser {
    t: Vec<Tok>,
    i: usize,
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.t[self.i]
    }
    fn next(&mut self) -> Tok {
        let t = self.t[self.i].clone();
        self.i += 1;
        t
    }
    fn eat_punct(&mut self, c: char) -> Result<(), String> {
        if self.peek() == &Tok::Punct(c) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected '{c}', found {:?}", self.peek()))
        }
    }
    fn is_punct(&self, c: char) -> bool {
        self.peek() == &Tok::Punct(c)
    }
    fn is_op(&self, s: &str) -> bool {
        matches!(self.peek(), Tok::Op(o) if o == s)
    }

    fn program(&mut self) -> Result<Vec<Stmt>, String> {
        let mut out = Vec::new();
        while self.peek() != &Tok::Eof {
            out.push(self.stmt()?);
        }
        Ok(out)
    }

    fn block_or_stmt(&mut self) -> Result<Vec<Stmt>, String> {
        if self.is_punct('{') {
            self.i += 1;
            let mut out = Vec::new();
            while !self.is_punct('}') {
                out.push(self.stmt()?);
            }
            self.eat_punct('}')?;
            Ok(out)
        } else if self.is_punct(';') {
            self.i += 1;
            Ok(Vec::new())
        } else {
            Ok(vec![self.stmt()?])
        }
    }

    fn stmt(&mut self) -> Result<Stmt, String> {
        // Leading modifiers `*` (disable), `!` (show-only root), `#` (highlight),
        // `%` (background). Capture them and honour their semantics at eval time.
        let mut mods = String::new();
        loop {
            let c = if self.is_op("*") {
                '*'
            } else if self.is_op("!") {
                '!'
            } else if self.is_op("#") {
                '#'
            } else if self.is_op("%") {
                '%'
            } else {
                break;
            };
            mods.push(c);
            self.i += 1;
        }
        let inner = self.stmt_body()?;
        Ok(if mods.is_empty() { inner } else { Stmt::Modified(mods, Box::new(inner)) })
    }

    fn stmt_body(&mut self) -> Result<Stmt, String> {
        if self.is_punct('{') {
            return Ok(Stmt::Block(self.block_or_stmt()?));
        }
        let ident = match self.peek().clone() {
            Tok::Ident(s) => s,
            other => return Err(format!("expected statement, found {other:?}")),
        };
        match ident.as_str() {
            "module" => {
                self.i += 1;
                let name = self.ident()?;
                let params = self.params()?;
                let body = self.block_or_stmt()?;
                Ok(Stmt::ModuleDef(name, params, body))
            }
            "function" => {
                self.i += 1;
                let name = self.ident()?;
                let params = self.params()?;
                if !self.is_op("=") {
                    return Err("expected '=' in function def".into());
                }
                self.i += 1;
                let body = self.expr()?;
                self.eat_punct(';')?;
                Ok(Stmt::FunctionDef(name, params, body))
            }
            "for" | "intersection_for" => {
                let is_isect = ident == "intersection_for";
                self.i += 1;
                self.eat_punct('(')?;
                let mut binds = Vec::new();
                loop {
                    let var = self.ident()?;
                    if !self.is_op("=") {
                        return Err("expected '=' in for".into());
                    }
                    self.i += 1;
                    binds.push((var, self.expr()?));
                    if self.is_punct(',') {
                        self.i += 1;
                    } else {
                        break;
                    }
                }
                self.eat_punct(')')?;
                let body = self.block_or_stmt()?;
                if is_isect {
                    // `intersection_for(...)` → a Call the evaluator folds by ∩.
                    let args = binds.into_iter().map(|(n, e)| Arg { name: Some(n), value: e }).collect();
                    return Ok(Stmt::Call("intersection_for".into(), args, body));
                }
                // Multiple bindings desugar to nested `for` (cartesian product).
                let mut acc = body;
                for (var, range) in binds.into_iter().rev() {
                    acc = vec![Stmt::For(var, range, acc)];
                }
                Ok(acc.into_iter().next().unwrap())
            }
            "if" => {
                self.i += 1;
                self.eat_punct('(')?;
                let cond = self.expr()?;
                self.eat_punct(')')?;
                let then = self.block_or_stmt()?;
                let els = if self.peek() == &Tok::Ident("else".into()) {
                    self.i += 1;
                    self.block_or_stmt()?
                } else {
                    Vec::new()
                };
                Ok(Stmt::If(cond, then, els))
            }
            _ => {
                // Assignment  `name = expr;`  or a module call.
                if self.t.get(self.i + 1) == Some(&Tok::Op("=".into())) {
                    self.i += 2;
                    let e = self.expr()?;
                    self.eat_punct(';')?;
                    Ok(Stmt::Assign(ident, e))
                } else {
                    self.i += 1;
                    let args = self.args()?;
                    let children = self.block_or_stmt()?;
                    Ok(Stmt::Call(ident, args, children))
                }
            }
        }
    }

    fn ident(&mut self) -> Result<String, String> {
        match self.next() {
            Tok::Ident(s) => Ok(s),
            o => Err(format!("expected identifier, found {o:?}")),
        }
    }

    fn params(&mut self) -> Result<Vec<Param>, String> {
        self.eat_punct('(')?;
        let mut out = Vec::new();
        while !self.is_punct(')') {
            let name = self.ident()?;
            let default = if self.is_op("=") {
                self.i += 1;
                Some(self.expr()?)
            } else {
                None
            };
            out.push(Param { name, default });
            if self.is_punct(',') {
                self.i += 1;
            }
        }
        self.eat_punct(')')?;
        Ok(out)
    }

    fn args(&mut self) -> Result<Vec<Arg>, String> {
        self.eat_punct('(')?;
        let mut out = Vec::new();
        while !self.is_punct(')') {
            // named?  ident = expr
            if let (Tok::Ident(n), Some(Tok::Op(eq))) = (self.peek().clone(), self.t.get(self.i + 1)) {
                if eq == "=" {
                    self.i += 2;
                    let value = self.expr()?;
                    out.push(Arg { name: Some(n), value });
                    if self.is_punct(',') {
                        self.i += 1;
                    }
                    continue;
                }
            }
            let value = self.expr()?;
            out.push(Arg { name: None, value });
            if self.is_punct(',') {
                self.i += 1;
            }
        }
        self.eat_punct(')')?;
        Ok(out)
    }

    fn starts_comp(&self) -> bool {
        matches!(self.peek(), Tok::Ident(s) if s == "for" || s == "let" || s == "each" || s == "if")
    }

    fn bindings(&mut self) -> Result<Vec<(String, Expr)>, String> {
        self.eat_punct('(')?;
        let mut out = Vec::new();
        while !self.is_punct(')') {
            let n = self.ident()?;
            if !self.is_op("=") {
                return Err("expected '=' in binding".into());
            }
            self.i += 1;
            out.push((n, self.expr()?));
            if self.is_punct(',') {
                self.i += 1;
            }
        }
        self.eat_punct(')')?;
        Ok(out)
    }

    fn list_elem(&mut self) -> Result<ListElem, String> {
        match self.peek().clone() {
            Tok::Ident(k) if k == "for" => {
                self.i += 1;
                let binds = self.bindings()?;
                Ok(ListElem::For(binds, Box::new(self.list_elem()?)))
            }
            Tok::Ident(k) if k == "let" => {
                self.i += 1;
                let binds = self.bindings()?;
                Ok(ListElem::Let(binds, Box::new(self.list_elem()?)))
            }
            Tok::Ident(k) if k == "if" => {
                self.i += 1;
                self.eat_punct('(')?;
                let cond = self.expr()?;
                self.eat_punct(')')?;
                let then = Box::new(self.list_elem()?);
                let els = if self.peek() == &Tok::Ident("else".into()) {
                    self.i += 1;
                    Some(Box::new(self.list_elem()?))
                } else {
                    None
                };
                Ok(ListElem::If(cond, then, els))
            }
            Tok::Ident(k) if k == "each" => {
                self.i += 1;
                Ok(ListElem::Each(self.expr()?))
            }
            _ => Ok(ListElem::Item(self.expr()?)),
        }
    }

    // --- expressions, precedence-climbing ---
    fn expr(&mut self) -> Result<Expr, String> {
        self.ternary()
    }
    fn ternary(&mut self) -> Result<Expr, String> {
        let c = self.binary(0)?;
        if self.is_op("?") {
            self.i += 1;
            let a = self.expr()?;
            if !self.is_op(":") {
                return Err("expected ':' in ternary".into());
            }
            self.i += 1;
            let b = self.expr()?;
            Ok(Expr::Ternary(Box::new(c), Box::new(a), Box::new(b)))
        } else {
            Ok(c)
        }
    }
    fn binary(&mut self, min_prec: u8) -> Result<Expr, String> {
        let mut lhs = self.unary()?;
        loop {
            let (op, prec) = match self.peek() {
                Tok::Op(o) => {
                    let p = match o.as_str() {
                        "||" => 1,
                        "&&" => 2,
                        "==" | "!=" => 3,
                        "<" | ">" | "<=" | ">=" => 4,
                        "+" | "-" => 5,
                        "*" | "/" | "%" => 6,
                        _ => break,
                    };
                    (o.clone(), p)
                }
                _ => break,
            };
            if prec < min_prec {
                break;
            }
            self.i += 1;
            let rhs = self.binary(prec + 1)?;
            lhs = Expr::Binary(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }
    fn unary(&mut self) -> Result<Expr, String> {
        if self.is_op("-") || self.is_op("!") || self.is_op("+") {
            let op = self.next();
            let e = self.unary()?;
            if let Tok::Op(o) = op {
                if o == "+" {
                    return Ok(e);
                }
                return Ok(Expr::Unary(o, Box::new(e)));
            }
        }
        self.postfix()
    }
    fn postfix(&mut self) -> Result<Expr, String> {
        let mut e = self.primary()?;
        loop {
            if self.is_punct('[') {
                self.i += 1;
                let idx = self.expr()?;
                self.eat_punct(']')?;
                e = Expr::Index(Box::new(e), Box::new(idx));
            } else if self.is_punct('.') {
                self.i += 1;
                let m = self.ident()?;
                let k = match m.as_str() {
                    "x" => 0.0,
                    "y" => 1.0,
                    "z" => 2.0,
                    _ => return Err(format!("unknown member .{m}")),
                };
                e = Expr::Index(Box::new(e), Box::new(Expr::Num(k)));
            } else if self.is_punct('(') {
                // Postfix call on an expression result: `fs[1](3)`, `(function(x)x)(4)`.
                let args = self.args()?;
                e = Expr::CallExpr(Box::new(e), args);
            } else {
                break;
            }
        }
        Ok(e)
    }
    fn primary(&mut self) -> Result<Expr, String> {
        match self.next() {
            Tok::Num(n) => Ok(Expr::Num(n)),
            Tok::Str(s) => Ok(Expr::Str(s)),
            Tok::Ident(id) => match id.as_str() {
                "true" => Ok(Expr::Num(1.0)),
                "false" => Ok(Expr::Num(0.0)),
                "undef" => Ok(Expr::Ident("undef".into())),
                "function" => {
                    // First-class function literal: `function (params) expr`.
                    let params = self.params()?;
                    let body = self.expr()?;
                    Ok(Expr::FuncLit(params, Box::new(body)))
                }
                "let" => {
                    self.eat_punct('(')?;
                    let mut binds = Vec::new();
                    while !self.is_punct(')') {
                        let n = self.ident()?;
                        if !self.is_op("=") {
                            return Err("expected '=' in let".into());
                        }
                        self.i += 1;
                        binds.push((n, self.expr()?));
                        if self.is_punct(',') {
                            self.i += 1;
                        }
                    }
                    self.eat_punct(')')?;
                    let body = self.expr()?;
                    Ok(Expr::Let(binds, Box::new(body)))
                }
                _ => {
                    if self.is_punct('(') {
                        let args = self.args()?;
                        Ok(Expr::Call(id, args))
                    } else {
                        Ok(Expr::Ident(id))
                    }
                }
            },
            Tok::Punct('(') => {
                let e = self.expr()?;
                self.eat_punct(')')?;
                Ok(e)
            }
            Tok::Punct('[') => {
                if self.is_punct(']') {
                    self.i += 1;
                    return Ok(Expr::List(Vec::new()));
                }
                if self.starts_comp() {
                    let mut elems = vec![self.list_elem()?];
                    while self.is_punct(',') {
                        self.i += 1;
                        if self.is_punct(']') {
                            break;
                        }
                        elems.push(self.list_elem()?);
                    }
                    self.eat_punct(']')?;
                    return Ok(Expr::List(elems));
                }
                let first = self.expr()?;
                if self.is_op(":") {
                    self.i += 1;
                    let second = self.expr()?;
                    if self.is_op(":") {
                        self.i += 1;
                        let third = self.expr()?;
                        self.eat_punct(']')?;
                        Ok(Expr::Range(Box::new(first), Some(Box::new(second)), Box::new(third)))
                    } else {
                        self.eat_punct(']')?;
                        Ok(Expr::Range(Box::new(first), None, Box::new(second)))
                    }
                } else {
                    let mut elems = vec![ListElem::Item(first)];
                    while self.is_punct(',') {
                        self.i += 1;
                        if self.is_punct(']') {
                            break;
                        }
                        elems.push(self.list_elem()?);
                    }
                    self.eat_punct(']')?;
                    Ok(Expr::List(elems))
                }
            }
            o => Err(format!("unexpected token {o:?}")),
        }
    }
}

// ---------------------------------------------------------------------------
// Values + evaluator
// ---------------------------------------------------------------------------

/// A first-class function value (a `function(params) body` literal) closing over
/// the scope where it was defined.
#[derive(Clone, Debug)]
struct FuncVal {
    params: Vec<Param>,
    body: Expr,
    captured: Scope,
}

#[derive(Clone, Debug)]
enum Value {
    Num(f64),
    Str(String),
    Vector(Vec<Value>),
    Range(f64, f64, f64),
    Func(std::rc::Rc<FuncVal>),
    Undef,
}

impl Value {
    fn num(&self) -> Result<f64, String> {
        match self {
            Value::Num(n) => Ok(*n),
            _ => Err(format!("expected number, got {self:?}")),
        }
    }
    fn truthy(&self) -> bool {
        match self {
            Value::Num(n) => *n != 0.0,
            Value::Undef => false,
            Value::Vector(v) => !v.is_empty(),
            Value::Str(s) => !s.is_empty(),
            Value::Range(..) => true,
            Value::Func(_) => true,
        }
    }
    fn vec3(&self, fill: f64) -> Result<[f32; 3], String> {
        match self {
            Value::Num(n) => Ok([*n as f32; 3]),
            Value::Vector(v) => {
                let g = |i: usize| v.get(i).and_then(|x| x.num().ok()).unwrap_or(fill) as f32;
                Ok([g(0), g(1), g(2)])
            }
            _ => Err("expected vector".into()),
        }
    }
}

#[derive(Default)]
struct Env {
    modules: HashMap<String, (Vec<Param>, Vec<Stmt>)>,
    functions: HashMap<String, (Vec<Param>, Expr)>,
    /// Directory that `import(…)` paths resolve against.
    base: std::path::PathBuf,
}

/// Build a [`Matrix4`] from an OpenSCAD (row-major) 4×4 / 4×3 matrix value.
fn matrix4_from_value(v: &Value) -> Matrix4 {
    let mut m = Matrix4::default(); // identity
    if let Value::Vector(rows) = v {
        for (r, row) in rows.iter().enumerate().take(4) {
            if let Value::Vector(cols) = row {
                for (c, val) in cols.iter().enumerate().take(4) {
                    if let Ok(x) = val.num() {
                        m.elements[c * 4 + r] = x as f32; // column-major storage
                    }
                }
            }
        }
    }
    m
}

type Scope = HashMap<String, Value>;

// Recursion guard for user function/module calls: matches OpenSCAD's behaviour
// of erroring at a depth limit rather than overflowing the stack. Combined with
// the large-stack eval thread, legitimately deep recursion completes while a
// runaway (infinite) recursion fails gracefully.
// Matches OpenSCAD's default recursion limit, and stays well within the 1 GB
// eval-thread stack (each eval level costs ~10 KB), so it errors before overflow.
const RECURSION_LIMIT: u32 = 20_000;
thread_local! {
    static RECUR_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}
struct RecurGuard;
impl Drop for RecurGuard {
    fn drop(&mut self) {
        RECUR_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}
/// Enter a user call frame; returns the depth-decrementing guard, or an error if
/// the recursion limit is exceeded.
fn enter_recursion() -> Result<RecurGuard, String> {
    let depth = RECUR_DEPTH.with(|d| {
        let v = d.get() + 1;
        d.set(v);
        v
    });
    if depth > RECURSION_LIMIT {
        RECUR_DEPTH.with(|d| d.set(d.get() - 1));
        Err(format!("recursion limit exceeded ({RECURSION_LIMIT})"))
    } else {
        Ok(RecurGuard)
    }
}

fn eval_expr(e: &Expr, sc: &Scope, env: &Env) -> Result<Value, String> {
    Ok(match e {
        Expr::Num(n) => Value::Num(*n),
        Expr::Str(s) => Value::Str(s.clone()),
        Expr::Ident(id) => match id.as_str() {
            "undef" => Value::Undef,
            "true" => Value::Num(1.0),
            "false" => Value::Num(0.0),
            "PI" => Value::Num(std::f64::consts::PI),
            _ => sc.get(id).cloned().unwrap_or_else(|| special_default(id)),
        },
        Expr::List(elems) => {
            let mut out = Vec::new();
            for el in elems {
                eval_list_elem(el, sc, env, &mut out)?;
            }
            Value::Vector(out)
        }
        Expr::Range(a, step, b) => {
            let a = eval_expr(a, sc, env)?.num()?;
            let b = eval_expr(b, sc, env)?.num()?;
            let s = match step {
                Some(s) => eval_expr(s, sc, env)?.num()?,
                None => 1.0,
            };
            Value::Range(a, s, b)
        }
        Expr::Unary(op, x) => {
            let v = eval_expr(x, sc, env)?;
            match op.as_str() {
                "-" => Value::Num(-v.num()?),
                "!" => Value::Num(if v.truthy() { 0.0 } else { 1.0 }),
                _ => return Err(format!("bad unary {op}")),
            }
        }
        Expr::Binary(op, a, b) => {
            let x = eval_expr(a, sc, env)?;
            let y = eval_expr(b, sc, env)?;
            eval_binary(op, x, y)?
        }
        Expr::Ternary(c, a, b) => {
            if eval_expr(c, sc, env)?.truthy() {
                eval_expr(a, sc, env)?
            } else {
                eval_expr(b, sc, env)?
            }
        }
        Expr::Index(v, i) => {
            let vv = eval_expr(v, sc, env)?;
            let idx = eval_expr(i, sc, env)?.num()? as usize;
            match vv {
                Value::Vector(items) => items.get(idx).cloned().unwrap_or(Value::Undef),
                Value::Str(s) => s.chars().nth(idx).map(|c| Value::Str(c.to_string())).unwrap_or(Value::Undef),
                _ => Value::Undef,
            }
        }
        Expr::Let(binds, body) => {
            let mut inner = sc.clone();
            for (n, e) in binds {
                let v = eval_expr(e, &inner, env)?;
                inner.insert(n.clone(), v);
            }
            eval_expr(body, &inner, env)?
        }
        Expr::FuncLit(params, body) => Value::Func(std::rc::Rc::new(FuncVal {
            params: params.clone(),
            body: (**body).clone(),
            captured: sc.clone(),
        })),
        Expr::Call(name, args) => eval_call(name, args, sc, env)?,
        Expr::CallExpr(callee, args) => match eval_expr(callee, sc, env)? {
            Value::Func(f) => call_function(&f, args, sc, env)?,
            other => return Err(format!("cannot call a non-function value: {other:?}")),
        },
    })
}

/// Invoke a first-class function value: bind args (evaluated in the calling
/// scope) over the function's captured scope, then evaluate its body.
fn call_function(f: &FuncVal, args: &[Arg], sc: &Scope, env: &Env) -> Result<Value, String> {
    let _g = enter_recursion()?;
    let mut inner = f.captured.clone();
    for p in &f.params {
        let v = match &p.default {
            Some(d) => eval_expr(d, &f.captured, env)?,
            None => Value::Undef,
        };
        inner.insert(p.name.clone(), v);
    }
    let mut pos = 0;
    for a in args {
        match &a.name {
            Some(nm) => {
                inner.insert(nm.clone(), eval_expr(&a.value, sc, env)?);
            }
            None => {
                if let Some(p) = f.params.get(pos) {
                    inner.insert(p.name.clone(), eval_expr(&a.value, sc, env)?);
                }
                pos += 1;
            }
        }
    }
    eval_expr(&f.body, &inner, env)
}

fn eval_list_elem(el: &ListElem, sc: &Scope, env: &Env, out: &mut Vec<Value>) -> Result<(), String> {
    match el {
        ListElem::Item(e) => out.push(eval_expr(e, sc, env)?),
        ListElem::Each(e) => match eval_expr(e, sc, env)? {
            Value::Vector(v) => out.extend(v),
            Value::Range(a, s, b) => out.extend(iterate(&Value::Range(a, s, b))),
            other => out.push(other),
        },
        ListElem::For(binds, body) => {
            // Nested cartesian product over the bindings' ranges.
            fn rec(
                binds: &[(String, Expr)],
                body: &ListElem,
                sc: &Scope,
                env: &Env,
                out: &mut Vec<Value>,
            ) -> Result<(), String> {
                if binds.is_empty() {
                    return eval_list_elem(body, sc, env, out);
                }
                let (name, range) = &binds[0];
                for item in iterate(&eval_expr(range, sc, env)?) {
                    let mut inner = sc.clone();
                    inner.insert(name.clone(), item);
                    rec(&binds[1..], body, &inner, env, out)?;
                }
                Ok(())
            }
            rec(binds, body, sc, env, out)?;
        }
        ListElem::Let(binds, body) => {
            let mut inner = sc.clone();
            for (n, e) in binds {
                let v = eval_expr(e, &inner, env)?;
                inner.insert(n.clone(), v);
            }
            eval_list_elem(body, &inner, env, out)?;
        }
        ListElem::If(cond, then, els) => {
            if eval_expr(cond, sc, env)?.truthy() {
                eval_list_elem(then, sc, env, out)?;
            } else if let Some(e) = els {
                eval_list_elem(e, sc, env, out)?;
            }
        }
    }
    Ok(())
}

fn eval_binary(op: &str, x: Value, y: Value) -> Result<Value, String> {
    // vector +/- and scalar ops
    if let (Value::Vector(a), Value::Vector(b)) = (&x, &y) {
        if op == "+" || op == "-" {
            let n = a.len().min(b.len());
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                out.push(eval_binary(op, a[i].clone(), b[i].clone())?);
            }
            return Ok(Value::Vector(out));
        }
    }
    let (a, b) = (x.num(), y.num());
    Ok(match op {
        "+" => Value::Num(a? + b?),
        "-" => Value::Num(a? - b?),
        "*" => Value::Num(a? * b?),
        "/" => Value::Num(a? / b?),
        "%" => Value::Num(a? % b?),
        "<" => Value::Num((a? < b?) as i32 as f64),
        ">" => Value::Num((a? > b?) as i32 as f64),
        "<=" => Value::Num((a? <= b?) as i32 as f64),
        ">=" => Value::Num((a? >= b?) as i32 as f64),
        "==" => Value::Num((a? == b?) as i32 as f64),
        "!=" => Value::Num((a? != b?) as i32 as f64),
        "&&" => Value::Num((x.truthy() && y.truthy()) as i32 as f64),
        "||" => Value::Num((x.truthy() || y.truthy()) as i32 as f64),
        _ => return Err(format!("bad operator {op}")),
    })
}

fn eval_call(name: &str, args: &[Arg], sc: &Scope, env: &Env) -> Result<Value, String> {
    // A variable bound to a first-class function value?
    if let Some(Value::Func(f)) = sc.get(name) {
        let f = f.clone();
        return call_function(&f, args, sc, env);
    }
    // User function?
    if let Some((params, body)) = env.functions.get(name) {
        let _g = enter_recursion()?;
        let inner = bind_args(params, args, sc, env)?;
        return eval_expr(body, &inner, env);
    }
    let a: Vec<Value> = args.iter().map(|x| eval_expr(&x.value, sc, env)).collect::<Result<_, _>>()?;
    let n = |i: usize| a.get(i).and_then(|v| v.num().ok()).unwrap_or(0.0);
    Ok(match name {
        "sin" => Value::Num(n(0).to_radians().sin()),
        "cos" => Value::Num(n(0).to_radians().cos()),
        "tan" => Value::Num(n(0).to_radians().tan()),
        "sqrt" => Value::Num(n(0).sqrt()),
        "abs" => Value::Num(n(0).abs()),
        "floor" => Value::Num(n(0).floor()),
        "ceil" => Value::Num(n(0).ceil()),
        "round" => Value::Num(n(0).round()),
        "exp" => Value::Num(n(0).exp()),
        "ln" => Value::Num(n(0).ln()),
        "pow" => Value::Num(n(0).powf(n(1))),
        "log" => Value::Num(n(0).log10()),
        "asin" => Value::Num(n(0).asin().to_degrees()),
        "acos" => Value::Num(n(0).acos().to_degrees()),
        "atan" => Value::Num(n(0).atan().to_degrees()),
        "atan2" => Value::Num(n(0).atan2(n(1)).to_degrees()),
        // rands(min, max, count[, seed]) — a list of `count` uniform values in
        // [min, max). Deterministic per `seed`; without one it advances a
        // thread-local xorshift so successive calls differ (no OS entropy needed,
        // so it works identically on wasm).
        "rands" => {
            let (lo, hi, cnt) = (n(0), n(1), n(2).max(0.0) as usize);
            let seeded = a.get(3).and_then(|v| v.num().ok());
            let mut state = match seeded {
                Some(s) => {
                    let b = s.to_bits();
                    if b == 0 { 0x9E37_79B9_7F4A_7C15 } else { b }
                }
                None => RAND_STATE.with(|c| c.get()),
            };
            let mut out = Vec::with_capacity(cnt);
            for _ in 0..cnt {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let u = (state >> 11) as f64 / 9_007_199_254_740_992.0; // /2^53 → [0,1)
                out.push(Value::Num(lo + (hi - lo) * u));
            }
            if seeded.is_none() {
                RAND_STATE.with(|c| c.set(state));
            }
            Value::Vector(out)
        }
        "sign" => Value::Num(n(0).signum() * (n(0) != 0.0) as i32 as f64),
        "min" => Value::Num(flat_nums(&a).into_iter().fold(f64::INFINITY, f64::min)),
        "max" => Value::Num(flat_nums(&a).into_iter().fold(f64::NEG_INFINITY, f64::max)),
        "norm" => Value::Num(match a.first() {
            Some(Value::Vector(v)) => v.iter().filter_map(|x| x.num().ok()).map(|x| x * x).sum::<f64>().sqrt(),
            _ => 0.0,
        }),
        "cross" => match (a.first(), a.get(1)) {
            (Some(Value::Vector(u)), Some(Value::Vector(w))) if u.len() >= 3 && w.len() >= 3 => {
                let g = |v: &[Value], i: usize| v[i].num().unwrap_or(0.0);
                Value::Vector(vec![
                    Value::Num(g(u, 1) * g(w, 2) - g(u, 2) * g(w, 1)),
                    Value::Num(g(u, 2) * g(w, 0) - g(u, 0) * g(w, 2)),
                    Value::Num(g(u, 0) * g(w, 1) - g(u, 1) * g(w, 0)),
                ])
            }
            _ => Value::Undef,
        },
        "concat" => {
            let mut out = Vec::new();
            for v in &a {
                match v {
                    Value::Vector(items) => out.extend(items.clone()),
                    other => out.push(other.clone()),
                }
            }
            Value::Vector(out)
        }
        "str" => Value::Str(a.iter().map(fmt_value).collect()),
        "len" => Value::Num(match a.first() {
            Some(Value::Vector(v)) => v.len() as f64,
            Some(Value::Str(s)) => s.chars().count() as f64,
            _ => 0.0,
        }),
        "is_undef" => Value::Num(matches!(a.first(), Some(Value::Undef) | None) as i32 as f64),
        "is_num" => Value::Num(matches!(a.first(), Some(Value::Num(_))) as i32 as f64),
        "is_list" => Value::Num(matches!(a.first(), Some(Value::Vector(_))) as i32 as f64),
        "is_string" => Value::Num(matches!(a.first(), Some(Value::Str(_))) as i32 as f64),
        "is_bool" => Value::Num(matches!(a.first(), Some(Value::Num(n)) if *n == 0.0 || *n == 1.0) as i32 as f64),
        "is_function" => Value::Num(matches!(a.first(), Some(Value::Func(_))) as i32 as f64),
        "chr" => Value::Str(flat_nums(&a).iter().filter_map(|&x| char::from_u32(x as u32)).collect()),
        "ord" => match a.first() {
            Some(Value::Str(s)) => s.chars().next().map(|c| Value::Num(c as u32 as f64)).unwrap_or(Value::Undef),
            _ => Value::Undef,
        },
        "version" => Value::Vector(vec![Value::Num(2021.0), Value::Num(1.0), Value::Num(0.0)]),
        "version_num" => Value::Num(2021.01),
        "lookup" => {
            let key = n(0);
            match a.get(1) {
                Some(Value::Vector(table)) => {
                    let mut kv: Vec<(f64, f64)> = table
                        .iter()
                        .filter_map(|e| match e {
                            Value::Vector(p) if p.len() >= 2 => Some((p[0].num().ok()?, p[1].num().ok()?)),
                            _ => None,
                        })
                        .collect();
                    kv.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
                    if kv.is_empty() {
                        Value::Undef
                    } else if key <= kv[0].0 {
                        Value::Num(kv[0].1)
                    } else if key >= kv[kv.len() - 1].0 {
                        Value::Num(kv[kv.len() - 1].1)
                    } else {
                        let mut r = kv[kv.len() - 1].1;
                        for w in kv.windows(2) {
                            if key >= w[0].0 && key <= w[1].0 {
                                let t = (key - w[0].0) / (w[1].0 - w[0].0).max(1e-12);
                                r = w[0].1 + t * (w[1].1 - w[0].1);
                                break;
                            }
                        }
                        Value::Num(r)
                    }
                }
                _ => Value::Undef,
            }
        }
        "search" => match (a.first(), a.get(1)) {
            (Some(Value::Str(m)), Some(Value::Str(s))) => {
                let sv: Vec<char> = s.chars().collect();
                Value::Vector(
                    m.chars()
                        .map(|c| match sv.iter().position(|&x| x == c) {
                            Some(i) => Value::Num(i as f64),
                            None => Value::Vector(vec![]),
                        })
                        .collect(),
                )
            }
            (Some(Value::Num(m)), Some(Value::Vector(list))) => {
                let mut out = Vec::new();
                for (i, e) in list.iter().enumerate() {
                    let hit = match e {
                        Value::Num(x) => *x == *m,
                        Value::Vector(p) => p.first().and_then(|x| x.num().ok()) == Some(*m),
                        _ => false,
                    };
                    if hit {
                        out.push(Value::Num(i as f64));
                    }
                }
                Value::Vector(out)
            }
            _ => Value::Vector(vec![]),
        },
        _ => return Err(format!("unknown function '{name}'")),
    })
}

/// Default values for OpenSCAD special variables when unset in scope.
fn special_default(id: &str) -> Value {
    match id {
        "$fn" => Value::Num(0.0),
        "$fa" => Value::Num(12.0),
        "$fs" => Value::Num(2.0),
        "$t" => Value::Num(0.0),
        "$preview" => Value::Num(1.0),
        "$vpr" => Value::Vector(vec![Value::Num(55.0), Value::Num(0.0), Value::Num(25.0)]),
        "$vpt" => Value::Vector(vec![Value::Num(0.0); 3]),
        "$vpd" => Value::Num(140.0),
        "$vpf" => Value::Num(22.5),
        "$children" => Value::Num(0.0),
        _ => Value::Undef,
    }
}

fn flat_nums(a: &[Value]) -> Vec<f64> {
    if let [Value::Vector(v)] = a {
        v.iter().filter_map(|x| x.num().ok()).collect()
    } else {
        a.iter().filter_map(|x| x.num().ok()).collect()
    }
}

fn fmt_value(v: &Value) -> String {
    match v {
        Value::Num(n) => {
            if n.fract() == 0.0 && n.abs() < 1e15 {
                format!("{}", *n as i64)
            } else {
                format!("{n}")
            }
        }
        Value::Str(s) => s.clone(),
        Value::Vector(items) => {
            format!("[{}]", items.iter().map(fmt_value).collect::<Vec<_>>().join(", "))
        }
        Value::Range(a, s, b) => format!("[{a} : {s} : {b}]"),
        Value::Func(_) => "function(…)".into(),
        Value::Undef => "undef".into(),
    }
}

fn bind_args(params: &[Param], args: &[Arg], sc: &Scope, env: &Env) -> Result<Scope, String> {
    let mut inner = sc.clone();
    let mut positional = 0;
    for p in params {
        if let Some(d) = &p.default {
            let v = eval_expr(d, sc, env)?;
            inner.insert(p.name.clone(), v);
        } else {
            inner.insert(p.name.clone(), Value::Undef);
        }
    }
    for arg in args {
        match &arg.name {
            Some(nm) => {
                let v = eval_expr(&arg.value, sc, env)?;
                inner.insert(nm.clone(), v);
            }
            None => {
                if let Some(p) = params.get(positional) {
                    let v = eval_expr(&arg.value, sc, env)?;
                    inner.insert(p.name.clone(), v);
                }
                positional += 1;
            }
        }
    }
    Ok(inner)
}

/// Look up a named/positional argument value.
fn arg<'a>(args: &'a [Arg], name: &str, pos: usize, sc: &Scope, env: &Env) -> Option<Value> {
    for a in args {
        if a.name.as_deref() == Some(name) {
            return eval_expr(&a.value, sc, env).ok();
        }
    }
    args.iter()
        .filter(|a| a.name.is_none())
        .nth(pos)
        .and_then(|a| eval_expr(&a.value, sc, env).ok())
}

// ---------------------------------------------------------------------------
// Geometry: 2D shapes and 3D solids flow through the evaluator together.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Poly {
    outer: Vec<[f32; 2]>,
    holes: Vec<Vec<[f32; 2]>>,
}
#[derive(Clone, Default)]
struct Shape {
    polys: Vec<Poly>,
}
#[derive(Clone)]
enum Geom {
    Solid(Solid),
    Shape(Shape),
}

impl Shape {
    fn one(outer: Vec<[f32; 2]>) -> Shape {
        Shape { polys: vec![Poly { outer, holes: Vec::new() }] }
    }
    fn map(&self, f: impl Fn([f32; 2]) -> [f32; 2] + Copy) -> Shape {
        Shape {
            polys: self
                .polys
                .iter()
                .map(|p| Poly {
                    outer: p.outer.iter().map(|&q| f(q)).collect(),
                    holes: p.holes.iter().map(|h| h.iter().map(|&q| f(q)).collect()).collect(),
                })
                .collect(),
        }
    }
    fn points(&self) -> Vec<[f32; 2]> {
        self.polys.iter().flat_map(|p| p.outer.iter().copied()).collect()
    }
}

fn square(size: [f32; 2], center: bool) -> Shape {
    let (w, h) = (size[0], size[1]);
    let r = vec![[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]];
    let s = Shape::one(r);
    if center {
        s.map(|p| [p[0] - w / 2.0, p[1] - h / 2.0])
    } else {
        s
    }
}

fn circle(r: f32, fnv: usize) -> Shape {
    let n = fnv.max(3);
    let ring = (0..n)
        .map(|i| {
            let a = i as f32 / n as f32 * std::f32::consts::TAU;
            [r * a.cos(), r * a.sin()]
        })
        .collect();
    Shape::one(ring)
}

/// Cylinder / cone / frustum with OpenSCAD's exact vertex placement: two rings of
/// `n` points at `φ = 360·i/n` (first vertex at angle 0), capped as n-gon fans.
/// Built directly so the facet *phase* matches OpenSCAD (not just the count).
fn cyl_mesh(r1: f32, r2: f32, h: f32, n: usize, center: bool) -> Solid {
    let n = n.max(3);
    let (z0, z1) = if center { (-h / 2.0, h / 2.0) } else { (0.0, h) };
    let ring = |r: f32, z: f32, verts: &mut Vec<[f32; 3]>| {
        for i in 0..n {
            let a = (i as f32 / n as f32) * std::f32::consts::TAU;
            verts.push([r * a.cos(), r * a.sin(), z]);
        }
    };
    let (has0, has1) = (r1 > 1e-9, r2 > 1e-9);
    let (nn, mut verts, mut faces) = (n as u32, Vec::with_capacity(2 * n), Vec::<Vec<u32>>::new());
    match (has0, has1) {
        // Frustum/cylinder: two rings + two caps.
        (true, true) => {
            ring(r1, z0, &mut verts); // 0..n   bottom
            ring(r2, z1, &mut verts); // n..2n  top
            for i in 0..n {
                let (i, j) = (i as u32, ((i + 1) % n) as u32);
                faces.push(vec![i, j, j + nn]);
                faces.push(vec![i, j + nn, i + nn]);
            }
            faces.push((0..nn).rev().collect()); // bottom → −Z
            faces.push((0..nn).map(|i| i + nn).collect()); // top → +Z
        }
        // Cone (r2==0): bottom ring fans up to a single apex — a zero-radius ring
        // would otherwise weld into n coincident verts and a non-manifold tip.
        (true, false) => {
            ring(r1, z0, &mut verts);
            let apex = verts.len() as u32;
            verts.push([0.0, 0.0, z1]);
            for i in 0..n {
                faces.push(vec![i as u32, ((i + 1) % n) as u32, apex]);
            }
            faces.push((0..nn).rev().collect()); // bottom cap
        }
        // Inverted cone (r1==0): single apex fans up to the top ring.
        (false, true) => {
            let apex = 0u32;
            verts.push([0.0, 0.0, z0]);
            ring(r2, z1, &mut verts); // 1..=n
            for i in 0..n {
                faces.push(vec![apex, 1 + i as u32, 1 + ((i + 1) % n) as u32]);
            }
            faces.push((1..=nn).collect()); // top cap
        }
        // Both radii ~0 → degenerate line, no solid.
        (false, false) => return polyhedron(&[], &[]),
    }
    polyhedron(&verts, &faces)
}

/// Sphere with OpenSCAD's exact tessellation: `(n+1)/2` latitude rings at
/// `φ = 180·(i+0.5)/rings`, each an `n`-gon at `θ = 360·j/n`, with flat n-gon
/// pole caps — so both the ring/fragment counts *and* the vertex phase match.
fn sphere_mesh(r: f32, n: usize) -> Solid {
    let n = n.max(3);
    let rings = (n + 1) / 2;
    let mut verts = Vec::with_capacity(rings * n);
    for i in 0..rings {
        let phi = (180.0 * (i as f32 + 0.5) / rings as f32).to_radians();
        let (rr, z) = (r * phi.sin(), r * phi.cos());
        for j in 0..n {
            let a = (j as f32 / n as f32) * std::f32::consts::TAU;
            verts.push([rr * a.cos(), rr * a.sin(), z]);
        }
    }
    let vid = |ring: usize, j: usize| (ring * n + j) as u32;
    let mut faces: Vec<Vec<u32>> = Vec::new();
    faces.push((0..n).map(|j| vid(0, j)).collect()); // north pole cap (+Z)
    for i in 0..rings - 1 {
        for j in 0..n {
            let jn = (j + 1) % n;
            faces.push(vec![vid(i, j), vid(i, jn), vid(i + 1, jn)]);
            faces.push(vec![vid(i, j), vid(i + 1, jn), vid(i + 1, j)]);
        }
    }
    faces.push((0..n).rev().map(|j| vid(rings - 1, j)).collect()); // south pole cap (−Z)
    polyhedron(&verts, &faces)
}

/// 2D convex hull (Andrew's monotone chain).
fn hull2d(mut pts: Vec<[f32; 2]>) -> Vec<[f32; 2]> {
    if pts.len() < 3 {
        return pts;
    }
    pts.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap().then(a[1].partial_cmp(&b[1]).unwrap()));
    pts.dedup();
    let cross = |o: [f32; 2], a: [f32; 2], b: [f32; 2]| {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };
    let mut h: Vec<[f32; 2]> = Vec::new();
    for &p in &pts {
        while h.len() >= 2 && cross(h[h.len() - 2], h[h.len() - 1], p) <= 0.0 {
            h.pop();
        }
        h.push(p);
    }
    let lower = h.len() + 1;
    for &p in pts.iter().rev() {
        while h.len() >= lower && cross(h[h.len() - 2], h[h.len() - 1], p) <= 0.0 {
            h.pop();
        }
        h.push(p);
    }
    h.pop();
    h
}

/// 3D convex hull (incremental) → an outward-wound triangle polyhedron `Solid`.
pub(super) fn hull3d(pts: &[[f32; 3]]) -> Option<Solid> {
    use crate::exact_csg::{cross, orient3d, sqlen, sub};
    // Mesh vertex soups repeat every corner once per incident face — dedup first,
    // else coincident points spawn degenerate overlapping hull faces.
    let mut p: Vec<[f64; 3]> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for q in pts {
        let k = ((q[0] * 1e5).round() as i64, (q[1] * 1e5).round() as i64, (q[2] * 1e5).round() as i64);
        if seen.insert(k) {
            p.push([q[0] as f64, q[1] as f64, q[2] as f64]);
        }
    }
    if p.len() < 4 {
        return None;
    }
    // Seed: find 4 non-coplanar points.
    let mut seed = vec![0usize];
    for i in 1..p.len() {
        if seed.len() == 1 && p[i] != p[0] {
            seed.push(i);
        } else if seed.len() == 2 {
            // non-collinear with seed[0..2]
            let d = cross(sub(p[seed[1]], p[seed[0]]), sub(p[i], p[seed[0]]));
            if sqlen(d) > 1e-12 {
                seed.push(i);
            }
        } else if seed.len() == 3 && orient3d(p[seed[0]], p[seed[1]], p[seed[2]], p[i]) != 0.0 {
            seed.push(i);
            break;
        }
    }
    if seed.len() < 4 {
        return None;
    }
    // Initial tetrahedron faces (outward-oriented).
    let mut faces: Vec<[usize; 3]> = Vec::new();
    let (a, b, c, d) = (seed[0], seed[1], seed[2], seed[3]);
    let add_face = |faces: &mut Vec<[usize; 3]>, x, y, z, apex| {
        // orient so `apex` is behind (negative side)
        if orient3d(p[x], p[y], p[z], p[apex]) > 0.0 {
            faces.push([x, z, y]);
        } else {
            faces.push([x, y, z]);
        }
    };
    add_face(&mut faces, a, b, c, d);
    add_face(&mut faces, a, b, d, c);
    add_face(&mut faces, a, c, d, b);
    add_face(&mut faces, b, c, d, a);

    for (i, pt) in p.iter().enumerate() {
        // Visible = pt strictly outside the face. Faces are wound outward (the
        // opposite seed is behind, `orient3d(face, seed) < 0`), so a point is in
        // front — and the face should be replaced — when `orient3d(face, pt) > 0`.
        let visible: Vec<usize> = faces
            .iter()
            .enumerate()
            .filter(|(_, f)| orient3d(p[f[0]], p[f[1]], p[f[2]], *pt) > 0.0)
            .map(|(k, _)| k)
            .collect();
        if visible.is_empty() {
            continue;
        }
        // horizon edges: edges of visible faces not shared with another visible face
        let vis: std::collections::HashSet<usize> = visible.iter().copied().collect();
        let mut horizon: Vec<(usize, usize)> = Vec::new();
        for &fi in &visible {
            let f = faces[fi];
            for e in 0..3 {
                let (u, v) = (f[e], f[(e + 1) % 3]);
                // is edge (v,u) in another visible face? if not → horizon
                let shared = visible.iter().any(|&gj| {
                    gj != fi && {
                        let g = faces[gj];
                        (0..3).any(|k| g[k] == v && g[(k + 1) % 3] == u)
                    }
                });
                let _ = &vis;
                if !shared {
                    horizon.push((u, v));
                }
            }
        }
        // remove visible faces
        let mut kept: Vec<[usize; 3]> = faces
            .iter()
            .enumerate()
            .filter(|(k, _)| !vis.contains(k))
            .map(|(_, f)| *f)
            .collect();
        // add new faces from horizon to pt
        for (u, v) in horizon {
            kept.push([u, v, i]);
        }
        faces = kept;
    }

    // Build a polyhedron Solid from the hull faces (points verbatim, auto-oriented).
    let uniq: Vec<[f32; 3]> = p.iter().map(|q| [q[0] as f32, q[1] as f32, q[2] as f32]).collect();
    let faces_u: Vec<Vec<u32>> = faces.iter().map(|f| vec![f[0] as u32, f[1] as u32, f[2] as u32]).collect();
    Some(polyhedron(&uniq, &faces_u))
}

// ---------------------------------------------------------------------------
// Cap triangulation for polygons with holes (earcut-style hole bridging). Used
// by both extrudes so a 2D shape with holes becomes a closed manifold *directly*
// — no float-CSG differencing (which stack-overflows on coaxial prisms).
// ---------------------------------------------------------------------------

/// Signed area (×2) of a ring given by indices into `pts` (>0 ⇒ CCW).
fn ring_signed_area(pts: &[[f64; 2]], ring: &[usize]) -> f64 {
    let n = ring.len();
    let mut a = 0.0;
    for i in 0..n {
        let p = pts[ring[i]];
        let q = pts[ring[(i + 1) % n]];
        a += p[0] * q[1] - q[0] * p[1];
    }
    a
}

/// Reorient a 2D ring to the requested handedness (`ccw=true` ⇒ counter-clockwise).
fn orient_ring(ring: &[[f32; 2]], ccw: bool) -> Vec<[f32; 2]> {
    let mut r = ring.to_vec();
    let n = r.len();
    let mut a = 0.0f32;
    for i in 0..n {
        let p = r[i];
        let q = r[(i + 1) % n];
        a += p[0] * q[1] - q[0] * p[1];
    }
    if (a > 0.0) != ccw {
        r.reverse();
    }
    r
}

/// earcut point-in-triangle (boundary-inclusive).
fn pit(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64, px: f64, py: f64) -> bool {
    (cx - px) * (ay - py) - (ax - px) * (cy - py) >= 0.0
        && (ax - px) * (by - py) - (bx - px) * (ay - py) >= 0.0
        && (bx - px) * (cy - py) - (cx - px) * (by - py) >= 0.0
}

/// earcut `locallyInside`: is the diagonal from outer vertex at position `i` to
/// point `m` locally inside the polygon?
fn locally_inside(pts: &[[f64; 2]], outer: &[usize], i: usize, m: usize) -> bool {
    let n = outer.len();
    let a = pts[outer[i]];
    let prev = pts[outer[(i + n - 1) % n]];
    let next = pts[outer[(i + 1) % n]];
    let b = pts[m];
    let ar = |p: [f64; 2], q: [f64; 2], r: [f64; 2]| {
        (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    };
    if ar(prev, a, next) < 0.0 {
        ar(a, b, next) >= 0.0 && ar(a, prev, b) >= 0.0
    } else {
        ar(a, b, prev) < 0.0 || ar(a, next, b) < 0.0
    }
}

/// Find the outer-ring position to bridge the hole vertex `m` to (earcut
/// `findHoleBridge`): cast a ray leftward from `m`, then refine by visibility.
fn find_hole_bridge(pts: &[[f64; 2]], outer: &[usize], m: usize) -> Option<usize> {
    let (hx, hy) = (pts[m][0], pts[m][1]);
    let n = outer.len();
    let mut qx = f64::NEG_INFINITY;
    let mut mpos: Option<usize> = None;
    for i in 0..n {
        let p = pts[outer[i]];
        let pn = pts[outer[(i + 1) % n]];
        if hy <= p[1] && hy >= pn[1] && pn[1] != p[1] {
            let x = p[0] + (hy - p[1]) / (pn[1] - p[1]) * (pn[0] - p[0]);
            if x <= hx && x > qx {
                qx = x;
                mpos = Some(if p[0] < pn[0] { i } else { (i + 1) % n });
                if x == hx {
                    return mpos;
                }
            }
        }
    }
    let mut mpos = mpos?;
    let mp = pts[outer[mpos]];
    let (mx, my) = (mp[0], mp[1]);
    let mut tan_min = f64::INFINITY;
    for i in 0..n {
        let p = pts[outer[i]];
        if hx >= p[0] && p[0] >= mx && hx != p[0] {
            let (t1x, t2x) = if hy < my { (hx, qx) } else { (qx, hx) };
            if pit(t1x, hy, mx, my, t2x, hy, p[0], p[1]) {
                let tan = (hy - p[1]).abs() / (hx - p[0]);
                let cur = pts[outer[mpos]][0];
                if locally_inside(pts, outer, i, m)
                    && (tan < tan_min || (tan == tan_min && p[0] > cur))
                {
                    mpos = i;
                    tan_min = tan;
                }
            }
        }
    }
    Some(mpos)
}

/// Splice a hole ring into the outer ring via a bridge (earcut `eliminateHole`).
fn eliminate_hole(pts: &[[f64; 2]], outer: &[usize], hole: &[usize]) -> Option<Vec<usize>> {
    let hstart = (0..hole.len())
        .min_by(|&a, &b| pts[hole[a]][0].partial_cmp(&pts[hole[b]][0]).unwrap())?;
    let m = hole[hstart];
    let bpos = find_hole_bridge(pts, outer, m)?;
    let bridge = outer[bpos];
    let rotated: Vec<usize> =
        hole[hstart..].iter().chain(hole[..hstart].iter()).copied().collect();
    let mut out = Vec::with_capacity(outer.len() + hole.len() + 2);
    out.extend_from_slice(&outer[..=bpos]);
    out.extend_from_slice(&rotated);
    out.push(m);
    out.push(bridge);
    out.extend_from_slice(&outer[bpos + 1..]);
    Some(out)
}

/// Ear-clip a (weakly simple) ring of `pts` indices into CCW triangles.
fn earclip(pts: &[[f64; 2]], ring: &[usize]) -> Vec<[usize; 3]> {
    let mut idx: Vec<usize> = ring.to_vec();
    if ring_signed_area(pts, &idx) < 0.0 {
        idx.reverse();
    }
    let cross = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };
    let mut tris = Vec::new();
    let mut guard = idx.len() * idx.len() + 4;
    while idx.len() > 3 && guard > 0 {
        guard -= 1;
        let n = idx.len();
        let mut clipped = false;
        for i in 0..n {
            let (ip, inx) = ((i + n - 1) % n, (i + 1) % n);
            let (a, b, c) = (pts[idx[ip]], pts[idx[i]], pts[idx[inx]]);
            if cross(a, b, c) <= 0.0 {
                continue; // reflex or degenerate corner
            }
            let coincide = |u: [f64; 2], v: [f64; 2]| {
                (u[0] - v[0]).abs() < 1e-9 && (u[1] - v[1]).abs() < 1e-9
            };
            let mut ok = true;
            for j in 0..n {
                if j == ip || j == i || j == inx {
                    continue;
                }
                let p = pts[idx[j]];
                // Skip vertices coincident with an ear corner (hole-bridge dupes);
                // block on any other vertex inside or on the closed ear triangle —
                // this catches reflex corners lying on an ear edge (concave polys).
                if coincide(p, a) || coincide(p, b) || coincide(p, c) {
                    continue;
                }
                if cross(a, b, p) >= 0.0 && cross(b, c, p) >= 0.0 && cross(c, a, p) >= 0.0 {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            tris.push([idx[ip], idx[i], idx[inx]]);
            idx.remove(i);
            clipped = true;
            break;
        }
        if !clipped {
            break;
        }
    }
    if idx.len() == 3 {
        tris.push([idx[0], idx[1], idx[2]]);
    }
    tris
}

/// Triangulate a polygon with holes. `outer` must be CCW and each hole CW.
/// Returns triangles indexing the concatenated list `outer ++ holes[0] ++ …`.
fn triangulate_holes(outer: &[[f32; 2]], holes: &[Vec<[f32; 2]>]) -> Vec<[usize; 3]> {
    let mut pts: Vec<[f64; 2]> = outer.iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
    let outer_idx: Vec<usize> = (0..outer.len()).collect();
    let mut hole_rings: Vec<Vec<usize>> = Vec::new();
    for h in holes {
        if h.len() < 3 {
            continue;
        }
        let base = pts.len();
        pts.extend(h.iter().map(|p| [p[0] as f64, p[1] as f64]));
        hole_rings.push((base..pts.len()).collect());
    }
    hole_rings.sort_by(|a, b| {
        let la = a.iter().map(|&i| pts[i][0]).fold(f64::INFINITY, f64::min);
        let lb = b.iter().map(|&i| pts[i][0]).fold(f64::INFINITY, f64::min);
        la.partial_cmp(&lb).unwrap()
    });
    let mut ring = outer_idx;
    for hole in &hole_rings {
        if let Some(merged) = eliminate_hole(&pts, &ring, hole) {
            ring = merged;
        }
    }
    earclip(&pts, &ring)
}

/// Append a closed prism (extruded polygon-with-holes) into shared buffers.
fn push_prism(
    verts: &mut Vec<[f32; 3]>,
    faces: &mut Vec<Vec<u32>>,
    outer: &[[f32; 2]],
    holes: &[Vec<[f32; 2]>],
    z0: f32,
    z1: f32,
) {
    let outer = orient_ring(outer, true);
    let holes: Vec<Vec<[f32; 2]>> =
        holes.iter().filter(|h| h.len() >= 3).map(|h| orient_ring(h, false)).collect();
    let tris = triangulate_holes(&outer, &holes);
    let mut flat = outer.clone();
    for h in &holes {
        flat.extend_from_slice(h);
    }
    let n = flat.len() as u32;
    let b = verts.len() as u32;
    for p in &flat {
        verts.push([p[0], p[1], z0]);
    }
    for p in &flat {
        verts.push([p[0], p[1], z1]);
    }
    for t in &tris {
        let (a0, a1, a2) = (t[0] as u32, t[1] as u32, t[2] as u32);
        faces.push(vec![b + a0, b + a2, b + a1]); // bottom cap → −Z
        faces.push(vec![b + n + a0, b + n + a1, b + n + a2]); // top cap → +Z
    }
    let mut wall = |start: usize, len: usize| {
        for k in 0..len {
            let a = b + (start + k) as u32;
            let bb = b + (start + (k + 1) % len) as u32;
            faces.push(vec![a, bb, bb + n]);
            faces.push(vec![a, bb + n, a + n]);
        }
    };
    wall(0, outer.len());
    let mut off = outer.len();
    for h in &holes {
        wall(off, h.len());
        off += h.len();
    }
}

/// Like [`push_prism`], but lofts the profile through `slices` intermediate layers
/// with a per-layer twist (radians, total over the height) and end scale (`scale`
/// applied linearly bottom→top) — OpenSCAD `linear_extrude(twist=, scale=, slices=)`.
fn push_loft(
    verts: &mut Vec<[f32; 3]>,
    faces: &mut Vec<Vec<u32>>,
    outer: &[[f32; 2]],
    holes: &[Vec<[f32; 2]>],
    z0: f32,
    z1: f32,
    twist: f32,
    scale: [f32; 2],
    slices: usize,
) {
    let outer = orient_ring(outer, true);
    let holes: Vec<Vec<[f32; 2]>> =
        holes.iter().filter(|h| h.len() >= 3).map(|h| orient_ring(h, false)).collect();
    let tris = triangulate_holes(&outer, &holes);
    let mut flat = outer.clone();
    for h in &holes {
        flat.extend_from_slice(h);
    }
    let n = flat.len() as u32;
    let b = verts.len() as u32;
    let ns = slices.max(1);
    // One ring of `n` transformed verts per layer (slice 0..=ns).
    for i in 0..=ns {
        let t = i as f32 / ns as f32;
        let z = z0 + (z1 - z0) * t;
        let ang = twist * t;
        let (sx, sy) = (1.0 + (scale[0] - 1.0) * t, 1.0 + (scale[1] - 1.0) * t);
        let (ca, sa) = (ang.cos(), ang.sin());
        for p in &flat {
            let (x, y) = (p[0] * sx, p[1] * sy);
            verts.push([x * ca - y * sa, x * sa + y * ca, z]);
        }
    }
    let bot = b; // slice 0 base
    let top = b + ns as u32 * n; // slice ns base
    for tr in &tris {
        let (a0, a1, a2) = (tr[0] as u32, tr[1] as u32, tr[2] as u32);
        faces.push(vec![bot + a0, bot + a2, bot + a1]); // bottom cap → −Z
        faces.push(vec![top + a0, top + a1, top + a2]); // top cap → +Z
    }
    let mut wall = |start: usize, len: usize| {
        for i in 0..ns as u32 {
            let (s0, s1) = (b + i * n, b + (i + 1) * n);
            for k in 0..len {
                let a = (start + k) as u32;
                let bb = (start + (k + 1) % len) as u32;
                faces.push(vec![s0 + a, s0 + bb, s1 + bb]);
                faces.push(vec![s0 + a, s1 + bb, s1 + a]);
            }
        }
    };
    wall(0, outer.len());
    let mut off = outer.len();
    for h in &holes {
        wall(off, h.len());
        off += h.len();
    }
}

/// Extrude a 2D shape along Z (OpenSCAD `linear_extrude`) as a closed manifold.
/// `twist` (degrees) and `scale` loft the profile through `slices` layers; a plain
/// extrude (no twist, unit scale) uses the cheaper straight-prism path.
fn linear_extrude_shape(
    s: &Shape,
    height: f32,
    center: bool,
    twist_deg: f32,
    scale: [f32; 2],
    slices: usize,
) -> Option<Solid> {
    let (z0, z1) = if center { (-height / 2.0, height / 2.0) } else { (0.0, height) };
    let straight = twist_deg.abs() < 1e-6
        && (scale[0] - 1.0).abs() < 1e-6
        && (scale[1] - 1.0).abs() < 1e-6;
    let mut verts = Vec::new();
    let mut faces = Vec::new();
    for poly in &s.polys {
        if poly.outer.len() >= 3 {
            if straight {
                push_prism(&mut verts, &mut faces, &poly.outer, &poly.holes, z0, z1);
            } else {
                push_loft(&mut verts, &mut faces, &poly.outer, &poly.holes, z0, z1, twist_deg.to_radians(), scale, slices);
            }
        }
    }
    (!faces.is_empty()).then(|| polyhedron(&verts, &faces))
}

/// Append a solid of revolution (profile-with-holes spun about the Y axis).
fn push_revolution(
    verts: &mut Vec<[f32; 3]>,
    faces: &mut Vec<Vec<u32>>,
    outer: &[[f32; 2]],
    holes: &[Vec<[f32; 2]>],
    ang: f32,
    segs: usize,
    full: bool,
) {
    let nsteps = if full { segs } else { segs + 1 };
    let outer = orient_ring(outer, true);
    let holes: Vec<Vec<[f32; 2]>> =
        holes.iter().filter(|h| h.len() >= 3).map(|h| orient_ring(h, false)).collect();
    let mut rings: Vec<&[[f32; 2]]> = vec![&outer[..]];
    for h in &holes {
        rings.push(&h[..]);
    }
    let mut ring_base = Vec::new();
    for ring in &rings {
        ring_base.push(verts.len());
        for &[r, hh] in ring.iter() {
            for j in 0..nsteps {
                let t = ang * (j as f32) / (segs as f32);
                verts.push([r * t.cos(), hh, r * t.sin()]);
            }
        }
    }
    for (ri, ring) in rings.iter().enumerate() {
        let base = ring_base[ri];
        let m = ring.len();
        let jsteps = if full { nsteps } else { nsteps - 1 };
        let vid = |kk: usize, jj: usize| (base + kk * nsteps + jj) as u32;
        for k in 0..m {
            let k1 = (k + 1) % m;
            for j in 0..jsteps {
                let jn = (j + 1) % nsteps;
                faces.push(vec![vid(k, j), vid(k, jn), vid(k1, jn)]);
                faces.push(vec![vid(k, j), vid(k1, jn), vid(k1, j)]);
            }
        }
    }
    if !full {
        let tris = triangulate_holes(&outer, &holes);
        let map_idx = |c: usize| -> (usize, usize) {
            if c < outer.len() {
                return (0, c);
            }
            let mut rem = c - outer.len();
            for (hi, h) in holes.iter().enumerate() {
                if rem < h.len() {
                    return (hi + 1, rem);
                }
                rem -= h.len();
            }
            (0, 0)
        };
        let vid_at = |c: usize, jj: usize| {
            let (ri, k) = map_idx(c);
            (ring_base[ri] + k * nsteps + jj) as u32
        };
        for t in &tris {
            faces.push(vec![vid_at(t[0], 0), vid_at(t[2], 0), vid_at(t[1], 0)]);
            let e = nsteps - 1;
            faces.push(vec![vid_at(t[0], e), vid_at(t[1], e), vid_at(t[2], e)]);
        }
    }
}

/// Revolve a 2D shape about the Y axis (OpenSCAD `rotate_extrude`) as a closed
/// manifold — full turns wrap seamlessly; partial turns are capped at both ends.
fn rotate_extrude_shape(s: &Shape, angle: f32, fnv: usize) -> Option<Solid> {
    let ang = angle.clamp(0.0, std::f32::consts::TAU);
    let full = (std::f32::consts::TAU - ang).abs() < 1e-4;
    let segs = fnv.max(3);
    let mut verts = Vec::new();
    let mut faces = Vec::new();
    for poly in &s.polys {
        if poly.outer.len() >= 3 {
            push_revolution(&mut verts, &mut faces, &poly.outer, &poly.holes, ang, segs, full);
        }
    }
    (!faces.is_empty()).then(|| polyhedron(&verts, &faces))
}

// ---------------------------------------------------------------------------
// resize / minkowski / offset / projection / 2D intersection
// ---------------------------------------------------------------------------

/// Axis-aligned bounds of a point cloud.
fn bbox3(pts: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for p in pts {
        for i in 0..3 {
            mn[i] = mn[i].min(p[i]);
            mx[i] = mx[i].max(p[i]);
        }
    }
    (mn, mx)
}

/// Per-axis factor that resizes a bounding box `cur` to `newsize` (OpenSCAD
/// `resize`): a zero target leaves the axis unscaled, unless `auto` copies the
/// factor from the first explicitly-sized axis.
fn resize_factor(cur: [f32; 3], newsize: [f32; 3], auto: [bool; 3]) -> [f32; 3] {
    let mut f = [1.0f32; 3];
    for i in 0..3 {
        if newsize[i] > 0.0 && cur[i] > 1e-9 {
            f[i] = newsize[i] / cur[i];
        }
    }
    let reff = (0..3).find(|&i| newsize[i] > 0.0 && cur[i] > 1e-9).map(|i| f[i]);
    for i in 0..3 {
        if newsize[i] <= 0.0 && auto[i] {
            if let Some(r) = reff {
                f[i] = r;
            }
        }
    }
    f
}

/// OpenSCAD `resize`: scale (about the origin) so the child's bounding box
/// matches `newsize`.
fn resize_geom(g: Geom, newsize: [f32; 3], auto: [bool; 3]) -> Geom {
    match g {
        Geom::Solid(s) => {
            let pts = solid_points(&s);
            if pts.is_empty() {
                return Geom::Solid(s);
            }
            let (mn, mx) = bbox3(&pts);
            let cur = [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]];
            let f = resize_factor(cur, newsize, auto);
            Geom::Solid(s.scale(f))
        }
        Geom::Shape(sh) => {
            let pts: Vec<[f32; 3]> = sh.points().iter().map(|p| [p[0], p[1], 0.0]).collect();
            if pts.is_empty() {
                return Geom::Shape(sh);
            }
            let (mn, mx) = bbox3(&pts);
            let cur = [mx[0] - mn[0], mx[1] - mn[1], 0.0];
            let f = resize_factor(cur, newsize, auto);
            Geom::Shape(sh.map(|p| [p[0] * f[0], p[1] * f[1]]))
        }
    }
}

/// Convex Minkowski sum of two 3D point sets = convex hull of pairwise sums.
/// (Exact for convex operands — the dominant "round a shape with a sphere" case.)
fn minkowski_sum3(a: &[[f32; 3]], b: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let mut out = Vec::with_capacity(a.len() * b.len());
    for p in a {
        for q in b {
            out.push([p[0] + q[0], p[1] + q[1], p[2] + q[2]]);
        }
    }
    out
}

/// Cover a 2D shape (with holes) by triangles (each convex).
fn shape_triangles(s: &Shape) -> Vec<[[f32; 2]; 3]> {
    let mut out = Vec::new();
    for poly in &s.polys {
        if poly.outer.len() < 3 {
            continue;
        }
        let outer = orient_ring(&poly.outer, true);
        let holes: Vec<Vec<[f32; 2]>> =
            poly.holes.iter().filter(|h| h.len() >= 3).map(|h| orient_ring(h, false)).collect();
        let tris = triangulate_holes(&outer, &holes);
        let mut flat = outer.clone();
        for h in &holes {
            flat.extend_from_slice(h);
        }
        for t in tris {
            out.push([flat[t[0]], flat[t[1]], flat[t[2]]]);
        }
    }
    out
}

/// Exact 2D Minkowski sum for arbitrary (possibly non-convex, holed) shapes:
/// `A ⊕ B = ⋃_{i,j} (triangleₐ ⊕ triangle_b)`, each pair being convex.
fn minkowski2d(a: &Shape, b: &Shape) -> Shape {
    let (ta, tb) = (shape_triangles(a), shape_triangles(b));
    let mut pieces: Vec<Shape> = Vec::with_capacity(ta.len() * tb.len());
    for x in &ta {
        for y in &tb {
            let mut pts = Vec::with_capacity(9);
            for p in x {
                for q in y {
                    pts.push([p[0] + q[0], p[1] + q[1]]);
                }
            }
            let h = hull2d(pts);
            if h.len() >= 3 {
                pieces.push(Shape::one(h));
            }
        }
    }
    union_shapes(&pieces)
}

/// Offset a single CCW/CW contour by `d` (outward for the ring's winding) with
/// rounded (`round=true`, arc joins) or mitered corners. Returns the offset ring.
fn offset_ring(ring: &[[f32; 2]], d: f32, round: bool, fnv: usize) -> Vec<[f32; 2]> {
    let n = ring.len();
    if n < 3 || d == 0.0 {
        return ring.to_vec();
    }
    // Signed area sets the interior side; outward normal of edge (a→b) for a CCW
    // ring is (dy, -dx). Positive `d` grows the region.
    let mut area = 0.0;
    for i in 0..n {
        let a = ring[i];
        let b = ring[(i + 1) % n];
        area += a[0] * b[1] - b[0] * a[1];
    }
    let ccw = area > 0.0;
    let sgn = if ccw { 1.0 } else { -1.0 };
    let edge_normal = |i: usize| {
        let a = ring[i];
        let b = ring[(i + 1) % n];
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let l = (dx * dx + dy * dy).sqrt().max(1e-9);
        [sgn * dy / l, -sgn * dx / l]
    };
    let mut out: Vec<[f32; 2]> = Vec::new();
    for i in 0..n {
        let prev = (i + n - 1) % n;
        let np = edge_normal(prev); // normal of edge into vertex i
        let nn = edge_normal(i); // normal of edge out of vertex i
        let v = ring[i];
        let p_in = [v[0] + np[0] * d, v[1] + np[1] * d];
        let p_out = [v[0] + nn[0] * d, v[1] + nn[1] * d];
        // Convex corner (outward turn) when cross(np, nn)*sgn... use the turn sign.
        let cross = np[0] * nn[1] - np[1] * nn[0];
        let convex = if d > 0.0 { cross * sgn > 1e-9 } else { cross * sgn < -1e-9 };
        if round && convex {
            // Arc from p_in to p_out around v.
            let a0 = (p_in[1] - v[1]).atan2(p_in[0] - v[0]);
            let mut a1 = (p_out[1] - v[1]).atan2(p_out[0] - v[0]);
            let dir = if d > 0.0 { sgn } else { -sgn };
            // ensure we sweep the short convex way in the correct direction
            if dir > 0.0 {
                while a1 < a0 {
                    a1 += std::f32::consts::TAU;
                }
            } else {
                while a1 > a0 {
                    a1 -= std::f32::consts::TAU;
                }
            }
            let steps = ((fnv as f32) * (a1 - a0).abs() / std::f32::consts::TAU).ceil().max(1.0) as usize;
            for k in 0..=steps {
                let t = a0 + (a1 - a0) * (k as f32 / steps as f32);
                out.push([v[0] + d.abs() * t.cos(), v[1] + d.abs() * t.sin()]);
            }
        } else {
            // Miter: intersection of the two offset edges, else average.
            if let Some(x) = line_intersize(p_in, np, p_out, nn) {
                out.push(x);
            } else {
                out.push([(p_in[0] + p_out[0]) * 0.5, (p_in[1] + p_out[1]) * 0.5]);
            }
        }
    }
    out
}

/// Intersection of the two offset lines through `p_in` (⟂ `n_in`) and `p_out`
/// (⟂ `n_out`) — i.e. lines with directions perpendicular to the given normals.
fn line_intersize(p_in: [f32; 2], n_in: [f32; 2], p_out: [f32; 2], n_out: [f32; 2]) -> Option<[f32; 2]> {
    let d1 = [-n_in[1], n_in[0]]; // direction of the incoming offset edge
    let d2 = [-n_out[1], n_out[0]];
    let denom = d1[0] * d2[1] - d1[1] * d2[0];
    if denom.abs() < 1e-9 {
        return None;
    }
    let t = ((p_out[0] - p_in[0]) * d2[1] - (p_out[1] - p_in[1]) * d2[0]) / denom;
    Some([p_in[0] + d1[0] * t, p_in[1] + d1[1] * t])
}

/// OpenSCAD `offset`: grow (`r`/`delta` > 0) or shrink (< 0) every contour.
/// Outer rings offset by `+d`, holes by `-d` (they follow the opposite winding).
fn offset_shape(sh: &Shape, d: f32, round: bool, fnv: usize) -> Shape {
    let polys = sh
        .polys
        .iter()
        .map(|p| Poly {
            outer: offset_ring(&p.outer, d, round, fnv),
            holes: p.holes.iter().map(|h| offset_ring(h, d, round, fnv)).collect(),
        })
        .collect();
    Shape { polys }
}

/// Slice a triangle mesh with the plane z = 0, returning cross-section polygons
/// (OpenSCAD `projection(cut=true)`). Segments are chained into closed loops.
fn slice_z0(g: &crate::core::BufferGeometry) -> Shape {
    let pos = match g.attributes.get("position") {
        Some(a) => &a.array,
        None => return Shape::default(),
    };
    let vert = |i: usize| [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]];
    let tri_count;
    let idx: Vec<usize> = if let Some(ix) = &g.index {
        tri_count = ix.len() / 3;
        ix.iter().map(|&x| x as usize).collect()
    } else {
        tri_count = pos.len() / 9;
        (0..tri_count * 3).collect()
    };
    let mut segs: Vec<([f32; 2], [f32; 2])> = Vec::new();
    for t in 0..tri_count {
        let v = [vert(idx[t * 3]), vert(idx[t * 3 + 1]), vert(idx[t * 3 + 2])];
        // collect edge crossings of z=0
        let mut hits: Vec<[f32; 2]> = Vec::new();
        for e in 0..3 {
            let (a, b) = (v[e], v[(e + 1) % 3]);
            if (a[2] <= 0.0 && b[2] > 0.0) || (b[2] <= 0.0 && a[2] > 0.0) {
                let t = a[2] / (a[2] - b[2]);
                hits.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
            }
        }
        if hits.len() == 2 {
            segs.push((hits[0], hits[1]));
        }
    }
    chain_segments(&segs)
}

/// Chain undirected segments into closed loops (endpoint matching by quantized
/// key). Loops shorter than 3 points are dropped.
fn chain_segments(segs: &[([f32; 2], [f32; 2])]) -> Shape {
    use std::collections::HashMap;
    let key = |p: [f32; 2]| ((p[0] * 1e4).round() as i64, (p[1] * 1e4).round() as i64);
    let mut adj: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, s) in segs.iter().enumerate() {
        adj.entry(key(s.0)).or_default().push(i);
        adj.entry(key(s.1)).or_default().push(i);
    }
    let mut used = vec![false; segs.len()];
    let mut polys = Vec::new();
    for start in 0..segs.len() {
        if used[start] {
            continue;
        }
        used[start] = true;
        let mut loop_pts = vec![segs[start].0, segs[start].1];
        let mut cur = segs[start].1;
        loop {
            let cands = match adj.get(&key(cur)) {
                Some(c) => c,
                None => break,
            };
            let next = cands.iter().copied().find(|&j| !used[j]);
            let j = match next {
                Some(j) => j,
                None => break,
            };
            used[j] = true;
            let (a, b) = segs[j];
            cur = if key(a) == key(cur) { b } else { a };
            if key(cur) == key(loop_pts[0]) {
                break;
            }
            loop_pts.push(cur);
        }
        if loop_pts.len() >= 3 {
            polys.push(Poly { outer: loop_pts, holes: Vec::new() });
        }
    }
    Shape { polys }
}

// ---------------------------------------------------------------------------
// 2D polygon boolean via arrangement + region classification. Every edge is
// split at all crossings; each sub-edge is classified by testing a point ε to
// its left and right; boundary sub-edges (region on the left) are kept and
// traced into loops. Handles non-convex shapes, holes, coincident/collinear
// edges, and self-intersections — one arrangement pass per n-ary operation.
// ---------------------------------------------------------------------------

type Seg = ([f64; 2], [f64; 2]);

fn ring_edges_into(out: &mut Vec<Seg>, ring: &[[f32; 2]]) {
    let n = ring.len();
    for i in 0..n {
        let a = ring[i];
        let b = ring[(i + 1) % n];
        out.push(([a[0] as f64, a[1] as f64], [b[0] as f64, b[1] as f64]));
    }
}

/// Directed boundary edges of a shape: outer CCW, holes CW (interior on left).
fn shape_edges(s: &Shape) -> Vec<Seg> {
    let mut e = Vec::new();
    for p in &s.polys {
        ring_edges_into(&mut e, &orient_ring(&p.outer, true));
        for h in &p.holes {
            ring_edges_into(&mut e, &orient_ring(h, false));
        }
    }
    e
}

fn cross_side(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

/// Even-odd point membership over an (undirected) edge set.
fn pip_evenodd(p: [f64; 2], edges: &[Seg]) -> bool {
    let mut c = false;
    for (a, b) in edges {
        if (a[1] > p[1]) != (b[1] > p[1]) {
            let x = a[0] + (p[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]);
            if p[0] < x {
                c = !c;
            }
        }
    }
    c
}

/// Nonzero-winding membership over directed edges (for self-union / offset clean).
fn pip_winding(p: [f64; 2], edges: &[Seg]) -> bool {
    let mut w = 0i32;
    for (a, b) in edges {
        if a[1] <= p[1] {
            if b[1] > p[1] && cross_side(*a, *b, p) > 0.0 {
                w += 1;
            }
        } else if b[1] <= p[1] && cross_side(*a, *b, p) < 0.0 {
            w -= 1;
        }
    }
    w != 0
}

/// Intersection points of two segments: proper crossing, endpoint touch, or the
/// interior endpoints of a collinear overlap.
fn seg_points(a1: [f64; 2], a2: [f64; 2], b1: [f64; 2], b2: [f64; 2]) -> Vec<[f64; 2]> {
    let r = [a2[0] - a1[0], a2[1] - a1[1]];
    let s = [b2[0] - b1[0], b2[1] - b1[1]];
    let denom = r[0] * s[1] - r[1] * s[0];
    let qp = [b1[0] - a1[0], b1[1] - a1[1]];
    let mut out = Vec::new();
    if denom.abs() > 1e-12 {
        let t = (qp[0] * s[1] - qp[1] * s[0]) / denom;
        let u = (qp[0] * r[1] - qp[1] * r[0]) / denom;
        if (-1e-9..=1.0 + 1e-9).contains(&t) && (-1e-9..=1.0 + 1e-9).contains(&u) {
            out.push([a1[0] + r[0] * t, a1[1] + r[1] * t]);
        }
    } else if (qp[0] * r[1] - qp[1] * r[0]).abs() < 1e-9 {
        let rr = r[0] * r[0] + r[1] * r[1];
        if rr > 1e-18 {
            for pt in [b1, b2] {
                let tt = ((pt[0] - a1[0]) * r[0] + (pt[1] - a1[1]) * r[1]) / rr;
                if tt > 1e-9 && tt < 1.0 - 1e-9 {
                    out.push(pt);
                }
            }
        }
    }
    out
}

fn qkey(p: [f64; 2], inv: f64) -> (i64, i64) {
    ((p[0] * inv).round() as i64, (p[1] * inv).round() as i64)
}

/// Split every segment at all crossings with the others (O(n²)).
fn split_segments(segs: &[Seg], inv: f64) -> Vec<Seg> {
    let mut out = Vec::new();
    for (i, &(a1, a2)) in segs.iter().enumerate() {
        let r = [a2[0] - a1[0], a2[1] - a1[1]];
        let rr = r[0] * r[0] + r[1] * r[1];
        if rr < 1e-18 {
            continue;
        }
        let mut ts: Vec<f64> = vec![0.0, 1.0];
        for (j, &(b1, b2)) in segs.iter().enumerate() {
            if i == j {
                continue;
            }
            for p in seg_points(a1, a2, b1, b2) {
                let t = ((p[0] - a1[0]) * r[0] + (p[1] - a1[1]) * r[1]) / rr;
                ts.push(t.clamp(0.0, 1.0));
            }
        }
        ts.sort_by(|x, y| x.partial_cmp(y).unwrap());
        for w in ts.windows(2) {
            if w[1] - w[0] < 1e-9 {
                continue;
            }
            let p = [a1[0] + r[0] * w[0], a1[1] + r[1] * w[0]];
            let q = [a1[0] + r[0] * w[1], a1[1] + r[1] * w[1]];
            if qkey(p, inv) != qkey(q, inv) {
                out.push((p, q));
            }
        }
    }
    out
}

/// Clockwise turn angle in [0, 2π) from incoming to outgoing direction.
fn turn_angle(indir: [f64; 2], od: [f64; 2]) -> f64 {
    let mut d = indir[1].atan2(indir[0]) - od[1].atan2(od[0]);
    while d < 0.0 {
        d += std::f64::consts::TAU;
    }
    while d >= std::f64::consts::TAU {
        d -= std::f64::consts::TAU;
    }
    d
}

/// Drop vertices collinear with their neighbours (the arrangement leaves split
/// points on straight edges; keeping them makes T-junctions in extruded caps).
fn simplify_collinear(r: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let n = r.len();
    if n < 3 {
        return r.to_vec();
    }
    let keep: Vec<bool> = (0..n)
        .map(|i| {
            let a = r[(i + n - 1) % n];
            let b = r[i];
            let c = r[(i + 1) % n];
            let cr = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            let s = (b[0] - a[0]).hypot(b[1] - a[1]) * (c[0] - b[0]).hypot(c[1] - b[1]);
            cr.abs() > 1e-7 * s.max(1e-9)
        })
        .collect();
    let out: Vec<[f64; 2]> = (0..n).filter(|&i| keep[i]).map(|i| r[i]).collect();
    if out.len() >= 3 {
        out
    } else {
        r.to_vec()
    }
}

fn ring_area64(r: &[[f64; 2]]) -> f64 {
    let n = r.len();
    let mut a = 0.0;
    for i in 0..n {
        let p = r[i];
        let q = r[(i + 1) % n];
        a += p[0] * q[1] - q[0] * p[1];
    }
    a * 0.5
}

/// Trace directed boundary edges into loops, then nest holes into outers.
fn trace_and_assemble(directed: &[Seg], inv: f64) -> Shape {
    use std::collections::HashMap;
    let mut start_map: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, e) in directed.iter().enumerate() {
        start_map.entry(qkey(e.0, inv)).or_default().push(i);
    }
    let mut used = vec![false; directed.len()];
    let mut loops: Vec<Vec<[f64; 2]>> = Vec::new();
    for s in 0..directed.len() {
        if used[s] {
            continue;
        }
        let start_key = qkey(directed[s].0, inv);
        let mut chain = vec![directed[s].0];
        let mut cur = s;
        let mut guard = directed.len() + 4;
        loop {
            used[cur] = true;
            let end = directed[cur].1;
            let indir = [end[0] - directed[cur].0[0], end[1] - directed[cur].0[1]];
            if qkey(end, inv) == start_key {
                break;
            }
            chain.push(end);
            let cands = match start_map.get(&qkey(end, inv)) {
                Some(c) => c,
                None => break,
            };
            let mut best: Option<usize> = None;
            let mut best_ang = f64::INFINITY;
            for &j in cands {
                if used[j] {
                    continue;
                }
                let od = [directed[j].1[0] - directed[j].0[0], directed[j].1[1] - directed[j].0[1]];
                let ang = turn_angle(indir, od);
                if ang < best_ang {
                    best_ang = ang;
                    best = Some(j);
                }
            }
            match best {
                Some(j) => cur = j,
                None => break,
            }
            guard -= 1;
            if guard == 0 {
                break;
            }
        }
        if chain.len() >= 3 {
            loops.push(chain);
        }
    }
    // Split into outers (CCW) and holes (CW); nest holes into containing outers.
    let mut outers: Vec<Vec<[f64; 2]>> = Vec::new();
    let mut holes: Vec<Vec<[f64; 2]>> = Vec::new();
    for l in loops {
        let l = simplify_collinear(&l);
        let a = ring_area64(&l);
        if a.abs() < 1e-9 || l.len() < 3 {
            continue;
        }
        if a >= 0.0 {
            outers.push(l);
        } else {
            holes.push(l);
        }
    }
    let f32ring = |r: &[[f64; 2]]| -> Vec<[f32; 2]> { r.iter().map(|p| [p[0] as f32, p[1] as f32]).collect() };
    let mut polys: Vec<Poly> =
        outers.iter().map(|o| Poly { outer: f32ring(o), holes: Vec::new() }).collect();
    for h in &holes {
        let hp = h[0];
        for poly in &mut polys {
            let mut oe = Vec::new();
            ring_edges_into(&mut oe, &poly.outer);
            if pip_evenodd(hp, &oe) {
                poly.holes.push(f32ring(h));
                break;
            }
        }
    }
    Shape { polys }
}

/// Extract the region `inside(·)` from the arrangement of `edges`.
fn arrange_extract(edges: &[Seg], inside: &dyn Fn([f64; 2]) -> bool) -> Shape {
    if edges.is_empty() {
        return Shape::default();
    }
    let mut mn = [f64::INFINITY; 2];
    let mut mx = [f64::NEG_INFINITY; 2];
    for (a, b) in edges {
        for p in [a, b] {
            for k in 0..2 {
                mn[k] = mn[k].min(p[k]);
                mx[k] = mx[k].max(p[k]);
            }
        }
    }
    let scale = (mx[0] - mn[0]).max(mx[1] - mn[1]).max(1e-6);
    let eps = scale * 1e-6;
    let inv = 1.0 / (scale * 1e-9);
    let mut seen = std::collections::HashSet::new();
    let mut directed: Vec<Seg> = Vec::new();
    for (p, q) in split_segments(edges, inv) {
        let (kp, kq) = (qkey(p, inv), qkey(q, inv));
        let key = if kp < kq { (kp, kq) } else { (kq, kp) };
        if !seen.insert(key) {
            continue; // coincident sub-edge (shared boundary) — classify once
        }
        let d = [q[0] - p[0], q[1] - p[1]];
        let l = (d[0] * d[0] + d[1] * d[1]).sqrt();
        if l < eps {
            continue;
        }
        let n = [-d[1] / l, d[0] / l];
        let m = [(p[0] + q[0]) * 0.5, (p[1] + q[1]) * 0.5];
        let li = inside([m[0] + n[0] * eps, m[1] + n[1] * eps]);
        let ri = inside([m[0] - n[0] * eps, m[1] - n[1] * eps]);
        if li == ri {
            continue;
        }
        directed.push(if li { (p, q) } else { (q, p) });
    }
    trace_and_assemble(&directed, inv)
}

/// n-ary 2D union.
fn union_shapes(ss: &[Shape]) -> Shape {
    let per: Vec<Vec<Seg>> = ss.iter().map(shape_edges).collect();
    let all: Vec<Seg> = per.iter().flatten().cloned().collect();
    let inside = move |p: [f64; 2]| per.iter().any(|e| pip_evenodd(p, e));
    arrange_extract(&all, &inside)
}

/// 2D difference: `first − rest…`.
fn diff_shapes(first: &Shape, rest: &[Shape]) -> Shape {
    let ef = shape_edges(first);
    let er: Vec<Vec<Seg>> = rest.iter().map(shape_edges).collect();
    let mut all = ef.clone();
    all.extend(er.iter().flatten().cloned());
    let inside = move |p: [f64; 2]| pip_evenodd(p, &ef) && !er.iter().any(|e| pip_evenodd(p, e));
    arrange_extract(&all, &inside)
}

/// n-ary 2D intersection.
fn inter_shapes(ss: &[Shape]) -> Shape {
    let per: Vec<Vec<Seg>> = ss.iter().map(shape_edges).collect();
    let all: Vec<Seg> = per.iter().flatten().cloned().collect();
    let inside = move |p: [f64; 2]| per.iter().all(|e| pip_evenodd(p, e));
    arrange_extract(&all, &inside)
}

/// Resolve self-intersections in a shape (nonzero winding = filled union).
fn simplify2d(s: &Shape) -> Shape {
    let e = shape_edges(s);
    let ec = e.clone();
    let inside = move |p: [f64; 2]| pip_winding(p, &ec);
    arrange_extract(&e, &inside)
}

/// Silhouette of a mesh onto z=0 (OpenSCAD `projection(cut=false)`): the union
/// of all projected triangles, via one arrangement.
fn project_union(g: &crate::core::BufferGeometry) -> Shape {
    let pos = match g.attributes.get("position") {
        Some(a) => &a.array,
        None => return Shape::default(),
    };
    let vert = |i: usize| [pos[i * 3] as f64, pos[i * 3 + 1] as f64];
    let (ntri, idx): (usize, Vec<usize>) = if let Some(ix) = &g.index {
        (ix.len() / 3, ix.iter().map(|&x| x as usize).collect())
    } else {
        (pos.len() / 9, (0..pos.len() / 3).collect())
    };
    let mut tris: Vec<[[f64; 2]; 3]> = Vec::with_capacity(ntri);
    let mut edges = Vec::new();
    for t in 0..ntri {
        let tri = [vert(idx[t * 3]), vert(idx[t * 3 + 1]), vert(idx[t * 3 + 2])];
        // drop degenerate (edge-on) triangles
        let a = (tri[1][0] - tri[0][0]) * (tri[2][1] - tri[0][1])
            - (tri[1][1] - tri[0][1]) * (tri[2][0] - tri[0][0]);
        if a.abs() < 1e-12 {
            continue;
        }
        for k in 0..3 {
            edges.push((tri[k], tri[(k + 1) % 3]));
        }
        tris.push(tri);
    }
    let in_tri = |p: [f64; 2], t: &[[f64; 2]; 3]| {
        let d = |i: usize, j: usize| {
            (t[j][0] - t[i][0]) * (p[1] - t[i][1]) - (t[j][1] - t[i][1]) * (p[0] - t[i][0])
        };
        let (d0, d1, d2) = (d(0, 1), d(1, 2), d(2, 0));
        (d0 >= 0.0 && d1 >= 0.0 && d2 >= 0.0) || (d0 <= 0.0 && d1 <= 0.0 && d2 <= 0.0)
    };
    let inside = move |p: [f64; 2]| tris.iter().any(|t| in_tri(p, t));
    arrange_extract(&edges, &inside)
}

// ---------------------------------------------------------------------------
// text() — glyph outlines from a TrueType font → filled 2D shape
// ---------------------------------------------------------------------------

/// Flatten a glyph's outline into separate contours (its `Path` merges every
/// contour with `move_to`s, so split wherever a curve doesn't continue the last).
fn glyph_contours(shape: &crate::curves::Shape, samples: usize) -> Vec<Vec<[f32; 2]>> {
    let mut out: Vec<Vec<[f32; 2]>> = Vec::new();
    let mut cur: Vec<[f32; 2]> = Vec::new();
    let mut last: Option<[f32; 2]> = None;
    for curve in shape.outline.curve_path.curves.iter() {
        let s = curve.get_point(0.0);
        let s = [s.x, s.y];
        if let Some(l) = last {
            if (l[0] - s[0]).abs() > 1e-3 || (l[1] - s[1]).abs() > 1e-3 {
                if cur.len() >= 3 {
                    out.push(std::mem::take(&mut cur));
                } else {
                    cur.clear();
                }
            }
        }
        if cur.is_empty() {
            cur.push(s);
        }
        for i in 1..=samples {
            let p = curve.get_point(i as f32 / samples as f32);
            cur.push([p.x, p.y]);
        }
        let e = curve.get_point(1.0);
        last = Some([e.x, e.y]);
    }
    if cur.len() >= 3 {
        out.push(cur);
    }
    for h in &shape.holes {
        let pts: Vec<[f32; 2]> = h.get_points(samples).iter().map(|p| [p.x, p.y]).collect();
        if pts.len() >= 3 {
            out.push(pts);
        }
    }
    out
}

/// Fill a set of glyph contours by the non-zero winding rule (preserving each
/// contour's TrueType orientation) into a shape with correctly nested holes.
fn contours_to_shape(contours: &[Vec<[f32; 2]>]) -> Shape {
    let mut edges = Vec::new();
    for c in contours {
        if c.len() >= 3 {
            ring_edges_into(&mut edges, c);
        }
    }
    if edges.is_empty() {
        return Shape::default();
    }
    let ec = edges.clone();
    let inside = move |p: [f64; 2]| pip_winding(p, &ec);
    arrange_extract(&edges, &inside)
}

/// Lay out `text` left-to-right from a font and return the filled 2D outline.
fn text_shape(
    text: &str,
    size: f32,
    font: &crate::TtfFont,
    segments: usize,
    spacing: f32,
    halign: &str,
    valign: &str,
) -> Shape {
    let upm = if font.units_per_em > 0 { font.units_per_em as f32 } else { 1000.0 };
    let scale = size / upm;
    let mut contours: Vec<Vec<[f32; 2]>> = Vec::new();
    let mut pen = 0.0f32;
    for ch in text.chars() {
        let gid = *font.cmap.get(&(ch as u32)).unwrap_or(&0) as usize;
        if let Some(g) = font.glyphs.get(gid) {
            for c in glyph_contours(&g.shape, segments) {
                contours.push(c.iter().map(|p| [p[0] * scale + pen, p[1] * scale]).collect());
            }
            pen += g.advance_width as f32 * scale * spacing;
        }
    }
    let sh = contours_to_shape(&contours);
    if sh.polys.is_empty() {
        return sh;
    }
    // Alignment: shift so the requested anchor sits at the origin.
    let pts = sh.points();
    let (mut mnx, mut mxx, mut mny, mut mxy) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for p in &pts {
        mnx = mnx.min(p[0]);
        mxx = mxx.max(p[0]);
        mny = mny.min(p[1]);
        mxy = mxy.max(p[1]);
    }
    let dx = match halign {
        "center" => -(mnx + mxx) / 2.0,
        "right" => -mxx,
        _ => 0.0,
    };
    let dy = match valign {
        "center" => -(mny + mxy) / 2.0,
        "top" => -mxy,
        "bottom" => -mny,
        _ => 0.0, // baseline
    };
    if dx != 0.0 || dy != 0.0 {
        sh.map(|p| [p[0] + dx, p[1] + dy])
    } else {
        sh
    }
}

/// Evaluate statements to their geometry (implicit union of the children).
fn eval_stmts(stmts: &[Stmt], sc: &Scope, env: &Env, kids: &[Geom]) -> Result<Vec<Geom>, String> {
    let mut geom = Vec::new();
    let mut sc = sc.clone();
    for s in stmts {
        match s {
            Stmt::Assign(n, e) => {
                let v = eval_expr(e, &sc, env)?;
                sc.insert(n.clone(), v);
            }
            Stmt::Block(b) => geom.extend(eval_stmts(b, &sc, env, kids)?),
            Stmt::For(var, range, body) => {
                for item in iterate(&eval_expr(range, &sc, env)?) {
                    let mut inner = sc.clone();
                    inner.insert(var.clone(), item);
                    geom.extend(eval_stmts(body, &inner, env, kids)?);
                }
            }
            Stmt::If(cond, then, els) => {
                let branch = if eval_expr(cond, &sc, env)?.truthy() { then } else { els };
                geom.extend(eval_stmts(branch, &sc, env, kids)?);
            }
            Stmt::ModuleDef(..) | Stmt::FunctionDef(..) => {} // hoisted below
            Stmt::Call(name, args, children) => {
                if let Some(g) = eval_call_stmt(name, args, children, &sc, env, kids)? {
                    geom.push(g);
                }
            }
            Stmt::Modified(mods, inner) => {
                // `*` (disable) and `%` (background) contribute no geometry to the
                // model — so they also drop out of a parent boolean's children.
                if mods.contains('*') || mods.contains('%') {
                    continue;
                }
                // `#` (highlight) is a no-op for geometry; `!` (show-only) evaluates
                // normally but also registers as the render root (see `parse_scad_in`).
                let g = eval_stmts(std::slice::from_ref(inner.as_ref()), &sc, env, kids)?;
                if mods.contains('!') {
                    ROOT_MOD.with(|r| r.borrow_mut().get_or_insert_with(Vec::new).extend(g.iter().cloned()));
                }
                geom.extend(g);
            }
        }
    }
    Ok(geom)
}

thread_local! {
    /// Geometry marked with the `!` (show-only / root) modifier. When any exists,
    /// the top level renders *only* these, ignoring the rest of the program.
    static ROOT_MOD: std::cell::RefCell<Option<Vec<Geom>>> = const { std::cell::RefCell::new(None) };

    /// Running state for unseeded `rands()` — a self-contained xorshift64 so results
    /// are reproducible within a run and identical on wasm (no OS entropy needed).
    static RAND_STATE: std::cell::Cell<u64> = const { std::cell::Cell::new(0x2545_F491_4F6C_DD1D) };
}

fn iterate(v: &Value) -> Vec<Value> {
    match v {
        Value::Range(a, s, b) => {
            let mut out = Vec::new();
            if *s == 0.0 {
                return out;
            }
            let mut x = *a;
            while (*s > 0.0 && x <= *b + 1e-9) || (*s < 0.0 && x >= *b - 1e-9) {
                out.push(Value::Num(x));
                x += *s;
            }
            out
        }
        Value::Vector(items) => items.clone(),
        _ => vec![v.clone()],
    }
}

/// Combine children by implicit union.
/// Split children into solids and 2D shapes (booleans never mix the two).
fn split_geoms(children: Vec<Geom>) -> (Vec<Solid>, Vec<Shape>) {
    let mut solids = Vec::new();
    let mut shapes = Vec::new();
    for g in children {
        match g {
            Geom::Solid(s) => solids.push(s),
            Geom::Shape(sh) => shapes.push(sh),
        }
    }
    (solids, shapes)
}

fn geom_union(children: Vec<Geom>) -> Option<Geom> {
    let (solids, shapes) = split_geoms(children);
    if !solids.is_empty() {
        let mut it = solids.into_iter();
        let mut acc = it.next().unwrap();
        for s in it {
            acc = acc.union(s);
        }
        Some(Geom::Solid(acc))
    } else if !shapes.is_empty() {
        let u = union_shapes(&shapes);
        (!u.polys.is_empty()).then_some(Geom::Shape(u))
    } else {
        None
    }
}

fn geom_difference(children: Vec<Geom>) -> Option<Geom> {
    let (solids, shapes) = split_geoms(children);
    if !solids.is_empty() {
        let mut it = solids.into_iter();
        let mut acc = it.next()?;
        for s in it {
            acc = acc.difference(s);
        }
        Some(Geom::Solid(acc))
    } else if let Some((first, rest)) = shapes.split_first() {
        let d = diff_shapes(first, rest);
        (!d.polys.is_empty()).then_some(Geom::Shape(d))
    } else {
        None
    }
}

fn geom_intersection(children: Vec<Geom>) -> Option<Geom> {
    let (solids, shapes) = split_geoms(children);
    if !solids.is_empty() {
        let mut it = solids.into_iter();
        let mut acc = it.next()?;
        for s in it {
            acc = acc.intersection(s);
        }
        Some(Geom::Solid(acc))
    } else if !shapes.is_empty() {
        let x = inter_shapes(&shapes);
        (!x.polys.is_empty()).then_some(Geom::Shape(x))
    } else {
        None
    }
}

fn geom_map(g: Geom, solid: impl FnOnce(Solid) -> Solid, pt: impl Fn([f32; 2]) -> [f32; 2] + Copy) -> Geom {
    match g {
        Geom::Solid(s) => Geom::Solid(solid(s)),
        Geom::Shape(sh) => Geom::Shape(sh.map(pt)),
    }
}

pub(super) fn solid_points(s: &Solid) -> Vec<[f32; 3]> {
    let g = s.clone().to_geometry();
    match g.attributes.get("position") {
        Some(a) => a.array.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
        None => Vec::new(),
    }
}

/// Is a solid convex? True when *every* vertex lies on its convex hull — a
/// tessellation-robust test (a faceted sphere/cylinder is convex; any reentrant
/// corner is not). Independent of facet count, unlike a volume comparison.
fn solid_is_convex(s: &Solid) -> bool {
    let g = s.clone().to_geometry();
    let pts: Vec<[f32; 3]> = match g.attributes.get("position") {
        Some(a) => a.array.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
        None => return true,
    };
    if pts.len() < 4 {
        return true;
    }
    let key = |p: &[f32; 3]| {
        ((p[0] * 1e4).round() as i64, (p[1] * 1e4).round() as i64, (p[2] * 1e4).round() as i64)
    };
    let hull = match hull3d(&pts) {
        Some(h) => h.to_geometry(),
        None => return true,
    };
    let hull_verts: std::collections::HashSet<(i64, i64, i64)> = match hull.attributes.get("position") {
        Some(a) => a.array.chunks_exact(3).map(|c| key(&[c[0], c[1], c[2]])).collect(),
        None => return true,
    };
    // Convex ⟺ no input vertex is strictly interior (all are hull vertices).
    pts.iter().all(|p| hull_verts.contains(&key(p)))
}

fn eval_call_stmt(
    name: &str,
    args: &[Arg],
    children: &[Stmt],
    sc: &Scope,
    env: &Env,
    mod_kids: &[Geom],
) -> Result<Option<Geom>, String> {
    // Circle resolution follows OpenSCAD's `$fn`/`$fa`/`$fs` rule, all three
    // dynamically scoped (explicit call arg wins, else the enclosing scope, else
    // the default). With `$fn` set (≥3) it's used verbatim; otherwise the count
    // is `ceil(max(min(360/$fa, 2πr/$fs), 5))`, i.e. radius-dependent.
    let special = |nm: &str, def: f64| -> f64 {
        arg(args, nm, 9999, sc, env)
            .and_then(|v| v.num().ok())
            .or_else(|| sc.get(nm).and_then(|v| v.num().ok()))
            .unwrap_or(def)
    };
    let dfn = special("$fn", 0.0);
    let dfa = special("$fa", 12.0).max(0.01);
    let dfs = special("$fs", 2.0).max(0.01);
    let frags = |r: f32| -> usize {
        if dfn >= 3.0 {
            return dfn as usize;
        }
        let r = (r.abs() as f64).max(0.0);
        if r < 1e-5 {
            return 3;
        }
        (((360.0 / dfa).min(r * 2.0 * std::f64::consts::PI / dfs)).ceil() as usize).max(5)
    };
    // The call's own block is evaluated in the *current* children context, so a
    // `children()` inside it resolves to the enclosing module's children.
    let kids = || eval_stmts(children, sc, env, mod_kids);
    let child = || -> Result<Option<Geom>, String> { Ok(geom_union(eval_stmts(children, sc, env, mod_kids)?)) };

    Ok(match name {
        // --- 2D primitives ---
        "square" => {
            let size = arg(args, "size", 0, sc, env).unwrap_or(Value::Num(1.0));
            let s = match &size {
                Value::Num(n) => [*n as f32; 2],
                _ => {
                    let v = size.vec3(1.0)?;
                    [v[0], v[1]]
                }
            };
            let center = arg(args, "center", 1, sc, env).map(|v| v.truthy()).unwrap_or(false);
            Some(Geom::Shape(square(s, center)))
        }
        "circle" => {
            let r = radius(args, "r", "d", 0, sc, env).unwrap_or(1.0);
            Some(Geom::Shape(circle(r, frags(r))))
        }
        "polygon" => {
            let pts = as_points2(&arg(args, "points", 0, sc, env).unwrap_or(Value::Undef));
            let paths = arg(args, "paths", 1, sc, env);
            Some(Geom::Shape(polygon_shape(pts, paths)))
        }
        // --- 3D primitives ---
        "cube" => {
            let size = arg(args, "size", 0, sc, env).unwrap_or(Value::Num(1.0));
            let s = if let Value::Num(n) = &size { [*n as f32; 3] } else { size.vec3(1.0)? };
            let center = arg(args, "center", 1, sc, env).map(|v| v.truthy()).unwrap_or(false);
            let c = cube([s[0], s[1], s[2]]);
            Some(Geom::Solid(if center { c } else { c.translate([s[0] / 2.0, s[1] / 2.0, s[2] / 2.0]) }))
        }
        "sphere" => {
            let r = radius(args, "r", "d", 0, sc, env).unwrap_or(1.0);
            Some(Geom::Solid(sphere_mesh(r, frags(r))))
        }
        "cylinder" => {
            let h = arg(args, "h", 0, sc, env).and_then(|v| v.num().ok()).unwrap_or(1.0) as f32;
            let center = arg(args, "center", 999, sc, env).map(|v| v.truthy()).unwrap_or(false);
            let (r1, r2) = cyl_radii(args, sc, env);
            Some(Geom::Solid(cyl_mesh(r1, r2, h, frags(r1.max(r2)), center)))
        }
        "polyhedron" => {
            let pts = as_points(&arg(args, "points", 0, sc, env).unwrap_or(Value::Undef));
            let faces = as_faces(&arg(args, "faces", 1, sc, env).unwrap_or(Value::Undef));
            Some(Geom::Solid(polyhedron(&pts, &faces)))
        }
        // --- extrusions (2D child → 3D) ---
        "linear_extrude" => {
            let h = arg(args, "height", 0, sc, env).and_then(|v| v.num().ok()).unwrap_or(1.0) as f32;
            let center = arg(args, "center", 999, sc, env).map(|v| v.truthy()).unwrap_or(false);
            let twist = arg(args, "twist", 999, sc, env).and_then(|v| v.num().ok()).unwrap_or(0.0) as f32;
            let scale = match arg(args, "scale", 999, sc, env) {
                Some(Value::Num(n)) => [n as f32, n as f32],
                Some(v) => {
                    let a = v.vec3(1.0).unwrap_or([1.0; 3]);
                    [a[0], a[1]]
                }
                None => [1.0, 1.0],
            };
            // Default slice count follows the twist magnitude (a layer every ~4°),
            // clamped; an explicit `slices=` overrides. No twist ⇒ 1 slice suffices.
            let slices = arg(args, "slices", 999, sc, env)
                .and_then(|v| v.num().ok())
                .map(|n| (n as usize).max(1))
                .unwrap_or_else(|| {
                    if twist.abs() > 1e-6 {
                        ((twist.abs() / 4.0).ceil() as usize).clamp(2, 200)
                    } else {
                        1
                    }
                });
            match child()? {
                Some(Geom::Shape(s)) => {
                    linear_extrude_shape(&s, h, center, twist, scale, slices).map(Geom::Solid)
                }
                _ => None,
            }
        }
        "rotate_extrude" => {
            let deg = arg(args, "angle", 0, sc, env).and_then(|v| v.num().ok()).unwrap_or(360.0) as f32;
            match child()? {
                Some(Geom::Shape(s)) => {
                    let rmax = s.points().iter().map(|p| p[0].abs()).fold(0.0f32, f32::max);
                    rotate_extrude_shape(&s, deg.to_radians(), frags(rmax)).map(Geom::Solid)
                }
                _ => None,
            }
        }
        // --- transforms ---
        "translate" => {
            let v = arg(args, "v", 0, sc, env).unwrap_or(Value::Undef).vec3(0.0).unwrap_or([0.0; 3]);
            child()?.map(|g| geom_map(g, |s| s.translate(v), move |p| [p[0] + v[0], p[1] + v[1]]))
        }
        "scale" => {
            let v = arg(args, "v", 0, sc, env).unwrap_or(Value::Num(1.0)).vec3(1.0).unwrap_or([1.0; 3]);
            child()?.map(|g| geom_map(g, |s| s.scale(v), move |p| [p[0] * v[0], p[1] * v[1]]))
        }
        "rotate" => {
            let a = arg(args, "a", 0, sc, env).unwrap_or(Value::Num(0.0));
            let v = match &a {
                Value::Num(deg) => [0.0, 0.0, (*deg as f32).to_radians()],
                _ => {
                    let d = a.vec3(0.0).unwrap_or([0.0; 3]);
                    [d[0].to_radians(), d[1].to_radians(), d[2].to_radians()]
                }
            };
            child()?.map(|g| {
                geom_map(
                    g,
                    |s| s.rotate_x(v[0]).rotate_y(v[1]).rotate_z(v[2]),
                    move |p| {
                        let (c, s2) = (v[2].cos(), v[2].sin());
                        [p[0] * c - p[1] * s2, p[0] * s2 + p[1] * c]
                    },
                )
            })
        }
        "mirror" => {
            let v = arg(args, "v", 0, sc, env).unwrap_or(Value::Undef).vec3(0.0).unwrap_or([1.0, 0.0, 0.0]);
            child()?.map(|g| {
                geom_map(g, |s| s.transform(reflection(v)), move |p| {
                    let l = (v[0] * v[0] + v[1] * v[1]).sqrt().max(1e-8);
                    let (nx, ny) = (v[0] / l, v[1] / l);
                    let d = 2.0 * (p[0] * nx + p[1] * ny);
                    [p[0] - d * nx, p[1] - d * ny]
                })
            })
        }
        "multmatrix" => {
            let m = matrix4_from_value(&arg(args, "m", 0, sc, env).unwrap_or(Value::Undef));
            let e = m.elements;
            child()?.map(|g| {
                geom_map(g, |s| s.transform(m), move |p| {
                    // apply the affine 2×2 + translation to a 2D point
                    [
                        e[0] * p[0] + e[4] * p[1] + e[12],
                        e[1] * p[0] + e[5] * p[1] + e[13],
                    ]
                })
            })
        }
        "import" => {
            let file = match arg(args, "file", 0, sc, env) {
                Some(Value::Str(s)) => s,
                _ => return Err("import() needs a file name".into()),
            };
            let path = env.base.join(&file);
            let lower = file.to_ascii_lowercase();
            let read_err = |e: std::io::Error| format!("import: cannot read {}: {e}", path.display());
            if lower.ends_with(".stl") {
                let bytes = read_file_bytes(&path).map_err(read_err)?;
                Some(Geom::Solid(Solid::Leaf(crate::StlLoader::parse(&bytes))))
            } else if lower.ends_with(".obj") {
                let text = read_file_string(&path).map_err(read_err)?;
                Some(Geom::Solid(Solid::Leaf(crate::ObjLoader::parse(&text))))
            } else if lower.ends_with(".off") {
                let text = read_file_string(&path).map_err(read_err)?;
                Some(Geom::Solid(parse_off(&text)?))
            } else if lower.ends_with(".dxf") {
                let text = read_file_string(&path).map_err(read_err)?;
                let sh = parse_dxf(&text);
                (!sh.polys.is_empty()).then_some(Geom::Shape(sh))
            } else if lower.ends_with(".svg") {
                let text = read_file_string(&path).map_err(read_err)?;
                let sh = parse_svg(&text);
                (!sh.polys.is_empty()).then_some(Geom::Shape(sh))
            } else if lower.ends_with(".3mf") {
                let bytes = read_file_bytes(&path).map_err(read_err)?;
                let solid = parse_3mf(&bytes).ok_or_else(|| format!("import: malformed 3MF '{file}'"))?;
                Some(Geom::Solid(solid))
            } else if lower.ends_with(".amf") {
                let text = read_file_string(&path).map_err(read_err)?;
                let solid = parse_amf(&text).ok_or_else(|| format!("import: malformed AMF '{file}'"))?;
                Some(Geom::Solid(solid))
            } else {
                return Err(format!("import: unsupported format '{file}' (STL/OBJ/OFF/3MF/AMF/DXF/SVG)"));
            }
        }
        "color" => child()?,
        // --- booleans ---
        "union" => child()?,
        "difference" => geom_difference(kids()?),
        "intersection" => geom_intersection(kids()?),
        "hull" => {
            let cg = kids()?;
            if cg.iter().any(|g| matches!(g, Geom::Solid(_))) {
                let mut pts = Vec::new();
                for g in &cg {
                    if let Geom::Solid(s) = g {
                        pts.extend(solid_points(s));
                    }
                }
                hull3d(&pts).map(Geom::Solid)
            } else {
                let mut p2 = Vec::new();
                for g in &cg {
                    if let Geom::Shape(s) = g {
                        p2.extend(s.points());
                    }
                }
                let h = hull2d(p2);
                (h.len() >= 3).then(|| Geom::Shape(Shape::one(h)))
            }
        }
        "echo" => {
            let parts: Vec<String> = args
                .iter()
                .map(|a| {
                    let v = eval_expr(&a.value, sc, env).unwrap_or(Value::Undef);
                    match &a.name {
                        Some(n) => format!("{n} = {}", fmt_value(&v)),
                        None => fmt_value(&v),
                    }
                })
                .collect();
            println!("ECHO: {}", parts.join(", "));
            child()?
        }
        "assert" => {
            let ok = arg(args, "condition", 0, sc, env).map(|v| v.truthy()).unwrap_or(true);
            if !ok {
                return Err("assertion failed".into());
            }
            child()?
        }
        // `render` / `let` / grouping wrappers — evaluate the children as-is.
        "render" | "group" => child()?,
        // `children()` / `children(i)` / `children([i:j])` / `children([a,b])` —
        // resolve against the enclosing module's children (works anywhere, incl.
        // inside `for`/`if`, because the context is threaded through evaluation).
        "children" => {
            let mut sel: Vec<Geom> = Vec::new();
            let pick = |i: f64, out: &mut Vec<Geom>| {
                if let Some(g) = mod_kids.get(i as usize) {
                    out.push(g.clone());
                }
            };
            if args.is_empty() {
                sel.extend(mod_kids.iter().cloned());
            } else {
                match eval_expr(&args[0].value, sc, env)? {
                    Value::Num(i) => pick(i, &mut sel),
                    Value::Range(a, s, b) => {
                        for it in iterate(&Value::Range(a, s, b)) {
                            if let Ok(i) = it.num() {
                                pick(i, &mut sel);
                            }
                        }
                    }
                    Value::Vector(items) => {
                        for it in items {
                            if let Ok(i) = it.num() {
                                pick(i, &mut sel);
                            }
                        }
                    }
                    _ => {}
                }
            }
            geom_union(sel)
        }
        // `let(a=…) child` binds in a new scope; `assign(a=…)` is the deprecated
        // synonym — both just introduce bindings for the child geometry.
        "let" | "assign" => {
            let mut inner = sc.clone();
            for a in args {
                if let Some(nm) = &a.name {
                    inner.insert(nm.clone(), eval_expr(&a.value, sc, env)?);
                }
            }
            geom_union(eval_stmts(children, &inner, env, mod_kids)?)
        }
        "intersection_for" => {
            // Loop like `for`, but intersect the per-iteration geometry.
            let mut acc: Option<Geom> = None;
            for a in args {
                let var = a.name.clone().unwrap_or_default();
                for item in iterate(&eval_expr(&a.value, sc, env)?) {
                    let mut inner = sc.clone();
                    inner.insert(var.clone(), item);
                    if let Some(g) = geom_union(eval_stmts(children, &inner, env, mod_kids)?) {
                        acc = Some(match acc {
                            Some(prev) => geom_intersection(vec![prev, g])
                                .unwrap_or_else(|| Geom::Shape(Shape::default())),
                            None => g,
                        });
                    }
                }
            }
            acc
        }
        "resize" => {
            let nv = arg(args, "newsize", 0, sc, env).unwrap_or(Value::Undef).vec3(0.0).unwrap_or([0.0; 3]);
            let auto = match arg(args, "auto", 1, sc, env) {
                Some(Value::Vector(v)) => {
                    let g = |i: usize| v.get(i).map(|x| x.truthy()).unwrap_or(false);
                    [g(0), g(1), g(2)]
                }
                Some(other) => [other.truthy(); 3],
                None => [false; 3],
            };
            child()?.map(|g| resize_geom(g, nv, auto))
        }
        "offset" => {
            let r = arg(args, "r", 0, sc, env).and_then(|v| v.num().ok());
            let delta = arg(args, "delta", 999, sc, env).and_then(|v| v.num().ok());
            let chamfer = arg(args, "chamfer", 999, sc, env).map(|v| v.truthy()).unwrap_or(false);
            let (d, round) = match (r, delta) {
                (Some(r), _) => (r as f32, !chamfer),
                (None, Some(dl)) => (dl as f32, false),
                (None, None) => (1.0, true),
            };
            match child()? {
                // Per-contour offset, then resolve any self-intersection it created.
                Some(Geom::Shape(s)) => {
                    let off = offset_shape(&s, d, round, frags(d));
                    let clean = simplify2d(&off);
                    Some(Geom::Shape(if clean.polys.is_empty() { off } else { clean }))
                }
                other => other, // offset of a 3D solid is undefined in OpenSCAD
            }
        }
        "fill" => {
            // fill() removes interior holes: union the 2D children, then drop
            // every hole so each region becomes solid. (3D children pass through.)
            match child()? {
                Some(Geom::Shape(s)) => {
                    let filled = Shape {
                        polys: s
                            .polys
                            .into_iter()
                            .map(|p| Poly { outer: p.outer, holes: Vec::new() })
                            .collect(),
                    };
                    Some(Geom::Shape(filled))
                }
                other => other,
            }
        }
        "minkowski" => {
            let cg = kids()?;
            if cg.iter().any(|g| matches!(g, Geom::Solid(_))) {
                // 3D Minkowski = convex hull of pairwise vertex sums — exact only
                // when both operands are convex. A general non-convex 3D sum needs
                // convex decomposition + 3D union (the CSG kernel); rather than
                // silently return the (wrong) convex hull, reject it clearly.
                let mut sets: Vec<Vec<[f32; 3]>> = Vec::new();
                for g in &cg {
                    match g {
                        Geom::Solid(s) => {
                            if !solid_is_convex(s) {
                                return Err("minkowski: non-convex 3D operands are not supported \
                                    (only convex solids like cube/sphere/cylinder)"
                                    .into());
                            }
                            sets.push(solid_points(s));
                        }
                        Geom::Shape(s) => {
                            sets.push(s.points().iter().map(|p| [p[0], p[1], 0.0]).collect())
                        }
                    }
                }
                let mut it = sets.into_iter();
                let mut acc = match it.next() {
                    Some(x) => x,
                    None => return Ok(None),
                };
                for s in it {
                    acc = minkowski_sum3(&acc, &s);
                }
                hull3d(&acc).map(Geom::Solid)
            } else {
                // 2D: exact for non-convex operands too — triangulate each shape,
                // convex-Minkowski every triangle pair (hull of vertex sums), and
                // union the pieces via the arrangement kernel.
                let shapes: Vec<&Shape> = cg
                    .iter()
                    .filter_map(|g| match g {
                        Geom::Shape(s) => Some(s),
                        _ => None,
                    })
                    .collect();
                let mut it = shapes.into_iter();
                let mut acc = match it.next() {
                    Some(x) => x.clone(),
                    None => return Ok(None),
                };
                for s in it {
                    acc = minkowski2d(&acc, s);
                }
                (!acc.polys.is_empty()).then_some(Geom::Shape(acc))
            }
        }
        "projection" => {
            let cut = arg(args, "cut", 0, sc, env).map(|v| v.truthy()).unwrap_or(false);
            match child()? {
                Some(Geom::Solid(s)) if cut => {
                    let sh = slice_z0(&s.to_geometry_exact());
                    (!sh.polys.is_empty()).then_some(Geom::Shape(sh))
                }
                Some(Geom::Solid(s)) => {
                    let sh = project_union(&s.to_geometry_exact());
                    (!sh.polys.is_empty()).then_some(Geom::Shape(sh))
                }
                other => other,
            }
        }
        "text" => {
            let txt = match arg(args, "text", 0, sc, env) {
                Some(Value::Str(s)) => s,
                Some(other) => fmt_value(&other),
                None => return Err("text() needs a string".into()),
            };
            let size = arg(args, "size", 999, sc, env).and_then(|v| v.num().ok()).unwrap_or(10.0) as f32;
            let spacing = arg(args, "spacing", 999, sc, env).and_then(|v| v.num().ok()).unwrap_or(1.0) as f32;
            let str_arg = |nm: &str, def: &str| match arg(args, nm, 999, sc, env) {
                Some(Value::Str(s)) => s,
                _ => def.to_string(),
            };
            let (halign, valign) = (str_arg("halign", "left"), str_arg("valign", "baseline"));
            // Font: an explicit `font="…ttf"` path, else the bundled default.
            let font = match arg(args, "font", 999, sc, env) {
                Some(Value::Str(f)) if f.to_ascii_lowercase().ends_with(".ttf") || f.to_ascii_lowercase().ends_with(".otf") => {
                    let bytes = read_file_bytes(&env.base.join(&f))
                        .map_err(|e| format!("text: cannot read font {f}: {e}"))?;
                    crate::TtfFont::parse(&bytes).map_err(|_| format!("text: cannot parse font {f}"))?
                }
                _ => crate::TtfFont::parse(DEFAULT_FONT).map_err(|_| "text: bundled font failed".to_string())?,
            };
            let segs = frags(size).clamp(4, 24);
            let sh = text_shape(&txt, size, &font, segs, spacing, &halign, &valign);
            (!sh.polys.is_empty()).then_some(Geom::Shape(sh))
        }
        "surface" => {
            let file = match arg(args, "file", 0, sc, env) {
                Some(Value::Str(s)) => s,
                _ => return Err("surface() needs a file name".into()),
            };
            let center = arg(args, "center", 999, sc, env).map(|v| v.truthy()).unwrap_or(false);
            let path = env.base.join(&file);
            let lower = file.to_ascii_lowercase();
            if lower.ends_with(".png") {
                let bytes = read_file_bytes(&path)
                    .map_err(|e| format!("surface: cannot read {}: {e}", path.display()))?;
                let grid = decode_png_luma(&bytes)
                    .ok_or("surface: unsupported PNG (need 8/16-bit greyscale/RGB, non-interlaced)")?;
                Some(Geom::Solid(surface_grid(grid, center)?))
            } else if lower.ends_with(".dat") {
                let text = read_file_string(&path)
                    .map_err(|e| format!("surface: cannot read {}: {e}", path.display()))?;
                Some(Geom::Solid(surface_dat(&text, center)?))
            } else {
                return Err("surface: only .dat and .png heightmaps are supported".into());
            }
        }
        // --- user module ---
        _ => {
            if let Some((params, body)) = env.modules.get(name) {
                let _g = enter_recursion()?;
                let mut inner = bind_args(params, args, sc, env)?;
                // The block passed to this call becomes the module's children,
                // evaluated in the *caller's* children context.
                let child_geom = kids()?;
                inner.insert("$children".into(), Value::Num(child_geom.len() as f64));
                geom_union(eval_stmts(body, &inner, env, &child_geom)?)
            } else {
                return Err(format!("unknown module '{name}'"));
            }
        }
    })
}

fn as_points2(v: &Value) -> Vec<[f32; 2]> {
    match v {
        Value::Vector(items) => items
            .iter()
            .map(|p| {
                let c = p.vec3(0.0).unwrap_or([0.0; 3]);
                [c[0], c[1]]
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn polygon_shape(pts: Vec<[f32; 2]>, paths: Option<Value>) -> Shape {
    match paths {
        Some(Value::Vector(rings)) if !rings.is_empty() => {
            let ring_of = |r: &Value| -> Vec<[f32; 2]> {
                match r {
                    Value::Vector(idx) => idx
                        .iter()
                        .filter_map(|x| x.num().ok().and_then(|n| pts.get(n as usize).copied()))
                        .collect(),
                    _ => Vec::new(),
                }
            };
            let outer = ring_of(&rings[0]);
            let holes = rings[1..].iter().map(ring_of).collect();
            Shape { polys: vec![Poly { outer, holes }] }
        }
        _ => Shape::one(pts),
    }
}

fn radius(args: &[Arg], rn: &str, dn: &str, pos: usize, sc: &Scope, env: &Env) -> Option<f32> {
    if let Some(d) = arg(args, dn, 9999, sc, env).and_then(|v| v.num().ok()) {
        return Some(d as f32 / 2.0);
    }
    arg(args, rn, pos, sc, env).and_then(|v| v.num().ok()).map(|r| r as f32)
}

fn cyl_radii(args: &[Arg], sc: &Scope, env: &Env) -> (f32, f32) {
    if let Some(r) = radius(args, "r", "d", 1, sc, env) {
        return (r, r);
    }
    let r1 = radius(args, "r1", "d1", 999, sc, env).unwrap_or(1.0);
    let r2 = radius(args, "r2", "d2", 999, sc, env).unwrap_or(1.0);
    (r1, r2)
}

fn as_points(v: &Value) -> Vec<[f32; 3]> {
    match v {
        Value::Vector(items) => items.iter().map(|p| p.vec3(0.0).unwrap_or([0.0; 3])).collect(),
        _ => Vec::new(),
    }
}
fn as_faces(v: &Value) -> Vec<Vec<u32>> {
    match v {
        Value::Vector(items) => items
            .iter()
            .filter_map(|f| match f {
                Value::Vector(idx) => Some(idx.iter().filter_map(|x| x.num().ok().map(|n| n as u32)).collect()),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// Reflection matrix across the plane through the origin with normal `n`.
fn reflection(n: [f32; 3]) -> Matrix4 {
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-8);
    let (x, y, z) = (n[0] / l, n[1] / l, n[2] / l);
    let mut m = Matrix4::identity();
    m.elements = [
        1.0 - 2.0 * x * x, -2.0 * x * y, -2.0 * x * z, 0.0,
        -2.0 * y * x, 1.0 - 2.0 * y * y, -2.0 * y * z, 0.0,
        -2.0 * z * x, -2.0 * z * y, 1.0 - 2.0 * z * z, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    m
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Parse and evaluate OpenSCAD source into a [`Solid`] (the implicit union of the
/// program's top-level geometry). Returns an error string on lex/parse failure.
/// Strip `include <…>` / `use <…>` directives from a line-based source, returning
/// the cleaned source and the list of `(is_use, path)` references (in order).
fn extract_directives(src: &str) -> (String, Vec<(bool, String)>) {
    let mut cleaned = String::new();
    let mut dirs = Vec::new();
    for line in src.lines() {
        let t = line.trim_start();
        let kw = if t.starts_with("include") {
            Some((false, 7))
        } else if t.starts_with("use") {
            Some((true, 3))
        } else {
            None
        };
        if let Some((is_use, off)) = kw {
            let rest = t[off..].trim_start();
            if let Some(open) = rest.find('<') {
                if let Some(close) = rest[open + 1..].find('>') {
                    let path = rest[open + 1..open + 1 + close].trim().to_string();
                    dirs.push((is_use, path));
                    continue; // drop the directive line
                }
            }
        }
        cleaned.push_str(line);
        cleaned.push('\n');
    }
    (cleaned, dirs)
}

/// Recursively inline `include <…>` files (textual) and collect `use <…>` paths.
fn inline_includes(
    src: &str,
    base: &std::path::Path,
    uses: &mut Vec<std::path::PathBuf>,
    seen: &mut std::collections::HashSet<std::path::PathBuf>,
) -> Result<String, String> {
    let (cleaned, dirs) = extract_directives(src);
    let mut out = String::new();
    for (is_use, path) in dirs {
        let full = base.join(&path);
        if is_use {
            uses.push(full);
            continue;
        }
        let canon = full.canonicalize().unwrap_or_else(|_| full.clone());
        if !seen.insert(canon) {
            continue; // include cycle — skip
        }
        let text = read_file_string(&full)
            .map_err(|e| format!("include: cannot read {}: {e}", full.display()))?;
        let child_base = full.parent().unwrap_or(base).to_path_buf();
        out.push_str(&inline_includes(&text, &child_base, uses, seen)?);
        out.push('\n');
    }
    out.push_str(&cleaned);
    Ok(out)
}

/// Hoist a program's `module`/`function` definitions into `env`.
fn hoist_defs(program: &[Stmt], env: &mut Env) {
    for s in program {
        match s {
            Stmt::ModuleDef(name, params, body) => {
                env.modules.insert(name.clone(), (params.clone(), body.clone()));
            }
            Stmt::FunctionDef(name, params, body) => {
                env.functions.insert(name.clone(), (params.clone(), body.clone()));
            }
            _ => {}
        }
    }
}

fn parse_program(src: &str) -> Result<Vec<Stmt>, String> {
    let toks = lex(src)?;
    Parser { t: toks, i: 0 }.program()
}

/// Parse OpenSCAD source, resolving `include`/`use`/`import` relative to `base`.
fn parse_scad_in(src: &str, base: &std::path::Path) -> Result<Solid, String> {
    let mut uses = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let combined = inline_includes(src, base, &mut uses, &mut seen)?;
    let program = parse_program(&combined)?;

    let mut env = Env { base: base.to_path_buf(), ..Env::default() };
    // `use` brings in only the definitions of the referenced files (no geometry).
    for up in &uses {
        if let Ok(text) = read_file_string(up) {
            let ubase = up.parent().unwrap_or(base).to_path_buf();
            let mut u2 = Vec::new();
            let mut s2 = std::collections::HashSet::new();
            if let Ok(uc) = inline_includes(&text, &ubase, &mut u2, &mut s2) {
                if let Ok(up_prog) = parse_program(&uc) {
                    hoist_defs(&up_prog, &mut env);
                }
            }
        }
    }
    hoist_defs(&program, &mut env);

    // Evaluate on a large stack so deep (but finite) recursive functions/modules
    // don't overflow — the recursion guard turns runaway recursion into an error.
    run_eval(move || {
        ROOT_MOD.with(|r| *r.borrow_mut() = None);
        let geom = eval_stmts(&program, &Scope::new(), &env, &[])?;
        // A `!` (show-only) modifier anywhere overrides the output with just its subtree.
        let geom = ROOT_MOD
            .with(|r| r.borrow_mut().take())
            .filter(|v| !v.is_empty())
            .unwrap_or(geom);
        match geom_union(geom) {
            Some(Geom::Solid(s)) => Ok(s),
            Some(Geom::Shape(_)) => {
                Err("program produced 2D geometry — wrap it in linear_extrude/rotate_extrude".into())
            }
            None => Err("program produced no geometry".into()),
        }
    })
}

/// Run scad evaluation on a 1 GB-stack worker thread (recursion can be deep).
/// On wasm (no threads) it runs inline.
#[cfg(not(target_arch = "wasm32"))]
fn run_eval(f: impl FnOnce() -> Result<Solid, String> + Send + 'static) -> Result<Solid, String> {
    std::thread::Builder::new()
        .name("scad-eval".into())
        .stack_size(1024 * 1024 * 1024)
        .spawn(f)
        .expect("spawn scad-eval thread")
        .join()
        .unwrap_or_else(|_| Err("evaluation failed (stack overflow or panic)".into()))
}
#[cfg(target_arch = "wasm32")]
fn run_eval(f: impl FnOnce() -> Result<Solid, String>) -> Result<Solid, String> {
    f()
}

/// Parse OpenSCAD source text. `include`/`use`/`import` paths resolve against the
/// current directory — use `parse_scad_file` to resolve against a file's folder.
pub fn parse_scad(src: &str) -> Result<Solid, String> {
    parse_scad_in(src, std::path::Path::new("."))
}

/// Parse an OpenSCAD `.scad` file, resolving `include`/`use`/`import` references
/// relative to the file's own directory.
pub fn parse_scad_file(path: impl AsRef<std::path::Path>) -> Result<Solid, String> {
    let path = path.as_ref();
    let src = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let base = path.parent().unwrap_or(std::path::Path::new("."));
    parse_scad_in(&src, base)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vol(s: Solid) -> f64 {
        vol_geom(&s.to_geometry())
    }
    fn vol_geom(g: &crate::BufferGeometry) -> f64 {
        let pos = &g.attributes.get("position").unwrap().array;
        let v = |i: usize| [pos[i * 3] as f64, pos[i * 3 + 1] as f64, pos[i * 3 + 2] as f64];
        let tri = |a: [f64; 3], b: [f64; 3], c: [f64; 3]| {
            a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2])
                + a[2] * (b[0] * c[1] - b[1] * c[0])
        };
        let mut s = 0.0;
        if let Some(idx) = &g.index {
            for t in idx.chunks_exact(3) {
                s += tri(v(t[0] as usize), v(t[1] as usize), v(t[2] as usize));
            }
        } else {
            for k in 0..pos.len() / 9 {
                s += tri(v(k * 3), v(k * 3 + 1), v(k * 3 + 2));
            }
        }
        (s / 6.0).abs()
    }

    #[test]
    fn cube_corner_origin() {
        // OpenSCAD cube(10) spans [0,10]^3 → volume 1000.
        assert!((vol(parse_scad("cube(10);").unwrap()) - 1000.0).abs() < 1e-2);
    }

    #[test]
    fn difference_and_translate() {
        // The box_slot corpus model.
        let src = "difference() { cube(10); translate([3,3,-1]) cube([4,4,12]); }";
        assert!((vol(parse_scad(src).unwrap()) - 840.0).abs() < 1e-1);
    }

    #[test]
    fn for_loop_union() {
        // Three unit cubes in a row → volume 3.
        let src = "for (i = [0:2]) translate([i*2, 0, 0]) cube(1);";
        assert!((vol(parse_scad(src).unwrap()) - 3.0).abs() < 1e-3);
    }

    #[test]
    fn module_and_expr() {
        let src = "module post(n) { for(i=[0:n-1]) translate([i*3,0,0]) cube([1,1,5]); } post(3);";
        assert!((vol(parse_scad(src).unwrap()) - 15.0).abs() < 1e-2);
    }

    #[test]
    fn function_def() {
        let src = "function d() = 2+3; cube(d());";
        assert!((vol(parse_scad(src).unwrap()) - 125.0).abs() < 1e-1);
    }

    #[test]
    fn if_else_and_ternary() {
        let src = "n = 3; if (n > 2) cube([2, 5, 1]); else cube(1);";
        assert!((vol(parse_scad(src).unwrap()) - 10.0).abs() < 1e-3);
        // ternary in an expression
        assert!((vol(parse_scad("m = 5; cube(m > 0 ? 2 : 1);").unwrap()) - 8.0).abs() < 1e-3);
    }

    #[test]
    fn rotate_preserves_volume() {
        // A rotated box has the same volume as an unrotated one.
        let r = vol(parse_scad("rotate([30, 40, 50]) cube([2, 3, 4], center = true);").unwrap());
        assert!((r - 24.0).abs() < 1e-2);
    }

    #[test]
    fn linear_extrude_square() {
        // 2D square extruded → a box. 4×3×2 = 24.
        assert!((vol(parse_scad("linear_extrude(2) square([4, 3]);").unwrap()) - 24.0).abs() < 1e-2);
    }

    #[test]
    fn extruded_annulus() {
        // 2D difference of circles, extruded → a tube: π(25−9)·2.
        let src = "linear_extrude(2) difference() { circle(5, $fn = 64); circle(3, $fn = 64); }";
        let v = vol(parse_scad(src).unwrap());
        let want = std::f64::consts::PI * (25.0 - 9.0) * 2.0;
        assert!((v - want).abs() / want < 0.02, "annulus {v} vs {want}");
    }

    #[test]
    fn rotate_extrude_washer() {
        // Square profile at radius [2,3], height 1, revolved 360° → washer: π(9−4)·1.
        let src = "rotate_extrude($fn = 64) translate([2, 0, 0]) square([1, 1]);";
        let v = vol(parse_scad(src).unwrap());
        let want = std::f64::consts::PI * (9.0 - 4.0);
        assert!((v - want).abs() / want < 0.03, "washer {v} vs {want}");
    }

    #[test]
    fn list_comprehension() {
        // Generate positions with a comprehension, place a cube at each.
        let src = "for (p = [for (i = [0:2]) [i * 2, 0, 0]]) translate(p) cube(1);";
        assert!((vol(parse_scad(src).unwrap()) - 3.0).abs() < 1e-3);
    }

    #[test]
    fn hull_of_two_cubes_is_the_bounding_box() {
        // Convex hull of two unit cubes offset in x → the [0,4]×[0,1]×[0,1] box, vol 4.
        let src = "hull() { cube(1); translate([3, 0, 0]) cube(1); }";
        let v = vol(parse_scad(src).unwrap());
        assert!((v - 4.0).abs() < 0.1, "hull vol {v}");
    }

    #[test]
    fn named_and_default_args() {
        // sphere via diameter; cube via named size; module default param.
        assert!(vol(parse_scad("sphere(d = 2, $fn = 32);").unwrap()) > 3.0); // ~4/3π ≈ 4.19
        // default $fn=0 → coarse sphere from $fa/$fs (OpenSCAD gives 5 fragments).
        assert!(vol(parse_scad("sphere(d = 2);").unwrap()) < 3.5);
        let src = "module slab(t = 2) cube([5, 5, t], center = true); slab();";
        assert!((vol(parse_scad(src).unwrap()) - 50.0).abs() < 1e-2);
    }

    #[test]
    fn resize_scales_bbox() {
        // auto=true scales the 0-axes by x's factor (×2) → 20³ = 8000.
        let v = vol(parse_scad("resize([20, 0, 0], auto = true) cube(10);").unwrap());
        assert!((v - 8000.0).abs() < 1.0, "resize auto {v}");
        // explicit newsize → 20×10×10 = 2000.
        let v = vol(parse_scad("resize([20, 10, 10]) cube(10);").unwrap());
        assert!((v - 2000.0).abs() < 1.0, "resize {v}");
    }

    #[test]
    fn minkowski_of_two_cubes_is_a_cube() {
        // Minkowski([0,2]³ ⊕ [0,1]³) = [0,3]³, volume 27.
        let v = vol(parse_scad("minkowski() { cube(2); cube(1); }").unwrap());
        assert!((v - 27.0).abs() < 0.1, "minkowski {v}");
    }

    #[test]
    fn offset_grows_2d_area() {
        // offset(delta=1) of a 10×10 square → 12×12 = 144 (mitred corners).
        let v = vol(parse_scad("linear_extrude(1) offset(delta = 1) square([10, 10]);").unwrap());
        assert!((v - 144.0).abs() < 0.5, "offset delta {v}");
        // offset(r=1) rounds corners: 100 + 40 + π ≈ 143.14.
        let v = vol(parse_scad("linear_extrude(1) offset(r = 1) square([10, 10]);").unwrap());
        assert!((v - 143.14).abs() < 3.0, "offset round {v}");
    }

    #[test]
    fn intersection_for_folds_by_intersection() {
        // ∩ of [0,2]³ and [1,3]×[0,2]×[0,2] → x∈[1,2] → volume 4.
        let src = "intersection_for(i = [0:1]) translate([i, 0, 0]) cube(2);";
        assert!((vol(parse_scad(src).unwrap()) - 4.0).abs() < 1e-2, "isect_for");
    }

    #[test]
    fn intersection_2d_via_clip() {
        // Two overlapping squares → [5,10]×[0,10] = 50.
        let src = "linear_extrude(1) intersection() { square([10, 10]); translate([5, 0]) square([10, 10]); }";
        assert!((vol(parse_scad(src).unwrap()) - 50.0).abs() < 1e-1, "isect2d");
    }

    #[test]
    fn multi_binding_for() {
        // for(i, j) is a cartesian product → 4 unit cubes, volume 4.
        let src = "for (i = [0:1], j = [0:1]) translate([i*2, j*2, 0]) cube(1);";
        assert!((vol(parse_scad(src).unwrap()) - 4.0).abs() < 1e-3, "multi-for");
    }

    #[test]
    fn projection_cut_slices_at_z0() {
        // A cube straddling z=0 sliced → 10×10 square, extruded to volume 100.
        let src = "linear_extrude(1) projection(cut = true) translate([0, 0, -5]) cube(10);";
        assert!((vol(parse_scad(src).unwrap()) - 100.0).abs() < 1.0, "projection");
    }

    #[test]
    fn children_index_selects_one() {
        let src = "module pick() { children(1); } pick() { cube(1); cube(2); }";
        assert!((vol(parse_scad(src).unwrap()) - 8.0).abs() < 1e-2, "children(1)");
    }

    #[test]
    fn children_in_control_flow() {
        // `children(i)` inside a for-loop over $children — the idiomatic pattern.
        // Places cube(1),cube(2),cube(3) → volumes 1+8+27 = 36.
        let src = "module m(){ for(i=[0:$children-1]) translate([i*4,0,0]) children(i); }
                   m(){ cube(1); cube(2); cube(3); }";
        assert!((vol(parse_scad(src).unwrap()) - 36.0).abs() < 1e-2, "children in for");
        // `children([0:1])` → a subset (two disjoint cube(2)s) = 16; the 3rd is dropped.
        let src = "module m(){ children([0:1]); }
                   m(){ cube(2); translate([5,0,0]) cube(2); translate([10,0,0]) cube(2); }";
        assert!((vol(parse_scad(src).unwrap()) - 16.0).abs() < 1e-2, "children range");
        // `children()` threaded through two nested modules.
        let src = "module outer(){ inner() children(); } module inner(){ children(); } outer() cube(2);";
        assert!((vol(parse_scad(src).unwrap()) - 8.0).abs() < 1e-2, "children nested");
    }

    #[test]
    fn builtins_and_constants() {
        // lookup interpolates; PI/true resolve.
        let v = vol(parse_scad("cube(lookup(1.5, [[0,0],[1,10],[2,20]]));").unwrap());
        assert!((v - 3375.0).abs() < 1.0, "lookup {v}"); // 15³
        assert!((vol(parse_scad("if (true) cube(2);").unwrap()) - 8.0).abs() < 1e-3);
        // chr/ord round-trip inside an expression (len of the string).
        assert!((vol(parse_scad("cube(len(chr(65, 66, 67)));").unwrap()) - 27.0).abs() < 1e-2);
    }

    #[test]
    fn modifier_semantics() {
        // `#` (highlight) is a no-op — normal geometry.
        assert!((vol(parse_scad("#cube(2);").unwrap()) - 8.0).abs() < 1e-3);
        // `*` (disable) and `%` (background) contribute nothing, so they drop out of
        // a parent boolean's children instead of wrongly subtracting.
        assert!((vol(parse_scad("difference(){ cube(2); %cube(3); }").unwrap()) - 8.0).abs() < 1e-3);
        assert!((vol(parse_scad("difference(){ cube(2); *cube(3); }").unwrap()) - 8.0).abs() < 1e-3);
        // A wholly-disabled program yields no geometry.
        assert!(parse_scad("*cube(2);").is_err());
        // `!` (show-only) renders just its subtree, ignoring the rest.
        assert!((vol(parse_scad("union(){ cube(4); !cube(2); }").unwrap()) - 8.0).abs() < 1e-3);
    }

    #[test]
    fn linear_extrude_twist_and_scale() {
        // A twist is a per-layer rotation, so cross-section area (and thus volume)
        // is preserved: base 10×10 × height 20 ≈ 2000.
        let t = vol(parse_scad(
            "linear_extrude(height=20, twist=180, slices=40) square([10,10], center=true);",
        )
        .unwrap());
        assert!((t - 2000.0).abs() < 120.0, "twisted extrude vol {t}");
        // `scale` tapers linearly to 0.3 → ∫(1-0.7t)² over the height ≈ 927.
        let s = vol(parse_scad(
            "linear_extrude(height=20, scale=0.3) square([10,10], center=true);",
        )
        .unwrap());
        assert!((s - 927.0).abs() < 25.0, "scaled extrude vol {s}");
    }

    #[test]
    fn rands_builtin() {
        // Seeded draws are deterministic: the same seed yields the same value.
        let src = "a=rands(0,9,1,7); b=rands(0,9,1,7); cube(a[0]==b[0]?2:1);";
        assert!((vol(parse_scad(src).unwrap()) - 8.0).abs() < 1e-3, "seeded rands not reproducible");
        // Count is the requested length; min==max pins every element to that value.
        let src = "r=rands(4,4,3); cube(len(r)==3 && r[0]==4 && r[2]==4 ? 2:1);";
        assert!((vol(parse_scad(src).unwrap()) - 8.0).abs() < 1e-3, "rands count/degenerate-range wrong");
        // Every draw lands in [min, max): sweep 64 of them, none may escape.
        let src = "r=rands(2,3,64); bad=len([for(x=r) if(x<2 || x>=3) 1]); cube(bad==0?2:1);";
        assert!((vol(parse_scad(src).unwrap()) - 8.0).abs() < 1e-3, "rands out of [min,max)");
    }

    #[test]
    fn fill_removes_holes() {
        // fill() drops interior holes: an annulus becomes a solid disk. Extrude by 1
        // so the 2D area reads out directly as a volume.
        let ring = vol(parse_scad(
            "linear_extrude(1) difference(){ circle(10,$fn=64); circle(5,$fn=64); }",
        )
        .unwrap());
        let disk = vol(parse_scad(
            "linear_extrude(1) fill() difference(){ circle(10,$fn=64); circle(5,$fn=64); }",
        )
        .unwrap());
        assert!((ring - 235.0).abs() < 12.0, "annulus vol {ring}"); // π(10²-5²)
        assert!((disk - 314.0).abs() < 12.0, "filled disk vol {disk}"); // π·10²
    }

    #[test]
    fn unsupported_ops_error_clearly() {
        assert!(parse_scad("text(\"hi\");").is_err());
        assert!(parse_scad("surface(\"h.dat\");").is_err());
    }

    #[test]
    fn first_class_functions() {
        // literal + call, closure capture, HOF, list-of-functions.
        assert!((vol(parse_scad("f = function(x) x*2; cube(f(4));").unwrap()) - 512.0).abs() < 1.0);
        assert!((vol(parse_scad("a=1; g=function(x) x+a; cube(g(3));").unwrap()) - 64.0).abs() < 1e-1);
        let src = "function ap(h,x)=h(x); cube(ap(function(y) y+1, 4));";
        assert!((vol(parse_scad(src).unwrap()) - 125.0).abs() < 1e-1); // cube(5)
        let src = "fs=[function(x) x, function(x) x*2]; cube(fs[1](3));";
        assert!((vol(parse_scad(src).unwrap()) - 216.0).abs() < 1.0); // cube(6)
        assert!(vol(parse_scad("f=function(x) x; cube(is_function(f)?2:1);").unwrap()) > 7.0);
    }

    #[test]
    fn deep_recursion_no_crash() {
        // Recursive functions past a few hundred deep used to overflow the stack.
        let src = "function sum(n)=n<=0?0:n+sum(n-1); cube(sum(3000)>0?2:1);";
        assert!((vol(parse_scad(src).unwrap()) - 8.0).abs() < 1e-2);
        // Runaway recursion errors gracefully (recursion limit) rather than crashing.
        assert!(parse_scad("function loop(n)=loop(n+1); x=loop(0); cube(x);").is_err());
    }

    #[test]
    fn minkowski_3d_rejects_nonconvex() {
        // Convex operands work; a non-convex operand errors (not silently wrong).
        assert!(parse_scad("minkowski(){ cube(4); sphere(1,$fn=12); }").is_ok());
        let src = "minkowski(){ difference(){cube(10); translate([5,5,-1]) cube([6,6,12]);} sphere(1,$fn=8); }";
        assert!(parse_scad(src).is_err());
    }

    #[test]
    fn import_dxf_and_svg() {
        let dir = std::env::temp_dir().join("threers_2dimport");
        std::fs::create_dir_all(&dir).unwrap();
        // DXF: a 10×10 closed LWPOLYLINE square.
        let dxf = "0\nSECTION\n2\nENTITIES\n0\nLWPOLYLINE\n70\n1\n10\n0\n20\n0\n10\n10\n20\n0\n10\n10\n20\n10\n10\n0\n20\n10\n0\nENDSEC\n0\nEOF\n";
        std::fs::write(dir.join("s.dxf"), dxf).unwrap();
        std::fs::write(dir.join("d.scad"), "linear_extrude(3) import(\"s.dxf\");").unwrap();
        let v = vol(parse_scad_file(dir.join("d.scad")).unwrap());
        assert!((v - 300.0).abs() < 1.0, "dxf {v}"); // 10×10×3
        // SVG: a 20×10 rect.
        std::fs::write(dir.join("r.svg"), "<svg><rect x=\"0\" y=\"0\" width=\"20\" height=\"10\"/></svg>").unwrap();
        std::fs::write(dir.join("v.scad"), "linear_extrude(2) import(\"r.svg\");").unwrap();
        let v = vol(parse_scad_file(dir.join("v.scad")).unwrap());
        assert!((v - 400.0).abs() < 1.0, "svg {v}"); // 20×10×2
    }

    #[test]
    fn import_3mf_and_amf() {
        let dir = std::env::temp_dir().join("threers_3d_import");
        std::fs::create_dir_all(&dir).unwrap();
        // A tetrahedron on the axes: volume = |det|/6 = 1000/6 ≈ 166.67.
        let tetra_tris = [(0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)];

        // --- 3MF (ZIP → 3dmodel.model XML): two objects, the second shifted +100
        // in x, so per-mesh index offsetting is exercised (each mesh's triangles
        // use LOCAL 0..3 indices). ---
        let mut model = String::from("<model><resources>");
        for (id, dx) in [(1, 0), (2, 100)] {
            model += &format!("<object id=\"{id}\"><mesh><vertices>");
            for [x, y, z] in [[dx, 0, 0], [dx + 10, 0, 0], [dx, 10, 0], [dx, 0, 10]] {
                model += &format!("<vertex x=\"{x}\" y=\"{y}\" z=\"{z}\"/>");
            }
            model += "</vertices><triangles>";
            for (a, b, c) in tetra_tris {
                model += &format!("<triangle v1=\"{a}\" v2=\"{b}\" v3=\"{c}\"/>");
            }
            model += "</triangles></mesh></object>";
        }
        model += "</resources></model>";
        std::fs::write(dir.join("t.3mf"), store_zip("3D/3dmodel.model", model.as_bytes())).unwrap();
        std::fs::write(dir.join("m.scad"), "import(\"t.3mf\");").unwrap();
        let g = parse_scad_file(dir.join("m.scad")).unwrap().to_geometry();
        // The far object reaches x = 110 and both are 4-triangle tetrahedra (8
        // total) only if each mesh's LOCAL indices resolved to its own vertices —
        // i.e. the per-mesh offset is right. (Volume is a poor check here: the
        // outward-orientation heuristic flips one disjoint tetra's winding.)
        let xs: Vec<f32> = g.attributes.get("position").unwrap().array.chunks_exact(3).map(|c| c[0]).collect();
        let (xmin, xmax) = (xs.iter().cloned().fold(f32::MAX, f32::min), xs.iter().cloned().fold(f32::MIN, f32::max));
        assert!((xmin - 0.0).abs() < 1e-3 && (xmax - 110.0).abs() < 1e-3, "3mf multi-object misindexed ({xmin}..{xmax})");
        let n_tri = g.index.as_ref().map(|i| i.len() / 3).unwrap_or(xs.len() / 3);
        assert_eq!(n_tri, 8, "3mf: expected 2 tetrahedra = 8 triangles");

        // --- AMF (uncompressed XML with nested element coordinates) ---
        let mut amf = String::from("<amf unit=\"millimeter\"><object id=\"0\"><mesh><vertices>");
        for [x, y, z] in [[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]] {
            amf += &format!("<vertex><coordinates><x>{x}</x><y>{y}</y><z>{z}</z></coordinates></vertex>");
        }
        amf += "</vertices><volume>";
        for (a, b, c) in tetra_tris {
            amf += &format!("<triangle><v1>{a}</v1><v2>{b}</v2><v3>{c}</v3></triangle>");
        }
        amf += "</volume></mesh></object></amf>";
        std::fs::write(dir.join("t.amf"), amf).unwrap();
        std::fs::write(dir.join("a.scad"), "import(\"t.amf\");").unwrap();
        let v = vol(parse_scad_file(dir.join("a.scad")).unwrap());
        assert!((v - 1000.0 / 6.0).abs() < 1.0, "amf vol {v}");
    }

    /// Minimal single-entry ZIP (store method, CRC left 0 — the reader doesn't
    /// verify it) so the 3MF test needs no external archiver.
    fn store_zip(name: &str, data: &[u8]) -> Vec<u8> {
        let (nl, dl) = (name.len() as u16, data.len() as u32);
        let mut out = Vec::new();
        out.extend_from_slice(b"PK\x03\x04");
        out.extend_from_slice(&[20, 0, 0, 0, 0, 0]); // version, flags, method=store
        out.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]); // time/date, crc
        out.extend_from_slice(&dl.to_le_bytes()); // compressed size
        out.extend_from_slice(&dl.to_le_bytes()); // uncompressed size
        out.extend_from_slice(&nl.to_le_bytes());
        out.extend_from_slice(&[0, 0]); // extra len
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(data);
        let cd_off = out.len() as u32;
        out.extend_from_slice(b"PK\x01\x02");
        out.extend_from_slice(&[20, 0, 20, 0, 0, 0, 0, 0]); // versions, flags, method
        out.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]); // time/date, crc
        out.extend_from_slice(&dl.to_le_bytes());
        out.extend_from_slice(&dl.to_le_bytes());
        out.extend_from_slice(&nl.to_le_bytes());
        out.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]); // extra, comment, disk, int-attr
        out.extend_from_slice(&[0, 0, 0, 0]); // external attrs
        out.extend_from_slice(&0u32.to_le_bytes()); // local header offset
        out.extend_from_slice(name.as_bytes());
        let cd_size = out.len() as u32 - cd_off;
        out.extend_from_slice(b"PK\x05\x06");
        out.extend_from_slice(&[0, 0, 0, 0, 1, 0, 1, 0]); // disks + entry counts (1)
        out.extend_from_slice(&cd_size.to_le_bytes());
        out.extend_from_slice(&cd_off.to_le_bytes());
        out.extend_from_slice(&[0, 0]); // comment len
        out
    }

    #[test]
    fn surface_png_heightmap() {
        let dir = std::env::temp_dir().join("threers_surface_png");
        std::fs::create_dir_all(&dir).unwrap();
        // Hand-build an 8×8 8-bit greyscale PNG, row ramp: row r → grey 40 + r·20.
        let (w, h) = (8usize, 8usize);
        std::fs::write(dir.join("ramp.png"), gray_png(w, h, |_x, y| 40 + (y * 20) as u8)).unwrap();
        std::fs::write(dir.join("s.scad"), "surface(file=\"ramp.png\");").unwrap();

        let g = parse_scad_file(dir.join("s.scad")).unwrap().to_geometry();
        let pos = &g.attributes.get("position").unwrap().array;
        let (mut zmin, mut zmax) = (f32::INFINITY, f32::NEG_INFINITY);
        for c in pos.chunks_exact(3) {
            zmin = zmin.min(c[2]);
            zmax = zmax.max(c[2]);
        }
        // Top peaks at the brightest row (180); the base sits 1 below the min (39).
        assert!((zmax - 180.0).abs() < 1.5, "zmax {zmax}");
        assert!((zmin - 39.0).abs() < 1.5, "zmin {zmin}");
    }

    /// Minimal 8-bit greyscale (colour type 0) PNG with one stored-zlib IDAT and
    /// dummy checksums — the decoder verifies neither CRC nor adler, so the test
    /// needs no compressor.
    fn gray_png(w: usize, h: usize, px: impl Fn(usize, usize) -> u8) -> Vec<u8> {
        let mut raw = Vec::with_capacity((w + 1) * h);
        for y in 0..h {
            raw.push(0); // filter: none
            for x in 0..w {
                raw.push(px(x, y));
            }
        }
        // zlib: header + one stored DEFLATE block + dummy adler32.
        let mut idat = vec![0x78, 0x01, 0x01];
        let len = raw.len() as u16;
        idat.extend_from_slice(&len.to_le_bytes());
        idat.extend_from_slice(&(!len).to_le_bytes());
        idat.extend_from_slice(&raw);
        idat.extend_from_slice(&[0, 0, 0, 0]);
        let mut png = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        let chunk = |png: &mut Vec<u8>, ty: &[u8; 4], data: &[u8]| {
            png.extend_from_slice(&(data.len() as u32).to_be_bytes());
            png.extend_from_slice(ty);
            png.extend_from_slice(data);
            png.extend_from_slice(&[0, 0, 0, 0]); // dummy CRC (unverified)
        };
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&(w as u32).to_be_bytes());
        ihdr.extend_from_slice(&(h as u32).to_be_bytes());
        ihdr.extend_from_slice(&[8, 0, 0, 0, 0]); // 8-bit, greyscale, no interlace
        chunk(&mut png, b"IHDR", &ihdr);
        chunk(&mut png, b"IDAT", &idat);
        chunk(&mut png, b"IEND", &[]);
        png
    }

    #[test]
    fn boolean_2d_union_overlapping() {
        // Two axis-aligned squares overlapping in [5,10]×[0,10] (collinear edges!):
        // 100 + 100 − 50 = 150.
        let src = "linear_extrude(1) union() { square([10,10]); translate([5,0]) square([10,10]); }";
        assert!((vol(parse_scad(src).unwrap()) - 150.0).abs() < 0.5, "union2d");
    }

    #[test]
    fn boolean_2d_difference_partial() {
        // [0,10]² − [5,15]×[0,10] = [0,5]×[0,10] = 50 (partial overlap, not a hole).
        let src = "linear_extrude(1) difference() { square([10,10]); translate([5,0]) square([10,10]); }";
        assert!((vol(parse_scad(src).unwrap()) - 50.0).abs() < 0.5, "diff2d partial");
    }

    #[test]
    fn boolean_2d_difference_makes_hole() {
        // A fully-interior subtraction leaves a hole: 10² − 2² = 96.
        let src = "linear_extrude(1) difference() { square(10); translate([4,4]) square(2); }";
        assert!((vol(parse_scad(src).unwrap()) - 96.0).abs() < 0.5, "diff2d hole");
    }

    #[test]
    fn boolean_2d_nonconvex_intersection() {
        // Intersection of an L (non-convex) with a square → the overlap region.
        // L = square(10) − translate([5,5]) square(6) (removes top-right) then
        // ∩ square([7,7]) → area = 49 − 4 (the [5,7]² bite) = 45.
        let src = "linear_extrude(1) intersection() {
            difference() { square(10); translate([5,5]) square(6); }
            square([7,7]);
        }";
        assert!((vol(parse_scad(src).unwrap()) - 45.0).abs() < 0.5, "isect2d nonconvex");
    }

    #[test]
    fn projection_no_cut_silhouette() {
        // Silhouette of two offset cubes = union of two squares: 100+100−25 = 175.
        let src = "linear_extrude(1) projection() union() { cube(10); translate([5,5,5]) cube(10); }";
        assert!((vol(parse_scad(src).unwrap()) - 175.0).abs() < 1.0, "projection nocut");
    }

    #[test]
    fn fa_fs_default_resolution() {
        // With $fn unset, circle(r) fragments = ceil(max(min(360/12, 2πr/2), 5)).
        // r=5 → min(30, 15.7) → 16; area of a 16-gon ≈ 76.7 (< the true 78.54).
        let a = vol(parse_scad("linear_extrude(1) circle(5);").unwrap());
        assert!((a - 76.7).abs() < 1.0, "default circle {a}");
        // Finer $fa and $fs → more fragments → closer to π·25 ≈ 78.54.
        let b = vol(parse_scad("linear_extrude(1) circle(5, $fs = 0.2, $fa = 1);").unwrap());
        assert!(b > a && (b - 78.54).abs() < 0.2, "fine circle {b}");
    }

    #[test]
    fn cylinder_facets_match_fn() {
        // $fn=6 → hexagonal prism: hexagon area (circumradius 5) × height 10.
        let hex = 1.5 * 3f64.sqrt() * 25.0; // (3√3/2)·r²
        let v = vol(parse_scad("cylinder(h = 10, r = 5, $fn = 6);").unwrap());
        assert!((v - hex * 10.0).abs() < 0.5, "hex prism {v} vs {}", hex * 10.0);
    }

    #[test]
    fn sphere_volume_converges() {
        // A fine sphere approaches 4/3·π·r³.
        let v = vol(parse_scad("sphere(5, $fn = 64);").unwrap());
        let exact = 4.0 / 3.0 * std::f64::consts::PI * 125.0;
        assert!((v - exact).abs() / exact < 0.01, "sphere {v} vs {exact}");
    }

    #[test]
    fn minkowski_2d_convex_square() {
        // square(4) ⊕ square(2) = square(6): area 36 (via triangulate + union).
        let src = "linear_extrude(1) minkowski() { square(4); square(2); }";
        assert!((vol(parse_scad(src).unwrap()) - 36.0).abs() < 0.1, "mink2d square");
    }

    #[test]
    fn surface_heightmap_volume() {
        let dir = std::env::temp_dir().join("threers_surf");
        std::fs::create_dir_all(&dir).unwrap();
        // 3×3 grid all height 2 → base at min−1=1 → a 2×2×1 box, volume 4.
        std::fs::write(dir.join("h.dat"), "2 2 2\n2 2 2\n2 2 2\n").unwrap();
        std::fs::write(dir.join("s.scad"), "surface(file = \"h.dat\");").unwrap();
        let v = vol(parse_scad_file(dir.join("s.scad")).unwrap());
        assert!((v - 4.0).abs() < 1e-2, "surface {v}");
    }

    #[test]
    fn import_off_tetra() {
        let dir = std::env::temp_dir().join("threers_off");
        std::fs::create_dir_all(&dir).unwrap();
        let off = "OFF\n4 4 0\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n3 0 1 2\n3 0 3 1\n3 0 2 3\n3 1 3 2\n";
        std::fs::write(dir.join("t.off"), off).unwrap();
        std::fs::write(dir.join("i.scad"), "import(\"t.off\");").unwrap();
        let v = vol(parse_scad_file(dir.join("i.scad")).unwrap());
        assert!((v - 1.0 / 6.0).abs() < 1e-3, "off tetra {v}");
    }

    #[test]
    fn text_produces_glyph_geometry() {
        // Uses the bundled font (won't match OpenSCAD's), but must be non-empty.
        let v = vol(parse_scad("linear_extrude(2) text(\"I\", size = 10);").unwrap());
        assert!(v > 1.0, "text vol {v}");
    }

    #[test]
    fn concave_polygon_extrudes_watertight() {
        // An L-shaped (concave) polygon → area 64, extruded to volume 64. Exercises
        // ear-clipping across reflex corners (the axis-aligned notch case).
        let src = "linear_extrude(1) polygon([[0,0],[10,0],[10,4],[4,4],[4,10],[0,10]]);";
        assert!((vol(parse_scad(src).unwrap()) - 64.0).abs() < 1e-2, "concave extrude");
    }

    #[test]
    fn multmatrix_scales_and_shears() {
        // Diagonal matrix scales x by 2 → volume 2.
        let src = "multmatrix([[2,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) cube(1);";
        assert!((vol(parse_scad(src).unwrap()) - 2.0).abs() < 1e-3, "multmatrix scale");
        // A shear preserves volume.
        let src = "multmatrix([[1,1,0,0],[0,1,0,0],[0,0,1,0]]) cube(2);";
        assert!((vol(parse_scad(src).unwrap()) - 8.0).abs() < 1e-2, "multmatrix shear");
    }

    #[test]
    fn import_stl_roundtrips() {
        let p = std::env::temp_dir().join("threers_import_cube.stl");
        std::fs::write(&p, super::cube([2.0, 2.0, 2.0]).to_stl()).unwrap();
        let src = format!("import(\"{}\");", p.display());
        let v = vol(parse_scad(&src).unwrap());
        assert!((v - 8.0).abs() < 1e-2, "import vol {v}");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn include_and_use_resolve_files() {
        let dir = std::env::temp_dir().join("threers_scad_incl");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("lib.scad"), "GAP = 4;\nmodule widget(s = 3) cube(s);\n").unwrap();
        // include: pulls in the variable GAP *and* the module widget.
        std::fs::write(dir.join("main_inc.scad"), "include <lib.scad>\ntranslate([GAP,0,0]) widget(2);").unwrap();
        let v = vol(parse_scad_file(dir.join("main_inc.scad")).unwrap());
        assert!((v - 8.0).abs() < 1e-2, "include vol {v}");
        // use: pulls in only the definitions (module widget), not GAP.
        std::fs::write(dir.join("main_use.scad"), "use <lib.scad>\nwidget(4);").unwrap();
        let v = vol(parse_scad_file(dir.join("main_use.scad")).unwrap());
        assert!((v - 64.0).abs() < 1e-1, "use vol {v}");
    }
}
