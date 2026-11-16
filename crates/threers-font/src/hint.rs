//! TrueType hinting: a bytecode interpreter that runs a face's font and
//! control value programs at a size, then each glyph's instructions on its
//! scaled points, with the results FreeType's version 40 interpreter gives
//! when hinting for grayscale (subpixel hinting, backward compatibility
//! with fonts that expect x moves to be ignored).
//!
//! The behaviour follows FreeType's (portions of this software are
//! copyright The FreeType Project, www.freetype.org); the code is written
//! from the TrueType instruction set specification.

use crate::fixed::{div_fix, mul_div, mul_div_no_round, mul_fix, pix_ceil, pix_floor, pix_round};
use crate::glyph::{self, Component, Glyph, Outline, RawGlyph, TAG_ON};
use crate::sfnt::{Error, Face};
use crate::size::Size;

const TOUCH_X: u8 = 0x08;
const TOUCH_Y: u8 = 0x10;

/// A failure that stops a program; FreeType carries on with the points as
/// they are then.
#[derive(Debug)]
struct Abort;

type Run<T> = Result<T, Abort>;

/// `TT_Round_*`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RoundState {
    HalfGrid,
    Grid,
    DoubleGrid,
    DownToGrid,
    UpToGrid,
    Off,
    Super,
    Super45,
}

/// `TT_GraphicsState`
#[derive(Clone, Copy, Debug)]
struct GraphicsState {
    rp0: usize,
    rp1: usize,
    rp2: usize,
    dual: (i64, i64),
    proj: (i64, i64),
    free: (i64, i64),
    loop_: i64,
    minimum_distance: i64,
    round_state: RoundState,
    auto_flip: bool,
    control_value_cutin: i64,
    single_width_cutin: i64,
    single_width_value: i64,
    delta_base: i64,
    delta_shift: i64,
    instruct_control: u8,
    gep0: u8,
    gep1: u8,
    gep2: u8,
}

impl Default for GraphicsState {
    /// `tt_default_graphics_state`
    fn default() -> Self {
        GraphicsState {
            rp0: 0,
            rp1: 0,
            rp2: 0,
            dual: (0x4000, 0),
            proj: (0x4000, 0),
            free: (0x4000, 0),
            loop_: 1,
            minimum_distance: 64,
            round_state: RoundState::Grid,
            auto_flip: true,
            control_value_cutin: 68,
            single_width_cutin: 0,
            single_width_value: 0,
            delta_base: 9,
            delta_shift: 3,
            instruct_control: 0,
            gep0: 1,
            gep1: 1,
            gep2: 1,
        }
    }
}

/// A glyph zone: the points as the font has them (`orus`), scaled
/// (`org`) and moved (`cur`), their tags and contour ends.
#[derive(Clone, Debug, Default)]
struct Zone {
    orus: Vec<(i64, i64)>,
    org: Vec<(i64, i64)>,
    cur: Vec<(i64, i64)>,
    tags: Vec<u8>,
    contours: Vec<usize>,
}

impl Zone {
    fn with_points(n: usize) -> Zone {
        Zone { orus: vec![(0, 0); n], org: vec![(0, 0); n], cur: vec![(0, 0); n], tags: vec![0; n], contours: Vec::new() }
    }

    fn len(&self) -> usize {
        self.cur.len()
    }
}

/// Which code a function or the running program is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Range {
    Font,
    Cvt,
    Glyph,
}

/// `TT_DefRecord`
#[derive(Clone, Copy, Debug)]
struct Def {
    range: Range,
    start: usize,
    end: usize,
}

/// A call in progress: where to come back to, the function, the times
/// left to run it.
#[derive(Clone, Copy, Debug)]
struct Call {
    range: Range,
    ip: usize,
    def: Def,
    count: i64,
}

/// A face's hinting state at a size: the CVT, storage, twilight zone,
/// graphics state and definitions its font and control value programs
/// left.
pub struct Hinter {
    cvt_units: Vec<i64>,
    cvt: Vec<i64>,
    storage: Vec<i64>,
    twilight: Zone,
    gs: GraphicsState,
    fdefs: Vec<Option<Def>>,
    idefs: Vec<Option<Def>>,
    fpgm: Vec<u8>,
    prep: Vec<u8>,
    max_stack: usize,
    /// Scale of the CVT and `WCVTF`: the larger ppem axis's.
    scale: i64,
    ppem: i64,
    point_size: i64,
    x_scale: i64,
    y_scale: i64,
    /// Whether the programs ran without error; hinting is skipped otherwise.
    ready: bool,
}

/// The interpreter running one program.
struct Exec<'a> {
    h: &'a mut Hinter,
    /// The CVT and storage the program reads and writes (copies for a
    /// glyph program).
    cvt: Vec<i64>,
    storage: Vec<i64>,
    pts: Zone,
    gs: GraphicsState,
    /// Whether each of zp0, zp1, zp2 is the glyph zone (else twilight).
    zp: [bool; 3],
    stack: Vec<i64>,
    calls: Vec<Call>,
    /// Where the program started (`iniRange`) and where it is now.
    ini_range: Range,
    range: Range,
    glyph_code: &'a [u8],
    ip: usize,
    opcode: u8,
    f_dot_p: i64,
    period: i64,
    phase: i64,
    threshold: i64,
    subpixel_hinting_lean: bool,
    grayscale_cleartype: bool,
    backward_compatibility: bool,
    iupx_called: bool,
    iupy_called: bool,
    is_composite: bool,
    x_scale: i64,
    y_scale: i64,
    /// Whether to log the program's moves and failures to stderr
    /// (`THREERS_FONT_TRACE`), for checking against another interpreter.
    trace: bool,
}

/// `TT_DotFix14`
fn dot_fix14(ax: i64, ay: i64, bx: i64, by: i64) -> i64 {
    let s = ax.wrapping_mul(bx).wrapping_add(ay.wrapping_mul(by));
    (s + 0x2000 + (s >> 63)) >> 14
}

/// `TT_MulFix14`
fn mul_fix14(a: i64, b: i64) -> i64 {
    let ab = a.wrapping_mul(b);
    (ab + 0x2000 + (ab >> 63)) >> 14
}

/// `FT_Vector_NormLen`: a vector to a 16.16 unit vector.
fn norm_len(x0: i64, y0: i64) -> (i64, i64) {
    let (x_, y_) = (x0 as i32, y0 as i32);
    let (sx, sy) = (if x_ < 0 { -1i64 } else { 1 }, if y_ < 0 { -1i64 } else { 1 });
    let mut x = x_.unsigned_abs();
    let mut y = y_.unsigned_abs();

    if x == 0 {
        return (0, if y > 0 { sy * 0x10000 } else { y0 });
    }

    if y == 0 {
        return (if x > 0 { sx * 0x10000 } else { x0 }, 0);
    }

    let mut l: u32 = if x > y { x.wrapping_add(y >> 1) } else { y.wrapping_add(x >> 1) };
    let msb = 31 - l.leading_zeros() as i32;
    let mut shift = 31 - msb;
    shift -= 15 + i32::from(l >= (0xAAAA_AAAAu32 >> shift));

    if shift > 0 {
        x <<= shift;
        y <<= shift;
        l = if x > y { x.wrapping_add(y >> 1) } else { y.wrapping_add(x >> 1) };
    } else {
        x >>= -shift;
        y >>= -shift;
        l >>= -shift;
    }

    let mut b: i32 = 0x10000 - l as i32;
    let (xi, yi) = (x as i32, y as i32);
    let (mut u, mut v);

    loop {
        u = (xi.wrapping_add(((i64::from(xi) * i64::from(b)) >> 16) as i32)) as u32;
        v = (yi.wrapping_add(((i64::from(yi) * i64::from(b)) >> 16) as i32)) as u32;
        let sq = u.wrapping_mul(u).wrapping_add(v.wrapping_mul(v)) as i32;
        let mut z: i32 = sq.wrapping_neg() / 0x200;
        z = ((i64::from(z) * i64::from((0x10000 + b) >> 8)) / 0x10000) as i32;
        b += z;

        if z <= 0 {
            break;
        }
    }

    (sx * i64::from(u), sy * i64::from(v))
}

/// `Normalize`: a vector to an F2Dot14 unit vector; (0, 0) leaves it.
fn normalize(x: i64, y: i64, out: &mut (i64, i64)) {
    if x == 0 && y == 0 {
        return;
    }

    let (nx, ny) = norm_len(x, y);
    *out = (i64::from((nx / 4) as i16), i64::from((ny / 4) as i16));
}

/// The size of the instruction at `code[ip]`, push data included.
fn instruction_len(code: &[u8], ip: usize) -> Option<usize> {
    let op = *code.get(ip)?;

    Some(match op {
        0x40 => 2 + usize::from(*code.get(ip + 1)?),
        0x41 => 2 + 2 * usize::from(*code.get(ip + 1)?),
        0xB0..=0xB7 => 2 + usize::from(op - 0xB0),
        0xB8..=0xBF => 3 + 2 * usize::from(op - 0xB8),
        _ => 1,
    })
}

impl Hinter {
    /// Whether the font and control value programs ran without error.
    pub fn ready(&self) -> bool {
        self.ready
    }

    /// `tt_size_init_bytecode` then `tt_size_ready_bytecode` at a size, the
    /// control value program run as the first hinted load runs it: once
    /// as monochrome, then again for grayscale subpixel hinting.
    pub fn new(face: &Face, size: &Size) -> Hinter {
        let cvt_units: Vec<i64> = face
            .table(b"cvt ")
            .map(|t| t.as_chunks::<2>().0.iter().map(|c| i64::from(i16::from_be_bytes([c[0], c[1]])) * 64).collect())
            .unwrap_or_default();

        let m = size.hinted;
        let (scale, ppem) = if m.x_ppem >= m.y_ppem { (m.x_scale, m.x_ppem) } else { (m.y_scale, m.y_ppem) };

        let mut h = Hinter {
            cvt: vec![0; cvt_units.len()],
            cvt_units,
            storage: vec![0; usize::from(face.max_storage)],
            twilight: Zone::with_points(usize::from(face.max_twilight_points) + 4),
            gs: GraphicsState::default(),
            fdefs: vec![None; usize::from(face.max_function_defs)],
            idefs: Vec::new(),
            fpgm: face.table(b"fpgm").unwrap_or(&[]).to_vec(),
            prep: face.table(b"prep").unwrap_or(&[]).to_vec(),
            max_stack: usize::from(face.max_stack_elements) + 32,
            scale,
            ppem: i64::from(ppem),
            point_size: 0,
            x_scale: m.x_scale,
            y_scale: m.y_scale,
            ready: true,
        };

        // TT_Size_Request: the point size MPS gives
        h.point_size = mul_div(h.ppem, 64 * 72, i64::from(size.resolution.max(1)));

        if !h.run_program(Range::Font, false) {
            h.ready = false;
            return h;
        }

        h.gs = GraphicsState::default();
        h.ready = h.run_prep(false) && h.run_prep(true);
        h
    }

    /// `tt_size_run_prep`: the CVT scaled afresh, then `prep`, then the
    /// graphics state it leaves (less vectors, reference points and zones)
    /// kept for glyphs.
    fn run_prep(&mut self, subpixel: bool) -> bool {
        let s = self.scale >> 6;

        for (c, u) in self.cvt.iter_mut().zip(&self.cvt_units) {
            *c = mul_fix(*u, s);
        }

        let ok = self.run_program(Range::Cvt, subpixel);
        let gs = &mut self.gs;
        gs.dual = (0x4000, 0);
        gs.proj = (0x4000, 0);
        gs.free = (0x4000, 0);
        gs.rp0 = 0;
        gs.rp1 = 0;
        gs.rp2 = 0;
        gs.gep0 = 1;
        gs.gep1 = 1;
        gs.gep2 = 1;
        gs.loop_ = 1;
        ok
    }

    /// Runs the font or control value program on the size's own state.
    fn run_program(&mut self, range: Range, subpixel: bool) -> bool {
        let gs = self.gs;
        let cvt = std::mem::take(&mut self.cvt);
        let storage = std::mem::take(&mut self.storage);
        let mut exec = Exec::new(self, cvt, storage, Zone::default(), gs, subpixel, false, 1 << 16, 1 << 16);
        exec.ini_range = range;
        exec.range = range;
        let ok = exec.run_code(&[]).is_ok();
        let (gs, cvt, storage) = (exec.gs, exec.cvt, exec.storage);
        self.gs = gs;
        self.cvt = cvt;
        self.storage = storage;
        ok
    }
}

impl<'a> Exec<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        h: &'a mut Hinter,
        cvt: Vec<i64>,
        storage: Vec<i64>,
        pts: Zone,
        gs: GraphicsState,
        subpixel: bool,
        is_composite: bool,
        x_scale: i64,
        y_scale: i64,
    ) -> Exec<'a> {
        let mut e = Exec {
            h,
            cvt,
            storage,
            pts,
            gs,
            zp: [true; 3],
            stack: Vec::new(),
            calls: Vec::new(),
            ini_range: Range::Glyph,
            range: Range::Glyph,
            glyph_code: &[],
            ip: 0,
            opcode: 0,
            f_dot_p: 0x4000,
            period: 64,
            phase: 0,
            threshold: 0,
            subpixel_hinting_lean: subpixel,
            grayscale_cleartype: subpixel,
            backward_compatibility: false,
            iupx_called: false,
            iupy_called: false,
            is_composite,
            x_scale,
            y_scale,
            trace: std::env::var_os("THREERS_FONT_TRACE").is_some(),
        };
        e.zp = [e.gs.gep0 == 1, e.gs.gep1 == 1, e.gs.gep2 == 1];
        e.compute_funcs();
        e
    }

    fn code(&self, range: Range) -> &[u8] {
        match range {
            Range::Font => &self.h.fpgm,
            Range::Cvt => &self.h.prep,
            Range::Glyph => self.glyph_code,
        }
    }

    fn zone(&self, i: usize) -> &Zone {
        if self.zp[i] { &self.pts } else { &self.h.twilight }
    }

    fn zone_mut(&mut self, i: usize) -> &mut Zone {
        if self.zp[i] { &mut self.pts } else { &mut self.h.twilight }
    }

    // --- projections and moves -------------------------------------

    /// `Compute_Funcs`
    fn compute_funcs(&mut self) {
        let (p, f) = (self.gs.proj, self.gs.free);

        // the dot product, rounded to F2Dot14
        self.f_dot_p = if f.0 == 0x4000 {
            p.0
        } else if f.1 == 0x4000 {
            p.1
        } else {
            dot_fix14(p.0, p.1, f.0, f.1)
        };

        // nearly perpendicular vectors would move points without bound
        if self.f_dot_p.abs() < 0x400 {
            self.f_dot_p = 0x4000;
        }
    }

    fn project(&self, dx: i64, dy: i64) -> i64 {
        let p = self.gs.proj;

        if p.0 == 0x4000 {
            dx
        } else if p.1 == 0x4000 {
            dy
        } else {
            dot_fix14(dx, dy, p.0, p.1)
        }
    }

    fn dual_project(&self, dx: i64, dy: i64) -> i64 {
        let p = self.gs.dual;

        if p.0 == 0x4000 {
            dx
        } else if p.1 == 0x4000 {
            dy
        } else {
            dot_fix14(dx, dy, p.0, p.1)
        }
    }

    fn project_pts(&self, a: (i64, i64), b: (i64, i64)) -> i64 {
        self.project(a.0.wrapping_sub(b.0), a.1.wrapping_sub(b.1))
    }

    fn dual_pts(&self, a: (i64, i64), b: (i64, i64)) -> i64 {
        self.dual_project(a.0.wrapping_sub(b.0), a.1.wrapping_sub(b.1))
    }

    /// Whether x moves are dropped (v40 backward compatibility).
    fn x_blocked(&self) -> bool {
        self.backward_compatibility
    }

    /// Whether y moves are dropped (after both IUPs in backward
    /// compatibility).
    fn y_blocked(&self) -> bool {
        self.backward_compatibility && self.iupx_called && self.iupy_called
    }

    /// `func_move` (`Direct_Move` and its axis cases)
    fn move_point(&mut self, z: usize, point: usize, distance: i64) {
        let (f, fdp) = (self.gs.free, self.f_dot_p);
        let axis_x = fdp == 0x4000 && f.0 == 0x4000;
        let axis_y = fdp == 0x4000 && !axis_x && f.1 == 0x4000;
        let (xb, yb) = (self.x_blocked(), self.y_blocked());
        let zone = self.zone_mut(z);

        if axis_x {
            if !xb {
                zone.cur[point].0 = zone.cur[point].0.wrapping_add(distance);
            }

            zone.tags[point] |= TOUCH_X;
            return;
        }

        if axis_y {
            if !yb {
                zone.cur[point].1 = zone.cur[point].1.wrapping_add(distance);
            }

            zone.tags[point] |= TOUCH_Y;
            return;
        }

        if f.0 != 0 {
            if !xb {
                zone.cur[point].0 = zone.cur[point].0.wrapping_add(mul_div(distance, f.0, fdp));
            }

            zone.tags[point] |= TOUCH_X;
        }

        if f.1 != 0 {
            if !yb {
                zone.cur[point].1 = zone.cur[point].1.wrapping_add(mul_div(distance, f.1, fdp));
            }

            zone.tags[point] |= TOUCH_Y;
        }
    }

    /// `func_move_orig` (`Direct_Move_Orig` and its axis cases)
    fn move_orig(&mut self, z: usize, point: usize, distance: i64) {
        let (f, fdp) = (self.gs.free, self.f_dot_p);
        let zone = self.zone_mut(z);

        if fdp == 0x4000 && f.0 == 0x4000 {
            zone.org[point].0 = zone.org[point].0.wrapping_add(distance);
        } else if fdp == 0x4000 && f.1 == 0x4000 {
            zone.org[point].1 = zone.org[point].1.wrapping_add(distance);
        } else {
            if f.0 != 0 {
                zone.org[point].0 = zone.org[point].0.wrapping_add(mul_div(distance, f.0, fdp));
            }

            if f.1 != 0 {
                zone.org[point].1 = zone.org[point].1.wrapping_add(mul_div(distance, f.1, fdp));
            }
        }
    }

    /// `Move_Zp2_Point`
    fn move_zp2(&mut self, point: usize, dx: i64, dy: i64, touch: bool) {
        let f = self.gs.free;
        let (xb, yb) = (self.x_blocked(), self.y_blocked());
        let zone = self.zone_mut(2);

        if f.0 != 0 {
            if !xb {
                zone.cur[point].0 = zone.cur[point].0.wrapping_add(dx);
            }

            if touch {
                zone.tags[point] |= TOUCH_X;
            }
        }

        if f.1 != 0 {
            if !yb {
                zone.cur[point].1 = zone.cur[point].1.wrapping_add(dy);
            }

            if touch {
                zone.tags[point] |= TOUCH_Y;
            }
        }
    }

    // --- rounding --------------------------------------------------

    fn round(&self, d: i64) -> i64 {
        self.round_with(self.gs.round_state, d)
    }

    /// `Round_To_Grid` and the others, compensation 0.
    fn round_with(&self, state: RoundState, d: i64) -> i64 {
        match state {
            RoundState::Grid => {
                if d >= 0 {
                    (d.wrapping_add(32) & -64).max(0)
                } else {
                    (-((32i64.wrapping_sub(d)) & -64)).min(0)
                }
            }
            RoundState::HalfGrid => {
                if d >= 0 {
                    let v = pix_floor(d) + 32;
                    if v < 0 { 32 } else { v }
                } else {
                    let v = -(pix_floor(d.wrapping_neg()) + 32);
                    if v > 0 { -32 } else { v }
                }
            }
            RoundState::DoubleGrid => {
                if d >= 0 {
                    (d.wrapping_add(16) & -32).max(0)
                } else {
                    (-((16i64.wrapping_sub(d)) & -32)).min(0)
                }
            }
            RoundState::DownToGrid => {
                if d >= 0 {
                    (d & -64).max(0)
                } else {
                    (-(d.wrapping_neg() & -64)).min(0)
                }
            }
            RoundState::UpToGrid => {
                if d >= 0 {
                    (d.wrapping_add(63) & -64).max(0)
                } else {
                    (-((63i64.wrapping_sub(d)) & -64)).min(0)
                }
            }
            RoundState::Off => Self::round_none(d),
            RoundState::Super => {
                if d >= 0 {
                    let v = (d.wrapping_add(self.threshold - self.phase) & -self.period).wrapping_add(self.phase);
                    if v < 0 { self.phase } else { v }
                } else {
                    let v = (-((self.threshold - self.phase).wrapping_sub(d) & -self.period)).wrapping_sub(self.phase);
                    if v > 0 { -self.phase } else { v }
                }
            }
            RoundState::Super45 => {
                if self.period == 0 {
                    return d;
                }

                if d >= 0 {
                    let v = (d.wrapping_add(self.threshold - self.phase) / self.period) * self.period + self.phase;
                    if v < 0 { self.phase } else { v }
                } else {
                    let v = -(((self.threshold - self.phase).wrapping_sub(d) / self.period) * self.period) - self.phase;
                    if v > 0 { -self.phase } else { v }
                }
            }
        }
    }

    /// `Round_None`
    fn round_none(d: i64) -> i64 {
        if d >= 0 { d.max(0) } else { d.min(0) }
    }

    /// `SetSuperRound`
    fn set_super_round(&mut self, grid_period: i64, selector: i64) {
        self.period = match selector & 0xC0 {
            0 => grid_period / 2,
            0x80 => grid_period * 2,
            _ => grid_period,
        };

        self.phase = match selector & 0x30 {
            0 => 0,
            0x10 => self.period / 4,
            0x20 => self.period / 2,
            _ => self.period * 3 / 4,
        };

        self.threshold = if selector & 0x0F == 0 { self.period - 1 } else { ((selector & 0x0F) - 4) * self.period / 8 };
        self.period >>= 8;
        self.phase >>= 8;
        self.threshold >>= 8;
    }

    // --- stack -------------------------------------------------------

    fn pop(&mut self) -> Run<i64> {
        self.stack.pop().ok_or(Abort)
    }

    fn push(&mut self, v: i64) -> Run<()> {
        if self.stack.len() >= self.h.max_stack {
            return Err(Abort);
        }

        self.stack.push(v);
        Ok(())
    }

    fn read_cvt(&self, i: i64) -> i64 {
        usize::try_from(i).ok().and_then(|i| self.cvt.get(i)).copied().unwrap_or(0)
    }

    /// A loop instruction that cannot run (too few arguments, a bad
    /// reference point): the stack left as it is, the loop reset.
    fn fail_loop(&mut self) -> Run<Option<usize>> {
        self.gs.loop_ = 1;
        Ok(None)
    }

    // --- the run loop --------------------------------------------------

    /// `TT_RunIns` over a program.
    fn run_code(&mut self, code: &'a [u8]) -> Run<()> {
        if self.range == Range::Glyph {
            self.glyph_code = code;
        }

        self.ip = 0;

        // backward compatibility as the font asks, for each program
        self.backward_compatibility = self.subpixel_hinting_lean && self.gs.instruct_control & 4 == 0;
        self.iupx_called = false;
        self.iupy_called = false;

        let mut steps: u64 = 0;

        loop {
            let code_len = self.code(self.range).len();

            if self.ip >= code_len {
                if self.calls.is_empty() {
                    return Ok(());
                }

                return Err(Abort);
            }

            steps += 1;

            if steps > 10_000_000 {
                return Err(Abort);
            }

            self.opcode = self.code(self.range)[self.ip];
            let len = instruction_len(self.code(self.range), self.ip).ok_or(Abort)?;

            if self.ip + len > code_len {
                return Err(Abort);
            }

            let trace = self.trace && self.ini_range == Range::Glyph;
            let before = if trace { self.pts.cur.clone() } else { Vec::new() };
            let (free_before, proj_before) = (self.gs.free, self.gs.proj);
            let stack_top: Vec<i64> = if trace { self.stack[self.stack.len().saturating_sub(4)..].to_vec() } else { Vec::new() };
            let next = match self.step(len) {
                Ok(next) => next,
                Err(e) => {
                    if self.trace {
                        eprintln!("abort: {:?} ip {} opcode {:#04x} stack {:?} calls {}", self.range, self.ip, self.opcode, &self.stack[self.stack.len().saturating_sub(8)..], self.calls.len());
                    }

                    return Err(e);
                }
            };
            if trace && (self.gs.free != free_before || self.gs.proj != proj_before) {
                eprintln!("{:?}@{} op {:#04x} args {:?}: free {:?} -> {:?} proj {:?} -> {:?}", self.range, self.ip, self.opcode, stack_top, free_before, self.gs.free, proj_before, self.gs.proj);
            }

            if trace {
                for (i, (a, b)) in before.iter().zip(&self.pts.cur).enumerate() {
                    if a != b {
                        eprintln!("{:?}@{} op {:#04x} args {:?}: point {} {:?} -> {:?} proj {:?} free {:?} dual {:?} fdp {}", self.range, self.ip, self.opcode, stack_top, i, a, b, self.gs.proj, self.gs.free, self.gs.dual, self.f_dot_p);
                    }
                }
            }

            self.ip = next.unwrap_or(self.ip + len);
        }
    }

    /// `SkipCode` from `ip`: the next instruction's opcode and place.
    fn skip(&self, ip: usize) -> Option<(u8, usize)> {
        let code = self.code(self.range);
        let next = ip + instruction_len(code, ip)?;
        code.get(next).map(|&op| (op, next))
    }

    /// Runs the instruction at `ip`; the next `ip` when it jumps.
    fn step(&mut self, len: usize) -> Run<Option<usize>> {
        let op = self.opcode;

        match op {
            0x00..=0x05 => {
                let a = if op & 1 != 0 { (0x4000, 0) } else { (0, 0x4000) };

                match op >> 1 {
                    0 => {
                        self.gs.proj = a;
                        self.gs.free = a;
                        self.gs.dual = a;
                    }
                    1 => {
                        self.gs.proj = a;
                        self.gs.dual = a;
                    }
                    _ => self.gs.free = a,
                }

                self.compute_funcs();
            }
            0x06..=0x09 => {
                let p2 = self.pop()? as u16 as usize;
                let p1 = self.pop()? as u16 as usize;
                self.sxvtl(p2, p1, op);
                self.compute_funcs();
            }
            0x0A | 0x0B => {
                let y = i64::from(self.pop()? as i16);
                let x = i64::from(self.pop()? as i16);

                if op == 0x0A {
                    normalize(x, y, &mut self.gs.proj);
                    self.gs.dual = self.gs.proj;
                } else {
                    normalize(x, y, &mut self.gs.free);
                }

                self.compute_funcs();
            }
            0x0C | 0x0D => {
                let v = if op == 0x0C { self.gs.proj } else { self.gs.free };
                self.push(v.0)?;
                self.push(v.1)?;
            }
            0x0E => {
                self.gs.free = self.gs.proj;
                self.compute_funcs();
            }
            0x0F => self.isect()?,
            0x10 => self.gs.rp0 = self.pop()? as u16 as usize,
            0x11 => self.gs.rp1 = self.pop()? as u16 as usize,
            0x12 => self.gs.rp2 = self.pop()? as u16 as usize,
            0x13..=0x16 => {
                let z = self.pop()?;

                if !(0..=1).contains(&z) {
                    return Ok(None);
                }

                let glyph = z == 1;

                match op {
                    0x13 => {
                        self.zp[0] = glyph;
                        self.gs.gep0 = z as u8;
                    }
                    0x14 => {
                        self.zp[1] = glyph;
                        self.gs.gep1 = z as u8;
                    }
                    0x15 => {
                        self.zp[2] = glyph;
                        self.gs.gep2 = z as u8;
                    }
                    _ => {
                        self.zp = [glyph; 3];
                        self.gs.gep0 = z as u8;
                        self.gs.gep1 = z as u8;
                        self.gs.gep2 = z as u8;
                    }
                }
            }
            0x17 => {
                let n = self.pop()?;

                if n < 0 {
                    return Err(Abort);
                }

                self.gs.loop_ = n.min(0xFFFF);
            }
            0x18 => self.gs.round_state = RoundState::Grid,
            0x19 => self.gs.round_state = RoundState::HalfGrid,
            0x1A => self.gs.minimum_distance = self.pop()?,
            0x1B => {
                // ELSE met while running: on past the matching EIF
                let mut nest = 1;
                let mut at = self.ip;

                while nest > 0 {
                    let (op, next) = self.skip(at).ok_or(Abort)?;
                    at = next;

                    match op {
                        0x58 => nest += 1,
                        0x59 => nest -= 1,
                        _ => {}
                    }
                }

                return Ok(Some(at + 1));
            }
            0x1C | 0x78 | 0x79 => {
                let (offset, jump) = if op == 0x1C {
                    let o = self.pop()?;
                    (o, true)
                } else {
                    let cond = self.pop()?;
                    let o = self.pop()?;
                    (o, (cond != 0) == (op == 0x78))
                };

                if jump {
                    if offset == 0 && op == 0x1C && self.stack.is_empty() {
                        return Err(Abort);
                    }

                    let target = self.ip as i64 + offset;

                    if target < 0 {
                        return Err(Abort);
                    }

                    if let Some(call) = self.calls.last()
                        && target as usize > call.def.end {
                            return Err(Abort);
                        }

                    return Ok(Some(target as usize));
                }
            }
            0x1D => self.gs.control_value_cutin = self.pop()?,
            0x1E => self.gs.single_width_cutin = self.pop()?,
            0x1F => {
                let v = self.pop()?;
                self.gs.single_width_value = mul_fix(v, self.h.scale);
            }
            0x20 => {
                let v = *self.stack.last().ok_or(Abort)?;
                self.push(v)?;
            }
            0x21 => {
                self.pop()?;
            }
            0x22 => self.stack.clear(),
            0x23 => {
                let n = self.stack.len();

                if n < 2 {
                    return Err(Abort);
                }

                self.stack.swap(n - 1, n - 2);
            }
            0x24 => {
                let n = self.stack.len() as i64;
                self.push(n)?;
            }
            0x25 => {
                let l = self.pop()?;
                let n = self.stack.len() as i64;
                let v = if l <= 0 || l > n { 0 } else { self.stack[(n - l) as usize] };
                self.push(v)?;
            }
            0x26 => {
                let l = self.pop()?;
                let n = self.stack.len() as i64;

                if l <= 0 || l > n {
                    return Ok(None);
                }

                let v = self.stack.remove((n - l) as usize);
                self.stack.push(v);
            }
            0x27 => {
                let p2 = self.pop()? as u16 as usize;
                let p1 = self.pop()? as u16 as usize;

                if p1 < self.zone(1).len() && p2 < self.zone(0).len() {
                    let d = self.project_pts(self.zone(0).cur[p2], self.zone(1).cur[p1]) / 2;
                    self.move_point(1, p1, d);
                    self.move_point(0, p2, d.wrapping_neg());
                }
            }
            0x29 => {
                let p = self.pop()? as u16 as usize;

                if p < self.zone(0).len() {
                    let mut mask = 0xFFu8;

                    if self.gs.free.0 != 0 {
                        mask &= !TOUCH_X;
                    }

                    if self.gs.free.1 != 0 {
                        mask &= !TOUCH_Y;
                    }

                    self.zone_mut(0).tags[p] &= mask;
                }
            }
            0x2A | 0x2B => {
                let f = self.pop()?;
                let count = if op == 0x2A { self.pop()? } else { 1 };
                let def = usize::try_from(f).ok().and_then(|f| self.h.fdefs.get(f).copied().flatten()).ok_or(Abort)?;

                if count <= 0 {
                    return Ok(None);
                }

                if self.calls.len() >= 32 {
                    return Err(Abort);
                }

                self.calls.push(Call { range: self.range, ip: self.ip + len, def, count });
                self.range = def.range;
                return Ok(Some(def.start));
            }
            0x2C => {
                let f = self.pop()?;

                if self.ini_range == Range::Glyph {
                    return Err(Abort);
                }

                let f = usize::try_from(f).map_err(|_| Abort)?;

                if f > 0xFFFF {
                    return Err(Abort);
                }

                if f >= self.h.fdefs.len() {
                    let used = self.h.fdefs.iter().filter(|d| d.is_some()).count();

                    if used >= self.h.fdefs.len() {
                        return Err(Abort);
                    }

                    self.h.fdefs.resize(f + 1, None);
                }

                let start = self.ip + 1;
                let mut at = self.ip;

                loop {
                    let (op, next) = self.skip(at).ok_or(Abort)?;
                    at = next;

                    match op {
                        0x89 | 0x2C => return Err(Abort),
                        0x2D => break,
                        _ => {}
                    }
                }

                self.h.fdefs[f] = Some(Def { range: self.range, start, end: at });
                return Ok(Some(at + 1));
            }
            0x2D => {
                let mut call = self.calls.pop().ok_or(Abort)?;
                call.count -= 1;

                if call.count > 0 {
                    let start = call.def.start;
                    self.calls.push(call);
                    return Ok(Some(start));
                }

                self.range = call.range;
                return Ok(Some(call.ip));
            }
            0x2E | 0x2F => {
                let p = self.pop()? as u16 as usize;

                if p < self.zone(0).len() {
                    let d = if op & 1 != 0 {
                        let c = self.zone(0).cur[p];
                        let cur = self.project(c.0, c.1);
                        self.round(cur).wrapping_sub(cur)
                    } else {
                        0
                    };

                    self.move_point(0, p, d);
                }

                self.gs.rp0 = p;
                self.gs.rp1 = p;
            }
            0x30 | 0x31 => self.iup(op & 1 != 0),
            0x32 | 0x33 => {
                if (self.stack.len() as i64) < self.gs.loop_ {
                    return self.fail_loop();
                }

                let Some((dx, dy, _, _)) = self.displacement() else { return Ok(None) };

                while self.gs.loop_ > 0 {
                    let p = self.pop()? as u16 as usize;

                    if p < self.zone(2).len() {
                        self.move_zp2(p, dx, dy, true);
                    }

                    self.gs.loop_ -= 1;
                }

                self.gs.loop_ = 1;
            }
            0x34 | 0x35 => {
                let contour = self.pop()? as u16 as usize;
                let bounds = if self.gs.gep2 == 0 { 1 } else { self.zone(2).contours.len() };

                if contour >= bounds {
                    return Ok(None);
                }

                let Some((dx, dy, refz, refp)) = self.displacement() else { return Ok(None) };
                let start = if contour == 0 { 0 } else { self.zone(2).contours[contour - 1] + 1 };
                let limit = if self.gs.gep2 == 0 { self.zone(2).len() } else { self.zone(2).contours[contour] + 1 };
                let same_zone = self.zp[refz] == self.zp[2];

                for i in start..limit.min(self.zone(2).len()) {
                    if !same_zone || refp != i {
                        self.move_zp2(i, dx, dy, true);
                    }
                }
            }
            0x36 | 0x37 => {
                let e = self.pop()?;

                if !(0..2).contains(&e) {
                    return Ok(None);
                }

                let Some((dx, dy, refz, refp)) = self.displacement() else { return Ok(None) };

                let limit = if self.gs.gep2 == 0 {
                    self.zone(2).len()
                } else if self.gs.gep2 == 1 && !self.zone(2).contours.is_empty() {
                    self.zone(2).contours.last().map_or(0, |c| c + 1)
                } else {
                    0
                };
                let same_zone = self.zp[refz] == self.zp[2];

                for i in 0..limit {
                    if !same_zone || refp != i {
                        self.move_zp2(i, dx, dy, false);
                    }
                }
            }
            0x38 => {
                let amount = self.pop()?;

                if (self.stack.len() as i64) < self.gs.loop_ {
                    return self.fail_loop();
                }

                let dx = mul_fix14(amount, self.gs.free.0);
                let dy = mul_fix14(amount, self.gs.free.1);
                let in_twilight = self.gs.gep0 == 0 && self.gs.gep1 == 0 && self.gs.gep2 == 0;

                while self.gs.loop_ > 0 {
                    let p = self.pop()? as u16 as usize;

                    if p >= self.zone(2).len() {
                        self.gs.loop_ -= 1;
                        continue;
                    }

                    if self.backward_compatibility {
                        let touched_y = self.zone(2).tags[p] & TOUCH_Y != 0;

                        if in_twilight
                            || (!(self.iupx_called && self.iupy_called) && ((self.is_composite && self.gs.free.1 != 0) || touched_y))
                        {
                            self.move_zp2(p, 0, dy, true);
                        }
                    } else {
                        self.move_zp2(p, dx, dy, true);
                    }

                    self.gs.loop_ -= 1;
                }

                self.gs.loop_ = 1;
            }
            0x39 => self.ip_instruction()?,
            0x3A | 0x3B => {
                let d = self.pop()?;
                let p = self.pop()? as u16 as usize;
                let rp0 = self.gs.rp0;

                if p >= self.zone(1).len() || rp0 >= self.zone(0).len() {
                    return Ok(None);
                }

                if self.gs.gep1 == 0 {
                    let o = self.zone(0).org[rp0];
                    self.zone_mut(1).org[p] = o;
                    self.move_orig(1, p, d);
                    let o = self.zone(1).org[p];
                    self.zone_mut(1).cur[p] = o;
                }

                let dist = self.project_pts(self.zone(1).cur[p], self.zone(0).cur[rp0]);
                self.move_point(1, p, d.wrapping_sub(dist));
                self.gs.rp1 = self.gs.rp0;
                self.gs.rp2 = p;

                if op & 1 != 0 {
                    self.gs.rp0 = p;
                }
            }
            0x3C => {
                if (self.stack.len() as i64) < self.gs.loop_ || self.gs.rp0 >= self.zone(0).len() {
                    return self.fail_loop();
                }

                while self.gs.loop_ > 0 {
                    let p = self.pop()? as u16 as usize;

                    if p >= self.zone(1).len() {
                        self.gs.loop_ -= 1;
                        continue;
                    }

                    let d = self.project_pts(self.zone(1).cur[p], self.zone(0).cur[self.gs.rp0]);
                    self.move_point(1, p, d.wrapping_neg());
                    self.gs.loop_ -= 1;
                }

                self.gs.loop_ = 1;
            }
            0x3D => self.gs.round_state = RoundState::DoubleGrid,
            0x3E | 0x3F => {
                let entry = self.pop()?;
                let p = self.pop()? as u16 as usize;

                if p >= self.zone(0).len() || entry < 0 || entry as usize >= self.cvt.len() {
                    self.gs.rp0 = p;
                    self.gs.rp1 = p;
                    return Ok(None);
                }

                let mut d = self.read_cvt(entry);

                if self.gs.gep0 == 0 {
                    let o = (mul_fix14(d, self.gs.free.0), mul_fix14(d, self.gs.free.1));
                    let z = self.zone_mut(0);
                    z.org[p] = o;
                    z.cur[p] = o;
                }

                let c = self.zone(0).cur[p];
                let org_dist = self.project(c.0, c.1);

                if op & 1 != 0 {
                    if (d - org_dist).abs() > self.gs.control_value_cutin {
                        d = org_dist;
                    }

                    d = self.round(d);
                }

                self.move_point(0, p, d.wrapping_sub(org_dist));
                self.gs.rp0 = p;
                self.gs.rp1 = p;
            }
            0x40 | 0x41 | 0xB0..=0xBF => {
                let code = self.code(self.range);
                let (n, words, at) = match op {
                    0x40 => (usize::from(code[self.ip + 1]), false, self.ip + 2),
                    0x41 => (usize::from(code[self.ip + 1]), true, self.ip + 2),
                    0xB0..=0xB7 => (usize::from(op - 0xB0) + 1, false, self.ip + 1),
                    _ => (usize::from(op - 0xB8) + 1, true, self.ip + 1),
                };
                let values: Vec<i64> = (0..n)
                    .map(|i| {
                        if words {
                            i64::from(i16::from_be_bytes([code[at + 2 * i], code[at + 2 * i + 1]]))
                        } else {
                            i64::from(code[at + i])
                        }
                    })
                    .collect();

                for v in values {
                    self.push(v)?;
                }
            }
            0x42 => {
                let v = self.pop()?;
                let i = self.pop()?;

                if let Some(s) = usize::try_from(i).ok().and_then(|i| self.storage.get_mut(i)) {
                    *s = v;
                }
            }
            0x43 => {
                let i = self.pop()?;
                let v = usize::try_from(i).ok().and_then(|i| self.storage.get(i)).copied().unwrap_or(0);
                self.push(v)?;
            }
            0x44 | 0x70 => {
                let v = self.pop()?;
                let i = self.pop()?;
                let v = if op == 0x70 { mul_fix(v, self.h.scale) } else { v };

                if let Some(c) = usize::try_from(i).ok().and_then(|i| self.cvt.get_mut(i)) {
                    *c = v;
                }
            }
            0x45 => {
                let i = self.pop()?;
                let v = self.read_cvt(i);
                self.push(v)?;
            }
            0x46 | 0x47 => {
                let l = self.pop()?;
                let v = match usize::try_from(l).ok().filter(|&l| l < self.zone(2).len()) {
                    Some(l) if op & 1 != 0 => {
                        let o = self.zone(2).org[l];
                        self.dual_project(o.0, o.1)
                    }
                    Some(l) => {
                        let c = self.zone(2).cur[l];
                        self.project(c.0, c.1)
                    }
                    None => 0,
                };
                self.push(v)?;
            }
            0x48 => {
                let value = self.pop()?;
                let k = self.pop()? as u16 as usize;

                if k < self.zone(2).len() {
                    let c = self.zone(2).cur[k];
                    let cur = self.project(c.0, c.1);
                    self.move_point(2, k, value.wrapping_sub(cur));

                    if self.gs.gep2 == 0 {
                        let c = self.zone(2).cur[k];
                        self.zone_mut(2).org[k] = c;
                    }
                }
            }
            0x49 | 0x4A => {
                let k = self.pop()? as u16 as usize;
                let l = self.pop()? as u16 as usize;

                let d = if l >= self.zone(0).len() || k >= self.zone(1).len() {
                    0
                } else if op & 1 != 0 {
                    self.project_pts(self.zone(0).cur[l], self.zone(1).cur[k])
                } else if self.gs.gep0 == 0 || self.gs.gep1 == 0 {
                    self.dual_pts(self.zone(0).org[l], self.zone(1).org[k])
                } else {
                    self.orus_distance(self.zone(0).orus[l], self.zone(1).orus[k])
                };

                self.push(d)?;
            }
            0x4B => self.push(self.h.ppem)?,
            0x4C => self.push(self.h.point_size)?,
            0x4D => self.gs.auto_flip = true,
            0x4E => self.gs.auto_flip = false,
            0x4F => return Err(Abort),
            0x50..=0x55 => {
                let b = self.pop()?;
                let a = self.pop()?;
                let r = match op {
                    0x50 => a < b,
                    0x51 => a <= b,
                    0x52 => a > b,
                    0x53 => a >= b,
                    0x54 => a == b,
                    _ => a != b,
                };
                self.push(i64::from(r))?;
            }
            0x56 | 0x57 => {
                let a = self.pop()?;
                let r = self.round(a) & 127;
                self.push(i64::from(if op == 0x56 { r == 64 } else { r == 0 }))?;
            }
            0x58 => {
                let cond = self.pop()?;

                if cond == 0 {
                    let mut nest = 1;
                    let mut at = self.ip;

                    loop {
                        let (op, next) = self.skip(at).ok_or(Abort)?;
                        at = next;

                        match op {
                            0x58 => nest += 1,
                            0x1B if nest == 1 => return Ok(Some(at + 1)),
                            0x59 => {
                                nest -= 1;

                                if nest == 0 {
                                    return Ok(Some(at + 1));
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            0x59 => {}
            0x5A | 0x5B => {
                let b = self.pop()?;
                let a = self.pop()?;
                self.push(i64::from(if op == 0x5A { a != 0 && b != 0 } else { a != 0 || b != 0 }))?;
            }
            0x5C => {
                let a = self.pop()?;
                self.push(i64::from(a == 0))?;
            }
            0x5D | 0x71 | 0x72 => self.deltap()?,
            0x5E => self.gs.delta_base = i64::from(self.pop()? as u16),
            0x5F => {
                let v = self.pop()?;

                if v as u64 > 6 {
                    return Err(Abort);
                }

                self.gs.delta_shift = v;
            }
            0x60..=0x63 | 0x8B | 0x8C => {
                let b = self.pop()?;
                let a = self.pop()?;
                let r = match op {
                    0x60 => a.wrapping_add(b),
                    0x61 => a.wrapping_sub(b),
                    0x62 => {
                        if b == 0 {
                            return Err(Abort);
                        }

                        mul_div_no_round(a, 64, b)
                    }
                    0x63 => mul_div(a, b, 64),
                    0x8B => a.max(b),
                    _ => a.min(b),
                };
                self.push(r)?;
            }
            0x64..=0x6F => {
                let a = self.pop()?;
                let r = match op {
                    0x64 => a.abs(),
                    0x65 => a.wrapping_neg(),
                    0x66 => pix_floor(a),
                    0x67 => pix_ceil(a),
                    0x68..=0x6B => self.round(a),
                    _ => Self::round_none(a),
                };
                self.push(r)?;
            }
            0x73..=0x75 => self.deltac()?,
            0x76 | 0x77 => {
                let s = self.pop()?;
                self.gs.round_state = if op == 0x76 { RoundState::Super } else { RoundState::Super45 };
                self.set_super_round(if op == 0x76 { 0x4000 } else { 0x2D41 }, s);
            }
            0x7A => self.gs.round_state = RoundState::Off,
            0x7C => self.gs.round_state = RoundState::UpToGrid,
            0x7D => self.gs.round_state = RoundState::DownToGrid,
            0x7E | 0x7F => {
                self.pop()?;
            }
            0x80 => {
                if self.backward_compatibility && self.iupx_called && self.iupy_called {
                    return self.fail_loop();
                }

                if (self.stack.len() as i64) < self.gs.loop_ {
                    return self.fail_loop();
                }

                while self.gs.loop_ > 0 {
                    let p = self.pop()? as u16 as usize;

                    if p < self.pts.len() {
                        self.pts.tags[p] ^= TAG_ON;
                    }

                    self.gs.loop_ -= 1;
                }

                self.gs.loop_ = 1;
            }
            0x81 | 0x82 => {
                let k = self.pop()? as u16 as usize;
                let l = self.pop()? as u16 as usize;

                if !(self.backward_compatibility && self.iupx_called && self.iupy_called) && k < self.pts.len() && l < self.pts.len() {
                    for tag in &mut self.pts.tags[l..=k] {
                        if op == 0x81 {
                            *tag |= TAG_ON;
                        } else {
                            *tag &= !TAG_ON;
                        }
                    }
                }
            }
            0x85 | 0x8D => {
                self.pop()?;
            }
            0x86 | 0x87 => self.sdpvtl()?,
            0x88 => {
                let a = self.pop()?;
                let k = self.getinfo(a);
                self.push(k)?;
            }
            0x89 => {
                let n = self.pop()?;

                if self.ini_range == Range::Glyph {
                    return Err(Abort);
                }

                let start = self.ip + 1;
                let mut at = self.ip;

                loop {
                    let (op, next) = self.skip(at).ok_or(Abort)?;
                    at = next;

                    match op {
                        0x89 | 0x2C => return Err(Abort),
                        0x2D => break,
                        _ => {}
                    }
                }

                if (0..256).contains(&n) {
                    let n = n as usize;

                    if self.h.idefs.len() <= n {
                        self.h.idefs.resize(n + 1, None);
                    }

                    self.h.idefs[n] = Some(Def { range: self.range, start, end: at });
                }

                return Ok(Some(at + 1));
            }
            0x8A => {
                let a = self.pop()?;
                let b = self.pop()?;
                let c = self.pop()?;
                self.push(b)?;
                self.push(a)?;
                self.push(c)?;
            }
            0x8E => {
                let k = self.pop()?;
                let l = self.pop()?;
                self.instctrl(k, l);
            }
            0xC0..=0xDF => self.mdrp()?,
            0xE0..=0xFF => self.mirp()?,
            _ => {
                // a user instruction (IDEF), else an invalid opcode
                let def = self.h.idefs.get(usize::from(op)).copied().flatten().ok_or(Abort)?;

                if self.calls.len() >= 32 {
                    return Err(Abort);
                }

                self.calls.push(Call { range: self.range, ip: self.ip + len, def, count: 1 });
                self.range = def.range;
                return Ok(Some(def.start));
            }
        }

        Ok(None)
    }

    /// A distance of original points in font units, scaled (MD and MDRP).
    fn orus_distance(&self, a: (i64, i64), b: (i64, i64)) -> i64 {
        if self.x_scale == self.y_scale {
            mul_fix(self.dual_pts(a, b), self.x_scale)
        } else {
            let v = (mul_fix(a.0.wrapping_sub(b.0), self.x_scale), mul_fix(a.1.wrapping_sub(b.1), self.y_scale));
            self.dual_project(v.0, v.1)
        }
    }

    /// IP's distances of original points: unscaled when both axes scale
    /// alike, as the ratio takes the scale out.
    fn orus_range(&self, a: (i64, i64), b: (i64, i64)) -> i64 {
        if self.x_scale == self.y_scale {
            self.dual_pts(a, b)
        } else {
            let v = (mul_fix(a.0.wrapping_sub(b.0), self.x_scale), mul_fix(a.1.wrapping_sub(b.1), self.y_scale));
            self.dual_project(v.0, v.1)
        }
    }

    /// `Ins_SxVTL` for SPVTL and SFVTL.
    fn sxvtl(&mut self, i1: usize, i2: usize, op: u8) {
        if i1 >= self.zone(2).len() || i2 >= self.zone(1).len() {
            return;
        }

        let p1 = self.zone(1).cur[i2];
        let p2 = self.zone(2).cur[i1];
        let (mut a, mut b) = (p1.0.wrapping_sub(p2.0), p1.1.wrapping_sub(p2.1));
        let mut rotate = op & 1 != 0;

        if a == 0 && b == 0 {
            a = 0x4000;
            rotate = false;
        }

        if rotate {
            let c = b;
            b = a;
            a = c.wrapping_neg();
        }

        if op <= 0x07 {
            normalize(a, b, &mut self.gs.proj);
            self.gs.dual = self.gs.proj;
        } else {
            normalize(a, b, &mut self.gs.free);
        }
    }

    /// `Ins_SDPVTL`
    fn sdpvtl(&mut self) -> Run<()> {
        let p2 = self.pop()? as u16 as usize;
        let p1 = self.pop()? as u16 as usize;

        if p2 >= self.zone(1).len() || p1 >= self.zone(2).len() {
            return Ok(());
        }

        let mut rotate = self.opcode & 1 != 0;
        let v1 = self.zone(1).org[p2];
        let v2 = self.zone(2).org[p1];
        let (mut a, mut b) = (v1.0.wrapping_sub(v2.0), v1.1.wrapping_sub(v2.1));

        if a == 0 && b == 0 {
            a = 0x4000;
            rotate = false;
        }

        if rotate {
            let c = b;
            b = a;
            a = c.wrapping_neg();
        }

        normalize(a, b, &mut self.gs.dual);

        let v1 = self.zone(1).cur[p2];
        let v2 = self.zone(2).cur[p1];
        let (mut a, mut b) = (v1.0.wrapping_sub(v2.0), v1.1.wrapping_sub(v2.1));

        if a == 0 && b == 0 {
            a = 0x4000;
            rotate = false;
        }

        if rotate {
            let c = b;
            b = a;
            a = c.wrapping_neg();
        }

        normalize(a, b, &mut self.gs.proj);
        self.compute_funcs();
        Ok(())
    }

    /// `Ins_ISECT`
    fn isect(&mut self) -> Run<()> {
        let b1 = self.pop()? as u16 as usize;
        let b0 = self.pop()? as u16 as usize;
        let a1 = self.pop()? as u16 as usize;
        let a0 = self.pop()? as u16 as usize;
        let p = self.pop()? as u16 as usize;

        if b0 >= self.zone(0).len() || b1 >= self.zone(0).len() || a0 >= self.zone(1).len() || a1 >= self.zone(1).len() || p >= self.zone(2).len() {
            return Ok(());
        }

        let (zb0, zb1) = (self.zone(0).cur[b0], self.zone(0).cur[b1]);
        let (za0, za1) = (self.zone(1).cur[a0], self.zone(1).cur[a1]);
        let (dbx, dby) = (zb1.0 - zb0.0, zb1.1 - zb0.1);
        let (dax, day) = (za1.0 - za0.0, za1.1 - za0.1);
        let (dx, dy) = (zb0.0 - za0.0, zb0.1 - za0.1);
        let discriminant = mul_div(dax, -dby, 0x40) + mul_div(day, dbx, 0x40);
        let dotproduct = mul_div(dax, dbx, 0x40) + mul_div(day, dby, 0x40);

        let point = if 19 * discriminant.abs() > dotproduct.abs() {
            let val = mul_div(dx, -dby, 0x40) + mul_div(dy, dbx, 0x40);
            (za0.0 + mul_div(val, dax, discriminant), za0.1 + mul_div(val, day, discriminant))
        } else {
            ((za0.0 + za1.0 + zb0.0 + zb1.0) / 4, (za0.1 + za1.1 + zb0.1 + zb1.1) / 4)
        };

        let z = self.zone_mut(2);
        z.cur[p] = point;
        z.tags[p] |= TOUCH_X | TOUCH_Y;
        Ok(())
    }

    /// `Compute_Point_Displacement` of SHP, SHC and SHZ: the move, the
    /// reference point's zone and number.
    fn displacement(&mut self) -> Option<(i64, i64, usize, usize)> {
        let (z, p) = if self.opcode & 1 != 0 { (0, self.gs.rp1) } else { (1, self.gs.rp2) };

        if p >= self.zone(z).len() {
            return None;
        }

        let zone = self.zone(z);
        let d = self.project_pts(zone.cur[p], zone.org[p]);
        Some((mul_div(d, self.gs.free.0, self.f_dot_p), mul_div(d, self.gs.free.1, self.f_dot_p), z, p))
    }

    /// `Ins_IP`
    fn ip_instruction(&mut self) -> Run<()> {
        if (self.stack.len() as i64) < self.gs.loop_ {
            self.gs.loop_ = 1;
            return Ok(());
        }

        let in_twilight = self.gs.gep0 == 0 || self.gs.gep1 == 0 || self.gs.gep2 == 0;
        let (rp1, rp2) = (self.gs.rp1, self.gs.rp2);

        if rp1 >= self.zone(0).len() {
            self.gs.loop_ = 1;
            return Ok(());
        }

        let orus_base = if in_twilight { self.zone(0).org[rp1] } else { self.zone(0).orus[rp1] };
        let cur_base = self.zone(0).cur[rp1];

        let (old_range, cur_range) = if rp2 >= self.zone(1).len() {
            (0, 0)
        } else {
            let old = if in_twilight {
                self.dual_pts(self.zone(1).org[rp2], orus_base)
            } else {
                self.orus_range(self.zone(1).orus[rp2], orus_base)
            };
            (old, self.project_pts(self.zone(1).cur[rp2], cur_base))
        };

        while self.gs.loop_ > 0 {
            let p = self.pop()? as u16 as usize;

            if p >= self.zone(2).len() {
                self.gs.loop_ -= 1;
                continue;
            }

            let org_dist = if in_twilight {
                self.dual_pts(self.zone(2).org[p], orus_base)
            } else {
                self.orus_range(self.zone(2).orus[p], orus_base)
            };
            let cur_dist = self.project_pts(self.zone(2).cur[p], cur_base);

            let new_dist = if org_dist != 0 {
                if old_range != 0 { mul_div(org_dist, cur_range, old_range) } else { org_dist }
            } else {
                0
            };

            self.move_point(2, p, new_dist.wrapping_sub(cur_dist));
            self.gs.loop_ -= 1;
        }

        self.gs.loop_ = 1;
        Ok(())
    }

    /// `Ins_MDRP`
    fn mdrp(&mut self) -> Run<()> {
        let p = self.pop()? as u16 as usize;
        let rp0 = self.gs.rp0;
        let op = self.opcode;

        if p < self.zone(1).len() && rp0 < self.zone(0).len() {
            let mut org_dist = if self.gs.gep0 == 0 || self.gs.gep1 == 0 {
                self.dual_pts(self.zone(1).org[p], self.zone(0).org[rp0])
            } else {
                self.orus_distance(self.zone(1).orus[p], self.zone(0).orus[rp0])
            };

            if (org_dist - self.gs.single_width_value).abs() < self.gs.single_width_cutin {
                org_dist = if org_dist >= 0 { self.gs.single_width_value } else { -self.gs.single_width_value };
            }

            let mut distance = if op & 4 != 0 { self.round(org_dist) } else { Self::round_none(org_dist) };

            if op & 8 != 0 {
                let md = self.gs.minimum_distance;

                if org_dist >= 0 {
                    if distance < md {
                        distance = md;
                    }
                } else if distance > -md {
                    distance = -md;
                }
            }

            let cur = self.project_pts(self.zone(1).cur[p], self.zone(0).cur[rp0]);

            if self.trace {
                eprintln!(
                    "MDRP p {p} rp0 {rp0} gep {} {} orus {:?} {:?} org {:?} {:?} cur {:?} {:?} org_dist {org_dist} distance {distance} cur_dist {cur} round {:?}",
                    self.gs.gep0, self.gs.gep1, self.zone(1).orus[p], self.zone(0).orus[rp0], self.zone(1).org[p], self.zone(0).org[rp0],
                    self.zone(1).cur[p], self.zone(0).cur[rp0], self.gs.round_state
                );
            }

            self.move_point(1, p, distance.wrapping_sub(cur));
        }

        self.gs.rp1 = self.gs.rp0;
        self.gs.rp2 = p;

        if op & 16 != 0 {
            self.gs.rp0 = p;
        }

        Ok(())
    }

    /// `Ins_MIRP`
    fn mirp(&mut self) -> Run<()> {
        let entry = self.pop()?.wrapping_add(1);
        let p = self.pop()? as u16 as usize;
        let rp0 = self.gs.rp0;
        let op = self.opcode;

        let fail = p >= self.zone(1).len() || entry < 0 || entry as u64 > self.cvt.len() as u64 || rp0 >= self.zone(0).len();

        if !fail {
            let mut cvt_dist = if entry == 0 { 0 } else { self.read_cvt(entry - 1) };

            if (cvt_dist - self.gs.single_width_value).abs() < self.gs.single_width_cutin {
                cvt_dist = if cvt_dist >= 0 { self.gs.single_width_value } else { -self.gs.single_width_value };
            }

            if self.gs.gep1 == 0 {
                let base = self.zone(0).org[rp0];
                let o = (base.0 + mul_fix14(cvt_dist, self.gs.free.0), base.1 + mul_fix14(cvt_dist, self.gs.free.1));
                let z = self.zone_mut(1);
                z.org[p] = o;
                z.cur[p] = o;
            }

            let org_dist = self.dual_pts(self.zone(1).org[p], self.zone(0).org[rp0]);
            let cur_dist = self.project_pts(self.zone(1).cur[p], self.zone(0).cur[rp0]);

            if self.gs.auto_flip && (org_dist ^ cvt_dist) < 0 {
                cvt_dist = cvt_dist.wrapping_neg();
            }

            let mut distance = if op & 4 != 0 {
                if self.gs.gep0 == self.gs.gep1 && (cvt_dist - org_dist).abs() > self.gs.control_value_cutin {
                    cvt_dist = org_dist;
                }

                self.round(cvt_dist)
            } else {
                Self::round_none(cvt_dist)
            };

            if op & 8 != 0 {
                let md = self.gs.minimum_distance;

                if org_dist >= 0 {
                    if distance < md {
                        distance = md;
                    }
                } else if distance > -md {
                    distance = -md;
                }
            }

            self.move_point(1, p, distance.wrapping_sub(cur_dist));
        }

        self.gs.rp1 = self.gs.rp0;

        if op & 16 != 0 {
            self.gs.rp0 = p;
        }

        self.gs.rp2 = p;
        Ok(())
    }

    /// `Ins_DELTAP`
    fn deltap(&mut self) -> Run<()> {
        let n = self.pop()?;
        let ppem = self.h.ppem;

        for _ in 0..n.max(0) {
            if self.stack.len() < 2 {
                self.stack.clear();
                return Ok(());
            }

            let a = self.pop()? as u16 as usize;
            let b = self.pop()?;

            if a >= self.zone(0).len() {
                continue;
            }

            let mut c = (b as u64 & 0xF0) as i64 >> 4;
            c += match self.opcode {
                0x71 => 16,
                0x72 => 32,
                _ => 0,
            };
            c += self.gs.delta_base;

            if ppem == c {
                let mut d = (b as u64 & 0xF) as i64 - 8;

                if d >= 0 {
                    d += 1;
                }

                d *= 1 << (6 - self.gs.delta_shift);

                if self.backward_compatibility {
                    let touched_y = self.zone(0).tags[a] & TOUCH_Y != 0;

                    if !(self.iupx_called && self.iupy_called) && ((self.is_composite && self.gs.free.1 != 0) || touched_y) {
                        self.move_point(0, a, d);
                    }
                } else {
                    self.move_point(0, a, d);
                }
            }
        }

        Ok(())
    }

    /// `Ins_DELTAC`
    fn deltac(&mut self) -> Run<()> {
        let n = self.pop()?;
        let ppem = self.h.ppem;

        for _ in 0..n.max(0) {
            if self.stack.len() < 2 {
                self.stack.clear();
                return Ok(());
            }

            let a = self.pop()?;
            let b = self.pop()?;

            if a < 0 || a as usize >= self.cvt.len() {
                return Ok(());
            }

            let mut c = (b as u64 & 0xF0) as i64 >> 4;
            c += match self.opcode {
                0x74 => 16,
                0x75 => 32,
                _ => 0,
            };
            c += self.gs.delta_base;

            if ppem == c {
                let mut d = (b as u64 & 0xF) as i64 - 8;

                if d >= 0 {
                    d += 1;
                }

                d *= 1 << (6 - self.gs.delta_shift);
                let cvt = &mut self.cvt[a as usize];
                *cvt = cvt.wrapping_add(d);
            }
        }

        Ok(())
    }

    /// `Ins_IUP`
    fn iup(&mut self, x: bool) {
        if self.pts.contours.is_empty() {
            return;
        }

        if self.backward_compatibility {
            if self.iupx_called && self.iupy_called {
                return;
            }

            if x {
                self.iupx_called = true;
            } else {
                self.iupy_called = true;
            }
        }

        let mask = if x { TOUCH_X } else { TOUCH_Y };
        let n = self.pts.len();
        let mut point = 0usize;
        let contours = self.pts.contours.clone();

        for &end in &contours {
            let end = end.min(n - 1);
            let first = point;

            while point <= end && self.pts.tags[point] & mask == 0 {
                point += 1;
            }

            if point <= end {
                let first_touched = point;
                let mut cur_touched = point;
                point += 1;

                while point <= end {
                    if self.pts.tags[point] & mask != 0 {
                        self.iup_interpolate(x, cur_touched + 1, point - 1, cur_touched, point);
                        cur_touched = point;
                    }

                    point += 1;
                }

                if cur_touched == first_touched {
                    self.iup_shift(x, first, end, cur_touched);
                } else {
                    self.iup_interpolate(x, cur_touched + 1, end, cur_touched, first_touched);

                    if first_touched > 0 {
                        self.iup_interpolate(x, first, first_touched - 1, cur_touched, first_touched);
                    }
                }
            }
        }
    }

    /// `_iup_worker_shift`
    fn iup_shift(&mut self, x: bool, p1: usize, p2: usize, p: usize) {
        let get = |v: (i64, i64)| if x { v.0 } else { v.1 };
        let d = get(self.pts.cur[p]).wrapping_sub(get(self.pts.org[p]));

        if d == 0 {
            return;
        }

        for i in (p1..p).chain(p + 1..=p2) {
            let c = &mut self.pts.cur[i];

            if x {
                c.0 = c.0.wrapping_add(d);
            } else {
                c.1 = c.1.wrapping_add(d);
            }
        }
    }

    /// `_iup_worker_interpolate`
    fn iup_interpolate(&mut self, x: bool, p1: usize, p2: usize, mut ref1: usize, mut ref2: usize) {
        if p1 > p2 {
            return;
        }

        let n = self.pts.len();

        if ref1 >= n || ref2 >= n {
            return;
        }

        let get = |v: (i64, i64)| if x { v.0 } else { v.1 };
        let mut orus1 = get(self.pts.orus[ref1]);
        let mut orus2 = get(self.pts.orus[ref2]);

        if orus1 > orus2 {
            std::mem::swap(&mut orus1, &mut orus2);
            std::mem::swap(&mut ref1, &mut ref2);
        }

        let (org1, org2) = (get(self.pts.org[ref1]), get(self.pts.org[ref2]));
        let (cur1, cur2) = (get(self.pts.cur[ref1]), get(self.pts.cur[ref2]));
        let (delta1, delta2) = (cur1.wrapping_sub(org1), cur2.wrapping_sub(org2));
        let mut scale = None;

        for i in p1..=p2 {
            let o = get(self.pts.org[i]);
            let v = if o <= org1 {
                o.wrapping_add(delta1)
            } else if o >= org2 {
                o.wrapping_add(delta2)
            } else if cur1 == cur2 || orus1 == orus2 {
                cur1
            } else {
                let s = *scale.get_or_insert_with(|| div_fix(cur2.wrapping_sub(cur1), orus2 - orus1));
                cur1.wrapping_add(mul_fix(get(self.pts.orus[i]) - orus1, s))
            };

            let c = &mut self.pts.cur[i];

            if x {
                c.0 = v;
            } else {
                c.1 = v;
            }
        }
    }

    /// `Ins_GETINFO` of FreeType's version 40 interpreter.
    fn getinfo(&self, a: i64) -> i64 {
        let mut k = 0;

        if a & 1 != 0 {
            k = 40;
        }

        if self.subpixel_hinting_lean {
            if a & 64 != 0 {
                k |= 1 << 13;
            }

            if a & 1024 != 0 {
                k |= 1 << 17;
            }

            if a & 2048 != 0 {
                k |= 1 << 18;
            }

            if a & 4096 != 0 && self.grayscale_cleartype {
                k |= 1 << 19;
            }
        }

        k
    }

    /// `Ins_INSTCTRL`
    fn instctrl(&mut self, k: i64, l: i64) {
        if !(1..=3).contains(&k) {
            return;
        }

        let kf = 1i64 << (k - 1);

        if l != 0 && l != kf {
            return;
        }

        if self.ini_range == Range::Cvt {
            self.gs.instruct_control &= !(kf as u8);
            self.gs.instruct_control |= l as u8;
        } else if self.ini_range == Range::Glyph && k == 3 {
            self.backward_compatibility = l != 4;
        }
    }
}

/// The points of a glyph as hinted, and its phantom points: hinted where
/// the interpreter may move them, else as scaled.
struct Loaded {
    outline: Outline,
    pp: [(i64, i64); 4],
}

/// The vertical phantom points' sources: top bearing and advance height
/// (`TT_Get_VMetrics` without `vmtx`).
fn v_metrics(face: &Face, y_max: i64) -> (i64, i64) {
    if let Some(os2) = face.os2 {
        let asc = i64::from(os2.typo_ascender);
        (asc - y_max, asc - i64::from(os2.typo_descender))
    } else {
        let asc = i64::from(face.ascender);
        (asc - y_max, asc - i64::from(face.descender))
    }
}

/// One `FT_Load_Glyph`: the face, the size's hinting state, and the CVT
/// and storage its glyph programs work on.
struct Loader<'a> {
    face: &'a Face,
    hinter: &'a mut Hinter,
    cvt: Vec<i64>,
    storage: Vec<i64>,
}

impl Loader<'_> {
    /// `TT_Hint_Glyph`: the program run on a zone of scaled points (the
    /// phantom points last, rounded); the hinted phantom points kept
    /// unless backward compatibility leaves the metrics alone.
    fn hint(&mut self, zone: &mut Zone, code: &[u8], is_composite: bool, pp: &mut [(i64, i64); 4]) {
        let n = zone.len();

        if !code.is_empty() {
            zone.org = zone.cur.clone();
        }

        let (xs, ys) = if is_composite {
            zone.orus = zone.cur.clone();
            (1 << 16, 1 << 16)
        } else {
            (self.hinter.x_scale, self.hinter.y_scale)
        };

        zone.cur[n - 4].0 = pix_round(zone.cur[n - 4].0);
        zone.cur[n - 3].0 = pix_round(zone.cur[n - 3].0);
        zone.cur[n - 2].1 = pix_round(zone.cur[n - 2].1);
        zone.cur[n - 1].1 = pix_round(zone.cur[n - 1].1);

        let gs = self.hinter.gs;
        let mut backward = gs.instruct_control & 4 == 0;

        if !code.is_empty() {
            let cvt = std::mem::take(&mut self.cvt);
            let storage = std::mem::take(&mut self.storage);
            let pts = std::mem::take(zone);
            let mut exec = Exec::new(self.hinter, cvt, storage, pts, gs, true, is_composite, xs, ys);
            exec.ini_range = Range::Glyph;
            exec.range = Range::Glyph;
            // TT_Run_Context
            exec.zp = [true; 3];
            exec.gs.gep0 = 1;
            exec.gs.gep1 = 1;
            exec.gs.gep2 = 1;
            exec.gs.proj = (0x4000, 0);
            exec.gs.free = (0x4000, 0);
            exec.gs.dual = (0x4000, 0);
            exec.gs.round_state = RoundState::Grid;
            exec.gs.loop_ = 1;
            exec.compute_funcs();
            let _ = exec.run_code(code);
            backward = exec.backward_compatibility;
            *zone = std::mem::take(&mut exec.pts);
            self.cvt = std::mem::take(&mut exec.cvt);
            self.storage = std::mem::take(&mut exec.storage);
        }

        if !backward {
            pp.copy_from_slice(&zone.cur[n - 4..]);
        }
    }

    /// `load_truetype_glyph` with hinting.
    fn load(&mut self, gid: u16, depth: u32) -> Result<Loaded, Error> {
        if depth > 16 {
            return Err(Error::Unsupported("composite glyphs nested too deep"));
        }

        let face = self.face;
        let (xs, ys) = (self.hinter.x_scale, self.hinter.y_scale);
        let (advance, lsb) = face.h_metrics(gid);
        let raw = glyph::parse(face, gid)?;

        let bbox = match &raw {
            RawGlyph::Empty => [0i16; 4],
            RawGlyph::Simple { bbox, .. } | RawGlyph::Composite { bbox, .. } => *bbox,
        };

        let (tsb, ah) = v_metrics(face, i64::from(bbox[3]));
        let pp1x = i64::from(bbox[0]) - i64::from(lsb);
        let top = tsb + i64::from(bbox[3]);
        let pp_units = [(pp1x, 0), (pp1x + i64::from(advance), 0), (0, top), (0, top - ah)];
        let scale = |p: (i64, i64)| (mul_fix(p.0, xs), mul_fix(p.1, ys));
        let mut pp = pp_units.map(scale);

        match raw {
            RawGlyph::Empty => Ok(Loaded { outline: Outline::default(), pp }),
            RawGlyph::Simple { points, flags, contours, instructions, .. } => {
                let n = points.len();
                let mut zone = Zone::with_points(n + 4);

                for (i, p) in points.iter().enumerate() {
                    zone.orus[i] = (i64::from(p.0), i64::from(p.1));
                    zone.cur[i] = scale(zone.orus[i]);
                    zone.tags[i] = flags[i] & TAG_ON;
                }

                for (i, p) in pp_units.iter().enumerate() {
                    zone.orus[n + i] = *p;
                    zone.cur[n + i] = pp[i];
                }

                zone.org = zone.cur.clone();
                zone.contours = contours.iter().map(|&c| usize::from(c)).collect();

                let code = if self.hinter.ready { instructions } else { &[] };
                self.hint(&mut zone, code, false, &mut pp);

                let outline = Outline {
                    points: zone.cur[..n].to_vec(),
                    tags: zone.tags[..n].iter().map(|t| t & TAG_ON).collect(),
                    contours,
                };

                Ok(Loaded { outline, pp })
            }
            RawGlyph::Composite { components, instructions, .. } => {
                let mut outline = Outline::default();

                for c in components {
                    self.add_component(&mut outline, &mut pp, c, depth)?;
                }

                if !instructions.is_empty() && self.hinter.ready {
                    let n = outline.points.len();
                    let mut zone = Zone::with_points(n + 4);
                    zone.cur[..n].copy_from_slice(&outline.points);
                    zone.cur[n..].copy_from_slice(&pp);
                    zone.cur[n].0 = pix_round(zone.cur[n].0);
                    zone.cur[n + 1].0 = pix_round(zone.cur[n + 1].0);
                    zone.cur[n + 2].1 = pix_round(zone.cur[n + 2].1);
                    zone.cur[n + 3].1 = pix_round(zone.cur[n + 3].1);

                    for (i, t) in outline.tags.iter().enumerate() {
                        zone.tags[i] = t & TAG_ON;
                    }

                    zone.org = zone.cur.clone();
                    zone.contours = outline.contours.iter().map(|&c| usize::from(c)).collect();
                    self.hint(&mut zone, instructions, true, &mut pp);
                    outline.points.copy_from_slice(&zone.cur[..n]);

                    for (t, z) in outline.tags.iter_mut().zip(&zone.tags) {
                        *t = z & TAG_ON;
                    }
                }

                Ok(Loaded { outline, pp })
            }
        }
    }

    /// A component placed in a composite (`TT_Process_Composite_Component`).
    fn add_component(&mut self, outline: &mut Outline, pp: &mut [(i64, i64); 4], c: Component, depth: u32) -> Result<(), Error> {
        let (xs, ys) = (self.hinter.x_scale, self.hinter.y_scale);
        let mut sub = self.load(c.gid, depth + 1)?;

        if c.flags & glyph::USE_MY_METRICS != 0 {
            *pp = sub.pp;
        }

        if let Some(m) = c.transform {
            for p in &mut sub.outline.points {
                *p = glyph::transform(*p, m);
            }
        }

        let (dx, dy) = if c.flags & glyph::ARGS_ARE_XY_VALUES != 0 {
            let (mut x, mut y) = (i64::from(c.arg1), i64::from(c.arg2));

            if let Some(m) = c.transform
                && c.flags & glyph::SCALED_COMPONENT_OFFSET != 0 && c.flags & glyph::UNSCALED_COMPONENT_OFFSET == 0 {
                    x = mul_fix(x, glyph::hypot(m[0], m[2]));
                    y = mul_fix(y, glyph::hypot(m[1], m[3]));
                }

            x = mul_fix(x, xs);
            y = mul_fix(y, ys);

            // subpixel hinting leaves x where it is
            if c.flags & glyph::ROUND_XY_TO_GRID != 0 {
                y = pix_round(y);
            }

            (x, y)
        } else {
            let (k, l) = (c.arg1 as usize, c.arg2 as usize);
            let base = outline.points.get(k).ok_or(Error::Unsupported("a component anchor past the points"))?;
            let own = sub.outline.points.get(l).ok_or(Error::Unsupported("a component anchor past the points"))?;
            (base.0 - own.0, base.1 - own.1)
        };

        let offset = outline.points.len() as u16;
        outline.contours.extend(sub.outline.contours.iter().map(|e| e + offset));
        outline.points.extend(sub.outline.points.iter().map(|p| (p.0 + dx, p.1 + dy)));
        outline.tags.extend(sub.outline.tags);
        Ok(())
    }
}

/// A glyph loaded with hinting, as `FT_Load_Glyph` with the default load
/// flags gives it.
pub(crate) fn load_hinted(face: &Face, size: &Size, hinter: &mut Hinter, gid: u16) -> Result<Glyph, Error> {
    // the control value program can turn hinting off for the size
    if hinter.gs.instruct_control & 1 != 0 {
        return Glyph::load(face, size, gid);
    }

    if hinter.gs.instruct_control & 2 != 0 {
        hinter.gs = GraphicsState::default();
    }

    let (cvt, storage) = (hinter.cvt.clone(), hinter.storage.clone());
    let mut loader = Loader { face, hinter, cvt, storage };
    let loaded = loader.load(gid, 0)?;
    let mut outline = loaded.outline;
    let pp1 = loaded.pp[0].0;

    if pp1 != 0 {
        for p in &mut outline.points {
            p.0 -= pp1;
        }
    }

    let advance = glyph::hinted_advance(loaded.pp[0].0, loaded.pp[1].0);
    let left_bearing = pix_floor(outline.points.iter().map(|p| p.0).min().unwrap_or(0));
    Ok(Glyph { outline, advance, left_bearing })
}
