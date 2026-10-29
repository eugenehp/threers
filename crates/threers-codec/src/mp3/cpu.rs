//! The machine the generated routines in [`super::built`] run on: AArch64
//! general registers, 128-bit vector registers kept as bit patterns, the Z
//! flag, and one byte-addressed memory arena (callers place their buffers in
//! it; the reference binary's constant pool sits at [`RODATA_BASE`]; the
//! stack grows down from the top).

pub const RODATA_BASE: u64 = 0x10_0000;
const ARENA: usize = 0x11_0000;

pub struct Cpu {
    pub x: [u64; 32],
    pub v: [[u32; 4]; 32],
    pub z: bool,
    pub mem: Vec<u8>,
}

impl Default for Cpu {
    fn default() -> Self {
        let mut mem = vec![0u8; ARENA];
        let ro = &super::built::RODATA;
        mem[RODATA_BASE as usize..RODATA_BASE as usize + ro.len()].copy_from_slice(ro);
        let mut x = [0u64; 32];
        x[31] = RODATA_BASE - 0x100; // sp, below the constant pool
        Cpu { x, v: [[0; 4]; 32], z: false, mem }
    }
}

impl Cpu {
    #[inline(always)]
    pub fn ld32(&self, a: u64) -> u32 {
        let a = a as usize;
        u32::from_le_bytes(self.mem[a..a + 4].try_into().unwrap())
    }
    #[inline(always)]
    pub fn st32(&mut self, a: u64, v: u32) {
        let a = a as usize;
        self.mem[a..a + 4].copy_from_slice(&v.to_le_bytes());
    }
    #[inline(always)]
    pub fn ld64(&self, a: u64) -> u64 {
        let a = a as usize;
        u64::from_le_bytes(self.mem[a..a + 8].try_into().unwrap())
    }
    #[inline(always)]
    pub fn st64(&mut self, a: u64, v: u64) {
        let a = a as usize;
        self.mem[a..a + 8].copy_from_slice(&v.to_le_bytes());
    }
    /// Scalar `s` register (lane 0).
    #[inline(always)]
    pub fn s(&self, r: usize) -> f32 {
        f32::from_bits(self.v[r][0])
    }
    /// Write an `s` register: the upper lanes are zeroed, as on AArch64.
    #[inline(always)]
    pub fn ss(&mut self, r: usize, x: f32) {
        self.v[r] = [x.to_bits(), 0, 0, 0];
    }
    #[inline(always)]
    pub fn vf(&self, r: usize) -> [f32; 4] {
        self.v[r].map(f32::from_bits)
    }
    #[inline(always)]
    pub fn setvf(&mut self, r: usize, x: [f32; 4]) {
        self.v[r] = x.map(f32::to_bits);
    }
    /// A 4-lane float op `d = f(n, m, d)`.
    #[inline(always)]
    pub fn vop4(&mut self, d: usize, n: usize, m: usize, f: impl Fn(f32, f32, f32) -> f32) {
        let (a, b, c) = (self.vf(n), self.vf(m), self.vf(d));
        self.setvf(d, [f(a[0], b[0], c[0]), f(a[1], b[1], c[1]), f(a[2], b[2], c[2]), f(a[3], b[3], c[3])]);
    }
    /// A 2-lane float op; the upper half of the destination is zeroed.
    #[inline(always)]
    pub fn vop2(&mut self, d: usize, n: usize, m: usize, f: impl Fn(f32, f32, f32) -> f32) {
        let (a, b, c) = (self.vf(n), self.vf(m), self.vf(d));
        self.setvf(d, [f(a[0], b[0], c[0]), f(a[1], b[1], c[1]), 0.0, 0.0]);
    }
    /// Copy floats into the arena at `addr`.
    pub fn put(&mut self, addr: u64, xs: &[f32]) {
        for (i, x) in xs.iter().enumerate() {
            self.st32(addr + 4 * i as u64, x.to_bits());
        }
    }
    pub fn get(&self, addr: u64, n: usize) -> Vec<f32> {
        (0..n).map(|i| f32::from_bits(self.ld32(addr + 4 * i as u64))).collect()
    }
}
