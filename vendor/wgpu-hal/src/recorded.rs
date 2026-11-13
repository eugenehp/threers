//! Metal shaders recorded once and replayed without naga.
//!
//! Not part of wgpu: a local change carried in threers' `vendor/` copy.
//!
//! A build with `msl-record` runs as usual, and when `WGPU_MSL_RECORD` names
//! a file it appends to that file every shader module's entry points and the
//! MSL that naga writes for every pipeline stage, with what the Metal
//! backend needs alongside it (work group sizes, the bindings whose sizes a
//! shader reads). A build with `msl-replay` never parses or translates a
//! shader: wgpu-core hands the backend the hash of a WGSL source, and the
//! backend looks the stage up in a recording [`install`]ed at startup. A
//! stage that was not recorded fails to build, as a shader with an error
//! would.
//!
//! A stage's key hashes everything naga's MSL depends on: the WGSL source,
//! the entry point, the vertex buffer layouts (Metal pulls vertices in the
//! shader), the pipeline layout's binding slots, the primitive class, the
//! pipeline constants and the runtime checks. The MSL is recorded for
//! language version 3.1 (`WGPU_MSL_RECORD_VERSION=major.minor` changes it),
//! the newest that iOS 17 compiles, so one recording serves every device.

use alloc::{string::String, vec::Vec};

/// 64-bit FNV-1a: the same on every machine and every run.
pub fn hash(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// A shader module's entry point.
#[derive(Clone, Debug)]
pub struct EntryPoint {
    pub name: String,
    /// 0 vertex, 1 fragment, 2 compute, 3 task, 4 mesh.
    pub stage: u8,
    pub workgroup_size: [u32; 3],
}

/// A pipeline stage, as the Metal backend builds it.
#[derive(Clone, Debug)]
pub struct Stage {
    /// The entry point's name in the MSL.
    pub function: String,
    pub workgroup_size: [u32; 3],
    pub workgroup_memory_sizes: Vec<u32>,
    /// (group, binding, array index) of each buffer whose size the shader
    /// reads.
    pub sized_bindings: Vec<(u32, u32, u32)>,
    pub immutable_buffer_mask: u64,
    pub msl: String,
}

const MODULE: u8 = b'M';
const STAGE: u8 = b'S';

#[cfg(feature = "msl-record")]
mod record {
    use super::*;
    use std::io::Write as _;

    fn put_u32(out: &mut Vec<u8>, v: u32) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    fn put_str(out: &mut Vec<u8>, s: &str) {
        put_u32(out, s.len() as u32);
        out.extend_from_slice(s.as_bytes());
    }

    fn append(bytes: &[u8]) {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let Some(path) = std::env::var_os("WGPU_MSL_RECORD") else {
            return;
        };
        let _held = LOCK.lock();
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path);
        match file.and_then(|mut f| f.write_all(bytes)) {
            Ok(()) => {}
            Err(e) => log::error!("WGPU_MSL_RECORD: {e}"),
        }
    }

    /// The MSL language version recordings are written for.
    pub fn version() -> (u8, u8) {
        std::env::var("WGPU_MSL_RECORD_VERSION")
            .ok()
            .and_then(|v| {
                let (a, b) = v.split_once('.')?;
                Some((a.parse().ok()?, b.parse().ok()?))
            })
            .unwrap_or((3, 1))
    }

    pub fn module(source_hash: u64, entry_points: &[EntryPoint]) {
        let mut out = vec![MODULE];
        out.extend_from_slice(&source_hash.to_le_bytes());
        put_u32(&mut out, entry_points.len() as u32);
        for ep in entry_points {
            put_str(&mut out, &ep.name);
            out.push(ep.stage);
            for v in ep.workgroup_size {
                put_u32(&mut out, v);
            }
        }
        append(&out);
    }

    pub fn stage(key: u64, stage: &Stage) {
        let mut out = vec![STAGE];
        out.extend_from_slice(&key.to_le_bytes());
        put_str(&mut out, &stage.function);
        for v in stage.workgroup_size {
            put_u32(&mut out, v);
        }
        put_u32(&mut out, stage.workgroup_memory_sizes.len() as u32);
        for &v in &stage.workgroup_memory_sizes {
            put_u32(&mut out, v);
        }
        put_u32(&mut out, stage.sized_bindings.len() as u32);
        for &(g, b, i) in &stage.sized_bindings {
            put_u32(&mut out, g);
            put_u32(&mut out, b);
            put_u32(&mut out, i);
        }
        out.extend_from_slice(&stage.immutable_buffer_mask.to_le_bytes());
        put_str(&mut out, &stage.msl);
        append(&out);
    }
}
#[cfg(feature = "msl-record")]
pub use record::{module as record_module, stage as record_stage, version as record_version};

#[cfg(feature = "msl-replay")]
mod replay {
    use super::*;
    use std::{collections::HashMap, sync::OnceLock};

    struct Tables {
        modules: HashMap<u64, Vec<EntryPoint>>,
        stages: HashMap<u64, Stage>,
    }
    static TABLES: OnceLock<Tables> = OnceLock::new();

    struct Reader<'a>(&'a [u8]);
    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            if self.0.len() < n {
                return None;
            }
            let (a, b) = self.0.split_at(n);
            self.0 = b;
            Some(a)
        }
        fn u8(&mut self) -> Option<u8> {
            Some(self.take(1)?[0])
        }
        fn u32(&mut self) -> Option<u32> {
            Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
        }
        fn u64(&mut self) -> Option<u64> {
            Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
        }
        fn string(&mut self) -> Option<String> {
            let n = self.u32()? as usize;
            String::from_utf8(self.take(n)?.to_vec()).ok()
        }
        fn u32s<const N: usize>(&mut self) -> Option<[u32; N]> {
            let mut a = [0; N];
            for v in &mut a {
                *v = self.u32()?;
            }
            Some(a)
        }
    }

    fn parse(bytes: &[u8]) -> Option<Tables> {
        let mut r = Reader(bytes);
        let mut t = Tables {
            modules: HashMap::new(),
            stages: HashMap::new(),
        };
        while !r.0.is_empty() {
            match r.u8()? {
                MODULE => {
                    let hash = r.u64()?;
                    let n = r.u32()?;
                    let mut eps = Vec::new();
                    for _ in 0..n {
                        eps.push(EntryPoint {
                            name: r.string()?,
                            stage: r.u8()?,
                            workgroup_size: r.u32s()?,
                        });
                    }
                    t.modules.insert(hash, eps);
                }
                STAGE => {
                    let key = r.u64()?;
                    let function = r.string()?;
                    let workgroup_size = r.u32s()?;
                    let n = r.u32()?;
                    let workgroup_memory_sizes = (0..n).map(|_| r.u32()).collect::<Option<_>>()?;
                    let n = r.u32()?;
                    let sized_bindings = (0..n)
                        .map(|_| Some((r.u32()?, r.u32()?, r.u32()?)))
                        .collect::<Option<_>>()?;
                    let immutable_buffer_mask = r.u64()?;
                    let msl = r.string()?;
                    t.stages.insert(
                        key,
                        Stage {
                            function,
                            workgroup_size,
                            workgroup_memory_sizes,
                            sized_bindings,
                            immutable_buffer_mask,
                            msl,
                        },
                    );
                }
                _ => return None,
            }
        }
        Some(t)
    }

    /// Takes a recording (the `WGPU_MSL_RECORD` file's bytes) for every
    /// device made afterwards. Returns the number of pipeline stages in it.
    /// Once per process: a second call is an error.
    pub fn install(bytes: &[u8]) -> Result<usize, &'static str> {
        let tables = parse(bytes).ok_or("not a WGPU_MSL_RECORD recording")?;
        let n = tables.stages.len();
        TABLES.set(tables).map_err(|_| "a recording is already installed")?;
        Ok(n)
    }

    pub fn module(source_hash: u64) -> Option<&'static [EntryPoint]> {
        TABLES.get()?.modules.get(&source_hash).map(Vec::as_slice)
    }

    pub fn stage(key: u64) -> Option<&'static Stage> {
        TABLES.get()?.stages.get(&key)
    }
}
#[cfg(feature = "msl-replay")]
pub use replay::{install, module, stage};
