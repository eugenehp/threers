//! Bytes shared between readers without copying: a buffer in memory, or a
//! file mapped into it.
//!
//! A 4K or 8K film runs to gigabytes. Reading one whole into a `Vec` before
//! the first picture means waiting for all of it and holding all of it.
//! Mapped, the operating system pages in what the demuxer and decoder touch,
//! as they touch it, and drops it again under memory pressure; the first
//! picture is ready after reading the index and one sample. Everything here
//! that takes a whole file ([`VideoReader`](crate::decode::VideoReader),
//! [`demux`](crate::decode::demux), the audio decoders) takes a slice, so a
//! mapping slots in anywhere a `Vec` did.
//!
//! ```no_run
//! use threers_codec::bytes::Bytes;
//!
//! let file = Bytes::map("film.mp4").unwrap();
//! println!("{} bytes, mapped: {}", file.len(), file.is_mapped());
//! ```

use std::ops::Deref;
use std::path::Path;
use std::sync::Arc;

/// Shared, immutable bytes: cheap to clone, from memory or a mapped file.
#[derive(Clone)]
pub struct Bytes(Arc<dyn AsRef<[u8]> + Send + Sync>, bool);

impl std::fmt::Debug for Bytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bytes")
            .field("len", &self.len())
            .field("mapped", &self.1)
            .finish()
    }
}

impl Bytes {
    /// Map the file at `path` read-only, where the platform can (Unix);
    /// elsewhere it is read into memory. The file must not be changed while
    /// mapped: like every program that maps files, this reads what is there.
    pub fn map(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        #[cfg(unix)]
        {
            if let Some(map) = mmap::Map::new(&file)? {
                return Ok(Self(Arc::new(map), true));
            }
        }
        use std::io::Read;
        let mut data = Vec::new();
        (&file).read_to_end(&mut data)?;
        Ok(Self::from(data))
    }

    /// Whether the bytes are a mapping of a file rather than a buffer.
    pub fn is_mapped(&self) -> bool {
        self.1
    }
}

impl Deref for Bytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        (*self.0).as_ref()
    }
}

impl AsRef<[u8]> for Bytes {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

impl From<Vec<u8>> for Bytes {
    fn from(v: Vec<u8>) -> Self {
        Self(Arc::new(v), false)
    }
}

impl From<Arc<Vec<u8>>> for Bytes {
    fn from(v: Arc<Vec<u8>>) -> Self {
        Self(Arc::new(VecRef(v)), false)
    }
}

struct VecRef(Arc<Vec<u8>>);

impl AsRef<[u8]> for VecRef {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

/// Picture buffers handed back to be filled again: the big byte buffers a
/// hardware decoder copies each picture into.
///
/// A fresh 50 MB buffer for every 8K picture costs more than the copy into
/// it: the system hands over new pages that are faulted in and zeroed on
/// first touch (measured on Linux, about 10 ms a picture against 2 ms for
/// the copy). A decoder that takes its buffers from here, and a player that
/// gives each picture's buffers back once it is off the screen, reuse a few
/// buffers for the whole film. Cheap to clone; every clone is the same pool.
/// It keeps at most a handful of buffers; the rest are freed.
#[derive(Clone, Default, Debug)]
pub struct FramePool(Arc<std::sync::Mutex<Vec<Vec<u8>>>>);

impl FramePool {
    /// How many buffers it keeps: a luma and a chroma buffer for each of the
    /// pictures a player holds at once (decoded ahead, due, on screen).
    const KEEP: usize = 12;

    /// An empty buffer with room for `len` bytes: the smallest given back
    /// that is big enough, or a new one.
    pub fn take(&self, len: usize) -> Vec<u8> {
        let mut held = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let fit = held
            .iter()
            .enumerate()
            .filter(|(_, b)| b.capacity() >= len)
            .min_by_key(|(_, b)| b.capacity())
            .map(|(i, _)| i);
        match fit {
            Some(i) => {
                let mut buf = held.swap_remove(i);
                buf.clear();
                buf
            }
            None => Vec::with_capacity(len),
        }
    }

    /// Give `buf` back, to be taken again.
    pub fn give(&self, buf: Vec<u8>) {
        if buf.capacity() == 0 {
            return;
        }
        let mut held = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if held.len() < Self::KEEP {
            held.push(buf);
        }
    }

    /// Buffers waiting to be taken.
    pub fn len(&self) -> usize {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Whether none are.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(unix)]
mod mmap {
    use std::ffi::c_void;
    use std::os::fd::AsRawFd;

    const PROT_READ: i32 = 1;
    const MAP_PRIVATE: i32 = 2;
    /// `MADV_SEQUENTIAL`, 2 on Linux and the BSDs (macOS included).
    const MADV_SEQUENTIAL: i32 = 2;

    unsafe extern "C" {
        fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut c_void;
        fn munmap(addr: *mut c_void, len: usize) -> i32;
        fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
    }

    /// A read-only private mapping of a whole file.
    pub(super) struct Map {
        ptr: *mut c_void,
        len: usize,
    }

    // Read-only memory nobody writes through.
    unsafe impl Send for Map {}
    unsafe impl Sync for Map {}

    impl Map {
        /// `None` for an empty file, which has nothing to map.
        pub(super) fn new(file: &std::fs::File) -> std::io::Result<Option<Self>> {
            let len = file.metadata()?.len();
            if len == 0 {
                return Ok(None);
            }
            let len = usize::try_from(len).map_err(|_| std::io::Error::other("file too large to map"))?;
            let ptr = unsafe { mmap(std::ptr::null_mut(), len, PROT_READ, MAP_PRIVATE, file.as_raw_fd(), 0) };
            if ptr as isize == -1 {
                return Err(std::io::Error::last_os_error());
            }
            // Demuxing and decoding read a video front to back: ask for
            // read-ahead. Advice only; a refusal changes nothing.
            unsafe { madvise(ptr, len, MADV_SEQUENTIAL) };
            Ok(Some(Self { ptr, len }))
        }
    }

    impl AsRef<[u8]> for Map {
        fn as_ref(&self) -> &[u8] {
            unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
        }
    }

    impl Drop for Map {
        fn drop(&mut self) {
            unsafe { munmap(self.ptr, self.len) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_mapped_file_reads_as_its_bytes() {
        let path = std::env::temp_dir().join(format!("threers-bytes-{}.bin", std::process::id()));
        let data: Vec<u8> = (0..100_000u32).map(|i| (i * 7 % 251) as u8).collect();
        std::fs::write(&path, &data).unwrap();
        let mapped = Bytes::map(&path).unwrap();
        assert_eq!(mapped.is_mapped(), cfg!(unix));
        assert_eq!(&mapped[..], &data[..]);
        let shared = mapped.clone();
        drop(mapped);
        assert_eq!(shared[99_999], data[99_999], "a clone keeps the mapping alive");
        drop(shared);
        std::fs::write(&path, b"").unwrap();
        assert!(Bytes::map(&path).unwrap().is_empty(), "an empty file maps to nothing");
        std::fs::remove_file(&path).unwrap();
        assert!(Bytes::from(vec![1, 2, 3]).iter().eq([1u8, 2, 3].iter()));
    }

    #[test]
    fn a_frame_pool_hands_back_the_closest_fit() {
        let pool = FramePool::default();
        let fresh = pool.take(100);
        assert!(fresh.capacity() >= 100 && fresh.is_empty());
        let (mut big, small) = (Vec::with_capacity(1000), Vec::<u8>::with_capacity(300));
        big.extend_from_slice(&[7; 10]);
        let (big_at, small_at) = (big.as_ptr(), small.as_ptr());
        pool.give(big);
        pool.give(small);
        pool.give(Vec::new());
        assert_eq!(pool.len(), 2, "an empty buffer is not kept");
        let got = pool.take(200);
        assert_eq!(got.as_ptr(), small_at, "the smallest that fits");
        let got = pool.take(500);
        assert_eq!(got.as_ptr(), big_at);
        assert!(got.is_empty(), "handed back empty");
        assert!(pool.is_empty());
        for _ in 0..20 {
            pool.give(vec![0; 4]);
        }
        assert_eq!(pool.len(), FramePool::KEEP, "the rest are freed");
    }
}
