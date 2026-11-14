//! Hardware H.264 encoding through a V4L2 memory-to-memory encoder (feature `v4l2`, Linux):
//! the Raspberry Pi's `bcm2835-codec` (`/dev/video11` on a Pi 4), or any stateful V4L2 encoder
//! with a multi-planar YU12 → H264 interface. I420 in, H.264 Annex B out, no CPU spent encoding
//! (a 1280x720 frame in ~11 ms of wall time on a Pi 4, a 320x180 one in ~2.4 ms).
//!
//! Multi-planar M2M through raw ioctls (the `v4l` crate covers capture devices): an OUTPUT
//! queue the frames go into and a CAPTURE queue the bitstream comes out of, both memory-mapped.
//! The fd is non-blocking and [`HwH264::encode`] returns whatever packets are ready, so the
//! encoder's pipeline delay never blocks the caller.
//!
//! Layouts are the kernel's (`linux/videodev2.h`) on 64-bit Linux; sizes are checked in tests.

#![allow(unsafe_code)] // ioctl and mmap: the kernel interface itself.

use std::io;
use std::os::fd::{AsRawFd, OwnedFd};

const BUF_TYPE_CAPTURE_MPLANE: u32 = 9;
const BUF_TYPE_OUTPUT_MPLANE: u32 = 10;
const MEMORY_MMAP: u32 = 1;
const FIELD_NONE: u32 = 1;
const FMT_YU12: u32 = u32::from_le_bytes(*b"YU12");
const FMT_H264: u32 = u32::from_le_bytes(*b"H264");

const CID_BITRATE: u32 = 0x0099_09cf;
const CID_GOP_SIZE: u32 = 0x0099_09cb;
const CID_REPEAT_SEQ_HEADER: u32 = 0x0099_09e2;
const CID_H264_I_PERIOD: u32 = 0x0099_0a66;
const CID_H264_PROFILE: u32 = 0x0099_0a6b;
/// `V4L2_MPEG_VIDEO_H264_PROFILE_CONSTRAINED_BASELINE`: decodable everywhere, no B-frames.
const PROFILE_CONSTRAINED_BASELINE: i32 = 1;

#[repr(C)]
#[derive(Clone, Copy)]
struct PlanePixFormat {
    sizeimage: u32,
    bytesperline: u32,
    reserved: [u16; 6],
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct PixFormatMplane {
    width: u32,
    height: u32,
    pixelformat: u32,
    field: u32,
    colorspace: u32,
    plane_fmt: [PlanePixFormat; 8],
    num_planes: u8,
    flags: u8,
    ycbcr_enc: u8,
    quantization: u8,
    xfer_func: u8,
    reserved: [u8; 7],
}

/// `struct v4l2_format`: the union is 8-byte aligned (it holds pointers elsewhere).
#[repr(C)]
struct Format {
    kind: u32,
    _pad: u32,
    pix_mp: PixFormatMplane,
    _rest: [u8; 200 - std::mem::size_of::<PixFormatMplane>()],
}

#[repr(C)]
struct RequestBuffers {
    count: u32,
    kind: u32,
    memory: u32,
    capabilities: u32,
    flags: u8,
    reserved: [u8; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Plane {
    bytesused: u32,
    length: u32,
    mem_offset: u64,
    data_offset: u32,
    reserved: [u32; 11],
}

#[repr(C)]
struct Buffer {
    index: u32,
    kind: u32,
    bytesused: u32,
    flags: u32,
    field: u32,
    timestamp: [i64; 2],
    timecode: [u32; 4],
    sequence: u32,
    memory: u32,
    planes: *mut Plane,
    length: u32,
    reserved2: u32,
    request_fd: i32,
}

#[repr(C)]
struct Control {
    id: u32,
    value: i32,
}

const fn iowr(nr: u32, size: usize) -> libc::c_ulong {
    ((3u32 << 30) | ((size as u32) << 16) | ((b'V' as u32) << 8) | nr) as libc::c_ulong
}
const fn iow(nr: u32, size: usize) -> libc::c_ulong {
    ((1u32 << 30) | ((size as u32) << 16) | ((b'V' as u32) << 8) | nr) as libc::c_ulong
}
const VIDIOC_S_FMT: libc::c_ulong = iowr(5, std::mem::size_of::<Format>());
const VIDIOC_REQBUFS: libc::c_ulong = iowr(8, std::mem::size_of::<RequestBuffers>());
const VIDIOC_QUERYBUF: libc::c_ulong = iowr(9, std::mem::size_of::<Buffer>());
const VIDIOC_QBUF: libc::c_ulong = iowr(15, std::mem::size_of::<Buffer>());
const VIDIOC_DQBUF: libc::c_ulong = iowr(17, std::mem::size_of::<Buffer>());
const VIDIOC_STREAMON: libc::c_ulong = iow(18, std::mem::size_of::<i32>());
const VIDIOC_STREAMOFF: libc::c_ulong = iow(19, std::mem::size_of::<i32>());
const VIDIOC_S_CTRL: libc::c_ulong = iowr(28, std::mem::size_of::<Control>());

fn ioctl<T>(fd: &OwnedFd, req: libc::c_ulong, arg: &mut T) -> io::Result<()> {
    loop {
        // SAFETY: `arg` is the struct `req` is defined for (sizes in the request number).
        let r = unsafe { libc::ioctl(fd.as_raw_fd(), req as _, arg as *mut T) };
        if r >= 0 {
            return Ok(());
        }
        let e = io::Error::last_os_error();
        if e.kind() != io::ErrorKind::Interrupted {
            return Err(e);
        }
    }
}

/// One memory-mapped buffer.
struct Mapped {
    ptr: *mut u8,
    len: usize,
}

impl Drop for Mapped {
    fn drop(&mut self) {
        // SAFETY: mapped by `map_buffers` with this length.
        unsafe { libc::munmap(self.ptr.cast(), self.len) };
    }
}

// The mappings are only touched through `&mut V4l2H264Encoder`.
unsafe impl Send for Mapped {}

/// The encoder, open and streaming.
pub struct V4l2H264Encoder {
    fd: OwnedFd,
    width: usize,
    height: usize,
    /// Y stride and the padded height the driver wants in the OUTPUT buffer.
    stride: usize,
    plane_height: usize,
    inputs: Vec<Mapped>,
    free_inputs: Vec<u32>,
    outputs: Vec<Mapped>,
    frames_in: u64,
    /// Frames dropped because every input buffer was still with the encoder.
    pub dropped: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct H264Options {
    pub bitrate: u32,
    /// Frames between key frames (each repeats SPS/PPS, so a stream can be joined there).
    pub key_interval: u32,
}

impl Default for H264Options {
    fn default() -> Self {
        Self { bitrate: 500_000, key_interval: 30 }
    }
}

/// The Raspberry Pi 4's encoder.
pub const DEVICE: &str = "/dev/video11";

impl V4l2H264Encoder {
    /// Open [`DEVICE`] for `width` x `height` I420 (even sides; the driver may pad).
    pub fn open(width: usize, height: usize, opts: H264Options) -> io::Result<Self> {
        Self::open_device(DEVICE, width, height, opts)
    }

    /// Open the encoder at `device` (another board's M2M encoder node).
    pub fn open_device(device: &str, width: usize, height: usize, opts: H264Options) -> io::Result<Self> {
        use std::os::fd::FromRawFd;
        let path = std::ffi::CString::new(device).map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "NUL in device path"))?;
        // SAFETY: a plain open(2); the fd is owned below.
        let raw = unsafe { libc::open(path.as_ptr(), libc::O_RDWR | libc::O_NONBLOCK | libc::O_CLOEXEC) };
        if raw < 0 {
            return Err(io::Error::new(io::Error::last_os_error().kind(), format!("{device}: {}", io::Error::last_os_error())));
        }
        // SAFETY: `raw` is a fresh fd nothing else owns.
        let fd = unsafe { OwnedFd::from_raw_fd(raw) };

        let set_fmt = |kind: u32, pixelformat: u32, sizeimage: u32| -> io::Result<PixFormatMplane> {
            // SAFETY: all-zero is a valid `v4l2_format`.
            let mut f: Format = unsafe { std::mem::zeroed() };
            f.kind = kind;
            f.pix_mp.width = width as u32;
            f.pix_mp.height = height as u32;
            f.pix_mp.pixelformat = pixelformat;
            f.pix_mp.field = FIELD_NONE;
            f.pix_mp.num_planes = 1;
            f.pix_mp.plane_fmt[0].sizeimage = sizeimage;
            ioctl(&fd, VIDIOC_S_FMT, &mut f)?;
            Ok(f.pix_mp)
        };
        let input = set_fmt(BUF_TYPE_OUTPUT_MPLANE, FMT_YU12, 0)?;
        set_fmt(BUF_TYPE_CAPTURE_MPLANE, FMT_H264, 512 * 1024)?;
        let plane0 = input.plane_fmt[0];
        let stride = (plane0.bytesperline as usize).max(width);
        let plane_height = (plane0.sizeimage as usize * 2 / 3 / stride).max(height);

        for (id, value) in [
            (CID_BITRATE, opts.bitrate as i32),
            (CID_GOP_SIZE, opts.key_interval as i32),
            (CID_H264_I_PERIOD, opts.key_interval as i32),
            (CID_REPEAT_SEQ_HEADER, 1),
            (CID_H264_PROFILE, PROFILE_CONSTRAINED_BASELINE),
        ] {
            let mut c = Control { id, value };
            // A driver without a control keeps its default; the stream is still H.264.
            let _ = ioctl(&fd, VIDIOC_S_CTRL, &mut c);
        }
        let inputs = map_buffers(&fd, BUF_TYPE_OUTPUT_MPLANE, 2)?;
        let outputs = map_buffers(&fd, BUF_TYPE_CAPTURE_MPLANE, 4)?;
        let enc = Self { fd, width, height, stride, plane_height, free_inputs: (0..inputs.len() as u32).collect(), inputs, outputs, frames_in: 0, dropped: 0 };
        for i in 0..enc.outputs.len() as u32 {
            enc.queue(BUF_TYPE_CAPTURE_MPLANE, i, 0)?;
        }
        for kind in [BUF_TYPE_OUTPUT_MPLANE, BUF_TYPE_CAPTURE_MPLANE] {
            let mut k = kind as i32;
            ioctl(&enc.fd, VIDIOC_STREAMON, &mut k)?;
        }
        Ok(enc)
    }

    fn queue(&self, kind: u32, index: u32, bytesused: u32) -> io::Result<()> {
        // SAFETY: zeroed POD.
        let mut plane: Plane = unsafe { std::mem::zeroed() };
        plane.bytesused = bytesused;
        let mut b = new_buffer(kind, index, &mut plane);
        ioctl(&self.fd, VIDIOC_QBUF, &mut b)
    }

    /// A finished buffer of `kind`: (index, bytes used), or `None` if none is ready.
    fn dequeue(&self, kind: u32) -> io::Result<Option<(u32, usize)>> {
        // SAFETY: zeroed POD.
        let mut plane: Plane = unsafe { std::mem::zeroed() };
        let mut b = new_buffer(kind, 0, &mut plane);
        match ioctl(&self.fd, VIDIOC_DQBUF, &mut b) {
            Ok(()) => Ok(Some((b.index, plane.bytesused as usize))),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Feed one I420 frame (`width * height * 3 / 2` bytes) and return the H.264 packets that
    /// are ready (Annex B; the first carries SPS/PPS). Waits up to `wait_ms` for the first.
    pub fn encode(&mut self, i420: &[u8], wait_ms: i32) -> io::Result<Vec<Vec<u8>>> {
        let (w, h) = (self.width, self.height);
        if i420.len() != w * h * 3 / 2 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("{} bytes for {w}x{h} I420", i420.len())));
        }
        // Reclaim input buffers the encoder has finished with.
        while let Some((i, _)) = self.dequeue(BUF_TYPE_OUTPUT_MPLANE)? {
            self.free_inputs.push(i);
        }
        if let Some(index) = self.free_inputs.pop() {
            let buf = &self.inputs[index as usize];
            // SAFETY: the mapping is `len` bytes and the driver does not own it (not queued).
            let dst = unsafe { std::slice::from_raw_parts_mut(buf.ptr, buf.len) };
            let (s, ph) = (self.stride, self.plane_height);
            let (y, uv) = i420.split_at(w * h);
            let (u, v) = uv.split_at(w * h / 4);
            for r in 0..h {
                dst[r * s..r * s + w].copy_from_slice(&y[r * w..(r + 1) * w]);
            }
            let (cs, cw, chh) = (s / 2, w / 2, h / 2);
            let (u0, v0) = (s * ph, s * ph + cs * (ph / 2));
            for r in 0..chh {
                dst[u0 + r * cs..u0 + r * cs + cw].copy_from_slice(&u[r * cw..(r + 1) * cw]);
                dst[v0 + r * cs..v0 + r * cs + cw].copy_from_slice(&v[r * cw..(r + 1) * cw]);
            }
            let used = (s * ph * 3 / 2).min(buf.len) as u32;
            self.queue(BUF_TYPE_OUTPUT_MPLANE, index, used)?;
            self.frames_in += 1;
        } else {
            self.dropped += 1;
        }
        self.collect(wait_ms)
    }

    /// Packets ready now, after waiting up to `wait_ms` for the first.
    pub fn collect(&mut self, wait_ms: i32) -> io::Result<Vec<Vec<u8>>> {
        let mut pfd = libc::pollfd { fd: self.fd.as_raw_fd(), events: libc::POLLIN, revents: 0 };
        // SAFETY: one valid pollfd.
        unsafe { libc::poll(&mut pfd, 1, wait_ms) };
        let mut packets = Vec::new();
        while let Some((i, used)) = self.dequeue(BUF_TYPE_CAPTURE_MPLANE)? {
            let buf = &self.outputs[i as usize];
            // SAFETY: the driver handed this buffer back; `used` <= its length.
            packets.push(unsafe { std::slice::from_raw_parts(buf.ptr, used.min(buf.len)) }.to_vec());
            self.queue(BUF_TYPE_CAPTURE_MPLANE, i, 0)?;
        }
        Ok(packets)
    }
}

impl Drop for V4l2H264Encoder {
    fn drop(&mut self) {
        for kind in [BUF_TYPE_OUTPUT_MPLANE, BUF_TYPE_CAPTURE_MPLANE] {
            let mut k = kind as i32;
            let _ = ioctl(&self.fd, VIDIOC_STREAMOFF, &mut k);
        }
    }
}

fn new_buffer(kind: u32, index: u32, plane: &mut Plane) -> Buffer {
    Buffer {
        index,
        kind,
        bytesused: 0,
        flags: 0,
        field: FIELD_NONE,
        timestamp: [0; 2],
        timecode: [0; 4],
        sequence: 0,
        memory: MEMORY_MMAP,
        planes: plane,
        length: 1,
        reserved2: 0,
        request_fd: 0,
    }
}

fn map_buffers(fd: &OwnedFd, kind: u32, count: u32) -> io::Result<Vec<Mapped>> {
    let mut req = RequestBuffers { count, kind, memory: MEMORY_MMAP, capabilities: 0, flags: 0, reserved: [0; 3] };
    ioctl(fd, VIDIOC_REQBUFS, &mut req)?;
    (0..req.count)
        .map(|i| {
            // SAFETY: zeroed POD.
            let mut plane: Plane = unsafe { std::mem::zeroed() };
            let mut b = new_buffer(kind, i, &mut plane);
            ioctl(fd, VIDIOC_QUERYBUF, &mut b)?;
            let len = plane.length as usize;
            // SAFETY: the offset and length the driver gave for this buffer.
            let ptr = unsafe {
                libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, fd.as_raw_fd(), plane.mem_offset as libc::off_t)
            };
            if ptr == libc::MAP_FAILED {
                return Err(io::Error::last_os_error());
            }
            Ok(Mapped { ptr: ptr.cast(), len })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_match_the_kernel() {
        assert_eq!(std::mem::size_of::<PixFormatMplane>(), 192);
        assert_eq!(std::mem::size_of::<Format>(), 208);
        assert_eq!(std::mem::size_of::<RequestBuffers>(), 20);
        assert_eq!(std::mem::size_of::<Plane>(), 64);
        assert_eq!(std::mem::size_of::<Buffer>(), 88);
        assert_eq!(std::mem::offset_of!(Buffer, timestamp), 24);
        assert_eq!(std::mem::offset_of!(Buffer, planes), 64);
        assert_eq!(VIDIOC_QBUF, 0xc058_560f);
        assert_eq!(VIDIOC_S_FMT, 0xc0d0_5605);
    }
}
