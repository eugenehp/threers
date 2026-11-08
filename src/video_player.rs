//! Video on the surface of a mesh: decode a file and show it in a texture.
//!
//! A [`VideoPlayer`] opens a video held in memory, decodes it with the
//! pure-Rust decoders, and writes each picture into a
//! [`VideoTexture`] when it is due. Put
//! [`texture`](VideoPlayer::texture) in a material's `map` (or `emissive_map`,
//! for a screen that glows), and call [`update`](VideoPlayer::update) once a
//! frame with the time that passed.
//!
//! ```no_run
//! use threers::video_player::VideoPlayer;
//! use threers::materials::{BasicMaterial, Material};
//! use threers::math::Color;
//!
//! let mut player = VideoPlayer::open(std::fs::read("clip.mp4").unwrap()).unwrap();
//! player.set_looping(true);
//! let mut screen = BasicMaterial::new(Color::WHITE);
//! screen.map = Some(player.texture());
//! let material = Material::Basic(screen);
//! // Each frame:
//! player.update(1.0 / 60.0);
//! # let _ = material;
//! ```
//!
//! The files are those [`VideoReader`]
//! reads: MP4/MOV with H.264, HEVC or AV1, WebM/Matroska and IVF with VP9 or
//! AV1. Natively the decoding runs on a thread of its own, a few frames ahead,
//! so a frame of the game never waits for it; in a browser it runs inline,
//! decoding each picture as it comes due.
//!
//! The soundtrack (AAC in MP4/MOV, Opus in WebM/Matroska) is decoded too, on
//! a thread of its own, and [`soundtrack`](VideoPlayer::soundtrack) hands it
//! over for whatever plays sound: threers itself has no audio output. To keep
//! it in step, start it at [`time`](VideoPlayer::time) and start it again
//! there whenever [`jumps`](VideoPlayer::jumps) moves on (a seek, a loop) or
//! playback pauses; `threers-ecs` does exactly that for a `Video`.
//!
//! Pictures go to the texture as the decoder's YUV planes, and the GPU
//! renderers convert them to RGB in a shader pass, so no CPU pass touches
//! every pixel: what makes 4K and 8K playable. With the `videotoolbox`
//! feature on macOS, H.264 and HEVC decode in hardware.
//!
//! Needs the `native-decode` or `video` feature.

use crate::audio::Audio;
use crate::textures::{Samples, Texture, VideoTexture, YuvFrame, YuvMatrix};
use std::sync::Arc;
use threers_codec::bytes::{Bytes, FramePool};
use threers_codec::decode::{ChromaFormat, ColorInfo, DecodedFrame, Matrix, PackedFrame, VideoReader};

/// A decoded picture ready to show: when, and the picture.
type Ready = (f64, Picture);

/// A decoded picture, as the decoder that made it holds it.
enum Picture {
    /// Planes, for the renderer to convert on the GPU.
    Yuv(YuvFrame),
    /// The browser's own frame (WebCodecs), for the renderer to copy on the
    /// GPU.
    #[cfg(target_arch = "wasm32")]
    Browser(crate::textures::BrowserImage),
}

/// Where a [`VideoPlayer`]'s pictures are decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerDecoder {
    /// On a thread of its own, in hardware where the machine has it
    /// ([`VideoReader::backend`] says which).
    Thread,
    /// On the calling thread.
    Inline,
    /// In the browser's own decoder (WebCodecs).
    Browser,
}

/// Plays a video into a [`VideoTexture`]. See the [module docs](self).
pub struct VideoPlayer {
    texture: VideoTexture,
    duration: f64,
    time: f64,
    speed: f64,
    playing: bool,
    looping: bool,
    /// Decoded and not yet due.
    next: Option<Ready>,
    /// The decoder has no more pictures until rewound.
    ended: bool,
    error: Option<String>,
    decode: Decode,
    sound: Sound,
    /// Seeks and loops so far.
    jumps: u64,
    /// Where the decoder's picture buffers come from; frames go back here
    /// once replaced on screen, so a few buffers serve the whole film.
    pool: FramePool,
}

/// The soundtrack: being decoded, decoded, or not there.
enum Sound {
    /// The file, to decode on first asking (no threads).
    Pending(Bytes),
    /// On a thread of its own.
    #[cfg(not(target_arch = "wasm32"))]
    Decoding(std::sync::mpsc::Receiver<Option<Audio>>),
    Ready(Option<Audio>),
}

/// Where the pictures come from.
enum Decode {
    /// On a thread of its own, a few pictures ahead.
    #[cfg(not(target_arch = "wasm32"))]
    Thread(worker::Worker),
    /// On the calling thread, one picture at a time as they come due.
    Inline(Box<VideoReader>),
    /// In the browser's decoder (WebCodecs), in hardware where it has it.
    #[cfg(target_arch = "wasm32")]
    Browser(Box<web::WebCodecs>),
}

impl VideoPlayer {
    /// Open a video held in memory and decode its first picture, which shows
    /// at once. Fails if the container or codec is not supported or nothing
    /// in it decodes.
    ///
    /// In a browser the decoding is the browser's (WebCodecs), in hardware
    /// where the machine has it, and the frames go GPU to GPU into the
    /// texture; the first shows as soon as the browser has decoded it. A
    /// browser without WebCodecs, or one that refuses the codec, gets the
    /// native decoders instead.
    pub fn open(data: Vec<u8>) -> Result<Self, String> {
        Self::open_bytes(data.into())
    }

    /// Open the video at `path`, mapped rather than read: the operating
    /// system pages in what decoding touches, as it touches it, so a film of
    /// many gigabytes starts at once and is never all in memory. Otherwise as
    /// [`open`](Self::open).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open_file(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let data = Bytes::map(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::open_bytes(data)
    }

    fn open_bytes(data: Bytes) -> Result<Self, String> {
        #[cfg(target_arch = "wasm32")]
        if let Some(browser) = web::WebCodecs::open(&data) {
            let opened = Opened {
                size: browser.size(),
                duration: browser.duration(),
                first: None,
                pool: FramePool::default(),
            };
            return Ok(Self::with(opened, Decode::Browser(Box::new(browser)), Sound::Pending(data)));
        }
        Self::start(data, cfg!(not(target_arch = "wasm32")))
    }

    /// As [`open`](Self::open), decoding on the calling thread even where
    /// threads exist: for tests that want every frame in step with
    /// [`update`](Self::update), and for programs that must not start threads.
    pub fn open_inline(data: Vec<u8>) -> Result<Self, String> {
        Self::start(data.into(), false)
    }

    fn start(data: Bytes, threaded: bool) -> Result<Self, String> {
        // The decoders are not `Send` (they can hold a hardware hook), so a
        // thread opens the file itself and reports what it found.
        #[cfg(not(target_arch = "wasm32"))]
        if threaded {
            let (worker, opened) = worker::Worker::start(data.clone())?;
            let (tx, rx) = std::sync::mpsc::channel();
            let sound = match std::thread::Builder::new()
                .name("threers soundtrack".into())
                .spawn(move || {
                    let _ = tx.send(decode_soundtrack(&data));
                }) {
                Ok(_) => Sound::Decoding(rx),
                Err(_) => Sound::Ready(None),
            };
            return Ok(Self::with(opened, Decode::Thread(worker), sound));
        }
        let _ = threaded;
        let mut reader = VideoReader::open_shared(data.clone())?;
        let opened = Opened::first_of(&mut reader)?;
        Ok(Self::with(opened, Decode::Inline(Box::new(reader)), Sound::Pending(data)))
    }

    fn with(opened: Opened, decode: Decode, sound: Sound) -> Self {
        let mut player = Self {
            texture: VideoTexture::new(opened.size.0, opened.size.1),
            duration: opened.duration,
            time: 0.0,
            speed: 1.0,
            playing: true,
            looping: false,
            next: opened.first,
            ended: false,
            error: None,
            decode,
            sound,
            jumps: 0,
            pool: opened.pool,
        };
        player.show_due();
        player
    }

    /// The texture the video shows in, for a material.
    pub fn texture(&self) -> Arc<Texture> {
        self.texture.texture()
    }

    /// The [`VideoTexture`] itself.
    pub fn video_texture(&self) -> &VideoTexture {
        &self.texture
    }

    /// Picture size in pixels.
    pub fn size(&self) -> (u32, u32) {
        self.texture.size()
    }

    /// How long the clip runs, in seconds.
    pub fn duration(&self) -> f64 {
        self.duration
    }

    /// Where the pictures are being decoded.
    pub fn decoder(&self) -> PlayerDecoder {
        match &self.decode {
            #[cfg(not(target_arch = "wasm32"))]
            Decode::Thread(_) => PlayerDecoder::Thread,
            Decode::Inline(_) => PlayerDecoder::Inline,
            #[cfg(target_arch = "wasm32")]
            Decode::Browser(_) => PlayerDecoder::Browser,
        }
    }

    /// Where playback is, in seconds from the start.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// The playback rate [`set_speed`](Self::set_speed) set.
    pub fn speed(&self) -> f64 {
        self.speed
    }

    /// How many times playback has jumped rather than run on: each
    /// [`seek`](Self::seek), and each time a looping clip goes round. A
    /// soundtrack playing alongside starts again at [`time`](Self::time)
    /// whenever this moves.
    pub fn jumps(&self) -> u64 {
        self.jumps
    }

    /// The clip's sound, interleaved, once decoded: `None` while it is still
    /// being decoded on its thread, and for good when the file has no sound
    /// track this crate reads. Without threads it is decoded on this call,
    /// the first time.
    pub fn soundtrack(&mut self) -> Option<&Audio> {
        match &self.sound {
            Sound::Pending(data) => {
                let audio = decode_soundtrack(data);
                self.sound = Sound::Ready(audio);
            }
            #[cfg(not(target_arch = "wasm32"))]
            Sound::Decoding(rx) => match rx.try_recv() {
                Ok(audio) => self.sound = Sound::Ready(audio),
                Err(std::sync::mpsc::TryRecvError::Empty) => return None,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => self.sound = Sound::Ready(None),
            },
            Sound::Ready(_) => {}
        }
        match &self.sound {
            Sound::Ready(audio) => audio.as_ref(),
            _ => None,
        }
    }

    /// Whether time moves on in [`update`](Self::update). True from the start.
    pub fn is_playing(&self) -> bool {
        self.playing
    }

    /// Whether it has reached the end and stopped. Never true while looping.
    pub fn is_finished(&self) -> bool {
        !self.looping && self.ended && self.next.is_none()
    }

    /// Why decoding stopped early, if it did. What was decoded stays shown.
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// Start again from the beginning when the end is reached.
    pub fn set_looping(&mut self, looping: bool) {
        self.looping = looping;
    }

    /// Playback rate: 1 as recorded, 0.5 half speed. Negative is taken as 0.
    pub fn set_speed(&mut self, speed: f64) {
        self.speed = speed.max(0.0);
    }

    /// Carry on from where it is.
    pub fn play(&mut self) {
        self.playing = true;
    }

    /// Hold the current picture until [`play`](Self::play).
    pub fn pause(&mut self) {
        self.playing = false;
    }

    /// Go to `seconds` from the start, decoding forward from the beginning to
    /// reach it. The picture there shows on the next [`update`](Self::update).
    pub fn seek(&mut self, seconds: f64) {
        self.restart();
        self.time = seconds.clamp(0.0, self.duration);
    }

    /// Move playback on by `seconds` (times the speed) and show the picture
    /// due then. Pictures that came due in between are skipped, not shown in
    /// a burst.
    pub fn update(&mut self, seconds: f64) {
        if !self.playing {
            return;
        }
        self.time += seconds.max(0.0) * self.speed;
        self.show_due();
    }

    /// Show the latest picture due at `self.time`, looping or stopping at the
    /// end.
    fn show_due(&mut self) {
        let mut due: Option<Ready> = None;
        loop {
            if self.next.is_none() && !self.ended {
                match self.poll() {
                    Poll::Frame(ready) => self.next = Some(ready),
                    Poll::End => self.ended = true,
                    Poll::Pending => break,
                }
            }
            match &self.next {
                Some((at, _)) if *at <= self.time => {
                    // A picture already due and now passed over is not
                    // shown; its buffers go straight back.
                    if let Some((_, skipped)) = std::mem::replace(&mut due, self.next.take()) {
                        self.recycle(skipped);
                    }
                }
                Some(_) => break,
                None if self.ended => {
                    if self.looping && self.duration > 0.0 && self.time >= self.duration {
                        let into_next_round = self.time % self.duration;
                        self.restart();
                        self.time = into_next_round;
                        continue;
                    }
                    if !self.looping {
                        self.time = self.time.min(self.duration);
                    }
                    break;
                }
                None => break,
            }
        }
        match due.map(|(_, picture)| picture) {
            Some(Picture::Yuv(frame)) => {
                match self.texture.stream().swap_yuv(frame) {
                    Ok(Some(old)) | Err(old) => self.recycle(Picture::Yuv(old)),
                    Ok(None) => {}
                }
            }
            #[cfg(target_arch = "wasm32")]
            Some(Picture::Browser(image)) => {
                self.texture.stream().write_browser(image);
            }
            None => {}
        }
    }

    /// Give a picture's byte buffers back to the decoder's pool.
    fn recycle(&self, picture: Picture) {
        match picture {
            Picture::Yuv(frame) => {
                for samples in [frame.y, frame.u, frame.v] {
                    if let Samples::Bytes(buf) = samples {
                        self.pool.give(buf);
                    }
                }
            }
            #[cfg(target_arch = "wasm32")]
            Picture::Browser(_) => {}
        }
    }

    /// Back to the first picture, dropping what was decoded ahead.
    fn restart(&mut self) {
        self.jumps += 1;
        self.next = None;
        self.ended = false;
        self.time = 0.0;
        match &mut self.decode {
            #[cfg(not(target_arch = "wasm32"))]
            Decode::Thread(worker) => worker.rewind(),
            Decode::Inline(reader) => {
                if let Err(e) = reader.rewind() {
                    self.error = Some(e);
                    self.ended = true;
                }
            }
            #[cfg(target_arch = "wasm32")]
            Decode::Browser(browser) => browser.rewind(),
        }
    }

    fn poll(&mut self) -> Poll {
        match &mut self.decode {
            #[cfg(not(target_arch = "wasm32"))]
            Decode::Thread(worker) => match worker.poll() {
                Ok(poll) => poll,
                Err(e) => {
                    self.error = Some(e);
                    Poll::End
                }
            },
            #[cfg(target_arch = "wasm32")]
            Decode::Browser(browser) => match browser.poll() {
                Ok(poll) => poll,
                // Refused before a single picture (a codec this browser does
                // not decode): the native decoders take over from the start.
                Err(e) if !browser.produced() => match VideoReader::open_shared(browser.file()) {
                    Ok(reader) => {
                        log::info!("WebCodecs refused the video ({e}); decoding it natively");
                        self.decode = Decode::Inline(Box::new(reader));
                        Poll::Pending
                    }
                    Err(_) => {
                        self.error = Some(e);
                        Poll::End
                    }
                },
                Err(e) => {
                    self.error = Some(e);
                    Poll::End
                }
            },
            Decode::Inline(reader) => match reader.next_picture() {
                Ok(Some((at, picture))) => Poll::Frame((at, Picture::Yuv(to_yuv(picture, &self.pool)))),
                Ok(None) => Poll::End,
                Err(e) => {
                    self.error = Some(e);
                    Poll::End
                }
            },
        }
    }
}

/// 8-bit samples held as `u16`, as bytes, into a buffer from `pool`. A big
/// picture is split across the cores: at 8K this is fifty million samples.
fn narrow(samples: &[u16], pool: &FramePool) -> Vec<u8> {
    let n = samples.len();
    let mut out = pool.take(n);
    {
        // Written, not zeroed first: every byte gets a sample.
        let spare = &mut out.spare_capacity_mut()[..n];
        let fill = |dst: &mut [std::mem::MaybeUninit<u8>], src: &[u16]| {
            dst.iter_mut().zip(src).for_each(|(d, &s)| {
                d.write(s as u8);
            })
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            const CHUNK: usize = 1 << 20;
            if n > 2 * CHUNK {
                let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(8);
                let per = n.div_ceil(threads);
                std::thread::scope(|scope| {
                    for (src, dst) in samples.chunks(per).zip(spare.chunks_mut(per)) {
                        scope.spawn(move || fill(dst, src));
                    }
                });
            } else {
                fill(spare, samples);
            }
        }
        #[cfg(target_arch = "wasm32")]
        fill(spare, samples);
    }
    // SAFETY: the first `n` bytes were all written just above.
    unsafe { out.set_len(n) };
    out
}

/// The sound track of `data`, if it has one this crate decodes, interleaved.
fn decode_soundtrack(data: &[u8]) -> Option<Audio> {
    let pcm = threers_codec::audio_file::read(data).ok()?;
    let channels = pcm.channels.len();
    let frames = pcm.channels.first().map_or(0, Vec::len);
    if channels == 0 || frames == 0 {
        return None;
    }
    let mut samples = Vec::with_capacity(channels * frames);
    for i in 0..frames {
        for channel in &pcm.channels {
            samples.push(channel.get(i).copied().unwrap_or(0.0));
        }
    }
    Some(Audio {
        buffer: Some(Arc::new(samples)),
        sample_rate: pcm.sample_rate,
        channels: channels as u16,
        ..Audio::default()
    })
}

/// What opening a file found: its first picture, where it is ready yet.
struct Opened {
    size: (u32, u32),
    duration: f64,
    first: Option<Ready>,
    /// The reader's picture buffers, to give shown frames back to.
    pool: FramePool,
}

impl Opened {
    fn first_of(reader: &mut VideoReader) -> Result<Self, String> {
        let (at, picture) = reader.next_picture()?.ok_or("the video has no pictures")?;
        Ok(Self {
            size: picture.size(),
            duration: reader.duration(),
            first: Some((at, Picture::Yuv(to_yuv(picture, &reader.frame_pool())))),
            pool: reader.frame_pool(),
        })
    }
}

/// What asking for the next picture gave.
enum Poll {
    Frame(Ready),
    /// None decoded yet; the thread is behind. Only a thread can be.
    Pending,
    End,
}

/// A decoder's picture as the texture takes it. A hardware decoder's NV12
/// goes as it is, its Cb/Cr pairs read by the GPU; P010 has its samples
/// moved down from the high bits; the native decoders' planes are moved, or
/// at 8 bits narrowed to a byte a sample, which halves what goes to the GPU.
/// This runs on the decoding thread, not the one that renders.
fn to_yuv(picture: threers_codec::decode::Picture, pool: &FramePool) -> YuvFrame {
    match picture {
        threers_codec::decode::Picture::Planes(frame) => planes_to_yuv(frame, pool),
        threers_codec::decode::Picture::Packed(frame) => packed_to_yuv(frame),
    }
}

fn packed_to_yuv(frame: PackedFrame) -> YuvFrame {
    let (y, uv) = if frame.bit_depth <= 8 {
        (Samples::Bytes(frame.y), Samples::Bytes(frame.uv))
    } else {
        let shift = 16 - frame.bit_depth.min(16) as u32;
        let down = |b: &[u8]| -> Samples {
            Samples::Words(b.chunks_exact(2).map(|s| u16::from_le_bytes([s[0], s[1]]) >> shift).collect())
        };
        (down(&frame.y), down(&frame.uv))
    };
    YuvFrame {
        width: frame.width,
        height: frame.height,
        bit_depth: frame.bit_depth,
        subsampling: (2, 2),
        y,
        u: uv,
        v: Samples::Bytes(Vec::new()),
        interleaved: true,
        matrix: matrix(&frame.color),
        full_range: frame.color.full_range,
    }
}

fn matrix(color: &ColorInfo) -> YuvMatrix {
    match color.matrix {
        Matrix::Bt601 => YuvMatrix::Bt601,
        Matrix::Bt709 => YuvMatrix::Bt709,
        Matrix::Bt2020 => YuvMatrix::Bt2020,
    }
}

fn planes_to_yuv(frame: DecodedFrame, pool: &FramePool) -> YuvFrame {
    let subsampling = match frame.chroma {
        ChromaFormat::Yuv420 => (2, 2),
        ChromaFormat::Yuv422 => (2, 1),
        ChromaFormat::Yuv440 => (1, 2),
        ChromaFormat::Yuv444 | ChromaFormat::Monochrome => (1, 1),
    };
    let monochrome = frame.chroma == ChromaFormat::Monochrome;
    let plane = |samples: Vec<u16>| -> Samples {
        if frame.bit_depth == 8 {
            Samples::Bytes(narrow(&samples, pool))
        } else {
            Samples::Words(samples)
        }
    };
    let (u, v) = if monochrome {
        (Samples::default(), Samples::default())
    } else {
        (plane(frame.u), plane(frame.v))
    };
    YuvFrame {
        width: frame.width,
        height: frame.height,
        bit_depth: frame.bit_depth,
        subsampling,
        y: plane(frame.y),
        u,
        v,
        interleaved: false,
        matrix: matrix(&frame.color),
        full_range: frame.color.full_range,
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod worker {
    //! The decoding thread: a few pictures ahead, rewound on request.

    use super::{to_yuv, Opened, Picture, Poll, Ready};
    use threers_codec::bytes::Bytes;
    use threers_codec::decode::VideoReader;
    use std::sync::mpsc::{self, Receiver, Sender, SyncSender, TryRecvError};
    use std::thread::JoinHandle;

    /// How many pictures the thread decodes ahead.
    const AHEAD: usize = 3;

    enum Control {
        /// Back to the start; `u64` is the new generation.
        Rewind(u64),
        Stop,
    }

    enum Message {
        Frame(u64, Ready),
        End(u64),
        Failed(u64, String),
    }

    pub(super) struct Worker {
        frames: Receiver<Message>,
        control: Sender<Control>,
        /// Messages from before the last rewind are dropped.
        generation: u64,
        _thread: JoinHandle<()>,
    }

    impl Worker {
        /// Open `data` on a new thread, wait for its first picture, and leave
        /// the thread decoding ahead.
        pub(super) fn start(data: Bytes) -> Result<(Self, Opened), String> {
            let (frame_tx, frames) = mpsc::sync_channel::<Message>(AHEAD);
            let (control, control_rx) = mpsc::channel::<Control>();
            let (opened_tx, opened_rx) = mpsc::channel::<Result<Opened, String>>();
            let thread = std::thread::Builder::new()
                .name("threers video".into())
                .spawn(move || {
                    let opened = VideoReader::open_shared(data)
                        .and_then(|mut reader| Opened::first_of(&mut reader).map(|o| (reader, o)));
                    match opened {
                        Ok((mut reader, opened)) => {
                            if opened_tx.send(Ok(opened)).is_ok() {
                                run(&mut reader, &frame_tx, &control_rx);
                            }
                        }
                        Err(e) => {
                            let _ = opened_tx.send(Err(e));
                        }
                    }
                })
                .map_err(|e| format!("no thread to decode video on: {e}"))?;
            let opened = opened_rx
                .recv()
                .map_err(|_| "the video thread stopped while opening".to_string())??;
            let worker = Self {
                frames,
                control,
                generation: 0,
                _thread: thread,
            };
            Ok((worker, opened))
        }

        pub(super) fn poll(&mut self) -> Result<Poll, String> {
            loop {
                match self.frames.try_recv() {
                    Ok(Message::Frame(g, ready)) if g == self.generation => return Ok(Poll::Frame(ready)),
                    Ok(Message::End(g)) if g == self.generation => return Ok(Poll::End),
                    Ok(Message::Failed(g, e)) if g == self.generation => return Err(e),
                    Ok(_) => {}
                    Err(TryRecvError::Empty) => return Ok(Poll::Pending),
                    Err(TryRecvError::Disconnected) => return Ok(Poll::End),
                }
            }
        }

        pub(super) fn rewind(&mut self) {
            self.generation += 1;
            let _ = self.control.send(Control::Rewind(self.generation));
        }
    }

    impl Drop for Worker {
        fn drop(&mut self) {
            // The thread ends on the stop, or when it next sends and finds the
            // queue gone; it is not joined, as it may be mid-picture.
            let _ = self.control.send(Control::Stop);
        }
    }

    fn run(
        reader: &mut VideoReader,
        frames: &SyncSender<Message>,
        control: &Receiver<Control>,
    ) {
        let pool = reader.frame_pool();
        let mut generation = 0;
        let mut at_end = false;
        loop {
            // Requests first; at the end of the stream, wait for one.
            let request = if at_end {
                match control.recv() {
                    Ok(c) => Some(c),
                    Err(_) => return,
                }
            } else {
                match control.try_recv() {
                    Ok(c) => Some(c),
                    Err(TryRecvError::Empty) => None,
                    Err(TryRecvError::Disconnected) => return,
                }
            };
            match request {
                Some(Control::Stop) => return,
                Some(Control::Rewind(g)) => {
                    generation = g;
                    at_end = false;
                    if let Err(e) = reader.rewind() {
                        at_end = true;
                        if frames.send(Message::Failed(generation, e)).is_err() {
                            return;
                        }
                    }
                    continue;
                }
                None => {}
            }
            let message = match reader.next_picture() {
                Ok(Some((at, picture))) => Message::Frame(generation, (at, Picture::Yuv(to_yuv(picture, &pool)))),
                Ok(None) => {
                    at_end = true;
                    Message::End(generation)
                }
                Err(e) => {
                    at_end = true;
                    Message::Failed(generation, e)
                }
            };
            if frames.send(message).is_err() {
                return;
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    //! The browser's video decoder (WebCodecs), fed by this crate's demuxer.
    //!
    //! The browser decodes in hardware where the machine has it, and its
    //! frames stay `VideoFrame`s: the renderer copies them into the texture
    //! GPU to GPU. Bound by hand, as web-sys has WebCodecs only behind its
    //! unstable-APIs flag.

    use super::{Picture, Poll};
    use crate::textures::BrowserImage;
    use std::cell::{Cell, RefCell};
    use std::collections::VecDeque;
    use std::rc::Rc;
    use threers_codec::bytes::Bytes;
    use threers_codec::decode::{demux, webcodecs, Demuxed};
    use wasm_bindgen::prelude::*;
    use wasm_bindgen::JsCast;

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_name = VideoDecoder)]
        type JsVideoDecoder;
        #[wasm_bindgen(constructor, js_class = "VideoDecoder", catch)]
        fn new(init: &js_sys::Object) -> Result<JsVideoDecoder, JsValue>;
        #[wasm_bindgen(method, catch)]
        fn configure(this: &JsVideoDecoder, config: &js_sys::Object) -> Result<(), JsValue>;
        #[wasm_bindgen(method, catch)]
        fn decode(this: &JsVideoDecoder, chunk: &JsEncodedVideoChunk) -> Result<(), JsValue>;
        #[wasm_bindgen(method, catch)]
        fn flush(this: &JsVideoDecoder) -> Result<js_sys::Promise, JsValue>;
        #[wasm_bindgen(method, catch)]
        fn reset(this: &JsVideoDecoder) -> Result<(), JsValue>;
        #[wasm_bindgen(method, catch)]
        fn close(this: &JsVideoDecoder) -> Result<(), JsValue>;
        #[wasm_bindgen(method, getter, js_name = decodeQueueSize)]
        fn decode_queue_size(this: &JsVideoDecoder) -> u32;

        #[wasm_bindgen(js_name = EncodedVideoChunk)]
        type JsEncodedVideoChunk;
        #[wasm_bindgen(constructor, js_class = "EncodedVideoChunk", catch)]
        fn new(init: &js_sys::Object) -> Result<JsEncodedVideoChunk, JsValue>;
    }

    /// Pictures the browser keeps decoded ahead, counting those in its queue.
    const AHEAD: usize = 4;

    fn set(o: &js_sys::Object, key: &str, value: &JsValue) {
        let _ = js_sys::Reflect::set(o, &JsValue::from_str(key), value);
    }

    /// What the callbacks fill, shared with them.
    #[derive(Default)]
    struct Shared {
        frames: RefCell<VecDeque<(f64, web_sys::VideoFrame)>>,
        failed: RefCell<Option<String>>,
        flushed: Cell<bool>,
        /// Bumped on a rewind, so a flush from before it does not end the
        /// next round.
        generation: Cell<u64>,
    }

    pub(super) struct WebCodecs {
        /// The callbacks, for a decoder made again.
        init: js_sys::Object,
        /// Hardware was refused and the browser's software decoder asked for.
        in_software: bool,
        file: Bytes,
        track: Demuxed,
        config: js_sys::Object,
        decoder: JsVideoDecoder,
        shared: Rc<Shared>,
        next: usize,
        flushing: bool,
        produced: bool,
        on_flush: Option<Closure<dyn FnMut(JsValue)>>,
        _output: Closure<dyn FnMut(JsValue)>,
        _error: Closure<dyn FnMut(JsValue)>,
    }

    impl WebCodecs {
        /// A decoder for `data`, or `None` where the browser has no
        /// WebCodecs or the file is not one to give it.
        pub(super) fn open(data: &Bytes) -> Option<Self> {
            let global = js_sys::global();
            if !js_sys::Reflect::has(&global, &JsValue::from_str("VideoDecoder")).unwrap_or(false) {
                return None;
            }
            let track = demux(data.clone()).ok()?;
            let codec = webcodecs::codec_string(&track)?;
            if track.samples.is_empty() || !webcodecs::is_keyframe(&track, 0) {
                return None;
            }
            let config = js_sys::Object::new();
            set(&config, "codec", &JsValue::from_str(&codec));
            set(&config, "codedWidth", &JsValue::from(track.width));
            set(&config, "codedHeight", &JsValue::from(track.height));
            set(&config, "hardwareAcceleration", &JsValue::from_str("prefer-hardware"));
            set(&config, "optimizeForLatency", &JsValue::TRUE);
            if let Some(record) = webcodecs::description(&track) {
                set(&config, "description", &js_sys::Uint8Array::from(record).into());
            }

            let shared = Rc::new(Shared::default());
            let sink = shared.clone();
            let output = Closure::<dyn FnMut(JsValue)>::new(move |frame: JsValue| {
                let Ok(frame) = frame.dyn_into::<web_sys::VideoFrame>() else { return };
                let at = frame.timestamp() / 1e6;
                sink.frames.borrow_mut().push_back((at, frame));
            });
            let sink = shared.clone();
            let error = Closure::<dyn FnMut(JsValue)>::new(move |e: JsValue| {
                let message = js_sys::Reflect::get(&e, &JsValue::from_str("message"))
                    .ok()
                    .and_then(|m| m.as_string())
                    .unwrap_or_else(|| format!("{e:?}"));
                *sink.failed.borrow_mut() = Some(format!("WebCodecs: {message}"));
            });
            let init = js_sys::Object::new();
            set(&init, "output", output.as_ref());
            set(&init, "error", error.as_ref());
            let decoder = JsVideoDecoder::new(&init).ok()?;
            decoder.configure(&config).ok()?;
            Some(Self {
                init,
                in_software: false,
                file: data.clone(),
                track,
                config,
                decoder,
                shared,
                next: 0,
                flushing: false,
                produced: false,
                on_flush: None,
                _output: output,
                _error: error,
            })
        }

        pub(super) fn size(&self) -> (u32, u32) {
            (self.track.width, self.track.height)
        }

        pub(super) fn duration(&self) -> f64 {
            self.track.duration()
        }

        /// Whether any picture has come out yet.
        pub(super) fn produced(&self) -> bool {
            self.produced
        }

        /// The file, for the native decoders to take over.
        pub(super) fn file(&self) -> Bytes {
            self.file.clone()
        }

        /// Keep the browser a few pictures ahead, and hand over the next
        /// one if it has come out.
        pub(super) fn poll(&mut self) -> Result<Poll, String> {
            let failed = self.shared.failed.borrow().clone();
            if let Some(e) = failed {
                // "prefer-hardware" is refused outright where the machine has
                // no hardware for the codec; the browser's own software
                // decoder still beats ours, so ask for that before giving up.
                if !self.produced && !self.in_software {
                    self.in_software = true;
                    set(&self.config, "hardwareAcceleration", &JsValue::from_str("no-preference"));
                    let decoder = JsVideoDecoder::new(&self.init).map_err(|_| e.clone())?;
                    decoder.configure(&self.config).map_err(|_| e.clone())?;
                    self.decoder = decoder;
                    *self.shared.failed.borrow_mut() = None;
                    self.rewind();
                    return Ok(Poll::Pending);
                }
                return Err(e);
            }
            let total = self.track.samples.len();
            while self.next < total && self.shared.frames.borrow().len() + (self.decoder.decode_queue_size() as usize) < AHEAD {
                let i = self.next;
                self.next += 1;
                let s = self.track.samples[i];
                let init = js_sys::Object::new();
                let key = i == 0 || webcodecs::is_keyframe(&self.track, i);
                set(&init, "type", &JsValue::from_str(if key { "key" } else { "delta" }));
                set(&init, "timestamp", &JsValue::from((s.seconds * 1e6).round()));
                set(&init, "data", &js_sys::Uint8Array::from(self.track.sample(i)).into());
                let chunk = JsEncodedVideoChunk::new(&init).map_err(|e| format!("EncodedVideoChunk: {e:?}"))?;
                self.decoder.decode(&chunk).map_err(|e| format!("decode: {e:?}"))?;
            }
            if self.next == total && !self.flushing {
                // The last pictures come out only on a flush.
                self.flushing = true;
                if let Ok(done) = self.decoder.flush() {
                    let shared = self.shared.clone();
                    let generation = shared.generation.get();
                    let on_done = Closure::<dyn FnMut(JsValue)>::new(move |_: JsValue| {
                        if shared.generation.get() == generation {
                            shared.flushed.set(true);
                        }
                    });
                    let _ = done.then(&on_done);
                    // Kept until the next flush replaces it.
                    self.on_flush = Some(on_done);
                }
            }
            if let Some((at, frame)) = self.shared.frames.borrow_mut().pop_front() {
                self.produced = true;
                return Ok(Poll::Frame((at, Picture::Browser(BrowserImage::Frame(frame)))));
            }
            Ok(if self.shared.flushed.get() { Poll::End } else { Poll::Pending })
        }

        /// Back to the first picture.
        pub(super) fn rewind(&mut self) {
            self.shared.generation.set(self.shared.generation.get() + 1);
            let _ = self.decoder.reset();
            let _ = self.decoder.configure(&self.config);
            for (_, frame) in self.shared.frames.borrow_mut().drain(..) {
                frame.close();
            }
            self.shared.flushed.set(false);
            self.next = 0;
            self.flushing = false;
        }
    }

    impl Drop for WebCodecs {
        fn drop(&mut self) {
            for (_, frame) in self.shared.frames.borrow_mut().drain(..) {
                frame.close();
            }
            let _ = self.decoder.close();
        }
    }
}
