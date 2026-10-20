//! How long the GPU actually spent on the main pass.
//!
//! Wall-clock timing round `render()` measures the wrong thing twice over. It
//! includes every CPU cost in the call — scene traversal, uniform packing,
//! bind-group churn — and it excludes most of the GPU cost, because submission
//! returns long before the work runs. On a loaded machine it stops meaning
//! anything at all: three identical renders of the same frame here came back
//! at 18, 31 and 35 ms, which is a measurement of the other fourteen processes
//! on the box.
//!
//! Timestamp queries sidestep both problems. The GPU writes a tick count at the
//! start and end of the pass, and the difference is time on the device,
//! unaffected by whatever the CPU was doing.
//!
//! # Why the result is a frame behind
//!
//! Reading the resolved timestamps back means mapping a buffer, and mapping
//! blocks until the queue has caught up. Waiting for that inside `render()`
//! would stall the pipeline every frame — the cost of measuring would exceed
//! what is being measured, and the number would be wrong as a result. So the
//! query is resolved into a staging buffer that is mapped without waiting, and
//! read on a later frame once it happens to be ready. What you get back is a
//! recent frame's time, not the last one, which is the right trade for a
//! statistic nobody makes per-frame decisions on.

/// Timestamp queries around one render pass.
///
/// `None` when the adapter has no `TIMESTAMP_QUERY` — every caller has to cope
/// with not being able to measure, so that is the same path as an unfinished
/// readback rather than a separate error.
pub struct GpuTimer {
    set: wgpu::QuerySet,
    /// Where `resolve_query_set` writes: one u64 tick per mark.
    resolved: wgpu::Buffer,
    /// MAP_READ copy of the above, mapped between frames.
    staging: wgpu::Buffer,
    /// Nanoseconds per tick, from the queue.
    period: f32,
    /// Set while `staging` is mapped or a map is in flight.
    mapping: std::sync::Arc<std::sync::atomic::AtomicU8>,
    /// Ticks from the last completed readback, one per mark.
    ticks: Vec<u64>,
    marks: u32,
    last_ms: f32,
}

/// `mapping` states. A plain bool cannot tell "never asked" from "asked and
/// still waiting", and mapping a buffer that is already mapped is a panic.
const IDLE: u8 = 0;
const IN_FLIGHT: u8 = 1;
const READY: u8 = 2;

impl GpuTimer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        Self::with_marks(device, queue, 2)
    }

    /// A timer with `marks` points on the GPU's clock rather than two.
    ///
    /// One pass is one `beginning`/`end` pair, so timing a *stage* made of
    /// several passes is not a matter of bracketing it — on hardware without
    /// `TIMESTAMP_QUERY_INSIDE_ENCODERS` there is nowhere to write a
    /// timestamp between two passes. But the ticks all come off one clock, so
    /// marking the first pass of each stage and the last pass of the frame
    /// gives the boundaries, and consecutive differences give the stages —
    /// gaps between passes included, which is the honest accounting.
    ///
    /// See [`mark`](Self::mark), [`mark_end`](Self::mark_end) and
    /// [`span_ms`](Self::span_ms).
    pub fn with_marks(device: &wgpu::Device, queue: &wgpu::Queue, marks: u32) -> Option<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let marks = marks.max(2);
        let bytes = (marks as u64) * 8;
        let set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("threers frame timer"),
            ty: wgpu::QueryType::Timestamp,
            count: marks,
        });
        Some(Self {
            set,
            resolved: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("threers frame timer resolve"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            staging: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("threers frame timer readback"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            period: queue.get_timestamp_period(),
            mapping: std::sync::Arc::new(std::sync::atomic::AtomicU8::new(IDLE)),
            ticks: vec![0; marks as usize],
            marks,
            last_ms: 0.0,
        })
    }

    /// Write the GPU's clock at the *start* of a pass, into slot `i`.
    ///
    /// Put it on the first pass of a stage; the next mark closes it.
    pub fn mark(&self, i: u32) -> wgpu::RenderPassTimestampWrites<'_> {
        wgpu::RenderPassTimestampWrites {
            query_set: &self.set,
            beginning_of_pass_write_index: Some(i.min(self.marks - 1)),
            end_of_pass_write_index: None,
        }
    }

    /// Write the GPU's clock at the *end* of a pass, into slot `i`. For the
    /// last stage, which has no following mark to close it.
    pub fn mark_end(&self, i: u32) -> wgpu::RenderPassTimestampWrites<'_> {
        wgpu::RenderPassTimestampWrites {
            query_set: &self.set,
            beginning_of_pass_write_index: None,
            end_of_pass_write_index: Some(i.min(self.marks - 1)),
        }
    }

    /// Write the clock at both ends of one pass, into slots `begin` and
    /// `end`. For the pass that closes the last stage: it opens its own and
    /// has nothing after it to close it.
    pub fn mark_both(&self, begin: u32, end: u32) -> wgpu::RenderPassTimestampWrites<'_> {
        wgpu::RenderPassTimestampWrites {
            query_set: &self.set,
            beginning_of_pass_write_index: Some(begin.min(self.marks - 1)),
            end_of_pass_write_index: Some(end.min(self.marks - 1)),
        }
    }

    /// Milliseconds between marks `i` and `i + 1`, from the last completed
    /// readback. Zero if either is missing or the pair came back out of
    /// order.
    pub fn span_ms(&self, i: usize) -> f32 {
        match (self.ticks.get(i), self.ticks.get(i + 1)) {
            (Some(&a), Some(&b)) if b > a => (b - a) as f32 * self.period / 1.0e6,
            _ => 0.0,
        }
    }

    /// How many marks this timer has.
    pub fn marks(&self) -> u32 {
        self.marks
    }

    /// Pass to `RenderPassDescriptor::timestamp_writes` to bracket a pass.
    pub fn writes(&self) -> wgpu::RenderPassTimestampWrites<'_> {
        wgpu::RenderPassTimestampWrites {
            query_set: &self.set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        }
    }

    /// Queue the resolve and the copy back. Call once per frame, after the
    /// bracketed pass has ended and before the encoder is submitted.
    ///
    /// Skipped while a readback is outstanding: overwriting the staging buffer
    /// under an in-flight map is a validation error, and one measurement per
    /// round trip is all this is for.
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        use std::sync::atomic::Ordering;
        if self.mapping.load(Ordering::Acquire) != IDLE {
            return;
        }
        let bytes = (self.marks as u64) * 8;
        encoder.resolve_query_set(&self.set, 0..self.marks, &self.resolved, 0);
        encoder.copy_buffer_to_buffer(&self.resolved, 0, &self.staging, 0, bytes);
    }

    /// Start the map, or collect one that has finished. Call once per frame,
    /// after submit. Never blocks.
    pub fn poll(&mut self, device: &wgpu::Device) {
        use std::sync::atomic::Ordering;
        match self.mapping.load(Ordering::Acquire) {
            IDLE => {
                let flag = self.mapping.clone();
                flag.store(IN_FLIGHT, Ordering::Release);
                self.staging
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |r| {
                        // A failed map leaves the buffer unmapped, so go back
                        // to idle and try again next frame rather than
                        // deadlocking on a state that never advances.
                        flag.store(if r.is_ok() { READY } else { IDLE }, Ordering::Release);
                    });
                // Nudge the queue along. Without this the callback only fires
                // when something else happens to poll, and on a headless
                // renderer nothing does.
                let _ = device.poll(wgpu::PollType::Poll);
            }
            READY => {
                if let Ok(view) = self.staging.slice(..).get_mapped_range() {
                    for (i, slot) in self.ticks.iter_mut().enumerate() {
                        *slot = u64::from_le_bytes(
                            view[i * 8..i * 8 + 8].try_into().unwrap_or([0; 8]),
                        );
                    }
                    // Timestamps can come back equal or out of order on a pass
                    // the driver reordered; report nothing rather than a
                    // negative or absurd figure.
                    if self.ticks[self.marks as usize - 1] > self.ticks[0] {
                        self.last_ms = (self.ticks[self.marks as usize - 1] - self.ticks[0]) as f32
                            * self.period
                            / 1.0e6;
                    }
                }
                self.staging.unmap();
                self.mapping.store(IDLE, Ordering::Release);
            }
            _ => {
                let _ = device.poll(wgpu::PollType::Poll);
            }
        }
    }

    /// Milliseconds the GPU spent between the first and last mark, from the
    /// most recent completed readback. Zero until the first one lands.
    pub fn last_ms(&self) -> f32 {
        self.last_ms
    }
}
