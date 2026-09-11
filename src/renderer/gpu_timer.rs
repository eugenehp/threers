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
    /// Where `resolve_query_set` writes: two u64 ticks.
    resolved: wgpu::Buffer,
    /// MAP_READ copy of the above, mapped between frames.
    staging: wgpu::Buffer,
    /// Nanoseconds per tick, from the queue.
    period: f32,
    /// Set while `staging` is mapped or a map is in flight.
    mapping: std::sync::Arc<std::sync::atomic::AtomicU8>,
    last_ms: f32,
}

/// `mapping` states. A plain bool cannot tell "never asked" from "asked and
/// still waiting", and mapping a buffer that is already mapped is a panic.
const IDLE: u8 = 0;
const IN_FLIGHT: u8 = 1;
const READY: u8 = 2;

impl GpuTimer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("threers frame timer"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        Some(Self {
            set,
            resolved: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("threers frame timer resolve"),
                size: 16,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            staging: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("threers frame timer readback"),
                size: 16,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            period: queue.get_timestamp_period(),
            mapping: std::sync::Arc::new(std::sync::atomic::AtomicU8::new(IDLE)),
            last_ms: 0.0,
        })
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
        encoder.resolve_query_set(&self.set, 0..2, &self.resolved, 0);
        encoder.copy_buffer_to_buffer(&self.resolved, 0, &self.staging, 0, 16);
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
                    let mut t = [0u64; 2];
                    for (i, slot) in t.iter_mut().enumerate() {
                        *slot = u64::from_le_bytes(
                            view[i * 8..i * 8 + 8].try_into().unwrap_or([0; 8]),
                        );
                    }
                    // Timestamps can come back equal or out of order on a pass
                    // the driver reordered; report nothing rather than a
                    // negative or absurd figure.
                    if t[1] > t[0] {
                        self.last_ms = (t[1] - t[0]) as f32 * self.period / 1.0e6;
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

    /// Milliseconds the GPU spent in the bracketed pass, from the most recent
    /// completed readback. Zero until the first one lands.
    pub fn last_ms(&self) -> f32 {
        self.last_ms
    }
}
