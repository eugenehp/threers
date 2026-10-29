//! Which mark is under the cursor, answered on the GPU.
//!
//! Every pickable mark is drawn again into an integer target, writing
//! `(layer, row)` instead of a colour, and the pixels around the cursor are
//! copied back. That costs one extra draw of what is visible, whatever the
//! number of instances — a CPU ray test against a million points costs a
//! million tests, or a spatial index that has to be kept in step with the data.
//!
//! The readback is asynchronous. [`MarkRenderer::pick`] records the work and
//! returns a [`PickRequest`]; poll the device and call
//! [`PickRequest::try_take`] on a later frame. Blocking for the answer
//! ([`PickRequest::wait`]) is native-only: waiting on the GPU from a browser's
//! main thread deadlocks the page, because the callback that would end the wait
//! runs on that same thread.

use super::{DepthKey, Mark, MarkRenderer, MarkView};

/// The integer target the ids are drawn into: `(layer id, row)`. Layer ids are
/// 1-based so that 0 means nothing was hit.
pub(crate) const PICK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Uint;

#[derive(Default)]
pub(crate) struct PickTargets {
    size: (u32, u32),
    color: Option<(wgpu::Texture, wgpu::TextureView)>,
    depth: Option<wgpu::TextureView>,
}

impl PickTargets {
    fn ensure(&mut self, device: &wgpu::Device, size: (u32, u32)) {
        if self.size == size && self.color.is_some() {
            return;
        }
        let mk = |label, format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let color = mk(
            "threers mark pick ids",
            PICK_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let depth = mk(
            "threers mark pick depth",
            super::DEPTH_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let view = color.create_view(&Default::default());
        self.color = Some((color, view));
        self.depth = Some(depth.create_view(&Default::default()));
        self.size = size;
    }
}

/// What was under the cursor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PickHit {
    /// Index into the slice of marks given to [`MarkRenderer::pick`].
    pub mark: usize,
    /// The instance's row. For [`SegmentMarks`](super::SegmentMarks) built
    /// from paths, map it back with
    /// [`path_of`](super::SegmentMarks::path_of); for
    /// [`TextLabels`](super::TextLabels) it is the label's index.
    pub index: u32,
    /// The pixel that was hit, top-left origin.
    pub pixel: (u32, u32),
    /// The row's id from
    /// [`MarkCommon::set_feature_ids`](super::MarkCommon::set_feature_ids),
    /// when the layer has them: stable across tiles, where rows are not.
    pub feature: Option<u64>,
}

/// A pick in flight.
pub struct PickRequest {
    pending: Option<Pending>,
}

struct Pending {
    buffer: wgpu::Buffer,
    rx: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    origin: (u32, u32),
    size: (u32, u32),
    padded_row: u32,
    center: (f32, f32),
    /// Each layer's feature ids, as they were when the pick was made.
    features: Vec<Option<std::sync::Arc<[u64]>>>,
}

/// A rectangle pick in flight: every distinct row drawn inside it.
pub struct PickRectRequest {
    pending: Option<Pending>,
}

impl PickRectRequest {
    /// The hits once the GPU has delivered them — one per distinct
    /// `(mark, row)`, at the first pixel found — or `None` while in flight.
    pub fn try_take(&mut self) -> Option<Vec<PickHit>> {
        let Some(p) = self.pending.as_ref() else {
            return Some(Vec::new());
        };
        match p.rx.try_recv() {
            Ok(Ok(())) => Some(self.pending.take().unwrap().all()),
            Ok(Err(_)) | Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.pending = None;
                Some(Vec::new())
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
        }
    }

    /// Block until the hits arrive. **Native only.**
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait(mut self, device: &wgpu::Device) -> Vec<PickHit> {
        loop {
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            if let Some(hits) = self.try_take() {
                return hits;
            }
        }
    }
}

impl PickRequest {
    /// The answer, once the GPU has delivered it: `None` while it is still in
    /// flight, `Some(None)` when nothing pickable was under the cursor.
    ///
    /// Nothing arrives unless the device is polled — `device.poll(PollType::Poll)`
    /// once a frame, which a browser does on its own.
    pub fn try_take(&mut self) -> Option<Option<PickHit>> {
        let Some(p) = self.pending.as_ref() else {
            return Some(None);
        };
        match p.rx.try_recv() {
            Ok(Ok(())) => {
                let p = self.pending.take().unwrap();
                Some(p.resolve())
            }
            Ok(Err(_)) | Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.pending = None;
                Some(None)
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
        }
    }

    /// Block until the answer arrives. **Native only.**
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait(mut self, device: &wgpu::Device) -> Option<PickHit> {
        self.pending.as_ref()?;
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        loop {
            if let Some(hit) = self.try_take() {
                return hit;
            }
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }
    }
}

impl Pending {
    fn hit(&self, id: u32, row: u32, pixel: (u32, u32)) -> PickHit {
        let mark = id as usize - 1;
        PickHit {
            mark,
            index: row,
            pixel,
            feature: self
                .features
                .get(mark)
                .and_then(|f| f.as_ref())
                .and_then(|f| f.get(row as usize).copied()),
        }
    }

    /// Every distinct hit in the region.
    fn all(self) -> Vec<PickHit> {
        let slice = self.buffer.slice(..);
        let Ok(bytes) = slice.get_mapped_range() else {
            return Vec::new();
        };
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for y in 0..self.size.1 {
            for x in 0..self.size.0 {
                let o = (y * self.padded_row + x * 8) as usize;
                let id = u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
                let row = u32::from_le_bytes(bytes[o + 4..o + 8].try_into().unwrap());
                if id != 0 && seen.insert((id, row)) {
                    out.push(self.hit(id, row, (self.origin.0 + x, self.origin.1 + y)));
                }
            }
        }
        drop(bytes);
        self.buffer.unmap();
        out
    }

    fn resolve(self) -> Option<PickHit> {
        let slice = self.buffer.slice(..);
        let bytes = slice.get_mapped_range().ok()?;
        let mut best: Option<(f32, PickHit)> = None;
        for y in 0..self.size.1 {
            for x in 0..self.size.0 {
                let o = (y * self.padded_row + x * 8) as usize;
                let id = u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
                if id == 0 {
                    continue;
                }
                let row = u32::from_le_bytes(bytes[o + 4..o + 8].try_into().unwrap());
                let (px, py) = (self.origin.0 + x, self.origin.1 + y);
                let dx = px as f32 + 0.5 - self.center.0;
                let dy = py as f32 + 0.5 - self.center.1;
                let d = dx * dx + dy * dy;
                if best.is_none_or(|(bd, _)| d < bd) {
                    best = Some((d, self.hit(id, row, (px, py))));
                }
            }
        }
        drop(bytes);
        self.buffer.unmap();
        best.map(|(_, h)| h)
    }
}

impl MarkRenderer {
    /// Find the mark nearest pixel `(x, y)` (top-left origin) within `radius`
    /// pixels.
    ///
    /// `occluder` is the frame's depth buffer — single-sampled and the
    /// viewport's size — so marks hidden behind meshes are not picked. Without
    /// it, marks hide only each other.
    #[allow(clippy::too_many_arguments)]
    pub fn pick(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &MarkView,
        marks: &mut [&mut dyn Mark],
        occluder: Option<&wgpu::TextureView>,
        x: f32,
        y: f32,
        radius: u32,
    ) -> PickRequest {
        let (w, h) = view.viewport;
        let cx = x.floor() as i64;
        let cy = y.floor() as i64;
        let r = radius as i64;
        let x0 = (cx - r).clamp(0, w as i64) as u32;
        let y0 = (cy - r).clamp(0, h as i64) as u32;
        let x1 = (cx + r + 1).clamp(0, w as i64) as u32;
        let y1 = (cy + r + 1).clamp(0, h as i64) as u32;
        PickRequest {
            pending: self.pick_region(
                device,
                queue,
                view,
                marks,
                occluder,
                [x0, y0, x1, y1],
                (x, y),
            ),
        }
    }

    /// Every distinct `(mark, row)` drawn in the pixel rectangle from
    /// `(x0, y0)` to `(x1, y1)` (top-left origin, exclusive) — box
    /// selection. `occluder` as for [`pick`](Self::pick).
    #[allow(clippy::too_many_arguments)]
    pub fn pick_rect(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &MarkView,
        marks: &mut [&mut dyn Mark],
        occluder: Option<&wgpu::TextureView>,
        (x0, y0): (u32, u32),
        (x1, y1): (u32, u32),
    ) -> PickRectRequest {
        let (w, h) = view.viewport;
        let r = [x0.min(w), y0.min(h), x1.min(w), y1.min(h)];
        let c = ((r[0] + r[2]) as f32 * 0.5, (r[1] + r[3]) as f32 * 0.5);
        PickRectRequest {
            pending: self.pick_region(device, queue, view, marks, occluder, r, c),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn pick_region(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &MarkView,
        marks: &mut [&mut dyn Mark],
        occluder: Option<&wgpu::TextureView>,
        [x0, y0, x1, y1]: [u32; 4],
        (x, y): (f32, f32),
    ) -> Option<Pending> {
        let (w, h) = view.viewport;
        if x0 >= x1 || y0 >= y1 {
            return None;
        }

        queue.write_buffer(
            &self.frame_uniform,
            0,
            bytemuck::bytes_of(&view.uniforms(false, 1, [0.0; 2])),
        );
        self.prepare_all(device, queue, view, marks);
        self.pick.ensure(device, (w, h));

        let use_occluder = occluder.is_some();
        let pickable = |m: &dyn Mark| {
            let s = &m.common().style;
            s.visible && s.pickable && m.instance_count() > 0 && m.kind() != super::MarkKind::Splat
        };
        let key_of = |m: &dyn Mark| {
            let mut k = Self::key_for(m, PICK_FORMAT, 1, true, true, view.reversed_z, view.flip_y);
            k.blend = super::MarkBlend::Alpha;
            k.depth = if use_occluder {
                // The frame's depth is read, never written: a pick must not
                // change what the next pass sees.
                DepthKey::Test {
                    test: true,
                    write: false,
                    reversed: view.reversed_z,
                }
            } else {
                DepthKey::Test {
                    test: true,
                    write: true,
                    reversed: view.reversed_z,
                }
            };
            k
        };
        for m in marks.iter_mut() {
            if pickable(&**m) {
                let key = key_of(&**m);
                self.ensure_pipeline(device, &key, &mut **m);
            }
        }

        let (color_tex, color_view) = self.pick.color.as_ref().unwrap();
        let depth_view = occluder.unwrap_or(self.pick.depth.as_ref().unwrap());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("threers mark pick"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("threers mark pick pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: if use_occluder {
                            wgpu::LoadOp::Load
                        } else {
                            wgpu::LoadOp::Clear(if view.reversed_z { 0.0 } else { 1.0 })
                        },
                        store: if use_occluder {
                            wgpu::StoreOp::Store
                        } else {
                            wgpu::StoreOp::Discard
                        },
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_scissor_rect(x0, y0, x1 - x0, y1 - y0);
            pass.set_bind_group(0, &self.frame_bind_group, &[]);
            for m in marks.iter_mut() {
                if !pickable(&**m) {
                    continue;
                }
                let key = key_of(&**m);
                Self::record(
                    &mut pass,
                    &self.pipelines[&key],
                    self.layer_bind_group(),
                    &mut **m,
                );
            }
        }
        let size = (x1 - x0, y1 - y0);
        let padded_row = (size.0 * 8).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threers mark pick readback"),
            size: (padded_row * size.1) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: color_tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x: x0, y: y0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(size.1),
                },
            },
            wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        Some(Pending {
            buffer,
            rx,
            origin: (x0, y0),
            size,
            padded_row,
            center: (x, y),
            features: marks
                .iter()
                .map(|m| m.common().feature_ids.clone())
                .collect(),
        })
    }
}
