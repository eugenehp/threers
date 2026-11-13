//! GPU resources for `ShaderMaterial` draws, kept from frame to frame.
//!
//! Every custom draw used to build a fresh uniform buffer, three storage
//! buffers (or data textures) and a bind group, every frame. For a material
//! carrying megabytes of per-object data that is megabytes uploaded per draw
//! per frame — and a tiled map hands one material to hundreds of meshes.
//!
//! What is kept, and what it is keyed on:
//!
//! - **per material value** ([`ShaderLive::id`](crate::materials::ShaderLive)):
//!   the uniform buffer, storage uploaded from the `storage0..2` fields, and
//!   one bind group per draw slot. The bind group has to be per draw because
//!   the mesh uniform shares its group; everything it points at is shared.
//! - **per live state** (shared by clones): storage set through
//!   `set_storage` / `set_storage_range`, so clones handed the same live
//!   storage upload it once.
//!
//! The uniform block is compared by value each frame and written when it
//! differs — 256 bytes, cheaper to compare than to version. A storage field is
//! recognised by its allocation and length, plus a revision the caller bumps
//! for in-place edits; live storage by its version, with recent ranges logged
//! so a range update uploads only the range. A buffer is recreated only when
//! the data outgrows it; a bind group only when something it binds changed —
//! a storage slot, or one of the resources the caller resolves each frame (the
//! mesh uniform's slice of the arena, the textures, the volumes, the layout),
//! compared by identity and held so an identity cannot be reused under it.
//!
//! Entries not drawn for [`KEEP_FRAMES`] renders are dropped.

use crate::materials::shader_material::LiveSlot;
use crate::materials::ShaderMaterial;
use std::collections::HashMap;

/// Renders an unused entry survives. Counted in `render` calls rather than
/// displayed frames, so a frame with reflection or cube-camera passes ages
/// entries faster; this is generous for that reason.
const KEEP_FRAMES: u64 = 240;

/// How many GPU objects the custom-shader path has created, and how much it
/// has uploaded, since the renderer was built. Exists so tests and profiling
/// can see that an unchanged material costs nothing per frame.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CustomShaderStats {
    /// Uniform and storage buffers created.
    pub buffers_created: u64,
    /// Storage data textures created (the WebGL2 path).
    pub textures_created: u64,
    /// `@group(1)` bind groups created.
    pub bind_groups_created: u64,
    /// Bytes written to uniform and storage buffers and data textures.
    pub bytes_uploaded: u64,
    /// Material values with resources currently cached.
    pub materials_cached: usize,
}

/// What the cache needs from the renderer to create things.
pub(crate) struct CustomCtx<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub data_placeholder: &'a wgpu::TextureView,
    pub sampler: &'a wgpu::Sampler,
    pub volume_sampler: &'a wgpu::Sampler,
    pub storage_buffers: bool,
}

/// What one draw's `@group(1)` binds besides the material's own uniform and
/// storage: resolved by the renderer each frame, and compared by identity.
pub(crate) struct DrawBinds<'a> {
    /// The mesh uniform: this draw's slice of the mesh arena.
    pub mesh_buffer: &'a wgpu::Buffer,
    pub mesh_offset: u64,
    pub mesh_size: u64,
    /// The layout: with the volume slots, or without them.
    pub bgl: &'a wgpu::BindGroupLayout,
    /// The four texture slots, placeholders already filled in.
    pub textures: [&'a wgpu::TextureView; 4],
    /// The two volume slots, where the layout has them.
    pub volumes: Option<[&'a wgpu::TextureView; 2]>,
}

impl DrawBinds<'_> {
    fn identity(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.mesh_buffer.hash(&mut h);
        (self.mesh_offset, self.mesh_size).hash(&mut h);
        self.bgl.hash(&mut h);
        self.textures.hash(&mut h);
        self.volumes.hash(&mut h);
        h.finish()
    }
}

#[derive(Default)]
pub(crate) struct CustomCache {
    frame: u64,
    next_generation: u64,
    stats: CustomShaderStats,
    /// Shader-source hash per material value, keyed by `(ptr, len)` of both
    /// sources so it is recomputed only when one is replaced.
    hashes: HashMap<u64, HashMemo>,
    mats: HashMap<u64, MatGpu>,
    /// Live storage, keyed by `(shared id, slot)`.
    live: HashMap<(u64, usize), LiveGpu>,
}

struct HashMemo {
    fp: [usize; 4],
    hash: u64,
    last_frame: u64,
}

struct MatGpu {
    last_frame: u64,
    uniform: wgpu::Buffer,
    /// What `uniform` holds; `None` until the first write.
    data: Option<[[f32; 4]; 16]>,
    fields: [FieldGpu; 3],
    bind_groups: HashMap<usize, BgEntry>,
}

struct FieldGpu {
    gpu: Option<SlotGpu>,
    fp: (usize, usize, u64),
}

struct LiveGpu {
    gpu: Option<SlotGpu>,
    version: u64,
    last_frame: u64,
}

enum SlotRes {
    Buffer(wgpu::Buffer),
    Texture(wgpu::Texture, wgpu::TextureView),
}

struct SlotGpu {
    res: SlotRes,
    /// Floats the buffer can hold.
    cap: usize,
    /// The texture's `(width, height)`.
    dims: (u32, u32),
    /// Floats currently meaningful — what a buffer binding is sized to.
    len: usize,
    /// Changes whenever the binding this slot contributes would change.
    generation: u64,
}

struct BgEntry {
    /// The storage slots' gens and the identity of what [`DrawBinds`] bound.
    key: [u64; 4],
    bg: wgpu::BindGroup,
    last_frame: u64,
    /// What the identity was computed from, held so none of it can be freed
    /// and its identity reused by something else while the key trusts it.
    _held: (wgpu::Buffer, wgpu::BindGroupLayout, Vec<wgpu::TextureView>),
}

/// The texture shape for `floats` floats: rows of at most 2048 texels.
///
/// 2048 is the smallest `max_texture_dimension_2d` anything still shipping
/// reports, so a row never has to be checked against the device.
fn data_tex_dims(floats: usize) -> (u32, u32) {
    let texels = floats.div_ceil(4).max(1);
    let width = texels.clamp(1, 2048) as u32;
    let height = (texels as u32).div_ceil(width).max(1);
    (width, height)
}

impl CustomCache {
    pub(crate) fn stats(&self) -> CustomShaderStats {
        CustomShaderStats {
            materials_cached: self.mats.len(),
            ..self.stats
        }
    }

    /// Start a render: age everything by one.
    pub(crate) fn begin_frame(&mut self) {
        self.frame += 1;
    }

    /// Drop what has not been drawn for a while.
    pub(crate) fn evict(&mut self) {
        let alive = |last: u64, now: u64| last + KEEP_FRAMES >= now;
        let now = self.frame;
        self.hashes.retain(|_, h| alive(h.last_frame, now));
        self.live.retain(|_, l| alive(l.last_frame, now));
        self.mats.retain(|_, m| alive(m.last_frame, now));
        for m in self.mats.values_mut() {
            m.bind_groups.retain(|_, b| alive(b.last_frame, now));
        }
    }

    /// The pipeline-cache hash of both shader stages (`custom_shader_hash`),
    /// recomputed only when one of the sources has been replaced rather than
    /// rehashing kilobytes of WGSL for every draw.
    pub(crate) fn shader_hash(&mut self, sm: &ShaderMaterial) -> u64 {
        let v = sm.vertex.as_deref();
        let fp = [
            sm.fragment.as_ptr() as usize,
            sm.fragment.len(),
            v.map_or(0, |v| v.as_ptr() as usize),
            v.map_or(usize::MAX, str::len),
        ];
        let frame = self.frame;
        let memo = self.hashes.entry(sm.live.id()).or_insert(HashMemo {
            fp: [0, 0, 0, 0],
            hash: 0,
            last_frame: frame,
        });
        memo.last_frame = frame;
        if memo.fp != fp {
            memo.fp = fp;
            memo.hash = super::renderer::custom_shader_hash(&sm.fragment, v);
        }
        memo.hash
    }

    /// The `@group(1)` bind group for `sm` drawn in slot `draw`, whose mesh
    /// uniform is `mesh_uniform`. Uploads whatever changed since the last frame
    /// and nothing else.
    pub(crate) fn bind_group(
        &mut self,
        ctx: &CustomCtx<'_>,
        sm: &ShaderMaterial,
        draw: usize,
        binds: &DrawBinds<'_>,
    ) -> wgpu::BindGroup {
        let frame = self.frame;
        let id = sm.live.id();
        if !self.mats.contains_key(&id) {
            let uniform = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("threers user uniform"),
                size: (16 * 16) as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.stats.buffers_created += 1;
            self.mats.insert(
                id,
                MatGpu {
                    last_frame: frame,
                    uniform,
                    data: None,
                    // A length no allocation has, so the first frame uploads.
                    fields: std::array::from_fn(|_| FieldGpu {
                        gpu: None,
                        fp: (0, usize::MAX, 0),
                    }),
                    bind_groups: HashMap::new(),
                },
            );
        }

        let live = sm.live.lock();
        let data = live.data.unwrap_or_else(|| sm.field_slots());
        let mut key = [0u64; 4];
        {
            let Self {
                mats,
                live: live_gpu,
                stats,
                next_generation,
                ..
            } = self;
            let mat = mats.get_mut(&id).expect("inserted above");
            mat.last_frame = frame;
            // Bitwise, so a NaN the shader is meant to see does not count as a
            // change every frame.
            let same = mat
                .data
                .is_some_and(|d| bytemuck::bytes_of(&d) == bytemuck::bytes_of(&data));
            if !same {
                ctx.queue
                    .write_buffer(&mat.uniform, 0, bytemuck::bytes_of(&data));
                stats.bytes_uploaded += 256;
                mat.data = Some(data);
            }
            for (slot, k) in key.iter_mut().take(3).enumerate() {
                *k = match &live.storage[slot] {
                    Some(ls) => {
                        let entry = live_gpu.entry((sm.live.shared_id(), slot));
                        sync_live(ctx, entry, ls, frame, next_generation, stats)
                    }
                    None => {
                        let field = sm.storage_field(slot);
                        let fp = (
                            field.as_ptr() as usize,
                            field.len(),
                            sm.live.field_rev(slot),
                        );
                        let f = &mut mat.fields[slot];
                        match &f.gpu {
                            Some(g) if f.fp == fp => g.generation,
                            _ => {
                                f.fp = fp;
                                upload(ctx, &mut f.gpu, field, None, next_generation, stats)
                            }
                        }
                    }
                };
            }
            key[3] = binds.identity();
            if let Some(b) = mat.bind_groups.get_mut(&draw)
                && b.key == key
            {
                b.last_frame = frame;
                return b.bg.clone();
            }
        }

        // Something it binds changed, or this slot is new: build it.
        let mat = &self.mats[&id];
        let slot_res = |slot: usize| -> &SlotGpu {
            match &live.storage[slot] {
                Some(_) => self.live[&(sm.live.shared_id(), slot)].gpu.as_ref(),
                None => mat.fields[slot].gpu.as_ref(),
            }
            .expect("synced above")
        };
        let slots = [slot_res(0), slot_res(1), slot_res(2)];
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: binds.mesh_buffer,
                    offset: binds.mesh_offset,
                    size: wgpu::BufferSize::new(binds.mesh_size),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: mat.uniform.as_entire_binding(),
            },
        ];
        for (i, s) in slots.iter().enumerate() {
            if let SlotRes::Buffer(b) = &s.res {
                // Sized to the data rather than the buffer, so a slot that
                // shrank into a bigger buffer still reports its own
                // `arrayLength`.
                entries.push(wgpu::BindGroupEntry {
                    binding: 2 + i as u32,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: b,
                        offset: 0,
                        size: wgpu::BufferSize::new((s.len * 4) as u64),
                    }),
                });
            }
        }
        for (i, view) in binds.textures.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: 5 + i as u32,
                resource: wgpu::BindingResource::TextureView(view),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: 9,
            resource: wgpu::BindingResource::Sampler(ctx.sampler),
        });
        // The data textures are only built where they are the only option; with
        // storage buffers present these bind a 1x1 placeholder nothing reads, so
        // the layout is the same shape on every backend and the module is not.
        for (i, s) in slots.iter().enumerate() {
            let view = match &s.res {
                SlotRes::Texture(_, v) => v,
                SlotRes::Buffer(_) => ctx.data_placeholder,
            };
            entries.push(wgpu::BindGroupEntry {
                binding: 10 + i as u32,
                resource: wgpu::BindingResource::TextureView(view),
            });
        }
        if let Some(volumes) = &binds.volumes {
            for (i, view) in volumes.iter().enumerate() {
                entries.push(wgpu::BindGroupEntry {
                    binding: 13 + i as u32,
                    resource: wgpu::BindingResource::TextureView(view),
                });
            }
            entries.push(wgpu::BindGroupEntry {
                binding: 15,
                resource: wgpu::BindingResource::Sampler(ctx.volume_sampler),
            });
        }
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers custom mesh+user bg"),
            layout: binds.bgl,
            entries: &entries,
        });
        drop(live);
        self.stats.bind_groups_created += 1;
        self.mats.get_mut(&id).expect("present").bind_groups.insert(
            draw,
            BgEntry {
                key,
                bg: bg.clone(),
                last_frame: frame,
                _held: (
                    binds.mesh_buffer.clone(),
                    binds.bgl.clone(),
                    binds
                        .textures
                        .iter()
                        .chain(binds.volumes.iter().flatten())
                        .map(|v| (*v).clone())
                        .collect(),
                ),
            },
        );
        bg
    }
}

/// Bring the GPU copy of a live slot up to date; returns its binding generation.
fn sync_live(
    ctx: &CustomCtx<'_>,
    entry: std::collections::hash_map::Entry<'_, (u64, usize), LiveGpu>,
    ls: &LiveSlot,
    frame: u64,
    next_generation: &mut u64,
    stats: &mut CustomShaderStats,
) -> u64 {
    use std::collections::hash_map::Entry;
    match entry {
        Entry::Occupied(mut o) => {
            let g = o.get_mut();
            g.last_frame = frame;
            if g.version != ls.version {
                // Ranges are enough when this copy is no older than the last
                // whole replacement and the log still reaches back to it.
                let covered =
                    g.version >= ls.base && ls.ranges.front().is_some_and(|r| r.0 <= g.version + 1);
                let ranges: Option<Vec<(usize, usize)>> = covered.then(|| {
                    ls.ranges
                        .iter()
                        .filter(|r| r.0 > g.version)
                        .map(|r| (r.1, r.2))
                        .collect()
                });
                upload(
                    ctx,
                    &mut g.gpu,
                    &ls.data,
                    ranges.as_deref(),
                    next_generation,
                    stats,
                );
                g.version = ls.version;
            }
            g.gpu.as_ref().map_or(0, |g| g.generation)
        }
        Entry::Vacant(v) => {
            let mut gpu = None;
            let generation = upload(ctx, &mut gpu, &ls.data, None, next_generation, stats);
            v.insert(LiveGpu {
                gpu,
                version: ls.version,
                last_frame: frame,
            });
            generation
        }
    }
}

/// Put `data` on the GPU, into the existing `slot` where it fits; returns the
/// slot's binding generation. `ranges` are `(start, end)` float ranges to write
/// instead of the whole thing, honoured only when the length has not changed.
fn upload(
    ctx: &CustomCtx<'_>,
    slot: &mut Option<SlotGpu>,
    data: &[f32],
    ranges: Option<&[(usize, usize)]>,
    next_generation: &mut u64,
    stats: &mut CustomShaderStats,
) -> u64 {
    // An empty slot still has to bind something: one zero.
    let data: &[f32] = if data.is_empty() { &[0.0] } else { data };
    let len = data.len();
    let mut new_generation = || {
        *next_generation += 1;
        *next_generation
    };
    if ctx.storage_buffers {
        if let Some(g) = slot
            .as_mut()
            .filter(|g| matches!(g.res, SlotRes::Buffer(_)) && g.cap >= len)
        {
            let SlotRes::Buffer(buf) = &g.res else {
                unreachable!()
            };
            match ranges.filter(|_| g.len == len) {
                Some(ranges) => {
                    for &(s, e) in ranges {
                        let e = e.min(len);
                        if s < e {
                            ctx.queue.write_buffer(
                                buf,
                                (s * 4) as u64,
                                bytemuck::cast_slice(&data[s..e]),
                            );
                            stats.bytes_uploaded += ((e - s) * 4) as u64;
                        }
                    }
                }
                None => {
                    ctx.queue.write_buffer(buf, 0, bytemuck::cast_slice(data));
                    stats.bytes_uploaded += (len * 4) as u64;
                }
            }
            if g.len != len {
                g.len = len;
                g.generation = new_generation();
            }
            return g.generation;
        }
        let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threers user storage"),
            size: (len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
        stats.buffers_created += 1;
        stats.bytes_uploaded += (len * 4) as u64;
        let generation = new_generation();
        *slot = Some(SlotGpu {
            res: SlotRes::Buffer(buf),
            cap: len,
            dims: (0, 0),
            len,
            generation,
        });
        return generation;
    }

    // No storage buffers (WebGL2): an `Rgba32Float` texture, four floats a
    // texel, read through `s0`/`s1`/`s2`. `textureLoad` needs no filtering and
    // no sampler, so this works on the baseline float-texture support every
    // WebGL2 context has. Reused only at the same shape, since `s_len` reports
    // the texture's size.
    let dims = data_tex_dims(len);
    if let Some(g) = slot
        .as_mut()
        .filter(|g| matches!(g.res, SlotRes::Texture(..)) && g.dims == dims)
    {
        let SlotRes::Texture(tex, _) = &g.res else {
            unreachable!()
        };
        let row = dims.0 as usize * 4;
        match ranges.filter(|_| g.len == len) {
            Some(ranges) => {
                for &(s, e) in ranges {
                    let e = e.min(len);
                    if s < e {
                        let (r0, r1) = (s / row, (e - 1) / row);
                        write_rows(ctx, tex, dims.0, r0 as u32, r1 as u32 + 1, data, stats);
                    }
                }
            }
            None => write_rows(ctx, tex, dims.0, 0, dims.1, data, stats),
        }
        if g.len != len {
            g.len = len;
            g.generation = new_generation();
        }
        return g.generation;
    }
    let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("threers user data texture"),
        size: wgpu::Extent3d {
            width: dims.0,
            height: dims.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    write_rows(ctx, &tex, dims.0, 0, dims.1, data, stats);
    stats.textures_created += 1;
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    let generation = new_generation();
    *slot = Some(SlotGpu {
        res: SlotRes::Texture(tex, view),
        cap: (dims.0 * dims.1 * 4) as usize,
        dims,
        len,
        generation,
    });
    generation
}

/// Write texel rows `r0..r1` of a data texture from `data`, zero-padding past
/// its end.
fn write_rows(
    ctx: &CustomCtx<'_>,
    tex: &wgpu::Texture,
    width: u32,
    r0: u32,
    r1: u32,
    data: &[f32],
    stats: &mut CustomShaderStats,
) {
    let row = width as usize * 4;
    let (start, end) = (r0 as usize * row, r1 as usize * row);
    let padded;
    let bytes: &[f32] = if end <= data.len() {
        &data[start..end]
    } else {
        let mut p = vec![0.0f32; end - start];
        let have = data.len().saturating_sub(start);
        p[..have].copy_from_slice(&data[start..start + have]);
        padded = p;
        &padded
    };
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: r0, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(bytes),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 16),
            rows_per_image: Some(r1 - r0),
        },
        wgpu::Extent3d {
            width,
            height: r1 - r0,
            depth_or_array_layers: 1,
        },
    );
    stats.bytes_uploaded += (bytes.len() * 4) as u64;
}
