use super::*;

impl MetalRenderer {
    /// Draw every opaque triangle mesh into the shadow map, from the light.
    ///
    /// Transparent items are skipped: a depth buffer cannot represent partial
    /// occlusion, so a glass panel would cast the shadow of a solid one. That
    /// is three.js's behaviour too, and the alternative — a second pass with
    /// alpha testing — is a lot of machinery for a case that usually wants no
    /// shadow at all.
    pub(super) unsafe fn encode_shadow(
        &mut self,
        pass_u: &ShadowPassGpu,
        items: &[DrawItem],
        slot: usize,
        map: Id,
        slice: usize,
        radial: bool,
        frustum: &crate::math::Frustum,
    ) -> Result<(), MetalError> {
        let _pool = AutoreleasePool::new();
        let chosen = if radial {
            self.point_shadow_pipeline.as_ref()
        } else {
            self.shadow_pipeline.as_ref()
        };
        let Some(pipeline) = chosen else {
            return Ok(());
        };
        let pipeline = pipeline.id();
        let cmd = self.device.command_buffer();
        if cmd.is_null() {
            return Err(MetalError::NoCommandQueue);
        }
        let desc: Id = unsafe { msg0(class!("MTLRenderPassDescriptor"), sel!("renderPassDescriptor")) };
        unsafe {
            let depth: Id = msg0(desc, sel!("depthAttachment"));
            let _: () = msg1(depth, sel!("setTexture:"), map);
            let _: () = msg1(depth, sel!("setLoadAction:"), load_action::CLEAR);
            let _: () = msg1(depth, sel!("setStoreAction:"), store_action::STORE);
            let _: () = msg1(depth, sel!("setClearDepth:"), 1.0f64);
            // Which cube face, for a point light. Zero for the flat maps.
            let _: () = msg1(depth, sel!("setSlice:"), slice as NSUInteger);
        }
        let encoder: Id = unsafe { msg1(cmd, sel!("renderCommandEncoderWithDescriptor:"), desc) };
        if encoder.is_null() {
            return Err(MetalError::NoCommandQueue);
        }
        let cut_pipeline = self.cut_shadow_pipeline.as_ref().map(|p| p.id());
        let mut current = pipeline;
        let mut last_cull = NSUInteger::MAX;
        unsafe {
            let _: () = msg1(encoder, sel!("setRenderPipelineState:"), pipeline);
            // Counter-clockwise front faces, as the colour pass draws them, so
            // a caster's `side` can choose which faces its shadow records.
            let _: () = msg1(encoder, sel!("setFrontFacingWinding:"), winding::COUNTER_CLOCKWISE);
            // Forward-Z `Less`, write on: what a shadow map wants (made in
            // `with_device`, so the lookup cannot miss).
            let _: () = msg1(
                encoder,
                sel!("setDepthStencilState:"),
                self.depth_states[&(compare::LESS, true)].id(),
            );
            set_bytes(encoder, true, pass_u, VB_FRAME);
            if radial {
                // `fs_shadow_point` reads the light's position and radius from
                // the same uniform, and the fragment stage has its own binding
                // table — binding it for the vertex stage alone leaves the
                // fragment reading zeros, which writes a depth of 0 everywhere
                // and shadows nothing.
                set_bytes(encoder, false, pass_u, VB_FRAME);
            }
        }

        for item in items {
            // Per-object, matching the wgpu path: `Object3D::cast_shadow` is
            // off by default and a caster has to say so. Shadowing everything
            // opaque instead looks right in a test scene and wrong in a real
            // one, where the floor casts onto itself.
            // A transparent caster casts too: three.js's shadow map ignores
            // blending (only `alphaTest` cuts a shadow).
            if !item.cast_shadow {
                continue;
            }
            if item.primitive != primitive::TRIANGLE {
                continue;
            }
            if self.item_sphere(item).is_some_and(|s| !frustum.intersects_sphere(&s)) {
                continue;
            }
            let key = Arc::as_ptr(&item.geometry) as usize;
            let Some(geom) = self.geometries.get(&key) else {
                continue;
            };
            let (vertices, indices, index_count, vertex_count, skin) = (
                geom.vertices.id(),
                geom.indices.as_ref().map(|b| b.id()),
                geom.index_count,
                geom.vertex_count,
                geom.skin.as_ref().map(|b| b.id()),
            );
            let instances = item
                .instance_slot
                .and_then(|i| self.transients[slot].get(i))
                .map(|b| b.id())
                .unwrap_or(self.identity_instance.id());
            let bones = item
                .bone_slot
                .and_then(|i| self.transients[slot].get(i))
                .map(|b| b.id())
                .unwrap_or(self.identity_instance.id());
            let morph_u = item_morph_uniforms(item, geom.morph.as_ref());
            let morph_buffer = geom
                .morph
                .as_ref()
                .filter(|_| morph_u.info[0] > 0)
                .map(|m| m.buffer.id())
                .unwrap_or(self.identity_instance.id());
            let (start, end) = (item.range.0 as usize, item.range.1 as usize);
            // The caster's map, for a cut-out test: `has_map` as the colour
            // pass computes it (anything but the white stand-in).
            let (map_texture, map_sampler) = self.material_texture(&item.material);
            let mut draw_u = draw_uniforms(
                item,
                map_texture != self.white.id(),
                item.instance_slot.is_some(),
                &item.geometry,
            );
            // `maps.w == 2` is how `vs_shadow` is told to skin; the fragment
            // stage never sees this uniform, so the slot is free.
            if item.bone_slot.is_some() && skin.is_some() {
                draw_u.maps[3] = 2;
            }
            // three.js's shadow depth material takes the caster's `map`,
            // `alphaMap` and `alphaTest`: a cut-out caster runs
            // `fs_shadow_cut` (the point pass's fragment stage tests it
            // already).
            let alpha = self.slot_texture(item.material.texture_slots().alpha_map.as_ref(), true);
            draw_u.extra[0] = alpha.is_some() as u32;
            let cut = item.material.alpha_test() > 0.0 && (draw_u.flags[1] == 1 || alpha.is_some());
            let want = match (radial, cut, cut_pipeline) {
                (false, true, Some(p)) => p,
                _ => pipeline,
            };
            // three.js's `shadowSide`: a front-sided caster records its back
            // faces, a back-sided one its front faces, a double-sided one both.
            let cull = shadow_cull(item.material.side(), radial);
            unsafe {
                if want != current {
                    let _: () = msg1(encoder, sel!("setRenderPipelineState:"), want);
                    current = want;
                }
                if cull != last_cull {
                    let _: () = msg1(encoder, sel!("setCullMode:"), cull);
                    last_cull = cull;
                }
                if cut || radial {
                    let _: () = msg2(encoder, sel!("setFragmentTexture:atIndex:"), map_texture, 0usize);
                    let _: () = msg2(encoder, sel!("setFragmentSamplerState:atIndex:"), map_sampler, 0usize);
                    let (alpha_tex, alpha_smp) = alpha.unwrap_or((self.white.id(), map_sampler));
                    let _: () = msg2(encoder, sel!("setFragmentTexture:atIndex:"), alpha_tex, 1usize);
                    let _: () = msg2(encoder, sel!("setFragmentSamplerState:atIndex:"), alpha_smp, 1usize);
                    set_bytes(encoder, false, &draw_u, VB_DRAW);
                }
                // The caster's displacement map, so its shadow has the relief
                // its surface has (`displace` in the shadow vertex stages).
                let disp = self.slot_texture(item.material.texture_slots().displacement_map.as_ref(), true);
                let _: () = msg2(encoder, sel!("setVertexTexture:atIndex:"), disp.map_or(self.white.id(), |(t, _)| t), 0usize);
                let _: () = msg2(encoder, sel!("setVertexSamplerState:atIndex:"), disp.map_or(map_sampler, |(_, s)| s), 0usize);
                let _: () = msg3(encoder, sel!("setVertexBuffer:offset:atIndex:"), vertices, 0usize, 0usize);
                let _: () = msg3(encoder, sel!("setVertexBuffer:offset:atIndex:"), instances, 0usize, VB_INSTANCES);
                let _: () = msg3(encoder, sel!("setVertexBuffer:offset:atIndex:"), skin.unwrap_or(vertices), 0usize, VB_SKIN);
                let _: () = msg3(encoder, sel!("setVertexBuffer:offset:atIndex:"), bones, 0usize, VB_BONES);
                let _: () = msg3(encoder, sel!("setVertexBuffer:offset:atIndex:"), morph_buffer, 0usize, VB_MORPH_DATA);
                set_bytes(encoder, true, &morph_u, VB_MORPH);
                set_bytes(encoder, true, &draw_u, VB_DRAW);
                let n = item.instance_count.max(1) as NSUInteger;
                match indices {
                    Some(ibuf) if index_count > 0 => {
                        let count = end.min(index_count).saturating_sub(start);
                        if count > 0 {
                            let _: () = msg6(
                                encoder,
                                sel!(
                                    "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:"
                                ),
                                primitive::TRIANGLE,
                                count as NSUInteger,
                                index_type::UINT32,
                                ibuf,
                                start * std::mem::size_of::<u32>(),
                                n,
                            );
                        }
                    }
                    _ => {
                        let count = end.min(vertex_count).saturating_sub(start);
                        if count > 0 {
                            let _: () = msg4(
                                encoder,
                                sel!("drawPrimitives:vertexStart:vertexCount:instanceCount:"),
                                primitive::TRIANGLE,
                                start,
                                count as NSUInteger,
                                n,
                            );
                        }
                    }
                }
            }
        }
        unsafe {
            let _: () = msg0(encoder, sel!("endEncoding"));
            let _: () = msg0(cmd, sel!("commit"));
        }
        Ok(())
    }

    /// Size the shadow arrays for this frame's casters, with their sampler
    /// and pipelines: a layer per directional and spot light, six per point
    /// light, at the size the lights of each kind ask for
    /// (`ShadowSettings::map_size`, the largest), shrunk when several share
    /// a kind (`shadow_layer_size`). Reallocated only to grow or resize.
    ///
    /// `map_size` used to be a field the wgpu path read and discarded; the
    /// failure mode there is worse than not offering it, since the caller sets
    /// 4096, sees no error and gets whatever was allocated. It is read here.
    pub(super) fn ensure_shadow_arrays(&mut self, casters: &Casters) -> Result<(), MetalError> {
        use crate::lights::shadow::shadow_layer_size;
        let count = |want: fn(&Caster) -> bool| casters.iter().filter(|c| want(c)).map(Caster::layers).sum::<usize>();
        let nd = count(|c| matches!(c, Caster::Dir { .. }));
        let ns = count(|c| matches!(c, Caster::Spot { .. }));
        let np = count(|c| matches!(c, Caster::Point { .. })) / 6;
        let max_size = |want: fn(&Caster) -> bool| {
            casters.iter().filter(|c| want(c)).map(|c| c.settings().map_size).max().unwrap_or(0)
        };
        if nd + ns + np == 0 {
            return Ok(());
        }
        let array = |device: &MetalDevice, size: u32, layers: usize| {
            crate::metal::resources::new_texture_array_levels(
                device,
                size,
                size,
                pixel_format::DEPTH32_FLOAT,
                texture_usage::RENDER_TARGET | texture_usage::SHADER_READ,
                layers as u32,
                1,
            )
        };
        if nd > 0 {
            let size = shadow_layer_size(max_size(|c| matches!(c, Caster::Dir { .. })), nd);
            if self.shadow_map.is_none() || self.shadow_size != size || self.shadow_layers[0] < nd {
                self.shadow_map = Some(array(&self.device, size, nd)?);
                self.shadow_size = size;
                self.shadow_layers[0] = nd;
            }
        }
        if ns > 0 {
            let size = shadow_layer_size(max_size(|c| matches!(c, Caster::Spot { .. })), ns);
            if self.spot_shadow_map.is_none() || self.spot_shadow_size != size || self.shadow_layers[1] < ns {
                self.spot_shadow_map = Some(array(&self.device, size, ns)?);
                self.spot_shadow_size = size;
                self.shadow_layers[1] = ns;
            }
        }
        if np > 0 {
            // 512 a face, matching the wgpu path. Six faces make a cube
            // expensive to grow — 1024 is four times the memory and four times
            // the fill — and a point light's shadow is usually small in frame.
            if self.point_shadow_map.is_none() || self.shadow_layers[2] < 6 * np {
                self.point_shadow_map = Some(array(&self.device, POINT_SHADOW_SIZE, 6 * np)?);
                self.shadow_layers[2] = 6 * np;
            }
            if self.point_shadow_pipeline.is_none() {
                self.point_shadow_pipeline =
                    Some(crate::metal::resources::point_shadow_pipeline(&self.device)?);
            }
        }
        self.ensure_shadow_support()
    }

    /// The pipeline and comparison sampler every shadow pass shares.
    pub(super) fn ensure_shadow_support(&mut self) -> Result<(), MetalError> {
        if self.shadow_sampler.is_none() {
            self.shadow_sampler = Some(crate::metal::resources::shadow_sampler(&self.device)?);
        }
        if self.shadow_pipeline.is_none() {
            self.shadow_pipeline = Some(crate::metal::resources::shadow_pipeline(&self.device)?);
        }
        if self.cut_shadow_pipeline.is_none() {
            self.cut_shadow_pipeline =
                Some(crate::metal::resources::cut_shadow_pipeline(&self.device)?);
        }
        Ok(())
    }
}
