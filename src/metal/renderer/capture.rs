use super::*;

impl MetalRenderer {
    /// three.js r165's `renderTransmissionPass`: the scene without its
    /// transmissive items drawn linear and untone-mapped into a half-float
    /// target the size of this pass, from the first view, copied into a
    /// mipmapped texture whose mips Metal generates; then the back faces of
    /// double-sided transmissive items drawn over it, reading that copy, and
    /// copied again. The colour pass binds the result (`capture_bound`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn capture_transmission(
        &mut self,
        pass: &PassAttachments,
        views: &[RenderView],
        frame_u: &FrameUniforms,
        items: &mut [DrawItem],
        scene: &Scene,
        slot: usize,
        stats: &mut MetalRenderStats,
    ) -> Result<(), MetalError> {
        // The size the views draw at: the pass, or the largest of their
        // viewports — three.js sizes the capture to the drawing buffer a view
        // renders into, and a side-by-side eye renders into half of one.
        let (w, h) = views.iter().fold((0u32, 0u32), |(w, h), v| match v.viewport {
            Some(vp) => (w.max(vp.width.round() as u32), h.max(vp.height.round() as u32)),
            None => (w.max(pass.width), h.max(pass.height)),
        });
        let (w, h) = (w.clamp(1, pass.width.max(1)), h.clamp(1, pass.height.max(1)));
        let slices = views.len().clamp(1, MAX_VIEWS) as u32;
        if self.transmission.as_ref().map(|c| (c.size, c.slices)) != Some(((w, h), slices)) {
            let levels = 32 - w.max(h).leading_zeros();
            let target = if slices > 1 {
                crate::metal::MetalRenderTarget::layered(&self.device, w, h, pixel_format::RGBA16_FLOAT, slices)?
            } else {
                crate::metal::MetalRenderTarget::new(&self.device, w, h, pixel_format::RGBA16_FLOAT, 1)?
            };
            let mipped = crate::metal::resources::new_texture_array_levels(
                &self.device,
                w,
                h,
                pixel_format::RGBA16_FLOAT,
                texture_usage::SHADER_READ | texture_usage::RENDER_TARGET,
                slices,
                levels,
            )?;
            self.transmission = Some(MetalCapture { target, mipped, size: (w, h), levels, slices });
        }
        let attachments = self.transmission.as_ref().expect("just made").target.attachments();
        // Every view captures the scene from where it stands, into its own
        // slice, over the whole capture — a view drawn into a viewport reads
        // its slice through its own projection, so where its rectangle sits
        // does not matter here.
        let capture_views: Vec<RenderView> = views[..slices as usize]
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let mut view = v.clone();
                view.color = NIL;
                view.depth = NIL;
                view.slice = i as u32;
                view.viewport = None;
                view
            })
            .collect();
        // Linear, untone-mapped, as three.js draws the transmission target.
        let mut fu = *frame_u;
        fu.view_info[0] = slices;
        fu.view_info[1] = 0;
        fu.view_info[2] = 0;
        fu.rect_counts[1] = 1;
        fu.viewport = [w as f32, h as f32, 1.0 / w as f32, 1.0 / h as f32];

        for it in items.iter_mut() {
            it.capture_skip = it.transmissive || it.screen_space;
        }
        self.capture_bound = None;
        // Its depth too, for `ss_depth_tex`.
        let store_depth = self.store_depth;
        self.store_depth |= items.iter().any(|i| i.screen_space);
        let drawn = unsafe { self.encode(&attachments, &capture_views, &fu, items, scene, slot, stats, None) };
        self.store_depth = store_depth;
        drawn?;
        self.copy_capture()?;

        if items.iter().any(|i| i.transmissive && i.material.side() == 2) {
            for it in items.iter_mut() {
                let back = it.transmissive && it.material.side() == 2;
                it.capture_skip = !back;
                it.side_override = back.then_some(1);
            }
            let cap = self.transmission.as_ref().expect("made above");
            self.capture_bound = Some(cap.mipped.id());
            fu.rect_counts[1] = cap.levels;
            let keep = Some((crate::renderer::viewport::ClearMode::None, None));
            unsafe { self.encode(&attachments, &capture_views, &fu, items, scene, slot, stats, keep) }?;
            self.copy_capture()?;
        }
        for it in items.iter_mut() {
            it.capture_skip = false;
            it.side_override = None;
        }
        self.capture_bound = self.transmission.as_ref().map(|c| c.mipped.id());
        Ok(())
    }

    /// The capture target's colour into level 0 of `mipped`, then its mips.
    pub(super) fn copy_capture(&self) -> Result<(), MetalError> {
        let Some(cap) = &self.transmission else { return Ok(()) };
        let _pool = AutoreleasePool::new();
        unsafe {
            let cmd = self.device.command_buffer();
            if cmd.is_null() {
                return Err(MetalError::NoCommandQueue);
            }
            let blit: Id = msg0(cmd, sel!("blitCommandEncoder"));
            let (w, h) = cap.size;
            for slice in 0..cap.slices as NSUInteger {
                let _: () = super::super::objc::msg9(
                    blit,
                    sel!("copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:"),
                    cap.target.color_texture(),
                    slice,
                    0usize,
                    MTLOrigin { x: 0, y: 0, z: 0 },
                    MTLSize { width: w as NSUInteger, height: h as NSUInteger, depth: 1 },
                    cap.mipped.id(),
                    slice,
                    0usize,
                    MTLOrigin { x: 0, y: 0, z: 0 },
                );
            }
            let _: () = msg1(blit, sel!("generateMipmapsForTexture:"), cap.mipped.id());
            let _: () = msg0(blit, sel!("endEncoding"));
            let _: () = msg0(cmd, sel!("commit"));
        }
        Ok(())
    }
}
