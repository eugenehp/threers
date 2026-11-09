use super::*;

impl MetalRenderer {
    /// The `MTLDepthStencilState` for one compare function and write flag,
    /// made on first use.
    pub(super) fn depth_state(&mut self, compare: NSUInteger, write: bool) -> Result<Id, MetalError> {
        if !self.depth_states.contains_key(&(compare, write)) {
            let state = depth_stencil_state(&self.device, compare, write)?;
            self.depth_states.insert((compare, write), state);
        }
        Ok(self.depth_states[&(compare, write)].id())
    }

    pub(super) fn pipeline(&mut self, key: PipelineKey) -> Result<Id, MetalError> {
        if !self.pipelines.contains_key(&key) {
            let state = match key.custom {
                0 => render_pipeline(&self.device, key)?,
                id => {
                    let library = self
                        .custom_libs
                        .get(&id)
                        .and_then(|l| l.as_ref())
                        .map(|l| l.id())
                        .ok_or_else(|| MetalError::Unsupported("custom shader not compiled".into()))?;
                    crate::metal::resources::custom_render_pipeline(&self.device, key, library)?
                }
            };
            self.pipelines.insert(key, state);
        }
        Ok(self.pipelines[&key].id())
    }

    /// A custom shader's id with its library compiled, or 0 to draw the
    /// fallback colour: the material's own [`msl`](crate::ShaderMaterial::msl)
    /// when it has one, otherwise its WGSL — fragment and any custom vertex
    /// stage — translated (`wgsl_material_to_msl`).
    pub(super) fn custom_shader(&mut self, material: &Material) -> u64 {
        let Material::Shader(m) = material else { return 0 };
        let msl: Arc<str> = match &m.msl {
            Some(msl) => msl.as_str().into(),
            None => {
                let key = custom_shader_id(&m.fragment) ^ m.vertex.is_some() as u64;
                let entry = self.translated.entry(key).or_insert_with(|| {
                    let result = crate::materials::shared_shader::wgsl_material_to_msl(
                        &m.fragment,
                        m.vertex.as_deref(),
                    );
                    match result {
                        Ok(msl) => Some(msl.into()),
                        Err(why) => {
                            eprintln!(
                                "threers: Metal backend: a ShaderMaterial draws its fallback colour: {why}"
                            );
                            None
                        }
                    }
                });
                match entry {
                    Some(msl) => msl.clone(),
                    None => return 0,
                }
            }
        };
        let id = custom_shader_id(&msl);
        if self.ensure_custom_library(id, &msl) {
            id
        } else {
            0
        }
    }

    /// Compile a custom shader's library the first time it is drawn. `false`
    /// when it does not compile — reported once, with the compiler's
    /// diagnostics — and the draw falls back to the unlit fallback colour.
    pub(super) fn ensure_custom_library(&mut self, id: u64, msl: &str) -> bool {
        if let Some(lib) = self.custom_libs.get(&id) {
            return lib.is_some();
        }
        let source = format!("{}\n{msl}", self.device.shader_source());
        let _pool = crate::metal::objc::AutoreleasePool::new();
        let lib = match unsafe { crate::metal::device::compile_library(self.device.device_id(), &source) } {
            Ok(lib) => Some(lib),
            Err(e) => {
                // The compiler names a line of the combined source, most of
                // which is the built-in library; quote the line itself.
                let text = e.to_string();
                let quoted = text
                    .split("program_source:")
                    .nth(1)
                    .and_then(|rest| rest.split(':').next())
                    .and_then(|n| n.parse::<usize>().ok())
                    .and_then(|n| source.lines().nth(n.saturating_sub(1)))
                    .map(|line| format!("\n    at: {}", line.trim()))
                    .unwrap_or_default();
                eprintln!(
                    "threers: Metal backend: a ShaderMaterial's MSL did not compile; drawing its fallback colour: {}{quoted}",
                    text.lines().next().unwrap_or(&text)
                );
                None
            }
        };
        let ok = lib.is_some();
        self.custom_libs.insert(id, lib);
        ok
    }
}
