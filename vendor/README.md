# vendor/

Local copies of upstream crates, used through the workspace's
`[patch.crates-io]`. Not published: a crate depending on threers from
crates.io builds against the released versions.

## wgpu-core, wgpu-hal (30.0.1)

As released, plus one change behind two features, neither on by default:

- **`msl-record`**: when `WGPU_MSL_RECORD` names a file, every WGSL module's
  entry points and the MSL naga writes for every Metal pipeline stage are
  appended to it, with what the backend needs beside the MSL (work group
  sizes, the bindings whose sizes a shader reads). The MSL is written for
  language version 3.1 (`WGPU_MSL_RECORD_VERSION` changes it), the newest
  iOS 17 compiles.
- **`msl-replay`**: no shader is parsed, validated or translated. wgpu-core
  hashes a WGSL source and treats the module as a passthrough module with
  the recorded entry points; the Metal backend looks each pipeline stage up
  in a recording installed with `wgpu_hal::recorded::install` and compiles
  its MSL. naga's WGSL front end, validator and MSL back end drop out of the
  binary (about 0.67 MB of an iPhone app). A stage that was not recorded
  fails to build, logging its key. The device makes neither of wgpu's own
  compute passes (indirect-call validation and timestamp normalisation),
  whose shaders are WGSL.

A stage's key hashes everything its MSL depends on: the WGSL source, the
entry point and stage, the vertex buffer layouts (the Metal backend pulls
vertices in the shader), the pipeline layout's binding slots and array
lengths, the primitive class, the pipeline constants and the runtime checks.

The change is in `wgpu-hal/src/recorded.rs` (new), `wgpu-hal/src/lib.rs`
(`ShaderInput::Recorded`, `NagaShader::source_hash`),
`wgpu-hal/src/metal/{device,mod}.rs`, one match arm in each other backend,
and `wgpu-core/src/device/resource.rs` (`create_recorded_shader_module`), with
the two internal passes switched off in `msl-replay`.

threers-ios's `precompiled-shaders` and `record-shaders` features turn them
on, and pack recordings small (`threers_ios::shaders`).

To move to a newer wgpu: copy the new release's `wgpu-core` and `wgpu-hal`
from the registry here and carry the change over; `diff -r` against the
registry's 30.0.1 shows it.
