# threers-audio

Audio for [threers](https://github.com/eugenehp/threers) scenes, modeled on
three.js: `Audio` (a buffer and how to play it), `AudioListener`,
`PositionalAudio` with its distance models and cone, `SpatialMix` (gain and
pan for one listener) and `AudioAnalyser`. It is the data and the maths;
playback is left to whatever the host has (cpal natively, Web Audio in a
browser, `threers-ecs`'s mixer).

`threers` re-exports it as `threers::audio` (the `audio` feature, on by
default), where a `PositionalAudio` is a scene-graph node.
