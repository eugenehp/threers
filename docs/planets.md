# Planets, Earth and lens flare

Part of the [threers](../README.md) documentation.

# Earth, sun and lens flare

```bash
cargo run --release --example earth_sun_flare                  # 4K maps
cargo run --release --example earth_sun_flare -- --texture 8192
cargo run --release --example earth_sun_flare -- --frames 120 --video out/earth.mp4
```

A full PBR scene with no downloaded assets. Five equirectangular maps — albedo,
normal, roughness, night lights and clouds — are generated from **one shared
elevation field**, so the coastlines in the colour map, the relief in the normal
map, the shine on the oceans and the shoreline glow of the city lights all line
up. At `--texture 8192` that is five 33.6-Mtexel maps, built in under a second
across cores.

The noise is sampled in 3D on the sphere rather than in UV space, which is what
keeps an equirectangular map free of both a date-line seam and the smearing that
2D noise suffers at the poles. The lens flare and the atmosphere rim are
composited in screen space after the render — a flare is a property of the
camera, not the scene, and doing it there is what lets it be occluded as Earth
passes in front of the sun.

Rendering it needs mip-mapped textures to not shimmer, which the renderer now
builds for every 2D texture.

# Planets (`planet`)

`earth_sun_flare` builds all of that by hand. The `planet` feature is the same
machinery as an API, so a believable planet is a builder rather than a thousand
lines of example:

```rust
// Cargo.toml: threers = { version = "0.0.5", features = ["planet"] }
use threers::planet::{EarthTextures, Planet, Starfield};

let mut scene = threers::Scene::new();
// Real NASA imagery where it is on disk, generated maps for whatever is not.
let maps = EarthTextures::from_dir("web/assets/earth").load();
let earth = Planet::earth().maps(maps).add_to(&mut scene);
Starfield::procedural(2048).add_to(&mut scene);
```

`Planet::add_to` puts three objects in the scene and returns their ids: the
body, a cloud shell above it, and an atmosphere above that. The details that are
easy to get wrong are handled — colour maps tagged sRGB and data maps not,
longitude wrapping while latitude clamps, cloud cover living in the alpha
channel, the night map gating the emissive so city lights do not flood the day
side, and the atmosphere shaded as a volume the view ray passes through rather
than as a surface.

```bash
scripts/fetch-earth-textures.sh --png --sky   # NASA imagery, public domain, gitignored
cargo run --release --features planet --example realistic_earth
cargo run --release --features planet --example realistic_earth -- --relief 0.05
cargo run --release --features planet --example realistic_earth -- --moon
cargo run --release --features planet --example realistic_earth -- --frames 120
```

The browser port (`web/examples/earth.html`) builds the globe from **eight
sphere patches**, each streaming its own NASA tile at 1024, 2048 or 4096 px
depending on how close the camera is and whether the patch faces it. That is a
16384x8192 globe at the top level — a single-texture globe cannot exceed
8192x4096, because that is all WebGPU guarantees for one texture. Sliders drive
the sun's longitude and height, city-light brightness, atmosphere strength,
cloud cover and relief.

The Moon is there at its real distance — 60.3 Earth radii, on a Keplerian
ellipse with the actual eccentricity, inclination and sidereal period, tidally
locked, sharing Earth's clock. That makes it four pixels wide from anywhere that
frames Earth, so the scene opens beside it and flies in: a held shot by the
Moon, a turn to Earth while it is still a distant blue marble, then an eased
descent into orbit that hands over to `OrbitControls`. `replay intro` runs it
again; `?flyover=0` skips it.

Without the download the example still runs: every map falls back to a
procedural one built from a shared elevation field. With it, you get Blue Marble
surface colour, Black Marble city lights, a MODIS cloud composite, relief and
ocean gloss derived from GEBCO elevation, the CGI Moon Kit Moon (LRO colour +
LOLA relief), and the Deep Star Maps sky — the last read straight out of its
ZIP-compressed OpenEXR as half-float, because over half of that map sits below
1/100th of full scale and clipping it to 8 bits leaves a black sky.
