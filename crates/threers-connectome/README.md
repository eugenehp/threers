# threers-connectome

Both *Drosophila* connectomes as **one node per neuron**, drawn in the browser by
the threers wasm build. Click a node and its `skeleton.swc` is read off the tree
and drawn as the actual arbour.

**350,033 nodes at 60 fps**, out of the 350,825 neurons the two indexes list.

| | |
|---|---|
| FlyWire FAFB v783 | 139,245 nodes — adult female brain |
| Male CNS v1.0 | 210,788 nodes — adult male brain + optic lobes + VNC |

The data is the [`connectome-fs`](https://github.com/eugenehp/threers) text
trees: one directory per neuron, `grep` and `cd` as the query language. This
crate reads them; it does not need a database, an API or a credential.

## Run it

```bash
cargo run --release -p threers-connectome --bin connectome-build   # ~5 s
cargo run --release -p threers-connectome --bin connectome-serve   # http://127.0.0.1:8787/
```

Needs a WebGPU browser (Chrome/Edge 113+, Safari 26+) and a `web/pkg` build of
threers (`web/build.sh`, or the committed one).

The tree defaults to `/Volumes/C4TB/connectome-fs`; override with
`--tree <dir>` or `CONNECTOME_FS`. `connectome-serve` also takes `--port`,
`--data` and `--web`.

## No dependencies

Not a stylistic point — each one would have been carrying weight it did not
need. The trees are TSV and SWC, which is `split('\t')` and `split_whitespace`.
The tables are little-endian typed arrays, which is `to_le_bytes`. `meta.json`
is written once and read by one consumer. The two routes are a file read and a
parse, and at a dozen requests per page load a server that answers with
`Connection: close` and a thread per socket is not the bottleneck — the external
volume is.

This crate also does not depend on `threers`: the renderer runs in the browser
as wasm, and the Rust side is a reader and a file server.

## Layout

| | |
|---|---|
| `src/tree.rs` | `index.tsv` and `skeleton.swc` → neurons and line segments |
| `src/tables.rs` | neurons → the binaries the browser fetches |
| `src/http.rs` | a small HTTP/1.1 server and the TSV helpers the routes need |
| `src/bin/build.rs` | `connectome-build` — writes `data/` (14 MB, gitignored) |
| `src/bin/serve.rs` | `connectome-serve` — hosts the viewer, reads the tree |
| `../../web/connectome/` | `index.html` + `viewer.js` |

## The node positions

Neither release publishes one position column that covers every neuron, and the
two do not publish the same one:

| Tree | Placed by | Recovered from `skeleton.swc` | Neither |
|---|---|---|---|
| FlyWire FAFB v783 | 118,104 somas | 21,141 centroids | 3 |
| Male CNS v1.0 | 210,788 centroids | — | 789 |

FlyWire ships a soma per neuron but 21,144 of them are blank, so `Tree::read`
walks those neurons' own skeletons and takes the mean of every 16th node — a
2,000-node arbour still contributes 125 samples, which lands the mean well inside
a micron. The Male CNS index ships a skeleton centroid already; its 789 gaps have
no skeleton either, so they cannot be placed at all. Their ids go into
`data/meta.json` and the count is shown in the viewer's **Coverage** panel — the
header reads *350,033 of 350,825* rather than implying the whole index is on
screen.

Coordinates stay in nanometres through the pipeline — both trees' own unit — and
are converted once, in the viewer.

## Two things about the renderer that shaped the viewer

**A node is a cross of line segments, not a point.** threers rasterises
`THREE.Points` as one camera-facing sprite draw *per point* — correct for the
tens of points three.js's `PointsMaterial.size` convention was written for, and
350,000 draw calls a frame here. `LineSegments` is a single cached draw for the
whole buffer. So each node is three axis-aligned segments, six vertices that read
as a dot from any angle, and the whole cloud is nine `LineSegments` objects of
40,000 nodes each. Skeletons are line segments too, so they cost one draw apiece.

**Vertex colours are linear.** The fragment shader ends in `linear_to_srgb`, so
the palette is sRGB-decoded on the way into the colour attribute; skipping that
step washes every hue out by roughly a stop.

## Axes

Both releases are EM volumes with *y* increasing downward, so *y* flips. Male CNS
additionally runs brain → VNC along **+z** — checked against the index, where
optic-lobe neurons average z ≈ 265 µm and VNC neurons z ≈ 780 µm — so its z
becomes the vertical and the animal hangs head-up the way the figures draw it.
Each tree is centred on its own bounding box and the two are laid out side by
side, so neither coordinate frame leaks into the other.

## Colour

The categorical palette is a validated dark-mode eight-slot set on surface
`#1a1a19` (worst adjacent CVD ΔE 8.4, worst adjacent normal-vision ΔE 19.3, all
eight ≥ 3:1 against the surface). A point cloud puts every pair on screen at
once, which eight hues cannot clear on their own, so identity never rests on
colour: every legend row carries its name and count, hovering a node names its
exact category, and clicking a legend row isolates that category. Past eight
categories — super class has 35 across both trees — the tail folds into one grey
**other** row that lists its members on hover, rather than generating a ninth hue.

`synapse count` switches to a one-hue blue ramp on a log scale, running dark →
light because on a dark surface magnitude has to read as brightness.

## Controls

| | |
|---|---|
| drag / right-drag / wheel | orbit / pan / zoom |
| click a node | load its skeleton and open its record |
| `f` | frame the selected neuron |
| `r` | reset the view |
| legend row | isolate that category |
| search | substring match on cell type, cell class, side, or id |

The record panel's partner lists come from the neuron's own `connections.tsv`;
clicking a partner id jumps to it in the same tree.

## The 18-digit trap

FlyWire root ids are 18 digits and 52,342 of them collide with another id when
parsed as a double. Ids are `u64` in Rust, `BigUint64Array` in the tables and
strings everywhere in JS — never numbers — and the server accepts an id only as
up to 20 ASCII digits and resolves it as a path component, so the float-collision
trap in the tree's own README cannot bite here.
