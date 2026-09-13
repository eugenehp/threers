# Getting started (native)

Part of the [threers](../README.md) documentation.

## Prelude

`use threers::prelude::*;` brings in the scene graph, the common geometries and
materials, lights, cameras, textures, the renderers, the vector maths, and the
`Camera` trait — which has to be in scope for `camera.view_matrix()` to resolve,
and is the sort of thing a prelude is for.

```rust
use threers::prelude::*;

let mut scene = Scene::new();
scene.add(Object3D::mesh(Mesh::new(
    BoxGeometry::new(1.0, 1.0, 1.0),
    Material::Standard(StandardMaterial::new(Color::from_hex(0xff8844))),
)));
scene.add_light(DirectionalLight::new(Color::WHITE, 3.0));

let mut camera = PerspectiveCamera::new(50.0, 16.0 / 9.0, 0.1, 100.0);
camera.position = Vector3::new(3.0, 2.0, 4.0);
camera.look_at(Vector3::ZERO);
```

It is a *curated* subset, not everything the crate exports: a glob import claims
every name in it, so the maths shapes (`Plane`, `Sphere`, `Triangle`, `Ray`,
`Box3`, …), the curve and path types (`Path`, `Shape`, …), and the subsystems you
reach for deliberately are left out of the root. Nested modules keep those
reachable without polluting the glob:

```rust
use threers::prelude::*;
use threers::prelude::controls::*;
use threers::prelude::animation::*;
```

All of them also remain at the crate root — `threers::Path`,
`threers::loaders::GltfLoader` — one `use` away. Feature-gated nested modules
(`prelude::csg`, `prelude::openscad`, `prelude::nurbs`, `prelude::raytrace`,
`prelude::captions`) appear when the matching Cargo feature is on. With the
`metal` feature on, the Metal renderers join the root prelude, because they are
renderers like the others.

Meta feature bundles (`cad`, `media`, `gi`, `apple`, `full`) turn on related leaf
flags in one go — see the feature table in the crate docs.

# Native (desktop)

Requires Rust 1.87+ (wgpu 30's minimum), a working wgpu backend (Metal/Vulkan/DX12), and dev dependencies from `Cargo.toml`.

```bash
cargo run --example cube          # spinning PBR cube
cargo run --example scene_graph   # hierarchy + lights
cargo run --example controls_orbit
cargo run --example shader_material
cargo run --release --features mesh-bvh --example simcity   # procedural city, running day/night clock
#   ^ mesh-bvh turns on the raycast light bake; --no-bake skips it
cargo run --release --example ocean_water   # spectral ocean: waves, foam, buoyancy
```

The generator lives in `examples/simcity/` as a module tree — fifteen files
rather than the six-thousand-line single include it started as, which had
become the main obstacle to changing anything in it. `cargo test --test
simcity` guards it: determinism, every layout building something, every park
kind building something at every size, no parcel cut into a sliver, nobody
leaving their stretch of road, corners swept rather than snapped, wildlife
staying where it was put, and traffic that does not quietly decay. Most of
those exist because the bug shipped first.

`simcity` generates a whole city from a seed, in one of seven street layouts
(`--layout manhattan|boulevard|oldtown|waterfront|parkway|gardencity|greenbelt`,
or `l` in the window). The layout is not a skin: it sets the grid metrics,
where the height piles up, how finely blocks are platted, where the water goes
— a river down one column, a bay along one edge, or nothing — and where the
green goes. On top of that come blocks platted into parcels by recursive
subdivision, towers that step back asymmetrically, nine kinds of designed park,
street trees in six species, lamps, traffic signals that actually cycle,
barges, woodland past the city limits, and cars, buses, lorries and several
hundred people who all stay on their own side of the water when their street
has no bridge. It arrives as ~40 merged meshes
rather than the ~7000 objects it looks like, so the whole thing draws in about
forty calls at a locked 60 fps; each building carries its own facade UVs, so
one 128×128 window tile serves towers of every size.

Parks are designed, not scattered. A green block picks a *kind* — formal
square, garden, pond, sports ground, hard court, playground, plaza, allotment,
meadow, grove, dog run — and is built to that kind's own rules: a square is symmetrical about a
fountain or a statue with one species of tree on a regular spacing, a sports
ground is mown in bands with markings, goals and floodlights and nothing in the
middle, an allotment is a patchwork of raised beds where no two neighbouring
plots carry the same crop. The kind follows from the block's size and how built
up its surroundings are, so downtown gets hard landscape and the outskirts get
allotments and meadows.

The fittings do as much work as the layouts. A square carries a fountain, a
statue or a bandstand — an octagon with a point on it is a bandstand and
nothing else, which makes it legible from the air in a way a lawn is not — and
is enclosed by cast-iron railings with a gate, because that enclosure is what
distinguishes a square from a patch of grass. A garden gets a kiosk with tables
and parasols outside it. Paths are lined with the small stuff: litter bins,
drinking fountains, fingerposts, bicycle hoops, notice boards. A sports ground
is a marked pitch or, on a tighter block, a fenced hard court with a net or a
pair of backboards. A dog run is gravel, agility equipment, dogs off the lead,
and a double gate — every dog run has an airlock, two gates with a pen between
them, and nothing else in a park is built that way.

Fences are rails, not panels. Drawn as a solid sheet — which is the obvious
thing, and what the first pass did — a chain-link fence reads as a compound
wall at any distance. Three thin rails and a sparse set of uprights cost about
the same and read correctly, and the gap between them is what says *mesh*.

A pond needs more room than its water. The walk round it and the benches facing
it reach five metres past the shore, so on a small block the walk lands on the
pavement and the benches stand in the road; below twenty-five metres a block
builds a garden instead. `parks_stay_inside_their_block` is what found that,
and it checks every kind at every size.

Where the green goes is a property of the layout, not of the dice. A park
pattern is one of scatter, one central square, a continuous belt at a fixed
radius, or four wedges running out from the middle; `greenbelt` and
`gardencity` exist to use the last two. Any designated green area is *graded*
from its middle outwards — water and rough grass at the heart, gardens and
squares where it meets the street — which is what makes eleven adjacent green
blocks read as one large park instead of as eleven small ones.

A pond needed a hole cutting in the ground. A block is a solid box from the
terrain up to the kerb, so anything sunk into it is simply hidden under the top
face; the surface is emitted as a ring of triangles round the pond instead,
with the square's four corner angles forced into the sample set so no corner
gets sliced off.

The horizon is graded rather than a wall of white. Rayleigh drives it, and the
horizon is where a wide shot spends most of its sky — measured looking along
the ground at noon, 0.33 clipped *86% of the sky* to flat white, 0.10 clipped
14%, 0.05 none. The old value was tuned on a steep aerial where the camera
barely sees the horizon at all, and it was hopeless the moment one did. Raising
Mie, which had fixed the same complaint at a steep angle, makes it worse here:
86% to 95%. It is now 27%, and what is left is the bit of sky right above the
horizon, which is legitimately bright.

The view runs a lot further. Haze used to close at 4.6 half-widths, which cut
the world off just past the city; it closes at 7.6 now, with the ground plate
and the camera's far plane grown to match. What fills the extra distance is
countryside at a coarsening grain: inside three half-widths a parcel is a
field with tramlines, hedges, woods and lakes, and beyond it a parcel is a
colour and a hedgerow and nothing else. At that range a tree is a pixel, and a
thousand pixels are not worth a hundred thousand triangles.

Four kinds of vessel, not one: the flat-decked barge that was already here, a
tug that is mostly wheelhouse with old tyres down its flanks, a container ship
with its boxes stacked in bays of uneven height and its island right aft, and a
yacht whose sail is the entire silhouette.

The city sits in something now. Everything past the last road was a flat green
plane running to the fog — fine at street level, and the first thing you notice
from the air, where it is half the frame. The ring around the built area is
laid out as parcels on a coarse grid: hedged fields in eight crop colours with
tramlines through them, stands of woodland, lakes with reed-fringed banks, and
rough pasture, with a barn or a farmhouse on one field in five.

Two things make it read as farmland rather than as a chequerboard. The colours
differ parcel to parcel, because a uniform green is a lawn however large it is.
And the parcels are inset independently on each of their four sides, with an
occasional deep bite out of one — a uniform inset leaves every field the same
size and the ring becomes a visible lattice, which is the one thing farmland
never looks like. Anything left too small to hedge is dropped to rough ground,
so the grid has gaps in it as well as variety.

All of it is drawn at the far level of detail. At that range a tree is a few
pixels, and paying near-detail prices for a thousand of them would cost more
than the city does.

Stars are round. They were square quads, and at two pixels across a square is
exactly what the eye reads as a cube. Each one is a fan now, with a solid core
and a ring falling off around it — and the falloff is *per-vertex colour*, not
alpha, because the material is opaque: the rim is set to the sky's own colour
so it fades into the background instead of painting a dark halo round every
star. The first attempt at this used a single fan from a bright point straight
to the sky colour, which is a gradient with no star in the middle of it; at
that size it averaged out to the background and the sky went empty.

Tall roofs get helipads: the deck, the touchdown circle, the H, green edge
lights after dark, and a windsock. `tall_roofs_get_helipads` checks the
geometry rather than a camera, because the placement is a chain of conditions —
top tier, tall enough, roof wide enough — and the top tier is set back, so
thresholds that read as reasonable can leave nothing qualifying at all. That
looks identical to a feature nobody wrote.

There is air traffic. Airliners cross high overhead, helicopters orbit at
working height, and quadcopters buzz about low over the streets — all of them
riding the same loop machinery as the pigeons, which is what makes them nearly
free. A plane's loop is simply two kilometres across and *centred so the city
sits on it*: over three hundred metres of town, that arc reads as a straight
line, which is what an airliner does, and it costs nothing extra to say so. A
helicopter is drawn with a rotor disc rather than blades, because from below
and from any distance the disc is the entire silhouette; two blades over the
top of it stop it reading as a propeller stopped for a photograph. Port and
starboard navigation lights are red and green on all three, as they should be.

Delivery trucks join the road fleets, taking them to fourteen: a box on a cab,
deliberately taller and squarer than the van's, with a roller shutter and a
folded tail lift at the back and a band of livery down each flank — a delivery
truck is a moving billboard and that is most of how it reads.

Both axes drive on the same side now. The lane offset was the same expression
on each — `center - dir * half * 0.44` — and that cannot be right for both,
because the sign flips between them: `Quaternion::from_axis_angle(UP, yaw)`
maps a vehicle's local `+X` to `(cos yaw, 0, -sin yaw)`, so travelling `+X` puts
its right at `-Z` while travelling `+Z` puts it at `+X`. Roads running along Z
had been driving on the wrong side. Nothing crashed, because each road is
internally consistent; it shows the moment a vehicle turns from one onto the
other and swaps sides doing it. (My first attempt flipped the wrong axis —
worked out from a cross product rather than from the engine's own rotation, and
the two disagree.)

Fixing it exposed a second stale count. `traffic_does_not_gridlock` measured
mean speed over `fleets.iter().take(4)` — a leftover from when there were four
road fleets. With thirteen it was sampling cars, hatchbacks, sports cars and
taxis, the four fastest, and ignoring the vans, buses and lorries they queue
behind: a biased sample and a far more volatile one. Exactly the mistake that
had been sitting in `follow_pass`.

Some plots are building sites. Every city here was finished and perfect, which
is most of why it read as a model rather than a place — nothing in progress,
nothing half-done. One plot in twenty gets a hoarding of painted panels with a
gate in it, a concrete frame going up with the top slab only part poured and
rebar standing out of the columns, a portacabin, skips, heaps of aggregate, and
a lattice tower crane with its jib on a random bearing and the hoist block
hanging somewhere down the rope. The crane is the point: it is visible from
anywhere in the city, and it is the one piece of a skyline that says the
skyline is still changing.

There is a night sky. There was none — after dark the Preetham dome goes to a
flat grey wash and that was the whole of it, brighter than any star drawn in
front of it. The dome is an analytic *daylight* model; it has no night in it.
So below the horizon it is retired, and what shows is the background — dark
navy with the city's own glow mixed in — with fourteen hundred stars on a shell
inside the camera's far plane, and a moon with a halo. The stars are
deliberately uneven: a third of them are pulled toward a band across the sky,
and magnitudes run mostly faint with a few bright, because a uniform scatter of
identical points reads as noise rather than as a sky.

The moving population is drawn larger than life, and the default camera has
come in to meet it. Metres are metres everywhere else here and for the
buildings that is right; for the things that move it was a mistake. Do the
arithmetic on the aerial view this example shipped with — camera at 1.6 times
the city's half-width, 36 degrees — and a person is about *two pixels*. Every
session spent on crowds, skin tones, seated figures, squirrels, leaf litter and
street furniture was spent below the resolution of the shot the example takes.
Every city game oversizes its traffic and its crowds, and not from sloppiness.
People are drawn at 1.45, wildlife at 1.6, vehicles at 1.22 — less, because
they queue against each other and share lanes, so `Vehicle::length` is scaled
with them and the following distances stay honest. The camera came in by about
a quarter, not by half: closer than that and the frame holds four blocks, which
trades the shape of the city for the detail in it.

The clock means something now. It was a slider with nothing behind it — the
same traffic at three in the morning as at nine, the same crowds — while the
machinery had existed for a while, since the animals already keep hours. The
population out on the streets follows two peaks, an hour or so after sunrise
and again before sunset, with a long trough overnight: 100% at rush hour, 75%
at midday, 20% at three in the morning. Pedestrians empty harder than traffic,
because there is always some traffic and at four in the morning there is nobody
walking. Who is absent is keyed on each mover's own seed rather than its index,
so the *same* vehicles are missing from frame to frame instead of flickering.

Finding that number honest took a second fix: the diagnostic that reports how
many movers are drawn ran *before* the clock was applied, so it printed the
same count at every hour of the day.

Nothing stands inside anything else. Every object in this city is placed by
scatter — a tree at a random point in the setback, a bin at a random point in
the pavement, a bench at a random point on a path — and each of those is
reasonable on its own while none of them knows about the others. So a lamp
column grew through a tree, a bin stood inside a bench, and a parked car
occupied the same two metres as a fire hydrant. At street level that is the
difference between a city and a pile of props.

There is now a claim. Before anything is drawn it asks for the ground it
needs, from a uniform grid of buckets holding rectangles, and it is only drawn
if that ground is free. Buildings take their plots first, so nothing scatters
into a wall. On a nine-block Manhattan that is 1074 footprints claimed and
**343 placements refused** — very nearly a quarter of everything scattered had
been overlapping something. The grid rides on `Batches` for the same reason the
bench positions do: every placement function already has those in hand, so
nothing needed a new parameter threaded through a dozen call sites.

Wheels are round. They had been `add_box` — axis-aligned boxes, and not even
square ones — which is invisible at a hundred metres and the first thing the
eye finds at ten, because a car is the one object in this city whose shape
everybody already knows. `add_limb` builds a cylinder between two points, so
putting the axis across the car costs nothing, and a hub set proud of the
tyre's outer face is what stops the result reading as a black disc.
`wheels_are_round` checks the geometry rather than a photograph: a box puts its
vertices at two heights, a cylinder puts them all the way round.

Thirteen kinds of road vehicle. Beyond the nine already there: an estate with
the roof carried back to the tail and bars on it, a squared-off 4x4 that sits a
head taller than everything else, an ambulance in green-and-yellow battenburg,
and a fire engine with lockers down its flanks and a ladder on the roof. The
last two carry beacons like the police car — the lamp is selected by a bitmask
of fleets now rather than a single index, and the mesh carries lenses at two
heights so one mesh serves a saloon roof and a box body alike.

A fleet costs draws for its animation, not for its paint. `InstancedMesh` now
carries a per-instance tint alongside its transforms — twenty floats an
instance instead of sixteen, multiplied into vertex colour in the shader — and
that is the difference between a palette being free and a palette being a draw
call each. It had been the latter: `spawn_fleet` built one mesh per (colour,
pose), so nine vehicle types with their palettes cost thirty-seven draws of
bodywork on their own, and the pedestrians' three skin tones across two poses
cost six more. Collapsing all of it took the city from 147 draws to 96 — and
this example's whole design premise is that draw calls are the budget, so
watching that number nearly triple over a session and doing nothing was the
real mistake. `a_fleets_draws_do_not_depend_on_its_palette` pins the
relationship rather than the number.

Night is five kinds of light, not one, and getting there meant finding a bug
that had been hiding in plain sight. `apply_sky` set the window emissive
intensity on `Material::Standard` — and the facades are `Material::Physical`,
because a curtain wall is glazing under a clearcoat lobe. The match arm never
fired, so every lit window in the city had been burning at whatever the
material default happened to be, identically, for the whole of the example's
life. With both arms handled the old constant of 1.6 blew the entire skyline
out, which is how you can tell it had never been applied.

Each facade idiom now has its own colour after dark, its own gain, and its own
warm bias — how far its windows lean toward incandescent. An office floor is
fluorescent and slightly green-blue with the odd late desk lamp; a brick
walk-up is table lamps almost all the way down. Nothing else distinguishes
those buildings at night, because at night the windows are all you can see.

The shopfronts were the other half of it. Ground-floor glazing was a box
wrapped round the whole plinth, a metre and a half tall, one cream colour on
every building in the city — and with bloom over it that ribbon was most of
what a night render was made of. It is a band of individual units now, one
every few metres, each its own colour and some of them shuttered: a butcher's
fluorescent white next to a bar's amber next to a dark one.

Cars are parked at the kerb. Static geometry, not instances: a parked car has
no simulation to do, so it merges into a batch and costs no draw call and no
per-frame transform. `MeshBuilder::append_at` stamps a copy of a body at a
position and turns it, repainting only the vertices that are *white* — the
convention the vehicle bodies already follow, so a red car does not get red
windows.

They only park where there is room for them. A side street's carriageway is
about nine metres, a running lane sits 0.44 of the half-width out from the
centre, and a parked car is 1.8 wide — the two do not both fit, and the first
pass had cars standing in the traffic lane. Below six metres of half-width
nothing parks, which in a real city is what the yellow lines are for. Nor do
they park across a junction or its crossings.

They draw from a random stream of their own, and that is not fussiness.
Decoration taking from the city's `rng` shifts every later decision — block
subdivision, building heights, where the traffic starts — so adding parked cars
silently rearranged the entire city, and one of the cities that came out of the
reshuffle gridlocked. The gridlock test caught it. Worth recording that the
fix removed *that* city rather than the fragility: the traffic model still
decays on some arrangements, and the test is the only thing watching.

Six building masses, not three. A box, an octagonal prism and a round tower
were the whole vocabulary, and anything on a plot that was not square-ish was
forced to a plain box however tall it got. There is now a hexagonal prism, a
square frustum — the obelisk profile — and a cruciform plan, a square core with
a bay pushed out on each face, which is what a great many pre-war towers
actually are and which is the one of the six that *wants* a long plot rather
than being ruled out by one. `every_building_form_gets_used` guards the trap
the facade styles fell into: a variant added to the end of an enum with no
branch selecting it is present, compiled and never once built.

Trees carry fruit and drop leaves. The litter is the cheaper half and does far
more work: a scatter of flat coloured quads under every broadleaf turns a lawn
with trees on it into a lawn *under* trees. Roughly one in five is in fruit,
hung round the outside of the crown where the light is and where it would be
visible, with windfalls beneath. Both only appear on the near level of detail,
which is where half these trees are anyway.

Flowers are a container, a band of green and a band of colour standing on it —
which is all one reads as at any distance — and that one shape does window
boxes under first-floor sills, hanging baskets on the lamp columns, beds
against a house wall and borders along a path. Houses have front gardens now as
well as back ones: a dwarf wall along the pavement with a gap for the gate, a
planted bed against the house and a clipped shrub or two, because a path to the
door and nothing else is a verge rather than a garden.

Three more insects. Dragonflies dart over the ponds and the river — a fast rate
on a very bent path, which is what separates one from a butterfly at the same
size — and hold their wings out flat at rest, which nothing else here does and
which is the entire recognition cue at that scale. Flies hang over the refuse
in knots, going nowhere, which is the point of them rather than any individual.
Butterflies were already there.

There is a railway, and it is the one piece of infrastructure here that
ignores the street grid — which is the point of it, since everything else in
this city is at ground level and parallel to something. A viaduct on arched
piers runs the full width above a street, with a station on the deck, and
tunnel portals in embankments at both ends. Where it crosses the river it takes
one clear span on a Warren truss: a pier in a navigable channel is wrong on its
own terms and, here, something for a barge to sail through.

The train is an ordinary mover on a lane of its own at deck height, and gets
that for free — the lane machinery already handles a long vehicle on a bounded
run, and a train is nothing but a very long vehicle that never turns. Its wrap
point sits *inside a tunnel*, so the one place the simulation teleports is the
one place nobody can see. It carries no lane id, which is what keeps it out of
the road queueing pass; without that it would brake for a bus passing
underneath it. Down at street level there are entrances to whatever runs below
— a stair, railings, and a roundel on a post, which is the one part of a metro
anybody recognises from across a street.

Boats work the reach between two bridges. A barge stands two and a half metres
out of the water and a bridge deck sits at kerb height a metre above it, so
nothing on this river can clear anything — and the boats had been sailing
straight through every crossing in the city. Rather than raise the roads, they
are now given a stretch of river between two bridges to work, which is what
craft that cannot pass under a low bridge actually do.

Five facade idioms, not three. Three was enough to tell a tower from a walk-up
and not enough to tell one tower from another: every building over thirteen
storeys had the same window in it. The new pair are ribbon glazing — horizontal
bands of window in deep spandrels, the only idiom here whose windows are wider
than they are tall — and a dark bronze curtain wall, which is what the eighties
put up next to the blue-green glass of the sixties.

Traffic is nine kinds, not four: car, hatchback, sports, taxi, pickup, van,
bus, lorry and a marked police car, weighted the way real traffic is — mostly
ordinary cars, a fifth commercial, the rarities rare. The paint list is
weighted too, white and silver and grey and black with a few strong colours in
it, because spreading evenly across a palette makes a street look like a bag of
sweets.

Lamps read the simulation rather than being decoration. Brake lights come on
when the traffic pass has pulled a vehicle below walking pace — queueing at a
red, or behind a bus — and indicators blink only while a corner is actually in
progress, both from state that already existed. The police car's light bar
alternates blue and red rather than blinking one colour, which is its whole
visual signature and the only thing in the city that does it. Brakes,
indicators and beacons stay lit in daylight; head and tail lamps and the beam
on the road do not.

Two things were wrong before that. Lamps were placed by cloning the car fleet
through `write_instances`, which put every lamp on every car unconditionally
and on *nothing else at all* — vans, buses and lorries drove around at night
with no lights on them. And a lamp mesh laid out for a four-metre car cannot
serve an eleven-metre bus, so the instance transform now stretches it along Z
by the vehicle's own length: one mesh, lights on the ends of whatever it is
attached to.

Houses vary. Two storeys was the whole of "house", and a pitched lid went on
everything; now it is one to three storeys under one of five roofs — hipped,
mansard with dormers in it, cross-gable, flat with a parapet, or plain gable —
because the shape of the roof is most of what tells one house from the next at
any distance. On the front: a porch on posts, sometimes a bay window, and an
attached garage with a door and a drive out to the kerb. Behind: a garden,
fenced with boards rather than a panel — a solid box reads as a wall and every
garden in the city would be a compound — with a patio against the house and one
of a shed, a washing line, a trampoline, a paddling pool, vegetable beds or a
tree and a table. Everything behind the houses used to be bare pavement, which
is the one part of a suburb nobody builds and everybody notices.

Refuse waits for collection against the walls, and there are rats on it after
dark. Both are small; the rat is the only animal here defined by what it is
*not* — a squirrel with the tail taken off it, long and bare and dragging, and
the whole animal dropped closer to the ground.

The lights glow, and fixing that meant fixing the crate. `bloom.rs`, `ssao.rs`
and a whole `EffectComposer` pass stack have always been there, and *no example
in the repository used any of it* — which turned out to be because the path did
not work. Bloom builds an sRGB view of its source to get the decode for free,
and the resolve target declared no view formats at all, so bloom panicked at
view creation on any target that was not already sRGB — that is, the default
one, and the one this example uses. And screen-space occlusion was handed the
renderer's own depth buffer when the non-MSAA path had written the *target's*,
so it read an untouched buffer and produced a uniformly white factor, which
looks exactly like "SSAO is on and subtle". Both are fixed.

Bloom is on by default now and it is what makes neon and a lamp lens read as
emitting rather than as bright paint. Its threshold follows the clock: a sign
at midnight should bleed, sunlit glass at noon should not, and one threshold
low enough for the first puts a haze round every tower in the second. Occlusion
is still off by default — with both bugs fixed it resolves to a constant
whatever radius it is given, so something further down is still wrong, and
shipping a flag that does nothing would be worse than saying so. `--ssao 1.0`
turns it on for anyone chasing it.

Some people are not going anywhere. Every bench, step and kerb in the city used
to be empty and every pedestrian was walking somewhere at a constant speed,
which reads less as a city than as a treadmill. Benches now record where they
are — `add_bench` already had the batches in hand, so they collect into
`Batches::seats` rather than being threaded back through a dozen call sites —
and a seated population is placed on them, with more standing about on the
plazas. None of it is simulated: placed once, never touched again, one
instanced mesh per variant. A seated figure is not a walker with the animation
paused; the weight is on the seat, the arms go somewhere, and a standing one
rests on one leg, because a figure with both feet square reads as a mannequin.

The city has wildlife, and it keeps hours. Two populations that barely overlap:
pigeons, squirrels, butterflies and off-lead dogs work daylight; foxes and bats
work the dark, and `Shift` decides which are awake on the same clock as the
street lamps. Drawing both at once is the single most obvious way to make a
night render look wrong, and swapping them is nearly free — the swarms are
already separate objects. Alongside them: flocks of birds circling, gulls
following the water, ducks and swans on the ponds, and about one pedestrian in
eleven walking a dog on a lead.

Each animal is built around whatever identifies it. A squirrel is its tail —
drawn as a tapered rod it is a rat, and only the plume arcing over the back
makes it a squirrel. A fox is the ears and the brush, on a dog's frame pulled
long and low. A bat is two membranes and almost no body, and it flies like
nothing else here: a bat that cruises is a bird, so it gets a fast rate and a
path that barely resembles a circle. Animals are not lane-bound the way
vehicles and pedestrians are — a `Mover` is a scalar on a one-dimensional run,
which is exactly what makes queueing and give-way tractable and exactly wrong
for a pigeon. Each animal instead walks a closed loop of its own: a circle bent
by two harmonics keyed on its seed, so position and heading are a pure function
of the clock. There is nothing to integrate, no neighbour queries and no state,
which is why several hundred of them cost one transform write each per frame.
The dogs are the exception and get it free: each one is a copy of a
pedestrian's mover pushed to one side of the pavement and set back a stride, so
it walks its owner's route at its owner's pace.

Light propagates, by raycasting. The rasteriser answers "is this lit by the
sun?" with a shadow map and "how much ambient reaches it?" with a constant —
and the second answer is the expensive one to be wrong about, because it means
the inside of a courtyard, the pavement under an awning and the middle of an
open plaza all get the same fill. So the real answer is computed once, offline,
with rays: for every vertex of the static city, integrate incoming radiance
over the hemisphere about its normal, rays that escape collecting sky and rays
that hit collecting what that surface re-emits — its own albedo times its own
sky access. That is one bounce of global illumination, and it is where the
colour comes from: a wall opposite a brick facade picks up the brick. Build
with `--features mesh-bvh` to get it; without, the city ships with flat ambient
as before.

Three things had to be true for that to work. Ground is built as one quad per
block, and four corners can only carry a bilinear ramp, so receivers are
subdivided to a three-metre edge first — cheap in a renderer that pays per draw
rather than per triangle. Rays are seeded from the vertex *position* rather than
a counter, so the duplicated vertices subdivision creates bake identically and
shared edges have no seam. And the result is normalised against the same rays
cast with nothing in the way, per vertex: normalise against a fixed up-facing
reference instead and an unoccluded vertical wall scores 0.55 for no reason but
its orientation, which darkens every facade in the city by half and looks
plausible while doing it.

The result folds into vertex colour, which the shader already multiplies into
base colour. That is the standard compromise and worth naming: it darkens
direct sun as well as ambient, so a sunlit courtyard wall comes out slightly
too dark. Against that, a courtyard that reads as a courtyard.

Baking a city took 97 seconds and now takes 6, which is two bugs in the crate's
own raycaster rather than anything clever. `MeshBvh` gave up whenever a split
plane failed to separate anything and collapsed the whole subtree into one leaf
— and a leaf is scanned linearly. That is not a rare case: a handful of
triangles far larger than the rest, which is to say a ground plane, drags the
root bounds out until every candidate that separates the real geometry falls
into one or two SAH bins. One degenerate split at the root cost every later
query the entire scene, measured at 6k rays per second per core. It falls back
to a median split now, behind `BuildOptions::split_degenerate`. Separately,
first-hit traversal ignored the distance it was given and always opened the left
child first; it culls by the best hit so far and descends into the nearer child
now. Together: 6k rays/s/core to 900k.
`cargo test --features mesh-bvh --test mesh_bvh_raycast` pins both, against
brute force for correctness and against a throughput floor for the collapse.

The fallback is **off by default**, which is worth explaining because it looks
like leaving a fix on the table. Two things downstream read the *shape* of the
tree rather than merely querying it: `bvhcast` enumerates leaf-against-leaf
pairs, and the CSG evaluator marks any *coplanar* pair it is handed as
intersecting whether or not the two triangles are anywhere near each other. So
a tighter tree hands the evaluator a smaller candidate set and the boolean comes
out different — on the parity meshes, 3816 pairs instead of 6240 and a window
frame of 751 vertices instead of 4074. Neither enumeration is wrong; the tighter
one was checked against brute force and misses no genuinely overlapping pair.
But the result of a boolean should not depend on how its accelerator was built,
and until the evaluator decides coplanarity from the geometry instead, the
default reproduces three-mesh-bvh's enumeration and the flag is for query trees —
raycast, closest-point — where nothing reads the shape.

Sunrise and sunset work now. `day` used to reach 1 at eleven degrees of
elevation, which this sun clears three percent of the way into the day, so the
golden hour was over before it began and `--time 0.05` already looked like
noon. Sun intensity went to zero exactly at the horizon rather than dimming and
reddening through the airmass. And the shadow camera is a box sized to the
city, which is right at noon and useless at dawn — a ninety-metre tower at ten
degrees throws its shadow half a kilometre, and ground that far out was not in
the map, so the one time of day shadows matter most had none at all. The box
grows as the sun drops. The dome's Mie coefficient was left at the library
default and clipped to flat white over a sixth of the sky at noon; raising it
fixes that, and the direction is worth recording because it is the opposite of
the intuitive one — more Mie means more extinction along the view ray, so the
sky gets darker, not brighter. Swept: 0.005 gave 10.0%/16.4% of the sky clipped
at sunrise/noon, 0.030 gives 3.0%/0.0%.

The ground plate also had to grow. It ran to 4.6 half-widths against a fog that
reached full strength at 5.2, so the terrain stopped *before* the haze had
finished swallowing it and left a hard band along the skyline at every hour.
Everything past the fog is fog-coloured, so a larger plate costs eight quads.

Streets are lit and sold to. Each lamp is a column, a swept arm and a shielded
luminaire, in sodium or LED chosen per road rather than per lamp — a real city
is part way through swapping one for the other and the mixture is visible from
any bridge. Under each is a translucent cone from the lens to the road, which
is the cheapest thing that makes a night street read as lit rather than as
tarmac with bright decals painted on it. Above them: fascia signs with neon on
them, blades projecting over the pavement, backlit panels on blank flanks and
hoardings on the roofs of low blocks. Each is built twice — structure in the
lit batch so it is a grey board by day, face in an unlit night-only batch so
the same artwork is backlit after dark — and about one in four blinks on the
traffic signals' own clock. The artwork is not text: a legible word needs a
texture and this is one draw shared by every sign in the city. What reads at
fifty metres is layout — a field, a bold band, a mark, a line of blocks
standing in for a strapline — which is all you get from a real poster at that
distance too.

Traffic turns, and it turns through a corner. A vehicle that reaches a junction
takes the crossing road about one time in five, adopting that road's lane,
bounds and signalling, with the choice coming from its own PRNG state so the
simulation stays deterministic without a shared generator. The *simulation*
switches lane instantly — that is what keeps the queueing one-dimensional — but
the drawn pose sweeps a quadratic Bezier through the corner, committed a corner
radius before the junction so there is road left to sweep through. The two legs
of that curve are laid out equal, which is the only way its tangents leave
along the old lane and arrive along the new one; unequal legs put a
sixty-degree kink back at the entry. A vehicle will not turn into a road it is
already at the end of, because driving off the end mid-corner aborts the arc
and snaps the heading by a full ninety degrees — which is what the whole change
was there to stop. That is also why lane grouping
is rebuilt every frame rather than precomputed: a vehicle that turns leaves its
lane, and a stale index would have it queueing behind traffic on a road it is
no longer on. One sort a frame gives both the grouping and the order within
each lane.

Traffic obeys the signals and gives way. Each vehicle holds station behind
whatever is in front of it — lanes span the car, van, bus and lorry fleets, so
a car queues behind a bus — stops at a red light, and yields at an unsignalled
crossroads to anything already crossing it. Pedestrians run the same pass with
their own constants, one lane per pavement per direction, so a crowd bunches
instead of walking through itself.

Only *moving* vehicles occupy a junction, which is not a detail: marking
stationary ones deadlocks the network within a minute — a car queued at a red
with its tail in the junction behind it blocks that cross street permanently,
that street backs up into its own junction, and it spreads. Measured, it took
234 vehicles from a mean 8.4 m/s to 2.8 and still falling; with the fix it
settles at 5.0–5.8 and holds. The whole pass is one sort per lane per frame and
allocates nothing.

The four street lamps nearest the camera get real spot lights, and the first of
them casts a shadow — the renderer allows exactly one spot caster. Nearest-first
rather than a fixed set: the ones that matter are the ones you are standing
among, and which those are changes as the camera moves. The unlit glow discs
still fake every other lamp in the city, pulled down to 62% since where both
apply they were doubling up.

The open water is moved by a WGSL compute shader that rewrites the mesh's
vertex buffer in place, position and normal, between frames: the surface never
crosses host memory. That uses the crate's two opt-in hooks for exactly this,
`BufferGeometry::gpu_writable` and `Renderer::vertex_buffer`, and raw wgpu for
the pipeline and the dispatch. `simcity_metal` draws the same `Scene` through
the Metal backend instead (`--features metal`) — no wgpu at all, which makes
it a standing cross-backend check; the water is flat there, since the pass
that moves it is WGSL.

Level of detail is done the way a draw-call-bound renderer wants it. Swapping
a simpler mesh in per level would *add* a draw call per level per colour,
which is backwards here — so the far level is a single shared impostor mesh
carrying every distant instance whatever its colour, one draw for the whole
population. Everything nearer than the fleet's LOD distance gets the full
model, which can then be far better than it could otherwise afford: cars with
mirrors and wheels, and people built on a table of adult proportions —
ankle, knee, hip, waist, shoulder, chin — out of tapered limbs with elliptical
cross-sections, swinging between two pose meshes once per 0.8 m walked. A
torso is about twice as wide as it is deep; drawn round it reads as a length
of pipe, and drawn with end caps its shoulders read as flat wings, so the
chest is open-ended with a deltoid rounding each joint. `MeshBuilder::add_cylinder` only stands
things up the Y axis, which is exactly why the first version of the figure was
nine boxes: a limb that swings cannot be axis-aligned. `add_limb` (a tapered
cylinder between two arbitrary points) is the primitive that fixed it, and it
does tree branches too. Skin, hair, hands and shoes are *separate* meshes, because an instanced mesh's
material colour multiplies every vertex — folding them into the clothes would
give each wearer a face the colour of their jacket. And one skin mesh for the
whole city would give everyone the same face, so the population is split across
three tones: six skin meshes, two poses by three tones, four extra draws. Hair
sits in the same mesh as a vertex-colour multiple of the tone, so it darkens
with it. Each mesh's transform list is *rebuilt* every frame with only the instances it
is actually drawing, packed to the front. A zero-scale matrix rasterises
nothing but is still uploaded and still runs the vertex shader over the whole
model — with two walk poses and a separate skin mesh that is four wasted copies
of every pedestrian in the city. Packing is what turns the LOD cut into a real
saving: on the twelve-block boulevard the default view draws 63 of 736 movers
at full detail rather than paying for all of them. The window title reports the
split live, and the stats line separates static triangles from the mover
ceiling, since what the movers cost depends on where the camera is standing.

The sky is the crate's Preetham model (`SkyMaterial`) on a dome, which has its
own pipeline — no cull, no depth write, pinned at the far plane — so it costs
one draw and needs neither fog nor tone mapping. Its published defaults clip to
white here, because that shader expects an exposure of about 0.5 and ACES after
it and this one writes straight to the framebuffer; the turbidity and rayleigh
used are the values that still read as sky. The haze colour was then *measured*
off the rendered dome rather than guessed — the Preetham horizon is a
desaturated grey-teal, much darker and far less blue than you would pick by
eye, and fading the terrain to the wrong one puts a bright band along the
skyline. Stars were tried on the same shell and cut: `Points` go through the fog like
any other geometry, and at a radius that clears the city they fade to exactly
the sky behind them.

Massing is three forms — rectangular, chamfered and round, the last two as
prisms inscribed in square-ish plots — with stepped deco crowns on the boxes
and tapered ones on the prisms, because a flat parapet at 90 m reads as an
unfinished box. One building a city is a landmark, well above the rest: a
purely statistical height distribution gives a plateau, not a skyline.

Trees are built around their branches rather than as a blob on a stick. The
trunk forks into three to five primaries, each with an elbow, and each carries
its own foliage cluster — so the outline is several lumps at different heights
with visible structure underneath, which is the difference between a tree and a
lollipop. Reach and cluster size are tuned against each other: too much reach
and the crown reads as a bunch of grapes, too little and it merges back into
one mass. Six species — spreading broadleaf; tiered conifer, whose notches between whorls
are its whole silhouette; columnar poplar; weeping, with the clusters hung
*below* the branch tips rather than sitting on them; bare, which is the
armature with twigs and nothing on it; and palm, a leaning stem with fronds
made from flattened tapered limbs, admitted only by the waterfront layout,
because one palm among the conifers of a temperate grid reads as a mistake.
Leaves are usually green, with a few per cent blossom and a slice turning
amber through rust — which is what stops an avenue of them reading as one
repeated asset. Crowns have their radius jittered per vertex. A perfectly smooth
ovoid never reads as foliage however it is coloured; the silhouette is what
gives it away. The jitter is hashed on `(ring, sector)` rather than on
position, so the duplicated seam vertices agree and the poles, where every
sector collapses to one point, stay closed. It costs no extra vertices, so it
stays on at the far level of detail, where the outline is the only part still
legible.

At street level there is a clutter layer — bins, bollards, hydrants, benches,
post boxes, cabinets, planters and cones on the pavement; skips, pallets and
crates in the yards behind; litter in the gutters and stains under the skips.
Without it a pavement is a grey ribbon, and no amount of work on the towers
fixes that. Standing water sits in the gutters as low-roughness patches, which
is the cheapest reflection a city has: it picks up the sky by day and the lamps
at night through the same environment map the glass uses.

Materials that have a second lobe get one. A curtain wall is glazing set in a
frame and car paint is a coloured base under lacquer — both are clearcoat, and
a single-lobe BRDF cannot make that sharper specular over the base at all. So
facades, car paint and water are `PhysicalMaterial` with a clearcoat weight
(a lot on glass, almost none on brick), and foliage carries sheen, since leaves
go pale and bright at grazing angles in a way plain diffuse cannot.

Facades carry three maps off one tile at 32 px a window cell: albedo with
mullions, transoms and course lines, blinds drawn to a different height in
about half the windows, and weather staining that runs *down* from each sill
and stops between them, the way it does on a real wall; the lit-window mask,
where a lit room behind a blind glows dimly rather than showing a bright pane;
and a roughness map, which is what separates the glass from the wall it is set
in and from the fabric hanging behind it. The ground batches
take a world-space UV override so one 128×128 asphalt or paving tile covers six
metres wherever it lands — per-quad UVs would put a 40 m road cell and a 3 m
paint stripe at the same scale.

Lighting is a day cycle with a prefiltered sky environment rebuilt as the sun
moves: shadows swing, glass reflects the sky it is actually standing under,
and at dusk the windows, the shopfronts, the street lamps and every car's
headlamps come on — the windows from an emissive mask that shares the facade's
UVs exactly, the headlamps throwing a pool on the tarmac ahead.
`simcity_render` is the same city headless —
`--view aerial|skyline|close|street`, `--strip` for four times of day in one
sheet, `--sheet` for all five layouts. With `native-codec` on it will also
write the whole cycle as an animation in one process — `--anim simcity.webm`
renders 96 orbiting frames and encodes them with the crate's own VP9, one
frame at a time, no ffmpeg anywhere.

`ocean_water` is a **GPU FFT ocean**: a JONSWAP spectrum on a full lattice,
inverse-transformed each frame into three tiling cascades (swell / waves /
ripples) by a butterfly compute pass. Everything reads the cascades through one
shared sampler — a compute pass displaces a camera-anchored disc from them, the
`ShaderMaterial` fragment samples them per pixel for normals and breaking-wave
foam, a second compute pass accumulates persistent foam and wake, and the CPU
answers buoyancy from the dominant modes with no readback. The mesh is uploaded
once and written in place on the GPU, so per-frame CPU cost is flat in vertex
count (~0.3 ms at every quality level). Eight sea-state presets, five mesh
densities:
`cargo run --release --example ocean_water -- storm --quality ultra`,
`-- tropical --png ocean.png`, `-- --bench`, or `-- --list`.

Waves **shoal** over the analytic sea bed — slowing, shortening, growing by
Green's law and breaking past 0.78 of their own depth, which is what draws the
surf zone. It also does screen-space reflection and refraction, caustics on a
shader-lit sea floor, persistent foam and wake, spray/rain/underwater-mote
particles, an underwater view with Snell's window, water masking, sparkle and
multi-point buoyancy.

The FFT's butterfly table is validated on the CPU against both a naive DFT and
**rlx's own FFT** before it ever reaches the GPU (`--features rlx`), because a
butterfly index off by one produces plausible noise rather than an error. rlx is
not in the per-frame path: measured, its CPU backend is ~6.6 ms a frame for three
cascades, and its GPU backend builds its own device rather than borrowing this
one, so results would cross host memory either way. (It used to be a different
major of wgpu as well; since the wgpu 30 upgrade both are the same major, which
is what made rlx's GPU backend usable on wasm32 at all.)

Four small opt-in APIs carry that, all reusable outside the example:

- `BufferGeometry::gpu_writable` adds `STORAGE` usage to a mesh's vertex buffer,
  and `Renderer::vertex_buffer` hands it back — so a compute shader can write
  positions and normals directly, vertex animation with no CPU round trip.
- `ShaderMaterial::textures` binds up to four of your own texture views as
  `u_tex0..3`, for sampling something a compute pass wrote this frame.
- `ShaderMaterial::screen_space` draws the material in the refraction pass, where
  the captured opaque colour and depth are available — the prerequisite for any
  custom SSR or refraction.
- `ShaderMaterial::side` is now honoured (it was previously ignored and always
  culled back faces, which silently hid anything meant to be seen from inside).
