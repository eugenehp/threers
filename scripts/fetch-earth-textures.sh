#!/usr/bin/env bash
# Fetch NASA Earth imagery for the `earth_sun_flare` example and its web port.
#
#   scripts/fetch-earth-textures.sh            # base set, ~4 MB
#   scripts/fetch-earth-textures.sh --tiles    # + the 500 m tiled globe, ~440 MB
#   scripts/fetch-earth-textures.sh --sky      # + Deep Star Maps and the CGI Moon Kit
#   scripts/fetch-earth-textures.sh --png      # also emit PNGs for the native example
#
# Everything here is NASA imagery, which is in the public domain — NASA material
# is not protected by copyright unless third-party content is noted, and none of
# these carry such a notice. Sources are recorded in the CREDITS file written
# alongside the images.
#
# The files are NOT committed: they are large, and a texture download does not
# belong in a source tree. Both the native example and the web demo fall back to
# their procedural planet when the directory is empty, so the repo works without
# ever running this.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/web/assets/earth"
BASE="https://eoimages.gsfc.nasa.gov/images/imagerecords"
WANT_TILES=0
WANT_PNG=0
WANT_SKY=0
# Per-tile edge length after resampling; eight tiles make a 4x2 globe.
TILE_PX="${TILE_PX:-4096}"
usage() {
    sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
}

for arg in "$@"; do
    case "$arg" in
        --tiles) WANT_TILES=1 ;;
        --sky)   WANT_SKY=1 ;;
        --png)   WANT_PNG=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown flag: $arg" >&2; echo >&2; usage >&2; exit 2 ;;
    esac
done

mkdir -p "$OUT"

# get <url> <destination>
get() {
    local url="$1" dest="$2"
    if [ -s "$dest" ]; then
        echo "  have $(basename "$dest")"
        return
    fi
    echo "  get  $(basename "$dest")"
    # `-C -` resumes a part-finished download; the tiles are big enough to care.
    curl -fSL --retry 3 --retry-delay 2 -C - -o "$dest" "$url"
}

echo "==> base textures → $OUT"
# Blue Marble Next Generation, December 2004, topography + bathymetry (2 km).
get "$BASE/73000/73909/world.topo.bathy.200412.3x5400x2700.jpg" "$OUT/day.jpg"
# Black Marble 2012 — night lights.
get "$BASE/79000/79765/dnb_land_ocean_ice.2012.3600x1800.jpg" "$OUT/night.jpg"
# MODIS cloud composite.
get "$BASE/57000/57747/cloud_combined_2048.jpg" "$OUT/clouds.jpg"
# GEBCO elevation, for the normal/relief map.
get "$BASE/73000/73934/gebco_08_rev_elev_21600x10800.png" "$OUT/elevation.png"
# The 2002 Blue Marble, for the Arctic. Blue Marble Next Generation is built
# from visible-light MODIS passes and the Arctic spends months in polar night,
# so every monthly mosaic pads the top of the map with the same near-black fill
# — the December and July releases are byte-identical down to about 82.6°N.
# This one has the sea ice, at full brightness right over the pole. Already a
# PNG, so no conversion step.
get "$BASE/57000/57730/land_ocean_ice_8192.png" "$OUT/ice_cap.png"

if [ "$WANT_TILES" = "1" ]; then
    echo "==> 500 m tiles (~440 MB; 8 × 21600²) → $OUT/tiles"
    mkdir -p "$OUT/tiles"
    # The globe is split 4 × 2: columns A–D run west→east from 180°W, rows 1–2
    # run north→south from 90°N. Each tile is one 90°×90° block.
    for t in A1 B1 C1 D1 A2 B2 C2 D2; do
        get "$BASE/73000/73909/world.topo.bathy.200412.3x21600x21600.$t.jpg" \
            "$OUT/tiles/day.$t.jpg"
    done

    # 21600² is past what a browser will decode — Chrome needs ~1.9 GB of RGBA
    # for one of these — so emit a resampled copy per tile. Eight of these at
    # 4096² compose a 16384x8192 globe, well beyond the 5400x2700 single file.
    echo "==> resampling tiles to ${TILE_PX}px"
    for t in A1 B1 C1 D1 A2 B2 C2 D2; do
        src="$OUT/tiles/day.$t.jpg"
        dst="$OUT/tiles/day.$t.$TILE_PX.jpg"
        [ -s "$src" ] || continue
        [ -s "$dst" ] && { echo "  have $(basename "$dst")"; continue; }
        if command -v sips >/dev/null 2>&1; then
            sips -Z "$TILE_PX" "$src" --out "$dst" >/dev/null
        elif command -v magick >/dev/null 2>&1; then
            magick "$src" -resize "${TILE_PX}x${TILE_PX}" "$dst"
        else
            echo "  !! need sips or ImageMagick to resample" >&2; break
        fi
        echo "  made $(basename "$dst")"
    done
fi

if [ "$WANT_SKY" = "1" ]; then
    # Two more NASA/SVS products, both public domain with a credit request:
    #
    #   Deep Star Maps 2020 — 1.7 billion stars from Hipparcos-2, Tycho-2 and
    #   Gaia DR2, plotted in celestial coordinates. Shipped as OpenEXR half-float
    #   so the dynamic range survives; the 4k is the smallest useful one.
    #
    #   CGI Moon Kit — LRO colour and LOLA elevation. TIFF, so it needs
    #   converting before a browser will touch it.
    #
    #   Colour comes at 8k (8192x4096) rather than 4k: 4k is 12 texels per
    #   degree, which on a close pass shows as mush. 8k is the largest a single
    #   texture can be and still fit WebGPU's guaranteed 8192 px.
    #
    #   Elevation at 16 pixels/degree (5760x2880) rather than 4 — the Moon is
    #   craters and nothing else, so the height field is most of its character.
    #   `ldem_64` exists at 506 MB if you want it.
    SVS="https://svs.gsfc.nasa.gov/vis/a000000"
    echo "==> sky and moon → $OUT"
    mkdir -p "$OUT/sky"
    get "$SVS/a004800/a004851/starmap_2020_4k.exr" "$OUT/sky/starmap_4k.exr"
    get "$SVS/a004700/a004720/lroc_color_poles_${MOON_COLOR:-8k}.tif" "$OUT/sky/moon_color.tif"
    # 16k for the tiled Moon: 16384x8192 is past what one texture can be, so it
    # is split into eight 4096² tiles the same way Blue Marble is.
    get "$SVS/a004700/a004720/lroc_color_poles_16k.tif" "$OUT/sky/moon_color_16k.tif"
    get "$SVS/a004700/a004720/ldem_${MOON_LDEM:-16}_uint.tif" "$OUT/sky/moon_height.tif"

    # TIFF and EXR are not browser formats. Convert what we can.
    for src in "$OUT/sky/moon_color.tif" "$OUT/sky/moon_height.tif"; do
        [ -s "$src" ] || continue
        dst="${src%.tif}.png"
        [ -s "$dst" ] && { echo "  have $(basename "$dst")"; continue; }
        if command -v sips >/dev/null 2>&1; then
            sips -s format png "$src" --out "$dst" >/dev/null 2>&1 && echo "  made $(basename "$dst")"
        elif command -v magick >/dev/null 2>&1; then
            magick "$src" "$dst" && echo "  made $(basename "$dst")"
        fi
    done
    # Split the 16k Moon into tiles. Columns A-D run west->east from 180°W,
    # rows 1-2 north->south — the same grid, and the same names, as the Blue
    # Marble tiles, so one loader serves both.
    if [ -s "$OUT/sky/moon_color_16k.tif" ] && command -v sips >/dev/null 2>&1; then
        echo "==> splitting the 16k Moon into 8 tiles → $OUT/sky/tiles"
        mkdir -p "$OUT/sky/tiles"
        col=0
        for c in A B C D; do
            row=0
            for r in 1 2; do
                dst="$OUT/sky/tiles/moon.$c$r.png"
                if [ -s "$dst" ]; then
                    echo "  have $(basename "$dst")"
                else
                    # `-s format png` is not optional: cropping alone leaves
                    # the data in the source's format, so `--out foo.png` would
                    # quietly write a TIFF that no browser will decode.
                    sips -s format png \
                        -c 4096 4096 --cropOffset $((row * 4096)) $((col * 4096)) \
                        "$OUT/sky/moon_color_16k.tif" --out "$dst" >/dev/null
                    echo "  made $(basename "$dst")"
                fi
                row=$((row + 1))
            done
            col=$((col + 1))
        done
    fi

    if [ -s "$OUT/sky/starmap_4k.exr" ] && [ ! -s "$OUT/sky/starmap_4k.png" ]; then
        # EXR is linear half-float; tone it down on the way to 8-bit or the
        # bright stars clip and the faint ones vanish.
        if command -v magick >/dev/null 2>&1; then
            magick "$OUT/sky/starmap_4k.exr" -colorspace RGB -evaluate multiply 3 \
                -colorspace sRGB "$OUT/sky/starmap_4k.png" \
                && echo "  made starmap_4k.png"
        else
            echo "  note: starmap is OpenEXR; the crate's ExrLoader reads it, but a"
            echo "        browser needs ImageMagick to convert it first."
        fi
    fi
fi

# The GEBCO elevation map is 21600x10800; decoding that in a browser needs
# ~930 MB of RGBA. Emit a small copy for the relief and water masks.
if [ -s "$OUT/elevation.png" ] && [ ! -s "$OUT/elevation.4096.png" ]; then
    echo "==> resampling elevation to 4096px"
    if command -v sips >/dev/null 2>&1; then
        sips -Z 4096 "$OUT/elevation.png" --out "$OUT/elevation.4096.png" >/dev/null
        echo "  made elevation.4096.png"
    elif command -v magick >/dev/null 2>&1; then
        magick "$OUT/elevation.png" -resize 4096x2048 "$OUT/elevation.4096.png"
        echo "  made elevation.4096.png"
    fi
fi

if [ "$WANT_PNG" = "1" ]; then
    # The native example only has a PNG decoder, so convert. `sips` ships with
    # macOS; elsewhere use ImageMagick.
    echo "==> converting to PNG for the native example"
    for f in "$OUT"/*.jpg; do
        [ -e "$f" ] || continue
        png="${f%.jpg}.png"
        [ -s "$png" ] && { echo "  have $(basename "$png")"; continue; }
        if command -v sips >/dev/null 2>&1; then
            sips -s format png "$f" --out "$png" >/dev/null
        elif command -v magick >/dev/null 2>&1; then
            magick "$f" "$png"
        else
            echo "  !! need sips or ImageMagick to convert $(basename "$f")" >&2
            continue
        fi
        echo "  made $(basename "$png")"
    done
fi

cat > "$OUT/CREDITS" <<'EOF'
NASA Earth imagery — public domain.

NASA material is generally not protected by copyright: see
https://www.nasa.gov/nasa-brand-center/images-and-media/ . None of the files
below carry a third-party copyright notice.

  day.jpg        Blue Marble: Next Generation, December 2004, topography and
                 bathymetry. NASA Earth Observatory (Reto Stöckli).
                 https://visibleearth.nasa.gov/images/73909
  tiles/day.*    The same product at 500 m, as the eight 21600x21600 tiles it
                 is distributed in.
  night.jpg      Black Marble 2012, VIIRS day/night band. NASA Earth
                 Observatory. https://visibleearth.nasa.gov/images/79765
  clouds.jpg     MODIS cloud composite. NASA Earth Observatory.
                 https://visibleearth.nasa.gov/images/57747
  elevation.png  GEBCO/SRTM elevation, 21600x10800. NASA Earth Observatory.
                 https://visibleearth.nasa.gov/images/73934
  ice_cap.png    Blue Marble 2002: land surface, ocean colour and sea ice,
                 8192x4096. Supplies the Arctic, which Blue Marble Next
                 Generation leaves as a near-black fill above about 82.6°N in
                 every month. NASA Earth Observatory (Reto Stöckli).
                 https://visibleearth.nasa.gov/images/57730
  sky/starmap_*  Deep Star Maps 2020 — 1.7 billion stars from Hipparcos-2,
                 Tycho-2 and Gaia DR2. NASA/Goddard Scientific Visualization
                 Studio; Gaia DR2: ESA/Gaia/DPAC. https://svs.gsfc.nasa.gov/4851
  sky/moon_*     CGI Moon Kit — LRO colour (8k) and LOLA elevation (16 px per
                 degree). NASA/Goddard Scientific Visualization Studio.
                 https://svs.gsfc.nasa.gov/4720
EOF

echo "==> done. $(du -sh "$OUT" | cut -f1) in $OUT"
echo "    The demo picks these up automatically; delete the directory to go back"
echo "    to the procedural planet."
