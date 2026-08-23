// Subtitles / captions drawn over a live threers scene in the browser.
//
// The renderer holds a `CaptionOverlay` and composites the active cue into each
// frame before presenting it, so there is no second canvas and no DOM text to
// keep in sync with the 3D view.

// Relative path so the page works under any static server root.
import THREE, { initThreers, CaptionTrack, CaptionOverlay } from '../threejs-shim.js';

const errEl = document.getElementById('err');
const statusEl = document.getElementById('status');

/** The demo dialogue, authored as WebVTT — cue settings and all. */
const VTT = `WEBVTT - Demo [en]

opening
00:00:00.000 --> 00:00:03.000
Captions are composited by the renderer,
straight onto the 3D frame.

wrap
00:00:03.000 --> 00:00:06.500
Long lines wrap inside the safe area, so a caption
can never run off the edge of the canvas no matter
how wide the text gets.

placed
00:00:06.500 --> 00:00:09.500 align:left line:8%
Cue settings move a caption out of the way —
this one is pinned to the top left.

closing
00:00:09.500 --> 00:00:12.000
The text is only rasterized when the cue changes.
`;

async function main() {
    await initThreers();
    statusEl.textContent = 'wasm ready';

    const canvas = document.getElementById('c');
    const renderer = await THREE.WebGLRenderer.create(canvas);
    renderer.setSize(canvas.width, canvas.height, false);

    // ---- scene ----
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d1018);

    const group = [];
    for (const [i, tint] of [0xd94452, 0x3fa9d9, 0xe6c04a].entries()) {
        const material = new THREE.MeshStandardMaterial({
            color: tint,
            roughness: 0.35,
            metalness: 0.1,
        });
        const mesh = new THREE.Mesh(new THREE.BoxGeometry(1.2, 1.2, 1.2), material);
        mesh.position.set(i * 2 - 2, 0, 0);
        scene.add(mesh);
        group.push(mesh);
    }
    scene.add(new THREE.AmbientLight(0xffffff, 0.35));
    const key = new THREE.DirectionalLight(0xfff4e6, 2.2);
    key.position.set(3, 4, 5);
    scene.add(key);

    const camera = new THREE.PerspectiveCamera(45, canvas.width / canvas.height, 0.1, 100);
    camera.position.set(0, 0.9, 7.5);
    camera.lookAt(0, 0, 0);

    // ---- captions ----
    const track = CaptionTrack.parse(VTT);
    const overlay = new CaptionOverlay(track, {
        fontSize: 34,
        color: '#ffffff',
        outlineColor: '#000000',
        outlineWidth: 2,
        background: '#00000096',
        anchor: 'bottom',
        align: 'center',
        maxWidth: 0.8,
    });
    // The style above is authored for 1080p; track the real canvas height.
    overlay.setAutoScale(true);
    renderer.setCaptions(overlay);

    // ---- playback clock ----
    const DURATION = Math.max(track.duration, 12);
    let time = 0;
    let playing = true;
    let last = performance.now();

    const playBtn = document.getElementById('play');
    const seek = document.getElementById('seek');
    const timeEl = document.getElementById('time');
    const cueEl = document.getElementById('cue');
    seek.max = String(DURATION);

    playBtn.addEventListener('click', () => {
        playing = !playing;
        playBtn.textContent = playing ? 'Pause' : 'Play';
    });
    seek.addEventListener('input', () => {
        time = Number(seek.value);
        playing = false;
        playBtn.textContent = 'Play';
    });

    // ---- style controls ----
    const restyle = (patch) => overlay.setStyle(patch);
    document.getElementById('anchor').addEventListener('change', (e) =>
        restyle({ anchor: e.target.value }));
    document.getElementById('align').addEventListener('change', (e) =>
        restyle({ align: e.target.value }));
    document.getElementById('size').addEventListener('input', (e) =>
        restyle({ fontSize: Number(e.target.value) }));
    document.getElementById('box').addEventListener('change', (e) =>
        restyle({ background: e.target.checked ? '#00000096' : '#00000000' }));

    // ---- downloads: the same cues, as a subtitle file ----
    const download = (text, name, type) => {
        const url = URL.createObjectURL(new Blob([text], { type }));
        const a = Object.assign(document.createElement('a'), { href: url, download: name });
        a.click();
        URL.revokeObjectURL(url);
    };
    document.getElementById('srt').addEventListener('click', () =>
        download(track.toSrt(), 'captions.srt', 'application/x-subrip'));
    document.getElementById('vtt').addEventListener('click', () =>
        download(track.toVtt(), 'captions.vtt', 'text/vtt'));

    // ---- frame loop ----
    function frame(now) {
        const dt = Math.min((now - last) / 1000, 0.1);
        last = now;
        if (playing) {
            time = (time + dt) % DURATION;
            seek.value = String(time);
        }

        for (const [i, mesh] of group.entries()) {
            mesh.rotation.y = 0.5 + i * 0.4 + time * 0.6;
            mesh.rotation.x = 0.3;
        }

        // The renderer draws the cue at `captionTime` over the frame.
        renderer.captionTime = time;
        renderer.render(scene, camera);

        timeEl.textContent = `${time.toFixed(1)} s`;
        cueEl.textContent = track.textAt(time) || '—';
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}

main().catch((e) => {
    errEl.textContent = String(e?.stack || e);
    statusEl.textContent = 'failed';
});
