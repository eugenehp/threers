// The Gaussian splat viewer's renderer: device, scene, camera, frame loop.
//
// It speaks only through a message port, so the same code runs in a Worker on
// an OffscreenCanvas (web/splats-render-worker.js; the page's own thread then
// does nothing but forward input) or, where a canvas cannot be handed to a
// worker, on the page itself through a MessageChannel. Sorting goes to a
// third thread when the page gives it one (web/splats-sort-worker.js): each
// frame this asks for an order along the new view axis if none is in flight,
// and draws with the latest that has come back, so a frame never waits on a
// sort.
//
// In:  init {canvas, width, height, dpr, params, sortPort?}
//      resize {width, height, dpr}  (CSS pixels)
//      load {bytes | url | galaxy, name}
//      orbit {daz, del} · zoom {factor}
// Out: status {status}  (what the page shows and tests read)
//      error {text}

import init, { WebRenderer, WebScene, WebCamera, WebColor } from "./pkg/threers.js";

/// A spiral galaxy as `.splat` bytes: 32 a splat — centre and scales as
/// float32s, then RGBA and a w-x-y-z rotation as bytes.
function galaxy(n) {
  const buf = new ArrayBuffer(n * 32), f = new Float32Array(buf), u = new Uint8Array(buf);
  let seed = 7;
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) >>> 8) / 16777216;
  const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-9)) * Math.cos(6.2831853 * rnd());
  for (let i = 0; i < n; i++) {
    const core = i < n * 0.15;
    const r = core ? Math.abs(gauss()) * 0.25 : 0.2 + Math.pow(rnd(), 0.7) * 1.8;
    const arm = (i % 2) * Math.PI;
    const a = arm + r * 2.6 + gauss() * (core ? 3 : 0.28);
    const h = gauss() * 0.04 * (core ? 2 : 1) / (0.4 + r);
    const o = i * 8;
    f[o] = Math.cos(a) * r; f[o + 1] = h; f[o + 2] = Math.sin(a) * r;
    const s = core ? 0.012 + rnd() * 0.02 : 0.004 + rnd() * 0.012;
    f[o + 3] = s * (1 + rnd() * 2); f[o + 4] = s * 0.3; f[o + 5] = s;
    const t = Math.min(r / 2, 1), b = i * 32 + 24;
    const star = rnd() < 0.02;
    u[b]     = star ? 255 : 255 * (1 - 0.55 * t);
    u[b + 1] = star ? 255 : 255 * (0.82 - 0.35 * t);
    u[b + 2] = star ? 255 : 255 * (0.55 + 0.4 * t);
    u[b + 3] = (star ? 0.95 : core ? 0.5 : 0.25) * 255;
    // Turned about y to follow the arm.
    const half = -a / 2;
    u[b + 4] = Math.cos(half) * 128 + 128; u[b + 5] = 128;
    u[b + 6] = Math.sin(half) * 128 + 128; u[b + 7] = 128;
  }
  return new Uint8Array(buf);
}

/// Run the viewer, taking its orders from `port`. In a Worker the canvas
/// comes in the `init` message, an OffscreenCanvas; on the page it is
/// `pageCanvas`, which a message port cannot carry.
export function run(port, inWorker, pageCanvas) {
  const send = (m, transfer) => port.postMessage(m, transfer ?? []);
  let params, renderer, canvas, scene, handle, frame = [0, 0, 0, 1];
  let sortPort = null, sortGen = 0, sortInFlight = false, sortedElsewhere = false, sentAxis = null;
  // Order buffers back from the sort worker, to send it again: two shuttle
  // between the threads, so a frame's sort allocates nothing to collect.
  let spare = [];
  let css = [1, 1], dpr = 1, w = 0, h = 0;
  let status = { ready: false };
  // Start on the +z side, near: a capture's cameras look down -z in the
  // y-up frame (a trainer's PLY turned over, or an .spz), and from much past
  // the splats' median distance the view leaves what they were trained to
  // show.
  const cam = { az: 0.25, el: 0.15, dist: 3, target: [0, 0, 0] };
  let camera, pinned, flipParam;
  let fixedScale, maxScale, scale;
  const gaps = [];
  let lastTick = 0, frames = 0, last = 0, fps = 0;
  let benchFrames = 0, bench = null;

  const raf = globalThis.requestAnimationFrame
    ? (f) => globalThis.requestAnimationFrame(f)
    : (f) => setTimeout(() => f(performance.now()), 16);

  /// Whether to turn a file over: trainers write y-down, but `.spz` (gzip,
  /// or version 4's "NGSP") is stored y-up.
  function flipFor(bytes) {
    if (flipParam !== null) return flipParam !== "0";
    const spz = (bytes[0] === 0x1f && bytes[1] === 0x8b) ||
      String.fromCharCode(...bytes.subarray(0, 4)) === "NGSP";
    return !spz;
  }

  function load(bytes, name, flip, extra) {
    const t = performance.now();
    scene?.free();
    scene = new WebScene();
    scene.background = new WebColor(0, 0, 0);
    try {
      handle = scene.addSplats(bytes);
    } catch (e) {
      send({ type: "error", text: `${name}: ${e.message ?? e}` });
      return;
    }
    if (flip) scene.setRotationEuler(handle, Math.PI, 0, 0, "XYZ");
    frame = Array.from(scene.splatFrame(handle));
    // The centre in world space: turned over with the object.
    cam.target = flip ? [frame[0], -frame[1], -frame[2]] : frame.slice(0, 3);
    // `?dist=` scales the distance `splatFrame` suggests.
    cam.dist = frame[3] * (Number(params.get("dist")) || 1);
    camera.setNear?.(frame[3] * 0.003);
    status = { ready: false, name, count: scene.splatCount(), loadMs: performance.now() - t,
               backend: renderer.backend(), worker: inWorker, sortWorker: !!sortPort, ...extra };
    // Sorting: the renderer sorts its own frames until the sort worker's
    // first order comes back for these splats, so none is drawn unsorted.
    sortedElsewhere = false;
    sentAxis = null;
    sortInFlight = false;
    spare = [];
    sortGen++;
    if (sortPort) {
      const centres = scene.splatCentres(handle);
      sortPort.postMessage({ type: "centres", centres, gen: sortGen }, [centres.buffer]);
    }
    bench = null;
    send({ type: "status", status });
  }

  async function fetchBytes(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url}: ${r.status}`);
    return new Uint8Array(await r.arrayBuffer());
  }

  async function onLoad(m) {
    try {
      if (m.galaxy) return load(galaxy(m.galaxy), "a generated galaxy", flipParam === "1");
      let bytes = m.bytes, extra = {};
      if (m.url) {
        const t0 = performance.now();
        bytes = await fetchBytes(m.url);
        extra.fetchMs = performance.now() - t0;
      }
      load(bytes, m.name, flipFor(bytes), extra);
    } catch (e) {
      send({ type: "error", text: `${m.name}: ${e.message ?? e}` });
    }
  }

  function onSortMessage(m) {
    if (m.type !== "order" || m.gen !== sortGen || !scene) return;
    sortInFlight = false;
    try {
      scene.setSplatOrder(handle, m.order);
      spare.push(m.order);
      if (!sortedElsewhere) {
        sortedElsewhere = true;
        scene.setSplatsSortedElsewhere(true);
      }
    } catch (e) {
      send({ type: "error", text: `sort: ${e.message ?? e}` });
    }
  }

  /// Ask the sort worker for the order along the current view axis, if it
  /// has turned since the last ask and no sort is under way.
  function requestSort() {
    if (!sortPort || sortInFlight || !scene) return;
    const axis = scene.splatViewAxis(handle, camera);
    if (axis.length !== 3) return;
    if (sentAxis && axis[0] === sentAxis[0] && axis[1] === sentAxis[1] && axis[2] === sentAxis[2]) return;
    sentAxis = axis;
    sortInFlight = true;
    const buffer = spare.pop();
    sortPort.postMessage({ type: "sort", axis: Array.from(axis), gen: sortGen, buffer },
                         buffer ? [buffer.buffer] : []);
  }

  /// Render scale (drawing-buffer pixels per CSS pixel). It starts at the
  /// screen's own, up to 2, and adapts to hold 30+ fps with room to spare:
  /// down a fifth while the median frame takes over 28 ms, back up a tenth
  /// when under 18 ms, never below 0.5. `?scale=` pins it.
  function adapt(t) {
    if (lastTick) gaps.push(t - lastTick);
    lastTick = t;
    if (fixedScale || gaps.length < 30) return;
    const median = [...gaps].sort((a, b) => a - b)[gaps.length >> 1];
    gaps.length = 0;
    if (median > 28 && scale > 0.5) scale = Math.max(0.5, scale * 0.8);
    else if (median < 18 && scale < maxScale) scale = Math.min(maxScale, scale * 1.1);
  }

  /// `?bench=<frames>`: once loaded, orbit by itself a hundredth of a radian
  /// a frame for that many frames, then report the median and 90th
  /// percentile CPU time (this thread's) and frame interval as
  /// `status.bench`. `&still=1` keeps the camera still: the GPU alone.
  function benchFrame(t, cpu) {
    if (!benchFrames || !status.ready || status.bench) return;
    bench ??= { cpu: [], gap: [], last: t, skip: 10 };
    if (params.get("still") !== "1") cam.az += 0.01;
    if (bench.skip-- > 0) { bench.last = t; return; }
    bench.cpu.push(cpu);
    bench.gap.push(t - bench.last);
    bench.last = t;
    if (bench.cpu.length < benchFrames) return;
    const q = (v, f) => [...v].sort((a, b) => a - b)[Math.floor(v.length * f)];
    status.bench = {
      cpuMs: q(bench.cpu, 0.5), cpuP90: q(bench.cpu, 0.9),
      frameMs: q(bench.gap, 0.5), frameP90: q(bench.gap, 0.9),
    };
    send({ type: "status", status });
  }

  function tick() {
    raf(tick);
    if (!scene) return;
    adapt(performance.now());
    const cw = Math.max(1, Math.round(css[0] * scale)), ch = Math.max(1, Math.round(css[1] * scale));
    if (cw !== w || ch !== h) {
      w = cw; h = ch;
      canvas.width = w; canvas.height = h;
      renderer.setSize(w, h);
      camera.setAspect(w / h);
    }
    if (pinned) {
      // `?cam=x,y,z,tx,ty,tz`: a fixed camera, to compare with other renderers.
      camera.setPosition(pinned[0], pinned[1], pinned[2]);
      camera.lookAt(pinned[3], pinned[4], pinned[5]);
    } else {
      const [tx, ty, tz] = cam.target;
      const c = Math.cos(cam.el);
      camera.setPosition(tx + Math.sin(cam.az) * c * cam.dist, ty + Math.sin(cam.el) * cam.dist,
                         tz + Math.cos(cam.az) * c * cam.dist);
      camera.lookAt(tx, ty, tz);
    }
    const t = performance.now();
    requestSort();
    renderer.render(scene, camera);
    const cpu = performance.now() - t;
    frames++;
    status.ready = true;
    status.cpuMs = cpu;
    status.scale = scale;
    status.maxCpuMs = Math.max(status.maxCpuMs ?? 0, cpu);
    status.sortedElsewhere = sortedElsewhere;
    if (t - last > 500) {
      fps = frames * 1000 / (t - last);
      frames = 0; last = t;
      status.fps = fps;
      status.fixedScale = !!fixedScale;
      send({ type: "status", status });
    }
    benchFrame(t, cpu);
  }

  port.onmessage = async ({ data: m }) => {
    if (m.type === "init") {
      params = new URLSearchParams(m.params);
      // `?backend=gl`: hide WebGPU so the renderer starts on WebGL2, as a
      // browser without it would.
      if (params.get("backend") === "gl") {
        Object.defineProperty(navigator, "gpu", { value: undefined, configurable: true });
      }
      pinned = params.get("cam")?.split(",").map(Number);
      flipParam = params.get("flip");
      fixedScale = Number(params.get("scale")) || 0;
      maxScale = fixedScale || Math.min(m.dpr, 2);
      scale = maxScale;
      benchFrames = Number(params.get("bench")) || 0;
      css = [m.width, m.height];
      dpr = m.dpr;
      canvas = m.canvas ?? pageCanvas;
      canvas.width = Math.max(1, Math.round(css[0] * scale));
      canvas.height = Math.max(1, Math.round(css[1] * scale));
      await init();
      try {
        renderer = inWorker ? await WebRenderer.newOffscreen(canvas) : await WebRenderer.new(canvas);
      } catch (e) {
        send({ type: "error", text: "This browser has neither WebGPU nor WebGL2: " + e });
        return;
      }
      camera = WebCamera.perspective(50, 1, 0.01, 1000);
      if (m.sortPort) {
        sortPort = m.sortPort;
        sortPort.onmessage = ({ data }) => onSortMessage(data);
      }
      send({ type: "ready" });
      raf(tick);
    } else if (m.type === "resize") {
      css = [m.width, m.height];
      if (m.dpr !== dpr && !fixedScale) {
        dpr = m.dpr;
        maxScale = Math.min(dpr, 2);
        scale = Math.min(scale, maxScale);
      }
    } else if (m.type === "load") {
      onLoad(m);
    } else if (m.type === "orbit") {
      cam.az -= m.daz;
      cam.el = Math.max(-1.5, Math.min(1.5, cam.el + m.del));
    } else if (m.type === "zoom") {
      cam.dist *= m.factor;
    }
  };
}
