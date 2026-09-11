#!/usr/bin/env python3
"""Codec benchmark: threers' native Rust encoders vs ffmpeg vs VideoToolbox.

Every arm is fed the identical planar `yuv420p` clip and every result is decoded
back to `yuv420p` and scored against that same reference, so what is measured is
the codec and not a colour conversion someone else did differently.

Three things this harness does that eyeballing a table does not:

  * **BD-rate.** Interpolating between two rate points answers a different
    question at every pair you pick. BD-rate integrates the whole curve over the
    quality range the two encoders share, and reports one number.
  * **Three metrics.** PSNR, SSIM and VMAF disagree, and which one disagrees is
    itself a finding — deblocking helps two of them and hurts the third.
  * **Both ffmpeg tunings.** x264 and x265 ship psychovisual RD and adaptive
    quantization on. Scoring against the defaults measures that trade, not codec
    efficiency, and on photographic content `-tune psnr` beats the defaults on
    *VMAF* too — by about 1.5 points at equal size, because it turns adaptive
    quantization off. That is the same effect measured on this project's own
    adaptive quantization, in a mature encoder, and it is large enough to flip
    the sign of a comparison.

Two GOP comparisons are reported because only both together are honest:

  * against **all-intra** ffmpeg (`-g 1`) — the like-for-like number, since the
    native encoders code every frame independently;
  * against **default-GOP** ffmpeg — what you would actually ship, and therefore
    the gap that matters to the file on disk.

Usage:
    scripts/codec_bench.py --clip frog
    scripts/codec_bench.py --clip all --out out/codec-bench
"""

import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target" / "release" / "examples" / "codec_bench"


def sh(cmd, capture_output=False, **kw):
    """Run `cmd`. `capture_output=True` keeps stdout as raw bytes."""
    if capture_output:
        return subprocess.run(cmd, capture_output=True, **kw)
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def ffmpeg(args):
    return sh(["ffmpeg", "-hide_banner", "-nostdin", "-y", *args])


# --------------------------------------------------------------------------
# Bjontegaard delta rate
# --------------------------------------------------------------------------


def bd_rate(anchor, test):
    """Average bitrate difference at equal quality, in percent.

    `anchor` and `test` are lists of `(bytes, metric)`. Negative means `test`
    needs fewer bits for the same quality.

    Eyeballing two rate points and interpolating between them — which is what
    the tables in `docs/codec-benchmark.md` did before this existed — answers a
    different question at every pair you pick. BD-rate is the standard answer:
    fit log(rate) as a cubic in the quality metric, integrate both curves over
    the quality range they share, and divide. It needs four points and it is
    only meaningful where the two curves actually overlap, so it returns
    `None` rather than a number when they barely do.
    """
    import numpy as np

    a = sorted((m, math.log10(b)) for b, m in anchor if b and m not in (None, float("inf")))
    t = sorted((m, math.log10(b)) for b, m in test if b and m not in (None, float("inf")))
    if len(a) < 4 or len(t) < 4:
        return None
    am, ar = np.array([p[0] for p in a]), np.array([p[1] for p in a])
    tm, tr = np.array([p[0] for p in t]), np.array([p[1] for p in t])

    lo = max(am.min(), tm.min())
    hi = min(am.max(), tm.max())
    # The overlap has to be judged against the metric's own scale: SSIM spans
    # 0.95 to 0.997 where PSNR spans 30 dB to 50, and a fixed threshold would
    # silently reject every SSIM curve.
    span = max(am.max() - am.min(), tm.max() - tm.min())
    if span <= 0 or (hi - lo) < 0.25 * span:
        return None

    pa = np.polyfit(am, ar, 3)
    pt = np.polyfit(tm, tr, 3)
    ia = np.polyval(np.polyint(pa), hi) - np.polyval(np.polyint(pa), lo)
    it = np.polyval(np.polyint(pt), hi) - np.polyval(np.polyint(pt), lo)
    return (10 ** ((it - ia) / (hi - lo)) - 1) * 100


def ssim_db(v):
    """SSIM on a decibel scale.

    Raw SSIM crowds everything above 0.95 into a sliver, so a polynomial fit
    across it is dominated by rounding. `-10 log10(1 - SSIM)` spreads it out the
    way PSNR already is, which is the usual convention for BD-rate on SSIM.
    """
    if v is None or v >= 1.0:
        return None
    return -10.0 * math.log10(max(1.0 - v, 1e-9))


def bd_table(rows, clip_name, metric):
    """BD-rate of each native encoder against its ffmpeg counterparts.

    Reported against ffmpeg's *defaults* and against `-tune psnr` separately,
    because they are different questions and the gap between them is large.
    x264 and x265 ship psychovisual rate-distortion and adaptive quantization
    on: both deliberately give up PSNR for how the result looks. Scoring PSNR
    against the defaults measures that trade, not codec efficiency — it is worth
    about 19% here, enough to flip the sign of the answer.
    """
    def pts(prefix):
        out = []
        for r in rows:
            if not (r.get("ok") and r["clip"] == clip_name and r["arm"].startswith(prefix)):
                continue
            v = r.get(metric)
            if metric == "ssim":
                v = ssim_db(v)
            if v is not None:
                out.append((r["bytes"], v))
        return out

    out = []
    for label, test_prefix, anchor_prefix in (
        ("native H.264 vs x264 all-intra (default)", "native h264 intra", "x264 all-intra crf"),
        ("native H.264 vs x264 all-intra (tune psnr)", "native h264 intra", "x264 all-intra psnr"),
        ("native HEVC  vs x265 all-intra (default)", "native hevc intra +residual", "x265 all-intra crf"),
        ("native HEVC  vs x265 all-intra (tune psnr)", "native hevc intra +residual", "x265 all-intra psnr"),
    ):
        bd = bd_rate(pts(anchor_prefix), pts(test_prefix))
        out.append((label, bd))
    return out


# --------------------------------------------------------------------------
# Clips
# --------------------------------------------------------------------------


def clip_from_pngs(pattern, name, fps, work):
    """Build a raw yuv420p clip from a numbered PNG sequence."""
    files = sorted(ROOT.glob(pattern))
    if not files:
        return None
    yuv = work / f"{name}.yuv"
    probe = sh(
        ["ffprobe", "-v", "error", "-select_streams", "v",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(files[0])]
    )
    w, h = (int(x) for x in probe.stdout.strip().split(","))
    # 4:2:0 needs even dimensions; crop rather than scale so no arm sees resampling.
    w, h = w - (w % 2), h - (h % 2)
    r = ffmpeg(["-framerate", str(fps), "-pattern_type", "glob",
                "-i", str(ROOT / pattern),
                "-vf", f"crop={w}:{h}:0:0", "-pix_fmt", "yuv420p",
                "-f", "rawvideo", str(yuv)])
    if r.returncode != 0:
        print(r.stderr[-800:], file=sys.stderr)
        return None
    return dict(name=name, yuv=yuv, w=w, h=h, fps=fps, frames=len(files))


def clip_from_refs(name, w, h, fps, work, tile=None):
    """A clip from the reference *photographs* in `out/`.

    Everything else here is content this project rendered, and a renderer's own
    output is the easy case: large flat areas, no sensor noise, no film grain.
    An encoder that has just learned to code flat regions cheaply looks far
    better on it than it deserves — on the `frog` clip the native HEVC encoder
    appears to beat x265 by 1.4-2.9x, which is not a believable result.

    These are photographs of finished models, taken with a camera. They are the
    harder distribution, and they are not self-generated. `tile` repeats the
    clip in a grid to reach a higher resolution while keeping the detail density
    of the original, which upscaling would not.
    """
    srcs = sorted(ROOT.glob("out/*-ref.jpg"))
    if not srcs:
        return None
    yuv = work / f"{name}.yuv"
    if yuv.exists():
        yuv.unlink()
    with yuv.open("wb") as fh:
        for src in srcs:
            vf = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
            r = sh(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-i", str(src),
                    "-vf", vf, "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"],
                   capture_output=True)
            if r.returncode != 0:
                return None
            fh.write(r.stdout if isinstance(r.stdout, bytes) else r.stdout.encode())
    frames = len(srcs)
    if tile:
        tw, th = tile
        big = work / f"{name}.t{tw}x{th}.yuv"
        r = ffmpeg(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}",
                    "-r", str(fps), "-i", str(yuv), "-vf", f"tile={tw}x{th}",
                    "-frames:v", str(frames // (tw * th)),
                    "-pix_fmt", "yuv420p", "-f", "rawvideo", str(big)])
        if r.returncode != 0 or not big.exists():
            return None
        return dict(name=name, yuv=big, w=w * tw, h=h * th,
                    fps=fps, frames=frames // (tw * th))
    return dict(name=name, yuv=yuv, w=w, h=h, fps=fps, frames=frames)


def clip_synthetic(name, kind, w, h, n, fps, work):
    """Procedural clips that isolate one encoder behaviour each.

    `static`  a mostly-still frame with a small moving element — the case where
              inter prediction wins by the largest margin, and the one rendered
              turntables and UI captures actually look like.
    `motion`  a full-frame pan, so every block moves and intra loses less.
    `noise`   per-frame film-grain-like noise over a gradient, standing in for
              an under-converged path-traced render.
    """
    yuv = work / f"{name}.yuv"
    if kind == "static":
        src = [
            "-f", "lavfi", "-i",
            f"testsrc2=size={w}x{h}:rate={fps}:duration={n/fps}",
            "-vf", "hue=s=0.6,boxblur=2:1",
        ]
    elif kind == "motion":
        src = [
            "-f", "lavfi", "-i",
            f"mandelbrot=size={w}x{h}:rate={fps}",
            "-frames:v", str(n),
        ]
    elif kind == "noise":
        src = [
            "-f", "lavfi", "-i",
            f"gradients=size={w}x{h}:rate={fps}:duration={n/fps}",
            "-vf", "noise=alls=28:allf=t+u",
        ]
    else:
        raise ValueError(kind)
    r = ffmpeg([*src, "-frames:v", str(n), "-pix_fmt", "yuv420p",
                "-f", "rawvideo", str(yuv)])
    if r.returncode != 0:
        print(r.stderr[-800:], file=sys.stderr)
        return None
    return dict(name=name, yuv=yuv, w=w, h=h, fps=fps, frames=n)


# --------------------------------------------------------------------------
# Quality
# --------------------------------------------------------------------------


def decode_to_yuv(path, clip, work):
    dec = work / (Path(path).stem + ".dec.yuv")
    r = ffmpeg(["-i", str(path), "-pix_fmt", "yuv420p", "-f", "rawvideo", str(dec)])
    if r.returncode != 0 or not dec.exists() or dec.stat().st_size == 0:
        return None, r.stderr[-500:]
    return dec, None


def raw_in(clip, path):
    return ["-f", "rawvideo", "-pix_fmt", "yuv420p",
            "-s", f"{clip['w']}x{clip['h']}", "-r", str(clip["fps"]), "-i", str(path)]


def measure_quality(dec, clip, work):
    """PSNR and VMAF of `dec` against the reference clip."""
    out = {}
    r = sh(["ffmpeg", "-hide_banner", "-nostdin",
            *raw_in(clip, dec), *raw_in(clip, clip["yuv"]),
            "-lavfi", "[0:v][1:v]psnr", "-f", "null", "-"])
    m = re.search(r"average:([0-9.]+|inf)", r.stderr)
    if m:
        out["psnr"] = float("inf") if m.group(1) == "inf" else float(m.group(1))

    r = sh(["ffmpeg", "-hide_banner", "-nostdin",
            *raw_in(clip, dec), *raw_in(clip, clip["yuv"]),
            "-lavfi", "[0:v][1:v]ssim", "-f", "null", "-"])
    m = re.search(r"All:([0-9.]+)", r.stderr)
    if m:
        out["ssim"] = float(m.group(1))

    log = work / "vmaf.json"
    r = sh(["ffmpeg", "-hide_banner", "-nostdin",
            *raw_in(clip, dec), *raw_in(clip, clip["yuv"]),
            "-lavfi",
            f"[0:v][1:v]libvmaf=log_fmt=json:log_path={log}",
            "-f", "null", "-"])
    if log.exists():
        try:
            out["vmaf"] = json.loads(log.read_text())["pooled_metrics"]["vmaf"]["mean"]
        except Exception:
            pass
        log.unlink(missing_ok=True)
    return out


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------


def cpu_seconds(cmd, repeats):
    """Minimum (user + sys) CPU seconds over `repeats` runs.

    Wall clock is not measurable on a shared machine: the ffmpeg baselines here
    moved 5x between runs minutes apart at a load average of 50 on 14 cores.
    CPU time is what survives that, and the minimum of several runs is what
    survives frequency scaling and cache contention on top of it. It measures
    total work rather than elapsed time, so a threaded encoder is not flattered
    by having cores available — which is the honest comparison for a
    single-threaded one.
    """
    best = None
    for _ in range(max(1, repeats)):
        r = subprocess.run(["/usr/bin/time", "-p", "/bin/sh", "-c",
                            " ".join(shlex.quote(c) for c in cmd) + " >/dev/null 2>&1"],
                           capture_output=True, text=True)
        u = re.search(r"^user\s+([0-9.]+)", r.stderr, re.M)
        sy = re.search(r"^sys\s+([0-9.]+)", r.stderr, re.M)
        if u and sy:
            t = float(u.group(1)) + float(sy.group(1))
            best = t if best is None else min(best, t)
    return best


def load_average():
    try:
        return os.getloadavg()[0]
    except OSError:
        return None


def run_native(clip, encoder, qp, work, repeats=1):
    out = work / f"{clip['name']}.{encoder}.qp{qp}.mp4"
    cmd = [str(BIN), "--input", str(clip["yuv"]), "--out", str(out),
           "--size", f"{clip['w']}x{clip['h']}", "--fps", str(clip["fps"]),
           "--encoder", encoder, "--qp", str(qp)]
    t0 = time.time()
    r = sh(cmd)
    wall = time.time() - t0
    if r.returncode != 0:
        return None, (r.stderr or r.stdout)[-500:]
    try:
        meta = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return None, r.stdout[-300:]
    cpu = cpu_seconds(cmd, repeats - 1) if repeats > 1 else None
    return dict(path=out, bytes=meta["bytes"], encode_s=meta["encode_ms"] / 1000.0,
                wall_s=wall, cpu_s=cpu), None


def run_ffmpeg(clip, label, codec_args, work, ext="mp4", repeats=1):
    out = work / f"{clip['name']}.{label}.{ext}"
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y",
           *raw_in(clip, clip["yuv"]), *codec_args, str(out)]
    t0 = time.time()
    r = ffmpeg([*raw_in(clip, clip["yuv"]), *codec_args, str(out)])
    wall = time.time() - t0
    if r.returncode != 0 or not out.exists():
        return None, r.stderr[-600:]
    cpu = cpu_seconds(cmd, repeats - 1) if repeats > 1 else None
    return dict(path=out, bytes=out.stat().st_size, encode_s=wall, wall_s=wall,
                cpu_s=cpu), None


# Six points per curve, spanning roughly 45 dB down to 33 dB on photographic
# content. BD-rate fits a cubic, so four is the minimum and the fit is only
# trustworthy where the two curves' quality ranges overlap.
NATIVE_QPS = (20, 24, 28, 32, 36, 40)
FFMPEG_CRFS = (16, 21, 26, 31, 36, 41)


def arms(clip):
    """(label, kind, spec) for every encoder configuration under test."""
    a = []
    # --- native Rust ---
    a.append(("native h264 I_PCM", "native", ("h264-pcm", 26)))
    a.append(("native hevc I_PCM", "native", ("hevc-pcm", 26)))
    for qp in NATIVE_QPS:
        a.append((f"native h264 intra qp{qp}", "native", ("h264-native", qp)))
    for qp in NATIVE_QPS:
        a.append((f"native hevc intra (pred-only) qp{qp}", "native", ("hevc-intra", qp)))
        a.append((f"native hevc intra +residual qp{qp}", "native",
                  ("hevc-intra-residual", qp)))

    # --- ffmpeg all-intra: the like-for-like comparison ---
    for crf in FFMPEG_CRFS:
        a.append((f"x264 all-intra crf{crf}", "ffmpeg",
                  ["-c:v", "libx264", "-preset", "medium", "-g", "1",
                   "-crf", str(crf), "-pix_fmt", "yuv420p"]))
        a.append((f"x265 all-intra crf{crf}", "ffmpeg",
                  ["-c:v", "libx265", "-preset", "medium", "-x265-params",
                   f"keyint=1:crf={crf}:log-level=none", "-tag:v", "hvc1",
                   "-pix_fmt", "yuv420p"]))

    # --- ffmpeg all-intra tuned for PSNR: the fair PSNR comparison ---
    # The defaults above optimise for how the picture looks, not for how close
    # it is; on PSNR that costs them about 19% here. Neither native encoder has
    # any psychovisual tooling, so this is the like-for-like arm.
    for crf in FFMPEG_CRFS:
        a.append((f"x264 all-intra psnr crf{crf}", "ffmpeg",
                  ["-c:v", "libx264", "-preset", "medium", "-tune", "psnr",
                   "-g", "1", "-crf", str(crf), "-pix_fmt", "yuv420p"]))
        a.append((f"x265 all-intra psnr crf{crf}", "ffmpeg",
                  ["-c:v", "libx265", "-preset", "medium", "-tune", "psnr",
                   "-x265-params", f"keyint=1:crf={crf}:log-level=none",
                   "-tag:v", "hvc1", "-pix_fmt", "yuv420p"]))

    # --- ffmpeg default GOP: what you would actually ship ---
    for crf in FFMPEG_CRFS:
        a.append((f"x264 crf{crf}", "ffmpeg",
                  ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                   "-pix_fmt", "yuv420p"]))
        a.append((f"x265 crf{crf}", "ffmpeg",
                  ["-c:v", "libx265", "-preset", "medium", "-x265-params",
                   f"crf={crf}:log-level=none", "-tag:v", "hvc1",
                   "-pix_fmt", "yuv420p"]))

    # --- VideoToolbox (hardware) ---
    px = clip["w"] * clip["h"] * clip["fps"]
    for mbps in (4, 10, 25):
        bitrate = f"{mbps}M"
        a.append((f"h264_vt {bitrate}", "ffmpeg",
                  ["-c:v", "h264_videotoolbox", "-b:v", bitrate,
                   "-pix_fmt", "yuv420p"]))
        a.append((f"hevc_vt {bitrate}", "ffmpeg",
                  ["-c:v", "hevc_videotoolbox", "-b:v", bitrate,
                   "-tag:v", "hvc1", "-pix_fmt", "yuv420p"]))
    # src/videotoolbox.rs before and after the retune, at matched bitrate so the
    # difference lands in the quality column rather than the size column.
    for mbps in (4, 10):
        a.append((f"hevc_vt OLD realtime+speed g=2s {mbps}M", "ffmpeg",
                  ["-c:v", "hevc_videotoolbox", "-b:v", f"{mbps}M",
                   "-realtime", "1", "-prio_speed", "1",
                   "-g", str(clip["fps"] * 2), "-tag:v", "hvc1", "-pix_fmt", "yuv420p"]))
        a.append((f"hevc_vt NEW export g=5s {mbps}M", "ffmpeg",
                  ["-c:v", "hevc_videotoolbox", "-b:v", f"{mbps}M",
                   "-realtime", "0", "-prio_speed", "0",
                   "-g", str(clip["fps"] * 5), "-tag:v", "hvc1", "-pix_fmt", "yuv420p"]))

    # What the new keyframe_interval / preset knobs on VideoOptions buy on x265.
    for g in (clip["fps"] * 1, clip["fps"] * 2, clip["fps"] * 5, 250):
        a.append((f"x265 crf23 keyint={g}", "ffmpeg",
                  ["-c:v", "libx265", "-preset", "medium", "-x265-params",
                   f"crf=23:keyint={g}:min-keyint={g}:log-level=none",
                   "-tag:v", "hvc1", "-pix_fmt", "yuv420p"]))
    return a


# --------------------------------------------------------------------------


def bench(clip, work, rows, repeats=1):
    ref_bytes = clip["w"] * clip["h"] * 3 // 2 * clip["frames"]
    # VMAF is an SVR fit, not a distance: on moving content it predicts under 100
    # even for a bit-identical decode. Score the reference against itself first so
    # every arm below is read against the ceiling that is actually reachable.
    ceiling = measure_quality(clip["yuv"], clip, work).get("vmaf")
    clip["vmaf_ceiling"] = ceiling
    print(f"\n=== {clip['name']}  {clip['w']}x{clip['h']}  "
          f"{clip['frames']} frames @ {clip['fps']}fps  "
          f"(raw yuv420p = {ref_bytes/1e6:.1f} MB"
          + (f", VMAF ceiling {ceiling:.2f})" if ceiling else ")") + " ===",
          flush=True)

    for label, kind, spec in arms(clip):
        if kind == "native":
            res, err = run_native(clip, spec[0], spec[1], work, repeats)
        else:
            res, err = run_ffmpeg(clip, re.sub(r"[^a-z0-9]+", "_", label.lower()),
                                  spec, work, repeats=repeats)
        if res is None:
            print(f"  {label:46s} FAILED  {err.splitlines()[-1] if err else ''}")
            rows.append(dict(clip=clip["name"], arm=label, ok=False))
            continue

        dec, derr = decode_to_yuv(res["path"], clip, work)
        q = measure_quality(dec, clip, work) if dec else {}
        if dec:
            dec.unlink(missing_ok=True)

        mbps = res["bytes"] * 8 / (clip["frames"] / clip["fps"]) / 1e6
        row = dict(clip=clip["name"], arm=label, ok=True,
                   bytes=res["bytes"], mbps=mbps,
                   vs_raw=ref_bytes / res["bytes"],
                   encode_s=res["encode_s"],
                   cpu_s=res.get("cpu_s"),
                   fps_enc=clip["frames"] / max(res["encode_s"], 1e-9),
                   psnr=q.get("psnr"), ssim=q.get("ssim"), vmaf=q.get("vmaf"),
                   vmaf_ceiling=clip.get("vmaf_ceiling"),
                   decoded=bool(dec))
        rows.append(row)
        psnr = row["psnr"]
        psnr_s = "  n/a " if psnr is None else ("  inf " if psnr == float("inf")
                                               else f"{psnr:6.2f}")
        vmaf_s = "  n/a " if row["vmaf"] is None else f"{row['vmaf']:6.2f}"
        print(f"  {label:46s} {res['bytes']/1e6:8.2f} MB  {mbps:8.1f} Mb/s  "
              f"PSNR {psnr_s}  VMAF {vmaf_s}  {row['fps_enc']:7.1f} fps"
              + ("" if dec else "  [DECODE FAILED]"), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="frog")
    ap.add_argument("--out", default="out/codec-bench")
    ap.add_argument("--keep", action="store_true", help="keep encoded files")
    ap.add_argument("--repeats", type=int, default=1,
                    help="re-run each encode this many times and keep the minimum "
                         "CPU time; 1 skips CPU timing entirely")
    args = ap.parse_args()

    if not BIN.exists():
        sys.exit(f"missing {BIN}\n  cargo build --release --example codec_bench "
                 f"--features native-codec")

    work = ROOT / args.out
    work.mkdir(parents=True, exist_ok=True)

    wanted = args.clip
    clips = []
    if wanted in ("frog", "all"):
        c = clip_from_pngs("out/origami-frog-*.png", "frog", 24, work)
        if c:
            clips.append(c)
    if wanted in ("static", "all"):
        clips.append(clip_synthetic("static", "static", 1280, 720, 48, 24, work))
    if wanted in ("motion", "all"):
        clips.append(clip_synthetic("motion", "motion", 1280, 720, 48, 24, work))
    if wanted in ("noise", "all"):
        clips.append(clip_synthetic("noise", "noise", 1280, 720, 48, 24, work))
    if wanted in ("photo", "all"):
        clips.append(clip_from_refs("photo", 1280, 720, 10, work))
    if wanted in ("photo4k",):
        clips.append(clip_from_refs("photo4k", 1280, 720, 10, work, tile=(3, 3)))
    clips = [c for c in clips if c]
    if not clips:
        sys.exit(f"no clips built for --clip {wanted}")

    la = load_average()
    if la is not None and la > os.cpu_count() * 0.5:
        print(f"WARNING: load average {la:.1f} on {os.cpu_count()} cores. Wall-clock "
              f"timings below are meaningless; use --repeats to get CPU time, and "
              f"read the quality columns, which are deterministic.", file=sys.stderr)

    rows = []
    for c in clips:
        bench(c, work, rows, repeats=args.repeats)

    csv = work / "results.csv"
    cols = ["clip", "arm", "ok", "bytes", "mbps", "vs_raw", "encode_s", "cpu_s",
            "fps_enc", "psnr", "ssim", "vmaf", "vmaf_ceiling", "decoded"]
    with csv.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(c) is None else str(r.get(c, ""))
                             for c in cols) + "\n")
    print(f"\nwrote {csv}")

    # --- BD-rate: one number per curve pair, instead of eyeballing two points ---
    for c in clips:
        header = False
        for metric in ("psnr", "ssim", "vmaf"):
            for label, bd in bd_table(rows, c["name"], metric):
                if bd is None:
                    continue
                if not header:
                    print(f"\nBD-rate on {c['name']} "
                          f"(negative = native needs fewer bits at equal quality):")
                    header = True
                print(f"  {metric.upper():5s}  {label:34s} {bd:+7.1f}%")
    if la is not None:
        print(f"\nload average during the run: {la:.1f} on {os.cpu_count()} cores")

    if not args.keep:
        for p in work.glob("*.mp4"):
            p.unlink()
        for p in work.glob("*.yuv"):
            p.unlink()


if __name__ == "__main__":
    main()
