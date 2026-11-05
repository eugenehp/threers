#!/usr/bin/env python3
"""Rebuild a game when its source changes and restart it where it was.

    python3 scripts/watch-game.py -p threers-ecs --example race --features physics,window,audio

Arguments are `cargo run`'s. The game keeps running while a change builds; only
once the build has worked is it stopped and the new one started, with
THREERS_HOT_RESUME set so that a game with the `HotReload` plugin picks up its
saved state. A build that fails prints its errors and leaves the game running.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WATCH = [ROOT / "src", ROOT / "crates"]
EXTENSIONS = {".rs", ".wgsl", ".toml", ".json", ".glb", ".gltf", ".ogg", ".wav", ".png"}
SKIP = ("target", "node_modules", ".venv", "dist")


def newest():
    stamp = 0.0
    for base in WATCH:
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in SKIP and not d.startswith(".")]
            for name in filenames:
                if Path(name).suffix in EXTENSIONS:
                    try:
                        stamp = max(stamp, os.stat(os.path.join(dirpath, name)).st_mtime)
                    except OSError:
                        pass
    return stamp


def build(args):
    started = time.time()
    done = subprocess.run(["cargo", "build", *args], cwd=ROOT)
    ok = done.returncode == 0
    print(f"[watch] build {'done' if ok else 'FAILED'} in {time.time() - started:.1f} s", flush=True)
    return ok


def start(args, resume):
    env = dict(os.environ)
    if resume:
        env["THREERS_HOT_RESUME"] = "1"
    print(f"[watch] starting{' where it was' if resume else ''}", flush=True)
    return subprocess.Popen(["cargo", "run", "-q", *args], cwd=ROOT, env=env)


def stop(game):
    if game and game.poll() is None:
        game.send_signal(signal.SIGTERM)
        try:
            game.wait(timeout=5)
        except subprocess.TimeoutExpired:
            game.kill()


def main(args):
    if not args:
        sys.exit(__doc__)
    if not build(args):
        sys.exit(1)
    game = start(args, resume=False)
    seen = newest()
    try:
        while True:
            time.sleep(0.5)
            stamp = newest()
            if stamp <= seen:
                if game.poll() is not None:
                    print("[watch] the game ended; waiting for a change", flush=True)
                    game = None
                    while newest() <= seen:
                        time.sleep(0.5)
                    stamp = newest()
                else:
                    continue
            seen = stamp
            print("[watch] change seen, building", flush=True)
            if build(args):
                stop(game)
                game = start(args, resume=True)
    except KeyboardInterrupt:
        stop(game)


if __name__ == "__main__":
    main(sys.argv[1:])
