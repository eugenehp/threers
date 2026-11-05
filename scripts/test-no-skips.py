#!/usr/bin/env python3
"""Run `cargo test` and fail if a test only passed by not testing anything.

Most suites here skip themselves when a tool, a GPU or a set of conformance
vectors is missing, and a skipped test reports `ok`. So does a test target
whose inner `#![cfg]` left it empty. A green `cargo test` on a machine without
those things proves very little, and nothing in its summary says so.

    scripts/test-no-skips.py -p threers-codec --release --features parallel
    scripts/test-no-skips.py --test av1_conformance -p threers-codec
    scripts/test-no-skips.py --check saved-output.log

Arguments go to `cargo test` unchanged; `--nocapture` is added, since the skip
messages are what this reads. A skip that is right for the machine belongs in
`scripts/allowed-skips.txt`, one Python regex per line, matched against
`<target>: <message>`. An empty target reads as `<target>: running 0 tests`.
`THREERS_ALLOWED_SKIPS` names a second such file, for what only one kind of
machine lacks: `scripts/allowed-skips-ci.txt` on a hosted runner.

A skip has to print for this to see it: `SKIP`, or `skipping` as a word.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

ALLOWED = Path(__file__).with_name("allowed-skips.txt")
SKIP = re.compile(r"(?:^|[\s:])(?:SKIP\b|[Ss]kipping\b)")
TARGET = re.compile(r"^\s*Running (?:unittests )?(\S+)")
DOC_TESTS = re.compile(r"^\s*Doc-tests (\S+)")
EMPTY = re.compile(r"^running 0 tests\s*$")


def allowed_patterns():
    paths = [ALLOWED]
    if extra := os.environ.get("THREERS_ALLOWED_SKIPS"):
        paths.append(Path(extra))
    patterns = []
    for path in paths:
        lines = (line.strip() for line in path.read_text().splitlines())
        patterns += [re.compile(line) for line in lines if line and not line.startswith("#")]
    return patterns


def findings(lines):
    """`<target>: <message>` for each skip and each empty integration test."""
    target = "?"
    found = []
    for line in lines:
        line = line.rstrip("\n")
        if m := TARGET.match(line):
            target = m.group(1)
        elif m := DOC_TESTS.match(line):
            target = f"doctests {m.group(1)}"
        elif EMPTY.match(line):
            # Only `tests/` files are expected to hold tests: a library or a
            # binary with none of its own is ordinary.
            if target.startswith("tests/"):
                found.append(f"{target}: running 0 tests")
        elif SKIP.search(line):
            found.append(f"{target}: {line.strip()}")
    return found


def report(lines):
    patterns = allowed_patterns()
    found = findings(lines)
    bad = [f for f in found if not any(p.search(f) for p in patterns)]
    # The same message from every test in a file is one fact, not fifty.
    counts = {}
    for f in bad:
        counts[f] = counts.get(f, 0) + 1
    for f, n in sorted(counts.items()):
        print(f"not tested: {f}" + (f" (x{n})" if n > 1 else ""), file=sys.stderr)
    print(f"{len(found)} skipped or empty, {len(found) - len(bad)} allowed, {len(bad)} not", file=sys.stderr)
    return bool(bad)


def main(args):
    if args[:1] == ["--check"]:
        if len(args) != 2:
            sys.exit("--check takes one log file")
        with open(args[1], errors="replace") as log:
            return 1 if report(log) else 0

    if "--" in args:
        cut = args.index("--")
        cargo, harness = args[:cut], args[cut + 1:]
    else:
        cargo, harness = args, []
    cmd = ["cargo", "test", "--no-fail-fast", *cargo, "--", "--nocapture", *harness]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    lines = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        lines.append(line)
    code = proc.wait()
    skipped = report(lines)
    return code or (1 if skipped else 0)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
