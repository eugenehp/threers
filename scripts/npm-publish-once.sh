#!/usr/bin/env bash
# Publish the npm package in the current directory, unless that exact version
# is already on the registry.
#
# npm refuses to replace a published version, so a release that stopped
# halfway (one platform package failed, the network dropped) could not be
# rerun: everything before the failure would now error. This makes a rerun
# carry on from where it stopped. Any other failure still fails.
#
# Usage: (cd <package dir> && scripts/npm-publish-once.sh [npm publish args])
# Default args: --access public. With NPM_ONCE_STATUS=<file>, writes
# "published" or "skipped" there, for a caller that reports which.
set -euo pipefail

status() { [[ -z "${NPM_ONCE_STATUS:-}" ]] || echo "$1" > "$NPM_ONCE_STATUS"; }

name=$(node -p 'require("./package.json").name')
version=$(node -p 'require("./package.json").version')

# `npm view` prints the version when it exists and fails with E404 when it does
# not; anything else (no network, a bad token) is an error, not "unpublished".
if out=$(npm view "$name@$version" version 2>&1); then
  if [[ "$out" == "$version" ]]; then
    echo "$name@$version is already on npm; skipping"
    status skipped
    exit 0
  fi
elif ! grep -q E404 <<<"$out"; then
  echo "could not ask npm about $name@$version:" >&2
  echo "$out" >&2
  exit 1
fi

if (( $# == 0 )); then
  set -- --access public
fi
echo "+ npm publish $* ($name@$version)"
# The registry can lag a fresh publish by minutes, so `npm view` above may miss
# a version that is up; npm then refuses the upload as a duplicate, which is
# the outcome this script exists for, not a failure.
log=$(mktemp)
trap 'rm -f "$log"' EXIT
if npm publish "$@" 2>&1 | tee "$log"; then
  status published
  exit 0
fi
if grep -q "cannot publish over the previously published versions" "$log"; then
  echo "$name@$version is already on npm; skipping"
  status skipped
  exit 0
fi
exit 1
