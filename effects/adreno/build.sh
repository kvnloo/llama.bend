#!/bin/sh
# Emit and cross-compile only. The Android executables must run through ADB, not proot.
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export BEND_NO_TELEMETRY=1
exec python3 "$HERE/run-device.py" build
