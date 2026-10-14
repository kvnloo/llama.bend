#!/usr/bin/env python3
"""Report generation latency from existing llama.bend demo/gen.bend output.

Usage:
  .scratch/gen data/smollm2-135m 'Once upon a time' 128 > run.log
  python3 scripts/latency_report.py run.log
  python3 scripts/latency_report.py --json < run.log

The t= stamps have 1 ms resolution and are measured from *after* model loading
and prompt tokenization. They include per-token printing overhead, so these
figures are diagnostic, not a substitute for controlled end-to-end benchmarks.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path


TOKEN = re.compile(r"^\s+id\s+\d+\s+logit\b.*\bt=(\d+(?:\.\d+)?)s\s*$")
LOAD = re.compile(r"^weights loaded in\s+(\d+(?:\.\d+)?)s\s*$")


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile (well-defined even for a short generation)."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def slope(values: list[float]) -> float | None:
    """Least-squares change in the per-token interval per generated position."""
    n = len(values)
    if n < 2:
        return None
    center = (n - 1) / 2
    mean = sum(values) / n
    return sum((i - center) * (v - mean) for i, v in enumerate(values)) / sum(
        (i - center) ** 2 for i in range(n)
    )


def finite_seconds(raw: str, source: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"nonfinite {source}")
    return value


def report(output: str) -> dict[str, float | int | None]:
    times: list[float] = []
    load_s = None
    for line in output.splitlines():
        if match := LOAD.match(line):
            load_s = finite_seconds(match.group(1), "load duration")
        if match := TOKEN.match(line):
            times.append(finite_seconds(match.group(1), "token timestamp"))

    if not times:
        raise ValueError("no generated token timestamps found (expected '  id ... t=...s')")
    intervals_ms = [(b - a) * 1000 for a, b in zip(times, times[1:])]
    if any(not math.isfinite(value) for value in intervals_ms):
        raise ValueError("nonfinite decode interval")
    if any(value < 0 for value in intervals_ms):
        raise ValueError("token timestamps are not monotonic")

    per_position_slope = slope(intervals_ms)
    metrics = {
        "load_s": load_s,
        "tokens": len(times),
        "first_generated_ms_since_generation_start": round(times[0] * 1000, 3),
        "decode_mean_ms": round(sum(intervals_ms) / len(intervals_ms), 3) if intervals_ms else None,
        "decode_p50_ms": round(percentile(intervals_ms, 0.5), 3) if intervals_ms else None,
        "decode_p95_ms": round(percentile(intervals_ms, 0.95), 3) if intervals_ms else None,
        "decode_slope_ms_per_position": round(per_position_slope, 5) if per_position_slope is not None else None,
    }
    if any(isinstance(value, float) and not math.isfinite(value) for value in metrics.values()):
        raise ValueError("nonfinite latency metric")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default="-", help="captured gen output (default: stdin)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    try:
        raw = sys.stdin.read() if args.output == "-" else Path(args.output).read_text(encoding="utf-8")
        result = report(raw)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    if args.json:
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    else:
        print(" ".join(f"{key}={value}" for key, value in result.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
