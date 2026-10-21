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
import struct
import sys
from pathlib import Path


# gen.bend emits both a numeric logit bit pattern and a rendered F32 value.
# Requiring both prevents truncated rows from being counted as valid samples.
TOKEN = re.compile(r"^\s+id\s+\d+\s+logit\s+(?P<bits>\d+)\s+(?P<logit>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+t=(?P<stamp>\d+(?:\.\d+)?)s\s*$", re.ASCII)
LOAD = re.compile(r"^weights loaded in\s+(\d+(?:\.\d+)?)s\s*$", re.ASCII)
# start2 emits positive-length decimal Nat IDs separated by single spaces.
# The native show_nats printer also emits one optional trailing space.
PROMPT = re.compile(r"^prompt ids: [0-9]+(?: [0-9]+)* ?$", re.ASCII)


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
    seen_prompt = False
    seen_text_footer = False
    for line in output.splitlines(keepends=True):
        # Decoded text may itself contain timestamp-looking lines. Native
        # IO.print terminates the footer line; an unterminated last line
        # is partial captured stdout, not proof generation completed.
        if line.startswith("text: "):
            if seen_prompt and not line.endswith("\n"):
                raise ValueError("incomplete generated output: unterminated decoded-text footer")
            seen_text_footer = True
            break
        # A malformed near-header must not bypass the full-run completion
        # requirement by being treated as unrelated stdout noise.
        if (line.lstrip().startswith("prompt ids:") or
                re.match(r"^\s*prompt\s+ids\s*:", line, re.ASCII)) and not line.startswith("prompt ids:"):
            raise ValueError("malformed prompt header")
        if line.startswith("prompt ids:"):
            # Native gen.bend emits exactly one header, before any token row.
            # Repeated/late headers must not blend distinct generation traces.
            if seen_prompt:
                raise ValueError("duplicate prompt header")
            if times:
                raise ValueError("prompt header appears after generated token output")
            if not PROMPT.fullmatch(line.rstrip("\r\n")):
                raise ValueError("malformed prompt header")
            seen_prompt = True
        # Native output prints the load clock at column zero. A leading
        # whitespace character must not hide a corrupt load receipt.
        if (line.lstrip().startswith("weights loaded in") or
                re.match(r"^\s*weights\s+loaded\s+in\b", line, re.ASCII)) and not line.startswith("weights loaded in"):
            raise ValueError("malformed load duration")
        if re.match(r"^weights loaded in\b", line):
            match = LOAD.fullmatch(line)
            if not match:
                raise ValueError("malformed load duration")
            if load_s is not None:
                raise ValueError("duplicate load duration")
            if seen_prompt or times:
                raise ValueError("load duration appears after prompt or token output")
            load_s = finite_seconds(match.group(1), "load duration")
        # A malformed generated-token row must invalidate the capture, not
        # silently disappear from the distribution or token count.
        if re.match(r"^\s*id\b", line):
            match = TOKEN.fullmatch(line)
            if not match:
                raise ValueError("malformed generated token row")
            # gen.bend renders the chosen F32's raw bits with U32.show.
            # A larger integer cannot be a genuine emission, even when the
            # row has an otherwise parseable timestamp.
            bits = match.group("bits")
            if len(bits) > 10 or int(bits) > 0xFFFFFFFF:
                raise ValueError("invalid generated token logit bits")
            # IEEE-754 F32 exponent 255 encodes NaN or infinity, not a
            # finite scalar. A plausible decimal field must not disguise
            # a nonfinite raw bit pattern into a latency sample.
            if int(bits) & 0x7F800000 == 0x7F800000:
                raise ValueError("nonfinite generated token logit bits")
            # The producer renders an F32, never an overflowing Python float.
            # For instance, 1e309 or 3.5e38 must not silently count as a
            # genuine generated token with a valid-looking timestamp.
            rendered = float(match.group("logit"))
            if not math.isfinite(rendered):
                raise ValueError("nonfinite generated token logit")
            try:
                display_bits = struct.unpack("<I", struct.pack("<f", rendered))[0]
            except OverflowError as error:
                raise ValueError("generated token logit outside F32 range") from error
            if display_bits != int(bits):
                raise ValueError("generated token logit bits/display mismatch")
            times.append(finite_seconds(match.group("stamp"), "token timestamp"))

    if not times:
        raise ValueError("no generated token timestamps found (expected '  id ... t=...s')")
    # Full generator output has a prompt header and a final decoded-text footer.
    # Never report partial stdout as a completed inference.
    if seen_prompt and not seen_text_footer:
        raise ValueError("incomplete generated output: missing decoded-text footer")
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
