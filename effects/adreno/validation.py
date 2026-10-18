"""Strict result boundary for the bounded 576 x 576 F32 capability example."""
import math
import re
import struct

ROWS = COLS = 576
TENSOR_BYTES = ROWS * COLS * 4
PATTERNS = ("ones", "alternating", "basis-last")
ABS_TOL = 2e-5
REL_TOL = 2e-4
RELATIVE_REPORT_FLOOR = 1e-30


class ValidationError(ValueError):
    pass


def parse_bits(stdout, target, pattern):
    """Require one complete, finite result; never accept a truncated prefix."""
    if target not in ("CPU", "ADRENO") or pattern not in range(len(PATTERNS)):
        raise ValidationError("unknown target or input pattern")
    prefix = target + "_GEMV_BITS"
    lines = [line for line in stdout.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise ValidationError(f"expected exactly one {prefix} result, found {len(lines)}")
    match = re.fullmatch(re.escape(prefix) + r" pattern=([0-2]) values=(.*)", lines[0])
    if match is None or int(match[1]) != pattern:
        raise ValidationError(f"{prefix}: malformed or mismatched pattern")
    words = match[2].split()
    if len(words) != ROWS:
        raise ValidationError(f"{prefix}: expected {ROWS} rows, found {len(words)}")
    bits = []
    for row, word in enumerate(words):
        if re.fullmatch(r"[0-9]+", word) is None or len(word) > 10:
            raise ValidationError(f"{prefix}: invalid U32 at row {row}")
        value = int(word)
        if value > 0xFFFFFFFF:
            raise ValidationError(f"{prefix}: out-of-range U32 at row {row}")
        bits.append(value)
    values = struct.unpack("<" + "f" * ROWS, struct.pack("<" + "I" * ROWS, *bits))
    for row, value in enumerate(values):
        if not math.isfinite(value):
            raise ValidationError(f"{prefix}: nonfinite output at row {row}")
    return values


def within_tolerance(error, reference):
    return error <= ABS_TOL + REL_TOL * abs(reference)


def compare_values(cpu, gpu):
    """Compare every row; relative error is diagnostic, not a second criterion."""
    if len(cpu) != ROWS or len(gpu) != ROWS:
        raise ValidationError(f"both outputs must have exactly {ROWS} rows")
    failed = []
    max_abs = max_rel = 0.0
    worst_abs_row = worst_rel_row = 0
    for row, (reference, actual) in enumerate(zip(cpu, gpu)):
        if not math.isfinite(reference) or not math.isfinite(actual):
            raise ValidationError(f"nonfinite output at row {row}")
        error = abs(actual - reference)
        relative = error / max(abs(reference), RELATIVE_REPORT_FLOOR)
        if error > max_abs:
            max_abs, worst_abs_row = error, row
        if relative > max_rel:
            max_rel, worst_rel_row = relative, row
        if not within_tolerance(error, reference):
            failed.append(row)
    return dict(acceptance_met=not failed, compared_rows=ROWS, failed_rows=failed,
                max_absolute_error=max_abs, max_relative_error=max_rel,
                worst_absolute_row=worst_abs_row, worst_relative_row=worst_rel_row,
                absolute_tolerance=ABS_TOL, relative_tolerance=REL_TOL,
                relative_report_floor=RELATIVE_REPORT_FLOOR)


def parse_metadata(stdout):
    """Read actual OpenCL identities and status fields, rejecting duplicates."""
    required = ("platform", "platform_vendor", "platform_version", "vendor", "device",
                "device_version", "driver_version", "device_type", "gpu_required",
                "cpu_fallback", "enqueue_ndrange_status", "global", "local",
                "kernel_event_execution_status", "blocking_readback",
                "tensor_bytes", "rows", "cols", "pattern", "kernel_build_status")
    fields = {}
    for line in stdout.splitlines():
        if line.startswith("ADRENO_GEMV_BITS"):
            continue
        starts = list(re.finditer(r"(?:^| )([a-z_]+)=", line))
        for index, match in enumerate(starts):
            key = match[1]
            if key not in required:
                continue
            end = starts[index + 1].start() if index + 1 < len(starts) else len(line)
            value = line[match.end():end].strip()
            if key in fields or not value:
                raise ValidationError(f"duplicate or empty OpenCL field: {key}")
            fields[key] = value
    missing = set(required) - fields.keys()
    if missing:
        raise ValidationError("missing OpenCL metadata: " + ", ".join(sorted(missing)))
    if not re.fullmatch(r"[0-9]+", fields["device_type"]):
        raise ValidationError("invalid OpenCL device type")
    device_type = int(fields["device_type"])
    if not device_type & 4 or device_type & 2:
        raise ValidationError("OpenCL device is not GPU-only")
    expected = {"gpu_required": "true", "cpu_fallback": "false",
                "enqueue_ndrange_status": "0", "global": "18432", "local": "256",
                "kernel_event_execution_status": "0", "blocking_readback": "success",
                "tensor_bytes": str(TENSOR_BYTES), "rows": str(ROWS), "cols": str(COLS),
                "kernel_build_status": "0"}
    for key, value in expected.items():
        if fields[key] != value:
            raise ValidationError(f"unaccepted OpenCL status: {key}={fields[key]}")
    if fields["pattern"] not in tuple(str(index) for index in range(len(PATTERNS))):
        raise ValidationError("invalid tensor input pattern")
    return fields


def compare_outputs(cpu_stdout, gpu_stdout, pattern):
    metadata = parse_metadata(gpu_stdout)
    if metadata["pattern"] != str(pattern):
        raise ValidationError("OpenCL metadata pattern does not match the requested case")
    result = compare_values(parse_bits(cpu_stdout, "CPU", pattern),
                            parse_bits(gpu_stdout, "ADRENO", pattern))
    result["opencl"] = metadata
    return result
