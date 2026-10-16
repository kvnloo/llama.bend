"""Deterministic OpenCL C candidates for a row-major 576x576 F32 GEMV.

Public interface:
    generate(spec) -> {id, spec, source, global_size, local_size}
    enumerate_specs() -> list of canonical spec dictionaries
    SPEC_SCHEMA -> JSON Schema for complete, model-proposed spec dictionaries

``family`` is required. Rowwise control/vector4 defaults are local_size=0,
unroll=1, rows_per_item=1; cooperative defaults are lanes=4, local_size=32,
unroll=1. Returned specs contain every applicable field. Unknown/inapplicable
fields, booleans in integer fields, and unsafe combinations raise ValueError.
Global/local sizes are one-dimensional integer work-item counts; local_size=0
means the driver chooses. The ID hashes the canonical spec, source and geometry.

Control retains the baseline's four independent scalar sums and pairwise finish.
Vector4 loads four adjacent values together, but retains those same four sums;
rows_per_item shares x loads across rows at the cost of more live registers.
Unrolling preserves each accumulator's term order and trades code size/registers
for less loop overhead. Cooperative4 distributes exactly those original four
sums across lanes and retains their pairwise finish. Larger cooperative widths
shorten serial sums but change term assignment and reduction order; they require
the same independent numerical acceptance as every other candidate. All sources
disable contraction and require ordinary, non-relaxed OpenCL C 1.2 build options.
No candidate assumes anything about the values of x or the matrix weights.
"""

import hashlib
import json


_ROWS = 576
_COLS = 576
_LOCAL_SIZES = (0, 32, 64, 128, 256)
_UNROLLS = (1, 2, 4, 8)
_ROWS_PER_ITEM = (1, 2, 4)
_LANES = (4, 8, 16, 32, 64, 128)

# Exact OpenCL source embedded in the verified effects/adreno/gemv.c baseline.
_CONTROL_SOURCE = """#pragma OPENCL FP_CONTRACT OFF
__kernel void gemv(__global const float *w, __global const float *x,
                   __global float *y) {
  uint row = (uint)get_global_id(0);
  if (row >= 576u) return;
  uint base = row * 576u;
  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
  for (uint k = 0u; k < 576u; k += 4u) {
    s0 = s0 + x[k] * w[base+k];
    s1 = s1 + x[k+1u] * w[base+k+1u];
    s2 = s2 + x[k+2u] * w[base+k+2u];
    s3 = s3 + x[k+3u] * w[base+k+3u];
  }
  y[row] = (s0 + s1) + (s2 + s3);
}
"""


def _schema_variant(family, lanes=None):
    properties = {"family": {"const": family}}
    if lanes is None:
        properties.update({
            "local_size": {"type": "integer", "enum": list(_LOCAL_SIZES)},
            "unroll": {"type": "integer", "enum": list(_UNROLLS)},
            "rows_per_item": {"type": "integer", "enum": list(_ROWS_PER_ITEM)},
        })
    else:
        properties.update({
            "lanes": {"type": "integer", "const": lanes},
            "local_size": {
                "type": "integer",
                "enum": [size for size in _LOCAL_SIZES if size >= lanes],
            },
            "unroll": {
                "type": "integer",
                "enum": [value for value in _UNROLLS if lanes * value <= _COLS],
            },
        })
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


SPEC_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Adreno fixed-shape GEMV candidate",
    "description": (
        "Complete specs for generate(). Cooperative lanes must fit evenly in an "
        "explicit workgroup; an unrolled block must fit within 576 columns. "
        "No custom source, launch dimensions, math flags or input values."
    ),
    "oneOf": [
        _schema_variant("control"),
        _schema_variant("vector4"),
        *[_schema_variant("cooperative", lanes) for lanes in _LANES],
    ],
}


def _normalize(spec):
    if type(spec) is not dict:
        raise ValueError("spec must be a JSON object")
    if any(type(key) is not str for key in spec):
        raise ValueError("spec keys must be strings")
    family = spec.get("family")
    if type(family) is not str or family not in ("control", "vector4", "cooperative"):
        raise ValueError("family must be control, vector4 or cooperative")
    if family == "cooperative":
        defaults = {"lanes": 4, "local_size": 32, "unroll": 1}
    else:
        defaults = {"local_size": 0, "unroll": 1, "rows_per_item": 1}
    unknown = set(spec) - {"family", *defaults}
    if unknown:
        raise ValueError("unknown or inapplicable fields: " + ", ".join(sorted(unknown)))
    normalized = {"family": family}
    for key, default in defaults.items():
        value = spec.get(key, default)
        if type(value) is not int:
            raise ValueError(key + " must be an integer, not a boolean or string")
        normalized[key] = value
    if normalized["local_size"] not in _LOCAL_SIZES:
        raise ValueError("local_size must be 0, 32, 64, 128 or 256")
    if normalized["unroll"] not in _UNROLLS:
        raise ValueError("unroll must be 1, 2, 4 or 8")
    if family == "cooperative":
        lanes = normalized["lanes"]
        local = normalized["local_size"]
        if lanes not in _LANES:
            raise ValueError("cooperative lanes must be 4, 8, 16, 32, 64 or 128")
        if local == 0 or local < lanes or local % lanes:
            raise ValueError("cooperative local_size must be explicit and divisible by lanes")
        if lanes * normalized["unroll"] > _COLS:
            raise ValueError("cooperative unroll block exceeds the column count")
        if (_ROWS * lanes) % local:
            raise ValueError("cooperative launch would contain an incomplete workgroup")
    elif normalized["rows_per_item"] not in _ROWS_PER_ITEM:
        raise ValueError("rows_per_item must be 1, 2 or 4")
    return normalized


def _rowwise_source(spec):
    family = spec["family"]
    rows = spec["rows_per_item"]
    unroll = spec["unroll"]
    if family == "control" and rows == 1 and unroll == 1:
        return _CONTROL_SOURCE
    lines = [
        "#pragma OPENCL FP_CONTRACT OFF",
        "__kernel void gemv(__global const float *w, __global const float *x,",
        "                   __global float *y) {",
        f"  uint first = (uint)get_global_id(0) * {rows}u;",
        "  if (first >= 576u) return;",
    ]
    # 576 is divisible by every allowed rows_per_item, so the guard covers
    # every row owned by this work item, including padded work items.
    for row in range(rows):
        lines.append(f"  uint base{row} = (first + {row}u) * 576u;")
        if family == "vector4":
            lines.append(f"  float4 s{row} = (float4)(0.0f);")
        else:
            lines.append(
                f"  float s{row}_0 = 0.0f, s{row}_1 = 0.0f, "
                f"s{row}_2 = 0.0f, s{row}_3 = 0.0f;"
            )
    lines.append(f"  for (uint k = 0u; k < 576u; k += {4 * unroll}u) {{")
    for step in range(unroll):
        offset = 4 * step
        if family == "vector4":
            lines.append(f"    float4 vx{step} = vload4(0, x + k + {offset}u);")
            for row in range(rows):
                lines.append(
                    f"    s{row} = s{row} + vx{step} * "
                    f"vload4(0, w + base{row} + k + {offset}u);"
                )
        else:
            for lane in range(4):
                lines.append(f"    float vx{step}_{lane} = x[k + {offset + lane}u];")
            for row in range(rows):
                for lane in range(4):
                    lines.append(
                        f"    s{row}_{lane} = s{row}_{lane} + vx{step}_{lane} * "
                        f"w[base{row} + k + {offset + lane}u];"
                    )
    lines.append("  }")
    for row in range(rows):
        if family == "vector4":
            value = f"(s{row}.s0 + s{row}.s1) + (s{row}.s2 + s{row}.s3)"
        else:
            value = f"(s{row}_0 + s{row}_1) + (s{row}_2 + s{row}_3)"
        lines.append(f"  y[first + {row}u] = {value};")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _cooperative_source(spec):
    lanes = spec["lanes"]
    local = spec["local_size"]
    unroll = spec["unroll"]
    stride = lanes * unroll
    full = (_COLS // stride) * stride
    lines = [
        "#pragma OPENCL FP_CONTRACT OFF",
        f"__attribute__((reqd_work_group_size({local}, 1, 1)))",
        "__kernel void gemv(__global const float *w, __global const float *x,",
        "                   __global float *y) {",
        "  uint lid = (uint)get_local_id(0);",
        f"  uint lane = lid % {lanes}u;",
        f"  uint row = (uint)get_group_id(0) * {local // lanes}u + lid / {lanes}u;",
        "  uint base = row * 576u;",
        f"  __local float partial[{local}];",
        "  float sum = 0.0f;",
        f"  for (uint k = lane; k < {full}u; k += {stride}u) {{",
    ]
    for step in range(unroll):
        offset = step * lanes
        lines.append(
            f"    sum = sum + x[k + {offset}u] * w[base + k + {offset}u];"
        )
    lines.append("  }")
    # Tail guards surround only arithmetic, never barriers. The full-block loop
    # bound guarantees that all its unrolled accesses are in range for all lanes.
    for offset in range(full, _COLS, lanes):
        lines.extend([
            f"  if (lane + {offset}u < 576u) {{",
            f"    sum = sum + x[lane + {offset}u] * w[base + lane + {offset}u];",
            "  }",
        ])
    lines.extend([
        "  partial[lid] = sum;",
        "  barrier(CLK_LOCAL_MEM_FENCE);",
    ])
    if lanes == 4:
        lines.extend([
            "  if (lane == 0u) {",
            "    y[row] = (partial[lid] + partial[lid + 1u]) +",
            "             (partial[lid + 2u] + partial[lid + 3u]);",
            "  }",
        ])
    else:
        offset = lanes // 2
        while offset > 1:
            lines.extend([
                f"  if (lane < {offset}u) {{",
                f"    partial[lid] = partial[lid] + partial[lid + {offset}u];",
                "  }",
                "  barrier(CLK_LOCAL_MEM_FENCE);",
            ])
            offset //= 2
        lines.append("  if (lane == 0u) y[row] = partial[lid] + partial[lid + 1u];")
    lines.append("}")
    return "\n".join(lines) + "\n"


def generate(spec):
    """Validate a JSON-like spec and return reproducible source and launch data.

    Validation is structural, not a claim that a device supports the chosen
    workgroup or that a changed reduction passes the independent Bend oracle.
    The caller must launch exactly the returned global/local sizes.
    """
    normalized = _normalize(spec)
    local = normalized["local_size"]
    if normalized["family"] == "cooperative":
        source = _cooperative_source(normalized)
        global_size = _ROWS * normalized["lanes"]
    else:
        source = _rowwise_source(normalized)
        global_size = _ROWS // normalized["rows_per_item"]
        if local:
            global_size = ((global_size + local - 1) // local) * local
    content = {
        "spec": normalized,
        "source": source,
        "global_size": global_size,
        "local_size": local,
    }
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    identity = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return {"id": identity, **content}


def enumerate_specs():
    """Return all 202 legal combinations in a fixed, baseline-first order.

    Width 128 excludes unroll=8: that would not contain a full unrolled block.
    Explicit rowwise workgroups may need guarded padding; cooperative launches
    contain exactly 576 rows, with complete row teams in every workgroup.
    """
    specs = []
    for family in ("control", "vector4"):
        for local in _LOCAL_SIZES:
            for rows in _ROWS_PER_ITEM:
                for unroll in _UNROLLS:
                    specs.append({
                        "family": family,
                        "local_size": local,
                        "unroll": unroll,
                        "rows_per_item": rows,
                    })
    for lanes in _LANES:
        for local in _LOCAL_SIZES:
            if local < lanes:
                continue
            for unroll in _UNROLLS:
                if lanes * unroll > _COLS:
                    continue
                specs.append({
                    "family": "cooperative",
                    "lanes": lanes,
                    "local_size": local,
                    "unroll": unroll,
                })
    return specs
