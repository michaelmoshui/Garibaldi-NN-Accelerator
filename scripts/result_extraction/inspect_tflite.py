#!/usr/bin/env python3
"""Inspect an INT8 LiteRT/TFLite model: operator inventory + tensor/quant summary.

Useful as a pre-flight check before packaging a LiteRT Micro workload — tells
you which op types appear, how tensors and quantisation parameters are laid
out, and where the multiply-accumulates (MACs) are.

Steps:
    1. load()           Load the .tflite with the LiteRT interpreter.
    2. op_inventory()   Histogram of op types + ordered op list with I/O tensors.
    3. tensor_summary() Per-tensor shape, dtype, and quantisation (scale/zp),
                        and a constants-vs-activations count.
    4. write_report()   Markdown report (model summary, per-op-type and
                        per-layer MACs/params, quantisation, tensor table)
                        under results/model_structure/<model>.md.

MACs are reported two ways: from tensor shapes (the model's arithmetic) and
as executed by TFLM's reference conv kernels, which skip filter taps that fall
outside the image under SAME padding (see conv_window()). The executed count
is the one that matches a Spike run of the TFLM workload.

Usage:
    python scripts/result_extraction/inspect_tflite.py --model <path/to/model.tflite> [--out DIR]
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "results" / "model_structure"
WEIGHTED_OPS = {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "BATCH_MATMUL"}


def load(model_path: str):
    """Load a .tflite with the LiteRT interpreter and return it."""
    try:
        from ai_edge_litert.interpreter import Interpreter, OpResolverType
    except ImportError:
        from tensorflow.lite import Interpreter  # type: ignore
        from tensorflow.lite.experimental import OpResolverType  # type: ignore

    print(f"[info] loading {model_path}")
    # Without default delegates: XNNPACK would otherwise append DELEGATE
    # pseudo-ops that wrap (and double-count) the real model ops.
    interp = Interpreter(
        model_path=model_path,
        experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interp.allocate_tensors()
    return interp


def op_inventory(interp) -> list:
    """Print the op-type histogram and the ordered operator list with I/O tensors."""
    ops = interp._get_ops_details()  # noqa: SLF001  (stable across TF/LiteRT 2.x)

    hist = Counter(op["op_name"] for op in ops)
    print(f"\n=== Operator inventory ({len(ops)} ops, {len(hist)} distinct types) ===")
    for name, n in hist.most_common():
        print(f"  {n:4d}  {name}")

    print("\n=== Ordered op list ===")
    details = {d["index"]: d for d in interp.get_tensor_details()}
    for op in ops:
        outs = op["outputs"]
        # outs is a numpy array — use len(), not truthiness (multi-element is ambiguous)
        out_shape = [int(x) for x in details[int(outs[0])]["shape"]] if len(outs) else []
        print(
            f"  #{op['index']:<3} {op['op_name']:<20} "
            f"in={[int(x) for x in op['inputs']]} "
            f"out={[int(x) for x in op['outputs']]} "
            f"out_shape={out_shape}"
        )
    return ops


def constant_indices(interp) -> set:
    """Op inputs never produced by any op and not model inputs.

    Restricting to op inputs excludes kernel scratch tensors (e.g. conv
    im2col buffers), which the interpreter also lists.
    """
    ops = interp._get_ops_details()  # noqa: SLF001
    op_inputs    = {int(idx) for op in ops for idx in op["inputs"] if idx >= 0}
    op_outputs   = {int(idx) for op in ops for idx in op["outputs"] if idx >= 0}
    model_inputs = {d["index"] for d in interp.get_input_details()}
    return op_inputs - op_outputs - model_inputs

def tensor_kind(idx: int, const_set: set, scratch_set: set) -> str:
    return "const" if idx in const_set else "scratch" if idx in scratch_set else "act"


def scratch_indices(interp) -> set:
    """Tensors that are neither op inputs nor op outputs (kernel temporaries)."""
    ops = interp._get_ops_details()  # noqa: SLF001
    used = {int(idx) for op in ops for key in ("inputs", "outputs") for idx in op[key] if idx >= 0}
    return {t["index"] for t in interp.get_tensor_details()} - used


def tensor_summary(interp) -> None:
    """Print per-tensor shape/dtype/quant and a constants-vs-activations count."""
    tensors = interp.get_tensor_details()
    const_set = constant_indices(interp)
    scratch_set = scratch_indices(interp)

    print(f"\n=== Tensors ({len(tensors)}) ===")
    for t in tensors:
        q      = t.get("quantization_parameters", {})
        scales = q.get("scales", [])
        zps    = q.get("zero_points", [])

        nscale    = len(scales)
        scale_str = (
            f"scale~{scales[0]:.3g}" if nscale == 1
            else f"{nscale} per-ch scales" if nscale > 1
            else "no-quant"
        )
        zp_str = f"zp={int(zps[0])}" if len(zps) else "zp=-"
        kind   = tensor_kind(t["index"], const_set, scratch_set)
        print(
            f"  [{t['index']:>3}] {kind:<7} {str([int(s) for s in t['shape']]):<20} "
            f"{t['dtype'].__name__:<8} {scale_str:<18} {zp_str}  {t['name']}"
        )

    n_act = len(tensors) - len(const_set) - len(scratch_set)
    print(f"\n[summary] {len(const_set)} constants, {n_act} activation tensors, "
          f"{len(scratch_set)} kernel scratch tensors")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def op_macs(name: str, shapes_in: list, shape_out: list) -> int:
    """MACs from tensor shapes (TFLite layouts: NHWC activations, OHWI weights)."""
    out = int(np.prod(shape_out)) if shape_out else 0
    if name == "CONV_2D" and len(shapes_in) > 1:
        _, kh, kw, cin = shapes_in[1]
        return out * kh * kw * cin
    if name == "DEPTHWISE_CONV_2D" and len(shapes_in) > 1:
        _, kh, kw, _ = shapes_in[1]
        return out * kh * kw
    if name == "FULLY_CONNECTED" and len(shapes_in) > 1:
        return out * shapes_in[1][-1]
    if name == "BATCH_MATMUL" and len(shapes_in) > 1:
        return out * shapes_in[0][-1]
    return 0


def infer_window(in_size: int, k: int, out_size: int) -> tuple[int, str, int]:
    """(stride, padding, pad_before) from one spatial dim, assuming dilation 1.

    VALID is preferred when both fit: the two only coincide when SAME needs
    no padding, so the executed-MAC count is the same either way.
    """
    for s in range(1, max(in_size, 1) + 1):
        if (in_size - k) // s + 1 == out_size and in_size >= k:
            return s, "VALID", 0
        if -(-in_size // s) == out_size:
            pad_total = max((out_size - 1) * s + k - in_size, 0)
            return s, "SAME", pad_total // 2
    return 1, "?", 0


def inside_taps(in_size: int, k: int, out_size: int, s: int, pad: int) -> int:
    """Filter taps that land inside the input, summed over output positions."""
    return sum(1 for o in range(out_size) for t in range(k) if 0 <= o * s - pad + t < in_size)


def conv_window(name: str, shape_in: list, shape_w: list, shape_out: list):
    """(stride, padding, executed MACs) for CONV_2D / DEPTHWISE_CONV_2D.

    TFLM's reference conv kernels skip filter taps outside the input image, so
    with SAME padding they execute fewer MACs than out * kh * kw * cin. With
    int8, a padded input equals the zero point, and (zp + input_offset) == 0,
    so the skipped products are all exactly zero.
    """
    n, ih, iw, _ = shape_in
    _, kh, kw, cin = shape_w
    _, oh, ow, cout = shape_out
    sh, pad_kind, ph = infer_window(ih, kh, oh)
    sw, _, pw = infer_window(iw, kw, ow)
    taps = inside_taps(ih, kh, oh, sh, ph) * inside_taps(iw, kw, ow, sw, pw)
    per_tap = cout * cin if name == "CONV_2D" else cout
    return sh, pad_kind, n * taps * per_tap


def quant_scheme(t: dict) -> str:
    q = t.get("quantization_parameters", {})
    scales, zps = q.get("scales", []), q.get("zero_points", [])
    if len(scales) == 0:
        return "none"
    sym = "symmetric" if not np.any(zps) else "asymmetric"
    return f"per-channel ({sym})" if len(scales) > 1 else f"per-tensor ({sym})"


def quant_str(t: dict) -> str:
    q = t.get("quantization_parameters", {})
    scales, zps = q.get("scales", []), q.get("zero_points", [])
    if len(scales) == 0:
        return "-"
    if len(scales) > 1:
        return f"{quant_scheme(t)} x{len(scales)}"
    return f"scale {scales[0]:.4g}, zp {int(zps[0])}"


def fmt_shape(shape) -> str:
    return "x".join(str(int(s)) for s in shape) if len(shape) else "scalar"


def pct(n: int, total: int) -> str:
    return f"{100.0 * n / total:5.1f}%" if total else "  n/a"


def write_report(interp, model_path: str, out_dir: Path) -> Path:
    ops = interp._get_ops_details()  # noqa: SLF001
    tensors = interp.get_tensor_details()
    details = {t["index"]: t for t in tensors}
    const_set = constant_indices(interp)
    scratch_set = scratch_indices(interp)
    act_set = set(details) - const_set - scratch_set

    def nbytes(i: int) -> int:
        return int(np.prod(details[i]["shape"])) * np.dtype(details[i]["dtype"]).itemsize

    def shape(i: int) -> list:
        return [int(x) for x in details[int(i)]["shape"]] if i >= 0 else []

    # Per-layer rows; parameters are the constant inputs of each op.
    layers = []
    for op in ops:
        ins = [int(i) for i in op["inputs"]]
        outs = [int(i) for i in op["outputs"]]
        shapes_in = [shape(i) for i in ins]
        shape_out = shape(outs[0]) if outs else []
        consts = [i for i in ins if i in const_set]
        acts = [i for i in ins if i >= 0 and i not in const_set]
        params = sum(int(np.prod(shape(i))) for i in consts)
        macs = op_macs(op["op_name"], shapes_in, shape_out)
        executed = macs
        stride = kernel = padding = ""
        if op["op_name"] in ("CONV_2D", "DEPTHWISE_CONV_2D") and len(shapes_in) > 1 and acts:
            kernel = f"{shapes_in[1][1]}x{shapes_in[1][2]}"
            s, padding, executed = conv_window(op["op_name"], shape(acts[0]), shapes_in[1], shape_out)
            stride = str(s)
        elif op["op_name"] in ("MAX_POOL_2D", "AVERAGE_POOL_2D") \
                and acts and len(shape(acts[0])) == 4 and len(shape_out) == 4 and shape_out[1]:
            stride = str(shape(acts[0])[1] // shape_out[1])
        layers.append({
            "index": op["index"], "op": op["op_name"],
            "in": fmt_shape(shape(acts[0])) if acts else "-",
            "weight": fmt_shape(shapes_in[1]) if op["op_name"] in WEIGHTED_OPS and len(ins) > 1 else "-",
            "out": fmt_shape(shape_out), "kernel": kernel, "stride": stride, "padding": padding,
            "macs": macs, "executed": executed, "params": params,
        })

    total_macs = sum(l["macs"] for l in layers)
    total_exec = sum(l["executed"] for l in layers)
    skipped = total_macs - total_exec
    total_params = sum(int(np.prod(details[i]["shape"])) for i in const_set)
    const_bytes = sum(nbytes(i) for i in const_set)
    params_by_dtype: dict[str, int] = defaultdict(int)
    for i in const_set:
        params_by_dtype[details[i]["dtype"].__name__] += int(np.prod(details[i]["shape"]))
    act_bytes = sum(nbytes(i) for i in act_set)
    peak_act = max((nbytes(i) for i in act_set), default=0)

    by_type: dict[str, dict] = defaultdict(lambda: {"count": 0, "macs": 0, "executed": 0, "params": 0})
    for l in layers:
        by_type[l["op"]]["count"] += 1
        by_type[l["op"]]["macs"] += l["macs"]
        by_type[l["op"]]["executed"] += l["executed"]
        by_type[l["op"]]["params"] += l["params"]

    name = Path(model_path).stem
    L = []
    L.append(f"# Model structure — {name}\n")
    L.append(f"Source: `{Path(model_path).as_posix()}`, inspected with the LiteRT interpreter "
             "(`scripts/result_extraction/inspect_tflite.py`). MACs are computed from tensor shapes.\n")
    L.append("| Quantity | Value |")
    L.append("|---|---:|")
    L.append(f"| Operators | {len(ops)} ({len(by_type)} distinct types) |")
    L.append(f"| Tensors | {len(tensors)} ({len(const_set)} constant, {len(act_set)} activation, "
             f"{len(scratch_set)} kernel scratch) |")
    L.append(f"| Parameters | {total_params:,} |")
    for dt, n in sorted(params_by_dtype.items(), key=lambda kv: -kv[1]):
        L.append(f"| &nbsp;&nbsp;{dt} parameters | {n:,} |")
    L.append(f"| Constant data | {const_bytes / 2**20:.2f} MiB |")
    L.append(f"| Total MACs / inference (from shapes) | {total_macs:,} ({total_macs / 1e9:.3f} GMAC) |")
    L.append(f"| MACs executed by TFLM reference kernels | {total_exec:,} ({total_exec / 1e9:.3f} GMAC, "
             f"{pct(skipped, total_macs).strip()} padding taps skipped) |")
    L.append(f"| Largest activation tensor | {peak_act / 1024:.1f} KiB |")
    L.append(f"| All activation tensors (no reuse) | {act_bytes / 2**20:.2f} MiB |")
    for d in interp.get_input_details():
        L.append(f"| Input `{d['name']}` | {fmt_shape(d['shape'])} {d['dtype'].__name__}, "
                 f"{quant_str(details[d['index']])} |")
    for d in interp.get_output_details():
        L.append(f"| Output `{d['name']}` | {fmt_shape(d['shape'])} {d['dtype'].__name__}, "
                 f"{quant_str(details[d['index']])} |")
    L.append("")

    L.append("## Shape MACs vs executed MACs\n")
    L.append("*MACs (from shapes)* is `output elements x kernel taps x input channels` — the model's "
             "arithmetic, including taps that land on padding. *Executed* is what TFLM's reference "
             "conv kernels actually compute: they test each filter tap against the input bounds and "
             "skip taps outside the image, so convolutions with SAME padding execute fewer MACs. "
             "The skip is exact for int8: a padded input equals the zero point, and "
             "`(zero_point + input_offset) == 0`, so every skipped product is zero.\n")
    L.append("Convolutions preceded by an explicit `PAD` op run VALID on a materialised padded tensor, "
             "so they execute every tap (the padded ones multiply zero). The executed count is the one "
             "to compare against a Spike run of the TFLM workload (e.g. instructions per MAC); an "
             "accelerated kernel that drops the bounds check would execute the shape count instead.\n")
    L.append(f"For this model: {skipped:,} MACs ({pct(skipped, total_macs).strip()}) are skipped, "
             f"all in SAME-padded convolutions.\n")

    L.append("## By operator type\n")
    L.append("| Op type | Count | MACs (from shapes) | Executed (TFLM) | MAC share | Params |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for op_name, s in sorted(by_type.items(), key=lambda kv: (-kv[1]["macs"], -kv[1]["count"])):
        L.append(f"| `{op_name}` | {s['count']} | {s['macs']:,} | {s['executed']:,} | "
                 f"{pct(s['macs'], total_macs)} | {s['params']:,} |")
    L.append("")

    L.append("## Per layer\n")
    L.append("Stride and padding are inferred from input/output shapes (dilation 1 assumed). "
             "Params are the op's constant inputs (weights + bias).\n")
    L.append("| # | Op | Input | Weight | Kernel | Stride | Padding | Output | MACs (from shapes) | "
             "Executed (TFLM) | MAC share | Params |")
    L.append("|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|")
    for l in layers:
        L.append(f"| {l['index']} | `{l['op']}` | {l['in']} | {l['weight']} | {l['kernel']} | {l['stride']} | "
                 f"{l['padding']} | {l['out']} | {l['macs']:,} | {l['executed']:,} | "
                 f"{pct(l['macs'], total_macs)} | {l['params']:,} |")
    L.append("")

    top = sorted(layers, key=lambda l: -l["macs"])[:10]
    L.append("## Top 10 layers by MACs\n")
    L.append("| # | Op | Weight | Output | MACs | MAC share |")
    L.append("|---:|---|---|---|---:|---:|")
    for l in top:
        L.append(f"| {l['index']} | `{l['op']}` | {l['weight']} | {l['out']} | {l['macs']:,} | {pct(l['macs'], total_macs)} |")
    L.append(f"\nTop 10 layers cover {pct(sum(l['macs'] for l in top), total_macs).strip()} of MACs.\n")

    L.append("## Quantisation\n")
    quant_kinds: Counter = Counter()
    for t in tensors:
        kind = tensor_kind(t["index"], const_set, scratch_set)
        quant_kinds[(kind, t["dtype"].__name__, quant_scheme(t))] += 1
    L.append("| Tensor kind | dtype | Scheme | Count |")
    L.append("|---|---|---|---:|")
    for (kind, dt, scheme), n in sorted(quant_kinds.items()):
        L.append(f"| {kind} | {dt} | {scheme} | {n} |")
    L.append("")

    L.append("## Tensors\n")
    L.append("<details><summary>All tensors</summary>\n")
    L.append("| Idx | Kind | Shape | dtype | Quantisation | Name |")
    L.append("|---:|---|---|---|---|---|")
    for t in tensors:
        kind = tensor_kind(t["index"], const_set, scratch_set)
        L.append(f"| {t['index']} | {kind} | {fmt_shape(t['shape'])} | {t['dtype'].__name__} | "
                 f"{quant_str(t)} | `{t['name']}` |")
    L.append("\n</details>\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / f"{name}.md"
    report.write_text("\n".join(L), encoding="utf-8")
    print(f"\n[report] {total_macs / 1e9:.3f} GMAC from shapes, {total_exec / 1e9:.3f} GMAC executed by TFLM, "
          f"{total_params:,} params -> {report}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True, help="path to the .tflite file")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"report directory (default: {DEFAULT_OUT})")
    args = ap.parse_args()

    interp = load(args.model)
    op_inventory(interp)
    tensor_summary(interp)
    write_report(interp, args.model, args.out)


if __name__ == "__main__":
    main()
