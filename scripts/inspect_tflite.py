#!/usr/bin/env python3
"""Inspect an INT8 LiteRT/TFLite model: operator inventory + tensor/quant summary.

Useful as a pre-flight check before running tflite_to_c.py — tells you which
op types appear (defines the C kernels you must implement) and how tensors and
quantisation parameters are laid out.

Steps:
    1. load()           Load the .tflite with the LiteRT interpreter.
    2. op_inventory()   Histogram of op types + ordered op list with I/O tensors.
    3. tensor_summary() Per-tensor shape, dtype, and quantisation (scale/zp),
                        and a constants-vs-activations count.

Usage:
    python scripts/inspect_tflite.py --model <path/to/model.tflite>
"""

import argparse
from collections import Counter


def load(model_path: str):
    """Load a .tflite with the LiteRT interpreter and return it."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter  # type: ignore

    print(f"[info] loading {model_path}")
    interp = Interpreter(model_path=model_path)
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


def tensor_summary(interp) -> None:
    """Print per-tensor shape/dtype/quant and a constants-vs-activations count."""
    tensors = interp.get_tensor_details()

    # Constants: tensors never produced by any op and not model inputs.
    ops = interp._get_ops_details()  # noqa: SLF001
    op_outputs   = {idx for op in ops for idx in op["outputs"] if idx >= 0}
    model_inputs = {d["index"] for d in interp.get_input_details()}
    all_indices  = {t["index"] for t in tensors}
    const_set    = all_indices - op_outputs - model_inputs

    n_const = 0
    print(f"\n=== Tensors ({len(tensors)}) ===")
    for t in tensors:
        q      = t.get("quantization_parameters", {})
        scales = q.get("scales", [])
        zps    = q.get("zero_points", [])
        is_const = t["index"] in const_set
        n_const += is_const

        nscale    = len(scales)
        scale_str = (
            f"scale~{scales[0]:.3g}" if nscale == 1
            else f"{nscale} per-ch scales" if nscale > 1
            else "no-quant"
        )
        zp_str = f"zp={int(zps[0])}" if len(zps) else "zp=-"
        kind   = "const" if is_const else "act"
        print(
            f"  [{t['index']:>3}] {kind:<5} {str(list(t['shape'])):<20} "
            f"{t['dtype'].__name__:<8} {scale_str:<18} {zp_str}  {t['name']}"
        )

    print(f"\n[summary] {n_const} constants, {len(tensors) - n_const} activation tensors")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True, help="path to the .tflite file")
    args = ap.parse_args()

    interp = load(args.model)
    op_inventory(interp)
    tensor_summary(interp)


if __name__ == "__main__":
    main()
