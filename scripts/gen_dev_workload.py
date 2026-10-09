#!/usr/bin/env python3
"""Generate the software_dev kernel-development workload (dev_net_int8).

dev_net is a synthetic W8A8 network with exactly one instance of every
operator type in resnet18_int8_channelwise, in the same order, quantisation
scheme (int8 per-channel symmetric weights, int8 asymmetric activations) and
flatbuffer encoding:

  #  op               input        weight     stride  padding  output
  0  TRANSPOSE        1x3x32x32    -                           1x32x32x3   NCHW -> NHWC
  1  PAD              1x32x32x3    -                           1x38x38x3   3 px each side
  2  CONV_2D 7x7      1x38x38x3    32x7x7x3   2       VALID    1x16x16x32  + ReLU
  3  MAX_POOL_2D 3x3  1x16x16x32   -          2       SAME     1x8x8x32
  4  CONV_2D 3x3      1x8x8x32     32x3x3x32  1       SAME     1x8x8x32
  5  ADD              1x8x8x32 x2  -                           1x8x8x32    residual (#4 + #3), + ReLU
  6  AVERAGE_POOL_2D  1x8x8x32     -          8       VALID    1x1x1x32
  7  FULLY_CONNECTED  1x1x1x32     10x32                       1x10

~1.8 MMAC per inference (resnet18: 1.8 GMAC).

Outputs under <out> (src/software_dev):
  tflm/                     TFLM runtime tree (create_tflm_tree.py, same pin as
                            gen_tflm_workload.py)
  model/dev_net_int8.tflite the model
  model/dev_net_int8_model_data.{cc,h}
                            generate_cc_arrays.py
  test_data/dev_net_int8_test_data.{cc,h}
                            int8 input + every op's output from the LiteRT
                            BUILTIN_REF kernels, for a per-layer check

The model is written directly with the TFLite flatbuffer schema from the
tflite-micro checkout (no TensorFlow). Weights are random; activation ranges
come from running a float32 copy of the graph on random inputs.

Requires Linux / WSL (create_tflm_tree.py drives tflite-micro's GNU Makefile).

Usage:
    python scripts/gen_dev_workload.py
    python scripts/gen_dev_workload.py --skip-runtime   # keep existing tflm/
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from gen_tflm_workload import (
    DEFAULT_TFLM_COMMIT,
    DEFAULT_TFLM_DIR,
    DEFAULT_TFLM_TARGET,
    REPO_ROOT,
    cc_array,
    checkout_tflm,
    export_runtime,
)

DEFAULT_OUT = REPO_ROOT / "src/software_dev"
MODEL_NAME = "dev_net_int8"

IN_C, IN_H, IN_W = 3, 32, 32
STEM_PAD = 3
CH = 32
NUM_CLASSES = 10
CALIB_SAMPLES = 64


class Graph:
    """Minimal single-subgraph TFLite model builder (flatbuffer object API)."""

    def __init__(self, schema):
        self.s = schema
        self.tensors = []
        self.buffers = [schema.BufferT()]  # buffer 0: empty, for activations
        self.opcodes = []
        self.operators = []
        self.index = {}       # tensor name -> index
        self.op_outputs = []  # (op name, output tensor index) in execution order

    def tensor(self, name, shape, dtype, data=None, scale=None, zero_point=None, qdim=0):
        s = self.s
        t = s.TensorT()
        t.name = name.encode()
        t.shape = list(shape)
        t.type = {"float32": s.TensorType.FLOAT32, "int8": s.TensorType.INT8,
                  "int32": s.TensorType.INT32}[dtype]
        t.buffer = 0
        if data is not None:
            buf = s.BufferT()
            buf.data = np.frombuffer(np.ascontiguousarray(data, dtype).tobytes(), np.uint8)
            self.buffers.append(buf)
            t.buffer = len(self.buffers) - 1
        if scale is not None:
            q = s.QuantizationParametersT()
            q.scale = [float(x) for x in np.atleast_1d(scale)]
            q.zeroPoint = [int(x) for x in np.atleast_1d(zero_point)]
            q.quantizedDimension = qdim
            t.quantization = q
        self.tensors.append(t)
        self.index[name] = len(self.tensors) - 1
        return self.index[name]

    def op(self, name, inputs, outputs, options_type=0, options=None):
        s = self.s
        code = getattr(s.BuiltinOperator, name)
        codes = [c.builtinCode for c in self.opcodes]
        if code not in codes:
            oc = s.OperatorCodeT()
            oc.builtinCode = code
            oc.deprecatedBuiltinCode = min(code, s.BuiltinOperator.PLACEHOLDER_FOR_GREATER_OP_CODES)
            oc.version = 1
            self.opcodes.append(oc)
            codes.append(code)
        o = s.OperatorT()
        o.opcodeIndex = codes.index(code)
        o.inputs = list(inputs)
        o.outputs = list(outputs)
        o.builtinOptionsType = options_type
        o.builtinOptions = options
        self.operators.append(o)
        self.op_outputs.append((name, outputs[0]))

    def serialize(self, inputs, outputs) -> bytes:
        import flatbuffers
        s = self.s
        sg = s.SubGraphT()
        sg.tensors = self.tensors
        sg.inputs = list(inputs)
        sg.outputs = list(outputs)
        sg.operators = self.operators
        sg.name = b"main"
        m = s.ModelT()
        m.version = 3
        m.description = b"garibaldi software_dev"
        m.operatorCodes = self.opcodes
        m.subgraphs = [sg]
        m.buffers = self.buffers
        b = flatbuffers.Builder(1024)
        b.Finish(m.Pack(b), file_identifier=b"TFL3")
        return bytes(b.Output())


def init_weights(rng) -> dict:
    """Random float weights (He-normal) and small biases."""
    def he(shape, fan_in):
        return (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)

    def bias(n):
        return (rng.standard_normal(n) * 0.1).astype(np.float32)

    return {
        "conv7_w": he((CH, 7, 7, IN_C), 7 * 7 * IN_C), "conv7_b": bias(CH),
        "conv3_w": he((CH, 3, 3, CH), 3 * 3 * CH), "conv3_b": bias(CH),
        "fc_w": he((NUM_CLASSES, CH), CH) * np.float32(0.5), "fc_b": bias(NUM_CLASSES),
    }


def build_model(schema, w: dict, act_q: dict = None):
    """Emit dev_net as float32 (act_q None) or int8 (act_q: name -> (scale, zp)).

    PAD, TRANSPOSE, MAX_POOL_2D and AVERAGE_POOL_2D keep their input's
    quantisation, as the TFLite converter requires; `shares` maps those
    tensors to the calibrated tensor whose parameters they reuse.
    """
    s = schema
    g = Graph(s)
    int8 = act_q is not None
    shares = {"x_nhwc": "x", "x_pad": "x", "maxpool": "conv7", "avgpool": "add"}

    def act(name, shape):
        if not int8:
            return g.tensor(name, shape, "float32")
        scale, zp = act_q[shares.get(name, name)]
        return g.tensor(name, shape, "int8", scale=scale, zero_point=zp)

    def weights(name, wf, bf, in_name):
        if not int8:
            return (g.tensor(f"{name}_w", wf.shape, "float32", wf),
                    g.tensor(f"{name}_b", bf.shape, "float32", bf))
        w_scale = np.maximum(np.abs(wf.reshape(len(wf), -1)).max(axis=1), 1e-8) / 127.0
        wq = np.clip(np.round(wf / w_scale.reshape(-1, *[1] * (wf.ndim - 1))), -127, 127)
        b_scale = act_q[shares.get(in_name, in_name)][0] * w_scale
        bq = np.round(bf / b_scale)
        zeros = np.zeros(len(wf), np.int64)
        return (g.tensor(f"{name}_w", wf.shape, "int8", wq, w_scale, zeros, 0),
                g.tensor(f"{name}_b", bf.shape, "int32", bq, b_scale, zeros, 0))

    def conv_opts(padding, stride, act_fn):
        o = s.Conv2DOptionsT()
        o.padding, o.strideW, o.strideH = padding, stride, stride
        o.fusedActivationFunction = act_fn
        o.dilationWFactor = o.dilationHFactor = 1
        return o

    def pool_opts(padding, stride, k):
        o = s.Pool2DOptionsT()
        o.padding, o.strideW, o.strideH = padding, stride, stride
        o.filterWidth = o.filterHeight = k
        return o

    VALID, SAME, RELU = s.Padding.VALID, s.Padding.SAME, s.ActivationFunctionType.RELU
    hp, wp = IN_H + 2 * STEM_PAD, IN_W + 2 * STEM_PAD
    h1, w1 = (hp - 7) // 2 + 1, (wp - 7) // 2 + 1
    h2, w2 = -(-h1 // 2), -(-w1 // 2)

    x = act("x", (1, IN_C, IN_H, IN_W))
    perm = g.tensor("perm_nchw_to_nhwc", (4,), "int32", np.array([0, 2, 3, 1]))
    x_nhwc = act("x_nhwc", (1, IN_H, IN_W, IN_C))
    g.op("TRANSPOSE", [x, perm], [x_nhwc], s.BuiltinOptions.TransposeOptions, s.TransposeOptionsT())

    paddings = g.tensor("stem_paddings", (4, 2), "int32",
                        np.array([[0, 0], [STEM_PAD, STEM_PAD], [STEM_PAD, STEM_PAD], [0, 0]]))
    x_pad = act("x_pad", (1, hp, wp, IN_C))
    g.op("PAD", [x_nhwc, paddings], [x_pad], s.BuiltinOptions.PadOptions, s.PadOptionsT())

    c7w, c7b = weights("conv7", w["conv7_w"], w["conv7_b"], "x_pad")
    conv7 = act("conv7", (1, h1, w1, CH))
    g.op("CONV_2D", [x_pad, c7w, c7b], [conv7], s.BuiltinOptions.Conv2DOptions,
         conv_opts(VALID, 2, RELU))

    maxpool = act("maxpool", (1, h2, w2, CH))
    g.op("MAX_POOL_2D", [conv7], [maxpool], s.BuiltinOptions.Pool2DOptions, pool_opts(SAME, 2, 3))

    c3w, c3b = weights("conv3", w["conv3_w"], w["conv3_b"], "maxpool")
    conv3 = act("conv3", (1, h2, w2, CH))
    g.op("CONV_2D", [maxpool, c3w, c3b], [conv3], s.BuiltinOptions.Conv2DOptions,
         conv_opts(SAME, 1, s.ActivationFunctionType.NONE))

    add = act("add", (1, h2, w2, CH))
    add_opts = s.AddOptionsT()
    add_opts.fusedActivationFunction = RELU
    g.op("ADD", [conv3, maxpool], [add], s.BuiltinOptions.AddOptions, add_opts)

    avgpool = act("avgpool", (1, 1, 1, CH))
    g.op("AVERAGE_POOL_2D", [add], [avgpool], s.BuiltinOptions.Pool2DOptions,
         pool_opts(VALID, h2, h2))

    fcw, fcb = weights("fc", w["fc_w"], w["fc_b"], "avgpool")
    logits = act("logits", (1, NUM_CLASSES))
    g.op("FULLY_CONNECTED", [avgpool, fcw, fcb], [logits],
         s.BuiltinOptions.FullyConnectedOptions, s.FullyConnectedOptionsT())

    return g.serialize([x], [logits]), g


def interpreter(model: bytes):
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    interp = Interpreter(model_content=model,
                         experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
                         experimental_preserve_all_tensors=True)
    interp.allocate_tensors()
    return interp


def calibrate(schema, w: dict, inputs: np.ndarray) -> dict:
    """Per-tensor int8 (scale, zero point) from float32 activation ranges."""
    model, g = build_model(schema, w)
    interp = interpreter(model)
    names = ["x", "conv7", "conv3", "add", "logits"]
    lo = {n: 0.0 for n in names}
    hi = {n: 0.0 for n in names}
    for x in inputs:
        interp.set_tensor(g.index["x"], x)
        interp.invoke()
        for n in names:
            t = interp.get_tensor(g.index[n])
            lo[n], hi[n] = min(lo[n], float(t.min())), max(hi[n], float(t.max()))
    q = {}
    for n in names:
        scale = max(hi[n] - lo[n], 1e-6) / 255.0
        q[n] = (scale, int(np.clip(round(-128 - lo[n] / scale), -128, 127)))
    return q


def write_test_data(out: Path, g: Graph, x_q: np.ndarray, layers: list) -> None:
    """test_data/<model>_test_data.{h,cc}: input + one golden tensor per op."""
    stem = f"{MODEL_NAME}_test_data"
    d = out / "test_data"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)

    def array(name, data):
        vals = [str(int(v)) for v in data.reshape(-1)]
        rows = [", ".join(vals[i:i + 16]) for i in range(0, len(vals), 16)]
        return (f"alignas(16) const int8_t {name}[{len(vals)}] = {{\n  "
                + ",\n  ".join(rows) + "\n};\n")

    (d / f"{stem}.h").write_text(
        "// Generated by scripts/gen_dev_workload.py. Do not edit.\n"
        "#pragma once\n\n#include <cstdint>\n\n"
        "struct DevGoldenTensor {\n"
        "  const char* op;      // builtin operator name\n"
        "  int tensor_index;    // op output tensor in the model's subgraph\n"
        "  const int8_t* data;  // LiteRT BUILTIN_REF output\n"
        "  int size;\n"
        "};\n\n"
        f"constexpr int g_{MODEL_NAME}_input_size = {x_q.size};\n"
        f"extern const int8_t g_{MODEL_NAME}_input[];\n\n"
        f"constexpr int g_{MODEL_NAME}_num_ops = {len(layers)};\n"
        f"extern const DevGoldenTensor g_{MODEL_NAME}_golden[];\n")

    body = [f'// Generated by scripts/gen_dev_workload.py. Do not edit.\n'
            f'#include "test_data/{stem}.h"\n\n', array(f"g_{MODEL_NAME}_input", x_q)]
    for i, (_, _, data) in enumerate(layers):
        body.append("\n" + array(f"g_{MODEL_NAME}_op{i}_output", data))
    body.append(f"\nconst DevGoldenTensor g_{MODEL_NAME}_golden[] = {{\n")
    for i, (op, idx, data) in enumerate(layers):
        body.append(f'  {{"{op}", {idx}, g_{MODEL_NAME}_op{i}_output, {data.size}}},\n')
    body.append("};\n")
    (d / f"{stem}.cc").write_text("".join(body))


def print_summary(g: Graph) -> None:
    total = 0
    for i, (op, idx) in enumerate(g.op_outputs):
        shape = list(g.tensors[idx].shape)
        macs = 0
        if op in ("CONV_2D", "FULLY_CONNECTED"):
            wt = g.tensors[g.operators[i].inputs[1]].shape
            macs = int(np.prod(shape)) * int(np.prod(wt[1:]))
        total += macs
        print(f"    #{i} {op:<16} -> {'x'.join(map(str, shape)):<12} {macs:>10,} MACs")
    print(f"    total {total:,} MACs")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"software_dev directory (default: {DEFAULT_OUT.relative_to(REPO_ROOT)})")
    ap.add_argument("--tflm-dir", type=Path, default=DEFAULT_TFLM_DIR,
                    help=f"tflite-micro checkout (default: {DEFAULT_TFLM_DIR})")
    ap.add_argument("--tflm-commit", default=DEFAULT_TFLM_COMMIT,
                    help="tflite-micro commit to pin")
    ap.add_argument("--tflm-target", default=DEFAULT_TFLM_TARGET,
                    help=f"TFLM Makefile TARGET for the export (default: {DEFAULT_TFLM_TARGET})")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for weights, calibration and golden input (default: 0)")
    ap.add_argument("--skip-runtime", action="store_true",
                    help="keep the existing <out>/tflm instead of re-exporting it")
    args = ap.parse_args()

    out = args.out.resolve()
    tflm_dir = args.tflm_dir.resolve()
    tools = tflm_dir / "tensorflow/lite/micro/tools"

    checkout_tflm(tflm_dir, args.tflm_commit)
    sys.path.insert(0, str(tflm_dir))
    from tensorflow.lite.python import schema_py_generated as schema

    env = {k: v for k, v in os.environ.items()
           if k not in ("MAKEFLAGS", "MFLAGS", "MAKELEVEL", "MAKEOVERRIDES")}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(tflm_dir), env.get("PYTHONPATH")]))

    if args.skip_runtime and (out / "tflm").is_dir():
        print("[1/4] Keeping existing LiteRT Micro runtime tree")
    else:
        print(f"[1/4] Exporting LiteRT Micro runtime tree (TARGET={args.tflm_target})")
        export_runtime(tflm_dir, tools, out, args.tflm_target, env)

    print(f"[2/4] Building {MODEL_NAME}.tflite (seed={args.seed})")
    rng = np.random.default_rng(args.seed)
    w = init_weights(rng)
    calib = rng.standard_normal((CALIB_SAMPLES, 1, IN_C, IN_H, IN_W)).astype(np.float32)
    act_q = calibrate(schema, w, calib)
    model, g = build_model(schema, w, act_q)
    (out / "model").mkdir(parents=True, exist_ok=True)
    model_path = out / "model" / f"{MODEL_NAME}.tflite"
    model_path.write_bytes(model)
    print_summary(g)

    print(f"[3/4] Embedding {model_path.name} as a C++ array")
    cc_array(tools, out, f"model/{MODEL_NAME}_model_data", model_path, env)

    print("[4/4] Generating per-op golden outputs with LiteRT reference kernels")
    x_scale, x_zp = act_q["x"]
    x_f = rng.standard_normal((1, IN_C, IN_H, IN_W))
    x_q = np.clip(np.round(x_f / x_scale) + x_zp, -128, 127).astype(np.int8)
    interp = interpreter(model)
    interp.set_tensor(g.index["x"], x_q)
    interp.invoke()
    layers = [(op, idx, interp.get_tensor(idx).copy()) for op, idx in g.op_outputs]
    write_test_data(out, g, x_q, layers)

    print(f"[ok] {MODEL_NAME} workload written to {out}")


if __name__ == "__main__":
    main()
