#!/usr/bin/env python3
"""Generate the "Baseline Workload Analysis" report (results/baseline_workload_analysis.md).

Covers Section 2 (Required Project Components) of project_guidelines.md for the
RV64IMC baseline: workload instantiation, project structure, and the Spike
workload characterisation. Every number is read from existing artefacts, so
`make baseline_report` runs `make test_spike` and `make model_inspect` first:

  results/spike_results/<run>.md              summary + instruction-class table + hot blocks
  results/spike_results/<run>_functions.csv   per-function retired-instruction mix
  results/spike_results/<run>_mnemonics.csv   per-mnemonic counts
  results/model_structure/<model>.md          parameters, MACs, op types, quantisation
  src/software/build/rv64/<run>.elf           code size (ELF section/symbol headers)
  src/software/build/rv64/<run>.stdout.txt    functional check (PASS, arena usage)

Standard library only; runs on Windows, WSL and Linux.

Usage:
  python scripts/result_extraction/gen_baseline_report.py
  python scripts/result_extraction/gen_baseline_report.py --top 10 \\
      --out results/baseline_workload_analysis.md
"""

from __future__ import annotations

import argparse
import csv
import re
import struct
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SPIKE = REPO_ROOT / "results/spike_results/resnet18_test"
DEFAULT_MODEL_REPORT = REPO_ROOT / "results/model_structure/resnet18_int8_channelwise.md"
DEFAULT_ELF = REPO_ROOT / "src/software/build/rv64/resnet18_test.elf"
DEFAULT_OUT = REPO_ROOT / "results/baseline_workload_analysis.md"

LOREM = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud "
    "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."
)

# Exclusive-time function -> model operator. Anything unmatched is setup/runtime.
OPERATOR_OF = [
    (re.compile(r"ConvEval"), "CONV_2D"),
    (re.compile(r"EvalAddQuantized"), "ADD"),
    (re.compile(r"MaxEval"), "MAX_POOL_2D"),
    (re.compile(r"AverageEval"), "AVERAGE_POOL_2D"),
    (re.compile(r"PadEval"), "PAD"),
    (re.compile(r"FullyConnectedEval"), "FULLY_CONNECTED"),
    (re.compile(r"TransposeImpl"), "TRANSPOSE"),
    (re.compile(r"MultiplyByQuantizedMultiplier"), "Requantization helper"),
]
SOFTFLOAT_RE = re.compile(
    r"^__((add|sub|mul|div|neg)[sd]f3|float(un)?[sdt]i[sd]f|fix(uns)?[sd]f[sdt]i|"
    r"extendsfdf2|truncdfsf2|(eq|ne|lt|le|gt|ge|unord)[sd]f2)$"
)

LOADS = {"lb": 1, "lbu": 1, "lh": 2, "lhu": 2, "lw": 4, "lwu": 4, "ld": 8}
STORES = {"sb": 1, "sh": 2, "sw": 4, "sd": 8}
RVC_MEM = {"c.lw": "lw", "c.ld": "ld", "c.lwsp": "lw", "c.ldsp": "ld",
           "c.sw": "sw", "c.sd": "sd", "c.swsp": "sw", "c.sdsp": "sd"}


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------
def md_table(text: str, heading: str | None = None) -> list[list[str]]:
    """Rows (header first) of the first Markdown table after `heading` (or in the file)."""
    lines = text.splitlines()
    start = 0
    if heading is not None:
        start = next(i for i, l in enumerate(lines)
                     if l.startswith("#") and l.lstrip("#").strip().startswith(heading)) + 1
    rows: list[list[str]] = []
    for line in lines[start:]:
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not all(set(c) <= set("-: ") for c in cells):
                rows.append(cells)
        elif rows:
            break
    if not rows:
        raise SystemExit(f"no table found after heading {heading!r}")
    return rows


def kv_table(text: str, heading: str | None = None) -> dict[str, str]:
    return {r[0].replace("&nbsp;", "").strip(): r[1] for r in md_table(text, heading)[1:]}


def num(s: str) -> int:
    return int(re.sub(r"[^\d]", "", s.split("(")[0]))


def hot_block(text: str) -> dict:
    m = re.search(
        r"^### `(.+?)` @ 0x([0-9a-f]+)\s+(\d+) insns x ([\d,]+) executions = ([\d,]+) "
        r"\(([\d.]+%)\)\s+```\n(.*?)```",
        text, re.M | re.S)
    if not m:
        raise SystemExit("no basic-block section found in the Spike report")
    code = [l.rstrip() for l in m.group(7).splitlines() if l.strip()]
    return {"func": m.group(1), "pc": m.group(2), "len": int(m.group(3)),
            "execs": num(m.group(4)), "insns": num(m.group(5)), "share": m.group(6),
            "code": code}


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def elf_layout(path: Path) -> tuple[list[dict], dict[str, int]]:
    """Allocatable sections and symbol sizes of a little-endian ELF64 file."""
    data = path.read_bytes()
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise SystemExit(f"{path}: not a little-endian ELF64 file")
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    raw = []
    for i in range(shnum):
        raw.append(struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize))

    def strtab_name(tab_off: int, idx: int) -> str:
        end = data.index(b"\0", tab_off + idx)
        return data[tab_off + idx:end].decode(errors="replace")

    names_off = raw[shstrndx][4]
    sections = []
    symbols: dict[str, int] = {}
    for name_idx, sh_type, flags, _addr, offset, size, link, _info, _al, entsize in raw:
        name = strtab_name(names_off, name_idx)
        if flags & 0x2:  # SHF_ALLOC
            kind = ("code" if flags & 0x4 else
                    "zero-initialised data" if sh_type == 8 else
                    "read-write data" if flags & 0x1 else "read-only data")
            sections.append({"name": name, "kind": kind, "size": size})
        if sh_type == 2:  # SHT_SYMTAB
            str_off = raw[link][4]
            for j in range(size // entsize):
                st_name, _i, _o, _shndx, _val, st_size = struct.unpack_from(
                    "<IBBHQQ", data, offset + j * entsize)
                if st_size:
                    symbols[strtab_name(str_off, st_name)] = st_size
    return sections, symbols


def operator_of(func: str) -> str:
    for pat, op in OPERATOR_OF:
        if pat.search(func):
            return op
    return "Setup / runtime"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def pct(n: float, total: float, digits: int = 1) -> str:
    return f"{100.0 * n / total:.{digits}f}%" if total else "n/a"


def size_str(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GiB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.2f} MiB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f} KiB"
    return f"{n} B"


def short(name: str, width: int = 70) -> str:
    name = name.replace("tflite::micro::", "").replace("(anonymous namespace)::", "")
    name = re.sub(r"\(.*\)", "()", name, count=1) if "(" in name else name
    return name if len(name) <= width else name[: width - 3] + "..."


def table(header: list[str], rows: list[list], align: str) -> list[str]:
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out + [""]


def rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spike", type=Path, default=DEFAULT_SPIKE,
                    help="inspect_spike.py output prefix (expects <prefix>.md, "
                         "<prefix>_functions.csv, <prefix>_mnemonics.csv)")
    ap.add_argument("--model-report", type=Path, default=DEFAULT_MODEL_REPORT,
                    help="inspect_tflite.py report for the model")
    ap.add_argument("--elf", type=Path, default=DEFAULT_ELF, help="bare-metal RV64 ELF")
    ap.add_argument("--stdout", type=Path, default=None,
                    help="Spike program stdout (default: <elf without .elf>.stdout.txt)")
    ap.add_argument("--top", type=int, default=10, help="functions in the top-function table")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="report path")
    args = ap.parse_args()

    spike_md = args.spike.with_suffix(".md").read_text(encoding="utf-8")
    model_md = args.model_report.read_text(encoding="utf-8")
    functions = read_csv(Path(f"{args.spike}_functions.csv"))
    mnemonics = read_csv(Path(f"{args.spike}_mnemonics.csv"))
    stdout_path = args.stdout or args.elf.with_suffix(".stdout.txt")
    stdout = stdout_path.read_text(errors="replace") if stdout_path.exists() else ""
    sections, symbols = elf_layout(args.elf)

    # ---- Spike numbers -------------------------------------------------------
    s = kv_table(spike_md)
    total = num(s["Retired instructions, whole run"])
    app = num(s["Application (ELF) instructions"])
    pk = num(s["Proxy-kernel / unmapped PCs"])
    invoke = num(s["`Invoke()` instret (CSR)"])
    rvc_share = s["Compressed (RVC) share of app insns"]
    outside_invoke = total - invoke

    class_rows = md_table(spike_md, "By instruction class")
    by_class = {r[0]: num(r[1]) for r in class_rows[1:]}
    loads, stores = by_class.get("load", 0), by_class.get("store", 0)
    mem = loads + stores

    hb = hot_block(spike_md)

    # ---- Model numbers -------------------------------------------------------
    m = kv_table(model_md)
    params = num(m["Parameters"])
    shape_macs = num(m["Total MACs / inference (from shapes)"])
    exec_macs = num(m["MACs executed by TFLM reference kernels"])
    op_rows = md_table(model_md, "By operator type")
    op_macs = {r[0].strip("`"): num(r[3]) for r in op_rows[1:]}
    op_count = {r[0].strip("`"): num(r[1]) for r in op_rows[1:]}
    quant_rows = md_table(model_md, "Quantisation")

    # ---- Per-function / per-operator aggregation -----------------------------
    cls_cols = ["load", "store", "alu", "mul", "branch"]
    funcs = sorted(functions, key=lambda r: -int(r["count"]))
    op_insns: dict[str, int] = defaultdict(int)
    softfloat = 0
    for r in funcs:
        op_insns[operator_of(r["function"])] += int(r["count"])
        if SOFTFLOAT_RE.match(r["function"]):
            softfloat += int(r["count"])

    # ---- Memory breakdown ----------------------------------------------------
    mem_by_mnem = {r["mnemonic"]: int(r["count"]) for r in mnemonics
                   if r["mnemonic"] in LOADS or r["mnemonic"] in STORES}
    bytes_loaded = sum(LOADS[k] * v for k, v in mem_by_mnem.items() if k in LOADS)
    bytes_stored = sum(STORES[k] * v for k, v in mem_by_mnem.items() if k in STORES)

    hb_mem = hb_stack = hb_operand = 0
    for line in hb["code"]:
        parts = line.split(None, 2)
        mnem, ops = (parts[1], parts[2] if len(parts) > 2 else "") if len(parts) > 1 else ("", "")
        base = RVC_MEM.get(mnem, mnem)
        if base in LOADS or base in STORES:
            hb_mem += 1
            if "(sp)" in ops:
                hb_stack += 1
            else:
                hb_operand += 1

    # ---- Code size -----------------------------------------------------------
    elf_bytes = args.elf.stat().st_size
    code_bytes = sum(x["size"] for x in sections if x["kind"] == "code")
    model_sym = next((k for k in symbols if re.search(r"model_data$", k)), None)
    arena_sym = next((k for k in symbols if "g_tensor_arena" in k), None)
    model_bytes = symbols.get(model_sym, 0) if model_sym else 0
    arena_bytes = symbols.get(arena_sym, 0) if arena_sym else 0
    arena_used = re.search(r"arena used:\s*(\d+)\s*/\s*(\d+)", stdout)
    passed = "PASS" in stdout
    mismatch = re.search(
        r"mismatches vs LiteRT reference:\s*(\d+)\s*/\s*(\d+)\s*\(max \|diff\| (\d+)\)", stdout)

    # ---- Write ---------------------------------------------------------------
    L: list[str] = []
    w = L.append

    w("# Baseline Workload Analysis\n")
    w(f"*Generated by `scripts/result_extraction/gen_baseline_report.py` from "
      f"`{rel(args.spike.with_suffix('.md'))}`, `{rel(args.model_report)}` and "
      f"`{rel(args.elf)}`. Re-run `make baseline_report` to refresh.*\n")

    w("## Contents\n")
    w("1. [Introduction](#1-introduction)")
    w("2. [Workload Instantiation](#2-workload-instantiation)")
    w("3. [Project Structure](#3-project-structure)")
    w("4. [Baseline Spike Simulation Analysis](#4-baseline-spike-simulation-analysis)")
    w("5. [Conclusion](#5-conclusion)\n")

    # 1 ---------------------------------------------------------------------
    w("## 1. Introduction\n")
    w("The Garibaldi NN Accelerator project designs and evaluates small architectural extensions "
      "that accelerate quantized neural-network inference on a **bare-metal RV64IMC** RISC-V "
      "processor, the CV-Wally core. The baseline has no floating-point or vector hardware. "
      "The workload is an INT8 (**W8A8**) network that is converted offline into a static C/C++ "
      "image and runs without an operating system. The project follows a "
      "*measurement → architecture proposal → implementation → evaluation* process:\n")
    w("1. Build a bare-metal baseline of the quantized model on RV64IMC "
      "(GEMM/convolution, requantization, nonlinear ops, static memory allocation).")
    w("2. Characterise the workload on the Spike ISA simulator and identify the largest "
      "architectural bottlenecks from measurements.")
    w("3. Add the minimum ISA extension (a packed 8×INT8 multiply-accumulate, `Xqmac8`, plus a "
      "requantization instruction) to Spike and to the kernels, verify against the reference, "
      "and measure the instruction reduction.")
    w("4. Implement at least one kernel on the CV-Wally RTL, measure cycles/CPI/Fmax and area "
      "(Synopsys DC / RTL Architect, SAED 32/28 nm), and report speedup per area.\n")
    w("This report covers **Section 2 (Required Project Components)** of the project guidelines: "
      "the bare-metal baseline (2.A) and the workload and architecture analysis (2.B). "
      "It describes the chosen workload and how it becomes a bare-metal application, "
      "explains the repository layout and build flow, and presents the baseline Spike "
      "characterisation that later ISA extensions will be measured against.\n")

    # 2 ---------------------------------------------------------------------
    w("## 2. Workload Instantiation\n")
    w("### 2.1 Model selection rationale\n")
    w(LOREM + "\n")
    w(LOREM + "\n")

    w("### 2.2 ResNet-18 architecture\n")
    w("ResNet-18 is an 18-layer residual convolutional network for ImageNet classification "
      "(224×224×3 input, 1000 classes). It is built from *basic blocks*: two 3×3 convolutions, "
      "each followed by batch normalisation, with ReLU after the first convolution and after "
      "the residual addition. A shortcut adds the block input to its output. When a block "
      "halves the spatial size and doubles the channels, the shortcut is a strided 1×1 "
      "convolution (the *downsample* path). The network is:\n")
    w_rows = [
        ["Stem", "7×7 conv, 64, stride 2 → 3×3 max-pool, stride 2", "112×112×64 → 56×56×64"],
        ["Stage 1 (`layer1`)", "2 basic blocks, 64 channels", "56×56×64"],
        ["Stage 2 (`layer2`)", "2 basic blocks, 128 channels, first block stride 2 + 1×1 downsample", "28×28×128"],
        ["Stage 3 (`layer3`)", "2 basic blocks, 256 channels, first block stride 2 + 1×1 downsample", "14×14×256"],
        ["Stage 4 (`layer4`)", "2 basic blocks, 512 channels, first block stride 2 + 1×1 downsample", "7×7×512"],
        ["Head", "global average pool → fully connected 512→1000", "1000 logits"],
    ]
    L.extend(table(["Stage", "Layers", "Output"], w_rows, "lll"))
    w(f"This gives 17 weight convolutions plus 3 downsample convolutions (20 `CONV_2D`), "
      f"and one fully connected layer. Together they hold **{params / 1e6:.2f} M parameters** "
      f"and need **{shape_macs / 1e9:.3f} GMAC** per inference, which is inside the guideline "
      f"range of 2–12 M parameters and 0.2–2 GMAC.\n")

    w("### 2.3 The quantized model and its bare-metal port\n")
    w("The model is `resnet18_int8_channelwise.tflite` from the HuggingFace "
      "[litert-community/resnet18](https://huggingface.co/litert-community/resnet18) "
      "repository. It is a PyTorch ResNet-18 converted to LiteRT (formerly TensorFlow Lite) "
      "and fully quantized with the `STATIC_WI8_AI8` recipe. The download is already W8A8, "
      "so no calibration or post-training quantization runs in this project "
      "(`make fetch_model`, `scripts/ml_workload/fetch_model.py`).\n")
    w("Conversion changed the graph in three ways that matter for the workload:\n")
    w("- **Batch norm and ReLU are folded into the convolutions.** Each conv's weights and "
      "int32 bias absorb the batch-norm scale and shift. The ReLU becomes the conv's "
      "output clamp: every post-ReLU activation has zero point −128, the int8 minimum.")
    w("- **Layout.** PyTorch's NCHW input is converted to LiteRT's NHWC layout by a leading "
      "`TRANSPOSE` op.")
    w("- **Explicit padding.** The stem conv, the max-pool, and the strided 3×3 convs get "
      "explicit `PAD` ops and then run with `VALID` padding. The stride-1 3×3 convs keep "
      "`SAME` padding.\n")
    op_table_rows = [[f"`{r[0].strip('`')}`", r[1], r[2], r[3], r[4], r[5]] for r in op_rows[1:]]
    L.extend(table(["Op type", "Count", "MACs (from shapes)", "MACs executed (TFLM)",
                    "MAC share", "Params"], op_table_rows, "lrrrrr"))
    w(f"The graph has {m['Operators'].replace(' (', ' operators (', 1)}. `CONV_2D` accounts for essentially all of the "
      f"arithmetic. *MACs executed* is lower than the shape-derived count because TFLM's "
      f"reference convolution skips filter taps that fall on `SAME` padding. Those products "
      f"are exactly zero for int8, so {shape_macs - exec_macs:,} MACs "
      f"({pct(shape_macs - exec_macs, shape_macs)}) are never computed. "
      f"Details are in `{rel(args.model_report)}`.\n")

    w("**From LiteRT to a bare-metal application.** The model runs on "
      "**LiteRT for Microcontrollers** (TFLM), LiteRT's interpreter for MCUs. TFLM has no "
      "dynamic memory, file system, or OS dependencies. `make model_workload` "
      "(`scripts/ml_workload/gen_tflm_workload.py`) does the conversion at a pinned "
      "tflite-micro commit in three steps:\n")
    w("1. **Runtime export.** `create_tflm_tree.py` copies the TFLM library sources and their "
      "third-party dependencies (flatbuffers, gemmlowp, ruy headers, and printf) into "
      "`src/software/workload/tflm/`. They are compiled as ordinary C++ with the portable "
      "*reference* kernels; there are no target-specific optimized kernels.")
    w("2. **Model embedding.** `generate_cc_arrays.py` turns the `.tflite` flatbuffer into a "
      "`const` C++ byte array (`model/resnet18_int8_channelwise_model_data.cc`). The weights "
      "and graph are linked into the ELF as read-only data, so nothing is loaded at run time.")
    w("3. **Op resolver.** `generate_micro_mutable_op_resolver_from_model.py` emits a "
      f"`MicroMutableOpResolver` header that registers only the {len(op_rows) - 1} op types "
      "the model uses, so the linker can drop every other kernel.\n")
    arena_txt = (f"{num(arena_used.group(1)):,} of {num(arena_used.group(2)):,} bytes"
                 if arena_used else "the arena")
    w("The application (`src/software/test/main.cc`) runs the model through `MicroInterpreter`. "
      "All activations and scratch buffers live in one statically allocated tensor arena "
      f"(`g_tensor_arena`, {size_str(arena_bytes) if arena_bytes else '3 MiB'}; the run "
      f"used {arena_txt}). The program does one inference on a random int8 input and checks "
      "the int8 logits against golden outputs from LiteRT's reference kernels. "
      "It also reads the `rdcycle`/`rdinstret` CSRs around `Invoke()`. It is built with "
      "`riscv64-unknown-elf-g++ -O2 -march=rv64imc -mabi=lp64` (soft-float ABI, newlib-nano) "
      "and runs on Spike through the proxy kernel `pk`. "
      "`TFLITE_EMULATE_FLOAT` makes TFLM compute its requantization multipliers with integer "
      "code. The few soft-float library calls left run only during model preparation, and "
      "the ELF contains no F/D instructions (`make -C src/software CONFIG=rv64 fp-audit`).\n")
    if passed:
        mm = (f": {mismatch.group(1)} of {mismatch.group(2)} logits differ, by at most "
              f"{mismatch.group(3)} LSB" if mismatch else "")
        w(f"On Spike the run **passes** the reference check{mm}. The test allows a ±1 LSB "
          "rounding difference between the x86 LiteRT reference and the RV64 scalar kernels.\n")

    w("### 2.4 Quantization scheme\n")
    w("The model is **W8A8**: INT8 weights and INT8 activations, with INT32 accumulation "
      "and INT32 biases.\n")
    q_rows = [[r[0], r[1], r[2], r[3]] for r in quant_rows[1:]]
    L.extend(table(["Tensor kind", "dtype", "Scheme", "Count"], q_rows, "lllr"))
    w("- **Weights**: signed INT8, **symmetric per output channel** (zero point 0, one scale "
      "per output channel of each conv/FC).")
    w("- **Activations**: signed INT8, **asymmetric per tensor** (one scale and one non-zero "
      f"zero point per tensor). For example, the input is `{m.get('Input `g0::x`', '')}`.")
    w("- **Biases**: INT32, per channel, with scale $s_{in} \\cdot s_w[c]$ and zero point 0, "
      "so they add straight into the accumulator.")
    w("- **Requantization** maps each INT32 accumulator to an INT8 output. During model "
      "preparation TFLM turns the real multiplier $M_c = s_{in} s_w[c] / s_{out}$ into a Q31 "
      "fixed-point mantissa and a shift. At run time each output computes\n")
    w("$$acc_c = bias_c + \\sum (x - z_{in}) \\cdot w_c, \\qquad "
      "y = \\mathrm{clamp}\\big(\\mathrm{MultiplyByQuantizedMultiplier}(acc_c, M_{0,c}, "
      "shift_c) + z_{out},\\ act_{min}, act_{max}\\big)$$\n")
    w("  The input zero point is subtracted from every activation inside the MAC loop "
      "(TFLM's `input_offset` $= -z_{in}$); weights need no correction because they are "
      "symmetric. `MultiplyByQuantizedMultiplier` performs a rounding doubling high multiply and a "
      "rounding right shift. The final clamp also applies the fused ReLU. This "
      "fixed-point multiply → round → shift → add zero point → saturate sequence is what "
      "the requantization instruction required by the project guidelines will accelerate.\n")

    # 3 ---------------------------------------------------------------------
    w("## 3. Project Structure\n")
    w("### 3.1 Repository hierarchy\n")
    w("```text")
    w("Garibaldi-NN-Accelerator/")
    w("├── Makefile                  top-level orchestration (`make help` lists every target)")
    w("├── project_guidelines.md     project specification")
    w("├── requirements.txt          Python environment (LiteRT, TFLM tools, HuggingFace hub)")
    w("├── scripts/")
    w("│   ├── ml_workload/          model download + TFLM workload / test / demo / dev generation")
    w("│   ├── result_extraction/    inspect_tflite.py, inspect_spike.py, gen_baseline_report.py")
    w("│   └── tool_installation/    Spike + pk, Verilator/Verible, CV-Wally checkout")
    w("├── src/")
    w("│   ├── software/             ResNet-18 bare-metal workload")
    w("│   │   ├── Makefile          one build for both apps (APP=test|demo, CONFIG=host|rv64)")
    w("│   │   ├── model/            downloaded .tflite (generated)")
    w("│   │   ├── workload/         TFLM runtime tree + model C array + op resolver (generated)")
    w("│   │   ├── test/             golden-vector test app: main.cc + test_data/ (generated)")
    w("│   │   ├── demo/             ImageNet photo demo: main.cc + demo_data/ (generated)")
    w("│   │   └── build/            host and rv64 objects, ELFs, Spike logs (generated)")
    w("│   ├── software_dev/         dev_net: one instance of each ResNet-18 op, for kernel work")
    w("│   │   └── kernels/          custom kernels that replace TFLM reference kernels")
    w("│   └── hardware/             CV-Wally RV64IMC config, project RTL, lint + Verilator sim")
    w("│       └── rtl/              garibaldi_xqmac8.sv, garibaldi_requant.sv, overrides/")
    w("└── results/")
    w("    ├── model_structure/      per-model op/MAC/quantisation reports (make model_inspect)")
    w("    ├── spike_results/        Spike instruction-mix reports + CSVs (make test_spike)")
    w("    └── baseline_workload_analysis.md   this report")
    w("```\n")
    w("Only hand-written sources are tracked in git. Generated artefacts (the model, the "
      "exported TFLM tree, test vectors, demo photos, build output, the CV-Wally checkout, "
      "and `results/`) are git-ignored and rebuilt by the Makefile. `src/software/` holds the "
      "full-model workload measured in this report. `src/software_dev/` is a small network "
      "(~1.8 MMAC) with the same op types, so custom kernels (for example the `Xqmac8` "
      "convolution) can be verified op by op in seconds. `src/hardware/` keeps upstream "
      "CV-Wally pristine and adds project RTL and overrides on top.\n")

    w("### 3.2 Building and running\n")
    w("The flow runs on Linux or WSL (Ubuntu 24.04). Each step in the top-level Makefile "
      "leaves a sentinel file, so later targets only redo stale steps.\n")
    build_rows = [
        ["`make setup`", "Create `.venv` and install `requirements.txt`"],
        ["`make get_spike`", "Build Spike + the proxy kernel `pk` under `~/.local/riscv`"],
        ["`make fetch_model`", "Download `resnet18_int8_channelwise.tflite` from HuggingFace"],
        ["`make model_inspect`", "Model structure report → `results/model_structure/`"],
        ["`make model_workload`", "Export the TFLM runtime, embed the model, generate the op resolver"],
        ["`make test_data`", "Random int8 input + LiteRT reference golden logits"],
        ["`make test_run`", "Native host build + bit-exact check against LiteRT"],
        ["`make test_compile`", "Cross-compile `src/software/build/rv64/resnet18_test.elf`"],
        ["`make test_spike`", "Spike `-g` PC histogram + disassembly → `results/spike_results/`"],
        ["`make baseline_report`", "Run `test_spike` + `model_inspect`, then regenerate this report"],
    ]
    L.extend(table(["Target", "What it does"], build_rows, "ll"))
    w("The demo pipeline (`make demo_data`, `demo_run`, `demo_compile`, `demo_spike`) runs "
      "the same runtime on 20 real ImageNet photos. The `dev_*` targets and the hardware "
      "targets (`get_hw_tools`, `hw_setup`, `hw_lint`, `hw_sim`) cover kernel development "
      "and the CV-Wally RTL flow (see `src/hardware/README.md`).\n")

    # 4 ---------------------------------------------------------------------
    w("## 4. Baseline Spike Simulation Analysis\n")
    w(f"All measurements are for one full ResNet-18 inference of the golden-vector test "
      f"(`resnet18_test.elf`) on Spike (`--isa=rv64imac_zicsr_zifencei_zicntr`, TFLM reference "
      f"kernels, `-O2`). Spike's `-g` flag counts how often each PC retires. "
      f"`inspect_spike.py` joins those counts with the ELF disassembly to get exact dynamic "
      f"counts per mnemonic, instruction class, function, and basic block.\n")
    w(f"**Setup overhead is negligible.** The `Invoke()` window (the actual network "
      f"evaluation) retires {invoke:,} instructions, {pct(invoke, total, 2)} of the whole "
      f"run. Model setup, interpreter construction, tensor allocation, the result check, "
      f"and the proxy kernel outside `Invoke()` together retire only {outside_invoke:,} "
      f"instructions ({pct(outside_invoke, total, 2)}). The soft-float helpers called during "
      f"model preparation retire {softfloat:,} ({pct(softfloat, app, 3)}). All figures below "
      f"therefore describe the NN workload itself, and setup is ignored in the bottleneck "
      f"analysis. Spike runs at one instruction per cycle, so its cycle count equals instret "
      f"and is not a timing estimate for CV-Wally. Cycle-level numbers will come from the RTL.\n")

    w("### 4.1 Total dynamic instruction count\n")
    L.extend(table(["Quantity", "Instructions", "Share of run"], [
        ["Retired instructions, whole run", f"{total:,}", "100.0%"],
        ["Application (ELF) instructions", f"{app:,}", pct(app, total)],
        ["Proxy kernel (`pk`) / unmapped PCs", f"{pk:,}", pct(pk, total)],
        ["`Invoke()` window (`minstret` CSR)", f"{invoke:,}", pct(invoke, total, 2)],
        ["Outside `Invoke()` (setup + check + exit)", f"{outside_invoke:,}", pct(outside_invoke, total, 2)],
    ], "lrr"))
    w(f"One inference retires **≈{total / 1e9:.1f} billion instructions**. "
      f"{rvc_share} of the application instructions are 16-bit compressed (RVC) encodings.\n")

    w("### 4.2 Instruction mix\n")
    w("Shares are of application instructions. Compressed instructions are counted as the "
      "base instruction they expand to.\n")
    L.extend(table(["Class", "Count", "Share"],
                   [[r[0], r[1], r[2].strip()] for r in class_rows[1:]], "lrr"))
    w("*Load* and *store* are the integer memory instructions. *alu* covers all integer "
      "register/immediate arithmetic and logic (`add[i][w]`, shifts, compares, `lui`, ...). "
      "*mul* is the M-extension multiplies, *branch* the conditional branches, and *jump* "
      "`jal`/`jalr`. Division and system instructions are listed separately for "
      "completeness and are negligible.\n")
    top_mnem = sorted(mnemonics, key=lambda r: -int(r["count"]))[:8]
    w("The eight most frequent mnemonics are "
      + ", ".join(f"`{r['mnemonic']}` ({pct(int(r['count']), app)})" for r in top_mnem)
      + f". Full counts are in `{rel(Path(f'{args.spike}_mnemonics.csv'))}`.\n")

    w("### 4.3 Dynamic memory instructions\n")
    mem_rows = []
    for k in list(LOADS) + list(STORES):
        if mem_by_mnem.get(k):
            kind = "load" if k in LOADS else "store"
            width = (LOADS.get(k) or STORES.get(k)) * 8
            mem_rows.append([f"`{k}`", kind, f"{width}-bit", f"{mem_by_mnem[k]:,}",
                             pct(mem_by_mnem[k], mem)])
    L.extend(table(["Mnemonic", "Type", "Width", "Count", "Share of memory insns"],
                   mem_rows, "lllrr"))
    L.extend(table(["Quantity", "Value"], [
        ["Loads", f"{loads:,} ({pct(loads, app)} of app insns)"],
        ["Stores", f"{stores:,} ({pct(stores, app)} of app insns)"],
        ["**Total memory instructions**", f"**{mem:,} ({pct(mem, app)} of app insns)**"],
        ["Load : store ratio", f"{loads / stores:.2f} : 1"],
        ["Memory instructions per executed MAC", f"{mem / exec_macs:.2f}"],
        ["Bytes loaded / stored", f"{size_str(bytes_loaded)} / {size_str(bytes_stored)}"],
        ["Average bytes per memory access", f"{(bytes_loaded + bytes_stored) / mem:.2f} B"],
    ], "lr"))
    w(f"Memory instructions are {pct(mem, app)} of the dynamic stream, about "
      f"**{mem / exec_macs:.1f} per MAC**. Most loads are single bytes (`lb`): the reference "
      f"kernel fetches each int8 activation and weight separately. The hottest basic block, "
      f"the convolution inner loop at `0x{hb['pc']}`, has {hb['len']} instructions and runs "
      f"{hb['execs']:,} times, which equals the {op_macs.get('CONV_2D', 0):,} MACs executed by "
      f"`CONV_2D`. So **one loop iteration computes one MAC**, and this block alone "
      f"retires {hb['share']} of all application instructions:\n")
    w("```")
    L.extend(hb["code"])
    w("```\n")
    w(f"Of its {hb_mem} memory instructions, {hb_operand} load the activation and weight "
      f"bytes and {hb_stack} are stack (`sp`-relative) accesses: a register spill/reload of "
      f"the loaded value and a reload of the input zero-point offset on every iteration.\n")

    w("### 4.4 MAC operations per retired instruction\n")
    L.extend(table(["Quantity", "Value"], [
        ["MACs executed per inference (TFLM reference kernels)", f"{exec_macs:,}"],
        ["MACs per inference (from tensor shapes)", f"{shape_macs:,}"],
        ["Total retired instructions (Spike, whole run)", f"{total:,}"],
        ["**MACs per retired instruction** (executed MACs)", f"**{exec_macs / total:.4f}**"],
        ["Retired instructions per MAC (executed MACs)", f"{total / exec_macs:.2f}"],
        ["MACs per retired instruction (shape MACs)", f"{shape_macs / total:.4f}"],
    ], "lr"))
    w(f"The baseline completes **{exec_macs / total:.3f} MAC per instruction**, roughly one "
      f"MAC every {total / exec_macs:.1f} instructions. The executed-MAC count is the fair "
      f"denominator because it counts the work the reference kernels actually perform; "
      f"the shape-based figure is given for comparison with kernels that do not skip "
      f"padding taps. For reference, a packed `Xqmac8` that performs 8 MACs per instruction "
      f"would reach 8 MAC/instruction in its inner loop alone.\n")

    w("### 4.5 Execution contribution of major operators\n")
    w(f"**Top {args.top} functions by exclusive retired instructions.** "
      f"The class columns give each function's own instruction mix.\n")
    f_rows = []
    cum = 0
    for i, r in enumerate(funcs[: args.top], 1):
        n = int(r["count"])
        cum += n
        mix = [pct(int(r[c]), n) for c in cls_cols]
        f_rows.append([i, f"`{short(r['function'])}`", operator_of(r["function"]),
                       f"{n:,}", pct(n, app, 2), *mix])
    L.extend(table(["#", "Function", "Operator", "Instructions", "Share", "load", "store",
                    "alu", "mul", "branch"], f_rows, "rllrrrrrrr"))
    w(f"The top {args.top} functions cover {pct(cum, app, 2)} of application instructions.\n")

    w("**Per operator.** Function counts are summed by model operator and compared with the "
      "operator's executed MACs. `MultiplyByQuantizedMultiplier` is TFLM's out-of-line "
      "requantization helper, called once per output element by the `CONV_2D` and "
      "`FULLY_CONNECTED` kernels. It is listed separately; the rest of each operator's "
      "requantization (zero-point add and clamp, and the inlined path in `ADD`) is counted "
      "inside the operator.\n")
    order = ["CONV_2D", "Requantization helper", "ADD", "MAX_POOL_2D", "PAD",
             "FULLY_CONNECTED", "TRANSPOSE", "AVERAGE_POOL_2D", "Setup / runtime"]
    o_rows = []
    for op in order + sorted(set(op_insns) - set(order)):
        if op not in op_insns:
            continue
        n = op_insns[op]
        macs = op_macs.get(op)
        o_rows.append([f"`{op}`" if op in op_macs else op,
                       op_count.get(op, "–"), f"{n:,}", pct(n, app, 2),
                       f"{macs:,}" if macs else "–",
                       f"{n / macs:.2f}" if macs else "–"])
    L.extend(table(["Operator", "Ops in graph", "Instructions", "Share",
                    "MACs executed", "Insns / MAC"], o_rows, "lrrrrr"))
    conv_n = op_insns.get("CONV_2D", 0)
    w(f"`CONV_2D` retires {pct(conv_n, app)} of all application instructions "
      f"({conv_n / max(op_macs.get('CONV_2D', 1), 1):.2f} instructions per MAC). Every other "
      f"operator together is under {pct(app - conv_n, app, 0) if app - conv_n else '0%'}. "
      f"The fully connected layer runs at a similar instructions-per-MAC rate but is only "
      f"{pct(op_macs.get('FULLY_CONNECTED', 0), exec_macs, 2)} of the MACs.\n")

    w("### 4.6 Code size\n")
    sec_rows = [[f"`{x['name']}`", x["kind"], f"{x['size']:,}", size_str(x["size"])]
                for x in sorted(sections, key=lambda x: -x["size"]) if x["size"]]
    L.extend(table(["Section", "Contents", "Bytes", "Size"], sec_rows, "llrr"))
    sym_rows = [["ELF file size", f"{elf_bytes:,}", size_str(elf_bytes)],
                ["**Code (executable sections)**", f"**{code_bytes:,}**", f"**{size_str(code_bytes)}**"]]
    if model_bytes:
        sym_rows.append([f"Embedded model (`{model_sym}`)", f"{model_bytes:,}", size_str(model_bytes)])
    if arena_bytes:
        sym_rows.append(["Tensor arena (`g_tensor_arena`, zero-initialised)", f"{arena_bytes:,}",
                         size_str(arena_bytes)])
    L.extend(table(["Quantity", "Bytes", "Size"], sym_rows, "lrr"))
    w(f"`{rel(args.elf)}` is {size_str(elf_bytes)} on disk, but most of it is the "
      f"embedded model flatbuffer"
      f"{f' ({pct(model_bytes, elf_bytes)} of the file)' if model_bytes else ''}. "
      f"The executable code is **{size_str(code_bytes)}**: the TFLM interpreter, the "
      f"{len(op_rows) - 1} reference kernels, the flatbuffer parser, and newlib-nano. This "
      f"is the baseline for the code-size change that ISA extensions must report.\n")

    w("### 4.7 Three largest architectural bottlenecks\n")
    for i in range(1, 4):
        w(f"**Bottleneck {i}: Lorem ipsum.** {LOREM}\n")

    # 5 ---------------------------------------------------------------------
    w("## 5. Conclusion\n")
    w(LOREM + "\n")
    w(LOREM + "\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L), encoding="utf-8")
    print(f"[gen_baseline_report] {total:,} retired insns, {exec_macs / total:.4f} MAC/insn, "
          f"code {size_str(code_bytes)}")
    print(f"[gen_baseline_report] report: {rel(args.out)}")


if __name__ == "__main__":
    main()
