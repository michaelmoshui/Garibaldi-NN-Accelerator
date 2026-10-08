#!/usr/bin/env python3
"""Dynamic instruction mix from a Spike PC histogram (spike -g).

Spike's -g flag counts how many times each PC retires and prints the table to
stderr on exit ("PC Histogram size:N" followed by "<hex pc> <count>" lines).
Joining those counts with the ELF disassembly gives the exact dynamic
instruction mix — per mnemonic, per instruction class, per function, and per
basic block — without writing a per-instruction commit log.

Usage:
  python3 scripts/insn_mix.py --hist pc_hist.txt --disasm disasm.txt \
      [--spike-out spike_stdout.txt] --out results/insn_mix/resnet18_test

The disassembly must come from `objdump -d -C -M no-aliases` so that
compressed instructions keep their c.* names and C++ symbols are demangled.
Writes <out>.md (report), <out>_mnemonics.csv and <out>_functions.csv.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

HIST_RE = re.compile(r"^([0-9a-f]+) (\d+)$")
FUNC_RE = re.compile(r"^([0-9a-f]+) <(.+)>:$")
INSN_RE = re.compile(r"^\s*([0-9a-f]+):\s+([0-9a-f]+)\s+(\S+)(?:\s+(.*))?$")
INSTRET_RE = re.compile(r"invoke instret:\s*(\d+)")
CYCLES_RE = re.compile(r"invoke cycles:\s*(\d+)")

# RVC mnemonic -> the base instruction it expands to.
RVC_BASE = {
    "c.addi": "addi", "c.addiw": "addiw", "c.addi16sp": "addi",
    "c.addi4spn": "addi", "c.li": "addi", "c.nop": "addi", "c.lui": "lui",
    "c.mv": "add", "c.add": "add", "c.addw": "addw", "c.sub": "sub",
    "c.subw": "subw", "c.and": "and", "c.andi": "andi", "c.or": "or",
    "c.xor": "xor", "c.slli": "slli", "c.srli": "srli", "c.srai": "srai",
    "c.lw": "lw", "c.ld": "ld", "c.lwsp": "lw", "c.ldsp": "ld",
    "c.sw": "sw", "c.sd": "sd", "c.swsp": "sw", "c.sdsp": "sd",
    "c.j": "jal", "c.jr": "jalr", "c.jalr": "jalr",
    "c.beqz": "beq", "c.bnez": "bne", "c.ebreak": "ebreak",
}

CLASSES = {
    "load": {"lb", "lbu", "lh", "lhu", "lw", "lwu", "ld"},
    "store": {"sb", "sh", "sw", "sd"},
    "mul": {"mul", "mulh", "mulhsu", "mulhu", "mulw"},
    "div": {"div", "divu", "rem", "remu", "divw", "divuw", "remw", "remuw"},
    "branch": {"beq", "bne", "blt", "bge", "bltu", "bgeu"},
    "jump": {"jal", "jalr"},
}
CLASS_OF = {m: c for c, ms in CLASSES.items() for m in ms}
CLASS_ORDER = ["load", "store", "alu", "mul", "div", "branch", "jump", "system"]


def classify(base: str) -> str:
    if base in CLASS_OF:
        return CLASS_OF[base]
    if base.startswith(("csr", "fence")) or base in {"ecall", "ebreak", "wfi", "mret", "sret"}:
        return "system"
    return "alu"


def parse_hist(path: Path) -> dict[int, int]:
    hist: dict[int, int] = {}
    seen_header = False
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("PC Histogram size:"):
            seen_header = True
            continue
        m = HIST_RE.match(line) if seen_header else None
        if m:
            hist[int(m.group(1), 16)] = int(m.group(2))
    if not seen_header:
        raise SystemExit(f"{path}: no 'PC Histogram size:' header — was Spike run with -g?")
    return hist


def parse_disasm(path: Path) -> dict[int, tuple[str, str, bool, str]]:
    """pc -> (function, mnemonic, is_compressed, disassembly text)."""
    insns = {}
    func = "?"
    for line in path.read_text(errors="replace").splitlines():
        m = FUNC_RE.match(line)
        if m:
            func = m.group(2)
            continue
        m = INSN_RE.match(line)
        if m:
            pc, enc, mnem, ops = m.groups()
            insns[int(pc, 16)] = (func, mnem, len(enc) == 4, f"{mnem} {ops or ''}".strip())
    return insns


def pct(n: int, total: int) -> str:
    return f"{100.0 * n / total:5.1f}%" if total else "  n/a"


def short(name: str, width: int = 90) -> str:
    return name if len(name) <= width else name[: width - 3] + "..."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hist", type=Path, required=True, help="Spike -g stderr output")
    ap.add_argument("--disasm", type=Path, required=True, help="objdump -d -C -M no-aliases output")
    ap.add_argument("--spike-out", type=Path, help="program stdout (for invoke cycles/instret)")
    ap.add_argument("--out", type=Path, required=True, help="output path prefix")
    ap.add_argument("--top", type=int, default=25, help="rows in the per-function / per-mnemonic tables")
    ap.add_argument("--blocks", type=int, default=10, help="hot basic blocks to print")
    args = ap.parse_args()

    hist = parse_hist(args.hist)
    insns = parse_disasm(args.disasm)

    total = sum(hist.values())
    unmapped = sum(n for pc, n in hist.items() if pc not in insns)
    user = total - unmapped

    by_mnem: dict[str, int] = defaultdict(int)
    by_class: dict[str, int] = defaultdict(int)
    by_func: dict[str, int] = defaultdict(int)
    func_class: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    compressed = 0
    for pc, n in hist.items():
        if pc not in insns:
            continue
        func, mnem, is_c, _ = insns[pc]
        base = RVC_BASE.get(mnem, mnem)
        cls = classify(base)
        by_mnem[base] += n
        by_class[cls] += n
        by_func[func] += n
        func_class[func][cls] += n
        if is_c:
            compressed += n

    # Basic blocks: maximal runs of consecutive instructions in one function
    # that all retired the same number of times.
    blocks = []
    cur: list[int] = []
    for pc in sorted(p for p in hist if p in insns):
        if cur:
            prev = cur[-1]
            prev_len = 2 if insns[prev][2] else 4
            if pc == prev + prev_len and hist[pc] == hist[prev] and insns[pc][0] == insns[prev][0]:
                cur.append(pc)
                continue
            blocks.append(cur)
        cur = [pc]
    if cur:
        blocks.append(cur)
    blocks.sort(key=lambda b: hist[b[0]] * len(b), reverse=True)

    instret = cycles = None
    if args.spike_out and args.spike_out.exists():
        text = args.spike_out.read_text(errors="replace")
        if m := INSTRET_RE.search(text):
            instret = int(m.group(1))
        if m := CYCLES_RE.search(text):
            cycles = int(m.group(1))

    # ---- report ----------------------------------------------------------
    L = []
    L.append(f"# Dynamic instruction mix — {args.out.name}\n")
    L.append("Source: Spike `-g` PC histogram joined with `objdump -d` of the ELF.\n")
    L.append("| Quantity | Count |")
    L.append("|---|---:|")
    L.append(f"| Retired instructions, whole run | {total:,} |")
    L.append(f"| Application (ELF) instructions | {user:,} ({pct(user, total).strip()}) |")
    L.append(f"| Proxy-kernel / unmapped PCs | {unmapped:,} ({pct(unmapped, total).strip()}) |")
    if instret is not None:
        L.append(f"| `Invoke()` instret (CSR) | {instret:,} |")
    if cycles is not None:
        L.append(f"| `Invoke()` cycles (CSR) | {cycles:,} |")
    L.append(f"| Compressed (RVC) share of app insns | {pct(compressed, user).strip()} |")
    L.append("")
    L.append("Percentages below are of application instructions.\n")

    L.append("## By instruction class\n")
    L.append("| Class | Count | Share |")
    L.append("|---|---:|---:|")
    for cls in CLASS_ORDER:
        if by_class.get(cls):
            L.append(f"| {cls} | {by_class[cls]:,} | {pct(by_class[cls], user)} |")
    L.append("")

    L.append(f"## Top {args.top} mnemonics (RVC folded into base op)\n")
    L.append("| Mnemonic | Class | Count | Share |")
    L.append("|---|---|---:|---:|")
    for mnem, n in sorted(by_mnem.items(), key=lambda kv: -kv[1])[: args.top]:
        L.append(f"| `{mnem}` | {classify(mnem)} | {n:,} | {pct(n, user)} |")
    L.append("")

    L.append(f"## Top {args.top} functions (exclusive)\n")
    L.append("| Function | Count | Share | load | store | alu | mul | branch |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    cum = 0
    for func, n in sorted(by_func.items(), key=lambda kv: -kv[1])[: args.top]:
        cum += n
        fc = func_class[func]
        mix = " | ".join(pct(fc.get(c, 0), n) for c in ("load", "store", "alu", "mul", "branch"))
        L.append(f"| `{short(func)}` | {n:,} | {pct(n, user)} | {mix} |")
    L.append(f"\nTop {args.top} functions cover {pct(cum, user).strip()} of application instructions.\n")

    L.append(f"## Top {args.blocks} basic blocks (by retired instructions)\n")
    for b in blocks[: args.blocks]:
        n = hist[b[0]]
        L.append(f"### `{short(insns[b[0]][0])}` @ 0x{b[0]:x}\n")
        L.append(f"{len(b)} insns x {n:,} executions = {len(b) * n:,} ({pct(len(b) * n, user).strip()})\n")
        L.append("```")
        for pc in b:
            L.append(f"{pc:8x}:  {insns[pc][3]}")
        L.append("```\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = args.out.with_suffix(".md")
    report.write_text("\n".join(L), encoding="utf-8")

    with open(f"{args.out}_mnemonics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mnemonic", "class", "count"])
        for mnem, n in sorted(by_mnem.items(), key=lambda kv: -kv[1]):
            w.writerow([mnem, classify(mnem), n])
    with open(f"{args.out}_functions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["function", "count", *CLASS_ORDER])
        for func, n in sorted(by_func.items(), key=lambda kv: -kv[1]):
            w.writerow([func, n, *(func_class[func].get(c, 0) for c in CLASS_ORDER)])

    print(f"[insn_mix] {total:,} retired ({user:,} app, {unmapped:,} pk/unmapped)")
    for cls in CLASS_ORDER:
        if by_class.get(cls):
            print(f"  {cls:7s} {pct(by_class[cls], user)}")
    print(f"[insn_mix] report: {report}")


if __name__ == "__main__":
    main()
