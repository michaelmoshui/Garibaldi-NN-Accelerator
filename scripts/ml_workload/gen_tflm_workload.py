#!/usr/bin/env python3
"""Package a .tflite model as a bare-metal LiteRT Micro (TFLM) workload.

Exports the shared components used by every app (test, demo, dev…):
  <out>/tflm/              TFLM runtime tree  (create_tflm_tree.py)
  <out>/model/*_model_data.*
                           C++ model array    (generate_cc_arrays.py)
  <out>/model/gen_micro_mutable_op_resolver.h
                           Op resolver header (gen_micro_mutable_op_resolver)

These outputs live under src/software/workload/ and are shared by both the
test app and the demo app.  Workload-specific data is generated separately:
  • Random golden test vectors → scripts/ml_workload/gen_test_data.py
  • Real ImageNet demo photos  → scripts/ml_workload/gen_demo_data.py

Requires Linux / WSL (create_tflm_tree.py drives tflite-micro's GNU Makefile).

Usage:
    python scripts/ml_workload/gen_tflm_workload.py
    python scripts/ml_workload/gen_tflm_workload.py \
        --model path/to/model.tflite --out path/to/workload
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_MODEL = REPO_ROOT / "src/software/model/resnet18_int8_channelwise.tflite"
DEFAULT_OUT = REPO_ROOT / "src/software/workload"
DEFAULT_TFLM_DIR = Path.home() / "tflite-micro"
DEFAULT_TFLM_COMMIT = "06f0b46177562d00c4b2e7b11ec377b5104f582e"
DEFAULT_TFLM_TARGET = "riscv32_generic"
TFLM_URL = "https://github.com/tensorflow/tflite-micro.git"


def run(cmd: list, cwd: Path = None, env: dict = None) -> None:
    """Run a command, echoing it, and abort on failure."""
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, check=True)


def checkout_tflm(tflm_dir: Path, commit: str) -> None:
    """Clone tflite-micro if needed and pin it to `commit`."""
    if not (tflm_dir / ".git").is_dir():
        run(["git", "clone", TFLM_URL, tflm_dir])
    head = subprocess.run(
        ["git", "-C", str(tflm_dir), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if head != commit:
        run(["git", "-C", tflm_dir, "fetch", "--depth", "1", "origin", commit])
        run(["git", "-C", tflm_dir, "checkout", "-q", commit])


def export_runtime(tflm_dir: Path, tools: Path, out: Path, target: str, env: dict) -> None:
    """Copy the TFLM library sources + third-party deps into out/tflm."""
    shutil.rmtree(out / "tflm", ignore_errors=True)
    run([sys.executable, tools / "project_generation/create_tflm_tree.py",
         f"--makefile_options=TARGET={target}", out / "tflm"],
        cwd=tflm_dir, env=env)


def cc_array(tools: Path, out: Path, rel_out: str, src: Path, env: dict) -> None:
    """Emit rel_out.{cc,h} from `src`.

    The path is passed relative to `out` because generate_cc_arrays.py writes
    it verbatim into the .cc's #include.
    """
    for ext in ("cc", "h"):
        run([sys.executable, tools / "generate_cc_arrays.py", f"{rel_out}.{ext}", src],
            cwd=out, env=env)


def gen_op_resolver(tflm_dir: Path, tools: Path, model: Path, out: Path, env: dict) -> None:
    """Emit model/gen_micro_mutable_op_resolver.h with only the model's ops."""
    run([sys.executable,
         tools / "gen_micro_mutable_op_resolver/generate_micro_mutable_op_resolver_from_model.py",
         f"--common_tflite_path={model.parent}",
         f"--input_tflite_files={model.name}",
         f"--output_dir={out / 'model'}",
         f"--verify_op_list_against_header="
         f"{tflm_dir / 'tensorflow/lite/micro/micro_mutable_op_resolver.h'}"],
        env=env)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                    help=f"input .tflite (default: {DEFAULT_MODEL.relative_to(REPO_ROOT)})")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"shared workload directory (default: {DEFAULT_OUT.relative_to(REPO_ROOT)})")
    ap.add_argument("--tflm-dir", type=Path, default=DEFAULT_TFLM_DIR,
                    help=f"tflite-micro checkout (default: {DEFAULT_TFLM_DIR})")
    ap.add_argument("--tflm-commit", default=DEFAULT_TFLM_COMMIT,
                    help="tflite-micro commit to pin")
    ap.add_argument("--tflm-target", default=DEFAULT_TFLM_TARGET,
                    help=f"TFLM Makefile TARGET for the export (default: {DEFAULT_TFLM_TARGET})")
    args = ap.parse_args()

    model = args.model.resolve()
    out = args.out.resolve()
    tflm_dir = args.tflm_dir.resolve()
    tools = tflm_dir / "tensorflow/lite/micro/tools"
    name = model.stem

    checkout_tflm(tflm_dir, args.tflm_commit)

    # tflite-micro's root-level tflite_micro.py maps `tflite_micro.*` imports
    # onto the source tree, so its Python tools run without Bazel.
    sys.path.insert(0, str(tflm_dir))
    # create_tflm_tree.py parses `make` stdout as file lists; inherited make
    # variables (when run from our Makefile) add "make[1]: Entering ..." lines.
    env = {k: v for k, v in os.environ.items()
           if k not in ("MAKEFLAGS", "MFLAGS", "MAKELEVEL", "MAKEOVERRIDES")}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(tflm_dir), env.get("PYTHONPATH")]))

    print(f"[1/3] Exporting LiteRT Micro runtime tree (TARGET={args.tflm_target})")
    export_runtime(tflm_dir, tools, out, args.tflm_target, env)

    print(f"[2/3] Embedding {model.name} as a C++ array")
    (out / "model").mkdir(parents=True, exist_ok=True)
    cc_array(tools, out, f"model/{name}_model_data", model, env)

    print("[3/3] Generating MicroMutableOpResolver for the model's operators")
    gen_op_resolver(tflm_dir, tools, model, out, env)

    print(f"[ok] Shared workload written to {out}")
    print(f"     Run 'make test_data' to generate random golden test vectors.")
    print(f"     Run 'make demo_data' to fetch ImageNet photo demo data.")


if __name__ == "__main__":
    main()
