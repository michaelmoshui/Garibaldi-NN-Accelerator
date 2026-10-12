#!/usr/bin/env python3
"""Generate random golden input/output vectors for the bare-metal test app.

Uses LiteRT's BUILTIN_REF kernels (via TFLM's TestDataGenerator) to produce
bit-exact reference outputs for a random int8 input.  The outputs land in
src/software/test/test_data/ and are compiled directly into the test binary.

This script is test-app-exclusive.  The shared TFLM runtime + model CC array
must be generated first (scripts/ml_workload/gen_tflm_workload.py / make model_workload).
Demo-specific data (real ImageNet photos) lives in gen_demo_data.py.

Requires Linux / WSL (uses the TFLM Python tools from the tflite-micro checkout).

Usage:
    python scripts/ml_workload/gen_test_data.py
    python scripts/ml_workload/gen_test_data.py \
        --model path/to/model.tflite --out path/to/test --seed 42
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

# gen_tflm_workload.py is in the same directory; Python adds the script's
# directory to sys.path so the import below works when invoked as a script.
from gen_tflm_workload import (
    DEFAULT_MODEL,
    DEFAULT_TFLM_COMMIT,
    DEFAULT_TFLM_DIR,
    REPO_ROOT,
    cc_array,
    checkout_tflm,
)

DEFAULT_OUT = REPO_ROOT / "src/software/test"


def gen_test_data(tools: Path, model: Path, out: Path, seed: int, env: dict) -> None:
    """Random int8 input + LiteRT reference-kernel output, as C++ arrays."""
    import numpy as np
    from tflite_micro.tensorflow.lite.micro.tools.generate_test_for_model import (
        TestDataGenerator,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy(model, tmp / model.name)
        # TestDataGenerator writes its CSVs next to the model path it is given.
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            np.random.seed(seed)
            TestDataGenerator(".", [model.name], [0]).generate_golden_single_in_single_out()
        finally:
            os.chdir(cwd)

        for csv in sorted(tmp.glob("*_int8.csv")):
            cc_array(tools, out, f"test_data/{csv.stem}_test_data", csv, env)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                    help=f"input .tflite (default: {DEFAULT_MODEL.relative_to(REPO_ROOT)})")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"test app directory for test_data/ "
                         f"(default: {DEFAULT_OUT.relative_to(REPO_ROOT)})")
    ap.add_argument("--tflm-dir", type=Path, default=DEFAULT_TFLM_DIR,
                    help=f"tflite-micro checkout (default: {DEFAULT_TFLM_DIR})")
    ap.add_argument("--tflm-commit", default=DEFAULT_TFLM_COMMIT,
                    help="tflite-micro commit to pin")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for the golden input (default: 0)")
    args = ap.parse_args()

    model = args.model.resolve()
    out = args.out.resolve()
    tflm_dir = args.tflm_dir.resolve()
    tools = tflm_dir / "tensorflow/lite/micro/tools"

    checkout_tflm(tflm_dir, args.tflm_commit)

    sys.path.insert(0, str(tflm_dir))
    env = {k: v for k, v in os.environ.items()
           if k not in ("MAKEFLAGS", "MFLAGS", "MAKELEVEL", "MAKEOVERRIDES")}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(tflm_dir), env.get("PYTHONPATH")]))

    print(f"[1/1] Generating golden input/output with LiteRT reference kernels (seed={args.seed})")
    shutil.rmtree(out / "test_data", ignore_errors=True)
    (out / "test_data").mkdir(parents=True, exist_ok=True)
    gen_test_data(tools, model, out, args.seed, env)

    print(f"[ok] Golden test vectors written to {out / 'test_data'}")


if __name__ == "__main__":
    main()
