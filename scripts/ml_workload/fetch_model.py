#!/usr/bin/env python3
"""Download a pre-quantized INT8 W8A8 TFLite model from HuggingFace.

Default model: resnet18_int8_channelwise.tflite (litert-community/resnet18)
  - Quantization recipe: STATIC_WI8_AI8
  - Weights:     signed INT8, symmetric per-channel quantization
  - Activations: signed INT8, asymmetric quantization   ← TRUE W8A8
  - No calibration or post-hoc quantization needed.

Usage:
    python scripts/ml_workload/fetch_model.py
    python scripts/ml_workload/fetch_model.py --repo litert-community/resnet18 \\
                                  --file resnet18_int8_channelwise.tflite \\
                                  --out  src/software/model/resnet18_int8_channelwise.tflite
"""

import argparse
import os
import shutil

DEFAULT_REPO = "litert-community/resnet18"
DEFAULT_FILE = "resnet18_int8_channelwise.tflite"
DEFAULT_OUT  = "src/software/model/resnet18_int8_channelwise.tflite"


def fetch(repo: str, filename: str, out_path: str) -> str:
    """Download `filename` from HuggingFace Hub repo `repo` and save to `out_path`."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "[error] huggingface_hub is not installed.\n"
            "        Run:  pip install huggingface_hub"
        ) from exc

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    print(f"[info] downloading {filename} from {repo} ...")
    cached = hf_hub_download(repo_id=repo, filename=filename)

    shutil.copy2(cached, out_path)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"[ok] saved → {out_path}  ({size_mb:.1f} MB)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help=f"HuggingFace repo id  (default: {DEFAULT_REPO})")
    ap.add_argument("--file", default=DEFAULT_FILE,
                    help=f"filename inside the repo  (default: {DEFAULT_FILE})")
    ap.add_argument("--out",  default=DEFAULT_OUT,
                    help=f"local output path  (default: {DEFAULT_OUT})")
    args = ap.parse_args()

    fetch(args.repo, args.file, args.out)
    print("\n[note] Model is W8A8 INT8:")
    print("         weights      — signed INT8, symmetric per-channel")
    print("         activations  — signed INT8, asymmetric (non-zero zero-points)")
    print("       Run 'make inspect' to view the op inventory and tensor layout.")


if __name__ == "__main__":
    main()
