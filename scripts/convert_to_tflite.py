#!/usr/bin/env python3
"""Convert EfficientFormer-L1 (timm / PyTorch) to an INT8 (W8A8) LiteRT/TFLite model.

Steps:
    1. load_model()           Load the pretrained EfficientFormer-L1 from HuggingFace
                              (via timm), set eval mode, and build a sample input
                              tensor used to trace tensor shapes during conversion.
    2. load_calibration_data()Load a class-balanced set of real images (resized +
                              ImageNet-normalized) used to measure activation ranges.
                              Falls back to random tensors if none are found.
    3. quantize_model()       Static PT2E post-training quantization to W8A8 INT8
                              (per-channel weights, static activations): export ->
                              prepare -> calibrate on the images -> convert.
    4. convert_to_litert()    Lower the quantized graph to LiteRT and export .tflite.

Prerequisite:
    Populate the calibration folder first (see CALIB_DIR):
        python scripts/fetch_calibration_images.py

Usage:
    python scripts/convert_to_tflite.py          # run all steps -> DEFAULT_OUT
"""

import glob
import os

import timm
import torch
from PIL import Image

import litert_torch
from litert_torch.quantize.pt2e_quantizer import (
    PT2EQuantizer,
    get_symmetric_quantization_config,
)
from litert_torch.quantize.quant_config import QuantConfig
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

MODEL_HF_ID = "efficientformer_l1.snap_dist_in1k"
DEFAULT_OUT = "src/software/model/efficientformer_l1_int8.tflite"

CALIB_DIR = "scripts/data/activation_calibration"
NUM_CALIB = 100

# ImageNet normalization (matches the checkpoint's preprocessing).
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_model():
    """Load the pretrained EfficientFormer-L1 from HuggingFace via timm (eval mode)."""

    print(f"[info] loading {MODEL_HF_ID} ...")
    model = timm.create_model(MODEL_HF_ID, pretrained=True)
    model.eval()

    # Input size the checkpoint expects (e.g. 224).
    cfg = getattr(model, "pretrained_cfg", {}) or {}
    input_size = cfg.get("input_size", (3, 224, 224))[-1]

    # A single example input, needed by the converter to trace shapes.
    sample_input = torch.zeros(1, 3, input_size, input_size)
    return model, sample_input


def load_calibration_data(input_size, calib_dir=CALIB_DIR, num_samples=NUM_CALIB):
    """Return a list of (1, 3, H, W) float tensors used to calibrate activation ranges.

    Loads real images from `calib_dir` (recursively). If none are found, falls back
    to a few random tensors so the pipeline still runs -- but INT8 accuracy will be
    meaningless until real ImageNet-val images are dropped into `calib_dir`.
    """
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPEG")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(calib_dir, "**", ext), recursive=True))
    files = sorted(files)[:num_samples]

    if not files:
        print(f"[warn] no images in {calib_dir!r}; using 8 RANDOM calibration samples "
              "(accuracy will be meaningless).")
        return [torch.randn(1, 3, input_size, input_size) for _ in range(8)]

    print(f"[info] loading {len(files)} calibration images from {calib_dir!r}")
    samples = []
    for path in files:
        img = Image.open(path).convert("RGB").resize((input_size, input_size), Image.BILINEAR)
        t = torch.frombuffer(img.tobytes(), dtype=torch.uint8).clone()
        t = t.view(input_size, input_size, 3).permute(2, 0, 1).float() / 255.0
        t = (t - IMAGENET_MEAN) / IMAGENET_STD
        samples.append(t.unsqueeze(0))
    return samples


def quantize_model(model, sample_input, calib_data):
    """Statically quantize the model to W8A8 (INT8) using PT2E, returning the
    quantized graph module and the quantizer (needed by the LiteRT converter)."""

    # Symmetric, per-channel weights, static activations (is_dynamic=False) -> full INT8.
    quantizer = PT2EQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_dynamic=False)
    )

    # Capture an ATen graph, insert observers, calibrate, then convert to int ops.
    exported = torch.export.export(model, (sample_input,)).module()
    prepared = prepare_pt2e(exported, quantizer)

    print(f"[info] calibrating on {len(calib_data)} samples ...")
    with torch.no_grad():
        for x in calib_data:
            prepared(x)

    quantized = convert_pt2e(prepared, fold_quantize=False)
    return quantized, quantizer


def convert_to_litert(model, sample_input, out_path, quantizer=None):
    """Convert a (quantized) PyTorch model to LiteRT and write it to `out_path`."""

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    quant_config = QuantConfig(pt2e_quantizer=quantizer) if quantizer is not None else None
    print("[info] converting PyTorch -> LiteRT ...")
    edge_model = litert_torch.convert(model, (sample_input,), quant_config=quant_config)
    edge_model.export(out_path)

    print(f"[ok] wrote LiteRT model -> {out_path}")
    return out_path


def main():
    model, sample_input = load_model()
    input_size = sample_input.shape[-1]

    calib_data = load_calibration_data(input_size)
    q_model, quantizer = quantize_model(model, sample_input, calib_data)
    convert_to_litert(q_model, sample_input, DEFAULT_OUT, quantizer)


if __name__ == "__main__":
    main()
