# =============================================================================
# Garibaldi NN Accelerator — top-level build orchestration
# Requires: GNU make (Git Bash / MSYS2 / WSL recommended on Windows)
# =============================================================================

# ---------------------------------------------------------------------------
# OS detection — adjust venv binary path for Windows vs Unix
# ---------------------------------------------------------------------------
ifeq ($(OS),Windows_NT)
    PYTHON   := python
    VENV_BIN := .venv/Scripts
else
    PYTHON   := python3
    VENV_BIN := .venv/bin
endif

VENV_PYTHON := $(VENV_BIN)/python
VENV_PIP    := $(VENV_BIN)/pip

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REQS            := requirements.txt

# Pre-quantized W8A8 INT8 TFLite model fetched directly from HuggingFace.
# litert-community/resnet18 — resnet18_int8_channelwise.tflite
#   Weights:     INT8, symmetric per-channel
#   Activations: INT8, asymmetric  (true W8A8 — no calibration needed)
MODEL_OUT       := src/software/model/resnet18_int8_channelwise.tflite
MODEL_GEN_DIR   := src/software/model

# Sentinel files (touched after each step succeeds; avoids redundant re-runs)
SETUP_SENTINEL     := .venv/.setup_done
MODEL_GEN_SENTINEL := $(MODEL_GEN_DIR)/.model_gen_done

# ---------------------------------------------------------------------------
# Phony targets
# ---------------------------------------------------------------------------
.PHONY: all setup fetch inspect model_gen clean help

## Default: run the full pipeline end-to-end
all: model_gen

# ---------------------------------------------------------------------------
# 1. Set up Python virtual environment
# ---------------------------------------------------------------------------
setup: $(SETUP_SENTINEL)

$(SETUP_SENTINEL): $(REQS)
	@echo "[setup] Creating virtual environment..."
	$(PYTHON) -m venv .venv
	@echo "[setup] Installing dependencies from $(REQS)..."
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r $(REQS)
	@echo "setup done" > $(SETUP_SENTINEL)
	@echo "[ok] Python environment ready."

# ---------------------------------------------------------------------------
# 2. Download the pre-quantized TFLite model from HuggingFace
# ---------------------------------------------------------------------------
fetch: $(MODEL_OUT)

$(MODEL_OUT): $(SETUP_SENTINEL) scripts/fetch_model.py
	@echo "[fetch] Downloading resnet18_int8_channelwise.tflite from HuggingFace..."
	$(VENV_PYTHON) scripts/fetch_model.py
	@echo "[ok] Model ready at $(MODEL_OUT)."

# ---------------------------------------------------------------------------
# 3. (Optional) Inspect the TFLite model — op inventory + tensor summary
# ---------------------------------------------------------------------------
inspect: $(MODEL_OUT) $(SETUP_SENTINEL)
	@echo "[inspect] Printing op inventory and tensor summary..."
	$(VENV_PYTHON) scripts/inspect_tflite.py --model $(MODEL_OUT)

# ---------------------------------------------------------------------------
# 4. Generate static C source files from the TFLite model
#    inspect_tflite.py runs first so the op inventory is visible in the log
#    and tflite_to_c.py dynamically filters OP_KERNEL to only present ops.
# ---------------------------------------------------------------------------
model_gen: $(MODEL_GEN_SENTINEL)

$(MODEL_GEN_SENTINEL): $(MODEL_OUT) $(SETUP_SENTINEL) scripts/inspect_tflite.py scripts/tflite_to_c.py
	@echo "[inspect] Op inventory for $(MODEL_OUT)..."
	$(VENV_PYTHON) scripts/inspect_tflite.py --model $(MODEL_OUT)
	@echo "[model_gen] Generating ISA-independent C from $(MODEL_OUT)..."
	$(VENV_PYTHON) scripts/tflite_to_c.py --model $(MODEL_OUT)
	@echo "model_gen done" > $(MODEL_GEN_SENTINEL)
	@echo "[ok] C sources written to $(MODEL_GEN_DIR)/."

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
clean:
	git clean -fdx .

help:
	@echo ""
	@echo "  make setup      — create .venv and install requirements"
	@echo "  make fetch      — download resnet18_int8_channelwise.tflite from HuggingFace"
	@echo "  make inspect    — print op inventory + tensor summary (optional)"
	@echo "  make model_gen  — generate static C from the TFLite model"
	@echo "  make all        — run the full pipeline (default)"
	@echo "  make clean      — remove venv, model outputs, generated C sources"
	@echo ""
	@echo "  Model: litert-community/resnet18 / resnet18_int8_channelwise.tflite"
	@echo "         W8A8: weights INT8 per-channel symmetric,"
	@echo "               activations INT8 asymmetric."
	@echo ""
