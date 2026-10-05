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
WORKLOAD_DIR    := src/software/workload

# Sentinel files (touched after each step succeeds; avoids redundant re-runs)
SETUP_SENTINEL     := .venv/.setup_done
WORKLOAD_SENTINEL  := $(WORKLOAD_DIR)/.workload_done

# ---------------------------------------------------------------------------
# Phony targets
# ---------------------------------------------------------------------------
.PHONY: all setup fetch inspect workload workload_run clean help

## Default: run the full pipeline end-to-end
all: workload

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
# 4. Package the model as a bare-metal LiteRT Micro (TFLM) workload
#    Uses only tflite-micro's own generators (runtime tree export, model
#    array, op resolver, golden test vectors). Linux / WSL only.
# ---------------------------------------------------------------------------
workload: $(WORKLOAD_SENTINEL)

$(WORKLOAD_SENTINEL): $(MODEL_OUT) $(SETUP_SENTINEL) scripts/gen_tflm_workload.py
	@echo "[workload] Generating LiteRT Micro workload from $(MODEL_OUT)..."
	$(VENV_PYTHON) scripts/gen_tflm_workload.py --model $(MODEL_OUT) --out $(WORKLOAD_DIR)
	@echo "workload done" > $(WORKLOAD_SENTINEL)
	@echo "[ok] Workload written to $(WORKLOAD_DIR)/."

## Native build + bit-exact check against the LiteRT reference output
workload_run: $(WORKLOAD_SENTINEL)
	$(MAKE) -C $(WORKLOAD_DIR) run

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
	@echo "  make workload   — generate the bare-metal LiteRT Micro workload"
	@echo "  make workload_run — build it natively and check against LiteRT"
	@echo "  make all        — run the full pipeline (default: workload)"
	@echo "  make clean      — remove venv, model outputs, generated workload"
	@echo ""
	@echo "  Model: litert-community/resnet18 / resnet18_int8_channelwise.tflite"
	@echo "         W8A8: weights INT8 per-channel symmetric,"
	@echo "               activations INT8 asymmetric."
	@echo ""
