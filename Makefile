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
REQS        := requirements.txt
CALIB_DIR   := scripts/data/activation_calibration
MODEL_OUT   := src/software/model/efficientformer_l1_int8.tflite
MODEL_GEN_DIR   := src/software/model

# Sentinel files (touched after each step succeeds; avoids redundant re-runs)
SETUP_SENTINEL   := .venv/.setup_done
CALIB_SENTINEL   := $(CALIB_DIR)/.fetch_done
MODEL_GEN_SENTINEL   := $(MODEL_GEN_DIR)/.model_gen_done

# ---------------------------------------------------------------------------
# Phony targets
# ---------------------------------------------------------------------------
.PHONY: all setup calibration convert inspect model_gen clean help

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
# 2. Fetch calibration images (Imagenette subset of ImageNet)
# ---------------------------------------------------------------------------
calibration: $(CALIB_SENTINEL)

$(CALIB_SENTINEL): $(SETUP_SENTINEL) scripts/fetch_calibration_images.py
	@echo "[calibration] Fetching calibration images..."
	$(VENV_PYTHON) scripts/fetch_calibration_images.py
	@echo "fetch done" > $(CALIB_SENTINEL)
	@echo "[ok] Calibration images ready in $(CALIB_DIR)."

# ---------------------------------------------------------------------------
# 3. Convert EfficientFormer-L1 to INT8 TFLite
# ---------------------------------------------------------------------------
convert: $(MODEL_OUT)

$(MODEL_OUT): $(CALIB_SENTINEL) scripts/convert_to_tflite.py
	@echo "[convert] Running PyTorch → INT8 LiteRT conversion..."
	$(VENV_PYTHON) scripts/convert_to_tflite.py
	@echo "[ok] Model written to $(MODEL_OUT)."

# ---------------------------------------------------------------------------
# 4. (Optional) Inspect the TFLite model — op inventory + tensor summary
# ---------------------------------------------------------------------------
inspect: $(MODEL_OUT) $(SETUP_SENTINEL)
	@echo "[inspect] Printing op inventory and tensor summary..."
	$(VENV_PYTHON) scripts/inspect_tflite.py --model $(MODEL_OUT)

# ---------------------------------------------------------------------------
# 5. Generate static C source files from the TFLite model
# ---------------------------------------------------------------------------
model_gen: $(MODEL_GEN_SENTINEL)

$(MODEL_GEN_SENTINEL): $(MODEL_OUT) $(SETUP_SENTINEL) scripts/tflite_to_c.py
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
	@echo "  make setup        — create .venv and install requirements"
	@echo "  make calibration  — fetch 100 Imagenette calibration images"
	@echo "  make convert      — quantise + export INT8 TFLite model"
	@echo "  make inspect      — print op inventory + tensor summary (optional)"
	@echo "  make model_gen        — generate static C from the TFLite model"
	@echo "  make all          — run the full pipeline (default)"
	@echo "  make clean        — remove venv, cached data, model outputs, C sources"
	@echo ""
