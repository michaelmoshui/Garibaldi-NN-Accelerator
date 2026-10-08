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
TEST_DIR        := src/software/test
DEMO_DIR        := src/software/demo

# Demo: number of photos to classify (empty = all 20), e.g. make demo_spike N=2
N ?=

# Spike ISA simulator (override on command line if not on PATH).
# `make get_spike` installs into $HOME/.local/riscv when apt has no packages.
SPIKE_PREFIX ?= $(HOME)/.local/riscv
SPIKE        ?= $(if $(wildcard $(SPIKE_PREFIX)/bin/spike),$(SPIKE_PREFIX)/bin/spike,spike)
PK           ?= $(if $(wildcard $(SPIKE_PREFIX)/riscv64-unknown-elf/bin/pk),$(SPIKE_PREFIX)/riscv64-unknown-elf/bin/pk,pk)

# Sentinel files (touched after each step succeeds; avoids redundant re-runs)
SETUP_SENTINEL     := .venv/.setup_done
WORKLOAD_SENTINEL  := $(WORKLOAD_DIR)/.workload_done
DEMO_SENTINEL      := $(DEMO_DIR)/.demo_done

# ---------------------------------------------------------------------------
# Phony targets
# ---------------------------------------------------------------------------
.PHONY: all setup fetch inspect workload workload_run compile get_spike spike_run spike_mix \
        demo_data demo demo_spike clean help

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
	$(VENV_PYTHON) scripts/gen_tflm_workload.py --model $(MODEL_OUT) --out $(WORKLOAD_DIR) \
	    --test-out $(TEST_DIR)
	@echo "workload done" > $(WORKLOAD_SENTINEL)
	@echo "[ok] Workload written to $(WORKLOAD_DIR)/, golden vectors to $(TEST_DIR)/test_data/."

## Native build + bit-exact check against the LiteRT reference output
workload_run: $(WORKLOAD_SENTINEL)
	$(MAKE) -C $(TEST_DIR) run

# ---------------------------------------------------------------------------
# 5. Compile the golden-vector test to a bare-metal RV64IMC ELF
#    Uses the RISC-V toolchain bundled inside the tflite-micro checkout.
#    Output: src/software/workload/build/rv64/resnet18_test.elf
# ---------------------------------------------------------------------------
## Compile the TFLM test program to a bare-metal RV64IMC ELF (for Spike / CV-Wally)
compile: $(WORKLOAD_SENTINEL)
	$(MAKE) -C $(TEST_DIR) all CONFIG=rv64

# ---------------------------------------------------------------------------
# 6. Spike RISC-V ISA simulator
# ---------------------------------------------------------------------------
## Install Spike + proxy kernel (pk) on WSL/Linux (source build; apt has no spike on Ubuntu 24.04)
get_spike:
	bash scripts/install_spike.sh

## Build the RV64 ELF and run one inference under Spike (prints cycles + instret)
spike_run: compile
	$(MAKE) -C $(TEST_DIR) spike SPIKE=$(SPIKE) PK=$(PK)

## Dynamic instruction mix (Spike -g PC histogram + disassembly) -> results/insn_mix/
spike_mix: compile
	$(MAKE) -C $(TEST_DIR) spike_mix SPIKE=$(SPIKE) PK=$(PK)

# ---------------------------------------------------------------------------
# 7. Demo: classify 20 real ImageNet photos (vs. the random golden vector
#    used by workload_run / spike_run during development)
# ---------------------------------------------------------------------------
## Download the photos, quantize them, and record LiteRT reference logits
demo_data: $(DEMO_SENTINEL)

$(DEMO_SENTINEL): $(MODEL_OUT) $(SETUP_SENTINEL) scripts/gen_demo_data.py
	@echo "[demo] Generating ImageNet photo demo data..."
	$(VENV_PYTHON) scripts/gen_demo_data.py --model $(MODEL_OUT) --out $(DEMO_DIR)
	@echo "demo done" > $(DEMO_SENTINEL)
	@echo "[ok] Demo data written to $(DEMO_DIR)/."

## Native build + classify the photos (fast; seconds per photo)
demo: $(WORKLOAD_SENTINEL) $(DEMO_SENTINEL)
	$(MAKE) -C $(DEMO_DIR) run N=$(N)

## Classify the photos on the bare-metal RV64 ELF under Spike (slow; use N=)
demo_spike: $(WORKLOAD_SENTINEL) $(DEMO_SENTINEL)
	$(MAKE) -C $(DEMO_DIR) spike N=$(N) SPIKE=$(SPIKE) PK=$(PK)

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
	@echo "  make compile    — cross-compile to bare-metal RV64IMC ELF"
	@echo "  make get_spike  — build Spike + pk from source (WSL/Linux)"
	@echo "  make spike_run  — run the RV64 ELF under Spike (cycles + instret)"
	@echo "  make spike_mix  — dynamic instruction mix under Spike -> results/insn_mix/"
	@echo "  make demo_data  — fetch 20 ImageNet photos + LiteRT reference logits"
	@echo "  make demo       — demo: classify the photos (native build) [N=count]"
	@echo "  make demo_spike — demo: classify the photos under Spike    [N=count]"
	@echo "  make all        — run the full pipeline (default: workload)"
	@echo "  make clean      — remove venv, model outputs, generated workload"
	@echo ""
	@echo "  Model: litert-community/resnet18 / resnet18_int8_channelwise.tflite"
	@echo "         W8A8: weights INT8 per-channel symmetric,"
	@echo "               activations INT8 asymmetric."
	@echo ""
