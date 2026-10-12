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
SW_DIR          := src/software
WORKLOAD_DIR    := src/software/workload
TEST_DIR        := src/software/test
DEMO_DIR        := src/software/demo
DEV_DIR         := src/software_dev
DEV_MODEL       := $(DEV_DIR)/model/dev_net_int8.tflite
HW_DIR          := src/hardware

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
TEST_DATA_SENTINEL := $(TEST_DIR)/.test_data_done
DEMO_SENTINEL      := $(DEMO_DIR)/.demo_done
DEV_SENTINEL       := $(DEV_DIR)/.dev_workload_done

# ---------------------------------------------------------------------------
# Phony targets
# ---------------------------------------------------------------------------
.PHONY: setup fetch_model model_inspect model_workload test_data test_run test_compile test_spike \
        get_spike baseline_report \
        demo_data demo_run demo_compile demo_spike \
        dev_workload dev_run dev_compile dev_spike dev_inspect dev_cvw_compile dev_cvw_run \
        get_hw_tools hw_setup hw_lint hw_sim clean help

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
fetch_model: $(MODEL_OUT)

$(MODEL_OUT): $(SETUP_SENTINEL) scripts/ml_workload/fetch_model.py
	@echo "[fetch_model] Downloading resnet18_int8_channelwise.tflite from HuggingFace..."
	$(VENV_PYTHON) scripts/ml_workload/fetch_model.py
	@echo "[ok] Model ready at $(MODEL_OUT)."

# ---------------------------------------------------------------------------
# 3. (Optional) Inspect the TFLite model — op inventory + tensor summary
# ---------------------------------------------------------------------------
model_inspect: $(MODEL_OUT) $(SETUP_SENTINEL)
	@echo "[model_inspect] Printing op inventory and tensor summary..."
	$(VENV_PYTHON) scripts/result_extraction/inspect_tflite.py --model $(MODEL_OUT)

# ---------------------------------------------------------------------------
# 4. Export the shared LiteRT Micro runtime tree, embed the model as a C++
#    array, and generate the MicroMutableOpResolver header.
#    Used by BOTH the test app and the demo app; neither app-specific dataset
#    is produced here (see test_data and demo_data targets below).
#    Linux / WSL only.
# ---------------------------------------------------------------------------
model_workload: $(WORKLOAD_SENTINEL)

$(WORKLOAD_SENTINEL): $(MODEL_OUT) $(SETUP_SENTINEL) scripts/ml_workload/gen_tflm_workload.py
	@echo "[model_workload] Generating shared LiteRT Micro runtime from $(MODEL_OUT)..."
	$(VENV_PYTHON) scripts/ml_workload/gen_tflm_workload.py \
	    --model $(MODEL_OUT) --out $(WORKLOAD_DIR)
	@echo "workload done" > $(WORKLOAD_SENTINEL)
	@echo "[ok] Shared workload written to $(WORKLOAD_DIR)/."

# ---------------------------------------------------------------------------
# 5. Generate random golden input/output vectors for the test app
#    (test-app-exclusive; not triggered by the demo or dev pipelines)
# ---------------------------------------------------------------------------
test_data: $(TEST_DATA_SENTINEL)

$(TEST_DATA_SENTINEL): $(MODEL_OUT) $(SETUP_SENTINEL) scripts/ml_workload/gen_test_data.py
	@echo "[test_data] Generating random golden test vectors..."
	$(VENV_PYTHON) scripts/ml_workload/gen_test_data.py \
	    --model $(MODEL_OUT) --out $(TEST_DIR)
	@echo "test_data done" > $(TEST_DATA_SENTINEL)
	@echo "[ok] Golden vectors written to $(TEST_DIR)/test_data/."

## Native build + bit-exact check against the LiteRT reference output
test_run: $(WORKLOAD_SENTINEL) $(TEST_DATA_SENTINEL)
	$(MAKE) -C $(SW_DIR) APP=test run

# ---------------------------------------------------------------------------
# 6. Compile the golden-vector test to a bare-metal RV64IMC ELF
#    Uses the RISC-V toolchain bundled inside the tflite-micro checkout.
#    Output: src/software/build/rv64/resnet18_test.elf
#    Note: dev_compile is the equivalent for the software_dev workload.
# ---------------------------------------------------------------------------
## Compile the TFLM test program to a bare-metal RV64IMC ELF (for Spike / CV-Wally)
test_compile: $(WORKLOAD_SENTINEL) $(TEST_DATA_SENTINEL)
	$(MAKE) -C $(SW_DIR) APP=test all CONFIG=rv64

## Compile the demo program to a bare-metal RV64IMC ELF
demo_compile: $(WORKLOAD_SENTINEL) $(DEMO_SENTINEL)
	$(MAKE) -C $(SW_DIR) APP=demo all CONFIG=rv64

## Compile the software_dev program to a bare-metal RV64IMC ELF
dev_compile: $(DEV_SENTINEL)
	$(MAKE) -C $(DEV_DIR) all CONFIG=rv64

# ---------------------------------------------------------------------------
# 7. Spike RISC-V ISA simulator
# ---------------------------------------------------------------------------
## Install Spike + proxy kernel (pk) on WSL/Linux (source build; apt has no spike on Ubuntu 24.04)
get_spike:
	bash scripts/tool_installation/install_spike.sh

## Dynamic instruction mix (Spike -g PC histogram + disassembly) -> results/spike_results/
test_spike: test_compile
	$(MAKE) -C $(SW_DIR) APP=test spike SPIKE=$(SPIKE) PK=$(PK)

## Baseline Workload Analysis report from the test_spike + model_inspect results
baseline_report: test_spike model_inspect
	$(PYTHON) scripts/result_extraction/gen_baseline_report.py

# ---------------------------------------------------------------------------
# 8. Demo: classify 20 real ImageNet photos (vs. the random golden vector
#    used by test_run / test_spike during development).
#    demo_data is demo-app-exclusive; it is NOT triggered by the test or dev
#    pipelines, keeping the random test data and real photo data independent.
# ---------------------------------------------------------------------------
## Download the photos, quantize them, and record LiteRT reference logits
demo_data: $(DEMO_SENTINEL)

$(DEMO_SENTINEL): $(MODEL_OUT) $(SETUP_SENTINEL) scripts/ml_workload/gen_demo_data.py
	@echo "[demo] Generating ImageNet photo demo data..."
	$(VENV_PYTHON) scripts/ml_workload/gen_demo_data.py \
	    --model $(MODEL_OUT) --out $(DEMO_DIR)
	@echo "demo done" > $(DEMO_SENTINEL)
	@echo "[ok] Demo data written to $(DEMO_DIR)/."

## Native build + classify the photos (fast; seconds per photo)
demo_run: $(WORKLOAD_SENTINEL) $(DEMO_SENTINEL)
	$(MAKE) -C $(SW_DIR) APP=demo run N=$(N)

## Dynamic instruction mix for the demo (Spike -g) -> results/spike_results/resnet18_demo.*
demo_spike: $(WORKLOAD_SENTINEL) $(DEMO_SENTINEL)
	$(MAKE) -C $(SW_DIR) APP=demo spike N=$(or $(N),1) SPIKE=$(SPIKE) PK=$(PK)

# ---------------------------------------------------------------------------
# 9. software_dev: kernel-development workload (dev_net, one instance of
#    each resnet18 op type, ~1.8 MMAC). Custom kernels live in
#    src/software_dev/kernels/; choose them with CUSTOM_KERNELS="CONV_2D ..."
#    (default CONV_2D, empty = all TFLM reference kernels).
#    dev_run is the software_dev equivalent of test_run; both run the
#    workload natively and check outputs against LiteRT reference.
# ---------------------------------------------------------------------------
## Build dev_net + TFLM runtime + per-op LiteRT golden outputs
dev_workload: $(DEV_SENTINEL)

$(DEV_SENTINEL): $(SETUP_SENTINEL) scripts/ml_workload/gen_dev_workload.py \
                 scripts/ml_workload/gen_tflm_workload.py
	@echo "[dev] Generating the software_dev workload..."
	$(VENV_PYTHON) scripts/ml_workload/gen_dev_workload.py --out $(DEV_DIR)
	@echo "dev workload done" > $(DEV_SENTINEL)
	@echo "[ok] software_dev workload written to $(DEV_DIR)/."

## Native build + per-op check against LiteRT
dev_run: $(DEV_SENTINEL)
	$(MAKE) -C $(DEV_DIR) run

## Dynamic instruction mix -> results/spike_results/dev_net_<variant>.*
dev_spike: $(DEV_SENTINEL)
	$(MAKE) -C $(DEV_DIR) spike SPIKE=$(SPIKE) PK=$(PK)

## Model structure report -> results/model_structure/dev_net_int8.md
dev_inspect: $(DEV_SENTINEL)
	$(VENV_PYTHON) scripts/result_extraction/inspect_tflite.py --model $(DEV_MODEL)

## Print-free bare-metal ELF for the CV-Wally testbench (software_dev CONFIG=cvw)
dev_cvw_compile: $(DEV_SENTINEL)
	$(MAKE) -C $(DEV_DIR) all CONFIG=cvw

## Run dev_net on the CV-Wally Verilator model: self-check + HPM counters
dev_cvw_run: $(DEV_SENTINEL)
	$(MAKE) -C $(DEV_DIR) cvw_run

# ---------------------------------------------------------------------------
# 10. Hardware: CV-Wally RTL + project RTL on Verilator (see src/hardware/README.md)
# ---------------------------------------------------------------------------
## Install Verilator 5.036 (source build), Verible, uv and GTKWave (WSL/Linux)
get_hw_tools:
	bash scripts/tool_installation/install_hw_tools.sh

## Clone CV-Wally at the pinned commit into src/hardware/cvw
hw_setup:
	bash scripts/tool_installation/setup_cvw.sh

## Verilator lint (full core + project RTL) + Verible style lint
hw_lint:
	$(MAKE) -C $(HW_DIR) lint

## Build the CV-Wally Verilator model (garibaldi_rv64imc config)
hw_sim:
	$(MAKE) -C $(HW_DIR) sim

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
clean:
	git clean -fdx .

help:
	@echo ""
	@echo "  === Setup & Installation ==="
	@echo "  make setup      — create .venv and install requirements"
	@echo "  make fetch_model   — download resnet18_int8_channelwise.tflite from HuggingFace"
	@echo "  make model_inspect — print ResNet-18 op inventory + tensor summary (optional)"
	@echo "  make get_spike  — build Spike + pk from source (WSL/Linux)"
	@echo "  make get_hw_tools  — hardware: Verilator 5.036, Verible, uv, GTKWave (WSL)"
	@echo "  make hw_setup      — hardware: clone CV-Wally (pinned) into src/hardware/cvw"
	@echo ""
	@echo "  === Shared ResNet-18 Workload (test + demo) ==="
	@echo "  make model_workload — export shared TFLM runtime + model CC array + op resolver"
	@echo ""
	@echo "  === Test Pipeline ==="
	@echo "  make test_data     — generate random golden test vectors (test-app-exclusive)"
	@echo "  make test_run      — native build + bit-exact check against LiteRT"
	@echo "  make test_compile  — cross-compile test app to bare-metal RV64IMC ELF"
	@echo "  make test_spike    — Spike run + dynamic instruction mix -> results/spike_results/"
	@echo "  make baseline_report — Baseline Workload Analysis -> results/baseline_workload_analysis.md"
	@echo ""
	@echo "  === Demo Pipeline ==="
	@echo "  make demo_data      — fetch 20 ImageNet photos + LiteRT reference logits"
	@echo "  make demo_run       — native build + classify the photos [N=count]"
	@echo "  make demo_compile   — cross-compile demo app to bare-metal RV64IMC ELF"
	@echo "  make demo_spike     — Spike run + dynamic instruction mix -> results/spike_results/ [N=count]"
	@echo ""
	@echo "  === software_dev Pipeline ==="
	@echo "  make dev_workload  — generate dev_net + golden outputs"
	@echo "  make dev_run       — native build + per-op check against LiteRT"
	@echo "  make dev_compile   — cross-compile dev workload to bare-metal RV64IMC ELF"
	@echo "  make dev_spike     — Spike run + dynamic instruction mix -> results/spike_results/"
	@echo "  make dev_inspect   — model structure report"
	@echo "  make dev_cvw_compile — print-free bare-metal ELF for the CV-Wally testbench"
	@echo "  make dev_cvw_run   — run on the CV-Wally Verilator model (self-check + HPM counters)"
	@echo "      [CUSTOM_KERNELS=\"CONV_2D ...\" (default CONV_2D; empty = reference)]"
	@echo ""
	@echo "  === Hardware ==="
	@echo "  make hw_lint       — Verilator + Verible lint"
	@echo "  make hw_sim        — build the CV-Wally Verilator model"
	@echo ""
	@echo "  make clean      — git clean -fdx: deletes ALL untracked files"
	@echo ""
	@echo "  Model: litert-community/resnet18 / resnet18_int8_channelwise.tflite"
	@echo "         W8A8: weights INT8 per-channel symmetric,"
	@echo "               activations INT8 asymmetric."
	@echo ""
