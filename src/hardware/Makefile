# =============================================================================
# Garibaldi hardware flow: CV-Wally + project RTL on Verilator (WSL / Linux)
#
#   make setup              clone CV-Wally at the pinned commit (scripts/tool_installation/setup_cvw.sh)
#   make lint               Verilator lint (full core + project RTL) + Verible style lint
#   make sim                build the CV-Wally Verilator model for CONFIG
#   make run ELF=foo.elf    run a bare-metal ELF on the RTL model
# =============================================================================
SHELL := /bin/bash

HW_DIR    := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT      := $(abspath $(HW_DIR)/../..)
CVW       ?= $(HW_DIR)/cvw
CONFIG    ?= garibaldi_rv64imc
BUILD     ?= $(HW_DIR)/build
JOBS      ?= $(shell nproc 2>/dev/null || echo 4)

# ---------------------------------------------------------------------------
# Tools (installed by scripts/tool_installation/install_hw_tools.sh / install_spike.sh)
# ---------------------------------------------------------------------------
HW_PREFIX    ?= $(HOME)/.local/riscv
VERILATOR    ?= $(if $(wildcard $(HW_PREFIX)/bin/verilator),$(HW_PREFIX)/bin/verilator,verilator)
VERIBLE_LINT ?= $(if $(wildcard $(HW_PREFIX)/bin/verible-verilog-lint),$(HW_PREFIX)/bin/verible-verilog-lint,verible-verilog-lint)
RISCV_TC_BIN ?= $(HOME)/.cache/garibaldi-nn/riscv-gnu-toolchain/bin
CROSS        := $(if $(wildcard $(RISCV_TC_BIN)/riscv64-unknown-elf-gcc),$(RISCV_TC_BIN)/,)riscv64-unknown-elf-
export PATH  := $(HW_PREFIX)/bin:$(RISCV_TC_BIN):$(PATH)

# ---------------------------------------------------------------------------
# RTL file list: CV-Wally sources, minus any file shadowed by a same-named
# file in rtl/overrides/, plus project RTL in rtl/.
# ---------------------------------------------------------------------------
CONFIG_DIR  := $(firstword $(wildcard $(HW_DIR)/config/$(CONFIG) $(CVW)/config/$(CONFIG) $(CVW)/config/deriv/$(CONFIG)))
OVERRIDES   := $(wildcard $(HW_DIR)/rtl/overrides/*.sv)
PROJECT_RTL := $(wildcard $(HW_DIR)/rtl/*.sv)
CVW_PKG     := $(CVW)/src/cvw.sv
CVW_RTL     := $(filter-out $(CVW_PKG) $(addprefix %/,$(notdir $(OVERRIDES))), \
                 $(wildcard $(CVW)/src/*/*.sv $(CVW)/src/*/*/*.sv))
CORE_RTL    := $(CVW_RTL) $(OVERRIDES) $(PROJECT_RTL)
INCLUDES    := -I$(CONFIG_DIR) -I$(CVW)/config/shared

# Upstream lint-wally waivers, applied to the full core
LINT_WAIVERS := -Wno-UNUSEDSIGNAL -Wno-VARHIDDEN -Wno-GENUNNAMED -Wno-PINCONNECTEMPTY

# ---------------------------------------------------------------------------
# Full-core Verilator model (mirrors cvw/sim/verilator/Makefile)
# SIM_PARAMS are testbench parameter overrides, e.g. "PrintHPMCounters=1 MAKE_VCD=1"
# ---------------------------------------------------------------------------
SIM_PARAMS ?= PrintHPMCounters=1
# testbench/common/loggers.sv trips width warnings when PrintHPMCounters is set;
# RTL width issues are still caught by `make lint_core`.
SIM_WAIVERS ?= -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC
SIM_DIR    := $(BUILD)/sim/$(CONFIG)
SIM_BIN    := $(SIM_DIR)/Vtestbench
SIM_STAMP  := $(SIM_DIR)/.params
TB_SRCS    := $(CVW)/testbench/testbench.sv $(wildcard $(CVW)/testbench/common/*.sv)

# Rebuild the model whenever SIM_PARAMS changes
$(shell mkdir -p $(SIM_DIR) && (echo '$(SIM_PARAMS)' | cmp -s - $(SIM_STAMP) || echo '$(SIM_PARAMS)' > $(SIM_STAMP)))

.SECONDARY:
.PHONY: help setup check_cvw lint lint_core lint_rtl lint_style sim run clean clean_sim

help:
	@sed -n '2,9p' $(firstword $(MAKEFILE_LIST))

setup:
	bash $(ROOT)/scripts/tool_installation/setup_cvw.sh

check_cvw:
	@test -f $(CVW_PKG) || { echo "CV-Wally not found at $(CVW); run 'make setup'"; exit 1; }
	@test -n "$(CONFIG_DIR)" || { echo "Unknown CONFIG=$(CONFIG)"; exit 1; }

# ---------------------------------------------------------------------------
# Lint
# ---------------------------------------------------------------------------
lint: lint_core lint_rtl lint_style

## Same checks as upstream bin/lint-wally, on CONFIG + overrides
lint_core: check_cvw
	@echo "[lint] CV-Wally core ($(CONFIG))"
	@$(VERILATOR) --lint-only --quiet -Wall $(LINT_WAIVERS) --top-module wallywrapper \
	  $(INCLUDES) $(CVW_PKG) $(CVW)/testbench/wallywrapper.sv $(CORE_RTL)

## Each project module standalone (not yet instantiated in the core)
lint_rtl:
	@for f in $(PROJECT_RTL); do \
	  echo "[lint] $$(basename $$f)"; \
	  $(VERILATOR) --lint-only --quiet -Wall --top-module $$(basename $$f .sv) $$f || exit 1; \
	done

## Style lint of project-owned SystemVerilog only (CV-Wally has its own style)
lint_style:
	@echo "[lint] verible: rtl/"
	@$(VERIBLE_LINT) --rules_config=$(HW_DIR)/lint/verible.rules $(PROJECT_RTL) $(OVERRIDES)

# ---------------------------------------------------------------------------
# Full-core simulation
# ---------------------------------------------------------------------------
sim: check_cvw $(SIM_BIN)

$(SIM_BIN): $(CVW_PKG) $(CORE_RTL) $(TB_SRCS) $(CONFIG_DIR)/config.vh $(SIM_STAMP)
	@echo "[sim] Verilating CV-Wally ($(CONFIG)) -> $(SIM_DIR)  (~3 min)"
	@$(VERILATOR) --Mdir $(SIM_DIR) -o Vtestbench --binary --trace -j $(JOBS) \
	  --assert --no-trace-top --top-module testbench --relative-includes $(SIM_WAIVERS) \
	  $(INCLUDES) $(CVW)/sim/verilator/wrapper.c $(patsubst %,-G%,$(SIM_PARAMS)) \
	  $(CVW_PKG) $(TB_SRCS) $(CORE_RTL)

# Sidecar files the CV-Wally testbench loads next to an ELF
%.elf.memfile: %.elf
	python3 $(CVW)/bin/elf2hex $< $@
%.elf.objdump: %.elf
	$(CROSS)objdump -S -D $< > $@
%.elf.objdump.addr: %.elf.objdump
	bash $(CVW)/bin/extractFunctionRadix.sh $<

## make run ELF=path/to/prog.elf [SIM_ARGS="+INSTR_LIMIT=1000000"]
run: sim
	@test -n "$(ELF)" || { echo "usage: make run ELF=path/to/prog.elf"; exit 1; }
	$(MAKE) --no-print-directory $(abspath $(ELF)).memfile $(abspath $(ELF)).objdump.addr
	@mkdir -p $(SIM_DIR)/logs
	cd $(SIM_DIR) && WALLY=$(CVW) RISCV=$(HW_PREFIX) ./Vtestbench \
	  +TEST=$(basename $(notdir $(ELF))) +ElfFile=$(abspath $(ELF)) \
	  +UART_LOG=1 +UART_LOG_FILE=$(abspath $(ELF)).uart.out $(SIM_ARGS)
	@echo "---- UART ($(abspath $(ELF)).uart.out) ----"; cat $(abspath $(ELF)).uart.out

clean_sim:
	rm -rf $(BUILD)/sim

clean:
	rm -rf $(BUILD)
