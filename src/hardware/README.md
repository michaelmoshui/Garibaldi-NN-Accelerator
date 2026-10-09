# Hardware: CV-Wally + Garibaldi extensions

RTL flow for the project. It builds [CV-Wally](https://github.com/openhwfoundation/cvw) in a bare-metal
**RV64IMC** configuration, simulates it with Verilator, and lints it.

Everything runs in **WSL (Ubuntu 24.04)** or native Linux.

## Quick start

From the repo root:

```bash
make get_spike        # Spike + RISC-V GCC (already used by the software flow)
make get_hw_tools     # Verilator 5.036, Verible, uv, GTKWave   (~10 min, needs sudo once)
export PATH="$HOME/.local/riscv/bin:$PATH"   # put this in ~/.bashrc

make hw_setup         # clone CV-Wally at the pinned commit -> src/hardware/cvw
make hw_lint          # Verilator lint (whole core) + Verible style lint
make hw_sim           # build the CV-Wally Verilator model    (~3 min)
make -C src/hardware run ELF=path/to/prog.elf
```

## Layout

```text
src/hardware/
  Makefile                    setup / lint / sim / run
  config/garibaldi_rv64imc/   RV64IMC CV-Wally config (derived from upstream rv64gc)
  rtl/                        new project modules (garibaldi_xqmac8.sv, garibaldi_requant.sv)
  rtl/overrides/              modified copies of CV-Wally files (shadow the upstream ones)
  lint/verible.rules          style-lint rules for project-owned SystemVerilog
  cvw/                        upstream CV-Wally checkout (git-ignored, pinned)
  build/                      all generated output (git-ignored)
```

## How the flow works

**Upstream stays pristine.** `scripts/setup_cvw.sh` checks out CV-Wally at a pinned commit. Project
changes never go into `cvw/`. New modules go in `rtl/`. To change an upstream file, copy it into
`rtl/overrides/` and edit the copy. The Makefile drops the upstream file with the same name and
compiles yours. See [rtl/overrides/README.md](rtl/overrides/README.md).

**Config.** `CONFIG=garibaldi_rv64imc` is the default. It's upstream `rv64gc` with the FPU, A, B,
crypto, S/U modes, virtual memory, and PMP removed. Caches, the branch predictor, and the
performance counters stay on. Any upstream config also works, for example `make sim CONFIG=rv64gc`.

**Simulation.** `make sim` Verilates CV-Wally's own `testbench.sv` with `PrintHPMCounters=1`, so
every run prints cycles, instret, branch mispredicts, cache misses, and stalls. `make run ELF=x.elf`
generates the `.memfile`/`.objdump.addr` sidecars the testbench needs and runs the program. UART
output (ns16550 at `0x10000000`) goes to `x.elf.uart.out`. The run ends when the program does a
`sw` to its `tohost` symbol or jumps to itself. The ELF must be linked at `0x80000000`.

## Tools

| tool | why | installed by |
|---|---|---|
| Verilator 5.036 | simulation + lint. CV-Wally needs ≥ 5.021, and apt's 5.020 in `/usr/bin` is too old, so keep `~/.local/riscv/bin` first on `PATH` | `get_hw_tools` |
| Verible | style lint/format for project SV (`verible-verilog-format` is handy too) | `get_hw_tools` |
| uv | runs CV-Wally's own scripts (`cvw/bin/wsim`, `lint-wally`, `regression-wally`) | `get_hw_tools` |
| GTKWave | waveforms: `make sim SIM_PARAMS="PrintHPMCounters=1 MAKE_VCD=1"`, then `make run ...` writes `testbench.vcd` in `build/sim/<config>/` | `get_hw_tools` |
| Spike + RISC-V GCC | reference model and cross-compiler | `get_spike` |

Optional, not installed:

- **Synopsys DC / RTL Architect + SAED 32/28** (required for the area numbers): `cvw/synthDC/` has a
  working DC flow. Point it at the same file list (CV-Wally sources + `rtl/overrides` + `rtl/`).
- **Yosys** (`apt install yosys`): quick open-source gate-count sanity check before running DC.
- **Surfer** (`cargo install surfer` or a release binary): a faster VCD/FST viewer than GTKWave.

## Notes

- Building on `/mnt/c` works but is slower than the WSL-native filesystem. The core model takes
  about 3 minutes to build here and only rebuilds when RTL, config, or `SIM_PARAMS` change.
  `BUILD=~/garibaldi-build` moves the output off `/mnt/c`.
- `garibaldi_xqmac8.sv` and `garibaldi_requant.sv` are combinational starting points. Encoding,
  operand packing, accumulator width, and pipelining are still open design decisions. Xqmac8 reads
  `rd` as a third source, but CV-Wally's integer register file has only two read ports.
