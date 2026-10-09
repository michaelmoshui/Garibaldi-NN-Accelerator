#!/usr/bin/env bash
# Install the open-source hardware tools used to build, lint and simulate
# CV-Wally (src/hardware/).
#
#   Verilator  — simulator + lint. Built from source: CV-Wally needs >= 5.021,
#                Ubuntu 24.04 apt ships 5.020. Pinned to the version upstream
#                CV-Wally installs (bin/installation/verilator-install.sh).
#   Verible    — SystemVerilog style linter / formatter (prebuilt static binary).
#   uv         — runs CV-Wally's own helper scripts (wsim, lint-wally, ...).
#   GTKWave    — waveform viewer (apt; opens via WSLg on Windows 11).
#
# Everything except the apt packages goes into the same prefix as Spike.
#
# Usage:
#   bash scripts/install_hw_tools.sh                 # first-time install
#   FORCE=1 bash scripts/install_hw_tools.sh         # rebuild everything
#   VERILATOR_VERSION=v5.040 bash ...                # override the pin
set -euo pipefail

HW_PREFIX="${HW_PREFIX:-$HOME/.local/riscv}"
SRC_DIR="${HW_SRC_DIR:-$HOME/.cache/garibaldi-nn}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
VERILATOR_VERSION="${VERILATOR_VERSION:-v5.036}"
VERIBLE_VERSION="${VERIBLE_VERSION:-v0.0-4296-g0f262651}"
VERILATOR_REPO="https://github.com/verilator/verilator.git"
VERIBLE_REPO="chipsalliance/verible"

log() { echo "[get_hw_tools] $*"; }
die() { echo "[get_hw_tools] ERROR: $*" >&2; exit 1; }

[[ "$(uname -s)" == "Linux" ]] || die "Run this script from WSL / Linux."
mkdir -p "${HW_PREFIX}/bin" "${SRC_DIR}"

# ---------------------------------------------------------------------------
# 1. apt packages: Verilator build deps + GTKWave
# ---------------------------------------------------------------------------
apt_pkgs=(git help2man perl python3 make autoconf g++ flex bison ccache
          libfl2 libfl-dev zlib1g zlib1g-dev numactl mold
          libgoogle-perftools-dev gtkwave curl)
missing=()
for p in "${apt_pkgs[@]}"; do
  dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
done
if ((${#missing[@]})); then
  log "Installing apt packages: ${missing[*]}"
  sudo apt-get update -qq
  sudo apt-get install -y "${missing[@]}"
else
  log "apt packages already installed."
fi

# ---------------------------------------------------------------------------
# 2. Verilator (source build)
# ---------------------------------------------------------------------------
verilator_bin="${HW_PREFIX}/bin/verilator"
want="${VERILATOR_VERSION#v}"
have="$("${verilator_bin}" --version 2>/dev/null | awk '{print $2}' || true)"
if [[ "${have}" == "${want}" && "${FORCE:-0}" != "1" ]]; then
  log "Verilator ${have} already at ${verilator_bin}."
else
  log "Building Verilator ${VERILATOR_VERSION} (takes ~5-10 min)..."
  vdir="${SRC_DIR}/verilator"
  if [[ ! -d "${vdir}/.git" ]]; then
    git clone "${VERILATOR_REPO}" "${vdir}"
  fi
  (
    cd "${vdir}"
    unset VERILATOR_ROOT
    git fetch --tags -q
    git checkout -q "${VERILATOR_VERSION}"
    git clean -fdxq
    autoconf
    ./configure --prefix="${HW_PREFIX}"
    make -j"${JOBS}"
    make install
  )
fi

# ---------------------------------------------------------------------------
# 3. Verible (prebuilt static release)
# ---------------------------------------------------------------------------
if [[ -x "${HW_PREFIX}/bin/verible-verilog-lint" && "${FORCE:-0}" != "1" ]]; then
  log "Verible already at ${HW_PREFIX}/bin."
else
  asset="verible-${VERIBLE_VERSION}-linux-static-x86_64.tar.gz"
  url="https://github.com/${VERIBLE_REPO}/releases/download/${VERIBLE_VERSION}/${asset}"
  log "Downloading Verible ${VERIBLE_VERSION}..."
  tmp="$(mktemp -d)"
  curl -fL --progress-bar "${url}" -o "${tmp}/${asset}"
  tar -xzf "${tmp}/${asset}" -C "${tmp}"
  cp "${tmp}"/verible-*/bin/verible-* "${HW_PREFIX}/bin/"
  rm -rf "${tmp}"
fi

# ---------------------------------------------------------------------------
# 4. uv (needed only for CV-Wally's own scripts: wsim, lint-wally, ...)
# ---------------------------------------------------------------------------
if command -v uv >/dev/null 2>&1 || [[ -x "${HW_PREFIX}/bin/uv" ]]; then
  log "uv already installed."
else
  log "Installing uv into ${HW_PREFIX}/bin..."
  curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="${HW_PREFIX}/bin" sh
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
"${verilator_bin}" --version
"${HW_PREFIX}/bin/verible-verilog-lint" --version | head -n1
log "Done. Put the tools first on PATH (apt's older Verilator is in /usr/bin):"
log "  export PATH=\"${HW_PREFIX}/bin:\$PATH\""
log "Next: make hw_setup && make hw_lint && make hw_sim"
