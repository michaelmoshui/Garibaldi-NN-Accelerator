#!/usr/bin/env bash
# Build Spike (riscv-isa-sim) and the proxy kernel (pk) from source.
#
# The RISC-V cross-compiler is downloaded as a prebuilt binary from
# riscv-collab/riscv-gnu-toolchain GitHub releases — no apt RISC-V packages.
# Only the minimal host build tools below are installed via apt.
#
# Usage:
#   bash scripts/tool_installation/install_spike.sh               # first-time install
#   FORCE=1 bash scripts/tool_installation/install_spike.sh       # rebuild everything
#   TC_VERSION=2024.12.10 bash ...              # pin toolchain release date
set -euo pipefail

SPIKE_PREFIX="${SPIKE_PREFIX:-$HOME/.local/riscv}"
SRC_DIR="${SPIKE_SRC_DIR:-$HOME/.cache/garibaldi-nn}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
TC_DIR="${SRC_DIR}/riscv-gnu-toolchain"
TC_REPO="riscv-collab/riscv-gnu-toolchain"
# Pinned release — newer releases use the shorter .tar.xz naming scheme.
TC_VERSION_DEFAULT="2026.08.27"
SPIKE_REPO="https://github.com/riscv-software-src/riscv-isa-sim.git"
PK_REPO="https://github.com/riscv-software-src/riscv-pk.git"

log() { echo "[get_spike] $*"; }
die() { echo "[get_spike] ERROR: $*" >&2; exit 1; }

[[ "$(uname -s)" == "Linux" ]] || die "Run this script from WSL / Linux."

spike_bin="${SPIKE_PREFIX}/bin/spike"
pk_bin="${SPIKE_PREFIX}/riscv64-unknown-elf/bin/pk"

if [[ -x "${spike_bin}" && -x "${pk_bin}" && "${FORCE:-0}" != "1" ]]; then
  log "Already installed under ${SPIKE_PREFIX}."
  log "  export PATH=\"${SPIKE_PREFIX}/bin:\$PATH\""
  log "Re-run with FORCE=1 to rebuild."
  exit 0
fi

# ---------------------------------------------------------------------------
# 1. Minimal host build tools (apt) — spike is a host C++ binary
# ---------------------------------------------------------------------------
need_apt=()
command -v g++       >/dev/null 2>&1 || need_apt+=(build-essential)
command -v git       >/dev/null 2>&1 || need_apt+=(git)
command -v curl      >/dev/null 2>&1 || need_apt+=(curl)
command -v dtc       >/dev/null 2>&1 || need_apt+=(device-tree-compiler)
command -v autoconf  >/dev/null 2>&1 || need_apt+=(autoconf)
command -v pkg-config >/dev/null 2>&1 || need_apt+=(pkg-config)
[[ -f /usr/include/boost/regex.hpp ]] || need_apt+=(libboost-regex-dev)
[[ -f /usr/include/boost/system/error_code.hpp ]] || need_apt+=(libboost-system-dev)

if ((${#need_apt[@]})); then
  log "Installing host build tools: ${need_apt[*]}"
  sudo apt-get update -qq
  sudo apt-get install -y "${need_apt[@]}"
fi

# ---------------------------------------------------------------------------
# 2. Prebuilt RISC-V bare-metal toolchain from GitHub releases
# ---------------------------------------------------------------------------
UBUNTU_VER="$(lsb_release -rs 2>/dev/null || echo 24.04)"
# Map Ubuntu version to the asset name used by riscv-collab releases.
case "${UBUNTU_VER}" in
  22*) ASSET_OS="ubuntu-22.04" ;;
  24*) ASSET_OS="ubuntu-24.04" ;;
  *)   ASSET_OS="ubuntu-24.04" ;;
esac

TC_TAG="${TC_VERSION:-${TC_VERSION_DEFAULT}}"
# Newer releases (2025+) dropped the -nightly-DATE-nightly suffix and use .tar.xz.
TC_ASSET="riscv64-elf-${ASSET_OS}-gcc.tar.xz"
TC_URL="https://github.com/${TC_REPO}/releases/download/${TC_TAG}/${TC_ASSET}"
log "Toolchain release: ${TC_TAG} — ${TC_ASSET} (override with TC_VERSION=YYYY.MM.DD)"

TC_TARBALL="${TC_DIR}/toolchain.tar.xz"
if [[ -x "${TC_DIR}/bin/riscv64-unknown-elf-gcc" && "${FORCE:-0}" != "1" ]]; then
  log "Toolchain already extracted at ${TC_DIR}."
else
  log "Downloading RISC-V toolchain (~170 MB)..."
  mkdir -p "${TC_DIR}"
  curl -fL --progress-bar "${TC_URL}" -o "${TC_TARBALL}"
  log "Extracting toolchain..."
  tar -xf "${TC_TARBALL}" -C "${TC_DIR}" --strip-components=1
  rm -f "${TC_TARBALL}"
fi

export PATH="${TC_DIR}/bin:${PATH}"
RISCV_GCC="${TC_DIR}/bin/riscv64-unknown-elf-gcc"
[[ -x "${RISCV_GCC}" ]] || die "riscv64-unknown-elf-gcc not found in ${TC_DIR}/bin"
log "Toolchain: ${RISCV_GCC} ($("${RISCV_GCC}" --version | head -n1))"

mkdir -p "${SPIKE_PREFIX}"

# ---------------------------------------------------------------------------
# 3. Build Spike (host binary)
# ---------------------------------------------------------------------------
if [[ -x "${spike_bin}" && "${FORCE:-0}" != "1" ]]; then
  log "Spike already at ${spike_bin}; skipping."
else
  if [[ ! -d "${SRC_DIR}/riscv-isa-sim/.git" ]]; then
    log "Cloning riscv-isa-sim..."
    git clone --depth 1 "${SPIKE_REPO}" "${SRC_DIR}/riscv-isa-sim"
  fi
  log "Building Spike..."
  mkdir -p "${SRC_DIR}/riscv-isa-sim/build"
  (
    cd "${SRC_DIR}/riscv-isa-sim/build"
    ../configure --prefix="${SPIKE_PREFIX}"
    make -j"${JOBS}"
    make install
  )
fi

# ---------------------------------------------------------------------------
# 4. Build riscv-pk (RISC-V ELF proxy kernel)
# ---------------------------------------------------------------------------
if [[ ! -d "${SRC_DIR}/riscv-pk/.git" ]]; then
  log "Cloning riscv-pk..."
  git clone --depth 1 "${PK_REPO}" "${SRC_DIR}/riscv-pk"
fi
log "Building pk..."
rm -rf "${SRC_DIR}/riscv-pk/build"
mkdir -p "${SRC_DIR}/riscv-pk/build"
(
  cd "${SRC_DIR}/riscv-pk/build"
  CC="${RISCV_GCC}" \
    ../configure --prefix="${SPIKE_PREFIX}" --host=riscv64-unknown-elf \
      --with-arch=rv64imac_zicsr_zifencei --with-abi=lp64
  make -j"${JOBS}"
  make install
)

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
[[ -x "${spike_bin}" ]] || die "spike not found at ${spike_bin} after install"
[[ -x "${pk_bin}" ]]    || die "pk not found at ${pk_bin} after install"

log "Done!"
log "  spike: ${spike_bin}"
log "  pk:    ${pk_bin}"
log ""
log "Add to PATH, then run:"
log "  export PATH=\"${SPIKE_PREFIX}/bin:\$PATH\""
log "  make test_spike"
