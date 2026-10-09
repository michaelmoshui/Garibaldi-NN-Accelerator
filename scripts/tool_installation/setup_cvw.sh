#!/usr/bin/env bash
# Check out CV-Wally (openhwfoundation/cvw) at a pinned commit into
# src/hardware/cvw. The checkout is treated as read-only upstream source:
# project RTL changes live in src/hardware/rtl/ (new modules) and
# src/hardware/rtl/overrides/ (modified copies of CV-Wally files).
#
# Submodules (riscv-arch-test, coremark, embench, ...) are skipped; they are
# only needed for upstream's own regression-wally flow:
#   WITH_SUBMODULES=1 bash scripts/tool_installation/setup_cvw.sh
#
# Usage:
#   bash scripts/tool_installation/setup_cvw.sh
#   CVW_COMMIT=<sha> bash scripts/tool_installation/setup_cvw.sh   # move the pin
#   CVW_REPO=https://github.com/<you>/cvw bash ...                 # use a fork
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CVW_DIR="${CVW_DIR:-${ROOT}/src/hardware/cvw}"
CVW_REPO="${CVW_REPO:-https://github.com/openhwfoundation/cvw.git}"
CVW_COMMIT="${CVW_COMMIT:-2064ca2bb8a88e3e3ec43753be93d00bef49345b}"  # main, 2026-10-06

log() { echo "[setup_cvw] $*"; }

if [[ ! -d "${CVW_DIR}/.git" ]]; then
  log "Cloning ${CVW_REPO} -> ${CVW_DIR}"
  git clone --filter=blob:none --no-checkout "${CVW_REPO}" "${CVW_DIR}"
fi

cd "${CVW_DIR}"
if [[ "$(git rev-parse HEAD 2>/dev/null || true)" != "${CVW_COMMIT}" ]]; then
  git cat-file -e "${CVW_COMMIT}^{commit}" 2>/dev/null || git fetch -q origin
  log "Checking out ${CVW_COMMIT}"
  git -c advice.detachedHead=false checkout -q "${CVW_COMMIT}"
fi

if [[ "${WITH_SUBMODULES:-0}" == "1" ]]; then
  log "Fetching submodules..."
  git submodule update --init --recursive --filter=blob:none
fi

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  log "WARNING: ${CVW_DIR} has local modifications. Move them to"
  log "         src/hardware/rtl/overrides/ so they are tracked by this repo."
fi

log "CV-Wally ready at ${CVW_DIR} ($(git log -1 --format='%h %cd' --date=short))"
