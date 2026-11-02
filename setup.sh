#!/usr/bin/env bash
# Build codemason and install it somewhere it can find its bundled ripgrep.
#
# WHY THIS EXISTS: `cargo build` alone is not an install. The binary looks for
# vendor/ripgrep/<arch-os>/rg next to itself, then one and two directories up —
# so it works from target/release inside this checkout (two up is the repo root,
# where vendor/ lives), and stops working the moment you copy just the exe
# somewhere else. Grep and Glob then fall back to a native directory walk, which
# is correct but much slower, and nothing tells you it happened.
#
# So the exe, vendor/ and tokenizer/ travel together. This copies all three.
#
# The Python original has no equivalent problem: it resolves vendor/ relative to
# its package directory, so pip carries it. A single Rust binary has no package
# directory, which is exactly why this script is here (Windows twin: setup.bat).
set -euo pipefail
cd "$(dirname "$0")"
REPO="$PWD"

PREFIX="${1:-$HOME/.local/share/codemason}"
BINDIR="${2:-$HOME/.local/bin}"

if ! command -v cargo >/dev/null 2>&1; then
  echo "[setup] cargo not found. Install Rust: https://rustup.rs" >&2
  exit 1
fi

echo "[setup] building release binary"
cargo build --release

echo "[setup] installing to $PREFIX"
mkdir -p "$PREFIX"
install -m 755 "$REPO/target/release/codemason" "$PREFIX/codemason"

# vendor/ sits BESIDE the binary, which is the first place it looks.
if [ -d "$REPO/vendor/ripgrep" ]; then
  rm -rf "$PREFIX/vendor"
  mkdir -p "$PREFIX/vendor"
  cp -a "$REPO/vendor/ripgrep" "$PREFIX/vendor/"
  echo "[setup] bundled ripgrep installed alongside the binary"
else
  echo "[setup] WARNING: vendor/ripgrep is missing from this checkout."
  echo "        Grep and Glob will fall back to a slower native walk unless"
  echo "        ripgrep is on PATH."
fi

# tokenizer/ resolves the same way vendor/ does — beside the binary, then one
# and two directories up. It decides every token count in the process (the
# auto-compact threshold, the context-window check, the usage readout), and a
# missing one does NOT fail loudly: the process falls back to a character
# heuristic. That is why this is copied here and not left to the checkout.
if [ -d "$REPO/tokenizer" ]; then
  rm -rf "$PREFIX/tokenizer"
  cp -a "$REPO/tokenizer" "$PREFIX/tokenizer"
  echo "[setup] model tokenizers installed alongside the binary"
else
  echo "[setup] WARNING: tokenizer/ is missing from this checkout."
  echo "        Token counts will fall back to a character heuristic, so"
  echo "        auto-compact and context-window checks become approximate."
fi

mkdir -p "$BINDIR"
ln -sfn "$PREFIX/codemason" "$BINDIR/cm2"

echo
echo "[setup] done."
echo "        binary : $PREFIX/codemason"
echo "        vendor : $PREFIX/vendor/ripgrep"
echo "        tokens : $PREFIX/tokenizer"
echo "        link   : $BINDIR/cm2"
case ":$PATH:" in
  *":$BINDIR:"*) : ;;
  *) echo "        NOTE: $BINDIR is not on your PATH." ;;
esac
