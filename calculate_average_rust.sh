#!/bin/bash
#
# Runs the Rust 1BRC implementation.
# Build first with:  cd src/rust && cargo build --release
#
set -e

INPUT="${1:-measurements.txt}"

BIN="./src/rust/target/release/calculate_average"

if [ ! -f "$BIN" ]; then
  echo "Binary not found. Building…"
  pushd src/rust > /dev/null
  cargo build --release
  popd > /dev/null
fi

exec "$BIN" "$INPUT"
