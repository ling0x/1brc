#!/bin/bash
#
# Runs the Rust 1BRC implementation.
# Always rebuilds (cargo is fast when nothing changed).
#
set -e

INPUT="${1:-measurements.txt}"

if [ ! -f "$INPUT" ]; then
  echo "ERROR: input file not found: $INPUT" >&2
  exit 1
fi

pushd src/rust > /dev/null
cargo build --release --quiet
popd > /dev/null

exec ./src/rust/target/release/calculate_average "$INPUT"
