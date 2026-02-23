#!/bin/bash
#
# Rust replacement for create_measurements.sh (no Java needed).
# Usage: ./create_measurements_rust.sh <count> [output_file]
#
#   ./create_measurements_rust.sh 1000000000          # -> measurements.txt
#   ./create_measurements_rust.sh 1000000 test.txt    # smaller test
#
set -e

COUNT="${1:-1000000000}"
OUTPUT="${2:-measurements.txt}"

pushd src/rust > /dev/null
cargo build --release --quiet
popd > /dev/null

exec ./src/rust/target/release/create_measurements "$COUNT" "$OUTPUT"
