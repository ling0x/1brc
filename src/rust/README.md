# 1BRC — Rust Implementation

A high-performance, multi-threaded Rust solution to the
[One Billion Row Challenge](https://github.com/gunnarmorling/1brc).

## Design

| Technique | Details |
|---|---|
| Memory-mapped I/O | The entire `measurements.txt` is mapped with `memmap2` — no buffered read loop |
| Parallel chunking | File split into `nCPUs` chunks, boundaries aligned to `\n` |
| Integer arithmetic | Temperatures stored as `i32` tenths-of-degree; no floats during aggregation |
| Per-thread `HashMap` | One `HashMap<Vec<u8>, Stats>` per thread, merged at the end |
| Sort & output | Entries sorted lexicographically; output formatted to one decimal place |

## Building

```bash
cd src/rust
cargo build --release
```

Requires **Rust 1.75+** (stable).

## Running

```bash
# From repo root (generates measurements.txt first if needed):
./create_measurements.sh 1000000000
./calculate_average_rust.sh

# Or directly:
./src/rust/target/release/calculate_average measurements.txt
```

## Dependencies

| Crate | Purpose |
|---|---|
| `memmap2` | Safe, cross-platform memory-mapped files |
