//! One-Billion-Row Challenge — Rust implementation
//!
//! Approach:
//!   1. Memory-map the input file.
//!   2. Split into N byte-aligned chunks (one per CPU thread).
//!   3. Each Rayon thread builds an owned HashMap<String, Stats>.
//!   4. Merge partial maps, sort by name, print.
//!
//! Temperatures are kept as i32 tenths-of-a-degree throughout to
//! avoid floating-point parsing.  E.g. "-12.3" → -123.

use memmap2::Mmap;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::{
    env, fs,
    io::{self, Write},
};

// ── Stats ────────────────────────────────────────────────────────

#[derive(Clone)]
struct Stats {
    min:   i32,
    max:   i32,
    sum:   i64,
    count: u64,
}

impl Stats {
    #[inline]
    fn new(val: i32) -> Self {
        Stats { min: val, max: val, sum: val as i64, count: 1 }
    }

    #[inline]
    fn update(&mut self, val: i32) {
        if val < self.min { self.min = val; }
        if val > self.max { self.max = val; }
        self.sum   += val as i64;
        self.count += 1;
    }

    #[inline]
    fn merge(&mut self, other: &Stats) {
        if other.min < self.min { self.min = other.min; }
        if other.max > self.max { self.max = other.max; }
        self.sum   += other.sum;
        self.count += other.count;
    }
}

// ── Temperature parser ───────────────────────────────────────────
//
// Handles: "-99.9" "-9.9" "9.9" "99.9"
// Input slice must contain exactly the characters between ';' and '\n'.

#[inline]
fn parse_temp(b: &[u8]) -> i32 {
    let mut i = 0usize;
    let neg = if b[i] == b'-' { i += 1; true } else { false };

    // Read up to two digits before the decimal point
    let mut v = (b[i] - b'0') as i32;
    i += 1;
    if b[i] != b'.' {
        v = v * 10 + (b[i] - b'0') as i32;
        i += 1;
    }
    // skip '.'
    i += 1;
    // one fractional digit
    v = v * 10 + (b[i] - b'0') as i32;

    if neg { -v } else { v }
}

// ── Chunk splitter ───────────────────────────────────────────────
//
// Divides `data` into `n` roughly equal slices, each ending on a
// newline boundary so no line is split across two chunks.

fn split_at_newlines(data: &[u8], n: usize) -> Vec<(usize, usize)> {
    if data.is_empty() { return vec![]; }
    let chunk_size = (data.len() / n).max(1);
    let mut ranges = Vec::with_capacity(n + 1);
    let mut start = 0usize;

    for _ in 0..n {
        if start >= data.len() { break; }
        let mut end = (start + chunk_size).min(data.len() - 1);
        // Advance to the byte just after the next '\n'
        while end < data.len() - 1 && data[end] != b'\n' {
            end += 1;
        }
        end += 1; // move past the '\n'
        ranges.push((start, end));
        start = end;
    }
    // Tail that didn't fit in an exact chunk
    if start < data.len() {
        ranges.push((start, data.len()));
    }
    ranges
}

// ── Per-chunk processing ─────────────────────────────────────────

fn process_chunk(chunk: &[u8]) -> FxHashMap<String, Stats> {
    let mut map: FxHashMap<String, Stats> =
        FxHashMap::with_capacity_and_hasher(1 << 11, Default::default());

    let mut pos = 0usize;
    let len = chunk.len();

    while pos < len {
        // ── find ';' ──
        let mut semi = pos;
        while semi < len && chunk[semi] != b';' {
            semi += 1;
        }
        if semi >= len { break; } // malformed / trailing newline

        // ── find end of line ──
        let temp_start = semi + 1;
        let mut eol = temp_start;
        while eol < len && chunk[eol] != b'\n' {
            eol += 1;
        }

        // Trim possible '\r' (CRLF files)
        let temp_end = if eol > temp_start && chunk[eol - 1] == b'\r' {
            eol - 1
        } else {
            eol
        };

        let name_bytes = &chunk[pos..semi];
        let temp_bytes = &chunk[temp_start..temp_end];

        if temp_bytes.is_empty() {
            pos = eol + 1;
            continue;
        }

        let val = parse_temp(temp_bytes);

        // Use raw_entry to avoid double-hashing on the hot path
        match map.get_mut(std::str::from_utf8(name_bytes).unwrap_or("")) {
            Some(s) => s.update(val),
            None => {
                let name = String::from_utf8_lossy(name_bytes).into_owned();
                map.insert(name, Stats::new(val));
            }
        }

        pos = eol + 1;
    }

    map
}

// ── Output helper ────────────────────────────────────────────────

#[inline]
fn fmt_temp(tenths: i32) -> String {
    let sign = if tenths < 0 { "-" } else { "" };
    let abs  = tenths.unsigned_abs();
    format!("{}{}.{}", sign, abs / 10, abs % 10)
}

// Round (sum_tenths / count) to nearest tenth, half-up.
#[inline]
fn mean_tenths(sum: i64, count: u64) -> i32 {
    // Multiply numerator by 10 to get hundredths, then round to tenths.
    // We want round(sum / count) where both are already in tenths.
    // Equivalent to: round(sum_tenths / count).
    let c = count as i64;
    if sum >= 0 {
        ((sum * 2 + c) / (2 * c)) as i32
    } else {
        -(((-sum) * 2 + c) / (2 * c)) as i32
    }
}

// ── main ─────────────────────────────────────────────────────────

fn main() -> io::Result<()> {
    let path = env::args().nth(1).unwrap_or_else(|| "measurements.txt".into());
    let file = fs::File::open(&path)
        .unwrap_or_else(|e| panic!("Cannot open '{}': {}", path, e));

    // SAFETY: we only read this mapping and never modify the backing file.
    let mmap  = unsafe { Mmap::map(&file)? };
    let data: &[u8] = &mmap;

    let nthreads = rayon::current_num_threads().max(1);
    let ranges   = split_at_newlines(data, nthreads);

    // Parallel processing — each thread owns its HashMap<String,Stats>
    let partial: Vec<FxHashMap<String, Stats>> = ranges
        .into_par_iter()
        .map(|(start, end)| process_chunk(&data[start..end]))
        .collect();

    // Merge
    let mut global: FxHashMap<String, Stats> =
        FxHashMap::with_capacity_and_hasher(1 << 14, Default::default());

    for map in partial {
        for (k, v) in map {
            global
                .entry(k)
                .and_modify(|s| s.merge(&v))
                .or_insert(v);
        }
    }

    // Sort by station name
    let mut entries: Vec<(String, Stats)> = global.into_iter().collect();
    entries.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));

    // Print
    let stdout = io::stdout();
    let mut out = io::BufWriter::with_capacity(1 << 20, stdout.lock());

    let last = entries.len().saturating_sub(1);
    out.write_all(b"{")? ;
    for (i, (name, s)) in entries.iter().enumerate() {
        let mean = mean_tenths(s.sum, s.count);
        let sep  = if i < last { ", " } else { "" };
        write!(
            out,
            "{}={}/{}/{}{}",
            name,
            fmt_temp(s.min),
            fmt_temp(mean),
            fmt_temp(s.max),
            sep
        )?;
    }
    out.write_all(b"}\n")?;
    Ok(())
}
