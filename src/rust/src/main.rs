//! One-Billion-Row Challenge — Rust implementation
//!
//! Approach:
//!   1. Memory-map the input file.
//!   2. Split into N byte-aligned chunks (one per CPU thread).
//!   3. Each Rayon thread builds an owned HashMap<String, Stats>.
//!   4. Merge partial maps, sort by name, print.
//!
//! Temperatures are kept as i32 tenths-of-a-degree throughout.
//! E.g. "-12.3" -> -123.

use memmap2::Mmap;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::{
    env, fs,
    io::{self, Write},
};

// ---------- Stats ---------------------------------------------------

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

// ---------- Temperature parser -------------------------------------
//
// Input: bytes between ';' and newline, e.g. "-12.3" or "4.5"
// Returns value in tenths-of-a-degree.

fn parse_temp(b: &[u8]) -> i32 {
    let mut i = 0;
    let neg = b[0] == b'-';
    if neg { i += 1; }

    let mut v: i32 = 0;
    while i < b.len() && b[i] != b'.' {
        v = v * 10 + (b[i] - b'0') as i32;
        i += 1;
    }
    i += 1; // skip '.'
    if i < b.len() {
        v = v * 10 + (b[i] - b'0') as i32;
    } else {
        v *= 10; // no fractional digit (shouldn't happen in spec)
    }

    if neg { -v } else { v }
}

// ---------- Chunk splitter -----------------------------------------
//
// Returns (start, end) byte ranges, each ending on a newline boundary.

fn split_at_newlines(data: &[u8], n: usize) -> Vec<(usize, usize)> {
    if data.is_empty() { return vec![]; }
    let chunk_size = (data.len() / n).max(1);
    let mut ranges = Vec::with_capacity(n + 1);
    let mut start  = 0usize;

    for _ in 0..n {
        if start >= data.len() { break; }
        let mut end = (start + chunk_size).min(data.len());
        // Walk forward to find the next newline
        while end < data.len() && data[end - 1] != b'\n' {
            end += 1;
        }
        if end > start {
            ranges.push((start, end));
        }
        start = end;
    }
    if start < data.len() {
        ranges.push((start, data.len()));
    }
    ranges
}

// ---------- Per-chunk processing -----------------------------------

fn process_chunk(chunk: &[u8]) -> FxHashMap<String, Stats> {
    let mut map: FxHashMap<String, Stats> =
        FxHashMap::with_capacity_and_hasher(1 << 11, Default::default());

    let mut pos = 0usize;
    let len = chunk.len();

    while pos < len {
        // Find ';'
        let mut semi = pos;
        while semi < len && chunk[semi] != b';' { semi += 1; }
        if semi >= len { break; }

        // Find newline
        let mut eol = semi + 1;
        while eol < len && chunk[eol] != b'\n' { eol += 1; }

        let name_bytes  = &chunk[pos..semi];
        let mut temp_end = eol;
        // Strip trailing \r
        if temp_end > semi + 1 && chunk[temp_end - 1] == b'\r' {
            temp_end -= 1;
        }
        let temp_bytes = &chunk[semi + 1..temp_end];

        if !name_bytes.is_empty() && !temp_bytes.is_empty() {
            let val  = parse_temp(temp_bytes);
            let name = unsafe { std::str::from_utf8_unchecked(name_bytes) };
            match map.get_mut(name) {
                Some(s) => s.update(val),
                None    => { map.insert(name.to_owned(), Stats::new(val)); }
            }
        }

        pos = eol + 1;
    }

    map
}

// ---------- Formatting ---------------------------------------------

#[inline]
fn fmt_temp(tenths: i32) -> String {
    let sign = if tenths < 0 { "-" } else { "" };
    let abs  = tenths.unsigned_abs();
    format!("{}{}.{}", sign, abs / 10, abs % 10)
}

/// Round (sum_tenths / count) to nearest tenth, half-up.
#[inline]
fn mean_tenths(sum: i64, count: u64) -> i32 {
    let c = count as i64;
    if sum >= 0 {
        ((sum * 2 + c) / (c * 2)) as i32
    } else {
        -((((-sum) * 2 + c) / (c * 2)) as i32)
    }
}

// ---------- main ---------------------------------------------------

fn main() -> io::Result<()> {
    let path = env::args().nth(1).unwrap_or_else(|| "measurements.txt".into());

    let file = fs::File::open(&path)
        .unwrap_or_else(|e| panic!("Cannot open '{}': {}", path, e));

    let meta = file.metadata()?;
    if meta.len() == 0 {
        eprintln!("WARNING: '{}' is empty.", path);
        println!("{{}}");
        return Ok(());
    }

    // SAFETY: read-only mapping; file is not modified during the run.
    let mmap = unsafe { Mmap::map(&file)? };
    let data: &[u8] = &mmap;

    let nthreads = rayon::current_num_threads().max(1);
    let ranges   = split_at_newlines(data, nthreads);

    eprintln!(
        "[1brc] file={} bytes={} threads={} chunks={}",
        path, data.len(), nthreads, ranges.len()
    );

    // Process in parallel — each thread owns its map
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

    eprintln!("[1brc] unique stations={}", global.len());

    // Sort
    let mut entries: Vec<(String, Stats)> = global.into_iter().collect();
    entries.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));

    // Output
    let stdout = io::stdout();
    let mut out = io::BufWriter::with_capacity(1 << 20, stdout.lock());

    let last = entries.len().saturating_sub(1);
    out.write_all(b"{")?;
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
