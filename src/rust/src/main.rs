//! One-Billion-Row Challenge – Rust implementation
//!
//! Strategy:
//!   • Memory-map the input file (zero-copy reads).
//!   • Split the mapped slice into N chunks (one per Rayon thread),
//!     each aligned to a newline boundary.
//!   • Each thread accumulates stats in a thread-local FxHashMap.
//!   • Merge all per-thread maps, then sort and print.
//!
//! Temperatures are stored as i32 tenths-of-a-degree (e.g. 12.3 → 123)
//! to avoid any floating-point parsing overhead.

use memmap2::Mmap;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::{
    env, fs,
    io::{self, Write},
    os::unix::io::FromRawFd,
};

// -----------------------------------------------------------------
// Per-station accumulator (integer tenths of a degree)
// -----------------------------------------------------------------
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

// -----------------------------------------------------------------
// Fast integer parser for signed values like "-12.3" or "4.5"
// Returns tenths-of-a-degree as i32.
// Input slice ends just before '\n' (or end-of-chunk).
// -----------------------------------------------------------------
#[inline]
fn parse_temp(bytes: &[u8]) -> i32 {
    let (neg, rest) = if bytes[0] == b'-' {
        (true, &bytes[1..])
    } else {
        (false, bytes)
    };

    // rest is one of:  "X.X"  "XX.X"
    let val = match rest {
        [a, b'.', c] => {
            ((*a - b'0') as i32) * 10 + ((*c - b'0') as i32)
        }
        [a, b, b'.', c] => {
            ((*a - b'0') as i32) * 100
                + ((*b - b'0') as i32) * 10
                + ((*c - b'0') as i32)
        }
        _ => panic!("unexpected temp format: {:?}", std::str::from_utf8(rest)),
    };

    if neg { -val } else { val }
}

// -----------------------------------------------------------------
// Process one chunk (a slice of bytes that starts and ends on line
// boundaries).  Returns a local FxHashMap<&[u8], Stats>.
// We use raw byte slices as keys to avoid allocation; the backing
// memory is the mmap which lives for the whole program.
// -----------------------------------------------------------------
fn process_chunk(chunk: &[u8]) -> FxHashMap<&[u8], Stats> {
    let mut map: FxHashMap<&[u8], Stats> =
        FxHashMap::with_capacity_and_hasher(1 << 10, Default::default());

    let mut pos = 0usize;
    let len = chunk.len();

    while pos < len {
        // Find ';'
        let semi = memchr(b';', &chunk[pos..]).unwrap_or(len - pos) + pos;
        let name = &chunk[pos..semi];

        // Find '\n'
        let nl_offset = semi + 1;
        let nl = memchr(b'\n', &chunk[nl_offset..]).unwrap_or(len - nl_offset)
            + nl_offset;
        let temp_bytes = &chunk[nl_offset..nl];

        let val = parse_temp(temp_bytes);

        map.entry(name)
            .and_modify(|s| s.update(val))
            .or_insert_with(|| Stats::new(val));

        pos = nl + 1;
    }

    map
}

// -----------------------------------------------------------------
// Tiny memchr helper (avoids pulling in the memchr crate)
// -----------------------------------------------------------------
#[inline]
fn memchr(needle: u8, haystack: &[u8]) -> Option<usize> {
    haystack.iter().position(|&b| b == needle)
}

// -----------------------------------------------------------------
// Split mmap into ~equal chunks aligned to newline boundaries
// -----------------------------------------------------------------
fn split_chunks(data: &[u8], n: usize) -> Vec<&[u8]> {
    if data.is_empty() { return vec![]; }
    let chunk_size = data.len().saturating_div(n).max(1);
    let mut chunks = Vec::with_capacity(n);
    let mut start = 0usize;

    for _ in 0..n {
        if start >= data.len() { break; }
        let mut end = (start + chunk_size).min(data.len());
        // Advance to the next newline boundary
        while end < data.len() && data[end] != b'\n' {
            end += 1;
        }
        if end < data.len() { end += 1; } // include the '\n'
        chunks.push(&data[start..end]);
        start = end;
    }
    // Any leftover bytes (last chunk)
    if start < data.len() {
        chunks.push(&data[start..]);
    }
    chunks
}

// -----------------------------------------------------------------
// Formatting helper: convert tenths-of-degree to "XX.X"
// -----------------------------------------------------------------
#[inline]
fn fmt_temp(tenths: i32) -> String {
    let sign = if tenths < 0 { "-" } else { "" };
    let abs = tenths.unsigned_abs();
    format!("{}{}.{}", sign, abs / 10, abs % 10)
}

// -----------------------------------------------------------------
// main
// -----------------------------------------------------------------
fn main() -> io::Result<()> {
    let path = env::args().nth(1).unwrap_or_else(|| "measurements.txt".into());
    let file = fs::File::open(&path)?;

    // Safety: we treat the file as read-only bytes for the entire run.
    let mmap = unsafe { Mmap::map(&file)? };
    let data: &[u8] = &mmap;

    let nthreads = rayon::current_num_threads().max(1);
    let chunks   = split_chunks(data, nthreads);

    // Process in parallel
    let partial_maps: Vec<FxHashMap<&[u8], Stats>> = chunks
        .into_par_iter()
        .map(|chunk| process_chunk(chunk))
        .collect();

    // Merge all partial maps into one
    let mut global: FxHashMap<&[u8], Stats> =
        FxHashMap::with_capacity_and_hasher(1 << 14, Default::default());

    for map in partial_maps {
        for (k, v) in map {
            global
                .entry(k)
                .and_modify(|s| s.merge(&v))
                .or_insert(v);
        }
    }

    // Sort by station name
    let mut entries: Vec<(&[u8], Stats)> = global.into_iter().collect();
    entries.sort_unstable_by_key(|(k, _)| *k);

    // Output
    let stdout = io::stdout();
    // Use a raw fd write for speed
    let mut out = io::BufWriter::with_capacity(1 << 20, stdout.lock());

    out.write_all(b"{")?
    ;
    let last = entries.len().saturating_sub(1);
    for (i, (name, s)) in entries.iter().enumerate() {
        // mean rounded to one decimal place (round-half-up)
        let mean_tenths = (s.sum * 10 + (s.count as i64 / 2) * (if s.sum >= 0 { 1 } else { -1 }))
            / s.count as i64;
        // Re-express as tenths (the *10 and /count give us the rounded 1-dp value in tenths)
        // simpler: just round normally
        let mean_tenths: i32 = {
            let raw = s.sum as f64 / s.count as f64;
            (raw * 10.0).round() as i32
        };
        let sep = if i < last { ", " } else { "" };
        write!(
            out,
            "{}={}/{}/{}{}",
            std::str::from_utf8(name).unwrap(),
            fmt_temp(s.min),
            fmt_temp(mean_tenths),
            fmt_temp(s.max),
            sep
        )?;
    }
    out.write_all(b"}\n")?;
    Ok(())
}
