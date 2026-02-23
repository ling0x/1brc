//! One Billion Row Challenge — Rust implementation
//!
//! Strategy:
//!  - Memory-map the entire file for zero-copy I/O.
//!  - Split the mapped region into N chunks (one per logical CPU).
//!    Each chunk boundary is nudged forward to the next newline so lines
//!    are never split across threads.
//!  - Each thread maintains a thread-local FxHashMap keyed on a raw byte
//!    slice (represented as a u64 hash + original slice for collision
//!    correctness) accumulating (min, max, sum, count) in integer tenths
//!    of a degree to avoid floating-point rounding during accumulation.
//!  - The per-thread maps are merged on the main thread and the result is
//!    printed in ascending station-name order.

use std::{
    collections::HashMap,
    env,
    fs::File,
    io::{self, Write},
    thread,
};

// Use memmap2 for cross-platform memory mapping.
use memmap2::Mmap;

// ---------------------------------------------------------------------------
// Aggregation record stored as integer tenths (i32) to avoid fp accumulation
// errors during the hot loop.
// ---------------------------------------------------------------------------
#[derive(Clone)]
struct Stats {
    min: i32,
    max: i32,
    sum: i64,
    count: u64,
}

impl Stats {
    #[inline]
    fn new(val: i32) -> Self {
        Self { min: val, max: val, sum: val as i64, count: 1 }
    }

    #[inline]
    fn merge(&mut self, other: &Stats) {
        if other.min < self.min { self.min = other.min; }
        if other.max > self.max { self.max = other.max; }
        self.sum += other.sum;
        self.count += other.count;
    }

    #[inline]
    fn update(&mut self, val: i32) {
        if val < self.min { self.min = val; }
        if val > self.max { self.max = val; }
        self.sum += val as i64;
        self.count += 1;
    }
}

// ---------------------------------------------------------------------------
// Fast ASCII-only temperature parser.
// Format: optional '-', 1-2 digit integer part, '.', 1 digit fraction.
// Returns value in tenths of a degree (e.g. "-12.3" → -123).
// ---------------------------------------------------------------------------
#[inline]
fn parse_temp(bytes: &[u8]) -> i32 {
    let (negative, bytes) = if bytes[0] == b'-' {
        (true, &bytes[1..])
    } else {
        (false, bytes)
    };

    let val = match bytes {
        // "X.Y"
        [a, b'.', c] => (*a - b'0') as i32 * 10 + (*c - b'0') as i32,
        // "XY.Z"
        [a, b, b'.', c] => (*a - b'0') as i32 * 100 + (*b - b'0') as i32 * 10 + (*c - b'0') as i32,
        _ => panic!("unexpected temperature format"),
    };

    if negative { -val } else { val }
}

// ---------------------------------------------------------------------------
// Process one chunk of bytes (guaranteed to start and end on line boundaries).
// ---------------------------------------------------------------------------
fn process_chunk(chunk: &[u8]) -> HashMap<Vec<u8>, Stats> {
    let mut map: HashMap<Vec<u8>, Stats> = HashMap::with_capacity(1 << 14);
    let mut pos = 0;
    let len = chunk.len();

    while pos < len {
        // Find ';'
        let start = pos;
        while pos < len && chunk[pos] != b';' {
            pos += 1;
        }
        let name = &chunk[start..pos];
        pos += 1; // skip ';'

        // Find newline
        let temp_start = pos;
        while pos < len && chunk[pos] != b'\n' {
            pos += 1;
        }
        let temp_bytes = &chunk[temp_start..pos];
        pos += 1; // skip '\n'

        let val = parse_temp(temp_bytes);

        match map.get_mut(name) {
            Some(s) => s.update(val),
            None => { map.insert(name.to_vec(), Stats::new(val)); }
        }
    }

    map
}

// ---------------------------------------------------------------------------
// Format a tenths-integer as "X.Y" with round-half-up (IEEE 754
// roundTowardPositive semantics for positive results).
// ---------------------------------------------------------------------------
#[inline]
fn fmt_temp(tenths: i32, buf: &mut Vec<u8>) {
    if tenths < 0 {
        buf.push(b'-');
        let t = -tenths;
        fmt_positive(t, buf);
    } else {
        fmt_positive(tenths, buf);
    }
}

#[inline]
fn fmt_positive(tenths: i32, buf: &mut Vec<u8>) {
    let int_part = tenths / 10;
    let frac = tenths % 10;
    // write integer part
    if int_part >= 10 {
        buf.push(b'0' + (int_part / 10) as u8);
    }
    buf.push(b'0' + (int_part % 10) as u8);
    buf.push(b'.');
    buf.push(b'0' + frac as u8);
}

// ---------------------------------------------------------------------------
// Round mean (stored as sum_in_tenths / count) toward positive infinity.
// ---------------------------------------------------------------------------
#[inline]
fn round_mean(sum: i64, count: u64) -> i32 {
    // We need round( sum/count ) with roundTowardPositive.
    // = floor( (sum + count - 1) / count )  when sum > 0 and exactly on .5
    // More precisely: round half away from zero for positive, toward zero for negative?
    // The spec says "roundTowardPositive" (IEEE 754) which means round toward +∞.
    // i.e. -0.05 rounds to 0.0, +0.05 rounds to 0.1.
    //
    // Multiply sum by 10 so we work in hundredths, then round toward +inf.
    let sum100 = sum * 10; // now in hundredths
    // floor division toward -inf, then correct for round-half-up toward +inf:
    // round_toward_positive(a/b) = floor((a + b - 1) / b)  only for positive.
    // For the general case:
    let q = sum100 / count as i64;
    let r = sum100 % count as i64;
    // if remainder >= half of count (and we're positive) or > half (negative), round up
    let rounded = if r * 2 >= count as i64 { q + 1 } else { q };
    rounded as i32
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
fn main() -> io::Result<()> {
    let path = env::args().nth(1).unwrap_or_else(|| "measurements.txt".to_string());

    let file = File::open(&path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let data: &[u8] = &mmap;

    let ncpus = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
        .min(64);

    // Split into chunks aligned to newline boundaries.
    let total = data.len();
    let chunk_size = (total + ncpus - 1) / ncpus;

    let mut offsets: Vec<(usize, usize)> = Vec::with_capacity(ncpus);
    let mut start = 0usize;
    for _ in 0..ncpus {
        if start >= total { break; }
        let mut end = (start + chunk_size).min(total);
        // Advance end to next newline
        while end < total && data[end - 1] != b'\n' {
            end += 1;
        }
        offsets.push((start, end));
        start = end;
    }

    // Spawn one thread per chunk.
    let results: Vec<HashMap<Vec<u8>, Stats>> = thread::scope(|s| {
        let handles: Vec<_> = offsets
            .iter()
            .map(|&(lo, hi)| {
                let chunk = &data[lo..hi];
                s.spawn(move || process_chunk(chunk))
            })
            .collect();

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Merge all thread-local maps.
    let mut global: HashMap<Vec<u8>, Stats> = HashMap::with_capacity(1 << 14);
    for partial in results {
        for (name, stats) in partial {
            match global.get_mut(&name) {
                Some(s) => s.merge(&stats),
                None => { global.insert(name, stats); }
            }
        }
    }

    // Sort by station name.
    let mut entries: Vec<(Vec<u8>, Stats)> = global.into_iter().collect();
    entries.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));

    // Output.
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());
    let mut buf: Vec<u8> = Vec::with_capacity(32);

    out.write_all(b"{");
    for (i, (name, stats)) in entries.iter().enumerate() {
        if i > 0 { out.write_all(b", ")?; }
        out.write_all(name)?;
        out.write_all(b"=")?;

        buf.clear();
        fmt_temp(stats.min, &mut buf);
        out.write_all(&buf)?;
        out.write_all(b"/")?;

        buf.clear();
        let mean = round_mean(stats.sum, stats.count);
        fmt_temp(mean, &mut buf);
        out.write_all(&buf)?;
        out.write_all(b"/")?;

        buf.clear();
        fmt_temp(stats.max, &mut buf);
        out.write_all(&buf)?;
    }
    out.write_all(b"}\n")?;
    out.flush()?;

    Ok(())
}
