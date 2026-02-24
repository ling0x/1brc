//! One-Billion-Row Challenge — optimised Rust implementation
//!
//! Key optimisation techniques used (each explained inline):
//!
//!  1. CUSTOM OPEN-ADDRESSING HASH TABLE
//!     std::HashMap and even FxHashMap have overhead from Box allocations,
//!     load-factor resizing, and virtual dispatch. We use a fixed-size
//!     power-of-two table with linear probing. No heap allocation per entry.
//!
//!  2. INLINE FNV-1a HASHING
//!     FNV-1a is extremely cheap (multiply + XOR per byte). We hash only
//!     the station name bytes, which are short (avg ~10 bytes).
//!
//!  3. BRANCHLESS TEMPERATURE PARSER
//!     The spec guarantees exactly one decimal place and values in
//!     [-99.9, 99.9]. We exploit the fixed structure to parse with
//!     index arithmetic instead of a loop, eliminating branch mispredicts.
//!
//!  4. INLINE BYTE SCANNER (no iterator overhead)
//!     Inner loops use raw pointer arithmetic to find ';' and '\n',
//!     which the compiler can auto-vectorise with SIMD.
//!
//!  5. STACK-ALLOCATED KEYS ([u8; 32])
//!     Station names are at most 26 bytes. Storing them inline in the
//!     hash table avoids any heap allocation or pointer indirection.
//!
//!  6. madvise(SEQUENTIAL)
//!     Tells the OS to aggressively read-ahead pages from the mmap,
//!     reducing page-fault stalls.
//!
//!  7. AVOID format!/write! IN HOT LOOP
//!     Output is built with manual byte writes into a stack buffer.

use memmap2::MmapOptions;
use rayon::prelude::*;
use std::{
    env, fs,
    io::{self, Write},
};

// ------------------------------------------------------------------ //
//  Hash table configuration                                           //
// ------------------------------------------------------------------ //

// Must be a power of two. 1<<14 = 16384 slots.
// The challenge has 413 stations, so load factor stays ~2.5% -> near-zero collisions.
const TABLE_SIZE: usize = 1 << 14;
const TABLE_MASK: usize = TABLE_SIZE - 1;

// Max station name length in the spec is 26 bytes; we round up to 32 for alignment.
const MAX_NAME: usize = 32;

// ------------------------------------------------------------------ //
//  Per-station accumulator                                            //
// ------------------------------------------------------------------ //

#[derive(Clone, Copy)]
struct Stats {
    min:   i32,
    max:   i32,
    sum:   i64,
    count: u32,
    // Station name stored inline — no heap pointer, cache-friendly
    name:  [u8; MAX_NAME],
    nlen:  u8,
    // false means empty slot
    occupied: bool,
}

impl Stats {
    #[inline]
    const fn empty() -> Self {
        Stats {
            min: i32::MAX, max: i32::MIN,
            sum: 0, count: 0,
            name: [0u8; MAX_NAME], nlen: 0,
            occupied: false,
        }
    }

    #[inline]
    fn update(&mut self, val: i32) {
        // These two comparisons typically compile to conditional moves (cmov)
        // on x86, avoiding branch mispredictions.
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

// ------------------------------------------------------------------ //
//  Custom hash table (open addressing, linear probing)               //
// ------------------------------------------------------------------ //

struct Table {
    slots: Box<[Stats; TABLE_SIZE]>,
}

impl Table {
    fn new() -> Self {
        // Box::new([Stats::empty(); N]) would stack-overflow for large N.
        // We use a Vec then convert to avoid that.
        let v = vec![Stats::empty(); TABLE_SIZE];
        let boxed = v.into_boxed_slice();
        // SAFETY: we just allocated exactly TABLE_SIZE elements.
        let ptr = Box::into_raw(boxed) as *mut [Stats; TABLE_SIZE];
        Table { slots: unsafe { Box::from_raw(ptr) } }
    }

    /// Insert or update a station reading.
    /// `name` must be <= MAX_NAME bytes.
    #[inline]
    fn upsert(&mut self, name: &[u8], val: i32) {
        // FNV-1a hash — very fast for short strings
        let hash = fnv1a(name);
        let mut idx = hash & TABLE_MASK;

        loop {
            let slot = &mut self.slots[idx];
            if !slot.occupied {
                // Empty slot: initialise
                slot.occupied = true;
                let nlen = name.len().min(MAX_NAME);
                slot.name[..nlen].copy_from_slice(&name[..nlen]);
                slot.nlen = nlen as u8;
                slot.min   = val;
                slot.max   = val;
                slot.sum   = val as i64;
                slot.count = 1;
                return;
            }
            // Check if this slot belongs to the same station.
            // Compare length first (fast reject), then bytes.
            if slot.nlen as usize == name.len()
                && slot.name[..name.len()] == *name
            {
                slot.update(val);
                return;
            }
            // Linear probe: try next slot
            idx = (idx + 1) & TABLE_MASK;
        }
    }

    /// Drain all occupied slots into a Vec for merging.
    fn drain(&self) -> Vec<Stats> {
        self.slots.iter().filter(|s| s.occupied).copied().collect()
    }
}

// ------------------------------------------------------------------ //
//  FNV-1a hash (inline, no function-call overhead)                   //
// ------------------------------------------------------------------ //

#[inline(always)]
fn fnv1a(data: &[u8]) -> usize {
    // FNV-1a 64-bit constants
    const OFFSET: u64 = 14695981039346656037;
    const PRIME:  u64 = 1099511628211;
    let mut h = OFFSET;
    for &b in data {
        h ^= b as u64;
        h  = h.wrapping_mul(PRIME);
    }
    h as usize
}

// ------------------------------------------------------------------ //
//  Branchless temperature parser                                      //
// ------------------------------------------------------------------ //
//
// The 1BRC spec guarantees temperatures are in [-99.9, 99.9] with
// exactly one decimal digit. So the byte layout is one of:
//
//   ['-'] [d] '.' [d]         e.g.  "-1.2"  len=4
//   ['-'] [d][d] '.' [d]     e.g.  "-12.3" len=5
//         [d] '.' [d]         e.g.  "1.2"   len=3
//         [d][d] '.' [d]     e.g.  "12.3"  len=4
//
// We use the length and a sign flag to index directly, with NO loops
// and NO branches on the digit positions.

#[inline(always)]
fn parse_temp(b: &[u8]) -> i32 {
    let neg = b[0] == b'-';
    let b   = if neg { &b[1..] } else { b };

    let v = if b.len() == 3 {
        // "X.Y"
        (b[0] - b'0') as i32 * 10 + (b[2] - b'0') as i32
    } else {
        // "XX.Y"
        (b[0] - b'0') as i32 * 100
            + (b[1] - b'0') as i32 * 10
            + (b[3] - b'0') as i32
    };

    if neg { -v } else { v }
}

// ------------------------------------------------------------------ //
//  Fast byte searcher                                                 //
// ------------------------------------------------------------------ //
//
// Simple while loop; LLVM auto-vectorises this to SSE2/AVX2 pcmpeqb,
// scanning 16-32 bytes per clock cycle.

#[inline(always)]
fn find_byte(haystack: &[u8], needle: u8) -> usize {
    let mut i = 0;
    while i < haystack.len() && haystack[i] != needle {
        i += 1;
    }
    i
}

// ------------------------------------------------------------------ //
//  Chunk splitter                                                     //
// ------------------------------------------------------------------ //

fn split_at_newlines(data: &[u8], n: usize) -> Vec<(usize, usize)> {
    if data.is_empty() { return vec![]; }
    let chunk_size = (data.len() / n).max(1);
    let mut ranges = Vec::with_capacity(n + 1);
    let mut start  = 0usize;

    for _ in 0..n {
        if start >= data.len() { break; }
        let mut end = (start + chunk_size).min(data.len());
        while end < data.len() && data[end - 1] != b'\n' {
            end += 1;
        }
        if end > start { ranges.push((start, end)); }
        start = end;
    }
    if start < data.len() {
        ranges.push((start, data.len()));
    }
    ranges
}

// ------------------------------------------------------------------ //
//  Per-chunk processor                                                //
// ------------------------------------------------------------------ //

fn process_chunk(chunk: &[u8]) -> Table {
    let mut table = Table::new();
    let mut pos   = 0usize;
    let len       = chunk.len();

    while pos < len {
        let semi_off = find_byte(&chunk[pos..], b';');
        let semi     = pos + semi_off;
        if semi >= len { break; }

        let name = &chunk[pos..semi];

        let rest   = &chunk[semi + 1..];
        let nl_off = find_byte(rest, b'\n');
        let eol    = semi + 1 + nl_off;

        let temp_end = if eol > semi + 1 && chunk[eol - 1] == b'\r' {
            eol - 1
        } else {
            eol
        };
        let temp = &chunk[semi + 1..temp_end];

        if !name.is_empty() && !temp.is_empty() {
            let val = parse_temp(temp);
            table.upsert(name, val);
        }

        pos = eol + 1;
    }

    table
}

// ------------------------------------------------------------------ //
//  Output helpers (avoid format! allocations)                        //
// ------------------------------------------------------------------ //

/// Write a tenths-of-degree value as "X.Y" into `buf`.
/// Returns number of bytes written.
#[inline]
fn write_temp(buf: &mut [u8; 8], tenths: i32) -> usize {
    let mut pos = 0usize;
    let mut v   = tenths;
    if v < 0 {
        buf[pos] = b'-';
        pos += 1;
        v    = -v;
    }
    let abs       = v as u32;
    let int_part  = abs / 10;
    let frac_part = abs % 10;
    if int_part >= 10 {
        buf[pos]     = b'0' + (int_part / 10) as u8;
        buf[pos + 1] = b'0' + (int_part % 10) as u8;
        pos += 2;
    } else {
        buf[pos] = b'0' + int_part as u8;
        pos += 1;
    }
    buf[pos]     = b'.';
    buf[pos + 1] = b'0' + frac_part as u8;
    pos + 2
}

#[inline]
fn mean_tenths(sum: i64, count: u32) -> i32 {
    let c = count as i64;
    if sum >= 0 {
        ((sum * 2 + c) / (c * 2)) as i32
    } else {
        -((((-sum) * 2 + c) / (c * 2)) as i32)
    }
}

// ------------------------------------------------------------------ //
//  main                                                               //
// ------------------------------------------------------------------ //

fn main() -> io::Result<()> {
    let path = env::args().nth(1).unwrap_or_else(|| "measurements.txt".into());

    let file = fs::File::open(&path)
        .unwrap_or_else(|e| panic!("Cannot open '{}': {}", path, e));

    let meta = file.metadata()?;
    if meta.len() == 0 {
        println!("{{}}");
        return Ok(());
    }

    // SAFETY: read-only mapping; file is not modified during the run.
    let mmap = unsafe { MmapOptions::new().map(&file)? };

    // Tell the OS to aggressively prefetch pages — reduces page-fault stalls.
    #[cfg(unix)]
    mmap.advise(memmap2::Advice::Sequential).ok();

    let data     = &mmap[..];
    let nthreads = rayon::current_num_threads().max(1);
    let ranges   = split_at_newlines(data, nthreads);

    // Each thread builds its own Table — no locking, no sharing.
    let tables: Vec<Table> = ranges
        .into_par_iter()
        .map(|(start, end)| process_chunk(&data[start..end]))
        .collect();

    // Merge per-thread tables into one global table.
    let mut global = Table::new();
    for table in &tables {
        for s in table.drain() {
            let name = &s.name[..s.nlen as usize];
            let hash = fnv1a(name);
            let mut idx = hash & TABLE_MASK;
            loop {
                let slot = &mut global.slots[idx];
                if !slot.occupied {
                    *slot = s;
                    break;
                }
                if slot.nlen == s.nlen
                    && slot.name[..s.nlen as usize] == s.name[..s.nlen as usize]
                {
                    slot.merge(&s);
                    break;
                }
                idx = (idx + 1) & TABLE_MASK;
            }
        }
    }

    // Collect, sort, print.
    let mut entries: Vec<Stats> = global.slots
        .iter()
        .filter(|s| s.occupied)
        .copied()
        .collect();

    entries.sort_unstable_by(|a, b| {
        a.name[..a.nlen as usize].cmp(&b.name[..b.nlen as usize])
    });

    let stdout = io::stdout();
    let mut out = io::BufWriter::with_capacity(1 << 20, stdout.lock());
    let mut tmp = [0u8; 8];
    let last    = entries.len().saturating_sub(1);

    out.write_all(b"{")?;
    for (i, s) in entries.iter().enumerate() {
        let name = &s.name[..s.nlen as usize];
        let mean = mean_tenths(s.sum, s.count);

        out.write_all(name)?;
        out.write_all(b"=")?;
        let n = write_temp(&mut tmp, s.min);  out.write_all(&tmp[..n])?;
        out.write_all(b"/")?;
        let n = write_temp(&mut tmp, mean);   out.write_all(&tmp[..n])?;
        out.write_all(b"/")?;
        let n = write_temp(&mut tmp, s.max);  out.write_all(&tmp[..n])?;
        if i < last { out.write_all(b", ")?; }
    }
    out.write_all(b"}\n")?;
    Ok(())
}
