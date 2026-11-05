//! Lossless JPEG (ITU T.81 process 14, "LJ92") encoding of 16-bit samples,
//! as DNG compression 7 stores raw tiles.
//!
//! A CFA tile `W` sensels wide is coded as a JPEG image `W / 2` wide with two
//! interleaved components, as DNG writers commonly do: the samples are the
//! same, row by row, but each one is predicted from the sensel two columns
//! to its left, which has the same colour in a Bayer row. Predictor 1 (the
//! left neighbour) is used throughout, with one Huffman table optimised for
//! the tile.

/// The lossless JPEG stream of `samples`, `width` × `height` in row-major
/// order, coded as `components` interleaved components (`width` must be a
/// multiple of `components`).
pub fn encode(samples: &[u16], width: usize, height: usize, components: usize) -> Vec<u8> {
    assert_eq!(samples.len(), width * height);
    assert!(components > 0 && width % components == 0);
    let differences = differences(samples, width, components);
    let mut histogram = [0u64; 17];
    for &d in &differences {
        histogram[category(d) as usize] += 1;
    }
    let table = HuffmanTable::optimal(&histogram);

    let mut out = Vec::with_capacity(samples.len() + 1024);
    out.extend([0xFF, 0xD8]); // SOI
    // SOF3: lossless, Huffman coded; 16-bit precision.
    out.extend([0xFF, 0xC3]);
    out.extend((8 + 3 * components as u16).to_be_bytes());
    out.push(16);
    out.extend((height as u16).to_be_bytes());
    out.extend(((width / components) as u16).to_be_bytes());
    out.push(components as u8);
    for id in 0..components as u8 {
        // Component id, sampling factors 1×1, no quantization table.
        out.extend([id, 0x11, 0]);
    }
    // DHT: one DC table, class 0, id 0.
    out.extend([0xFF, 0xC4]);
    out.extend((2 + 1 + 16 + table.values.len() as u16).to_be_bytes());
    out.push(0x00);
    out.extend(table.counts);
    out.extend(&table.values);
    // SOS: every component uses table 0; predictor 1, no point transform.
    out.extend([0xFF, 0xDA]);
    out.extend((6 + 2 * components as u16).to_be_bytes());
    out.push(components as u8);
    for id in 0..components as u8 {
        out.extend([id, 0x00]);
    }
    out.extend([1, 0, 0]);

    let mut bits = BitWriter::new(&mut out);
    for &d in &differences {
        let ssss = category(d);
        let (code, length) = table.codes[ssss as usize];
        bits.put(u64::from(code), length);
        // Category 16 (the difference 32768) has no extra bits.
        if ssss > 0 && ssss < 16 {
            // Negative differences are sent as d - 1 in `ssss` bits.
            let extra = if d < 0 { d - 1 } else { d };
            bits.put((extra as u64) & ((1 << ssss) - 1), ssss);
        }
    }
    bits.finish();
    out.extend([0xFF, 0xD9]); // EOI
    out
}

/// Prediction differences in coding order, modulo 2¹⁶ and centred on 0.
fn differences(samples: &[u16], width: usize, components: usize) -> Vec<i32> {
    samples
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let (row, column) = (i / width, i % width);
            let prediction = if column >= components {
                samples[i - components]
            } else if row > 0 {
                // First pixel of a row: the sample above.
                samples[i - width]
            } else {
                // First pixel of the image: half the range.
                1 << 15
            };
            i32::from(x.wrapping_sub(prediction) as i16)
        })
        .collect()
}

/// The magnitude category SSSS of a difference: its bit length.
fn category(d: i32) -> u8 {
    (32 - d.unsigned_abs().leading_zeros()) as u8
}

/// A Huffman table for categories 0..=16.
struct HuffmanTable {
    /// BITS: number of codes of each length 1..=16.
    counts: [u8; 16],
    /// HUFFVAL: categories in order of increasing code length.
    values: Vec<u8>,
    /// Code and length of each category (length 0 when unused).
    codes: [(u16, u8); 17],
}

impl HuffmanTable {
    /// The optimal table for `histogram`, limited to 16-bit codes and with
    /// no code of all ones (ITU T.81 Annex K.2).
    fn optimal(histogram: &[u64; 17]) -> Self {
        // A reserved symbol of frequency 1 claims the all-ones code.
        const RESERVED: usize = 17;
        let mut freq = [0u64; 18];
        freq[..17].copy_from_slice(histogram);
        freq[RESERVED] = 1;
        let mut size = [0usize; 18];
        let mut others = [None::<usize>; 18];
        loop {
            // The two least frequent remaining symbols; ties go to the larger
            // symbol, as in the reference procedure.
            let mut least: Option<usize> = None;
            let mut second: Option<usize> = None;
            for v in 0..18 {
                if freq[v] == 0 {
                    continue;
                }
                if least.is_none_or(|l| freq[v] <= freq[l]) {
                    second = least;
                    least = Some(v);
                } else if second.is_none_or(|s| freq[v] <= freq[s]) {
                    second = Some(v);
                }
            }
            let (Some(v1), Some(v2)) = (least, second) else {
                break;
            };
            freq[v1] += freq[v2];
            freq[v2] = 0;
            // Lengthen every code in both chains and join the chains.
            let mut v = v1;
            loop {
                size[v] += 1;
                match others[v] {
                    Some(next) => v = next,
                    None => break,
                }
            }
            others[v] = Some(v2);
            let mut v = v2;
            loop {
                size[v] += 1;
                match others[v] {
                    Some(next) => v = next,
                    None => break,
                }
            }
        }
        // Count codes per length, then limit lengths to 16 (Figure K.3).
        let mut bits = [0u32; 33];
        for &s in &size {
            if s > 0 {
                bits[s] += 1;
            }
        }
        let mut i = 32;
        while i > 16 {
            while bits[i] > 0 {
                let mut j = i - 2;
                while bits[j] == 0 {
                    j -= 1;
                }
                bits[i] -= 2;
                bits[i - 1] += 1;
                bits[j + 1] += 2;
                bits[j] -= 1;
            }
            i -= 1;
        }
        // Drop the reserved symbol's code, the longest.
        let mut i = 16;
        while bits[i] == 0 {
            i -= 1;
        }
        bits[i] -= 1;

        let mut order: Vec<usize> = (0..17).filter(|&v| size[v] > 0).collect();
        order.sort_by_key(|&v| (size[v], v));
        let counts: [u8; 16] = std::array::from_fn(|i| bits[i + 1] as u8);
        let values: Vec<u8> = order.iter().map(|&v| v as u8).collect();
        // Canonical codes from the counts (Annex C).
        let mut codes = [(0u16, 0u8); 17];
        let mut code = 0u16;
        let mut k = 0;
        for length in 1..=16u8 {
            for _ in 0..counts[usize::from(length) - 1] {
                codes[usize::from(values[k])] = (code, length);
                code += 1;
                k += 1;
            }
            code <<= 1;
        }
        Self {
            counts,
            values,
            codes,
        }
    }
}

/// Entropy-coded bits, most significant first, with 0xFF bytes stuffed.
struct BitWriter<'a> {
    out: &'a mut Vec<u8>,
    buffer: u64,
    count: u32,
}

impl<'a> BitWriter<'a> {
    fn new(out: &'a mut Vec<u8>) -> Self {
        Self {
            out,
            buffer: 0,
            count: 0,
        }
    }

    /// Append the low `length` bits of `value` (at most 32).
    fn put(&mut self, value: u64, length: u8) {
        self.buffer = (self.buffer << length) | value;
        self.count += u32::from(length);
        while self.count >= 8 {
            self.count -= 8;
            let byte = (self.buffer >> self.count) as u8;
            self.out.push(byte);
            if byte == 0xFF {
                self.out.push(0);
            }
        }
        self.buffer &= (1 << self.count) - 1;
    }

    /// Pad the last byte with one bits.
    fn finish(mut self) {
        if self.count > 0 {
            let pad = 8 - self.count as u8;
            self.put((1 << pad) - 1, pad);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal decoder for the streams `encode` writes.
    fn decode(data: &[u8]) -> (Vec<u16>, usize, usize, usize) {
        let be16 = |at: usize| usize::from(u16::from_be_bytes([data[at], data[at + 1]]));
        let mut at = 2;
        let (mut width, mut height, mut components) = (0, 0, 0);
        let mut table: Vec<(u16, u8, u8)> = Vec::new();
        loop {
            let marker = data[at + 1];
            let length = be16(at + 2);
            let body = at + 4;
            match marker {
                0xC3 => {
                    assert_eq!(data[body], 16);
                    height = be16(body + 1);
                    width = be16(body + 3);
                    components = usize::from(data[body + 5]);
                }
                0xC4 => {
                    let counts = &data[body + 1..body + 17];
                    let mut values = data[body + 17..].iter();
                    let mut code = 0u16;
                    for (i, &n) in counts.iter().enumerate() {
                        for _ in 0..n {
                            table.push((code, i as u8 + 1, *values.next().unwrap()));
                            code += 1;
                        }
                        code <<= 1;
                    }
                }
                0xDA => {
                    at = body + length - 2;
                    break;
                }
                _ => panic!("unexpected marker {marker:x}"),
            }
            at = body + length - 2;
        }
        // Unstuff the entropy-coded segment up to EOI.
        let mut bytes = Vec::new();
        while !(data[at] == 0xFF && data[at + 1] == 0xD9) {
            bytes.push(data[at]);
            at += if data[at] == 0xFF { 2 } else { 1 };
        }
        let mut position = 0usize;
        let mut bit = || {
            let b = (bytes[position / 8] >> (7 - position % 8)) & 1;
            position += 1;
            u16::from(b)
        };
        let full_width = width * components;
        let mut out: Vec<u16> = Vec::with_capacity(full_width * height);
        for i in 0..full_width * height {
            let (mut code, mut length) = (0u16, 0u8);
            let ssss = loop {
                code = code << 1 | bit();
                length += 1;
                if let Some(&(_, _, v)) = table.iter().find(|e| e.0 == code && e.1 == length) {
                    break v;
                }
            };
            let d: i32 = match ssss {
                0 => 0,
                16 => 32768,
                s => {
                    let mut v = 0i32;
                    for _ in 0..s {
                        v = v << 1 | i32::from(bit());
                    }
                    if v < 1 << (s - 1) {
                        v - (1 << s) + 1
                    } else {
                        v
                    }
                }
            };
            let (row, column) = (i / full_width, i % full_width);
            let prediction = if column >= components {
                out[i - components]
            } else if row > 0 {
                out[i - full_width]
            } else {
                1 << 15
            };
            out.push(prediction.wrapping_add(d as u16));
        }
        (out, width, height, components)
    }

    fn round_trip(samples: &[u16], width: usize, height: usize) {
        let data = encode(samples, width, height, 2);
        let (decoded, w, h, c) = decode(&data);
        assert_eq!((w, h, c), (width / 2, height, 2));
        assert_eq!(decoded, samples);
    }

    #[test]
    fn random_and_extreme_samples_survive() {
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let random: Vec<u16> = (0..64 * 16).map(|_| next() as u16).collect();
        round_trip(&random, 64, 16);
        // Differences of exactly ±32768 and runs of 0xFF bytes.
        let extreme: Vec<u16> = (0..32 * 4)
            .map(|i| if (i / 2) % 2 == 0 { 0 } else { 0x8000 })
            .collect();
        round_trip(&extreme, 32, 4);
        round_trip(&[0xFFFF; 16 * 2], 16, 2);
    }

    #[test]
    fn a_flat_tile_still_gets_a_valid_table() {
        round_trip(&[1000; 16 * 16], 16, 16);
        let data = encode(&[1000; 16 * 16], 16, 16, 2);
        // Smooth data compresses: here to well under a bit per sample.
        assert!(data.len() < 16 * 16 / 8 + 64, "{} bytes", data.len());
    }

    #[test]
    fn skewed_histograms_need_long_codes() {
        // Frequencies doubling per category force lengths past 16 before
        // limiting.
        let mut histogram = [0u64; 17];
        for (i, f) in histogram.iter_mut().enumerate() {
            *f = 1 << (2 * i);
        }
        let table = HuffmanTable::optimal(&histogram);
        assert!(
            table
                .codes
                .iter()
                .all(|&(_, length)| (1..=16).contains(&length))
        );
        // Kraft: the codes leave room for the reserved all-ones code.
        let kraft: f64 = table
            .codes
            .iter()
            .map(|&(_, l)| 0.5f64.powi(l.into()))
            .sum();
        assert!(kraft < 1.0);
        let all_ones = |&(code, length): &(u16, u8)| u32::from(code) == (1 << length) - 1;
        assert!(!table.codes.iter().any(all_ones));
    }
}
