//! Exact Euclidean distance transform (Felzenszwalb & Huttenlocher, 2012).
//!
//! The squared distance transform is separable: run a 1D transform along x,
//! then along y on the result, then along z. Each 1D pass computes the lower
//! envelope of parabolas `f(q) + (p − q)²` in linear time. Lines are
//! independent, so each pass is parallel over lines with no shared writes,
//! which keeps the result identical at any thread count.

use rayon::prelude::*;

/// Marker for "no obstacle on this line yet".
pub const INF: f32 = f32::INFINITY;

/// 1D squared distance transform of `f` into `d`. `f[q]` is 0 at obstacles
/// and `INF` elsewhere (or a partial result from a previous pass).
/// Scratch buffers `v` (parabola positions) and `z` (envelope boundaries) are
/// passed in so the hot loop does not allocate.
pub fn edt_1d(f: &[f32], d: &mut [f32], v: &mut Vec<usize>, z: &mut Vec<f64>) {
    let n = f.len();
    v.clear();
    z.clear();
    // Only finite samples are parabolas; INF ones never touch the envelope.
    for q in 0..n {
        if f[q] == INF {
            continue;
        }
        let fq = f[q] as f64;
        let qf = q as f64;
        loop {
            let Some(&p) = v.last() else {
                v.push(q);
                z.push(f64::NEG_INFINITY);
                break;
            };
            let pf = p as f64;
            // Intersection of parabolas rooted at p and q.
            let s = ((fq + qf * qf) - (f[p] as f64 + pf * pf)) / (2.0 * (qf - pf));
            if let Some(&zl) = z.last() {
                if s <= zl {
                    v.pop();
                    z.pop();
                    continue;
                }
            }
            v.push(q);
            z.push(s);
            break;
        }
    }
    if v.is_empty() {
        d.fill(INF);
        return;
    }
    let mut k = 0;
    for (q, out) in d.iter_mut().enumerate() {
        let qf = q as f64;
        while k + 1 < v.len() && z[k + 1] < qf {
            k += 1;
        }
        let p = v[k];
        let diff = qf - p as f64;
        *out = (diff * diff + f[p] as f64) as f32;
    }
}

/// In-place 3D squared EDT on an x-fastest grid of `dims`.
/// Input: 0 at obstacle voxels, `INF` elsewhere. Output: squared distance
/// (in voxel units) to the nearest obstacle voxel center, or `INF` if none.
pub fn edt_3d(grid: &mut [f32], dims: [usize; 3]) {
    let [nx, ny, nz] = dims;
    // Pass 1: along x (contiguous lines).
    grid.par_chunks_mut(nx).for_each_init(
        || (vec![0f32; nx], Vec::new(), Vec::new()),
        |(buf, v, z), line| {
            buf.copy_from_slice(line);
            edt_1d(buf, line, v, z);
        },
    );
    // Pass 2: along y, one z-slice per task.
    grid.par_chunks_mut(nx * ny).for_each_init(
        || (vec![0f32; ny], vec![0f32; ny], Vec::new(), Vec::new()),
        |(fbuf, dbuf, v, z), slice| {
            for i in 0..nx {
                for j in 0..ny {
                    fbuf[j] = slice[i + nx * j];
                }
                edt_1d(fbuf, dbuf, v, z);
                for j in 0..ny {
                    slice[i + nx * j] = dbuf[j];
                }
            }
        },
    );
    // Pass 3: along z. Columns are strided across slices, so transform into a
    // column-major scratch buffer (parallel over columns), then copy back.
    let src: &[f32] = grid;
    let mut cols = vec![0f32; src.len()];
    cols.par_chunks_mut(nz).enumerate().for_each_init(
        || (vec![0f32; nz], Vec::new(), Vec::new()),
        |(fbuf, v, z), (col, out)| {
            let (i, j) = (col % nx, col / nx);
            for k in 0..nz {
                fbuf[k] = src[i + nx * (j + ny * k)];
            }
            edt_1d(fbuf, out, v, z);
        },
    );
    grid.par_chunks_mut(nx * ny)
        .enumerate()
        .for_each(|(k, slice)| {
            for (col, val) in slice.iter_mut().enumerate() {
                *val = cols[col * nz + k];
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn brute(obst: &[bool], dims: [usize; 3]) -> Vec<f32> {
        let [nx, ny, nz] = dims;
        let pts: Vec<(i64, i64, i64)> = (0..obst.len())
            .filter(|&i| obst[i])
            .map(|i| {
                (
                    (i % nx) as i64,
                    ((i / nx) % ny) as i64,
                    (i / (nx * ny)) as i64,
                )
            })
            .collect();
        (0..nx * ny * nz)
            .map(|i| {
                let (x, y, z) = (
                    (i % nx) as i64,
                    ((i / nx) % ny) as i64,
                    (i / (nx * ny)) as i64,
                );
                pts.iter()
                    .map(|&(a, b, c)| ((x - a).pow(2) + (y - b).pow(2) + (z - c).pow(2)) as f32)
                    .fold(INF, f32::min)
            })
            .collect()
    }

    proptest! {
        #[test]
        fn matches_brute_force(
            nx in 1usize..9, ny in 1usize..9, nz in 1usize..9,
            seed in prop::collection::vec(any::<bool>(), 512),
            density in 0u8..4,
        ) {
            let dims = [nx, ny, nz];
            let n = nx * ny * nz;
            // Sparse obstacles: keep a bit only if `density` of 4 bits agree.
            let obst: Vec<bool> = (0..n).map(|i| seed[i % 512] && (i as u8 % 4) <= density && (i * 7 % 5 == 0 || density == 3)).collect();
            let mut g: Vec<f32> = obst.iter().map(|&o| if o { 0.0 } else { INF }).collect();
            edt_3d(&mut g, dims);
            prop_assert_eq!(g, brute(&obst, dims));
        }
    }

    #[test]
    fn empty_and_single() {
        let mut g = vec![INF; 27];
        edt_3d(&mut g, [3, 3, 3]);
        assert!(g.iter().all(|v| *v == INF));
        let mut g = vec![INF; 27];
        g[13] = 0.0;
        edt_3d(&mut g, [3, 3, 3]);
        assert_eq!(g[0], 3.0);
        assert_eq!(g[13], 0.0);
        assert_eq!(g[12], 1.0);
    }
}
