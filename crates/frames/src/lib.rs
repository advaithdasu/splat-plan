//! Blur scoring and keyframe selection for drone video frames.
//!
//! Structure-from-motion and splat training both want *sharp* frames with
//! *enough parallax* between them. Video at 10 fps gives thousands of
//! near-duplicate, sometimes motion-blurred frames. This crate picks a few
//! hundred good ones:
//!
//! 1. [`laplacian_variance`]: a classic sharpness score. The Laplacian
//!    responds to edges; blur removes edges, so its variance drops.
//! 2. [`motion_profile`]: how much the image changes between consecutive
//!    frames (mean absolute difference of small grayscale thumbnails), a
//!    cheap stand-in for camera motion / parallax.
//! 3. [`select_keyframes`]: split the video into `target` windows of equal
//!    *cumulative motion*, and keep the sharpest frame of each window. Fast
//!    passages get more keyframes per second, hovering gets fewer.
#![forbid(unsafe_code)]

use serde::Serialize;

/// A grayscale image in `0..=1`, row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Gray {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Pixel values.
    pub data: Vec<f32>,
}

impl Gray {
    /// Value at `(x, y)` (caller keeps coordinates in range).
    pub fn at(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }

    /// Box-downsampled copy whose width is at most `max_w` (aspect kept).
    pub fn thumbnail(&self, max_w: usize) -> Gray {
        let f = self.width.div_ceil(max_w.max(1)).max(1);
        let (w, h) = ((self.width / f).max(1), (self.height / f).max(1));
        let mut data = vec![0.0; w * h];
        for (ty, row) in data.chunks_mut(w).enumerate() {
            for (tx, v) in row.iter_mut().enumerate() {
                let mut s = 0.0;
                let mut n = 0.0;
                for y in ty * f..((ty + 1) * f).min(self.height) {
                    for x in tx * f..((tx + 1) * f).min(self.width) {
                        s += self.at(x, y);
                        n += 1.0;
                    }
                }
                *v = if n > 0.0 { s / n } else { 0.0 };
            }
        }
        Gray {
            width: w,
            height: h,
            data,
        }
    }
}

/// Variance of the 4-neighbor Laplacian over the interior pixels.
/// Higher is sharper. Returns 0 for images smaller than 3×3.
pub fn laplacian_variance(img: &Gray) -> f64 {
    if img.width < 3 || img.height < 3 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut n = 0.0f64;
    for y in 1..img.height - 1 {
        for x in 1..img.width - 1 {
            let l = (img.at(x - 1, y) + img.at(x + 1, y) + img.at(x, y - 1) + img.at(x, y + 1)
                - 4.0 * img.at(x, y)) as f64;
            sum += l;
            sum2 += l * l;
            n += 1.0;
        }
    }
    let mean = sum / n;
    (sum2 / n - mean * mean).max(0.0)
}

/// Mean absolute difference between two same-sized thumbnails.
pub fn frame_difference(a: &Gray, b: &Gray) -> f64 {
    if a.width != b.width || a.height != b.height || a.data.is_empty() {
        return 0.0;
    }
    a.data
        .iter()
        .zip(&b.data)
        .map(|(x, y)| (x - y).abs() as f64)
        .sum::<f64>()
        / a.data.len() as f64
}

/// `motion[i]` = difference between frame `i − 1` and frame `i` (`motion[0] = 0`).
pub fn motion_profile(thumbs: &[Gray]) -> Vec<f64> {
    let mut m = vec![0.0; thumbs.len()];
    for i in 1..thumbs.len() {
        m[i] = frame_difference(&thumbs[i - 1], &thumbs[i]);
    }
    m
}

/// Result of keyframe selection.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Selection {
    /// Indices of the chosen frames, increasing.
    pub chosen: Vec<usize>,
    /// Blur score of every input frame.
    pub sharpness: Vec<f64>,
    /// Frames rejected as too blurry before selection.
    pub rejected_blurry: usize,
}

/// Picks up to `target` keyframes.
///
/// Frames whose sharpness is below `min_rel_sharpness` × the median are
/// discarded first. The remaining sequence is cut into `target` windows of
/// equal cumulative motion; in each window the sharpest frame wins (ties:
/// earliest). Deterministic.
pub fn select_keyframes(
    sharpness: &[f64],
    motion: &[f64],
    target: usize,
    min_rel_sharpness: f64,
) -> Selection {
    let n = sharpness.len();
    let mut sorted = sharpness.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = sorted.get(n / 2).copied().unwrap_or(0.0);
    let ok: Vec<bool> = sharpness
        .iter()
        .map(|s| *s >= min_rel_sharpness * median)
        .collect();
    let rejected_blurry = ok.iter().filter(|o| !**o).count();
    if n == 0 || target == 0 {
        return Selection {
            chosen: vec![],
            sharpness: sharpness.to_vec(),
            rejected_blurry,
        };
    }
    // Cumulative motion; a tiny constant keeps windows well defined when the camera hovers.
    let mut cum = Vec::with_capacity(n);
    let mut acc = 0.0;
    for i in 0..n {
        acc += motion.get(i).copied().unwrap_or(0.0) + 1e-9;
        cum.push(acc);
    }
    let total = acc;
    let mut chosen = Vec::new();
    let mut start = 0;
    for w in 0..target {
        let end_value = total * (w + 1) as f64 / target as f64;
        let mut end = start;
        while end < n && cum[end] <= end_value {
            end += 1;
        }
        if w + 1 == target {
            end = n;
        }
        let best =
            (start..end)
                .filter(|&i| ok[i])
                .fold(None, |best: Option<usize>, i| match best {
                    Some(b) if sharpness[b] >= sharpness[i] => Some(b),
                    _ => Some(i),
                });
        if let Some(b) = best {
            chosen.push(b);
        }
        start = end;
    }
    chosen.dedup();
    Selection {
        chosen,
        sharpness: sharpness.to_vec(),
        rejected_blurry,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    /// Random texture, then a box blur of the given radius.
    fn textured(seed: u64, blur: usize, shift: usize) -> Gray {
        let (w, h) = (96, 64);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let base: Vec<f32> = (0..(w + 64) * h).map(|_| rng.gen_range(0.0..1.0)).collect();
        let mut data = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut s = 0.0;
                let mut n = 0.0;
                for dy in -(blur as i64)..=blur as i64 {
                    for dx in -(blur as i64)..=blur as i64 {
                        let (xx, yy) = (x as i64 + dx + shift as i64, y as i64 + dy);
                        if xx >= 0 && yy >= 0 && (xx as usize) < w + 64 && (yy as usize) < h {
                            s += base[yy as usize * (w + 64) + xx as usize];
                            n += 1.0;
                        }
                    }
                }
                data[y * w + x] = s / n;
            }
        }
        Gray {
            width: w,
            height: h,
            data,
        }
    }

    #[test]
    fn blur_lowers_sharpness_monotonically() {
        let s: Vec<f64> = (0..4)
            .map(|b| laplacian_variance(&textured(1, b, 0)))
            .collect();
        assert!(s.windows(2).all(|w| w[0] > w[1]), "{s:?}");
        assert_eq!(
            laplacian_variance(&Gray {
                width: 2,
                height: 2,
                data: vec![0.0; 4]
            }),
            0.0
        );
    }

    #[test]
    fn selection_prefers_sharp_frames_and_spreads_over_motion() {
        // 60 frames panning steadily; every third frame is blurred.
        let frames: Vec<Gray> = (0..60)
            .map(|i| textured(2, if i % 3 == 0 { 0 } else { 2 }, i))
            .collect();
        let sharp: Vec<f64> = frames.iter().map(laplacian_variance).collect();
        let thumbs: Vec<Gray> = frames.iter().map(|f| f.thumbnail(32)).collect();
        let motion = motion_profile(&thumbs);
        let sel = select_keyframes(&sharp, &motion, 10, 0.5);
        assert_eq!(sel.chosen.len(), 10);
        assert!(
            sel.chosen.iter().all(|i| i % 3 == 0),
            "picked blurry frames: {:?}",
            sel.chosen
        );
        assert!(
            sel.chosen.windows(2).all(|w| w[1] - w[0] >= 3),
            "{:?}",
            sel.chosen
        );
        // Deterministic.
        assert_eq!(sel, select_keyframes(&sharp, &motion, 10, 0.5));
    }

    #[test]
    fn hovering_gets_fewer_keyframes() {
        // Frames 0-29 static (hover), 30-59 moving: most keyframes land in the moving half.
        let frames: Vec<Gray> = (0..60)
            .map(|i| textured(3, 0, if i < 30 { 0 } else { (i - 29) * 2 }))
            .collect();
        let sharp: Vec<f64> = frames.iter().map(laplacian_variance).collect();
        let motion = motion_profile(&frames.iter().map(|f| f.thumbnail(32)).collect::<Vec<_>>());
        let sel = select_keyframes(&sharp, &motion, 10, 0.0);
        let moving = sel.chosen.iter().filter(|i| **i >= 30).count();
        assert!(moving >= 8, "{:?}", sel.chosen);
        assert!(select_keyframes(&[], &[], 5, 0.5).chosen.is_empty());
    }
}
