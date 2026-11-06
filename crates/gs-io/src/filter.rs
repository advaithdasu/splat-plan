//! Simple, deterministic cleanup filters for real-world splats.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use splat_types::Gaussian;

use crate::inspect::percentile;

/// Keeps Gaussians with opacity at least `min_opacity`.
pub fn filter_opacity(gaussians: &[Gaussian], min_opacity: f32) -> Vec<Gaussian> {
    gaussians
        .iter()
        .filter(|g| g.opacity >= min_opacity)
        .copied()
        .collect()
}

/// Settings for [`filter_outliers`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OutlierFilter {
    /// Lower percentile (0..100) of means, per axis, defining the core box.
    pub low_percentile: f64,
    /// Upper percentile (0..100) of means, per axis.
    pub high_percentile: f64,
    /// Padding added to the core box, as a fraction of its largest edge.
    pub pad_fraction: f64,
    /// Gaussians whose largest scale exceeds this multiple of the median
    /// largest scale are dropped (huge background "skybox" splats).
    pub max_scale_ratio: f64,
}

impl Default for OutlierFilter {
    fn default() -> Self {
        OutlierFilter {
            low_percentile: 1.0,
            high_percentile: 99.0,
            pad_fraction: 0.25,
            max_scale_ratio: 50.0,
        }
    }
}

/// Removes far-away floaters and oversized background Gaussians.
///
/// Uses per-axis percentiles of the means (robust to a few distant
/// outliers) padded by `pad_fraction`, plus a cap on the largest scale.
pub fn filter_outliers(gaussians: &[Gaussian], cfg: &OutlierFilter) -> Vec<Gaussian> {
    if gaussians.is_empty() {
        return Vec::new();
    }
    let mut lo = [0.0f64; 3];
    let mut hi = [0.0f64; 3];
    for axis in 0..3 {
        let vals: Vec<f64> = gaussians.iter().map(|g| g.mean[axis] as f64).collect();
        lo[axis] = percentile(&vals, cfg.low_percentile);
        hi[axis] = percentile(&vals, cfg.high_percentile);
    }
    let largest_edge = (0..3).map(|a| hi[a] - lo[a]).fold(0.0, f64::max);
    let pad = largest_edge * cfg.pad_fraction;
    let max_scales: Vec<f64> = gaussians.iter().map(|g| g.scale.max() as f64).collect();
    let scale_cap = percentile(&max_scales, 50.0) * cfg.max_scale_ratio;

    gaussians
        .iter()
        .filter(|g| {
            (0..3).all(|a| {
                let m = g.mean[a] as f64;
                m >= lo[a] - pad && m <= hi[a] + pad
            }) && (g.scale.max() as f64) <= scale_cap
        })
        .copied()
        .collect()
}

/// For each Gaussian, true if at least `min_neighbors` other Gaussian
/// means lie within `radius` of its mean (same units as the means).
///
/// Trained splats leave "floaters": stray Gaussians in empty air, mostly
/// near the cameras where training had few views to pin them down. Real
/// surfaces are covered by many overlapping Gaussians, so a Gaussian with
/// almost no neighbors is a floater far more often than it is geometry.
/// Thin real objects (a cable, a chair leg) still have neighbors along their
/// length, so a small radius keeps them.
///
/// Uses a hash grid with cell size `radius`; the result does not depend on
/// iteration order, so it is deterministic.
pub fn isolation_mask(gaussians: &[Gaussian], radius: f64, min_neighbors: usize) -> Vec<bool> {
    if min_neighbors == 0 || !(radius > 0.0 && radius.is_finite()) {
        return vec![true; gaussians.len()];
    }
    let cell = |g: &Gaussian| -> [i64; 3] {
        [0, 1, 2].map(|a| (g.mean[a] as f64 / radius).floor() as i64)
    };
    let mut grid: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
    for (i, g) in gaussians.iter().enumerate() {
        grid.entry(cell(g)).or_default().push(i as u32);
    }
    let r2 = radius * radius;
    gaussians
        .iter()
        .enumerate()
        .map(|(i, g)| {
            let c = cell(g);
            let p = g.mean_f64();
            let mut n = 0usize;
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let Some(bucket) = grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) else {
                            continue;
                        };
                        for &j in bucket {
                            if j as usize != i
                                && (gaussians[j as usize].mean_f64() - p).norm_squared() <= r2
                            {
                                n += 1;
                                if n >= min_neighbors {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
            false
        })
        .collect()
}

/// Keep-mask for floater removal: a Gaussian is dropped only if it is both
/// fainter than `max_opacity` and isolated (fewer than `min_neighbors`
/// other means within `radius`, see [`isolation_mask`]).
///
/// Either condition alone removes real geometry: coarse splats cover blank
/// walls with a few large, opaque Gaussians whose means are far apart
/// (isolated but solid), and a surface that trained faint, like glass, is
/// many faint Gaussians close together (faint but dense).
pub fn floater_mask(
    gaussians: &[Gaussian],
    radius: f64,
    min_neighbors: usize,
    max_opacity: f32,
) -> Vec<bool> {
    isolation_mask(gaussians, radius, min_neighbors)
        .into_iter()
        .zip(gaussians)
        .map(|(not_isolated, g)| not_isolated || g.opacity >= max_opacity)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use splat_types::Vec3f;

    #[test]
    fn removes_far_floater_and_huge_splat() {
        let mut gs: Vec<Gaussian> = (0..1000)
            .map(|i| {
                let f = i as f32 / 1000.0;
                Gaussian::isotropic(Vec3f::new(f, 1.0 - f, f * 0.5), 0.01, 0.9)
            })
            .collect();
        gs.push(Gaussian::isotropic(Vec3f::new(100.0, 0.0, 0.0), 0.01, 0.9));
        gs.push(Gaussian::isotropic(Vec3f::new(0.5, 0.5, 0.2), 5.0, 0.9));
        let kept = filter_outliers(&gs, &OutlierFilter::default());
        assert_eq!(kept.len(), 1000);
        assert_eq!(filter_opacity(&gs, 0.95).len(), 0);
        assert!(filter_outliers(&[], &OutlierFilter::default()).is_empty());
    }

    #[test]
    fn isolation_keeps_clusters_and_drops_strays() {
        // A 10 x 10 sheet at 1 cm spacing, a 1 m "cable" at 2 cm spacing,
        // and three lone floaters.
        let mut gs = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                gs.push(Gaussian::isotropic(
                    Vec3f::new(i as f32 * 0.01, j as f32 * 0.01, 0.0),
                    0.005,
                    0.9,
                ));
            }
        }
        for i in 0..50 {
            gs.push(Gaussian::isotropic(
                Vec3f::new(2.0 + i as f32 * 0.02, 0.0, 1.0),
                0.005,
                0.9,
            ));
        }
        for p in [[5.0, 5.0, 5.0], [0.5, 0.5, 0.5], [-3.0, 0.0, 1.0]] {
            gs.push(Gaussian::isotropic(Vec3f::new(p[0], p[1], p[2]), 0.05, 0.1));
        }
        let keep = isolation_mask(&gs, 0.1, 5);
        assert!(keep[..150].iter().all(|k| *k));
        assert!(keep[150..].iter().all(|k| !*k));
        // Disabled settings keep everything.
        assert!(isolation_mask(&gs, 0.1, 0).iter().all(|k| *k));
        assert!(isolation_mask(&gs, 0.0, 5).iter().all(|k| *k));
        assert!(isolation_mask(&[], 0.1, 5).is_empty());
        // Opaque strays survive floater removal; faint ones do not.
        let mut opaque = gs.clone();
        opaque[150].opacity = 0.9;
        let keep = floater_mask(&opaque, 0.1, 5, 0.2);
        assert!(keep[..151].iter().all(|k| *k));
        assert!(keep[151..].iter().all(|k| !*k));
    }
}
