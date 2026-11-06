//! Human-readable summary of a splat (`splatplan inspect`).

use std::fmt;

use serde::Serialize;
use splat_types::{Gaussian, Report};

/// Number of bins in the opacity histogram.
pub const OPACITY_BINS: usize = 10;

/// Summary statistics for a set of Gaussians.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct InspectReport {
    /// Number of Gaussians.
    pub count: usize,
    /// Minimum of the means, per axis.
    pub bounds_min: [f64; 3],
    /// Maximum of the means, per axis.
    pub bounds_max: [f64; 3],
    /// 1st percentile of the means, per axis (robust bounds).
    pub p01: [f64; 3],
    /// 99th percentile of the means, per axis.
    pub p99: [f64; 3],
    /// Percentiles (1, 50, 99) of each Gaussian's largest linear scale.
    pub max_scale_percentiles: [f64; 3],
    /// Percentiles (1, 50, 99) of each Gaussian's smallest linear scale.
    pub min_scale_percentiles: [f64; 3],
    /// Opacity histogram over `[0, 1]` in equal-width bins.
    pub opacity_histogram: [usize; OPACITY_BINS],
}

impl Report for InspectReport {}

/// Linear-interpolated percentile `p` (0..100) of `values`. Returns 0 for an empty slice.
pub(crate) fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let pos = (p.clamp(0.0, 100.0) / 100.0) * (v.len() - 1) as f64;
    let i = pos.floor() as usize;
    let j = (i + 1).min(v.len() - 1);
    let f = pos - i as f64;
    v[i] * (1.0 - f) + v[j] * f
}

/// Computes an [`InspectReport`].
pub fn inspect(gaussians: &[Gaussian]) -> InspectReport {
    let mut r = InspectReport {
        count: gaussians.len(),
        bounds_min: [0.0; 3],
        bounds_max: [0.0; 3],
        p01: [0.0; 3],
        p99: [0.0; 3],
        max_scale_percentiles: [0.0; 3],
        min_scale_percentiles: [0.0; 3],
        opacity_histogram: [0; OPACITY_BINS],
    };
    if gaussians.is_empty() {
        return r;
    }
    for axis in 0..3 {
        let vals: Vec<f64> = gaussians.iter().map(|g| g.mean[axis] as f64).collect();
        r.bounds_min[axis] = vals.iter().copied().fold(f64::INFINITY, f64::min);
        r.bounds_max[axis] = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        r.p01[axis] = percentile(&vals, 1.0);
        r.p99[axis] = percentile(&vals, 99.0);
    }
    let maxs: Vec<f64> = gaussians.iter().map(|g| g.scale.max() as f64).collect();
    let mins: Vec<f64> = gaussians.iter().map(|g| g.scale.min() as f64).collect();
    for (k, p) in [1.0, 50.0, 99.0].into_iter().enumerate() {
        r.max_scale_percentiles[k] = percentile(&maxs, p);
        r.min_scale_percentiles[k] = percentile(&mins, p);
    }
    for g in gaussians {
        let bin =
            ((g.opacity.clamp(0.0, 1.0) * OPACITY_BINS as f32) as usize).min(OPACITY_BINS - 1);
        r.opacity_histogram[bin] += 1;
    }
    r
}

impl fmt::Display for InspectReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Gaussians: {}", self.count)?;
        writeln!(f, "Bounds (min..max of means):")?;
        for (a, name) in ["x", "y", "z"].iter().enumerate() {
            writeln!(
                f,
                "  {name}: {:>10.4} .. {:>10.4}   (p1..p99: {:>10.4} .. {:>10.4})",
                self.bounds_min[a], self.bounds_max[a], self.p01[a], self.p99[a]
            )?;
        }
        writeln!(
            f,
            "Largest scale  p1/p50/p99: {:.5} / {:.5} / {:.5}",
            self.max_scale_percentiles[0],
            self.max_scale_percentiles[1],
            self.max_scale_percentiles[2]
        )?;
        writeln!(
            f,
            "Smallest scale p1/p50/p99: {:.5} / {:.5} / {:.5}",
            self.min_scale_percentiles[0],
            self.min_scale_percentiles[1],
            self.min_scale_percentiles[2]
        )?;
        writeln!(f, "Opacity histogram:")?;
        let peak = self
            .opacity_histogram
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .max(1);
        for (i, n) in self.opacity_histogram.iter().enumerate() {
            let bar = "#".repeat(n * 40 / peak);
            writeln!(
                f,
                "  {:.1}-{:.1} {:>9} {bar}",
                i as f64 / OPACITY_BINS as f64,
                (i + 1) as f64 / OPACITY_BINS as f64,
                n
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use splat_types::Vec3f;

    #[test]
    fn percentile_basic() {
        let v = [3.0, 1.0, 2.0, 4.0, 5.0];
        assert_eq!(percentile(&v, 0.0), 1.0);
        assert_eq!(percentile(&v, 50.0), 3.0);
        assert_eq!(percentile(&v, 100.0), 5.0);
        assert_eq!(percentile(&v, 25.0), 2.0);
        assert_eq!(percentile(&[], 50.0), 0.0);
    }

    #[test]
    fn inspect_counts_and_histogram() {
        let gs: Vec<Gaussian> = (0..10)
            .map(|i| {
                Gaussian::isotropic(Vec3f::new(i as f32, 0.0, 0.0), 0.1, i as f32 / 10.0 + 0.05)
            })
            .collect();
        let r = inspect(&gs);
        assert_eq!(r.count, 10);
        assert_eq!(r.bounds_max[0], 9.0);
        assert_eq!(r.opacity_histogram, [1; 10]);
        assert!(r.to_string().contains("Gaussians: 10"));
        assert_eq!(inspect(&[]).count, 0);
    }
}
