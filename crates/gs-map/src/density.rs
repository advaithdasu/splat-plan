//! Density grid: sum of Gaussian "opacity mass" at every voxel center.

use rayon::prelude::*;
use splat_types::Gaussian;
use splat_types::nalgebra::Matrix3;

use crate::grid::GridSpec;

/// Number of z-layers per work unit. Each slab is owned by exactly one task,
/// so no two threads ever add into the same voxel.
const SLAB: usize = 4;

/// Per-voxel density `Σ α · exp(−½ (x−μ)ᵀ Σ⁻¹ (x−μ))`.
#[derive(Debug, Clone, PartialEq)]
pub struct DensityGrid {
    /// Grid geometry.
    pub spec: GridSpec,
    /// Density at each voxel center, x-fastest.
    pub density: Vec<f32>,
}

/// Precomputed per-Gaussian data for accumulation.
struct Prepared {
    mean: [f64; 3],
    inv: Matrix3<f64>,
    opacity: f64,
    /// Inclusive voxel index range per axis.
    lo: [usize; 3],
    hi: [usize; 3],
}

impl DensityGrid {
    /// Accumulates the density of `gaussians` on `spec`.
    ///
    /// - Only voxels inside each Gaussian's `k_sigma` ellipsoid receive mass.
    /// - **Anti-aliasing**: the covariance is widened by `(res/2)²·I` before
    ///   evaluation. Without it, thin "disc" Gaussians (normal std-dev of a few
    ///   millimetres, common in trained splats) fall between voxel centers and
    ///   surfaces vanish from the map. Widening can only *add* occupancy, so it
    ///   errs on the safe side.
    /// - **Determinism**: z-slabs are processed in parallel, but inside a
    ///   slab Gaussians are added in their input order, so every voxel's sum is
    ///   computed in the same order at any thread count.
    pub fn accumulate(
        spec: GridSpec,
        gaussians: &[Gaussian],
        k_sigma: f64,
        min_opacity: f32,
    ) -> Self {
        let prepared: Vec<Prepared> = gaussians
            .iter()
            .filter(|g| g.opacity >= min_opacity)
            .filter_map(|g| prepare(&spec, g, k_sigma))
            .collect();

        // Bucket Gaussian indices by slab (input order preserved within a bucket).
        let n_slabs = spec.dims[2].div_ceil(SLAB);
        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); n_slabs];
        for (i, p) in prepared.iter().enumerate() {
            for bucket in &mut buckets[(p.lo[2] / SLAB)..=(p.hi[2] / SLAB)] {
                bucket.push(i as u32);
            }
        }

        let layer = spec.dims[0] * spec.dims[1];
        let mut density = vec![0.0f32; spec.len()];
        let k2 = k_sigma * k_sigma;
        density
            .par_chunks_mut(layer * SLAB)
            .zip(buckets.par_iter())
            .enumerate()
            .for_each(|(slab, (chunk, bucket))| {
                let z0 = slab * SLAB;
                let z1 = (z0 + SLAB).min(spec.dims[2]) - 1;
                for &gi in bucket {
                    let p = &prepared[gi as usize];
                    for k in p.lo[2].max(z0)..=p.hi[2].min(z1) {
                        let z = spec.origin.z + (k as f64 + 0.5) * spec.res - p.mean[2];
                        for j in p.lo[1]..=p.hi[1] {
                            let y = spec.origin.y + (j as f64 + 0.5) * spec.res - p.mean[1];
                            let row = (k - z0) * layer + j * spec.dims[0];
                            for i in p.lo[0]..=p.hi[0] {
                                let x = spec.origin.x + (i as f64 + 0.5) * spec.res - p.mean[0];
                                let m = &p.inv;
                                // Mahalanobis distance squared, written out (Σ⁻¹ is symmetric).
                                let e = m[(0, 0)] * x * x
                                    + m[(1, 1)] * y * y
                                    + m[(2, 2)] * z * z
                                    + 2.0
                                        * (m[(0, 1)] * x * y
                                            + m[(0, 2)] * x * z
                                            + m[(1, 2)] * y * z);
                                if e <= k2 {
                                    chunk[row + i] += (p.opacity * (-0.5 * e).exp()) as f32;
                                }
                            }
                        }
                    }
                }
            });
        DensityGrid { spec, density }
    }
}

fn prepare(spec: &GridSpec, g: &Gaussian, k_sigma: f64) -> Option<Prepared> {
    let aa = 0.25 * spec.res * spec.res;
    let cov = g.covariance() + Matrix3::identity() * aa;
    let inv = cov.try_inverse()?;
    let mean = g.mean_f64();
    let half = [0, 1, 2].map(|a| k_sigma * cov[(a, a)].sqrt());
    let mut lo = [0usize; 3];
    let mut hi = [0usize; 3];
    for a in 0..3 {
        // Voxel centers c satisfy |c − μ| ≤ half  ⇔  index range below.
        let qlo = ((mean[a] - half[a] - spec.origin[a]) / spec.res - 0.5).ceil();
        let qhi = ((mean[a] + half[a] - spec.origin[a]) / spec.res - 0.5).floor();
        if qhi < 0.0 || qlo > (spec.dims[a] - 1) as f64 || qlo > qhi {
            return None;
        }
        lo[a] = qlo.max(0.0) as usize;
        hi[a] = (qhi as usize).min(spec.dims[a] - 1);
    }
    Some(Prepared {
        mean: [mean.x, mean.y, mean.z],
        inv,
        opacity: g.opacity as f64,
        lo,
        hi,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use splat_types::{Aabb, Vec3, Vec3f};

    #[test]
    fn single_gaussian_matches_formula() {
        let spec = GridSpec::covering(&Aabb::new(Vec3::repeat(-1.0), Vec3::repeat(1.0)), 0.1);
        let g = Gaussian::isotropic(Vec3f::new(0.05, 0.05, 0.05), 0.2, 0.8);
        let d = DensityGrid::accumulate(spec, &[g], 3.0, 0.0);
        // Peak at the voxel whose center is the mean.
        let c = spec.cell_of(&Vec3::new(0.05, 0.05, 0.05)).unwrap();
        assert!((d.density[spec.index(c)] - 0.8).abs() < 1e-6);
        // One voxel over: σ² = 0.04 + 0.0025 (anti-aliasing).
        let n = spec.index([c[0] + 1, c[1], c[2]]);
        let expect = 0.8 * (-0.5 * 0.01 / 0.0425f64).exp();
        assert!((d.density[n] as f64 - expect).abs() < 1e-6);
        // Far away: zero (outside 3σ).
        assert_eq!(d.density[spec.index([0, 0, 0])], 0.0);
    }

    #[test]
    fn thin_disc_is_not_lost_between_voxel_centers() {
        // A disc 2 mm thick sitting exactly between two voxel layers.
        let spec = GridSpec::covering(&Aabb::new(Vec3::repeat(-0.5), Vec3::repeat(0.5)), 0.05);
        let mut gs = Vec::new();
        for i in -5..5 {
            for j in -5..5 {
                let mut g = Gaussian::isotropic(
                    Vec3f::new(i as f32 * 0.02, j as f32 * 0.02, 0.0),
                    0.01,
                    0.9,
                );
                g.scale.z = 0.002;
                gs.push(g);
            }
        }
        let d = DensityGrid::accumulate(spec, &gs, 3.0, 0.0);
        let c = spec.cell_of(&Vec3::new(0.0, 0.0, 0.01)).unwrap();
        assert!(d.density[spec.index(c)] > 0.2);
    }
}
