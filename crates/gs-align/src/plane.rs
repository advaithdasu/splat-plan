//! RANSAC ground-plane fit with a camera up-vector prior.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use splat_types::nalgebra::{Matrix3, SymmetricEigen};
use splat_types::{CameraPose, Vec3};

use crate::{AlignConfig, AlignError};

/// A plane `n·x = d` with unit normal `n`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Plane {
    /// Unit normal.
    pub normal: Vec3,
    /// Offset: `normal · x = offset` on the plane.
    pub offset: f64,
}

impl Plane {
    /// Signed distance (positive on the normal's side).
    pub fn signed_distance(&self, p: &Vec3) -> f64 {
        self.normal.dot(p) - self.offset
    }

    /// Plane through three points, or `None` if they are (nearly) collinear.
    fn through(a: &Vec3, b: &Vec3, c: &Vec3) -> Option<Plane> {
        let n = (b - a).cross(&(c - a)).try_normalize(1e-12)?;
        Some(Plane {
            normal: n,
            offset: n.dot(a),
        })
    }

    /// Least-squares plane (smallest-eigenvalue direction of the covariance).
    fn fit(points: &[Vec3]) -> Option<Plane> {
        if points.len() < 3 {
            return None;
        }
        let n = points.len() as f64;
        let mean: Vec3 = points.iter().sum::<Vec3>() / n;
        let mut cov = Matrix3::zeros();
        for p in points {
            let d = p - mean;
            cov += d * d.transpose();
        }
        let eig = SymmetricEigen::new(cov / n);
        let i = eig.eigenvalues.imin();
        let normal = eig
            .eigenvectors
            .column(i)
            .into_owned()
            .try_normalize(1e-12)?;
        Some(Plane {
            normal,
            offset: normal.dot(&mean),
        })
    }
}

/// The plane with its normal flipped, if needed, to point along `up`.
fn oriented(plane: Plane, up: &Vec3) -> Plane {
    if plane.normal.dot(up) < 0.0 {
        Plane {
            normal: -plane.normal,
            offset: -plane.offset,
        }
    } else {
        plane
    }
}

/// True if at least half of the cameras are above `plane` (normal pointing
/// up). A floor is below the cameras that filmed it; a ceiling or the top of
/// a tall shelf is above them. With no cameras every plane passes.
fn below_cameras(plane: &Plane, cameras: &[CameraPose]) -> bool {
    let above = cameras
        .iter()
        .filter(|c| plane.signed_distance(&c.center()) > 0.0)
        .count();
    2 * above >= cameras.len()
}

/// Fits the ground plane. Returns the plane (normal oriented along the
/// prior, so the cameras are above it) and the inlier ratio on the sampled
/// points.
///
/// Candidate planes must be within `max_prior_angle_deg` of the prior and
/// below at least half of the cameras. The prior only fixes the axis up to
/// sign, so without the camera test a ceiling (often the largest flat
/// surface indoors, with blank walls and floors around it) can win and turn
/// the scene upside down.
pub fn fit_ground_plane(
    means: &[Vec3],
    prior: &Vec3,
    cameras: &[CameraPose],
    cfg: &AlignConfig,
) -> Result<(Plane, f64), AlignError> {
    if means.len() < 3 {
        return Err(AlignError::TooFewPoints(means.len()));
    }
    // Deterministic subsample (every k-th point) keeps RANSAC fast.
    let step = means.len().div_ceil(cfg.max_points).max(1);
    let pts: Vec<Vec3> = means.iter().step_by(step).copied().collect();

    // Scale-free inlier threshold: a fraction of the robust diagonal.
    let diag = robust_diagonal(&pts);
    let thr = (cfg.inlier_fraction * diag).max(1e-12);
    let cos_max = cfg.max_prior_angle_deg.to_radians().cos();

    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    let mut best: Option<(Plane, usize)> = None;
    for _ in 0..cfg.ransac_iters {
        let i = rng.gen_range(0..pts.len());
        let j = rng.gen_range(0..pts.len());
        let k = rng.gen_range(0..pts.len());
        let Some(plane) = Plane::through(&pts[i], &pts[j], &pts[k]) else {
            continue;
        };
        let plane = oriented(plane, prior);
        if plane.normal.dot(prior) < cos_max || !below_cameras(&plane, cameras) {
            continue;
        }
        let count = pts
            .iter()
            .filter(|p| plane.signed_distance(p).abs() <= thr)
            .count();
        // Strictly greater: ties keep the earlier candidate (deterministic).
        if best.as_ref().is_none_or(|(_, c)| count > *c) {
            best = Some((plane, count));
        }
    }
    let ratio_of = |c: usize| c as f64 / pts.len() as f64;
    let (plane, count) = best.ok_or(AlignError::NoGroundPlane {
        max_angle_deg: cfg.max_prior_angle_deg,
        best_ratio: 0.0,
    })?;
    if ratio_of(count) < cfg.min_inlier_ratio {
        return Err(AlignError::NoGroundPlane {
            max_angle_deg: cfg.max_prior_angle_deg,
            best_ratio: ratio_of(count),
        });
    }

    // Refit on inliers, twice, tightening onto the plane.
    let mut plane = plane;
    for _ in 0..2 {
        let inliers: Vec<Vec3> = pts
            .iter()
            .filter(|p| plane.signed_distance(p).abs() <= thr)
            .copied()
            .collect();
        if let Some(p) = Plane::fit(&inliers) {
            plane = oriented(p, prior);
        }
    }
    if plane.normal.dot(prior) < cos_max || !below_cameras(&plane, cameras) {
        return Err(AlignError::NoGroundPlane {
            max_angle_deg: cfg.max_prior_angle_deg,
            best_ratio: ratio_of(count),
        });
    }
    let final_count = pts
        .iter()
        .filter(|p| plane.signed_distance(p).abs() <= thr)
        .count();
    Ok((plane, ratio_of(final_count)))
}

/// Diagonal of the 1st-99th percentile box of the points.
fn robust_diagonal(pts: &[Vec3]) -> f64 {
    let mut d2 = 0.0;
    for axis in 0..3 {
        let mut v: Vec<f64> = pts.iter().map(|p| p[axis]).collect();
        v.sort_by(f64::total_cmp);
        let lo = v[(v.len() - 1) / 100];
        let hi = v[(v.len() - 1) * 99 / 100];
        d2 += (hi - lo) * (hi - lo);
    }
    d2.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_recovers_tilted_plane_and_rejects_off_prior() {
        // A tilted plane plus a vertical wall with more points.
        let n = Vec3::new(0.1, 0.0, 1.0).normalize();
        let mut pts = Vec::new();
        for i in 0..40 {
            for j in 0..40 {
                let p = Vec3::new(i as f64 * 0.1, j as f64 * 0.1, 0.0);
                pts.push(p - n * (n.dot(&p) - 0.5));
            }
        }
        for i in 0..60 {
            for j in 0..60 {
                pts.push(Vec3::new(10.0, i as f64 * 0.1, j as f64 * 0.1));
            }
        }
        let cfg = AlignConfig::default();
        let (plane, _) = fit_ground_plane(&pts, &Vec3::z(), &[], &cfg).unwrap();
        assert!(plane.normal.angle(&n) < 1e-6);
        assert!((plane.offset - 0.5).abs() < 1e-9);
        // With an x-axis prior the wall wins.
        let (wall, _) = fit_ground_plane(&pts, &Vec3::x(), &[], &cfg).unwrap();
        assert!(wall.normal.angle(&Vec3::x()) < 1e-6);
        assert!(matches!(
            fit_ground_plane(&pts[..2], &Vec3::z(), &[], &cfg),
            Err(AlignError::TooFewPoints(2))
        ));
    }

    /// Indoors the ceiling can be the biggest flat surface. With the cameras
    /// between floor and ceiling, the floor must win.
    #[test]
    fn ceiling_above_the_cameras_is_not_ground() {
        let mut pts = Vec::new();
        for i in 0..40 {
            for j in 0..40 {
                pts.push(Vec3::new(i as f64 * 0.1, j as f64 * 0.1, 0.0));
            }
        }
        // Ceiling at 2.5 with twice the points of the floor.
        for i in 0..80 {
            for j in 0..40 {
                pts.push(Vec3::new(i as f64 * 0.05, j as f64 * 0.1, 2.5));
            }
        }
        let cams: Vec<CameraPose> = (0..5)
            .map(|i| {
                CameraPose::look_at(
                    format!("{i}"),
                    Vec3::new(1.0 + i as f64 * 0.5, 2.0, 1.4),
                    Vec3::new(1.0 + i as f64 * 0.5, 4.0, 1.4),
                    Vec3::z(),
                    splat_types::PinholeIntrinsics::from_fov_x(640, 480, 1.2),
                )
                .unwrap()
            })
            .collect();
        let cfg = AlignConfig::default();
        let (plane, _) = fit_ground_plane(&pts, &Vec3::z(), &cams, &cfg).unwrap();
        assert!(plane.normal.angle(&Vec3::z()) < 1e-6);
        assert!(plane.offset.abs() < 1e-9, "offset {}", plane.offset);
    }
}
