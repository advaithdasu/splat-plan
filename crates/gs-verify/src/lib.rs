//! Independent trajectory verifier that checks clearance against the raw Gaussians.
//!
//! **The planner never grades itself.** This crate depends on nothing in
//! the workspace except `splat-types`. It does not use the voxel
//! map, the distance field, or any planner code; it re-derives safety from
//! the raw Gaussians with a different method:
//!
//! - **Clearance**: each Gaussian above an opacity floor is treated as its
//!   solid `k`σ ellipsoid; a provable lower bound on the distance to the
//!   nearest one (see [`Ellipsoid::lower_bound`]) is computed with a BVH.
//! - **Sampling**: the trajectory is sampled every `dt`. Between samples the
//!   robot moves at most `v·dt/2` from the nearest sample, so each sample
//!   needs clearance `r + margin + v·dt/2`.
//! - **Support**: every sample must be within `max_support_dist` of a
//!   training camera (a different notion of "observed" than the map's ray
//!   casting, so the two cross-check each other).
//! - **Line of sight** (re-derived without voxels): the robot body at every
//!   sample (its center and six points at `robot_radius` along the axes) must
//!   be visible from at least `min_views` training cameras: inside the image
//!   and with no Gaussian ellipsoid on the segment from the camera. A camera
//!   that saw through a point is the only evidence that it is empty; the map
//!   proves the same with voxel ray casting, this check with exact
//!   segment–ellipsoid tests, so a bug in either is caught by the other.
//! - **Limits**: speed and acceleration within the configured maxima.
//!
//! The result is a [`Certificate`] (`certificate.json`).
#![forbid(unsafe_code)]

mod bound;
mod bvh;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use splat_types::{CameraPose, Gaussian, PolyTrajectory, Report, Scale, Vec3};

pub use bound::Ellipsoid;
pub use bvh::Bvh;

/// Verifier settings. Lengths in meters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifierConfig {
    /// Gaussians are solid out to this many standard deviations.
    pub k_sigma: f64,
    /// Gaussians with lower opacity are ignored.
    pub opacity_floor: f32,
    /// Sampling period (s).
    pub dt: f64,
    /// Robot radius (m).
    pub robot_radius: f64,
    /// Safety margin (m).
    pub margin: f64,
    /// Maximum allowed speed (m/s).
    pub max_speed: f64,
    /// Maximum allowed acceleration (m/s²).
    pub max_accel: f64,
    /// Every sample must be within this distance of a training camera (m).
    pub max_support_dist: f64,
    /// Every probe point of the robot body must be seen by at least this
    /// many training cameras (0 disables the line-of-sight check).
    pub min_views: usize,
}

impl Default for VerifierConfig {
    fn default() -> Self {
        VerifierConfig {
            k_sigma: 3.0,
            opacity_floor: 0.05,
            dt: 0.02,
            robot_radius: 0.15,
            margin: 0.10,
            max_speed: 1.5,
            max_accel: 1.5,
            max_support_dist: 6.0,
            min_views: 1,
        }
    }
}

/// Overall verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Verdict {
    /// Every check passed and distances are metric.
    Safe,
    /// At least one check failed.
    Unsafe,
    /// All checks passed but distances are not metric: never `Safe`.
    Unscaled,
}

/// Which check a violation failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ViolationReason {
    /// Too close to a Gaussian.
    Clearance,
    /// Too far from every training camera.
    Support,
    /// Speed or acceleration over the limit.
    Limits,
    /// Part of the robot body was not seen by enough training cameras.
    Unobserved,
}

/// One failed sample.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Violation {
    /// Time along the trajectory (s).
    pub t: f64,
    /// Position (scene units).
    pub position: [f64; 3],
    /// Failed check.
    pub reason: ViolationReason,
    /// Clearance lower bound at this sample (scene units).
    pub clearance: f64,
    /// Center of the Gaussian that produced the clearance bound (scene units).
    /// Lets a planner place a keep-out at the obstacle rather than at itself.
    pub obstacle: Option<[f64; 3]>,
}

/// Settings recorded in the certificate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CertificateConfig {
    /// See [`VerifierConfig::k_sigma`].
    pub k_sigma: f64,
    /// See [`VerifierConfig::opacity_floor`].
    pub opacity_floor: f32,
    /// See [`VerifierConfig::dt`].
    pub dt: f64,
    /// See [`VerifierConfig::robot_radius`].
    pub robot_radius: f64,
    /// See [`VerifierConfig::margin`].
    pub margin: f64,
    /// See [`VerifierConfig::max_support_dist`].
    pub max_support_dist: f64,
    /// See [`VerifierConfig::min_views`]. Absent in certificates written
    /// before the check existed (read as 0: not checked).
    #[serde(default)]
    pub min_views: usize,
    /// Gaussians actually checked (above the opacity floor).
    pub gaussians_checked: usize,
}

/// Maximum violations listed in a certificate (the count is always exact).
pub const MAX_LISTED_VIOLATIONS: usize = 500;

/// The verifier's output (`certificate.json`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Certificate {
    /// Overall verdict.
    pub verdict: Verdict,
    /// Whether distances are in meters.
    pub metric: bool,
    /// Smallest clearance lower bound over all samples (m; clamped at 0).
    pub min_clearance_m: f64,
    /// Clearance each sample needed: `r + margin + v·dt/2` (m).
    pub margin_required_m: f64,
    /// Number of samples checked.
    pub samples: usize,
    /// Maximum sampled speed (m/s).
    pub max_speed: f64,
    /// Maximum sampled acceleration (m/s²).
    pub max_accel: f64,
    /// True if every sample passed the support check.
    pub support_ok: bool,
    /// True if every sample passed the line-of-sight check (false when read
    /// from a certificate written before the check existed).
    #[serde(default)]
    pub observed_ok: bool,
    /// Total number of violating samples.
    pub violation_count: usize,
    /// The first [`MAX_LISTED_VIOLATIONS`] violations.
    pub violations: Vec<Violation>,
    /// Settings used.
    pub config: CertificateConfig,
    /// Which splat file the Gaussians came from (and whether it was
    /// cleaned). The verifier only sees Gaussians, so the caller fills this in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub splat: Option<String>,
}

impl Report for Certificate {}

/// Per-sample results, computed in parallel then combined in order.
struct SampleCheck {
    t: f64,
    pos: Vec3,
    clearance: f64,
    obstacle: Option<usize>,
    support_dist: f64,
    observed: bool,
    speed: f64,
    accel: f64,
}

/// The verifier: a BVH over the scene's Gaussians plus camera positions.
#[derive(Debug, Clone)]
pub struct Verifier {
    cfg: VerifierConfig,
    bvh: Bvh,
    /// Gaussian centers by source index (for violation reports).
    centers: Vec<Vec3>,
    cameras: Vec<Vec3>,
    /// Full poses, for the line-of-sight check.
    poses: Vec<CameraPose>,
    scale: Scale,
    unit: f64,
}

impl Verifier {
    /// Prepares a verifier.
    ///
    /// `units_per_meter` converts the config's meters into scene units. It
    /// must be 1 for metric scenes (it is forced to 1 when `scale` is
    /// metric); for unscaled scenes it is the caller's guess, and the verdict
    /// can never be `Safe` anyway.
    pub fn new(
        gaussians: &[Gaussian],
        cameras: &[CameraPose],
        scale: Scale,
        units_per_meter: f64,
        cfg: VerifierConfig,
    ) -> Self {
        let items: Vec<Ellipsoid> = gaussians
            .iter()
            .enumerate()
            .filter(|(_, g)| g.opacity >= cfg.opacity_floor)
            .map(|(i, g)| Ellipsoid::from_gaussian(g, cfg.k_sigma, i))
            .collect();
        let unit = if scale.is_metric() || !(units_per_meter > 0.0 && units_per_meter.is_finite()) {
            1.0
        } else {
            units_per_meter
        };
        Verifier {
            bvh: Bvh::build(items),
            centers: gaussians.iter().map(|g| g.mean_f64()).collect(),
            cameras: cameras.iter().map(|c| c.center()).collect(),
            poses: cameras.to_vec(),
            scale,
            unit,
            cfg,
        }
    }

    /// The configuration.
    pub fn config(&self) -> &VerifierConfig {
        &self.cfg
    }

    /// Lower bound on the distance from `p` to any Gaussian (scene units),
    /// with the index of the closest Gaussian.
    pub fn clearance_lower_bound(&self, p: &Vec3) -> (f64, Option<usize>) {
        self.bvh.nearest_lower_bound(p)
    }

    /// Distance from `p` to the nearest training camera (scene units).
    pub fn support_distance(&self, p: &Vec3) -> f64 {
        self.cameras
            .iter()
            .map(|c| (c - p).norm())
            .fold(f64::INFINITY, f64::min)
    }

    /// True if no ellipsoid touches the segment `from → to` (the
    /// line-of-sight test used by [`Verifier::views`], without the image test).
    pub fn line_of_sight(&self, from: &Vec3, to: &Vec3) -> bool {
        !self.bvh.segment_blocked(from, to)
    }

    /// Number of training cameras that see `p` (stops counting at `limit`):
    /// `p` projects inside the image and no ellipsoid touches the segment
    /// from the camera center to `p`.
    pub fn views(&self, p: &Vec3, limit: usize) -> usize {
        let mut n = 0;
        for cam in &self.poses {
            if n >= limit {
                break;
            }
            let Some((u, v, _)) = cam.project(p) else {
                continue;
            };
            let k = &cam.intrinsics;
            if !(u >= 0.0 && v >= 0.0 && u < f64::from(k.width) && v < f64::from(k.height)) {
                continue;
            }
            if self.line_of_sight(&cam.center(), p) {
                n += 1;
            }
        }
        n
    }

    /// True if the robot body at `p` passes the line-of-sight check: the
    /// center and the six points at `robot_radius` along ±x, ±y, ±z are each
    /// seen by at least `min_views` cameras.
    pub fn body_observed(&self, p: &Vec3) -> bool {
        let need = self.cfg.min_views;
        if need == 0 {
            return true;
        }
        let r = self.cfg.robot_radius * self.unit;
        let mut probes = [*p; 7];
        for a in 0..3 {
            probes[1 + 2 * a][a] += r;
            probes[2 + 2 * a][a] -= r;
        }
        probes.iter().all(|q| self.views(q, need) >= need)
    }

    /// Checks a trajectory and issues a certificate.
    pub fn certify(&self, traj: &PolyTrajectory) -> Certificate {
        let cfg = &self.cfg;
        let u = self.unit;
        let samples = traj.sample(cfg.dt);
        // par_iter().map().collect() preserves order: deterministic.
        let checks: Vec<SampleCheck> = samples
            .par_iter()
            .map(|s| {
                let (clearance, obstacle) = self.bvh.nearest_lower_bound(&s.pos);
                SampleCheck {
                    t: s.t,
                    pos: s.pos,
                    clearance,
                    obstacle,
                    support_dist: self.support_distance(&s.pos),
                    observed: self.body_observed(&s.pos),
                    speed: s.vel.norm(),
                    accel: s.acc.norm(),
                }
            })
            .collect();

        let max_speed = checks.iter().map(|c| c.speed).fold(0.0, f64::max);
        let max_accel = checks.iter().map(|c| c.accel).fold(0.0, f64::max);
        // Movement between samples is bounded by the larger of the allowed
        // and the observed speed (so a too-fast trajectory cannot sneak
        // through between samples).
        let v_bound = (cfg.max_speed * u).max(max_speed);
        let required = (cfg.robot_radius + cfg.margin) * u + v_bound * cfg.dt / 2.0;
        // Tiny relative slack for limits: time scaling targets the limit exactly.
        let slack = 1.0 + 1e-6;

        let centers = &self.centers;
        let mut violations = Vec::new();
        let mut count = 0usize;
        let mut support_ok = true;
        let mut observed_ok = true;
        let mut min_clear = f64::INFINITY;
        for c in &checks {
            min_clear = min_clear.min(c.clearance);
            let mut push = |reason| {
                count += 1;
                if violations.len() < MAX_LISTED_VIOLATIONS {
                    violations.push(Violation {
                        t: c.t,
                        position: c.pos.into(),
                        reason,
                        clearance: c.clearance,
                        obstacle: c
                            .obstacle
                            .and_then(|i| centers.get(i))
                            .map(|p| [p.x, p.y, p.z]),
                    });
                }
            };
            if c.clearance < required {
                push(ViolationReason::Clearance);
            }
            if c.support_dist > cfg.max_support_dist * u {
                support_ok = false;
                push(ViolationReason::Support);
            }
            if !c.observed {
                observed_ok = false;
                push(ViolationReason::Unobserved);
            }
            if c.speed > cfg.max_speed * u * slack || c.accel > cfg.max_accel * u * slack {
                push(ViolationReason::Limits);
            }
        }
        let verdict = if count > 0 || checks.is_empty() {
            Verdict::Unsafe
        } else if !self.scale.is_metric() {
            Verdict::Unscaled
        } else {
            Verdict::Safe
        };
        Certificate {
            verdict,
            metric: self.scale.is_metric(),
            min_clearance_m: min_clear.max(0.0) / u,
            margin_required_m: required / u,
            samples: checks.len(),
            max_speed: max_speed / u,
            max_accel: max_accel / u,
            support_ok,
            observed_ok,
            violation_count: count,
            violations,
            config: CertificateConfig {
                k_sigma: cfg.k_sigma,
                opacity_floor: cfg.opacity_floor,
                dt: cfg.dt,
                robot_radius: cfg.robot_radius,
                margin: cfg.margin,
                max_support_dist: cfg.max_support_dist,
                min_views: cfg.min_views,
                gaussians_checked: self.bvh.len(),
            },
            splat: None,
        }
    }
}
