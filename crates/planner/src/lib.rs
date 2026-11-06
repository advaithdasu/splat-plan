//! Motion planner: A*, shortcutting, minimum-snap smoothing and a verify-repair loop.
//!
//! Stages:
//!
//! 1. [`astar`]: shortest 26-connected voxel path through observed-free
//!    space with clearance (unknown voxels are never entered).
//! 2. [`shortcut`]: greedy line-of-sight pruning, every segment checked by
//!    conservative distance-field sampling.
//! 3. [`smooth`]: minimum-snap trajectory through the pruned waypoints,
//!    time-scaled to the speed/acceleration limits, subdivided until it
//!    keeps its clearance.
//! 4. [`plan_and_certify`]: hands the trajectory to the independent verifier
//!    (`gs-verify`); on rejection, the violation *positions* come back as
//!    keep-out spheres and planning repeats, a bounded number of times.
//!
//! 5. [`standoff_orbit`]: a closed inspection loop at a fixed distance from
//!    the largest object, certified against the scene and the object.
//!
//! Every failure is an `Err(PlanError)` with a diagnostic: refuse rather
//! than guess. The planner never decides on its own that a
//! trajectory is safe; only the verifier's certificate does.
#![forbid(unsafe_code)]

mod astar;
mod minsnap;
mod orbit;
mod repair;
mod retime;
mod shortcut;
mod smooth;
mod workspace;

use gs_map::{Map, VoxelState};
use serde::{Deserialize, Serialize};
use splat_types::{PolyTrajectory, Vec3};
use thiserror::Error;

pub use astar::{astar, cell_ok};
pub use minsnap::{MinSnapError, min_snap, trapezoid_time};
pub use orbit::{OrbitConfig, OrbitPlan, SkippedRay, largest_object, standoff_orbit};
pub use repair::{PlanOutcome, plan_and_certify, plan_and_certify_via};
pub use retime::{calm, dimensionless_jerk};
pub use shortcut::{segment_clear, shortcut};
pub use smooth::smooth;
pub use workspace::{ExtraField, KeepOut, Workspace};

/// Why the planner refused.
#[derive(Debug, Clone, Error, PartialEq)]
pub enum PlanError {
    /// A query point lies outside the mapped volume.
    #[error("{which} is outside the map")]
    OutsideMap {
        /// `"start"` or `"goal"`.
        which: &'static str,
    },
    /// A query point is in unknown or occupied space, or too close to it.
    #[error(
        "{which} is not traversable: state {state:?}, clearance {clearance:.3} < required {required:.3}"
    )]
    NotTraversable {
        /// `"start"` or `"goal"`.
        which: &'static str,
        /// Voxel state at the point.
        state: VoxelState,
        /// Conservative clearance at the point.
        clearance: f64,
        /// Required clearance.
        required: f64,
    },
    /// No path exists (or none was found within the limits).
    #[error("no path: {reason} ({expanded} nodes expanded)")]
    NoPath {
        /// Explanation.
        reason: String,
        /// A* expansions performed.
        expanded: usize,
    },
    /// Start and goal coincide.
    #[error("start and goal are the same point")]
    Degenerate,
    /// Smoothing failed.
    #[error("trajectory generation failed: {0}")]
    Trajectory(String),
    /// The verifier kept rejecting the plan.
    #[error(
        "verifier rejected all {attempts} attempts (last: {violations} violating samples, first reason {first_reason})"
    )]
    VerifierRejected {
        /// Attempts made.
        attempts: usize,
        /// Violations in the last certificate.
        violations: usize,
        /// Reason of the first violation in the last certificate.
        first_reason: String,
    },
    /// The configuration is invalid.
    #[error("invalid planner config: {0}")]
    BadConfig(String),
}

impl PlanError {
    /// Short machine-readable category, used to tally refusal reasons.
    pub fn kind(&self) -> &'static str {
        match self {
            PlanError::OutsideMap { .. } => "outside_map",
            PlanError::NotTraversable {
                state: VoxelState::Unknown,
                ..
            } => "endpoint_unknown",
            PlanError::NotTraversable {
                state: VoxelState::Occupied,
                ..
            } => "endpoint_occupied",
            PlanError::NotTraversable { .. } => "endpoint_too_close",
            PlanError::NoPath { .. } => "no_path",
            PlanError::Degenerate => "degenerate",
            PlanError::Trajectory(_) => "trajectory",
            PlanError::VerifierRejected { .. } => "verifier_rejected",
            PlanError::BadConfig(_) => "bad_config",
        }
    }
}

/// Planner settings. Lengths are meters (converted with `Map::length_unit`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlannerConfig {
    /// Robot radius (m).
    pub robot_radius: f64,
    /// Safety margin on top of the radius (m).
    pub margin: f64,
    /// Extra clearance used while searching, so later smoothing has slack (m).
    pub plan_buffer: f64,
    /// Maximum speed (m/s).
    pub max_speed: f64,
    /// Maximum acceleration (m/s²).
    pub max_accel: f64,
    /// A* expansion limit.
    pub max_expansions: usize,
    /// Verify-repair rounds after the first attempt.
    pub max_repairs: usize,
    /// Radius of keep-out spheres placed at verifier violations (m).
    pub keepout_radius: f64,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        PlannerConfig {
            robot_radius: 0.15,
            margin: 0.10,
            plan_buffer: 0.05,
            max_speed: 1.5,
            max_accel: 1.5,
            max_expansions: 5_000_000,
            max_repairs: 4,
            keepout_radius: 0.05,
        }
    }
}

impl PlannerConfig {
    /// Clearance (scene units) every point of a plan must have in the map.
    pub fn search_clearance(&self, map: &Map) -> f64 {
        (self.robot_radius + self.margin + self.plan_buffer) * map.length_unit
    }

    fn validate(&self) -> Result<(), PlanError> {
        let vals = [
            self.robot_radius,
            self.margin,
            self.plan_buffer,
            self.max_speed,
            self.max_accel,
            self.keepout_radius,
        ];
        if vals.iter().any(|v| !v.is_finite() || *v < 0.0)
            || self.max_speed <= 0.0
            || self.max_accel <= 0.0
        {
            return Err(PlanError::BadConfig(format!("{self:?}")));
        }
        Ok(())
    }
}

/// Checks that a query point is usable, with a precise diagnostic if not.
pub fn check_endpoint(
    ws: &Workspace,
    p: &Vec3,
    required: f64,
    which: &'static str,
) -> Result<(), PlanError> {
    let map = ws.map;
    if map.spec.cell_of(p).is_none() {
        return Err(PlanError::OutsideMap { which });
    }
    let state = map.state_at(p);
    let clearance = ws.clearance(p);
    // One voxel of slack: see `shortcut::segment_clear` and `astar::cell_ok`.
    if state != VoxelState::FreeObserved || clearance < required + map.spec.res {
        return Err(PlanError::NotTraversable {
            which,
            state,
            clearance,
            required: required + map.spec.res,
        });
    }
    Ok(())
}

/// Plans a polyline in a workspace (map + keep-outs). Every segment of the
/// result is clear with [`PlannerConfig::search_clearance`].
pub fn plan_path_in(
    ws: &Workspace,
    start: &Vec3,
    goal: &Vec3,
    cfg: &PlannerConfig,
) -> Result<Vec<Vec3>, PlanError> {
    cfg.validate()?;
    if (start - goal).norm() < 1e-9 {
        return Err(PlanError::Degenerate);
    }
    let need = cfg.search_clearance(ws.map);
    check_endpoint(ws, start, need, "start")?;
    check_endpoint(ws, goal, need, "goal")?;
    // A* runs one voxel stricter than the segment check, so consecutive
    // voxel centers always pass it (see `segment_clear`).
    let cells = astar(ws, start, goal, need + ws.map.spec.res, cfg.max_expansions)?;
    let mut raw = Vec::with_capacity(cells.len() + 2);
    raw.push(*start);
    raw.extend(cells);
    raw.push(*goal);
    raw.dedup_by(|a, b| (*a - *b).norm() < 1e-12);
    shortcut(ws, &raw, need).ok_or_else(|| PlanError::NoPath {
        reason: "voxel path failed the conservative segment check".into(),
        expanded: 0,
    })
}

/// Plans a collision-free polyline from `start` to `goal` on `map`.
pub fn plan_path(
    map: &Map,
    start: &Vec3,
    goal: &Vec3,
    cfg: &PlannerConfig,
) -> Result<Vec<Vec3>, PlanError> {
    plan_path_in(&Workspace::new(map), start, goal, cfg)
}

/// Plans a smooth trajectory (not yet verified; use [`plan_and_certify`]).
pub fn plan(
    map: &Map,
    start: &Vec3,
    goal: &Vec3,
    cfg: &PlannerConfig,
) -> Result<PolyTrajectory, PlanError> {
    let ws = Workspace::new(map);
    let poly = plan_path_in(&ws, start, goal, cfg)?;
    smooth(&ws, &poly, cfg, 1.0)
}
