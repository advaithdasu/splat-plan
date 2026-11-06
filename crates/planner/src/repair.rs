//! The verify-repair loop.

use gs_map::Map;
use gs_verify::{Certificate, Verifier, ViolationReason};
use serde::Serialize;
use splat_types::{PolyTrajectory, Vec3};

use crate::smooth::smooth;
use crate::workspace::{KeepOut, Workspace};
use crate::{PlanError, PlannerConfig, plan_path_in};

/// Plans one polyline through every point of `via` in order: each leg is
/// an ordinary [`plan_path_in`] query (A* plus shortcutting within the leg),
/// so every via point stays on the path.
fn plan_via_in(ws: &Workspace, via: &[Vec3], cfg: &PlannerConfig) -> Result<Vec<Vec3>, PlanError> {
    let mut poly: Vec<Vec3> = Vec::new();
    for w in via.windows(2) {
        let leg = plan_path_in(ws, &w[0], &w[1], cfg)?;
        let skip = usize::from(!poly.is_empty());
        poly.extend(leg.into_iter().skip(skip));
    }
    Ok(poly)
}

/// A certified plan.
#[derive(Debug, Clone, Serialize)]
pub struct PlanOutcome {
    /// The trajectory the verifier accepted.
    pub trajectory: PolyTrajectory,
    /// The verifier's certificate (verdict `SAFE`, or `UNSCALED` for unscaled scenes).
    pub certificate: Certificate,
    /// The waypoint polyline the trajectory was built from.
    pub waypoints: Vec<Vec3>,
    /// Plan/verify rounds used (1 = accepted first time).
    pub attempts: usize,
    /// Keep-out spheres added from verifier feedback.
    pub keepouts: Vec<KeepOut>,
}

/// Plans, then asks the independent verifier; repairs on rejection.
///
/// The verifier's only influence on the planner is *data*: for each
/// clearance violation, the position of the offending Gaussian and the
/// clearance it left become a keep-out sphere around that Gaussian; support
/// violations become keep-outs at the sample; limit violations lower the
/// speed. After `cfg.max_repairs` failed repairs the planner refuses with
/// [`PlanError::VerifierRejected`]; it never returns a rejected trajectory.
///
/// Once the verifier has rejected an attempt, a failure to re-plan around
/// the new keep-outs (for example a keep-out covering the start) is also
/// reported as [`PlanError::VerifierRejected`]: the verifier's verdict is the
/// root cause, and reporting the keep-out as "start not traversable" would
/// misattribute the refusal.
pub fn plan_and_certify(
    map: &Map,
    verifier: &Verifier,
    start: &Vec3,
    goal: &Vec3,
    cfg: &PlannerConfig,
) -> Result<PlanOutcome, PlanError> {
    plan_and_certify_via(map, verifier, &[*start, *goal], cfg)
}

/// [`plan_and_certify`] through a sequence of via points: one trajectory
/// that passes every via point in order without stopping (the minimum-snap
/// curve is smooth through interior waypoints). Keep-outs from a rejection
/// apply to every leg when the whole route is re-planned.
pub fn plan_and_certify_via(
    map: &Map,
    verifier: &Verifier,
    via: &[Vec3],
    cfg: &PlannerConfig,
) -> Result<PlanOutcome, PlanError> {
    if via.len() < 2 {
        return Err(PlanError::Degenerate);
    }
    let mut ws = Workspace::new(map);
    let unit = map.length_unit;
    let radius = cfg.keepout_radius * unit;
    // Violations closer than this to an existing keep-out are redundant.
    let spacing = (cfg.robot_radius + cfg.margin) * unit * 0.25;
    let mut speed_factor = 1.0;
    let mut last: Option<Certificate> = None;
    for attempt in 1..=cfg.max_repairs + 1 {
        let replanned = plan_via_in(&ws, via, cfg)
            .and_then(|poly| smooth(&ws, &poly, cfg, speed_factor).map(|t| (poly, t)));
        let (poly, traj) = match (replanned, &last) {
            (Ok(x), _) => x,
            (Err(e), None) => return Err(e),
            (Err(e), Some(c)) => {
                tracing::debug!(attempt, error = %e, "re-plan after rejection failed");
                return Err(rejected(attempt - 1, c));
            }
        };
        let cert = verifier.certify(&traj);
        if cert.violation_count == 0 {
            return Ok(PlanOutcome {
                trajectory: traj,
                certificate: cert,
                waypoints: poly,
                attempts: attempt,
                keepouts: ws.keepouts,
            });
        }
        tracing::debug!(
            attempt,
            violations = cert.violation_count,
            "verifier rejected; repairing"
        );
        if cert
            .violations
            .iter()
            .any(|v| v.reason == ViolationReason::Limits)
        {
            speed_factor *= 0.9;
        }
        for v in &cert.violations {
            let x = Vec3::from(v.position);
            let keepout = match (v.reason, v.obstacle) {
                (ViolationReason::Limits, _) => continue,
                // The verifier saw an obstacle centered at `o` whose surface is
                // about `|x − o| − clearance` from its center in this direction:
                // model it as a sphere of that radius.
                (ViolationReason::Clearance, Some(o)) => {
                    let o = Vec3::from(o);
                    KeepOut {
                        center: o,
                        radius: ((x - o).norm() - v.clearance).max(radius),
                    }
                }
                // No obstacle reported (support or line-of-sight check): keep
                // away from the sample.
                _ => KeepOut { center: x, radius },
            };
            // A violation at an obstacle that already has a keep-out means the
            // sphere was too small in this direction (a big Gaussian reaches
            // farther one way than another): grow it instead of ignoring
            // the violation, or the next plan is the same plan.
            match ws
                .keepouts
                .iter_mut()
                .find(|k| (k.center - keepout.center).norm() <= spacing)
            {
                Some(k) => k.radius = k.radius.max(keepout.radius),
                None => ws.keepouts.push(keepout),
            }
        }
        last = Some(cert);
    }
    match &last {
        Some(c) => Err(rejected(cfg.max_repairs + 1, c)),
        // Unreachable: the loop runs at least once and either returns or sets `last`.
        None => Err(PlanError::VerifierRejected {
            attempts: 0,
            violations: 0,
            first_reason: String::new(),
        }),
    }
}

/// The refusal for a plan whose last certificate was `cert`.
fn rejected(attempts: usize, cert: &Certificate) -> PlanError {
    PlanError::VerifierRejected {
        attempts,
        violations: cert.violation_count,
        first_reason: cert
            .violations
            .first()
            .map(|v| format!("{:?}", v.reason).to_lowercase())
            .unwrap_or_default(),
    }
}
