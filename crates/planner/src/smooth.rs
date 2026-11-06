//! Turning a safe polyline into a smooth, dynamically feasible trajectory.

use splat_types::{PolyTrajectory, Vec3};

use crate::minsnap::{min_snap, trapezoid_time};
use crate::workspace::Workspace;
use crate::{PlanError, PlannerConfig};

/// Maximum rounds of "insert a waypoint where the curve strays, re-solve".
const MAX_SUBDIVISIONS: usize = 12;
/// Trajectories are timed to reach this fraction of the limits (headroom
/// for sampling between checks).
const LIMIT_TARGET: f64 = 0.97;

/// Smooths `polyline` (whose straight segments are already clear) into a
/// minimum-snap trajectory that
///
/// - stays within the speed/acceleration limits (uniform time scaling), and
/// - keeps `(robot_radius + margin)` of conservative map clearance.
///
/// Minimum-snap curves bulge away from the straight segments. Wherever the
/// curve loses clearance, the offending segment gets its midpoint inserted as
/// a new waypoint (Richter et al.), which pulls the curve back toward the
/// verified-clear straight line. This repeats a bounded number of times; if
/// the curve is still not clear the planner refuses.
pub fn smooth(
    ws: &Workspace,
    polyline: &[Vec3],
    cfg: &PlannerConfig,
    speed_factor: f64,
) -> Result<PolyTrajectory, PlanError> {
    let unit = ws.map.length_unit;
    let v_max = cfg.max_speed * unit * speed_factor;
    let a_max = cfg.max_accel * unit * speed_factor;
    let need = (cfg.robot_radius + cfg.margin) * unit;
    let mut wps: Vec<Vec3> = polyline.to_vec();
    if wps.len() < 2 || (wps[0] - wps[wps.len() - 1]).norm() < 1e-9 && wps.len() == 2 {
        return Err(PlanError::Degenerate);
    }
    for _ in 0..MAX_SUBDIVISIONS {
        let durations: Vec<f64> = wps
            .windows(2)
            .map(|w| trapezoid_time((w[1] - w[0]).norm(), v_max, a_max).max(1e-3))
            .collect();
        let traj = time_scaled(&wps, durations, v_max, a_max)?;
        let bad = straying_segments(ws, &traj, need, v_max);
        if bad.is_empty() {
            return Ok(traj);
        }
        // Insert midpoints of offending segments (back to front keeps indices valid).
        for &i in bad.iter().rev() {
            let mid = (wps[i] + wps[i + 1]) * 0.5;
            wps.insert(i + 1, mid);
        }
    }
    Err(PlanError::Trajectory(format!(
        "smoothed trajectory still violates map clearance after {MAX_SUBDIVISIONS} subdivisions"
    )))
}

/// Solves min-snap, then uniformly rescales time so the peak speed and
/// acceleration sit at `LIMIT_TARGET` of the limits. Uniform scaling of all
/// durations by `k` is exact for min-snap: the new optimum is the old curve
/// played `k` times slower, so speeds scale by `1/k` and accelerations by `1/k²`.
fn time_scaled(
    wps: &[Vec3],
    mut durations: Vec<f64>,
    v_max: f64,
    a_max: f64,
) -> Result<PolyTrajectory, PlanError> {
    let solve = |d: &[f64]| min_snap(wps, d).map_err(|e| PlanError::Trajectory(format!("{e:?}")));
    let mut traj = solve(&durations)?;
    for _ in 0..4 {
        let dt = (traj.duration() / 4000.0).max(1e-4);
        let (v, a) = traj.max_speed_accel(dt);
        let k = (v / (LIMIT_TARGET * v_max)).max((a / (LIMIT_TARGET * a_max)).sqrt());
        if !(k.is_finite() && k > 0.0) {
            return Err(PlanError::Trajectory("non-finite time scale".into()));
        }
        if (k - 1.0).abs() < 0.005 {
            break;
        }
        for d in &mut durations {
            *d *= k;
        }
        traj = solve(&durations)?;
    }
    // Final guarantee: never return a trajectory over the limits.
    let (v, a) = traj.max_speed_accel((traj.duration() / 4000.0).max(1e-4));
    if v > v_max || a > a_max {
        return Err(PlanError::Trajectory(format!(
            "time scaling did not converge (v {v:.3}/{v_max:.3}, a {a:.3}/{a_max:.3})"
        )));
    }
    Ok(traj)
}

/// Indices of segments whose curve comes closer than `need` to obstacles.
///
/// Samples are taken so consecutive positions are at most `δ` apart (using
/// the speed bound `v_max`), and each sample must have `need + δ/2` of
/// clearance, so the whole curve is covered.
fn straying_segments(ws: &Workspace, traj: &PolyTrajectory, need: f64, v_max: f64) -> Vec<usize> {
    let delta = ws.map.spec.res * 0.2;
    let dt = delta / v_max;
    let mut bad = Vec::new();
    for (i, seg) in traj.segments().iter().enumerate() {
        let n = ((seg.duration / dt).ceil() as usize).max(1);
        let ok = (0..=n).all(|k| {
            let p = seg.eval_derivative(seg.duration * k as f64 / n as f64, 0);
            ws.clearance(&p) >= need + delta / 2.0
        });
        if !ok {
            bad.push(i);
        }
    }
    bad
}
