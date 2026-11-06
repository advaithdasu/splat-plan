//! Calmer flying along a path that is already certified.
//!
//! Minimum-snap timing in [`crate::smooth`] times every polyline segment as
//! if the drone started and stopped on it, so the finished trajectory
//! surges and crawls (peak speed 1.45 m/s, dipping to 0.3 m/s between).
//! [`calm`] takes a certified trajectory, keeps its route (optionally
//! rounding the corners a little) and flies it with one cruise speed, an
//! even ramp up and down at the ends, and no surges. Each candidate goes
//! back to the independent verifier; the original stays unless a candidate
//! certifies and is measurably smoother.

use gs_map::Map;
use gs_verify::Verifier;
use splat_types::{PolyTrajectory, Vec3};

use crate::minsnap::min_snap;
use crate::repair::PlanOutcome;
use crate::{PlanError, PlannerConfig};

/// Distance between resampled route points (m).
const ROUTE_STEP_M: f64 = 0.05;
/// Distance between the waypoints min-snap is asked to pass (m). Short
/// enough that the curve stays on the route, long enough to stay smooth.
const KNOT_M: f64 = 0.4;
/// Fraction of the speed and acceleration limits a retimed path may use.
const LIMIT_TARGET: f64 = 0.9;
/// Ramp acceleration, as a fraction of the acceleration limit.
const RAMP_FRACTION: f64 = 0.35;

/// Cruise speeds (m/s) and corner-rounding passes tried.
const CRUISE_MPS: [f64; 3] = [0.7, 0.9, 1.1];
const ROUNDING_PASSES: [usize; 8] = [0, 20, 40, 80, 160, 320, 640, 1280];
/// A calmer path may stray at most this far (m) from the certified route.
const MAX_DRIFT_M: f64 = 0.35;

/// Dimensionless jerk, `T⁵/L² ∫|jerk|² dt`: lower is smoother, and it does
/// not reward going slowly or short.
pub fn dimensionless_jerk(traj: &PolyTrajectory) -> f64 {
    let dt = 0.01;
    let s = traj.sample(dt);
    if s.len() < 4 {
        return f64::INFINITY;
    }
    let t = traj.duration();
    let len = traj.length(dt);
    let int: f64 = s
        .windows(2)
        .map(|w| ((w[1].acc - w[0].acc) / dt).norm_squared() * dt)
        .sum();
    t.powi(5) / (len * len) * int
}

/// The route of `traj` as points `step` apart (arc length), end points kept.
fn resample(traj: &PolyTrajectory, step: f64) -> Vec<Vec3> {
    let dense: Vec<Vec3> = traj
        .sample((traj.duration() / 20_000.0).max(1e-4))
        .iter()
        .map(|s| s.pos)
        .collect();
    let mut out = vec![dense[0]];
    let mut carried = 0.0;
    for w in dense.windows(2) {
        let seg = (w[1] - w[0]).norm();
        if seg <= 0.0 {
            continue;
        }
        let mut at = step - carried;
        while at <= seg {
            out.push(w[0] + (w[1] - w[0]) * (at / seg));
            at += step;
        }
        carried = seg - (at - step);
    }
    let last = dense[dense.len() - 1];
    if (out[out.len() - 1] - last).norm() < step * 0.5 && out.len() > 1 {
        let n = out.len();
        out[n - 1] = last;
    } else {
        out.push(last);
    }
    out
}

/// Farthest any point of `traj` is from the nearest point of `route`.
fn drift(traj: &PolyTrajectory, route: &[Vec3]) -> f64 {
    traj.sample(0.05)
        .iter()
        .map(|s| {
            route
                .iter()
                .map(|r| (r - s.pos).norm())
                .fold(f64::INFINITY, f64::min)
        })
        .fold(0.0, f64::max)
}

/// Rounds corners: each interior point moves halfway toward the average of
/// its neighbors, `passes` times. The end points stay put.
fn round_corners(mut pts: Vec<Vec3>, passes: usize) -> Vec<Vec3> {
    for _ in 0..passes {
        let prev = pts.clone();
        for i in 1..pts.len() - 1 {
            pts[i] = prev[i] * 0.5 + (prev[i - 1] + prev[i + 1]) * 0.25;
        }
    }
    pts
}

/// Time to have covered arc length `s` of a route of length `len`: speed
/// ramps up at constant acceleration `a` to `v`, cruises, and ramps down.
fn time_at(s: f64, len: f64, v: f64, a: f64) -> f64 {
    let ramp = v * v / (2.0 * a);
    let (v, ramp) = if 2.0 * ramp > len {
        // Too short to reach the cruise speed.
        ((a * len).sqrt(), len / 2.0)
    } else {
        (v, ramp)
    };
    let t_ramp = v / a;
    let up = |x: f64| (2.0 * x / a).sqrt();
    let total = 2.0 * t_ramp + (len - 2.0 * ramp) / v;
    if s <= ramp {
        up(s)
    } else if s <= len - ramp {
        t_ramp + (s - ramp) / v
    } else {
        total - up((len - s).max(0.0))
    }
}

/// `route` flown at `cruise_mps` along its whole length.
fn fly(
    route: &[Vec3],
    cruise: f64,
    cfg: &PlannerConfig,
    unit: f64,
) -> Result<PolyTrajectory, PlanError> {
    let mut arc = vec![0.0];
    for w in route.windows(2) {
        arc.push(arc[arc.len() - 1] + (w[1] - w[0]).norm());
    }
    let len = arc[arc.len() - 1];
    // Knots about KNOT_M apart, always including both ends.
    let every = ((KNOT_M * unit) / (ROUTE_STEP_M * unit)).round().max(1.0) as usize;
    let mut idx: Vec<usize> = (0..route.len()).step_by(every).collect();
    if idx.last() != Some(&(route.len() - 1)) {
        if route.len() - 1 - idx[idx.len() - 1] < every / 2 && idx.len() > 1 {
            let n = idx.len();
            idx[n - 1] = route.len() - 1;
        } else {
            idx.push(route.len() - 1);
        }
    }
    let v_max = cfg.max_speed * unit * LIMIT_TARGET;
    let a_max = cfg.max_accel * unit * LIMIT_TARGET;
    let v = (cruise * unit).min(v_max);
    let a = (cfg.max_accel * RAMP_FRACTION * unit).min(a_max);
    let knots: Vec<Vec3> = idx.iter().map(|&i| route[i]).collect();
    let mut dur: Vec<f64> = idx
        .windows(2)
        .map(|w| (time_at(arc[w[1]], len, v, a) - time_at(arc[w[0]], len, v, a)).max(1e-3))
        .collect();
    let solve = |d: &[f64]| {
        min_snap(&knots, d).map_err(|e| PlanError::Trajectory(format!("retime: {e:?}")))
    };
    let mut traj = solve(&dur)?;
    // Slow down (never speed up) until the limits hold. Scaling every
    // duration by k scales speed by 1/k and acceleration by 1/k².
    for _ in 0..6 {
        let (vm, am) = traj.max_speed_accel((traj.duration() / 4000.0).max(1e-4));
        let k = (vm / v_max).max((am / a_max).sqrt());
        if k <= 1.0 {
            return Ok(traj);
        }
        for d in &mut dur {
            *d *= k * 1.01;
        }
        traj = solve(&dur)?;
    }
    Err(PlanError::Trajectory("retime: limits not reached".into()))
}

/// Replaces a certified trajectory with a calmer one along the same route,
/// if the verifier certifies one that is smoother. Returns the input
/// unchanged otherwise.
pub fn calm(
    map: &Map,
    verifier: &Verifier,
    outcome: PlanOutcome,
    cfg: &PlannerConfig,
) -> PlanOutcome {
    let unit = map.length_unit;
    let base = dimensionless_jerk(&outcome.trajectory);
    let route = resample(&outcome.trajectory, ROUTE_STEP_M * unit);
    let mut best: Option<(f64, PolyTrajectory, gs_verify::Certificate)> = None;
    for &passes in &ROUNDING_PASSES {
        let rounded = round_corners(route.clone(), passes);
        for &cruise in &CRUISE_MPS {
            let Ok(traj) = fly(&rounded, cruise, cfg, unit) else {
                continue;
            };
            let score = dimensionless_jerk(&traj);
            if best.as_ref().is_some_and(|b| score >= b.0) {
                continue;
            }
            if drift(&traj, &route) > MAX_DRIFT_M * unit {
                continue;
            }
            let cert = verifier.certify(&traj);
            if cert.violation_count == 0 {
                tracing::debug!(passes, cruise, score, "calmer candidate certified");
                best = Some((score, traj, cert));
            }
        }
    }
    match best {
        Some((score, trajectory, certificate)) if score < base => PlanOutcome {
            trajectory,
            certificate,
            ..outcome
        },
        _ => outcome,
    }
}
