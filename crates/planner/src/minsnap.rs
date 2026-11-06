//! Minimum-snap piecewise polynomials (Mellinger & Kumar 2011; Richter et al. 2013).
//!
//! For each axis and segment `i` we fit a degree-7 polynomial and minimize
//! the total squared snap `Σ ∫ (d⁴p/dt⁴)² dt` subject to:
//!
//! - passing through the waypoints,
//! - zero velocity, acceleration and jerk at the start and the end,
//! - continuous velocity, acceleration and jerk at interior waypoints.
//!
//! This is an equality-constrained quadratic program, solved in closed form
//! through its KKT system. Each segment is parameterized in normalized time
//! `s = τ/T ∈ [0, 1]`, which keeps the matrix well conditioned whatever the
//! segment durations are; the coefficients are converted back to real time
//! at the end.

use splat_types::nalgebra::DMatrix;
use splat_types::{PolySegment, PolyTrajectory, Vec3};

/// Coefficients per segment and axis (degree 7).
const N: usize = 8;
/// Derivative orders constrained at the ends and kept continuous inside.
const CONT: usize = 3;

/// `k! / (k − n)!` (0 if `n > k`).
fn falling(k: usize, n: usize) -> f64 {
    if n > k {
        return 0.0;
    }
    ((k - n + 1)..=k).map(|v| v as f64).product()
}

/// Row vector of the `n`-th derivative (w.r.t. `s`) at `s ∈ {0, 1}`.
fn deriv_row(n: usize, at_end: bool) -> [f64; N] {
    let mut r = [0.0; N];
    for (k, v) in r.iter_mut().enumerate() {
        // At s = 1 every power contributes; at s = 0 only the s^n term does.
        *v = if at_end || k == n { falling(k, n) } else { 0.0 };
    }
    r
}

/// Errors from the solver.
#[derive(Debug, Clone, PartialEq)]
pub enum MinSnapError {
    /// Fewer than two waypoints, or durations/waypoints mismatched.
    BadInput(String),
    /// The KKT system could not be solved.
    Singular,
}

/// Solves the minimum-snap problem for `waypoints` with segment `durations`.
pub fn min_snap(waypoints: &[Vec3], durations: &[f64]) -> Result<PolyTrajectory, MinSnapError> {
    let m = durations.len();
    if waypoints.len() < 2 || waypoints.len() != m + 1 {
        return Err(MinSnapError::BadInput(format!(
            "{} waypoints, {} durations",
            waypoints.len(),
            m
        )));
    }
    if durations.iter().any(|t| !(t.is_finite() && *t > 0.0)) {
        return Err(MinSnapError::BadInput("durations must be positive".into()));
    }
    let nv = N * m;
    let nc = 2 * m + 2 * CONT + CONT * (m - 1);
    let dim = nv + nc;
    let mut kkt = DMatrix::<f64>::zeros(dim, dim);
    let mut rhs = DMatrix::<f64>::zeros(dim, 3);

    // Cost block: Q_jk = j!/(j−4)! · k!/(k−4)! / (j + k − 7), weighted by 1/T⁷.
    for (i, t) in durations.iter().enumerate() {
        let w = 1.0 / t.powi(7);
        for j in 4..N {
            for k in 4..N {
                let q = falling(j, 4) * falling(k, 4) / (j + k - 7) as f64;
                kkt[(i * N + j, i * N + k)] = q * w;
            }
        }
    }

    // Constraint rows A (placed at rows nv.., and transposed into columns).
    let mut row = nv;
    let put = |kkt: &mut DMatrix<f64>, row: usize, seg: usize, coeffs: &[f64; N], scale: f64| {
        for (k, c) in coeffs.iter().enumerate() {
            let v = c * scale;
            kkt[(row, seg * N + k)] += v;
            kkt[(seg * N + k, row)] += v;
        }
    };
    // Positions: start and end of every segment.
    for i in 0..m {
        put(&mut kkt, row, i, &deriv_row(0, false), 1.0);
        for a in 0..3 {
            rhs[(row, a)] = waypoints[i][a];
        }
        row += 1;
        put(&mut kkt, row, i, &deriv_row(0, true), 1.0);
        for a in 0..3 {
            rhs[(row, a)] = waypoints[i + 1][a];
        }
        row += 1;
    }
    // Rest at both ends: derivatives 1..=3 are zero (rhs already zero).
    for n in 1..=CONT {
        put(
            &mut kkt,
            row,
            0,
            &deriv_row(n, false),
            1.0 / durations[0].powi(n as i32),
        );
        row += 1;
        put(
            &mut kkt,
            row,
            m - 1,
            &deriv_row(n, true),
            1.0 / durations[m - 1].powi(n as i32),
        );
        row += 1;
    }
    // Continuity of derivatives 1..=3 at interior waypoints (in real time).
    for i in 0..m.saturating_sub(1) {
        for n in 1..=CONT {
            put(
                &mut kkt,
                row,
                i,
                &deriv_row(n, true),
                1.0 / durations[i].powi(n as i32),
            );
            put(
                &mut kkt,
                row,
                i + 1,
                &deriv_row(n, false),
                -1.0 / durations[i + 1].powi(n as i32),
            );
            row += 1;
        }
    }
    debug_assert_eq!(row, dim);

    let sol = kkt.lu().solve(&rhs).ok_or(MinSnapError::Singular)?;
    if sol.iter().any(|v| !v.is_finite()) {
        return Err(MinSnapError::Singular);
    }
    let segments = (0..m)
        .map(|i| {
            let t = durations[i];
            let mut coeffs = [[0.0; N]; 3];
            for (a, axis) in coeffs.iter_mut().enumerate() {
                for (k, c) in axis.iter_mut().enumerate() {
                    // p(τ) = Σ c_k (τ/T)^k  ⇒  real-time coefficient c_k / T^k.
                    *c = sol[(i * N + k, a)] / t.powi(k as i32);
                }
            }
            PolySegment {
                duration: t,
                coeffs,
            }
        })
        .collect();
    PolyTrajectory::new(segments).map_err(|e| MinSnapError::BadInput(e.to_string()))
}

/// Segment time for distance `len` under a trapezoidal velocity profile
/// that starts and ends at rest (the classic initial guess).
pub fn trapezoid_time(len: f64, v_max: f64, a_max: f64) -> f64 {
    if len >= v_max * v_max / a_max {
        len / v_max + v_max / a_max
    } else {
        2.0 * (len / a_max).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_continuity(tr: &PolyTrajectory) {
        for i in 1..tr.segments().len() {
            let prev = &tr.segments()[i - 1];
            let next = &tr.segments()[i];
            for n in 0..=3 {
                let a = prev.eval_derivative(prev.duration, n);
                let b = next.eval_derivative(0.0, n);
                assert!(
                    (a - b).norm() < 1e-6 * (1.0 + a.norm()),
                    "derivative {n} jumps at {i}: {a:?} vs {b:?}"
                );
            }
        }
    }

    #[test]
    fn interpolates_and_is_continuous_through_jerk() {
        let wps = [
            Vec3::zeros(),
            Vec3::new(1.0, 0.5, 0.0),
            Vec3::new(2.0, -0.5, 0.5),
            Vec3::new(4.0, 0.0, 1.0),
        ];
        let tr = min_snap(&wps, &[1.0, 1.5, 2.0]).unwrap();
        let mut t = 0.0;
        for (i, seg) in tr.segments().iter().enumerate() {
            assert!((tr.eval(t).pos - wps[i]).norm() < 1e-9);
            t += seg.duration;
        }
        assert!((tr.end() - wps[3]).norm() < 1e-9);
        check_continuity(&tr);
        // At rest at both ends.
        for s in [tr.eval(0.0), tr.eval(tr.duration())] {
            assert!(s.vel.norm() < 1e-9 && s.acc.norm() < 1e-9 && s.jerk.norm() < 1e-9);
        }
    }

    #[test]
    fn single_segment_and_long_durations_are_well_conditioned() {
        let wps = [
            Vec3::zeros(),
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(10.0, 10.0, 0.0),
        ];
        let tr = min_snap(&wps, &[30.0, 0.05]).unwrap();
        assert!((tr.end() - wps[2]).norm() < 1e-6);
        check_continuity(&tr);
        let one = min_snap(&wps[..2], &[2.0]).unwrap();
        assert!((one.eval(1.0).pos - Vec3::new(5.0, 0.0, 0.0)).norm() < 1e-9);
    }

    #[test]
    fn bad_inputs() {
        assert!(min_snap(&[Vec3::zeros()], &[]).is_err());
        assert!(min_snap(&[Vec3::zeros(), Vec3::x()], &[0.0]).is_err());
        assert!(min_snap(&[Vec3::zeros(), Vec3::x()], &[1.0, 1.0]).is_err());
    }

    #[test]
    fn trapezoid() {
        assert!((trapezoid_time(3.0, 1.5, 1.5) - 3.0).abs() < 1e-12);
        assert!((trapezoid_time(1.0, 1.5, 1.0) - 2.0).abs() < 1e-12);
    }
}
