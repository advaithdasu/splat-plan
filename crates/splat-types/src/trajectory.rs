use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::Vec3;

/// Number of coefficients of a degree-7 polynomial.
pub const N_COEFFS: usize = 8;

/// Errors constructing a [`PolyTrajectory`].
#[derive(Debug, Error, PartialEq)]
pub enum TrajectoryError {
    /// No segments were given.
    #[error("trajectory has no segments")]
    Empty,
    /// A segment duration was zero, negative or not finite.
    #[error("segment {index} has invalid duration {duration}")]
    BadDuration {
        /// Segment index.
        index: usize,
        /// The offending duration.
        duration: f64,
    },
    /// A coefficient was NaN or infinite.
    #[error("segment {index} has a non-finite coefficient")]
    NonFinite {
        /// Segment index.
        index: usize,
    },
}

/// Position and its first three derivatives at one instant.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryState {
    /// Time since the trajectory start (s).
    pub t: f64,
    /// Position.
    pub pos: Vec3,
    /// Velocity.
    pub vel: Vec3,
    /// Acceleration.
    pub acc: Vec3,
    /// Jerk.
    pub jerk: Vec3,
}

/// One polynomial piece. For axis `a` and local time `τ ∈ [0, duration]`,
/// `p_a(τ) = Σ_k coeffs[a][k] · τ^k`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolySegment {
    /// Segment duration in seconds.
    pub duration: f64,
    /// Coefficients per axis (x, y, z), lowest degree first.
    pub coeffs: [[f64; N_COEFFS]; 3],
}

impl PolySegment {
    /// A straight-line segment from `a` to `b` at constant velocity.
    pub fn linear(a: Vec3, b: Vec3, duration: f64) -> Self {
        let mut coeffs = [[0.0; N_COEFFS]; 3];
        for axis in 0..3 {
            coeffs[axis][0] = a[axis];
            coeffs[axis][1] = (b[axis] - a[axis]) / duration;
        }
        PolySegment { duration, coeffs }
    }

    /// Evaluates derivative `order` (0 = position) of every axis at local time `tau`.
    pub fn eval_derivative(&self, tau: f64, order: usize) -> Vec3 {
        let mut out = Vec3::zeros();
        for axis in 0..3 {
            out[axis] = eval_poly_derivative(&self.coeffs[axis], tau, order);
        }
        out
    }
}

/// Evaluates the `order`-th derivative of `Σ c_k τ^k` at `tau` (Horner's rule).
pub fn eval_poly_derivative(c: &[f64; N_COEFFS], tau: f64, order: usize) -> f64 {
    if order >= N_COEFFS {
        return 0.0;
    }
    let mut acc = 0.0;
    for k in (order..N_COEFFS).rev() {
        // d^order/dτ^order τ^k = k!/(k-order)! τ^(k-order)
        let mut falling = 1.0;
        for j in 0..order {
            falling *= (k - j) as f64;
        }
        acc = acc * tau + c[k] * falling;
    }
    acc
}

/// A piecewise degree-7 polynomial trajectory, continuous by construction of
/// whoever built it (the planner tests continuity through jerk).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolyTrajectory {
    segments: Vec<PolySegment>,
    /// Cumulative start time of each segment (same length as `segments`).
    starts: Vec<f64>,
}

impl PolyTrajectory {
    /// Builds a trajectory, validating durations and coefficients.
    pub fn new(segments: Vec<PolySegment>) -> Result<Self, TrajectoryError> {
        if segments.is_empty() {
            return Err(TrajectoryError::Empty);
        }
        let mut starts = Vec::with_capacity(segments.len());
        let mut t = 0.0;
        for (index, s) in segments.iter().enumerate() {
            if !(s.duration.is_finite() && s.duration > 0.0) {
                return Err(TrajectoryError::BadDuration {
                    index,
                    duration: s.duration,
                });
            }
            if s.coeffs.iter().flatten().any(|c| !c.is_finite()) {
                return Err(TrajectoryError::NonFinite { index });
            }
            starts.push(t);
            t += s.duration;
        }
        Ok(PolyTrajectory { segments, starts })
    }

    /// Straight-line, constant-speed trajectory through `points` (used in tests
    /// and as a fallback representation of a polyline).
    pub fn polyline(points: &[Vec3], speed: f64) -> Result<Self, TrajectoryError> {
        let segs = points
            .windows(2)
            .map(|w| {
                let len = (w[1] - w[0]).norm();
                PolySegment::linear(w[0], w[1], len / speed)
            })
            .collect();
        Self::new(segs)
    }

    /// The segments.
    pub fn segments(&self) -> &[PolySegment] {
        &self.segments
    }

    /// Total duration in seconds.
    pub fn duration(&self) -> f64 {
        match (self.starts.last(), self.segments.last()) {
            (Some(s), Some(seg)) => s + seg.duration,
            _ => 0.0,
        }
    }

    /// Start time of segment `i`.
    pub fn segment_start(&self, i: usize) -> Option<f64> {
        self.starts.get(i).copied()
    }

    /// Finds the segment containing `t` (clamped to `[0, duration]`) and the local time.
    fn locate(&self, t: f64) -> (usize, f64) {
        let t = t.clamp(0.0, self.duration());
        // Index of the last segment whose start is <= t.
        let i = self.starts.partition_point(|s| *s <= t).saturating_sub(1);
        let tau = (t - self.starts[i]).min(self.segments[i].duration);
        (i, tau)
    }

    /// Evaluates the full state at time `t` (clamped to the trajectory's time range).
    pub fn eval(&self, t: f64) -> TrajectoryState {
        let (i, tau) = self.locate(t);
        let s = &self.segments[i];
        TrajectoryState {
            t: t.clamp(0.0, self.duration()),
            pos: s.eval_derivative(tau, 0),
            vel: s.eval_derivative(tau, 1),
            acc: s.eval_derivative(tau, 2),
            jerk: s.eval_derivative(tau, 3),
        }
    }

    /// Start position.
    pub fn start(&self) -> Vec3 {
        self.eval(0.0).pos
    }

    /// End position.
    pub fn end(&self) -> Vec3 {
        self.eval(self.duration()).pos
    }

    /// Samples at uniform `dt`, always including the final instant.
    /// Returns an empty vector if `dt` is not positive.
    pub fn sample(&self, dt: f64) -> Vec<TrajectoryState> {
        if !(dt.is_finite() && dt > 0.0) {
            return Vec::new();
        }
        let total = self.duration();
        let n = (total / dt).ceil() as usize;
        let mut out = Vec::with_capacity(n + 1);
        for k in 0..n {
            out.push(self.eval(k as f64 * dt));
        }
        out.push(self.eval(total));
        out
    }

    /// Maximum speed and acceleration magnitude over samples at `dt`.
    pub fn max_speed_accel(&self, dt: f64) -> (f64, f64) {
        self.sample(dt)
            .iter()
            .fold((0.0_f64, 0.0_f64), |(v, a), s| {
                (v.max(s.vel.norm()), a.max(s.acc.norm()))
            })
    }

    /// Polyline length estimated from samples at `dt`.
    pub fn length(&self, dt: f64) -> f64 {
        self.sample(dt)
            .windows(2)
            .map(|w| (w[1].pos - w[0].pos).norm())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derivative_of_known_polynomial() {
        // p(τ) = 1 + 2τ + 3τ² + τ⁷
        let c = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let t: f64 = 1.5;
        assert!((eval_poly_derivative(&c, t, 0) - (1.0 + 3.0 + 6.75 + t.powi(7))).abs() < 1e-9);
        assert!((eval_poly_derivative(&c, t, 1) - (2.0 + 6.0 * t + 7.0 * t.powi(6))).abs() < 1e-9);
        assert!((eval_poly_derivative(&c, t, 2) - (6.0 + 42.0 * t.powi(5))).abs() < 1e-9);
        assert!((eval_poly_derivative(&c, t, 3) - 210.0 * t.powi(4)).abs() < 1e-9);
        assert_eq!(eval_poly_derivative(&c, t, 8), 0.0);
    }

    #[test]
    fn polyline_eval_and_sampling() {
        let pts = [
            Vec3::zeros(),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 2.0, 0.0),
        ];
        let tr = PolyTrajectory::polyline(&pts, 1.0).unwrap();
        assert!((tr.duration() - 4.0).abs() < 1e-12);
        assert!((tr.eval(1.0).pos - Vec3::new(1.0, 0.0, 0.0)).norm() < 1e-12);
        assert!((tr.eval(3.0).pos - Vec3::new(2.0, 1.0, 0.0)).norm() < 1e-12);
        assert!((tr.end() - pts[2]).norm() < 1e-12);
        // Clamping beyond the ends.
        assert!((tr.eval(-1.0).pos - pts[0]).norm() < 1e-12);
        assert!((tr.eval(99.0).pos - pts[2]).norm() < 1e-12);
        let s = tr.sample(0.3);
        assert!((s.last().unwrap().t - 4.0).abs() < 1e-12);
        let (v, a) = tr.max_speed_accel(0.1);
        assert!((v - 1.0).abs() < 1e-9 && a.abs() < 1e-9);
        assert!((tr.length(0.01) - 4.0).abs() < 1e-9);
        assert!(tr.sample(0.0).is_empty());
    }

    #[test]
    fn rejects_bad_segments() {
        assert_eq!(PolyTrajectory::new(vec![]), Err(TrajectoryError::Empty));
        let bad = PolySegment::linear(Vec3::zeros(), Vec3::x(), 0.0);
        assert!(matches!(
            PolyTrajectory::new(vec![bad]),
            Err(TrajectoryError::BadDuration { .. })
        ));
        let mut nan = PolySegment::linear(Vec3::zeros(), Vec3::x(), 1.0);
        nan.coeffs[1][3] = f64::NAN;
        assert!(matches!(
            PolyTrajectory::new(vec![nan]),
            Err(TrajectoryError::NonFinite { .. })
        ));
    }

    #[test]
    fn json_roundtrip() {
        let tr = PolyTrajectory::polyline(&[Vec3::zeros(), Vec3::x()], 2.0).unwrap();
        let s = serde_json::to_string(&tr).unwrap();
        let back: PolyTrajectory = serde_json::from_str(&s).unwrap();
        assert_eq!(tr, back);
    }
}
