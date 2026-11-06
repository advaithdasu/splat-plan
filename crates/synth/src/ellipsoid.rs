//! Numerically exact distance from a point to a solid ellipsoid.
//!
//! Used as ground truth ("oracle") for random Gaussian fields and to check
//! the verifier's cheap lower bound. Follows D. Eberly, "Distance from a
//! Point to an Ellipse, an Ellipsoid, or a Hyperellipsoid" (bisection on the
//! Lagrange multiplier), which is robust for every configuration.

use splat_types::Vec3;

/// Distance from `q` (already in the ellipsoid's local frame, centered at
/// the origin) to the solid ellipsoid with semi-axes `a`. Zero inside.
///
/// All semi-axes must be positive.
pub fn distance_to_ellipsoid_local(q: &Vec3, a: &Vec3) -> f64 {
    // Work in the first octant: the problem is symmetric in each axis' sign.
    let y = q.abs();
    let inside = (y.x / a.x).powi(2) + (y.y / a.y).powi(2) + (y.z / a.z).powi(2);
    if inside <= 1.0 {
        return 0.0;
    }
    // Closest point x_i = a_i² y_i / (t + a_i²) for the unique root t > 0 of
    // F(t) = Σ (a_i y_i / (t + a_i²))² − 1, which is strictly decreasing for t ≥ 0.
    let f = |t: f64| -> f64 {
        (0..3)
            .map(|i| (a[i] * y[i] / (t + a[i] * a[i])).powi(2))
            .sum::<f64>()
            - 1.0
    };
    let mut lo = 0.0;
    // F(‖a∘y‖) ≤ 0, so the root is in [0, ‖a∘y‖].
    let mut hi = a.component_mul(&y).norm();
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break; // converged to machine precision
        }
        if f(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = 0.5 * (lo + hi);
    let x = Vec3::new(
        a.x * a.x * y.x / (t + a.x * a.x),
        a.y * a.y * y.y / (t + a.y * a.y),
        a.z * a.z * y.z / (t + a.z * a.z),
    );
    (y - x).norm()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn sphere_and_axis_cases() {
        let a = Vec3::new(1.0, 1.0, 1.0);
        assert!((distance_to_ellipsoid_local(&Vec3::new(3.0, 0.0, 0.0), &a) - 2.0).abs() < 1e-12);
        let b = Vec3::new(3.0, 2.0, 1.0);
        assert!((distance_to_ellipsoid_local(&Vec3::new(0.0, 0.0, 5.0), &b) - 4.0).abs() < 1e-12);
        assert!((distance_to_ellipsoid_local(&Vec3::new(-7.0, 0.0, 0.0), &b) - 4.0).abs() < 1e-12);
        assert_eq!(
            distance_to_ellipsoid_local(&Vec3::new(0.5, 0.5, 0.1), &b),
            0.0
        );
    }

    proptest! {
        /// The returned distance is attained (closest point lies on the surface)
        /// and no surface sample is closer.
        #[test]
        fn matches_dense_surface_sampling(
            a in prop::array::uniform3(0.05f64..3.0),
            q in prop::array::uniform3(-5.0f64..5.0),
        ) {
            let a = Vec3::new(a[0], a[1], a[2]);
            let q = Vec3::new(q[0], q[1], q[2]);
            let d = distance_to_ellipsoid_local(&q, &a);
            // Dense parametric sampling gives an upper bound on the true distance.
            let mut best = f64::INFINITY;
            let n = 120;
            for i in 0..=n {
                let th = std::f64::consts::PI * i as f64 / n as f64;
                for j in 0..(2 * n) {
                    let ph = std::f64::consts::PI * j as f64 / n as f64;
                    let s = Vec3::new(a.x * th.sin() * ph.cos(), a.y * th.sin() * ph.sin(), a.z * th.cos());
                    best = best.min((q - s).norm());
                }
            }
            let inside = (q.x / a.x).powi(2) + (q.y / a.y).powi(2) + (q.z / a.z).powi(2) <= 1.0;
            if inside {
                prop_assert_eq!(d, 0.0);
            } else {
                prop_assert!(d <= best + 1e-9, "exact {} > sampled {}", d, best);
                // Sampling resolution: arc step ≈ π·a_max/n.
                prop_assert!(best - d <= std::f64::consts::PI * a.max() / n as f64 + 1e-9);
            }
        }
    }
}
