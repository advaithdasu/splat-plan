//! Cheap, provable lower bound on the distance from a point to an ellipsoid.

use splat_types::nalgebra::Matrix3;
use splat_types::{Gaussian, Vec3};

/// Smallest semi-axis allowed. Degenerate (zero-thickness) Gaussians are
/// widened to this; a bigger ellipsoid contains the original one, so bounds
/// computed for it remain valid (just slightly looser).
const MIN_AXIS: f64 = 1e-9;

/// The `k`σ ellipsoid of a Gaussian, prepared for fast distance bounds.
#[derive(Debug, Clone, PartialEq)]
pub struct Ellipsoid {
    /// Center `μ`.
    pub center: Vec3,
    /// `Rᵀ` (world → local).
    pub world_to_local: Matrix3<f64>,
    /// Semi-axes `a = k·s`.
    pub axes: Vec3,
    /// Half extents of the world-space axis-aligned bounding box.
    pub aabb_half: Vec3,
    /// Index of the source Gaussian.
    pub source: usize,
}

impl Ellipsoid {
    /// The `k`σ ellipsoid of `g`.
    pub fn from_gaussian(g: &Gaussian, k: f64, source: usize) -> Self {
        let r = g.rot.cast::<f64>().to_rotation_matrix().into_inner();
        let axes = g.scale.cast::<f64>().map(|s| (k * s).max(MIN_AXIS));
        // AABB of a rotated ellipsoid: half extent_i = sqrt(Σ_j (R_ij a_j)²).
        let mut aabb_half = Vec3::zeros();
        for i in 0..3 {
            aabb_half[i] = (0..3)
                .map(|j| (r[(i, j)] * axes[j]).powi(2))
                .sum::<f64>()
                .sqrt();
        }
        Ellipsoid {
            center: g.mean_f64(),
            world_to_local: r.transpose(),
            axes,
            aabb_half,
            source,
        }
    }

    /// Lower bound on the Euclidean distance from `p` to the solid ellipsoid.
    ///
    /// Two valid bounds, the larger is used:
    ///
    /// 1. **Scaled bound**. With `q = Rᵀ(p − μ)`, `D = diag(a)`
    ///    and `m = ‖D⁻¹q‖`: for any point `x` on the ellipsoid `‖D⁻¹x‖ = 1`,
    ///    so `‖D⁻¹(q − x)‖ ≥ m − 1` (triangle inequality), and since
    ///    `‖D⁻¹v‖ ≤ ‖v‖ / a_min`, we get `‖q − x‖ ≥ a_min · (m − 1)`.
    /// 2. **Sphere bound**. The ellipsoid lies inside the ball of radius
    ///    `a_max` around `μ`, so `dist ≥ ‖q‖ − a_max`.
    ///
    /// Bound 1 is tight along the shortest axis; bound 2 is tight far away
    /// and along the longest axis (bound 1 alone is very loose sideways from
    /// a flat "disc" Gaussian). Negative values mean "possibly touching".
    pub fn lower_bound(&self, p: &Vec3) -> f64 {
        let q = self.world_to_local * (p - self.center);
        let m = q.component_div(&self.axes).norm();
        let scaled = self.axes.min() * (m - 1.0);
        let sphere = q.norm() - self.axes.max();
        scaled.max(sphere)
    }

    /// True if the segment `a → b` may touch the solid ellipsoid.
    ///
    /// Exact up to rounding, then erring towards "touches": in the scaled
    /// local frame (`q = D⁻¹Rᵀ(x − μ)`) the ellipsoid is the unit ball and
    /// the segment stays a segment, so the test is the closest point of a
    /// segment to the origin, compared against a slightly inflated radius.
    /// Used for line of sight, where "touches" means "blocks the view": the
    /// conservative answer.
    pub fn segment_may_hit(&self, a: &Vec3, b: &Vec3) -> bool {
        let qa = (self.world_to_local * (a - self.center)).component_div(&self.axes);
        let qb = (self.world_to_local * (b - self.center)).component_div(&self.axes);
        let d = qb - qa;
        let dd = d.norm_squared();
        let t = if dd > 0.0 {
            (-qa.dot(&d) / dd).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (qa + d * t).norm_squared() <= (1.0 + 1e-9) * (1.0 + 1e-9)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use splat_types::Vec3f;

    #[test]
    fn exact_on_axes_of_a_sphere() {
        let g = Gaussian::isotropic(Vec3f::new(1.0, 2.0, 3.0), 0.1, 1.0);
        let e = Ellipsoid::from_gaussian(&g, 3.0, 0);
        assert!((e.lower_bound(&Vec3::new(1.0, 2.0, 4.0)) - 0.7).abs() < 1e-6);
        assert!(e.lower_bound(&Vec3::new(1.0, 2.0, 3.0)) < 0.0);
        assert!((e.aabb_half - Vec3::repeat(0.3)).norm() < 1e-6);
    }

    #[test]
    fn segment_hits_match_geometry() {
        // Flat disc in the z = 0 plane, radius 0.3 (3σ of 0.1).
        let mut g = Gaussian::isotropic(Vec3f::zeros(), 0.1, 1.0);
        g.scale.z = 0.0;
        let e = Ellipsoid::from_gaussian(&g, 3.0, 0);
        let up = Vec3::new(0.0, 0.0, 1.0);
        let down = Vec3::new(0.0, 0.0, -1.0);
        assert!(e.segment_may_hit(&up, &down), "through the disc");
        assert!(
            !e.segment_may_hit(&up, &Vec3::new(0.0, 0.0, 0.01)),
            "stops short"
        );
        let off = Vec3::new(0.31, 0.0, 0.0);
        assert!(
            !e.segment_may_hit(&(off + up), &(off + down)),
            "beside the disc"
        );
        let on = Vec3::new(0.29, 0.0, 0.0);
        assert!(
            e.segment_may_hit(&(on + up), &(on + down)),
            "edge of the disc"
        );
        // Degenerate segment: a point.
        assert!(e.segment_may_hit(&Vec3::zeros(), &Vec3::zeros()));
        assert!(!e.segment_may_hit(&up, &up));
    }

    #[test]
    fn degenerate_scale_is_safe() {
        let mut g = Gaussian::isotropic(Vec3f::zeros(), 0.1, 1.0);
        g.scale.z = 0.0;
        let e = Ellipsoid::from_gaussian(&g, 3.0, 0);
        let b = e.lower_bound(&Vec3::new(0.0, 0.0, 1.0));
        assert!(b.is_finite() && b <= 1.0);
    }
}
