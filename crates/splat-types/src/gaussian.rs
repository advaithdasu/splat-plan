use nalgebra::{Matrix3, UnitQuaternion, Vector3};

use crate::{Vec3, Vec3f};

/// Zeroth-order spherical-harmonic constant: `rgb = 0.5 + SH_C0 · f_dc`.
pub const SH_C0: f32 = 0.282_094_8;

/// One 3D Gaussian splat primitive, in "activated" (human-meaningful) units.
///
/// The PLY file stores log-scales and logit-opacity; the loader converts them
/// so that nothing downstream has to remember to call `exp`/`sigmoid`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gaussian {
    /// Center of the Gaussian.
    pub mean: Vec3f,
    /// Standard deviations along the local axes (linear, not log).
    pub scale: Vec3f,
    /// Rotation from local axes to world axes (unit quaternion).
    pub rot: UnitQuaternion<f32>,
    /// Opacity in `0..=1` (already passed through the sigmoid).
    pub opacity: f32,
    /// Degree-0 spherical-harmonic coefficients (the "DC" color term).
    pub sh_dc: [f32; 3],
}

impl Gaussian {
    /// An axis-aligned Gaussian; handy for tests and synthetic scenes.
    pub fn isotropic(mean: Vec3f, sigma: f32, opacity: f32) -> Self {
        Gaussian {
            mean,
            scale: Vector3::repeat(sigma),
            rot: UnitQuaternion::identity(),
            opacity,
            sh_dc: [0.0; 3],
        }
    }

    /// Mean as `f64`.
    pub fn mean_f64(&self) -> Vec3 {
        self.mean.cast::<f64>()
    }

    /// Covariance `Σ = R · diag(s²) · Rᵀ` in `f64`.
    pub fn covariance(&self) -> Matrix3<f64> {
        let r = self.rot.cast::<f64>().to_rotation_matrix().into_inner();
        let s = self.scale.cast::<f64>();
        let d = Matrix3::from_diagonal(&s.component_mul(&s));
        r * d * r.transpose()
    }

    /// Display color from the DC term, clamped to `0..=1`.
    pub fn color_rgb(&self) -> [f32; 3] {
        self.sh_dc.map(|c| (0.5 + SH_C0 * c).clamp(0.0, 1.0))
    }

    /// Sets the DC term so that [`Gaussian::color_rgb`] returns `rgb`.
    pub fn with_color(mut self, rgb: [f32; 3]) -> Self {
        self.sh_dc = rgb.map(|c| (c - 0.5) / SH_C0);
        self
    }

    /// Half-extents of the axis-aligned box containing the `k`-sigma ellipsoid.
    ///
    /// For an ellipsoid `xᵀ Σ⁻¹ x ≤ k²` the AABB half-extent along axis `i` is
    /// `k · sqrt(Σ_ii)`.
    pub fn aabb_half_extent(&self, k: f64) -> Vec3 {
        let c = self.covariance();
        Vec3::new(c[(0, 0)].sqrt(), c[(1, 1)].sqrt(), c[(2, 2)].sqrt()) * k
    }

    /// True if every field is finite and the scales/opacity are in range.
    pub fn is_valid(&self) -> bool {
        self.mean.iter().all(|v| v.is_finite())
            && self.scale.iter().all(|v| v.is_finite() && *v >= 0.0)
            && self.rot.coords.iter().all(|v| v.is_finite())
            && self.opacity.is_finite()
            && (0.0..=1.0).contains(&self.opacity)
            && self.sh_dc.iter().all(|v| v.is_finite())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covariance_of_rotated_gaussian() {
        // Scale 2 along local x, 1 elsewhere; rotate 90° about z, so world y
        // should carry the variance 4.
        let g = Gaussian {
            mean: Vec3f::zeros(),
            scale: Vec3f::new(2.0, 1.0, 1.0),
            rot: UnitQuaternion::from_axis_angle(&Vector3::z_axis(), std::f32::consts::FRAC_PI_2),
            opacity: 1.0,
            sh_dc: [0.0; 3],
        };
        let c = g.covariance();
        assert!((c[(0, 0)] - 1.0).abs() < 1e-6);
        assert!((c[(1, 1)] - 4.0).abs() < 1e-6);
        assert!((c[(2, 2)] - 1.0).abs() < 1e-6);
        let h = g.aabb_half_extent(3.0);
        assert!((h.y - 6.0).abs() < 1e-5);
    }

    #[test]
    fn color_roundtrip() {
        let g = Gaussian::isotropic(Vec3f::zeros(), 1.0, 1.0).with_color([0.2, 0.5, 0.9]);
        let c = g.color_rgb();
        for (a, b) in c.iter().zip([0.2, 0.5, 0.9]) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
