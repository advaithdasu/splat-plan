//! Shared geometry types for SplatPlan.
//!
//! Every other crate in the workspace speaks in these types, so they are kept
//! small and plain:
//!
//! - [`Gaussian`]: one splat primitive (mean, linear scales, rotation, opacity, color).
//! - [`CameraPose`] / [`PinholeIntrinsics`]: COLMAP-convention cameras.
//! - [`Scale`]: whether coordinates are in meters.
//! - [`Similarity`]: `x' = s·R·x + t`, applied consistently to points, Gaussians and cameras.
//! - [`PolyTrajectory`]: piecewise degree-7 polynomial trajectory.
//! - [`Aabb`]: axis-aligned box.
//! - [`Report`]: anything that produces a JSON metrics report.
//!
//! Conventions: meters, right-handed, +Z up after alignment. `f32` for bulk
//! storage, `f64` for trajectory and verification math.
#![forbid(unsafe_code)]

mod aabb;
mod camera;
mod gaussian;
mod report;
mod scale;
mod similarity;
mod trajectory;

pub use aabb::Aabb;
pub use camera::{CameraPose, PinholeIntrinsics};
pub use gaussian::{Gaussian, SH_C0};
pub use report::Report;
pub use scale::Scale;
pub use similarity::Similarity;
pub use trajectory::{PolySegment, PolyTrajectory, TrajectoryError, TrajectoryState};

/// Re-exported so downstream crates use the exact same `nalgebra` version.
pub use nalgebra;

/// 3D vector in `f64` (trajectory / verifier math).
pub type Vec3 = nalgebra::Vector3<f64>;
/// 3D vector in `f32` (bulk storage).
pub type Vec3f = nalgebra::Vector3<f32>;

/// A scene: Gaussians, the cameras they were trained from, and the scale state.
///
/// This is the unit that flows through the pipeline (load → align → map → plan → verify).
#[derive(Debug, Clone, PartialEq)]
pub struct Scene {
    /// The splat primitives.
    pub gaussians: Vec<Gaussian>,
    /// Training cameras (world-to-camera poses plus intrinsics).
    pub cameras: Vec<CameraPose>,
    /// Whether coordinates are metric.
    pub scale: Scale,
}

impl Scene {
    /// Applies one similarity to Gaussians and cameras together, so they can
    /// never end up in different frames. `result_scale` records whether the
    /// result is in meters (metric only when `sim` carries a known metric scale,
    /// or the input already was metric and `sim` is rigid).
    pub fn transformed(&self, sim: &Similarity, result_scale: Scale) -> Scene {
        Scene {
            gaussians: self
                .gaussians
                .iter()
                .map(|g| sim.apply_gaussian(g))
                .collect(),
            cameras: self.cameras.iter().map(|c| sim.apply_camera(c)).collect(),
            scale: result_scale,
        }
    }
}
