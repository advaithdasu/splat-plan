//! Ground-plane fit, gravity alignment and metric scale recovery.
//!
//! A COLMAP reconstruction lives in an arbitrary frame with an arbitrary
//! scale. Planning needs "+Z is up, units are meters". This crate produces
//! one [`Similarity`] that does both, and applies it to the Gaussians and the
//! cameras together.
//!
//! 1. **Up prior**: the mean of the cameras' "up" vectors. A drone orbiting a
//!    car keeps its camera roughly level, so this points roughly skyward.
//! 2. **Ground plane**: RANSAC over Gaussian means, accepting only planes
//!    whose normal is within `max_prior_angle_deg` (30°) of the prior and
//!    that lie below most cameras (so a ceiling can't win), then a
//!    least-squares refit on the inliers.
//! 3. **Gravity alignment**: rotate the plane normal onto +Z, put the ground
//!    at `z = 0`, center the scene horizontally.
//! 4. **Scale**: from `scale.json`, or from
//!    `annotations.json` (two endpoints of a known distance, each clicked in
//!    ≥ 2 images and triangulated with the COLMAP poses). Without either, the
//!    result is [`Scale::Unscaled`].
#![forbid(unsafe_code)]

mod annotations;
mod plane;

use serde::{Deserialize, Serialize};
use splat_types::nalgebra::UnitQuaternion;
use splat_types::{Report, Scale, Scene, Similarity, Vec3};
use thiserror::Error;

pub use annotations::{Annotations, Click, ScaleFile, scale_from_annotations, triangulate};
pub use plane::{Plane, fit_ground_plane};

/// Errors from alignment.
#[derive(Debug, Error)]
pub enum AlignError {
    /// The scene has no cameras, so there is no up prior.
    #[error("scene has no cameras; cannot estimate the up direction")]
    NoCameras,
    /// Too few Gaussians to fit a plane.
    #[error("scene has only {0} Gaussians; need at least 3")]
    TooFewPoints(usize),
    /// No plane within the prior cone had enough support.
    #[error(
        "no ground plane found within {max_angle_deg}° of the camera up prior and below the cameras (best inlier ratio {best_ratio:.3})"
    )]
    NoGroundPlane {
        /// Allowed angle.
        max_angle_deg: f64,
        /// Best inlier ratio seen.
        best_ratio: f64,
    },
    /// Scale annotations are unusable.
    #[error("bad scale annotation: {0}")]
    BadAnnotation(String),
}

/// Alignment settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignConfig {
    /// RANSAC iterations.
    pub ransac_iters: usize,
    /// Inlier distance as a fraction of the robust scene diagonal (scale-free).
    pub inlier_fraction: f64,
    /// Plane normal must be within this angle of the camera up prior.
    pub max_prior_angle_deg: f64,
    /// Minimum fraction of (subsampled) Gaussians on the plane.
    pub min_inlier_ratio: f64,
    /// Maximum Gaussians used for RANSAC (deterministic subsample).
    pub max_points: usize,
    /// RNG seed, so the fit is reproducible.
    pub seed: u64,
}

impl Default for AlignConfig {
    fn default() -> Self {
        AlignConfig {
            ransac_iters: 500,
            inlier_fraction: 0.004,
            max_prior_angle_deg: 30.0,
            min_inlier_ratio: 0.05,
            max_points: 50_000,
            seed: 0,
        }
    }
}

/// Where the metric scale came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ScaleSource {
    /// No scale information: output stays [`Scale::Unscaled`].
    None,
    /// Multiply input lengths by this factor (from `scale.json`).
    Manual(f64),
    /// Triangulated known-distance annotation.
    Annotations(Annotations),
}

/// What alignment did, for the report and for reproducing it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AlignReport {
    /// The transform applied (input frame → aligned frame).
    pub similarity: Similarity,
    /// Output scale state.
    pub scale: Scale,
    /// How the scale was obtained.
    pub scale_method: String,
    /// Mean camera up vector (input frame, unit).
    pub up_prior: [f64; 3],
    /// Ground normal (input frame, unit, pointing up).
    pub ground_normal: [f64; 3],
    /// Angle between prior and ground normal (degrees).
    pub prior_to_normal_deg: f64,
    /// Fraction of sampled Gaussians within the inlier band of the ground.
    pub ground_inlier_ratio: f64,
    /// Set when the capture's `ground.json` moved z = 0 from the fitted plane
    /// down to the real floor (the fit can lock onto desk tops when the floor
    /// is plain carpet). Already included in `similarity`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ground_correction: Option<String>,
}

impl Report for AlignReport {}

/// Mean camera "up" direction (unit), the gravity prior.
pub fn up_prior(scene: &Scene) -> Result<Vec3, AlignError> {
    let sum: Vec3 = scene.cameras.iter().map(|c| c.up()).sum();
    sum.try_normalize(1e-9).ok_or(AlignError::NoCameras)
}

/// Aligns a scene: ground at `z = 0` with +Z up, horizontally centered, and
/// metric if `scale` provides it.
pub fn align_scene(
    scene: &Scene,
    cfg: &AlignConfig,
    scale: &ScaleSource,
) -> Result<(Scene, AlignReport), AlignError> {
    if scene.cameras.is_empty() {
        return Err(AlignError::NoCameras);
    }
    let prior = up_prior(scene)?;
    let means: Vec<Vec3> = scene.gaussians.iter().map(|g| g.mean_f64()).collect();
    let (plane, ratio) = fit_ground_plane(&means, &prior, &scene.cameras, cfg)?;

    let (s, method) = match scale {
        ScaleSource::None => (1.0, "none (UNSCALED)".to_string()),
        ScaleSource::Manual(f) => {
            if !(f.is_finite() && *f > 0.0) {
                return Err(AlignError::BadAnnotation(format!(
                    "metric_scale {f} must be > 0"
                )));
            }
            (*f, "scale.json".to_string())
        }
        ScaleSource::Annotations(a) => (
            scale_from_annotations(a, &scene.cameras)?,
            format!("annotations.json ({})", a.label),
        ),
    };

    // Rotation taking the ground normal to +Z (shortest arc; the yaw is
    // otherwise arbitrary, which is fine for planning).
    let rot = UnitQuaternion::rotation_between(&plane.normal, &Vec3::z())
        .unwrap_or_else(|| UnitQuaternion::from_axis_angle(&Vec3::x_axis(), std::f64::consts::PI));
    // Horizontal center: median of means projected onto the plane. Height:
    // the plane itself goes to z = 0.
    let center = robust_center(&means, &plane);
    let sim = Similarity {
        scale: s,
        rot,
        trans: -(rot * center) * s,
    };
    let result_scale = match (scale, scene.scale) {
        (ScaleSource::None, Scale::Metric(m)) => Scale::Metric(m),
        (ScaleSource::None, Scale::Unscaled) => Scale::Unscaled,
        _ => Scale::Metric(s),
    };
    let aligned = scene.transformed(&sim, result_scale);
    let report = AlignReport {
        similarity: sim,
        scale: result_scale,
        scale_method: method,
        up_prior: prior.into(),
        ground_normal: plane.normal.into(),
        prior_to_normal_deg: prior.angle(&plane.normal).to_degrees(),
        ground_inlier_ratio: ratio,
        ground_correction: None,
    };
    Ok((aligned, report))
}

/// Per-axis median of the means, then dropped onto the plane.
fn robust_center(means: &[Vec3], plane: &Plane) -> Vec3 {
    let mut c = Vec3::zeros();
    for axis in 0..3 {
        let mut v: Vec<f64> = means.iter().map(|p| p[axis]).collect();
        v.sort_by(f64::total_cmp);
        c[axis] = v.get(v.len() / 2).copied().unwrap_or(0.0);
    }
    c - plane.normal * plane.signed_distance(&c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use synth::{CarConfig, car_scene, random_similarity};

    /// Synthetic annotation: two ground markers 3 m apart, clicked in several images.
    fn tape_annotations(scene: &synth::SynthScene, colmap_scene: &Scene) -> Annotations {
        let a = Vec3::new(-1.5, -2.2, 0.0);
        let b = Vec3::new(1.5, -2.2, 0.0);
        let clicks = |p: &Vec3| -> Vec<Click> {
            scene
                .cameras
                .iter()
                .zip(&colmap_scene.cameras)
                .filter_map(|(metric_cam, cam)| {
                    let (x, y, _) = metric_cam.project(p)?;
                    let k = &cam.intrinsics;
                    (x >= 0.0 && y >= 0.0 && x < k.width as f64 && y < k.height as f64).then(|| {
                        Click {
                            image: cam.name.clone(),
                            x,
                            y,
                        }
                    })
                })
                .take(6)
                .collect()
        };
        Annotations {
            known_distance_m: 3.0,
            label: "synthetic tape".into(),
            endpoint_a: clicks(&a),
            endpoint_b: clicks(&b),
        }
    }

    #[test]
    fn recovers_gravity_and_scale_under_random_similarity() {
        let synth_scene = car_scene(&CarConfig::default());
        for seed in 0..5u64 {
            let sim = random_similarity(seed);
            let colmap = synth_scene.to_scene().transformed(&sim, Scale::Unscaled);
            let ann = tape_annotations(&synth_scene, &colmap);
            let (aligned, rep) = align_scene(
                &colmap,
                &AlignConfig::default(),
                &ScaleSource::Annotations(ann),
            )
            .unwrap();
            // Gravity: true up in the COLMAP frame is R_sim · z.
            let true_up = sim.rot * Vec3::z();
            let err_deg = true_up.angle(&Vec3::from(rep.ground_normal)).to_degrees();
            assert!(err_deg < 1.0, "seed {seed}: gravity error {err_deg}°");
            // Scale: product with the true scale should be 1.
            let s = match rep.scale {
                Scale::Metric(s) => s,
                Scale::Unscaled => panic!("expected metric"),
            };
            let rel = (s * sim.scale - 1.0).abs();
            assert!(rel < 0.005, "seed {seed}: scale error {rel}");
            // In the aligned frame the camera heights match the metric rig heights.
            let h_true = synth_scene.cameras[0].center().z;
            let h = aligned.cameras[0].center().z;
            assert!(
                (h - h_true).abs() < 0.05,
                "seed {seed}: height {h} vs {h_true}"
            );
        }
    }

    #[test]
    fn without_scale_output_is_unscaled() {
        let synth_scene = car_scene(&CarConfig::default());
        let colmap = synth_scene
            .to_scene()
            .transformed(&random_similarity(42), Scale::Unscaled);
        let (aligned, rep) =
            align_scene(&colmap, &AlignConfig::default(), &ScaleSource::None).unwrap();
        assert_eq!(aligned.scale, Scale::Unscaled);
        assert_eq!(rep.scale, Scale::Unscaled);
        // Manual scale makes it metric.
        let (_, rep2) =
            align_scene(&colmap, &AlignConfig::default(), &ScaleSource::Manual(2.0)).unwrap();
        assert_eq!(rep2.scale, Scale::Metric(2.0));
        assert!(align_scene(&colmap, &AlignConfig::default(), &ScaleSource::Manual(-1.0)).is_err());
    }

    #[test]
    fn errors_without_cameras_or_points() {
        let mut s = car_scene(&CarConfig::default()).to_scene();
        let cams = std::mem::take(&mut s.cameras);
        assert!(matches!(
            align_scene(&s, &AlignConfig::default(), &ScaleSource::None),
            Err(AlignError::NoCameras)
        ));
        let empty = Scene {
            gaussians: vec![],
            cameras: cams,
            scale: Scale::Unscaled,
        };
        assert!(matches!(
            align_scene(&empty, &AlignConfig::default(), &ScaleSource::None),
            Err(AlignError::TooFewPoints(0))
        ));
    }
}
