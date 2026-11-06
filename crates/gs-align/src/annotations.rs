//! Metric scale from `scale.json` or from known-distance annotations.

use serde::{Deserialize, Serialize};
use splat_types::nalgebra::Matrix3;
use splat_types::{CameraPose, Vec3};

use crate::AlignError;

/// `scale.json`: multiply reconstruction lengths by `metric_scale` to get meters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScaleFile {
    /// Meters per reconstruction unit.
    pub metric_scale: f64,
    /// How it was measured.
    #[serde(default)]
    pub method: String,
    /// Free-form notes.
    #[serde(default)]
    pub notes: String,
}

/// One pixel click on an endpoint in one image.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Click {
    /// Image file name (must match a COLMAP image name).
    pub image: String,
    /// Pixel x.
    pub x: f64,
    /// Pixel y.
    pub y: f64,
}

/// `annotations.json`: two endpoints of a known distance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Annotations {
    /// True distance between the endpoints, in meters.
    pub known_distance_m: f64,
    /// What was measured.
    #[serde(default)]
    pub label: String,
    /// Clicks on endpoint A (≥ 2 images).
    pub endpoint_a: Vec<Click>,
    /// Clicks on endpoint B (≥ 2 images).
    pub endpoint_b: Vec<Click>,
}

/// Least-squares intersection of rays: the point minimizing the summed
/// squared distance to every ray. Each ray is `(origin, unit direction)`.
///
/// Solves `Σ (I − d dᵀ) x = Σ (I − d dᵀ) c`. Returns `None` for fewer than
/// two rays or (near-)parallel rays.
pub fn triangulate(rays: &[(Vec3, Vec3)]) -> Option<Vec3> {
    if rays.len() < 2 {
        return None;
    }
    let mut a = Matrix3::zeros();
    let mut b = Vec3::zeros();
    for (c, d) in rays {
        let m = Matrix3::identity() - d * d.transpose();
        a += m;
        b += m * c;
    }
    // Reject ill-conditioned systems (all rays nearly parallel).
    let eig = a.symmetric_eigen();
    let (lo, hi) = (eig.eigenvalues.min(), eig.eigenvalues.max());
    if lo <= 1e-6 * hi {
        return None;
    }
    a.try_inverse().map(|inv| inv * b)
}

fn endpoint(clicks: &[Click], cameras: &[CameraPose], which: &str) -> Result<Vec3, AlignError> {
    let mut rays = Vec::new();
    for c in clicks {
        let cam = cameras
            .iter()
            .find(|cam| cam.name == c.image)
            .ok_or_else(|| {
                AlignError::BadAnnotation(format!("image `{}` not in the COLMAP model", c.image))
            })?;
        rays.push((cam.center(), cam.pixel_ray(c.x, c.y)));
    }
    if rays.len() < 2 {
        return Err(AlignError::BadAnnotation(format!(
            "endpoint {which} needs clicks in at least 2 images (got {})",
            rays.len()
        )));
    }
    triangulate(&rays).ok_or_else(|| {
        AlignError::BadAnnotation(format!("endpoint {which}: rays are (nearly) parallel"))
    })
}

/// Meters per reconstruction unit from a known-distance annotation.
pub fn scale_from_annotations(a: &Annotations, cameras: &[CameraPose]) -> Result<f64, AlignError> {
    if !(a.known_distance_m.is_finite() && a.known_distance_m > 0.0) {
        return Err(AlignError::BadAnnotation(
            "known_distance_m must be > 0".into(),
        ));
    }
    let pa = endpoint(&a.endpoint_a, cameras, "a")?;
    let pb = endpoint(&a.endpoint_b, cameras, "b")?;
    let d = (pa - pb).norm();
    if d <= 0.0 || !d.is_finite() {
        return Err(AlignError::BadAnnotation(
            "endpoints triangulate to the same point".into(),
        ));
    }
    Ok(a.known_distance_m / d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangulates_exactly() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        let origins = [
            Vec3::zeros(),
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(0.0, 6.0, 1.0),
        ];
        let rays: Vec<_> = origins.iter().map(|c| (*c, (p - c).normalize())).collect();
        assert!((triangulate(&rays).unwrap() - p).norm() < 1e-9);
        assert!(triangulate(&rays[..1]).is_none());
        let parallel = [(Vec3::zeros(), Vec3::x()), (Vec3::y(), Vec3::x())];
        assert!(triangulate(&parallel).is_none());
    }

    #[test]
    fn annotation_errors() {
        let a = Annotations {
            known_distance_m: 3.0,
            label: String::new(),
            endpoint_a: vec![Click {
                image: "missing.jpg".into(),
                x: 0.0,
                y: 0.0,
            }],
            endpoint_b: vec![],
        };
        assert!(scale_from_annotations(&a, &[]).is_err());
        let json =
            r#"{"known_distance_m": 3.0, "label": "tape", "endpoint_a": [], "endpoint_b": []}"#;
        let parsed: Annotations = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.label, "tape");
        let s: ScaleFile = serde_json::from_str(r#"{"metric_scale": 1.5}"#).unwrap();
        assert_eq!(s.metric_scale, 1.5);
    }
}
