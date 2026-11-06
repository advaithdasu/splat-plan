//! Camera rigs that imitate drone capture paths.

use splat_types::{CameraPose, PinholeIntrinsics, Vec3};

/// Default synthetic camera: 320×240, 75° horizontal field of view.
pub fn default_intrinsics() -> PinholeIntrinsics {
    PinholeIntrinsics::from_fov_x(320, 240, 75f64.to_radians())
}

/// Rings of cameras orbiting `target`, all looking at it.
///
/// One ring per `(radius, height)` entry, `per_ring` cameras evenly spaced in
/// azimuth (starting at +X, counter-clockwise). Images are named `00001.jpg`,
/// `00002.jpg`, ... in capture order.
pub fn orbit_rig(
    target: Vec3,
    rings: &[(f64, f64)],
    per_ring: usize,
    intrinsics: PinholeIntrinsics,
) -> Vec<CameraPose> {
    let mut out = Vec::with_capacity(rings.len() * per_ring);
    for (radius, height) in rings {
        for i in 0..per_ring {
            let az = 2.0 * std::f64::consts::PI * i as f64 / per_ring as f64;
            let c = Vec3::new(
                target.x + radius * az.cos(),
                target.y + radius * az.sin(),
                *height,
            );
            let name = format!("{:05}.jpg", out.len() + 1);
            // look_at only fails if the camera sits on the target or looks
            // straight down the up axis; skip such degenerate placements.
            if let Some(cam) = CameraPose::look_at(name, c, target, Vec3::z(), intrinsics) {
                out.push(cam);
            }
        }
    }
    out
}

/// Azimuth (radians, in `(-π, π]`) of a camera center around `target`.
pub fn azimuth(cam: &CameraPose, target: &Vec3) -> f64 {
    let d = cam.center() - target;
    d.y.atan2(d.x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orbit_looks_at_target() {
        let t = Vec3::new(0.0, 0.0, 0.8);
        let cams = orbit_rig(t, &[(5.0, 1.5), (4.0, 3.0)], 12, default_intrinsics());
        assert_eq!(cams.len(), 24);
        assert_eq!(cams[0].name, "00001.jpg");
        for c in &cams {
            let (u, v, _) = c.project(&t).unwrap();
            assert!((u - 160.0).abs() < 1e-6 && (v - 120.0).abs() < 1e-6);
        }
        assert!(azimuth(&cams[3], &t) > 1.5 && azimuth(&cams[3], &t) < 1.6);
    }
}
