//! Camera paths for flythrough videos (`camera_path.json`).

use serde::{Deserialize, Serialize};
use splat_types::nalgebra::{Matrix3, Rotation3, UnitQuaternion};
use splat_types::{CameraPose, PinholeIntrinsics, PolyTrajectory, Vec3};

/// One frame of a camera path.
///
/// `position` is the camera center; `rotation_wxyz` is the **world-to-camera**
/// rotation as a `(w, x, y, z)` quaternion, the same convention as COLMAP's
/// `images.txt`, so frames can be fed straight back into the renderer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraPathFrame {
    /// Time (s).
    pub t: f64,
    /// Camera center.
    pub position: [f64; 3],
    /// World-to-camera rotation `(w, x, y, z)`.
    pub rotation_wxyz: [f64; 4],
}

/// A camera path: shared intrinsics plus one pose per frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraPath {
    /// Frames per second.
    pub fps: f64,
    /// Intrinsics for every frame.
    pub intrinsics: PinholeIntrinsics,
    /// Frames in order.
    pub frames: Vec<CameraPathFrame>,
}

impl CameraPath {
    /// The pose of frame `i`.
    pub fn pose(&self, i: usize) -> Option<CameraPose> {
        let f = self.frames.get(i)?;
        let q = f.rotation_wxyz;
        let rot_cw = UnitQuaternion::from_quaternion(splat_types::nalgebra::Quaternion::new(
            q[0], q[1], q[2], q[3],
        ));
        let c = Vec3::from(f.position);
        Some(CameraPose {
            name: format!("{i:05}.png"),
            rot_cw,
            trans_cw: -(rot_cw * c),
            intrinsics: self.intrinsics,
        })
    }
}

/// World-to-camera rotation looking along `forward` with +Z as up
/// (falls back to looking along +X if `forward` is vertical or zero).
pub fn look_rotation(forward: &Vec3) -> UnitQuaternion<f64> {
    let f = forward
        .try_normalize(1e-9)
        .filter(|f| f.cross(&Vec3::z()).norm() > 1e-6)
        .unwrap_or_else(Vec3::x);
    let right = f.cross(&Vec3::z()).normalize();
    let down = f.cross(&right);
    let r = Matrix3::from_rows(&[right.transpose(), down.transpose(), f.transpose()]);
    UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r))
}

/// Samples `traj` at `fps`. With `look_at = Some(p)` the camera keeps `p`
/// centered (good for showing the object); otherwise it looks along the
/// direction of travel (from positions half a second apart, so it is
/// defined even when stopped at the ends), pitched down by `pitch_deg`.
pub fn camera_path_along(
    traj: &PolyTrajectory,
    fps: f64,
    intrinsics: PinholeIntrinsics,
    look_at: Option<Vec3>,
    pitch_deg: f64,
) -> CameraPath {
    let n = ((traj.duration() * fps).floor() as usize).max(1);
    let frames = (0..=n)
        .map(|i| {
            let t = (i as f64 / fps).min(traj.duration());
            let pos = traj.eval(t).pos;
            let forward = match look_at {
                Some(target) => target - pos,
                None => {
                    let ahead = traj.eval(t + 0.5).pos - traj.eval(t - 0.5).pos;
                    let horiz = Vec3::new(ahead.x, ahead.y, 0.0);
                    let n = horiz.norm().max(1e-9);
                    horiz / n * pitch_deg.to_radians().cos()
                        - Vec3::z() * pitch_deg.to_radians().sin()
                }
            };
            let q = look_rotation(&forward);
            let qq = q.quaternion();
            CameraPathFrame {
                t,
                position: pos.into(),
                rotation_wxyz: [qq.w, qq.i, qq.j, qq.k],
            }
        })
        .collect();
    CameraPath {
        fps,
        intrinsics,
        frames,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_frames_look_forward_and_roundtrip() {
        let traj =
            PolyTrajectory::polyline(&[Vec3::zeros(), Vec3::new(3.0, 0.0, 0.0)], 1.0).unwrap();
        let k = PinholeIntrinsics::from_fov_x(64, 48, 1.0);
        let path = camera_path_along(&traj, 10.0, k, None, 0.0);
        assert_eq!(path.frames.len(), 31);
        let pose = path.pose(10).unwrap();
        assert!((pose.center() - Vec3::new(1.0, 0.0, 0.0)).norm() < 1e-9);
        assert!((pose.forward() - Vec3::x()).norm() < 1e-9);
        assert!((pose.up() - Vec3::z()).norm() < 1e-9);
        // Looking at a target keeps it centered.
        let target = Vec3::new(1.5, 2.0, 0.5);
        let p2 = camera_path_along(&traj, 10.0, k, Some(target), 0.0);
        let (u, v, _) = p2.pose(3).unwrap().project(&target).unwrap();
        assert!((u - 32.0).abs() < 1e-6 && (v - 24.0).abs() < 1e-6);
        let json = serde_json::to_string(&path).unwrap();
        let back: CameraPath = serde_json::from_str(&json).unwrap();
        assert_eq!(back, path);
    }
}
