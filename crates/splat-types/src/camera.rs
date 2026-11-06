use nalgebra::{Matrix3, Rotation3, UnitQuaternion};
use serde::{Deserialize, Serialize};

use crate::Vec3;

/// Pinhole intrinsics for an undistorted image.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PinholeIntrinsics {
    /// Focal length in pixels along x.
    pub fx: f64,
    /// Focal length in pixels along y.
    pub fy: f64,
    /// Principal point x (pixels).
    pub cx: f64,
    /// Principal point y (pixels).
    pub cy: f64,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

impl PinholeIntrinsics {
    /// Intrinsics for a given horizontal field of view (radians), principal point centered.
    pub fn from_fov_x(width: u32, height: u32, fov_x: f64) -> Self {
        let fx = (width as f64 / 2.0) / (fov_x / 2.0).tan();
        PinholeIntrinsics {
            fx,
            fy: fx,
            cx: width as f64 / 2.0,
            cy: height as f64 / 2.0,
            width,
            height,
        }
    }

    /// Same camera at a different resolution (focal lengths and principal point scale with it).
    pub fn rescaled(&self, factor: f64) -> Self {
        PinholeIntrinsics {
            fx: self.fx * factor,
            fy: self.fy * factor,
            cx: self.cx * factor,
            cy: self.cy * factor,
            width: ((self.width as f64) * factor).round().max(1.0) as u32,
            height: ((self.height as f64) * factor).round().max(1.0) as u32,
        }
    }
}

/// A camera in the COLMAP convention.
///
/// Camera axes: x right, y down, z forward. COLMAP's `images.txt` stores the
/// **world-to-camera** rotation `R` and translation `t`, so a world point `X`
/// has camera coordinates `R·X + t`. The camera center is therefore
/// `C = -Rᵀ·t` (the classic place to get the sign or transpose wrong).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraPose {
    /// Image file name this pose belongs to.
    pub name: String,
    /// World-to-camera rotation `R`.
    pub rot_cw: UnitQuaternion<f64>,
    /// World-to-camera translation `t`.
    pub trans_cw: Vec3,
    /// Pinhole intrinsics.
    pub intrinsics: PinholeIntrinsics,
}

impl CameraPose {
    /// Camera center in world coordinates: `C = -Rᵀ t`.
    pub fn center(&self) -> Vec3 {
        -(self.rot_cw.inverse() * self.trans_cw)
    }

    /// World point to camera coordinates: `R·X + t`.
    pub fn world_to_camera(&self, p: &Vec3) -> Vec3 {
        self.rot_cw * p + self.trans_cw
    }

    /// Camera-frame direction expressed in world coordinates.
    pub fn camera_dir_to_world(&self, d: &Vec3) -> Vec3 {
        self.rot_cw.inverse() * d
    }

    /// Viewing direction (camera +z) in world coordinates.
    pub fn forward(&self) -> Vec3 {
        self.camera_dir_to_world(&Vec3::z())
    }

    /// Camera "up" (camera −y, because image y points down) in world coordinates.
    pub fn up(&self) -> Vec3 {
        -self.camera_dir_to_world(&Vec3::y())
    }

    /// Builds a camera at `center` looking at `target`, with image "up"
    /// as close to `world_up` as possible.
    ///
    /// Returns `None` if `target == center` or the view direction is parallel to `world_up`.
    pub fn look_at(
        name: impl Into<String>,
        center: Vec3,
        target: Vec3,
        world_up: Vec3,
        intrinsics: PinholeIntrinsics,
    ) -> Option<Self> {
        let forward = (target - center).try_normalize(1e-12)?;
        // Looking along +x with up +z, "right" must be −y: forward × up gives exactly that.
        let right = forward.cross(&world_up).try_normalize(1e-9)?;
        let down = forward.cross(&right);
        // Rows of R are the camera axes expressed in world coordinates.
        let r = Matrix3::from_rows(&[right.transpose(), down.transpose(), forward.transpose()]);
        let rot_cw = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r));
        let trans_cw = -(rot_cw * center);
        Some(CameraPose {
            name: name.into(),
            rot_cw,
            trans_cw,
            intrinsics,
        })
    }

    /// Projects a world point to pixel coordinates `(u, v, depth)`.
    /// Returns `None` for points at or behind the camera plane.
    pub fn project(&self, p: &Vec3) -> Option<(f64, f64, f64)> {
        let c = self.world_to_camera(p);
        if c.z <= 1e-9 {
            return None;
        }
        let k = &self.intrinsics;
        Some((k.fx * c.x / c.z + k.cx, k.fy * c.y / c.z + k.cy, c.z))
    }

    /// World-space unit ray direction through pixel `(u, v)`.
    pub fn pixel_ray(&self, u: f64, v: f64) -> Vec3 {
        let k = &self.intrinsics;
        let d = Vec3::new((u - k.cx) / k.fx, (v - k.cy) / k.fy, 1.0).normalize();
        self.camera_dir_to_world(&d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    fn k() -> PinholeIntrinsics {
        PinholeIntrinsics::from_fov_x(640, 480, 1.2)
    }

    #[test]
    fn center_is_minus_r_transpose_t() {
        // Hand-built pose: rotate 90° about z, translate (1,2,3).
        let rot_cw = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.5);
        let trans_cw = Vec3::new(1.0, 2.0, 3.0);
        let cam = CameraPose {
            name: "a".into(),
            rot_cw,
            trans_cw,
            intrinsics: k(),
        };
        let r = rot_cw.to_rotation_matrix().into_inner();
        let expected = -(r.transpose() * trans_cw);
        assert!((cam.center() - expected).norm() < 1e-12);
        // The center maps to the camera-frame origin.
        assert!(cam.world_to_camera(&cam.center()).norm() < 1e-12);
    }

    #[test]
    fn look_at_points_forward_and_up() {
        let c = Vec3::new(5.0, 0.0, 2.0);
        let cam = CameraPose::look_at("x", c, Vec3::new(0.0, 0.0, 2.0), Vec3::z(), k()).unwrap();
        assert!((cam.center() - c).norm() < 1e-12);
        assert!((cam.forward() - Vec3::new(-1.0, 0.0, 0.0)).norm() < 1e-12);
        assert!((cam.up() - Vec3::z()).norm() < 1e-12);
        // The target projects to the principal point.
        let (u, v, d) = cam.project(&Vec3::new(0.0, 0.0, 2.0)).unwrap();
        assert!((u - 320.0).abs() < 1e-9 && (v - 240.0).abs() < 1e-9 && (d - 5.0).abs() < 1e-12);
        // A point above the target appears higher in the image (smaller v).
        let (_, v_up, _) = cam.project(&Vec3::new(0.0, 0.0, 3.0)).unwrap();
        assert!(v_up < 240.0);
        // Behind the camera: no projection.
        assert!(cam.project(&Vec3::new(10.0, 0.0, 2.0)).is_none());
        assert!(CameraPose::look_at("y", c, c, Vec3::z(), k()).is_none());
    }

    #[test]
    fn pixel_ray_inverts_project() {
        let cam = CameraPose::look_at(
            "x",
            Vec3::new(3.0, -4.0, 1.5),
            Vec3::new(0.0, 0.0, 0.5),
            Vec3::z(),
            k(),
        )
        .unwrap();
        let p = Vec3::new(0.3, 0.2, 0.9);
        let (u, v, depth) = cam.project(&p).unwrap();
        let ray = cam.pixel_ray(u, v);
        let along = ray * (p - cam.center()).norm();
        assert!((cam.center() + along - p).norm() < 1e-9);
        assert!(depth > 0.0);
    }
}
