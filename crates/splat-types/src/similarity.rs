use nalgebra::UnitQuaternion;
use serde::{Deserialize, Serialize};

use crate::{CameraPose, Gaussian, Vec3};

/// A similarity transform `x' = s·R·x + t` (uniform scale, rotation, translation).
///
/// Alignment produces exactly one of these, and it is applied to the Gaussians
/// and the cameras together (see [`crate::Scene::transformed`]) so that the two
/// can never drift into different frames.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Similarity {
    /// Uniform scale `s > 0`.
    pub scale: f64,
    /// Rotation `R`.
    pub rot: UnitQuaternion<f64>,
    /// Translation `t`.
    pub trans: Vec3,
}

impl Default for Similarity {
    fn default() -> Self {
        Self::identity()
    }
}

impl Similarity {
    /// The identity transform.
    pub fn identity() -> Self {
        Similarity {
            scale: 1.0,
            rot: UnitQuaternion::identity(),
            trans: Vec3::zeros(),
        }
    }

    /// Transforms a point.
    pub fn apply_point(&self, p: &Vec3) -> Vec3 {
        self.rot * p * self.scale + self.trans
    }

    /// Transforms a direction/displacement (no translation).
    pub fn apply_vector(&self, v: &Vec3) -> Vec3 {
        self.rot * v * self.scale
    }

    /// Transforms a Gaussian: the mean moves, scales grow by `s`, and the
    /// rotation is composed with `R` (so `Σ' = s² R Σ Rᵀ`).
    pub fn apply_gaussian(&self, g: &Gaussian) -> Gaussian {
        let mean = self.apply_point(&g.mean_f64()).cast::<f32>();
        let rot = self.rot.cast::<f32>() * g.rot;
        Gaussian {
            mean,
            scale: g.scale * self.scale as f32,
            rot,
            opacity: g.opacity,
            sh_dc: g.sh_dc,
        }
    }

    /// Transforms a camera. The center moves like a point; the world-to-camera
    /// rotation becomes `R_cw · Rᵀ` so that the camera keeps seeing the same
    /// (transformed) world. Intrinsics are unchanged.
    pub fn apply_camera(&self, c: &CameraPose) -> CameraPose {
        let center = self.apply_point(&c.center());
        let rot_cw = c.rot_cw * self.rot.inverse();
        let trans_cw = -(rot_cw * center);
        CameraPose {
            name: c.name.clone(),
            rot_cw,
            trans_cw,
            intrinsics: c.intrinsics,
        }
    }

    /// `self ∘ other`: first apply `other`, then `self`.
    pub fn compose(&self, other: &Similarity) -> Similarity {
        Similarity {
            scale: self.scale * other.scale,
            rot: self.rot * other.rot,
            trans: self.rot * other.trans * self.scale + self.trans,
        }
    }

    /// The inverse transform. Returns `None` if the scale is not positive and finite.
    pub fn inverse(&self) -> Option<Similarity> {
        if !(self.scale.is_finite() && self.scale > 0.0) {
            return None;
        }
        let inv_rot = self.rot.inverse();
        let inv_scale = 1.0 / self.scale;
        Some(Similarity {
            scale: inv_scale,
            rot: inv_rot,
            trans: -(inv_rot * self.trans) * inv_scale,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PinholeIntrinsics;
    use nalgebra::Vector3;
    use proptest::prelude::*;

    fn sim(s: f64, ax: [f64; 3], ang: f64, t: [f64; 3]) -> Similarity {
        let axis = Vector3::new(ax[0], ax[1], ax[2] + 1e-3);
        Similarity {
            scale: s,
            rot: UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(axis), ang),
            trans: Vec3::new(t[0], t[1], t[2]),
        }
    }

    proptest! {
        #[test]
        fn inverse_and_compose_roundtrip(
            s in 0.1f64..10.0, ax in prop::array::uniform3(-1.0f64..1.0), ang in -3.0f64..3.0,
            t in prop::array::uniform3(-10.0f64..10.0), p in prop::array::uniform3(-10.0f64..10.0),
        ) {
            let a = sim(s, ax, ang, t);
            let b = sim(1.0 / s + 0.5, [ax[1], ax[2], ax[0]], ang * 0.3, [t[2], t[0], t[1]]);
            let p = Vec3::new(p[0], p[1], p[2]);
            let back = a.inverse().unwrap().apply_point(&a.apply_point(&p));
            prop_assert!((back - p).norm() < 1e-9 * (1.0 + p.norm()));
            let ab = a.compose(&b).apply_point(&p);
            let seq = a.apply_point(&b.apply_point(&p));
            prop_assert!((ab - seq).norm() < 1e-9 * (1.0 + seq.norm()));
        }

        #[test]
        fn camera_projection_invariant_under_similarity(
            s in 0.2f64..5.0, ax in prop::array::uniform3(-1.0f64..1.0), ang in -3.0f64..3.0,
            t in prop::array::uniform3(-5.0f64..5.0),
        ) {
            // A camera and a point transformed together must project to the same pixel.
            let k = PinholeIntrinsics::from_fov_x(320, 240, 1.0);
            let cam = CameraPose::look_at("c", Vec3::new(4.0, 1.0, 2.0), Vec3::zeros(), Vec3::z(), k).unwrap();
            let p = Vec3::new(0.2, -0.1, 0.3);
            let a = sim(s, ax, ang, t);
            let (u0, v0, d0) = cam.project(&p).unwrap();
            let cam2 = a.apply_camera(&cam);
            let (u1, v1, d1) = cam2.project(&a.apply_point(&p)).unwrap();
            prop_assert!((u0 - u1).abs() < 1e-6 && (v0 - v1).abs() < 1e-6);
            prop_assert!((d1 - s * d0).abs() < 1e-6 * (1.0 + d1));
            prop_assert!((cam2.center() - a.apply_point(&cam.center())).norm() < 1e-9 * (1.0 + cam2.center().norm()));
        }
    }

    #[test]
    fn gaussian_covariance_transforms() {
        let a = sim(2.0, [0.3, 0.1, 0.9], 0.7, [1.0, 2.0, 3.0]);
        let g = Gaussian {
            mean: crate::Vec3f::new(1.0, 0.0, 0.0),
            scale: crate::Vec3f::new(0.1, 0.2, 0.3),
            rot: UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.4),
            opacity: 0.5,
            sh_dc: [0.0; 3],
        };
        let g2 = a.apply_gaussian(&g);
        let r = a.rot.to_rotation_matrix().into_inner();
        let expected = r * g.covariance() * r.transpose() * 4.0;
        assert!((g2.covariance() - expected).norm() < 1e-5);
        assert!(a.inverse().is_some());
        assert!(
            Similarity {
                scale: 0.0,
                ..Similarity::identity()
            }
            .inverse()
            .is_none()
        );
    }
}
