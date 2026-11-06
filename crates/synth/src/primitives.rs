//! Analytic solids with exact distance functions, and their Gaussian "splat".

use serde::{Deserialize, Serialize};
use splat_types::nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};
use splat_types::{Gaussian, Vec3, Vec3f};

use crate::ellipsoid::distance_to_ellipsoid_local;

/// A solid shape. Distances are measured to the *solid* (zero inside).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Primitive {
    /// Box rotated by `yaw` (radians) about +Z.
    Box {
        /// Center.
        center: Vec3,
        /// Half extents in the box frame.
        half: Vec3,
        /// Rotation about +Z.
        yaw: f64,
    },
    /// Finite solid cylinder.
    Cylinder {
        /// Center.
        center: Vec3,
        /// Unit axis direction.
        axis: Vec3,
        /// Radius.
        radius: f64,
        /// Half the length along the axis.
        half_len: f64,
    },
    /// The half-space `z <= height` (infinite ground).
    Ground {
        /// Ground height.
        height: f64,
    },
    /// Closed axis-aligned box shell with a sealed, empty cavity inside.
    HollowBox {
        /// Center.
        center: Vec3,
        /// Outer half extents.
        outer_half: Vec3,
        /// Wall thickness.
        wall: f64,
    },
    /// Solid ellipsoid (used for random fields: one per Gaussian's kσ surface).
    Ellipsoid {
        /// Center.
        center: Vec3,
        /// Semi-axes in the local frame.
        semi_axes: Vec3,
        /// Rotation local→world.
        rot: UnitQuaternion<f64>,
    },
}

fn yaw_rot(yaw: f64) -> UnitQuaternion<f64> {
    UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw)
}

/// Distance from a point (in box frame) to a solid axis-aligned box with half extents `h`.
fn box_distance_local(q: &Vec3, h: &Vec3) -> f64 {
    let d = q.abs() - h;
    d.sup(&Vec3::zeros()).norm()
}

impl Primitive {
    /// Exact Euclidean distance from `p` to this solid (0 inside or on it).
    pub fn distance(&self, p: &Vec3) -> f64 {
        match self {
            Primitive::Box { center, half, yaw } => {
                let q = yaw_rot(*yaw).inverse() * (p - center);
                box_distance_local(&q, half)
            }
            Primitive::Cylinder {
                center,
                axis,
                radius,
                half_len,
            } => {
                let d = p - center;
                let along = d.dot(axis);
                let radial = (d - axis * along).norm();
                let dr = (radial - radius).max(0.0);
                let dz = (along.abs() - half_len).max(0.0);
                (dr * dr + dz * dz).sqrt()
            }
            Primitive::Ground { height } => (p.z - height).max(0.0),
            Primitive::HollowBox {
                center,
                outer_half,
                wall,
            } => {
                let q = p - center;
                let outside = box_distance_local(&q, outer_half);
                if outside > 0.0 {
                    return outside;
                }
                let inner = outer_half - Vec3::repeat(*wall);
                let a = q.abs();
                if (0..3).all(|i| a[i] < inner[i]) {
                    // In the cavity: distance to the nearest inner wall.
                    (0..3)
                        .map(|i| inner[i] - a[i])
                        .fold(f64::INFINITY, f64::min)
                } else {
                    0.0
                }
            }
            Primitive::Ellipsoid {
                center,
                semi_axes,
                rot,
            } => distance_to_ellipsoid_local(&(rot.inverse() * (p - center)), semi_axes),
        }
    }

    /// Covers the surface with flat "disc" Gaussians on a lattice of the given spacing.
    ///
    /// Each Gaussian has tangential std-dev `0.5·spacing` and normal std-dev
    /// `0.15·spacing`, lying exactly on the surface, so the splat reproduces
    /// the solid the way a well-trained 3DGS would. `Ground` is sampled over
    /// the square `[-ground_half, ground_half]²`.
    pub fn splat(
        &self,
        spacing: f64,
        ground_half: f64,
        color: [f32; 3],
        opacity: f32,
    ) -> Vec<Gaussian> {
        let mut out = Vec::new();
        let st = 0.5 * spacing;
        let sn = 0.15 * spacing;
        let push = |pos: Vec3, u: Vec3, v: Vec3, n: Vec3, out: &mut Vec<Gaussian>| {
            let m = Matrix3::from_columns(&[u, v, n]);
            let rot = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(m));
            out.push(
                Gaussian {
                    mean: pos.cast::<f32>(),
                    scale: Vec3f::new(st as f32, st as f32, sn as f32),
                    rot: rot.cast::<f32>(),
                    opacity,
                    sh_dc: [0.0; 3],
                }
                .with_color(color),
            );
        };
        match self {
            Primitive::Box { center, half, yaw } => {
                let r = yaw_rot(*yaw);
                box_faces(half, spacing, |q, u, v, n| {
                    push(center + r * q, r * u, r * v, r * n, &mut out)
                });
            }
            Primitive::HollowBox {
                center,
                outer_half,
                wall,
            } => {
                box_faces(outer_half, spacing, |q, u, v, n| {
                    push(center + q, u, v, n, &mut out)
                });
                let inner = outer_half - Vec3::repeat(*wall);
                // Inner faces point into the cavity (normal flipped).
                box_faces(&inner, spacing, |q, u, v, n| {
                    push(center + q, v, u, -n, &mut out)
                });
            }
            Primitive::Cylinder {
                center,
                axis,
                radius,
                half_len,
            } => {
                let w = axis.normalize();
                let helper = if w.x.abs() < 0.9 {
                    Vec3::x()
                } else {
                    Vec3::y()
                };
                let e1 = w.cross(&helper).normalize();
                let e2 = w.cross(&e1);
                let n_theta = ((2.0 * std::f64::consts::PI * radius) / spacing)
                    .ceil()
                    .max(6.0) as usize;
                let n_len = ((2.0 * half_len) / spacing).ceil().max(1.0) as usize;
                for i in 0..n_theta {
                    let th = 2.0 * std::f64::consts::PI * (i as f64 + 0.5) / n_theta as f64;
                    let radial = e1 * th.cos() + e2 * th.sin();
                    let tangent = w.cross(&radial);
                    for j in 0..n_len {
                        let s = -half_len + 2.0 * half_len * (j as f64 + 0.5) / n_len as f64;
                        push(
                            center + radial * *radius + w * s,
                            tangent,
                            w,
                            radial,
                            &mut out,
                        );
                    }
                }
                // End caps.
                let n_r = (radius / spacing).ceil() as i64;
                for sign in [-1.0, 1.0] {
                    for a in -n_r..=n_r {
                        for b in -n_r..=n_r {
                            let (x, y) = (a as f64 * spacing, b as f64 * spacing);
                            if x * x + y * y <= radius * radius {
                                let n = w * sign;
                                push(
                                    center + w * (sign * half_len) + e1 * x + e2 * y,
                                    e1,
                                    n.cross(&e1),
                                    n,
                                    &mut out,
                                );
                            }
                        }
                    }
                }
            }
            Primitive::Ground { height } => {
                let n = (ground_half / spacing).ceil() as i64;
                for a in -n..n {
                    for b in -n..n {
                        let x = (a as f64 + 0.5) * spacing;
                        let y = (b as f64 + 0.5) * spacing;
                        push(
                            Vec3::new(x, y, *height),
                            Vec3::x(),
                            Vec3::y(),
                            Vec3::z(),
                            &mut out,
                        );
                    }
                }
            }
            Primitive::Ellipsoid {
                center,
                semi_axes,
                rot,
            } => {
                // One Gaussian whose 3σ surface is the ellipsoid.
                out.push(
                    Gaussian {
                        mean: center.cast::<f32>(),
                        scale: (semi_axes / 3.0).cast::<f32>(),
                        rot: rot.cast::<f32>(),
                        opacity,
                        sh_dc: [0.0; 3],
                    }
                    .with_color(color),
                );
            }
        }
        out
    }
}

/// Calls `f(point, u, v, outward_normal)` for lattice points on the six faces
/// of an axis-aligned box with half extents `h` centered at the origin.
fn box_faces(h: &Vec3, spacing: f64, mut f: impl FnMut(Vec3, Vec3, Vec3, Vec3)) {
    for axis in 0..3 {
        let (ua, va) = ((axis + 1) % 3, (axis + 2) % 3);
        let nu = ((2.0 * h[ua]) / spacing).ceil().max(1.0) as usize;
        let nv = ((2.0 * h[va]) / spacing).ceil().max(1.0) as usize;
        for sign in [-1.0, 1.0] {
            let mut n = Vec3::zeros();
            n[axis] = sign;
            let mut u = Vec3::zeros();
            u[ua] = 1.0;
            let mut v = Vec3::zeros();
            v[va] = sign; // keeps (u, v, n) right-handed
            for i in 0..nu {
                for j in 0..nv {
                    let mut q = Vec3::zeros();
                    q[axis] = sign * h[axis];
                    q[ua] = -h[ua] + 2.0 * h[ua] * (i as f64 + 0.5) / nu as f64;
                    q[va] = -h[va] + 2.0 * h[va] * (j as f64 + 0.5) / nv as f64;
                    f(q, u, v, n);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distances() {
        let b = Primitive::Box {
            center: Vec3::new(0.0, 0.0, 1.0),
            half: Vec3::new(1.0, 2.0, 1.0),
            yaw: std::f64::consts::FRAC_PI_2,
        };
        // After a 90° yaw the long axis is x.
        assert!((b.distance(&Vec3::new(3.0, 0.0, 1.0)) - 1.0).abs() < 1e-12);
        assert!((b.distance(&Vec3::new(0.0, 3.0, 1.0)) - 2.0).abs() < 1e-12);
        assert_eq!(b.distance(&Vec3::new(0.0, 0.0, 1.0)), 0.0);

        let c = Primitive::Cylinder {
            center: Vec3::zeros(),
            axis: Vec3::y(),
            radius: 0.5,
            half_len: 0.1,
        };
        assert!((c.distance(&Vec3::new(1.5, 0.0, 0.0)) - 1.0).abs() < 1e-12);
        assert!((c.distance(&Vec3::new(0.0, 1.1, 0.0)) - 1.0).abs() < 1e-12);
        assert!((c.distance(&Vec3::new(3.5, 4.1, 0.0)) - 5.0).abs() < 1e-12);

        assert_eq!(
            Primitive::Ground { height: 0.0 }.distance(&Vec3::new(5.0, 5.0, -1.0)),
            0.0
        );

        let h = Primitive::HollowBox {
            center: Vec3::zeros(),
            outer_half: Vec3::repeat(1.0),
            wall: 0.1,
        };
        assert!((h.distance(&Vec3::zeros()) - 0.9).abs() < 1e-12);
        assert_eq!(h.distance(&Vec3::new(0.95, 0.0, 0.0)), 0.0);
        assert!((h.distance(&Vec3::new(2.0, 0.0, 0.0)) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn splat_gaussians_lie_on_surface() {
        let prims = [
            Primitive::Box {
                center: Vec3::new(0.3, -0.2, 1.0),
                half: Vec3::new(1.0, 0.5, 0.4),
                yaw: 0.4,
            },
            Primitive::Cylinder {
                center: Vec3::zeros(),
                axis: Vec3::new(0.0, 1.0, 0.0),
                radius: 0.3,
                half_len: 0.1,
            },
        ];
        for p in &prims {
            let gs = p.splat(0.05, 0.0, [1.0, 0.0, 0.0], 0.9);
            assert!(gs.len() > 50);
            for g in &gs {
                // On the surface: distance to the solid is ~0, and a point pushed
                // out along the disc normal is at distance ~push.
                let m = g.mean_f64();
                assert!(p.distance(&m) < 1e-6);
                let normal = g.rot.cast::<f64>() * Vec3::z();
                assert!((p.distance(&(m + normal * 0.01)) - 0.01).abs() < 1e-6);
            }
        }
    }
}
