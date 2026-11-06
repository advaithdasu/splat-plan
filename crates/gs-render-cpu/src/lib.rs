//! CPU reference Gaussian splat rasterizer (color and expected depth).
//!
//! Follows the original 3DGS forward pass (Kerbl et al. 2023), written for
//! clarity rather than speed:
//!
//! 1. **Project** each Gaussian with the EWA approximation: the 3D covariance
//!    `Σ` becomes the 2D covariance `Σ' = J W Σ Wᵀ Jᵀ` (top-left 2×2), where
//!    `W` is the world-to-camera rotation and `J` the Jacobian of the
//!    perspective projection at the Gaussian's center. `0.3 px²` is added to
//!    the diagonal (a low-pass filter so tiny splats still cover a pixel).
//! 2. **Bin** splats into 16×16-pixel tiles and **sort** each tile's list by
//!    depth (ties by index, so the result is deterministic).
//! 3. **Composite** front to back per pixel: `α = opacity · exp(−½ dᵀ Σ'⁻¹ d)`,
//!    skip `α < 1/255`, clamp `α ≤ 0.99`, stop when transmittance `T < 1e-4`.
//!
//! Color uses only the DC spherical-harmonic term (view-independent).
//! Depth is the expected depth `Σ wᵢ zᵢ / Σ wᵢ` with `wᵢ = T·α`.
#![forbid(unsafe_code)]

mod path;

use rayon::prelude::*;
use splat_types::nalgebra::{Matrix2, Matrix2x3, Matrix3, Vector2};
use splat_types::{CameraPose, Gaussian};

pub use path::{CameraPath, CameraPathFrame, camera_path_along, look_rotation};

/// Tile edge length in pixels.
pub const TILE: usize = 16;
/// Splats nearer than this (camera-space z) are culled (as in 3DGS).
pub const NEAR: f64 = 0.2;
/// The Jacobian is evaluated with the view direction clamped to this
/// multiple of the half field of view (as in 3DGS). Without the clamp, a
/// splat far outside the frustum but barely in front of the camera gets a
/// gigantic `x/z²` term and smears an opaque ellipse over the whole image.
pub const JACOBIAN_FOV_CLAMP: f64 = 1.3;

/// One shaded pixel: (color, accumulated alpha, expected depth).
type Shaded = ([f32; 3], f32, f32);

/// A rendered image (row-major, top row first).
#[derive(Debug, Clone, PartialEq)]
pub struct Rendered {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Linear RGB in `0..=1`.
    pub rgb: Vec<[f32; 3]>,
    /// Accumulated opacity `1 − T` per pixel.
    pub alpha: Vec<f32>,
    /// Expected depth per pixel (camera z), or `f32::INFINITY` where `alpha` is ~0.
    pub depth: Vec<f32>,
}

impl Rendered {
    /// RGB as 8-bit bytes (row-major, 3 bytes per pixel).
    pub fn rgb8(&self) -> Vec<u8> {
        self.rgb
            .iter()
            .flat_map(|c| c.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8))
            .collect()
    }
}

/// Render settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderConfig {
    /// Background color.
    pub background: [f32; 3],
    /// Value added to the 2D covariance diagonal (px²).
    pub dilation: f64,
}

impl Default for RenderConfig {
    fn default() -> Self {
        RenderConfig {
            background: [0.0, 0.0, 0.0],
            dilation: 0.3,
        }
    }
}

/// A Gaussian after projection to the image.
#[derive(Debug, Clone, Copy)]
pub struct Splat2D {
    /// Index of the source Gaussian (depth-sort tie-breaker).
    pub index: usize,
    /// Pixel center (continuous; pixel `i` spans `[i, i+1)`).
    pub center: Vector2<f64>,
    /// Inverse 2D covariance (the "conic").
    pub conic: Matrix2<f64>,
    /// 2D covariance (px²), after dilation.
    pub cov: Matrix2<f64>,
    /// Camera-space depth.
    pub depth: f64,
    /// Opacity.
    pub opacity: f64,
    /// DC color.
    pub color: [f32; 3],
    /// Pixel bounding box (inclusive): 3σ of the larger eigenvalue.
    pub bbox: [i64; 4],
}

/// EWA projection of one Gaussian. `None` if behind the near plane,
/// transparent, or off-screen.
pub fn project(
    g: &Gaussian,
    index: usize,
    cam: &CameraPose,
    cfg: &RenderConfig,
) -> Option<Splat2D> {
    if (g.opacity as f64) < 1.0 / 255.0 {
        return None;
    }
    let t = cam.world_to_camera(&g.mean_f64());
    if t.z < NEAR {
        return None;
    }
    let k = &cam.intrinsics;
    let z = t.z;
    let lim_x = JACOBIAN_FOV_CLAMP * k.width as f64 / (2.0 * k.fx);
    let lim_y = JACOBIAN_FOV_CLAMP * k.height as f64 / (2.0 * k.fy);
    let x = (t.x / z).clamp(-lim_x, lim_x) * z;
    let y = (t.y / z).clamp(-lim_y, lim_y) * z;
    // Jacobian of (fx·x/z + cx, fy·y/z + cy) with respect to (x, y, z).
    let j = Matrix2x3::new(
        k.fx / z,
        0.0,
        -k.fx * x / (z * z),
        0.0,
        k.fy / z,
        -k.fy * y / (z * z),
    );
    let w: Matrix3<f64> = cam.rot_cw.to_rotation_matrix().into_inner();
    let cov3 = w * g.covariance() * w.transpose();
    let mut cov = j * cov3 * j.transpose();
    cov[(0, 0)] += cfg.dilation;
    cov[(1, 1)] += cfg.dilation;
    let conic = cov.try_inverse()?;
    // The true (unclamped) center.
    let center = Vector2::new(k.fx * t.x / z + k.cx, k.fy * t.y / z + k.cy);
    // Largest eigenvalue of the symmetric 2×2 covariance.
    let mid = 0.5 * (cov[(0, 0)] + cov[(1, 1)]);
    let det = cov.determinant();
    let lambda = mid + (mid * mid - det).max(0.0).sqrt();
    let r = 3.0 * lambda.sqrt();
    let bbox = [
        (center.x - r).floor() as i64,
        (center.y - r).floor() as i64,
        (center.x + r).ceil() as i64,
        (center.y + r).ceil() as i64,
    ];
    if bbox[2] < 0 || bbox[3] < 0 || bbox[0] >= k.width as i64 || bbox[1] >= k.height as i64 {
        return None;
    }
    Some(Splat2D {
        index,
        center,
        conic,
        cov,
        depth: z,
        opacity: g.opacity as f64,
        color: g.color_rgb(),
        bbox,
    })
}

/// Renders `gaussians` from `cam`.
pub fn render(gaussians: &[Gaussian], cam: &CameraPose, cfg: &RenderConfig) -> Rendered {
    let (w, h) = (
        cam.intrinsics.width as usize,
        cam.intrinsics.height as usize,
    );
    let splats: Vec<Splat2D> = gaussians
        .par_iter()
        .enumerate()
        .filter_map(|(i, g)| project(g, i, cam, cfg))
        .collect();

    // Bin splat indices into tiles.
    let (tx, ty) = (w.div_ceil(TILE), h.div_ceil(TILE));
    let mut bins: Vec<Vec<u32>> = vec![Vec::new(); tx * ty];
    for (si, s) in splats.iter().enumerate() {
        let x0 = (s.bbox[0].max(0) as usize) / TILE;
        let y0 = (s.bbox[1].max(0) as usize) / TILE;
        let x1 = ((s.bbox[2].min(w as i64 - 1)) as usize) / TILE;
        let y1 = ((s.bbox[3].min(h as i64 - 1)) as usize) / TILE;
        for by in y0..=y1 {
            for bx in x0..=x1 {
                bins[by * tx + bx].push(si as u32);
            }
        }
    }

    // Each tile is rendered independently (deterministic in parallel).
    let tiles: Vec<(usize, Vec<Shaded>)> = bins
        .into_par_iter()
        .enumerate()
        .map(|(ti, mut list)| {
            list.sort_by(|a, b| {
                let (sa, sb) = (&splats[*a as usize], &splats[*b as usize]);
                sa.depth.total_cmp(&sb.depth).then(sa.index.cmp(&sb.index))
            });
            let (bx, by) = (ti % tx, ti / tx);
            let mut px = Vec::with_capacity(TILE * TILE);
            for y in by * TILE..((by + 1) * TILE).min(h) {
                for x in bx * TILE..((bx + 1) * TILE).min(w) {
                    px.push(shade(&splats, &list, x, y, cfg));
                }
            }
            (ti, px)
        })
        .collect();

    let mut out = Rendered {
        width: w,
        height: h,
        rgb: vec![[0.0; 3]; w * h],
        alpha: vec![0.0; w * h],
        depth: vec![f32::INFINITY; w * h],
    };
    for (ti, px) in tiles {
        let (bx, by) = (ti % tx, ti / tx);
        let mut it = px.into_iter();
        for y in by * TILE..((by + 1) * TILE).min(h) {
            for x in bx * TILE..((bx + 1) * TILE).min(w) {
                if let Some((c, a, d)) = it.next() {
                    out.rgb[y * w + x] = c;
                    out.alpha[y * w + x] = a;
                    out.depth[y * w + x] = d;
                }
            }
        }
    }
    out
}

/// Front-to-back compositing at pixel `(x, y)`; returns (color, alpha, depth).
fn shade(
    splats: &[Splat2D],
    list: &[u32],
    x: usize,
    y: usize,
    cfg: &RenderConfig,
) -> ([f32; 3], f32, f32) {
    let p = Vector2::new(x as f64 + 0.5, y as f64 + 0.5);
    let mut t = 1.0f64;
    let mut c = [0.0f64; 3];
    let mut depth_acc = 0.0f64;
    for &si in list {
        let s = &splats[si as usize];
        let d = p - s.center;
        let power = -0.5 * (d.transpose() * s.conic * d)[(0, 0)];
        if power > 0.0 {
            continue; // numerical guard: conic should be positive definite
        }
        let alpha = (s.opacity * power.exp()).min(0.99);
        if alpha < 1.0 / 255.0 {
            continue;
        }
        let wgt = t * alpha;
        for (acc, col) in c.iter_mut().zip(s.color) {
            *acc += wgt * col as f64;
        }
        depth_acc += wgt * s.depth;
        t *= 1.0 - alpha;
        if t < 1e-4 {
            break;
        }
    }
    let a = 1.0 - t;
    let color = [0, 1, 2].map(|k| (c[k] + t * cfg.background[k] as f64) as f32);
    let depth = if a > 1e-6 {
        (depth_acc / a) as f32
    } else {
        f32::INFINITY
    };
    (color, a as f32, depth)
}

/// Peak signal-to-noise ratio (dB) between two equally sized RGB images in `0..=1`.
/// Returns `None` if sizes differ or the images are empty; `∞` if identical.
pub fn psnr(a: &[[f32; 3]], b: &[[f32; 3]]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mse: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (0..3).map(|k| ((x[k] - y[k]) as f64).powi(2)).sum::<f64>())
        .sum::<f64>()
        / (3 * a.len()) as f64;
    Some(if mse == 0.0 {
        f64::INFINITY
    } else {
        -10.0 * mse.log10()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use splat_types::nalgebra::{UnitQuaternion, Vector3};
    use splat_types::{PinholeIntrinsics, Vec3, Vec3f};

    fn cam() -> CameraPose {
        let k = PinholeIntrinsics {
            fx: 200.0,
            fy: 200.0,
            cx: 64.0,
            cy: 48.0,
            width: 128,
            height: 96,
        };
        // At the origin looking down +x... use look_at for clarity.
        CameraPose::look_at("c", Vec3::new(-5.0, 0.0, 0.0), Vec3::zeros(), Vec3::z(), k).unwrap()
    }

    #[test]
    fn single_gaussian_projects_to_expected_ellipse() {
        // Anisotropic Gaussian at distance 5: σ_y = 0.05 (image x), σ_z = 0.02 (image y).
        let g = Gaussian {
            mean: Vec3f::zeros(),
            scale: Vec3f::new(0.01, 0.05, 0.02),
            rot: UnitQuaternion::identity(),
            opacity: 0.8,
            sh_dc: [0.0; 3],
        }
        .with_color([1.0, 1.0, 1.0]);
        let c = cam();
        let s = project(&g, 0, &c, &RenderConfig::default()).unwrap();
        // Expected: (f·σ/d)² + 0.3.
        let ex = (200.0f64 * 0.05 / 5.0).powi(2) + 0.3;
        let ey = (200.0f64 * 0.02 / 5.0).powi(2) + 0.3;
        assert!((s.cov[(0, 0)] - ex).abs() < 1e-6, "{}", s.cov);
        assert!((s.cov[(1, 1)] - ey).abs() < 1e-6);
        assert!(s.cov[(0, 1)].abs() < 1e-9);
        assert!((s.center - Vector2::new(64.0, 48.0)).norm() < 1e-9);

        // The rendered alpha follows the same ellipse.
        let r = render(&[g], &c, &RenderConfig::default());
        let at = |x: usize, y: usize| r.alpha[y * r.width + x] as f64;
        let alpha_model = |dx: f64, dy: f64| 0.8 * (-0.5 * (dx * dx / ex + dy * dy / ey)).exp();
        // Pixel (63, 47) has center (63.5, 47.5): offset (−0.5, −0.5).
        assert!((at(63, 47) - alpha_model(-0.5, -0.5)).abs() < 1e-5);
        assert!((at(66, 47) - alpha_model(2.5, -0.5)).abs() < 1e-5);
        assert!((at(63, 49) - alpha_model(-0.5, 1.5)).abs() < 1e-5);
        // Expected depth of a lone splat is its depth.
        assert!((r.depth[47 * r.width + 63] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn rotated_gaussian_has_correlated_screen_covariance() {
        let g = Gaussian {
            mean: Vec3f::zeros(),
            scale: Vec3f::new(0.01, 0.1, 0.01),
            rot: UnitQuaternion::from_axis_angle(&Vector3::x_axis(), std::f32::consts::FRAC_PI_4),
            opacity: 0.8,
            sh_dc: [0.0; 3],
        };
        let s = project(&g, 0, &cam(), &RenderConfig::default()).unwrap();
        // 45° in the image plane: equal variances, strong correlation.
        // Relative tolerance: the rotation is stored as an f32 quaternion (~1e-7).
        assert!(
            (s.cov[(0, 0)] - s.cov[(1, 1)]).abs() < 1e-5 * s.cov[(0, 0)],
            "{}",
            s.cov
        );
        assert!(s.cov[(0, 1)].abs() > 0.4 * s.cov[(0, 0)]);
    }

    #[test]
    fn two_gaussian_ordering() {
        let c = cam();
        let red =
            Gaussian::isotropic(Vec3f::new(-1.0, 0.0, 0.0), 0.2, 0.99).with_color([1.0, 0.0, 0.0]);
        let blue =
            Gaussian::isotropic(Vec3f::new(1.0, 0.0, 0.0), 0.2, 0.99).with_color([0.0, 0.0, 1.0]);
        let center = 48 * 128 + 64;
        for list in [vec![red, blue], vec![blue, red]] {
            let r = render(&list, &c, &RenderConfig::default());
            let px = r.rgb[center];
            assert!(px[0] > 0.9 && px[2] < 0.05, "red must be in front: {px:?}");
            assert!((r.depth[center] - 4.0).abs() < 0.05);
        }
        // Move red behind blue: blue wins.
        let mut red2 = red;
        red2.mean.x = 2.0;
        let r = render(&[red2, blue], &c, &RenderConfig::default());
        assert!(r.rgb[center][2] > 0.9);
    }

    #[test]
    fn culls_behind_and_transparent() {
        let c = cam();
        let behind = Gaussian::isotropic(Vec3f::new(-6.0, 0.0, 0.0), 0.1, 1.0);
        // Just in front of the camera but far off to the side: must not smear.
        let side = Gaussian::isotropic(Vec3f::new(-4.7, 0.0, -3.0), 0.05, 1.0);
        let faint = Gaussian::isotropic(Vec3f::zeros(), 0.1, 0.001);
        assert!(project(&behind, 0, &c, &RenderConfig::default()).is_none());
        assert!(project(&faint, 0, &c, &RenderConfig::default()).is_none());
        let r = render(&[behind, faint, side], &c, &RenderConfig::default());
        assert!(r.alpha.iter().all(|a| *a == 0.0));
        assert!(psnr(&r.rgb, &r.rgb).unwrap().is_infinite());
        assert!(psnr(&r.rgb, &[]).is_none());
    }

    #[test]
    fn renders_synthetic_car_deterministically() {
        let s = synth::car_scene(&synth::CarConfig::default());
        let cam = &s.cameras[5];
        let a = render(&s.gaussians, cam, &RenderConfig::default());
        let b = render(&s.gaussians, cam, &RenderConfig::default());
        assert_eq!(a, b);
        // The car (target) projects to the image center and is opaque there.
        let center = (cam.intrinsics.height as usize / 2) * a.width + a.width / 2;
        assert!(a.alpha[center] > 0.99);
        let d = (cam.center() - s.target).norm();
        assert!(
            (a.depth[center] as f64) < d,
            "surface is in front of the target point"
        );
    }
}
