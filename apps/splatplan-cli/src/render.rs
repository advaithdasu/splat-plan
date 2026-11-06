//! `splatplan render` (flythrough frames) and `splatplan psnr` (renderer check).

use std::path::Path;

use anyhow::{Context, Result, bail};
use gs_render_cpu::{CameraPath, RenderConfig, Rendered, camera_path_along, psnr, render};
use image::RgbImage;
use serde::Serialize;
use splat_types::{PolyTrajectory, Report, Scene, Vec3};

use crate::provenance::Provenance;

fn to_image(r: &Rendered) -> Result<RgbImage> {
    RgbImage::from_raw(r.width as u32, r.height as u32, r.rgb8())
        .context("image buffer size mismatch")
}

/// Renders a flythrough along `traj`: `out/frames/NNNNN.png` + `out/camera_path.json`.
pub fn flythrough(
    scene: &Scene,
    traj: &PolyTrajectory,
    out: &Path,
    fps: f64,
    width: u32,
    look_at: Option<Vec3>,
) -> Result<CameraPath> {
    let first = scene
        .cameras
        .first()
        .context("scene has no cameras for intrinsics")?;
    let k = first
        .intrinsics
        .rescaled(width as f64 / first.intrinsics.width as f64);
    let path = camera_path_along(traj, fps, k, look_at, 10.0);
    let frames = out.join("frames");
    std::fs::create_dir_all(&frames)?;
    std::fs::write(
        out.join("camera_path.json"),
        serde_json::to_string_pretty(&path)?,
    )?;
    let cfg = RenderConfig {
        background: [0.75, 0.82, 0.92],
        ..RenderConfig::default()
    };
    for i in 0..path.frames.len() {
        let Some(cam) = path.pose(i) else { continue };
        let r = render(&scene.gaussians, &cam, &cfg);
        to_image(&r)?.save(frames.join(format!("{i:05}.png")))?;
    }
    Ok(path)
}

/// PSNR of renders against the training photos.
#[derive(Debug, Serialize)]
pub struct PsnrReport {
    /// Images compared.
    pub images: usize,
    /// Mean PSNR (dB).
    pub mean_db: f64,
    /// Per-image `(name, dB)`.
    pub per_image: Vec<(String, f64)>,
    /// Data label (real captures only).
    pub data: String,
    /// How it was produced.
    pub provenance: Provenance,
}

impl Report for PsnrReport {}

/// Renders up to `max` training cameras and compares them with the photos in `images_dir`.
pub fn psnr_check(
    scene: &Scene,
    images_dir: &Path,
    max: usize,
    command: &str,
) -> Result<PsnrReport> {
    let mut per = Vec::new();
    let step = (scene.cameras.len() / max.max(1)).max(1);
    for cam in scene.cameras.iter().step_by(step).take(max) {
        let path = images_dir.join(&cam.name);
        let Ok(photo) = image::open(&path) else {
            continue;
        };
        let photo = photo.to_rgb8();
        if photo.width() != cam.intrinsics.width || photo.height() != cam.intrinsics.height {
            bail!(
                "{}: photo is {}x{}, camera is {}x{} (use the undistorted images)",
                path.display(),
                photo.width(),
                photo.height(),
                cam.intrinsics.width,
                cam.intrinsics.height
            );
        }
        let r = render(&scene.gaussians, cam, &RenderConfig::default());
        let p: Vec<[f32; 3]> = photo
            .pixels()
            .map(|px| px.0.map(|v| v as f32 / 255.0))
            .collect();
        if let Some(db) = psnr(&r.rgb, &p) {
            per.push((cam.name.clone(), db));
        }
    }
    if per.is_empty() {
        bail!("no training images found in {}", images_dir.display());
    }
    let mean_db = per.iter().map(|(_, d)| d).sum::<f64>() / per.len() as f64;
    Ok(PsnrReport {
        images: per.len(),
        mean_db,
        per_image: per,
        data: "real".into(),
        provenance: Provenance::capture(command),
    })
}
