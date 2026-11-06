//! `splatplan export`: a capture's splat and its certified paths, packaged
//! for the web viewer in `web/`.
//!
//! Writes, into the output directory:
//!
//! - `scene.splat`: the aligned splat (meters, +Z up, ground at z = 0) in
//!   the common 32-byte `.splat` layout, cropped to the region the cameras
//!   covered (display only; planning used the whole splat).
//! - `paths.json`: certified trajectories, sampled for drawing and fly-through.
//! - `scene.json`: what the files are and where they came from.
//!
//! Every exported path is re-verified against exactly this capture's
//! Gaussians before it is written. If one is no longer `SAFE` (the splat was
//! re-cleaned or retrained since planning), the export refuses.
//!
//! At most [`MAX_SHOWN_PATHS`] paths are written, ranked by [`ExportedPath::score`]:
//! long paths that go around many obstacles first.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use gs_verify::{Verdict, Verifier};
use rayon::prelude::*;
use serde::Serialize;
use splat_types::{Gaussian, PolyTrajectory, Report, Scene, Vec3};

use crate::capture::Capture;
use crate::pipeline::Robot;
use crate::provenance::Provenance;

/// Export settings.
pub struct ExportConfig {
    /// Keep Gaussians within this horizontal distance (m) of some camera.
    pub crop_m: f64,
    /// Directories with `certified/<id>/trajectory.json` (pipeline, replan,
    /// showcase or weave outputs); empty for a splat-only scene.
    pub paths: Vec<PathBuf>,
    /// Only these path ids (the `NNN` directory names); empty means all.
    pub only: Vec<String>,
    /// Title shown in the viewer.
    pub title: String,
    /// Directory with a `flight.json` from `splatplan flight`, if any.
    pub flight: Option<PathBuf>,
    /// Keep the best this many paths (at most [`MAX_SHOWN_PATHS`]).
    pub max_paths: usize,
}

/// The viewer never shows more planned paths than this.
pub const MAX_SHOWN_PATHS: usize = 5;

/// One exported path.
#[derive(Debug, Serialize)]
pub struct ExportedPath {
    /// Directory name under `certified/`.
    pub id: String,
    /// Length of the sampled trajectory (m).
    pub length_m: f64,
    /// Flight time (s).
    pub duration_s: f64,
    /// Verdict of the re-verification at export time.
    pub verdict: String,
    /// Smallest clearance to any Gaussian along the path (m).
    pub min_clearance_m: f64,
    /// Clearance the verifier required (m).
    pub required_m: f64,
    /// `[t, x, y, z]` every 0.1 s, plus the final point.
    pub samples: Vec<[f64; 4]>,
    /// The training camera that sees most of this path (set after loading).
    pub view: Option<HomeView>,
    /// What blocks the straight line, for showcase paths (`showcase.json`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avoidance: Option<Avoidance>,
    /// How a weave tour threads between obstacles (`weave.json`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weave: Option<Weave>,
    /// For paths between real flight points (`replan.json`): which ones,
    /// and what the drone actually did between them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub real: Option<RealTrip>,
    /// Ranking score, see [`ExportedPath::score`].
    pub score: f64,
}

/// A planned path between two places the drone really was.
#[derive(Debug, Serialize)]
pub struct RealTrip {
    /// `start`, `end` or `hover at m:ss`.
    pub from_label: String,
    pub to_label: String,
    /// How far each end was moved to a valid endpoint (m).
    pub from_snap_m: f64,
    pub to_snap_m: f64,
    /// Where the drone really was at each end (m, aligned frame). The
    /// viewer draws the way from there to the planned path as uncertified.
    pub from_xyz: [f64; 3],
    pub to_xyz: [f64; 3],
    /// Video time the drone left the start and reached the goal (s).
    pub t0: f64,
    pub t1: f64,
    /// Length of the recovered flight between those times (m) and time
    /// spent in gaps, where it is unknown (s).
    pub flown_m: f64,
    pub unknown_s: f64,
}

/// Real-trip info by path id, from `dir/replan.json`.
fn load_real(dir: &Path) -> Result<std::collections::HashMap<String, RealTrip>> {
    let mut out = std::collections::HashMap::new();
    let f = dir.join("replan.json");
    if !f.exists() {
        return Ok(out);
    }
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&f)?)
        .with_context(|| format!("parsing {}", f.display()))?;
    let xyz = |k: Option<u64>| -> Option<[f64; 3]> {
        let r = &v["points"][k? as usize]["real"];
        Some([r[0].as_f64()?, r[1].as_f64()?, r[2].as_f64()?])
    };
    for p in v["paths"].as_array().into_iter().flatten() {
        let num = |v: &serde_json::Value| v.as_f64();
        let fl = &p["flown"];
        if let (
            Some(id),
            Some(from),
            Some(to),
            Some(fs),
            Some(ts),
            Some(t0),
            Some(t1),
            Some(m),
            Some(u),
            Some(fx),
            Some(tx),
        ) = (
            p["id"].as_str(),
            p["from_label"].as_str(),
            p["to_label"].as_str(),
            num(&p["from_snap_m"]),
            num(&p["to_snap_m"]),
            num(&fl["t0"]),
            num(&fl["t1"]),
            num(&fl["length_m"]),
            num(&fl["unknown_s"]),
            xyz(p["from"].as_u64()),
            xyz(p["to"].as_u64()),
        ) {
            out.insert(
                id.to_string(),
                RealTrip {
                    from_label: from.into(),
                    to_label: to.into(),
                    from_snap_m: fs,
                    to_snap_m: ts,
                    from_xyz: fx,
                    to_xyz: tx,
                    t0,
                    t1,
                    flown_m: m,
                    unknown_s: u,
                },
            );
        }
    }
    Ok(out)
}

impl ExportedPath {
    /// Obstacles the path goes around: the ones blocking its straight line
    /// (showcase) or the ones it threads between (weave); 0 if unknown.
    pub fn obstacles_avoided(&self) -> u64 {
        match (&self.avoidance, &self.weave) {
            (Some(a), _) => a.obstacles,
            (None, Some(w)) => w.obstacles_threaded,
            (None, None) => 0,
        }
    }

    /// Length times (1 + obstacles avoided): long paths that weave around
    /// things beat short ones and straight ones.
    pub fn score(&self) -> f64 {
        self.length_m * (1 + self.obstacles_avoided()) as f64
    }
}

/// Sorts paths best first (ties by id) and keeps the top `max`.
fn rank_paths(paths: &mut Vec<ExportedPath>, max: usize) {
    for p in paths.iter_mut() {
        p.score = p.score();
    }
    paths.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
    paths.truncate(max);
}

/// How a weave tour threads between obstacles.
#[derive(Debug, Serialize)]
pub struct Weave {
    /// Gaps between two obstacles the tour flies through.
    pub gates: u64,
    /// Distinct obstacles (walls not counted) the gaps lie between.
    pub obstacles_threaded: u64,
    /// Distinct obstacles (walls not counted) within 0.6 m of the drone's body.
    pub obstacles_passed: u64,
    /// Highest the drone's center was allowed to fly (m above the floor).
    pub ceiling_m: f64,
}

/// Weave measures by path id, from `dir/weave.json`.
fn load_weave(dir: &Path) -> Result<std::collections::HashMap<String, Weave>> {
    let mut out = std::collections::HashMap::new();
    let f = dir.join("weave.json");
    if !f.exists() {
        return Ok(out);
    }
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&f)?)
        .with_context(|| format!("parsing {}", f.display()))?;
    let ceiling_m = v["config"]["ceiling_m"].as_f64().unwrap_or(f64::NAN);
    for p in v["paths"].as_array().into_iter().flatten() {
        if let (Some(id), Some(gates), Some(obstacles_threaded), Some(obstacles_passed)) = (
            p["id"].as_str(),
            p["gates"].as_array().map(|g| g.len() as u64),
            p["obstacles_threaded"].as_u64(),
            p["obstacles_passed"].as_u64(),
        ) {
            out.insert(
                id.to_string(),
                Weave {
                    gates,
                    obstacles_threaded,
                    obstacles_passed,
                    ceiling_m,
                },
            );
        }
    }
    Ok(out)
}

/// How a showcase path goes around what blocks its straight line.
#[derive(Debug, Serialize)]
pub struct Avoidance {
    /// Straight-line distance from start to goal (m).
    pub straight_m: f64,
    /// Separate solid pieces the straight line passes through.
    pub obstacles: u64,
    /// How much of the straight line lies inside solid voxels (m).
    pub blocked_m: f64,
    /// Farthest the path swings sideways off the straight line (m).
    pub side_offset_m: f64,
}

/// Obstacle measures by path id, from `dir/showcase.json` or `dir/replan.json`.
fn load_avoidance(dir: &Path) -> Result<std::collections::HashMap<String, Avoidance>> {
    let mut out = std::collections::HashMap::new();
    let Some(f) = ["showcase.json", "replan.json"]
        .iter()
        .map(|n| dir.join(n))
        .find(|f| f.exists())
    else {
        return Ok(out);
    };
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&f)?)
        .with_context(|| format!("parsing {}", f.display()))?;
    for p in v["paths"].as_array().into_iter().flatten() {
        let num = |k: &str| p[k].as_f64();
        if let (Some(id), Some(straight_m), Some(obstacles), Some(blocked_m), Some(side_offset_m)) = (
            p["id"].as_str(),
            num("straight_m"),
            p["obstacles"].as_u64(),
            num("blocked_m"),
            num("side_offset_m"),
        ) {
            out.insert(
                id.to_string(),
                Avoidance {
                    straight_m,
                    obstacles,
                    blocked_m,
                    side_offset_m,
                },
            );
        }
    }
    Ok(out)
}

/// The viewer's starting view: one of the training cameras, where a splat
/// looks its best.
#[derive(Debug, Serialize)]
pub struct HomeView {
    /// Camera position (m).
    pub position: [f64; 3],
    /// A point 2 m in front of it.
    pub target: [f64; 3],
    /// Vertical field of view (degrees), from the camera intrinsics.
    pub fov_deg: f64,
    /// Image name of the chosen camera.
    pub camera: String,
}

/// `scene.json`.
#[derive(Debug, Serialize)]
pub struct SceneInfo {
    /// Capture name.
    pub name: String,
    /// Viewer title.
    pub title: String,
    /// True if lengths are meters.
    pub metric: bool,
    /// How the scale was obtained.
    pub scale_method: String,
    /// Splat used for planning (and whether it was cleaned).
    pub splat: String,
    /// Gaussians in that splat.
    pub gaussians_total: usize,
    /// Gaussians written to `scene.splat` after cropping.
    pub gaussians_exported: usize,
    /// The crop, in words.
    pub crop: String,
    /// Camera centers (aligned frame).
    pub cameras: Vec<[f64; 3]>,
    /// Starting view.
    pub home: Option<HomeView>,
    /// Drone radius (m).
    pub robot_radius_m: f64,
    /// Safety margin (m).
    pub margin_m: f64,
    /// Exported paths (count).
    pub paths: usize,
    /// True if `flight.json` (the camera's recovered flight path) was written.
    pub flight: bool,
    /// How this export was produced.
    pub provenance: Provenance,
}

impl Report for SceneInfo {}

/// Writes one Gaussian in the 32-byte `.splat` layout: position (3 × f32),
/// linear scale (3 × f32), RGBA (4 × u8, alpha = opacity), rotation (4 × u8,
/// `w, x, y, z` mapped from [-1, 1] to [0, 255]). Little-endian.
fn push_splat(buf: &mut Vec<u8>, g: &Gaussian) {
    for v in g.mean.iter().chain(g.scale.iter()) {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    let c = g.color_rgb();
    buf.extend_from_slice(&[byte(c[0]), byte(c[1]), byte(c[2]), byte(g.opacity)]);
    let q = g.rot.into_inner();
    for v in [q.w, q.i, q.j, q.k] {
        buf.push((v * 128.0 + 128.0).round().clamp(0.0, 255.0) as u8);
    }
}

fn percentile(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    v[((p / 100.0) * (v.len() - 1) as f64).round() as usize]
}

/// Gaussians kept for display: within `crop` (horizontally) of a point in
/// `around` (camera positions), and between the 1st and 99th height
/// percentiles (padded by 0.25 m).
fn crop(scene: &Scene, cams: &[Vec3], crop: f64) -> Vec<Gaussian> {
    let mut z: Vec<f64> = scene.gaussians.iter().map(|g| g.mean.z as f64).collect();
    let (z0, z1) = (
        percentile(&mut z, 1.0) - 0.25,
        percentile(&mut z, 99.0) + 0.25,
    );
    let r2 = crop * crop;
    scene
        .gaussians
        .par_iter()
        .filter(|g| {
            let p = g.mean_f64();
            p.z >= z0
                && p.z <= z1
                && cams
                    .iter()
                    .any(|c| (c.x - p.x).powi(2) + (c.y - p.y).powi(2) <= r2)
        })
        .copied()
        .collect()
}

/// The training camera that sees the most of `points` (inside the image,
/// 0.3 to 8 m in front). Ties keep the first camera, so it is deterministic.
fn home_view(scene: &Scene, points: &[Vec3], unit: f64) -> Option<HomeView> {
    let (_, cam) = scene
        .cameras
        .iter()
        .map(|cam| {
            let k = &cam.intrinsics;
            let n = points
                .iter()
                .filter(|p| {
                    cam.project(p).is_some_and(|(u, v, z)| {
                        z > 0.3 * unit
                            && z < 8.0 * unit
                            && u >= 0.0
                            && v >= 0.0
                            && u < f64::from(k.width)
                            && v < f64::from(k.height)
                    })
                })
                .count();
            (n, cam)
        })
        .fold(None, |best: Option<(usize, _)>, (n, cam)| match best {
            Some((b, _)) if b >= n => best,
            _ => Some((n, cam)),
        })?;
    let c = cam.center() / unit;
    let k = &cam.intrinsics;
    Some(HomeView {
        position: c.into(),
        target: (c + cam.forward() * 2.0).into(),
        fov_deg: (2.0 * (f64::from(k.height) / (2.0 * k.fy)).atan()).to_degrees(),
        camera: cam.name.clone(),
    })
}

/// Loads, re-verifies and samples the certified paths in `dir/certified/`.
fn load_paths(
    dir: &Path,
    only: &[String],
    verifier: &Verifier,
    unit: f64,
) -> Result<Vec<ExportedPath>> {
    let root = dir.join("certified");
    let mut ids: Vec<String> = std::fs::read_dir(&root)
        .with_context(|| {
            format!(
                "reading {} (run `splatplan replan` or `pipeline` first)",
                root.display()
            )
        })?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("trajectory.json").exists())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    ids.sort();
    if !only.is_empty() {
        ids.retain(|id| only.contains(id));
    }
    let mut avoidance = load_avoidance(dir)?;
    let mut weave = load_weave(dir)?;
    let mut real = load_real(dir)?;
    let mut out = Vec::new();
    for id in ids {
        let f = root.join(&id).join("trajectory.json");
        let t: PolyTrajectory = serde_json::from_str(&std::fs::read_to_string(&f)?)
            .with_context(|| format!("parsing {}", f.display()))?;
        let cert = verifier.certify(&t);
        if cert.verdict != Verdict::Safe {
            bail!(
                "path {id} is {:?} against this capture's splat ({} violations); \
                 the splat changed since planning: plan again",
                cert.verdict,
                cert.violation_count
            );
        }
        let mut samples: Vec<[f64; 4]> = t
            .sample(0.1)
            .iter()
            .map(|s| [s.t, s.pos.x / unit, s.pos.y / unit, s.pos.z / unit])
            .collect();
        // `sample` may stop short of the end: always include the last point.
        let end = t.sample(t.duration()).last().map(|s| s.pos);
        let short = samples.last().is_some_and(|l| l[0] < t.duration());
        if let (Some(e), true) = (end, short) {
            samples.push([t.duration(), e.x / unit, e.y / unit, e.z / unit]);
        }
        let length_m = samples
            .windows(2)
            .map(|w| Vec3::new(w[1][1] - w[0][1], w[1][2] - w[0][2], w[1][3] - w[0][3]).norm())
            .sum();
        let avoid = avoidance.remove(&id);
        let weave_info = weave.remove(&id);
        let real_trip = real.remove(&id);
        out.push(ExportedPath {
            id,
            length_m,
            duration_s: t.duration(),
            verdict: "SAFE".into(),
            min_clearance_m: cert.min_clearance_m,
            required_m: cert.margin_required_m,
            samples,
            view: None,
            avoidance: avoid,
            weave: weave_info,
            real: real_trip,
            score: 0.0,
        });
    }
    Ok(out)
}

/// Runs the export.
pub fn run(
    dir: &Path,
    raw_splat: bool,
    robot: &Robot,
    out: &Path,
    cfg: &ExportConfig,
) -> Result<SceneInfo> {
    if !(cfg.crop_m > 0.0 && cfg.crop_m.is_finite()) {
        bail!("crop must be > 0 m, got {}", cfg.crop_m);
    }
    if cfg.max_paths > MAX_SHOWN_PATHS {
        bail!(
            "--max-paths {} is more than the viewer shows ({MAX_SHOWN_PATHS})",
            cfg.max_paths
        );
    }
    if cfg.only.len() > cfg.max_paths {
        bail!(
            "--only names {} paths but at most {} are exported (--max-paths)",
            cfg.only.len(),
            cfg.max_paths
        );
    }
    let capture = Capture::open(dir, raw_splat)?;
    let (scene, align) = capture.load_aligned()?;
    let metric = scene.scale.is_metric();
    let unit = if metric {
        1.0
    } else {
        gs_map::length_unit_guess(&scene)
    };
    let mut paths = match cfg.paths.as_slice() {
        [] => Vec::new(),
        dirs => {
            if !metric {
                bail!(
                    "{} is UNSCALED: it has no certified paths to export",
                    capture.name
                );
            }
            let verifier = Verifier::new(
                &scene.gaussians,
                &scene.cameras,
                scene.scale,
                unit,
                robot.verifier(),
            );
            let mut all = Vec::new();
            for d in dirs {
                all.extend(load_paths(d, &cfg.only, &verifier, unit)?);
            }
            for id in &cfg.only {
                if !all.iter().any(|p| &p.id == id) {
                    bail!("no certified path {id} in {dirs:?}");
                }
            }
            let mut seen = std::collections::HashSet::new();
            if let Some(dup) = all.iter().find(|p| !seen.insert(p.id.clone())) {
                bail!("two --paths directories both have a path {}", dup.id);
            }
            rank_paths(&mut all, cfg.max_paths);
            all
        }
    };
    let flight = match &cfg.flight {
        None => None,
        Some(d) => Some(crate::flight::load_checked(
            d,
            &capture.name,
            &align.similarity,
        )?),
    };
    // Display crop: around the keyframes and, with a flight, every pose.
    let mut around: Vec<Vec3> = scene.cameras.iter().map(|c| c.center()).collect();
    if let Some(fr) = &flight {
        around.extend(
            fr.poses
                .iter()
                .step_by(5)
                .map(|p| Vec3::new(p[1], p[2], p[3]) * unit),
        );
    }
    let kept = crop(&scene, &around, cfg.crop_m * unit);
    // Start where the paths are, or where the most of the scene is in view.
    let focus: Vec<Vec3> = if paths.is_empty() {
        kept.iter().step_by(50).map(|g| g.mean_f64()).collect()
    } else {
        paths
            .iter()
            .flat_map(|p| p.samples.iter().map(|s| Vec3::new(s[1], s[2], s[3]) * unit))
            .collect()
    };
    let home = home_view(&scene, &focus, unit);
    for p in &mut paths {
        let pts: Vec<Vec3> = p
            .samples
            .iter()
            .map(|s| Vec3::new(s[1], s[2], s[3]) * unit)
            .collect();
        p.view = home_view(&scene, &pts, unit);
    }
    let mut buf = Vec::with_capacity(kept.len() * 32);
    for g in &kept {
        let mut g = *g;
        g.mean /= unit as f32;
        g.scale /= unit as f32;
        push_splat(&mut buf, &g);
    }
    std::fs::create_dir_all(out)?;
    std::fs::write(out.join("scene.splat"), &buf)?;
    std::fs::write(out.join("paths.json"), serde_json::to_string(&paths)?)?;
    match &flight {
        Some(fr) => std::fs::write(
            out.join(crate::flight::FLIGHT_JSON),
            serde_json::to_string(fr)?,
        )?,
        None if out.join(crate::flight::FLIGHT_JSON).exists() => {
            std::fs::remove_file(out.join(crate::flight::FLIGHT_JSON))?
        }
        None => {}
    }
    let vc = robot.verifier();
    let info = SceneInfo {
        name: capture.name.clone(),
        title: cfg.title.clone(),
        metric,
        scale_method: align.scale_method,
        splat: capture.splat_label(),
        gaussians_total: scene.gaussians.len(),
        gaussians_exported: kept.len(),
        crop: format!(
            "display only: Gaussians within {} m (horizontally) of a {} and between the 1st and 99th height percentiles",
            cfg.crop_m,
            if flight.is_some() {
                "keyframe or recovered flight pose"
            } else {
                "camera"
            }
        ),
        cameras: scene
            .cameras
            .iter()
            .map(|c| (c.center() / unit).into())
            .collect(),
        home,
        robot_radius_m: vc.robot_radius,
        margin_m: vc.margin,
        paths: paths.len(),
        flight: flight.is_some(),
        provenance: Provenance::capture(&format!(
            "splatplan export {} --out {} --crop {} --max-paths {}{}{}{}{}{}",
            dir.display(),
            out.display(),
            cfg.crop_m,
            cfg.max_paths,
            if cfg.paths.is_empty() {
                String::new()
            } else {
                format!(
                    " --paths {}",
                    cfg.paths
                        .iter()
                        .map(|p| p.display().to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                )
            },
            if cfg.only.is_empty() {
                String::new()
            } else {
                format!(" --only {}", cfg.only.join(","))
            },
            cfg.flight
                .as_ref()
                .map(|d| format!(" --flight {}", d.display()))
                .unwrap_or_default(),
            if raw_splat { " --raw-splat" } else { "" },
            robot.flags()
        )),
    };
    std::fs::write(out.join("scene.json"), info.to_json()?)?;
    Ok(info)
}

#[cfg(test)]
mod tests {
    use super::*;
    use splat_types::Vec3f;
    use splat_types::nalgebra::{UnitQuaternion, Vector3};

    #[test]
    fn splat_record_layout() {
        let g = Gaussian {
            mean: Vec3f::new(1.0, -2.0, 0.5),
            scale: Vec3f::new(0.1, 0.2, 0.3),
            rot: UnitQuaternion::from_axis_angle(&Vector3::z_axis(), std::f32::consts::FRAC_PI_2),
            opacity: 0.5,
            sh_dc: [0.0; 3],
        };
        let mut buf = Vec::new();
        push_splat(&mut buf, &g);
        assert_eq!(buf.len(), 32);
        let f = |i: usize| f32::from_le_bytes([buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]);
        assert_eq!((f(0), f(4), f(8)), (1.0, -2.0, 0.5));
        assert_eq!((f(12), f(16), f(20)), (0.1, 0.2, 0.3));
        // DC 0 is mid-gray; alpha is the opacity.
        assert_eq!(&buf[24..28], &[128, 128, 128, 128]);
        // 90° about z: w = x... = cos 45°, z = sin 45°.
        assert_eq!(&buf[28..32], &[219, 128, 128, 219]);
    }

    fn path(id: &str, length_m: f64, obstacles: Option<u64>) -> ExportedPath {
        ExportedPath {
            id: id.into(),
            length_m,
            duration_s: 1.0,
            verdict: "SAFE".into(),
            min_clearance_m: 0.3,
            required_m: 0.25,
            samples: Vec::new(),
            view: None,
            avoidance: obstacles.map(|obstacles| Avoidance {
                straight_m: 1.0,
                obstacles,
                blocked_m: 0.5,
                side_offset_m: 1.0,
            }),
            weave: None,
            real: None,
            score: 0.0,
        }
    }

    #[test]
    fn ranking_prefers_long_paths_around_obstacles() {
        let mut v = vec![
            path("a", 10.0, None),   // long but straight: 10
            path("b", 4.0, Some(3)), // short, weaves: 16
            path("c", 8.0, Some(2)), // 24
            path("d", 2.0, Some(1)), // 4
            path("e", 6.0, Some(1)), // 12
            path("f", 1.0, None),    // 1
            path("g", 3.0, Some(3)), // 12, ties with e: by id
        ];
        rank_paths(&mut v, MAX_SHOWN_PATHS);
        let ids: Vec<&str> = v.iter().map(|p| p.id.as_str()).collect();
        assert_eq!(ids, ["c", "b", "e", "g", "a"]);
        assert_eq!(v[0].score, 24.0);
    }
}
