//! `splatplan flight`: the path the camera actually took, recovered from
//! every frame of the capture video.
//!
//! `scripts/register_all.sh` registers all video frames into the COLMAP
//! model the splat was trained from, with the keyframes held fixed, so the
//! poses are already in the splat's frame. This applies the capture's
//! alignment and drops bad poses with two checks:
//!
//! - **Render vs. video**: the splat rendered from the pose must correlate
//!   with the video frame.
//! - **Speed**: in rooms with repeated features (identical desks), a few
//!   frames land in the matching spot across the room. The largest
//!   time-ordered subset that never moves faster than `max_speed` is kept.
//!
//! Missing stretches are reported as gaps and never interpolated.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use splat_types::{CameraPose, Gaussian, Report, Similarity, Vec3};

use crate::capture::Capture;
use crate::provenance::Provenance;

/// File name of the flight report.
pub const FLIGHT_JSON: &str = "flight.json";

/// Flight-path settings.
pub struct FlightConfig {
    /// Text model with all registered frames (`work/flight/model_txt`).
    pub model: PathBuf,
    /// Frame rate of `frames_all/` (ingest extracts 10 fps).
    pub fps: f64,
    /// Fastest believable camera speed (m/s).
    pub max_speed: f64,
    /// Position noise allowed on top of `max_speed` (m).
    pub slack_m: f64,
    /// A longer stretch without a pose is a gap (s).
    pub gap_s: f64,
    /// Width (px) of the renders compared with the video frames; 0 skips
    /// the photometric check.
    pub check_width: u32,
    /// Frames whose render-vs-video correlation (median over ±0.25 s) is
    /// lower are dropped: the splat does not look like the video from there.
    pub min_match: f64,
}

/// A continuous stretch of recovered poses.
#[derive(Debug, Serialize, Deserialize)]
pub struct Segment {
    /// First and last frame time (s).
    pub t0: f64,
    pub t1: f64,
    /// Frames in it.
    pub frames: usize,
    /// Polyline length (m).
    pub length_m: f64,
}

/// A stretch of frames dropped by either check.
#[derive(Debug, Serialize, Deserialize)]
pub struct Rejected {
    /// First and last frame time (s).
    pub t0: f64,
    pub t1: f64,
    /// Frames in it.
    pub frames: usize,
    /// Farthest any of them was from the kept path at the nearest kept time (m).
    pub max_offset_m: f64,
    /// Keyframes (frames the splat was trained on) among them.
    pub keyframes: usize,
    /// Median render-vs-video correlation of these frames (NaN if unchecked).
    pub median_match: f64,
    /// Why: `"render mismatch"` (the splat seen from these poses does not
    /// look like the video) or `"impossible jump"` (too fast to be real).
    pub reason: String,
}

/// `flight.json`.
#[derive(Debug, Serialize, Deserialize)]
pub struct FlightReport {
    /// Capture name.
    pub name: String,
    /// True if lengths are meters.
    pub metric: bool,
    /// The alignment applied to the COLMAP poses (must match the splat's).
    pub alignment: Similarity,
    /// Frame rate of the extracted video frames.
    pub fps: f64,
    /// Video frames extracted.
    pub video_frames: usize,
    /// Keyframes the splat was trained on.
    pub keyframes: usize,
    /// Frames COLMAP registered (keyframes included).
    pub registered: usize,
    /// Frames kept after both checks.
    pub kept: usize,
    /// Frames dropped because their render did not match the video.
    pub mismatched: usize,
    /// Correlation threshold used (NaN if the check was skipped).
    pub min_match: f64,
    /// Largest distance between a keyframe in the flight model and the same
    /// keyframe in the splat's model, after alignment (m). Near zero when the
    /// registrator left the keyframes alone.
    pub keyframe_residual_m: f64,
    /// Speed limit used by the consistency check (m/s).
    pub max_speed_mps: f64,
    /// Gap threshold (s).
    pub gap_s: f64,
    /// Video length covered by a pose, within `gap_s` (s).
    pub covered_s: f64,
    /// Video length (s).
    pub duration_s: f64,
    /// Total length of the recovered segments (m).
    pub length_m: f64,
    /// Median and 95th-percentile speed between kept frames (m/s).
    pub median_speed_mps: f64,
    pub p95_speed_mps: f64,
    /// Lowest and highest camera height (m above the floor).
    pub z_range_m: [f64; 2],
    /// Vertical field of view of the video (degrees).
    pub fov_deg: f64,
    /// Continuous stretches.
    pub segments: Vec<Segment>,
    /// Stretches with no pose: `[t0, t1]`, last known to next known time (s).
    pub gaps: Vec<[f64; 2]>,
    /// Stretches of frames dropped by either check.
    pub rejected: Vec<Rejected>,
    /// Kept poses: `[t, x, y, z, fx, fy, fz, ux, uy, uz, key, match]`:
    /// position, viewing direction, image-up direction, 1 for a keyframe,
    /// and the render-vs-video correlation (`photo_score`; -2 if unchecked).
    pub poses: Vec<[f64; 12]>,
    /// How it was made.
    pub provenance: Provenance,
}

impl Report for FlightReport {}

/// Reads `dir/flight.json` and checks it was made in exactly the frame
/// `align` puts this capture's splat in (same capture, same similarity).
pub fn load_checked(dir: &Path, capture: &str, align: &Similarity) -> Result<FlightReport> {
    let f = dir.join(FLIGHT_JSON);
    let fr: FlightReport = serde_json::from_str(
        &std::fs::read_to_string(&f)
            .with_context(|| format!("reading {} (run `splatplan flight` first)", f.display()))?,
    )
    .with_context(|| format!("parsing {}", f.display()))?;
    let (a, b) = (fr.alignment, *align);
    let same = fr.name == capture
        && (a.scale - b.scale).abs() <= 1e-9 * b.scale
        && a.rot.angle_to(&b.rot) <= 1e-9
        && (a.trans - b.trans).norm() <= 1e-9 * (1.0 + b.trans.norm());
    if !same {
        bail!(
            "{} was made with a different alignment than {capture}'s splat (rescaled or \
             re-leveled since?): rerun `splatplan flight`",
            f.display()
        );
    }
    Ok(fr)
}

struct Frame {
    t: f64,
    pos: Vec3,
    fwd: Vec3,
    up: Vec3,
    key: bool,
    /// Correlation between the splat rendered at this pose and the video
    /// frame (NaN if not checked).
    score: f64,
}

/// Renders the splat at `pose` (`width` px wide) and returns the normalized
/// cross-correlation of its brightness with the video frame `img`. Near 1:
/// the splat looks like the video from here. Near 0: it doesn't.
fn photo_score(gs: &[Gaussian], pose: &CameraPose, img: &Path, width: u32) -> Result<f64> {
    let k = pose.intrinsics;
    let mut cam = pose.clone();
    cam.intrinsics = k.rescaled(f64::from(width) / f64::from(k.width));
    let r = gs_render_cpu::render(gs, &cam, &gs_render_cpu::RenderConfig::default());
    let frame = image::open(img)
        .with_context(|| format!("reading {}", img.display()))?
        .to_rgb8();
    let frame = image::imageops::resize(
        &frame,
        r.width as u32,
        r.height as u32,
        image::imageops::FilterType::Triangle,
    );
    let luma = |c: [f32; 3]| 0.299 * c[0] as f64 + 0.587 * c[1] as f64 + 0.114 * c[2] as f64;
    let a: Vec<f64> = r.rgb.iter().map(|&c| luma(c)).collect();
    let b: Vec<f64> = frame
        .pixels()
        .map(|p| {
            luma([
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
            ])
        })
        .collect();
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(&b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    Ok(if saa > 0.0 && sbb > 0.0 {
        sab / (saa * sbb).sqrt()
    } else {
        0.0
    })
}

/// Frame number from an ffmpeg-style name (`00042.jpg` → 42).
fn frame_number(name: &str) -> Option<u64> {
    Path::new(name).file_stem()?.to_str()?.parse().ok()
}

/// Indices of the largest time-ordered subset in which every step between
/// neighbors is at most `max_speed · dt + slack`. `frames` is sorted by time.
#[cfg(test)]
fn consistent(frames: &[Frame], max_speed: f64, slack: f64) -> Vec<usize> {
    consistent_refs(&frames.iter().collect::<Vec<_>>(), max_speed, slack)
}

fn consistent_refs(frames: &[&Frame], max_speed: f64, slack: f64) -> Vec<usize> {
    let n = frames.len();
    let mut best = vec![1usize; n];
    let mut prev = vec![usize::MAX; n];
    for j in 0..n {
        for i in 0..j {
            let dt = frames[j].t - frames[i].t;
            if best[i] + 1 > best[j]
                && (frames[j].pos - frames[i].pos).norm() <= max_speed * dt + slack
            {
                best[j] = best[i] + 1;
                prev[j] = i;
            }
        }
    }
    let Some(mut j) = (0..n).max_by_key(|&j| (best[j], std::cmp::Reverse(j))) else {
        return Vec::new();
    };
    let mut out = vec![j];
    while prev[j] != usize::MAX {
        j = prev[j];
        out.push(j);
    }
    out.reverse();
    out
}

fn percentile(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[((p / 100.0) * (v.len() - 1) as f64).round() as usize]
}

/// Builds the flight report.
pub fn run(dir: &Path, raw_splat: bool, cfg: &FlightConfig) -> Result<FlightReport> {
    if !(cfg.fps > 0.0 && cfg.max_speed > 0.0 && cfg.slack_m >= 0.0 && cfg.gap_s > 0.0) {
        bail!("fps, max speed and gap must be > 0 and slack >= 0");
    }
    let capture = Capture::open(dir, raw_splat)?;
    let (scene, align) = capture.load_aligned()?;
    let metric = scene.scale.is_metric();
    let unit = if metric {
        1.0
    } else {
        gs_map::length_unit_guess(&scene)
    };
    let sim = align.similarity;

    // Intrinsics do not matter for poses; reuse the splat model's camera so
    // the distorted camera in the flight model need not be parsed.
    let cams = colmap_io::parse_cameras(
        &std::fs::read_to_string(capture.colmap.join("cameras.txt"))
            .context("reading the splat model's cameras.txt")?,
    )?;
    let k = *cams.values().next().context("no camera in cameras.txt")?;
    let text = std::fs::read_to_string(cfg.model.join("images.txt")).with_context(|| {
        format!(
            "reading {} (run scripts/register_all.sh first)",
            cfg.model.join("images.txt").display()
        )
    })?;
    // Every camera id in the file gets those intrinsics (2D-point lines may
    // add spurious ids; they are never looked up).
    let map: std::collections::BTreeMap<u32, _> = text
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .filter_map(|l| l.split_whitespace().nth(8)?.parse().ok())
        .map(|id| (id, k))
        .collect();
    let images = colmap_io::parse_images(&text, &map)?;

    // The keyframes must sit exactly where the splat's model has them.
    let splat_cams: HashMap<&str, Vec3> = scene
        .cameras
        .iter()
        .map(|c| (c.name.as_str(), c.center()))
        .collect();
    let mut residual: f64 = 0.0;
    let mut frames = Vec::new();
    let scores: Vec<f64> = if cfg.check_width > 0 {
        images
            .par_iter()
            .map(|im| {
                photo_score(
                    &scene.gaussians,
                    &sim.apply_camera(&im.pose),
                    &dir.join("frames_all").join(&im.pose.name),
                    cfg.check_width,
                )
            })
            .collect::<Result<_>>()?
    } else {
        vec![f64::NAN; images.len()]
    };
    for (im, &score) in images.iter().zip(&scores) {
        let pose = sim.apply_camera(&im.pose);
        let key = splat_cams.get(im.pose.name.as_str());
        if let Some(c) = key {
            residual = residual.max((pose.center() - c).norm() / unit);
        }
        let n = frame_number(&im.pose.name)
            .with_context(|| format!("frame name {} is not a number", im.pose.name))?;
        frames.push(Frame {
            t: (n.saturating_sub(1)) as f64 / cfg.fps,
            pos: pose.center() / unit,
            fwd: pose.forward(),
            up: -pose.camera_dir_to_world(&Vec3::y()),
            key: key.is_some(),
            score,
        });
    }
    let matched = frames.iter().filter(|f| f.key).count();
    if matched * 2 < scene.cameras.len() {
        bail!(
            "only {matched} of the splat's {} keyframes are in {}: wrong model?",
            scene.cameras.len(),
            cfg.model.display()
        );
    }
    if residual > 0.01 {
        bail!(
            "keyframes moved by up to {residual:.3} m between the splat's model and {}: \
             the flight model was re-adjusted and is no longer in the splat's frame",
            cfg.model.display()
        );
    }
    frames.sort_by(|a, b| a.t.total_cmp(&b.t));

    // Photometric check, smoothed over ±0.25 s so one blurred frame does not
    // punch a hole in a good stretch.
    let smoothed: Vec<f64> = frames
        .iter()
        .map(|f| {
            let mut v: Vec<f64> = frames
                .iter()
                .filter(|g| (g.t - f.t).abs() <= 0.25 + 1e-9)
                .map(|g| g.score)
                .collect();
            percentile(&mut v, 50.0)
        })
        .collect();
    let looks_right: Vec<bool> = smoothed
        .iter()
        .map(|&m| m.is_nan() || m >= cfg.min_match) // NaN: unchecked
        .collect();
    let candidates: Vec<usize> = (0..frames.len()).filter(|&i| looks_right[i]).collect();
    let sub: Vec<&Frame> = candidates.iter().map(|&i| &frames[i]).collect();
    let keep: Vec<usize> = consistent_refs(&sub, cfg.max_speed, cfg.slack_m)
        .into_iter()
        .map(|j| candidates[j])
        .collect();
    let kept: Vec<&Frame> = keep.iter().map(|&i| &frames[i]).collect();

    // Rejected stretches: runs of frames not in `keep`.
    let mut is_kept = vec![false; frames.len()];
    for &i in &keep {
        is_kept[i] = true;
    }
    let near_kept = |t: f64| -> Option<Vec3> {
        kept.iter()
            .min_by(|a, b| (a.t - t).abs().total_cmp(&(b.t - t).abs()))
            .map(|f| f.pos)
    };
    let mut rejected: Vec<Rejected> = Vec::new();
    let mut i = 0;
    while i < frames.len() {
        if is_kept[i] {
            i += 1;
            continue;
        }
        let start = i;
        while i < frames.len() && !is_kept[i] {
            i += 1;
        }
        let run = &frames[start..i];
        rejected.push(Rejected {
            t0: run[0].t,
            t1: run[run.len() - 1].t,
            frames: run.len(),
            max_offset_m: run
                .iter()
                .filter_map(|f| near_kept(f.t).map(|p| (f.pos - p).norm()))
                .fold(0.0, f64::max),
            keyframes: run.iter().filter(|f| f.key).count(),
            median_match: percentile(&mut run.iter().map(|f| f.score).collect::<Vec<_>>(), 50.0),
            reason: if looks_right[start..i].iter().filter(|&&ok| !ok).count() * 2 >= run.len() {
                "render mismatch".into()
            } else {
                "impossible jump".into()
            },
        });
    }

    // Segments and gaps.
    let mut segments = Vec::new();
    let mut gaps = Vec::new();
    let mut speeds = Vec::new();
    let video_frames = std::fs::read_dir(dir.join("frames_all"))
        .map(|d| {
            d.filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|x| x == "jpg"))
                .count()
        })
        .unwrap_or(0);
    let duration_s = (video_frames.max(1) - 1) as f64 / cfg.fps;
    let mut covered = 0.0;
    let mut s0 = 0;
    for j in 0..kept.len() {
        let end = j + 1 == kept.len() || kept[j + 1].t - kept[j].t > cfg.gap_s;
        if j > s0 {
            let (a, b) = (kept[j - 1], kept[j]);
            speeds.push((b.pos - a.pos).norm() / (b.t - a.t));
        }
        if end {
            let seg = &kept[s0..=j];
            segments.push(Segment {
                t0: seg[0].t,
                t1: seg[seg.len() - 1].t,
                frames: seg.len(),
                length_m: seg.windows(2).map(|w| (w[1].pos - w[0].pos).norm()).sum(),
            });
            covered += seg[seg.len() - 1].t - seg[0].t + 1.0 / cfg.fps;
            if j + 1 < kept.len() {
                gaps.push([kept[j].t, kept[j + 1].t]);
            }
            s0 = j + 1;
        }
    }
    if let Some(f) = kept.first().filter(|f| f.t > cfg.gap_s) {
        gaps.insert(0, [0.0, f.t]);
    }
    if let Some(l) = kept.last().filter(|l| duration_s - l.t > cfg.gap_s) {
        gaps.push([l.t, duration_s]);
    }
    let z: Vec<f64> = kept.iter().map(|f| f.pos.z).collect();
    let r = |v: f64| (v * 1e4).round() / 1e4;
    let poses = kept
        .iter()
        .map(|f| {
            [
                r(f.t),
                r(f.pos.x),
                r(f.pos.y),
                r(f.pos.z),
                r(f.fwd.x),
                r(f.fwd.y),
                r(f.fwd.z),
                r(f.up.x),
                r(f.up.y),
                r(f.up.z),
                if f.key { 1.0 } else { 0.0 },
                if f.score.is_finite() {
                    r(f.score)
                } else {
                    -2.0
                },
            ]
        })
        .collect();
    Ok(FlightReport {
        name: capture.name.clone(),
        metric,
        alignment: sim,
        fps: cfg.fps,
        video_frames,
        keyframes: scene.cameras.len(),
        registered: frames.len(),
        kept: kept.len(),
        mismatched: looks_right.iter().filter(|&&ok| !ok).count(),
        min_match: if cfg.check_width > 0 {
            cfg.min_match
        } else {
            f64::NAN
        },
        keyframe_residual_m: residual,
        max_speed_mps: cfg.max_speed,
        gap_s: cfg.gap_s,
        covered_s: covered.min(duration_s),
        duration_s,
        length_m: segments.iter().map(|s| s.length_m).sum(),
        median_speed_mps: percentile(&mut speeds.clone(), 50.0),
        p95_speed_mps: percentile(&mut speeds, 95.0),
        z_range_m: [
            z.iter().copied().fold(f64::INFINITY, f64::min),
            z.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        ],
        fov_deg: (2.0 * (f64::from(k.height) / (2.0 * k.fy)).atan()).to_degrees(),
        segments,
        gaps,
        rejected,
        poses,
        provenance: Provenance::capture(&format!(
            "splatplan flight {} --model {} --fps {} --max-speed {} --slack {} --gap {} --check-width {} --min-match {}{}",
            dir.display(),
            cfg.model.display(),
            cfg.fps,
            cfg.max_speed,
            cfg.slack_m,
            cfg.gap_s,
            cfg.check_width,
            cfg.min_match,
            if raw_splat { " --raw-splat" } else { "" }
        )),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(t: f64, x: f64) -> Frame {
        Frame {
            t,
            pos: Vec3::new(x, 0.0, 1.0),
            fwd: Vec3::x(),
            up: Vec3::z(),
            key: false,
            score: f64::NAN,
        }
    }

    #[test]
    fn frame_numbers() {
        assert_eq!(frame_number("00042.jpg"), Some(42));
        assert_eq!(frame_number("a.jpg"), None);
    }

    /// A flight at 0.5 m/s with a two-frame excursion 8 m away: the excursion
    /// goes, everything else stays.
    #[test]
    fn drops_aliased_jumps() {
        let mut frames: Vec<Frame> = (0..30)
            .map(|i| f(i as f64 * 0.1, i as f64 * 0.05))
            .collect();
        frames[12].pos.x += 8.0;
        frames[13].pos.x += 8.0;
        let keep = consistent(&frames, 2.0, 0.1);
        assert_eq!(keep.len(), 28);
        assert!(!keep.contains(&12) && !keep.contains(&13));
    }

    /// After a long gap the camera may be anywhere within max_speed · gap.
    #[test]
    fn keeps_both_sides_of_a_gap() {
        let frames = vec![f(0.0, 0.0), f(0.1, 0.05), f(10.0, 6.0), f(10.1, 6.05)];
        assert_eq!(consistent(&frames, 2.0, 0.1), vec![0, 1, 2, 3]);
    }
}
