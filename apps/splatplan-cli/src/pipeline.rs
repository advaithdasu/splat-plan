//! `splatplan pipeline`: align → map → plan → verify on a capture, with reports.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use gs_align::AlignReport;
use gs_map::{Map, MapConfig, MapReport, VoxelState, build_map};
use gs_verify::{Verifier, VerifierConfig};
use planner::{PlanOutcome, PlannerConfig, Workspace, check_endpoint, plan_and_certify};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::Serialize;
use splat_types::{Report, Scale, Scene, Vec3};

use crate::capture::Capture;
use crate::clean::CleanRecord;
use crate::provenance::Provenance;
use crate::viz::write_slice;

/// Drone size, shared by the planner and the verifier so the two always
/// agree. `None` keeps the library defaults.
#[derive(Debug, Clone, Copy, Default)]
pub struct Robot {
    /// Robot radius (m).
    pub radius: Option<f64>,
    /// Safety margin on top of the radius (m).
    pub margin: Option<f64>,
}

impl Robot {
    /// Planner settings for this robot.
    pub fn planner(&self) -> PlannerConfig {
        let d = PlannerConfig::default();
        PlannerConfig {
            robot_radius: self.radius.unwrap_or(d.robot_radius),
            margin: self.margin.unwrap_or(d.margin),
            ..d
        }
    }

    /// Verifier settings for this robot.
    pub fn verifier(&self) -> VerifierConfig {
        let d = VerifierConfig::default();
        VerifierConfig {
            robot_radius: self.radius.unwrap_or(d.robot_radius),
            margin: self.margin.unwrap_or(d.margin),
            ..d
        }
    }

    /// The flags that reproduce this robot, for provenance.
    pub fn flags(&self) -> String {
        let mut s = String::new();
        if let Some(r) = self.radius {
            s += &format!(" --robot-radius {r}");
        }
        if let Some(m) = self.margin {
            s += &format!(" --margin {m}");
        }
        s
    }
}

/// A scene ready for planning: aligned, mapped, and a verifier built.
pub struct Prepared {
    /// Aligned scene.
    pub scene: Scene,
    /// Alignment report.
    pub align: AlignReport,
    /// Collision map.
    pub map: Map,
    /// Map report.
    pub map_report: MapReport,
    /// Independent verifier.
    pub verifier: Verifier,
}

/// Loads, aligns and maps a capture.
pub fn prepare(capture: &Capture, map_cfg: &MapConfig, robot: &Robot) -> Result<Prepared> {
    let (scene, align) = capture.load_aligned()?;
    let (map, map_report) = build_map(&scene, map_cfg)?;
    let verifier = Verifier::new(
        &scene.gaussians,
        &scene.cameras,
        scene.scale,
        map.length_unit,
        robot.verifier(),
    );
    Ok(Prepared {
        scene,
        align,
        map,
        map_report,
        verifier,
    })
}

/// Region for random queries: a cylinder around the scene center, inside
/// the camera orbit (at 85% of the median camera distance), from
/// 0.5 m up to 2.5 m (or the median camera height if lower).
pub fn query_cylinder(scene: &Scene, unit: f64) -> (f64, f64, f64) {
    let mut radial: Vec<f64> = scene
        .cameras
        .iter()
        .map(|c| c.center().xy().norm())
        .collect();
    let mut heights: Vec<f64> = scene.cameras.iter().map(|c| c.center().z).collect();
    radial.sort_by(f64::total_cmp);
    heights.sort_by(f64::total_cmp);
    let med = |v: &[f64]| v.get(v.len() / 2).copied().unwrap_or(0.0);
    let radius = 0.85 * med(&radial);
    let z_lo = 0.5 * unit;
    let z_hi = (2.5 * unit).min(med(&heights)).max(z_lo + 0.5 * unit);
    (radius, z_lo, z_hi)
}

/// `n` seeded random query pairs in the query cylinder.
pub fn random_queries(scene: &Scene, unit: f64, seed: u64, n: usize) -> Vec<(Vec3, Vec3)> {
    let (r, z0, z1) = query_cylinder(scene, unit);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut point = || loop {
        let x = rng.gen_range(-r..=r);
        let y = rng.gen_range(-r..=r);
        let z = rng.gen_range(z0..=z1);
        if x * x + y * y <= r * r {
            return Vec3::new(x, y, z);
        }
    };
    (0..n).map(|_| (point(), point())).collect()
}

/// True if `p` is a usable endpoint for the whole system: the planner's
/// endpoint check passes, and at `p` the verifier's own checks pass too
/// (clearance with one voxel to spare, so a keep-out from verifier feedback
/// cannot later invalidate it; near a camera; seen by the cameras).
pub(crate) fn system_endpoint_ok(ws: &Workspace, verifier: &Verifier, need: f64, p: &Vec3) -> bool {
    let vc = verifier.config();
    let unit = ws.map.length_unit;
    check_endpoint(ws, p, need, "start").is_ok()
        && verifier.clearance_lower_bound(p).0 >= need + ws.map.spec.res
        && verifier.support_distance(p) <= vc.max_support_dist * unit
        && verifier.body_observed(p)
}

/// Height band for valid endpoints (m above the ground): drone flight height.
const VALID_Z: (f64, f64) = (0.5, 2.5);

/// Voxel centers in the valid height band that pass the planner's endpoint
/// check: the pool both the benchmark and the showcase draw endpoints from.
pub(crate) fn endpoint_candidates(map: &Map, ws: &Workspace, need: f64) -> Vec<Vec3> {
    let (z0, z1) = (VALID_Z.0 * map.length_unit, VALID_Z.1 * map.length_unit);
    // `dist` at a voxel center bounds the clearance there from above, so
    // this cheap test only skips voxels that would fail anyway.
    (0..map.spec.len())
        .filter(|&i| {
            map.state[i] == VoxelState::FreeObserved && map.dist[i] as f64 >= need + map.spec.res
        })
        .map(|i| map.spec.center(map.spec.coords(i)))
        .filter(|c| c.z >= z0 && c.z <= z1 && check_endpoint(ws, c, need, "start").is_ok())
        .collect()
}

/// `n` seeded query pairs whose endpoints both pass the planner's and the
/// verifier's endpoint checks, so the benchmark measures whether the planner
/// can connect valid endpoints.
///
/// Endpoints come from [`endpoint_candidates`] rather than a fixed region:
/// in a fly-through capture the free space is a thin band along the camera
/// path. Pairs with no valid endpoint after 2000 draws are skipped.
pub fn valid_queries(p: &Prepared, cfg: &PlannerConfig, seed: u64, n: usize) -> Vec<(Vec3, Vec3)> {
    let map = &p.map;
    let ws = Workspace::new(map);
    let need = cfg.search_clearance(map);
    let candidates = endpoint_candidates(map, &ws, need);
    if candidates.is_empty() {
        return Vec::new();
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x5eed);
    let mut point = || {
        for _ in 0..2000 {
            let q = candidates[rng.gen_range(0..candidates.len())];
            if system_endpoint_ok(&ws, &p.verifier, need, &q) {
                return Some(q);
            }
        }
        None
    };
    // In a small valid region both endpoints can land on the same voxel;
    // draw the goal again rather than ask for a zero-length plan.
    (0..n)
        .filter_map(|_| {
            let a = point()?;
            (0..20)
                .find_map(|_| point().filter(|b| *b != a))
                .map(|b| (a, b))
        })
        .collect()
}

/// Aggregate statistics over a batch of plan attempts.
#[derive(Debug, Clone, Serialize, Default)]
pub struct PlanStats {
    /// Queries attempted.
    pub queries: usize,
    /// Queries that produced a certified trajectory.
    pub successes: usize,
    /// `successes / queries`.
    pub success_rate: f64,
    /// Refusals by reason.
    pub refusals: BTreeMap<String, usize>,
    /// Certificates by verdict (successful queries only).
    pub verdicts: BTreeMap<String, usize>,
    /// Smallest verifier clearance over all certified trajectories (m or units).
    pub min_verified_clearance: Option<f64>,
    /// Clearance the verifier required (r + margin + v·dt/2).
    pub margin_required: Option<f64>,
    /// Largest speed over certified trajectories.
    pub max_speed: Option<f64>,
    /// Largest acceleration over certified trajectories.
    pub max_accel: Option<f64>,
    /// Plan + verify wall time per query (ms): mean.
    pub plan_ms_mean: f64,
    /// Plan + verify wall time per query (ms): max.
    pub plan_ms_max: f64,
    /// Mean verify-repair attempts per success.
    pub mean_attempts: f64,
    /// Mean length of the certified paths (m or units; waypoint polyline).
    pub path_length_mean: Option<f64>,
    /// Longest certified path (m or units).
    pub path_length_max: Option<f64>,
}

/// Plans all queries (in parallel; results are combined in input order, so
/// the statistics are deterministic apart from the timings).
pub fn run_queries(
    p: &Prepared,
    queries: &[(Vec3, Vec3)],
    cfg: &PlannerConfig,
) -> (PlanStats, Vec<Option<PlanOutcome>>) {
    let results: Vec<(Result<PlanOutcome, planner::PlanError>, f64)> = queries
        .par_iter()
        .map(|(a, b)| {
            let t = Instant::now();
            let r = plan_and_certify(&p.map, &p.verifier, a, b, cfg);
            (r, t.elapsed().as_secs_f64() * 1e3)
        })
        .collect();
    let mut st = PlanStats {
        queries: queries.len(),
        ..PlanStats::default()
    };
    let mut attempts = 0usize;
    let mut total_len = 0.0;
    let mut outcomes = Vec::with_capacity(results.len());
    for (r, ms) in results {
        st.plan_ms_mean += ms / queries.len().max(1) as f64;
        st.plan_ms_max = st.plan_ms_max.max(ms);
        match r {
            Ok(o) => {
                st.successes += 1;
                attempts += o.attempts;
                let c = &o.certificate;
                let v = format!("{:?}", c.verdict).to_uppercase();
                *st.verdicts.entry(v).or_default() += 1;
                let upd = |x: &mut Option<f64>, v: f64, min: bool| {
                    *x = Some(match *x {
                        None => v,
                        Some(o) if min => o.min(v),
                        Some(o) => o.max(v),
                    })
                };
                upd(&mut st.min_verified_clearance, c.min_clearance_m, true);
                upd(&mut st.margin_required, c.margin_required_m, false);
                upd(&mut st.max_speed, c.max_speed, false);
                upd(&mut st.max_accel, c.max_accel, false);
                let len = o
                    .waypoints
                    .windows(2)
                    .map(|w| (w[1] - w[0]).norm())
                    .sum::<f64>()
                    / p.map.length_unit;
                total_len += len;
                upd(&mut st.path_length_max, len, false);
                outcomes.push(Some(o));
            }
            Err(e) => {
                *st.refusals.entry(e.kind().to_string()).or_default() += 1;
                outcomes.push(None);
            }
        }
    }
    st.success_rate = st.successes as f64 / st.queries.max(1) as f64;
    st.mean_attempts = attempts as f64 / st.successes.max(1) as f64;
    st.path_length_mean = (st.successes > 0).then(|| total_len / st.successes as f64);
    (st, outcomes)
}

/// Everything `splatplan pipeline` reports.
#[derive(Debug, Serialize)]
pub struct PipelineReport {
    /// Capture name.
    pub capture: String,
    /// True if the capture was generated by `synth` (has `ground_truth.json`).
    pub synthetic: bool,
    /// The splat file used, and whether it was cleaned.
    pub splat: String,
    /// The cleaning record, if the cleaned splat was used.
    pub cleaning: Option<CleanRecord>,
    /// Scale state; UNSCALED captures can never be certified SAFE.
    pub scale: Scale,
    /// Banner repeated in the report when unscaled.
    pub warning: Option<String>,
    /// Alignment.
    pub align: AlignReport,
    /// Map build.
    pub map: MapReport,
    /// Planning statistics on uniformly random endpoints (many invalid by design).
    pub plans_random: PlanStats,
    /// Planning statistics on endpoints sampled from traversable space.
    pub plans_valid_endpoints: PlanStats,
    /// Planner settings.
    pub planner: PlannerConfig,
    /// Total wall time (ms).
    pub total_ms: f64,
    /// How this report was produced.
    pub provenance: Provenance,
}

impl Report for PipelineReport {}

/// Runs the pipeline on a capture and writes everything to `out`.
pub fn run(
    capture_dir: &Path,
    raw_splat: bool,
    robot: &Robot,
    out: &Path,
    n_queries: usize,
    seed: u64,
) -> Result<PipelineReport> {
    let t0 = Instant::now();
    let capture = Capture::open(capture_dir, raw_splat)?;
    std::fs::create_dir_all(out)?;
    let p = prepare(&capture, &MapConfig::default(), robot)?;
    let cfg = robot.planner();
    let queries = random_queries(&p.scene, p.map.length_unit, seed, n_queries);
    let (stats, _) = run_queries(&p, &queries, &cfg);
    let valid = valid_queries(&p, &cfg, seed, n_queries);
    let (valid_stats, mut outcomes) = run_queries(&p, &valid, &cfg);
    for o in outcomes.iter_mut().flatten() {
        o.certificate.splat = Some(capture.splat_label());
    }

    // Artifacts: first certified plan, and a slice with every certified path.
    let paths: Vec<Vec<Vec3>> = outcomes
        .iter()
        .flatten()
        .map(|o| o.trajectory.sample(0.05).iter().map(|s| s.pos).collect())
        .collect();
    let z = 1.0 * p.map.length_unit;
    write_slice(&p.map, z, &paths, &out.join("slice_z1.0m.png"))?;
    if let Some(o) = outcomes.iter().flatten().next() {
        std::fs::write(
            out.join("trajectory.json"),
            serde_json::to_string_pretty(&o.trajectory)?,
        )?;
        std::fs::write(out.join("certificate.json"), o.certificate.to_json()?)?;
        std::fs::write(
            out.join("waypoints.json"),
            serde_json::to_string_pretty(&o.waypoints)?,
        )?;
    }
    // Every certified plan, numbered by query, so each can be re-verified
    // or shown on its own. Stale files from an earlier run are removed first.
    let all = out.join("certified");
    if all.exists() {
        std::fs::remove_dir_all(&all)?;
    }
    for (i, o) in outcomes.iter().enumerate() {
        let Some(o) = o else { continue };
        let dir = all.join(format!("{i:03}"));
        std::fs::create_dir_all(&dir)?;
        std::fs::write(
            dir.join("trajectory.json"),
            serde_json::to_string_pretty(&o.trajectory)?,
        )?;
        std::fs::write(dir.join("certificate.json"), o.certificate.to_json()?)?;
        std::fs::write(
            dir.join("waypoints.json"),
            serde_json::to_string_pretty(&o.waypoints)?,
        )?;
    }
    let report = PipelineReport {
        capture: capture.name.clone(),
        synthetic: capture.dir.join("ground_truth.json").exists(),
        splat: capture.splat_label(),
        cleaning: capture.cleaning.clone(),
        scale: p.scene.scale,
        warning: (!p.scene.scale.is_metric()).then(|| Scale::UNSCALED_BANNER.to_string()),
        align: p.align,
        map: p.map_report,
        plans_random: stats,
        plans_valid_endpoints: valid_stats,
        planner: cfg,
        total_ms: t0.elapsed().as_secs_f64() * 1e3,
        provenance: Provenance::capture(&format!(
            "splatplan pipeline {} --out {} --queries {n_queries} --seed {seed}{}{}",
            capture_dir.display(),
            out.display(),
            if raw_splat { " --raw-splat" } else { "" },
            robot.flags()
        )),
    };
    std::fs::write(out.join("pipeline.json"), report.to_json()?)?;
    for (label, st) in [
        ("random endpoints", &report.plans_random),
        ("valid endpoints", &report.plans_valid_endpoints),
    ] {
        println!(
            "{} [{label}]: {}/{} certified, verdicts {:?}, refusals {:?}{}",
            report.capture,
            st.successes,
            st.queries,
            st.verdicts,
            st.refusals,
            match (st.path_length_mean, st.path_length_max) {
                (Some(m), Some(x)) => format!(", path length mean {m:.2} max {x:.2}"),
                _ => String::new(),
            }
        );
    }
    if report.warning.is_some() {
        eprintln!("\n{}\n", Scale::UNSCALED_BANNER);
    }
    Ok(report)
}
