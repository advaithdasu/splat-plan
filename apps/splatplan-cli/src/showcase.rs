//! `splatplan showcase`: a few long certified paths, picked for display.
//!
//! `pipeline` is the benchmark: 50 random start/goal pairs, most of them
//! close together, reported as statistics. This picks paths worth looking
//! at instead. It draws validated endpoints (the same checks the benchmark
//! uses), plans and verifies many far-apart pairs, and keeps the best few
//! whose straight line runs through an obstacle (the planner had to go
//! around it), ranked by `--rank`. Near-duplicates are dropped.
//!
//! Every path kept went through the same planner and independent verifier
//! as any other; only the choice of which queries to show is different.

use std::path::Path;

use anyhow::{Result, bail};
use gs_map::{Map, MapConfig, VoxelState};
use planner::PlanOutcome;
use planner::Workspace;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use splat_types::{Report, Vec3};

use crate::capture::Capture;
use crate::pipeline::{Robot, endpoint_candidates, prepare, run_queries, system_endpoint_ok};
use crate::provenance::Provenance;
use crate::viz::write_slice;

/// What to look for.
#[derive(Debug, Clone, Serialize)]
pub struct ShowcaseConfig {
    /// Paths to keep.
    pub count: usize,
    /// Start/goal pairs to plan and verify.
    pub tries: usize,
    /// Smallest straight-line distance between start and goal (m).
    pub min_dist_m: f64,
    /// Keep endpoints and whole paths inside this box (m): xmin, ymin, xmax, ymax.
    pub region: Option<[f64; 4]>,
    /// Endpoint height band (m above the floor); the path itself may climb or dip.
    pub height: Option<[f64; 2]>,
    /// Smallest length of straight line (m) that must run through solid voxels.
    pub min_blocked_m: f64,
    /// How to rank.
    pub rank: Rank,
    /// Two paths whose starts and goals are both within this distance (m) count as the same path.
    pub dedup_m: f64,
    /// Seed for endpoint and pair sampling.
    pub seed: u64,
}

/// Validated endpoints drawn before pairing them up.
const ENDPOINTS: usize = 400;

/// How a certified path relates to the straight line between its ends.
#[derive(Debug, Clone, Serialize)]
pub struct Candidate {
    /// Query index among the pairs tried.
    pub index: usize,
    /// Length of the certified path (m, waypoint polyline).
    pub length_m: f64,
    /// Straight-line distance from start to goal (m).
    pub straight_m: f64,
    /// `length / straight`.
    pub detour: f64,
    /// How much of the straight line lies inside occupied voxels (m).
    pub blocked_m: f64,
    /// Separate solid pieces the straight line passes through.
    pub obstacles: usize,
    /// Farthest the path gets from the straight line (m).
    pub max_offset_m: f64,
    /// Farthest it gets sideways, seen from above (m).
    pub side_offset_m: f64,
}

/// How to rank the candidates.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, clap::ValueEnum)]
pub enum Rank {
    /// Long and bending: length × detour.
    Length,
    /// Going around things: how far the path swings sideways off the
    /// straight line, times how much solid stuff that line would have hit.
    Avoidance,
}

impl Rank {
    fn score(self, c: &Candidate) -> f64 {
        match self {
            Rank::Length => c.length_m * c.detour,
            Rank::Avoidance => c.side_offset_m * c.blocked_m.min(2.0) * c.length_m.sqrt(),
        }
    }
}

/// One path that was kept.
#[derive(Debug, Clone, Serialize)]
pub struct ShowcasePath {
    /// Directory name under `certified/`.
    pub id: String,
    /// Length, detour and obstacle measures.
    #[serde(flatten)]
    pub measures: Candidate,
    /// Smallest clearance the verifier measured along the path (m).
    pub min_clearance_m: f64,
    /// Clearance the verifier required (m).
    pub required_m: f64,
    /// Start and goal (m, aligned frame).
    pub start: [f64; 3],
    /// Goal.
    pub goal: [f64; 3],
}

/// Everything `splatplan showcase` reports.
#[derive(Debug, Serialize)]
pub struct ShowcaseReport {
    /// Capture name.
    pub capture: String,
    /// The splat file used.
    pub splat: String,
    /// Settings.
    pub config: ShowcaseConfig,
    /// Validated endpoints found (inside the region, if any).
    pub endpoints: usize,
    /// Pairs planned and verified.
    pub pairs_tried: usize,
    /// Pairs that came back certified.
    pub certified: usize,
    /// Certified paths that stayed inside the region and whose straight line is blocked.
    pub candidates: usize,
    /// The paths kept, best first.
    pub paths: Vec<ShowcasePath>,
    /// Every certified, in-region path that was measured (for tuning the ranking).
    pub all_candidates: Vec<Candidate>,
    /// How this report was produced.
    pub provenance: Provenance,
}

impl Report for ShowcaseReport {}

fn in_region(region: &Option<[f64; 4]>, unit: f64, p: &Vec3) -> bool {
    match region {
        None => true,
        Some([x0, y0, x1, y1]) => {
            p.x >= x0 * unit && p.x <= x1 * unit && p.y >= y0 * unit && p.y <= y1 * unit
        }
    }
}

/// Length, detour and obstacle measures of a certified path from `a` to `b`.
pub fn measure(map: &Map, a: &Vec3, b: &Vec3, o: &PlanOutcome, index: usize) -> Candidate {
    let unit = map.length_unit;
    let (a, b) = (*a, *b);
    let samples = o.trajectory.sample(0.05);
    let length = o
        .waypoints
        .windows(2)
        .map(|w| (w[1] - w[0]).norm())
        .sum::<f64>()
        / unit;
    let straight = (b - a).norm() / unit;
    // Walk the straight line: how much of it is inside solid voxels, and
    // in how many separate pieces (each piece is an obstacle in the way).
    let n = ((b - a).norm() / (0.5 * map.spec.res)).ceil().max(1.0) as usize;
    let step = (b - a).norm() / n as f64 / unit;
    let (mut solid, mut pieces, mut inside) = (0.0, 0, false);
    for k in 0..=n {
        let occ = map.state_at(&(a + (b - a) * (k as f64 / n as f64))) == VoxelState::Occupied;
        if occ {
            solid += step;
            if !inside {
                pieces += 1;
            }
        }
        inside = occ;
    }
    // Farthest the flown path gets from the straight line.
    let ab = b - a;
    let max_offset = samples
        .iter()
        .map(|s| {
            let t = ((s.pos - a).dot(&ab) / ab.norm_squared()).clamp(0.0, 1.0);
            (s.pos - (a + ab * t)).norm()
        })
        .fold(0.0, f64::max)
        / unit;
    // The same, seen from above: swinging sideways around something,
    // as opposed to hopping over it.
    let ab2 = Vec3::new(ab.x, ab.y, 0.0);
    let side_offset = samples
        .iter()
        .map(|s| {
            let d = Vec3::new(s.pos.x - a.x, s.pos.y - a.y, 0.0);
            let t = if ab2.norm_squared() > 0.0 {
                (d.dot(&ab2) / ab2.norm_squared()).clamp(0.0, 1.0)
            } else {
                0.0
            };
            (d - ab2 * t).norm()
        })
        .fold(0.0, f64::max)
        / unit;
    Candidate {
        index,
        length_m: length,
        straight_m: straight,
        detour: length / straight,
        blocked_m: solid,
        obstacles: pieces,
        max_offset_m: max_offset,
        side_offset_m: side_offset,
    }
}

/// Picks long certified paths and writes them to `out/certified/NNN/`
/// (the layout `export` reads), with `showcase.json` and a map slice.
pub fn run(
    capture_dir: &Path,
    raw_splat: bool,
    robot: &Robot,
    out: &Path,
    cfg: &ShowcaseConfig,
) -> Result<ShowcaseReport> {
    let capture = Capture::open(capture_dir, raw_splat)?;
    let p = prepare(&capture, &MapConfig::default(), robot)?;
    if !p.scene.scale.is_metric() {
        bail!(
            "{} is UNSCALED: nothing can be certified, so there is nothing to show",
            capture.name
        );
    }
    let planner_cfg = robot.planner();
    let map = &p.map;
    let unit = map.length_unit;
    let ws = Workspace::new(map);
    let need = planner_cfg.search_clearance(map);
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed ^ 0x5c0c);

    // Endpoints: validated the same way as the benchmark's.
    let mut pool: Vec<Vec3> = endpoint_candidates(map, &ws, need)
        .into_iter()
        .filter(|c| in_region(&cfg.region, unit, c))
        .filter(|c| {
            cfg.height
                .is_none_or(|[lo, hi]| c.z >= lo * unit && c.z <= hi * unit)
        })
        .collect();
    pool.shuffle(&mut rng);
    let endpoints: Vec<Vec3> = pool
        .into_iter()
        .filter(|q| system_endpoint_ok(&ws, &p.verifier, need, q))
        .take(ENDPOINTS)
        .collect();

    // Far-apart pairs, a random subset of them.
    let min_d = cfg.min_dist_m * unit;
    let mut pairs: Vec<(Vec3, Vec3)> = Vec::new();
    for (i, a) in endpoints.iter().enumerate() {
        for b in &endpoints[i + 1..] {
            if (b - a).norm() >= min_d {
                pairs.push((*a, *b));
            }
        }
    }
    pairs.shuffle(&mut rng);
    pairs.truncate(cfg.tries);
    if pairs.is_empty() {
        bail!(
            "no endpoint pairs at least {} m apart ({} endpoints found)",
            cfg.min_dist_m,
            endpoints.len()
        );
    }

    let (_, outcomes) = run_queries(&p, &pairs, &planner_cfg);
    let certified = outcomes.iter().flatten().count();

    // Measure the certified paths that stay in the region.
    let mut cands: Vec<Candidate> = Vec::new();
    for (i, o) in outcomes.iter().enumerate() {
        let Some(o) = o else { continue };
        let samples = o.trajectory.sample(0.05);
        if !samples.iter().all(|s| in_region(&cfg.region, unit, &s.pos)) {
            continue;
        }
        let (a, b) = pairs[i];
        cands.push(measure(map, &a, &b, o, i));
    }
    // Only paths whose straight line runs through something solid: a long
    // straight run down an open aisle shows nothing about planning.
    let mut scored: Vec<Candidate> = cands
        .iter()
        .filter(|c| c.blocked_m >= cfg.min_blocked_m && c.obstacles > 0)
        .cloned()
        .collect();
    let candidates = scored.len();
    scored.sort_by(|x, y| {
        cfg.rank
            .score(y)
            .total_cmp(&cfg.rank.score(x))
            .then(x.index.cmp(&y.index))
    });

    let dedup = cfg.dedup_m * unit;
    let same = |(a1, b1): (Vec3, Vec3), (a2, b2): (Vec3, Vec3)| {
        ((a1 - a2).norm() < dedup && (b1 - b2).norm() < dedup)
            || ((a1 - b2).norm() < dedup && (b1 - a2).norm() < dedup)
    };
    let mut kept: Vec<Candidate> = Vec::new();
    for c in scored {
        if kept.len() == cfg.count {
            break;
        }
        if kept.iter().all(|k| !same(pairs[k.index], pairs[c.index])) {
            kept.push(c);
        }
    }

    // Write, best first, as certified/000, 001, ...
    let all = out.join("certified");
    if all.exists() {
        std::fs::remove_dir_all(&all)?;
    }
    let mut paths = Vec::new();
    let mut drawn = Vec::new();
    for (rank, c) in kept.iter().enumerate() {
        let Some(mut o) = outcomes[c.index].clone() else {
            continue;
        };
        o.certificate.splat = Some(capture.splat_label());
        let id = format!("{rank:03}");
        let dir = all.join(&id);
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
        let (a, b) = pairs[c.index];
        paths.push(ShowcasePath {
            id,
            measures: c.clone(),
            min_clearance_m: o.certificate.min_clearance_m,
            required_m: o.certificate.margin_required_m,
            start: [a.x / unit, a.y / unit, a.z / unit],
            goal: [b.x / unit, b.y / unit, b.z / unit],
        });
        drawn.push(o.trajectory.sample(0.05).iter().map(|s| s.pos).collect());
    }
    write_slice(map, 1.0 * unit, &drawn, &out.join("slice_z1.0m.png"))?;

    let region_flag = cfg
        .region
        .map(|r| format!(" --region {},{},{},{}", r[0], r[1], r[2], r[3]))
        .unwrap_or_default()
        + &cfg
            .height
            .map(|h| format!(" --height {},{}", h[0], h[1]))
            .unwrap_or_default();
    let report = ShowcaseReport {
        capture: capture.name.clone(),
        splat: capture.splat_label(),
        config: cfg.clone(),
        endpoints: endpoints.len(),
        pairs_tried: pairs.len(),
        certified,
        candidates,
        paths,
        all_candidates: cands,
        provenance: Provenance::capture(&format!(
            "splatplan showcase {} --out {} --count {} --tries {} --min-dist {} --min-blocked {} --rank {} --dedup {} --seed {}{region_flag}{}{}",
            capture_dir.display(),
            out.display(),
            cfg.count,
            cfg.tries,
            cfg.min_dist_m,
            cfg.min_blocked_m,
            match cfg.rank {
                Rank::Length => "length",
                Rank::Avoidance => "avoidance",
            },
            cfg.dedup_m,
            cfg.seed,
            if raw_splat { " --raw-splat" } else { "" },
            robot.flags()
        )),
    };
    std::fs::write(out.join("showcase.json"), report.to_json()?)?;
    println!(
        "{}: {} endpoints, {} pairs at least {} m apart planned, {} certified, {} in region with the straight line blocked; kept {}:",
        report.capture,
        report.endpoints,
        report.pairs_tried,
        cfg.min_dist_m,
        report.certified,
        report.candidates,
        report.paths.len()
    );
    for s in &report.paths {
        let m = &s.measures;
        println!(
            "  {}  {:.1} m (straight {:.1} m, detour {:.2}), straight line hits {} obstacle(s) over {:.2} m, swings {:.2} m sideways; clearance {:.3} m (needed {:.3})",
            s.id,
            m.length_m,
            m.straight_m,
            m.detour,
            m.obstacles,
            m.blocked_m,
            m.side_offset_m,
            s.min_clearance_m,
            s.required_m
        );
    }
    Ok(report)
}
