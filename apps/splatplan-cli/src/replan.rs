//! `splatplan replan`: certified paths between places the drone really was.
//!
//! `showcase` draws random endpoints. This takes them from the recovered
//! flight (`splatplan flight`) instead: where the video starts, where it
//! ends, and every spot the drone hovered. Each one is lifted straight up
//! to the lowest point that passes the planner's and the verifier's endpoint
//! checks (a takeoff spot on the floor is too close to the floor to count).
//! Every time-ordered pair is planned and verified, and the best few are
//! kept, ranked like `export` ranks: length × (1 + obstacles avoided).
//!
//! Each path records how far the drone actually flew between the same two
//! moments, so the viewer can put the planned path next to the real one.

use std::path::Path;

use anyhow::{Result, bail};
use gs_map::MapConfig;
use planner::Workspace;
use serde::Serialize;
use splat_types::{Report, Vec3};

use crate::capture::Capture;
use crate::pipeline::{Robot, endpoint_candidates, prepare, run_queries, system_endpoint_ok};
use crate::provenance::Provenance;
use crate::showcase::{Candidate, measure};
use crate::viz::write_slice;

/// Settings.
#[derive(Debug, Clone, Serialize)]
pub struct ReplanConfig {
    /// Directory with `flight.json`.
    pub flight: std::path::PathBuf,
    /// Paths to keep (at most 5, what the viewer shows).
    pub count: usize,
    /// A hover: the drone stays within this distance (m) of where it stopped...
    pub hover_radius_m: f64,
    /// ...for at least this long (s).
    pub hover_s: f64,
    /// Points closer than this (m) to an earlier one are the same place.
    pub merge_m: f64,
    /// Farthest a real point may be moved to make it a valid endpoint (m).
    pub max_snap_m: f64,
    /// Smallest straight-line distance between the two ends of a path (m).
    pub min_dist_m: f64,
}

/// A place the drone really was.
#[derive(Debug, Clone, Serialize)]
pub struct RealPoint {
    /// `start`, `end` or `hover at m:ss` (when it got there).
    pub label: String,
    /// Video time the drone got there and left (s).
    pub t0: f64,
    pub t1: f64,
    /// Where it was (m, aligned frame).
    pub real: [f64; 3],
    /// The endpoint used for planning, if one was found (m).
    pub snapped: Option<[f64; 3]>,
    /// Distance from `real` to `snapped` (m).
    pub snap_m: Option<f64>,
    /// Why there is no endpoint here, if there isn't.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skipped: Option<String>,
}

/// What the drone really did between the two ends of a planned path.
#[derive(Debug, Clone, Serialize)]
pub struct Flown {
    /// Video time it left the start and reached the goal (s).
    pub t0: f64,
    pub t1: f64,
    /// Length of the recovered flight between them (m), gaps not counted.
    pub length_m: f64,
    /// Time inside gaps, where the path is unknown (s).
    pub unknown_s: f64,
}

/// One kept path.
#[derive(Debug, Clone, Serialize)]
pub struct ReplanPath {
    /// Directory name under `certified/`.
    pub id: String,
    /// Indices into `points`.
    pub from: usize,
    pub to: usize,
    /// Labels of the two ends, for display.
    pub from_label: String,
    pub to_label: String,
    /// How far each end was lifted or moved to a valid endpoint (m).
    pub from_snap_m: f64,
    pub to_snap_m: f64,
    /// Length, detour and obstacle measures.
    #[serde(flatten)]
    pub measures: Candidate,
    /// The real flight between the same two moments.
    pub flown: Flown,
    /// Smallest clearance the verifier measured (m) and what it required.
    pub min_clearance_m: f64,
    pub required_m: f64,
}

/// Everything `splatplan replan` reports.
#[derive(Debug, Serialize)]
pub struct ReplanReport {
    /// Capture name.
    pub capture: String,
    /// The splat file used.
    pub splat: String,
    /// Settings.
    pub config: ReplanConfig,
    /// Every real point considered, with its endpoint or why it has none.
    pub points: Vec<RealPoint>,
    /// Pairs planned and verified.
    pub pairs_tried: usize,
    /// Pairs that came back certified.
    pub certified: usize,
    /// Why the other pairs have no certified path.
    pub refusals: std::collections::BTreeMap<String, usize>,
    /// The paths kept, best first.
    pub paths: Vec<ReplanPath>,
    /// How this report was produced.
    pub provenance: Provenance,
}

impl Report for ReplanReport {}

/// Start, hovers and end of a flight, in time order. `poses` are flight.json
/// rows (`[t, x, y, z, ...]`, meters); `gap_s` splits the flight into runs.
pub fn real_points(poses: &[[f64; 12]], gap_s: f64, cfg: &ReplanConfig) -> Vec<RealPoint> {
    let pos = |p: &[f64; 12]| Vec3::new(p[1], p[2], p[3]);
    let mut out: Vec<RealPoint> = Vec::new();
    let (Some(first), Some(last)) = (poses.first(), poses.last()) else {
        return out;
    };
    let point = |label: String, t0: f64, t1: f64, at: Vec3| RealPoint {
        label,
        t0,
        t1,
        real: [at.x, at.y, at.z],
        snapped: None,
        snap_m: None,
        skipped: None,
    };
    let mut cand = vec![point("start".into(), first[0], first[0], pos(first))];
    // Hovers: runs of poses that stay near where the run began.
    let mut i = 0;
    while i < poses.len() {
        let anchor = pos(&poses[i]);
        let mut j = i;
        while j + 1 < poses.len()
            && poses[j + 1][0] - poses[j][0] <= gap_s
            && (pos(&poses[j + 1]) - anchor).norm() <= cfg.hover_radius_m
        {
            j += 1;
        }
        if poses[j][0] - poses[i][0] >= cfg.hover_s {
            let n = (j - i + 1) as f64;
            let mean = poses[i..=j].iter().map(pos).sum::<Vec3>() / n;
            cand.push(point(String::new(), poses[i][0], poses[j][0], mean));
            i = j + 1;
        } else {
            i += 1;
        }
    }
    cand.push(point("end".into(), last[0], last[0], pos(last)));
    // One point per place. A point near the previous one continues it (the
    // drone sitting on the pad, then hovering above it: one point that lasts
    // until it leaves); a point near an older one is a revisit and dropped.
    for c in cand {
        let at = Vec3::from(c.real);
        let near = |k: &RealPoint| (Vec3::from(k.real) - at).norm() < cfg.merge_m;
        if let Some(prev) = out.last_mut().filter(|p| near(p)) {
            prev.t1 = prev.t1.max(c.t1);
            if c.label == "end" {
                prev.label = c.label;
            }
        } else if c.label == "end" || !out.iter().any(near) {
            // The end always counts, even back where the drone started.
            out.push(c);
        }
    }
    for p in &mut out {
        if p.label.is_empty() {
            p.label = format!("hover at {}", clock(p.t0));
        }
    }
    out
}

/// Video time as `m:ss`.
fn clock(t: f64) -> String {
    let s = t.round() as u64;
    format!("{}:{:02}", s / 60, s % 60)
}

/// Length of the recovered flight between video times `t0` and `t1`, and
/// how much of that time falls in gaps.
pub fn flown_between(poses: &[[f64; 12]], gap_s: f64, t0: f64, t1: f64) -> Flown {
    let mut length = 0.0;
    let mut unknown = 0.0;
    for w in poses.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        if a[0] < t0 || b[0] > t1 {
            continue;
        }
        if b[0] - a[0] > gap_s {
            unknown += b[0] - a[0];
        } else {
            length += (Vec3::new(b[1], b[2], b[3]) - Vec3::new(a[1], a[2], a[3])).norm();
        }
    }
    Flown {
        t0,
        t1,
        length_m: length,
        unknown_s: unknown,
    }
}

/// Plans between the real points and writes the best paths to
/// `out/certified/NNN/` (the layout `export` reads), with `replan.json`.
pub fn run(
    capture_dir: &Path,
    raw_splat: bool,
    robot: &Robot,
    out: &Path,
    cfg: &ReplanConfig,
) -> Result<ReplanReport> {
    if cfg.count == 0 || cfg.count > crate::export::MAX_SHOWN_PATHS {
        bail!(
            "--count must be 1 to {}, got {}",
            crate::export::MAX_SHOWN_PATHS,
            cfg.count
        );
    }
    let capture = Capture::open(capture_dir, raw_splat)?;
    let p = prepare(&capture, &MapConfig::default(), robot)?;
    if !p.scene.scale.is_metric() {
        bail!(
            "{} is UNSCALED: nothing can be certified, so there is nothing to plan",
            capture.name
        );
    }
    let fr = crate::flight::load_checked(&cfg.flight, &capture.name, &p.align.similarity)?;
    let planner_cfg = robot.planner();
    let map = &p.map;
    let unit = map.length_unit;
    let ws = Workspace::new(map);
    let need = planner_cfg.search_clearance(map);

    // Real points, each moved to the nearest valid endpoint: straight up
    // first (how a drone leaves the floor), else the closest valid voxel.
    let mut points = real_points(&fr.poses, fr.gap_s, cfg);
    let pool = endpoint_candidates(map, &ws, need);
    let ok = |q: &Vec3| system_endpoint_ok(&ws, &p.verifier, need, q);
    for pt in &mut points {
        let real = Vec3::from(pt.real) * unit;
        let steps = (cfg.max_snap_m * unit / map.spec.res).ceil() as usize;
        let up = (0..=steps)
            .map(|k| real + Vec3::new(0.0, 0.0, k as f64 * map.spec.res))
            .find(&ok);
        let snapped = up.or_else(|| {
            let mut near: Vec<&Vec3> = pool
                .iter()
                .filter(|c| (*c - real).norm() <= cfg.max_snap_m * unit)
                .collect();
            near.sort_by(|a, b| (*a - real).norm().total_cmp(&(*b - real).norm()));
            near.into_iter().find(|q| ok(q)).copied()
        });
        match snapped {
            Some(q) => {
                pt.snapped = Some([q.x / unit, q.y / unit, q.z / unit]);
                pt.snap_m = Some((q - real).norm() / unit);
            }
            None => {
                pt.skipped = Some(format!(
                    "no valid endpoint within {} m (too close to something, or unseen space)",
                    cfg.max_snap_m
                ))
            }
        }
    }

    // Every time-ordered pair of usable points far enough apart.
    let usable: Vec<usize> = (0..points.len())
        .filter(|&i| points[i].snapped.is_some())
        .collect();
    // `usable` only holds points with an endpoint.
    let at = |i: usize| Vec3::from(points[i].snapped.unwrap_or_default()) * unit;
    let mut pairs_idx: Vec<(usize, usize)> = Vec::new();
    for (k, &i) in usable.iter().enumerate() {
        for &j in &usable[k + 1..] {
            if (at(j) - at(i)).norm() >= cfg.min_dist_m * unit {
                pairs_idx.push((i, j));
            }
        }
    }
    if pairs_idx.is_empty() {
        bail!(
            "{}: {} real points, {} with a valid endpoint, no two at least {} m apart",
            capture.name,
            points.len(),
            usable.len(),
            cfg.min_dist_m
        );
    }
    let pairs: Vec<(Vec3, Vec3)> = pairs_idx.iter().map(|&(i, j)| (at(i), at(j))).collect();
    let (stats, outcomes) = run_queries(&p, &pairs, &planner_cfg);
    let certified = outcomes.iter().flatten().count();

    // Rank like export does. The takeoff-to-landing trip always goes first;
    // after that, no point is used by more than two paths.
    let mut cands: Vec<Candidate> = outcomes
        .iter()
        .enumerate()
        .filter_map(|(k, o)| {
            o.as_ref()
                .map(|o| measure(map, &pairs[k].0, &pairs[k].1, o, k))
        })
        .collect();
    let score = |c: &Candidate| c.length_m * (1 + c.obstacles) as f64;
    cands.sort_by(|a, b| score(b).total_cmp(&score(a)).then(a.index.cmp(&b.index)));
    let (first, last) = (0, points.len() - 1);

    // A kept path: its measures, the certified plan, where it really goes
    // (`points` indices) and the real points it is routed through.
    struct Kept {
        cand: Candidate,
        outcome: planner::PlanOutcome,
        from: usize,
        to: usize,
    }
    let mut kept: Vec<Kept> = Vec::new();
    let mut landing_leg: Option<f64> = None;
    let direct = cands
        .iter()
        .position(|c| pairs_idx[c.index] == (first, last));
    if let Some(k) = direct {
        let c = cands.remove(k);
        if let Some(outcome) = outcomes[c.index].clone() {
            kept.push(Kept {
                cand: c,
                outcome,
                from: first,
                to: last,
            });
        }
    } else {
        // No path certifies all the way (the landing spot is often too tight): the trip
        // is the certified path from the takeoff that stops closest to it,
        // within `max_snap_m`. The viewer draws the rest of the way, to the
        // real landing spot, as uncertified.
        let land = Vec3::from(points[last].real) * unit;
        let gap = |c: &Candidate| (pairs[c.index].1 - land).norm() / unit;
        let best = cands
            .iter()
            .enumerate()
            .filter(|(_, c)| pairs_idx[c.index].0 == first && gap(c) <= cfg.max_snap_m)
            .min_by(|a, b| gap(a.1).total_cmp(&gap(b.1)))
            .map(|(k, _)| k);
        if let Some(k) = best {
            let c = cands.remove(k);
            landing_leg = Some(gap(&c));
            if let Some(outcome) = outcomes[c.index].clone() {
                kept.push(Kept {
                    cand: c,
                    outcome,
                    from: first,
                    to: last,
                });
            }
        }
    }
    let mut uses = vec![0usize; points.len()];
    for k in &kept {
        uses[k.from] += 1;
        uses[k.to] += 1;
    }
    for c in cands {
        if kept.len() == cfg.count {
            break;
        }
        let (i, j) = pairs_idx[c.index];
        if uses[i] < 2 && uses[j] < 2 {
            if let Some(outcome) = outcomes[c.index].clone() {
                uses[i] += 1;
                uses[j] += 1;
                kept.push(Kept {
                    cand: c,
                    outcome,
                    from: i,
                    to: j,
                });
            }
        }
    }

    let all = out.join("certified");
    if all.exists() {
        std::fs::remove_dir_all(&all)?;
    }
    let mut paths = Vec::new();
    let mut drawn = Vec::new();
    for (rank, k) in kept.iter().enumerate() {
        // Same route, flown more evenly (only if the verifier certifies it).
        let mut o = planner::calm(&p.map, &p.verifier, k.outcome.clone(), &planner_cfg);
        let (a, b) = pairs[k.cand.index];
        let measures = measure(&p.map, &a, &b, &o, k.cand.index);
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
        let (i, j) = (k.from, k.to);
        let mut to_snap_m = points[j].snap_m.unwrap_or(0.0);
        // A trip that stops short ends `gap` m before the real landing spot.
        if let Some(gap) = landing_leg.filter(|_| rank == 0) {
            to_snap_m = gap;
        }
        // Leave the start when the drone left it; reach the goal when it arrived.
        let flown = flown_between(&fr.poses, fr.gap_s, points[i].t1, points[j].t0);
        paths.push(ReplanPath {
            id,
            from: i,
            to: j,
            from_label: points[i].label.clone(),
            to_label: points[j].label.clone(),
            from_snap_m: points[i].snap_m.unwrap_or(0.0),
            to_snap_m,
            measures,
            flown,
            min_clearance_m: o.certificate.min_clearance_m,
            required_m: o.certificate.margin_required_m,
        });
        drawn.push(o.trajectory.sample(0.05).iter().map(|s| s.pos).collect());
    }
    std::fs::create_dir_all(out)?;
    write_slice(map, 1.0 * unit, &drawn, &out.join("slice_z1.0m.png"))?;

    let report = ReplanReport {
        capture: capture.name.clone(),
        splat: capture.splat_label(),
        config: cfg.clone(),
        points,
        pairs_tried: pairs.len(),
        certified,
        refusals: stats.refusals.clone(),
        paths,
        provenance: Provenance::capture(&format!(
            "splatplan replan {} --flight {} --out {} --count {} --hover-radius {} --hover {} --merge {} --max-snap {} --min-dist {}{}{}",
            capture_dir.display(),
            cfg.flight.display(),
            out.display(),
            cfg.count,
            cfg.hover_radius_m,
            cfg.hover_s,
            cfg.merge_m,
            cfg.max_snap_m,
            cfg.min_dist_m,
            if raw_splat { " --raw-splat" } else { "" },
            robot.flags()
        )),
    };
    std::fs::write(out.join("replan.json"), report.to_json()?)?;
    let fmt = |t: f64| format!("{}:{:04.1}", (t / 60.0).floor(), t % 60.0);
    println!("{}: real points", report.capture);
    for pt in &report.points {
        match (&pt.snap_m, &pt.skipped) {
            (Some(s), _) => println!(
                "  {:<9} {}-{}  z {:.2} m, moved {:.2} m to a valid endpoint",
                pt.label,
                fmt(pt.t0),
                fmt(pt.t1),
                pt.real[2],
                s
            ),
            (None, Some(why)) => println!("  {:<9} {}  skipped: {why}", pt.label, fmt(pt.t0)),
            _ => {}
        }
    }
    println!(
        "{} pairs planned, {} certified; kept {}:",
        report.pairs_tried,
        report.certified,
        report.paths.len()
    );
    for (why, n) in &report.refusals {
        println!("  refused {n}: {why}");
    }
    for s in &report.paths {
        let m = &s.measures;
        println!(
            "  {}  {} -> {}: planned {:.1} m around {} obstacle(s) (straight {:.1} m); flown {:.1} m in {:.0} s{}; clearance {:.3} m (needed {:.3})",
            s.id,
            s.from_label,
            s.to_label,
            m.length_m,
            m.obstacles,
            m.straight_m,
            s.flown.length_m,
            s.flown.t1 - s.flown.t0,
            if s.flown.unknown_s > 0.0 {
                format!(" ({:.0} s unknown)", s.flown.unknown_s)
            } else {
                String::new()
            },
            s.min_clearance_m,
            s.required_m
        );
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ReplanConfig {
        ReplanConfig {
            flight: "unused".into(),
            count: 5,
            hover_radius_m: 0.35,
            hover_s: 1.5,
            merge_m: 0.75,
            max_snap_m: 1.0,
            min_dist_m: 2.0,
        }
    }

    fn pose(t: f64, x: f64, y: f64) -> [f64; 12] {
        let mut p = [0.0; 12];
        p[0] = t;
        p[1] = x;
        p[2] = y;
        p[3] = 1.0;
        p
    }

    #[test]
    fn finds_start_hover_end() {
        // Fly 0 -> 4 m along x (1 m/s), hover 3 s, fly on to 8 m.
        let mut poses = Vec::new();
        let mut t = 0.0;
        for k in 0..=40 {
            poses.push(pose(t, k as f64 * 0.1, 0.0));
            t += 0.1;
        }
        for _ in 0..30 {
            poses.push(pose(t, 4.0, 0.0));
            t += 0.1;
        }
        for k in 1..=40 {
            poses.push(pose(t, 4.0 + k as f64 * 0.1, 0.0));
            t += 0.1;
        }
        let pts = real_points(&poses, 0.5, &cfg());
        let labels: Vec<&str> = pts.iter().map(|p| p.label.as_str()).collect();
        assert_eq!(labels, ["start", "hover at 0:04", "end"]);
        assert!((pts[1].real[0] - 4.0).abs() < 0.2);
        assert!(pts[1].t1 - pts[1].t0 >= 2.9);
    }

    #[test]
    fn flown_skips_gaps() {
        let poses = vec![
            pose(0.0, 0.0, 0.0),
            pose(0.1, 0.1, 0.0),
            pose(5.0, 3.0, 0.0), // 4.9 s gap
            pose(5.1, 3.1, 0.0),
        ];
        let f = flown_between(&poses, 0.5, 0.0, 5.1);
        assert!((f.length_m - 0.2).abs() < 1e-9);
        assert!((f.unknown_s - 4.9).abs() < 1e-9);
    }
}
