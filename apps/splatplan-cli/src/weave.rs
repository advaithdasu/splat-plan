//! `splatplan weave`: long certified tours that thread between obstacles.
//!
//! Shortest paths in a room full of desks mostly fly over them. Weaving
//! makes the drone go around instead:
//!
//! 1. **Cap the height** at `--ceiling` by marking everything above it solid.
//! 2. **Find obstacles from above**: anything occupied between
//!    `--min-height` and the ceiling. Its footprint is made solid at every
//!    height. Areas touching the map edge are walls.
//! 3. **Find gates**: where two obstacles' Voronoi regions meet in free
//!    space, at the narrowest point the drone fits through.
//! 4. **Chain gates** with a beam search. Consecutive gates share an
//!    obstacle, and tours may not cross or double back.
//! 5. **Certify** each tour as one trajectory through all its gates, checked
//!    by the verifier like any other path.

use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap};
use std::path::Path;

use anyhow::{Context, Result, bail};
use gs_map::{Map, MapConfig, VoxelState};
use image::{Rgb, RgbImage};
use planner::{PlanOutcome, PlannerConfig, Workspace, cell_ok, plan_and_certify_via};
use rayon::prelude::*;
use serde::Serialize;
use splat_types::{Report, Vec3};

use crate::capture::Capture;
use crate::pipeline::{Prepared, Robot, prepare, system_endpoint_ok};
use crate::provenance::Provenance;

/// Settings.
#[derive(Debug, Clone, Serialize)]
pub struct WeaveConfig {
    /// Tours to keep.
    pub count: usize,
    /// Highest the drone's center may fly (m above the floor).
    pub ceiling_m: f64,
    /// Lowest gate height considered (m above the floor).
    pub floor_m: f64,
    /// Anything occupied above this height (m) and below the ceiling is an
    /// obstacle to go around; lower clutter is ignored.
    pub min_height_m: f64,
    /// A tour stops growing once it is this long (m, path distance).
    pub max_length_m: f64,
    /// Longest leg between consecutive gates (m, path distance).
    pub max_leg_m: f64,
    /// Certified tours shorter than this (m) are not kept.
    pub min_length_m: f64,
    /// Solid areas smaller than this (m², seen from above) are noise.
    pub min_obstacle_m2: f64,
    /// Tours kept per step of the beam search.
    pub beam: usize,
    /// Best varied tours handed to the planner and verifier.
    pub candidates: usize,
}

/// A gap between two obstacles that the drone fits through.
#[derive(Debug, Clone, Serialize)]
pub struct Gate {
    /// Where the tour passes (m, aligned frame).
    pub pos: [f64; 3],
    /// The two obstacles (ids; see `obstacles` in the report).
    pub between: [usize; 2],
    /// Free width around the gate's center (m): twice its 2D clearance.
    pub width_m: f64,
}

/// An obstacle, seen from above.
#[derive(Debug, Clone, Serialize)]
pub struct Obstacle {
    /// Id (index into the report's list).
    pub id: usize,
    /// Footprint (m²).
    pub area_m2: f64,
    /// Footprint center (m).
    pub center: [f64; 2],
    /// Part of the room's boundary (touches the edge of the map).
    pub wall: bool,
}

/// One certified tour.
#[derive(Debug, Clone, Serialize)]
pub struct WeavePath {
    /// Directory name under `certified/`.
    pub id: String,
    /// Length of the certified path (m, waypoint polyline).
    pub length_m: f64,
    /// Flight time (s).
    pub duration_s: f64,
    /// Gates threaded, in order (indices into the report's `gates`).
    pub gates: Vec<usize>,
    /// Distinct obstacles (walls not counted) the gates lie between.
    pub obstacles_threaded: usize,
    /// Distinct obstacles (walls not counted) within 0.6 m of the drone's body.
    pub obstacles_passed: usize,
    /// Total change of heading seen from above (degrees).
    pub turning_deg: f64,
    /// Highest and lowest the drone's center flies (m above the floor).
    pub max_z_m: f64,
    /// Lowest.
    pub min_z_m: f64,
    /// Smallest clearance the verifier measured along the path (m).
    pub min_clearance_m: f64,
    /// Clearance the verifier required (m).
    pub required_m: f64,
}

/// Everything `splatplan weave` reports.
#[derive(Debug, Serialize)]
pub struct WeaveReport {
    /// Capture name.
    pub capture: String,
    /// The splat file used.
    pub splat: String,
    /// Settings.
    pub config: WeaveConfig,
    /// Obstacles found (noise blobs excluded).
    pub obstacles: Vec<Obstacle>,
    /// Gates found.
    pub gates: Vec<Gate>,
    /// Tours handed to the planner and verifier.
    pub tours_tried: usize,
    /// The certified tours kept, best first.
    pub paths: Vec<WeavePath>,
    /// How this report was produced.
    pub provenance: Provenance,
}

impl Report for WeaveReport {}

/// Dijkstra queue entry on the 2D grid (pops the smallest distance; ties
/// to the smaller index, so results are reproducible).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Q {
    d: f64,
    i: usize,
}
impl Eq for Q {}
impl Ord for Q {
    fn cmp(&self, o: &Self) -> Ordering {
        o.d.total_cmp(&self.d).then(o.i.cmp(&self.i))
    }
}
impl PartialOrd for Q {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

const NB8: [(i64, i64); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

/// The map seen from above, in the drone's height band.
struct Plan2d {
    nx: usize,
    ny: usize,
    res: f64,
    /// Per column: the band voxel A* may use with the most clearance (k, dist).
    free: Vec<Option<(usize, f64)>>,
    /// Per column: obstacle id, if solid and part of a kept obstacle.
    obstacle: Vec<Option<usize>>,
    /// Per column: nearest obstacle (Voronoi label) and distance to it.
    nearest: Vec<Option<(usize, f64)>>,
    obstacles: Vec<Obstacle>,
}

impl Plan2d {
    fn idx(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }
    fn nb(&self, c: usize, d: (i64, i64)) -> Option<usize> {
        let (i, j) = ((c % self.nx) as i64 + d.0, (c / self.nx) as i64 + d.1);
        (i >= 0 && j >= 0 && (i as usize) < self.nx && (j as usize) < self.ny)
            .then(|| self.idx(i as usize, j as usize))
    }
}

/// A copy of `map` with everything above `ceiling_m` (m above the floor),
/// plus the planning clearance, marked solid, so the drone's center stays at
/// or below the ceiling. The verifier still checks the raw Gaussians.
fn cap_map(map: &Map, ceiling_m: f64, cfg: &PlannerConfig) -> Map {
    let mut m = map.clone();
    let top = ceiling_m * m.length_unit + cfg.search_clearance(&m) + m.spec.res;
    for i in 0..m.spec.len() {
        if m.spec.center(m.spec.coords(i)).z > top {
            m.state[i] = VoxelState::Occupied;
        }
    }
    m.dist = gs_map::distance_field(&m.spec, &m.state);
    m
}

/// Marks solid every free voxel between heights `z0` and `z1` (scene
/// units) that might touch one of the verifier's Gaussian ellipsoids.
///
/// The map and the verifier disagree where faint Gaussians are thick, so a
/// long tour with fixed gates would usually be refused somewhere. Planning on
/// the union of both obstacle sets only makes the planner stricter; the
/// verifier still has the final say.
fn fold_verifier_solids(map: &mut Map, verifier: &gs_verify::Verifier, z0: f64, z1: f64) -> usize {
    let spec = map.spec;
    let half = spec.res * 3f64.sqrt() / 2.0;
    let hit: Vec<usize> = (0..spec.len())
        .into_par_iter()
        .filter(|&i| {
            if map.state[i] != VoxelState::FreeObserved {
                return false;
            }
            let c = spec.center(spec.coords(i));
            c.z >= z0 && c.z <= z1 && verifier.clearance_lower_bound(&c).0 <= half
        })
        .collect();
    for &i in &hit {
        map.state[i] = VoxelState::Occupied;
    }
    map.dist = gs_map::distance_field(&map.spec, &map.state);
    hit.len()
}

/// Column index range of heights `[z0, z1]` (scene units), clamped to the grid.
fn k_range(map: &Map, z0: f64, z1: f64) -> (usize, usize) {
    let spec = &map.spec;
    let nz = spec.dims[2] as f64;
    let kz = |z: f64| {
        ((z - spec.origin.z) / spec.res)
            .floor()
            .clamp(0.0, nz - 1.0) as usize
    };
    (kz(z0), kz(z1))
}

/// Obstacles seen from above: a column is solid if anything occupied sits
/// between `--min-height` and the top of the drone's body at the ceiling, or
/// if the whole flight band is unknown. Connected solid areas are
/// obstacles; small ones are left out as noise.
fn find_obstacles(map: &Map, need: f64, cfg: &WeaveConfig) -> Plan2d {
    let spec = &map.spec;
    let unit = map.length_unit;
    let [nx, ny, _] = spec.dims;
    let res = spec.res;
    let (k0, k1) = k_range(map, cfg.floor_m * unit, cfg.ceiling_m * unit);
    let (s0, s1) = k_range(map, cfg.min_height_m * unit, cfg.ceiling_m * unit + need);
    let solid: Vec<bool> = (0..nx * ny)
        .into_par_iter()
        .map(|c| {
            let (i, j) = (c % nx, c / nx);
            let occupied =
                (s0..=s1).any(|k| map.state[spec.index([i, j, k])] == VoxelState::Occupied);
            let all_unknown =
                (k0..=k1).all(|k| map.state[spec.index([i, j, k])] == VoxelState::Unknown);
            occupied || all_unknown
        })
        .collect();
    let mut g = Plan2d {
        nx,
        ny,
        res,
        free: vec![None; nx * ny],
        obstacle: vec![None; nx * ny],
        nearest: vec![None; nx * ny],
        obstacles: Vec::new(),
    };
    // Connected solid areas (8-connected), in index order.
    let mut comp = vec![usize::MAX; nx * ny];
    let min_cells = (cfg.min_obstacle_m2 * unit * unit / (res * res)).ceil() as usize;
    let mut ncomp = 0;
    for s in 0..nx * ny {
        if !solid[s] || comp[s] != usize::MAX {
            continue;
        }
        let mut stack = vec![s];
        let mut cells = Vec::new();
        comp[s] = ncomp;
        let mut wall = false;
        while let Some(c) = stack.pop() {
            cells.push(c);
            let (i, j) = (c % nx, c / nx);
            wall |= i == 0 || j == 0 || i + 1 == nx || j + 1 == ny;
            for d in NB8 {
                if let Some(n) = g.nb(c, d) {
                    if solid[n] && comp[n] == usize::MAX {
                        comp[n] = ncomp;
                        stack.push(n);
                    }
                }
            }
        }
        ncomp += 1;
        if cells.len() < min_cells {
            continue;
        }
        let id = g.obstacles.len();
        let (mut cx, mut cy) = (0.0, 0.0);
        for &c in &cells {
            g.obstacle[c] = Some(id);
            let p = spec.center([c % nx, c / nx, 0]);
            cx += p.x;
            cy += p.y;
        }
        let n = cells.len() as f64;
        g.obstacles.push(Obstacle {
            id,
            area_m2: n * res * res / (unit * unit),
            center: [cx / n / unit, cy / n / unit],
            wall,
        });
    }
    g
}

/// Extrudes every obstacle's footprint from the floor to the ceiling cap:
/// the planner can then only go around obstacles, never over them.
/// Returns the number of voxels newly marked solid.
fn extrude(map: &mut Map, g: &Plan2d) -> usize {
    let spec = map.spec;
    let mut n = 0;
    for c in 0..g.nx * g.ny {
        if g.obstacle[c].is_none() {
            continue;
        }
        for k in 0..spec.dims[2] {
            let i = spec.index([c % g.nx, c / g.nx, k]);
            if map.state[i] == VoxelState::FreeObserved {
                map.state[i] = VoxelState::Occupied;
                n += 1;
            }
        }
    }
    map.dist = gs_map::distance_field(&map.spec, &map.state);
    n
}

/// Fills in the traversable columns (the band voxel A* may use with the
/// most clearance) and each column's nearest obstacle.
fn free_and_nearest(g: &mut Plan2d, map: &Map, ws: &Workspace, need: f64, cfg: &WeaveConfig) {
    let unit = map.length_unit;
    let (nx, res) = (g.nx, g.res);
    let (k0, k1) = k_range(map, cfg.floor_m * unit, cfg.ceiling_m * unit);
    g.free = (0..g.nx * g.ny)
        .into_par_iter()
        .map(|c| {
            let (i, j) = (c % nx, c / nx);
            let mut best: Option<(usize, f64)> = None;
            for k in k0..=k1 {
                if cell_ok(ws, [i, j, k], need + res) {
                    let d = map.dist_at([i, j, k]);
                    if best.is_none_or(|(_, b)| d > b) {
                        best = Some((k, d));
                    }
                }
            }
            best
        })
        .collect();
    // Nearest obstacle for every column (multi-source Dijkstra).
    let ny = g.ny;
    let mut heap = BinaryHeap::new();
    for c in 0..nx * ny {
        if let Some(o) = g.obstacle[c] {
            g.nearest[c] = Some((o, 0.0));
            heap.push(Q { d: 0.0, i: c });
        }
    }
    while let Some(Q { d, i: c }) = heap.pop() {
        let Some((lab, dc)) = g.nearest[c] else {
            continue;
        };
        if d > dc {
            continue;
        }
        for dd in NB8 {
            let Some(n) = g.nb(c, dd) else { continue };
            let step = if dd.0 != 0 && dd.1 != 0 {
                std::f64::consts::SQRT_2
            } else {
                1.0
            } * res;
            let nd = d + step;
            if g.nearest[n].is_none_or(|(_, old)| nd < old) {
                g.nearest[n] = Some((lab, nd));
                heap.push(Q { d: nd, i: n });
            }
        }
    }
}

/// Gates: where two obstacles' Voronoi regions meet in traversable columns.
/// One per connected stretch of boundary, at its narrowest usable point.
fn find_gates(g: &Plan2d, map: &Map, ok: &(dyn Fn(&Vec3) -> bool + Sync)) -> Vec<(Gate, usize)> {
    let spec = &map.spec;
    let unit = map.length_unit;
    // Boundary columns per obstacle pair.
    let mut by_pair: std::collections::BTreeMap<(usize, usize), Vec<usize>> = Default::default();
    for c in 0..g.nx * g.ny {
        if g.free[c].is_none() {
            continue;
        }
        let Some((a, _)) = g.nearest[c] else { continue };
        for d in [(1, 0), (0, 1), (-1, 0), (0, -1)] {
            let Some(n) = g.nb(c, d) else { continue };
            if g.free[n].is_none() {
                continue;
            }
            let Some((b, _)) = g.nearest[n] else { continue };
            if a != b && !(g.obstacles[a].wall && g.obstacles[b].wall) {
                by_pair.entry((a.min(b), a.max(b))).or_default().push(c);
                break;
            }
        }
    }
    let mut gates = Vec::new();
    for ((a, b), cells) in by_pair {
        let mut seen = vec![false; cells.len()];
        let pos_of: std::collections::BTreeMap<usize, usize> =
            cells.iter().enumerate().map(|(k, &c)| (c, k)).collect();
        for s in 0..cells.len() {
            if seen[s] {
                continue;
            }
            // One connected stretch (8-connected, gaps of one column bridged).
            let mut piece = Vec::new();
            let mut stack = vec![s];
            seen[s] = true;
            while let Some(k) = stack.pop() {
                let c = cells[k];
                piece.push(c);
                for di in -2i64..=2 {
                    for dj in -2i64..=2 {
                        let kn = g.nb(c, (di, dj)).and_then(|n| pos_of.get(&n).copied());
                        if let Some(kn) = kn.filter(|&kn| !seen[kn]) {
                            seen[kn] = true;
                            stack.push(kn);
                        }
                    }
                }
            }
            // A stretch shorter than a voxel or two is a corner touch, not a gap.
            if piece.len() < 3 {
                continue;
            }
            // Narrowest first: the gap proper, not the open space beyond it.
            piece.sort_by(|x, y| {
                g.nearest[*x]
                    .map(|n| n.1)
                    .unwrap_or(0.0)
                    .total_cmp(&g.nearest[*y].map(|n| n.1).unwrap_or(0.0))
                    .then(x.cmp(y))
            });
            for &c in &piece {
                let Some((k, _)) = g.free[c] else { continue };
                let p = spec.center([c % g.nx, c / g.nx, k]);
                if ok(&p) {
                    let w = g.nearest[c].map_or(0.0, |n| n.1);
                    gates.push((
                        Gate {
                            pos: [p.x / unit, p.y / unit, p.z / unit],
                            between: [a, b],
                            width_m: 2.0 * w / unit,
                        },
                        c,
                    ));
                    break;
                }
            }
        }
    }
    gates
}

/// Path distance through traversable columns from each gate to the others
/// within `max` (scene units).
fn gate_links(g: &Plan2d, gates: &[(Gate, usize)], max: f64) -> Vec<Vec<(usize, f64)>> {
    let mut at = vec![None; g.nx * g.ny];
    for (k, (_, c)) in gates.iter().enumerate() {
        at[*c] = Some(k);
    }
    gates
        .par_iter()
        .enumerate()
        .map(|(k, (_, s))| {
            let mut dist: std::collections::HashMap<usize, f64> = Default::default();
            let mut heap = BinaryHeap::new();
            dist.insert(*s, 0.0);
            heap.push(Q { d: 0.0, i: *s });
            let mut out = Vec::new();
            while let Some(Q { d, i: c }) = heap.pop() {
                if d > dist[&c] {
                    continue;
                }
                if let Some(h) = at[c].filter(|&h| h != k) {
                    out.push((h, d));
                }
                for dd in NB8 {
                    let Some(n) = g.nb(c, dd) else { continue };
                    if g.free[n].is_none() {
                        continue;
                    }
                    let step = if dd.0 != 0 && dd.1 != 0 {
                        std::f64::consts::SQRT_2
                    } else {
                        1.0
                    } * g.res;
                    let nd = d + step;
                    if nd <= max && dist.get(&n).is_none_or(|old| nd < *old) {
                        dist.insert(n, nd);
                        heap.push(Q { d: nd, i: n });
                    }
                }
            }
            out.sort_by(|x, y| x.0.cmp(&y.0));
            out
        })
        .collect()
}

#[derive(Debug, Clone)]
struct Tour {
    seq: Vec<usize>,
    length: f64,
    obstacles: BTreeSet<usize>,
    score: f64,
}

fn score(t: &Tour, unit: f64) -> f64 {
    t.obstacles.len() as f64 + 0.5 * t.seq.len() as f64 + 0.15 * t.length / unit
}

fn segments_cross(p1: [f64; 2], p2: [f64; 2], q1: [f64; 2], q2: [f64; 2]) -> bool {
    let o = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    };
    let (d1, d2) = (o(q1, q2, p1), o(q1, q2, p2));
    let (d3, d4) = (o(p1, p2, q1), o(p1, p2, q2));
    d1 * d2 < 0.0 && d3 * d4 < 0.0
}

/// Beam search for long, non-crossing gate tours; returns the best found,
/// best first.
fn search_tours(
    gates: &[(Gate, usize)],
    obstacles: &[Obstacle],
    links: &[Vec<(usize, f64)>],
    unit: f64,
    cfg: &WeaveConfig,
) -> Vec<Tour> {
    let xy = |k: usize| [gates[k].0.pos[0], gates[k].0.pos[1]];
    let real = |o: usize| !obstacles[o].wall;
    // Consecutive gates must share a real obstacle: the leg between them
    // rounds it. (Sharing only the wall would let tours zig-zag across
    // open floor from one side of the room to the other.)
    let shares = |a: usize, b: usize| {
        let (x, y) = (gates[a].0.between, gates[b].0.between);
        x.iter().any(|&o| real(o) && y.contains(&o))
    };
    // No doubling back at a gate: the heading may turn at most this much.
    let max_turn = 120f64.to_radians();
    let turn = |a: usize, b: usize, c: usize| {
        let (p, q, r) = (xy(a), xy(b), xy(c));
        let (u, v) = ([q[0] - p[0], q[1] - p[1]], [r[0] - q[0], r[1] - q[1]]);
        (u[0] * v[1] - u[1] * v[0])
            .atan2(u[0] * v[0] + u[1] * v[1])
            .abs()
    };
    let add_obs = |set: &mut BTreeSet<usize>, k: usize| {
        for o in gates[k].0.between {
            if real(o) {
                set.insert(o);
            }
        }
    };
    let max_len = cfg.max_length_m * unit;
    let min_leg = 0.5 * unit;
    let near = 0.8; // m: a tour may not come back this close to an earlier gate
    // A beam search from every gate (in parallel), so the tours start all
    // over the room rather than all growing out of the few best openings.
    let from = |k: usize| -> Vec<Tour> {
        let mut first = BTreeSet::new();
        add_obs(&mut first, k);
        let mut t0 = Tour {
            seq: vec![k],
            length: 0.0,
            obstacles: first,
            score: 0.0,
        };
        t0.score = score(&t0, unit);
        let mut beam = vec![t0];
        let mut archive: Vec<Tour> = Vec::new();
        while !beam.is_empty() {
            let mut next: Vec<Tour> = Vec::new();
            for t in &beam {
                let Some(&last) = t.seq.last() else { continue };
                for &(h, d) in &links[last] {
                    let ok = d >= min_leg
                        && shares(last, h)
                        && (t.seq.len() < 2 || turn(t.seq[t.seq.len() - 2], last, h) <= max_turn)
                        && !t.seq.contains(&h)
                        && t.seq[..t.seq.len() - 1].iter().all(|&q| {
                            let (a, b) = (xy(q), xy(h));
                            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt() > near
                        })
                        && t.seq
                            .windows(2)
                            .take(t.seq.len().saturating_sub(2))
                            .all(|w| !segments_cross(xy(w[0]), xy(w[1]), xy(last), xy(h)));
                    if !ok {
                        continue;
                    }
                    let mut n = t.clone();
                    n.seq.push(h);
                    n.length += d;
                    add_obs(&mut n.obstacles, h);
                    n.score = score(&n, unit);
                    next.push(n);
                }
            }
            next.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.seq.cmp(&b.seq)));
            // One tour per (set of gates, last gate): the rest are reorderings.
            let mut seen = BTreeSet::new();
            next.retain(|t| {
                let mut key = t.seq.clone();
                let last = key.pop();
                key.sort_unstable();
                key.extend(last);
                seen.insert(key)
            });
            next.truncate(cfg.beam);
            archive.extend(next.iter().filter(|t| t.seq.len() >= 4).cloned());
            next.retain(|t| t.length < max_len);
            beam = next;
        }
        archive.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.seq.cmp(&b.seq)));
        archive.truncate(3);
        archive
    };
    let mut pool: Vec<Tour> = (0..gates.len())
        .into_par_iter()
        .flat_map_iter(from)
        .collect();
    pool.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.seq.cmp(&b.seq)));
    // Varied: pick greedily by what each tour adds (gates and obstacles
    // not already covered by earlier picks), so later tours explore other
    // parts of the room instead of repeating the best one.
    let mut picked: Vec<Tour> = Vec::new();
    let mut used_gates = BTreeSet::new();
    let mut used_obs = BTreeSet::new();
    while picked.len() < cfg.candidates && !pool.is_empty() {
        let gain = |t: &Tour| {
            let new_g = t.seq.iter().filter(|k| !used_gates.contains(*k)).count();
            let new_o = t
                .obstacles
                .iter()
                .filter(|o| !used_obs.contains(*o))
                .count();
            let frac = new_g as f64 / t.seq.len() as f64;
            new_o as f64 + 0.5 * new_g as f64 + 0.15 * frac * t.length / unit
        };
        let (k, g) = pool
            .iter()
            .enumerate()
            .map(|(k, t)| (k, gain(t)))
            .fold((0, f64::NEG_INFINITY), |b, x| if x.1 > b.1 { x } else { b });
        if g < 2.0 {
            break;
        }
        let t = pool.swap_remove(k);
        used_gates.extend(t.seq.iter().copied());
        used_obs.extend(t.obstacles.iter().copied());
        picked.push(t);
    }
    picked
}

/// Certifies a tour as one trajectory through all its gates. If the
/// verifier refuses it, the tour is rebuilt gate by gate: each gate is
/// kept only if the trajectory so far, extended to it, still certifies.
/// Returns the gates kept (indices into `via`) and the certified plan.
fn certify_tour(
    map: &Map,
    p: &Prepared,
    via: &[Vec3],
    cfg: &PlannerConfig,
) -> Option<(Vec<usize>, PlanOutcome)> {
    match plan_and_certify_via(map, &p.verifier, via, cfg) {
        Ok(o) => return Some(((0..via.len()).collect(), o)),
        Err(e) => tracing::info!(gates = via.len(), error = %e, "tour refused whole; rebuilding"),
    }
    let mut kept: Vec<usize> = vec![0];
    let mut best: Option<PlanOutcome> = None;
    for k in 1..via.len() {
        let mut pts: Vec<Vec3> = kept.iter().map(|&i| via[i]).collect();
        pts.push(via[k]);
        if let Ok(o) = plan_and_certify_via(map, &p.verifier, &pts, cfg) {
            kept.push(k);
            best = Some(o);
        } else if kept.len() == 1 {
            // Nothing certified yet: start over from this gate instead.
            kept = vec![k];
        }
    }
    let o = best?;
    (kept.len() >= 4).then_some((kept, o))
}

fn color(k: usize) -> Rgb<u8> {
    const C: [[u8; 3]; 8] = [
        [230, 40, 40],
        [40, 200, 255],
        [255, 200, 30],
        [60, 230, 90],
        [230, 90, 230],
        [255, 130, 40],
        [150, 120, 255],
        [255, 255, 255],
    ];
    Rgb(C[k % C.len()])
}

/// Top view: obstacles (each its own shade), walls, traversable band,
/// gates, and the kept tours.
fn write_top(
    g: &Plan2d,
    map: &Map,
    gates: &[(Gate, usize)],
    tours: &[Vec<Vec3>],
    out: &Path,
) -> Result<()> {
    const Z: u32 = 2;
    let (w, h) = (g.nx as u32 * Z, g.ny as u32 * Z);
    let mut img = RgbImage::new(w, h);
    for c in 0..g.nx * g.ny {
        let px = match (g.obstacle[c], g.free[c]) {
            (Some(o), _) if g.obstacles[o].wall => Rgb([45, 45, 45]),
            (Some(o), _) => {
                let s = (o * 2654435761) % 997;
                Rgb([
                    90 + (s % 7) as u8 * 15,
                    50 + (s % 5) as u8 * 12,
                    40 + (s % 3) as u8 * 20,
                ])
            }
            (None, Some(_)) => Rgb([40, 70, 130]),
            (None, None) => Rgb([20, 20, 30]),
        };
        let (i, j) = ((c % g.nx) as u32, (c / g.nx) as u32);
        for dx in 0..Z {
            for dy in 0..Z {
                img.put_pixel(i * Z + dx, h - 1 - (j * Z + dy), px);
            }
        }
    }
    let to_px = |p: &Vec3| -> Option<(u32, u32)> {
        let q = map.spec.continuous(p);
        let x = q.x * Z as f64;
        let y = h as f64 - 1.0 - q.y * Z as f64;
        (x >= 0.0 && y >= 0.0 && x < w as f64 && y < h as f64).then_some((x as u32, y as u32))
    };
    let dot = |img: &mut RgbImage, p: &Vec3, r: i64, c: Rgb<u8>| {
        if let Some((x, y)) = to_px(p) {
            for dx in -r..=r {
                for dy in -r..=r {
                    let (px, py) = (x as i64 + dx, y as i64 + dy);
                    if px >= 0 && py >= 0 && (px as u32) < w && (py as u32) < h {
                        img.put_pixel(px as u32, py as u32, c);
                    }
                }
            }
        }
    };
    let unit = map.length_unit;
    for (gate, _) in gates {
        let p = Vec3::new(gate.pos[0], gate.pos[1], gate.pos[2]) * unit;
        dot(&mut img, &p, 1, Rgb([200, 200, 200]));
    }
    for (k, t) in tours.iter().enumerate() {
        for p in t {
            dot(&mut img, p, 0, color(k));
        }
        if let Some(p) = t.first() {
            dot(&mut img, p, 3, color(k));
        }
    }
    img.save(out)
        .with_context(|| format!("writing {}", out.display()))
}

/// Finds, certifies and writes weaving tours to `out/certified/NNN/`.
pub fn run(
    capture_dir: &Path,
    raw_splat: bool,
    robot: &Robot,
    out: &Path,
    cfg: &WeaveConfig,
) -> Result<WeaveReport> {
    if cfg.floor_m.partial_cmp(&cfg.ceiling_m) != Some(Ordering::Less) {
        bail!("--floor must be below --ceiling");
    }
    let capture = Capture::open(capture_dir, raw_splat)?;
    let p = prepare(&capture, &MapConfig::default(), robot)?;
    if !p.scene.scale.is_metric() {
        bail!("{} is UNSCALED: nothing can be certified", capture.name);
    }
    let pcfg = robot.planner();
    let mut map = cap_map(&p.map, cfg.ceiling_m, &pcfg);
    let unit = map.length_unit;
    let need = pcfg.search_clearance(&map);
    let mut g = find_obstacles(&map, need, cfg);
    let extruded = extrude(&mut map, &g);
    tracing::info!(extruded, "obstacle footprints extruded to the ceiling");
    let res = map.spec.res;
    let folded = fold_verifier_solids(
        &mut map,
        &p.verifier,
        cfg.floor_m * unit - need - res,
        cfg.ceiling_m * unit + need + res,
    );
    tracing::info!(folded, "verifier solids marked in the planning map");
    let map = map;
    let ws = Workspace::new(&map);
    free_and_nearest(&mut g, &map, &ws, need, cfg);
    std::fs::create_dir_all(out)?;

    let free_cols = g.free.iter().filter(|f| f.is_some()).count();
    let ok = |q: &Vec3| system_endpoint_ok(&ws, &p.verifier, need, q);
    let gates = find_gates(&g, &map, &ok);
    let real = g.obstacles.iter().filter(|o| !o.wall).count();
    println!(
        "{}: below {:.2} m: {:.1} m² traversable, {} obstacles (+{} wall pieces), {} gates",
        capture.name,
        cfg.ceiling_m,
        free_cols as f64 * g.res * g.res / (unit * unit),
        real,
        g.obstacles.len() - real,
        gates.len()
    );
    if gates.len() < 4 {
        write_top(&g, &map, &gates, &[], &out.join("weave_top.png"))?;
        bail!(
            "only {} gates below {:.2} m: nothing to weave through (see {})",
            gates.len(),
            cfg.ceiling_m,
            out.join("weave_top.png").display()
        );
    }
    let links = gate_links(&g, &gates, cfg.max_leg_m * unit);
    let tours = search_tours(&gates, &g.obstacles, &links, unit, cfg);
    tracing::info!(
        links = links.iter().map(Vec::len).sum::<usize>(),
        tours = tours.len(),
        lens = ?tours.iter().map(|t| t.seq.len()).collect::<Vec<_>>(),
        "tour search"
    );

    // Certify the candidate tours (in parallel; results in candidate order).
    let results: Vec<Option<(Vec<usize>, PlanOutcome)>> = tours
        .par_iter()
        .map(|t| {
            let via: Vec<Vec3> = t
                .seq
                .iter()
                .map(|&k| Vec3::new(gates[k].0.pos[0], gates[k].0.pos[1], gates[k].0.pos[2]) * unit)
                .collect();
            certify_tour(&map, &p, &via, &pcfg)
                .map(|(kept, o)| (kept.iter().map(|&i| t.seq[i]).collect(), o))
        })
        .collect();

    // Measure and rank the certified ones.
    let body = (pcfg.robot_radius + 0.6) * unit;
    let mut done: Vec<(WeavePath, PlanOutcome)> = Vec::new();
    for (seq, o) in results.into_iter().flatten() {
        let length = o
            .waypoints
            .windows(2)
            .map(|w| (w[1] - w[0]).norm())
            .sum::<f64>()
            / unit;
        if length < cfg.min_length_m {
            continue;
        }
        let samples = o.trajectory.sample(0.05);
        let mut passed = BTreeSet::new();
        for s in &samples {
            if let Some(c) = map.spec.cell_of(&s.pos) {
                let col = g.idx(c[0], c[1]);
                if let Some((lab, _)) =
                    g.nearest[col].filter(|&(lab, d)| d <= body && !g.obstacles[lab].wall)
                {
                    passed.insert(lab);
                }
            }
        }
        let mut threaded = BTreeSet::new();
        for &k in &seq {
            for ob in gates[k].0.between {
                if !g.obstacles[ob].wall {
                    threaded.insert(ob);
                }
            }
        }
        let mut turning = 0.0;
        let pts: Vec<Vec3> =
            samples
                .iter()
                .map(|s| s.pos)
                .fold(Vec::new(), |mut v: Vec<Vec3>, q| {
                    if v.last().is_none_or(|l| ((q - l).xy()).norm() > 0.1 * unit) {
                        v.push(q);
                    }
                    v
                });
        for w in pts.windows(3) {
            let (a, b) = ((w[1] - w[0]).xy(), (w[2] - w[1]).xy());
            turning += (a.x * b.y - a.y * b.x).atan2(a.dot(&b)).abs();
        }
        let zs = samples.iter().map(|s| s.pos.z / unit);
        done.push((
            WeavePath {
                id: String::new(),
                length_m: length,
                duration_s: o.trajectory.duration(),
                gates: seq,
                obstacles_threaded: threaded.len(),
                obstacles_passed: passed.len(),
                turning_deg: turning.to_degrees(),
                max_z_m: zs.clone().fold(f64::NEG_INFINITY, f64::max),
                min_z_m: zs.fold(f64::INFINITY, f64::min),
                min_clearance_m: o.certificate.min_clearance_m,
                required_m: o.certificate.margin_required_m,
            },
            o,
        ));
    }
    let rank =
        |w: &WeavePath| w.obstacles_passed as f64 + 0.5 * w.gates.len() as f64 + 0.15 * w.length_m;
    done.sort_by(|a, b| {
        rank(&b.0)
            .total_cmp(&rank(&a.0))
            .then(a.0.gates.cmp(&b.0.gates))
    });
    // Drop tours that mostly fly where a better one already does: half of
    // their 0.5 m floor cells already covered.
    let cells = |o: &PlanOutcome| -> BTreeSet<(i64, i64)> {
        o.trajectory
            .sample(0.05)
            .iter()
            .map(|s| {
                (
                    (s.pos.x / (0.5 * unit)).floor() as i64,
                    (s.pos.y / (0.5 * unit)).floor() as i64,
                )
            })
            .collect()
    };
    let mut kept: Vec<(WeavePath, PlanOutcome)> = Vec::new();
    let mut covered: BTreeSet<(i64, i64)> = BTreeSet::new();
    for (w, o) in done {
        if kept.len() == cfg.count {
            break;
        }
        let mine = cells(&o);
        let repeat = mine.iter().filter(|c| covered.contains(*c)).count();
        if 2 * repeat <= mine.len() {
            covered.extend(mine);
            kept.push((w, o));
        }
    }

    let all = out.join("certified");
    if all.exists() {
        std::fs::remove_dir_all(&all)?;
    }
    let mut paths = Vec::new();
    let mut drawn = Vec::new();
    for (rank, (mut w, mut o)) in kept.into_iter().enumerate() {
        o.certificate.splat = Some(capture.splat_label());
        w.id = format!("{rank:03}");
        let dir = all.join(&w.id);
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
        drawn.push(o.trajectory.sample(0.05).iter().map(|s| s.pos).collect());
        paths.push(w);
    }
    write_top(&g, &map, &gates, &drawn, &out.join("weave_top.png"))?;

    let report = WeaveReport {
        capture: capture.name.clone(),
        splat: capture.splat_label(),
        config: cfg.clone(),
        obstacles: g.obstacles.clone(),
        gates: gates.iter().map(|(g, _)| g.clone()).collect(),
        tours_tried: tours.len(),
        paths,
        provenance: Provenance::capture(&format!(
            "splatplan weave {} --out {} --count {} --ceiling {} --floor {} --min-height {} --max-length {} --max-leg {} --min-length {} --min-obstacle {} --beam {} --candidates {}{}{}",
            capture_dir.display(),
            out.display(),
            cfg.count,
            cfg.ceiling_m,
            cfg.floor_m,
            cfg.min_height_m,
            cfg.max_length_m,
            cfg.max_leg_m,
            cfg.min_length_m,
            cfg.min_obstacle_m2,
            cfg.beam,
            cfg.candidates,
            if raw_splat { " --raw-splat" } else { "" },
            robot.flags()
        )),
    };
    std::fs::write(out.join("weave.json"), report.to_json()?)?;
    println!(
        "{} tours tried, {} certified and kept:",
        report.tours_tried,
        report.paths.len()
    );
    for w in &report.paths {
        println!(
            "  {}  {:.1} m, {:.0} s, {} gates between {} obstacles, passes {} obstacles, turns {:.0}°, z {:.2}-{:.2} m; clearance {:.3} m (needed {:.3})",
            w.id,
            w.length_m,
            w.duration_s,
            w.gates.len(),
            w.obstacles_threaded,
            w.obstacles_passed,
            w.turning_deg,
            w.min_z_m,
            w.max_z_m,
            w.min_clearance_m,
            w.required_m
        );
    }
    println!("top view: {}", out.join("weave_top.png").display());
    Ok(report)
}
