//! 26-connected A* over the voxel grid, with deterministic tie-breaking.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use gs_map::VoxelState;
use splat_types::Vec3;

use crate::PlanError;
use crate::workspace::Workspace;

/// Heap entry. Ordered so the `BinaryHeap` (a max-heap) pops the smallest
/// `f`; ties go to the smaller `h` (closer to the goal), then to the smaller
/// voxel index. With a total order there is exactly one possible expansion
/// sequence, so plans are reproducible.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Node {
    f: f64,
    h: f64,
    idx: u32,
}

impl Eq for Node {}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .f
            .total_cmp(&self.f)
            .then_with(|| other.h.total_cmp(&self.h))
            .then_with(|| other.idx.cmp(&self.idx))
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A voxel is traversable for A* if it is observed-free and its center has
/// at least `clearance` of conservative clearance (map and keep-outs).
pub fn cell_ok(ws: &Workspace, c: [usize; 3], clearance: f64) -> bool {
    let map = ws.map;
    let idx = map.spec.index(c);
    map.state[idx] == VoxelState::FreeObserved
        && map.dist[idx] as f64 - map.spec.res * 3f64.sqrt() / 2.0 >= clearance
        && ws.keepout_distance(&map.spec.center(c)) >= clearance
}

/// Finds a shortest 26-connected path of voxel centers from the voxel of
/// `start` to the voxel of `goal`, through voxels satisfying [`cell_ok`]
/// (the start and goal voxels themselves are exempt; see the body).
///
/// Returns voxel centers (start and goal voxels included).
pub fn astar(
    ws: &Workspace,
    start: &Vec3,
    goal: &Vec3,
    clearance: f64,
    max_expansions: usize,
) -> Result<Vec<Vec3>, PlanError> {
    let spec = &ws.map.spec;
    let s = spec
        .cell_of(start)
        .ok_or(PlanError::OutsideMap { which: "start" })?;
    let g = spec
        .cell_of(goal)
        .ok_or(PlanError::OutsideMap { which: "goal" })?;
    // The start and goal voxels are not required to pass `cell_ok`: the
    // query points themselves were validated by the caller, and the final
    // segment check (`shortcut`) re-verifies the connection from each point
    // to its voxel center. Requiring it here would refuse valid endpoints
    // whose own voxel center happens to sit a few millimetres closer to an
    // obstacle than the point.
    let goal_c = spec.center(g);
    let h_of = |c: [usize; 3]| (spec.center(c) - goal_c).norm();

    let s_idx = spec.index(s) as u32;
    let g_idx = spec.index(g) as u32;
    // Only point lookups are used, never iteration, so HashMap order cannot
    // leak into the result.
    let mut best: HashMap<u32, (f64, u32)> = HashMap::new();
    let mut closed: HashMap<u32, ()> = HashMap::new();
    let mut heap = BinaryHeap::new();
    best.insert(s_idx, (0.0, s_idx));
    heap.push(Node {
        f: h_of(s),
        h: h_of(s),
        idx: s_idx,
    });
    let mut nbrs = Vec::with_capacity(26);
    let mut expanded = 0usize;
    while let Some(Node { idx, .. }) = heap.pop() {
        if closed.insert(idx, ()).is_some() {
            continue; // stale duplicate
        }
        if idx == g_idx {
            // Walk parents back to the start.
            let mut path = vec![spec.center(spec.coords(idx as usize))];
            let mut cur = idx;
            while cur != s_idx {
                cur = best[&cur].1;
                path.push(spec.center(spec.coords(cur as usize)));
            }
            path.reverse();
            return Ok(path);
        }
        expanded += 1;
        if expanded > max_expansions {
            return Err(PlanError::NoPath {
                reason: format!("search limit of {max_expansions} expansions reached"),
                expanded,
            });
        }
        let g_cur = best[&idx].0;
        let c = spec.coords(idx as usize);
        spec.neighbors26(c, &mut nbrs);
        for &(n, d) in &nbrs {
            let n_idx = spec.index(n) as u32;
            if closed.contains_key(&n_idx) || (n_idx != g_idx && !cell_ok(ws, n, clearance)) {
                continue;
            }
            let step = ((d[0] as f64).powi(2) + (d[1] as f64).powi(2) + (d[2] as f64).powi(2))
                .sqrt()
                * spec.res;
            let g_new = g_cur + step;
            if best.get(&n_idx).is_none_or(|(g_old, _)| g_new < *g_old) {
                best.insert(n_idx, (g_new, idx));
                let h = h_of(n);
                heap.push(Node {
                    f: g_new + h,
                    h,
                    idx: n_idx,
                });
            }
        }
    }
    Err(PlanError::NoPath {
        reason: "start and goal are not connected through observed-free space with the required clearance".into(),
        expanded,
    })
}
