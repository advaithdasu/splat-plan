//! Line-of-sight shortcutting with conservative distance-field sampling.

use splat_types::Vec3;

use crate::workspace::Workspace;

/// Sample spacing along segments, as a fraction of the voxel size.
const STEP_FRACTION: f64 = 0.2;

/// True if every point of segment `a`–`b` has at least `clearance` of
/// conservative clearance.
///
/// Samples are spaced `δ ≤ res/5` apart. Any point of the segment is within
/// `δ/2` of a sample, and clearance is 1-Lipschitz, so requiring
/// `clearance + δ/2` at every sample guarantees `clearance` everywhere.
pub fn segment_clear(ws: &Workspace, a: &Vec3, b: &Vec3, clearance: f64) -> bool {
    let len = (b - a).norm();
    let max_step = ws.map.spec.res * STEP_FRACTION;
    let n = ((len / max_step).ceil() as usize).max(1);
    let step = len / n as f64;
    let need = clearance + step / 2.0;
    (0..=n).all(|k| {
        let p = a + (b - a) * (k as f64 / n as f64);
        ws.clearance(&p) >= need
    })
}

/// Greedy shortcutting: from each kept point, jump to the farthest later
/// point that is visible (clear segment). Returns `None` if even consecutive
/// points are not connected by a clear segment, in which case the caller
/// must refuse rather than return an unchecked path.
pub fn shortcut(ws: &Workspace, path: &[Vec3], clearance: f64) -> Option<Vec<Vec3>> {
    let first = *path.first()?;
    let mut out = vec![first];
    let mut i = 0;
    while i + 1 < path.len() {
        let mut next = None;
        for j in (i + 1..path.len()).rev() {
            if segment_clear(ws, &path[i], &path[j], clearance) {
                next = Some(j);
                break;
            }
        }
        let j = next?;
        out.push(path[j]);
        i = j;
    }
    Some(out)
}
