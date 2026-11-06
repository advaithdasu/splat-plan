//! Morphological closing (dilate, then erode) of the occupancy grid.
//!
//! Glossy car paint produces sparse or hollow bodywork in a splat. Closing
//! with a small ball fills pin-holes and thin gaps in surfaces without
//! growing objects overall.
//!
//! Both steps reuse the exact distance transform, so they are parallel,
//! deterministic, and use a true Euclidean ball as structuring element:
//!
//! - dilate: `D = { v : dist(v, occupied) ≤ r }`
//! - erode:  `E = { v : dist(v, not D) > r }`
//!
//! For erosion, cells outside the grid are *not* counted as "not D", so
//! objects touching the grid boundary are not eaten away from that side.

use crate::edt::{INF, edt_3d};

/// Closing with a ball of radius `r` voxels (`r <= 0` returns the input).
/// `r = 1.5` reaches face and edge neighbors, enough to bridge one-voxel
/// holes in a surface (a unit ball, face neighbors only, cannot).
pub fn close(occ: &[bool], dims: [usize; 3], r: f64) -> Vec<bool> {
    if r <= 0.0 {
        return occ.to_vec();
    }
    let r2 = (r * r) as f32;
    // Squared distance to the nearest occupied voxel.
    let mut d: Vec<f32> = occ.iter().map(|o| if *o { 0.0 } else { INF }).collect();
    edt_3d(&mut d, dims);
    let dilated: Vec<bool> = d.iter().map(|v| *v <= r2).collect();
    // Squared distance to the nearest voxel outside the dilated set.
    let mut e: Vec<f32> = dilated.iter().map(|o| if *o { INF } else { 0.0 }).collect();
    edt_3d(&mut e, dims);
    e.iter().map(|v| *v > r2).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closing_fills_hole_and_keeps_shape() {
        let dims = [9, 9, 9];
        let idx = |x: usize, y: usize, z: usize| x + 9 * (y + 9 * z);
        // A 5x5 wall at z = 4 with a one-voxel hole.
        let mut occ = vec![false; 729];
        for x in 2..7 {
            for y in 2..7 {
                occ[idx(x, y, 4)] = true;
            }
        }
        occ[idx(4, 4, 4)] = false;
        let c = close(&occ, dims, 1.5);
        assert!(c[idx(4, 4, 4)], "hole filled");
        assert!(!c[idx(4, 4, 3)] && !c[idx(4, 4, 5)], "wall not thickened");
        assert!(!c[idx(1, 4, 4)], "wall not widened");
        assert_eq!(close(&occ, dims, 0.0), occ);
        // Closing is extensive: it never removes occupied voxels.
        assert!(occ.iter().zip(&c).all(|(a, b)| !*a || *b));
    }

    #[test]
    fn closing_never_creates_occupancy_at_the_grid_boundary() {
        // Regression: an earlier erosion returned `true` whenever its window
        // touched the grid edge, turning a shell around the grid occupied.
        let dims = [6, 5, 4];
        let empty = vec![false; 120];
        assert_eq!(close(&empty, dims, 1.5), empty);
        assert_eq!(close(&empty, dims, 2.5), empty);
        // An object touching the boundary is kept, not eroded away.
        let mut occ = vec![false; 120];
        for y in 0..5 {
            for z in 0..4 {
                occ[6 * (y + 5 * z)] = true; // the x = 0 face
            }
        }
        assert_eq!(close(&occ, dims, 1.5), occ);
    }

    #[test]
    fn closing_bridges_a_two_voxel_gap_with_radius_two() {
        let dims = [12, 3, 3];
        let idx = |x: usize, y: usize, z: usize| x + 12 * (y + 3 * z);
        let mut occ = vec![false; 108];
        for x in (0..5).chain(7..12) {
            occ[idx(x, 1, 1)] = true;
        }
        assert!(!close(&occ, dims, 1.0)[idx(5, 1, 1)]);
        let c = close(&occ, dims, 2.0);
        assert!(c[idx(5, 1, 1)] && c[idx(6, 1, 1)]);
    }
}
