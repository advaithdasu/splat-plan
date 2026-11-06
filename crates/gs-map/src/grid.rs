//! Voxel grid geometry: index math shared by every layer of the map.

use serde::{Deserialize, Serialize};
use splat_types::{Aabb, Vec3};

/// Axis-aligned voxel grid. Voxel `(i, j, k)` covers
/// `origin + [i, i+1) × [j, j+1) × [k, k+1)` times `res`, with its center at
/// `origin + (i + ½, j + ½, k + ½)·res`. Storage is x-fastest:
/// `index = i + nx·(j + ny·k)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GridSpec {
    /// Minimum corner of voxel (0, 0, 0).
    pub origin: Vec3,
    /// Voxel edge length.
    pub res: f64,
    /// Number of voxels along x, y, z.
    pub dims: [usize; 3],
}

impl GridSpec {
    /// Grid covering `bounds` with voxels of size `res` (rounded up).
    pub fn covering(bounds: &Aabb, res: f64) -> Self {
        let e = bounds.extent();
        let dims = [0, 1, 2].map(|a| ((e[a] / res).ceil() as usize).max(1));
        GridSpec {
            origin: bounds.min,
            res,
            dims,
        }
    }

    /// Total voxel count.
    pub fn len(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    /// True if the grid has no voxels.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Linear index of `(i, j, k)` (caller guarantees it is in range).
    pub fn index(&self, c: [usize; 3]) -> usize {
        c[0] + self.dims[0] * (c[1] + self.dims[1] * c[2])
    }

    /// `(i, j, k)` of a linear index.
    pub fn coords(&self, idx: usize) -> [usize; 3] {
        let i = idx % self.dims[0];
        let j = (idx / self.dims[0]) % self.dims[1];
        let k = idx / (self.dims[0] * self.dims[1]);
        [i, j, k]
    }

    /// Center of voxel `(i, j, k)`.
    pub fn center(&self, c: [usize; 3]) -> Vec3 {
        self.origin + Vec3::new(c[0] as f64 + 0.5, c[1] as f64 + 0.5, c[2] as f64 + 0.5) * self.res
    }

    /// Continuous voxel coordinates of a point (voxel centers at `n + 0.5`).
    pub fn continuous(&self, p: &Vec3) -> Vec3 {
        (p - self.origin) / self.res
    }

    /// Voxel containing `p`, or `None` outside the grid.
    pub fn cell_of(&self, p: &Vec3) -> Option<[usize; 3]> {
        let q = self.continuous(p);
        let mut c = [0usize; 3];
        for a in 0..3 {
            if !(q[a] >= 0.0 && q[a] < self.dims[a] as f64) {
                return None;
            }
            c[a] = q[a] as usize;
        }
        Some(c)
    }

    /// World-space bounds of the whole grid.
    pub fn bounds(&self) -> Aabb {
        let max = self.origin
            + Vec3::new(
                self.dims[0] as f64,
                self.dims[1] as f64,
                self.dims[2] as f64,
            ) * self.res;
        Aabb {
            min: self.origin,
            max,
        }
    }

    /// Up to 26 neighbors of `c` that are inside the grid, with their step vectors.
    pub fn neighbors26(&self, c: [usize; 3], out: &mut Vec<([usize; 3], [i8; 3])>) {
        out.clear();
        for dz in -1i8..=1 {
            for dy in -1i8..=1 {
                for dx in -1i8..=1 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    let d = [dx, dy, dz];
                    let mut n = [0usize; 3];
                    let mut ok = true;
                    for a in 0..3 {
                        let v = c[a] as i64 + d[a] as i64;
                        if v < 0 || v >= self.dims[a] as i64 {
                            ok = false;
                            break;
                        }
                        n[a] = v as usize;
                    }
                    if ok {
                        out.push((n, d));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_roundtrip_and_cells() {
        let g = GridSpec::covering(
            &Aabb::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.5, 0.3)),
            0.1,
        );
        assert_eq!(g.dims, [20, 5, 3]);
        for idx in [0, 7, 99, g.len() - 1] {
            assert_eq!(g.index(g.coords(idx)), idx);
        }
        assert_eq!(g.cell_of(&Vec3::new(-0.95, 0.05, 0.05)), Some([0, 0, 0]));
        assert_eq!(g.cell_of(&Vec3::new(1.5, 0.0, 0.0)), None);
        assert!((g.center([0, 0, 0]) - Vec3::new(-0.95, 0.05, 0.05)).norm() < 1e-12);
        let mut n = Vec::new();
        g.neighbors26([0, 0, 0], &mut n);
        assert_eq!(n.len(), 7);
        g.neighbors26([5, 2, 1], &mut n);
        assert_eq!(n.len(), 26);
    }
}
