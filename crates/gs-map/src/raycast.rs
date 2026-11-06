//! Observed-free space: what the training cameras actually saw through.
//!
//! A splat only describes surfaces. Empty air and never-seen space look the
//! same (no Gaussians). The only evidence that a voxel is *empty* is that a
//! camera looked through it and saw something farther away. So: from every
//! training camera, cast rays through a lattice of pixels and mark each voxel
//! the ray crosses as free, stopping at the first occupied voxel.
//! Everything never marked stays unknown.

use rayon::prelude::*;
use splat_types::{CameraPose, Vec3};

use crate::grid::GridSpec;

/// Packed bitset over voxels.
#[derive(Debug, Clone, PartialEq)]
pub struct BitSet {
    words: Vec<u64>,
}

impl BitSet {
    /// All-zero bitset with `n` bits.
    pub fn new(n: usize) -> Self {
        BitSet {
            words: vec![0; n.div_ceil(64)],
        }
    }
    /// Sets bit `i`.
    pub fn set(&mut self, i: usize) {
        self.words[i / 64] |= 1 << (i % 64);
    }
    /// Reads bit `i`.
    pub fn get(&self, i: usize) -> bool {
        (self.words[i / 64] >> (i % 64)) & 1 == 1
    }
    /// In-place union. OR is commutative and associative, so the result of a
    /// parallel reduction does not depend on thread count or order.
    pub fn union_with(&mut self, other: &BitSet) {
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a |= *b;
        }
    }
}

/// Ray-cast settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayCastConfig {
    /// Neighboring rays are at most this far apart (scene units) at
    /// `dense_range` from the camera. Usually one voxel.
    pub ray_spacing: f64,
    /// Distance up to which rays are dense enough to touch every voxel.
    pub dense_range: f64,
    /// Stop rays after this distance from the camera.
    pub max_range: f64,
}

impl RayCastConfig {
    /// Pixel step for a camera with focal length `f` (pixels): rays one
    /// `ray_spacing` apart at `dense_range` are `f · spacing / range` pixels
    /// apart. Never finer than a quarter pixel nor coarser than 64 pixels.
    pub fn pixel_step(&self, f: f64) -> f64 {
        (f * self.ray_spacing / self.dense_range).clamp(0.25, 64.0)
    }
}

/// Marks voxels seen through by `cameras` (stopping at `occupied` voxels).
pub fn observed_free(
    spec: &GridSpec,
    occupied: &BitSet,
    cameras: &[CameraPose],
    cfg: &RayCastConfig,
) -> BitSet {
    cameras
        .par_iter()
        .fold(
            || BitSet::new(spec.len()),
            |mut free, cam| {
                let k = &cam.intrinsics;
                let origin = cam.center();
                // Ray lattice spaced so neighboring rays are ≤ ray_spacing
                // apart out to dense_range: sparser lattices leave speckles
                // of never-crossed (Unknown) voxels between rays.
                let step = cfg.pixel_step(k.fx.min(k.fy));
                let nu = (k.width as f64 / step).ceil() as usize;
                let nv = (k.height as f64 / step).ceil() as usize;
                for iv in 0..nv {
                    let v = (iv as f64 + 0.5) * step;
                    for iu in 0..nu {
                        let u = (iu as f64 + 0.5) * step;
                        let dir = cam.pixel_ray(u, v);
                        cast_ray(spec, occupied, &origin, &dir, cfg.max_range, &mut free);
                    }
                }
                free
            },
        )
        .reduce(
            || BitSet::new(spec.len()),
            |mut a, b| {
                a.union_with(&b);
                a
            },
        )
}

/// Amanatides–Woo 3D DDA. Visits every voxel the ray passes through, in
/// order, marking each free until it reaches an occupied voxel, leaves the
/// grid or exceeds `max_range`. Returns the number of voxels marked.
pub fn cast_ray(
    spec: &GridSpec,
    occupied: &BitSet,
    origin: &Vec3,
    dir: &Vec3,
    max_range: f64,
    free: &mut BitSet,
) -> usize {
    let b = spec.bounds();
    // Slab test: parameter interval where the ray is inside the grid box.
    let mut t_enter: f64 = 0.0;
    let mut t_exit = max_range;
    for a in 0..3 {
        if dir[a].abs() < 1e-15 {
            if origin[a] < b.min[a] || origin[a] >= b.max[a] {
                return 0;
            }
        } else {
            let t0 = (b.min[a] - origin[a]) / dir[a];
            let t1 = (b.max[a] - origin[a]) / dir[a];
            t_enter = t_enter.max(t0.min(t1));
            t_exit = t_exit.min(t0.max(t1));
        }
    }
    if t_enter >= t_exit {
        return 0;
    }
    // Start cell: nudge slightly inside to avoid sitting on a face.
    let start = origin + dir * (t_enter + 1e-9 * spec.res);
    let q = spec.continuous(&start);
    let mut cell = [0i64; 3];
    let mut step = [0i64; 3];
    let mut t_max = [f64::INFINITY; 3];
    let mut t_delta = [f64::INFINITY; 3];
    for a in 0..3 {
        cell[a] = (q[a].floor() as i64).clamp(0, spec.dims[a] as i64 - 1);
        if dir[a] > 0.0 {
            step[a] = 1;
            let boundary = spec.origin[a] + (cell[a] + 1) as f64 * spec.res;
            t_max[a] = (boundary - origin[a]) / dir[a];
            t_delta[a] = spec.res / dir[a];
        } else if dir[a] < 0.0 {
            step[a] = -1;
            let boundary = spec.origin[a] + cell[a] as f64 * spec.res;
            t_max[a] = (boundary - origin[a]) / dir[a];
            t_delta[a] = -spec.res / dir[a];
        }
    }
    let mut marked = 0;
    let mut t = t_enter;
    // Linear index maintained incrementally: one add per step instead of a
    // multiply-add per axis (this loop runs billions of times per map).
    let strides = [
        1i64,
        spec.dims[0] as i64,
        (spec.dims[0] * spec.dims[1]) as i64,
    ];
    let mut idx = spec.index([cell[0] as usize, cell[1] as usize, cell[2] as usize]) as i64;
    let dims = [
        spec.dims[0] as i64,
        spec.dims[1] as i64,
        spec.dims[2] as i64,
    ];
    loop {
        if occupied.get(idx as usize) {
            break;
        }
        free.set(idx as usize);
        marked += 1;
        // Advance along the axis whose boundary comes first (ties: lowest axis).
        let mut a = 0;
        if t_max[1] < t_max[a] {
            a = 1;
        }
        if t_max[2] < t_max[a] {
            a = 2;
        }
        t = t.max(t_max[a]);
        if t >= t_exit {
            break;
        }
        cell[a] += step[a];
        if cell[a] < 0 || cell[a] >= dims[a] {
            break;
        }
        idx += step[a] * strides[a];
        t_max[a] += t_delta[a];
    }
    marked
}

#[cfg(test)]
mod tests {
    use super::*;
    use splat_types::Aabb;

    fn spec() -> GridSpec {
        GridSpec::covering(&Aabb::new(Vec3::zeros(), Vec3::new(1.0, 1.0, 1.0)), 0.1)
    }

    #[test]
    fn axis_ray_stops_at_obstacle() {
        let s = spec();
        let mut occ = BitSet::new(s.len());
        occ.set(s.index([6, 5, 5]));
        let mut free = BitSet::new(s.len());
        let n = cast_ray(
            &s,
            &occ,
            &Vec3::new(-1.0, 0.55, 0.55),
            &Vec3::x(),
            100.0,
            &mut free,
        );
        assert_eq!(n, 6);
        for i in 0..6 {
            assert!(free.get(s.index([i, 5, 5])));
        }
        assert!(!free.get(s.index([6, 5, 5])));
        assert!(!free.get(s.index([7, 5, 5])));
    }

    #[test]
    fn range_limit_and_miss() {
        let s = spec();
        let occ = BitSet::new(s.len());
        let mut free = BitSet::new(s.len());
        // Starts inside, 0.22 range: covers cells 0,1,2 along x.
        let n = cast_ray(
            &s,
            &occ,
            &Vec3::new(0.05, 0.05, 0.05),
            &Vec3::x(),
            0.25,
            &mut free,
        );
        assert_eq!(n, 3);
        // Misses the grid entirely.
        let n = cast_ray(
            &s,
            &occ,
            &Vec3::new(-1.0, 5.0, 0.5),
            &Vec3::x(),
            100.0,
            &mut free,
        );
        assert_eq!(n, 0);
    }

    #[test]
    fn diagonal_ray_is_face_connected() {
        // Every consecutive pair of visited cells must share a face: no
        // diagonal leaks through obstacle corners.
        let s = spec();
        let occ = BitSet::new(s.len());
        let mut free = BitSet::new(s.len());
        let dir = Vec3::new(1.0, 0.7, 0.3).normalize();
        let n = cast_ray(
            &s,
            &occ,
            &Vec3::new(0.01, 0.02, 0.03),
            &dir,
            100.0,
            &mut free,
        );
        // A ray crossing x: 10 cells, y: 7, z: 3 → 1 + 9 + 7 + 3 − overlap ≈ 18-20 cells.
        assert!((17..=21).contains(&n), "visited {n}");
        // Block a cell on the path and check nothing beyond it is marked.
        let mut occ2 = BitSet::new(s.len());
        let blocked = s
            .cell_of(&(Vec3::new(0.01, 0.02, 0.03) + dir * 0.5))
            .unwrap();
        occ2.set(s.index(blocked));
        let mut free2 = BitSet::new(s.len());
        let n2 = cast_ray(
            &s,
            &occ2,
            &Vec3::new(0.01, 0.02, 0.03),
            &dir,
            100.0,
            &mut free2,
        );
        assert!(n2 < n);
        let far = s
            .cell_of(&(Vec3::new(0.01, 0.02, 0.03) + dir * 0.9))
            .unwrap();
        assert!(!free2.get(s.index(far)));
    }
}
