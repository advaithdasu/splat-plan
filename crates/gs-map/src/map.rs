//! The collision map: voxel states plus a Euclidean distance field.

use std::time::Instant;

use serde::{Deserialize, Serialize};
use splat_types::{Aabb, Report, Scale, Scene, Vec3};
use thiserror::Error;

use crate::density::DensityGrid;
use crate::edt::{INF, edt_3d};
use crate::grid::GridSpec;
use crate::morph::close;
use crate::raycast::{BitSet, RayCastConfig, observed_free};

/// Three-valued voxel state. Unknown is not free.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum VoxelState {
    /// Density above threshold: something is here.
    Occupied = 0,
    /// A training camera saw through this voxel: known empty.
    FreeObserved = 1,
    /// Never observed: treated exactly like an obstacle.
    Unknown = 2,
}

/// Errors building a map.
#[derive(Debug, Error, PartialEq)]
pub enum MapError {
    /// No Gaussians to build from.
    #[error("scene has no Gaussians")]
    Empty,
    /// The grid would be too large.
    #[error(
        "grid of {dims:?} = {voxels} voxels exceeds the limit of {limit}; increase the voxel size"
    )]
    TooLarge {
        /// Grid dimensions.
        dims: [usize; 3],
        /// Voxel count.
        voxels: usize,
        /// Configured limit.
        limit: usize,
    },
    /// A configuration value is out of range.
    #[error("invalid map config: {0}")]
    BadConfig(String),
}

/// Map-building settings. Lengths are in meters for metric scenes; for
/// unscaled scenes they are multiplied by a guessed unit (see [`MapReport::length_unit`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MapConfig {
    /// Voxel edge length.
    pub voxel: f64,
    /// `Occupied` iff density ≥ this.
    pub density_threshold: f32,
    /// Gaussians contribute inside their `k_sigma` ellipsoid.
    pub k_sigma: f64,
    /// Gaussians below this opacity are ignored.
    pub min_opacity: f32,
    /// Morphological closing with a ball of this radius in voxels (0 disables).
    pub closing_radius: f64,
    /// Bounds use the `p`-th and `(100 − p)`-th percentiles of the means.
    pub bounds_percentile: f64,
    /// Padding added to the percentile box on every side except the top.
    pub pad: f64,
    /// Extra room above the top percentile (flight space).
    pub headroom: f64,
    /// Rays are at most one voxel apart up to this distance from each camera.
    pub dense_ray_range: f64,
    /// Maximum ray length from a camera.
    pub max_ray_range: f64,
    /// Refuse to build grids bigger than this.
    pub max_voxels: usize,
}

impl Default for MapConfig {
    fn default() -> Self {
        MapConfig {
            voxel: 0.05,
            density_threshold: 0.2,
            k_sigma: 3.0,
            min_opacity: 0.02,
            closing_radius: 1.5,
            bounds_percentile: 1.0,
            pad: 1.0,
            headroom: 2.5,
            dense_ray_range: 7.0,
            max_ray_range: 15.0,
            max_voxels: 60_000_000,
        }
    }
}

/// Timings and counts from a map build.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MapReport {
    /// Grid dimensions.
    pub dims: [usize; 3],
    /// Voxel size (scene units).
    pub voxel: f64,
    /// Grid origin.
    pub origin: [f64; 3],
    /// Scale state of the scene.
    pub scale: Scale,
    /// Scene units per "config meter": 1 for metric scenes, a guess otherwise.
    pub length_unit: f64,
    /// Gaussians in the input.
    pub gaussians: usize,
    /// Cameras used for ray casting.
    pub cameras: usize,
    /// Voxel counts.
    pub occupied: usize,
    /// Voxels seen through by a camera.
    pub free_observed: usize,
    /// Voxels never observed.
    pub unknown: usize,
    /// Wall-clock milliseconds per stage (not deterministic; excluded from hashes).
    pub timing_ms: StageTimings,
}

/// Per-stage wall-clock times in milliseconds.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StageTimings {
    /// Density accumulation.
    pub density: f64,
    /// Thresholding + closing.
    pub occupancy: f64,
    /// Observed-free ray cast.
    pub raycast: f64,
    /// Distance transform.
    pub esdf: f64,
    /// Total.
    pub total: f64,
}

impl Report for MapReport {}

/// Voxel states plus the distance field.
#[derive(Debug, Clone, PartialEq)]
pub struct Map {
    /// Grid geometry.
    pub spec: GridSpec,
    /// Per-voxel state.
    pub state: Vec<VoxelState>,
    /// Distance (scene units) from each voxel center to the nearest
    /// non-`FreeObserved` voxel center or the outside of the grid. 0 for
    /// non-free voxels.
    pub dist: Vec<f32>,
    /// Scale of the scene the map was built from.
    pub scale: Scale,
    /// Scene units per meter: 1 for metric scenes, a rough guess for
    /// unscaled ones (so robot sizes can still be applied; outputs stay UNSCALED).
    pub length_unit: f64,
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn percentile_sorted(v: &[f64], p: f64) -> f64 {
    let pos = (p / 100.0) * (v.len() - 1) as f64;
    v[pos.round() as usize]
}

/// Per-axis `p`-th to `(100 − p)`-th percentile box of the Gaussian means.
fn percentile_box(scene: &Scene, p: f64) -> Option<Aabb> {
    if scene.gaussians.is_empty() {
        return None;
    }
    let mut lo = Vec3::zeros();
    let mut hi = Vec3::zeros();
    for a in 0..3 {
        let mut v: Vec<f64> = scene.gaussians.iter().map(|g| g.mean[a] as f64).collect();
        v.sort_by(f64::total_cmp);
        lo[a] = percentile_sorted(&v, p);
        hi[a] = percentile_sorted(&v, 100.0 - p);
    }
    Some(Aabb::new(lo, hi))
}

/// Map bounds: the percentile box of the Gaussian means grown to contain
/// every camera center, padded on every side, plus headroom above.
///
/// The cameras matter for fly-through captures: at the end of a hallway
/// the camera stands where few Gaussians are, outside the percentile box,
/// and a map that excludes the camera path cannot plan along it.
pub fn scene_bounds(scene: &Scene, cfg: &MapConfig) -> Option<Aabb> {
    let core = percentile_box(scene, cfg.bounds_percentile)?;
    let core = Aabb::from_points(
        [core.min, core.max]
            .into_iter()
            .chain(scene.cameras.iter().map(|c| c.center())),
    )?;
    let mut b = core.padded(cfg.pad);
    b.max.z = core.max.z + cfg.headroom;
    Some(b)
}

/// Guessed scene units per meter for unscaled scenes: assume the robust
/// extent of the scene is about 15 m (the synthetic car scene). Only used to
/// pick sensible voxel sizes; every output is still marked UNSCALED.
pub fn length_unit_guess(scene: &Scene) -> f64 {
    match percentile_box(scene, MapConfig::default().bounds_percentile) {
        Some(b) => (b.extent().max() / 15.0).max(1e-9),
        None => 1.0,
    }
}

impl MapConfig {
    /// Config with every length multiplied by `unit`.
    fn scaled(&self, unit: f64) -> MapConfig {
        MapConfig {
            voxel: self.voxel * unit,
            pad: self.pad * unit,
            headroom: self.headroom * unit,
            dense_ray_range: self.dense_ray_range * unit,
            max_ray_range: self.max_ray_range * unit,
            ..self.clone()
        }
    }
}

/// Builds the collision map for a scene.
pub fn build_map(scene: &Scene, cfg: &MapConfig) -> Result<(Map, MapReport), MapError> {
    let t_total = Instant::now();
    if !(cfg.voxel > 0.0 && cfg.voxel.is_finite()) {
        return Err(MapError::BadConfig(format!("voxel {}", cfg.voxel)));
    }
    if scene.gaussians.is_empty() {
        return Err(MapError::Empty);
    }
    let unit = match scene.scale {
        Scale::Metric(_) => 1.0,
        Scale::Unscaled => {
            tracing::warn!("{}", Scale::UNSCALED_BANNER);
            length_unit_guess(scene)
        }
    };
    let cfg = cfg.scaled(unit);
    let bounds = scene_bounds(scene, &cfg).ok_or(MapError::Empty)?;
    let spec = GridSpec::covering(&bounds, cfg.voxel);
    let voxels = spec
        .dims
        .iter()
        .try_fold(1usize, |acc, d| acc.checked_mul(*d));
    match voxels {
        Some(v) if v <= cfg.max_voxels => {}
        _ => {
            return Err(MapError::TooLarge {
                dims: spec.dims,
                voxels: voxels.unwrap_or(usize::MAX),
                limit: cfg.max_voxels,
            });
        }
    }

    let t = Instant::now();
    let density = DensityGrid::accumulate(spec, &scene.gaussians, cfg.k_sigma, cfg.min_opacity);
    let t_density = ms(t);

    let t = Instant::now();
    let raw: Vec<bool> = density
        .density
        .iter()
        .map(|d| *d >= cfg.density_threshold)
        .collect();
    let occ_bool = close(&raw, spec.dims, cfg.closing_radius);
    let mut occupied = BitSet::new(spec.len());
    for (i, o) in occ_bool.iter().enumerate() {
        if *o {
            occupied.set(i);
        }
    }
    let t_occ = ms(t);

    let t = Instant::now();
    let free = observed_free(
        &spec,
        &occupied,
        &scene.cameras,
        &RayCastConfig {
            ray_spacing: cfg.voxel,
            dense_range: cfg.dense_ray_range,
            max_range: cfg.max_ray_range,
        },
    );
    let state: Vec<VoxelState> = (0..spec.len())
        .map(|i| {
            if occ_bool[i] {
                VoxelState::Occupied
            } else if free.get(i) {
                VoxelState::FreeObserved
            } else {
                VoxelState::Unknown
            }
        })
        .collect();
    let t_ray = ms(t);

    let t = Instant::now();
    let dist = distance_field(&spec, &state);
    let t_esdf = ms(t);

    let count = |s: VoxelState| state.iter().filter(|v| **v == s).count();
    let report = MapReport {
        dims: spec.dims,
        voxel: spec.res,
        origin: spec.origin.into(),
        scale: scene.scale,
        length_unit: unit,
        gaussians: scene.gaussians.len(),
        cameras: scene.cameras.len(),
        occupied: count(VoxelState::Occupied),
        free_observed: count(VoxelState::FreeObserved),
        unknown: count(VoxelState::Unknown),
        timing_ms: StageTimings {
            density: t_density,
            occupancy: t_occ,
            raycast: t_ray,
            esdf: t_esdf,
            total: ms(t_total),
        },
    };
    tracing::info!(?report.dims, report.occupied, report.free_observed, report.unknown, "map built");
    Ok((
        Map {
            spec,
            state,
            dist,
            scale: scene.scale,
            length_unit: unit,
        },
        report,
    ))
}

/// Exact distance (scene units) from each voxel center to the nearest
/// non-free voxel center, also bounded by the distance to the cells just
/// outside the grid (the outside is unknown, hence an obstacle).
pub fn distance_field(spec: &GridSpec, state: &[VoxelState]) -> Vec<f32> {
    let mut d2: Vec<f32> = state
        .iter()
        .map(|s| {
            if *s == VoxelState::FreeObserved {
                INF
            } else {
                0.0
            }
        })
        .collect();
    edt_3d(&mut d2, spec.dims);
    let res = spec.res as f32;
    d2.iter()
        .enumerate()
        .map(|(idx, v)| {
            let c = spec.coords(idx);
            // Nearest outside cell along each axis: index −1 or dims[a].
            let edge = (0..3)
                .map(|a| (c[a] + 1).min(spec.dims[a] - c[a]))
                .min()
                .unwrap_or(0) as f32;
            v.sqrt().min(edge) * res
        })
        .collect()
}

/// Conservative lower bound on the distance from `p` to the obstacle set of
/// a voxel distance field `dist` (distance from each voxel center to the
/// nearest obstacle voxel center, scene units): the best of
/// `dist(c) − |p − c|` over the 8 voxel centers around `p`, minus the
/// obstacle voxels' half-diagonal. See [`Map::clearance`].
pub fn lipschitz_clearance(spec: &GridSpec, dist: &[f32], p: &Vec3) -> f64 {
    let half_diag = spec.res * 3f64.sqrt() / 2.0;
    let q = spec.continuous(p) - Vec3::repeat(0.5);
    let base = [q.x.floor(), q.y.floor(), q.z.floor()];
    let mut best = f64::NEG_INFINITY;
    for corner in 0..8 {
        let mut c = [0usize; 3];
        let mut inside = true;
        for a in 0..3 {
            let v = base[a] + ((corner >> a) & 1) as f64;
            if v < 0.0 || v >= spec.dims[a] as f64 {
                inside = false;
                break;
            }
            c[a] = v as usize;
        }
        if !inside {
            continue; // outside cells have distance 0: they cannot raise the bound
        }
        let d = dist[spec.index(c)] as f64 - (spec.center(c) - p).norm();
        best = best.max(d);
    }
    if best == f64::NEG_INFINITY {
        return -half_diag;
    }
    best - half_diag
}

impl Map {
    /// State of the voxel containing `p` (`Unknown` outside the grid).
    pub fn state_at(&self, p: &Vec3) -> VoxelState {
        match self.spec.cell_of(p) {
            Some(c) => self.state[self.spec.index(c)],
            None => VoxelState::Unknown,
        }
    }

    /// Distance stored at voxel `c`.
    pub fn dist_at(&self, c: [usize; 3]) -> f64 {
        self.dist[self.spec.index(c)] as f64
    }

    /// **Conservative clearance**: a lower bound on the distance from `p` to
    /// the nearest non-free region (occupied, unknown, or outside the grid).
    ///
    /// For each of the 8 voxel centers `c` surrounding `p`, the distance field
    /// is 1-Lipschitz, so `D(p) ≥ d(c) − |p − c|`; obstacle voxels are cubes,
    /// so subtract their half-diagonal `res·√3/2`. At a voxel center this is
    /// exactly "distance minus res·√3/2"; between centers it stays
    /// a guaranteed bound where plain trilinear interpolation is not.
    /// Returns a value ≤ 0 when no clearance is guaranteed.
    pub fn clearance(&self, p: &Vec3) -> f64 {
        lipschitz_clearance(&self.spec, &self.dist, p)
    }

    /// True if `p` is in observed-free space with at least `radius` of
    /// conservative clearance. This is the planner's only notion of "safe".
    pub fn traversable(&self, p: &Vec3, radius: f64) -> bool {
        self.state_at(p) == VoxelState::FreeObserved && self.clearance(p) >= radius
    }

    /// Stable 64-bit FNV-1a hash of the states and distances, for
    /// determinism tests.
    pub fn content_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |b: u8| {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        };
        for s in &self.state {
            eat(*s as u8);
        }
        for d in &self.dist {
            for b in d.to_bits().to_le_bytes() {
                eat(b);
            }
        }
        h
    }

    /// Horizontal slice of the distance field at height `z` (row-major, y rows),
    /// with negative values marking occupied (−1) and unknown (−2) voxels.
    pub fn slice_z(&self, z: f64) -> Option<(usize, usize, Vec<f32>)> {
        let k = ((z - self.spec.origin.z) / self.spec.res).floor();
        if k < 0.0 || k >= self.spec.dims[2] as f64 {
            return None;
        }
        let k = k as usize;
        let [nx, ny, _] = self.spec.dims;
        let mut out = Vec::with_capacity(nx * ny);
        for j in 0..ny {
            for i in 0..nx {
                let idx = self.spec.index([i, j, k]);
                out.push(match self.state[idx] {
                    VoxelState::Occupied => -1.0,
                    VoxelState::Unknown => -2.0,
                    VoxelState::FreeObserved => self.dist[idx],
                });
            }
        }
        Some((nx, ny, out))
    }
}
