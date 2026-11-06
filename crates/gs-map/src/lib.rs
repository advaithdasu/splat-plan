//! Collision map: density grid, observed-free ray casting and Euclidean distance field.
//!
//! Pipeline ([`build_map`]):
//!
//! 1. [`DensityGrid`]: sum Gaussian densities at voxel centers (deterministic, z-slab parallel).
//! 2. Threshold to `Occupied`, then morphological closing to patch holes.
//! 3. [`observed_free`]: cast rays from the training cameras with 3D DDA; voxels
//!    seen through become `FreeObserved`. Everything else stays `Unknown`.
//! 4. [`distance_field`]: exact Euclidean distance transform (Felzenszwalb) to
//!    the nearest voxel that is *not* `FreeObserved`.
//! 5. [`Map::clearance`] / [`Map::traversable`]: conservative queries for the planner.
#![forbid(unsafe_code)]

mod density;
mod edt;
mod grid;
mod map;
mod morph;
mod raycast;

pub use density::DensityGrid;
pub use edt::{edt_1d, edt_3d};
pub use grid::GridSpec;
pub use map::{
    Map, MapConfig, MapError, MapReport, StageTimings, VoxelState, build_map, distance_field,
    length_unit_guess, lipschitz_clearance, scene_bounds,
};
pub use morph::close;
pub use raycast::{BitSet, RayCastConfig, cast_ray, observed_free};
