//! Synthetic scenes with exact distance oracles, camera rigs.
//!
//! Real captures are noisy and have no ground truth, so every claim this
//! project makes is first proven on synthetic scenes where the true geometry
//! is known analytically:
//!
//! - [`Primitive`]: boxes, cylinders, ground, sealed hollow boxes and
//!   ellipsoids, each with an exact distance function.
//! - [`SynthScene::true_clearance`]: the **oracle**, distance to the true solids.
//! - [`car_scene`]: the synthetic car (body, cabin, four wheels, ground).
//! - [`random_field`]: random ellipsoids over a ground plane.
//! - [`orbit_rig`]: drone-like orbit cameras.
//! - [`export_capture`]: writes the same files a real capture would have.
//!
//! Everything is seeded (`rand_chacha`) and therefore reproducible.
#![forbid(unsafe_code)]

mod ellipsoid;
mod export;
mod primitives;
mod rig;
mod scenes;

pub use ellipsoid::distance_to_ellipsoid_local;
pub use export::{ExportError, GroundTruth, export_capture, random_similarity};
pub use primitives::Primitive;
pub use rig::{azimuth, default_intrinsics, orbit_rig};
pub use scenes::{
    CarConfig, POCKET_CENTER, POCKET_HALF, SynthScene, adversarial_scene, car_scene, random_field,
};
