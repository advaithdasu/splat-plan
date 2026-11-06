//! The planner's view of free space: the map plus temporary keep-out spheres.

use gs_map::{Map, VoxelState, lipschitz_clearance};
use serde::{Deserialize, Serialize};
use splat_types::Vec3;

/// A sphere the planner must stay away from (fed back from the verifier).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct KeepOut {
    /// Center (scene units).
    pub center: Vec3,
    /// Radius (scene units).
    pub radius: f64,
}

/// An extra distance field on the map's grid: points must keep
/// `field(p) − offset` of clearance (used to hold a standoff from an object).
#[derive(Debug, Clone)]
pub struct ExtraField<'a> {
    /// Distance from each voxel center to the field's obstacle (scene units).
    pub dist: &'a [f32],
    /// Subtracted from the field's distance.
    pub offset: f64,
}

/// Map plus keep-out spheres. All planner geometry queries go through here.
#[derive(Debug, Clone)]
pub struct Workspace<'a> {
    /// The collision map.
    pub map: &'a Map,
    /// Extra obstacles.
    pub keepouts: Vec<KeepOut>,
    /// Optional extra distance constraint.
    pub extra: Option<ExtraField<'a>>,
}

impl<'a> Workspace<'a> {
    /// Workspace with no keep-outs.
    pub fn new(map: &'a Map) -> Self {
        Workspace {
            map,
            keepouts: Vec::new(),
            extra: None,
        }
    }

    /// Distance from `p` to the nearest keep-out sphere surface, or to the
    /// extra field's obstacle minus its offset (∞ if neither exists).
    pub fn keepout_distance(&self, p: &Vec3) -> f64 {
        let spheres = self
            .keepouts
            .iter()
            .map(|k| (p - k.center).norm() - k.radius)
            .fold(f64::INFINITY, f64::min);
        match &self.extra {
            Some(f) => spheres.min(lipschitz_clearance(&self.map.spec, f.dist, p) - f.offset),
            None => spheres,
        }
    }

    /// Conservative clearance: the smaller of the map's bound and the exact
    /// keep-out distance. Both are lower bounds of 1-Lipschitz distances.
    pub fn clearance(&self, p: &Vec3) -> f64 {
        self.map.clearance(p).min(self.keepout_distance(p))
    }

    /// State of the voxel containing `p`.
    pub fn state_at(&self, p: &Vec3) -> VoxelState {
        self.map.state_at(p)
    }
}
