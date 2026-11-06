//! Standoff orbit: a closed inspection loop at a fixed distance from the main object.

use std::collections::VecDeque;

use gs_map::{Map, VoxelState, edt_3d, lipschitz_clearance};
use gs_verify::{Certificate, Verifier, VerifierConfig};
use serde::{Deserialize, Serialize};
use splat_types::{CameraPose, Gaussian, PolyTrajectory, Scale, Vec3};

use crate::shortcut::segment_clear;
use crate::smooth::smooth;
use crate::workspace::{ExtraField, Workspace};
use crate::{PlanError, PlannerConfig, plan_path_in};

/// Orbit settings (meters, converted with `Map::length_unit`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrbitConfig {
    /// Desired distance from the object's surface.
    pub standoff: f64,
    /// Flight heights above the ground plane (one loop per height).
    pub heights: Vec<f64>,
    /// Waypoints per loop (evenly spaced in azimuth).
    pub waypoints_per_loop: usize,
    /// The certified clearance to the object must be ≥ `standoff − tolerance`.
    pub tolerance: f64,
    /// Occupied voxels below this height count as ground, not object.
    pub ground_band: f64,
}

impl Default for OrbitConfig {
    fn default() -> Self {
        OrbitConfig {
            standoff: 1.5,
            heights: vec![1.2, 2.2],
            waypoints_per_loop: 24,
            tolerance: 0.15,
            ground_band: 0.15,
        }
    }
}

/// A ray that did not produce a waypoint, and why (logged, not fatal).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SkippedRay {
    /// Loop height (m).
    pub height: f64,
    /// Azimuth (degrees).
    pub azimuth_deg: f64,
    /// Reason.
    pub reason: String,
}

/// A certified standoff orbit.
#[derive(Debug, Clone, Serialize)]
pub struct OrbitPlan {
    /// The trajectory.
    pub trajectory: PolyTrajectory,
    /// Certificate against the whole scene.
    pub certificate: Certificate,
    /// Certificate against the object's Gaussians only (its min clearance is
    /// the verified standoff).
    pub object_certificate: Certificate,
    /// Object center (look-at point for the camera).
    pub center: Vec3,
    /// Voxels in the object component.
    pub object_voxels: usize,
    /// Waypoints used (before smoothing).
    pub waypoints: Vec<Vec3>,
    /// Rays that never reached a usable standoff point.
    pub skipped: Vec<SkippedRay>,
}

/// Labels the largest 26-connected component of occupied voxels above the
/// ground band. Returns a mask over all voxels (`true` = object).
pub fn largest_object(map: &Map, ground_band: f64) -> Vec<bool> {
    let spec = &map.spec;
    let n = spec.len();
    let above = |i: usize| spec.center(spec.coords(i)).z > ground_band;
    let mut label = vec![u32::MAX; n];
    let mut best: (usize, u32) = (0, u32::MAX);
    let mut next = 0u32;
    let mut queue = VecDeque::new();
    let mut nbrs = Vec::with_capacity(26);
    // Scan in index order: deterministic labels and tie-breaking.
    for seed in 0..n {
        if label[seed] != u32::MAX || map.state[seed] != VoxelState::Occupied || !above(seed) {
            continue;
        }
        let id = next;
        next += 1;
        let mut size = 0usize;
        label[seed] = id;
        queue.push_back(seed);
        while let Some(i) = queue.pop_front() {
            size += 1;
            spec.neighbors26(spec.coords(i), &mut nbrs);
            for (c, _) in &nbrs {
                let j = spec.index(*c);
                if label[j] == u32::MAX && map.state[j] == VoxelState::Occupied && above(j) {
                    label[j] = id;
                    queue.push_back(j);
                }
            }
        }
        if size > best.0 {
            best = (size, id);
        }
    }
    label.iter().map(|l| *l == best.1 && best.0 > 0).collect()
}

/// Plans and certifies a standoff orbit around the largest object.
#[allow(clippy::too_many_arguments)]
pub fn standoff_orbit(
    map: &Map,
    gaussians: &[Gaussian],
    cameras: &[CameraPose],
    scale: Scale,
    verifier_cfg: &VerifierConfig,
    cfg: &PlannerConfig,
    ocfg: &OrbitConfig,
) -> Result<OrbitPlan, PlanError> {
    let u = map.length_unit;
    let spec = &map.spec;
    let mask = largest_object(map, ocfg.ground_band * u);
    let object_voxels = mask.iter().filter(|m| **m).count();
    if object_voxels == 0 {
        return Err(PlanError::NoPath {
            reason: "no occupied object above the ground".into(),
            expanded: 0,
        });
    }
    // Object center: mean of its voxel centers (horizontal), mid-height.
    let mut center = Vec3::zeros();
    let (mut zmin, mut zmax) = (f64::INFINITY, f64::NEG_INFINITY);
    for (i, _) in mask.iter().enumerate().filter(|(_, m)| **m) {
        let c = spec.center(spec.coords(i));
        center += c;
        zmin = zmin.min(c.z);
        zmax = zmax.max(c.z);
    }
    center /= object_voxels as f64;
    center.z = 0.5 * (zmin + zmax);

    // Exact distance to the object only.
    let mut d2: Vec<f32> = mask
        .iter()
        .map(|m| if *m { 0.0 } else { f32::INFINITY })
        .collect();
    edt_3d(&mut d2, spec.dims);
    let obj_dist = |p: &Vec3| -> Option<f64> {
        spec.cell_of(p)
            .map(|c| (d2[spec.index(c)] as f64).sqrt() * spec.res)
    };

    let need = cfg.search_clearance(map);
    // Distances in scene units for the workspace's extra field: everything,
    // including detours, must stay `standoff` from the object.
    let obj_field: Vec<f32> = d2.iter().map(|v| v.sqrt() * spec.res as f32).collect();
    let mut ws = Workspace::new(map);
    // Conservative distance to the object (same bound the workspace uses).
    let obj_clear = |p: &Vec3| lipschitz_clearance(spec, &obj_field, p);
    // One voxel beyond the standoff: the endpoint slack the detour planner requires.
    let target = ocfg.standoff * u + spec.res;
    let mut skipped = Vec::new();
    let mut waypoints: Vec<Vec3> = Vec::new();
    for &h in &ocfg.heights {
        let mut ring = Vec::new();
        for k in 0..ocfg.waypoints_per_loop {
            let az = std::f64::consts::TAU * k as f64 / ocfg.waypoints_per_loop as f64;
            let dir = Vec3::new(az.cos(), az.sin(), 0.0);
            let mut r = 0.0;
            let found = loop {
                let p = Vec3::new(center.x, center.y, h * u) + dir * r;
                if spec.cell_of(&p).is_none() {
                    break Err("ray left the map before reaching the standoff");
                }
                match obj_clear(&p) {
                    d if d >= target => {
                        break if ws.state_at(&p) != VoxelState::FreeObserved {
                            Err("standoff point is not observed-free")
                        } else if ws.clearance(&p) < need + spec.res {
                            Err("standoff point too close to other obstacles")
                        } else {
                            Ok(p)
                        };
                    }
                    _ => r += spec.res * 0.5,
                }
            };
            match found {
                Ok(p) => ring.push(p),
                Err(reason) => {
                    tracing::info!(h, az = az.to_degrees(), reason, "orbit ray skipped");
                    skipped.push(SkippedRay {
                        height: h,
                        azimuth_deg: az.to_degrees(),
                        reason: reason.into(),
                    });
                }
            }
        }
        if ring.len() < 3 {
            continue;
        }
        let first = ring[0];
        waypoints.extend(ring);
        waypoints.push(first); // close the loop
    }
    if waypoints.len() < 4 {
        return Err(PlanError::NoPath {
            reason: format!(
                "orbit has too few usable waypoints ({} rays skipped)",
                skipped.len()
            ),
            expanded: 0,
        });
    }
    waypoints.dedup_by(|a, b| (*a - *b).norm() < 1e-9);
    // Where rays were skipped, neighboring waypoints can be far apart and the
    // straight chord may cut through an obstacle or come too close to the
    // object: route around with A*, holding the standoff as an extra field.
    ws.extra = Some(ExtraField {
        dist: &obj_field,
        offset: ocfg.standoff * u - need,
    });
    let mut routed = vec![waypoints[0]];
    for w in waypoints.windows(2) {
        if !segment_clear(&ws, &w[0], &w[1], need) {
            let detour = plan_path_in(&ws, &w[0], &w[1], cfg)?;
            routed.extend_from_slice(&detour[1..detour.len() - 1]);
        }
        routed.push(w[1]);
    }
    let waypoints = routed;
    let trajectory = smooth(&ws, &waypoints, cfg, 1.0)?;

    let verifier = Verifier::new(gaussians, cameras, scale, u, verifier_cfg.clone());
    let certificate = verifier.certify(&trajectory);
    if certificate.violation_count > 0 {
        return Err(PlanError::VerifierRejected {
            attempts: 1,
            violations: certificate.violation_count,
            first_reason: certificate
                .violations
                .first()
                .map(|v| format!("{:?}", v.reason).to_lowercase())
                .unwrap_or_default(),
        });
    }
    // Standoff check with the same independent verifier code, fed only the
    // Gaussians whose centers lie in (or next to) the object's voxels.
    let near_object = |g: &Gaussian| {
        let p = g.mean_f64();
        obj_dist(&p).is_some_and(|d| d <= spec.res * 1.5)
    };
    let object_gaussians: Vec<Gaussian> = gaussians
        .iter()
        .filter(|g| near_object(g))
        .copied()
        .collect();
    let object_verifier = Verifier::new(&object_gaussians, cameras, scale, u, verifier_cfg.clone());
    let object_certificate = object_verifier.certify(&trajectory);
    let standoff_ok = object_certificate.min_clearance_m >= ocfg.standoff - ocfg.tolerance;
    if !standoff_ok {
        return Err(PlanError::Trajectory(format!(
            "verified standoff {:.3} m < required {:.3} m",
            object_certificate.min_clearance_m,
            ocfg.standoff - ocfg.tolerance
        )));
    }
    Ok(OrbitPlan {
        trajectory,
        certificate,
        object_certificate,
        center,
        object_voxels,
        waypoints,
        skipped,
    })
}
