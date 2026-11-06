//! Ready-made synthetic scenes with exact oracles.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use splat_types::nalgebra::{Unit, UnitQuaternion};
use splat_types::{Aabb, CameraPose, Gaussian, Scale, Scene, Vec3};

use crate::primitives::Primitive;
use crate::rig::{default_intrinsics, orbit_rig};

/// A synthetic scene: analytic geometry (the truth), its Gaussian splat, and
/// the training cameras.
#[derive(Debug, Clone)]
pub struct SynthScene {
    /// Short name used in reports.
    pub name: String,
    /// The true solids. [`SynthScene::true_clearance`] measures to these.
    pub primitives: Vec<Primitive>,
    /// The splat representing the solids.
    pub gaussians: Vec<Gaussian>,
    /// Training cameras.
    pub cameras: Vec<CameraPose>,
    /// Region in which random start/goal queries are drawn.
    pub query_region: Aabb,
    /// Point the cameras orbit (center of the main object).
    pub target: Vec3,
}

impl SynthScene {
    /// **Exact oracle**: distance from `p` to the nearest true solid (0 inside).
    pub fn true_clearance(&self, p: &Vec3) -> f64 {
        self.primitives
            .iter()
            .map(|s| s.distance(p))
            .fold(f64::INFINITY, f64::min)
    }

    /// The pipeline-facing view: Gaussians + cameras, metric (scale 1).
    pub fn to_scene(&self) -> Scene {
        Scene {
            gaussians: self.gaussians.clone(),
            cameras: self.cameras.clone(),
            scale: Scale::Metric(1.0),
        }
    }

    /// `n` deterministic random `(start, goal)` pairs inside the query region.
    /// Some will fall inside obstacles or unknown space on purpose: the
    /// planner must refuse those.
    pub fn random_queries(&self, seed: u64, n: usize) -> Vec<(Vec3, Vec3)> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let r = &self.query_region;
        let mut pt = || {
            Vec3::new(
                rng.gen_range(r.min.x..=r.max.x),
                rng.gen_range(r.min.y..=r.max.y),
                rng.gen_range(r.min.z..=r.max.z),
            )
        };
        (0..n).map(|_| (pt(), pt())).collect()
    }
}

/// Parameters of the synthetic car scene (defaults model a ~4.5 m sedan).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CarConfig {
    /// Car heading about +Z (radians).
    pub yaw: f64,
    /// Lattice spacing for car surfaces (m).
    pub spacing: f64,
    /// Lattice spacing for the ground (m).
    pub ground_spacing: f64,
    /// Half-size of the square ground patch (m).
    pub ground_half: f64,
    /// Orbit rings as `(radius, height)` in meters.
    pub rings: Vec<(f64, f64)>,
    /// Cameras per ring.
    pub per_ring: usize,
    /// Add a sealed hollow box (a cavity no camera can see into).
    pub sealed_pocket: bool,
    /// Per-Gaussian color noise amplitude (0 = flat colors). Real surfaces
    /// have texture; structure-from-motion needs it to find features.
    pub color_noise: f32,
    /// Seed for the color noise.
    pub texture_seed: u64,
}

impl Default for CarConfig {
    fn default() -> Self {
        CarConfig {
            yaw: 0.0,
            spacing: 0.04,
            ground_spacing: 0.08,
            ground_half: 7.0,
            // Two low orbits, one mid, one high pass.
            rings: vec![(5.5, 1.2), (5.5, 2.4), (5.0, 3.6), (3.5, 5.5)],
            per_ring: 36,
            sealed_pocket: false,
            color_noise: 0.0,
            texture_seed: 0,
        }
    }
}

/// Where the sealed pocket box sits when enabled (beside the car).
pub const POCKET_CENTER: [f64; 3] = [0.0, 3.2, 0.75];
/// Outer half-size of the sealed pocket box.
pub const POCKET_HALF: f64 = 0.75;

/// The synthetic car: box body + cabin, four cylindrical wheels, ground plane,
/// all represented by Gaussians, plus orbit cameras.
pub fn car_scene(cfg: &CarConfig) -> SynthScene {
    let r = UnitQuaternion::from_axis_angle(&Vec3::z_axis(), cfg.yaw);
    let body = Primitive::Box {
        center: Vec3::new(0.0, 0.0, 0.75),
        half: Vec3::new(2.25, 0.9, 0.4),
        yaw: cfg.yaw,
    };
    let cabin = Primitive::Box {
        center: r * Vec3::new(-0.2, 0.0, 1.4),
        half: Vec3::new(1.1, 0.8, 0.25),
        yaw: cfg.yaw,
    };
    let mut colored: Vec<(Primitive, [f32; 3], f64)> = vec![
        (body, [0.75, 0.1, 0.1], cfg.spacing),
        (cabin, [0.25, 0.35, 0.5], cfg.spacing),
    ];
    for (x, y) in [(1.4, 0.85), (1.4, -0.85), (-1.4, 0.85), (-1.4, -0.85)] {
        colored.push((
            Primitive::Cylinder {
                center: r * Vec3::new(x, y, 0.33),
                axis: r * Vec3::y(),
                radius: 0.33,
                half_len: 0.11,
            },
            [0.08, 0.08, 0.08],
            cfg.spacing,
        ));
    }
    if cfg.sealed_pocket {
        colored.push((
            Primitive::HollowBox {
                center: Vec3::from(POCKET_CENTER),
                outer_half: Vec3::repeat(POCKET_HALF),
                wall: 0.1,
            },
            [0.8, 0.6, 0.2],
            cfg.spacing,
        ));
    }
    let mut gaussians = Vec::new();
    let mut primitives = Vec::new();
    for (p, color, spacing) in colored {
        gaussians.extend(p.splat(spacing, cfg.ground_half, color, 0.9));
        primitives.push(p);
    }
    // Checkerboard ground (texture helps the renderer demo; geometry is flat).
    let ground = Primitive::Ground { height: 0.0 };
    for g in ground.splat(cfg.ground_spacing, cfg.ground_half, [0.5, 0.5, 0.5], 0.9) {
        let cell = ((g.mean.x.floor() as i64) + (g.mean.y.floor() as i64)).rem_euclid(2);
        let c = if cell == 0 { 0.55 } else { 0.4 };
        gaussians.push(g.with_color([c, c, c * 0.9]));
    }
    primitives.push(ground);
    if cfg.color_noise > 0.0 {
        let mut rng = ChaCha8Rng::seed_from_u64(cfg.texture_seed);
        for g in &mut gaussians {
            let c = g.color_rgb();
            let n = cfg.color_noise;
            *g = g.with_color(c.map(|v| (v + rng.gen_range(-n..=n)).clamp(0.0, 1.0)));
        }
    }

    let target = Vec3::new(0.0, 0.0, 0.8);
    let cameras = orbit_rig(target, &cfg.rings, cfg.per_ring, default_intrinsics());
    SynthScene {
        name: if cfg.sealed_pocket {
            "synth-car-pocket".into()
        } else {
            "synth-car".into()
        },
        primitives,
        gaussians,
        cameras,
        // Inside the camera orbit: space at or beyond the ring (r = 5.5 m) is
        // never looked *through*, so it is correctly Unknown to the map.
        query_region: Aabb::new(Vec3::new(-3.2, -3.2, 0.4), Vec3::new(3.2, 3.2, 2.8)),
        target,
    }
}

/// A random field of ellipsoidal Gaussians above a ground plane. The oracle is
/// the exact distance to each Gaussian's 3σ ellipsoid (and the ground).
pub fn random_field(seed: u64, n: usize) -> SynthScene {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut primitives = Vec::with_capacity(n + 1);
    let mut gaussians = Vec::new();
    for _ in 0..n {
        let center = Vec3::new(
            rng.gen_range(-3.0..3.0),
            rng.gen_range(-3.0..3.0),
            rng.gen_range(0.3..2.5),
        );
        let semi_axes = Vec3::new(
            rng.gen_range(0.05..0.6),
            rng.gen_range(0.05..0.6),
            rng.gen_range(0.05..0.6),
        );
        let axis = Unit::new_normalize(Vec3::new(
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(0.01..1.0),
        ));
        let rot = UnitQuaternion::from_axis_angle(&axis, rng.gen_range(0.0..std::f64::consts::TAU));
        let p = Primitive::Ellipsoid {
            center,
            semi_axes,
            rot,
        };
        gaussians.extend(p.splat(0.0, 0.0, [0.2, 0.6, 0.3], 0.95));
        primitives.push(p);
    }
    let ground = Primitive::Ground { height: 0.0 };
    gaussians.extend(ground.splat(0.08, 6.0, [0.5, 0.5, 0.5], 0.9));
    primitives.push(ground);
    let target = Vec3::new(0.0, 0.0, 1.2);
    let cameras = orbit_rig(
        target,
        &[(5.0, 1.0), (5.0, 2.5), (4.0, 4.5)],
        36,
        default_intrinsics(),
    );
    SynthScene {
        name: format!("synth-field-{seed}"),
        primitives,
        gaussians,
        cameras,
        query_region: Aabb::new(Vec3::new(-3.5, -3.5, 0.3), Vec3::new(3.5, 3.5, 3.0)),
        target,
    }
}

/// Adversarial scene: things that are easy to miss or to squeeze through.
///
/// - a thin vertical pole (radius 2 cm) and a thin horizontal wire (radius 1 cm);
/// - two walls with a 0.5 m gap (narrower than 2 × (radius + margin + buffer)
///   with the default 0.15 m robot, 0.10 m margin and 0.05 m buffer);
/// - every Gaussian duplicated with a sub-millimetre offset (near-duplicates,
///   common after densification in trained splats).
pub fn adversarial_scene() -> SynthScene {
    let mut colored: Vec<(Primitive, [f32; 3])> = vec![
        (
            Primitive::Cylinder {
                center: Vec3::new(1.5, 0.5, 1.5),
                axis: Vec3::z(),
                radius: 0.02,
                half_len: 1.5,
            },
            [0.9, 0.9, 0.1],
        ),
        (
            Primitive::Cylinder {
                center: Vec3::new(0.0, 0.0, 1.6),
                axis: Vec3::y(),
                radius: 0.01,
                half_len: 2.5,
            },
            [0.1, 0.9, 0.9],
        ),
    ];
    for y in [-1.25, 1.25] {
        colored.push((
            Primitive::Box {
                center: Vec3::new(-1.8, y, 1.25),
                half: Vec3::new(0.15, 1.0, 1.25),
                yaw: 0.0,
            },
            [0.6, 0.3, 0.8],
        ));
    }
    let mut gaussians = Vec::new();
    let mut primitives = Vec::new();
    for (p, c) in colored {
        gaussians.extend(p.splat(0.04, 0.0, c, 0.9));
        primitives.push(p);
    }
    let ground = Primitive::Ground { height: 0.0 };
    gaussians.extend(ground.splat(0.08, 6.0, [0.5, 0.5, 0.5], 0.9));
    primitives.push(ground);
    let dups: Vec<Gaussian> = gaussians
        .iter()
        .map(|g| {
            let mut d = *g;
            d.mean += splat_types::Vec3f::new(3e-4, -2e-4, 1e-4);
            d
        })
        .collect();
    gaussians.extend(dups);
    let target = Vec3::new(0.0, 0.0, 1.2);
    let cameras = orbit_rig(
        target,
        &[(5.0, 1.0), (5.0, 2.2), (4.0, 4.0)],
        36,
        default_intrinsics(),
    );
    SynthScene {
        name: "synth-adversarial".into(),
        primitives,
        gaussians,
        cameras,
        query_region: Aabb::new(Vec3::new(-3.2, -3.2, 0.5), Vec3::new(3.2, 3.2, 2.6)),
        target,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn car_oracle_and_size() {
        let s = car_scene(&CarConfig::default());
        // Above the roof: cabin top at z = 1.65.
        assert!((s.true_clearance(&Vec3::new(-0.2, 0.0, 2.65)) - 1.0).abs() < 1e-9);
        // Inside the body.
        assert_eq!(s.true_clearance(&Vec3::new(0.0, 0.0, 0.8)), 0.0);
        // Beside the body: side at y = 0.9 is nearer than the ground (z = 1.0)
        // and the cabin edge (y = 0.8, z = 1.15).
        assert!((s.true_clearance(&Vec3::new(0.0, 1.5, 1.0)) - 0.6).abs() < 1e-9);
        assert!(s.gaussians.len() > 20_000 && s.gaussians.len() < 200_000);
        assert_eq!(s.cameras.len(), 4 * 36);
        assert!(s.gaussians.iter().all(Gaussian::is_valid));
    }

    #[test]
    fn deterministic_generation() {
        let a = random_field(5, 50);
        let b = random_field(5, 50);
        assert_eq!(a.gaussians, b.gaussians);
        assert_eq!(a.random_queries(1, 10), b.random_queries(1, 10));
        assert_ne!(random_field(6, 50).gaussians, a.gaussians);
    }

    #[test]
    fn pocket_cavity_has_positive_clearance() {
        let s = car_scene(&CarConfig {
            sealed_pocket: true,
            ..CarConfig::default()
        });
        let c = Vec3::from(POCKET_CENTER);
        assert!((s.true_clearance(&c) - (POCKET_HALF - 0.1)).abs() < 1e-9);
    }
}
