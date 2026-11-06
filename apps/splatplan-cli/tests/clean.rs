//! Floater removal (`splatplan clean`) judged against synthetic ground truth.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use splat_types::{Gaussian, Vec3, Vec3f};
use synth::{CarConfig, SynthScene, adversarial_scene, car_scene, random_field};

/// The `splatplan clean` defaults (meters).
const RADIUS: f64 = 0.10;
const MIN_NEIGHBORS: usize = 10;
const MAX_OPACITY: f32 = 0.2;

/// `n` faint floaters in free air (at least 0.5 m from every true solid).
fn floaters(scene: &SynthScene, seed: u64, n: usize) -> Vec<Gaussian> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let r = scene.query_region;
    let mut out = Vec::new();
    while out.len() < n {
        let p = Vec3::new(
            rng.gen_range(r.min.x..=r.max.x),
            rng.gen_range(r.min.y..=r.max.y),
            rng.gen_range(r.min.z..=r.max.z),
        );
        if scene.true_clearance(&p) < 0.5 {
            continue;
        }
        let mut g = Gaussian::isotropic(p.cast::<f32>(), rng.gen_range(0.01..0.15), 0.1);
        g.opacity = rng.gen_range(0.05..MAX_OPACITY);
        // Flat discs, like many trained floaters.
        g.scale.z *= 0.05;
        out.push(g);
    }
    out
}

/// Every true-surface Gaussian is kept and every injected floater removed.
fn check(scene: &SynthScene, extra: &[Gaussian]) {
    let n = scene.gaussians.len();
    let mut all = scene.gaussians.clone();
    all.extend_from_slice(extra);
    let keep = gs_io::floater_mask(&all, RADIUS, MIN_NEIGHBORS, MAX_OPACITY);
    let lost = keep[..n].iter().filter(|k| !**k).count();
    assert_eq!(
        lost, 0,
        "{}: {lost} true-surface Gaussians removed",
        scene.name
    );
    let missed = keep[n..].iter().filter(|k| **k).count();
    assert_eq!(missed, 0, "{}: {missed} floaters kept", scene.name);
}

#[test]
fn removes_floaters_and_keeps_every_true_surface() {
    let car = car_scene(&CarConfig::default());
    check(&car, &floaters(&car, 1, 300));
    for seed in [19, 24, 35, 1, 2] {
        let field = random_field(seed, 60);
        check(&field, &floaters(&field, seed, 300));
    }
    check(&adversarial_scene(), &[]);
}

/// A surface that trained faint, like glass: many faint Gaussians close
/// together. Dense, so it is kept even though every Gaussian is faint.
#[test]
fn faint_but_dense_surface_is_kept() {
    let mut car = car_scene(&CarConfig::default());
    let (lo, hi) = (Vec3f::new(-1.35, -0.85, 1.18), Vec3f::new(0.95, 0.85, 1.7));
    let mut faint = 0;
    for g in &mut car.gaussians {
        if (0..3).all(|a| g.mean[a] >= lo[a] && g.mean[a] <= hi[a]) {
            g.opacity = 0.12;
            faint += 1;
        }
    }
    assert!(faint > 500, "{faint}");
    check(&car, &floaters(&car, 7, 100));
}

/// An isolated Gaussian that is opaque is kept: coarse splats cover blank
/// walls with a few large opaque Gaussians whose means are far apart.
#[test]
fn isolated_but_opaque_is_kept() {
    let mut gs: Vec<Gaussian> = (0..20)
        .map(|i| Gaussian::isotropic(Vec3f::new(i as f32 * 0.5, 0.0, 1.0), 0.2, 0.8))
        .collect();
    gs.push(Gaussian::isotropic(Vec3f::new(0.0, 5.0, 1.0), 0.2, 0.15));
    let keep = gs_io::floater_mask(&gs, RADIUS, MIN_NEIGHBORS, MAX_OPACITY);
    assert!(keep[..20].iter().all(|k| *k));
    assert!(!keep[20]);
}
