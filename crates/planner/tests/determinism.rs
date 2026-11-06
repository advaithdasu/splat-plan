//! Determinism: map → plan → certify gives byte-identical outputs at 1, 2
//! and N rayon threads (same machine).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use gs_map::{MapConfig, build_map};
use gs_verify::{Verifier, VerifierConfig};
use planner::{PlannerConfig, plan_and_certify};
use splat_types::Vec3;
use synth::{CarConfig, car_scene};

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ *b as u64).wrapping_mul(0x0100_0000_01b3)
    })
}

fn run_once() -> u64 {
    let s = car_scene(&CarConfig {
        per_ring: 18,
        ..CarConfig::default()
    });
    let scene = s.to_scene();
    let cfg = MapConfig {
        voxel: 0.08,
        ..MapConfig::default()
    };
    let (map, _) = build_map(&scene, &cfg).unwrap();
    let verifier = Verifier::new(
        &scene.gaussians,
        &scene.cameras,
        scene.scale,
        1.0,
        VerifierConfig::default(),
    );
    let mut out = Vec::new();
    for (a, b) in [
        (Vec3::new(-3.0, 0.5, 1.5), Vec3::new(3.0, -0.5, 1.6)),
        (Vec3::new(0.0, -2.6, 1.5), Vec3::new(0.0, 2.6, 1.7)),
    ] {
        let o = plan_and_certify(&map, &verifier, &a, &b, &PlannerConfig::default()).unwrap();
        out.extend(serde_json::to_vec(&o.trajectory).unwrap());
        out.extend(serde_json::to_vec(&o.certificate).unwrap());
    }
    fnv(&out) ^ map.content_hash().rotate_left(1)
}

#[test]
fn plan_and_certify_identical_across_thread_counts() {
    let n = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .max(3);
    let hashes: Vec<u64> = [1, 2, n]
        .iter()
        .map(|&t| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap()
                .install(run_once)
        })
        .collect();
    assert!(hashes.windows(2).all(|w| w[0] == w[1]), "{hashes:x?}");
}
