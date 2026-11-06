//! Adversarial scene: thin pole, thin wire, too-narrow gap, near-duplicate Gaussians.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use gs_map::{MapConfig, build_map};
use gs_verify::{Verdict, Verifier, VerifierConfig};
use planner::{PlannerConfig, plan_and_certify};
use rayon::prelude::*;
use splat_types::Vec3;
use synth::adversarial_scene;

#[test]
fn thin_and_narrow_obstacles_never_produce_oracle_unsafe_plans() {
    let s = adversarial_scene();
    let scene = s.to_scene();
    let (map, _) = build_map(&scene, &MapConfig::default()).unwrap();
    let verifier = Verifier::new(
        &scene.gaussians,
        &scene.cameras,
        scene.scale,
        1.0,
        VerifierConfig::default(),
    );
    let cfg = PlannerConfig::default();
    let need = cfg.robot_radius + cfg.margin;

    // The 2 cm pole and 1 cm wire must be in the map.
    assert_eq!(
        map.state_at(&Vec3::new(1.5, 0.5, 1.5)),
        gs_map::VoxelState::Occupied
    );
    assert_eq!(
        map.state_at(&Vec3::new(0.0, 1.0, 1.6)),
        gs_map::VoxelState::Occupied
    );

    // Straight through the 0.5 m gap is impossible: the plan must go around
    // or over the walls.
    let a = Vec3::new(-3.0, 0.0, 1.0);
    let b = Vec3::new(-0.6, 0.0, 1.0);
    if let Ok(o) = plan_and_certify(&map, &verifier, &a, &b, &cfg) {
        // Over the 2.5 m walls is fine; between them below the top is not.
        let samples = o.trajectory.sample(0.01);
        let through_gap = samples
            .iter()
            .any(|x| (x.pos.x + 1.8).abs() < 0.15 && x.pos.y.abs() < 0.25 && x.pos.z < 2.5);
        assert!(
            !through_gap,
            "squeezed through a gap narrower than the robot needs"
        );
        let c = samples
            .iter()
            .map(|x| s.true_clearance(&x.pos))
            .fold(f64::INFINITY, f64::min);
        assert!(c >= need, "gap query: oracle clearance {c}");
    }

    let queries = s.random_queries(11, 120);
    let results: Vec<_> = queries
        .par_iter()
        .map(|(a, b)| plan_and_certify(&map, &verifier, a, b, &cfg))
        .collect();
    let mut ok = 0;
    for (i, r) in results.iter().enumerate() {
        if let Ok(o) = r {
            assert_eq!(o.certificate.verdict, Verdict::Safe);
            let c = o
                .trajectory
                .sample(0.005)
                .iter()
                .map(|x| s.true_clearance(&x.pos))
                .fold(f64::INFINITY, f64::min);
            assert!(c >= need, "query {i}: oracle clearance {c:.4}");
            ok += 1;
        }
    }
    println!("adversarial: {ok}/120 certified, all oracle-safe");
    assert!(ok >= 10);
}
