//! `calm` only ever returns a path the verifier certifies, and never a rougher one.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use gs_map::{MapConfig, build_map};
use gs_verify::{Verdict, Verifier, VerifierConfig};
use planner::{PlannerConfig, calm, dimensionless_jerk, plan_and_certify};
use synth::adversarial_scene;

#[test]
fn calmer_paths_are_certified_smoother_and_within_limits() {
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

    let mut calmed = 0;
    for (a, b) in s.random_queries(5, 12) {
        let Ok(before) = plan_and_certify(&map, &verifier, &a, &b, &cfg) else {
            continue;
        };
        let (start, end) = (before.trajectory.start(), before.trajectory.end());
        let jerk_before = dimensionless_jerk(&before.trajectory);
        let after = calm(&map, &verifier, before, &cfg);
        // Same ends, certified, inside the limits, never rougher.
        assert!((after.trajectory.start() - start).norm() < 1e-6);
        assert!((after.trajectory.end() - end).norm() < 1e-6);
        assert_eq!(after.certificate.verdict, Verdict::Safe);
        assert_eq!(after.certificate.violation_count, 0);
        let (v, acc) = after.trajectory.max_speed_accel(0.01);
        assert!(v <= cfg.max_speed + 1e-6 && acc <= cfg.max_accel + 1e-6);
        assert!(dimensionless_jerk(&after.trajectory) <= jerk_before + 1e-9);
        let c = after
            .trajectory
            .sample(0.005)
            .iter()
            .map(|x| s.true_clearance(&x.pos))
            .fold(f64::INFINITY, f64::min);
        assert!(c >= need, "oracle clearance {c}");
        calmed += 1;
    }
    assert!(calmed > 0, "no query planned");
}
