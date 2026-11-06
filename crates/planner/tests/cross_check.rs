//! The verifier's line-of-sight check catches a map that claims to have seen
//! space the training cameras never saw (a ray-casting bug inside `gs-map`).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use gs_map::{MapConfig, VoxelState, build_map};
use gs_verify::{Verdict, Verifier, VerifierConfig};
use planner::{PlanError, PlannerConfig, plan_and_certify};
use splat_types::{CameraPose, Scale, Scene, Vec3};
use synth::{Primitive, default_intrinsics};

/// Seven cameras at `x`, looking at the wall center.
fn cameras(x: f64) -> Vec<CameraPose> {
    (0..7)
        .map(|i| {
            let y = -1.5 + 0.5 * f64::from(i);
            CameraPose::look_at(
                format!("cam{x}_{i}"),
                Vec3::new(x, y, 1.5),
                Vec3::new(0.0, 0.0, 1.5),
                Vec3::z(),
                default_intrinsics(),
            )
            .unwrap()
        })
        .collect()
}

#[test]
fn map_seeing_more_than_the_verifier_is_refused() {
    let wall = Primitive::Box {
        center: Vec3::new(0.0, 0.0, 1.5),
        half: Vec3::new(0.05, 3.0, 1.5),
        yaw: 0.0,
    };
    let mut gaussians = wall.splat(0.04, 0.0, [0.5, 0.5, 0.5], 0.9);
    // Ground, so the map's bounds cover both sides of the wall.
    gaussians.extend(Primitive::Ground { height: 0.0 }.splat(0.08, 4.0, [0.4, 0.4, 0.4], 0.9));
    let front = cameras(-4.0);
    let all: Vec<CameraPose> = front.iter().cloned().chain(cameras(4.0)).collect();

    // The map is built from cameras on both sides; the verifier only gets
    // the front ones, so behind the wall is "observed" for the map only.
    let scene = Scene {
        gaussians: gaussians.clone(),
        cameras: all,
        scale: Scale::Metric(1.0),
    };
    let (map, _) = build_map(&scene, &MapConfig::default()).unwrap();
    let behind_a = Vec3::new(1.5, -1.0, 1.5);
    let behind_b = Vec3::new(1.5, 1.0, 1.5);
    assert_eq!(map.state_at(&behind_a), VoxelState::FreeObserved);
    let verifier = Verifier::new(
        &gaussians,
        &front,
        Scale::Metric(1.0),
        1.0,
        VerifierConfig::default(),
    );
    let cfg = PlannerConfig::default();

    let front_plan = plan_and_certify(
        &map,
        &verifier,
        &Vec3::new(-1.5, -1.0, 1.5),
        &Vec3::new(-1.5, 1.0, 1.5),
        &cfg,
    )
    .unwrap();
    assert_eq!(front_plan.certificate.verdict, Verdict::Safe);
    assert!(front_plan.certificate.observed_ok);

    match plan_and_certify(&map, &verifier, &behind_a, &behind_b, &cfg) {
        // Reported as the verifier's refusal, not as the start point
        // being blocked by the keep-outs the repair loop added.
        Err(PlanError::VerifierRejected {
            first_reason,
            attempts,
            ..
        }) => {
            assert_eq!(first_reason, "unobserved");
            assert!(attempts >= 1);
        }
        other => panic!("expected a line-of-sight refusal, got {other:?}"),
    }
}
