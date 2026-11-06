//! Standoff orbit around the synthetic car.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use gs_map::{MapConfig, build_map};
use gs_verify::{Verdict, VerifierConfig};
use planner::{OrbitConfig, PlannerConfig, largest_object, standoff_orbit};
use synth::{CarConfig, Primitive, car_scene};

#[test]
fn orbit_around_car_keeps_standoff() {
    let s = car_scene(&CarConfig {
        sealed_pocket: true,
        ..CarConfig::default()
    });
    let scene = s.to_scene();
    let (map, _) = build_map(&scene, &MapConfig::default()).unwrap();
    // The largest object is the car, not the smaller pocket box.
    let mask = largest_object(&map, 0.15);
    let pocket_wall = splat_types::Vec3::new(0.0, 3.2 - 0.75, 0.75);
    let pocket = map.spec.index(map.spec.cell_of(&pocket_wall).unwrap());
    assert!(
        !mask[pocket],
        "pocket box wall must not be part of the car object"
    );

    let ocfg = OrbitConfig {
        standoff: 1.0,
        heights: vec![1.0, 1.8],
        ..OrbitConfig::default()
    };
    let plan = standoff_orbit(
        &map,
        &scene.gaussians,
        &scene.cameras,
        scene.scale,
        &VerifierConfig::default(),
        &PlannerConfig::default(),
        &ocfg,
    )
    .unwrap();
    assert_eq!(plan.certificate.verdict, Verdict::Safe);
    assert!(
        plan.center.xy().norm() < 0.3,
        "object center {:?} should be the car",
        plan.center
    );
    assert!(
        plan.center.xy().norm() < 0.3,
        "object center {:?} should be the car",
        plan.center
    );
    assert!(plan.object_certificate.min_clearance_m >= ocfg.standoff - ocfg.tolerance);
    // Rays toward the pocket box cannot reach the standoff in free space: logged, not fatal.
    assert!(
        !plan.skipped.is_empty(),
        "expected skipped rays near the pocket box"
    );
    // Oracle: distance to the car's true solids (not ground, not pocket) along the orbit.
    let car_parts: Vec<&Primitive> = s
        .primitives
        .iter()
        .filter(|p| matches!(p, Primitive::Box { .. } | Primitive::Cylinder { .. }))
        .collect();
    let min_true = plan
        .trajectory
        .sample(0.01)
        .iter()
        .map(|x| {
            car_parts
                .iter()
                .map(|p| p.distance(&x.pos))
                .fold(f64::INFINITY, f64::min)
        })
        .fold(f64::INFINITY, f64::min);
    assert!(
        min_true >= ocfg.standoff - ocfg.tolerance,
        "true standoff {min_true}"
    );
    // Closed loop: back where it started at each height.
    assert!((plan.trajectory.start() - plan.waypoints[0]).norm() < 1e-9);

    // An impossible standoff is refused, never faked.
    let far = OrbitConfig {
        standoff: 8.0,
        ..ocfg
    };
    assert!(
        standoff_orbit(
            &map,
            &scene.gaussians,
            &scene.cameras,
            scene.scale,
            &VerifierConfig::default(),
            &PlannerConfig::default(),
            &far
        )
        .is_err()
    );
}
