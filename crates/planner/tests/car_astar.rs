//! A* + shortcut on the synthetic car.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use gs_map::{Map, MapConfig, VoxelState, build_map};
use planner::{PlanError, PlannerConfig, plan_path};
use splat_types::Vec3;
use synth::{CarConfig, POCKET_CENTER, SynthScene, azimuth, car_scene};

fn build(cfg: &CarConfig) -> (SynthScene, Map) {
    let s = car_scene(cfg);
    let (map, _) = build_map(&s.to_scene(), &MapConfig::default()).unwrap();
    (s, map)
}

/// Dense oracle check of a polyline.
fn min_oracle_clearance(s: &SynthScene, path: &[Vec3]) -> f64 {
    let mut m = f64::INFINITY;
    for w in path.windows(2) {
        let n = ((w[1] - w[0]).norm() / 0.005).ceil() as usize + 1;
        for k in 0..=n {
            let p = w[0] + (w[1] - w[0]) * (k as f64 / n as f64);
            m = m.min(s.true_clearance(&p));
        }
    }
    m
}

#[test]
fn plans_around_car_and_respects_oracle() {
    let (s, map) = build(&CarConfig {
        sealed_pocket: true,
        ..CarConfig::default()
    });
    let cfg = PlannerConfig::default();
    let start = Vec3::new(-4.0, 0.3, 1.0);
    let goal = Vec3::new(4.0, -0.2, 1.0);
    let path = plan_path(&map, &start, &goal, &cfg).unwrap();
    assert!((path[0] - start).norm() < 1e-12 && (path.last().unwrap() - goal).norm() < 1e-12);
    assert!(path.len() >= 3, "must go around the car: {path:?}");
    let c = min_oracle_clearance(&s, &path);
    assert!(c >= cfg.robot_radius + cfg.margin, "oracle clearance {c}");

    // Determinism: same inputs, same path.
    assert_eq!(path, plan_path(&map, &start, &goal, &cfg).unwrap());

    // Start inside the sealed pocket: refused, and the reason is unknown space.
    match plan_path(&map, &Vec3::from(POCKET_CENTER), &goal, &cfg) {
        Err(PlanError::NotTraversable {
            which: "start",
            state: VoxelState::Unknown,
            ..
        }) => {}
        other => panic!("expected unknown-start refusal, got {other:?}"),
    }
    // Goal inside the car body: refused.
    assert!(matches!(
        plan_path(&map, &start, &Vec3::new(0.5, 0.0, 0.8), &cfg),
        Err(PlanError::NotTraversable { which: "goal", .. })
    ));
    // Goal outside the map.
    assert!(matches!(
        plan_path(&map, &start, &Vec3::new(50.0, 0.0, 1.0), &cfg),
        Err(PlanError::OutsideMap { which: "goal" })
    ));
    // Goal too close to the ground.
    assert!(matches!(
        plan_path(&map, &start, &Vec3::new(3.0, 3.0, 0.2), &cfg),
        Err(PlanError::NotTraversable { .. })
    ));
}

#[test]
fn unobserved_region_is_refused() {
    // Only cameras facing the -y side (azimuth -150°..-30°): part of the far
    // side of the car is never seen through, although it is open air.
    let mut s = car_scene(&CarConfig::default());
    let full = build_map(&s.to_scene(), &MapConfig::default()).unwrap().0;
    let target = s.target;
    s.cameras.retain(|c| {
        let a = azimuth(c, &target).to_degrees();
        (-150.0..-30.0).contains(&a)
    });
    let (map, _) = build_map(&s.to_scene(), &MapConfig::default()).unwrap();
    // Find open-air points (true clearance >= 0.5 m) on the hidden side that are
    // free in the full map but unknown with half the cameras.
    let mut hidden = None;
    'search: for j in 0..14 {
        for i in 0..31 {
            for k in 0..12 {
                let p = Vec3::new(
                    -1.5 + 0.1 * i as f64,
                    1.3 + 0.1 * j as f64,
                    0.65 + 0.05 * k as f64,
                );
                if s.true_clearance(&p) >= 0.5
                    && full.state_at(&p) == VoxelState::FreeObserved
                    && map.state_at(&p) == VoxelState::Unknown
                {
                    hidden = Some(p);
                    break 'search;
                }
            }
        }
    }
    let hidden = hidden.expect("removing cameras should leave some open air unobserved");
    let r = plan_path(
        &map,
        &Vec3::new(0.0, -4.0, 1.5),
        &hidden,
        &PlannerConfig::default(),
    );
    assert!(
        matches!(
            r,
            Err(PlanError::NotTraversable {
                which: "goal",
                state: VoxelState::Unknown,
                ..
            })
        ),
        "{r:?}"
    );
}

#[test]
fn blocked_when_clearance_too_large() {
    let (_, map) = build(&CarConfig::default());
    let cfg = PlannerConfig {
        robot_radius: 3.0,
        ..PlannerConfig::default()
    };
    assert!(
        plan_path(
            &map,
            &Vec3::new(-4.0, 0.0, 1.0),
            &Vec3::new(4.0, 0.0, 1.0),
            &cfg
        )
        .is_err()
    );
    let bad = PlannerConfig {
        max_speed: -1.0,
        ..PlannerConfig::default()
    };
    assert!(matches!(
        plan_path(
            &map,
            &Vec3::new(-4.0, 0.0, 1.0),
            &Vec3::new(4.0, 0.0, 1.0),
            &bad
        ),
        Err(PlanError::BadConfig(_))
    ));
}
