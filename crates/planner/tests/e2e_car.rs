//! End-to-end safety on the synthetic car: for N = 200 seeded
//! random start/goal pairs, every outcome is either Err or a SAFE certificate
//! whose trajectory clears the exact oracle by at least robot_radius + margin.
//! "Verifier says SAFE but the oracle says unsafe" must never happen.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use std::collections::BTreeMap;

use gs_map::{MapConfig, build_map};
use gs_verify::{Verdict, Verifier, VerifierConfig};
use planner::{PlannerConfig, plan_and_certify};
use rayon::prelude::*;
use splat_types::{Gaussian, PolyTrajectory, Vec3, Vec3f};
use synth::{CarConfig, SynthScene, car_scene};

fn oracle_min_clearance(s: &SynthScene, t: &PolyTrajectory) -> f64 {
    t.sample(0.005)
        .iter()
        .map(|x| s.true_clearance(&x.pos))
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn two_hundred_random_queries_never_unsafe() {
    let s = car_scene(&CarConfig {
        sealed_pocket: true,
        ..CarConfig::default()
    });
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
    let queries = s.random_queries(2026, 200);

    let results: Vec<_> = queries
        .par_iter()
        .map(|(a, b)| plan_and_certify(&map, &verifier, a, b, &cfg))
        .collect();

    let mut ok = 0;
    let mut refusals: BTreeMap<&str, usize> = BTreeMap::new();
    let mut worst = f64::INFINITY;
    for (i, r) in results.iter().enumerate() {
        match r {
            Ok(out) => {
                assert_eq!(out.certificate.verdict, Verdict::Safe, "query {i}");
                let c = oracle_min_clearance(&s, &out.trajectory);
                worst = worst.min(c);
                assert!(
                    c >= need,
                    "UNSAFE vs oracle: query {i} clearance {c:.4} < {need}"
                );
                let (v, a) = out.trajectory.max_speed_accel(0.005);
                assert!(v <= cfg.max_speed + 1e-9 && a <= cfg.max_accel + 1e-9);
                ok += 1;
            }
            Err(e) => *refusals.entry(e.kind()).or_default() += 1,
        }
    }
    println!("success {ok}/200, refusals {refusals:?}, worst oracle clearance {worst:.3} m");
    // Random endpoints land inside the car, near the ground or in unknown
    // space about half the time (so ~25-30% of pairs have two valid
    // endpoints). Refusals for any reason *other* than an invalid endpoint
    // would mean the planner failed on a feasible query; track that too.
    let endpoint = |k: &str| k.starts_with("endpoint_") || k == "outside_map";
    let other: usize = refusals
        .iter()
        .filter(|(k, _)| !endpoint(k))
        .map(|(_, v)| v)
        .sum();
    assert!(ok >= 40, "only {ok} successes; refusals {refusals:?}");
    assert!(
        other * 10 <= ok,
        "too many non-endpoint refusals: {refusals:?}"
    );

    // Determinism: re-running a few queries gives byte-identical trajectories.
    for (a, b) in queries.iter().take(10) {
        let x = plan_and_certify(&map, &verifier, a, b, &cfg).map(|o| o.trajectory);
        let y = plan_and_certify(&map, &verifier, a, b, &cfg).map(|o| o.trajectory);
        assert_eq!(x, y);
    }
}

#[test]
fn repair_loop_routes_around_floater_only_the_verifier_sees() {
    // The map is built without the floater; the verifier sees it. The first
    // plan flies through it, the verifier rejects, and the keep-out sphere
    // makes the second plan go around.
    let s = car_scene(&CarConfig::default());
    let scene = s.to_scene();
    let (map, _) = build_map(&scene, &MapConfig::default()).unwrap();
    let start = Vec3::new(-3.2, -2.5, 1.5);
    let goal = Vec3::new(3.2, -2.5, 1.5);
    let mut gs = scene.gaussians.clone();
    gs.push(Gaussian::isotropic(Vec3f::new(0.0, -2.5, 1.5), 0.05, 0.9));
    let verifier = Verifier::new(
        &gs,
        &scene.cameras,
        scene.scale,
        1.0,
        VerifierConfig::default(),
    );
    let cfg = PlannerConfig::default();
    let out = plan_and_certify(&map, &verifier, &start, &goal, &cfg).unwrap();
    assert!(
        out.attempts >= 2,
        "expected at least one repair, got {}",
        out.attempts
    );
    assert!(!out.keepouts.is_empty());
    assert_eq!(out.certificate.verdict, Verdict::Safe);
    // With zero repairs allowed the planner refuses instead of returning the bad plan.
    let strict = PlannerConfig {
        max_repairs: 0,
        ..cfg
    };
    let r = plan_and_certify(&map, &verifier, &start, &goal, &strict);
    assert!(
        matches!(r, Err(planner::PlanError::VerifierRejected { .. })),
        "{r:?}"
    );
}

/// A route through via points: certified pieces chained from several valid
/// queries pass every via point in order, without stopping between them,
/// and clear the oracle like any single query. With two points it is
/// exactly `plan_and_certify`.
#[test]
fn via_routes_pass_every_point_and_stay_safe() {
    use planner::plan_and_certify_via;
    let s = car_scene(&CarConfig::default());
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
    // Endpoints of queries that certify on their own, chained into routes.
    let pts: Vec<Vec3> = s
        .random_queries(7, 200)
        .into_iter()
        .filter_map(|(a, b)| {
            plan_and_certify(&map, &verifier, &a, &b, &cfg)
                .ok()
                .map(|_| a)
        })
        .take(12)
        .collect();
    assert!(pts.len() >= 4, "need some valid points, got {}", pts.len());

    let pair = plan_and_certify(&map, &verifier, &pts[0], &pts[1], &cfg).unwrap();
    let via2 = plan_and_certify_via(&map, &verifier, &pts[0..2], &cfg).unwrap();
    assert_eq!(pair.waypoints, via2.waypoints);

    let mut certified = 0;
    for w in pts.windows(4) {
        let Ok(o) = plan_and_certify_via(&map, &verifier, w, &cfg) else {
            continue;
        };
        certified += 1;
        assert_eq!(o.certificate.verdict, Verdict::Safe);
        assert!(oracle_min_clearance(&s, &o.trajectory) >= need);
        for v in w {
            assert!(
                o.waypoints.iter().any(|p| (p - v).norm() < 1e-9),
                "via point {v:?} is not on the route"
            );
        }
    }
    assert!(certified > 0, "no 4-point route certified");
}
