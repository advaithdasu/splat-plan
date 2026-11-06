//! Criterion benchmarks: A* + shortcut, smoothing, and plan + verify on the synthetic car.
#![allow(missing_docs, clippy::unwrap_used)]

use criterion::{Criterion, criterion_group, criterion_main};
use gs_map::{MapConfig, build_map};
use gs_verify::{Verifier, VerifierConfig};
use planner::{PlannerConfig, Workspace, plan_and_certify, plan_path, smooth};
use splat_types::Vec3;
use synth::{CarConfig, car_scene};

fn bench(c: &mut Criterion) {
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
    // Around the car, end to end: a typical query.
    let (a, b) = (Vec3::new(-3.2, 0.5, 1.0), Vec3::new(3.2, -0.5, 1.4));
    let poly = plan_path(&map, &a, &b, &cfg).unwrap();
    let mut g = c.benchmark_group("planner");
    g.sample_size(20);
    g.bench_function("astar_shortcut", |bch| {
        bch.iter(|| plan_path(&map, &a, &b, &cfg).unwrap())
    });
    g.bench_function("min_snap_smooth", |bch| {
        bch.iter(|| smooth(&Workspace::new(&map), &poly, &cfg, 1.0).unwrap())
    });
    g.bench_function("plan_and_certify", |bch| {
        bch.iter(|| plan_and_certify(&map, &verifier, &a, &b, &cfg).unwrap())
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
