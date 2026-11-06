//! Release-mode performance budgets. Compiled only without debug
//! assertions, i.e. by `cargo test --release`.
//!
//! Budgets are deliberately generous and per platform: CI runners are much
//! slower than the developer's Apple Silicon laptop.
#![cfg(not(debug_assertions))]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use std::time::Instant;

use gs_map::{MapConfig, build_map, edt_3d};
use gs_verify::{Verifier, VerifierConfig};
use planner::{PlannerConfig, plan_and_certify, plan_path};
use splat_types::Vec3;
use synth::{CarConfig, car_scene};

/// (esdf_10m_s, plan_median_ms, plan_verify_max_ms)
fn budgets() -> (f64, f64, f64) {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        // Apple Silicon (developer laptop and the macOS CI runner).
        (2.0, 150.0, 1000.0)
    } else {
        (4.0, 400.0, 3000.0)
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[test]
fn perf_budgets() {
    let (esdf_budget, plan_budget, pv_budget) = budgets();

    // ESDF on a 10M-voxel grid (216³) with sparse obstacles; best of 3.
    let dims = [216, 216, 216];
    let n = dims[0] * dims[1] * dims[2];
    let base: Vec<f32> = (0..n)
        .map(|i| if i % 997 == 0 { 0.0 } else { f32::INFINITY })
        .collect();
    let mut esdf = f64::INFINITY;
    for _ in 0..3 {
        let mut g = base.clone();
        let t = Instant::now();
        edt_3d(&mut g, dims);
        esdf = esdf.min(t.elapsed().as_secs_f64());
    }
    println!("ESDF {n} voxels: {esdf:.3} s (budget {esdf_budget} s)");
    assert!(esdf < esdf_budget);

    // Plans on the synthetic car, run one at a time.
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
    let queries = [
        (Vec3::new(-3.2, 0.5, 1.0), Vec3::new(3.2, -0.5, 1.4)),
        (Vec3::new(0.0, -2.5, 1.2), Vec3::new(0.0, 2.5, 1.2)),
        (Vec3::new(-2.5, -2.5, 2.0), Vec3::new(2.5, 2.5, 0.9)),
        (Vec3::new(3.0, 2.0, 1.5), Vec3::new(-3.0, -2.0, 1.0)),
        (Vec3::new(-1.0, 2.6, 0.8), Vec3::new(1.0, -2.6, 2.4)),
    ];
    let mut plan_ms = Vec::new();
    let mut pv_ms = Vec::new();
    for (a, b) in &queries {
        let t = Instant::now();
        plan_path(&map, a, b, &cfg).unwrap();
        plan_ms.push(t.elapsed().as_secs_f64() * 1e3);
        let t = Instant::now();
        plan_and_certify(&map, &verifier, a, b, &cfg).unwrap();
        pv_ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let pm = median(plan_ms);
    let pv = pv_ms.iter().copied().fold(0.0, f64::max);
    println!(
        "plan median {pm:.1} ms (budget {plan_budget}), plan+verify max {pv:.1} ms (budget {pv_budget})"
    );
    assert!(pm < plan_budget);
    assert!(pv < pv_budget);
}
