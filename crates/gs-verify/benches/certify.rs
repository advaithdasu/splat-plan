//! Criterion benchmarks: verifier construction (BVH) and certification.
#![allow(missing_docs, clippy::unwrap_used)]

use criterion::{Criterion, criterion_group, criterion_main};
use gs_verify::{Verifier, VerifierConfig};
use splat_types::{PolyTrajectory, Vec3};
use synth::{CarConfig, car_scene};

fn bench(c: &mut Criterion) {
    let s = car_scene(&CarConfig::default());
    let scene = s.to_scene();
    let traj = PolyTrajectory::polyline(
        &[
            Vec3::new(-3.5, -2.5, 1.5),
            Vec3::new(3.5, -2.5, 1.5),
            Vec3::new(3.5, 2.5, 1.5),
        ],
        1.0,
    )
    .unwrap();
    let mut g = c.benchmark_group("verifier");
    g.sample_size(20);
    g.bench_function("build_bvh", |b| {
        b.iter(|| {
            Verifier::new(
                &scene.gaussians,
                &scene.cameras,
                scene.scale,
                1.0,
                VerifierConfig::default(),
            )
        })
    });
    let v = Verifier::new(
        &scene.gaussians,
        &scene.cameras,
        scene.scale,
        1.0,
        VerifierConfig::default(),
    );
    g.bench_function("certify_12m_path", |b| b.iter(|| v.certify(&traj)));
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
