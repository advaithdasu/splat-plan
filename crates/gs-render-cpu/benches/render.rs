//! Criterion benchmark: one 320x240 frame of the synthetic car.
#![allow(missing_docs, clippy::unwrap_used)]

use criterion::{Criterion, criterion_group, criterion_main};
use gs_render_cpu::{RenderConfig, render};
use synth::{CarConfig, car_scene};

fn bench(c: &mut Criterion) {
    let s = car_scene(&CarConfig::default());
    let cam = &s.cameras[5];
    let mut g = c.benchmark_group("render");
    g.sample_size(20);
    g.bench_function("car_320x240", |b| {
        b.iter(|| render(&s.gaussians, cam, &RenderConfig::default()))
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
