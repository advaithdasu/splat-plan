//! Criterion benchmarks for each map-building stage on the synthetic car.
#![allow(missing_docs, clippy::unwrap_used)]

use criterion::{Criterion, criterion_group, criterion_main};
use gs_map::{
    BitSet, DensityGrid, GridSpec, MapConfig, RayCastConfig, build_map, edt_3d, observed_free,
    scene_bounds,
};
use synth::{CarConfig, car_scene};

fn stages(c: &mut Criterion) {
    let scene = car_scene(&CarConfig::default()).to_scene();
    let cfg = MapConfig::default();
    let spec = GridSpec::covering(&scene_bounds(&scene, &cfg).unwrap(), cfg.voxel);
    let mut g = c.benchmark_group("map");
    g.sample_size(10);
    g.bench_function("density", |b| {
        b.iter(|| DensityGrid::accumulate(spec, &scene.gaussians, 3.0, 0.02))
    });
    let dens = DensityGrid::accumulate(spec, &scene.gaussians, 3.0, 0.02);
    let mut occ = BitSet::new(spec.len());
    for (i, d) in dens.density.iter().enumerate() {
        if *d >= 0.2 {
            occ.set(i);
        }
    }
    let rc = RayCastConfig {
        ray_spacing: 0.05,
        dense_range: 7.0,
        max_range: 15.0,
    };
    g.bench_function("raycast", |b| {
        b.iter(|| observed_free(&spec, &occ, &scene.cameras, &rc))
    });
    g.bench_function("esdf", |b| {
        b.iter(|| {
            let mut d: Vec<f32> = (0..spec.len())
                .map(|i| if occ.get(i) { 0.0 } else { f32::INFINITY })
                .collect();
            edt_3d(&mut d, spec.dims);
            d
        })
    });
    g.bench_function("build_map_total", |b| {
        b.iter(|| build_map(&scene, &cfg).unwrap())
    });
    g.finish();
}

criterion_group!(benches, stages);
criterion_main!(benches);
