//! Integration tests of the map on the synthetic car scene.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use gs_map::{MapConfig, VoxelState, build_map};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use splat_types::Vec3;
use synth::{CarConfig, POCKET_CENTER, car_scene};

fn pocket_scene() -> synth::SynthScene {
    car_scene(&CarConfig {
        sealed_pocket: true,
        ..CarConfig::default()
    })
}

#[test]
fn car_map_states_and_conservative_clearance() {
    let s = pocket_scene();
    let (map, rep) = build_map(&s.to_scene(), &MapConfig::default()).unwrap();
    println!("{rep:#?}");
    // Inside the car body: never observed -> unknown or occupied, never free.
    assert_ne!(
        map.state_at(&Vec3::new(0.0, 0.0, 0.75)),
        VoxelState::FreeObserved
    );
    // Inside the sealed pocket: unknown.
    assert_eq!(
        map.state_at(&Vec3::from(POCKET_CENTER)),
        VoxelState::Unknown
    );
    // Open air beside the car, in view of the cameras: free.
    assert_eq!(
        map.state_at(&Vec3::new(0.0, -3.0, 1.5)),
        VoxelState::FreeObserved
    );
    assert_eq!(
        map.state_at(&Vec3::new(0.0, 0.0, 2.6)),
        VoxelState::FreeObserved
    );
    // Below ground: never free.
    assert_ne!(
        map.state_at(&Vec3::new(2.0, 2.0, -0.3)),
        VoxelState::FreeObserved
    );

    // The conservative clearance never exceeds the true clearance.
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let mut checked = 0;
    for _ in 0..20_000 {
        let p = Vec3::new(
            rng.gen_range(-6.0..6.0),
            rng.gen_range(-6.0..6.0),
            rng.gen_range(-0.5..3.5),
        );
        let c = map.clearance(&p);
        if c > 0.0 {
            checked += 1;
            let truth = s.true_clearance(&p);
            assert!(
                c <= truth + 1e-6,
                "map clearance {c} > true {truth} at {p:?}"
            );
        }
    }
    assert!(checked > 1000, "only {checked} free samples");
}

#[test]
fn deterministic_across_thread_counts() {
    // A smaller scene keeps this quick: fewer cameras.
    let s = car_scene(&CarConfig {
        per_ring: 12,
        ..CarConfig::default()
    });
    let scene = s.to_scene();
    let cfg = MapConfig {
        voxel: 0.1,
        ..MapConfig::default()
    };
    let n = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .max(3);
    let mut hashes = Vec::new();
    for threads in [1, 2, n] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let (map, _) = pool.install(|| build_map(&scene, &cfg)).unwrap();
        hashes.push(map.content_hash());
    }
    // And twice at the same thread count.
    let (again, _) = build_map(&scene, &cfg).unwrap();
    hashes.push(again.content_hash());
    assert!(hashes.windows(2).all(|w| w[0] == w[1]), "{hashes:x?}");
}

#[test]
fn unscaled_scene_still_builds_with_guessed_unit() {
    let s = car_scene(&CarConfig {
        per_ring: 6,
        ..CarConfig::default()
    });
    let mut scene = s.to_scene();
    let sim = splat_types::Similarity {
        scale: 3.0,
        ..splat_types::Similarity::identity()
    };
    scene = scene.transformed(&sim, splat_types::Scale::Unscaled);
    let cfg = MapConfig {
        voxel: 0.1,
        ..MapConfig::default()
    };
    let (map, rep) = build_map(&scene, &cfg).unwrap();
    assert_eq!(map.scale, splat_types::Scale::Unscaled);
    assert!(
        rep.length_unit > 2.0 && rep.length_unit < 4.0,
        "{}",
        rep.length_unit
    );
}

#[test]
fn rejects_empty_and_huge() {
    let mut scene = car_scene(&CarConfig::default()).to_scene();
    let tiny = MapConfig {
        voxel: 0.001,
        ..MapConfig::default()
    };
    assert!(matches!(
        build_map(&scene, &tiny),
        Err(gs_map::MapError::TooLarge { .. })
    ));
    scene.gaussians.clear();
    assert_eq!(
        build_map(&scene, &MapConfig::default()).unwrap_err(),
        gs_map::MapError::Empty
    );
}

/// A camera standing outside the percentile box of the Gaussians (the end
/// of a hallway) must still be inside the map, with padding around it.
#[test]
fn bounds_contain_every_camera() {
    let mut scene = car_scene(&CarConfig {
        per_ring: 6,
        ..CarConfig::default()
    })
    .to_scene();
    let far = Vec3::new(25.0, -3.0, 1.5);
    let mut cam = scene.cameras[0].clone();
    cam.trans_cw = -(cam.rot_cw * far);
    scene.cameras.push(cam);
    let cfg = MapConfig::default();
    let b = gs_map::scene_bounds(&scene, &cfg).unwrap();
    assert!(b.contains(&far), "{b:?}");
    assert!(b.max.x >= far.x + cfg.pad - 1e-9);
    // The unit guess for unscaled scenes ignores cameras.
    let mut without = scene.clone();
    without.cameras.pop();
    assert_eq!(
        gs_map::length_unit_guess(&scene),
        gs_map::length_unit_guess(&without)
    );
}
