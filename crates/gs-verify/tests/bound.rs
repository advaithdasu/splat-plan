//! The clearance lower bound never exceeds the exact distance.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use gs_verify::{Bvh, Ellipsoid};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use splat_types::nalgebra::{Quaternion, UnitQuaternion};
use splat_types::{Gaussian, Vec3, Vec3f};
use synth::distance_to_ellipsoid_local;

fn random_gaussian(rng: &mut ChaCha8Rng) -> Gaussian {
    Gaussian {
        mean: Vec3f::new(
            rng.gen_range(-2.0..2.0),
            rng.gen_range(-2.0..2.0),
            rng.gen_range(-2.0..2.0),
        ),
        // Includes very flat discs (ratio up to 1000:1), as in trained splats.
        scale: Vec3f::new(
            10f32.powf(rng.gen_range(-3.0..-0.5)),
            10f32.powf(rng.gen_range(-3.0..-0.5)),
            10f32.powf(rng.gen_range(-3.0..-0.5)),
        ),
        rot: UnitQuaternion::from_quaternion(Quaternion::new(
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        )),
        opacity: 1.0,
        sh_dc: [0.0; 3],
    }
}

fn exact(e: &Ellipsoid, p: &Vec3) -> f64 {
    distance_to_ellipsoid_local(&(e.world_to_local * (p - e.center)), &e.axes)
}

#[test]
fn lower_bound_never_exceeds_exact_distance_10k() {
    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    let mut ratios = Vec::new();
    for i in 0..10_000 {
        let g = random_gaussian(&mut rng);
        let e = Ellipsoid::from_gaussian(&g, 3.0, i);
        // Query points at a range of distances, some inside.
        let p = g.mean_f64()
            + Vec3::new(
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            ) * 10f64.powf(rng.gen_range(-3.0..0.5));
        let lb = e.lower_bound(&p);
        let d = exact(&e, &p);
        // Tolerance covers f32→f64 rounding of the stored rotation only.
        assert!(
            lb <= d + 1e-9 * (1.0 + d),
            "case {i}: bound {lb} > exact {d}"
        );
        if d > 1e-3 {
            ratios.push(lb.max(0.0) / d);
        }
    }
    ratios.sort_by(f64::total_cmp);
    let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
    let p05 = ratios[ratios.len() / 20];
    let p50 = ratios[ratios.len() / 2];
    println!(
        "tightness (bound / exact, outside points, n={}): mean {mean:.3}, p5 {p05:.3}, median {p50:.3}, min {:.3}",
        ratios.len(),
        ratios[0]
    );
    assert!(mean > 0.5, "bound unexpectedly loose: mean ratio {mean}");
}

#[test]
fn bvh_result_is_a_valid_bound_and_not_looser_than_needed() {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let items: Vec<Ellipsoid> = (0..2000)
        .map(|i| Ellipsoid::from_gaussian(&random_gaussian(&mut rng), 3.0, i))
        .collect();
    let bvh = Bvh::build(items.clone());
    for _ in 0..500 {
        let p = Vec3::new(
            rng.gen_range(-3.0..3.0),
            rng.gen_range(-3.0..3.0),
            rng.gen_range(-3.0..3.0),
        );
        let (b, who) = bvh.nearest_lower_bound(&p);
        let brute_lb = items
            .iter()
            .map(|e| e.lower_bound(&p))
            .fold(f64::INFINITY, f64::min);
        let brute_exact = items
            .iter()
            .map(|e| exact(e, &p))
            .fold(f64::INFINITY, f64::min);
        assert!(b <= brute_exact + 1e-9, "bvh {b} > exact {brute_exact}");
        assert!(b >= brute_lb - 1e-12, "bvh {b} < brute lb {brute_lb}");
        assert!(who.is_some());
    }
    assert_eq!(
        Bvh::build(vec![]).nearest_lower_bound(&Vec3::zeros()).1,
        None
    );
}

fn random_point(rng: &mut ChaCha8Rng, r: f64) -> Vec3 {
    Vec3::new(
        rng.gen_range(-r..r),
        rng.gen_range(-r..r),
        rng.gen_range(-r..r),
    )
}

/// Line of sight: a segment that passes through an ellipsoid is always
/// reported as blocked, and a reported hit is never far from the ellipsoid.
#[test]
fn segment_hit_never_misses_an_intersection() {
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let steps = 2000;
    let (mut hits, mut misses) = (0, 0);
    for i in 0..3000 {
        let g = random_gaussian(&mut rng);
        let e = Ellipsoid::from_gaussian(&g, 3.0, i);
        // Segments near the ellipsoid, many passing through it.
        let c = g.mean_f64();
        let a = c + random_point(&mut rng, 0.6);
        let b = c + random_point(&mut rng, 0.6);
        let hit = e.segment_may_hit(&a, &b);
        // Exact distance along a dense sampling of the segment.
        let min_d = (0..=steps)
            .map(|k| exact(&e, &(a + (b - a) * (k as f64 / steps as f64))))
            .fold(f64::INFINITY, f64::min);
        if min_d == 0.0 {
            assert!(hit, "case {i}: segment enters the ellipsoid but no hit");
        }
        if hit {
            let step_len = (b - a).norm() / steps as f64;
            assert!(
                min_d <= step_len + 1e-9,
                "case {i}: hit reported but segment stays {min_d} away"
            );
            hits += 1;
        } else {
            misses += 1;
        }
    }
    assert!(hits > 300 && misses > 300, "{hits} hits, {misses} misses");
}

#[test]
fn bvh_segment_query_matches_brute_force() {
    let mut rng = ChaCha8Rng::seed_from_u64(12);
    let items: Vec<Ellipsoid> = (0..2000)
        .map(|i| Ellipsoid::from_gaussian(&random_gaussian(&mut rng), 3.0, i))
        .collect();
    let bvh = Bvh::build(items.clone());
    let mut blocked = 0;
    for _ in 0..1000 {
        let a = random_point(&mut rng, 3.0);
        // Short and long segments.
        let b = a + random_point(&mut rng, 1.0) * 10f64.powf(rng.gen_range(-1.0..0.6));
        let brute = items.iter().any(|e| e.segment_may_hit(&a, &b));
        assert_eq!(bvh.segment_blocked(&a, &b), brute, "segment {a:?} -> {b:?}");
        blocked += usize::from(brute);
    }
    assert!(blocked > 100 && blocked < 900, "{blocked} blocked");
    assert!(!Bvh::build(vec![]).segment_blocked(&Vec3::zeros(), &Vec3::x()));
}
