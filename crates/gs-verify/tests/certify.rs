//! Certificates on the synthetic car.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use gs_verify::{Verdict, Verifier, VerifierConfig, ViolationReason};
use splat_types::{PolyTrajectory, Report, Scale, Vec3};
use synth::{CarConfig, car_scene};

fn verifier(scale: Scale) -> Verifier {
    let s = car_scene(&CarConfig::default());
    Verifier::new(
        &s.gaussians,
        &s.cameras,
        scale,
        1.0,
        VerifierConfig::default(),
    )
}

#[test]
fn certifies_safe_and_rejects_unsafe() {
    let v = verifier(Scale::Metric(1.0));
    // Well clear of the car, low speed.
    let safe = PolyTrajectory::polyline(
        &[Vec3::new(-3.5, -2.5, 1.5), Vec3::new(3.5, -2.5, 1.5)],
        1.0,
    )
    .unwrap();
    let c = v.certify(&safe);
    assert_eq!(c.verdict, Verdict::Safe, "{c:#?}");
    assert!(c.min_clearance_m >= c.margin_required_m);
    assert!(c.metric && c.support_ok);
    let d = VerifierConfig::default();
    let need = d.robot_radius + d.margin + d.max_speed * d.dt / 2.0;
    assert!((c.margin_required_m - need).abs() < 1e-12);

    // Straight through the car body.
    let through =
        PolyTrajectory::polyline(&[Vec3::new(-3.5, 0.0, 1.0), Vec3::new(3.5, 0.0, 1.0)], 1.0)
            .unwrap();
    let c = v.certify(&through);
    assert_eq!(c.verdict, Verdict::Unsafe);
    assert!(
        c.violations
            .iter()
            .any(|x| x.reason == ViolationReason::Clearance)
    );
    assert_eq!(c.min_clearance_m, 0.0);

    // Too fast.
    let fast = PolyTrajectory::polyline(
        &[Vec3::new(-3.5, -2.5, 1.5), Vec3::new(3.5, -2.5, 1.5)],
        3.0,
    )
    .unwrap();
    let c = v.certify(&fast);
    assert_eq!(c.verdict, Verdict::Unsafe);
    assert!(
        c.violations
            .iter()
            .all(|x| x.reason == ViolationReason::Limits)
    );

    // Far from every camera (but in open air above the scene).
    let far = PolyTrajectory::polyline(
        &[Vec3::new(0.0, -2.5, 12.0), Vec3::new(1.0, -2.5, 12.0)],
        1.0,
    )
    .unwrap();
    let c = v.certify(&far);
    assert!(!c.support_ok);
    assert!(
        c.violations
            .iter()
            .any(|x| x.reason == ViolationReason::Support)
    );
}

#[test]
fn unscaled_is_never_safe() {
    let v = verifier(Scale::Unscaled);
    let safe = PolyTrajectory::polyline(
        &[Vec3::new(-3.5, -2.5, 1.5), Vec3::new(3.5, -2.5, 1.5)],
        1.0,
    )
    .unwrap();
    let c = v.certify(&safe);
    assert_eq!(c.verdict, Verdict::Unscaled);
    assert!(!c.metric);
    let json = c.to_json().unwrap();
    assert!(json.contains("\"verdict\": \"UNSCALED\""));
    assert!(json.contains("\"metric\": false"));
}

#[test]
fn certificate_json_has_appendix_d_fields() {
    let v = verifier(Scale::Metric(1.0));
    let t = PolyTrajectory::polyline(&[Vec3::new(-3.5, 0.0, 1.0), Vec3::new(3.5, 0.0, 1.0)], 1.0)
        .unwrap();
    let json: serde_json::Value = serde_json::from_str(&v.certify(&t).to_json().unwrap()).unwrap();
    for key in [
        "verdict",
        "metric",
        "min_clearance_m",
        "margin_required_m",
        "samples",
        "max_speed",
        "max_accel",
        "support_ok",
        "violations",
        "config",
    ] {
        assert!(json.get(key).is_some(), "missing {key}");
    }
    let viol = &json["violations"][0];
    assert!(viol["t"].is_number() && viol["position"].is_array());
    assert_eq!(viol["reason"], "clearance");
    for key in ["k_sigma", "opacity_floor", "dt"] {
        assert!(json["config"].get(key).is_some(), "missing config.{key}");
    }
}

/// A wall at x = 0 with every camera on the −x side: space behind it was
/// never seen, contains no Gaussians, and must not be certified.
#[test]
fn line_of_sight_rejects_space_hidden_behind_a_wall() {
    use splat_types::CameraPose;
    use synth::{Primitive, default_intrinsics};
    let wall = Primitive::Box {
        center: Vec3::new(0.0, 0.0, 1.5),
        half: Vec3::new(0.05, 3.0, 1.5),
        yaw: 0.0,
    };
    let gaussians = wall.splat(0.04, 0.0, [0.5, 0.5, 0.5], 0.9);
    let cameras: Vec<CameraPose> = (0..7)
        .map(|i| {
            let y = -1.5 + 0.5 * f64::from(i);
            CameraPose::look_at(
                format!("cam{i}"),
                Vec3::new(-4.0, y, 1.5),
                Vec3::new(0.0, 0.0, 1.5),
                Vec3::z(),
                default_intrinsics(),
            )
            .unwrap()
        })
        .collect();
    let v = Verifier::new(
        &gaussians,
        &cameras,
        Scale::Metric(1.0),
        1.0,
        VerifierConfig::default(),
    );
    let path = |x: f64| {
        PolyTrajectory::polyline(&[Vec3::new(x, -1.0, 1.5), Vec3::new(x, 1.0, 1.5)], 1.0).unwrap()
    };

    let seen = v.certify(&path(-1.5));
    assert_eq!(seen.verdict, Verdict::Safe, "{seen:#?}");
    assert!(seen.observed_ok && seen.support_ok);
    assert_eq!(seen.config.min_views, 1);

    // Same clearance and camera distance, but behind the wall.
    let hidden = v.certify(&path(1.5));
    assert_eq!(hidden.verdict, Verdict::Unsafe);
    assert!(!hidden.observed_ok);
    assert!(hidden.support_ok, "fails only on line of sight");
    assert!(hidden.min_clearance_m > 1.0);
    assert!(
        hidden
            .violations
            .iter()
            .all(|x| x.reason == ViolationReason::Unobserved)
    );
    assert_eq!(hidden.violation_count, hidden.samples);

    // Disabling the check certifies it: proves the check, not something
    // else, is what refuses the hidden path.
    let off = Verifier::new(
        &gaussians,
        &cameras,
        Scale::Metric(1.0),
        1.0,
        VerifierConfig {
            min_views: 0,
            ..VerifierConfig::default()
        },
    );
    assert_eq!(off.certify(&path(1.5)).verdict, Verdict::Safe);

    // Asking for more views than cameras exist rejects even the seen path.
    let strict = Verifier::new(
        &gaussians,
        &cameras,
        Scale::Metric(1.0),
        1.0,
        VerifierConfig {
            min_views: 8,
            ..VerifierConfig::default()
        },
    );
    assert!(!strict.certify(&path(-1.5)).observed_ok);
}

/// Certificates written before the line-of-sight check still parse, and
/// read as "not checked" rather than "passed".
#[test]
fn old_certificates_parse_as_unchecked() {
    let v = verifier(Scale::Metric(1.0));
    let safe = PolyTrajectory::polyline(
        &[Vec3::new(-3.5, -2.5, 1.5), Vec3::new(3.5, -2.5, 1.5)],
        1.0,
    )
    .unwrap();
    let mut json: serde_json::Value =
        serde_json::from_str(&v.certify(&safe).to_json().unwrap()).unwrap();
    let obj = json.as_object_mut().unwrap();
    obj.remove("observed_ok");
    obj["config"].as_object_mut().unwrap().remove("min_views");
    let old: gs_verify::Certificate = serde_json::from_value(json).unwrap();
    assert!(!old.observed_ok);
    assert_eq!(old.config.min_views, 0);
}
