//! End-to-end CLI test: synthetic capture → pipeline → reports.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use std::path::PathBuf;
use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_splatplan"))
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("splatplan-cli-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn run(cmd: &mut Command) -> String {
    let out = cmd.output().unwrap();
    assert!(
        out.status.success(),
        "command failed: {}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn synth_then_pipeline_metric_and_unscaled() {
    let root = tmp("e2e");
    let metric = root.join("car");
    let unscaled = root.join("car-unscaled");
    run(bin()
        .args(["synth", "car", "--seed", "4", "--write-scale", "--out"])
        .arg(&metric));
    run(bin()
        .args(["synth", "car", "--seed", "4", "--out"])
        .arg(&unscaled));

    for (cap, expect_metric) in [(&metric, true), (&unscaled, false)] {
        let out = cap.join("report");
        run(bin()
            .arg("pipeline")
            .arg(cap)
            .arg("--out")
            .arg(&out)
            .args(["--queries", "6"]));
        let r: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(out.join("pipeline.json")).unwrap())
                .unwrap();
        assert_eq!(r["synthetic"], true);
        assert_eq!(
            r["scale"]["kind"],
            if expect_metric { "metric" } else { "unscaled" }
        );
        let verdicts = &r["plans_valid_endpoints"]["verdicts"];
        if expect_metric {
            assert!(verdicts["SAFE"].as_u64().unwrap_or(0) >= 1, "{verdicts}");
            assert!(verdicts.get("UNSCALED").is_none());
        } else {
            // Unscaled captures are never certified SAFE.
            assert!(verdicts.get("SAFE").is_none(), "{verdicts}");
            assert!(r["warning"].as_str().unwrap().contains("UNSCALED"));
        }
        assert!(out.join("slice_z1.0m.png").exists());
        let cert: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(out.join("certificate.json")).unwrap())
                .unwrap();
        assert_eq!(cert["metric"], expect_metric);

        // `verify` re-checks the saved trajectory independently.
        let v = bin()
            .arg("verify")
            .arg(cap)
            .arg(out.join("trajectory.json"))
            .output()
            .unwrap();
        assert_eq!(v.status.success(), expect_metric);
    }

    // Flythrough along the certified trajectory.
    let render_dir = root.join("render");
    run(bin()
        .arg("render")
        .arg(&metric)
        .arg("--trajectory")
        .arg(metric.join("report/trajectory.json"))
        .args(["--fps", "2", "--width", "64", "--out"])
        .arg(&render_dir));
    let path: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(render_dir.join("camera_path.json")).unwrap(),
    )
    .unwrap();
    let n = path["frames"].as_array().unwrap().len();
    assert!(n >= 2);
    assert!(render_dir.join("frames/00000.png").exists());
    assert_eq!(path["intrinsics"]["width"], 64);

    // `plan` with an endpoint inside the car is refused with a reason.
    let refused = bin()
        .arg("plan")
        .arg(&metric)
        .args(["--start", "0,0,0.8", "--goal", "0,3,1.5", "--out"])
        .arg(root.join("refused"))
        .output()
        .unwrap();
    assert!(!refused.status.success());
    assert!(String::from_utf8_lossy(&refused.stderr).contains("planner refused"));
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn inspect_prints_summary() {
    let root = tmp("inspect");
    run(bin().args(["synth", "field", "--out"]).arg(&root));
    let s = run(bin().arg("inspect").arg(root.join("splat/point_cloud.ply")));
    assert!(s.contains("Gaussians:") && s.contains("Opacity histogram"));
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn clean_is_recorded_reproducible_and_checked_for_staleness() {
    let root = tmp("clean");
    let cap = root.join("car");
    run(bin()
        .args(["synth", "car", "--seed", "2", "--write-scale", "--out"])
        .arg(&cap));
    run(bin().arg("clean").arg(&cap));
    let ply = std::fs::read(cap.join("splat/point_cloud.clean.ply")).unwrap();
    let rec: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(cap.join("splat/clean.json")).unwrap())
            .unwrap();
    assert_eq!(rec["input"], "splat/point_cloud.ply");
    assert!(rec["metric"].as_bool().unwrap());
    // Same input, same output, byte for byte.
    run(bin().arg("clean").arg(&cap));
    assert_eq!(
        ply,
        std::fs::read(cap.join("splat/point_cloud.clean.ply")).unwrap()
    );

    // Reports and certificates say which splat they used.
    let out = root.join("report");
    run(bin()
        .arg("pipeline")
        .arg(&cap)
        .arg("--out")
        .arg(&out)
        .args(["--queries", "4"]));
    let r: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("pipeline.json")).unwrap()).unwrap();
    assert!(
        r["splat"]
            .as_str()
            .unwrap()
            .contains("point_cloud.clean.ply")
    );
    assert_eq!(r["cleaning"]["input_hash"], rec["input_hash"]);
    let cert: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("certificate.json")).unwrap())
            .unwrap();
    assert!(cert["splat"].as_str().unwrap().contains("cleaned from"));
    let raw = run(bin()
        .arg("verify")
        .arg(&cap)
        .arg(out.join("trajectory.json"))
        .arg("--raw-splat"));
    assert!(raw.contains("(raw, not cleaned)"), "{raw}");

    // Export for the web viewer: every certified path is re-verified.
    let web = root.join("web");
    run(bin()
        .arg("export")
        .arg(&cap)
        .arg("--out")
        .arg(&web)
        .arg("--paths")
        .arg(&out));
    let paths: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(web.join("paths.json")).unwrap()).unwrap();
    let n = std::fs::read_dir(out.join("certified")).unwrap().count();
    assert_eq!(paths.as_array().unwrap().len(), n);
    assert!(paths[0]["view"]["position"].is_array());
    let info: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(web.join("scene.json")).unwrap()).unwrap();
    let splat_len = std::fs::metadata(web.join("scene.splat")).unwrap().len();
    assert_eq!(splat_len, 32 * info["gaussians_exported"].as_u64().unwrap());
    assert!(info["splat"].as_str().unwrap().contains("cleaned"));

    // Retraining (a changed raw splat) makes the cleaned file stale: refuse.
    let mut bytes = std::fs::read(cap.join("splat/point_cloud.ply")).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    std::fs::write(cap.join("splat/point_cloud.ply"), bytes).unwrap();
    let stale = bin().arg("align").arg(&cap).output().unwrap();
    assert!(!stale.status.success());
    assert!(String::from_utf8_lossy(&stale.stderr).contains("changed since"));
    std::fs::remove_dir_all(&root).unwrap();
}
