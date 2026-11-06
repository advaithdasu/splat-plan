# splat-types

The shared vocabulary of the workspace. Every other crate speaks in these types.

| Type | What it is |
| --- | --- |
| `Gaussian` | One splat: mean, linear scales, rotation, opacity (0..1), DC color. Activated units (no logs/logits). |
| `CameraPose` | COLMAP convention: world-to-camera `R`, `t`; center `C = -Rᵀt`; x right, y down, z forward. |
| `Scale` | `Metric(f64)` or `Unscaled`. Unscaled results are never certified safe. |
| `Similarity` | `x' = s·R·x + t`, applied to Gaussians and cameras together via `Scene::transformed`. |
| `PolyTrajectory` | Piecewise degree-7 polynomials; `eval(t)` gives position through jerk. |
| `Report` | Trait for anything that writes a JSON metrics report. |

Read `camera.rs` first: the `C = -Rᵀt` test is the classic bug this project guards against.
