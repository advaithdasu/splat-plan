# splatplan (CLI)

`splatplan <command>`. Commands that read a capture take its directory (`data/captures/<name>`).

| Command | What it does |
| --- | --- |
| `inspect` | Summary of a `.ply`. |
| `clean` | Removes floaters (faint *and* isolated Gaussians) into `splat/point_cloud.clean.ply`, logged in `splat/clean.json`. |
| `align` | Alignment report: leveling and scale. |
| `flight` | The drone's real path from the video frames (run `scripts/register_all.sh` first). Drops poses that don't match the video or move faster than 2 m/s. |
| `export` | Splat, flight path and up to 5 planned paths for the web viewer. Re-verifies every path. |
| `render` / `psnr` | Flythrough frames; renderer check against the photos. |
| `synth` | Writes a synthetic capture (car, car-pocket, field). |

Planning (optional):

| Command | What it does |
| --- | --- |
| `map` | Map report and distance-field slices. |
| `plan` / `verify` | Plan and certify one path; re-verify a saved trajectory. |
| `orbit` | Certified orbit around the largest object. |
| `pipeline` | Align, map, plan 50 random trips and verify them. Writes `reports/<name>/`. |
| `replan` | Certified paths between places the drone really was (start, end, hovers), each with how far the drone actually flew between them. |
| `showcase` | Keeps a few long certified paths between random endpoints whose straight line is blocked. |
| `weave` | Long tours that go around obstacles, through the gaps between them, never over. |

Run `splatplan <command> --help` for the options.

Once `clean` has run, every command uses the cleaned splat and says so in its reports (`--raw-splat` to opt out). If the original splat changes, commands refuse until you clean again. For planning, `--robot-radius` and `--margin` (defaults 0.15 m and 0.10 m) set the drone size for both planner and verifier.
