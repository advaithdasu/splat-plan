# SplatPlan

I flew a drone through two rooms, turned each video into a 3D Gaussian splat, and recovered the path the drone actually flew by placing the video frames back in the splat. The web viewer plays the video and the path together, and can show the splat from where the drone was at any moment.

**[Live demo](https://splatplan-demo.vercel.app)**: both rooms, with the video and flight path playing together.

## The footage

Runs used nothing but the camera on the DJI Neo I flew in each room. No flight logs, GPS, IMU or depth data are used.

## How it works

1. **Video to splat.** Pick sharp keyframes, recover their camera poses with [COLMAP](https://github.com/colmap/colmap), and train a splat with [Brush](https://github.com/ArthurBrussee/brush). Runs on a Mac, about 10 minutes of training per room.
2. **Clean, level, scale.** Remove faint floaters, level the scene, and set the scale in meters from one known distance (tape marks, or a desk of known length).
3. **Recover the flight path.** Register 10 frames per second of the video into the same COLMAP model, so every pose is in the splat's frame. A pose is dropped if the splat rendered from it doesn't match the video frame, or if the drone would have had to move faster than 2 m/s to get there. Missing stretches stay gaps. They are never filled in.
4. **Show it.** Export the splat and the flight path, add a small copy of the video, and open them in a static web viewer.

Everything runs on one Mac. COLMAP runs on the CPU, and Brush trains on the Apple GPU. The Rust steps (cleaning, leveling, checking poses against the video, planning and verifying) are CPU-only, and the heavy ones run in parallel with [rayon](https://github.com/rayon-rs/rayon). The map and planner give identical results on 1 thread or many; tests check that.

| Room | Frames placed | Flown | Gaps |
| --- | --- | --- | --- |
| Engineering loft | 1,481 of 1,522 | 27 m in 2:32 | one, 3.6 s |
| Basement workspace | 1,527 of 1,684 | 49 m in 2:48 | six, 16 s total |

Distances count only the recovered path, so the real flights were a bit longer: nothing is added for the gaps.

## Try it without footage

The footage isn't in this repo, but the planning half runs on a synthetic scene with known geometry (macOS or Linux, [Rust](https://rustup.rs) required):

```bash
cargo test --workspace
cargo run --release -p splatplan-cli -- synth car-pocket --out out/car --write-scale
cargo run --release -p splatplan-cli -- pipeline out/car --out out/car-report
```

That writes a map slice, a trajectory and a safety certificate to `out/car-report/`.

## Your own footage

This needs an Apple Silicon Mac and Rust.

```bash
scripts/setup_macos.sh                         # ffmpeg, COLMAP, Brush
scripts/ingest.sh data/captures/<name>         # video -> splat
# give it a scale: scale.json or annotations.json, see data/README.md
splatplan clean  data/captures/<name>          # remove floaters
scripts/register_all.sh data/captures/<name>   # place every video frame (~10-15 min)
splatplan flight data/captures/<name>          # check the poses -> flight path
splatplan replan data/captures/<name>          # planned paths between real flight points
splatplan export data/captures/<name> --out web/data/<scene> --flight reports/<name>/flight --crop 8 \
  --paths reports/<name>/replan
```

(`splatplan` is `cargo run --release -p splatplan-cli --`.) Filming tips and setting the scale: [data/README.md](data/README.md). Adding the video and running the viewer: [web/README.md](web/README.md). All commands: [apps/splatplan-cli](apps/splatplan-cli).

## Planned paths

The project started as a drone path planner, and that part still works. `splatplan replan` plans new paths between points the drone really visited:

1. **Pick endpoints.** Use the start and end of the video, plus every spot the drone hovered for 1.5 s or more. Each point is lifted straight up until there's room for the drone.
2. **Plan.** Build an occupancy map from the splat, search it with A\*, and smooth the result into a flyable trajectory. Space only counts as free if a camera looked through it.
3. **Verify.** A separate verifier checks each trajectory against the raw Gaussians. It shares no code with the planner or the map, and CI checks that. Without a real-world scale, nothing is called safe.
4. **Show it.** Keep up to five paths per room, ranked by length times (1 + obstacles the straight line runs into). The viewer draws the drone's real route beside the selected one. A takeoff-to-landing path is kept first if it certifies. If none does, the trip is the certified path from takeoff that ends closest to the real landing spot, and the uncertified last bit is drawn faded.

| Room | Pairs certified | Best path |
| --- | --- | --- |
| Basement | 83 of 133 | Takeoff to the hover above the landing spot: 9.6 m around 5 obstacles, in 11 s (the drone flew 48.6 m) |
| Loft | 6 of 49 | Takeoff to landing: 4.1 m certified, plus faded ends to the real takeoff and landing (the drone flew 26 m) |

`splatplan showcase` (random endpoints) and `splatplan weave` (tours through the gaps between obstacles) are still there too.

## Crates

| Crate | Does |
| --- | --- |
| [`frames`](crates/frames) | Picks sharp keyframes from a video |
| [`colmap-io`](crates/colmap-io), [`gs-io`](crates/gs-io) | Read and write COLMAP models and splat `.ply` files |
| [`gs-align`](crates/gs-align) | Levels the scene and sets the scale |
| [`gs-render-cpu`](crates/gs-render-cpu) | Small CPU splat renderer (used to check flight poses against the video) |
| [`gs-map`](crates/gs-map) | Occupancy grid and distance field (planning) |
| [`planner`](crates/planner) | A\*, shortcutting, minimum-snap trajectories (planning) |
| [`gs-verify`](crates/gs-verify) | The independent safety checker (planning) |
| [`synth`](crates/synth) | Synthetic scenes with known geometry, for tests |
| [`splat-types`](crates/splat-types) | Shared types |
| [`splatplan-cli`](apps/splatplan-cli) | The `splatplan` command |

## Limits

- Plain carpet barely reconstructs, so indoors the leveling can lock onto desk tops instead of the floor. `ground.json` sets the real floor height.
- Rooms full of identical desks can fool feature matching. In the basement, matching every pair of keyframes folded parts of the room onto each other. Dropping long-range matches with under 30 inliers fixed it, but `ingest.sh` doesn't do that step yet.
- Glass and screens reconstruct poorly.
- No planned path has been flown. `SAFE` means the path is clear in the reconstructed model, not in the real room. A hole in the splat looks like free space, and the verifier ignores Gaussians below an opacity floor (0.05, recorded in every certificate).

## License

MIT
