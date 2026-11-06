# Web viewer

A static page that shows each splat with the drone's recovered flight path and its video. It can also draw planned paths as a layer. Uses [three.js](https://threejs.org) and [Spark](https://sparkjs.dev) from jsDelivr, so there's no build step.

## Make the data

```bash
splatplan clean  data/captures/<name>
scripts/register_all.sh data/captures/<name>     # ~10-15 min on CPU
splatplan flight data/captures/<name>
splatplan export data/captures/<name> --out web/data/<scene> --flight reports/<name>/flight --crop 8
ffmpeg -i data/captures/<name>/raw/video.mp4 -an -vf scale=640:-2 -c:v libx264 \
  -crf 30 -g 15 -pix_fmt yuv420p -movflags +faststart web/data/<scene>/video.mp4
```

(`splatplan` here is `cargo run --release -p splatplan-cli --`.)

`flight` drops a pose if the splat rendered from it doesn't match the video frame, or if the drone would have moved faster than 2 m/s. Stretches with no pose are drawn as gaps, not filled in. `--crop` only trims what's displayed.

To add planned paths, run `splatplan replan data/captures/<name>` (paths between the start, end and hover points of the real flight) and add `--paths reports/<name>/replan` to `export`. `showcase` and `weave` outputs work too. `export` re-checks every path with the verifier and refuses if one no longer passes (say, the splat was retrained). It keeps at most 5, ranked by length × (1 + obstacles avoided). `--max-paths` lowers the cap. The viewer also never draws more than 5.

Each scene in `web/data/<scene>/`:

| File | What it is |
| --- | --- |
| `scene.splat` | The splat (32-byte `.splat` layout, meters, +Z up) |
| `scene.json` | Where it came from and the starting view |
| `flight.json`, `video.mp4` | The drone's recovered path and the video |
| `paths.json` | Planned paths, best first, sampled every 0.1 s, with the real trip they match (may be empty) |

`web/data/` is git-ignored. Each scene button in `index.html` names its folder (`data-scene`) and the title shown in the panel (`data-title`).

## Run it

```bash
npx serve web
```

Use a server that supports HTTP range requests, or video seeking won't work (`python3 -m http.server` doesn't). To publish, copy `web/` with its data to any static host, or run `scripts/deploy_demo.sh` to deploy it to Vercel (needs the `vercel` CLI). It needs WebGL 2.
