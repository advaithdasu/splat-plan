# gs-io

Reads and writes the standard 3DGS binary PLY.

- Parses the header and finds properties **by name** (exporters differ in order and in optional properties such as `f_rest_*` or normals).
- Converts log-scales with `exp`, logit opacity with a sigmoid, and normalizes the `(w, x, y, z)` quaternion on load.
- Truncated or malformed files return `Err`, never panic (fuzzed with proptest).
- `inspect` prints counts, robust bounds, scale percentiles and an opacity histogram (`splatplan inspect file.ply`).
- `floater_mask` picks out floaters for `splatplan clean`: Gaussians that are both faint and isolated. Either test alone removes real geometry (blank walls are isolated, glass is faint).
