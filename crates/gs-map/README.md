# gs-map

The collision map. Bounds are the 1st–99th percentile box of the Gaussian means, grown to include every camera, then padded.

1. **Density grid**: sum of Gaussian densities at voxel centers (z-slab parallel, deterministic; covariance widened by `(res/2)²` so thin discs are not lost).
2. **Occupancy**: density ≥ τ, then morphological closing (two distance transforms).
3. **Observed-free**: rays from every training camera (3D DDA) mark voxels they pass through; everything else stays **Unknown**, never free.
4. **ESDF**: exact Euclidean distance transform (Felzenszwalb) to the nearest voxel that is *not* observed-free.
5. **Queries**: `Map::clearance(p)` is a guaranteed lower bound; `Map::traversable(p, r)` is the planner's notion of safe.

Read `edt.rs` for the parabola-envelope trick, and `raycast.rs` for Amanatides–Woo.
