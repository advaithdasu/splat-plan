# gs-verify

The independent judge. It depends on nothing in the workspace except `splat-types`, and a test enforces this.

It ignores the voxel map entirely and re-derives safety from the raw Gaussians:

- each Gaussian above an opacity floor is a solid 3σ ellipsoid;
- a provable lower bound on the distance to an ellipsoid (`bound.rs` has the proof) is minimized with a BVH;
- samples every `dt` need `r + margin + v·dt/2`; every sample must be near a training camera (support); speed and acceleration within limits.
- **line of sight** (re-derived without voxels): the robot body at every sample (center plus six points at `robot_radius`) must be seen by at least `min_views` training cameras, meaning inside the image with no ellipsoid on the segment from the camera (`Bvh::segment_blocked`, exact segment–ellipsoid tests). The map proves "observed" by voxel ray casting; this check proves it independently, so a ray-casting bug in `gs-map` shows up as `unobserved` violations (test `planner/tests/cross_check.rs`).

Output: `Certificate` with verdict `SAFE`, `UNSAFE` or `UNSCALED` (never `SAFE` without metric scale). The caller records which splat file the Gaussians came from (raw, or cleaned by `splatplan clean`) in the certificate's `splat` field.
