# synth

Synthetic scenes with **exact ground truth**, the backbone of every safety claim.

- `Primitive`: box, cylinder, ground, sealed hollow box, ellipsoid, each with an exact distance function.
- `car_scene`: body, cabin, four wheels and ground, all represented by Gaussians; optional sealed pocket.
- `random_field`: random ellipsoids.
- `SynthScene::true_clearance(p)`: the oracle.
- `orbit_rig`, `export_capture` (writes a capture directory in a random COLMAP-like frame).
