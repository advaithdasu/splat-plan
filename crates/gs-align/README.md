# gs-align

Turns COLMAP's arbitrary frame into "+Z up, ground at z = 0, meters".

1. Up prior = mean of the cameras' up vectors (a drone keeps its camera roughly level).
2. RANSAC plane over the Gaussian means, accepting only normals within 30° of the prior and planes below most cameras (so a ceiling can't win); least-squares refit.
3. Scale from `scale.json` or from tape-marker clicks (`annotations.json`) triangulated with the camera poses.
4. One `Similarity` applied to Gaussians and cameras together.

Without a scale the output is `Scale::Unscaled` and every downstream result says so.
