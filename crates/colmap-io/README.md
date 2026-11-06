# colmap-io

Parses and writes the COLMAP **text** model (`cameras.txt`, `images.txt`, `points3D.txt`).

Only undistorted `PINHOLE` / `SIMPLE_PINHOLE` cameras are accepted; any distortion model is an explicit error telling you to run `colmap image_undistorter`. The parser keeps empty 2D-point lines in `images.txt` (a common source of off-by-one bugs).
