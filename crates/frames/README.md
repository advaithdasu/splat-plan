# frames

Picks a few hundred sharp, well-spaced keyframes from a drone video.

`frames select frames_all/ frames/ --target 200`: sharpness is the variance of the Laplacian; motion is the difference between consecutive thumbnails; the video is split into equal-motion windows and the sharpest frame in each wins.
