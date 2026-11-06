# gs-render-cpu

A CPU reference splat rasterizer, written for clarity.

EWA projection (`Σ' = J W Σ Wᵀ Jᵀ` + 0.3 px²), 16×16 tiles, per-tile depth sort, front-to-back alpha compositing, expected depth. Also `camera_path.json` export for flythroughs and a PSNR helper for checking renders against training photos.
