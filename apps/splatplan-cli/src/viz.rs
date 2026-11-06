//! PNG visualizations: distance-field slices with planned paths on top.

use std::path::Path;

use anyhow::{Context, Result};
use gs_map::Map;
use image::{Rgb, RgbImage};
use splat_types::Vec3;

/// Pixels per voxel in the output image.
const ZOOM: u32 = 2;

/// Distance → color: dark blue (close) to yellow (≥ `max_d`).
fn ramp(d: f32, max_d: f32) -> Rgb<u8> {
    let t = (d / max_d).clamp(0.0, 1.0);
    let r = (40.0 + 215.0 * t) as u8;
    let g = (40.0 + 200.0 * t) as u8;
    let b = (140.0 - 100.0 * t) as u8;
    Rgb([r, g, b])
}

/// Writes a horizontal slice of the map at height `z`. Black = occupied,
/// gray = unknown, blue→yellow = distance to the nearest non-free voxel.
/// Each path is drawn in red (start green, goal magenta) as projected onto the slice.
pub fn write_slice(map: &Map, z: f64, paths: &[Vec<Vec3>], out: &Path) -> Result<()> {
    let Some((nx, ny, vals)) = map.slice_z(z) else {
        anyhow::bail!("height {z} is outside the map");
    };
    let max_d = (2.0 * map.length_unit) as f32;
    let (w, h) = (nx as u32 * ZOOM, ny as u32 * ZOOM);
    let mut img = RgbImage::new(w, h);
    for j in 0..ny {
        for i in 0..nx {
            let v = vals[j * nx + i];
            let c = if v == -1.0 {
                Rgb([0, 0, 0])
            } else if v == -2.0 {
                Rgb([110, 110, 110])
            } else {
                ramp(v, max_d)
            };
            // Image rows go down; flip so +y is up.
            for dy in 0..ZOOM {
                for dx in 0..ZOOM {
                    img.put_pixel(i as u32 * ZOOM + dx, h - 1 - (j as u32 * ZOOM + dy), c);
                }
            }
        }
    }
    let to_px = |p: &Vec3| -> Option<(u32, u32)> {
        let q = map.spec.continuous(p);
        let x = q.x * ZOOM as f64;
        let y = h as f64 - 1.0 - q.y * ZOOM as f64;
        (x >= 0.0 && y >= 0.0 && x < w as f64 && y < h as f64).then_some((x as u32, y as u32))
    };
    for path in paths {
        for p in path {
            if let Some((x, y)) = to_px(p) {
                img.put_pixel(x, y, Rgb([230, 30, 30]));
            }
        }
        for (p, color) in [
            (path.first(), Rgb([30, 220, 30])),
            (path.last(), Rgb([220, 30, 220])),
        ] {
            if let Some((x, y)) = p.and_then(to_px) {
                for dx in 0..5u32 {
                    for dy in 0..5u32 {
                        let (px, py) = ((x + dx).saturating_sub(2), (y + dy).saturating_sub(2));
                        if px < w && py < h {
                            img.put_pixel(px, py, color);
                        }
                    }
                }
            }
        }
    }
    img.save(out)
        .with_context(|| format!("writing {}", out.display()))
}
