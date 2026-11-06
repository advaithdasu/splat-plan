//! `frames select <in_dir> <out_dir> --target N`: copy the sharpest,
//! motion-spaced frames from a directory of extracted video frames.
#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use frames::{Gray, laplacian_variance, motion_profile, select_keyframes};
use rayon::prelude::*;

/// Keyframe selection for drone video frames.
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Select keyframes from `input` (jpg/png) and copy them into `output`.
    Select {
        /// Directory of extracted frames, in video order by file name.
        input: PathBuf,
        /// Output directory (created; existing files with the same names are overwritten).
        output: PathBuf,
        /// Number of keyframes to keep.
        #[arg(long, default_value_t = 200)]
        target: usize,
        /// Discard frames whose sharpness is below this fraction of the median.
        #[arg(long, default_value_t = 0.5)]
        min_rel_sharpness: f64,
    },
}

fn load_gray(path: &Path) -> Result<Gray> {
    let img = image::open(path)
        .with_context(|| format!("reading {}", path.display()))?
        .to_luma8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let full = Gray {
        width: w,
        height: h,
        data: img
            .into_raw()
            .into_iter()
            .map(|v| v as f32 / 255.0)
            .collect(),
    };
    // Score at ≤ 640 px wide: comparable across resolutions, and fast.
    Ok(full.thumbnail(640))
}

fn main() -> Result<()> {
    let Command::Select {
        input,
        output,
        target,
        min_rel_sharpness,
    } = Cli::parse().command;
    let mut files: Vec<PathBuf> = std::fs::read_dir(&input)
        .with_context(|| format!("listing {}", input.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png"))
        })
        .collect();
    files.sort();
    if files.is_empty() {
        bail!("no .jpg/.png frames in {}", input.display());
    }
    // Score each frame and keep only a 64-px thumbnail, so memory stays
    // small even for thousands of frames.
    let scored: Vec<(f64, Gray)> = files
        .par_iter()
        .map(|f| {
            let g = load_gray(f)?;
            Ok((laplacian_variance(&g), g.thumbnail(64)))
        })
        .collect::<Result<_>>()?;
    let (sharp, thumbs): (Vec<f64>, Vec<Gray>) = scored.into_iter().unzip();
    let motion = motion_profile(&thumbs);
    let sel = select_keyframes(&sharp, &motion, target, min_rel_sharpness);
    std::fs::create_dir_all(&output)?;
    for &i in &sel.chosen {
        let name = files[i].file_name().context("frame without file name")?;
        std::fs::copy(&files[i], output.join(name))?;
    }
    std::fs::write(
        output.join("selection.json"),
        serde_json::to_string_pretty(&sel)?,
    )?;
    println!(
        "selected {} of {} frames ({} rejected as blurry) -> {}",
        sel.chosen.len(),
        files.len(),
        sel.rejected_blurry,
        output.display()
    );
    Ok(())
}
