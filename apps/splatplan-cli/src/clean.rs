//! `splatplan clean`: prune floaters into a new splat file, and record what
//! was removed.
//!
//! The cleaned splat is a modeling decision, so it is never silent: the
//! record (`splat/clean.json`) names the input file and its hash, the rule,
//! and the counts, and every report and certificate made from the cleaned
//! file says so. The verifier still checks the raw Gaussians *of that file*;
//! cleaning only decides which file that is.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use splat_types::{Report, Scale};

use crate::capture::Capture;
use crate::provenance::Provenance;

/// Raw splat, relative to the capture directory.
pub const RAW_PLY: &str = "splat/point_cloud.ply";
/// Cleaned splat, relative to the capture directory.
pub const CLEAN_PLY: &str = "splat/point_cloud.clean.ply";
/// Cleaning record, relative to the capture directory.
pub const CLEAN_JSON: &str = "splat/clean.json";

/// Cleaning settings (lengths in meters for metric captures).
///
/// A Gaussian is removed only if it is both **isolated** (fewer than
/// `min_neighbors` other means within `radius_m`) and **faint** (opacity
/// below `max_opacity`). Either test alone removes real geometry: blank
/// walls are a few large opaque Gaussians far apart (isolation alone
/// removed ~95% of the occupied map on real captures), and glass is many
/// faint Gaussians close together.
///
/// The default `max_opacity` is the map's density threshold (0.2): a lone
/// Gaussian that faint can't occupy a voxel on its own, but the verifier
/// would still treat it as solid.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CleanConfig {
    /// Neighborhood radius (m).
    pub radius_m: f64,
    /// A Gaussian with fewer than this many other means within the radius is isolated.
    pub min_neighbors: usize,
    /// Only isolated Gaussians with opacity below this are removed.
    pub max_opacity: f32,
}

impl Default for CleanConfig {
    fn default() -> Self {
        CleanConfig {
            radius_m: 0.10,
            min_neighbors: 10,
            max_opacity: 0.2,
        }
    }
}

/// `splat/clean.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanRecord {
    /// Input splat (relative to the capture).
    pub input: String,
    /// FNV-1a 64 hash of the input file's bytes (hex). A mismatch means the
    /// splat was retrained after cleaning and the cleaned file is stale.
    pub input_hash: String,
    /// Output splat (relative to the capture).
    pub output: String,
    /// The rule, in words.
    pub method: String,
    /// Settings.
    pub config: CleanConfig,
    /// The radius in the splat file's own units.
    pub radius_units: f64,
    /// False if the capture is unscaled (the radius then uses a guessed unit).
    pub metric: bool,
    /// Gaussians in the input.
    pub input_gaussians: usize,
    /// Gaussians removed.
    pub removed: usize,
    /// Gaussians written.
    pub kept: usize,
    /// How the record was produced.
    pub provenance: Provenance,
}

impl Report for CleanRecord {}

impl CleanRecord {
    /// One line for reports and certificates.
    pub fn summary(&self) -> String {
        format!(
            "{} (cleaned from {}: {} of {} Gaussians removed, {})",
            self.output, self.input, self.removed, self.input_gaussians, self.method
        )
    }
}

/// Stable FNV-1a 64 hash of a file, as hex.
pub fn file_hash(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    Ok(format!("{h:016x}"))
}

/// Cleans the raw splat of the capture in `dir` and writes the cleaned file
/// and its record. Always reads the raw splat, never an earlier cleaned one.
pub fn run(dir: &Path, cfg: &CleanConfig) -> Result<CleanRecord> {
    if !(cfg.radius_m > 0.0 && cfg.radius_m.is_finite()) {
        bail!("radius must be > 0, got {}", cfg.radius_m);
    }
    if !(0.0..=1.0).contains(&cfg.max_opacity) {
        bail!("max opacity must be in [0, 1], got {}", cfg.max_opacity);
    }
    let capture = Capture::open(dir, true)?;
    let (aligned, align) = capture.load_aligned()?;
    let metric = aligned.scale.is_metric();
    // Neighbor counts are invariant under the similarity, so count in the
    // aligned frame (meters, or guessed meters) and apply the mask to the
    // raw file's Gaussians, which are in the same order.
    let unit = match aligned.scale {
        Scale::Metric(_) => 1.0,
        Scale::Unscaled => gs_map::length_unit_guess(&aligned),
    };
    let isolated_ok =
        gs_io::isolation_mask(&aligned.gaussians, cfg.radius_m * unit, cfg.min_neighbors);
    let keep: Vec<bool> = isolated_ok
        .iter()
        .zip(&aligned.gaussians)
        .map(|(ok, g)| *ok || g.opacity >= cfg.max_opacity)
        .collect();
    let raw = gs_io::load_ply(&capture.ply)
        .with_context(|| format!("loading {}", capture.ply.display()))?;
    if raw.len() != keep.len() {
        bail!("splat changed while cleaning");
    }
    let kept: Vec<_> = raw
        .iter()
        .zip(&keep)
        .filter(|(_, k)| **k)
        .map(|(g, _)| *g)
        .collect();
    gs_io::write_ply(dir.join(CLEAN_PLY), &kept)
        .with_context(|| format!("writing {}", dir.join(CLEAN_PLY).display()))?;
    let record = CleanRecord {
        input: RAW_PLY.into(),
        input_hash: file_hash(&capture.ply)?,
        output: CLEAN_PLY.into(),
        method: format!(
            "dropped Gaussians with opacity below {} and fewer than {} other means within {:.3} {}",
            cfg.max_opacity,
            cfg.min_neighbors,
            cfg.radius_m,
            if metric { "m" } else { "guessed m (UNSCALED)" }
        ),
        config: *cfg,
        radius_units: cfg.radius_m * unit / align.similarity.scale,
        metric,
        input_gaussians: raw.len(),
        removed: raw.len() - kept.len(),
        kept: kept.len(),
        provenance: Provenance::capture(&format!(
            "splatplan clean {} --radius {} --min-neighbors {} --max-opacity {}",
            dir.display(),
            cfg.radius_m,
            cfg.min_neighbors,
            cfg.max_opacity
        )),
    };
    std::fs::write(dir.join(CLEAN_JSON), record.to_json()?)?;
    Ok(record)
}
