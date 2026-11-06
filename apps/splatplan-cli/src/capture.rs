//! Loading a capture directory (`data/captures/<name>/`) and aligning it.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use gs_align::{AlignConfig, AlignReport, Annotations, ScaleFile, ScaleSource, align_scene};
use splat_types::{Scale, Scene};

use crate::clean::{CLEAN_JSON, CleanRecord, RAW_PLY, file_hash};

/// Optional per-capture ground correction.
pub const GROUND_JSON: &str = "ground.json";

/// `ground.json`: where the real floor is relative to the fitted ground plane.
/// The fit picks the largest flat surface below the cameras, which can be the
/// desk tops when the floor is plain carpet that barely reconstructs.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct GroundFile {
    /// How far the floor is below the fitted plane (m).
    pub floor_below_fit_m: f64,
    /// What the fitted plane is, and how the offset was measured.
    #[serde(default)]
    pub reason: String,
}

/// Files of a capture, resolved.
pub struct Capture {
    /// Capture name (directory name).
    pub name: String,
    /// Root directory.
    pub dir: PathBuf,
    /// The splat that is loaded: the cleaned one if `splatplan clean` was
    /// run (and the raw splat has not changed since), else the raw one.
    pub ply: PathBuf,
    /// The cleaning record, when `ply` is the cleaned splat.
    pub cleaning: Option<CleanRecord>,
    /// COLMAP text model directory.
    pub colmap: PathBuf,
}

impl Capture {
    /// Resolves the standard layout, accepting `colmap/` or `colmap/sparse/0/`.
    ///
    /// Uses the cleaned splat when `splat/clean.json` exists, unless
    /// `raw_splat` is set. A cleaned splat made from a different raw splat
    /// (retrained since) is an error rather than a silent fallback.
    pub fn open(dir: &Path, raw_splat: bool) -> Result<Self> {
        let name = dir
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "capture".into());
        let raw = dir.join(RAW_PLY);
        if !raw.exists() {
            bail!(
                "{} not found (train a splat first: see data/README.md)",
                raw.display()
            );
        }
        let mut colmap = dir.join("colmap");
        if !colmap.join("cameras.txt").exists() && colmap.join("sparse/0/cameras.txt").exists() {
            colmap = colmap.join("sparse/0");
        }
        if !colmap.join("cameras.txt").exists() {
            bail!("no COLMAP text model in {}", dir.join("colmap").display());
        }
        let record_path = dir.join(CLEAN_JSON);
        let (ply, cleaning) = if !raw_splat && record_path.exists() {
            let rec: CleanRecord = serde_json::from_str(&std::fs::read_to_string(&record_path)?)
                .with_context(|| format!("parsing {}", record_path.display()))?;
            if file_hash(&raw)? != rec.input_hash {
                bail!(
                    "{} changed since `splatplan clean` ran; run it again (or pass --raw-splat)",
                    raw.display()
                );
            }
            let out = dir.join(&rec.output);
            if !out.exists() {
                bail!(
                    "{} is missing; run `splatplan clean` again (or pass --raw-splat)",
                    out.display()
                );
            }
            (out, Some(rec))
        } else {
            (raw, None)
        };
        Ok(Capture {
            name,
            dir: dir.to_path_buf(),
            ply,
            cleaning,
            colmap,
        })
    }

    /// Which splat file is used, in one line (for reports and certificates).
    pub fn splat_label(&self) -> String {
        match &self.cleaning {
            Some(rec) => rec.summary(),
            None => format!("{RAW_PLY} (raw, not cleaned)"),
        }
    }

    /// Scale source: `scale.json` wins over `annotations.json`; neither means UNSCALED.
    pub fn scale_source(&self) -> Result<ScaleSource> {
        let scale = self.dir.join("scale.json");
        if scale.exists() {
            let s: ScaleFile = serde_json::from_str(&std::fs::read_to_string(&scale)?)
                .with_context(|| format!("parsing {}", scale.display()))?;
            return Ok(ScaleSource::Manual(s.metric_scale));
        }
        let ann = self.dir.join("annotations.json");
        if ann.exists() {
            let a: Annotations = serde_json::from_str(&std::fs::read_to_string(&ann)?)
                .with_context(|| format!("parsing {}", ann.display()))?;
            return Ok(ScaleSource::Annotations(a));
        }
        Ok(ScaleSource::None)
    }

    /// The optional ground correction, `ground.json`.
    pub fn ground_file(&self) -> Result<Option<GroundFile>> {
        let path = self.dir.join(GROUND_JSON);
        if !path.exists() {
            return Ok(None);
        }
        let g = serde_json::from_str(&std::fs::read_to_string(&path)?)
            .with_context(|| format!("parsing {}", path.display()))?;
        Ok(Some(g))
    }

    /// Loads the raw (COLMAP-frame, unscaled) scene.
    pub fn load_raw(&self) -> Result<Scene> {
        let gaussians = gs_io::load_ply(&self.ply)
            .with_context(|| format!("loading {}", self.ply.display()))?;
        let model = colmap_io::load_model(&self.colmap)
            .with_context(|| format!("loading {}", self.colmap.display()))?;
        Ok(Scene {
            gaussians,
            cameras: model.poses(),
            scale: Scale::Unscaled,
        })
    }

    /// Loads and aligns (gravity, ground at z = 0, metric if a scale is known).
    pub fn load_aligned(&self) -> Result<(Scene, AlignReport)> {
        let raw = self.load_raw()?;
        let source = self.scale_source()?;
        let (mut scene, mut report) = align_scene(&raw, &AlignConfig::default(), &source)?;
        if let Some(g) = self.ground_file()? {
            if !scene.scale.is_metric() {
                bail!("{GROUND_JSON} needs a metric scale (its offset is in meters)");
            }
            if !(g.floor_below_fit_m.is_finite() && g.floor_below_fit_m.abs() < 5.0) {
                bail!("{GROUND_JSON}: floor_below_fit_m must be a distance in meters");
            }
            // One similarity, applied once to the raw scene, so Gaussians and
            // cameras stay in the same frame. Raising everything by the offset
            // puts the floor at z = 0.
            let mut sim = report.similarity;
            sim.trans.z += g.floor_below_fit_m;
            scene = raw.transformed(&sim, scene.scale);
            report.similarity = sim;
            report.ground_correction = Some(format!(
                "{GROUND_JSON}: the fitted plane is {} m above the floor ({})",
                g.floor_below_fit_m, g.reason
            ));
        }
        if !scene.scale.is_metric() {
            eprintln!("\n{}\n", Scale::UNSCALED_BANNER);
        }
        Ok((scene, report))
    }
}
