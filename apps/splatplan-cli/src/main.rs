//! `splatplan`: command-line front end for the SplatPlan pipeline.
//!
//! Every subcommand that reads a capture takes the capture directory
//! (`data/captures/<name>/`, layout in `data/README.md`).
#![forbid(unsafe_code)]

mod capture;
mod clean;
mod export;
mod flight;
mod pipeline;
mod provenance;
mod render;
mod replan;
mod showcase;
mod viz;
mod weave;

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use gs_map::{MapConfig, build_map};
use planner::plan_and_certify;
use splat_types::{PolyTrajectory, Report, Vec3};

use crate::capture::Capture;

/// SplatPlan: drone video to Gaussian splats, recovered flight paths and verified planned paths.
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
    /// Use the raw splat even if `splatplan clean` has written a cleaned one.
    #[arg(long, global = true)]
    raw_splat: bool,
    /// Drone radius in meters, for both planner and verifier (default 0.15).
    #[arg(long, global = true)]
    robot_radius: Option<f64>,
    /// Safety margin in meters on top of the radius (default 0.10).
    #[arg(long, global = true)]
    margin: Option<f64>,
}

/// Which synthetic scene to generate.
#[derive(Clone, Copy, ValueEnum)]
enum SynthKind {
    /// Synthetic car on a ground plane.
    Car,
    /// Car plus a sealed hollow box (an unobservable cavity).
    CarPocket,
    /// Random ellipsoids over a ground plane.
    Field,
}

#[derive(Subcommand)]
enum Command {
    /// Print a summary of a 3DGS `.ply` file (count, bounds, scales, opacity histogram).
    Inspect {
        /// Path to the `.ply` file.
        ply: PathBuf,
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
    },
    /// Write a synthetic capture (splat + COLMAP model) in a random "COLMAP" frame.
    Synth {
        /// Scene to generate.
        kind: SynthKind,
        /// Output capture directory.
        #[arg(long)]
        out: PathBuf,
        /// Seed for the random frame (and the field scene).
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Also write scale.json with the true metric scale.
        #[arg(long)]
        write_scale: bool,
    },
    /// Prune floaters (isolated, faint Gaussians) into
    /// splat/point_cloud.clean.ply and record what was removed in
    /// splat/clean.json. Later commands use the cleaned splat and say so in
    /// their reports and certificates.
    Clean {
        /// Capture directory.
        capture: PathBuf,
        /// Neighborhood radius (m).
        #[arg(long, default_value_t = 0.10)]
        radius: f64,
        /// A Gaussian with fewer than this many others within the radius is isolated.
        #[arg(long, default_value_t = 10)]
        min_neighbors: usize,
        /// Only isolated Gaussians fainter than this are removed.
        #[arg(long, default_value_t = 0.2)]
        max_opacity: f32,
    },
    /// Align a capture (gravity, ground at z = 0, metric scale) and print the report.
    Align {
        /// Capture directory.
        capture: PathBuf,
    },
    /// Build the collision map and write a report plus distance-field slices.
    Map {
        /// Capture directory.
        capture: PathBuf,
        /// Output directory.
        #[arg(long, default_value = "out/map")]
        out: PathBuf,
        /// Voxel size (m).
        #[arg(long, default_value_t = 0.05)]
        voxel: f64,
        /// Density threshold for "occupied".
        #[arg(long, default_value_t = 0.2)]
        threshold: f32,
    },
    /// Plan and certify one trajectory between two points (aligned frame).
    Plan {
        /// Capture directory.
        capture: PathBuf,
        /// Start as x,y,z (meters in the aligned frame).
        #[arg(long, value_parser = parse_vec3, allow_hyphen_values = true)]
        start: Vec3,
        /// Goal as x,y,z.
        #[arg(long, value_parser = parse_vec3, allow_hyphen_values = true)]
        goal: Vec3,
        /// Output directory.
        #[arg(long, default_value = "out/plan")]
        out: PathBuf,
    },
    /// Re-verify a saved trajectory against the raw Gaussians of a capture.
    Verify {
        /// Capture directory.
        capture: PathBuf,
        /// trajectory.json written by `plan` or `pipeline`.
        trajectory: PathBuf,
    },
    /// Render a flythrough along a trajectory (PNG frames + camera_path.json).
    Render {
        /// Capture directory.
        capture: PathBuf,
        /// trajectory.json (aligned frame).
        #[arg(long)]
        trajectory: PathBuf,
        /// Output directory (frames/ and camera_path.json).
        #[arg(long, default_value = "out/render")]
        out: PathBuf,
        /// Frames per second.
        #[arg(long, default_value_t = 30.0)]
        fps: f64,
        /// Output width in pixels (intrinsics of the first training camera, rescaled).
        #[arg(long, default_value_t = 640)]
        width: u32,
        /// Keep this point (x,y,z, aligned frame) centered instead of looking along the path.
        #[arg(long, value_parser = parse_vec3, allow_hyphen_values = true)]
        look_at: Option<Vec3>,
    },
    /// Renderer check: PSNR of renders vs. the (undistorted) training photos.
    Psnr {
        /// Capture directory.
        capture: PathBuf,
        /// Directory with the undistorted training images.
        #[arg(long)]
        images: PathBuf,
        /// Maximum number of images to compare.
        #[arg(long, default_value_t = 20)]
        max: usize,
    },
    /// Certified standoff orbit around the largest object (optionally rendered).
    Orbit {
        /// Capture directory.
        capture: PathBuf,
        /// Standoff from the object's surface (m).
        #[arg(long, default_value_t = 1.5)]
        standoff: f64,
        /// Loop heights above the ground (m), comma-separated.
        #[arg(long, value_delimiter = ',', default_value = "1.2,2.2")]
        heights: Vec<f64>,
        /// Output directory.
        #[arg(long, default_value = "out/orbit")]
        out: PathBuf,
        /// Also render a flythrough looking at the object (frames at this fps).
        #[arg(long)]
        render_fps: Option<f64>,
    },
    /// Package a capture for the web viewer in web/: the aligned splat
    /// (cropped for display), its recovered flight path, and up to 5
    /// certified paths, re-verified on export.
    Export {
        /// Capture directory.
        capture: PathBuf,
        /// Output directory (e.g. web/data/<name>).
        #[arg(long)]
        out: PathBuf,
        /// Output with certified/ paths (e.g. reports/<name>/replan); omit for a splat-only scene.
        #[arg(long, value_delimiter = ',')]
        paths: Vec<PathBuf>,
        /// Only these certified path ids, comma-separated (e.g. 001,004).
        #[arg(long, value_delimiter = ',')]
        only: Vec<String>,
        /// Keep Gaussians within this horizontal distance (m) of a camera.
        #[arg(long, default_value_t = 2.5)]
        crop: f64,
        /// Title shown in the viewer.
        #[arg(long, default_value = "")]
        title: String,
        /// Output of `splatplan flight` (directory with flight.json) to include.
        #[arg(long)]
        flight: Option<PathBuf>,
        /// Keep this many paths (at most 5), ranked by length x (1 + obstacles avoided).
        #[arg(long, default_value_t = export::MAX_SHOWN_PATHS)]
        max_paths: usize,
    },
    /// Recover the camera's real flight path from every video frame
    /// (after scripts/register_all.sh), in the splat's aligned frame.
    Flight {
        /// Capture directory.
        capture: PathBuf,
        /// Text model with all registered frames (default: <capture>/work/flight/model_txt).
        #[arg(long)]
        model: Option<PathBuf>,
        /// Output directory (default: reports/<capture name>/flight).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Frame rate of frames_all/.
        #[arg(long, default_value_t = 10.0)]
        fps: f64,
        /// Fastest believable camera speed (m/s); faster jumps are registration errors.
        #[arg(long, default_value_t = 2.0)]
        max_speed: f64,
        /// Position noise allowed on top of the speed limit (m).
        #[arg(long, default_value_t = 0.2)]
        slack: f64,
        /// A longer stretch without a pose is shown as a gap (s).
        #[arg(long, default_value_t = 0.5)]
        gap: f64,
        /// Width (px) of the splat renders compared with each video frame; 0 skips the check.
        #[arg(long, default_value_t = 192)]
        check_width: u32,
        /// Drop frames whose render correlates with the video frame less than this.
        #[arg(long, default_value_t = 0.6)]
        min_match: f64,
    },
    /// Align, map, plan random queries and verify them; write reports.
    Pipeline {
        /// Capture directory.
        capture: PathBuf,
        /// Output directory (default: reports/<capture name>).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Number of random start/goal queries.
        #[arg(long, default_value_t = 50)]
        queries: usize,
        /// Seed for the queries.
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Pick a few long certified paths for display (the pipeline's 50 queries are a benchmark).
    Showcase {
        /// Capture directory.
        capture: PathBuf,
        /// Output directory (default: reports/<capture name>/showcase).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Paths to keep.
        #[arg(long, default_value_t = 6)]
        count: usize,
        /// Start/goal pairs to plan and verify.
        #[arg(long, default_value_t = 300)]
        tries: usize,
        /// Smallest straight-line distance between start and goal (m).
        #[arg(long, default_value_t = 3.0)]
        min_dist: f64,
        /// Keep endpoints and paths inside xmin,ymin,xmax,ymax (m, aligned frame).
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        region: Option<Vec<f64>>,
        /// Endpoint height band lo,hi (m above the floor); default: the benchmark's 0.5,2.5.
        #[arg(long, value_delimiter = ',')]
        height: Option<Vec<f64>>,
        /// Smallest length (m) of the straight line that must run through solid voxels.
        #[arg(long, default_value_t = 0.05)]
        min_blocked: f64,
        /// Ranking: `length` (length × detour) or `avoidance` (how far the path swings around what blocks the straight line).
        #[arg(long, value_enum, default_value_t = showcase::Rank::Length)]
        rank: showcase::Rank,
        /// Paths whose starts and goals are both this close (m) count as duplicates.
        #[arg(long, default_value_t = 1.5)]
        dedup: f64,
        /// Seed for endpoint and pair sampling.
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Certified paths between places the drone really was: the start and
    /// end of its recovered flight and every spot it hovered.
    Replan {
        /// Capture directory.
        capture: PathBuf,
        /// Output of `splatplan flight` (default: reports/<capture name>/flight).
        #[arg(long)]
        flight: Option<PathBuf>,
        /// Output directory (default: reports/<capture name>/replan).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Paths to keep (1 to 5).
        #[arg(long, default_value_t = 5)]
        count: usize,
        /// A hover: the drone stays within this distance (m)...
        #[arg(long, default_value_t = 0.35)]
        hover_radius: f64,
        /// ...for at least this long (s).
        #[arg(long, default_value_t = 1.5)]
        hover: f64,
        /// Points closer than this (m) to an earlier one count as the same place.
        #[arg(long, default_value_t = 0.75)]
        merge: f64,
        /// Farthest a real point may be moved (straight up first) to a valid endpoint (m).
        #[arg(long, default_value_t = 1.5)]
        max_snap: f64,
        /// Smallest straight-line distance between the ends of a path (m).
        #[arg(long, default_value_t = 2.0)]
        min_dist: f64,
    },
    /// Long certified tours that thread the gaps between obstacles, kept low so the drone goes around things, not over them.
    Weave {
        /// Capture directory.
        capture: PathBuf,
        /// Output directory (default: reports/<capture name>/weave).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Tours to keep.
        #[arg(long, default_value_t = 4)]
        count: usize,
        /// Highest the drone's center may fly (m above the floor).
        #[arg(long, default_value_t = 1.5)]
        ceiling: f64,
        /// Lowest the gates may be (m above the floor).
        #[arg(long, default_value_t = 0.8)]
        floor: f64,
        /// Anything solid between this height (m) and the ceiling is an obstacle the drone goes around, never over.
        #[arg(long, default_value_t = 0.5)]
        min_height: f64,
        /// A tour stops growing at this length (m).
        #[arg(long, default_value_t = 30.0)]
        max_length: f64,
        /// Longest leg between consecutive gates (m, through free space).
        #[arg(long, default_value_t = 4.0)]
        max_leg: f64,
        /// Certified tours shorter than this (m) are not kept.
        #[arg(long, default_value_t = 5.0)]
        min_length: f64,
        /// Solid areas smaller than this (m², from above) are noise, not obstacles.
        #[arg(long, default_value_t = 0.05)]
        min_obstacle: f64,
        /// Tours kept per step of the search.
        #[arg(long, default_value_t = 300)]
        beam: usize,
        /// Best varied tours handed to the planner and verifier.
        #[arg(long, default_value_t = 24)]
        candidates: usize,
    },
}

fn parse_vec3(s: &str) -> Result<Vec3, String> {
    let v: Vec<f64> = s
        .split(',')
        .map(|t| t.trim().parse::<f64>().map_err(|e| format!("`{t}`: {e}")))
        .collect::<Result<_, _>>()?;
    match v.as_slice() {
        [x, y, z] => Ok(Vec3::new(*x, *y, *z)),
        _ => Err(format!("expected x,y,z, got `{s}`")),
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();
    let cli = Cli::parse();
    let raw = cli.raw_splat;
    let robot = pipeline::Robot {
        radius: cli.robot_radius,
        margin: cli.margin,
    };
    match cli.command {
        Command::Inspect { ply, json } => {
            let gs = gs_io::load_ply(&ply).with_context(|| format!("loading {}", ply.display()))?;
            let report = gs_io::inspect(&gs);
            if json {
                print!("{}", report.to_json()?);
            } else {
                print!("{report}");
            }
        }
        Command::Synth {
            kind,
            out,
            seed,
            write_scale,
        } => {
            let scene = match kind {
                SynthKind::Car => synth::car_scene(&synth::CarConfig::default()),
                SynthKind::CarPocket => synth::car_scene(&synth::CarConfig {
                    sealed_pocket: true,
                    ..synth::CarConfig::default()
                }),
                SynthKind::Field => synth::random_field(seed, 60),
            };
            let sim = synth::random_similarity(seed);
            synth::export_capture(&scene, &out, &sim, write_scale)?;
            println!(
                "wrote {} ({} Gaussians, {} cameras){}",
                out.display(),
                scene.gaussians.len(),
                scene.cameras.len(),
                if write_scale {
                    ", with scale.json"
                } else {
                    ", UNSCALED (no scale.json)"
                }
            );
        }
        Command::Clean {
            capture,
            radius,
            min_neighbors,
            max_opacity,
        } => {
            let rec = clean::run(
                &capture,
                &clean::CleanConfig {
                    radius_m: radius,
                    min_neighbors,
                    max_opacity,
                },
            )?;
            println!(
                "removed {} of {} Gaussians ({:.1}%): {}",
                rec.removed,
                rec.input_gaussians,
                100.0 * rec.removed as f64 / rec.input_gaussians.max(1) as f64,
                rec.method
            );
            println!(
                "wrote {} and {}",
                capture.join(clean::CLEAN_PLY).display(),
                capture.join(clean::CLEAN_JSON).display()
            );
        }
        Command::Align { capture } => {
            let (_, report) = Capture::open(&capture, raw)?.load_aligned()?;
            print!("{}", report.to_json()?);
        }
        Command::Map {
            capture,
            out,
            voxel,
            threshold,
        } => {
            let (scene, _) = Capture::open(&capture, raw)?.load_aligned()?;
            let cfg = MapConfig {
                voxel,
                density_threshold: threshold,
                ..MapConfig::default()
            };
            let (map, report) = build_map(&scene, &cfg)?;
            std::fs::create_dir_all(&out)?;
            std::fs::write(out.join("map.json"), report.to_json()?)?;
            for z in [0.5, 1.0, 1.5, 2.0] {
                let name = format!("slice_z{z:.1}m.png");
                viz::write_slice(&map, z * map.length_unit, &[], &out.join(name))?;
            }
            print!("{}", report.to_json()?);
        }
        Command::Plan {
            capture,
            start,
            goal,
            out,
        } => {
            let c = Capture::open(&capture, raw)?;
            let p = pipeline::prepare(&c, &MapConfig::default(), &robot)?;
            let u = p.map.length_unit;
            let mut o = plan_and_certify(
                &p.map,
                &p.verifier,
                &(start * u),
                &(goal * u),
                &robot.planner(),
            )
            .map_err(|e| anyhow::anyhow!("planner refused: {e}"))?;
            o.certificate.splat = Some(c.splat_label());
            std::fs::create_dir_all(&out)?;
            std::fs::write(
                out.join("trajectory.json"),
                serde_json::to_string_pretty(&o.trajectory)?,
            )?;
            std::fs::write(out.join("certificate.json"), o.certificate.to_json()?)?;
            let path: Vec<Vec3> = o.trajectory.sample(0.05).iter().map(|s| s.pos).collect();
            viz::write_slice(&p.map, start.z * u, &[path], &out.join("slice.png"))?;
            print!("{}", o.certificate.to_json()?);
        }
        Command::Verify {
            capture,
            trajectory,
        } => {
            let c = Capture::open(&capture, raw)?;
            let p = pipeline::prepare(&c, &MapConfig::default(), &robot)?;
            let t: PolyTrajectory = serde_json::from_str(&std::fs::read_to_string(&trajectory)?)?;
            let mut cert = p.verifier.certify(&t);
            cert.splat = Some(c.splat_label());
            print!("{}", cert.to_json()?);
            if cert.verdict != gs_verify::Verdict::Safe {
                bail!("verdict {:?}", cert.verdict);
            }
        }
        Command::Render {
            capture,
            trajectory,
            out,
            fps,
            width,
            look_at,
        } => {
            let (scene, _) = Capture::open(&capture, raw)?.load_aligned()?;
            let t: PolyTrajectory = serde_json::from_str(&std::fs::read_to_string(&trajectory)?)?;
            let path = render::flythrough(&scene, &t, &out, fps, width, look_at)?;
            println!(
                "rendered {} frames to {} (encode with scripts/encode.sh {} {fps})",
                path.frames.len(),
                out.join("frames").display(),
                out.display()
            );
        }
        Command::Psnr {
            capture,
            images,
            max,
        } => {
            // PSNR needs the photos' frame: use the raw (unaligned) scene,
            // whose cameras match the images exactly.
            let scene = Capture::open(&capture, raw)?.load_raw()?;
            let cmd = format!(
                "splatplan psnr {} --images {} --max {max}",
                capture.display(),
                images.display()
            );
            let r = render::psnr_check(&scene, &images, max, &cmd)?;
            print!("{}", r.to_json()?);
        }
        Command::Orbit {
            capture,
            standoff,
            heights,
            out,
            render_fps,
        } => {
            let c = Capture::open(&capture, raw)?;
            let p = pipeline::prepare(&c, &MapConfig::default(), &robot)?;
            let ocfg = planner::OrbitConfig {
                standoff,
                heights,
                ..planner::OrbitConfig::default()
            };
            let mut plan = planner::standoff_orbit(
                &p.map,
                &p.scene.gaussians,
                &p.scene.cameras,
                p.scene.scale,
                &robot.verifier(),
                &robot.planner(),
                &ocfg,
            )
            .map_err(|e| anyhow::anyhow!("orbit refused: {e}"))?;
            plan.certificate.splat = Some(c.splat_label());
            plan.object_certificate.splat = Some(c.splat_label());
            std::fs::create_dir_all(&out)?;
            std::fs::write(
                out.join("trajectory.json"),
                serde_json::to_string_pretty(&plan.trajectory)?,
            )?;
            std::fs::write(out.join("certificate.json"), plan.certificate.to_json()?)?;
            std::fs::write(
                out.join("object_certificate.json"),
                plan.object_certificate.to_json()?,
            )?;
            std::fs::write(
                out.join("skipped_rays.json"),
                serde_json::to_string_pretty(&plan.skipped)?,
            )?;
            let path: Vec<Vec3> = plan.trajectory.sample(0.05).iter().map(|s| s.pos).collect();
            let z = ocfg.heights.first().copied().unwrap_or(1.0) * p.map.length_unit;
            viz::write_slice(&p.map, z, &[path], &out.join("slice.png"))?;
            println!(
                "orbit: verdict {:?}, verified standoff {:.3} m (required {:.3}), {} rays skipped, {:.1} s",
                plan.certificate.verdict,
                plan.object_certificate.min_clearance_m,
                standoff - ocfg.tolerance,
                plan.skipped.len(),
                plan.trajectory.duration()
            );
            if let Some(fps) = render_fps {
                render::flythrough(
                    &p.scene,
                    &plan.trajectory,
                    &out,
                    fps,
                    640,
                    Some(plan.center),
                )?;
                println!(
                    "frames in {} (scripts/encode.sh {} {fps})",
                    out.join("frames").display(),
                    out.display()
                );
            }
        }
        Command::Export {
            capture,
            out,
            paths,
            only,
            crop,
            title,
            flight,
            max_paths,
        } => {
            let info = export::run(
                &capture,
                raw,
                &robot,
                &out,
                &export::ExportConfig {
                    crop_m: crop,
                    paths,
                    only,
                    title,
                    flight,
                    max_paths,
                },
            )?;
            println!(
                "exported {} of {} Gaussians and {} certified paths to {}",
                info.gaussians_exported,
                info.gaussians_total,
                info.paths,
                out.display()
            );
        }
        Command::Flight {
            capture,
            model,
            out,
            fps,
            max_speed,
            slack,
            gap,
            check_width,
            min_match,
        } => {
            let name = capture
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "capture".into());
            let out = out.unwrap_or_else(|| PathBuf::from("reports").join(&name).join("flight"));
            let report = flight::run(
                &capture,
                raw,
                &flight::FlightConfig {
                    model: model.unwrap_or_else(|| capture.join("work/flight/model_txt")),
                    fps,
                    max_speed,
                    slack_m: slack,
                    gap_s: gap,
                    check_width,
                    min_match,
                },
            )?;
            std::fs::create_dir_all(&out)?;
            std::fs::write(out.join(flight::FLIGHT_JSON), report.to_json()?)?;
            let gap_s: f64 = report.gaps.iter().map(|g| g[1] - g[0]).sum();
            let longest = report.gaps.iter().map(|g| g[1] - g[0]).fold(0.0, f64::max);
            println!(
                "{}: {} of {} video frames registered, {} kept ({} rejected: {} render mismatch, the rest impossible jumps)",
                report.name,
                report.registered,
                report.video_frames,
                report.kept,
                report.registered - report.kept,
                report.mismatched,
            );
            println!(
                "  covered {:.1} of {:.1} s; {} gaps totaling {:.1} s (longest {:.1} s); keyframe residual {:.2e} m",
                report.covered_s,
                report.duration_s,
                report.gaps.len(),
                gap_s,
                longest,
                report.keyframe_residual_m
            );
            println!(
                "  {:.1} m flown in {} segments; speed median {:.2} m/s, p95 {:.2} m/s; height {:.2} to {:.2} m",
                report.length_m,
                report.segments.len(),
                report.median_speed_mps,
                report.p95_speed_mps,
                report.z_range_m[0],
                report.z_range_m[1]
            );
            for r in &report.rejected {
                println!(
                    "  rejected {:.1}-{:.1} s ({}): {} frames, up to {:.1} m off the path, {} keyframes, match {:.2}",
                    r.t0, r.t1, r.reason, r.frames, r.max_offset_m, r.keyframes, r.median_match
                );
            }
            println!("wrote {}", out.join(flight::FLIGHT_JSON).display());
        }
        Command::Pipeline {
            capture,
            out,
            queries,
            seed,
        } => {
            let name = capture
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "capture".into());
            let out = out.unwrap_or_else(|| PathBuf::from("reports").join(name));
            pipeline::run(&capture, raw, &robot, &out, queries, seed)?;
        }
        Command::Showcase {
            capture,
            out,
            count,
            tries,
            min_dist,
            region,
            height,
            min_blocked,
            rank,
            dedup,
            seed,
        } => {
            let name = capture
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "capture".into());
            let out = out.unwrap_or_else(|| PathBuf::from("reports").join(name).join("showcase"));
            let region = match region.as_deref() {
                None => None,
                Some([x0, y0, x1, y1]) if x0 < x1 && y0 < y1 => Some([*x0, *y0, *x1, *y1]),
                Some(r) => bail!("--region wants xmin,ymin,xmax,ymax with min < max, got {r:?}"),
            };
            let height = match height.as_deref() {
                None => None,
                Some([lo, hi]) if lo < hi => Some([*lo, *hi]),
                Some(h) => bail!("--height wants lo,hi with lo < hi, got {h:?}"),
            };
            let cfg = showcase::ShowcaseConfig {
                count,
                tries,
                min_dist_m: min_dist,
                region,
                height,
                min_blocked_m: min_blocked,
                rank,
                dedup_m: dedup,
                seed,
            };
            showcase::run(&capture, raw, &robot, &out, &cfg)?;
        }
        Command::Replan {
            capture,
            flight,
            out,
            count,
            hover_radius,
            hover,
            merge,
            max_snap,
            min_dist,
        } => {
            let name = capture
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "capture".into());
            let base = PathBuf::from("reports").join(name);
            let cfg = replan::ReplanConfig {
                flight: flight.unwrap_or_else(|| base.join("flight")),
                count,
                hover_radius_m: hover_radius,
                hover_s: hover,
                merge_m: merge,
                max_snap_m: max_snap,
                min_dist_m: min_dist,
            };
            let out = out.unwrap_or_else(|| base.join("replan"));
            replan::run(&capture, raw, &robot, &out, &cfg)?;
        }
        Command::Weave {
            capture,
            out,
            count,
            ceiling,
            floor,
            min_height,
            max_length,
            max_leg,
            min_length,
            min_obstacle,
            beam,
            candidates,
        } => {
            let name = capture
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "capture".into());
            let out = out.unwrap_or_else(|| PathBuf::from("reports").join(name).join("weave"));
            let cfg = weave::WeaveConfig {
                count,
                ceiling_m: ceiling,
                floor_m: floor,
                min_height_m: min_height,
                max_length_m: max_length,
                max_leg_m: max_leg,
                min_length_m: min_length,
                min_obstacle_m2: min_obstacle,
                beam,
                candidates,
            };
            weave::run(&capture, raw, &robot, &out, &cfg)?;
        }
    }
    Ok(())
}
