//! Parser and writer for the COLMAP **text** model.
//!
//! A COLMAP sparse model in text form is three files:
//!
//! - `cameras.txt`: `CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]`
//! - `images.txt`: two lines per image:
//!   `IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME`, then the 2D points line
//!   (which may be empty).
//! - `points3D.txt`: `POINT3D_ID X Y Z R G B ERROR TRACK[]`
//!
//! The pose in `images.txt` is **world-to-camera** (`X_cam = R·X_world + t`),
//! so the camera center is `C = -Rᵀ·t`; [`splat_types::CameraPose`] keeps that
//! convention. Only undistorted pinhole cameras are accepted (`PINHOLE`,
//! `SIMPLE_PINHOLE`): splat trainers expect undistorted images, and silently
//! ignoring distortion would put rays in the wrong place.
#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use splat_types::nalgebra::{Quaternion, UnitQuaternion};
use splat_types::{CameraPose, PinholeIntrinsics, Vec3};
use thiserror::Error;

/// Errors from reading or writing a COLMAP text model.
#[derive(Debug, Error)]
pub enum ColmapError {
    /// Underlying I/O failure (with the file involved).
    #[error("{path}: {source}")]
    Io {
        /// File path.
        path: String,
        /// The I/O error.
        source: std::io::Error,
    },
    /// A line could not be parsed.
    #[error("{file}:{line}: {msg}")]
    Parse {
        /// File name.
        file: &'static str,
        /// 1-based line number.
        line: usize,
        /// What went wrong.
        msg: String,
    },
    /// The camera uses a distortion model this project does not support.
    #[error(
        "camera {camera_id} uses model `{model}`; only undistorted PINHOLE / SIMPLE_PINHOLE are supported (run `colmap image_undistorter` first)"
    )]
    UnsupportedModel {
        /// Camera id.
        camera_id: u32,
        /// Model name as written in the file.
        model: String,
    },
    /// An image refers to a camera id that is not in `cameras.txt`.
    #[error("image `{image}` refers to unknown camera id {camera_id}")]
    UnknownCamera {
        /// Image name.
        image: String,
        /// Missing camera id.
        camera_id: u32,
    },
}

/// One reconstructed 3D point.
#[derive(Debug, Clone, PartialEq)]
pub struct Point3D {
    /// COLMAP point id.
    pub id: u64,
    /// Position in the model frame.
    pub xyz: Vec3,
    /// Color.
    pub rgb: [u8; 3],
    /// Mean reprojection error (pixels).
    pub error: f64,
}

/// One registered image: its COLMAP ids and its pose.
#[derive(Debug, Clone, PartialEq)]
pub struct Image {
    /// COLMAP image id.
    pub id: u32,
    /// COLMAP camera id.
    pub camera_id: u32,
    /// Pose and intrinsics (intrinsics looked up from `cameras.txt`).
    pub pose: CameraPose,
}

/// A parsed text model.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Model {
    /// Intrinsics by camera id.
    pub cameras: BTreeMap<u32, PinholeIntrinsics>,
    /// Images sorted by image id.
    pub images: Vec<Image>,
    /// 3D points sorted by id (empty if `points3D.txt` is absent).
    pub points: Vec<Point3D>,
}

impl Model {
    /// Just the camera poses, in image-id order.
    pub fn poses(&self) -> Vec<CameraPose> {
        self.images.iter().map(|i| i.pose.clone()).collect()
    }
}

fn perr(file: &'static str, line: usize, msg: impl Into<String>) -> ColmapError {
    ColmapError::Parse {
        file,
        line,
        msg: msg.into(),
    }
}

fn num<T: std::str::FromStr>(
    tok: Option<&str>,
    file: &'static str,
    line: usize,
    what: &str,
) -> Result<T, ColmapError> {
    let t = tok.ok_or_else(|| perr(file, line, format!("missing {what}")))?;
    t.parse::<T>()
        .map_err(|_| perr(file, line, format!("bad {what} `{t}`")))
}

fn finite(v: f64, file: &'static str, line: usize, what: &str) -> Result<f64, ColmapError> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(perr(file, line, format!("{what} is not finite")))
    }
}

/// Parses `cameras.txt` content.
pub fn parse_cameras(text: &str) -> Result<BTreeMap<u32, PinholeIntrinsics>, ColmapError> {
    const F: &str = "cameras.txt";
    let mut out = BTreeMap::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        let ln = i + 1;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut tok = line.split_whitespace();
        let id: u32 = num(tok.next(), F, ln, "camera id")?;
        let model = tok.next().ok_or_else(|| perr(F, ln, "missing model"))?;
        let width: u32 = num(tok.next(), F, ln, "width")?;
        let height: u32 = num(tok.next(), F, ln, "height")?;
        let params: Vec<f64> = tok
            .map(|t| {
                t.parse::<f64>()
                    .map_err(|_| perr(F, ln, format!("bad parameter `{t}`")))
            })
            .collect::<Result<_, _>>()?;
        let k = match (model, params.as_slice()) {
            ("PINHOLE", [fx, fy, cx, cy]) => PinholeIntrinsics {
                fx: *fx,
                fy: *fy,
                cx: *cx,
                cy: *cy,
                width,
                height,
            },
            ("SIMPLE_PINHOLE", [f, cx, cy]) => PinholeIntrinsics {
                fx: *f,
                fy: *f,
                cx: *cx,
                cy: *cy,
                width,
                height,
            },
            ("PINHOLE" | "SIMPLE_PINHOLE", p) => {
                return Err(perr(
                    F,
                    ln,
                    format!(
                        "{model} expects {} params, got {}",
                        if model == "PINHOLE" { 4 } else { 3 },
                        p.len()
                    ),
                ));
            }
            (other, _) => {
                return Err(ColmapError::UnsupportedModel {
                    camera_id: id,
                    model: other.to_string(),
                });
            }
        };
        if !(k.fx > 0.0 && k.fy > 0.0 && k.fx.is_finite() && k.fy.is_finite()) {
            return Err(perr(F, ln, "focal length must be positive and finite"));
        }
        if out.insert(id, k).is_some() {
            return Err(perr(F, ln, format!("duplicate camera id {id}")));
        }
    }
    Ok(out)
}

/// Parses `images.txt` content, attaching intrinsics from `cameras`.
pub fn parse_images(
    text: &str,
    cameras: &BTreeMap<u32, PinholeIntrinsics>,
) -> Result<Vec<Image>, ColmapError> {
    const F: &str = "images.txt";
    let mut out = Vec::new();
    // Iterate over non-comment lines but keep empty ones: the 2D-points line
    // that follows each image line may legitimately be empty.
    let mut lines = text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim_start().starts_with('#'));
    while let Some((i, raw)) = lines.next() {
        let line = raw.trim();
        let ln = i + 1;
        if line.is_empty() {
            continue;
        }
        let mut tok = line.split_whitespace();
        let id: u32 = num(tok.next(), F, ln, "image id")?;
        let mut q = [0.0f64; 4];
        for v in q.iter_mut() {
            *v = finite(num(tok.next(), F, ln, "quaternion")?, F, ln, "quaternion")?;
        }
        let mut t = [0.0f64; 3];
        for v in t.iter_mut() {
            *v = finite(num(tok.next(), F, ln, "translation")?, F, ln, "translation")?;
        }
        let camera_id: u32 = num(tok.next(), F, ln, "camera id")?;
        let name: Vec<&str> = tok.collect();
        if name.is_empty() {
            return Err(perr(F, ln, "missing image name"));
        }
        let name = name.join(" ");
        // Skip the 2D-points line (possibly absent at end of file).
        let _ = lines.next();

        let quat = Quaternion::new(q[0], q[1], q[2], q[3]);
        if quat.norm() < 1e-12 {
            return Err(perr(F, ln, "zero quaternion"));
        }
        let intrinsics = *cameras
            .get(&camera_id)
            .ok_or_else(|| ColmapError::UnknownCamera {
                image: name.clone(),
                camera_id,
            })?;
        out.push(Image {
            id,
            camera_id,
            pose: CameraPose {
                name,
                rot_cw: UnitQuaternion::from_quaternion(quat),
                trans_cw: Vec3::new(t[0], t[1], t[2]),
                intrinsics,
            },
        });
    }
    out.sort_by_key(|im| im.id);
    Ok(out)
}

/// Parses `points3D.txt` content.
pub fn parse_points(text: &str) -> Result<Vec<Point3D>, ColmapError> {
    const F: &str = "points3D.txt";
    let mut out = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        let ln = i + 1;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut tok = line.split_whitespace();
        let id: u64 = num(tok.next(), F, ln, "point id")?;
        let x = finite(num(tok.next(), F, ln, "x")?, F, ln, "x")?;
        let y = finite(num(tok.next(), F, ln, "y")?, F, ln, "y")?;
        let z = finite(num(tok.next(), F, ln, "z")?, F, ln, "z")?;
        let r: u8 = num(tok.next(), F, ln, "r")?;
        let g: u8 = num(tok.next(), F, ln, "g")?;
        let b: u8 = num(tok.next(), F, ln, "b")?;
        let error: f64 = num(tok.next(), F, ln, "error")?;
        out.push(Point3D {
            id,
            xyz: Vec3::new(x, y, z),
            rgb: [r, g, b],
            error,
        });
    }
    out.sort_by_key(|p| p.id);
    Ok(out)
}

fn read(dir: &Path, name: &str) -> Result<Option<String>, ColmapError> {
    let p = dir.join(name);
    match std::fs::read_to_string(&p) {
        Ok(s) => Ok(Some(s)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(ColmapError::Io {
            path: p.display().to_string(),
            source,
        }),
    }
}

/// Loads a text model from `dir` (`cameras.txt` and `images.txt` required,
/// `points3D.txt` optional).
pub fn load_model(dir: impl AsRef<Path>) -> Result<Model, ColmapError> {
    let dir = dir.as_ref();
    let missing = |name: &str| ColmapError::Io {
        path: dir.join(name).display().to_string(),
        source: std::io::Error::new(std::io::ErrorKind::NotFound, "file not found"),
    };
    let cams_txt = read(dir, "cameras.txt")?.ok_or_else(|| missing("cameras.txt"))?;
    let imgs_txt = read(dir, "images.txt")?.ok_or_else(|| missing("images.txt"))?;
    let cameras = parse_cameras(&cams_txt)?;
    let images = parse_images(&imgs_txt, &cameras)?;
    let points = match read(dir, "points3D.txt")? {
        Some(t) => parse_points(&t)?,
        None => Vec::new(),
    };
    Ok(Model {
        cameras,
        images,
        points,
    })
}

/// Serializes poses and points as a COLMAP text model.
///
/// Cameras with identical intrinsics share one camera id. Returns the
/// contents of `(cameras.txt, images.txt, points3D.txt)`.
pub fn format_model(poses: &[CameraPose], points: &[Point3D]) -> (String, String, String) {
    let mut cam_ids: Vec<PinholeIntrinsics> = Vec::new();
    let mut cams = String::from(
        "# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n",
    );
    let mut imgs = String::from(
        "# Image list with two lines of data per image:\n#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[] as (X, Y, POINT3D_ID)\n",
    );
    for (i, p) in poses.iter().enumerate() {
        let cam_id = match cam_ids.iter().position(|k| *k == p.intrinsics) {
            Some(j) => j + 1,
            None => {
                cam_ids.push(p.intrinsics);
                let k = p.intrinsics;
                let _ = writeln!(
                    cams,
                    "{} PINHOLE {} {} {:.17} {:.17} {:.17} {:.17}",
                    cam_ids.len(),
                    k.width,
                    k.height,
                    k.fx,
                    k.fy,
                    k.cx,
                    k.cy
                );
                cam_ids.len()
            }
        };
        let q = p.rot_cw.quaternion();
        let t = p.trans_cw;
        let _ = writeln!(
            imgs,
            "{} {:.17} {:.17} {:.17} {:.17} {:.17} {:.17} {:.17} {} {}\n",
            i + 1,
            q.w,
            q.i,
            q.j,
            q.k,
            t.x,
            t.y,
            t.z,
            cam_id,
            p.name
        );
    }
    let mut pts = String::from(
        "# 3D point list with one line of data per point:\n#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n",
    );
    for p in points {
        let _ = writeln!(
            pts,
            "{} {:.9} {:.9} {:.9} {} {} {} {:.6}",
            p.id, p.xyz.x, p.xyz.y, p.xyz.z, p.rgb[0], p.rgb[1], p.rgb[2], p.error
        );
    }
    (cams, imgs, pts)
}

/// Writes a text model into `dir` (created if needed).
pub fn write_model(
    dir: impl AsRef<Path>,
    poses: &[CameraPose],
    points: &[Point3D],
) -> Result<(), ColmapError> {
    let dir = dir.as_ref();
    let io = |p: &Path, source| ColmapError::Io {
        path: p.display().to_string(),
        source,
    };
    std::fs::create_dir_all(dir).map_err(|e| io(dir, e))?;
    let (c, i, p) = format_model(poses, points);
    for (name, body) in [("cameras.txt", c), ("images.txt", i), ("points3D.txt", p)] {
        let path = dir.join(name);
        std::fs::write(&path, body).map_err(|e| io(&path, e))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny model in the exact layout COLMAP writes, including an empty
    /// 2D-points line and comments.
    const CAMERAS: &str =
        "# Camera list\n1 PINHOLE 640 480 500 510 320 240\n2 SIMPLE_PINHOLE 100 50 80 50 25\n";
    const IMAGES: &str = "# Image list\n# two lines per image\n\
2 1 0 0 0 0 0 5 2 b.jpg\n\n\
1 0.7071067811865476 0 0.7071067811865476 0 1 2 3 1 a.jpg\n10.0 20.0 -1 30.0 40.0 7\n";
    const POINTS: &str = "# 3D points\n7 1.0 2.0 3.0 255 0 10 0.5 1 0 2 1\n";

    #[test]
    fn parses_sample_model() {
        let cams = parse_cameras(CAMERAS).unwrap();
        assert_eq!(cams[&1].fx, 500.0);
        assert_eq!(cams[&2].fy, 80.0);
        let imgs = parse_images(IMAGES, &cams).unwrap();
        assert_eq!(imgs.len(), 2);
        assert_eq!(imgs[0].id, 1);
        assert_eq!(imgs[0].pose.name, "a.jpg");
        assert_eq!(imgs[1].pose.name, "b.jpg");
        // Image 2: identity rotation, t = (0,0,5)  =>  C = -t = (0,0,-5).
        assert!((imgs[1].pose.center() - Vec3::new(0.0, 0.0, -5.0)).norm() < 1e-12);
        // Image 1: 90° about y, t = (1,2,3). C = -Rᵀ t.
        let r = imgs[0].pose.rot_cw.to_rotation_matrix().into_inner();
        let c = -(r.transpose() * Vec3::new(1.0, 2.0, 3.0));
        assert!((imgs[0].pose.center() - c).norm() < 1e-12);
        // Rᵀ for +90° about y maps (1,2,3) to (-3,2,1); negate -> (3,-2,-1).
        assert!((imgs[0].pose.center() - Vec3::new(3.0, -2.0, -1.0)).norm() < 1e-9);
        let pts = parse_points(POINTS).unwrap();
        assert_eq!(pts[0].rgb, [255, 0, 10]);
    }

    #[test]
    fn rejects_distortion_and_bad_input() {
        assert!(matches!(
            parse_cameras("1 OPENCV 640 480 500 500 320 240 0.1 0 0 0"),
            Err(ColmapError::UnsupportedModel { .. })
        ));
        assert!(matches!(
            parse_cameras("1 PINHOLE 640 480 500"),
            Err(ColmapError::Parse { .. })
        ));
        assert!(matches!(
            parse_cameras("1 PINHOLE 640 480 0 1 1 1"),
            Err(ColmapError::Parse { .. })
        ));
        let cams = parse_cameras(CAMERAS).unwrap();
        assert!(matches!(
            parse_images("1 1 0 0 0 0 0 0 9 x.jpg\n\n", &cams),
            Err(ColmapError::UnknownCamera { .. })
        ));
        assert!(matches!(
            parse_images("1 1 0 0 0 0 0 nan 1 x.jpg\n\n", &cams),
            Err(ColmapError::Parse { .. })
        ));
        assert!(matches!(
            parse_images("1 0 0 0 0 0 0 0 1 x.jpg\n\n", &cams),
            Err(ColmapError::Parse { .. })
        ));
        assert!(matches!(
            parse_images("1 1 0 0 0 0 0 0 1\n\n", &cams),
            Err(ColmapError::Parse { .. })
        ));
        assert!(load_model("/nonexistent/dir").is_err());
    }

    #[test]
    fn write_then_load_roundtrip() {
        let k = PinholeIntrinsics::from_fov_x(320, 240, 1.0);
        let poses: Vec<CameraPose> = (0..5)
            .map(|i| {
                let a = i as f64;
                CameraPose::look_at(
                    format!("{i:05}.jpg"),
                    Vec3::new(4.0 * a.cos(), 4.0 * a.sin(), 1.5),
                    Vec3::zeros(),
                    Vec3::z(),
                    k,
                )
                .unwrap()
            })
            .collect();
        let pts = vec![Point3D {
            id: 1,
            xyz: Vec3::new(0.1, 0.2, 0.3),
            rgb: [1, 2, 3],
            error: 0.25,
        }];
        let dir = std::env::temp_dir().join(format!("colmap-io-{}", std::process::id()));
        write_model(&dir, &poses, &pts).unwrap();
        let m = load_model(&dir).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(m.cameras.len(), 1);
        assert_eq!(m.images.len(), 5);
        for (a, b) in poses.iter().zip(m.poses()) {
            assert_eq!(a.name, b.name);
            assert!((a.center() - b.center()).norm() < 1e-12);
            assert!(a.rot_cw.angle_to(&b.rot_cw) < 1e-12);
        }
        assert_eq!(m.points.len(), 1);
    }
}
