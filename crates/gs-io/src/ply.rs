//! Binary PLY reader/writer for 3DGS files.

use std::path::Path;

use splat_types::nalgebra::{Quaternion, UnitQuaternion, Vector3};
use splat_types::{Gaussian, Vec3f};
use thiserror::Error;

/// Headers longer than this are rejected (real ones are a few KB).
const MAX_HEADER_BYTES: usize = 1 << 16;

/// Errors from reading or writing PLY files.
#[derive(Debug, Error)]
pub enum PlyError {
    /// Underlying I/O failure.
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
    /// The header is malformed.
    #[error("bad PLY header: {0}")]
    Header(String),
    /// The file is valid PLY but uses a feature this reader does not support.
    #[error("unsupported PLY feature: {0}")]
    Unsupported(String),
    /// A property every 3DGS file must have is absent.
    #[error("missing required vertex property `{0}`")]
    MissingProperty(String),
    /// The body is shorter than the header promises.
    #[error("file truncated: header promises {expected} bytes of vertex data, found {found}")]
    Truncated {
        /// Bytes required by the header.
        expected: usize,
        /// Bytes actually available.
        found: usize,
    },
    /// A value is NaN/infinite or otherwise unusable (e.g. a zero quaternion).
    #[error("invalid value in vertex {index}: {reason}")]
    InvalidValue {
        /// Vertex index.
        index: usize,
        /// What is wrong.
        reason: String,
    },
}

/// Scalar property types allowed by the PLY spec.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ScalarType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl ScalarType {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "char" | "int8" => ScalarType::I8,
            "uchar" | "uint8" => ScalarType::U8,
            "short" | "int16" => ScalarType::I16,
            "ushort" | "uint16" => ScalarType::U16,
            "int" | "int32" => ScalarType::I32,
            "uint" | "uint32" => ScalarType::U32,
            "float" | "float32" => ScalarType::F32,
            "double" | "float64" => ScalarType::F64,
            _ => return None,
        })
    }

    fn size(self) -> usize {
        match self {
            ScalarType::I8 | ScalarType::U8 => 1,
            ScalarType::I16 | ScalarType::U16 => 2,
            ScalarType::I32 | ScalarType::U32 | ScalarType::F32 => 4,
            ScalarType::F64 => 8,
        }
    }

    /// Reads one little-endian value. `b` must hold at least `size()` bytes
    /// (guaranteed by the caller's bounds check).
    fn read_le(self, b: &[u8]) -> f64 {
        let arr = |n: usize| -> [u8; 8] {
            let mut a = [0u8; 8];
            a[..n].copy_from_slice(&b[..n]);
            a
        };
        match self {
            ScalarType::I8 => i8::from_le_bytes([b[0]]) as f64,
            ScalarType::U8 => b[0] as f64,
            ScalarType::I16 => {
                let a = arr(2);
                i16::from_le_bytes([a[0], a[1]]) as f64
            }
            ScalarType::U16 => {
                let a = arr(2);
                u16::from_le_bytes([a[0], a[1]]) as f64
            }
            ScalarType::I32 => {
                let a = arr(4);
                i32::from_le_bytes([a[0], a[1], a[2], a[3]]) as f64
            }
            ScalarType::U32 => {
                let a = arr(4);
                u32::from_le_bytes([a[0], a[1], a[2], a[3]]) as f64
            }
            ScalarType::F32 => {
                let a = arr(4);
                f32::from_le_bytes([a[0], a[1], a[2], a[3]]) as f64
            }
            ScalarType::F64 => f64::from_le_bytes(arr(8)),
        }
    }
}

#[derive(Debug)]
struct Element {
    name: String,
    count: usize,
    /// `(name, type)`; `None` type marks a list property (unsupported in data we read).
    props: Vec<(String, Option<ScalarType>)>,
}

impl Element {
    /// Byte size of one record, or `None` if it contains list properties.
    fn stride(&self) -> Option<usize> {
        self.props
            .iter()
            .map(|(_, t)| t.map(ScalarType::size))
            .sum()
    }
}

/// Parses the ASCII header; returns the elements and the byte offset of the body.
fn parse_header(bytes: &[u8]) -> Result<(Vec<Element>, usize), PlyError> {
    const END: &[u8] = b"end_header\n";
    let search = &bytes[..bytes.len().min(MAX_HEADER_BYTES)];
    let end = search
        .windows(END.len())
        .position(|w| w == END)
        .ok_or_else(|| PlyError::Header("no `end_header` line found".into()))?;
    let body_start = end + END.len();
    let text = std::str::from_utf8(&bytes[..end])
        .map_err(|_| PlyError::Header("header is not valid UTF-8/ASCII".into()))?;

    let mut lines = text.lines();
    if lines.next().map(str::trim) != Some("ply") {
        return Err(PlyError::Header("file does not start with `ply`".into()));
    }
    let mut format_seen = false;
    let mut elements: Vec<Element> = Vec::new();
    for (lineno, raw) in lines.enumerate() {
        let line = raw.trim();
        let mut tok = line.split_whitespace();
        match tok.next() {
            None | Some("comment") | Some("obj_info") => {}
            Some("format") => {
                match tok.next() {
                    Some("binary_little_endian") => {}
                    Some(f @ ("ascii" | "binary_big_endian")) => {
                        return Err(PlyError::Unsupported(format!(
                            "format `{f}` (only binary_little_endian)"
                        )));
                    }
                    other => {
                        return Err(PlyError::Header(format!("unknown format {other:?}")));
                    }
                }
                format_seen = true;
            }
            Some("element") => {
                let name = tok.next().ok_or_else(|| {
                    PlyError::Header(format!("line {}: element name", lineno + 2))
                })?;
                let count = tok
                    .next()
                    .and_then(|c| c.parse::<usize>().ok())
                    .ok_or_else(|| {
                        PlyError::Header(format!("line {}: element count", lineno + 2))
                    })?;
                elements.push(Element {
                    name: name.to_string(),
                    count,
                    props: Vec::new(),
                });
            }
            Some("property") => {
                let el = elements.last_mut().ok_or_else(|| {
                    PlyError::Header(format!("line {}: property before element", lineno + 2))
                })?;
                let ty = tok.next().ok_or_else(|| {
                    PlyError::Header(format!("line {}: property type", lineno + 2))
                })?;
                if ty == "list" {
                    // `property list <count type> <item type> <name>`
                    let name = tok.nth(2).ok_or_else(|| {
                        PlyError::Header(format!("line {}: list property name", lineno + 2))
                    })?;
                    el.props.push((name.to_string(), None));
                } else {
                    let st = ScalarType::parse(ty).ok_or_else(|| {
                        PlyError::Header(format!("line {}: unknown type `{ty}`", lineno + 2))
                    })?;
                    let name = tok.next().ok_or_else(|| {
                        PlyError::Header(format!("line {}: property name", lineno + 2))
                    })?;
                    el.props.push((name.to_string(), Some(st)));
                }
            }
            Some(other) => {
                return Err(PlyError::Header(format!(
                    "line {}: unexpected keyword `{other}`",
                    lineno + 2
                )));
            }
        }
    }
    if !format_seen {
        return Err(PlyError::Header("missing `format` line".into()));
    }
    Ok((elements, body_start))
}

/// Property names a 3DGS vertex must carry, in the order we read them.
const REQUIRED: [&str; 14] = [
    "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
];

/// Parses a 3DGS PLY from memory. Never panics on malformed input.
pub fn parse_ply(bytes: &[u8]) -> Result<Vec<Gaussian>, PlyError> {
    let (elements, body_start) = parse_header(bytes)?;

    // The vertex block starts after every element declared before it.
    let mut offset = body_start;
    let mut vertex: Option<&Element> = None;
    for el in &elements {
        if el.name == "vertex" {
            vertex = Some(el);
            break;
        }
        let stride = el.stride().ok_or_else(|| {
            PlyError::Unsupported(format!("list properties in element `{}`", el.name))
        })?;
        offset = el
            .count
            .checked_mul(stride)
            .and_then(|n| n.checked_add(offset))
            .ok_or_else(|| PlyError::Header("element sizes overflow".into()))?;
    }
    let vertex = vertex.ok_or_else(|| PlyError::Header("no `vertex` element".into()))?;
    let stride = vertex
        .stride()
        .ok_or_else(|| PlyError::Unsupported("list property in vertex element".into()))?;

    // Byte offset and type of each required property within one vertex record.
    let mut fields = [(0usize, ScalarType::F32); REQUIRED.len()];
    for (slot, want) in fields.iter_mut().zip(REQUIRED) {
        let mut off = 0usize;
        let mut found = None;
        for (name, ty) in &vertex.props {
            // stride() succeeded, so every type is Some.
            let ty = ty.unwrap_or(ScalarType::F32);
            if name == want {
                found = Some((off, ty));
                break;
            }
            off += ty.size();
        }
        *slot = found.ok_or_else(|| PlyError::MissingProperty(want.to_string()))?;
    }

    let expected = vertex
        .count
        .checked_mul(stride)
        .ok_or_else(|| PlyError::Header("vertex count overflows".into()))?;
    let available = bytes.len().saturating_sub(offset);
    if available < expected {
        return Err(PlyError::Truncated {
            expected,
            found: available,
        });
    }

    let body = &bytes[offset..offset + expected];
    let mut out = Vec::with_capacity(vertex.count);
    let mut v = [0.0f64; REQUIRED.len()];
    for (index, rec) in body
        .chunks_exact(stride.max(1))
        .take(vertex.count)
        .enumerate()
    {
        for (dst, (off, ty)) in v.iter_mut().zip(fields.iter()) {
            *dst = ty.read_le(&rec[*off..]);
        }
        if v.iter().any(|x| !x.is_finite()) {
            return Err(PlyError::InvalidValue {
                index,
                reason: "non-finite value".into(),
            });
        }
        let q = Quaternion::new(v[10], v[11], v[12], v[13]);
        if q.norm() < 1e-12 {
            return Err(PlyError::InvalidValue {
                index,
                reason: "zero-length rotation quaternion".into(),
            });
        }
        let rot = UnitQuaternion::from_quaternion(q).cast::<f32>();
        // Activation functions: scales are stored as logs, opacity as a logit.
        let scale = Vector3::new(v[7].exp(), v[8].exp(), v[9].exp()).cast::<f32>();
        let opacity = (1.0 / (1.0 + (-v[6]).exp())) as f32;
        out.push(Gaussian {
            mean: Vec3f::new(v[0] as f32, v[1] as f32, v[2] as f32),
            scale,
            rot,
            opacity,
            sh_dc: [v[3] as f32, v[4] as f32, v[5] as f32],
        });
    }
    Ok(out)
}

/// Loads a 3DGS PLY file from disk.
pub fn load_ply(path: impl AsRef<Path>) -> Result<Vec<Gaussian>, PlyError> {
    let bytes = std::fs::read(path)?;
    parse_ply(&bytes)
}

/// Encodes Gaussians as a standard 3DGS binary PLY (no `f_rest` coefficients).
pub fn encode_ply(gaussians: &[Gaussian]) -> Vec<u8> {
    let props = [
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0",
        "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ];
    let mut header = format!(
        "ply\nformat binary_little_endian 1.0\ncomment written by splatplan gs-io\nelement vertex {}\n",
        gaussians.len()
    );
    for p in props {
        header.push_str("property float ");
        header.push_str(p);
        header.push('\n');
    }
    header.push_str("end_header\n");

    let mut out = Vec::with_capacity(header.len() + gaussians.len() * props.len() * 4);
    out.extend_from_slice(header.as_bytes());
    for g in gaussians {
        // Inverse activations; clamp so the logit and log stay finite.
        let o = g.opacity.clamp(1e-6, 1.0 - 1e-6);
        let logit = (o / (1.0 - o)).ln();
        let ls = g.scale.map(|s| s.max(f32::MIN_POSITIVE).ln());
        let q = g.rot.quaternion();
        let vals = [
            g.mean.x, g.mean.y, g.mean.z, 0.0, 0.0, 0.0, g.sh_dc[0], g.sh_dc[1], g.sh_dc[2], logit,
            ls.x, ls.y, ls.z, q.w, q.i, q.j, q.k,
        ];
        for v in vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// Writes Gaussians to `path` as a standard 3DGS binary PLY.
pub fn write_ply(path: impl AsRef<Path>, gaussians: &[Gaussian]) -> Result<(), PlyError> {
    std::fs::write(path, encode_ply(gaussians))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn random_gaussians(n: usize, seed: u64) -> Vec<Gaussian> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..n)
            .map(|_| Gaussian {
                mean: Vec3f::new(
                    rng.gen_range(-5.0..5.0),
                    rng.gen_range(-5.0..5.0),
                    rng.gen_range(0.0..1.0),
                ),
                scale: Vec3f::new(
                    rng.gen_range(0.001..0.5),
                    rng.gen_range(0.001..0.5),
                    rng.gen_range(0.001..0.5),
                ),
                rot: UnitQuaternion::from_quaternion(Quaternion::new(
                    rng.gen_range(-1.0..1.0),
                    rng.gen_range(-1.0..1.0),
                    rng.gen_range(-1.0..1.0),
                    rng.gen_range(0.1..1.0),
                )),
                opacity: rng.gen_range(0.01..0.99),
                sh_dc: [
                    rng.gen_range(-2.0..2.0),
                    rng.gen_range(0.0..1.0),
                    rng.gen_range(0.0..1.0),
                ],
            })
            .collect()
    }

    #[test]
    fn roundtrip() {
        let gs = random_gaussians(500, 7);
        let back = parse_ply(&encode_ply(&gs)).unwrap();
        assert_eq!(back.len(), gs.len());
        for (a, b) in gs.iter().zip(&back) {
            assert_eq!(a.mean, b.mean);
            assert_eq!(a.sh_dc, b.sh_dc);
            assert!((a.scale - b.scale).norm() <= 1e-5 * a.scale.norm());
            assert!((a.opacity - b.opacity).abs() < 1e-5);
            assert!(a.rot.angle_to(&b.rot) < 1e-3);
        }
    }

    #[test]
    fn roundtrip_via_file() {
        let dir = std::env::temp_dir().join(format!("gsio-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.ply");
        let gs = random_gaussians(10, 1);
        write_ply(&path, &gs).unwrap();
        assert_eq!(load_ply(&path).unwrap().len(), 10);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A header in the INRIA layout, with `f_rest_*`, normals and an unusual
    /// property order, plus a preceding element, parses by name.
    #[test]
    fn parses_by_name_not_offset() {
        let mut h = String::from(
            "ply\nformat binary_little_endian 1.0\nelement camera 1\nproperty double fx\nelement vertex 1\n",
        );
        let mut names: Vec<String> = vec!["rot_0".into(), "x".into(), "y".into(), "z".into()];
        names.extend((0..45).map(|i| format!("f_rest_{i}")));
        names.extend(
            [
                "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_1",
                "rot_2", "rot_3",
            ]
            .map(String::from),
        );
        for n in &names {
            h.push_str(&format!("property float {n}\n"));
        }
        h.push_str("end_header\n");
        let mut bytes = h.into_bytes();
        bytes.extend_from_slice(&123.0f64.to_le_bytes()); // camera element
        for n in &names {
            let v: f32 = match n.as_str() {
                "x" => 1.0,
                "y" => 2.0,
                "z" => 3.0,
                "rot_0" => 2.0,   // unnormalized w
                "opacity" => 0.0, // sigmoid(0) = 0.5
                "scale_0" | "scale_1" | "scale_2" => (0.1f32).ln(),
                _ => 0.0,
            };
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let g = parse_ply(&bytes).unwrap();
        assert_eq!(g[0].mean, Vec3f::new(1.0, 2.0, 3.0));
        assert!((g[0].opacity - 0.5).abs() < 1e-6);
        assert!((g[0].scale.x - 0.1).abs() < 1e-6);
        assert!(g[0].rot.angle() < 1e-6);
    }

    #[test]
    fn errors_on_malformed_input() {
        let good = encode_ply(&random_gaussians(3, 2));
        // Truncated body.
        assert!(matches!(
            parse_ply(&good[..good.len() - 1]),
            Err(PlyError::Truncated { .. })
        ));
        // Truncated header.
        assert!(matches!(parse_ply(&good[..40]), Err(PlyError::Header(_))));
        assert!(matches!(parse_ply(b""), Err(PlyError::Header(_))));
        assert!(matches!(
            parse_ply(b"plx\nend_header\n"),
            Err(PlyError::Header(_))
        ));
        assert!(matches!(
            parse_ply(b"ply\nformat ascii 1.0\nend_header\n"),
            Err(PlyError::Unsupported(_))
        ));
        assert!(matches!(
            parse_ply(b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nend_header\n"),
            Err(PlyError::MissingProperty(_))
        ));
        assert!(matches!(
            parse_ply(b"ply\nformat binary_little_endian 1.0\nelement vertex 99999999999999999999\nend_header\n"),
            Err(PlyError::Header(_))
        ));
        // Huge count that would overflow count * stride.
        let huge = format!(
            "ply\nformat binary_little_endian 1.0\nelement vertex {}\n{}end_header\n",
            usize::MAX / 2,
            REQUIRED.map(|n| format!("property double {n}\n")).concat()
        );
        assert!(matches!(
            parse_ply(huge.as_bytes()),
            Err(PlyError::Header(_))
        ));
    }

    #[test]
    fn rejects_nan_and_zero_quaternion() {
        let mut g = random_gaussians(1, 3);
        g[0].mean.x = f32::NAN;
        assert!(matches!(
            parse_ply(&encode_ply(&g)),
            Err(PlyError::InvalidValue { index: 0, .. })
        ));
        let mut bytes = encode_ply(&random_gaussians(1, 3));
        let n = bytes.len();
        // rot_0..3 are the last 16 bytes.
        for b in &mut bytes[n - 16..] {
            *b = 0;
        }
        assert!(matches!(
            parse_ply(&bytes),
            Err(PlyError::InvalidValue { .. })
        ));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]
        /// Fuzz: arbitrary corruption of a valid file must yield Ok or Err, never a panic.
        #[test]
        fn fuzz_corruption_never_panics(
            cut in 0usize..2000,
            flips in prop::collection::vec((0usize..2000, any::<u8>()), 0..8),
        ) {
            let mut bytes = encode_ply(&random_gaussians(4, 9));
            for (pos, val) in flips {
                let n = bytes.len();
                bytes[pos % n] = val;
            }
            let cut = cut.min(bytes.len());
            let _ = parse_ply(&bytes[..cut]);
            let _ = parse_ply(&bytes);
        }

        /// Fuzz: random header text never panics.
        #[test]
        fn fuzz_header_text(s in "ply\n(format binary_little_endian 1.0\n)?((element|property|comment) [a-z0-9_ ]{0,20}\n){0,6}end_header\n") {
            let _ = parse_ply(s.as_bytes());
        }
    }
}
