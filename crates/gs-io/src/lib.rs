//! Reading, writing, filtering and summarizing standard 3D Gaussian Splatting PLY files.
//!
//! The standard 3DGS PLY (as written by the original INRIA code, Brush,
//! gsplat, OpenSplat, ...) is a binary little-endian PLY whose `vertex`
//! element carries, per Gaussian:
//!
//! ```text
//! x y z  nx ny nz  f_dc_0..2  f_rest_0..44  opacity  scale_0..2  rot_0..3
//! ```
//!
//! with **log** scales, **logit** opacity and an unnormalized `(w, x, y, z)`
//! quaternion. Exporters differ in which optional properties they include, so
//! this reader parses the header and looks properties up **by name**, never by
//! fixed byte offset. Anything required that is missing is an error.
#![forbid(unsafe_code)]

mod filter;
mod inspect;
mod ply;

pub use filter::{OutlierFilter, filter_opacity, filter_outliers, floater_mask, isolation_mask};
pub use inspect::{InspectReport, inspect};
pub use ply::{PlyError, encode_ply, load_ply, parse_ply, write_ply};
