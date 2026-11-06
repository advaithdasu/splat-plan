use serde::{Deserialize, Serialize};

/// Whether scene coordinates are in meters (no meters without scale).
///
/// COLMAP reconstructions have an arbitrary scale. Until a metric scale is
/// known (tape markers, a known wheelbase, a manual `scale.json`), every
/// distance the system computes is in unknown units. `Unscaled` must therefore
/// propagate all the way to the certificate, which then reads `UNSCALED`
/// instead of `SAFE`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "meters_per_unit", rename_all = "snake_case")]
pub enum Scale {
    /// Coordinates are in meters. The value is the factor that was applied to
    /// the original reconstruction units to obtain meters (kept for provenance).
    Metric(f64),
    /// Coordinates are in arbitrary reconstruction units.
    Unscaled,
}

impl Scale {
    /// True only for [`Scale::Metric`].
    pub fn is_metric(&self) -> bool {
        matches!(self, Scale::Metric(_))
    }

    /// The loud banner every output prints when distances are not metric.
    pub const UNSCALED_BANNER: &'static str = "*** UNSCALED: distances are in arbitrary reconstruction units, NOT meters. \
         Clearances and speeds below are meaningless until a metric scale is provided \
         (scale.json or annotations.json). No trajectory can be certified SAFE. ***";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_shape() {
        let s = serde_json::to_string(&Scale::Metric(2.0)).unwrap();
        assert_eq!(s, r#"{"kind":"metric","meters_per_unit":2.0}"#);
        let u = serde_json::to_string(&Scale::Unscaled).unwrap();
        assert_eq!(u, r#"{"kind":"unscaled"}"#);
        assert!(!Scale::Unscaled.is_metric());
    }
}
