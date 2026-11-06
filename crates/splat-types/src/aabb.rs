use serde::{Deserialize, Serialize};

use crate::Vec3;

/// Axis-aligned bounding box in `f64`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Aabb {
    /// Minimum corner.
    pub min: Vec3,
    /// Maximum corner.
    pub max: Vec3,
}

impl Aabb {
    /// Creates a box from two corners (they are sorted per axis).
    pub fn new(a: Vec3, b: Vec3) -> Self {
        Aabb {
            min: a.inf(&b),
            max: a.sup(&b),
        }
    }

    /// Smallest box containing all points, or `None` for an empty iterator.
    pub fn from_points<I: IntoIterator<Item = Vec3>>(points: I) -> Option<Self> {
        let mut it = points.into_iter();
        let first = it.next()?;
        let mut b = Aabb {
            min: first,
            max: first,
        };
        for p in it {
            b.min = b.min.inf(&p);
            b.max = b.max.sup(&p);
        }
        Some(b)
    }

    /// Box grown by `pad` on every side.
    pub fn padded(&self, pad: f64) -> Self {
        let d = Vec3::repeat(pad);
        Aabb {
            min: self.min - d,
            max: self.max + d,
        }
    }

    /// Edge lengths.
    pub fn extent(&self) -> Vec3 {
        self.max - self.min
    }

    /// Center point.
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// True if `p` is inside or on the boundary.
    pub fn contains(&self, p: &Vec3) -> bool {
        (0..3).all(|i| p[i] >= self.min[i] && p[i] <= self.max[i])
    }

    /// Euclidean distance from `p` to the box (0 inside).
    pub fn distance_to(&self, p: &Vec3) -> f64 {
        let mut d2 = 0.0;
        for i in 0..3 {
            let v = if p[i] < self.min[i] {
                self.min[i] - p[i]
            } else if p[i] > self.max[i] {
                p[i] - self.max[i]
            } else {
                0.0
            };
            d2 += v * v;
        }
        d2.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_points_and_distance() {
        let b = Aabb::from_points([Vec3::new(1.0, 2.0, 3.0), Vec3::new(-1.0, 0.0, 5.0)]).unwrap();
        assert_eq!(b.min, Vec3::new(-1.0, 0.0, 3.0));
        assert_eq!(b.max, Vec3::new(1.0, 2.0, 5.0));
        assert_eq!(b.distance_to(&Vec3::new(0.0, 1.0, 4.0)), 0.0);
        assert!((b.distance_to(&Vec3::new(4.0, 1.0, 4.0)) - 3.0).abs() < 1e-12);
        assert!(Aabb::from_points(std::iter::empty()).is_none());
    }
}
