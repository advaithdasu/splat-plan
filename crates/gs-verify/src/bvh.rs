//! A small bounding-volume hierarchy over ellipsoid bounding boxes.
//!
//! Built once per scene by recursive median splits; queried with
//! branch-and-bound for the smallest clearance lower bound.

use splat_types::Vec3;

use crate::bound::Ellipsoid;

/// Ellipsoids per leaf.
const LEAF_SIZE: usize = 8;

#[derive(Debug, Clone)]
struct Node {
    min: Vec3,
    max: Vec3,
    /// Leaf: `items[start..end]`. Internal: `left`/`right` node indices.
    kind: NodeKind,
}

#[derive(Debug, Clone)]
enum NodeKind {
    Leaf { start: usize, end: usize },
    Internal { left: usize, right: usize },
}

/// Bounding-volume hierarchy over a set of ellipsoids.
#[derive(Debug, Clone)]
pub struct Bvh {
    nodes: Vec<Node>,
    items: Vec<Ellipsoid>,
}

fn box_distance(p: &Vec3, min: &Vec3, max: &Vec3) -> f64 {
    let mut d2 = 0.0;
    for a in 0..3 {
        let v = (min[a] - p[a]).max(0.0).max(p[a] - max[a]);
        d2 += v * v;
    }
    d2.sqrt()
}

impl Bvh {
    /// Builds the hierarchy. Deterministic: splits sort by centroid with the
    /// source index as tie-breaker.
    pub fn build(mut items: Vec<Ellipsoid>) -> Self {
        let mut nodes = Vec::new();
        if !items.is_empty() {
            let n = items.len();
            build_rec(&mut nodes, &mut items, 0, n);
        }
        Bvh { nodes, items }
    }

    /// Number of ellipsoids.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// True if there are no ellipsoids.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// A lower bound on the distance from `p` to the nearest ellipsoid, and
    /// the source index of the ellipsoid that produced it.
    ///
    /// Why pruning keeps this a valid bound: a subtree is skipped only when
    /// its box is at least `best` away, and every ellipsoid inside that box is
    /// then at least `best` away too. So the result is ≤ the true distance to
    /// every ellipsoid, visited or not. Returns `(∞, None)` for an empty set.
    pub fn nearest_lower_bound(&self, p: &Vec3) -> (f64, Option<usize>) {
        let mut best = f64::INFINITY;
        let mut who = None;
        if self.nodes.is_empty() {
            return (best, who);
        }
        let mut stack: Vec<(usize, f64)> = Vec::with_capacity(64);
        stack.push((0, box_distance(p, &self.nodes[0].min, &self.nodes[0].max)));
        while let Some((ni, d)) = stack.pop() {
            if d >= best {
                continue;
            }
            match self.nodes[ni].kind {
                NodeKind::Leaf { start, end } => {
                    for e in &self.items[start..end] {
                        let lb = e.lower_bound(p);
                        if lb < best {
                            best = lb;
                            who = Some(e.source);
                        }
                    }
                }
                NodeKind::Internal { left, right } => {
                    let dl = box_distance(p, &self.nodes[left].min, &self.nodes[left].max);
                    let dr = box_distance(p, &self.nodes[right].min, &self.nodes[right].max);
                    // Push the farther child first so the nearer one is explored first.
                    if dl <= dr {
                        stack.push((right, dr));
                        stack.push((left, dl));
                    } else {
                        stack.push((left, dl));
                        stack.push((right, dr));
                    }
                }
            }
        }
        (best, who)
    }

    /// True if the segment `a → b` may touch any ellipsoid (it is then
    /// treated as blocked: the conservative answer for line of sight).
    ///
    /// Subtrees are skipped only when the segment misses their box, and
    /// every ellipsoid lies inside its box, so no hit is ever missed.
    pub fn segment_blocked(&self, a: &Vec3, b: &Vec3) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let mut stack: Vec<usize> = Vec::with_capacity(64);
        stack.push(0);
        while let Some(ni) = stack.pop() {
            let node = &self.nodes[ni];
            if !segment_meets_box(a, b, &node.min, &node.max) {
                continue;
            }
            match node.kind {
                NodeKind::Leaf { start, end } => {
                    if self.items[start..end]
                        .iter()
                        .any(|e| e.segment_may_hit(a, b))
                    {
                        return true;
                    }
                }
                NodeKind::Internal { left, right } => {
                    stack.push(right);
                    stack.push(left);
                }
            }
        }
        false
    }
}

/// Slab test: does the segment `a → b` meet the box (padded by a tiny
/// tolerance so rounding can only report extra contacts)?
fn segment_meets_box(a: &Vec3, b: &Vec3, min: &Vec3, max: &Vec3) -> bool {
    let d = b - a;
    let mut t0: f64 = 0.0;
    let mut t1: f64 = 1.0;
    for k in 0..3 {
        let pad = 1e-9 * (1.0 + min[k].abs().max(max[k].abs()));
        let (lo, hi) = (min[k] - pad, max[k] + pad);
        if d[k].abs() < 1e-300 {
            if a[k] < lo || a[k] > hi {
                return false;
            }
        } else {
            let inv = 1.0 / d[k];
            let (mut ta, mut tb) = ((lo - a[k]) * inv, (hi - a[k]) * inv);
            if ta > tb {
                std::mem::swap(&mut ta, &mut tb);
            }
            t0 = t0.max(ta);
            t1 = t1.min(tb);
            if t0 > t1 {
                return false;
            }
        }
    }
    true
}

fn build_rec(nodes: &mut Vec<Node>, items: &mut [Ellipsoid], start: usize, end: usize) -> usize {
    let slice = &mut items[start..end];
    let mut min = Vec3::repeat(f64::INFINITY);
    let mut max = Vec3::repeat(f64::NEG_INFINITY);
    let mut cmin = min;
    let mut cmax = max;
    for e in slice.iter() {
        min = min.inf(&(e.center - e.aabb_half));
        max = max.sup(&(e.center + e.aabb_half));
        cmin = cmin.inf(&e.center);
        cmax = cmax.sup(&e.center);
    }
    let id = nodes.len();
    nodes.push(Node {
        min,
        max,
        kind: NodeKind::Leaf { start, end },
    });
    if end - start <= LEAF_SIZE {
        return id;
    }
    // Split at the median along the longest axis of the centroid box.
    let ext = cmax - cmin;
    let axis = if ext.x >= ext.y && ext.x >= ext.z {
        0
    } else if ext.y >= ext.z {
        1
    } else {
        2
    };
    slice.sort_by(|a, b| {
        a.center[axis]
            .total_cmp(&b.center[axis])
            .then(a.source.cmp(&b.source))
    });
    let mid = start + (end - start) / 2;
    let left = build_rec(nodes, items, start, mid);
    let right = build_rec(nodes, items, mid, end);
    nodes[id].kind = NodeKind::Internal { left, right };
    id
}
