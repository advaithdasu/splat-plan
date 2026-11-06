# planner

From two points to a certified trajectory.

1. `astar`: 26-connected A* through observed-free voxels with clearance; total-order heap for deterministic ties.
2. `shortcut`: greedy line-of-sight pruning; every segment checked at res/5 spacing with Lipschitz slack.
3. `smooth` / `minsnap`: degree-7 minimum-snap through the waypoints (closed-form KKT), time-scaled to the limits, midpoints inserted where the curve strays.
4. `plan_and_certify`: ask `gs-verify`; on rejection, the obstacles it reports become keep-out spheres; bounded retries, then refuse.
5. `standoff_orbit`: closed inspection loops at a fixed distance from the largest object.

Every failure is a `PlanError` with a reason (refuse rather than guess).
