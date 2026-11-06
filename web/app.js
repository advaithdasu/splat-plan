// SplatPlan web viewer: a splat (Spark), the drone's recovered flight path,
// and an optional layer of certified planned paths (three.js).
// Data comes from `splatplan export` into data/<scene>/:
//   scene.splat  aligned splat (meters, +Z up)
//   scene.json   metadata (cameras, drone size, provenance)
//   flight.json  the drone's real path, recovered from the video
//                (`splatplan flight`), with video.mp4 beside it
//   paths.json   certified planned paths, [t, x, y, z] samples, best first
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

// Never show more planned paths than this (export already ranks and caps).
const MAX_PATHS = 5;
const COLORS = [0xff5a3c, 0x3cc8ff, 0xffd23c, 0x7dff6a, 0xd06bff, 0xff8fc8, 0x4cffd2, 0xffffff];

const container = document.getElementById("viewer");
if (!document.createElement("canvas").getContext("webgl2")) {
  document.getElementById("loading").textContent =
    "This viewer needs WebGL 2, which this browser doesn't provide. Try a recent Chrome, Edge, Firefox or Safari.";
  throw new Error("WebGL 2 unavailable");
}
const renderer = new THREE.WebGLRenderer({ antialias: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);
const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 200);
camera.up.set(0, 0, 1); // exported scenes are +Z up
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new SparkRenderer({ renderer }));

const ui = {
  loading: document.getElementById("loading"),
  title: document.getElementById("scene-title"),
  facts: document.getElementById("scene-facts"),
  pathSection: document.getElementById("path-section"),
  showFlight: document.getElementById("show-flight"),
  list: document.getElementById("path-list"),
  detail: document.getElementById("path-detail"),
  fly: document.getElementById("fly"),
  showCams: document.getElementById("show-cams"),
  showPaths: document.getElementById("show-paths"),
  flightSection: document.getElementById("flight-section"),
  flightFacts: document.getElementById("flight-facts"),
  fplay: document.getElementById("fplay"),
  camFree: document.getElementById("cam-free"),
  camDrone: document.getElementById("cam-drone"),
  badge: document.getElementById("viewbadge"),
  badgeText: document.getElementById("viewbadge-text"),
  hint: document.getElementById("hint"),
  fslider: document.getElementById("fslider"),
  ftime: document.getElementById("ftime"),
  fvideoBox: document.getElementById("fvideo-box"),
  fvideo: document.getElementById("fvideo"),
  fvideoCap: document.getElementById("fvideo-cap"),
};

let current = null; // { name, info, paths, splat, group, cams, home, rec }
let selected = -1;
let flight = null; // { path, start, marker }
let loadToken = 0;

function resize() {
  const w = container.clientWidth;
  const h = container.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(container);

// Position along a path at time t (linear between 0.1 s samples).
function at(path, t) {
  const s = path.samples;
  if (t <= s[0][0]) return new THREE.Vector3(s[0][1], s[0][2], s[0][3]);
  const last = s[s.length - 1];
  if (t >= last[0]) return new THREE.Vector3(last[1], last[2], last[3]);
  let lo = 0;
  let hi = s.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (s[mid][0] <= t) lo = mid; else hi = mid;
  }
  const a = s[lo];
  const b = s[hi];
  const u = (t - a[0]) / (b[0] - a[0]);
  return new THREE.Vector3(a[1] + u * (b[1] - a[1]), a[2] + u * (b[2] - a[2]), a[3] + u * (b[3] - a[3]));
}

function pathCurve(path) {
  return new THREE.CatmullRomCurve3(path.samples.map((p) => new THREE.Vector3(p[1], p[2], p[3])));
}

// replan.json labels: "start", "end" or "hover at m:ss".
const placeName = (label) =>
  ({ start: "Takeoff", end: "Landing" })[label] ?? label.charAt(0).toUpperCase() + label.slice(1);

function tripName(p) {
  const r = p.real;
  return r ? `${placeName(r.from_label)} → ${placeName(r.to_label)}` : null;
}

// The drone's real route between two video times, split at gaps.
function realRoute(rec, t0, t1) {
  const g = new THREE.Group();
  if (!rec) return g;
  let run = [];
  const flush = () => {
    if (run.length > 1) {
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(run),
        new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9, depthTest: false }),
      );
      line.renderOrder = 10;
      g.add(line);
    }
    run = [];
  };
  let prev = null;
  for (const p of rec.poses) {
    if (p[0] < t0 || p[0] > t1) continue;
    if (prev && p[0] - prev[0] > rec.gap_s) flush();
    run.push(new THREE.Vector3(p[1], p[2], p[3]));
    prev = p;
  }
  flush();
  return g;
}

// A path with its uncertified ends added: from where the drone really started
// to where it really stopped. The ends are flown at the path's average speed.
function withEnds(p) {
  const speed = p.length_m / Math.max(p.duration_s, 1e-6) || 1;
  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  const pts = p.samples.map((q) => q.slice(1));
  const { from_xyz: from, to_xyz: to } = p.real ?? {};
  if (from && dist(from, pts[0]) > 0.05) pts.unshift(from);
  if (to && dist(to, pts[pts.length - 1]) > 0.05) pts.push(to);
  const samples = [];
  let t = 0;
  pts.forEach((q, k) => {
    if (k) t += dist(q, pts[k - 1]) / speed;
    samples.push([t, ...q]);
  });
  return { samples, duration_s: t };
}

function buildPaths(info, paths, rec) {
  const group = new THREE.Group();
  const envelope = info.robot_radius_m + info.margin_m;
  paths.forEach((p, i) => {
    const color = COLORS[i % COLORS.length];
    const curve = pathCurve(p);
    const segs = Math.max(16, p.samples.length * 2);
    // Drawn over the splat, like the flight trail, so the whole path shows.
    const line = new THREE.Mesh(
      new THREE.TubeGeometry(curve, segs, 0.025, 6, false),
      new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true }),
    );
    line.renderOrder = 11;
    const body = new THREE.Mesh(
      new THREE.TubeGeometry(curve, segs, envelope, 16, false),
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.18, depthWrite: false }),
    );
    body.visible = false;
    // Ends: where the drone really was, joined to the certified path by a
    // dashed leg when the planner had to start or stop elsewhere (a takeoff
    // spot on the floor is too close to it to certify).
    const ends = new THREE.Group();
    const v3 = (s) => new THREE.Vector3(s[1], s[2], s[3]);
    const first = v3(p.samples[0]), last = v3(p.samples[p.samples.length - 1]);
    const at = (xyz, fallback) => (xyz ? new THREE.Vector3(...xyz) : fallback);
    const from = at(p.real?.from_xyz, first), to = at(p.real?.to_xyz, last);
    for (const [pos, c] of [[from, 0x30e070], [to, 0xe030e0]]) {
      const dot = new THREE.Mesh(new THREE.SphereGeometry(0.05, 12, 8), new THREE.MeshBasicMaterial({ color: c, depthTest: false, transparent: true }));
      dot.position.copy(pos);
      dot.renderOrder = 12;
      ends.add(dot);
    }
    // Lighter tubes: the stretch the checker did not certify.
    for (const [a, b] of [[from, first], [last, to]]) {
      if (a.distanceTo(b) < 0.05) continue;
      const leg = new THREE.Mesh(
        new THREE.TubeGeometry(new THREE.LineCurve3(a, b), 4, 0.025, 6, false),
        new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.45 }),
      );
      leg.renderOrder = 11;
      ends.add(leg);
    }
    // The straight line from start to goal, dashed: for showcase paths it
    // runs through something solid, which is why the path bends away from it.
    const a = p.samples[0], b = p.samples[p.samples.length - 1];
    const direct = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(a[1], a[2], a[3]), new THREE.Vector3(b[1], b[2], b[3])]),
      new THREE.LineDashedMaterial({ color: 0xffffff, dashSize: 0.12, gapSize: 0.08, transparent: true, opacity: 0.85 }),
    );
    direct.computeLineDistances();
    direct.visible = false;
    // For paths between real flight points: what the drone actually flew.
    const real = p.real
      ? realRoute(rec, p.real.from_label === "start" ? 0 : p.real.t0, p.real.to_label === "end" ? 1e9 : p.real.t1)
      : new THREE.Group();
    real.visible = false;
    const item = new THREE.Group();
    item.add(line, body, ends, direct, real);
    item.userData = { line, body, color, direct, real };
    group.add(item);
  });
  return group;
}

function buildCams(info) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(info.cameras.flat(), 3));
  const pts = new THREE.Points(g, new THREE.PointsMaterial({ color: 0xffffff, size: 0.05 }));
  pts.visible = ui.showCams.checked;
  return pts;
}

// Default view: a training camera (exported), else a raised view of the camera region.
function homeView(info) {
  if (info.home) {
    return {
      position: new THREE.Vector3(...info.home.position),
      target: new THREE.Vector3(...info.home.target),
      fov: info.home.fov_deg,
    };
  }
  const box = new THREE.Box3();
  for (const c of info.cameras) box.expandByPoint(new THREE.Vector3(...c));
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const r = Math.max(size.x, size.y, 2.0);
  return {
    target: new THREE.Vector3(center.x, center.y, Math.max(center.z - 0.3, 0.3)),
    position: new THREE.Vector3(center.x - 0.55 * r, center.y - 0.75 * r, center.z + 0.9 * r),
    fov: 60,
  };
}

function goHome() {
  if (!current) return;
  if (riding) setRide(false);
  stopFlight();
  // Start inside the room, at the drone's first camera view. Only with no
  // exported view fall back to the overhead view of the whole flight.
  if (current.rec && !current.info.home) { overview(); return; }
  camera.fov = current.home.fov;
  camera.updateProjectionMatrix();
  camera.position.copy(current.home.position);
  controls.target.copy(current.home.target);
  controls.update();
}

function describe(info) {
  const n = info.gaussians_exported.toLocaleString();
  return `${n} Gaussians, trained on ${info.cameras.length} keyframes from the drone video.`;
}

// Long paths that go around many obstacles first (same rule as export).
function pathScore(p) {
  const n = p.avoidance?.obstacles ?? p.weave?.obstacles_threaded ?? 0;
  return p.score ?? p.length_m * (1 + n);
}

function select(i) {
  if (!current) return;
  selected = i;
  // With a path selected, show only that path and its drone-size tube.
  current.group.children.forEach((item, k) => {
    item.visible = i < 0 || k === i;
    item.userData.body.visible = k === i;
    item.userData.direct.visible = k === i && !!current.paths[k].avoidance && !current.paths[k].real;
    item.userData.real.visible = k === i;
  });
  [...ui.list.children].forEach((li, k) => li.classList.toggle("selected", k === i));
  const p = current.paths[i];
  stopFlight();
  if (p && riding) setRide(false);
  if (p && !current.group.visible) {
    current.group.visible = true;
    ui.showPaths.checked = true;
    ui.pathSection.style.display = "block";
  }
  if (p) viewPath(p);
  // Cue the video to the moment the drone left the start of this trip.
  if (p?.real && current.rec) { ui.fvideo.pause(); ui.fvideo.currentTime = p.real.t0; }
  ui.fly.disabled = !p;
  const legs = p?.real && Math.max(p.real.from_snap_m, p.real.to_snap_m) > 0.05;
  const longer = p?.full && p.full.duration_s > p.duration_s + 0.5;
  ui.detail.innerHTML = p
    ? `<b>SAFE</b> · ${p.length_m.toFixed(1)} m, closest ${p.min_clearance_m.toFixed(2)} m to anything` +
      (longer ? "<br>Faded ends (takeoff, landing) aren't certified." : "") +
      (p.real ? `<br>White: the ${p.real.flown_m.toFixed(0)} m the drone really flew${p.real.unknown_s > 0 ? `, plus ${p.real.unknown_s.toFixed(1)} s with no pose` : ""}.` : "") +
      (p.avoidance && !p.real ? `<br>Dashed: the straight line, through ${p.avoidance.obstacles} obstacle${p.avoidance.obstacles === 1 ? "" : "s"}.` : "") +
      (p.weave ? `<br>Threads ${p.weave.gates} gaps.` : "")
    : "";
}

// Playback: a marker the size of the drone (radius + margin) moves along
// the selected path in real time. The camera stays where it is, at a view
// the drone actually filmed, because a splat looks wrong from far-off angles.
function stopFlight() {
  if (!flight) return;
  scene.remove(flight.marker);
  flight = null;
  ui.fly.classList.remove("on");
  ui.fly.setAttribute("aria-label", "Play path");
}

ui.fly.addEventListener("click", () => {
  if (flight) { stopFlight(); return; }
  const path = current?.paths[selected];
  if (!path) return;
  const r = current.info.robot_radius_m + current.info.margin_m;
  const marker = new THREE.Group();
  marker.add(new THREE.Mesh(new THREE.SphereGeometry(current.info.robot_radius_m, 20, 14),
    new THREE.MeshBasicMaterial({ color: 0xffffff })));
  marker.add(new THREE.Mesh(new THREE.SphereGeometry(r, 24, 16),
    new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.2, depthWrite: false })));
  scene.add(marker);
  flight = { path: path.full ?? path, start: performance.now(), marker };
  ui.fly.classList.add("on");
  ui.fly.setAttribute("aria-label", "Stop");
});

function updateFlight() {
  if (!flight) return;
  const t = (performance.now() - flight.start) / 1000;
  flight.marker.position.copy(at(flight.path, t));
  if (t > flight.path.duration_s + 0.8) stopFlight();
}

function viewPath(p) {
  // Frame the whole route, ends included, from above and behind.
  const box = new THREE.Box3();
  for (const q of (p.full ?? p).samples) box.expandByPoint(new THREE.Vector3(q[1], q[2], q[3]));
  const c = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const r = Math.max(size.x, size.y, 2.0);
  camera.fov = 50;
  camera.updateProjectionMatrix();
  controls.target.copy(c);
  camera.position.set(c.x - 0.2 * r, c.y - 1.15 * r, c.z + 0.95 * r + 0.5);
  controls.update();
}


// ---- Recovered flight path ------------------------------------------------
// flight.json poses: [t, x, y, z, fx, fy, fz, ux, uy, uz, key, match]. The video is
// the clock: the pose shown is the one at video.currentTime. Between two
// poses closer than gap_s apart the pose is interpolated; in a gap there is
// no pose, and the marker hides rather than guess.

const RAMP = [0x3cc8ff, 0x7dff6a, 0xffd23c, 0xff5a3c].map((c) => new THREE.Color(c));
function rampColor(u) {
  const x = Math.min(Math.max(u, 0), 1) * (RAMP.length - 1);
  const i = Math.min(Math.floor(x), RAMP.length - 2);
  return RAMP[i].clone().lerp(RAMP[i + 1], x - i);
}

function fmtTime(t) {
  const m = Math.floor(t / 60);
  return `${m}:${(t - 60 * m).toFixed(1).padStart(4, "0")}`;
}

// Pose at time t, or null inside a gap or outside the recovered span.
function poseAt(rec, t) {
  const p = rec.poses;
  if (!p.length || t < p[0][0] - rec.gap_s || t > p[p.length - 1][0] + rec.gap_s) return null;
  let lo = 0;
  let hi = p.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (p[mid][0] <= t) lo = mid; else hi = mid;
  }
  const a = p[lo];
  const b = p[hi];
  let u;
  if (t <= a[0]) u = 0;
  else if (t >= b[0]) u = 1;
  else if (b[0] - a[0] > rec.gap_s) return null;
  else u = (t - a[0]) / (b[0] - a[0]);
  const v = (k) => new THREE.Vector3(a[k] + u * (b[k] - a[k]), a[k + 1] + u * (b[k + 1] - a[k + 1]), a[k + 2] + u * (b[k + 2] - a[k + 2]));
  return { pos: v(1), fwd: v(4).normalize(), up: v(7).normalize() };
}

// Continuous runs of poses (split where they are more than gap_s apart).
function runs(rec) {
  const out = [];
  let cur = [];
  rec.poses.forEach((p, i) => {
    if (i && p[0] - rec.poses[i - 1][0] > rec.gap_s) { out.push(cur); cur = []; }
    cur.push(p);
  });
  if (cur.length) out.push(cur);
  return out;
}

function buildFlight(rec) {
  const group = new THREE.Group();
  const trail = new THREE.Group(); // the drawn path (hidden in drone view)
  group.add(trail);
  const T = Math.max(rec.duration_s, 1e-6);
  const rs = runs(rec);
  for (const run of rs) {
    if (run.length < 2) continue;
    const pts = run.map((p) => new THREE.Vector3(p[1], p[2], p[3]));
    const curve = new THREE.CatmullRomCurve3(pts);
    const segs = Math.max(8, run.length * 2);
    const radial = 6;
    const geo = new THREE.TubeGeometry(curve, segs, 0.022, radial, false);
    const colors = [];
    for (let i = 0; i <= segs; i++) {
      const t = run[0][0] + (i / segs) * (run[run.length - 1][0] - run[0][0]);
      const c = rampColor(t / T);
      for (let j = 0; j <= radial; j++) colors.push(c.r, c.g, c.b);
    }
    geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    // Drawn over the splat: from above, the ceiling would hide it.
    const tube = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ vertexColors: true, depthTest: false, transparent: true, opacity: 0.9 }));
    tube.renderOrder = 10;
    trail.add(tube);
  }
  // Gaps: where the camera went is unknown. A faint dashed line joins the
  // last pose before the gap to the first one after it.
  for (let k = 1; k < rs.length; k++) {
    const a = rs[k - 1][rs[k - 1].length - 1];
    const b = rs[k][0];
    const line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(a[1], a[2], a[3]), new THREE.Vector3(b[1], b[2], b[3])]),
      new THREE.LineDashedMaterial({ color: 0xffffff, dashSize: 0.06, gapSize: 0.08, transparent: true, opacity: 0.6, depthTest: false }),
    );
    line.renderOrder = 10;
    line.computeLineDistances();
    trail.add(line);
  }
  // The camera itself: a small frustum (16:9) that follows the video.
  const d = 0.22;
  const h = d * Math.tan(THREE.MathUtils.degToRad(rec.fov_deg / 2));
  const w = h * 16 / 9;
  const corners = [[-w, h], [w, h], [w, -h], [-w, -h]].map(([x, y]) => new THREE.Vector3(x, y, -d));
  const o = new THREE.Vector3();
  const seg = [];
  for (let i = 0; i < 4; i++) seg.push(o, corners[i], corners[i], corners[(i + 1) % 4]);
  seg.push(new THREE.Vector3(-w * 0.4, h, -d), new THREE.Vector3(0, h * 1.5, -d), new THREE.Vector3(0, h * 1.5, -d), new THREE.Vector3(w * 0.4, h, -d));
  const frustum = new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(seg),
    new THREE.LineBasicMaterial({ color: 0xffffff, depthTest: false, transparent: true }),
  );
  frustum.renderOrder = 11;
  const dot = new THREE.Mesh(new THREE.SphereGeometry(0.03, 12, 8), new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false, transparent: true }));
  dot.renderOrder = 11;
  frustum.add(dot);
  frustum.visible = false;
  group.add(frustum);
  return { group, trail, frustum };
}

let riding = false;
let sliderDrag = false;

function syncTrail() {
  const rec = current?.rec;
  if (rec) rec.view.group.visible = ui.showFlight.checked || riding;
  if (rec) rec.view.trail.visible = !riding;
}

function placeFrustum(rec, pose) {
  const f = rec.view.frustum;
  f.visible = !!pose && !riding;
  if (!pose) return;
  // three.js cameras look down -Z with +Y up; build that basis from the pose.
  const z = pose.fwd.clone().negate();
  const x = new THREE.Vector3().crossVectors(pose.up, z).normalize();
  const y = new THREE.Vector3().crossVectors(z, x);
  f.matrix.makeBasis(x, y, z).setPosition(pose.pos);
  f.matrixAutoUpdate = false;
  f.matrixWorldNeedsUpdate = true;
}

function updateRecorded() {
  const rec = current?.rec;
  if (!rec) return;
  const t = ui.fvideo.currentTime || 0;
  const pose = poseAt(rec, t);
  placeFrustum(rec, pose);
  if (riding && pose) {
    camera.position.copy(pose.pos);
    camera.up.copy(pose.up);
    camera.lookAt(pose.pos.clone().add(pose.fwd));
  }
  if (!sliderDrag) ui.fslider.value = Math.round((1000 * t) / Math.max(rec.duration_s, 1e-6));
  ui.ftime.textContent = `${fmtTime(t)} / ${fmtTime(rec.duration_s)}` +
    (pose ? (pose.pos.z >= 0 ? ` · ${pose.pos.z.toFixed(2)} m above the floor` : "") : " · no pose here (gap)");
  ui.fvideoCap.textContent = pose ? "Drone video" : "Drone video · position unknown here";
}

// One place that says which camera is live: the switch, the badge on the
// canvas, and the hint (dragging does nothing while riding the drone).
function updateViewUI() {
  const has = !!current?.rec;
  ui.camFree.setAttribute("aria-pressed", String(!riding));
  ui.camDrone.setAttribute("aria-pressed", String(riding));
  ui.badge.style.display = has ? "flex" : "none";
  ui.badge.classList.toggle("drone", riding);
  ui.badgeText.textContent = riding ? "Drone's view" : "Free view";
  ui.hint.textContent = riding
    ? "Following the drone camera · switch to Free to look around"
    : "Drag to orbit · right-drag to pan · scroll to zoom";
}

function setRide(on) {
  const rec = current?.rec;
  riding = on && !!rec;
  controls.enabled = !riding;
  syncTrail();
  updateViewUI();
  if (riding) {
    stopFlight();
    camera.fov = rec.fov_deg;
    // Training leaves faint floaters right in front of the camera; clip
    // the first 0.2 m like the CPU renderer used for the pose check does.
    camera.near = 0.2;
    camera.updateProjectionMatrix();
  } else {
    camera.near = 0.01;
    camera.updateProjectionMatrix();
    // Hand the view back to the orbit controls without a jump.
    const dir = new THREE.Vector3();
    camera.getWorldDirection(dir);
    camera.up.set(0, 0, 1);
    controls.target.copy(camera.position).addScaledVector(dir, 2);
    controls.update();
  }
}

function overview() {
  const rec = current?.rec;
  if (!rec) return;
  setRide(false);
  stopFlight();
  const box = new THREE.Box3();
  for (const p of rec.poses) box.expandByPoint(new THREE.Vector3(p[1], p[2], p[3]));
  const c = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const r = Math.max(size.x, size.y, 2.0);
  camera.fov = 50;
  camera.updateProjectionMatrix();
  controls.target.set(c.x, c.y, 0.6);
  camera.position.set(c.x - 0.15 * r, c.y - 0.55 * r, 1.0 * r + 1.5);
  controls.update();
}

function describeFlight(rec) {
  const gap = rec.gaps.reduce((sum, [a, b]) => sum + b - a, 0);
  return `${rec.length_m.toFixed(0)} m recovered from a ${fmtTime(rec.duration_s)} flight` +
    (gap > 0 ? `, ${gap.toFixed(1)} s with no pose` : "") +
    `. <span class="legend"></span> start → end`;
}

ui.fplay.addEventListener("click", () => {
  if (!current?.rec) return;
  if (ui.fvideo.paused) ui.fvideo.play(); else ui.fvideo.pause();
});
function syncPlayBtn() {
  const playing = !ui.fvideo.paused;
  ui.fplay.classList.toggle("on", playing);
  ui.fplay.setAttribute("aria-label", playing ? "Pause" : "Play");
}
ui.fvideo.addEventListener("play", syncPlayBtn);
ui.fvideo.addEventListener("pause", syncPlayBtn);
ui.camFree.addEventListener("click", () => setRide(false));
ui.camDrone.addEventListener("click", () => setRide(true));
ui.fslider.addEventListener("input", () => {
  if (!current?.rec) return;
  sliderDrag = true;
  ui.fvideo.currentTime = (ui.fslider.value / 1000) * current.rec.duration_s;
});
ui.fslider.addEventListener("change", () => { sliderDrag = false; });

async function loadScene(name) {
  const token = ++loadToken;
  stopFlight();
  setRide(false);
  ui.fvideo.pause();
  if (current) {
    scene.remove(current.splat, current.group, current.cams);
    if (current.rec) scene.remove(current.rec.view.group);
    current.splat.dispose();
    current = null;
  }
  ui.loading.style.display = "block";
  ui.loading.textContent = "Loading splat…";
  const base = `data/${name}/`;
  let info, allPaths;
  try {
    [info, allPaths] = await Promise.all(["scene.json", "paths.json"].map(async (f) => {
      const r = await fetch(base + f);
      if (!r.ok) throw new Error(`${f}: HTTP ${r.status}`);
      return r.json();
    }));
  } catch (e) {
    if (token === loadToken) ui.loading.textContent = `No data for this scene (${e.message}). Scene data isn't in the repo: see web/README.md to export it.`;
    return;
  }
  const paths = [...allPaths].sort((a, b) => pathScore(b) - pathScore(a)).slice(0, MAX_PATHS);
  paths.forEach((p) => { p.full = withEnds(p); });
  const rec = info.flight ? await fetch(base + "flight.json").then((r) => r.json()) : null;
  if (token !== loadToken) return;
  const splat = new SplatMesh({
    url: base + "scene.splat",
    onProgress: (e) => {
      if (e.lengthComputable) ui.loading.textContent = `Loading splat… ${Math.round((100 * e.loaded) / e.total)}%`;
    },
  });
  const group = buildPaths(info, paths, rec);
  const cams = buildCams(info);
  scene.add(splat, group, cams);
  current = { name, info, paths, splat, group, cams, home: homeView(info), rec };
  if (rec) {
    rec.view = buildFlight(rec);
    scene.add(rec.view.group);
  }
  ui.flightSection.style.display = rec ? "block" : "none";
  ui.showFlight.parentElement.style.display = rec ? "block" : "none";
  syncTrail();
  // The flight path is the main thing; planned paths are an opt-in layer.
  ui.showPaths.checked = !rec && paths.length > 0;
  ui.showPaths.disabled = !paths.length;
  group.visible = ui.showPaths.checked;
  ui.pathSection.style.display = ui.showPaths.checked ? "block" : "none";
  ui.fvideoBox.style.display = rec ? "block" : "none";
  if (rec) {
    ui.flightFacts.innerHTML = describeFlight(rec);
    ui.fvideo.src = base + "video.mp4";
    ui.fvideo.currentTime = 0;
  } else {
    ui.fvideo.removeAttribute("src");
    ui.fvideo.load();
  }

  const tab = document.querySelector(`#tabs button[data-scene="${name}"]`);
  ui.title.textContent = tab?.dataset.title || info.title || info.name;
  ui.facts.textContent = describe(info);
  ui.list.innerHTML = "";
  paths.forEach((p, i) => {
    const li = document.createElement("li");
    const hex = "#" + COLORS[i % COLORS.length].toString(16).padStart(6, "0");
    li.innerHTML = `<span class="swatch" style="background:${hex}"></span>${tripName(p) ?? `Path ${i + 1}`} · ${p.length_m.toFixed(1)} m`;
    li.addEventListener("click", () => select(i === selected ? -1 : i));
    ui.list.appendChild(li);
  });
  select(-1);
  goHome();
  updateViewUI();

  await splat.initialized;
  if (token === loadToken) ui.loading.style.display = "none";
}

document.querySelectorAll("#tabs button").forEach((b) => {
  b.addEventListener("click", () => {
    document.querySelectorAll("#tabs button").forEach((x) => x.classList.toggle("active", x === b));
    loadScene(b.dataset.scene);
  });
});
document.getElementById("reset").addEventListener("click", goHome);
ui.showPaths.addEventListener("change", () => {
  if (!current) return;
  current.group.visible = ui.showPaths.checked;
  ui.pathSection.style.display = ui.showPaths.checked ? "block" : "none";
  if (!ui.showPaths.checked) select(-1);
  // One layer at a time reads best: the two kinds of path overlap.
  ui.showFlight.checked = !ui.showPaths.checked;
  syncTrail();
});
ui.showFlight.addEventListener("change", syncTrail);
ui.showCams.addEventListener("change", () => {
  if (current) current.cams.visible = ui.showCams.checked;
});

resize();
loadScene("loft");
renderer.setAnimationLoop(() => {
  updateFlight();
  updateRecorded();
  if (!riding) controls.update();
  renderer.render(scene, camera);
});
