/**
 * Static fallback for Hero.astro: renders the canvas animation's exact first
 * frame (the state before the first walk burst) to hero_voronoi.png.
 *
 * This is a line-for-line port of the drawing code in ../Hero.astro — the
 * seeded RNG (mulberry32(7)), seed layout, walker selection, sub-partitions,
 * and draw passes are identical, so the PNG matches the canvas's static
 * frame. Everything the first frame shows is deterministic: Math.random()
 * only enters during walk bursts, which start WALK_EVERY seconds after load.
 *
 * Rendered at CSS width 1200 with devicePixelRatio 2 (the retina-desktop
 * code path in Hero.astro's resize()), giving a 2400x800 bitmap for the
 * blog page's widths=[768, 1536, 2400] srcset. Keep this in sync with
 * Hero.astro if the drawing logic there ever changes.
 *
 * Run (d3-delaunay is a repo dependency; the canvas lib is not):
 *   npm install --no-save @napi-rs/canvas
 *   node "content/posts/Work Order SIDs for Generative Recommendation/assets/hero_voronoi.mjs"
 */
import { createCanvas } from "@napi-rs/canvas";
import { Delaunay } from "d3-delaunay";
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// Deterministic RNG so the composition is stable across visits
function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const ASPECT = 3; // logical space is ASPECT x 1
const N = 150;
const WALKERS = 5;

// The retina-desktop path of Hero.astro's resize(): dpr-capped 2x bitmap
const CSS_W = 1200;
const DPR = 2;

const rand = mulberry32(7);

// Clustered candidate seeds (like a PCA projection of embeddings),
// thinned to a minimum separation so no degenerate sliver cells
const cands = [];
const centers = Array.from({ length: 8 }, () => [
  0.15 + rand() * (ASPECT - 0.3),
  0.12 + rand() * 0.76,
]);
for (const [cx, cy] of centers) {
  const n = 22 + Math.floor(rand() * 16);
  for (let i = 0; i < n; i++) {
    const gx = (rand() + rand() + rand() - 1.5) / 1.5;
    const gy = (rand() + rand() + rand() - 1.5) / 1.5;
    cands.push({ x: cx + gx * 0.26, y: cy + gy * 0.18 });
  }
}
for (let i = 0; i < 90; i++) cands.push({ x: rand() * ASPECT, y: rand() });
const seeds = [];
for (const c of cands) {
  if (seeds.length >= N) break;
  c.x = Math.min(ASPECT - 0.01, Math.max(0.01, c.x));
  c.y = Math.min(0.99, Math.max(0.01, c.y));
  if (seeds.every((s) => Math.hypot(s.x - c.x, s.y - c.y) > 0.085)) seeds.push(c);
}
for (const s of seeds) {
  s.hue = rand() * 360;
  s.light = 87 + rand() * 7;
  s.sat = 35 + rand() * 35;
  // satellite dots (ops) at fixed offsets from the seed
  s.dots = Array.from({ length: 2 + Math.floor(rand() * 3) }, () => [
    (rand() - 0.5) * 0.09,
    (rand() - 0.5) * 0.06,
  ]);
}

// Static logical-space geometry: areas for the initial walker selection
const points = seeds.map((s) => [s.x, s.y]);
const delaunayL = Delaunay.from(points);
const vorL = delaunayL.voronoi([0, 0, ASPECT, 1]);
const byArea = seeds
  .map((s, i) => {
    const poly = vorL.cellPolygon(i);
    let a = 0;
    if (poly)
      for (let k = 0; k < poly.length - 1; k++)
        a += poly[k][0] * poly[k + 1][1] - poly[k + 1][0] * poly[k][1];
    return { i, area: Math.abs(a) / 2 };
  })
  .sort((p, q) => q.area - p.area);

// Walkers start on large, well-separated cells — these are the first
// frame's selected (subdivided) cells
const walkers = [];
for (const { i } of byArea) {
  if (walkers.length >= WALKERS) break;
  const s = seeds[i];
  if (
    walkers.every((w) => Math.hypot(seeds[w.cell].x - s.x, seeds[w.cell].y - s.y) > 0.55)
  )
    walkers.push({ cell: i });
}

// Sub-partition seeds per cell: offsets are FRACTIONS of the cell's
// half-extents so one recipe fits any cell it lands on
const subFor = (i) => {
  const r = mulberry32(1000 + i);
  return Array.from({ length: 8 }, () => ({
    fx: (r() - 0.5) * 1.7,
    fy: (r() - 0.5) * 1.7,
    light: 66 + r() * 22,
    dots: [
      [0, 0],
      ...Array.from({ length: 1 + Math.floor(r() * 3) }, () => [
        (r() - 0.5) * 0.28,
        (r() - 0.5) * 0.28,
      ]),
    ],
  }));
};

const cssW = CSS_W;
const cssH = CSS_W / ASPECT;
const canvas = createCanvas(Math.round(cssW * DPR), Math.round(cssH * DPR));
const ctx = canvas.getContext("2d");
ctx.setTransform(DPR, 0, 0, DPR, 0, 0);

const S = cssW / ASPECT;
const pxPts = points.map(([x, y]) => [x * S, y * S]);
const vorPx = Delaunay.from(pxPts).voronoi([0, 0, cssW, cssH]);

const tracePoly = (c, poly) => {
  c.beginPath();
  c.moveTo(poly[0][0], poly[0][1]);
  for (const [x, y] of poly) c.lineTo(x, y);
  c.closePath();
};

const drawDots = (c, x, y, offsets, sx, sy, hue) => {
  c.fillStyle = `hsla(${hue}, 70%, 45%, 0.85)`;
  for (const [dx, dy] of offsets) {
    c.beginPath();
    c.arc(x + dx * sx, y + dy * sy, 1.4, 0, Math.PI * 2);
    c.fill();
  }
};

// Base layer: tessellation + dots (Hero.astro renders this into an
// offscreen canvas; here it draws straight onto the output)
ctx.lineJoin = "round";
for (let i = 0; i < seeds.length; i++) {
  const poly = vorPx.cellPolygon(i);
  if (!poly) continue;
  const s = seeds[i];
  tracePoly(ctx, poly);
  ctx.fillStyle = `hsl(${s.hue}, ${s.sat}%, ${s.light}%)`;
  ctx.fill();
  ctx.strokeStyle = "#3a3a3a";
  ctx.lineWidth = 0.8;
  ctx.stroke();
}
for (let i = 0; i < seeds.length; i++)
  drawDots(ctx, pxPts[i][0], pxPts[i][1], seeds[i].dots, S, S, seeds[i].hue);

// Each walker's cell: bold outline + sub-partition
function drawSelection(i) {
  const poly = vorPx.cellPolygon(i);
  if (!poly) return;
  const s = seeds[i];
  let minx = 1e9, miny = 1e9, maxx = -1e9, maxy = -1e9;
  for (const [x, y] of poly) {
    minx = Math.min(minx, x);
    miny = Math.min(miny, y);
    maxx = Math.max(maxx, x);
    maxy = Math.max(maxy, y);
  }
  const cx = pxPts[i][0];
  const cy = pxPts[i][1];
  const hw = (maxx - minx) / 2;
  const hh = (maxy - miny) / 2;
  const subSeeds = subFor(i);
  const sp = subSeeds.map((u) => [cx + u.fx * hw, cy + u.fy * hh]);

  ctx.save();
  tracePoly(ctx, poly);
  ctx.clip();
  const sub = Delaunay.from(sp).voronoi([minx, miny, maxx, maxy]);
  for (let k = 0; k < sp.length; k++) {
    const q = sub.cellPolygon(k);
    if (!q) continue;
    tracePoly(ctx, q);
    ctx.fillStyle = `hsl(${s.hue}, 55%, ${subSeeds[k].light}%)`;
    ctx.fill();
    ctx.strokeStyle = "#3a3a3a";
    ctx.lineWidth = 0.6;
    ctx.stroke();
  }
  // each sub-partition's own ops, drawn after every sub-cell fill so a
  // neighboring fill can't paint over a dot near a shared edge
  for (let k = 0; k < sp.length; k++)
    drawDots(ctx, sp[k][0], sp[k][1], subSeeds[k].dots, hw, hh, s.hue);
  // the sub-cells also painted over the base layer's op dots: redraw
  // every dot that can land in this cell (still clipped to it)
  const pad = 0.1 * S; // max dot offset from its seed
  for (let j = 0; j < seeds.length; j++) {
    const [jx, jy] = pxPts[j];
    if (jx < minx - pad || jx > maxx + pad || jy < miny - pad || jy > maxy + pad)
      continue;
    drawDots(ctx, jx, jy, seeds[j].dots, S, S, seeds[j].hue);
  }
  ctx.restore();
  tracePoly(ctx, poly);
  ctx.strokeStyle = "#1a1a1a";
  ctx.lineWidth = 1.8;
  ctx.stroke();
}

ctx.lineJoin = "round";
for (const w of walkers) drawSelection(w.cell);

const out = join(dirname(fileURLToPath(import.meta.url)), "hero_voronoi.png");
writeFileSync(out, canvas.toBuffer("image/png"));
console.log(out, `${canvas.width}x${canvas.height}`);
