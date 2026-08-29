"""Decorative hero banner: hierarchical Voronoi tessellation (the RQ-VAE motif).

Purely generative — no project data. Level-0 cells in the pastel palette of the
codebook figure; a few cells subdivided into level-1 sub-cells shaded in
lightness variations of the parent hue (residual quantization as recursive
partition). No text, no legend, banner aspect.
"""
import colorsys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import Voronoi
from shapely.geometry import Point, Polygon, box

rng = np.random.default_rng(7)

W, H = 3.0, 1.0  # banner aspect
BBOX = box(0, 0, W, H)

# clustered seeds -> varied cell sizes, like a PCA projection of embeddings
centers = rng.uniform([0.1, 0.1], [W - 0.1, H - 0.1], size=(9, 2))
weights = rng.uniform(0.4, 1.6, size=9)
pts = []
for c, w in zip(centers, weights):
    n = int(22 * w)
    pts.append(rng.normal(c, [0.22, 0.13], size=(n, 2)))
pts.append(rng.uniform([0, 0], [W, H], size=(40, 2)))
pts = np.vstack(pts)
pts = pts[(pts[:, 0] > 0.01) & (pts[:, 0] < W - 0.01) & (pts[:, 1] > 0.01) & (pts[:, 1] < H - 0.01)]

# mirror points across edges so border cells close cleanly
mirrored = [pts]
for axis, lo, hi in [(0, 0, W), (1, 0, H)]:
    for bound in (lo, hi):
        m = pts.copy()
        m[:, axis] = 2 * bound - m[:, axis]
        mirrored.append(m)
vor = Voronoi(np.vstack(mirrored))

# pastel palette sampled like the figure: high lightness, low-mid saturation, full hue wheel
def pastel(h, light=0.90, sat=0.55):
    r, g, b = colorsys.hls_to_rgb(h, light, sat)
    return (r, g, b)

fig, ax = plt.subplots(figsize=(15, 5), dpi=200)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.subplots_adjust(0, 0, 1, 1)

cells = []
for i in range(len(pts)):
    region = vor.regions[vor.point_region[i]]
    if -1 in region or len(region) == 0:
        continue
    poly = Polygon(vor.vertices[region]).intersection(BBOX)
    if poly.is_empty or poly.area < 1e-6:
        continue
    cells.append((i, poly))

hues = rng.uniform(0, 1, size=len(pts))
areas = np.array([p.area for _, p in cells])

for i, poly in cells:
    face = pastel(hues[i], light=rng.uniform(0.87, 0.94), sat=rng.uniform(0.35, 0.7))
    ax.add_patch(MplPolygon(np.asarray(poly.exterior.coords), facecolor=face,
                            edgecolor="#3a3a3a", linewidth=0.7, joinstyle="round"))

# subdivide a few of the larger cells: level-1 sub-partition in shades of parent hue
big = [cells[j] for j in np.argsort(areas)[-14:]]
chosen = [big[j] for j in rng.choice(len(big), size=6, replace=False)]
for i, poly in chosen:
    minx, miny, maxx, maxy = poly.bounds
    sub_pts = []
    while len(sub_pts) < 9:
        p = rng.uniform([minx, miny], [maxx, maxy])
        if poly.contains(Point(p[0], p[1]).buffer(0.001)):
            sub_pts.append(p)
    sub_pts = np.array(sub_pts)
    # mirror sub-seeds across the cell bbox for closed regions, then clip to the cell
    sm = [sub_pts]
    for axis, lo, hi in [(0, minx, maxx), (1, miny, maxy)]:
        for bound in (lo, hi):
            m = sub_pts.copy()
            m[:, axis] = 2 * bound - m[:, axis]
            sm.append(m)
    svor = Voronoi(np.vstack(sm))
    shades = np.linspace(0.62, 0.88, len(sub_pts))
    rng.shuffle(shades)
    for k in range(len(sub_pts)):
        region = svor.regions[svor.point_region[k]]
        if -1 in region or len(region) == 0:
            continue
        sp = Polygon(svor.vertices[region]).intersection(poly)
        if sp.is_empty:
            continue
        geoms = sp.geoms if sp.geom_type == "MultiPolygon" else [sp]
        for g in geoms:
            if g.geom_type != "Polygon" or g.area < 1e-7:
                continue
            face = pastel(hues[i], light=shades[k], sat=0.55)
            ax.add_patch(MplPolygon(np.asarray(g.exterior.coords), facecolor=face,
                                    edgecolor="#3a3a3a", linewidth=0.5, joinstyle="round"))
    # heavier outline on the subdivided (zoomed) cell, like the figure's selected cell
    ax.add_patch(MplPolygon(np.asarray(poly.exterior.coords), facecolor="none",
                            edgecolor="#1a1a1a", linewidth=1.6, joinstyle="round"))

# sparse saturated dots: ops living in the cells
dot_n = 700
dots = rng.uniform([0, 0], [W, H], size=(dot_n, 2))

dot_c = []
keep = []
for d in dots:
    p = Point(d)
    for i, poly in cells:
        if poly.contains(p):
            r, g, b = colorsys.hls_to_rgb(hues[i], 0.45, 0.75)
            dot_c.append((r, g, b))
            keep.append(d)
            break
keep = np.array(keep)
ax.scatter(keep[:, 0], keep[:, 1], s=2.2, c=dot_c, alpha=0.85, linewidths=0)

out = __file__.replace(".py", ".png")
fig.savefig(out, dpi=200)
print(out)
