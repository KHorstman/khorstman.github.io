"""Schematic: observing the spectrum of a K giant (with a candidate
Neptune-mass planet) next to an F star, with a laser frequency comb below,
on a space-themed background."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle
from matplotlib.colors import hsv_to_rgb, to_rgb

fig, ax = plt.subplots(figsize=(15, 7.5))
ax.set_xlim(0, 15)
ax.set_ylim(0, 7.5)
ax.set_aspect("equal")
ax.axis("off")

# --- Palette ------------------------------------------------------------------
BG_DARK, BG_LIGHT = "#070b1f", "#1c2452"
TEXT, LINE = "#e6e9ff", "#8f9bd0"
K_FACE, K_EDGE, K_GLOW = "#ef7a45", "#ffb48a", "#ff9a5c"
F_FACE, F_EDGE, F_GLOW = "#fff3da", "#fffaf0", "#ffe9b8"
P_FACE, P_EDGE = "#5c86e6", "#b7caff"
CONE = "#ffb48a"
PANEL_FACE, PANEL_EDGE, SPEC = "#10173a", "#6d79b8", "#f6eadb"

# --- Space background: soft radial gradient plus a field of stars ------------
gx, gy = np.meshgrid(np.linspace(0, 15, 600), np.linspace(0, 7.5, 300))
r = np.hypot((gx - 11.5) / 15, (gy - 5) / 7.5)
mix = np.clip(1 - r / 0.9, 0, 1)[..., None] ** 1.5
c0 = np.array(to_rgb(BG_DARK)); c1 = np.array(to_rgb(BG_LIGHT))
ax.imshow(c0 + (c1 - c0) * mix, extent=(0, 15, 0, 7.5), origin="lower", zorder=-10)
srng = np.random.default_rng(3)
n_bg = 450
sxs, sys_ = srng.uniform(0, 15, n_bg), srng.uniform(0, 7.5, n_bg)
star_cols = srng.choice(["#ffffff", "#cfdcff", "#fff1d6", "#ffd9c2"], n_bg, p=[.5, .25, .15, .1])
sizes = np.minimum(srng.pareto(2.5, n_bg) * 1.5 + 0.3, 6)
ax.scatter(sxs, sys_, s=sizes, c=star_cols, alpha=srng.uniform(0.35, 1, n_bg), lw=0, zorder=-9)
bright = sizes > 3.5
ax.scatter(sxs[bright], sys_[bright], s=sizes[bright] * 4, c=star_cols[bright], alpha=0.08, lw=0, zorder=-9)

# --- Stars ------------------------------------------------------------------
kx, ky, kr = 12.0, 5.2, 1.45          # K giant
fx, fy, fr = 13.75, 3.35, 0.55        # F star
for k in range(12):                                     # layered glows
    ax.add_patch(Circle((kx, ky), kr * (1 + 0.04 * (k + 1)), color=K_GLOW, alpha=0.045, lw=0, zorder=2.5))
    ax.add_patch(Circle((fx, fy), fr * (1 + 0.08 * (k + 1)), color=F_GLOW, alpha=0.05, lw=0, zorder=3))
ax.add_patch(Circle((kx, ky), kr, facecolor=K_FACE, edgecolor=K_EDGE, lw=1.5, zorder=3))
ax.add_patch(Circle((kx - 0.25, ky + 0.25), kr * 0.7, color="#ffa36b", alpha=0.35, lw=0, zorder=3))
ax.add_patch(Circle((fx, fy), fr, facecolor=F_FACE, edgecolor=F_EDGE, lw=1.2, zorder=4))
ax.text(kx, ky - kr - 0.35, "K giant", ha="center", va="top", fontsize=15, weight="bold", color=TEXT)
ax.text(fx, fy - fr - 0.2, "F star", ha="center", va="top", fontsize=13, weight="bold", color=TEXT)

# Orbit of the planet (tilted ellipse); back half behind the star, front half in front
orb_w, orb_h, tilt = 4.3, 1.0, 12
theta = np.linspace(0, 2 * np.pi, 400)
ox = orb_w / 2 * np.cos(theta)
oy = orb_h / 2 * np.sin(theta)
t = np.deg2rad(tilt)
rx, ry = kx + ox * np.cos(t) - oy * np.sin(t), ky + ox * np.sin(t) + oy * np.cos(t)
back = np.sin(theta) > 0
ax.plot(np.where(back, rx, np.nan), np.where(back, ry, np.nan), color=LINE, lw=1.2, ls="--", zorder=2)
ax.plot(np.where(~back, rx, np.nan), np.where(~back, ry, np.nan), color=LINE, lw=1.2, ls="--", zorder=5)

# Neptune-mass planet with a question mark
pa = np.deg2rad(305)
px = kx + orb_w / 2 * np.cos(pa) * np.cos(t) - orb_h / 2 * np.sin(pa) * np.sin(t)
py = ky + orb_w / 2 * np.cos(pa) * np.sin(t) + orb_h / 2 * np.sin(pa) * np.cos(t)
ax.add_patch(Circle((px, py), 0.33, facecolor=P_FACE, edgecolor=P_EDGE, lw=1.5, zorder=6))
ax.text(px, py - 0.02, "?", ha="center", va="center", fontsize=20, weight="bold", color="white", zorder=7)
ax.annotate("Neptune-mass\nplanet?", xy=(px + 0.15, py + 0.3), xytext=(px + 0.5, py + 1.75),
            ha="center", fontsize=12, color=TEXT, arrowprops=dict(arrowstyle="-", color=LINE, lw=1))

# --- Spectrum panel -----------------------------------------------------------
bx0, by0, bw, bh = 0.4, 0.4, 7.2, 4.6
# Light "pyramid" from the K giant to the panel (as in the lux diagram)
apex = (kx - 0.9, ky - 0.25)
corners = [(bx0, by0 + bh), (bx0 + bw, by0 + bh), (bx0 + bw, by0), (bx0, by0)]
ax.add_patch(Polygon([apex, corners[0], corners[1]], color=CONE, alpha=0.08, lw=0, zorder=0))
ax.add_patch(Polygon([apex, corners[1], corners[2]], color=CONE, alpha=0.15, lw=0, zorder=0))
for c in corners:
    ax.plot([apex[0], c[0]], [apex[1], c[1]], ls=":", color=LINE, lw=1.2, zorder=1)

ax.add_patch(Rectangle((bx0, by0), bw, bh, facecolor=PANEL_FACE, edgecolor=PANEL_EDGE, lw=1.5, zorder=2))

# Approximate H-band spectrum of the K giant (shape traced from the data/model comparison)
rng = np.random.default_rng(7)
w0, w1 = 1.6432, 1.6657                                   # microns
wl = np.linspace(w0, w1, 4000)
knots_w = [1.6432, 1.646, 1.648, 1.650, 1.6535, 1.656, 1.658, 1.660, 1.6625, 1.6657]
knots_f = [7000, 8300, 9500, 10300, 10500, 10200, 8800, 7400, 6200, 4800]
flux = np.polyval(np.polyfit(knots_w, knots_f, 4), wl)    # smooth continuum (blaze-like hump)
strong = [(1.6433, .12), (1.6438, .20), (1.6443, .40), (1.6450, .40), (1.6459, .15), (1.6469, .38),
          (1.6473, .12), (1.6484, .30), (1.6487, .15), (1.6490, .25), (1.6498, .25), (1.6502, .20),
          (1.6511, .20), (1.6519, .12), (1.6534, .20), (1.6555, .10), (1.6566, .10), (1.6574, .15),
          (1.6586, .25), (1.6592, .25), (1.6596, .20), (1.6600, .40), (1.6605, .15), (1.6611, .12),
          (1.6618, .12), (1.6625, .15), (1.6634, .45), (1.6637, .15), (1.6643, .12), (1.6650, .10)]
weak = [(l, d) for l, d in zip(rng.uniform(w0, w1, 90), rng.uniform(0.02, 0.08, 90))]
for l, d in strong + weak:
    flux *= 1 - d * np.exp(-0.5 * ((wl - l) / 0.00006) ** 2)
flux += rng.normal(0, 60, wl.size)                       # a little noise, like real data

sx0, sx1 = bx0 + 0.35, bx0 + bw - 0.35
sy0, sy1 = by0 + 1.55, by0 + bh - 0.3
xs = sx0 + (wl - w0) / (w1 - w0) * (sx1 - sx0)
ys = sy0 + (flux - flux.min()) / (flux.max() - flux.min()) * (sy1 - sy0)
ax.plot(xs, ys, color=SPEC, lw=1.0, zorder=3)

# Laser frequency comb that doubles as a ruler: the comb teeth rise from a ruler body,
# each tooth lining up with a major tick, with minor ticks in between
n_teeth = 34
tx = np.linspace(sx0 + 0.1, sx1 - 0.1, n_teeth)
step = tx[1] - tx[0]
rx0, rx1 = tx[0] - step / 2, tx[-1] + step / 2
ry0, rh = by0 + 0.18, 0.4                                 # ruler body
ry1 = ry0 + rh
hues = np.linspace(0.80, 0.0, n_teeth)                   # violet -> red
cols = hsv_to_rgb(np.stack([hues, np.full_like(hues, 0.55), np.ones_like(hues)], axis=-1))

body = FancyBboxPatch((rx0, ry0), rx1 - rx0, rh, boxstyle="round,pad=0,rounding_size=0.08",
                      facecolor="none", edgecolor=PANEL_EDGE, lw=1.3, zorder=6)
ax.add_patch(body)
grad_h = np.linspace(0.80, 0.0, 512)
grad = hsv_to_rgb(np.stack([grad_h, np.full_like(grad_h, 0.55), np.ones_like(grad_h)], axis=-1))[None]
im = ax.imshow(grad, extent=(rx0, rx1, ry0, ry1), aspect="auto", alpha=0.45, zorder=4)
im.set_clip_path(body)
ax.plot([rx0 + 0.05, rx1 - 0.05], [ry1 - 0.02, ry1 - 0.02], color="white", alpha=0.25, lw=1, zorder=5)  # glassy edge

for i, x in enumerate(np.linspace(rx0 + step / 2, rx1 - step / 2, 5 * (n_teeth - 1) + 1)):
    major = i % 5 == 0
    tick = 0.24 if major else (0.15 if i % 5 == 2 or i % 5 == 3 else 0.1)
    ax.plot([x, x], [ry1, ry1 - tick], color=TEXT, alpha=0.85 if major else 0.55,
            lw=1.1 if major else 0.7, zorder=6)

u = np.linspace(0, 1, n_teeth)
env = 0.12 + 0.88 * np.exp(-0.5 * ((u - 0.55) / 0.24) ** 2)
ch = 0.85
for x, h, c in zip(tx, env, cols):
    ax.plot([x, x], [ry1, ry1 + h * ch], color=c, lw=9, alpha=0.18, solid_capstyle="round", zorder=5)
    ax.plot([x, x], [ry1, ry1 + h * ch], color=c, lw=3.5, solid_capstyle="round", zorder=5)

out = __file__.replace("scripts/kgiant_spectrum.py", "images/kgiant_spectrum.png")
fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0, facecolor=BG_DARK)
print("saved", out)
