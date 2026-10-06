"""Schematic: observing the spectrum of a K giant (with a candidate
Neptune-mass planet) next to an F star, with a wavelength "ruler" below."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle
from matplotlib.colors import hsv_to_rgb

fig, ax = plt.subplots(figsize=(15, 7.5))
ax.set_xlim(0, 15)
ax.set_ylim(0, 7.5)
ax.set_aspect("equal")
ax.axis("off")

# --- Stars ------------------------------------------------------------------
kx, ky, kr = 12.0, 5.2, 1.45          # K giant
fx, fy, fr = 13.75, 3.35, 0.55        # F star
ax.add_patch(Circle((kx, ky), kr * 1.08, color="#e0662a", alpha=0.25, lw=0))
ax.add_patch(Circle((kx, ky), kr, facecolor="#d9531e", edgecolor="#5a2a10", lw=1.5, zorder=3))
ax.add_patch(Circle((fx, fy), fr * 1.15, color="#fff4c9", alpha=0.6, lw=0, zorder=3))
ax.add_patch(Circle((fx, fy), fr, facecolor="#fff2bf", edgecolor="#555", lw=1.2, zorder=4))
ax.text(kx, ky - kr - 0.35, "K giant", ha="center", va="top", fontsize=15, weight="bold")
ax.text(fx, fy - fr - 0.2, "F star", ha="center", va="top", fontsize=13, weight="bold")

# Orbit of the planet (tilted ellipse); back half behind the star, front half in front
orb_w, orb_h, tilt = 4.3, 1.0, 12
theta = np.linspace(0, 2 * np.pi, 400)
ox = orb_w / 2 * np.cos(theta)
oy = orb_h / 2 * np.sin(theta)
t = np.deg2rad(tilt)
rx, ry = kx + ox * np.cos(t) - oy * np.sin(t), ky + ox * np.sin(t) + oy * np.cos(t)
back = np.sin(theta) > 0
ax.plot(np.where(back, rx, np.nan), np.where(back, ry, np.nan), color="#444", lw=1.2, ls="--", zorder=2)
ax.plot(np.where(~back, rx, np.nan), np.where(~back, ry, np.nan), color="#444", lw=1.2, ls="--", zorder=5)

# Neptune-mass planet with a question mark
pa = np.deg2rad(305)
px = kx + orb_w / 2 * np.cos(pa) * np.cos(t) - orb_h / 2 * np.sin(pa) * np.sin(t)
py = ky + orb_w / 2 * np.cos(pa) * np.sin(t) + orb_h / 2 * np.sin(pa) * np.cos(t)
ax.add_patch(Circle((px, py), 0.33, facecolor="#3d6fd6", edgecolor="#1b2f66", lw=1.5, zorder=6))
ax.text(px, py - 0.02, "?", ha="center", va="center", fontsize=20, weight="bold", color="white", zorder=7)
ax.annotate("Neptune-mass\nplanet?", xy=(px + 0.15, py + 0.3), xytext=(px + 0.9, py + 1.75),
            ha="center", fontsize=12, arrowprops=dict(arrowstyle="-", color="#444", lw=1))

# --- Spectrum panel -----------------------------------------------------------
bx0, by0, bw, bh = 0.4, 0.4, 7.2, 4.6
# Light "pyramid" from the K giant to the panel (as in the lux diagram)
apex = (kx - 0.9, ky - 0.25)
corners = [(bx0, by0 + bh), (bx0 + bw, by0 + bh), (bx0 + bw, by0), (bx0, by0)]
ax.add_patch(Polygon([apex, corners[0], corners[1]], color="#f6c88f", alpha=0.25, lw=0, zorder=0))
ax.add_patch(Polygon([apex, corners[1], corners[2]], color="#f6c88f", alpha=0.45, lw=0, zorder=0))
for c in corners:
    ax.plot([apex[0], c[0]], [apex[1], c[1]], ls=":", color="#555", lw=1.2, zorder=1)

ax.add_patch(Rectangle((bx0, by0), bw, bh, facecolor="white", edgecolor="#444", lw=1.5, zorder=2))

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
sy0, sy1 = by0 + 1.4, by0 + bh - 0.3
xs = sx0 + (wl - w0) / (w1 - w0) * (sx1 - sx0)
ys = sy0 + (flux - flux.min()) / (flux.max() - flux.min()) * (sy1 - sy0)
ax.plot(xs, ys, color="black", lw=1.0, zorder=3)
ax.text(bx0 + bw / 2, by0 + bh + 0.12, "H-band spectrum of the K giant", ha="center", va="bottom", fontsize=14, weight="bold")

# Rainbow ruler underneath (violet on the left to red on the right, matching wavelength)
rx0, ry0, rw, rh = sx0, by0 + 0.3, sx1 - sx0, 0.55
hues = np.linspace(0.80, 0.0, 512)                       # violet -> red
grad = hsv_to_rgb(np.stack([hues, np.full_like(hues, 0.85), np.ones_like(hues)], axis=-1))[None, :, :]
ruler = FancyBboxPatch((rx0, ry0), rw, rh, boxstyle="round,pad=0,rounding_size=0.12",
                       facecolor="none", edgecolor="#333", lw=1.2, zorder=5)
ax.add_patch(ruler)
im = ax.imshow(grad, extent=(rx0, rx0 + rw, ry0, ry0 + rh), aspect="auto", zorder=4)
im.set_clip_path(ruler)
for i, x in enumerate(np.linspace(rx0 + 0.08, rx0 + rw - 0.08, 61)):
    tick = 0.22 if i % 10 == 0 else (0.15 if i % 5 == 0 else 0.09)
    ax.plot([x, x], [ry0 + rh, ry0 + rh - tick], color="#222", lw=0.8, zorder=6)
ax.text(rx0, ry0 - 0.08, f"{w0:.3f} µm", ha="left", va="top", fontsize=10)
ax.text(rx0 + rw, ry0 - 0.08, f"{w1:.3f} µm", ha="right", va="top", fontsize=10)

out = __file__.replace("scripts/kgiant_spectrum.py", "images/kgiant_spectrum.png")
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("saved", out)
