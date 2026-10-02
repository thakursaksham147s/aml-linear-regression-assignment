"""
Simple Linear Regression - animated gradient descent  (by Sako)

Three linked panels:
  1. data, current line, residuals (the "squares" whose sum we minimise)
  2. loss surface J(b0, b1) as contours + the path gradient descent takes
  3. loss vs iteration
The dashed green line / star is the closed-form OLS answer, so you can see
gradient descent converge to it.

Run:  python animate.py          -> regression.gif, regression.mp4 (if ffmpeg), figures/frame1-4.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

# ---------------- data ----------------
rng = np.random.default_rng(7)
n = 40
x = np.sort(rng.uniform(0, 5, n))
y = 1.5 * x + 2.0 + rng.normal(0, 0.6, n)

# ---------------- closed form (OLS) ----------------
xb, yb = x.mean(), y.mean()
b1_ols = np.sum((x - xb) * (y - yb)) / np.sum((x - xb) ** 2)
b0_ols = yb - b1_ols * xb

def loss(b0, b1):
    return np.mean((y - (b0 + b1 * x)) ** 2)

# ---------------- gradient descent ----------------
alpha, steps = 0.05, 160
b0, b1 = 0.0, 0.0
path = [(b0, b1)]
for _ in range(steps):
    r = y - (b0 + b1 * x)
    g0 = -2 * r.mean()
    g1 = -2 * (x * r).mean()
    b0 -= alpha * g0
    b1 -= alpha * g1
    path.append((b0, b1))
path = np.array(path)
losses = np.array([loss(*p) for p in path])

# ---------------- figure ----------------
BG, FG, ACC, ACC2, GRN = "#0f1626", "#e8eefc", "#4cc9f0", "#f72585", "#80ed99"
plt.rcParams.update({"text.color": FG, "axes.labelcolor": FG, "xtick.color": FG,
                     "ytick.color": FG, "axes.edgecolor": "#3a4668", "font.size": 10})
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=BG,
                                    gridspec_kw={"width_ratios": [1.15, 1.05, 0.9]})
for a in (ax1, ax2, ax3):
    a.set_facecolor(BG)
    a.grid(alpha=0.15)

# panel 1
ax1.scatter(x, y, s=28, c=ACC, edgecolor="white", linewidth=0.4, zorder=3)
xs = np.linspace(-0.2, 5.2, 50)
ax1.plot(xs, b0_ols + b1_ols * xs, "--", c=GRN, lw=1.3, alpha=0.8, label="OLS (closed form)")
line, = ax1.plot([], [], c=ACC2, lw=2.8, label="current fit", zorder=4)
resid = [ax1.plot([], [], c=ACC2, lw=0.8, alpha=0.5)[0] for _ in range(n)]
ax1.set_xlim(-0.2, 5.2); ax1.set_ylim(-0.5, 11)
ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_title("Fitting the line", color=FG)
ax1.legend(facecolor=BG, edgecolor="#3a4668", loc="upper left")
txt = ax1.text(0.98, 0.04, "", transform=ax1.transAxes, ha="right", va="bottom", color=FG,
               fontsize=10, family="monospace")

# panel 2
B0, B1 = np.meshgrid(np.linspace(-1, 5, 120), np.linspace(-1, 4, 120))
Z = np.array([[loss(a, b) for a, b in zip(r0, r1)] for r0, r1 in zip(B0, B1)])
ax2.contourf(B0, B1, Z, levels=30, cmap="magma", alpha=0.9)
ax2.contour(B0, B1, Z, levels=12, colors="white", linewidths=0.3, alpha=0.4)
ax2.plot(b0_ols, b1_ols, "*", c=GRN, ms=16, mec="white", zorder=5)
trail, = ax2.plot([], [], c=ACC, lw=1.6)
dot, = ax2.plot([], [], "o", c="white", ms=8, mec=ACC2, mew=2, zorder=6)
ax2.set_xlabel(r"intercept $\beta_0$"); ax2.set_ylabel(r"slope $\beta_1$")
ax2.set_title(r"Loss surface $J(\beta_0,\beta_1)$", color=FG)

# panel 3
ax3.set_xlim(0, steps); ax3.set_yscale("log"); ax3.set_ylim(loss(b0_ols, b1_ols) * 0.8, losses[0] * 1.5)
ax3.set_xlabel("iteration"); ax3.set_ylabel("MSE (log scale)"); ax3.set_title("Loss going down", color=FG)
ax3.axhline(loss(b0_ols, b1_ols), c=GRN, ls="--", lw=1)
lcurve, = ax3.plot([], [], c=ACC, lw=2.2)
ldot, = ax3.plot([], [], "o", c=ACC2, ms=7)

fig.suptitle("Simple Linear Regression: gradient descent finds the least-squares line",
             color=FG, fontsize=13, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])

# skip frames so the GIF stays small but early steps (where the action is) are dense
frames = sorted(set(list(range(0, 30)) + list(range(30, steps + 1, 3)) + [steps]))

def draw(k):
    p0, p1 = path[k]
    line.set_data(xs, p0 + p1 * xs)
    yh = p0 + p1 * x
    for i, rl in enumerate(resid):
        rl.set_data([x[i], x[i]], [y[i], yh[i]])
    trail.set_data(path[:k + 1, 0], path[:k + 1, 1])
    dot.set_data([p0], [p1])
    lcurve.set_data(range(k + 1), losses[:k + 1])
    ldot.set_data([k], [losses[k]])
    txt.set_text(f"iter {k:3d}\nb0={p0:5.2f}  b1={p1:5.2f}\nMSE={losses[k]:6.3f}")
    return []

def save_frame(k, name):
    draw(k)
    fig.savefig(f"figures/{name}", dpi=150, facecolor=BG)

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    print(f"OLS: b0={b0_ols:.3f}  b1={b1_ols:.3f}   GD final: b0={path[-1,0]:.3f} b1={path[-1,1]:.3f}")
    # 4 key frames for the report
    for k, name in [(0, "frame1.png"), (4, "frame2.png"), (30, "frame3.png"), (steps, "frame4.png")]:
        save_frame(k, name)
    anim = FuncAnimation(fig, draw, frames=frames, interval=60, blit=False)
    anim.save("regression.gif", writer=PillowWriter(fps=16), dpi=80, savefig_kwargs={"facecolor": BG})
    print("saved regression.gif")
    try:
        anim.save("regression.mp4", writer=FFMpegWriter(fps=16), dpi=120, savefig_kwargs={"facecolor": BG})
        print("saved regression.mp4")
    except Exception as e:
        print("mp4 skipped:", e)
