"""Builds the paper's figures from the project's own results and screenshots."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8, "axes.linewidth": 0.6})
DARK, MID, LIGHT, ACCENT = "#1f3a5f", "#5b7aa6", "#dde5f0", "#b3261e"
W = 3.45  # IEEE column width in inches


def box(ax, x, y, w, h, title, sub, fill, text="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fill, ec="none"))
    ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", color=text,
            fontsize=7.5, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", color=text, fontsize=5.6)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#333333", mutation_scale=7))


# Fig. 1: architecture
fig, ax = plt.subplots(figsize=(W, 3.4))
ax.set_xlim(0, 10)
ax.set_ylim(1.9, 11.2)
ax.axis("off")
bw, bh = 4.3, 1.15
left, right = 0.25, 5.45
rows = [9.9, 8.3, 6.7, 5.1]
box(ax, left, rows[0], bw, bh, "CCTV cameras", "files or RTSP, one place each", DARK)
box(ax, right, rows[0], bw, bh, "Camera worker", "one thread and model per feed", DARK)
box(ax, right, rows[1], bw, bh, "YOLO11s-pose", "boxes and 17 keypoints", DARK)
box(ax, left, rows[1], bw, bh, "ByteTrack", "one ID per person", DARK)
box(ax, left, rows[2], bw, bh, "Fall decision", "torso angle, height drop, timers", ACCENT)
box(ax, right, rows[2], bw, bh, "Temporal network", "optional, 2 s of keypoints", MID)
box(ax, left, rows[3], bw, bh, "Alert store (SQLite)", "photo, place, status, repeats", ACCENT)
box(ax, right, rows[3], bw, bh, "Flask server", "dashboard, signed-token API", DARK)
outs = [(left, "Android app", "push via FCM, staff respond"),
        (right, "Telegram", "backup copy with photo")]
for x, t, sub in outs:
    box(ax, x, 3.0, bw, 1.15, t, sub, LIGHT, text="#1a1a1a")
arrow(ax, left + bw, rows[0] + bh / 2, right, rows[0] + bh / 2)
arrow(ax, right + bw / 2, rows[0], right + bw / 2, rows[1] + bh)
arrow(ax, right, rows[1] + bh / 2, left + bw, rows[1] + bh / 2)
arrow(ax, left + bw / 2, rows[1], left + bw / 2, rows[2] + bh)
arrow(ax, right, rows[2] + bh / 2, left + bw, rows[2] + bh / 2)
arrow(ax, left + bw / 2, rows[2], left + bw / 2, rows[3] + bh)
arrow(ax, left + bw, rows[3] + bh / 2, right, rows[3] + bh / 2)
arrow(ax, left + bw / 2, rows[3], left + bw / 2, 3.0 + 1.15)
arrow(ax, left + bw * 0.85, rows[3], right + bw / 2, 3.0 + 1.15)
ax.text(5, 2.25, "The phone reaches the server through a Cloudflare tunnel.", ha="center",
        fontsize=6.5, color="#444444", style="italic")
fig.savefig("fig1_architecture.png", dpi=400, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# Fig. 2: dashboard screenshot, cropped to the camera grid and alert panel
shot = Image.open("media/s13_image4.png").convert("RGB")
top, bottom = shot.crop((0, 70, 1112, 185)), shot.crop((0, 462, 1112, 750))
out = Image.new("RGB", (1112, top.height + bottom.height + 8), "#0d1117")
out.paste(top, (0, 0))
out.paste(bottom, (0, top.height + 8))
out.save("fig2_dashboard.png")

# Fig. 3: Telegram alert and multi-person composite
Image.open("media/s14_image7.jpg").convert("RGB").save("fig3_multi.png")

# Fig. 4: held-out results (from results.json, results_pose.json, results_realtime.json, results_learned.json)
names = ["Prototype", "Rules,\nevery frame", "Rules,\n7.5 fps", "Network,\n7.5 fps"]
caught = [9, 15, 11, 14]
false = [6, 1, 1, 6]
fig, ax = plt.subplots(figsize=(W, 2.0))
x = range(len(names))
b1 = ax.bar([i - 0.19 for i in x], [c / 15 * 100 for c in caught], 0.36, color=DARK, label="Falls caught (of 15)")
b2 = ax.bar([i + 0.19 for i in x], [f / 20 * 100 for f in false], 0.36, color=ACCENT, label="False alarms (of 20)")
for bars, counts in ((b1, caught), (b2, false)):
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, str(n), ha="center", fontsize=7)
ax.set_xticks(list(x))
ax.set_xticklabels(names)
ax.set_ylabel("Clips (%)")
ax.set_ylim(0, 115)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, lw=0.4, color="#dddddd")
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.13))
fig.savefig("fig4_results.png", dpi=400, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# Fig. 5: six-camera throughput by PyTorch threads per camera (45 s runs, unpaced)
labels = ["Default", "4", "2", "1"]
fps = [4.4, 5.0, 4.7, 4.1]
p99 = [359, 250, 265, 406]
fig, ax = plt.subplots(figsize=(W, 1.75))
bars = ax.bar(labels, fps, 0.5, color=DARK)
for bar, v in zip(bars, fps):
    ax.text(bar.get_x() + bar.get_width() / 2, v / 2, f"{v:.1f}", ha="center", va="center",
            fontsize=7.5, color="white", fontweight="bold")
ax.set_ylabel("Frames/s per camera")
ax.set_xlabel("PyTorch threads per camera")
ax.set_ylim(0, 6.2)
ax.spines[["top", "right"]].set_visible(False)
ax2 = ax.twinx()
ax2.plot(labels, p99, color=ACCENT, marker="o", ms=3.5, lw=1)
ax2.set_ylabel("99th percentile gap (ms)", color=ACCENT)
ax2.set_ylim(0, 480)
ax2.tick_params(axis="y", colors=ACCENT)
ax2.spines[["top"]].set_visible(False)
fig.savefig("fig5_speed.png", dpi=400, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("figures written")
