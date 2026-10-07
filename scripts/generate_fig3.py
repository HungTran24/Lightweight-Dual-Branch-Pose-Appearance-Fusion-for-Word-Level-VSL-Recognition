"""Generate the compact signer-error chart used in the paper."""

from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "writing-paper" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

signers = ["006", "020", "028"]
errors = [75, 123, 225]
shares = [17.7, 29.1, 53.2]
colors = ["#9aa6b2", "#6f8fa8", "#d97745"]

plt.rcParams.update(
    {
        "font.family": "serif",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.7,
    }
)

fig, ax = plt.subplots(figsize=(3.35, 1.75), dpi=300)
bars = ax.barh(signers, errors, color=colors, height=0.55)

for bar, count, share in zip(bars, errors, shares, strict=True):
    ax.text(
        count + 5,
        bar.get_y() + bar.get_height() / 2,
        f"{count} ({share:g}%)",
        va="center",
        fontsize=7.5,
    )

ax.invert_yaxis()
ax.set_xlim(0, 270)
ax.set_xlabel("Top-1 errors")
ax.set_ylabel("Validation signer")
ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.45)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout(pad=0.4)

for suffix in ("pdf", "png", "svg"):
    fig.savefig(OUT_DIR / f"fig3-error-analysis.{suffix}", dpi=300, bbox_inches="tight")
plt.close(fig)
