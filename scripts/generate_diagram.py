"""Generate the Climate-Pulse system architecture diagram."""
from __future__ import annotations

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

os.makedirs("screenshots", exist_ok=True)


def draw_box(ax, x: float, y: float, w: float, h: float, label: str, color: str, fontsize: int = 9) -> None:
    box = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor="black",
        linewidth=1.2,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, fontweight="bold", wrap=True)


def draw_arrow(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 1.5},
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")
    fig.patch.set_facecolor("#f9f9f9")
    ax.set_facecolor("#f9f9f9")

    plt.title("Climate-Pulse — System Architecture", fontsize=15, fontweight="bold", pad=16)

    # Client
    draw_box(ax, 1.5, 6.5, 2.2, 0.9, "Client\n(HTTP / curl)", "#d0e8ff")

    # FastAPI
    draw_box(ax, 4.5, 6.5, 2.4, 1.1, "FastAPI\n/predict /metrics\n/drift /retrain", "#c8f5d8")

    # Middleware
    draw_box(ax, 4.5, 4.8, 2.4, 0.9, "Middleware\nRateLimit · CorrID", "#fff3cc")

    # Feature Pipeline
    draw_box(ax, 8.0, 6.5, 2.4, 1.1, "Feature Pipeline\nLag · Rolling · Ratio\nSeasonal · DewPoint", "#ffe0cc")

    # ML Ensemble
    draw_box(ax, 11.5, 6.5, 2.6, 1.1, "ML Ensemble\nXGBoost+LGBM+RF\nTemp · Precip · Extreme", "#f0d0ff")

    # Monitoring
    draw_box(ax, 8.0, 4.5, 2.4, 1.0, "Monitoring\nKS-Drift · Logs\nDrift Reports", "#ffe0e0")

    # Database
    draw_box(ax, 11.5, 4.5, 2.6, 1.0, "Database\nSQLite (dev)\nPostgreSQL (prod)", "#d8e8ff")

    # Airflow
    draw_box(ax, 4.5, 2.5, 2.4, 0.9, "Airflow DAG\nWeekly Retrain", "#e8f8e0")

    # Models on disk
    draw_box(ax, 8.0, 2.5, 2.4, 0.9, "Model Store\ntemp · precip\nextreme models", "#fff0f0")

    # Docker
    draw_box(ax, 11.5, 2.5, 2.6, 0.9, "Docker Compose\nAPI + PostgreSQL", "#f0f0f0")

    # Arrows
    draw_arrow(ax, 2.6, 6.5, 3.3, 6.5)
    draw_arrow(ax, 5.7, 6.5, 6.8, 6.5)
    draw_arrow(ax, 9.2, 6.5, 10.2, 6.5)
    draw_arrow(ax, 4.5, 6.0, 4.5, 5.3)
    draw_arrow(ax, 8.0, 6.0, 8.0, 5.0)
    draw_arrow(ax, 11.5, 6.0, 11.5, 5.0)
    draw_arrow(ax, 4.5, 4.3, 4.5, 3.0)
    draw_arrow(ax, 6.5, 2.5, 7.0, 2.5)
    draw_arrow(ax, 10.2, 2.5, 10.2, 2.5)

    plt.tight_layout()
    plt.savefig("screenshots/architecture.png", dpi=150, bbox_inches="tight")
    print("Architecture diagram saved to screenshots/architecture.png")


if __name__ == "__main__":
    main()
