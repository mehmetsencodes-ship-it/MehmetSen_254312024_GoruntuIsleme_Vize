"""Rapor şekilleri için görselleştirme yardımcıları."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from skimage.color import label2rgb
from skimage.segmentation import find_boundaries


def display_gain(img, p=99.5):
    """Yalnızca gösterim amaçlı: koyu görüntüleri görünür kılmak için doğrusal kazanç."""
    hi = max(np.percentile(img, p), 1)
    return np.clip(img.astype(np.float32) / hi, 0, 1)


def overlay_labels(img, labels, alpha=0.35):
    base = display_gain(img)
    rgb = label2rgb(labels, image=base, bg_label=0, alpha=alpha, image_alpha=1)
    rgb[find_boundaries(labels, mode="outer")] = (1, 1, 0)
    return rgb


def overlay_markers(ax, tra_labels, color="red"):
    """Referans (TRA) hücre kimliklerinin merkezlerini işaretler."""
    for lab in np.unique(tra_labels)[1:]:
        ys, xs = np.nonzero(tra_labels == lab)
        ax.plot(xs.mean(), ys.mean(), "+", color=color, ms=9, mew=2)


def show_with_hist(axes_img, axes_hist, img, title):
    axes_img.imshow(img, cmap="gray", vmin=0, vmax=255)
    axes_img.set_title(title, fontsize=9)
    axes_img.axis("off")
    axes_hist.hist(img.ravel(), bins=256, range=(0, 255), color="steelblue", log=True)
    axes_hist.set_xlim(0, 255)
    axes_hist.tick_params(labelsize=7)


def flowchart(steps, title="Nihai pipeline", path=None):
    fig, ax = plt.subplots(figsize=(4.2, 0.9 * len(steps) + 0.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(steps))
    ax.axis("off")
    for i, (head, detail) in enumerate(steps):
        y = len(steps) - i - 0.5
        ax.add_patch(FancyBboxPatch((0.05, y - 0.32), 0.9, 0.64, boxstyle="round,pad=0.02",
                                    fc="#e8f0fb", ec="#2a5caa", lw=1.2))
        ax.text(0.5, y + 0.1, head, ha="center", va="center", fontsize=9, weight="bold")
        ax.text(0.5, y - 0.15, detail, ha="center", va="center", fontsize=7.5)
        if i < len(steps) - 1:
            ax.annotate("", xy=(0.5, y - 0.55), xytext=(0.5, y - 0.34),
                        arrowprops=dict(arrowstyle="->", color="#2a5caa", lw=1.4))
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def config_to_steps(cfg):
    from .pipeline import describe
    d = describe(cfg)
    return [
        ("Girdi", "8-bit gri seviye TIFF (1024x1024)"),
        ("Yoğunluk dönüşümü", d["intensity"]),
        ("Uzamsal filtre", d["filter"]),
        ("Segmentasyon", d["segmentation"]),
        ("Morfoloji", d["morphology"]),
        ("Temas eden hücre ayrıştırma", d["separation"]),
        ("Alan filtresi + sayım", f"min_area = {d['min_area']} px, etiket sayısı"),
    ]
