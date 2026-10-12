from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)


def _fig(ncols: int = 1, width: float = 7.0):
    fig, axes = plt.subplots(1, ncols, figsize=(width, 4.2), facecolor=SURFACE)
    return fig, np.atleast_1d(axes)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


PCT1 = plt.FuncFormatter(lambda v, _: f"{v:.1%}")
PCT0 = plt.FuncFormatter(lambda v, _: f"{v:.0%}")


def _line(ax, x, y, i: int, label: str, err=None, end_label: bool = True) -> None:
    ax.plot(x, y, color=SERIES[i], linewidth=2, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
    if err is not None:
        ax.fill_between(x, np.asarray(y) - err, np.asarray(y) + err, color=SERIES[i], alpha=0.15, linewidth=0)
    if end_label:
        ax.annotate(f"{y[-1]:.1%}", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK_2, fontsize=9)


def plot_k_tuning(ks, acc_mean, acc_std, latency_ms, best_k: int, path: Path) -> None:
    fig, (a1, a2) = _fig(2, 10)
    _line(a1, ks, acc_mean, 0, "CV accuracy", err=acc_std, end_label=False)
    i = list(ks).index(best_k)
    a1.annotate(f"wybrane k={best_k} ({acc_mean[i]:.2%})", (best_k, acc_mean[i]), xytext=(40, 6),
                textcoords="offset points", color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    _style(a1, "Dokładność 5-fold CV vs k", "k (liczba sąsiadów)", "accuracy")
    a1.yaxis.set_major_formatter(PCT1)
    _line(a2, ks, latency_ms, 0, "czas predykcji", end_label=False)
    _style(a2, "Czas predykcji 1 wektora vs k", "k (liczba sąsiadów)", "mediana [ms]")
    a2.set_ylim(bottom=0)
    _save(fig, path)


def plot_padding(levels, acc_clean_model, acc_aware_model, dummy_acc: float, path: Path) -> None:
    fig, (ax,) = _fig(1, 7.5)
    _line(ax, levels, acc_clean_model, 0, "k-NN trenowany na czystym ruchu", end_label=False)
    _line(ax, levels, acc_aware_model, 1, "k-NN trenowany z paddingiem U{1..100} (obrona)", end_label=False)
    ax.axhline(dummy_acc, color=MUTED, linestyle="--", linewidth=1.2)
    ax.annotate("losowy baseline (Dummy)", (levels[0], dummy_acc), xytext=(0, 5), textcoords="offset points",
                color=INK_2, fontsize=9)
    ax.axhline(0.9, color=MUTED, linestyle=":", linewidth=1.2)
    ax.annotate("próg zaliczenia 90%", (levels[-1], 0.9), xytext=(-110, 5), textcoords="offset points",
                color=INK_2, fontsize=9)
    _style(ax, "Atak padding: dokładność vs maks. narzut na pakiet", "padding: U{1..X} bajtów na pakiet (X)",
           "accuracy (zbiór testowy)")
    ax.set_ylim(0, 1.04)
    ax.yaxis.set_major_formatter(PCT0)
    ax.legend(frameon=False, loc="lower left", labelcolor=INK_2, fontsize=9)
    _save(fig, path)


def plot_prefix(ns, acc, path: Path) -> None:
    fig, (ax,) = _fig(1, 7.0)
    _line(ax, ns, acc, 0, "k-NN")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.minorticks_off()
    _style(ax, "Wczesna klasyfikacja: dokładność vs liczba pierwszych pakietów",
           "liczba obserwowanych pakietów przepływu (skala log)", "accuracy (zbiór testowy)")
    ax.yaxis.set_major_formatter(PCT1)
    _save(fig, path)


def plot_confusion(cm: np.ndarray, classes: list[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4), facecolor=SURFACE)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=9,
                    color=SURFACE if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predykcja", color=INK_2)
    ax.set_ylabel("prawdziwa klasa", color=INK_2)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=10)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(classes)), minor=True)
    ax.set_yticks(np.arange(-.5, len(classes)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    _save(fig, path)
