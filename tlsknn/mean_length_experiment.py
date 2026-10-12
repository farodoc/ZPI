"""Ewaluacja baseline MeanPacketLengthClassifier — osobne artefakty, bez nadpisywania train."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from . import plots
from .baselines import MeanPacketLengthClassifier
from .dataset import Flow
from .experiment import ATTACK_PAD, ATTACK_REPEATS, PAD_LEVELS, PREFIX_LEVELS, _log
from .features import DEFAULT_MAX_PACKETS, mean_packet_length_many, pad_sizes
from .model import TrafficClassifier, category


def _split_flows(flows: list[Flow], model_path: Path, seed: int) -> tuple[list[Flow], list[Flow]]:
    meta = TrafficClassifier.load(model_path).meta
    test_names = set(meta.get("test_flow_names", []))
    if test_names:
        te = [f for f in flows if f"{f.label}/{f.name}" in test_names]
        tr = [f for f in flows if f"{f.label}/{f.name}" not in test_names]
        if len(te) and len(tr):
            return tr, te
    y = np.array([f.label for f in flows])
    idx_tr, idx_te = train_test_split(np.arange(len(flows)), test_size=0.2, stratify=y, random_state=seed)
    return [flows[i] for i in idx_tr], [flows[i] for i in idx_te]


def _mean_length_matrix(flows: list[Flow], max_packets: int, pad_high: int = 0, seed: int = 0,
                        pad_low: int = 1) -> np.ndarray:
    if not pad_high:
        return mean_packet_length_many(flows, max_packets)
    rng = np.random.default_rng(seed)
    rows = []
    for f in flows:
        s = f.sizes[:max_packets]
        s = pad_sizes(s, rng, pad_low, pad_high)
        rows.append([float(s.mean()) if s.size else 0.0])
    return np.asarray(rows, dtype=np.float64)


def _write_report(r: dict, path: Path) -> None:
    t, pa = r["test"], r["padding_attack"]

    def pct(v):
        return f"{v * 100:.2f}%"

    lines = [
        "# Baseline: klasyfikator średniej długości pakietu",
        "",
        f"*Wygenerowano automatycznie: {time.strftime('%Y-%m-%d %H:%M')}*",
        "",
        "Dummy własny (`MeanPacketLengthClassifier`): jedna cecha — średni rozmiar pakietu "
        f"(z pierwszych {r['features']['max_packets']} pakietów). Predykcja = klasa o najbliższej "
        "średniej tej cechy na zbiorze treningowym.",
        "",
        f"Podział train/test zgodny z modelem głównym: `{r['reference_model']}` (bez ponownego treningu k-NN).",
        "",
        "## Podsumowanie",
        "",
        "| Miara | Wartość |",
        "|---|---|",
        f"| Accuracy (czysty test) | **{pct(t['accuracy'])}** |",
        f"| Macro-F1 | {t['macro_f1']:.4f} |",
        f"| Accuracy po ataku padding U{{1..100}} B | {pct(pa['accuracy_mean'])} ± {pa['accuracy_std'] * 100:.2f} pp |",
        f"| k-NN główny (czysty test, z metrics.json) | {pct(r['reference']['knn_clean_accuracy'])} |",
        "",
        "## Wyniki na czystym zbiorze testowym",
        "",
        "| Klasa | Kategoria | Precision | Recall | F1 | Liczność |",
        "|---|---|---|---|---|---|",
        *[f"| {c} | {category(c)} | {t['per_class'][c]['precision']:.4f} | {t['per_class'][c]['recall']:.4f} | "
          f"{t['per_class'][c]['f1-score']:.4f} | {int(t['per_class'][c]['support'])} |" for c in t["classes"]],
        "",
        "![cm](confusion_clean.png)",
        "",
        "## Wczesna klasyfikacja (ile pakietów wystarczy?)",
        "",
        "| Pierwsze N pakietów | Accuracy |",
        "|---|---|",
        *[f"| {p['packets']} | {pct(p['accuracy'])} |" for p in r["prefix"]],
        "",
        "![prefix](prefix.png)",
        "",
        "## Atak padding",
        "",
        "| Maks. padding X [B] | baseline średniej długości | k-NN (odniesienie) |",
        "|---|---|---|",
        *[f"| {s['max_pad']} | {pct(s['acc_mean_length'])} | {pct(s['acc_knn_ref'])} |" for s in pa["sweep"]],
        "",
        "![padding](padding_attack.png)",
        "",
        "![cm-pad](confusion_padding.png)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_mean_packet_length(flows: list[Flow], out_dir: Path, model_path: Path,
                           main_metrics_path: Path | None = None,
                           max_packets: int = DEFAULT_MAX_PACKETS, seed: int = 42) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Brak modelu głównego: {model_path}")

    main_metrics_path = Path(main_metrics_path or Path("results") / "metrics.json")
    knn_ref = {}
    if main_metrics_path.is_file():
        knn_ref = json.loads(main_metrics_path.read_text(encoding="utf-8"))

    tr, te = _split_flows(flows, model_path, seed)
    y_tr = np.array([f.label for f in tr])
    y_te = np.array([f.label for f in te])
    classes = sorted(set(y_te))

    _log(f"Baseline średniej długości pakietu — train={len(tr)} test={len(te)} "
         f"(split z {model_path.name})")

    clf = MeanPacketLengthClassifier(max_packets=max_packets).fit(
        mean_packet_length_many(tr, max_packets), y_tr)
    pred = clf.predict(mean_packet_length_many(te, max_packets))
    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred, average="macro")
    cm = confusion_matrix(y_te, pred, labels=classes)
    report = classification_report(y_te, pred, labels=classes, digits=4, output_dict=True)
    _log(f"  accuracy = {acc:.4f}   macro-F1 = {f1:.4f}")
    _log(f"  centroidy [B]: " + ", ".join(f"{c}={clf.centroids_[i]:.1f}" for i, c in enumerate(clf.classes_)))

    _log("\n  Dokładność vs liczba pierwszych pakietów:")
    prefix = []
    for n in PREFIX_LEVELS:
        m = MeanPacketLengthClassifier(max_packets=n).fit(mean_packet_length_many(tr, n), y_tr)
        a = accuracy_score(y_te, m.predict(mean_packet_length_many(te, n)))
        prefix.append({"packets": n, "accuracy": a})
        _log(f"    N={n:>4}: {a:.4f}")

    _log(f"\nAtak padding U{{{ATTACK_PAD[0]}..{ATTACK_PAD[1]}}} B ({ATTACK_REPEATS} powtórzeń):")
    attack_preds = []
    for r in range(ATTACK_REPEATS):
        Xp = _mean_length_matrix(te, max_packets, ATTACK_PAD[1], seed + 1000 + r, ATTACK_PAD[0])
        attack_preds.append(clf.predict(Xp))
    attack_acc = np.array([accuracy_score(y_te, p) for p in attack_preds])
    cm_pad = confusion_matrix(y_te, attack_preds[0], labels=classes)
    _log(f"  accuracy po ataku = {attack_acc.mean():.4f} ± {attack_acc.std():.4f}")

    knn_sweep = {s["max_pad"]: s["acc_clean_model"] for s in knn_ref.get("padding_attack", {}).get("sweep", [])}
    knn_clean = float(knn_ref.get("test", {}).get("accuracy", float("nan")))

    _log("\n  Krzywa odporności (padding U{1..X}):")
    sweep = []
    for high in PAD_LEVELS:
        Xp = _mean_length_matrix(te, max_packets, high, seed + 2000 + high) if high else mean_packet_length_many(te, max_packets)
        a = accuracy_score(y_te, clf.predict(Xp))
        ref = knn_sweep.get(high, float("nan"))
        sweep.append({"max_pad": high, "acc_mean_length": a, "acc_knn_ref": ref})
        _log(f"    X={high:>3} B: baseline {a:.4f} | k-NN (ref) {ref:.4f}")

    results = {
        "classifier": "MeanPacketLengthClassifier",
        "reference_model": str(model_path),
        "reference_metrics": str(main_metrics_path) if main_metrics_path.is_file() else None,
        "reference": {"knn_clean_accuracy": knn_clean},
        "dataset": {"flows": len(flows), "train": len(tr), "test": len(te), "seed": seed,
                    "split": "test_flow_names from main model meta"},
        "features": {"max_packets": max_packets, "dim": 1, "name": "mean_packet_length_bytes"},
        "centroids_bytes": {str(c): float(clf.centroids_[i]) for i, c in enumerate(clf.classes_)},
        "test": {"accuracy": acc, "macro_f1": f1, "per_class": report, "confusion_matrix": cm.tolist(),
                 "classes": classes},
        "prefix": prefix,
        "padding_attack": {"pad_range": ATTACK_PAD, "repeats": ATTACK_REPEATS,
                           "accuracy_mean": float(attack_acc.mean()), "accuracy_std": float(attack_acc.std()),
                           "confusion_matrix": cm_pad.tolist(), "sweep": sweep},
    }
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    plots.plot_prefix(list(PREFIX_LEVELS), [p["accuracy"] for p in prefix], out_dir / "prefix.png")
    plots.plot_mean_length_padding(list(PAD_LEVELS), [s["acc_mean_length"] for s in sweep],
                                   [s["acc_knn_ref"] for s in sweep] if knn_sweep else None,
                                   out_dir / "padding_attack.png")
    plots.plot_confusion(cm, classes, f"Macierz pomyłek — średnia długość pakietu (acc {acc:.2%})",
                         out_dir / "confusion_clean.png")
    plots.plot_confusion(cm_pad, classes, f"Macierz pomyłek — po ataku padding (acc {attack_acc[0]:.2%})",
                         out_dir / "confusion_padding.png")
    _write_report(results, out_dir / "report.md")
    _log(f"\nZapisano artefakty baseline -> {out_dir}")
    return results
