from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from . import plots
from .dataset import Flow
from .features import DEFAULT_MAX_PACKETS, extract, extract_many, feature_names, pad_sizes
from .model import TrafficClassifier, build_pipeline, category

K_GRID = (1, 3, 5, 7, 9, 11, 15, 21, 31, 51)
PAD_LEVELS = (0, 10, 25, 50, 100, 200, 400)
PREFIX_LEVELS = (5, 10, 20, 50, 100, 200, 500, 1000)
STAGE_LEVELS = (10, 20, 50)
ATTACK_PAD = (1, 100)
ATTACK_REPEATS = 5
K_TOLERANCE = 0.002  # mniejsze k wygrywa, jeśli traci < 0.2 pp


def _log(msg: str) -> None:
    print(msg, flush=True)


def padded_features(flows: list[Flow], high: int, seed: int, max_packets: int, low: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for f in flows:
        s = f.sizes[:max_packets]
        out.append(extract(f.times[:max_packets], pad_sizes(s, rng, low, high) if high else s,
                           f.dirs[:max_packets], max_packets))
    return np.vstack(out)


def single_latency_ms(clf: TrafficClassifier, flows: list[Flow], n: int = 500) -> np.ndarray:
    lat = [clf.predict_flow(f.times, f.sizes, f.dirs).latency_ms for f in flows[:n]]
    return np.array(lat)


def tune_k(X: np.ndarray, y: np.ndarray, seed: int) -> dict:
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    rows = []
    for k in K_GRID:
        r = cross_validate(build_pipeline(k), X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        m = build_pipeline(k).fit(X, y)
        xs = X[:200]
        m.predict(xs[:1])  # warm-up
        t0 = time.perf_counter()
        for x in xs:
            m.predict(x[None, :])
        lat = (time.perf_counter() - t0) / len(xs) * 1e3
        rows.append({"k": k, "cv_acc_mean": float(r["test_score"].mean()), "cv_acc_std": float(r["test_score"].std()),
                     "single_predict_ms": lat})
        _log(f"  k={k:>2}  CV acc = {r['test_score'].mean():.4f} ± {r['test_score'].std():.4f}  |  {lat:.3f} ms/wektor")
    best = max(r["cv_acc_mean"] for r in rows)
    chosen = min(r["k"] for r in rows if r["cv_acc_mean"] >= best - K_TOLERANCE)
    return {"grid": rows, "best_cv_acc": best, "chosen_k": chosen}


def run(flows: list[Flow], out_dir: Path, model_path: Path, max_packets: int = DEFAULT_MAX_PACKETS,
        seed: int = 42, k: int | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    y = np.array([f.label for f in flows])
    classes = sorted(set(y))
    _log(f"Dane: {len(flows)} przepływów, {len(classes)} klas: "
         + ", ".join(f"{c} ({(y == c).sum()})" for c in classes))

    idx_tr, idx_te = train_test_split(np.arange(len(flows)), test_size=0.2, stratify=y, random_state=seed)
    tr = [flows[i] for i in idx_tr]
    te = [flows[i] for i in idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]
    _log(f"Hold-out: train={len(tr)}  test={len(te)}  (cechy z pierwszych {max_packets} pakietów)")

    t0 = time.perf_counter()
    X_tr, X_te = extract_many(tr, max_packets), extract_many(te, max_packets)
    feat_ms = (time.perf_counter() - t0) / len(flows) * 1e3
    _log(f"Wymiar wektora cech: {X_tr.shape[1]}  (ekstrakcja {feat_ms:.2f} ms/przepływ)")

    _log("\nDobór hiperparametru k (5-fold CV na zbiorze treningowym):")
    tuning = tune_k(X_tr, y_tr, seed)
    if k is None:
        k = tuning["chosen_k"]
    _log(f"  => wybrane k = {k} (najmniejsze k w granicy {K_TOLERANCE * 100:.1f} pp od najlepszego wyniku CV)")

    t0 = time.perf_counter()
    pipe = build_pipeline(k).fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0

    _log("\nEwaluacja na czystym zbiorze testowym:")
    t0 = time.perf_counter()
    pred = pipe.predict(X_te)
    batch_s = time.perf_counter() - t0
    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred, average="macro")
    cm = confusion_matrix(y_te, pred, labels=classes)
    report = classification_report(y_te, pred, labels=classes, digits=4, output_dict=True)
    _log(f"  k-NN accuracy = {acc:.4f}   macro-F1 = {f1:.4f}")
    _log(classification_report(y_te, pred, labels=classes, digits=4))

    baselines = {}
    for strategy in ("uniform", "stratified", "most_frequent"):
        d = DummyClassifier(strategy=strategy, random_state=seed).fit(X_tr, y_tr)
        p = d.predict(X_te)
        baselines[strategy] = {"accuracy": accuracy_score(y_te, p), "macro_f1": f1_score(y_te, p, average="macro")}
        _log(f"  Dummy ({strategy:<13}) accuracy = {baselines[strategy]['accuracy']:.4f}")

    # modele dla N ze STAGE_LEVELS idą od razu do klasyfikatora wieloetapowego
    _log("\n  Dokładność vs liczba pierwszych pakietów przepływu (osobny model k-NN dla każdego N):")
    prefix, stages = [], {max_packets: pipe}
    for n in PREFIX_LEVELS:
        if n == max_packets:
            m, a = pipe, acc
        else:
            m = build_pipeline(k).fit(extract_many(tr, n), y_tr)
            a = accuracy_score(y_te, m.predict(extract_many(te, n)))
        if n in STAGE_LEVELS and n < max_packets:
            stages[n] = m
        prefix.append({"packets": n, "accuracy": a, "stage": n in stages})
        _log(f"    N={n:>4}: {a:.4f}{'  (etap klasyfikatora w locie)' if n in stages else ''}")

    clf = TrafficClassifier(stages)
    lat = single_latency_ms(clf, te)
    perf = {
        "fit_seconds": fit_s,
        "feature_extraction_ms_per_flow": feat_ms,
        "single_flow_latency_ms": {"median": float(np.median(lat)), "p95": float(np.percentile(lat, 95)),
                                   "p99": float(np.percentile(lat, 99)), "max": float(lat.max())},
        "batch_throughput_flows_per_s": len(te) / batch_s,
    }
    _log(f"\n  Czas predykcji 1 przepływu: mediana {perf['single_flow_latency_ms']['median']:.2f} ms, "
         f"p99 {perf['single_flow_latency_ms']['p99']:.2f} ms;  wsadowo {perf['batch_throughput_flows_per_s']:.0f} przepływów/s")

    _log(f"\nAtak padding: +U{{{ATTACK_PAD[0]}..{ATTACK_PAD[1]}}} B do każdego pakietu testowego "
         f"({ATTACK_REPEATS} powtórzeń):")
    attack_runs = []
    for r in range(ATTACK_REPEATS):
        Xp = padded_features(te, ATTACK_PAD[1], seed + 1000 + r, max_packets, ATTACK_PAD[0])
        attack_runs.append(pipe.predict(Xp))
    attack_acc = np.array([accuracy_score(y_te, p) for p in attack_runs])
    cm_pad = confusion_matrix(y_te, attack_runs[0], labels=classes)
    _log(f"  accuracy po ataku = {attack_acc.mean():.4f} ± {attack_acc.std():.4f}  "
         f"(spadek o {(acc - attack_acc.mean()) * 100:.1f} pp)")

    # obrona: trening na danych + ich kopii z paddingiem
    y_tr_aug = np.r_[y_tr, y_tr]
    stages_aware = {
        n: build_pipeline(k).fit(np.vstack([extract_many(tr, n),
                                            padded_features(tr, ATTACK_PAD[1], seed + 7, n, ATTACK_PAD[0])]), y_tr_aug)
        for n in stages}
    pipe_aware = stages_aware[max_packets]

    _log("\n  Krzywa odporności (padding U{1..X}):")
    sweep = []
    for high in PAD_LEVELS:
        Xp = padded_features(te, high, seed + 2000 + high, max_packets) if high else X_te
        a_clean = accuracy_score(y_te, pipe.predict(Xp))
        a_aware = accuracy_score(y_te, pipe_aware.predict(Xp))
        sweep.append({"max_pad": high, "acc_clean_model": a_clean, "acc_padding_aware_model": a_aware})
        _log(f"    X={high:>3} B: k-NN {a_clean:.4f} | k-NN z obroną {a_aware:.4f}")

    hypothesis = bool(acc > 0.90)
    _log(f"\nHIPOTEZA (accuracy > 90% na czystym zbiorze testowym): "
         f"{'POTWIERDZONA' if hypothesis else 'ODRZUCONA'} ({acc:.2%})")

    clf.meta = {
        "k": k, "classes": classes, "trained_on": len(tr), "test_accuracy": acc,
        "test_flow_names": [f"{f.label}/{f.name}" for f in te], "seed": seed,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    clf.save(model_path)
    TrafficClassifier(stages_aware, {**clf.meta, "padding_aware": True}).save(
        model_path.with_name(model_path.stem + "_padding_aware" + model_path.suffix))

    results = {
        "dataset": {"flows": len(flows), "classes": {c: int((y == c).sum()) for c in classes},
                    "train": len(tr), "test": len(te), "split": "hold-out 80/20, stratified", "seed": seed},
        "features": {"max_packets": max_packets, "dim": int(X_tr.shape[1]), "names": feature_names()},
        "k_tuning": tuning, "k": k,
        "test": {"accuracy": acc, "macro_f1": f1, "per_class": report, "confusion_matrix": cm.tolist(),
                 "classes": classes},
        "baselines": baselines,
        "performance": perf,
        "prefix": prefix,
        "padding_attack": {"pad_range": ATTACK_PAD, "repeats": ATTACK_REPEATS,
                           "accuracy_mean": float(attack_acc.mean()), "accuracy_std": float(attack_acc.std()),
                           "drop_pp": float((acc - attack_acc.mean()) * 100),
                           "confusion_matrix": cm_pad.tolist(), "sweep": sweep},
        "hypothesis_accepted": hypothesis,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "platform": platform.platform(), "cpu": platform.processor()},
    }
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    plots.plot_k_tuning([r["k"] for r in tuning["grid"]], [r["cv_acc_mean"] for r in tuning["grid"]],
                        np.array([r["cv_acc_std"] for r in tuning["grid"]]),
                        [r["single_predict_ms"] for r in tuning["grid"]], k, out_dir / "k_tuning.png")
    plots.plot_padding(list(PAD_LEVELS), [s["acc_clean_model"] for s in sweep],
                       [s["acc_padding_aware_model"] for s in sweep], baselines["uniform"]["accuracy"],
                       out_dir / "padding_attack.png")
    plots.plot_prefix(list(PREFIX_LEVELS), [p["accuracy"] for p in prefix], out_dir / "prefix.png")
    plots.plot_confusion(cm, classes, f"Macierz pomyłek - czysty test (acc {acc:.2%})", out_dir / "confusion_clean.png")
    plots.plot_confusion(cm_pad, classes, f"Macierz pomyłek - po ataku padding (acc {attack_acc[0]:.2%})",
                         out_dir / "confusion_padding.png")
    write_report(results, out_dir / "report.md")
    _log(f"\nZapisano: model -> {model_path}, wyniki -> {out_dir}")
    return results


def write_report(r: dict, path: Path) -> None:
    t, perf, pa = r["test"], r["performance"], r["padding_attack"]
    lat = perf["single_flow_latency_ms"]

    def pct(v):
        return f"{v * 100:.2f}%"

    lines = [
        "# Raport z eksperymentu - k-NN dla ruchu szyfrowanego (UC Davis QUIC)",
        "",
        f"*Wygenerowano automatycznie: {time.strftime('%Y-%m-%d %H:%M')}*",
        "",
        "## Podsumowanie",
        "",
        "| Miara | Wartość |",
        "|---|---|",
        f"| Accuracy k-NN (czysty test) | **{pct(t['accuracy'])}** |",
        f"| Macro-F1 | {t['macro_f1']:.4f} |",
        f"| Accuracy Dummy (losowy, uniform) | {pct(r['baselines']['uniform']['accuracy'])} |",
        f"| Accuracy po ataku padding U{{1..100}} B | {pct(pa['accuracy_mean'])} ± {pa['accuracy_std'] * 100:.2f} pp |",
        f"| Spadek skuteczności po ataku | {pa['drop_pp']:.1f} pp |",
        f"| Wybrane k | {r['k']} |",
        f"| Czas predykcji 1 przepływu (mediana / p99) | {lat['median']:.2f} ms / {lat['p99']:.2f} ms |",
        f"| Hipoteza (accuracy > 90%) | **{'POTWIERDZONA' if r['hypothesis_accepted'] else 'ODRZUCONA'}** |",
        "",
        "## Dane",
        "",
        f"- Przepływów: {r['dataset']['flows']}, podział {r['dataset']['split']} "
        f"(train {r['dataset']['train']}, test {r['dataset']['test']}), seed {r['dataset']['seed']}.",
        "- Klasy: " + ", ".join(f"{c} - {category(c)} ({n})" for c, n in r["dataset"]["classes"].items()) + ".",
        f"- Cechy: {r['features']['dim']}-wymiarowy wektor statystyk rozmiarów pakietów i inter-arrival times "
        f"(osobno dla kierunku upstream/downstream) z pierwszych {r['features']['max_packets']} pakietów przepływu.",
        "",
        "## Dobór k (5-fold CV na zbiorze treningowym)",
        "",
        "| k | CV accuracy | odch. std. | czas predykcji 1 wektora [ms] |",
        "|---|---|---|---|",
        *[f"| {'**' + str(g['k']) + '**' if g['k'] == r['k'] else g['k']} | {g['cv_acc_mean']:.4f} | {g['cv_acc_std']:.4f} | "
          f"{g['single_predict_ms']:.3f} |" for g in r["k_tuning"]["grid"]],
        "",
        f"Reguła wyboru: najmniejsze k, którego dokładność CV jest nie gorsza niż {K_TOLERANCE * 100:.1f} pp od najlepszej "
        "(mniejsze k = mniej sąsiadów do głosowania i mniejsza wrażliwość na nierównowagę klas).",
        "",
        "![k](k_tuning.png)",
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
        "## Grupa kontrolna (baseline)",
        "",
        "| Klasyfikator | Accuracy | Macro-F1 |",
        "|---|---|---|",
        f"| k-NN (k={r['k']}) | {pct(t['accuracy'])} | {t['macro_f1']:.4f} |",
        *[f"| Dummy ({s}) | {pct(v['accuracy'])} | {v['macro_f1']:.4f} |" for s, v in r["baselines"].items()],
        "",
        "## Wczesna klasyfikacja (ile pakietów wystarczy?)",
        "",
        "Dla każdego N trenowany jest osobny model k-NN na cechach z pierwszych N pakietów. Modele oznaczone "
        "jako etap są częścią klasyfikatora w locie: dają wstępną decyzję zanim przepływ osiągnie "
        f"{r['features']['max_packets']} pakietów.",
        "",
        "| Pierwsze N pakietów | Accuracy | Etap klasyfikatora w locie |",
        "|---|---|---|",
        *[f"| {p['packets']} | {pct(p['accuracy'])} | {'tak' if p['stage'] else ''} |" for p in r["prefix"]],
        "",
        "![prefix](prefix.png)",
        "",
        "## Atak padding",
        "",
        f"Do rozmiaru każdego pakietu w zbiorze testowym dodano losowo od {pa['pad_range'][0]} do {pa['pad_range'][1]} B "
        f"({pa['repeats']} powtórzeń z różnym ziarnem). Model trenowany na czystym ruchu: "
        f"**{pct(pa['accuracy_mean'])}** (spadek o {pa['drop_pp']:.1f} pp).",
        "",
        "| Maks. padding X [B] | k-NN (trening czysty) | k-NN (trening z paddingiem) |",
        "|---|---|---|",
        *[f"| {s['max_pad']} | {pct(s['acc_clean_model'])} | {pct(s['acc_padding_aware_model'])} |" for s in pa["sweep"]],
        "",
        "![padding](padding_attack.png)",
        "",
        "![cm-pad](confusion_padding.png)",
        "",
        "## Wydajność",
        "",
        f"- Trening (zapamiętanie {r['dataset']['train']} wektorów + skaler): {perf['fit_seconds'] * 1e3:.1f} ms",
        f"- Ekstrakcja cech: {perf['feature_extraction_ms_per_flow']:.2f} ms / przepływ",
        f"- Predykcja pojedynczego przepływu (ekstrakcja + k-NN): mediana {lat['median']:.2f} ms, "
        f"p95 {lat['p95']:.2f} ms, p99 {lat['p99']:.2f} ms, max {lat['max']:.2f} ms",
        f"- Przepustowość wsadowa: {perf['batch_throughput_flows_per_s']:.0f} przepływów/s",
        f"- Środowisko: Python {r['environment']['python']}, scikit-learn {r['environment']['sklearn']}, "
        f"{r['environment']['platform']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
