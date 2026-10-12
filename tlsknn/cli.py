from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DEFAULT_CACHE = Path("data/flows_cache.npz")
DEFAULT_RAW = Path("data/raw/pretraining.zip")
DEFAULT_MODEL = Path("models/knn.joblib")
DEFAULT_RESULTS = Path("results")
DEFAULT_MEAN_LENGTH_RESULTS = Path("results/baseline_mean_packet_length")

# Google Drive, folder 1Pvev0hJ82usPh6dWDlz7Lv8L6h3JpWhE
DATASET_FILES = {
    "pretraining": ("1o3clzI1iptkSyiDJC94TTo44Qe_bn5ge", "pretraining.zip"),
    "script": ("16UMnu7JMZGtabFL0DuP_5ufH_BJKI8E2", "Retraining(script-triggered).zip"),
    "human": ("1RuvcfMNE9vzaEZXI-2t1WP7lmd03SSJ8", "Retraining(human-triggered).zip"),
}


def _load_flows(args):
    from .dataset import load_cache

    if not Path(args.cache).exists():
        sys.exit(f"Brak {args.cache}. Najpierw uruchom: python -m tlsknn prepare")
    return load_cache(args.cache)


def _load_model(path):
    from .model import TrafficClassifier

    if not Path(path).exists():
        sys.exit(f"Brak modelu {path}. Najpierw uruchom: python -m tlsknn train")
    return TrafficClassifier.load(path)


def cmd_download(args):
    import gdown

    file_id, name = DATASET_FILES[args.part]
    out = Path(args.out) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"{out} już istnieje - pomijam.")
        return
    print(f"Pobieranie {name} (UC Davis QUIC dataset) -> {out}")
    gdown.download(id=file_id, output=str(out), quiet=False)


def cmd_prepare(args):
    from .dataset import load_any, save_cache

    print(f"Wczytywanie {args.source} ...")
    flows = load_any(args.source, limit=args.cache_packets, max_per_class=args.max_per_class)
    Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
    save_cache(flows, args.cache)
    labels, counts = np.unique([f.label for f in flows], return_counts=True)
    print(f"Zapisano {len(flows)} przepływów do {args.cache}: "
          + ", ".join(f"{a}={b}" for a, b in zip(labels, counts)))


def cmd_train(args):
    from .experiment import run

    run(_load_flows(args), Path(args.results), Path(args.model), max_packets=args.max_packets,
        seed=args.seed, k=args.k)


def cmd_baseline_mean_length(args):
    from .mean_length_experiment import run_mean_packet_length

    run_mean_packet_length(_load_flows(args), Path(args.results), Path(args.model),
                           Path(args.main_metrics) if args.main_metrics else None,
                           max_packets=args.max_packets, seed=args.seed)


def cmd_classify(args):
    from .dataset import load_flow_file, load_ucdavis
    from .pcap import flows_from_pcap

    clf = _load_model(args.model)
    for target in args.files:
        p = Path(target)
        if p.suffix.lower() in (".pcap", ".pcapng", ".cap"):
            flows = flows_from_pcap(p, limit=clf.max_packets)
        elif p.is_dir() or p.suffix.lower() == ".zip":
            flows = load_ucdavis(p, limit=clf.max_packets)
        else:
            flows = [load_flow_file(p, clf.max_packets)]
        for f in flows:
            r = clf.predict_flow(f.times, f.sizes, f.dirs)
            print(f"{f.name:<40} {len(f):>5} pkt -> {r.label:<13} ({r.category}, {r.confidence:.0%})  "
                  f"{r.latency_ms:.2f} ms")


def cmd_pcap(args):
    from .monitor import Monitor
    from .pcap import iter_pcap

    mon = Monitor(_load_model(args.model), checkpoints=args.checkpoints, quiet=args.quiet,
                  idle_timeout=args.idle_timeout)
    for m in iter_pcap(args.file, frozenset(args.ports)):
        mon.packet(m)
    mon.finish()
    mon.print_summary()


def cmd_live(args):
    from .monitor import Monitor
    from .pcap import sniff_live

    mon = Monitor(_load_model(args.model), checkpoints=args.checkpoints, quiet=args.quiet,
                  idle_timeout=args.idle_timeout)
    print(f"Nasłuch na {args.iface or 'domyślnym interfejsie'} (porty {args.ports}); Ctrl+C kończy.", flush=True)
    try:
        sniff_live(mon.packet, iface=args.iface, ports=frozenset(args.ports), timeout=args.duration)
    except KeyboardInterrupt:
        pass
    except (RuntimeError, OSError, PermissionError) as e:
        sys.exit(f"Przechwytywanie nieudane: {e}\nNa Windows wymagany Npcap (https://npcap.com) i konsola "
                 "administratora; alternatywnie użyj 'pcap' lub 'replay'.")
    mon.finish()
    mon.print_summary()


def cmd_replay(args):
    from .monitor import replay

    clf = _load_model(args.model)
    flows = _load_flows(args)
    test_names = set(clf.meta.get("test_flow_names", []))
    pool = [f for f in flows if f"{f.label}/{f.name}" in test_names] or flows
    if not test_names:
        print("UWAGA: model nie zawiera listy przepływów testowych - replay na całym zbiorze.")
    rng = np.random.default_rng(args.seed)
    chosen = [pool[i] for i in rng.choice(len(pool), size=min(args.flows, len(pool)), replace=False)]
    print(f"Replay {len(chosen)} przepływów ze zbioru testowego, "
          f"model k={clf.k}, decyzja po {clf.max_packets} pkt\n")
    mon = replay(clf, chosen, speed=args.speed, spread=args.spread, seed=args.seed, quiet=args.quiet,
                 idle_timeout=args.idle_timeout)
    mon.print_summary()


def cmd_info(args):
    clf = _load_model(args.model)
    m = clf.meta
    print(f"Model: {args.model}\n  k = {clf.k}\n  decyzja końcowa po {clf.max_packets} pakietach\n"
          f"  etapy (pakiety): {', '.join(map(str, clf.stages))}\n"
          f"  klasy: {', '.join(clf.classes)}\n  trening: {m.get('trained_on')} przepływów, "
          f"accuracy (test) = {m.get('test_accuracy', float('nan')):.4f}\n  utworzony: {m.get('created')}"
          + ("\n  wariant: odporny na padding" if m.get("padding_aware") else ""))


def build_parser() -> argparse.ArgumentParser:
    from .features import DEFAULT_MAX_PACKETS

    p = argparse.ArgumentParser(prog="python -m tlsknn",
                                description="Identyfikacja aplikacji w szyfrowanym ruchu HTTPS/TLS/QUIC "
                                            "na podstawie rozmiarów pakietów i czasów (k-NN).")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("download", help="pobierz UC Davis QUIC dataset")
    s.add_argument("--part", choices=sorted(DATASET_FILES), default="pretraining")
    s.add_argument("--out", default=str(DEFAULT_RAW.parent))
    s.set_defaults(func=cmd_download)

    s = sub.add_parser("prepare", help="wczytaj dataset (UC Davis lub katalog z PCAP) do cache")
    s.add_argument("--source", default=str(DEFAULT_RAW))
    s.add_argument("--cache", default=str(DEFAULT_CACHE))
    s.add_argument("--cache-packets", type=int, default=5000, help="ile pierwszych pakietów przepływu zachować")
    s.add_argument("--max-per-class", type=int, default=None, help="ogranicz próbkę (np. 500 przepływów/klasę)")
    s.set_defaults(func=cmd_prepare)

    s = sub.add_parser("train", help="dobór k, trening, ewaluacja, atak padding, raport")
    s.add_argument("--cache", default=str(DEFAULT_CACHE))
    s.add_argument("--model", default=str(DEFAULT_MODEL))
    s.add_argument("--results", default=str(DEFAULT_RESULTS))
    s.add_argument("--max-packets", type=int, default=DEFAULT_MAX_PACKETS)
    s.add_argument("--k", type=int, default=None, help="wymuś k (domyślnie: dobór przez CV)")
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("baseline-mean-length",
                       help="ewaluacja dummy: tylko średnia długość pakietu (osobne wyniki, bez treningu k-NN)")
    s.add_argument("--cache", default=str(DEFAULT_CACHE))
    s.add_argument("--model", default=str(DEFAULT_MODEL), help="model główny — tylko meta podziału testowego")
    s.add_argument("--main-metrics", default=str(DEFAULT_RESULTS / "metrics.json"),
                   help="metrics.json k-NN do linii odniesienia na wykresach")
    s.add_argument("--results", default=str(DEFAULT_MEAN_LENGTH_RESULTS))
    s.add_argument("--max-packets", type=int, default=DEFAULT_MAX_PACKETS)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_baseline_mean_length)

    def monitor_opts(s):
        s.add_argument("--model", default=str(DEFAULT_MODEL))
        s.add_argument("--checkpoints", type=int, nargs="*", default=None,
                       help="progi liczby pakietów dla wstępnych predykcji (domyślnie: etapy modelu)")
        s.add_argument("--quiet", action="store_true", help="tylko podsumowanie")
        s.add_argument("--idle-timeout", type=float, default=15.0,
                       help="po ilu sekundach ciszy przepływ uznać za zakończony")

    s = sub.add_parser("replay", help="odtwarzanie przepływów testowych jak ruchu na żywo")
    monitor_opts(s)
    s.add_argument("--cache", default=str(DEFAULT_CACHE))
    s.add_argument("--flows", type=int, default=20)
    s.add_argument("--speed", type=float, default=0.0, help="mnożnik czasu rzeczywistego (0 = bez czekania)")
    s.add_argument("--spread", type=float, default=5.0, help="rozrzut startów przepływów [s]")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_replay)

    s = sub.add_parser("pcap", help="klasyfikacja pliku PCAP")
    s.add_argument("file")
    s.add_argument("--ports", type=int, nargs="*", default=[443])
    monitor_opts(s)
    s.set_defaults(func=cmd_pcap)

    s = sub.add_parser("live", help="przechwytywanie z interfejsu sieciowego")
    s.add_argument("--iface", default=None)
    s.add_argument("--ports", type=int, nargs="*", default=[443])
    s.add_argument("--duration", type=float, default=None, help="czas nasłuchu [s] (domyślnie do Ctrl+C)")
    monitor_opts(s)
    s.set_defaults(func=cmd_live)

    s = sub.add_parser("classify", help="klasyfikacja plików .txt (UC Davis) lub PCAP")
    s.add_argument("files", nargs="+")
    s.add_argument("--model", default=str(DEFAULT_MODEL))
    s.set_defaults(func=cmd_classify)

    s = sub.add_parser("info", help="informacje o modelu")
    s.add_argument("--model", default=str(DEFAULT_MODEL))
    s.set_defaults(func=cmd_info)
    return p


def main(argv=None) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
