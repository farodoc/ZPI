import os

import numpy as np
import pytest

from tlsknn.dataset import Flow, load_cache, save_cache
from tlsknn.features import extract, extract_many, feature_names, pad_sizes
from tlsknn.model import TrafficClassifier, build_pipeline
from tlsknn.monitor import replay
from tlsknn.pcap import FlowTracker, flows_from_pcap, load_scapy


def synthetic_flows(n_per_class=40, seed=0):
    rng = np.random.default_rng(seed)
    flows = []
    for i in range(n_per_class):
        n = int(rng.integers(30, 150))
        flows.append(Flow("video", f"v{i}", np.cumsum(rng.exponential(0.002, n)),
                          rng.integers(1200, 1400, n), (rng.random(n) < 0.1).astype(np.int8)))
        flows.append(Flow("web", f"w{i}", np.cumsum(rng.exponential(0.05, n)),
                          rng.integers(60, 600, n), (rng.random(n) < 0.5).astype(np.int8)))
    return flows


def test_feature_vector_shape_and_finite():
    f = synthetic_flows(1)[0]
    x = extract(f.times, f.sizes, f.dirs)
    assert x.shape == (len(feature_names()),)
    assert np.isfinite(x).all()


def test_features_robust_to_edge_cases():
    assert np.isfinite(extract(np.array([0.0]), np.array([100]), np.array([1]))).all()
    assert np.isfinite(extract(np.array([0.0, 0.5, 0.2]), np.array([100, 50, 70]), np.array([0, 0, 0]))).all()


def test_padding_adds_between_1_and_100_bytes():
    s = np.full(10_000, 500)
    p = pad_sizes(s, np.random.default_rng(0))
    assert (p - s).min() >= 1 and (p - s).max() <= 100


def test_cache_roundtrip(tmp_path):
    flows = synthetic_flows(3)
    save_cache(flows, tmp_path / "c.npz")
    back = load_cache(tmp_path / "c.npz")
    assert [f.label for f in back] == [f.label for f in flows]
    np.testing.assert_array_equal(back[0].sizes, flows[0].sizes)


def test_multistage_classifier_save_load(tmp_path):
    flows = synthetic_flows()
    y = [f.label for f in flows]
    stages = {n: build_pipeline(3).fit(extract_many(flows, n), y) for n in (10, 100)}
    clf = TrafficClassifier(stages, {"k": 3})
    clf.save(tmp_path / "m.joblib")
    clf = TrafficClassifier.load(tmp_path / "m.joblib")
    assert clf.stage_for(5) == 10 and clf.stage_for(37) == 100 and clf.stage_for(5000) == 100
    test = synthetic_flows(10, seed=1)
    acc = np.mean([clf.predict_flow(f.times, f.sizes, f.dirs).label == f.label for f in test])
    assert acc > 0.9


def test_replay_matches_labels():
    flows = synthetic_flows()
    y = [f.label for f in flows]
    clf = TrafficClassifier({n: build_pipeline(1).fit(extract_many(flows, n), y) for n in (10, 100)})
    mon = replay(clf, synthetic_flows(10, seed=2), quiet=True, idle_timeout=1e9)
    s = mon.summary()
    assert s["flows"] == 20 and s["accuracy"] > 0.9


def test_tracker_checkpoints_and_close():
    events = []
    tr = FlowTracker(lambda f, r: events.append(r), checkpoints=(2, 4), max_packets=4)
    from tlsknn.pcap import PacketMeta
    key = ("UDP", ("1.1.1.1", 443), ("10.0.0.1", 5000))
    for i in range(5):
        src, dst = (("10.0.0.1", 5000), ("1.1.1.1", 443)) if i % 2 == 0 else (("1.1.1.1", 443), ("10.0.0.1", 5000))
        tr.add(PacketMeta(float(i), key, src, dst, 100 + i))
    tr.flush()
    assert events == ["2 pkt", "4 pkt"]


def test_pcap_roundtrip_tcp_and_udp(tmp_path):
    sc = load_scapy()
    pkts = []
    for i, size in enumerate([300, 1200, 50, 900]):
        up = i % 2 == 0
        ip = sc.IP(src="10.0.0.2", dst="8.8.8.8") if up else sc.IP(src="8.8.8.8", dst="10.0.0.2")
        l4 = sc.UDP(sport=50000, dport=443) if up else sc.UDP(sport=443, dport=50000)
        p = sc.Ether(src="02:00:00:00:00:01", dst="02:00:00:00:00:02") / ip / l4 / os.urandom(size)
        p.time = 100 + i * 0.01
        pkts.append(p)
    for i, size in enumerate([500, 0, 700]):  # 0 = czysty ACK
        p = (sc.Ether(src="02:00:00:00:00:01", dst="02:00:00:00:00:02") / sc.IP(src="10.0.0.3", dst="1.2.3.4")
             / sc.TCP(sport=40000, dport=443, flags="PA" if size else "A") / os.urandom(size))
        p.time = 200 + i * 0.01
        pkts.append(p)
    path = tmp_path / "t.pcap"
    sc.wrpcap(str(path), pkts)
    flows = sorted(flows_from_pcap(path), key=lambda f: f.name)
    assert len(flows) == 2
    tcp, udp = flows
    assert tcp.name.startswith("TCP") and list(tcp.sizes) == [500, 700]
    assert list(udp.sizes) == [300, 1200, 50, 900] and list(udp.dirs) == [1, 0, 1, 0]
    assert udp.times[0] == 0 and udp.times[-1] == pytest.approx(0.03)
