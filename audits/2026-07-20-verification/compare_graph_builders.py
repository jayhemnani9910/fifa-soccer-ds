"""Throwaway comparison of build_track_graph vs build_track_graph_optimized.
Read-only against the repo; not written into the repo.
"""
import random
import sys

sys.path.insert(0, "/home/po/projects/work/fifa-soccer-ds")

from src.graph.build_graph import build_track_graph, build_track_graph_optimized
from src.track.bytetrack_runtime import Tracklet, Tracklets


def make_windows(n_frames, tracks_per_frame, seed=0, jitter=True):
    rng = random.Random(seed)
    windows = []
    for f in range(n_frames):
        items = []
        for t in range(tracks_per_frame):
            base_x = (t % 5) * 30.0
            base_y = (t // 5) * 30.0
            if jitter:
                base_x += rng.uniform(-2, 2)
                base_y += rng.uniform(-2, 2)
            bbox = [base_x, base_y, base_x + 20, base_y + 20]
            items.append(Tracklet(track_id=t, bbox=bbox, score=0.9))
        windows.append(Tracklets(frame_id=f, items=items))
    return windows


def compare(label, n_frames, tracks_per_frame, distance_threshold=80.0, window=30):
    windows = make_windows(n_frames, tracks_per_frame)
    r1 = build_track_graph(windows, window=window, distance_threshold=distance_threshold)
    r2 = build_track_graph_optimized(windows, window=window, distance_threshold=distance_threshold)

    n1, n2 = r1.x.shape[0], r2.x.shape[0]
    e1, e2 = r1.edge_index.shape[1], r2.edge_index.shape[1]
    same_nodes = n1 == n2
    same_edges = e1 == e2

    # compare meta sets
    meta1 = set(r1.meta)
    meta2 = set(r2.meta)
    same_meta = meta1 == meta2

    # compare edge sets by (frame,track) pairs rather than raw indices,
    # since node ordering / indexing differs subtly between the two builders
    def edge_pairs_by_meta(data):
        idx_to_meta = data.meta
        pairs = set()
        ei = data.edge_index
        for k in range(ei.shape[1]):
            a, b = ei[0, k].item(), ei[1, k].item()
            ma, mb = idx_to_meta[a], idx_to_meta[b]
            pairs.add(frozenset((ma, mb)) if ma != mb else (ma, mb))
        return pairs

    edges1 = edge_pairs_by_meta(r1)
    edges2 = edge_pairs_by_meta(r2)
    same_edge_semantics = edges1 == edges2

    print(f"[{label}] nodes: {n1} vs {n2} (equal={same_nodes})")
    print(f"[{label}] raw edge count: {e1} vs {e2} (equal={same_edges})")
    print(f"[{label}] meta sets equal: {same_meta}")
    print(f"[{label}] edge semantics (by frame/track pairs) equal: {same_edge_semantics}")
    if not same_edge_semantics:
        print(f"  only in original: {len(edges1 - edges2)} pairs, only in optimized: {len(edges2 - edges1)} pairs")
    print()


if __name__ == "__main__":
    print("=== Small case (well within limits) ===")
    compare("small-5x10", n_frames=5, tracks_per_frame=10)

    print("=== Dense frame case (>50 nodes/frame triggers original's 'skip dense frame' path) ===")
    compare("dense-3x60", n_frames=3, tracks_per_frame=60)

    print("=== Larger multi-frame case ===")
    compare("multi-30x20", n_frames=30, tracks_per_frame=20)
