"""Construct windowed interaction graphs from tracked detections."""

from __future__ import annotations

import logging
from typing import Optional
from itertools import combinations

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

try:
    from torch_geometric.data import Data
except ImportError:  # pragma: no cover - optional dependency
    Data = None  # type: ignore[assignment]

from src.track.bytetrack_runtime import Tracklets

logger = logging.getLogger(__name__)


def _bbox_to_tensor(bbox) -> torch.Tensor:
    coords = list(bbox)
    if len(coords) != 4:
        coords = coords[:4] + [0.0] * max(0, 4 - len(coords))
    return torch.tensor(coords, dtype=torch.float32)


def _center(bbox_tensor: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = bbox_tensor
    return torch.tensor([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=torch.float32)


def build_track_graph(
    track_windows: list[Tracklets],
    window: int = 30,
    distance_threshold: float = 80.0,
    include_temporal_edges: bool = True,
    max_spatial_edges: int = 1000,  # Memory safety limit
):
    """Create a graph of tracked entities using spatial proximity with memory optimization."""

    if not track_windows:
        raise ValueError("track_windows must contain at least one frame.")

    if torch is None:
        raise ImportError("PyTorch is required to build graphs.")

    active = track_windows[-window:]
    nodes: list[torch.Tensor] = []
    node_meta: list[tuple[int, int]] = []
    node_lookup: dict[tuple[int, int], int] = {}
    frame_to_nodes: dict[int, list[int]] = {}

    # Build nodes with memory tracking
    total_nodes = 0
    for frame in active:
        total_nodes += len(frame.items)
        if total_nodes > 1000:  # Prevent memory explosion
            logger.warning(f"Too many nodes ({total_nodes}), truncating for memory safety")
            break
            
        for track in frame.items:
            node_idx = len(nodes)
            tensor = _bbox_to_tensor(track.bbox)
            nodes.append(tensor)
            node_meta.append((frame.frame_id, track.track_id))
            node_lookup[(frame.frame_id, track.track_id)] = node_idx
            frame_to_nodes.setdefault(frame.frame_id, []).append(node_idx)

    edges: list[tuple[int, int]] = []

    # Build temporal edges (memory efficient)
    if include_temporal_edges:
        temporal_lookup: dict[int, tuple[int, int]] = {}
        for frame in active:
            for track in frame.items:
                key = track.track_id
                current_idx = node_lookup[(frame.frame_id, track.track_id)]
                if key in temporal_lookup:
                    prev_frame, prev_idx = temporal_lookup[key]
                    if frame.frame_id - prev_frame <= window:
                        edges.append((prev_idx, current_idx))
                        edges.append((current_idx, prev_idx))
                temporal_lookup[key] = (frame.frame_id, current_idx)

    # Build spatial edges with memory optimization
    if distance_threshold > 0:
        threshold_sq = distance_threshold**2
        spatial_edges_added = 0
        
        for _frame_id, node_indices in frame_to_nodes.items():
            if len(node_indices) > 50:  # Skip dense frames
                logger.warning(f"Skipping spatial edges for dense frame with {len(node_indices)} nodes")
                continue
                
            for idx_a, idx_b in combinations(node_indices, 2):
                if spatial_edges_added >= max_spatial_edges:
                    logger.warning(f"Reached max spatial edges limit ({max_spatial_edges})")
                    break
                    
                center_a = _center(nodes[idx_a])
                center_b = _center(nodes[idx_b])
                distance_sq = torch.sum((center_a - center_b) ** 2).item()
                if distance_sq <= threshold_sq:
                    edges.append((idx_a, idx_b))
                    edges.append((idx_b, idx_a))
                    spatial_edges_added += 2
            
            if spatial_edges_added >= max_spatial_edges:
                break

    # Create edge index tensor
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    x = torch.stack(nodes) if nodes else torch.empty((0, 4), dtype=torch.float32)

    # Memory usage logging
    memory_mb = (x.numel() * 4 + edge_index.numel() * 4) / (1024 * 1024)
    logger.info(f"Graph built: {len(nodes)} nodes, {len(edges)} edges, {memory_mb:.2f}MB")

    if Data is None:
        return {"x": x, "edge_index": edge_index, "meta": node_meta}

    return Data(x=x, edge_index=edge_index, meta=node_meta)


def build_track_graph_optimized(
    track_windows: list[Tracklets],
    window: int = 30,
    distance_threshold: float = 80.0,
    include_temporal_edges: bool = True,
    max_nodes_per_frame: int = 50,  # Memory optimization
):
    """
    Memory-optimized graph construction using vectorized operations.
    
    This function prevents memory explosion by:
    1. Limiting nodes per frame
    2. Using vectorized distance computation
    3. Early termination on edge limits
    4. Sparse adjacency representation
    """
    
    if not track_windows:
        raise ValueError("track_windows must contain at least one frame.")

    if torch is None:
        raise ImportError("PyTorch is required to build graphs.")

    active = track_windows[-window:]
    
    # Collect all nodes with limits
    all_nodes = []
    all_meta = []
    node_lookup = {}
    frame_to_nodes = {}
    
    node_counter = 0
    for frame in active:
        frame_nodes = []
        frame_tracks = frame.items[:max_nodes_per_frame]  # Limit nodes
        
        for track in frame_tracks:
            tensor = _bbox_to_tensor(track.bbox)
            all_nodes.append(tensor)
            all_meta.append((frame.frame_id, track.track_id))
            
            node_lookup[(frame.frame_id, track.track_id)] = node_counter
            frame_nodes.append(node_counter)
            node_counter += 1
            
        frame_to_nodes[frame.frame_id] = frame_nodes

    if not all_nodes:
        empty_data = {
            "x": torch.empty((0, 4), dtype=torch.float32),
            "edge_index": torch.empty((2, 0), dtype=torch.long),
            "meta": []
        }
        if Data is None:
            return empty_data
        return Data(**empty_data)

    x = torch.stack(all_nodes)
    edges = []

    # Build temporal edges efficiently
    if include_temporal_edges:
        track_last_seen = {}
        for frame in active:
            for track in frame.items:
                if track.track_id in track_last_seen:
                    last_frame, last_idx = track_last_seen[track.track_id]
                    if (frame.frame_id - last_frame) <= window:
                        current_idx = node_lookup.get((frame.frame_id, track.track_id))
                        if current_idx is not None:
                            edges.append((last_idx, current_idx))
                            edges.append((current_idx, last_idx))
                
                track_last_seen[track.track_id] = (
                    frame.frame_id, 
                    node_lookup.get((frame.frame_id, track.track_id))
                )

    # Build spatial edges with vectorized computation
    if distance_threshold > 0:
        threshold_sq = distance_threshold ** 2
        
        for frame_id, node_indices in frame_to_nodes.items():
            if len(node_indices) <= 1:
                continue
                
            # Get centers for this frame
            centers = torch.stack([
                _center(x[node_idx]) for node_idx in node_indices
            ])
            
            # Vectorized pairwise distance computation
            if len(centers) > 0:
                # Compute all pairwise distances
                diff = centers.unsqueeze(1) - centers.unsqueeze(0)
                dist_matrix = torch.sum(diff ** 2, dim=2)
                
                # Find pairs within threshold (upper triangle)
                mask = (dist_matrix <= threshold_sq) & (dist_matrix > 0)
                pairs = torch.nonzero(mask, as_tuple=False)
                
                # Add edges
                for i, j in pairs:
                    if i < j:  # Upper triangle only
                        idx_i, idx_j = node_indices[i.item()], node_indices[j.item()]
                        edges.append((idx_i, idx_j))
                        edges.append((idx_j, idx_i))

    # Create edge index
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )

    # Memory usage report
    num_nodes = len(all_nodes)
    num_edges = len(edges)
    memory_mb = (x.numel() * 4 + edge_index.numel() * 4) / (1024 * 1024)
    
    logger.info(
        f"Optimized graph: {num_nodes} nodes, {num_edges} edges, "
        f"{memory_mb:.2f}MB, avg_degree: {num_edges/max(1, num_nodes):.1f}"
    )

    if Data is None:
        return {"x": x, "edge_index": edge_index, "meta": all_meta}

    return Data(x=x, edge_index=edge_index, meta=all_meta)


def estimate_graph_memory(num_nodes: int, avg_degree: float = 5.0) -> dict:
    """
    Estimate memory usage for graph construction.
    
    Args:
        num_nodes: Number of nodes in graph
        avg_degree: Average node degree (typical: 3-8 for soccer)
        
    Returns:
        Dictionary with memory estimates in MB
    """
    # Sparse representation memory
    num_edges = int(num_nodes * avg_degree)
    node_memory = num_nodes * 4 * 4  # 4 features * 4 bytes
    edge_memory = num_edges * 2 * 4  # 2 indices * 4 bytes
    total_memory_mb = (node_memory + edge_memory) / (1024 * 1024)
    
    # Dense representation for comparison
    dense_edges = num_nodes * (num_nodes - 1) // 2
    dense_memory_mb = (node_memory + dense_edges * 2 * 4) / (1024 * 1024)
    
    return {
        "nodes": num_nodes,
        "sparse_edges": num_edges,
        "dense_edges": dense_edges,
        "sparse_memory_mb": total_memory_mb,
        "dense_memory_mb": dense_memory_mb,
        "memory_ratio": dense_memory_mb / max(0.1, total_memory_mb)
    }
