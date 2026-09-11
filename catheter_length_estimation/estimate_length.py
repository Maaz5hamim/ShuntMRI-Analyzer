import pandas as pd
import numpy as np
import ants
import os
import joblib
import networkx as nx
from tqdm import tqdm
from scipy.spatial import distance_matrix
from scipy.ndimage import binary_closing
from scipy.interpolate import splprep, splev

def downsample_coords(coords_mm, step_mm=2.0):
    if len(coords_mm) == 0:
        return coords_mm
    kept = [coords_mm[0]]
    for pt in coords_mm[1:]:
        if np.linalg.norm(pt - kept[-1]) >= step_mm:
            kept.append(pt)
    return np.array(kept)

def mask_to_spine(cath_mask, spacing, bridge_dist_mm=2.0, downsample_mm=2.0):
    coords = np.argwhere(cath_mask > 0)
    if len(coords) < 2:
        return None
    coords_mm = downsample_coords(coords * spacing, step_mm=downsample_mm)
    if len(coords_mm) < 2:
        return None

    dist_mat = distance_matrix(coords_mm, coords_mm)
    adj = (dist_mat <= bridge_dist_mm) & (dist_mat > 0)
    G = nx.from_numpy_array(adj.astype(float))
    for u, v in G.edges():
        G[u][v]['weight'] = dist_mat[u, v]

    if not nx.is_connected(G):
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        G = G.subgraph(components[0]).copy()
        coords_mm = coords_mm[list(components[0])]
        G = nx.convert_node_labels_to_integers(G)

    T = nx.minimum_spanning_tree(G, weight='weight')
    dist_a = nx.single_source_shortest_path_length(T, 0)
    node_a = max(dist_a, key=dist_a.get)
    dist_b = nx.single_source_shortest_path_length(T, node_a)
    node_b = max(dist_b, key=dist_b.get)
    return coords_mm[nx.shortest_path(T, node_a, node_b)]

def smooth_and_measure(path_coords, smoothing_factor=100):
    if len(path_coords) < 4:
        arc_len = sum(np.linalg.norm(path_coords[j+1] - path_coords[j])
                      for j in range(len(path_coords)-1))
        return arc_len, None
    try:
        tck, _ = splprep(path_coords.T.tolist(), s=smoothing_factor, k=3)
    except Exception:
        arc_len = sum(np.linalg.norm(path_coords[j+1] - path_coords[j])
                      for j in range(len(path_coords)-1))
        return arc_len, None
    pts = np.array(splev(np.linspace(0, 1, 1000), tck)).T
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))), tck

def get_artifact_side_tip(path_coords, artifact_mask, spacing):
    art_mm = np.argwhere(artifact_mask > 0) * spacing
    if len(art_mm) == 0:
        return path_coords[-1], path_coords
    da = np.min(np.linalg.norm(art_mm - path_coords[0],  axis=1))
    db = np.min(np.linalg.norm(art_mm - path_coords[-1], axis=1))
    if da < db:
        path_coords = path_coords[::-1]
    return path_coords[-1], path_coords

def find_artifact_entry_point(path_coords, artifact_mask, spacing):
    for i in range(len(path_coords)-1, -1, -1):
        vox = np.clip(np.round(path_coords[i] / spacing).astype(int),
                      0, np.array(artifact_mask.shape)-1)
        if artifact_mask[vox[0], vox[1], vox[2]] == 0:
            return i, path_coords[i]
    return 0, path_coords[0]

def extract_features(visible_coords, visible_len, tck, art_mask, spacing):
    feats = {}
    feats['visible_len'] = visible_len

    if len(visible_coords) >= 3:
        angles = []
        for i in range(1, len(visible_coords)-1):
            v1 = visible_coords[i]   - visible_coords[i-1]
            v2 = visible_coords[i+1] - visible_coords[i]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 0 and n2 > 0:
                cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
                angles.append(np.degrees(np.arccos(cos_a)))
        feats['mean_curvature_deg'] = float(np.mean(angles)) if angles else 0.0
        feats['max_curvature_deg']  = float(np.max(angles))  if angles else 0.0
    else:
        feats['mean_curvature_deg'] = 0.0
        feats['max_curvature_deg']  = 0.0

    if tck is not None:
        dx, dy, dz = splev(1.0, tck, der=1)
        tip_dir = np.array([dx, dy, dz], dtype=float)
    else:
        tip_dir = visible_coords[-1] - visible_coords[-2]
    tip_dir /= (np.linalg.norm(tip_dir) + 1e-8)

    bulk_center = visible_coords[:max(1, len(visible_coords)//2)].mean(axis=0)
    if np.dot(tip_dir, visible_coords[-1] - bulk_center) < 0:
        tip_dir = -tip_dir

    art_mm = np.argwhere(art_mask > 0) * spacing
    if len(art_mm) == 0:
        feats['art_volume_mm3']       = 0.0
        feats['art_extent_along_tip'] = 0.0
        feats['art_tip_dist_mm']      = 999.0
        feats['art_n_voxels']         = 0
    else:
        feats['art_volume_mm3']  = float(len(art_mm) * np.prod(spacing))
        feats['art_n_voxels']    = len(art_mm)
        tip_pt = visible_coords[-1]
        proj   = (art_mm - tip_pt) @ tip_dir
        feats['art_extent_along_tip'] = float(proj[proj > 0].max()) if (proj > 0).any() else 0.0
        feats['art_tip_dist_mm']      = float(np.min(np.linalg.norm(art_mm - tip_pt, axis=1)))

    feats['endpoint_dist_mm'] = float(np.linalg.norm(visible_coords[-1] - visible_coords[0]))
    feats['n_spine_pts']      = len(visible_coords)

    return feats


FEATURE_COLS = [
    'visible_len',
    'mean_curvature_deg',
    'max_curvature_deg',
    'art_volume_mm3',
    'art_extent_along_tip',
    'art_tip_dist_mm',
    'art_n_voxels',
    'endpoint_dist_mm',
    'n_spine_pts',
]

def run_inference(raw_mask, art_mask, spacing, model_path,
                   smoothing_factor=100, downsample_mm=2.0, bridge_dist_mm=2.0):

    model, feat_cols = joblib.load(model_path)

    cath_mask = binary_closing(raw_mask > 0,
                                structure=np.ones((3,3,3))).astype(np.uint8)

    # Spine extraction
    path_coords = mask_to_spine(cath_mask, spacing,
                                    bridge_dist_mm=bridge_dist_mm,
                                    downsample_mm=downsample_mm)
    if path_coords is None or len(path_coords) < 2:
        return 0

    _, path_coords = get_artifact_side_tip(path_coords, art_mask, spacing)
    entry_idx, _   = find_artifact_entry_point(path_coords, art_mask, spacing)
    visible_coords = path_coords[:entry_idx + 1]
    if len(visible_coords) < 2:
        return visible_coords

    visible_len, tck = smooth_and_measure(visible_coords,
                                            smoothing_factor=smoothing_factor)

    # Feature extraction → model prediction
    feats      = extract_features(visible_coords, visible_len, tck,
                                    art_mask, spacing)
    feat_vec   = np.array([[feats[c] for c in feat_cols]])
    hidden_est = float(np.clip(model.predict(feat_vec)[0], 0, 40))

    total_pred = visible_len + hidden_est

    return total_pred
