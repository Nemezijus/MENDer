# find_SVD_ensembles.py
# Python 3.10
# Requires:
# numpy==1.23.5, scipy==1.8.1, pandas==1.5.3, scikit-learn==1.2.2,
# matplotlib==3.6.3, h5py==3.8.0, joblib==1.2.0, threadpoolctl==3.1.0, allensdk>=2.16,<3

import os
import sys
import math
import warnings
from typing import Dict, Tuple, List, Optional

import numpy as np
import scipy.io as sio
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
matplotlib.use("Agg")  # for non-interactive save
import matplotlib.pyplot as plt


# ---------------------------
# Utility: MATLAB-like helpers
# ---------------------------

def _ensure_binary(a: np.ndarray) -> np.ndarray:
    """Convert to {0,1} int array."""
    return (a != 0).astype(np.int8)

def _histc_like(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """MATLAB histc-like: counts per bin edge list (including rightmost edge).
    Returns counts same length as bins.
    """
    # numpy histogram excludes rightmost edge by default; emulate MATLAB-ish behavior
    counts, edges = np.histogram(x, bins=np.append(bins, bins[-1] + (bins[-1] - bins[-2] if len(bins) > 1 else 1)))
    return counts.astype(float)

def _cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """1 - cosine distance, for columns-as-samples."""
    # X: (features x samples) -> we want pairwise similarity among columns
    # cdist expects rows as observations, so transpose
    d = cdist(X.T, X.T, metric='cosine')  # distance
    sim = 1.0 - d
    return sim

def _cosine_similarity_to_vector(X: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute 1 - cosine distance between each column of X and vector v (same feature length)."""
    # cdist expects rows as obs; build a 1xN vector
    d = cdist(X.T, v.reshape(1, -1), metric='cosine').ravel()
    sim = 1.0 - d
    # If v is all-zeros, cosine distance is undefined -> set sim to zeros
    if np.all(v == 0):
        sim[:] = 0.0
    # Replace NaNs from numerical issues with 0
    sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)
    return sim

def _jaccard_similarity_matrix(B: np.ndarray) -> np.ndarray:
    """1 - jaccard distance for binary matrices; compares rows as observations, so we adapt accordingly.
       We need frame-frame jaccard on a binary matrix of neighbor sets S_tib (T' x T').
    """
    # cdist with 'jaccard' expects boolean or 0/1; compares rows
    # We want similarity among rows of B => directly:
    d = cdist(B, B, metric='jaccard')
    return 1.0 - d

def _sortrows_matlab(A: np.ndarray) -> np.ndarray:
    """Emulate MATLAB sortrows(A) for 2D array by lexicographic sort on columns ascending."""
    # A shape: (rows, cols)
    if A.size == 0:
        return A
    # lexsort uses last key first; provide keys in reverse column order
    keys = [A[:, i] for i in reversed(range(A.shape[1]))]
    idx = np.lexsort(keys)
    return A[idx, :]


# ---------------------------
# Shuffling (surrogates)
# ---------------------------

def shuffle(x: np.ndarray, method: str = 'frames') -> np.ndarray:
    """
    Shuffles spike raster data (N x T) with multiple methods.
    Reimplements MATLAB version.

    Methods:
      'frames'     - permute columns (frames)
      'time'       - permute each neuron's time independently (preserve per-cell spike count)
      'time_shift' - circularly shift each neuron's time series by random amount
      'isi'        - shuffle inter-spike intervals within each cell
      'cell'       - permute spikes across cells for each frame (preserve per-frame count)
      'exchange'   - exchange pairs of spikes across cells (preserve per-cell and per-frame totals)

    Note: Only 'time' is required by your pipeline, but the others are included for completeness.
    """
    if method not in {'frames', 'time', 'time_shift', 'isi', 'cell', 'exchange'}:
        method = 'frames'

    x = _ensure_binary(x)
    N, T = x.shape
    shuffled = x.copy()

    if method == 'frames':
        perm = np.random.permutation(T)
        shuffled = shuffled[:, perm]

    elif method == 'time':
        for i in range(N):
            shuffled[i, :] = shuffled[i, np.random.permutation(T)]

    elif method == 'time_shift':
        for i in range(N):
            shift = np.random.randint(T) if T > 0 else 0
            if shift > 0:
                shuffled[i, :] = np.concatenate([x[i, -shift:], x[i, :-shift]])
            else:
                shuffled[i, :] = x[i, :]

    elif method == 'isi':
        # reconstruct spikes from randomly permuted ISIs
        shuffled = np.zeros_like(x)
        for i in range(N):
            idx = np.flatnonzero(x[i, :])
            # emulate: isi = diff(find([1 x(i,:) 1]))
            # Build augmented with sentinel spikes at 0 and T+1 to get ISIs
            aug = np.concatenate(([0], idx + 1, [T + 1]))
            isi = np.diff(aug)
            if isi.size == 0:
                continue
            # Randomize ISIs
            isi_perm = np.random.permutation(isi)
            # Reconstruct spike times (exclude the last boundary ISI)
            cum = np.cumsum(isi_perm)[:-1]
            # place spikes at cum-1 (back to 0-based)
            valid = (cum > 0) & (cum <= T)
            shuffled[i, (cum[valid] - 1).astype(int)] = 1

    elif method == 'cell':
        for t in range(T):
            perm = np.random.permutation(N)
            col = x[:, t]
            temp = np.column_stack((perm, col))
            temp = temp[temp[:, 0].argsort()]  # sortrows by perm
            shuffled[:, t] = temp[:, 1]

    elif method == 'exchange':
        # slow; preserves row and column sums by swapping spike positions
        r, c = np.nonzero(shuffled)
        n = len(r)
        if n == 0:
            return shuffled
        for _ in range(2 * n):
            a, b = np.random.randint(0, n, size=2)
            # enforce they’re different and no collisions
            if a == b:
                continue
            ra, ca = r[a], c[a]
            rb, cb = r[b], c[b]
            if ra == rb or ca == cb:
                continue
            if shuffled[rb, ca] or shuffled[ra, cb]:
                continue
            # swap
            shuffled[rb, ca] = 1
            shuffled[ra, ca] = 0
            shuffled[ra, cb] = 1
            shuffled[rb, cb] = 0
        # Note: r,c not updated iteratively for speed; acceptable approximation of behavior

    return shuffled


# ---------------------------
# TF-IDF
# ---------------------------

def calcTFIDF(data: np.ndarray) -> np.ndarray:
    """
    TF-IDF normalization of raster data (N x T).
    Mirrors your MATLAB code including your custom fix for infs.
    """
    data = data.astype(float, copy=False)
    N, T = data.shape

    # term frequency per frame (normalize each column by its sum)
    colsum = np.sum(data, axis=0, keepdims=True)  # 1 x T
    with np.errstate(divide='ignore', invalid='ignore'):
        tf = data / np.where(colsum == 0, 1.0, colsum)

    # inverse document frequency: T / (df), then log
    rowsum = np.sum(data, axis=1, keepdims=True)  # N x 1
    with np.errstate(divide='ignore', invalid='ignore'):
        idf = T / np.where(rowsum == 0, np.nan, rowsum)
    idf = np.log(idf)
    # custom fix from your MATLAB: set NaN/Inf to 0
    idf = np.where(np.isfinite(idf), idf, 0.0)

    data_tfidf = tf * idf
    # Replace residual NaNs from zero-division with zeros
    data_tfidf = np.nan_to_num(data_tfidf, nan=0.0, posinf=0.0, neginf=0.0)
    return data_tfidf


# ---------------------------
# Thresholds from shuffled nulls
# ---------------------------

def findActiveFramesCustom(data: np.ndarray, pks: Optional[float]) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Determine frame activity threshold pks (if None) via shuffled similarity,
    then return data restricted to frames with sum>=pks, and their indices.
    """
    num_shuff = 100
    p = 0.98
    N, T = data.shape

    if pks is None:
        # Precompute shuffles
        data_shuff = np.zeros((N, T, num_shuff), dtype=np.int8)
        for ii in range(num_shuff):
            data_shuff[:, :, ii] = shuffle(data, method='time')

        max_sum = int(np.max(np.sum(data, axis=0)))
        # Start from 3 as in MATLAB
        for n in range(3, max_sum + 1):
            idx = np.where(np.sum(data, axis=0) >= n)[0]
            if idx.size < 2:
                continue
            data_high = data[:, idx]
            S_real = _cosine_similarity_matrix(data_high)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                S_out = np.zeros(num_shuff, dtype=float)
                for ii in range(num_shuff):
                    dh = data_shuff[:, :, ii]
                    idx_r = np.where(np.sum(dh, axis=0) >= n)[0]
                    if idx_r.size < 2:
                        S_out[ii] = 0.0
                        continue
                    dr = dh[:, idx_r]
                    S_rd = _cosine_similarity_matrix(dr)
                    S_out[ii] = np.nanmean(S_rd)

            # Determine scut as percentile of shuffled means
            try:
                bins = np.arange(0.0, np.nanmax(S_out) + 0.002, 0.002)
                if bins.size == 0:
                    bins = np.array([0.0])
                cd = _histc_like(S_out, bins)
                cd = np.cumsum(cd / np.sum(cd))
                scut = bins[np.argmax(cd > p)]
            except Exception:
                # Fallback with adaptive bin step (as in your MATLAB try/catch)
                start_step = 0.02
                scut = 0.0
                while True:
                    bins = np.arange(0.0, np.nanmax(S_out) + start_step, start_step)
                    if bins.size > 1:
                        cd = _histc_like(S_out, bins)
                        cd = np.cumsum(cd / np.sum(cd))
                        scut = bins[np.argmax(cd > p)]
                        break
                    start_step /= 10.0
                    if start_step < 1e-10:
                        break

            if float(np.nanmean(S_real)) > float(scut):
                pks = float(n)
                break

    if pks is None:
        # If auto never set (extremely sparse data), default to 3
        pks = 3.0

    pks = float(pks)
    pks_frame = np.where(np.sum(data, axis=0) >= pks)[0]
    data_high = data[:, pks_frame]
    return data_high, pks_frame, pks


def calc_scut(data_tfidf: np.ndarray) -> float:
    """Compute cosine similarity cutoff from time-shuffled null (p=0.98)."""
    num_shuff = 20
    p = 0.98
    N, T = data_tfidf.shape
    sims = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(num_shuff):
            shuff = shuffle(_ensure_binary(data_tfidf > 0), method='time')  # shuffle by time using spike pattern
            # Rebuild TF-IDF on shuffled spikes to mimic MATLAB behavior (they shuffle data before cosine)
            shuff_tfidf = calcTFIDF(shuff.astype(float))
            S = _cosine_similarity_matrix(shuff_tfidf)
            sims.append(S.ravel())

    sims = np.concatenate(sims) if sims else np.array([0.0])
    bins = np.arange(0.0, 1.0 + 0.01, 0.01)
    cd = _histc_like(sims, bins)
    cd = np.cumsum(cd / np.sum(cd))
    scut = bins[np.argmax(cd > p)]
    return float(scut)


def calc_jcut(S_tib: np.ndarray) -> float:
    """Compute Jaccard similarity cutoff from time-shuffled null (p=0.99)."""
    num_shuff = 20
    p = 0.99
    Tprime = S_tib.shape[0]
    sims = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(num_shuff):
            shuff = shuffle(S_tib, method='time')  # shuffles per row across columns
            d = cdist(shuff, shuff, metric='jaccard')
            S = 1.0 - d
            sims.append(S.ravel())

    sims = np.concatenate(sims) if sims else np.array([0.0])
    bins = np.arange(0.0, 1.0 + 0.01, 0.01)
    cd = _histc_like(sims, bins)
    cd = np.cumsum(cd / np.sum(cd))
    jcut = bins[np.argmax(cd > p)]
    return float(jcut)


# ---------------------------
# SVD state extraction
# ---------------------------

def SVDStateBinary(S_bin: np.ndarray, state_cut: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Mirror of MATLAB SVDStateBinary:
      - SVD on binary similarity matrix
      - threshold rank-1 reconstructions by fac_cut
      - keep factors covering >= p fraction (by sqrt(area) share)
      - raise fac_cut until no frame is assigned to more than one state
    Returns: (state_raster [T' x K], state_pks [1 x T'], fac_cut)
    """
    p = 0.025  # per MATLAB code (comment says 5%, but code uses 2.5%)

    sz = S_bin.shape[0]
    # SVD; for symmetric S, V is enough
    # numpy returns U, s, Vt such that S = U * diag(s) * Vt
    U, s, Vt = np.linalg.svd(S_bin.astype(float), full_matrices=True)
    V = Vt.T  # columns are right singular vectors
    S_svd = s  # singular values

    fac_cut = 0.4
    dlt = 1
    state_raster = np.zeros((sz, 0), dtype=int)
    state_pks = np.zeros((sz,), dtype=int)

    while dlt > 0:
        fac_count = np.zeros(state_cut, dtype=int)

        max_n = min(state_cut, V.shape[1], len(S_svd))
        for n in range(max_n):
            v = V[:, n]
            block = (np.outer(v, v) * S_svd[n]) > fac_cut
            fac_count[n] = int(np.sum(block))

        # Estimate cluster sizes by sqrt(area)
        sizes = np.floor(np.sqrt(fac_count)).astype(int)
        total = np.sum(sizes)
        if total == 0:
            # no states at this fac_cut; tighten threshold down would add overlap, but spec keeps upward only
            # fall back: no states
            state_pks_num = np.zeros((0, sz), dtype=int)
            num_state = 0
        else:
            share = sizes / total
            keep_idx = np.where(share >= p)[0]
            num_state = len(keep_idx)

            svd_sig = np.zeros((sz, sz, num_state), dtype=int)
            for i, k in enumerate(keep_idx):
                v = V[:, k]
                svd_sig[:, :, i] = (np.outer(v, v) * S_svd[k]) > fac_cut

            state_pks_num = np.zeros((num_state, sz), dtype=int)
            for i in range(num_state):
                # assign frame j if any link in column j is present
                col_has = np.sum(svd_sig[:, :, i], axis=0) > 0
                state_pks_num[i, col_has] = 1

        # Check overlap: sum across states per frame
        overlaps = np.sum(state_pks_num, axis=0) if state_pks_num.size else np.zeros((sz,), dtype=int)
        if np.max(overlaps, initial=0) > 1:
            fac_cut += 0.01
            dlt = 1
            continue
        else:
            dlt = 0
            # Build state_raster and state_pks as in MATLAB
            # state_raster = rot90(sortrows(state_pks_num)')'
            B = _sortrows_matlab(state_pks_num)  # sort rows
            C = B.T  # sz x num_state
            # rot90(C) ccw
            C_rot = np.rot90(C)
            state_raster = C_rot.T.astype(int)  # sz x num_state

            # state indices per frame (column index where '1' is)
            if state_raster.size:
                label_matrix = state_raster * np.arange(1, state_raster.shape[1] + 1)[np.newaxis, :]
                state_pks = np.sum(label_matrix, axis=1).astype(int)
            else:
                state_pks = np.zeros((sz,), dtype=int)

    return state_raster, state_pks, float(fac_cut)


# ---------------------------
# Main ensemble finder (Python mirror of MATLAB function)
# ---------------------------

def find_svd_ensembles(
    data: np.ndarray,
    coords: Optional[np.ndarray] = None,
    param: Optional[Dict] = None,
    save_dir: Optional[str] = None
) -> Tuple[List[np.ndarray], np.ndarray, Dict, List[str], List[np.ndarray]]:
    """
    Python mirror of findSVDensembleCustom.
    Returns:
      core_svd (list of 1D arrays),
      state_pks_full (1 x T array),
      param_out (dict),
      fig_paths (list of saved figure paths),
      pool_svd (list of 1D arrays)
    """
    # Defaults
    if param is None:
        param = {}
    pks = param.get('pks', 4)
    ticut = param.get('ticut', 0.22)
    jcut = param.get('jcut', 0.06)
    state_cut = param.get('state_cut', int(round(data.shape[0] / 4)))

    if state_cut is None:
        state_cut = int(round(data.shape[0] / 4))

    # Ensure binary matrix
    data = _ensure_binary(data)
    N, T = data.shape

    # Save directory
    if save_dir is None:
        save_dir = os.getcwd()

    # 1) Find high-activity frames
    data_active, pk_indx, pks_val = findActiveFramesCustom(data, None if param.get('pks', None) in (None, []) else pks)

    # 2) TF-IDF on active
    data_tfidf = calcTFIDF(data_active.astype(float))

    # 3) Cosine similarity among active frames
    S_ti = _cosine_similarity_matrix(data_tfidf)

    # 4) ticut from null if empty
    if param.get('ticut', None) in (None, []):
        ticut_val = calc_scut(data_tfidf)
    else:
        ticut_val = ticut

    S_tib = (S_ti > ticut_val).astype(int)

    # 5) jcut from null if empty
    if param.get('jcut', None) in (None, []):
        jcut_val = calc_jcut(S_tib.astype(int))
    else:
        jcut_val = jcut

    js = _jaccard_similarity_matrix(S_tib.astype(int))
    S_bin = (js > jcut_val).astype(int)

    # 6) SVD states
    state_raster, state_pks, fac_cut = SVDStateBinary(S_bin, state_cut)
    num_state = state_raster.shape[1] if state_raster.ndim == 2 else 0

    # 7) Map states back to full timeline
    state_pks_full = np.zeros((T,), dtype=int)
    if pk_indx.size > 0:
        state_pks_full[pk_indx] = state_pks

    # 8) Plot raster of states
    fig_paths = []
    plt.figure(figsize=(8, 4))
    plt.imshow((state_raster.T == 0), aspect='auto', cmap='gray', interpolation='nearest')
    plt.xlabel('Frame (active subset)')
    plt.ylabel('Ensemble index')
    plt.title('Ensemble activity (white = active)')
    path1 = os.path.join(save_dir, 'ensemble_rasters.png')
    plt.tight_layout()
    plt.savefig(path1, dpi=150)
    plt.close()
    fig_paths.append(path1)

    # 9) Find most significant cells (core) via AUC over ti thresholds
    ti_vec = np.arange(0.01, 0.101, 0.01)
    core_svd: List[np.ndarray] = []
    pool_svd: List[np.ndarray] = []
    state_member_raster = np.zeros((N, num_state), dtype=int)

    # Cross-validation figure (grid MxN)
    if num_state > 0:
        Ngrid = int(math.ceil(math.sqrt(num_state)))
        Mgrid = int(math.ceil(num_state / Ngrid))
    else:
        Ngrid = Mgrid = 1

    plt.figure(figsize=(3 * Ngrid, 3 * Mgrid))
    for ii in range(num_state):
        # pull out all activities in a state
        in_state = (state_pks == (ii + 1))
        # sum TF-IDF across those frames (N x T') -> 1 x N
        if np.any(in_state):
            state_ti_hist = np.sum(data_tfidf[:, in_state], axis=1).astype(float)
        else:
            state_ti_hist = np.zeros((N,), dtype=float)
        maxv = np.max(state_ti_hist) if state_ti_hist.size else 0.0
        if maxv > 0:
            state_ti_hist = (state_ti_hist / maxv)
        else:
            state_ti_hist = np.zeros_like(state_ti_hist)

        aucs = np.zeros((ti_vec.size,), dtype=float)

        ax = plt.subplot(Mgrid, Ngrid, ii + 1)
        for n, ti in enumerate(ti_vec):
            core_vec = np.zeros((N,), dtype=float)
            core_vec[state_ti_hist > ti] = 1.0

            # similarity of each active frame to the core
            # data_active: N x T' ; labels over T'
            sim_core = _cosine_similarity_to_vector(data_active.astype(float), core_vec)

            y_true = in_state.astype(int)
            # Guard: if only one class present, AUC is undefined -> set 0.5
            if np.unique(y_true).size < 2:
                auc = 0.5
                fpr = np.array([0, 1])
                tpr = np.array([0, 1])
            else:
                # Build ROC and AUC
                fpr, tpr, _ = roc_curve(y_true, sim_core)
                try:
                    auc = roc_auc_score(y_true, sim_core)
                except ValueError:
                    auc = 0.5
            aucs[n] = auc
            ax.plot(fpr, tpr, linewidth=1)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title(f'state {ii + 1}')

        best_indx = int(np.argmax(aucs))
        ti_cut = float(ti_vec[best_indx])
        state_member_raster[:, ii] = (state_ti_hist > ti_cut).astype(int)
        core_indices = np.where(state_member_raster[:, ii] > 0)[0]
        pool_indices = np.where(state_ti_hist > 0)[0]
        core_svd.append(core_indices)
        pool_svd.append(pool_indices)

    path2 = os.path.join(save_dir, 'cross_validation.png')
    plt.tight_layout()
    plt.savefig(path2, dpi=150)
    plt.close()
    fig_paths.append(path2)

    # 10) Plot core neurons (if coords provided)
    if coords is not None and coords.ndim == 2 and coords.shape[0] == N and coords.shape[1] == 2 and num_state > 0:
        mksz = 30
        plt.figure(figsize=(3 * Ngrid, 3 * Mgrid))
        for ii in range(num_state):
            ax = plt.subplot(Mgrid, Ngrid, ii + 1)
            ax.scatter(coords[:, 0], -coords[:, 1], s=mksz, c='k', edgecolors='none')
            if pool_svd[ii].size > 0:
                ax.scatter(coords[pool_svd[ii], 0], -coords[pool_svd[ii], 1], s=mksz, c=(1.0, 0.8, 0.8))
            if core_svd[ii].size > 0:
                ax.scatter(coords[core_svd[ii], 0], -coords[core_svd[ii], 1], s=mksz, c=(1.0, 0.2, 0.2))
            ax.set_title(f'ensemble #{ii + 1}')
            ax.axis('off')
            ax.set_aspect('equal', adjustable='box')
        path3 = os.path.join(save_dir, 'neurons.png')
        plt.tight_layout()
        plt.savefig(path3, dpi=150)
        plt.close()
        fig_paths.append(path3)

    # 11) Param out
    param_out = {
        'pks': pks_val,
        'ticut': ticut_val,
        'jcut': jcut_val,
        'state_cut': int(state_cut),
        'fac_cut': fac_cut
    }

    return core_svd, state_pks_full, param_out, fig_paths, pool_svd


# ---------------------------
# .mat I/O and entry point
# ---------------------------

def run_from_mat(mat_path: str):
    """
    Entry function: load raster .mat, run ensemble detection, save plots next to the .mat.
    Expects at least a variable named 'data' (N x T, binary). If missing, tries 'raster'.
    Optional: 'coords' (N x 2), 'param' (struct-like dict with pks/ticut/jcut/state_cut).
    """
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"File not found: {mat_path}")

    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    # Try to find data
    data = None
    for key in ('data', 'raster'):
        if key in mat and isinstance(mat[key], np.ndarray):
            data = mat[key]
            break
    if data is None:
        raise KeyError("Could not find 'data' or 'raster' in the .mat file.")

    # Ensure 2D
    if data.ndim != 2:
        raise ValueError(f"'data' must be 2D (N x T). Got shape {data.shape}")

    # Optional coords
    coords = mat.get('coords', None)
    if isinstance(coords, np.ndarray) and coords.ndim == 2 and coords.shape[0] == data.shape[0] and coords.shape[1] == 2:
        pass
    else:
        coords = None

    # Optional param
    param = {}
    if 'param' in mat:
        p = mat['param']
        # If it's a MATLAB struct, it may come as a simple object or dict-like
        # Try to read fields if present
        def _getf(obj, name):
            try:
                return getattr(obj, name)
            except Exception:
                try:
                    return obj[name]
                except Exception:
                    return None
        for key in ('pks', 'ticut', 'jcut', 'state_cut'):
            val = _getf(p, key)
            # normalize empty to None
            if val is None:
                continue
            try:
                if np.size(val) == 0:
                    continue
                param[key] = float(val) if key != 'state_cut' else int(val)
            except Exception:
                # ignore non-numeric
                pass

    save_dir = os.path.dirname(os.path.abspath(mat_path))
    core_svd, state_pks_full, param_out, fig_paths, pool_svd = find_svd_ensembles(
        data=data, coords=coords, param=param, save_dir=save_dir
    )

    # Save a small .npz with outputs next to figures (optional, helps comparison)
    out_npz = os.path.join(save_dir, 'svd_ensembles_output.npz')
    # Convert lists of arrays to object dtype for saving
    core_obj = np.array(core_svd, dtype=object)
    pool_obj = np.array(pool_svd, dtype=object)
    np.savez(
        out_npz,
        core_svd=core_obj,
        pool_svd=pool_obj,
        state_pks_full=state_pks_full,
        pks=param_out['pks'],
        ticut=param_out['ticut'],
        jcut=param_out['jcut'],
        state_cut=param_out['state_cut'],
        fac_cut=param_out['fac_cut'],
        fig_paths=np.array(fig_paths, dtype=object)
    )

    print("Finished SVD ensemble detection.")
    print("Saved figures:")
    for p in fig_paths:
        print("  -", p)
    print("Saved outputs:", out_npz)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python find_SVD_ensembles.py /path/to/raster.mat")
        sys.exit(1)
    run_from_mat(sys.argv[1])
