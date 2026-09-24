# Created by gonzalezroy at 6/17/24
import time
import numpy as np
import mdtraj as md
from numba import njit, prange
from tqdm import tqdm

import compass.descriptors.correlations as corr
from compass.descriptors.geometry import calc_dist, calc_min_dist, find_hb, find_sb

def get_xyz_chunks(trajs, topo, chunk_size=500, max_frames=None):
    n_loaded = 0
    for traj in trajs:
        for chunk in md.iterload(traj, top=topo, chunk=chunk_size):
            xyz = chunk.xyz
            if max_frames is not None:
                remaining = max_frames - n_loaded
                if remaining <= 0:
                    return
                if xyz.shape[0] > remaining:
                    xyz = xyz[:remaining]
            yield xyz
            n_loaded += xyz.shape[0]
            if max_frames is not None and n_loaded >= max_frames:
                return

def _pad_index_map(index_map, n_resids):
    """Pad a residue -> atom-index map into (n_resids, max_len) arrays for Numba."""
    rows = []
    max_len = 1
    for i in range(n_resids):
        vals = index_map[i] if i in index_map else ()
        arr = np.asarray(vals, dtype=np.int32).reshape(-1)
        if arr.size > max_len:
            max_len = arr.size
        rows.append(arr)
    padded = np.full((n_resids, max_len), -1, dtype=np.int32)
    counts = np.zeros(n_resids, dtype=np.int32)
    for i, arr in enumerate(rows):
        counts[i] = arr.size
        if arr.size:
            padded[i, :arr.size] = arr
    return padded, counts

def _calpha_index_array(calphas, n_resids):
    ca_idx = np.empty(n_resids, dtype=np.int32)
    for i in range(n_resids):
        ca_idx[i] = calphas[i]
    return ca_idx

def compute_descriptors(arg, resids_to_atoms, resids_to_noh, calphas, oxy, nitro, donors, hydros, acceptors, first_timer):
    n_resids = len(resids_to_atoms)

    min_dist_sum = np.zeros((n_resids, n_resids))
    nb_sum = np.zeros((n_resids, n_resids)) # non-bonded interactions
    sb_sum = np.zeros((n_resids, n_resids)) # salt-bridge interactions
    hb_sum = np.zeros((n_resids, n_resids)) # hydrogen bonds
    int_sum = np.zeros((n_resids, n_resids)) # interactions (salt-bridge + hydrogen bonds)

    # Welford online variance accumulators for CP
    cp_mean = np.zeros((n_resids, n_resids)) # communication propensity
    cp_m2 = np.zeros((n_resids, n_resids)) # communication propensity variance

    # Backbone coords collected per frame for MI/GC
    corr_list = []

    ca_idx = _calpha_index_array(calphas, n_resids)
    noh_idx, noh_n = _pad_index_map(resids_to_noh, n_resids)
    oxy_idx, oxy_n = _pad_index_map(oxy, n_resids)
    nitro_idx, nitro_n = _pad_index_map(nitro, n_resids)
    donors_idx, donors_n = _pad_index_map(donors, n_resids)
    hydros_idx, hydros_n = _pad_index_map(hydros, n_resids)
    acceptors_idx, acceptors_n = _pad_index_map(acceptors, n_resids)

    chunk_size = 50
    n_frames = 0
    max_frames = arg.n_frames
    if max_frames is not None:
        print(f" 📋 System details: using first {max_frames} frames across all trajectories")
    for chunk_xyz in tqdm(get_xyz_chunks(arg.traj.split(), arg.topo, chunk_size=chunk_size, max_frames=max_frames)):
        n_chunk_frames = chunk_xyz.shape[0]
        for f in range(n_chunk_frames):
            frame = np.ascontiguousarray(chunk_xyz[f])
            f_mindist, f_cp, f_nb, f_sb, f_hb, f_int = get_frame_info(
                frame, noh_idx, noh_n, ca_idx, arg.nb_cut, arg.sb_cut, arg.da_cut,
                arg.ha_cut, arg.dha_cut, oxy_idx, oxy_n, nitro_idx, nitro_n,
                donors_idx, donors_n, hydros_idx, hydros_n, acceptors_idx, acceptors_n)

            min_dist_sum += f_mindist
            nb_sum += f_nb
            sb_sum += f_sb
            hb_sum += f_hb
            int_sum += f_int

            # Welford online update for CP variance
            delta = f_cp - cp_mean
            cp_mean += delta / (n_frames + 1)
            delta2 = f_cp - cp_mean
            cp_m2 += delta * delta2

            # Collect backbone coordinates for correlation
            corr_list.append(frame[ca_idx].copy())

            n_frames += 1

    if n_frames == 0:
        raise ValueError("No frames were loaded from the trajectory files")
    if max_frames is not None and n_frames < max_frames:
        print(f" ⚠️  Only {n_frames} frames were available (requested {max_frames})")

    # Compute average values (mdtraj coordinates are in nm; convert to angstrom)
    ave_min_dist = (min_dist_sum / n_frames) * 10
    occ_nb = nb_sum / n_frames
    occ_sb = sb_sum / n_frames
    occ_hb = hb_sum / n_frames
    occ_int = int_sum / n_frames

    # Communication Propensity = variance (nm^2 -> angstrom^2), then invert
    cp = (cp_m2 / n_frames) * 100
    cp = abs(cp - np.max(cp))

    # MI & GC from backbone coordinates
    corr_coords = np.stack(corr_list)
    mi, gc = corr.compute_gc_matrix(corr_coords)

    running_time = round(time.time() - first_timer, 2)
    print(f" 📋 System details: number of frames are {n_frames}")
    print(f" ⏱️  Until descriptors computed: {running_time} s")

    return ave_min_dist, occ_nb, cp, occ_sb, occ_hb, occ_int, mi, gc

@njit(parallel=True, fastmath=True, cache=True)
def get_frame_info(frame_coords, noh_idx, noh_n, ca_idx, nb_cut, sb_cut, da_cut,
                   ha_cut, dha_cut, oxy_idx, oxy_n, nitro_idx, nitro_n,
                   donors_idx, donors_n, hydros_idx, hydros_n,
                   acceptors_idx, acceptors_n):
    n_resids = noh_n.shape[0]

    mat_min_dist = np.zeros((n_resids, n_resids))
    mat_nb = np.zeros((n_resids, n_resids))
    mat_cp = np.zeros((n_resids, n_resids))
    mat_sb = np.zeros((n_resids, n_resids))
    mat_hb = np.zeros((n_resids, n_resids))
    mat_int = np.zeros((n_resids, n_resids))

    for i in prange(n_resids):
        coords_i = frame_coords[noh_idx[i, :noh_n[i]]]
        calpha_i = frame_coords[ca_idx[i]]
        for j in range(i + 1, n_resids):
            coords_j = frame_coords[noh_idx[j, :noh_n[j]]]
            calpha_j = frame_coords[ca_idx[j]]

            min_dist = calc_min_dist(coords_i, coords_j)
            mat_min_dist[i, j] = mat_min_dist[j, i] = min_dist
            nb_val = 1.0 if min_dist < nb_cut else 0.0
            mat_nb[i, j] = mat_nb[j, i] = nb_val
            cp_val = calc_dist(calpha_i, calpha_j)
            mat_cp[i, j] = mat_cp[j, i] = cp_val

            sb = 0
            if (min_dist < sb_cut) and (i + 1 != j):
                if (oxy_n[i] > 0) and (nitro_n[j] > 0):
                    sb += find_sb(frame_coords, oxy_idx[i, :oxy_n[i]],
                                  nitro_idx[j, :nitro_n[j]], sb_cut)
                if (oxy_n[j] > 0) and (nitro_n[i] > 0):
                    sb += find_sb(frame_coords, oxy_idx[j, :oxy_n[j]],
                                  nitro_idx[i, :nitro_n[i]], sb_cut)
            if sb:
                mat_sb[i, j] = mat_sb[j, i] = 1.0

            hb = 0
            if min_dist <= da_cut:
                if (donors_n[i] > 0) and (acceptors_n[j] > 0):
                    hb += find_hb(frame_coords, donors_idx[i, :donors_n[i]],
                                  hydros_idx[i, :hydros_n[i]],
                                  acceptors_idx[j, :acceptors_n[j]],
                                  da_cut, ha_cut, dha_cut)
                if (donors_n[j] > 0) and (acceptors_n[i] > 0):
                    hb += find_hb(frame_coords, donors_idx[j, :donors_n[j]],
                                  hydros_idx[j, :hydros_n[j]],
                                  acceptors_idx[i, :acceptors_n[i]],
                                  da_cut, ha_cut, dha_cut)
            if hb:
                mat_hb[i, j] = mat_hb[j, i] = 1.0

            if sb or hb:
                mat_int[i, j] = mat_int[j, i] = 1.0

    return mat_min_dist, mat_cp, mat_nb, mat_sb, mat_hb, mat_int
