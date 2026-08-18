# Created by gonzalezroy at 6/17/24
import time
import numpy as np
import mdtraj as md
from numba import njit, prange

import compass.descriptors.correlations as corr
from compass.descriptors.geometry import calc_dist, calc_min_dist, find_hb, find_sb


def get_xyz_chunks(trajs, topo, chunk_size=500):
    for traj in trajs:
        for chunk in md.iterload(traj, top=topo, chunk=chunk_size):
            yield chunk.xyz

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

def compute_descriptors(arg, resids_to_atoms, resids_to_noh, atoms_to_resids, calphas, oxy, nitro, donors, hydros, acceptors, first_timer):
    # Initialize containers
    n_resids = len(resids_to_atoms)
    n_pairs = int(n_resids * (n_resids - 1) / 2)
    pair_min_dist_sum = np.zeros(n_pairs) # minimum distance between every pair of residues
    pair_cp_sum = np.zeros(n_pairs) # distance between calpha atoms of every pair of residues
    pair_nb_sum = np.zeros(n_pairs) # non-bonded contacts between every pair of residues
    pair_sb_sum = np.zeros(n_pairs) # salt bridges between every pair of residues
    pair_hb_sum = np.zeros(n_pairs) # hydrogen bonds between every pair of residues
    pair_int_sum = np.zeros(n_pairs) # interactions between every pair of residues

    ca_idx = _calpha_index_array(calphas, n_resids)
    noh_idx, noh_n = _pad_index_map(resids_to_noh, n_resids)
    oxy_idx, oxy_n = _pad_index_map(oxy, n_resids)
    nitro_idx, nitro_n = _pad_index_map(nitro, n_resids)
    donors_idx, donors_n = _pad_index_map(donors, n_resids)
    hydros_idx, hydros_n = _pad_index_map(hydros, n_resids)
    acceptors_idx, acceptors_n = _pad_index_map(acceptors, n_resids)

    n_frames = 0
    for chunk in get_xyz_chunks(arg.traj.split(), arg.topo, chunk_size=1):
        n_frames += 1
        frame = np.ascontiguousarray(chunk[0])
        pair_min_dist, pair_cp, pair_nb, pair_sb, pair_hb, pair_int = get_frame_info(
            frame, noh_idx, noh_n, ca_idx, arg.nb_cut, arg.sb_cut, arg.da_cut,
            arg.ha_cut, arg.dha_cut, oxy_idx, oxy_n, nitro_idx, nitro_n,
            donors_idx, donors_n, hydros_idx, hydros_n, acceptors_idx, acceptors_n)
        pair_min_dist_sum += pair_min_dist
        pair_cp_sum += pair_cp
        pair_nb_sum += pair_nb
        pair_sb_sum += pair_sb
        pair_hb_sum += pair_hb
        pair_int_sum += pair_int

    # Compute average values
    ave_min_dist = (pair_min_dist_sum / n_frames) * 10 # convert to nanometers
    ave_pair_cp = pair_cp_sum / n_frames
    occ_nb = pair_nb_sum / n_frames
    occ_sb = pair_sb_sum / n_frames
    occ_hb = pair_hb_sum / n_frames
    occ_int = pair_int_sum / n_frames

    # Do a 2nd pass to compute cp & extract coords for correlation matrices
    pair_cp_sum2 = np.zeros(n_pairs)
    corr_coords = np.zeros((n_frames, n_resids, 3))

    n_frames = 0
    for chunk in get_xyz_chunks(arg.traj.split(), arg.topo, chunk_size=1):
        n_frames += 1
        frame = np.ascontiguousarray(chunk[0])
        pair_cp2 = get_chunk_cp(frame, ca_idx, ave_pair_cp)
        pair_cp_sum2 += pair_cp2

        # Get correlation coordinates
        corr_coords[n_frames - 1, :] = frame[ca_idx]
    
    # Compute Communication Propensity
    cp = (pair_cp_sum2 / n_frames) * 100
    cp = abs(cp - max(cp))  # Invert CP matrix
    
    # Compute MI & GC
    mi, gc = corr.compute_gc_matrix(corr_coords, num_atoms_per_residue=3)

    running_time = round(time.time() - first_timer, 2)
    print(f" 📋 System details: number of frames are {n_frames}")
    print(f" ⏱️  Until descriptors computed: {running_time} s")

    return ave_min_dist, occ_nb, cp, occ_sb, occ_hb, occ_int, mi, gc

@njit(parallel=True, fastmath=True, cache=True)
def get_chunk_cp(traj_coords, ca_idx, ave_pair_cp):
    n_resids = ca_idx.shape[0]
    n_pairs = n_resids * (n_resids - 1) // 2
    triangle = np.zeros(n_pairs)

    for i in prange(n_resids):
        calpha_i = traj_coords[ca_idx[i]]
        base = i * n_resids - i * (i + 1) // 2
        for j in range(i + 1, n_resids):
            index = base + (j - i - 1)
            triangle[index] = calc_dist(calpha_i, traj_coords[ca_idx[j]])
    return (triangle - ave_pair_cp) ** 2

@njit(parallel=True, fastmath=True, cache=True)
def get_frame_info(frame_coords, noh_idx, noh_n, ca_idx, nb_cut, sb_cut, da_cut,
                   ha_cut, dha_cut, oxy_idx, oxy_n, nitro_idx, nitro_n,
                   donors_idx, donors_n, hydros_idx, hydros_n,
                   acceptors_idx, acceptors_n):
    n_resids = noh_n.shape[0]
    n_pairs = n_resids * (n_resids - 1) // 2

    pair_min_dists = np.zeros(n_pairs)
    pair_nb = np.zeros(n_pairs)
    pair_cp = np.zeros(n_pairs)
    pair_sb = np.zeros(n_pairs)
    pair_hb = np.zeros(n_pairs)
    pair_int = np.zeros(n_pairs)

    for i in prange(n_resids):
        coords_i = frame_coords[noh_idx[i, :noh_n[i]]]
        calpha_i = frame_coords[ca_idx[i]]
        base = i * n_resids - i * (i + 1) // 2
        for j in range(i + 1, n_resids):
            index = base + (j - i - 1)
            coords_j = frame_coords[noh_idx[j, :noh_n[j]]]
            calpha_j = frame_coords[ca_idx[j]]

            min_dist = calc_min_dist(coords_i, coords_j)
            pair_min_dists[index] = min_dist
            pair_nb[index] = 1.0 if min_dist < nb_cut else 0.0
            pair_cp[index] = calc_dist(calpha_i, calpha_j)

            sb = 0
            if (min_dist < sb_cut) and (i + 1 != j):
                if (oxy_n[i] > 0) and (nitro_n[j] > 0):
                    sb += find_sb(frame_coords, oxy_idx[i, :oxy_n[i]],
                                  nitro_idx[j, :nitro_n[j]], sb_cut)
                if (oxy_n[j] > 0) and (nitro_n[i] > 0):
                    sb += find_sb(frame_coords, oxy_idx[j, :oxy_n[j]],
                                  nitro_idx[i, :nitro_n[i]], sb_cut)
            if sb:
                pair_sb[index] = 1.0

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
                pair_hb[index] = 1.0

            if sb or hb:
                pair_int[index] = 1.0

    return pair_min_dists, pair_nb, pair_cp, pair_sb, pair_hb, pair_int
