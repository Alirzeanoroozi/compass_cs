# Created by gonzalezroy at 6/17/24
import os
import time
from os.path import join
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from numba import njit

@njit(cache=True, fastmath=True)
def calc_dist(atom1_coords, atom2_coords):
    """
    Computes the Euclidean distance between two atoms in a molecule

    Args:
        atom1_coords: 3D coordinate array of the first atom
        atom2_coords: 3D coordinate array of the second atom

    Returns:
        float: the Euclidean distance between the two atoms
    """
    dx = atom1_coords[0] - atom2_coords[0]
    dy = atom1_coords[1] - atom2_coords[1]
    dz = atom1_coords[2] - atom2_coords[2]
    return np.sqrt(dx * dx + dy * dy + dz * dz)

@njit(cache=True, fastmath=True)
def calc_min_dist(coords1, coords2):
    """
    Get the minimumm distance between two sets of coordinates

    Args:
        coords1: coordinates of the first residue
        coords2: coordinates of the second residue

    Returns:
        The minimum distance between two sets of coordinates
    """
    n1 = coords1.shape[0]
    n2 = coords2.shape[0]
    min_dist_squared = 1.0e300
    for i in range(n1):
        x1 = coords1[i, 0]
        y1 = coords1[i, 1]
        z1 = coords1[i, 2]
        for j in range(n2):
            dx = x1 - coords2[j, 0]
            dy = y1 - coords2[j, 1]
            dz = z1 - coords2[j, 2]
            dist_squared = dx * dx + dy * dy + dz * dz
            if dist_squared < min_dist_squared:
                min_dist_squared = dist_squared
    return np.sqrt(min_dist_squared)

@njit(cache=True, fastmath=True)
def calc_single_angle(d, h, a):
    """
    Computes the angle between three atoms

    Args:
        d (donor): Coordinates of the first atom (x, y, z)
        h (hydrogen): Coordinates of the second atom (x, y, z).
        a (acceptor): Coordinates of the third atom (x, y, z).

    Returns:
        angle_deg: the angle in degrees
    """
    dhx = d[0] - h[0]
    dhy = d[1] - h[1]
    dhz = d[2] - h[2]
    ahx = a[0] - h[0]
    ahy = a[1] - h[1]
    ahz = a[2] - h[2]
    dot_product = dhx * ahx + dhy * ahy + dhz * ahz
    dh_norm = np.sqrt(dhx * dhx + dhy * dhy + dhz * dhz)
    ah_norm = np.sqrt(ahx * ahx + ahy * ahy + ahz * ahz)
    angle_rad = np.arccos(dot_product / (dh_norm * ah_norm))
    return angle_rad * 180.0 / np.pi

@njit(cache=True, fastmath=True)
def find_sb(frame_coords, oxy_i, nitro_j, k):
    """
    Find a single salt bridge between two residues

    Args:
        frame_coords: 3D coordinates of the frame
        oxy_i: oxygen atom index of the first residue
        nitro_j: nitrogen atom index of the second residue
        k: cutoff distance

    Returns:
        int: 1 if a salt bridge is found, 0 otherwise
    """
    n1 = oxy_i.shape[0]
    n2 = nitro_j.shape[0]
    min_dist_squared = k * k
    for i in range(n1):
        oi = oxy_i[i]
        x1 = frame_coords[oi, 0]
        y1 = frame_coords[oi, 1]
        z1 = frame_coords[oi, 2]
        for j in range(n2):
            nj = nitro_j[j]
            dx = x1 - frame_coords[nj, 0]
            dy = y1 - frame_coords[nj, 1]
            dz = z1 - frame_coords[nj, 2]
            if dx * dx + dy * dy + dz * dz < min_dist_squared:
                return 1
    return 0

@njit(cache=True, fastmath=True)
def find_hb(frame_coords, donors_i, hydros_i, acceptors_j, da_cut, ha_cut, dha_cut):
    """
    Find a single hydrogen bond between two residues
    Args:
        frame_coords: 3D coordinates of the frame
        donors_i: donor atom indices of the first residue
        hydros_i: hydrogen atom indices of the first residue
        acceptors_j: acceptor atom indices of the second residue
        da_cut: distance cutoff for the donor-acceptor distance
        ha_cut: distance cutoff for the hydrogen-acceptor distance
        dha_cut: angle cutoff for the donor-hydrogen-acceptor angle

    Returns:
        int: 1 if a hydrogen bond is found, 0 otherwise
    """
    n1 = donors_i.shape[0]
    n2 = acceptors_j.shape[0]
    for i in range(n1):
        di = donors_i[i]
        hi = hydros_i[i]
        coords_d = frame_coords[di]
        coords_h = frame_coords[hi]
        for j in range(n2):
            coords_a = frame_coords[acceptors_j[j]]
            da_dist = calc_dist(coords_d, coords_a)
            if da_dist < da_cut:
                ha_dist = calc_dist(coords_h, coords_a)
                if ha_dist < ha_cut:
                    angle = calc_single_angle(coords_d, coords_h, coords_a)
                    if angle > dha_cut:
                        return 1
    return 0

def to_matrix(one_dim_array, n):
    matrix = np.zeros((n, n))
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = one_dim_array[k]
            k += 1

    matrix += matrix.T
    return matrix

def save_matrix(arr, n, out_name, norm=False, prec=2):
    # Convert to matrix if needed
    matrix = to_matrix(arr, n) if len(arr.shape) == 1 else arr

    # Normalize if requested
    if norm:
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        matrix = (matrix - min_val) / (max_val - min_val)

    # Save matrix
    np.savetxt(out_name, matrix, fmt=f"%.{prec}f")
    return matrix

def get_matrix_name(out_dir, title, suffix):
    matrix_dir = join(out_dir, "matrices")
    os.makedirs(matrix_dir, exist_ok=True)
    return join(matrix_dir, f"{title}_{suffix}.mat")

def plot_matrix(matrix, matrix_title, output_name):
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(matrix, cmap="jet")
    plt.title(matrix_title)
    plt.xlabel("Residue Index")
    plt.ylabel("Residue Index")
    plt.savefig(output_name)
    plt.close()

def process_matrices(arg, n, ave_min_dist, occ_nb, cp, occ_sb, occ_hb, occ_int, mi, gc, first_timer):
    # Declare matrices to process
    matrices = {
        "MINDIST": {"data": ave_min_dist, "norm": False, "prec": 4, "title": "Pairwise Minimum Distances"},
        "NONBOND": {"data": occ_nb, "norm": False, "prec": 4, "title": "Non-Bonded Interactions"},
        "SALTBRIDGES": {"data": occ_sb, "norm": False, "prec": 4, "title": "Salt Bridges"},
        "HBONDS": {"data": occ_hb, "norm": False, "prec": 4, "title": "Hydrogen Bonds"},
        "INTERACTIONS": {"data": occ_int, "norm": False, "prec": 2, "title": "Interactions"},
        "COMMPROP": {"data": cp, "norm": True, "prec": 4, "title": "Communication Properties"},
        "MI": {"data": mi, "norm": True, "prec": 4, "title": "Mutual Information"},
        "GC": {"data": gc, "norm": True, "prec": 4, "title": "Generalized Correlation"},
    }

    # Process matrices
    matrices_name = {}
    for matrix in matrices:
        data = matrices[matrix]["data"]
        normalize = matrices[matrix]["norm"]
        precision = matrices[matrix]["prec"]

        # Save matrices
        matrix_name = get_matrix_name(arg.out_dir, arg.title, matrix)
        matrices_name.update({matrix: matrix_name})
        matrix_data = save_matrix(data, n, matrix_name, norm=normalize, prec=precision)
        matrices[matrix].update({"data": matrix_data})

        # Plot matrices
        plot_name = matrix_name.replace(".mat", ".png")
        matrix_title = matrices[matrix]["title"]
        plot_matrix(matrix_data, matrix_title, plot_name)

    saving_time = round(time.time() - first_timer, 2)
    print(f" ⏱️  Until saving & plotting matrices: {saving_time} s")
    return matrices, matrices_name
