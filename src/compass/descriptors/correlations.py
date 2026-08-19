# Created by gonzalezroy at 6/24/24
import numpy as np


def compute_gc_matrix(corr_coords):
    """
    Compute Mutual Information (MI) and Generalized Correlation (GC) matrices
    from per-residue backbone (CA) coordinates.

    Args:
        corr_coords: (n_frames, n_residues, 3) array of CA xyz per frame

    Returns:
        MI_scores: (n_residues, n_residues) mutual information matrix
        GC_matrix: (n_residues, n_residues) generalized correlation matrix
    """
    n_frames, n_res, _ = corr_coords.shape

    # Covariance over the 3*n_res flattened coordinate vector
    flat = corr_coords.reshape(n_frames, -1)  # (F, 3R)
    cov = np.cov(flat, rowvar=False)           # (3R, 3R)

    # Reshape into residue-pair 3x3 blocks: cov_blocks[i,j] is the 3x3
    # cross-covariance between residue i and residue j
    cov_blocks = cov.reshape(n_res, 3, n_res, 3)

    # var_i = mean of the 3 diagonal xyz variances for residue i
    var = np.einsum('ikik->i', cov_blocks) / 3.0   # (n_res,)

    # cov_ij = mean of the 3x3 cross-covariance block
    cov_ij = np.einsum('ikjl->ij', cov_blocks) / 9.0  # (n_res, n_res)

    # MI = 0.5 * ln(1 + cov_ij^2 / (var_i * var_j))
    var_product = np.outer(var, var)
    div_term = (cov_ij ** 2) / var_product
    MI_scores = (0.5 * np.log(1.0 + div_term)).astype(np.float32)

    # GC = sqrt(1 - exp(-2 * MI))
    GC_matrix = np.sqrt(1.0 - np.exp(-2.0 * MI_scores)).astype(np.float32)

    return MI_scores, GC_matrix
