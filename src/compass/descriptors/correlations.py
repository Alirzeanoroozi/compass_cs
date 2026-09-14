# Created by gonzalezroy at 6/24/24
import numpy as np

def compute_gc_matrix(corr_coords, block_size=128):
    """
    Compute Mutual Information (MI) and Generalized Correlation (GC) matrices
    from per-residue backbone (CA) coordinates.

    MI uses the Gaussian estimator

        I_ij = 1/2 [ln det(cov_i) + ln det(cov_j) - ln det(cov_ij)]

    where cov_i and cov_j are the 3x3 covariances of residues i and j and
    cov_ij is their 6x6 joint covariance. The generalized correlation is

        GC_ij = sqrt(1 - exp(-2 I_ij / 3))

    with 3 the dimensionality of the coordinate vectors.

    Args:
        corr_coords: (n_frames, n_residues, 3) array of CA xyz per frame
        block_size: number of residue rows whose joint covariances are
                    assembled at once, bounding peak memory

    Returns:
        MI_scores: (n_residues, n_residues) mutual information matrix
        GC_matrix: (n_residues, n_residues) generalized correlation matrix
    """
    n_frames, n_res, _ = corr_coords.shape
    if n_frames < 2:
        raise ValueError(
            "Mutual information and generalized correlation need at least 2 "
            f"frames to estimate covariance; got {n_frames}. Increase n_frames "
            "in the [generals] section of the config file."
        )

    # (3R, 3R) covariance viewed as (R, R, 3, 3) residue-pair blocks, so that
    # cov_blocks[i, j] is the 3x3 cross-covariance between residues i and j
    cov = np.cov(corr_coords.reshape(n_frames, -1), rowvar=False)
    cov_blocks = cov.reshape(n_res, 3, n_res, 3).transpose(0, 2, 1, 3)

    # Ridge keeps the determinants finite for near-degenerate residue motion
    ridge = 1e-10 * np.trace(cov) / (3 * n_res)
    diag = np.arange(n_res)
    self_blocks = cov_blocks[diag, diag] + ridge * np.eye(3)
    logdet_self = np.linalg.slogdet(self_blocks)[1]

    MI_scores = np.empty((n_res, n_res))
    joint = np.empty((block_size, n_res, 6, 6))
    for start in range(0, n_res, block_size):
        stop = min(start + block_size, n_res)
        rows = joint[:stop - start]
        rows[:, :, :3, :3] = self_blocks[start:stop, None]
        rows[:, :, :3, 3:] = cov_blocks[start:stop]
        rows[:, :, 3:, :3] = cov_blocks[start:stop].transpose(0, 1, 3, 2)
        rows[:, :, 3:, 3:] = self_blocks[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            logdet_joint = np.linalg.slogdet(rows)[1]
        MI_scores[start:stop] = 0.5 * (logdet_self[start:stop, None]
                                       + logdet_self[None, :] - logdet_joint)

    np.clip(MI_scores, 0.0, None, out=MI_scores)

    # I_ii diverges by construction (the joint covariance of a residue with
    # itself is singular). Cap those entries so the matrix stays usable once
    # it is normalised and saved downstream.
    diverged = ~np.isfinite(MI_scores)
    MI_scores[diverged] = 0.0
    MI_scores[diverged] = MI_scores.max()

    GC_matrix = np.sqrt(1.0 - np.exp(-2.0 * MI_scores / 3.0))
    np.fill_diagonal(GC_matrix, 1.0)

    return MI_scores.astype(np.float32), GC_matrix.astype(np.float32)
