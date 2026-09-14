# Created by gonzalezroy at 6/26/24
import time
import numpy as np
from numpy import concatenate as concat
from scipy.sparse import lil_matrix
from sklearn.decomposition import PCA

from compass.descriptors import geometry as geom

def reshape_matrices(matrices):
    """
    Reshape the matrices to be concatenated for PCA

    Args:
        matrices: List of matrices to be reshaped

    Returns:
        data: Concatenated and reshaped matrices
    """
    # Ensure all matrices have the same shape
    shapes = [matrix.shape for matrix in matrices]
    if len(set(shapes)) > 1:
        raise ValueError("All matrices must have the same shape")

    # Concatenate the flattened matrices for PCA
    num_rows = shapes[0][0]
    flatened = [matrix.reshape(num_rows, -1) for matrix in matrices]
    data = concat(flatened, axis=1)
    return data

def calc_adjacency_matrix(data, threshold=0.3):
    pca = PCA(n_components=2, svd_solver="full")
    pca_result = pca.fit_transform(data)
    n_samples = pca_result.shape[0]
    adjacency_matrix = lil_matrix((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i, n_samples):
            dist = np.linalg.norm(pca_result[i] - pca_result[j])
            # Only store distances above threshold
            # This ensures distant points get larger values
            if dist > threshold:
                adjacency_matrix[i, j] = dist
                adjacency_matrix[j, i] = dist

    return adjacency_matrix

def run_pca(arg, matrices, n, first_timer):
    gc_mat = matrices["GC"]["data"]
    int_mat = matrices["INTERACTIONS"]["data"]
    cp_mat = matrices["COMMPROP"]["data"]
    matrices = [gc_mat, int_mat, cp_mat]

    data = reshape_matrices(matrices)

    # Perform PCA & generate adjacency matrix
    adj_mat_raw = calc_adjacency_matrix(data)
    adj_mat = adj_mat_raw.toarray()
    adj_mat = 1 - adj_mat # from distance to similarity
    adj_name = geom.get_matrix_name(arg.out_dir, arg.title, "ADJACENCY")
    adj_mat = geom.save_matrix(adj_mat, n, adj_name, norm=True, prec=4)
    matrix_name = "Adjacency matrix"
    geom.plot_matrix(adj_mat, matrix_name, adj_name.replace(".mat", ".png"))

    pca_time = round(time.time() - first_timer, 2)
    print(f" ⏱️  Until PCA & Adjacency matrix computing: {pca_time} s")
    return adj_name
