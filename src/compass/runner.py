import os
import sys
import time
from os.path import join

import compass.descriptors.config as cfg
import compass.descriptors.geometry as geom
import compass.descriptors.main as mm
import compass.descriptors.pca as pca
import compass.descriptors.topo_traj as tt
import compass.network.generals as gn

def runner():
    first_timer = time.time()
    config_path = sys.argv[1]
    arg = cfg.parse_params(config_path)

    (
        resids_to_atoms,
        resids_to_noh,
        calphas,
        oxy,
        nitro,
        donors,
        hydros,
        acceptors,
    ) = tt.prepare_datastructures(
        arg.topo,
        arg.out_dir,
        arg.heavies,
        first_timer,
    )

    (
        ave_min_dist,
        occ_nb,
        cp,
        occ_sb,
        occ_hb,
        occ_int,
        mi,
        gc,
    ) = mm.compute_descriptors(
        arg,
        resids_to_atoms,
        resids_to_noh,
        calphas,
        oxy,
        nitro,
        donors,
        hydros,
        acceptors,
        first_timer,
    )

    n = len(resids_to_atoms)
    matrices, matrix_files = geom.process_matrices(
        arg,
        n,
        ave_min_dist,
        occ_nb,
        cp,
        occ_sb,
        occ_hb,
        occ_int,
        mi,
        gc,
        first_timer,
    )

    min_dist_matrix_file = matrix_files["MINDIST"]
    adjacency_file = pca.run_pca(arg, matrices, n, first_timer)

    arg.network_dir = join(arg.out_dir, "network")
    os.makedirs(arg.network_dir, exist_ok=True)
    dist_cutoffs = [arg.dist_graph, arg.dist_clique]

    gn.process_graphs(arg, min_dist_matrix_file, adjacency_file, dist_cutoffs)

    gn.process_graph_files(arg.network_dir, dist_cutoffs[0])
    gn.process_graph_files_for_communities_and_cliques(arg.network_dir, dist_cutoffs[0], dist_cutoffs[1])

    gn.generate_pymol_scripts(arg.network_dir, arg.topo, dist_cutoffs[0], dist_cutoffs[1])

    print(f" ⏳  Wall Clock Time: {time.time() - first_timer:.2f} seconds")
    print(f"**** -------Normal Termination -------****")
