from argparse import ArgumentParser
import os
import time
from os.path import join

import compass.descriptors.config as cfg
import compass.descriptors.geometry as geom
import compass.descriptors.main as mm
import compass.descriptors.pca as pca
import compass.descriptors.topo_traj as tt
import compass.network.generals as gn

def _parse_cli_arguments():
    parser = ArgumentParser(
        description="Compute ComPASS descriptors and network analyses."
    )
    parser.add_argument("config_path", help="Path to the ComPASS configuration file.")
    parser.add_argument(
        "--network-only",
        action="store_true",
        help=(
            "Skip descriptor and PCA calculations and use the existing "
            "MINDIST and ADJACENCY matrices in the output directory."
        ),
    )
    return parser.parse_args()

def _run_network_pipeline(arg, min_dist_matrix_file, adjacency_file, first_timer):
    required_matrices = (min_dist_matrix_file, adjacency_file)
    missing_matrices = [
        matrix_file
        for matrix_file in required_matrices
        if not os.path.isfile(matrix_file)
    ]
    if missing_matrices:
        missing_files = "\n".join(f"  - {path}" for path in missing_matrices)
        raise FileNotFoundError(
            "Network analysis requires these existing matrix files:\n"
            f"{missing_files}"
        )

    arg.network_dir = join(arg.out_dir, "network")
    os.makedirs(arg.network_dir, exist_ok=True)
    dist_cutoffs = [arg.dist_graph, arg.dist_clique]

    gn.process_graphs(
        arg,
        min_dist_matrix_file,
        adjacency_file,
        dist_cutoffs,
    )
    print(
        f" ⏳  Until graphs construction: "
        f"{round(time.time() - first_timer, 2)} s"
    )

    gn.process_graph_files(arg.network_dir, dist_cutoffs[0])
    print(
        f" ⏳  Until network parameters computed: "
        f"{round(time.time() - first_timer, 2)} s"
    )

    gn.process_graph_files_for_communities_and_cliques(
        arg.network_dir,
        dist_cutoffs[0],
        dist_cutoffs[1],
    )
    print(
        f" ⏳  Until communities and cliques detection: "
        f"{round(time.time() - first_timer, 2)} s"
    )

    gn.generate_pymol_scripts(
        arg.network_dir,
        arg.topo,
        dist_cutoffs[0],
        dist_cutoffs[1],
    )

def runner():
    first_timer = time.time()

    cli_arguments = _parse_cli_arguments()
    arg = cfg.parse_params(
        cli_arguments.config_path,
        validate_trajectories=not cli_arguments.network_only,
    )

    if cli_arguments.network_only:
        min_dist_matrix_file = geom.get_matrix_name(
            arg.out_dir, arg.title, "MINDIST"
        )
        adjacency_file = geom.get_matrix_name(
            arg.out_dir, arg.title, "ADJACENCY"
        )
    else:
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

    _run_network_pipeline(
        arg,
        min_dist_matrix_file,
        adjacency_file,
        first_timer,
    )

    print(f" ⏳  Wall Clock Time: {time.time() - first_timer:.2f} seconds")
    print(f"**** -------Normal Termination -------****")
