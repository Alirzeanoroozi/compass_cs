import os
import time
from os.path import join
import sys
import mdtraj as md

import compass.descriptors.config as cfg
import compass.descriptors.geometry as geom
import compass.descriptors.main as mm
import compass.descriptors.pca as pca
import compass.descriptors.topo_traj as tt
import compass.network.generals as gn

def runner():
    first_timer = time.time()

    # Parse configuration file
    if len(sys.argv) != 2:
        raise ValueError('\ncompass syntax is: compass path-to-config-file')
    config_path = sys.argv[1]
    arg = cfg.parse_params(config_path)

    # Prepare data structures
    resids_to_atoms, resids_to_noh, atoms_to_resids, calphas, oxy, nitro, donors, hydros, acceptors = tt.prepare_datastructures(arg.traj, arg.topo, arg.out_dir, arg.heavies, first_timer)

    # Compute descriptors
    ave_min_dist, occ_nb, cp, occ_sb, occ_hb, occ_int, mi, gc = mm.compute_descriptors(arg, resids_to_atoms, resids_to_noh, atoms_to_resids, calphas, oxy, nitro, donors, hydros, acceptors, first_timer)

    # Save matrices
    n = len(resids_to_atoms)
    matrices, matrices_names = geom.process_matrices(arg, n, ave_min_dist, occ_nb, cp, occ_sb, occ_hb, occ_int, mi, gc, first_timer)

    # Perform PCA & generate adjacency matrix from PCA results
    adj_name = pca.run_pca(arg, matrices, n, first_timer)

    # Construct graphs
    arg.adjacency_file = adj_name
    arg.min_dist_matrix_file = matrices_names["MINDIST"]
    arg.network_dir = join(arg.out_dir, 'network')
    os.makedirs(arg.network_dir, exist_ok=True)
    dist_cutoffs = [arg.dist_graph, arg.dist_clique]

    # Create a PDB for the network visualization under the network output folder
    filename = os.path.basename(arg.topo)
    pdb_name = join(arg.network_dir, f'{os.path.splitext(filename)[0]}_internal.pdb')
    parsed = next(md.iterload(arg.traj.split()[0], top=arg.topo, chunk=1))
    parsed.save_pdb(pdb_name)
    arg.pdb_file_path = pdb_name

    gn.process_graphs(arg, dist_cutoffs)
    graph_time = round(time.time() - first_timer, 2)
    print(f' ⏳  Until graphs construction: {graph_time} s')

    # Compute network parameters
    gn.process_graph_files(arg.network_dir, dist_cutoffs[0])
    network_time = round(time.time() - first_timer, 2)
    print(f' ⏳  Until network parameters computed: {network_time} s')

    # Compute communities and cliques
    gn.process_graph_files_for_communities_and_cliques(arg.network_dir, dist_cutoffs[0], dist_cutoffs[1])
    clique_time = round(time.time() - first_timer, 2)
    print(f' ⏳  Until communities and cliques detection: {clique_time} s')

    # Generate PyMOL scripts
    gn.generate_pymol_scripts(arg.network_dir, arg.pdb_file_path, dist_cutoffs[0], dist_cutoffs[1])

    # # Find paths
    # if dict_arg["paths"]["find_path"] == 'True':
    #     source_residues = dict_arg["paths"]["sources"].split(",")
    #     target_residues = dict_arg["paths"]["targets"].split(",")

    #     for source_residue in source_residues:
    #         for target_residue in target_residues:
    #             gn.find_paths(arg.pdb_file_path, arg.network_dir, dist_cutoffs[0], source_residue.strip(), target_residue.strip())

    print(f" ⏳  Wall Clock Time: {time.time() - first_timer:.2f} seconds")
    print(f"**** -------Normal Termination -------****")
