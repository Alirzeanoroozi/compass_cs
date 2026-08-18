# Created by rglez at 9/13/24
import os
import time
import shutil
import json

from compass.network.communities_cliques import CliqueDetector, CommunityDetector
from compass.network.graph_constructor import GraphConstructor
from compass.network.networkparameters import NetworkParameters
from compass.network.pymol_visualizer import PyMOLVisualizer
from compass.network.read_files import ReadFiles

def _graph_json_name(dist_cutoff):
    return f"graph_cutoff_{dist_cutoff}.json"

def process_graphs(param_space, distance_cutoffs):
    # Initialize GraphConstructor
    graph_constructor = GraphConstructor(
        distance_file=param_space.min_dist_matrix_file,
        adjacency_file=param_space.adjacency_file,
        distance_cutoffs=distance_cutoffs
    )
    atom_mapping, _ = graph_constructor.reader.atom_mapping(param_space.pdb_file_path)

    # Iterate over each distance cutoff to build and process the graph
    for distance_cutoff in graph_constructor.distance_cutoffs:
        start_time = time.time()
        G = graph_constructor.build_graph_from_matrices(distance_cutoff)
        graph_filename = _graph_json_name(distance_cutoff)
        output_graph_file = os.path.join(param_space.network_dir, graph_filename)
        # Save the graph and atom mapping
        graph_constructor.save_graph_and_mapping(G, atom_mapping, output_file=output_graph_file)
        # Plot and save histogram
        output_file_prefix = os.path.join(param_space.network_dir, f"graph_cutoff_{distance_cutoff}")
        graph_constructor.plot_and_save_histogram(G, output_file_prefix=output_file_prefix)
        print(f" 🖥️  Processed graph for cutoff {distance_cutoff} in {round(time.time() - start_time, 2)} seconds")

def process_graph_files(results_dir, dist_cutoff_graph):
    graph_filename = _graph_json_name(dist_cutoff_graph)
    json_path = os.path.join(results_dir, graph_filename)

    graph, atom_mapping = ReadFiles().load_graph_and_mapping(json_path)
    network_parameters = NetworkParameters(G=graph, atom_mapping=atom_mapping)
    prefix = graph_filename.replace('.json', '')
    shortest_paths_file = os.path.join(results_dir, f"{prefix}_shortest_paths.json")
    centralities_file = os.path.join(results_dir, f"{prefix}_centralities.json")
    edge_betweenness_file = os.path.join(results_dir, f"{prefix}_edge_betweenness.json")
    top_nodes_file = os.path.join(results_dir, f"{prefix}_top_5_percent_nodes.json")
    top_shortest_paths_file = os.path.join(results_dir, f"{prefix}_top_10_shortest_paths.json")

    network_parameters.compute_shortest_paths(shortest_paths_file, top_shortest_paths_file)
    centralities = network_parameters.calculate_centralities()
    network_parameters.save_centrality_measures(centralities, centralities_file)
    edge_betweenness = network_parameters.calculate_edge_betweenness()
    network_parameters.save_edge_betweenness(edge_betweenness, edge_betweenness_file)
    network_parameters.identify_top_10_percent_nodes(centralities, top_nodes_file)

def find_paths(pdb_file, results_dir, dist_cutoff_graph, source_res, target_res):
    graph_filename = _graph_json_name(dist_cutoff_graph)
    json_path = os.path.join(results_dir, graph_filename)

    graph, atom_mapping = ReadFiles().load_graph_and_mapping(json_path)
    network_parameters = NetworkParameters(G=graph, atom_mapping=atom_mapping)
    visualizer = PyMOLVisualizer(pdb_file=pdb_file, atom_mapping=atom_mapping, graph=graph)
    prefix = graph_filename.replace('.json', '')
    alt_paths_file = os.path.join(results_dir, f"{prefix}_alt_paths.json")
    network_parameters.find_alternative_paths(source_res, target_res, alt_paths_file)

    output_pml_file = os.path.join(results_dir, f"{prefix}_alt_paths.pml")
    visualizer.write_pml_script_for_alternative_paths(alt_paths_file, output_pml_file)

def process_graph_files_for_communities_and_cliques(results_dir, dist_cutoff_graph, dist_cutoff_clique):
    """
    Processes graph files in the specified directory to detect communities and cliques.

    Args:
        results_dir (str): Directory containing the graph files.
        method (str): Community detection method ('leiden' or 'girvan').
    """

    graph_filename = _graph_json_name(dist_cutoff_graph)
    json_path = os.path.join(results_dir, graph_filename)
    if os.path.exists(json_path):
        graph, atom_mapping = ReadFiles().load_graph_and_mapping(json_path)
        community_detector = CommunityDetector(G=graph,
                                               atom_mapping=atom_mapping)
        prefix = graph_filename.replace('.json', '')
        communities_leiden, modularity_leiden = community_detector.detect_communities_leiden()
        communities_file = os.path.join(results_dir,
                                        f"{prefix}_communities_leiden.json")
        community_detector.save_communities_to_file(communities_leiden,
                                                    communities_file)
        print(
            f" 🧩  Communities detected using Leiden algorithm saved to {communities_file}")

    clique_filename = _graph_json_name(dist_cutoff_clique)
    clique_json_path = os.path.join(results_dir, clique_filename)
    if os.path.exists(clique_json_path):
        graph, atom_mapping = ReadFiles().load_graph_and_mapping(clique_json_path)
        clique_detector = CliqueDetector(G=graph, atom_mapping=atom_mapping)
        prefix = clique_filename.replace('.json', '')
        cliques_file = os.path.join(results_dir, f"{prefix}_cliques.json")
        cliques = clique_detector.detect_cliques()
        clique_detector.save_cliques_to_file(cliques, cliques_file)

def generate_pymol_scripts(results_dir, pdb_file, dist_cutoff_graph, dist_cutoff_clique):
    graph_filename = _graph_json_name(dist_cutoff_graph)
    json_path = os.path.join(results_dir, graph_filename)
    prefix = graph_filename.replace('.json', '')
    graph, atom_mapping = ReadFiles().load_graph_and_mapping(json_path)

    # Get the file paths for the result files
    communities_file = os.path.join(results_dir, f"{prefix}_communities_leiden.json")
    centrality_file = os.path.join(results_dir, f"{prefix}_centralities.json")
    edge_betweenness_file = os.path.join(results_dir, f"{prefix}_edge_betweenness.json")
    top_nodes_file = os.path.join(results_dir, f"{prefix}_top_5_percent_nodes.json")
    paths_file = os.path.join(results_dir, f"{prefix}_paths.json")
    output_pml_file = os.path.join(results_dir, f"{prefix}_top_5_percent.pml")
    output_pml_communities = os.path.join(results_dir, f"{prefix}_communities.pml")
    output_top_paths = os.path.join(results_dir, f"{prefix}_top_paths.pml")
    top_15_file = os.path.join(results_dir, f"{prefix}_top_10_shortest_paths.json")
    pdb_output_path = os.path.join(results_dir, os.path.basename(pdb_file))

    # Prefer the PDB inside the network folder so .pml scripts resolve locally
    if os.path.abspath(pdb_file) != os.path.abspath(pdb_output_path):
        shutil.copy(pdb_file, pdb_output_path)

    visualizer = PyMOLVisualizer(pdb_file=pdb_output_path, atom_mapping=atom_mapping, graph=graph)

    visualizer.communities_pml(communities_file, output_pml_communities)
    visualizer.graph_pml(centrality_file, edge_betweenness_file, os.path.join(results_dir, f"{prefix}_graph"))
    visualizer.highlight_top_nodes_pml(pdb_output_path, atom_mapping, top_nodes_file, output_pml_file)

    with open(paths_file, 'r') as f:
        paths_data = json.load(f)
    residue_list = paths_data if isinstance(paths_data, list) else paths_data.get("residues", [])
    visualizer.write_pml_script_for_residue_paths(residue_list, os.path.join(results_dir, f"{prefix}_paths.pml"))
    visualizer.write_pml_script_for_top_shortest_paths(top_15_file, edge_betweenness_file, output_top_paths)

    clique_filename = _graph_json_name(dist_cutoff_clique)
    clique_json_path = os.path.join(results_dir, clique_filename)
    prefix = clique_filename.replace('.json', '')
    graph, atom_mapping = ReadFiles().load_graph_and_mapping(clique_json_path)
    pdb_output_path = os.path.join(results_dir, os.path.basename(pdb_file))
    visualizer = PyMOLVisualizer(pdb_file=pdb_output_path, atom_mapping=atom_mapping, graph=graph)
    cliques_file = os.path.join(results_dir, f"{prefix}_cliques.json")
    output_pml_cliques = os.path.join(results_dir, f"{prefix}_cliques.pml")
    visualizer.cliques_pml(cliques_file, output_pml_cliques)

    print(f" 🖥️  PyMOL scripts generated.")
