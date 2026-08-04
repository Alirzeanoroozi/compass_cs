from typing import Dict, List
import json
import igraph as ig
import leidenalg as la
import networkx as nx


class CommunityDetector:
    """
    A class for detecting communities in a graph using various algorithms.

    Attributes:
        G (nx.Graph): The input graph for community detection.
    """

    def __init__(self, G, atom_mapping=None):
        """
        Initialize the NetworkParameters class.

        Args:
            G (nx.Graph): Network graph with weighted edges.
            atom_mapping (dict, optional): Dictionary mapping node indices to atom information.
                Expected format: {node_id: (chain, residue_name, residue_number, atom_name)}
        """
        self.G = G
        self.atom_mapping = atom_mapping if atom_mapping else {}

    def detect_communities_leiden(self):
        """
        Detects communities using the Leiden algorithm.

        Returns:
            tuple: A tuple containing:
                - communities (dict): A dictionary mapping nodes to their community index.
                - modularity (float): The modularity of the partition.
        """
        edges = list(self.G.edges())
        nodes = list(self.G.nodes())

        g = ig.Graph()
        g.add_vertices(len(nodes))
        node_mapping = {node: idx for idx, node in enumerate(nodes)}
        edge_list = [(node_mapping[edge[0]], node_mapping[edge[1]]) for edge in
                     edges]
        g.add_edges(edge_list)

        partition = la.find_partition(g, la.ModularityVertexPartition)
        modularity = partition.modularity

        communities = {}
        for idx, community in enumerate(partition):
            for node_idx in community:
                original_node = nodes[node_idx]
                communities[original_node] = idx

        print(
            f" 🧩  Leiden detected {len(set(communities.values()))} communities with modularity {modularity:.4f}.")

        return communities, modularity

    @staticmethod
    def _collect_members_partition(self, communities):
        """
        Organizes communities by community index.

        Args:
            communities (dict): Dictionary mapping nodes to their community index.

        Returns:
            dict: Dictionary mapping community indices to lists of member records.
        """
        community_groups = {}
        for node, comm_idx in communities.items():
            if comm_idx not in community_groups:
                community_groups[comm_idx] = []
            res_name, atom_name, res_num, chain_id = self.atom_mapping.get(
                str(node), ("Unknown", "Unknown", "Unknown", "Unknown"))
            community_groups[comm_idx].append({
                "node": int(node),
                "chain_id": chain_id,
                "res_num": res_num,
                "label": f"{chain_id}_{res_num}",
            })
        return community_groups

    def save_communities_to_file(self, communities, output_file):
        """
        Saves communities to a JSON file.

        Args:
            communities (dict): A dictionary mapping nodes to their community index.
            output_file (str): Path to the output file.
        """
        try:
            community_groups = self._collect_members_partition(self,
                                                               communities)
            payload = {
                "description": (
                    "Communities file. Members are listed with chain_id and "
                    "residue_number."
                ),
                "communities": {
                    str(comm_idx): members
                    for comm_idx, members in sorted(community_groups.items())
                },
            }
            with open(output_file, 'w') as f:
                json.dump(payload, f, indent=2)
            print(f" 🧩  Communities saved to {output_file}")
        except Exception as e:
            print(f"Error writing communities to {output_file}: {e}")
            raise


class CliqueDetector:
    """
    A class for detecting cliques in a graph and saving them to files.

    Attributes:
        G (nx.Graph): The input graph for clique detection.
    """

    def __init__(self, G, atom_mapping=None):
        """
        Initialize the NetworkParameters class.

        Args:
            G (nx.Graph): Network graph with weighted edges.
            atom_mapping (dict, optional): Dictionary mapping node indices to atom information.
                Expected format: {node_id: (chain, residue_name, residue_number, atom_name)}
        """
        self.G = G
        self.atom_mapping = atom_mapping if atom_mapping else {}

    def detect_cliques(self) -> Dict[int, List]:
        """
        Detects cliques in the graph and returns a dictionary of cliques.

        Returns:
            dict: A dictionary where keys are clique indices and values are lists of nodes in each clique.
        """
        try:
            all_cliques = list(nx.find_cliques(self.G))
            large_cliques = [clique for clique in all_cliques if
                             len(clique) > 2]
            sorted_large_cliques = sorted(large_cliques, key=len, reverse=True)
            selected_cliques = []
            used_nodes = set()

            for clique in sorted_large_cliques:
                if not any(node in used_nodes for node in clique):
                    selected_cliques.append(clique)
                    used_nodes.update(clique)

            clique_dict = self._collect_members_cliques(selected_cliques)

            return clique_dict
        except Exception as e:
            print(f"Error detecting cliques: {e}")
            raise

    @staticmethod
    def _collect_members_cliques(cliques: List[List]) -> Dict[int, List]:
        """
        Collects cliques into a dictionary format.

        Args:
            cliques (list): List of cliques detected by NetworkX.

        Returns:
            dict: Dictionary mapping clique indices to lists of nodes in each clique.
        """
        return {idx: sorted(clique) for idx, clique in enumerate(cliques)}

    def save_cliques_to_file(self, cliques: Dict[int, List],
                             output_file: str) -> None:
        """
        Saves cliques to a JSON file.

        Args:
            cliques (dict): A dictionary where keys are clique indices and values are
                           lists of nodes in each clique.
            output_file (str): Path to the file where all cliques will be saved.

        Raises:
            IOError: If there's an error writing to the files.
        """

        def get_details(member):
            res_name, atom_name, res_num, chain_id = self.atom_mapping.get(
                str(member), ("Unknown", "Unknown", "Unknown", "Unknown"))
            return {
                "node": int(member),
                "chain_id": chain_id,
                "res_num": res_num,
                "label": f"{chain_id}_{res_num}",
            }

        try:
            payload = {
                "description": (
                    "Cliques file. Members are listed with chain_id and "
                    "residue_number."
                ),
                "cliques": {
                    str(clique_idx): [get_details(member) for member in members]
                    for clique_idx, members in cliques.items()
                },
            }
            with open(output_file, 'w') as all_cliques_file:
                json.dump(payload, all_cliques_file, indent=2)

            print(f" 🧩  Cliques saved to {output_file}")

        except IOError as e:
            print(f"Error writing cliques to file: {e}")
            raise
