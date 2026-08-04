import json
import mdtraj as md
import networkx as nx
import numpy as np
import pandas as pd


class ReadFiles:
    """
    A class to handle reading and parsing of files for network analysis, including matrices, structures, and centrality values.
    """

    def read_matrix(self, file_path):
        """
        Reads a matrix from a .txt file.

        Args:
            file_path (str): Path to the matrix file.

        Returns:
            np.ndarray: The matrix read from the file.
        """
        return np.loadtxt(file_path)

    def atom_mapping(self, file_path):
        """
        Extracts CA atoms for amino acids and P or O5' atoms for nucleic acids from a topology.

        Returns:
            tuple: A tuple containing:
                - atom_mapping (dict): Mapping of atom indices to atom information.
                - atoms (list): List of atom tuples (residue name, atom name, residue id, chain id).
        """
        # Load the PDB file using MDTraj
        trajectory = md.load(file_path)
        topology = trajectory.topology

        from compass.descriptors.topo_traj import select_backbone_atoms
        all_atoms = select_backbone_atoms(topology)

        atom_mapping = {}  # Maps node index to atom information
        atoms = []
        index_counter = 0

        amino_acid_count = 0
        nucleic_acid_count = 0

        # Process selected atoms to build atom_mapping
        for atom_index in all_atoms:
            atom = topology.atom(int(atom_index))
            residue = atom.residue
            chain_id = residue.chain.chain_id if residue.chain.chain_id is not None else ''
            residue_name = residue.name
            residue_id = residue.resSeq
            atom_name = atom.name

            if atom_name in ('CA', 'GC'):
                amino_acid_count += 1
            elif atom_name in ("C5'", 'C5X'):
                nucleic_acid_count += 1

            atoms.append((residue_name, atom_name, residue_id, chain_id))
            atom_mapping[index_counter] = (
            residue_name, atom_name, residue_id, chain_id)
            index_counter += 1

        print(f" 🔍  Processing matrices for graph construction")
        print(f" 📦  Processed {amino_acid_count} amino acid residues.")
        print(f" 🧬  Processed {nucleic_acid_count} nucleic acid residues.")
        print(f" ⚙️   Total residues processed: {len(atoms)}.")
        print(f" 🕸️  Graph network construction is complete.")
        return atom_mapping, atoms

    def parse_mapping(atom_mapping):
        """
        Parses atom mapping to extract residues information.

        Args:
            atom_mapping (dict): A dictionary mapping atom indices to atom information.

        Returns:
            list: A list of tuples (chain_id, res_num, atom_name).
        """
        residues = []
        for index, (
        res_name, atom_name, res_num, chain_id) in atom_mapping.items():
            residues.append((chain_id, res_num, atom_name))
        return residues

    def load_graph_and_mapping(self, input_file):
        """
        Loads the graph and atom mapping from a JSON file.

        Args:
            input_file (str): Path to the JSON file containing the graph and atom mapping.

        Returns:
            tuple: A tuple containing:
                - G (nx.Graph): The loaded graph.
                - atom_mapping (dict): The loaded atom mapping.
        """
        with open(input_file, 'r') as f:
            data = json.load(f)

        G = nx.readwrite.json_graph.node_link_graph(data['graph'])
        atom_mapping = data['atom_mapping']
        return G, atom_mapping

    def read_centrality_from_file(file_path):
        """
        Processes a JSON file containing node centrality metrics.

        Args:
            file_path (str): Path to the input file.

        Returns:
            pd.DataFrame: A DataFrame containing parsed node metrics.
        """
        with open(file_path, 'r') as f:
            payload = json.load(f)

        data = []
        for node in payload.get("nodes", []):
            data.append({
                "Node_Res_Num": int(node["res_num"]),
                "Chain_ID": node.get("chain_id", ""),
                "Betweenness": float(node["betweenness"]),
                "Closeness": float(node["closeness"]),
                "Degree": int(node["degree"]),
            })
        return pd.DataFrame(data)

    def read_edge_betweenness_from_file(file_path):
        """
        Processes a JSON file containing edge betweenness metrics.

        Args:
            file_path (str): Path to the input file.

        Returns:
            pd.DataFrame: A DataFrame containing parsed edge metrics.
        """
        with open(file_path, 'r') as f:
            payload = json.load(f)

        edges = []
        for edge in payload.get("edges", []):
            edges.append({
                "Res1": int(edge["res_num1"]),
                "Chain1": edge.get("chain_id1", ""),
                "Res2": int(edge["res_num2"]),
                "Chain2": edge.get("chain_id2", ""),
                "Betweenness": float(edge["betweenness"]),
            })
        return pd.DataFrame(edges)
