import json
import mdtraj as md
import networkx as nx
import numpy as np
import pandas as pd

from compass.descriptors.topo_traj import select_backbone_atoms

def read_matrix(file_path):
    return np.loadtxt(file_path)

def read_atom_mapping(file_path):
    trajectory = md.load(file_path)
    topology = trajectory.topology

    backbone_atoms = select_backbone_atoms(topology)

    atom_mapping = {}
    atoms = []
    index_counter = 0

    amino_acid_count = 0
    nucleic_acid_count = 0

    for atom_index in backbone_atoms:
        atom = topology.atom(atom_index)
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
        atom_mapping[index_counter] = (residue_name, atom_name, residue_id, chain_id)
        index_counter += 1

    print(f" 🔍  Processing matrices for graph construction")
    print(f" 📦  Processed {amino_acid_count} amino acid residues.")
    print(f" 🧬  Processed {nucleic_acid_count} nucleic acid residues.")
    print(f" ⚙️   Total residues processed: {len(atoms)}.")
    print(f" 🕸️  Graph network construction is complete.")
    return atom_mapping, atoms

def load_graph_and_mapping(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    G = nx.readwrite.json_graph.node_link_graph(data['graph'])
    atom_mapping = data['atom_mapping']
    return G, atom_mapping

def read_centrality_from_file(file_path):
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
