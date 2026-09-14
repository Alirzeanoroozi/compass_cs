import colorsys
import json
import math
import random

from compass.network.read_files import read_centrality_from_file, read_edge_betweenness_from_file

class PyMOLVisualizer:
    def __init__(self, pdb_file, atom_mapping, graph):
        self.pdb_file = pdb_file
        self.atom_mapping = atom_mapping
        self.graph = graph

    def parse_communities_file(self, communities_file):
        with open(communities_file, 'r') as f:
            payload = json.load(f)

        communities = {}
        for community_idx, members in payload.get("communities", {}).items():
            labels = []
            for member in members:
                if isinstance(member, dict):
                    labels.append(member.get(
                        "label",
                        f"{member.get('chain_id', '')}_{member.get('res_num', '')}"
                    ))
                else:
                    labels.append(str(member))
            communities[str(community_idx)] = labels
        return communities

    def communities_pml(self, communities_file, output_pml_file):
        communities = self.parse_communities_file(communities_file)

        standard_colors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
            [1.0, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]

        color_rng = random.Random(42)

        def generate_random_color():
            return [
                color_rng.random(),
                color_rng.random(),
                color_rng.random(),
            ]

        community_colors = {}
        for i, community in enumerate(communities):
            if i < len(standard_colors):
                community_colors[community] = standard_colors[i]
            else:
                community_colors[community] = generate_random_color()

        with open(output_pml_file, 'w') as f:
            f.write(f"load {self.pdb_file} \n")

            for community, color in community_colors.items():
                f.write(
                    f"set_color color{community}, [{', '.join(map(str, color))}]\n")
            for i, (community, nodes) in enumerate(communities.items()):
                community_residues = []
                color = f"color{community}"
                for node in nodes:
                    chain_id, res_num = node.split('_')[:2]
                    if chain_id:
                        f.write(f"color {color}, chain {chain_id} and resi {res_num}\n")
                        community_residues.append(f"chain {chain_id} and resi {res_num}")
                    else:
                        f.write(f"color {color}, resi {res_num}\n")
                        community_residues.append(f"resi {res_num}")
                if community_residues:
                    f.write(f"select community_{i}, {' + '.join(community_residues)}\n")

            f.write("show cartoon\n")
            f.write("bg_color white\n")

        print(f" 🧊  PyMOL script for communities saved to {output_pml_file}")

    def parse_cliques_file(self, cliques_file):
        with open(cliques_file, 'r') as f:
            payload = json.load(f)

        cliques = {}
        for clique_idx, members in payload.get("cliques", {}).items():
            labels = []
            for member in members:
                if isinstance(member, dict):
                    labels.append(member.get(
                        "label",
                        f"{member.get('chain_id', '')}_{member.get('res_num', '')}"
                    ))
                else:
                    labels.append(str(member).strip().replace(',', ''))
            cliques[int(clique_idx)] = labels
        return cliques

    def cliques_pml(self, cliques_file, output_pml):
        cliques = self.parse_cliques_file(cliques_file)
        color_rng = random.Random(42)

        with open(output_pml, 'w') as f:
            f.write(f"load {self.pdb_file}\n")
            f.write("show cartoon\n")
            f.write("set cartoon_color, grey90\n")

            for i, (clique, nodes) in enumerate(cliques.items(), start=1):
                color = [
                    color_rng.random(),
                    color_rng.random(),
                    color_rng.random(),
                ]
                f.write(
                    f"set_color clique_{i}, [{', '.join(map(str, color))}]\n")
                selection_residues = []
                for node in nodes:
                    chain_id, res_num = node.split('_')[:2]
                    res_num = int(str(res_num).strip().replace(',', ''))
                    selection_residues.append(f"chain {chain_id} and resi {res_num}")
                    f.write(f"color clique_{i}, chain {chain_id} and resi {res_num}\n")
                    f.write(
                        f"show spheres, chain {chain_id} and resi {res_num} "
                        f"and (name CA or name C5' or name GC or name C5X)\n"
                    )
                if selection_residues:
                    f.write(
                        f"select clique_{i}, {' + '.join(selection_residues)}\n")
            f.write("set dash_gap, 0\n")
            f.write("set dash_color, grey10\n")
            f.write("hide labels\n")
            f.write("set sphere_scale, 0.7\n")
            f.write("bg_color white\n")

        print(f" 🧊  PyMOL script for cliques saved to {output_pml}")

    def graph_pml(self, centrality_file, edge_betweenness_file, output_pml):
        centrality_df = read_centrality_from_file(centrality_file)
        betweenness_df = read_edge_betweenness_from_file(edge_betweenness_file)
        out_file = open(output_pml, 'w')


        backbone = "(name CA or name C5' or name GC or name C5X)"
        out_file.write(f"load {self.pdb_file}\n")
        out_file.write(f"show_as cartoon, structure\n")
        out_file.write(f"set cartoon_transparency, 0.6\n")

        max_centrality = centrality_df['Betweenness'].max()
        for _, row in centrality_df.iterrows():
            res_num = row['Node_Res_Num']
            chain_id = row['Chain_ID']
            centrality_value = row['Betweenness']
            norm_centrality = centrality_value / max_centrality if max_centrality else 0
            sphere_scale = 0.3 + 1 * norm_centrality
            out_file.write(f"show spheres, chain {chain_id} and resi {res_num} and {backbone}\n")
            out_file.write(f"set sphere_scale, {sphere_scale:.2f}, chain {chain_id} and resi {res_num} and {backbone}\n")

        max_betweenness = betweenness_df['Betweenness'].max()
        for _, row in betweenness_df.iterrows():
            res1 = int(row['Res1'])
            res2 = int(row['Res2'])
            chain1 = str(row['Chain1'])
            chain2 = str(row['Chain2'])
            betweenness_value = row['Betweenness']
            norm_betweenness = betweenness_value / max_betweenness if max_betweenness else 0
            thickness = 0.5 + 10 * norm_betweenness

            out_file.write(f"distance edge_{res1}_{res2}, chain {chain1} and resi {res1} and {backbone}, chain {chain2} and resi {res2} and {backbone}\n")
            out_file.write(f"set dash_width, {thickness:.2f}, edge_{res1}_{res2}\n")

        out_file.write("hide labels\n")
        out_file.write("set dash_gap, 0\n")
        out_file.write("set dash_color, black\n")
        out_file.write("bg_color white\n")
        out_file.close()

        print(f" 🧊  PyMOL script for graph attributes saved to {output_pml}")

    def highlight_top_nodes_pml(self, pdb_file, atom_mapping, nodes_file, output_pml_file):
        """
        Generates a PyMOL script to highlight residues corresponding to nodes from a file.

        Args:
            pdb_file (str): Path to the PDB file.
            atom_mapping (str): Path to the PDB atom mapping file.
            nodes_file (str): Path to the file containing node indices or names.
            output_pml_file (str): Path to the output PyMOL script file.
        """
        try:
            with open(nodes_file, 'r') as f:
                payload = json.load(f)

            residue_info = []
            for node in payload.get("nodes", []):
                residue_info.append((int(node["res_num"]), node.get("chain_id", "")))

            with open(output_pml_file, 'w') as f:
                f.write(f"load {pdb_file}\n")
                f.write("set cartoon_color, grey90\n")
                f.write("set_color highlight_color, [1.0, 0.0, 0.0]\n")
                residue_selections = []
                for res_num, chain_id in residue_info:
                    selection = (
                        f"(chain {chain_id} and resi {res_num} and "
                        f"(name CA or name C5' or name GC or name C5X))"
                    )
                    residue_selections.append(selection)
                    f.write(f"show spheres, {selection}\n")
                    f.write(f"color highlight_color, {selection}\n")

                if residue_selections:
                    selection_string = f"sele hotspot_residues, {' or '.join(residue_selections)}\n"
                    f.write(selection_string)
                f.write("set sphere_scale, 0.7\n")
                f.write("bg_color white\n")

            print(
                f" 🧊 PyMOL script to highlight top nodes saved to {output_pml_file}")

        except FileNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def write_pml_script_for_residue_paths(self, paths_file, output_pml_file):
        with open(paths_file, 'r') as f:
            paths_list = json.load(f)['paths']

        backbone = "(name CA or name C5' or name GC or name C5X)"
        with open(output_pml_file, 'w') as f:
            f.write(f"load {self.pdb_file}\n")
            f.write("set cartoon_color, grey90\n")
            f.write("set_color black, [0.0, 0.0, 0.0]\n")
            f.write("bg_color white\n")

            for path in paths_list:
                source_residue, target_residue = int(path['source']) + 1, int(path['target']) + 1
                f.write(f"select resi {source_residue}, resi {target_residue}\n")
                f.write(f"distance path_{source_residue}_{target_residue}, resi {source_residue} and {backbone}, resi {target_residue} and {backbone}\n")
                f.write(f"set gap_width, 0, path_{source_residue}_{target_residue}\n")
                f.write(f"color black, path_{source_residue}_{target_residue}\n")
                f.write(f"hide labels, path_{source_residue}_{target_residue}\n")

        print(f" 🧊  PyMOL script for residue paths saved to {output_pml_file}")

    def write_pml_script_for_shortest_paths(self, paths_file, edge_betweenness_file, output_pml_file):
        betweenness_df = read_edge_betweenness_from_file(edge_betweenness_file)

        def edge_key(res_num1, chain_id1, res_num2, chain_id2):
            residue1 = (str(chain_id1), int(res_num1))
            residue2 = (str(chain_id2), int(res_num2))
            return tuple(sorted((residue1, residue2)))

        betweenness_by_edge = {}
        finite_betweenness = []
        for row in betweenness_df.itertuples(index=False):
            betweenness = float(row.Betweenness)
            betweenness_by_edge[
                edge_key(row.Res1, row.Chain1, row.Res2, row.Chain2)
            ] = betweenness
            if math.isfinite(betweenness):
                finite_betweenness.append(betweenness)
        max_betweenness = max(finite_betweenness, default=0.0)

        with open(paths_file, 'r') as f:
            payload = json.load(f)
        path_entries = [
            entry
            for entry in payload.get("paths", [])
            if len(entry.get("path", [])) >= 2
        ]

        with open(output_pml_file, 'w') as f:
            f.write(f'load "{self.pdb_file}", structure\n')
            f.write("show cartoon, structure\n")
            f.write("color grey80, structure\n")
            f.write("set cartoon_transparency, 0.6, structure\n")
            f.write("bg_color white\n")

            for path_index, entry in enumerate(path_entries, start=1):
                path = entry["path"]
                mapped_path = [
                    self.atom_mapping[str(node)]
                    for node in path
                ]
                _, _, source_res_num, source_chain_id = mapped_path[0]
                _, _, target_res_num, target_chain_id = mapped_path[-1]
                source_chain = str(source_chain_id) or "none"
                target_chain = str(target_chain_id) or "none"
                path_name = (
                    f"path_{path_index:02d}_"
                    f"{source_chain}{source_res_num}_to_"
                    f"{target_chain}{target_res_num}"
                )
                color_name = f"path_color_{path_index:02d}"
                hue = ((path_index - 1) * 0.61803398875) % 1.0
                red, green, blue = colorsys.hsv_to_rgb(hue, 0.75, 0.95)

                path_edges = []
                path_betweenness = []
                for node_index in range(len(mapped_path) - 1):
                    _, atom_name1, res_num1, chain_id1 = mapped_path[node_index]
                    _, atom_name2, res_num2, chain_id2 = mapped_path[node_index + 1]
                    path_edges.append((
                        f"chain {chain_id1} and resi {res_num1} "
                        f"and name {atom_name1}",
                        f"chain {chain_id2} and resi {res_num2} "
                        f"and name {atom_name2}",
                    ))
                    betweenness = betweenness_by_edge.get(
                        edge_key(res_num1, chain_id1, res_num2, chain_id2)
                    )
                    if betweenness is not None and math.isfinite(betweenness):
                        path_betweenness.append(betweenness)

                mean_betweenness = (
                    sum(path_betweenness) / len(path_betweenness)
                    if path_betweenness
                    else 0.0
                )
                normalized_betweenness = (
                    mean_betweenness / max_betweenness
                    if max_betweenness > 0
                    else 0.0
                )
                thickness = 2.0 + 4.0 * normalized_betweenness

                f.write(
                    f"set_color {color_name}, "
                    f"[{red:.3f}, {green:.3f}, {blue:.3f}]\n"
                )
                for selection1, selection2 in path_edges:
                    f.write(
                        f"distance {path_name}, {selection1}, {selection2}\n"
                    )
                f.write(f"hide labels, {path_name}\n")
                f.write(f"set dash_gap, 0, {path_name}\n")
                f.write(f"set dash_width, {thickness:.2f}, {path_name}\n")
                f.write(f"set dash_color, {color_name}, {path_name}\n")
                f.write(f"group shortest_paths, {path_name}\n")

            print(f"🧊 PyMOL script for shortest paths saved to {output_pml_file}")


    def write_pml_script_for_alternative_paths(self, alternative_paths_file,
                                               output_pml_file):
        """
        Generates a PyMOL script to draw lines connecting consecutive residues in alternative paths.

        Args:
            alternative_paths_file (str): Path to the alternative paths JSON file.
            output_pml_file (str): Path to the output PyMOL script file.
        """
        backbone = "(name CA or name C5' or name GC or name C5X)"
        try:
            with open(alternative_paths_file, 'r') as f:
                payload = json.load(f)

            paths = []
            if payload.get("shortest_path"):
                paths.append([
                    (str(node["res_num"]), str(node.get("chain_id", "")))
                    for node in payload["shortest_path"]
                ])
            for alt in payload.get("alternative_paths", []):
                paths.append([
                    (str(node["res_num"]), str(node.get("chain_id", "")))
                    for node in alt
                ])
        except Exception as e:
            print(f"Error reading alternative paths file: {e}")
            return

        try:
            with open(output_pml_file, 'w') as f:
                f.write(f"load {self.pdb_file}\n")
                for path in paths:
                    for i in range(len(path) - 1):
                        res_num1, chain_id1 = path[i]
                        res_num2, chain_id2 = path[i + 1]
                        f.write(
                            f"select resi_{res_num1}, chain {chain_id1} and resi {res_num1} and {backbone}\n")
                        f.write(
                            f"select resi_{res_num2}, chain {chain_id2} and resi {res_num2} and {backbone}\n")
                        f.write(
                            f"show spheres, chain {chain_id1} and resi {res_num1} and {backbone} or chain {chain_id2} and resi {res_num2} and {backbone}\n")
                        f.write(
                            f"distance edge_{res_num1}_{res_num2}, "
                            f"chain {chain_id1} and resi {res_num1} and {backbone}, "
                            f"chain {chain_id2} and resi {res_num2} and {backbone}\n"
                        )
                f.write("hide labels \n")
                f.write("set dash_gap, 0 \n")
                f.write("set dash_color, grey10 \n")
                f.write("bg_color white\n")
                f.write("set sphere_transparency, 0.3 \n")
        except Exception as e:
            print(f"Error writing PyMOL script to file: {e}")
