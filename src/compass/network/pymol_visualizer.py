import random
import json

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

        def generate_random_color():
            return [random.random(), random.random(), random.random()]

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

        with open(output_pml, 'w') as f:
            f.write(f"load {self.pdb_file}\n")
            f.write("show cartoon\n")
            f.write("set cartoon_color, grey90\n")

            for i, (clique, nodes) in enumerate(cliques.items(), start=1):
                color = [random.random(), random.random(), random.random()]
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
        output_files = {
            "all": open(f"{output_pml}_all.pml", 'w'),
        }

        backbone = "(name CA or name C5' or name GC or name C5X)"
        for file in output_files.values():
            file.write(f"load {self.pdb_file}\n")
            file.write(f"show_as cartoon, structure\n")
            file.write(f"set cartoon_transparency, 0.6\n")

        max_centrality = centrality_df['Betweenness'].max()
        for _, row in centrality_df.iterrows():
            res_num = row['Node_Res_Num']
            chain_id = row['Chain_ID']
            centrality_value = row['Betweenness']
            norm_centrality = centrality_value / max_centrality if max_centrality else 0
            sphere_scale = 0.3 + 1 * norm_centrality
            for file in output_files.values():
                file.write(
                    f"show spheres, chain {chain_id} and resi {res_num} and {backbone}\n")
                file.write(
                    f"set sphere_scale, {sphere_scale:.2f}, chain {chain_id} and resi {res_num} and {backbone}\n")

        max_betweenness = betweenness_df['Betweenness'].max()
        for _, row in betweenness_df.iterrows():
            res1 = int(row['Res1'])
            res2 = int(row['Res2'])
            chain1 = str(row['Chain1'])
            chain2 = str(row['Chain2'])
            betweenness_value = row['Betweenness']
            norm_betweenness = betweenness_value / max_betweenness if max_betweenness else 0
            thickness = 0.5 + 10 * norm_betweenness

            output_files["all"].write(
                f"distance edge_{res1}_{res2}, chain {chain1} and resi {res1} and {backbone}, chain {chain2} and resi {res2} and {backbone}\n"
            )
            output_files["all"].write(
                f"set dash_width, {thickness:.2f}, edge_{res1}_{res2}\n"
            )

        for file in output_files.values():
            file.write("hide labels\n")
            file.write("set dash_gap, 0\n")
            file.write("set dash_color, black\n")
            file.write("bg_color white\n")
            file.close()

        print(
            f" 🧊  PyMOL script for graph attributes saved with prefix {output_pml}")

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

    def write_pml_script_for_top_shortest_paths(self, top_file, edge_betweenness_file, output_pml_file):
        betweenness_df = read_edge_betweenness_from_file(edge_betweenness_file)
        max_betweenness = betweenness_df['Betweenness'].max()

        with open(top_file, 'r') as f:
            payload = json.load(f)
        paths = [entry["path"] for entry in payload.get("paths", []) if "path" in entry]

        with open(output_pml_file, 'w') as f:
            written_selections = set()
            written_distances = set()
            written_spheres = set()
            f.write(f"load {self.pdb_file}\n")
            for path in paths:
                for i in range(len(path) - 1):
                    node1 = path[i]
                    node2 = path[i + 1]
                    try:
                        res_name1, atom_name1, res_num1, chain_id1 = self.atom_mapping[str(node1)]
                        res_name2, atom_name2, res_num2, chain_id2 = self.atom_mapping[str(node2)]
                    except KeyError as e:
                        print(f"Warning: Node {e} not found in atom mappings.")
                        continue

                    edge_betweenness_row = betweenness_df[((betweenness_df['Res1'] == res_num1) & (betweenness_df['Res2'] == res_num2)) | ((betweenness_df['Res1'] == res_num2) & (betweenness_df['Res2'] == res_num1))]
                    if edge_betweenness_row.empty:
                        print(f"Warning: No betweenness data for edge ({node1}, {node2}).")
                        continue

                    betweenness_value = edge_betweenness_row.iloc[0]['Betweenness']
                    norm_betweenness = betweenness_value / max_betweenness if max_betweenness else 0
                    thickness = 1 + 10 * norm_betweenness

                    selection1 = f"select resi_{res_num1}, chain {chain_id1} and resi {res_num1} and name {atom_name1}"
                    selection2 = f"select resi_{res_num2}, chain {chain_id2} and resi {res_num2} and name {atom_name2}"
                    if selection1 not in written_selections:
                        f.write(f"{selection1}\n")
                        written_selections.add(selection1)
                    if selection2 not in written_selections:
                        f.write(f"{selection2}\n")
                        written_selections.add(selection2)

                    sphere_cmd1 = f"show spheres, resi {res_num1} and chain {chain_id1} and name {atom_name1}"
                    sphere_cmd2 = f"show spheres, resi {res_num2} and chain {chain_id2} and name {atom_name2}"
                    if sphere_cmd1 not in written_spheres:
                        f.write(f"{sphere_cmd1}\n")
                        written_spheres.add(sphere_cmd1)
                    if sphere_cmd2 not in written_spheres:
                        f.write(f"{sphere_cmd2}\n")
                        written_spheres.add(sphere_cmd2)

                    sorted_res = sorted([
                        (res_num1, chain_id1, atom_name1),
                        (res_num2, chain_id2, atom_name2),
                    ])
                    distance_key = f"edge_{sorted_res[0][0]}_{sorted_res[1][0]}"
                    distance_cmd = (
                        f"distance {distance_key}, "
                        f"chain {sorted_res[0][1]} and resi {sorted_res[0][0]} and name {sorted_res[0][2]}, "
                        f"chain {sorted_res[1][1]} and resi {sorted_res[1][0]} and name {sorted_res[1][2]}"
                    )
                    if distance_cmd not in written_distances:
                        f.write(f"{distance_cmd}\n")
                        f.write(
                            f"set dash_width, {thickness:.2f}, {distance_key}\n")
                        written_distances.add(distance_cmd)
                break
            f.write("hide labels\n")
            f.write("set dash_gap, 0\n")
            f.write("set dash_color, grey10\n")
            f.write("bg_color white\n")
            f.write("set sphere_transparency, 0.3\n")
            f.write("set sphere_scale, 0.5\n")

            print(f"🧊 PyMOL script for top shortest paths saved to {output_pml_file}")


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
