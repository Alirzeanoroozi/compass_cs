import heapq
import json
import time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.patches import Patch


class NetworkParameters:
    def __init__(self, G, atom_mapping=None):
        self.G = G
        self.atom_mapping = atom_mapping

    def _atom_info(self, node):
        res_name, atom_name, res_num, chain_id = self.atom_mapping.get(str(node), ("Unknown", "Unknown", "Unknown", "Unknown"))
        return {
            "res_name": res_name,
            "atom_name": atom_name,
            "res_num": res_num,
            "chain_id": chain_id,
        }

    def compute_shortest_paths(self, all_paths_file, heatmap_file=None):
        nodes = sorted(list(self.G.nodes()))
        shortest_paths, path_lengths = self._compute_all_shortest_paths(nodes)
        self._save_paths(all_paths_file, shortest_paths, path_lengths)
        self.save_paths_and_create_heatmap(path_lengths, heatmap_file, title=f"All-pairs shortest-path lengths ({len(nodes)} × {len(nodes)})", cbar_label="Weighted shortest-path length")

    def _compute_all_shortest_paths(self, nodes):
        start_time = time.time()
        shortest_paths = {}
        path_lengths = {}

        for source, (distances, paths) in nx.all_pairs_dijkstra(self.G, weight="weight"):
            for target in nodes:
                if target > source and target in paths:
                    shortest_paths[(source, target)] = paths[target]
                    path_lengths[(source, target)] = distances[target]

        end_time = time.time()
        print(f" 📐  Shortest paths computation completed in {end_time - start_time:.2f} seconds")
        return shortest_paths, path_lengths

    def _save_paths(self, all_paths_file, shortest_paths, path_lengths):
        all_paths = []
        covered_edges = set()
        sorted_path_lengths = sorted(
            path_lengths.items(), key=lambda item: item[1], reverse=True
        )

        for (source, target), length in sorted_path_lengths:
            path = shortest_paths[(source, target)]
            if not path:
                continue

            path_edges = {
                tuple(sorted((node1, node2)))
                for node1, node2 in zip(path, path[1:])
            }
            if not path_edges - covered_edges:
                continue

            all_paths.append({
                "source": int(source),
                "target": int(target),
                "path": [int(node) for node in path],
                "length": float(length),
            })
            covered_edges.update(path_edges)

            if len(all_paths) == 50:
                break

        with open(all_paths_file, 'w') as file:
            json.dump({"paths": all_paths}, file)

    def calculate_shortest_path_between_residues(self, residue1, residue2):
        try:
            length, path = nx.single_source_dijkstra(self.G, residue1, target=residue2)
            return length, path
        except nx.NetworkXNoPath:
            return float('inf'), []

    def save_paths_and_create_heatmap(self, shortest_path_lengths, heatmap_file, title, cbar_label):
        """
        Create an N × N heatmap of all-pairs shortest-path lengths.

        Args:
            shortest_path_lengths (dict): Maps ``(source, target)`` node pairs
                to their weighted shortest-path length.
            heatmap_file (str): Path to the file where the heatmap will be saved.
            title (str): Title of the heatmap.
            cbar_label (str): Label for the color bar.
        """
        start_time = time.time()

        nodes = sorted(self.G.nodes())
        num_nodes = len(nodes)
        node_positions = {node: position for position, node in enumerate(nodes)}
        data_matrix = np.full((num_nodes, num_nodes), np.nan, dtype=float)
        np.fill_diagonal(data_matrix, 0.0)

        for (source, target), length in shortest_path_lengths.items():
            source_position = node_positions[source]
            target_position = node_positions[target]
            data_matrix[source_position, target_position] = length
            data_matrix[target_position, source_position] = length

        colormap = plt.colormaps["viridis"].copy()
        colormap.set_bad("#d9d9d9")
        figure, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
        image = axis.imshow(
            data_matrix,
            cmap=colormap,
            interpolation="nearest",
            origin="upper",
            aspect="equal",
        )
        colorbar = figure.colorbar(image, ax=axis)
        colorbar.set_label(cbar_label)

        legend_handles = []
        furthest_rows = []
        furthest_columns = []
        for row_position, row in enumerate(data_matrix):
            reachable_positions = np.flatnonzero(np.isfinite(row))
            reachable_positions = reachable_positions[
                reachable_positions != row_position
            ]
            if reachable_positions.size:
                furthest_position = reachable_positions[
                    np.argmax(row[reachable_positions])
                ]
                furthest_rows.append(row_position)
                furthest_columns.append(furthest_position)

        if furthest_rows:
            furthest_marker = axis.scatter(
                furthest_columns,
                furthest_rows,
                marker="x",
                s=90,
                linewidths=2,
                color="#ff2d55",
                label="Furthest node in each row",
            )
            legend_handles.append(furthest_marker)

        if num_nodes:
            tick_count = min(num_nodes, 12)
            tick_positions = np.unique(
                np.linspace(0, num_nodes - 1, tick_count, dtype=int)
            )
            tick_labels = [str(nodes[position]) for position in tick_positions]
            axis.set_xticks(tick_positions, labels=tick_labels, rotation=45, ha="right")
            axis.set_yticks(tick_positions, labels=tick_labels)

        axis.set_title(title)
        axis.set_xlabel("Target node")
        axis.set_ylabel("Source node")
        if np.isnan(data_matrix).any():
            legend_handles.append(Patch(facecolor="#d9d9d9", label="No path"))
        if legend_handles:
            axis.legend(
                handles=legend_handles,
                loc="upper left",
                bbox_to_anchor=(1.01, 0),
                borderaxespad=0,
            )

        figure.savefig(heatmap_file, dpi=300)
        plt.close(figure)
        end_time = time.time()
        print(
            f" 📊  Shortest-path heatmap saved to {heatmap_file} "
            f"in {end_time - start_time:.2f} seconds"
        )

    def generate_paths_chunk(self, start_nodes, source_node, target_node):
        """
        Separate method for path generation to enable pickling

        Args:
            start_nodes (list): Nodes to start path generation from
            source_node (int): Source node
            target_node (int): Target node

        Returns:
            list: Paths found in this chunk
        """
        chunk_paths = []
        for start_node in start_nodes:
            try:
                paths = list(nx.all_simple_paths(
                    self.G,
                    source=int(source_node),
                    target=int(target_node),
                    cutoff=None
                ))
                chunk_paths.extend(paths)
            except nx.NetworkXNoPath:
                continue
        return chunk_paths

    def find_alternative_paths(self, source_residue, target_residue,
                               alt_paths_file, k=2):
        """
        Finds the top k alternative paths using Yen's algorithm.

        Args:
            source_residue (str): The residue identifier for the source node.
            target_residue (str): The residue identifier for the target node.
            alt_paths_file (str): File to store alternative paths.
            k (int): Number of alternative paths to find.

        Returns:
            list: A list of alternative paths.
        """
        source_res_num, source_chain_id = source_residue.split(":")
        target_res_num, target_chain_id = target_residue.split(":")

        node1 = None
        node2 = None
        for key, values in self.atom_mapping.items():
            if values[-2] == int(source_res_num) and values[
                -1] == source_chain_id:
                node1 = key
            if values[-2] == int(target_res_num) and values[
                -1] == target_chain_id:
                node2 = key
            if node1 is not None and node2 is not None:
                break
        node1, node2 = int(source_res_num), int(target_res_num)

        def find_yen_k_paths():
            """Implementation of Yen's k shortest paths algorithm."""
            A = []
            B = []

            try:
                path = nx.shortest_path(self.G, node1, node2, weight='weight')
                A.append(path)
            except nx.NetworkXNoPath:
                return []

            for _ in range(1, k):
                prev_path = A[-1]

                for i in range(len(prev_path) - 1):
                    spur_node = prev_path[i]
                    root_path = prev_path[:i + 1]

                    removed_edges = []

                    for path in A:
                        if len(path) > i and path[:i + 1] == root_path:
                            u, v = path[i], path[i + 1]
                            if self.G.has_edge(u, v):
                                removed_edges.append(
                                    (u, v, self.G[u][v].copy()))
                                self.G.remove_edge(u, v)

                    try:
                        spur_path = nx.shortest_path(self.G, spur_node, node2,
                                                     weight='weight')
                        total_path = root_path[:-1] + spur_path
                        if total_path not in B:
                            heapq.heappush(B, total_path)
                    except nx.NetworkXNoPath:
                        pass

                    for u, v, data in removed_edges:
                        self.G.add_edge(u, v, **data)

                if not B:
                    break

                new_path = heapq.heappop(B)
                A.append(new_path)

            return A

        paths = find_yen_k_paths()

        def map_path(path):
            mapped = []
            for node in path:
                info = self._atom_info(node)
                mapped.append({
                    "res_num": info["res_num"],
                    "chain_id": info["chain_id"],
                })
            return mapped

        payload = {
            "source_residue": source_residue,
            "target_residue": target_residue,
            "shortest_path": map_path(paths[0]) if paths else [],
            "alternative_paths": [map_path(p) for p in paths[1:k + 1]],
        }
        with open(alt_paths_file, 'w') as f:
            json.dump(payload, f, indent=2)

        return paths[:k + 1]

    def calculate_centralities(self):
        """
        Calculates various centrality measures for the graph.

        Returns:
            tuple: A tuple containing dictionaries for betweenness, closeness, and degree centralities.
        """
        betweenness = nx.betweenness_centrality(self.G, weight='weight')
        closeness = nx.closeness_centrality(self.G, distance='weight')
        degree = dict(self.G.degree())
        return betweenness, closeness, degree

    def save_centrality_measures(self, centralities, output_file):
        """
        Saves centrality measures to a JSON file.

        Args:
            centralities (tuple): A tuple containing dictionaries for betweenness, closeness, and degree centralities.
            output_file (str): Path to the file where centrality measures will be saved.
        """
        betweenness, closeness, degree = centralities
        nodes = []
        for node in betweenness.keys():
            info = self._atom_info(node)
            nodes.append({
                "node": int(node),
                "res_num": info["res_num"],
                "chain_id": info["chain_id"],
                "betweenness": float(betweenness[node]),
                "closeness": float(closeness[node]),
                "degree": int(degree[node]),
            })
        with open(output_file, 'w') as f:
            json.dump({"nodes": nodes}, f, indent=2)

    def calculate_edge_betweenness(self):
        return nx.edge_betweenness_centrality(self.G, weight='weight', normalized=True)

    def save_edge_betweenness(self, edge_betweenness, output_file):
        """
        Saves edge betweenness centrality measures to a JSON file.

        Args:
            edge_betweenness (dict): A dictionary mapping edges to their betweenness centrality value.
            output_file (str): Path to the file where edge betweenness centralities will be saved.
        """
        edges = []
        for edge, centrality in edge_betweenness.items():
            info0 = self._atom_info(edge[0])
            info1 = self._atom_info(edge[1])
            edges.append({
                "res_num1": info0["res_num"],
                "chain_id1": info0["chain_id"],
                "res_num2": info1["res_num"],
                "chain_id2": info1["chain_id"],
                "betweenness": float(centrality),
            })
        with open(output_file, 'w') as f:
            json.dump({"edges": edges}, f, indent=2)

    def save_interchain_edge_betweenness(
        self, edge_betweenness, output_file
    ):
        """Save cross-chain edge betweenness for structures with two chains."""
        chain_ids = {
            self._atom_info(node)["chain_id"] for node in self.G.nodes()
        }
        if len(chain_ids) != 2:
            return False

        edges = []
        for edge, centrality in edge_betweenness.items():
            info0 = self._atom_info(edge[0])
            info1 = self._atom_info(edge[1])
            if (
                centrality == 0
                or info0["chain_id"] == info1["chain_id"]
            ):
                continue
            edges.append({
                "res_num1": info0["res_num"],
                "chain_id1": info0["chain_id"],
                "res_num2": info1["res_num"],
                "chain_id2": info1["chain_id"],
                "betweenness": float(centrality),
            })

        with open(output_file, 'w') as f:
            json.dump({"edges": edges}, f, indent=2)
        return True

    def save_graph_matrix_plots(
        self,
        edge_betweenness,
        adjacency_file,
        edge_weights_file,
        edge_betweenness_file,
    ):
        """Save N × N plots of graph adjacency, edge cost, and betweenness."""
        nodes = sorted(self.G.nodes())
        node_positions = {node: position for position, node in enumerate(nodes)}
        shape = (len(nodes), len(nodes))
        adjacency = np.zeros(shape, dtype=np.uint8)
        edge_weights = np.full(shape, np.nan, dtype=float)
        betweenness = np.full(shape, np.nan, dtype=float)

        for source, target, edge_data in self.G.edges(data=True):
            source_position = node_positions[source]
            target_position = node_positions[target]
            weight = float(edge_data.get("weight", 1.0))
            centrality = edge_betweenness.get(
                (source, target),
                edge_betweenness.get((target, source), 0.0),
            )

            adjacency[source_position, target_position] = 1
            adjacency[target_position, source_position] = 1
            edge_weights[source_position, target_position] = weight
            edge_weights[target_position, source_position] = weight
            betweenness[source_position, target_position] = centrality
            betweenness[target_position, source_position] = centrality

        self._save_node_pair_matrix(
            adjacency,
            nodes,
            adjacency_file,
            title=f"Graph adjacency ({len(nodes)} × {len(nodes)})",
            cbar_label="Edge present",
            cmap="Greys",
            vmin=0,
            vmax=1,
            colorbar_ticks=[0, 1],
        )
        self._save_node_pair_matrix(
            edge_weights,
            nodes,
            edge_weights_file,
            title=f"Graph edge weights ({len(nodes)} × {len(nodes)})",
            cbar_label="Edge cost: −log(coupling)",
            cmap="plasma",
        )

        finite_betweenness = betweenness[np.isfinite(betweenness)]
        max_betweenness = (
            float(np.max(finite_betweenness))
            if finite_betweenness.size
            else 0.0
        )
        betweenness_norm = (
            PowerNorm(gamma=0.35, vmin=0.0, vmax=max_betweenness)
            if max_betweenness > 0
            else None
        )
        self._save_node_pair_matrix(
            betweenness,
            nodes,
            edge_betweenness_file,
            title=f"Edge betweenness centrality ({len(nodes)} × {len(nodes)})",
            cbar_label="Edge betweenness",
            cmap="viridis",
            norm=betweenness_norm,
        )

    @staticmethod
    def _save_node_pair_matrix(
        matrix,
        nodes,
        output_file,
        title,
        cbar_label,
        cmap,
        norm=None,
        vmin=None,
        vmax=None,
        colorbar_ticks=None,
    ):
        """Save one node-pair matrix with readable labels for large graphs."""
        colormap = plt.colormaps[cmap].copy()
        colormap.set_bad("#f2f2f2")
        figure, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
        image_options = {
            "cmap": colormap,
            "interpolation": "nearest",
            "origin": "upper",
            "aspect": "equal",
        }
        if norm is not None:
            image_options["norm"] = norm
        else:
            image_options["vmin"] = vmin
            image_options["vmax"] = vmax
        image = axis.imshow(matrix, **image_options)
        colorbar = figure.colorbar(image, ax=axis, ticks=colorbar_ticks)
        colorbar.set_label(cbar_label)

        if nodes:
            tick_count = min(len(nodes), 12)
            tick_positions = np.unique(
                np.linspace(0, len(nodes) - 1, tick_count, dtype=int)
            )
            tick_labels = [str(nodes[position]) for position in tick_positions]
            axis.set_xticks(
                tick_positions, labels=tick_labels, rotation=45, ha="right"
            )
            axis.set_yticks(tick_positions, labels=tick_labels)

        axis.set_title(title)
        axis.set_xlabel("Target node")
        axis.set_ylabel("Source node")
        figure.savefig(output_file, dpi=300)
        plt.close(figure)
        print(f" 📊  Graph matrix plot saved to {output_file}")

    def identify_top_10_percent_nodes(self, centralities, output_file):
        """
        Identifies the top 5% nodes based on centrality measures and saves them as allosteric hotspots.

        Args:
            centralities (tuple): A tuple containing dictionaries for betweenness, closeness, and degree centralities.
            output_file (str): Path to the file where top nodes will be saved.
        """
        betweenness, closeness, degree = centralities
        num_nodes = len(betweenness)
        top_n = max(1, num_nodes // 20)
        sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1],
                                    reverse=True)[:top_n]
        sorted_closeness = sorted(closeness.items(), key=lambda x: x[1],
                                  reverse=True)[:top_n]
        sorted_degree = sorted(degree.items(), key=lambda x: x[1],
                               reverse=True)[:top_n]
        top_nodes = set([node for node, _ in sorted_betweenness] +
                        [node for node, _ in sorted_closeness] +
                        [node for node, _ in sorted_degree])

        nodes_out = []
        for node in top_nodes:
            info = self._atom_info(node)
            nodes_out.append({
                "node": int(node),
                "res_num": info["res_num"],
                "chain_id": info["chain_id"],
            })

        with open(output_file, 'w') as f:
            json.dump({
                "description": "Top 5% Nodes (Allosteric Hotspots)",
                "nodes": nodes_out,
            }, f, indent=2)

        print(
            f" 📥  Top 5% nodes identified and saved as allosteric hotspots to {output_file} ")
