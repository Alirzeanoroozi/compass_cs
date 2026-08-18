import heapq
import json
import time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

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

    def compute_shortest_paths(self, all_paths_file, top_file):
        """
        Compute shortest paths between nodes until reaching 20% of total residues.

        The method uses an optimized sequential approach that:
        1. Computes shortest paths between all node pairs
        2. Sorts paths by length
        3. Collects paths until reaching the residue threshold
        4. Saves results to specified files
        """
        nodes = sorted(list(self.G.nodes()))
        shortest_paths, path_lengths = self._compute_all_shortest_paths(nodes)
        collected_paths = self._collect_paths_until_threshold(nodes, shortest_paths, path_lengths)
        self._save_paths(all_paths_file, shortest_paths)
        self._write_top_paths_with_mapping(collected_paths, shortest_paths, top_file)

        return path_lengths

    def _compute_all_shortest_paths(self, nodes):
        start_time = time.time()
        shortest_paths = {}
        path_lengths = {}

        for source in nodes:
            try:
                distances, paths = nx.single_source_dijkstra(self.G, source, weight='weight')
                for target in (n for n in nodes if n > source):
                    if target in paths:
                        shortest_paths[(source, target)] = paths[target]
                        path_lengths[(source, target)] = distances[target]
            except nx.NetworkXNoPath:
                continue

        end_time = time.time()
        print(f" 📐  Shortest paths computation completed in {end_time - start_time:.2f} seconds")
        return shortest_paths, path_lengths

    def _collect_paths_until_threshold(self, nodes, shortest_paths, path_lengths):
        total_residues = len(nodes)
        residue_threshold = 0.2 * total_residues

        path_list = [(source, target, length) for (source, target), length in path_lengths.items()]
        path_list.sort(key=lambda x: x[2], reverse=True)

        collected_paths = []
        unique_residues = set()

        for source, target, length in path_list:
            path = shortest_paths.get((source, target))
            if path is None:
                continue

            collected_paths.append((source, target, length))
            unique_residues.update(path)

            if len(unique_residues) >= residue_threshold:
                break

        return collected_paths

    def _save_paths(self, all_paths_file, shortest_paths):
        all_paths = []
        for (source, target), path in shortest_paths.items():
            if not path:
                continue
            mapped_path = []
            for node in path:
                info = self._atom_info(node)
                mapped_path.append({"res_num": info["res_num"], "chain_id": info["chain_id"]})
            all_paths.append({"source": int(source), "target": int(target), "path": [int(n) for n in path], "mapped_path": mapped_path})

        with open(all_paths_file, 'w') as file:
            json.dump({"paths": all_paths}, file)

    def _write_top_paths_with_mapping(self, top_paths, shortest_paths, top_file):
        unique_top_paths = set()
        paths_out = []

        for source, target, length in top_paths:
            if source > target:
                source, target = target, source

            if (source, target) in unique_top_paths:
                continue
            try:
                path = shortest_paths.get((source, target))
                if not path:
                    continue

                mapped_path = []
                for node in path:
                    info = self._atom_info(node)
                    mapped_path.append({"res_num": info["res_num"], "chain_id": info["chain_id"]})
                paths_out.append({"source": int(source), "target": int(target), "length": float(length), "path": [int(n) for n in path], "mapped_path": mapped_path})
                unique_top_paths.add((source, target))
            except Exception as e:
                print(f"Error processing path {source} -> {target}: {str(e)}")

        with open(top_file, 'w') as file:
            json.dump({"description": "Top shortest paths", "paths": paths_out}, file, indent=2)

        print(f" 📥  Top 50 shortest paths with node and residue mapping written to {top_file}")

    def calculate_shortest_path_between_residues(self, residue1, residue2):
        try:
            length, path = nx.single_source_dijkstra(self.G, residue1, target=residue2)
            return length, path
        except nx.NetworkXNoPath:
            return float('inf'), []

    def save_paths_and_create_heatmap(self, shortest_path_lengths, heatmap_file, title, cbar_label):
        """
        Creates a heatmap of the shortest paths.

        Args:
            shortest_path_lengths (dict): A dictionary of shortest path lengths.
            heatmap_file (str): Path to the file where the heatmap will be saved.
            title (str): Title of the heatmap.
            cbar_label (str): Label for the color bar.
        """
        start_time = time.time()

        num_nodes = len(self.G.nodes())
        data_matrix = np.zeros((num_nodes, num_nodes))

        for source, paths in shortest_path_lengths.items():
            for target, length in paths.items():
                data_matrix[int(source)][int(target)] = length
                data_matrix[int(target)][int(source)] = length

        data_matrix[np.isinf(data_matrix)] = 0
        vmin = np.min(data_matrix)
        vmax = np.max(data_matrix)
        plt.figure(figsize=(10, 8))
        sns.heatmap(data_matrix, annot=False, fmt=".2f", cmap="viridis", cbar_kws={'label': cbar_label}, vmin=vmin, vmax=vmax)
        plt.title(title)
        plt.xlabel("Node Index")
        plt.ylabel("Node Index")
        plt.savefig(heatmap_file)
        plt.close()
        end_time = time.time()
        print(f"Heatmap created and saved in {end_time - start_time:.2f} seconds")

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
        """
        Calculates edge betweenness centrality for the graph.

        Returns:
            dict: A dictionary mapping edges to their betweenness centrality value.
        """
        edge_betweenness = nx.edge_betweenness_centrality(self.G,
                                                          weight='weight')
        return edge_betweenness

    def save_edge_betweenness(self, edge_betweenness, output_file):
        """
        Saves edge betweenness centrality measures to a JSON file.

        Args:
            edge_betweenness (dict): A dictionary mapping edges to their betweenness centrality value.
            output_file (str): Path to the file where edge betweenness centralities will be saved.
        """
        edges = []
        for edge, centrality in edge_betweenness.items():
            if self.G.has_edge(*edge):
                info0 = self._atom_info(edge[0])
                info1 = self._atom_info(edge[1])
                edges.append({
                    "node1": int(edge[0]),
                    "node2": int(edge[1]),
                    "res_num1": info0["res_num"],
                    "chain_id1": info0["chain_id"],
                    "res_num2": info1["res_num"],
                    "chain_id2": info1["chain_id"],
                    "betweenness": float(centrality),
                })
        with open(output_file, 'w') as f:
            json.dump({"edges": edges}, f, indent=2)

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
