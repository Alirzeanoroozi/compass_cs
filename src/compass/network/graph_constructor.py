import json
import matplotlib.pyplot as plt
import networkx as nx

from compass.network.read_files import read_matrix

class GraphConstructor:
    def __init__(self, distance_file, adjacency_file, distance_cutoffs):
        self.distance_file = distance_file
        self.adjacency_file = adjacency_file
        self.distance_cutoffs = distance_cutoffs

    def build_graph_from_matrices(self, distance_cutoff):
        min_dist_matrix = read_matrix(self.distance_file)
        adjacency_matrix = read_matrix(self.adjacency_file)

        G = nx.Graph()
        num_nodes = len(min_dist_matrix)

        for i in range(num_nodes):
            G.add_node(i)

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if min_dist_matrix[i, j] < float(distance_cutoff) and adjacency_matrix[i, j] > 0:
                    G.add_edge(i, j, weight=adjacency_matrix[i, j])
        return G

    def save_graph_and_mapping(self, G, atom_mapping, output_file):
        data = {'graph': nx.readwrite.json_graph.node_link_data(G), 'atom_mapping': atom_mapping}
        with open(output_file, 'w') as f:
            json.dump(data, f)
        print(f" 📥  Graph and atom mapping saved to {output_file}")

    def plot_and_save_histogram(self, G, output_file_prefix):
        weights = []
        for u, v, data in G.edges(data=True):
            try:
                weights.append(data['weight'])
            except KeyError:
                weights.append(0)

        plt.figure(figsize=(10, 6))
        plt.hist(weights, bins=10, edgecolor='black', alpha=0.7)
        plt.title('Histogram of Edge Weights')
        plt.xlabel('Weight')
        plt.ylabel('Frequency')

        histogram_file = f"{output_file_prefix}_histogram.png"
        plt.savefig(histogram_file)
        plt.close()
        print(f" 📈  Histogram of edge weights saved to {histogram_file}")
