import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap


class NetworkGenerator:
    def __init__(self, num_nodes, num_core=5):
        """
        Initializes the NetworkGenerator class.

        Parameters:
        - num_nodes: Number of nodes (entities) in the network.
        - num_core: Number of core nodes in the network.
        """
        self.num_nodes = num_nodes
        self.num_core = num_core
        self.core_nodes = np.arange(num_core)
        self.periphery_nodes = np.arange(num_core, num_nodes)

    def er_network(self, p=0.4, max_liability=20):
        """
        Generates an E-R random liability network and calculates external assets.

        Parameters:
        - p: Probability of a directed edge (v_i, v_j) being present, default is 0.1.
        - max_liability: Maximum value for liability (l_{ij}), sampled from U(0, max_liability).

        Returns:
        - adj_matrix: Adjacency matrix of the generated network.
        - liabs: 2D array representing liabilities (l_{ij}) between entities.
        - ext_a: 1D array of external assets (e_i) for each entity.
        """

        adj_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=int)
        liabs = np.zeros((self.num_nodes, self.num_nodes), dtype=float)

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j and np.random.rand() < p:
                    adj_matrix[i, j] = 1
                    liabs[i, j] = np.random.uniform(0, max_liability)

        ext_a = np.zeros(self.num_nodes)

        for i in range(self.num_nodes):
            total_inter_liabs = np.sum(liabs[i, :])
            total_liabs = total_inter_liabs / 0.6
            total_ext_liabs = total_liabs - total_inter_liabs
            total_a = total_liabs * np.random.uniform(0.7, 0.9)
            inter_a = np.sum(liabs[:, i])
            temp_ext_a = total_a - inter_a
            #             Net external assets
            ext_a[i] = max(0, temp_ext_a - total_ext_liabs)

        return adj_matrix, liabs, ext_a

    def core_periphery_network(self):
        """
        Generates a core-periphery random network according to specified probabilities and uniform distributions.

        Returns:
        - adj_matrix: Adjacency matrix of the generated network.
        - liabs: 2D array representing liabilities (l_{ij}) between entities.
        - ext_a: 1D array of external assets (e_i) for each entity.
        """

        adj_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=int)
        liabs = np.zeros((self.num_nodes, self.num_nodes), dtype=float)

        prob_cc = 0.6
        prob_pc = 0.2
        prob_cp = 0.125
        prob_pp = 0.075

        range_cc = (60, 100)
        range_pc = (20, 40)
        range_cp = (20, 40)
        range_pp = (10, 20)

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    if i in self.core_nodes and j in self.core_nodes:
                        if np.random.rand() < prob_cc:
                            adj_matrix[i, j] = 1
                            liabs[i, j] = np.random.uniform(*range_cc)
                    elif i in self.periphery_nodes and j in self.core_nodes:
                        if np.random.rand() < prob_pc:
                            adj_matrix[i, j] = 1
                            liabs[i, j] = np.random.uniform(*range_pc)
                    elif i in self.core_nodes and j in self.periphery_nodes:
                        if np.random.rand() < prob_cp:
                            adj_matrix[i, j] = 1
                            liabs[i, j] = np.random.uniform(*range_cp)
                    elif i in self.periphery_nodes and j in self.periphery_nodes:
                        if np.random.rand() < prob_pp:
                            adj_matrix[i, j] = 1
                            liabs[i, j] = np.random.uniform(*range_pp)

        ext_a = np.zeros(self.num_nodes)

        for i in range(self.num_nodes):
            total_inter_liabs = np.sum(liabs[i, :])
            total_liabs = total_inter_liabs / np.random.uniform(0.6, 0.8)
            #             total_liabs=total_inter_liabs / 0.6
            total_ext_liabs = total_liabs - total_inter_liabs
            total_a = total_liabs * np.random.uniform(0.7, 1)
            inter_a = np.sum(liabs[:, i])
            temp_ext_a = total_a - inter_a
            ext_a[i] = max(0, temp_ext_a - total_ext_liabs)

        return adj_matrix, liabs, ext_a

    def binary_tree_network(self, S):
        """
        Generates a Binary Tree Network with S levels.

        Parameters:
        - S: Number of levels in the binary tree.

        Returns:
        - adj_matrix: Adjacency matrix of the generated network.
        - liabs: 2D array representing liabilities (l_{ij}) between entities.
        - ext_a: 1D array of external assets (e_i) for each entity.
        """

        # Calculate total number of nodes N in the binary tree
        N = 2 ** S - 1

        adj_matrix = np.zeros((N, N), dtype=int)
        liabs = np.zeros((N, N), dtype=float)

        # Populate the adjacency matrix and liabilities
        for i in range(N):
            left_child = 2 * i + 1
            right_child = 2 * i + 2
            if left_child < N:
                adj_matrix[i, left_child] = 1
                liabs[i, left_child] = 2 ** (S - (i // 2 + 1))
            if right_child < N:
                adj_matrix[i, right_child] = 1
                liabs[i, right_child] = 2 ** (S - (i // 2 + 1))
                ext_a = np.zeros(self.num_nodes)

        ext_a = np.zeros(N)
        #         for i in range(self.num_nodes):
        #             total_inter_liabs = np.sum(liabs[i, :])
        #             total_liabs=total_inter_liabs / np.random.uniform(0.6, 0.8)
        # #             total_liabs=total_inter_liabs / 0.6
        #             total_ext_liabs= total_liabs-total_inter_liabs
        #             total_a = total_liabs * np.random.uniform(0.7, 1)
        #             inter_a = np.sum(liabs[:, i])
        #             temp_ext_a=total_a-inter_a
        #             ext_a[i]=max(0,temp_ext_a-total_ext_liabs)

        return adj_matrix, liabs, ext_a

def visualize_network(adj_matrix, ext_a):
    """
    Visualizes a network using NetworkX.

    Parameters:
    - adj_matrix: Adjacency matrix of the network.
    - ext_a: 1D array of external assets (e_i) for each entity.
    """

    num_nodes = adj_matrix.shape[0]
    G = nx.DiGraph()

    # Add nodes
    G.add_nodes_from(range(num_nodes))

    # Add edges based on adjacency matrix
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i, j] == 1:
                G.add_edge(i, j)

    # Calculate positions for nodes with adjusted parameters
    pos = nx.spring_layout(G, scale=500, k=0.1)

    # Determine unique values of ext_a and map them to colors
    ext_a_values = ext_a.copy()
    ext_a_values.sort()
    colormap = plt.cm.get_cmap('viridis', len(ext_a_values))  # Choose colormap and number of colors

    # Normalize ext_a_values to map to colormap
    norm = Normalize(vmin=ext_a_values[0], vmax=ext_a_values[-1])

    # Create a color dictionary for nodes based on ext_a values
    color_map = {}
    for i, value in enumerate(ext_a):
        color_map[i] = colormap(norm(value))

    # Draw nodes with colors based on external assets
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color=list(color_map.values()), cmap=colormap)

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, arrows=True)

    # Draw node indices
    node_labels = {i: str(i) for i in range(num_nodes)}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_color='black')

    plt.title('Core-periphery Network Visualization')
    plt.axis('off')
    plt.show()


# Example usage:
if __name__ == "__main__":
    num_nodes = 10
    network_generator = NetworkGenerator(num_nodes, num_core=3)
    # adj_matrix, y, ext_a = network_generator.er_network()
    adj_matrix,y, ext_a = network_generator.core_periphery_network()
    visualize_network(adj_matrix, ext_a)
    print('external assets:', ext_a)
    print('liabilities matrix', y)
    print('adj_zero_one', adj_matrix)



