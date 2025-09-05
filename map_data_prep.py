# map_data_prep.py

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def get_map_data():
    """
    Defines locations and a dummy cost/distance matrix.
    In a real scenario, this would involve calling a mapping API.
    """
    N = 4 # N = number of nodes (0=Depot, 1=A, 2=B, 3=C)

    # Dummy Cost/Distance Matrix (e.g., travel time in minutes)
    # C[i][j] = cost to go from i to j
    C = np.array([
        [0, 10, 20, 15],  # From Depot (0) to 0, A, B, C
        [10, 0, 5, 12],   # From A (1) to Depot, A, B, C
        [20, 5, 0, 8],    # From B (2) to Depot, A, B, C
        [15, 12, 8, 0]    # From C (3) to Depot, A, B, C
    ])

    # For visualization: define approximate geographic positions
    positions = {
        0: (0, 0),    # Depot
        1: (1, 2),    # Customer A
        2: (3, 1),    # Customer B
        3: (2, -1)    # Customer C
    }

    return N, C, positions

def visualize_network(N, C, positions):
    """
    Visualizes the initial network with travel costs.
    """
    G = nx.DiGraph()
    for i in range(N):
        for j in range(N):
            if i != j:
                G.add_edge(i, j, weight=C[i,j])

    plt.figure(figsize=(7, 6))
    nx.draw_networkx_nodes(G, positions, node_color='lightblue', node_size=700)
    nx.draw_networkx_labels(G, positions, labels={i: str(i) for i in range(N)}, font_size=12)
    nx.draw_networkx_edges(G, positions, arrowstyle='->', arrowsize=20, edge_color='gray', width=1)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, positions, edge_labels=edge_labels, font_color='red')
    plt.title("Sample Map with Travel Costs")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    N_nodes, cost_matrix, node_positions = get_map_data()
    print("Map Data Prepared.")
    print(f"Number of nodes: {N_nodes}")
    print("Cost Matrix:\n", cost_matrix)
    visualize_network(N_nodes, cost_matrix, node_positions)