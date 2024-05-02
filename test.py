import numpy as np
from scipy.sparse.csgraph import connected_components
from numpy.linalg import matrix_power
import matplotlib.pyplot as plt
import networkx as nx

# Define the transition matrix P for a Markov Chain with 4 states
P = np.array([
    [0.06, 0.41, 0.53, 0.00],
    [0.07, 0.42, 0.47, 0.04],
    [0.08, 0.40, 0.47, 0.05],
    [0.40, 0.40, 0.10, 0.10]
])

# Normalize P to ensure each row sums to 1
P = P / P.sum(axis=1, keepdims=True)

# Check if the Markov chain is ergodic
def is_ergodic(P):
    # Check irreducibility
    n = P.shape[0]
    graph = (P > 0).astype(int)
    n_components, labels = connected_components(csgraph=graph, directed=True, connection='strong')
    if n_components != 1:
        return False  # Not irreducible if there are isolated subgraphs
    
    # Check aperiodicity
    periods = set()
    for i in range(n):
        for j in range(1, n+1):  # Check matrix powers from 1 to n
            if matrix_power(P, j)[i, i] > 0:
                periods.add(j)
    
    # Calculate the gcd of the cycle lengths
    from math import gcd
    from functools import reduce
    overall_gcd = reduce(gcd, periods)
    return overall_gcd == 1

# calculate the steady state probability distribution using the eigenvalue method
def find_steady_state_eigen(P):
    # First check if the matrix is ergodic
    if not is_ergodic(P):
        raise ValueError("The Markov chain is not ergodic.")
    
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    index = np.argmin(np.abs(eigenvalues - 1))
    steady_state = np.abs(eigenvectors[:, index].real)
    steady_state /= steady_state.sum()
    return steady_state

# visualize the markov chain
def visualize_markov_chain(P, save_path=None):
    G = nx.DiGraph()
    n = P.shape[0]
    labels = {}
    for i in range(n):
        for j in range(n):
            if P[i, j] > 0:
                G.add_edge(i, j, weight=P[i, j])
                labels[(i, j)] = f"{P[i, j]:.2f}"
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

try:
    steady_state_eigen = find_steady_state_eigen(P)
    print("Normalized Steady State Probability Distribution (Eigen):")
    print(steady_state_eigen)
    visualize_markov_chain(P, save_path='/home/moyed/or2/markov_chain_plot.png')
    
except ValueError as e:
    print(e)
