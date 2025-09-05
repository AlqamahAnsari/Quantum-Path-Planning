# quantum_solver.py

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Corrected imports for Qiskit algorithms
from qiskit.algorithms.minimum_eigensolvers import QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit import BasicAer
from qiskit_optimization.algorithms import MinimumEigenOptimizer, GurobiOptimizer # GurobiOptimizer is in qiskit_optimization

# Import functions from our other files
from map_data_prep import get_map_data, visualize_network
from qubo_formulation import create_path_planning_qubo

def solve_with_qaoa(qp_problem):
    """
    Solves the given QuadraticProgram using Qiskit's QAOA algorithm on a simulator.

    Args:
        qp_problem (QuadraticProgram): The QUBO problem to solve.

    Returns:
        qiskit_optimization.algorithms.OptimizationResult: The solution found by QAOA.
    """
    # Set up the quantum instance (simulator)
    algorithm_globals.random_seed = 123
    quantum_instance = QuantumInstance(
        BasicAer.get_backend('qasm_simulator'),
        seed_simulator=algorithm_globals.random_seed,
        seed_transpiler=algorithm_globals.random_seed,
        shots=1024 # Number of times to run the circuit
    )

    # Choose an optimizer for QAOA
    optimizer = COBYLA(maxiter=100) # COBYLA is good for smaller problems

    # Create a QAOA instance
    # 'reps' is the QAOA depth, higher depth can lead to better solutions but more complex circuits
    qaoa_mes = QAOA(optimizer=optimizer, reps=1, quantum_instance=quantum_instance)

    # Create MinimumEigenOptimizer to integrate QAOA with QuadraticProgram
    qaoa_optimizer = MinimumEigenOptimizer(qaoa_mes)

    # Solve the problem
    result = qaoa_optimizer.solve(qp_problem)
    return result

def decode_and_calculate_cost(result, x_vars, N_nodes, cost_matrix):
    """
    Decodes the QAOA result into an optimal path and calculates its total cost.
    """
    binary_solution = result.x
    
    path = []
    num_steps = N_nodes # Assuming path has N_nodes steps (e.g., 0->1->2->0 is 4 steps if indexed 0 to 3)
    
    # Reconstruct the path from the binary solution
    for p in range(num_steps):
        found_node_at_step = False
        for i in range(N_nodes):
            if round(x_vars[(i, p)].evaluate(binary_solution)) == 1:
                path.append(i)
                found_node_at_step = True
                break
        if not found_node_at_step:
            print(f"Warning: No node found at step {p}. Solution might be invalid.")

    # Calculate the total cost of the found path
    total_cost = 0
    if len(path) > 1:
        for k in range(len(path) - 1):
            total_cost += cost_matrix[path[k], path[k+1]]
        # Add the cost for the final return to the depot if the path doesn't automatically close in cost calc
        # This part depends on how the objective function was precisely built
        # Given our current obj. function, the C[i,0] * x[i, N-1] * x[0,0] term implies this is covered.
        # But for safety, let's explicitly check and add if the path is not a cycle in a direct sense.
        if path[-1] != path[0]:
             total_cost += cost_matrix[path[-1], path[0]] # This should ideally be zero if constraints work perfectly
    
    return path, total_cost

def visualize_optimized_path(N_nodes, cost_matrix, positions, path, total_cost):
    """
    Visualizes the optimized path on the network.
    """
    G = nx.DiGraph()
    for i in range(N_nodes):
        for j in range(N_nodes):
            if i != j:
                G.add_edge(i, j, weight=cost_matrix[i,j])

    plt.figure(figsize=(7, 6))
    nx.draw_networkx_nodes(G, positions, node_color='lightblue', node_size=700)
    nx.draw_networkx_labels(G, positions, labels={i: str(i) for i in range(N_nodes)}, font_size=12)

    # Draw the optimized path
    path_edges = []
    for k in range(len(path) - 1):
        path_edges.append((path[k], path[k+1]))
    # Add the closing edge if the path is a cycle (from last node back to first)
    if path and path[-1] != path[0]:
        path_edges.append((path[-1], path[0]))

    nx.draw_networkx_edges(G, positions, edgelist=path_edges, edge_color='green', width=3, arrowstyle='->', arrowsize=25)
    
    # Only show labels for edges in the optimized path
    optimized_edge_labels = {(u,v): cost_matrix[u,v] for u,v in path_edges}
    nx.draw_networkx_edge_labels(G, positions, edge_labels=optimized_edge_labels, font_color='darkgreen', font_size=10)

    plt.title(f"Optimized Path: {path} (Cost: {total_cost})")
    plt.axis('off')
    plt.show()

def main():
    # 1. Get map data
    N_nodes, cost_matrix, node_positions = get_map_data()
    print("Map Data Prepared.")
    visualize_network(N_nodes, cost_matrix, node_positions)

    # 2. Formulate QUBO
    qp_problem, x_vars = create_path_planning_qubo(N_nodes, cost_matrix)
    print("\n--- QUBO Model Generated ---")
    print(f"Number of binary variables: {qp_problem.get_num_vars()}")
    print(f"Number of constraints: {qp_problem.get_num_linear_constraints()}")
    # print(qp_problem.prettyprint()) # Uncomment for full QUBO printout

    # 3. Solve with Qiskit's QAOA
    print("\n--- Solving with QAOA ---")
    qaoa_result = solve_with_qaoa(qp_problem)
    print("QAOA Solution Result:", qaoa_result)

    # 4. Decode and visualize the path
    optimized_path, total_cost = decode_and_calculate_cost(qaoa_result, x_vars, N_nodes, cost_matrix)
    print(f"\nOptimal Path (QAOA): {optimized_path}")
    print(f"Total Cost of Path: {total_cost}")

    visualize_optimized_path(N_nodes, cost_matrix, node_positions, optimized_path, total_cost)

    # --- Optional: Compare with a classical exact solver (e.g., Gurobi) ---
    try:
        gurobi_optimizer = GurobiOptimizer()
        gurobi_result = gurobi_optimizer.solve(qp_problem)
        classical_path, classical_total_cost = decode_and_calculate_cost(gurobi_result, x_vars, N_nodes, cost_matrix)
        print("\n--- Classical (Gurobi) Solution ---")
        print(f"Classical Optimal Path: {classical_path}")
        print(f"Classical Total Cost: {classical_total_cost}")
        # visualize_optimized_path(N_nodes, cost_matrix, node_positions, classical_path, classical_total_cost) # Uncomment to see classical path
    except Exception as e:
        print(f"\nCould not run Gurobi (likely not installed or configured): {e}")


if __name__ == "__main__":
    main()