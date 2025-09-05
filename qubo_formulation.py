# qubo_formulation.py

from qiskit_optimization import QuadraticProgram

def create_path_planning_qubo(N, C):
    """
    Formulates the path planning problem as a QuadraticProgram (QUBO).

    Args:
        N (int): Number of nodes (including depot).
        C (np.array): NxN cost matrix where C[i][j] is cost from i to j.

    Returns:
        tuple: (QuadraticProgram, dict) where dict maps (node, step) to Qiskit variable.
    """
    num_nodes = N
    num_steps = N # Path length is N steps, e.g., 0 -> 1 -> 2 -> 0

    mod = QuadraticProgram(name="path_planning_qubo")

    # Create binary variables x_i_p (node i at step p)
    x = {}
    for i in range(num_nodes):
        for p in range(num_steps):
            x[(i, p)] = mod.binary_var(name=f'x_{i}_{p}')

    # --- Objective Function: Minimize total travel cost ---
    objective_terms = 0
    # Cost for transitions between steps 0 to N-2
    for p in range(num_steps - 1):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    objective_terms += C[i, j] * x[(i, p)] * x[(j, p+1)]

    # Cost for returning to the depot (from the last visited node at step N-1 to node 0)
    # This term works because we constrain x_0_0 and x_0_(N-1) to be 1
    for i in range(num_nodes):
        if i != 0:
            objective_terms += C[i, 0] * x[(i, num_steps-1)] * x[(0, 0)]

    mod.minimize(objective=objective_terms)

    # --- Constraints ---
    # 1. Each customer location (1 to N-1) is visited exactly once
    for i in range(1, num_nodes): # For each customer node
        linear_coeffs = {x[(i, p)]: 1 for p in range(num_steps)}
        mod.linear_constraint(linear=linear_coeffs, sense="==", rhs=1, name=f'node_{i}_visited_once')

    # 2. Exactly one location is visited at each step
    for p in range(num_steps):
        linear_coeffs = {x[(i, p)]: 1 for i in range(num_nodes)}
        mod.linear_constraint(linear=linear_coeffs, sense="==", rhs=1, name=f'step_{p}_one_node')

    # 3. Starting at Depot (node 0 at step 0)
    mod.linear_constraint(linear={x[(0, 0)]: 1}, sense="==", rhs=1, name='start_at_depot')

    # 4. Ending at Depot (node 0 at step N-1)
    mod.linear_constraint(linear={x[(0, num_steps-1)]: 1}, sense="==", rhs=1, name='end_at_depot')

    return mod, x

if __name__ == "__main__":
    from map_data_prep import get_map_data
    N_nodes, cost_matrix, _ = get_map_data()
    
    qp_problem, x_vars = create_path_planning_qubo(N_nodes, cost_matrix)
    
    print("\n--- QUBO Model Generated ---")
    print(f"Number of binary variables: {qp_problem.get_num_vars()}")
    print(f"Number of constraints: {qp_problem.get_num_linear_constraints()}")
    print(qp_problem.prettyprint())