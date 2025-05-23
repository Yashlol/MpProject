from django.shortcuts import render
from django.http import HttpResponse
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
from scipy.optimize import linprog
import requests

def graphical_method_view(request):
    if request.method == 'POST':
        try:
            optimization_type = request.POST.get('optimization_type', 'maximize')
            print("Optimization type received:", optimization_type)  # Debug print

            objective_function_input = request.POST.get('objective_function', '').strip()
            if not objective_function_input:
                raise ValueError("Objective function cannot be empty.")

            objective_coefficients = [float(x) for x in objective_function_input.split(',')]
            constraints, A, b = parse_constraints(request)

            graph_image, optimal_point, optimal_value = solve_linear_program(
                objective_coefficients, np.array(A), np.array(b), constraints, optimization_type
            )

            return render(request, 'graphical.html', {
                'optimal_point': optimal_point,
                'optimal_value': optimal_value,
                'graph_image': graph_image,
            })
        except ValueError as e:
            return render(request, 'graphical.html', {'error_message': str(e)})

    return render(request, 'graphical.html')

def parse_constraints(request):
    constraints = []
    A = []
    b = []
    i = 1
    while True:
        constraint_data = request.POST.getlist(f'constraint_{i}[]')
        if not constraint_data:
            break
        
        try:
            x1_coeff = float(constraint_data[0])
            x2_coeff = float(constraint_data[1])
            inequality_type = constraint_data[2]
            rhs_value = float(constraint_data[3])
            
            constraints.append({
                'coefficients': [x1_coeff, x2_coeff],
                'inequality_type': inequality_type,
                'rhs': rhs_value
            })
            
            A.append([x1_coeff, x2_coeff])
            b.append(rhs_value)
        except (ValueError, IndexError):
            raise ValueError(f"Invalid input in constraint {i}")
        i += 1

    # Add non-negativity constraints automatically
    constraints.append({'coefficients': [1, 0], 'inequality_type': '>=', 'rhs': 0})
    A.append([1, 0])
    b.append(0)
    constraints.append({'coefficients': [0, 1], 'inequality_type': '>=', 'rhs': 0})
    A.append([0, 1])
    b.append(0)
    
    return constraints, A, b

def solve_linear_program(c, A, b, constraints, optimization_type):
    vertices = compute_feasible_region(A, b, constraints)
    print("Computed vertices:", vertices)  # Debug print

    optimal_vertex, optimal_value = None, None

    if len(vertices) > 0:
        # Calculate the objective function value at each vertex
        z_values = np.dot(vertices, c)
        print("Objective values for vertices:", z_values)  # Debug print
        
        if optimization_type == 'maximize':
            optimal_index = np.argmax(z_values)
        else:
            optimal_index = np.argmin(z_values)
        optimal_value = z_values[optimal_index]
        optimal_vertex = vertices[optimal_index]
        print("Selected vertex:", optimal_vertex, "with objective value:", optimal_value)  # Debug print
    
    graph_image = plot_constraints(constraints, vertices, optimal_vertex)
    return graph_image, optimal_vertex.tolist() if optimal_vertex is not None else None, optimal_value

def compute_feasible_region(A, b, constraints):
    vertices = []
    num_constraints = len(A)
    tol = 1e-10  # small tolerance to allow near-zero values

    # Find intersections between every pair of constraints
    for i in range(num_constraints):
        for j in range(i + 1, num_constraints):
            A_ = np.array([A[i], A[j]])
            b_ = np.array([b[i], b[j]])
            try:
                vertex = np.linalg.solve(A_, b_)
                if all_constraints_satisfied(vertex, A, b, constraints) and (vertex >= -tol).all():
                    vertices.append(vertex)
            except np.linalg.LinAlgError:
                continue

    # Check intersections with the axes
    for i in range(num_constraints):
        # Intersection with x-axis (y = 0)
        if A[i][0] != 0:
            x_val = b[i] / A[i][0]
            vertex = np.array([x_val, 0])
            if all_constraints_satisfied(vertex, A, b, constraints) and x_val >= -tol:
                vertices.append(vertex)
        # Intersection with y-axis (x = 0)
        if A[i][1] != 0:
            y_val = b[i] / A[i][1]
            vertex = np.array([0, y_val])
            if all_constraints_satisfied(vertex, A, b, constraints) and y_val >= -tol:
                vertices.append(vertex)

    # Include the origin if it is feasible
    origin = np.array([0, 0])
    if all_constraints_satisfied(origin, A, b, constraints):
        vertices.append(origin)

    return np.unique(np.round(vertices, decimals=10), axis=0) if vertices else np.array([[0, 0]])

def all_constraints_satisfied(vertex, A, b, constraints):
    for k in range(len(A)):
        lhs_value = np.dot(A[k], vertex)
        if constraints[k]['inequality_type'] == '<=' and lhs_value > b[k]:
            return False
        elif constraints[k]['inequality_type'] == '>=' and lhs_value < b[k]:
            return False
        elif constraints[k]['inequality_type'] == '=' and not np.isclose(lhs_value, b[k]):
            return False
    return True

def plot_constraints(constraints, feasible_region, optimal_vertex):
    plt.figure(figsize=(10, 8))
    plt.clf()
    x = np.linspace(0, max(np.max(feasible_region), 10), 1000)
    
    for constraint in constraints:
        coeff, b_val = constraint['coefficients'], constraint['rhs']
        if coeff[1] != 0:
            y = (b_val - coeff[0] * x) / coeff[1]
            plt.plot(x, y, '-', label=f"{coeff[0]}x₁ + {coeff[1]}x₂ {constraint['inequality_type']} {b_val}")
        else:
            plt.axvline(x=b_val / coeff[0], color='r', linestyle='--', label=f"x₁ = {b_val}")
    
    if len(feasible_region) >= 3:
        hull = ConvexHull(feasible_region)
        polygon = Polygon(feasible_region[hull.vertices], alpha=0.2, color='lightblue', label='Feasible Region')
        plt.gca().add_patch(polygon)
    
    for point in feasible_region:
        plt.plot(point[0], point[1], 'bo', markersize=5)
    
    if optimal_vertex is not None:
        plt.plot(optimal_vertex[0], optimal_vertex[1], 'o', color='#F472B6', markersize=10, label='Optimal Solution')
        plt.annotate(f'({optimal_vertex[0]:.1f}, {optimal_vertex[1]:.1f})',
                     (optimal_vertex[0], optimal_vertex[1]),
                     xytext=(10, 10), textcoords='offset points')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel("x₁", fontsize=12)
    plt.ylabel("x₂", fontsize=12)
    plt.title("Linear Programming: Graphical Method", fontsize=14, pad=20)
    plt.legend()
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

'''
SIMPLEX STARTS FROM HERE!!
1
1
1
1
1
1
1
1
1
1
'''

def generate_dual_problem(c, A, b):
    A_dual = A.T
    b_dual = c
    c_dual = b
    return c_dual, A_dual, b_dual
def simplex_method_view(request):
    if request.method == 'POST':
        try:
            optimization_type = request.POST.get('optimization_type', 'maximize')
            solve_dual = request.POST.get('solve_dual') == 'on'

            objective_function_input = request.POST.get('objective_function', '').strip()
            if not objective_function_input:
                raise ValueError("Objective function cannot be empty.")
            objective_function_list = [float(x) for x in objective_function_input.split(',')]

            A_list, b_list, constraints_data = parse_constraints_simplex(request)
            A = np.array(A_list)
            b = np.array(b_list)

            # Solve primal
            if optimization_type == 'maximize':
                result = linprog([-coef for coef in objective_function_list],
                                 A_ub=A, b_ub=b, method="highs")
                if not result.success:
                    raise ValueError("No optimal solution found.")
                solution = result.x
                optimal_value = -result.fun
            else:
                result = linprog(objective_function_list,
                                 A_ub=A, b_ub=b, method="highs")
                if not result.success:
                    raise ValueError("No optimal solution found.")
                solution = result.x
                optimal_value = result.fun

            dual_solution = None
            dual_optimal = None
            if solve_dual:
                # Generate and solve dual
                c_dual, A_dual, b_dual = generate_dual_problem(np.array(objective_function_list), A, b)
                result_dual = linprog(c_dual, A_ub=-A_dual, b_ub=-b_dual, method="highs")
                if result_dual.success:
                    dual_solution = result_dual.x
                    dual_optimal = result_dual.fun
                else:
                    raise ValueError("Dual problem could not be solved.")

            context = {
                'objective_function': objective_function_list,
                'constraints': constraints_data,
                'optimal_point': solution.tolist(),
                'optimal_value': optimal_value,
                'dual_solution': dual_solution.tolist() if dual_solution is not None else None,
                'dual_optimal': dual_optimal,
                'dual_solved': solve_dual,
            }
            return render(request, 'result2.html', context)
        except ValueError as e:
            return render(request, 'result2.html', {'error_message': str(e)})
    return render(request, 'result2.html')


def parse_constraints_simplex(request):
    """
    Parses constraints for the simplex method.
    Expected format for each constraint (from the form):
         constraint_i[]: [ "coef1,coef2,...", inequality, rhs ]
    
    If the inequality is ">=" or "≥", the constraint is converted to an equivalent
    <= constraint by multiplying the coefficients and the RHS by -1.
    
    Returns:
      - A: list of lists (processed coefficients for the solver)
      - b: list (processed RHS values for the solver)
      - constraints_data: list of dictionaries with original (or converted)
        constraint values for display.
    """
    A = []
    b = []
    constraints_data = []
    i = 1
    while True:
        constraint_data = request.POST.getlist(f'constraint_{i}[]')
        if not constraint_data:
            break
        try:
            # Expect exactly 3 items: coefficients string, inequality, and rhs.
            coeff_str = constraint_data[0]
            coeffs = [float(x.strip()) for x in coeff_str.split(',')]
            inequality_type = constraint_data[1].strip()
            rhs_value = float(constraint_data[2])
            # If inequality is ">=" (or "≥"), convert to <= by multiplying by -1.
            if inequality_type in [">=", "≥"]:
                coeffs_converted = [-c for c in coeffs]
                rhs_value_converted = -rhs_value
            elif inequality_type in ["<=", "≤"]:
                coeffs_converted = coeffs
                rhs_value_converted = rhs_value
            else:
                raise ValueError("Inequality must be <=, ≤, >=, or ≥.")
            A.append(coeffs_converted)
            b.append(rhs_value_converted)
            # Save original (or appropriately converted) data for display.
            constraints_data.append({
                "coefficients": coeffs,
                "inequality": inequality_type,
                "rhs": rhs_value
            })
        except (ValueError, IndexError):
            raise ValueError(f"Invalid input in constraint {i}")
        i += 1
    return A, b, constraints_data

def simplex(c, A, b):
    """
    Solves the LP using the Simplex Method:
      Maximize: Z = c^T * x
      Subject to: A * x <= b, x >= 0

    Parameters:
      - c: 1D numpy array of objective function coefficients
      - A: 2D numpy array of constraint coefficients
      - b: 1D numpy array of RHS values

    Returns:
      - solution: Optimal solution for the decision variables (numpy array)
      - optimal_value: Optimal objective function value (float)
    """
    num_constraints, num_variables = A.shape

    # Add slack variables to convert inequalities to equalities.
    slack_vars = np.eye(num_constraints)
    tableau = np.hstack((A, slack_vars, b.reshape(-1, 1)))

    # Add the objective function row (with a minus sign for maximization).
    obj_row = np.hstack((-c, np.zeros(num_constraints + 1)))
    tableau = np.vstack((tableau, obj_row))

    num_total_vars = num_variables + num_constraints

    # Simplex iterations: use np.all() to check optimality.
    while True:
        if np.all(tableau[-1, :-1] >= 0):
            break

        # Determine the entering variable (most negative coefficient).
        pivot_col = np.argmin(tableau[-1, :-1])

        # Determine the leaving variable (minimum positive ratio of RHS / pivot_col value).
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_col]
        ratios[ratios <= 0] = np.inf  # Ignore non-positive ratios.
        pivot_row = np.argmin(ratios)

        if np.all(ratios == np.inf):
            raise ValueError("The problem is unbounded.")

        # Pivot operation.
        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element
        for i in range(tableau.shape[0]):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

    # Extract the solution.
    solution = np.zeros(num_total_vars)
    for i in range(num_constraints):
        # Use np.isclose for a robust comparison to 1.
        basic_var_index = np.where(np.isclose(tableau[i, :-1], 1))[0]
        if len(basic_var_index) == 1 and basic_var_index[0] < num_total_vars:
            solution[basic_var_index[0]] = tableau[i, -1]

    optimal_value = tableau[-1, -1]
    return solution[:num_variables], optimal_value

def index(request):
    return render(request, 'index.html')

def transportation_method_view(request):
    if request.method == 'POST':
        try:
            # Parse supply points
            supply_input = request.POST.get('supply', '').strip()
            supply = [float(x.strip()) for x in supply_input.split(',')]

            # Parse demand points
            demand_input = request.POST.get('demand', '').strip()
            demand = [float(x.strip()) for x in demand_input.split(',')]

            # Parse cost matrix
            cost_matrix_input = request.POST.get('cost_matrix', '').strip()
            cost_matrix = [
                [float(x.strip()) for x in row.split(',')]
                for row in cost_matrix_input.split(';')
            ]

            # Solve transportation problem
            result = solve_transportation_problem(cost_matrix, supply, demand)

            if result["solution"] is not None:
                context = {
                    'optimal_solution': result["solution"].tolist(),
                    'total_cost': result["total_cost"],
                    'status': result["status"],
                    'supply': supply,
                    'demand': demand,
                    'cost_matrix': cost_matrix,
                }
            else:
                raise ValueError(f"Failed to find solution: {result['status']}")

            return render(request, 'transportation.html', context)

        except (ValueError, IndexError) as e:
            return render(request, 'transportation.html', {
                'error_message': str(e)
            })

    return render(request, 'transportation.html')

def solve_transportation_problem(cost_matrix, supply, demand):
    """
    Solves the transportation problem using linear programming.
    """
    try:
        # Convert inputs to numpy arrays
        cost_matrix = np.array(cost_matrix)
        supply = np.array(supply)
        demand = np.array(demand)

        # Verify problem is balanced
        if supply.sum() != demand.sum():
            raise ValueError("Supply and demand must be balanced")

        # Get dimensions
        m, n = cost_matrix.shape  # m sources, n destinations

        # Flatten cost matrix for linprog
        c = cost_matrix.flatten()

        # Create equality constraints matrix and vector
        A_eq = []
        b_eq = []

        # Supply constraints (row-wise)
        for i in range(m):
            row = np.zeros(m * n)
            row[i*n:(i+1)*n] = 1
            A_eq.append(row)
            b_eq.append(supply[i])

        # Demand constraints (column-wise)
        for j in range(n):
            col = np.zeros(m * n)
            col[j::n] = 1
            A_eq.append(col)
            b_eq.append(demand[j])

        # Convert to numpy arrays
        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)

        # Solve using linprog
        result = linprog(
            c=c,
            A_eq=A_eq,
            b_eq=b_eq,
            method='highs',
            bounds=(0, None)
        )

        if result.success:
            return {
                "solution": result.x.reshape(m, n),
                "total_cost": result.fun,
                "status": "Optimal solution found"
            }
        else:
            return {
                "solution": None,
                "total_cost": None,
                "status": result.message
            }

    except Exception as e:
        return {
            "solution": None,
            "total_cost": None,
            "status": str(e)
        }

def knapsack_solver(request):
    result = None
    if request.method == 'POST':
        try:
            values = list(map(int, request.POST.get('values').split(',')))
            weights = list(map(int, request.POST.get('weights').split(',')))
            capacity = int(request.POST.get('capacity'))

            n = len(values)
            dp = [[0] * (capacity + 1) for _ in range(n + 1)]

            # Dynamic programming to solve 0/1 knapsack
            for i in range(1, n + 1):
                for w in range(capacity + 1):
                    if weights[i - 1] <= w:
                        dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
                    else:
                        dp[i][w] = dp[i - 1][w]

            # Backtrack to find selected items
            selected_items = []
            w = capacity
            for i in range(n, 0, -1):
                if dp[i][w] != dp[i - 1][w]:
                    selected_items.append(i - 1)
                    w -= weights[i - 1]

            result = {
                'items': list(reversed(selected_items)),
                'value': dp[n][capacity]
            }

        except Exception as e:
            result = {'items': [], 'value': 0, 'error': str(e)}

    return render(request, 'knapsack.html', {'result': result})

"""






All of this is genetic algorithm







"""
from django.shortcuts import render
import random

# Helper functions
import random
from django.shortcuts import render

def fitness(individual):
    # Basic fitness: maximize number of 1s
    return sum(individual)

def selection(population):
    return sorted(population, key=fitness, reverse=True)[:2]

def crossover(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:
        point = random.randint(1, len(parent1)-1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    else:
        return parent1[:], parent2[:]

def mutate(individual, mutation_rate):
    return [bit if random.random() > mutation_rate else 1-bit for bit in individual]

# Main view
def genetic_algorithm(request):
    if request.method == 'POST':
        # Collect form inputs
        gene_length = int(request.POST.get('gene_length'))
        pop_size = int(request.POST.get('population_size'))
        generations = int(request.POST.get('generations'))
        crossover_rate = float(request.POST.get('crossover_rate'))
        mutation_rate = float(request.POST.get('mutation_rate'))

        # Initialize population with random 0s and 1s
        population = [
            [random.randint(0, 1) for _ in range(gene_length)]
            for _ in range(pop_size)
        ]

        best_individual = None
        best_fitness = 0
        history = []

        for gen in range(generations):
            population = sorted(population, key=fitness, reverse=True)
            best_candidate = population[0]
            current_fitness = fitness(best_candidate)
            history.append({'generation': gen+1, 'best': best_candidate, 'fitness': current_fitness})

            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_individual = best_candidate

            next_generation = []

            while len(next_generation) < pop_size:
                parent1, parent2 = selection(population)
                child1, child2 = crossover(parent1, parent2, crossover_rate)
                next_generation.append(mutate(child1, mutation_rate))
                if len(next_generation) < pop_size:
                    next_generation.append(mutate(child2, mutation_rate))

            population = next_generation

        # Extract labels and fitness values for chart rendering
        labels = [h['generation'] for h in history]
        fitness_data = [h['fitness'] for h in history]

        return render(request, 'genetic_algorithm.html', {
            'best_individual': best_individual,
            'best_fitness': best_fitness,
            'history': history,
            'labels': labels,
            'fitness_data': fitness_data,
        })

    return render(request, 'genetic_algorithm.html')



import numpy as np
from scipy.optimize import linprog
from django.shortcuts import render
import copy

import numpy as np
import copy
from scipy.optimize import linprog
from django.shortcuts import render

def is_integer_array(arr):
    return all(float(x).is_integer() for x in arr)

def branch_and_bound(c, A, b):
    best_solution = None
    best_value = None
    nodes = [(A, b)]  # Stack of subproblems

    while nodes:
        A_curr, b_curr = nodes.pop()
        res = linprog(c=-np.array(c), A_ub=A_curr, b_ub=b_curr, method='highs')

        if not res.success:
            continue

        x = res.x
        val = -res.fun

        if is_integer_array(x):
            if best_value is None or val > best_value:
                best_solution = x
                best_value = val
        else:
            frac_indices = [i for i, xi in enumerate(x) if not float(xi).is_integer()]
            if not frac_indices:
                continue

            idx = frac_indices[0]
            floor_constraint = copy.deepcopy(A_curr)
            ceil_constraint = copy.deepcopy(A_curr)

            floor_b = copy.deepcopy(b_curr)
            ceil_b = copy.deepcopy(b_curr)

            new_row_floor = [0] * len(x)
            new_row_floor[idx] = 1
            floor_constraint = np.vstack([floor_constraint, new_row_floor])
            floor_b.append(np.floor(x[idx]))

            new_row_ceil = [0] * len(x)
            new_row_ceil[idx] = -1
            ceil_constraint = np.vstack([ceil_constraint, new_row_ceil])
            ceil_b.append(-np.ceil(x[idx]))

            nodes.append((floor_constraint, floor_b))
            nodes.append((ceil_constraint, ceil_b))

    return best_solution, best_value

def run_branch_and_bound(c, A, b, signs):
    # Handle inequalities if needed (currently only handles ≤)
    # Convert all constraints to ≤ type if required
    A_ub = []
    b_ub = []

    for i in range(len(signs)):
        if signs[i] == "<=":
            A_ub.append(A[i])
            b_ub.append(b[i])
        elif signs[i] == ">=":
            A_ub.append([-a for a in A[i]])
            b_ub.append(-b[i])
        elif signs[i] == "=":
            # Split into two inequalities
            A_ub.append(A[i])
            b_ub.append(b[i])
            A_ub.append([-a for a in A[i]])
            b_ub.append(-b[i])
        else:
            raise ValueError("Unsupported constraint sign.")

    solution, value = branch_and_bound(c, A_ub, b_ub)

    if solution is None:
        return {"error": "No feasible integer solution found."}

    return {
        "optimal_solution": [round(val) for val in solution],
        "optimal_value": round(value, 2)
    }

def branch_and_bound_view(request):
    if request.method == "POST":
        try:
            # Get objective function coefficients
            c = list(map(float, request.POST.get("objective_function").split(",")))

            # Get constraints
            A = []
            b = []
            signs = []

            i = 1
            while True:
                coeffs = request.POST.getlist(f"constraint_{i}[]")
                if not coeffs or len(coeffs) < 3:
                    break
                A.append(list(map(float, coeffs[0].split(","))))
                signs.append(coeffs[1].strip())
                b.append(float(coeffs[2]))
                i += 1

            result = run_branch_and_bound(c, A, b, signs)

            return render(request, "branch_and_bound_form.html", {
                "result": result
            })
        except Exception as e:
            return render(request, "branch_and_bound_form.html", {
                "error": str(e)
            })

    return render(request, "branch_and_bound_form.html")

from sympy import symbols, Eq, diff, solve
from django.shortcuts import render

def kkt_solver(request):
    result = None

    if request.method == 'POST':
        x, y, l = symbols('x y lambda')

        obj_expr = request.POST['objective']
        constraint_expr = request.POST['constraint']

        # Convert strings to symbolic expressions
        f = eval(obj_expr)  # Objective function
        g = eval(constraint_expr)  # Constraint

        # KKT conditions
        L = f + l * g
        dL_dx = diff(L, x)
        dL_dy = diff(L, y)
        dL_dl = g  # The constraint itself

        solutions = solve([dL_dx, dL_dy, dL_dl], (x, y, l), dict=True)
        result = solutions

    return render(request, 'kkt_solver.html', {'result': result})

from django.conf import settings

def simulated_annealing_view(request):
    return render(request, "simulated_annealing.html", context)

import json
import math
import random
from django.shortcuts import render

def haversine(coord1, coord2):
    R = 6371
    lat1, lon1 = coord1['lat'], coord1['lng']
    lat2, lon2 = coord2['lat'], coord2['lng']
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(d_lambda/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

def total_distance(route):
    return sum(haversine(route[i], route[i+1]) for i in range(len(route)-1))

def simulated_annealing_solver(points, T=10000, alpha=0.995, stopping_T=1e-8):
    current = points[:]
    best = points[:]
    current_cost = best_cost = total_distance(current)
    while T > stopping_T:
        i, j = sorted(random.sample(range(len(points)), 2))
        candidate = current[:]
        candidate[i], candidate[j] = candidate[j], candidate[i]
        candidate_cost = total_distance(candidate)
        if candidate_cost < current_cost or random.random() < math.exp((current_cost - candidate_cost) / T):
            current = candidate
            current_cost = candidate_cost
            if current_cost < best_cost:
                best = current
                best_cost = candidate_cost
        T *= alpha
    return best, best_cost

def route_optimizer(request):
    result = None
    if request.method == 'POST':
        points_json = request.POST.get('waypoints')
        if points_json:
            points = json.loads(points_json)
            if len(points) >= 2:
                best_path, best_cost = simulated_annealing_solver(points)
                result = {
                    'path': best_path,
                    'cost': round(best_cost, 2)
                }
            else:
                result = {'error': 'Please select at least 2 waypoints.'}
    return render(request, 'simulated_annealing.html', {'result': result})
