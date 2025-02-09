from django.shortcuts import render
from django.http import HttpResponse
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

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

            return render(request, 'result.html', {
                'optimal_point': optimal_point,
                'optimal_value': optimal_value,
                'graph_image': graph_image,
            })
        except ValueError as e:
            return render(request, 'result.html', {'error_message': str(e)})

    return render(request, 'result.html')

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

def index(request):
    return render(request, 'index.html')
