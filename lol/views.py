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
            # Get optimization type
            optimization_type = request.POST.get('optimization_type', 'maximize')

            # Objective Function input
            objective_function_input = request.POST.get('objective_function', '').strip()
            if not objective_function_input:
                raise ValueError("Objective function cannot be empty.")
            
            # Parse coefficients from the objective function
            objective_coefficients = [float(x) for x in objective_function_input.split(',')]

            # Collect all constraints
            constraints = []
            A = []  # Coefficients matrix
            b = []  # RHS values
            i = 1
            while True:
                constraint_data = request.POST.getlist(f'constraint_{i}[]')
                if not constraint_data:
                    break
                
                try:
                    # Get coefficients, inequality type, and RHS value
                    x1_coeff = float(constraint_data[0])
                    x2_coeff = float(constraint_data[1])
                    inequality_type = constraint_data[2]  # '<=', '>=', or '='
                    rhs_value = float(constraint_data[3])
                    
                    constraints.append({
                        'coefficients': [x1_coeff, x2_coeff],
                        'inequality_type': inequality_type,
                        'rhs': rhs_value
                    })
                    
                    # Store in matrix form for solving
                    A.append([x1_coeff, x2_coeff])
                    b.append(rhs_value)

                except (ValueError, IndexError):
                    raise ValueError(f"Invalid input in constraint {i}")
                i += 1

            # Call function to solve LP
            graph_image, optimal_point, optimal_value = solve_linear_program(objective_coefficients, A, b,constraints, optimization_type)

            return render(request, 'result.html', {
                'optimal_point': optimal_point,
                'optimal_value': optimal_value,
                'graph_image': graph_image,
            })

        except ValueError as e:
            return render(request, 'result.html', {
                'error_message': str(e),
            })

    return render(request, 'result.html')


def solve_linear_program(c, A, b, constraints, optimization_type):
    """Solves the linear programming problem and generates a graph."""
    # Dynamic bounds based on constraints
    x_values = [abs(row[0]) for row in A] + [abs(row[1]) for row in A]
    bounds = [0, max(x_values) + 5]  # Slightly larger than max constraint

    # Solve using vertices of the feasible region
    vertices = []
    num_constraints = len(A)
    for i in range(num_constraints):
        for j in range(i + 1, num_constraints):
            # Find intersection of two lines
            A_ = np.array([A[i], A[j]])
            b_ = np.array([b[i], b[j]])
            try:
                vertex = np.linalg.solve(A_, b_)
                
                # Check feasibility considering inequality type
                lhs_values = np.dot(A, vertex)
                feasible_conditions = [
                    (lhs_values <= b).all() if constraint['inequality_type'] == '<=' else
                    (lhs_values >= b).all() if constraint['inequality_type'] == '>=' else
                    np.isclose(lhs_values, b).all()
                    for constraint in constraints
                ]
                
                if all(feasible_conditions) and (vertex >= 0).all():
                    vertices.append(vertex)

            except np.linalg.LinAlgError:
                continue

    # Filter unique vertices
    feasible_vertices = np.unique(vertices, axis=0)

    # Evaluate the objective function at each vertex
    optimal_vertex = None
    optimal_value = None
    if len(feasible_vertices) > 0:
        z_values = np.dot(feasible_vertices, c)
        if optimization_type == 'maximize':
            optimal_index = np.argmax(z_values)
        else:  # Minimization case
            optimal_index = np.argmin(z_values)
        
        optimal_value = z_values[optimal_index]
        optimal_vertex = feasible_vertices[optimal_index]

    # Generate graph
    graph_image = plot_constraints(constraints, bounds, feasible_region=feasible_vertices, optimal_vertex=optimal_vertex)
    print("Feasible Vertices:", feasible_vertices)
    print("Optimal Vertex:", optimal_vertex)
    print("Optimal Value:", optimal_value)

    return graph_image, optimal_vertex, optimal_value



def plot_constraints(constraints, bounds, feasible_region=None, optimal_vertex=None):
    """Plots the constraints, feasible region, and optimal solution."""
    x = np.linspace(bounds[0], bounds[1], 400)
    plt.figure(figsize=(10, 8))

    # Plot constraints as lines
    for constraint in constraints:
        coeff, b = constraint['coefficients'], constraint['rhs']
        if coeff[1] != 0:  # Plot lines with a slope
            y = (b - coeff[0] * x) / coeff[1]
            plt.plot(x, y, label=f"{coeff[0]}x1 + {coeff[1]}x2 {constraint['inequality_type']} {b}")
        else:  # Vertical line
            x_val = b / coeff[0]
            plt.axvline(x_val, color='r', linestyle='--', label=f"x1 = {x_val}")

    # Highlight feasible region
    # Highlight feasible region
    if feasible_region is not None and len(feasible_region) >= 3:
        try:
            hull = ConvexHull(feasible_region)
            polygon = Polygon(feasible_region[hull.vertices], closed=True, color='lightgreen', alpha=0.5, label='Feasible Region')
            plt.gca().add_patch(polygon)
        except QhullError:
            print("QhullError: Not enough points to construct a valid feasible region.")
    elif len(feasible_region) > 0:
        print("Warning: Less than 3 feasible points. Feasible region not plotted.")


    # Highlight corner points
    if feasible_region is not None:
        for point in feasible_region:
            plt.plot(point[0], point[1], 'bo')  # Mark corners

    # Highlight the optimal solution
    if optimal_vertex is not None:
        plt.plot(optimal_vertex[0], optimal_vertex[1], 'ro', label='Optimal Solution')

    plt.xlim(bounds)
    plt.ylim(bounds)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Linear Programming: Graphical Method")
    plt.legend()
    plt.grid()

    # Convert plot to PNG image and encode in Base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()

    graph_image = base64.b64encode(image_png).decode('utf-8')

    return graph_image


def index(request):
    """Renders the index page to choose the method."""
    return render(request, 'index.html')
