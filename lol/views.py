from django.shortcuts import render
from django.http import HttpResponse
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

def graphical_method_view(request):
    if request.method == 'POST':
        try:
            # Objective Function input
            objective_function_input = request.POST.get('objective_function', '').strip()

            if not objective_function_input:
                raise ValueError("Objective function cannot be empty.")

            # Parse coefficients from the objective function (assuming format "3, 4")
            objective_coefficients = [float(x) for x in objective_function_input.split(',')]

            # Constraints input
            constraints = []
            constraint_prefix = 'constraint_1'  # This should match the name attributes for constraints
            i = 1
            while True:
                constraint_1 = request.POST.getlist(f'{constraint_prefix}[]')
                if not constraint_1:
                    break
                try:
                    # Coefficients and right-hand side values
                    constraint_coeffs = [float(x) for x in constraint_1[:2]]
                    rhs_value = float(constraint_1[2])
                    constraints.append((constraint_coeffs, rhs_value))
                except ValueError:
                    raise ValueError(f"Invalid input in constraint {i}, please check the values.")
                i += 1

            # Graph generation (Just an example, you can customize this part)
            fig, ax = plt.subplots()
            x = np.linspace(-10, 10, 400)
            y = (objective_coefficients[0] * x) / objective_coefficients[1]
            ax.plot(x, y, label='Objective Function')

            for constraint in constraints:
                constraint_coeffs, rhs_value = constraint
                y_constraint = (rhs_value - constraint_coeffs[0] * x) / constraint_coeffs[1]
                ax.plot(x, y_constraint, label=f'Constraint: {constraint_coeffs[0]}x + {constraint_coeffs[1]}y <= {rhs_value}')

            # Format the graph for embedding into HTML
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.legend()

            # Save the plot to a BytesIO object and then encode it to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Send the optimal point and value for now as placeholders
            optimal_point = (0, 0)  # Replace with your actual logic for optimal point calculation
            optimal_value = sum(objective_coefficients)  # Replace with actual optimal value logic

            return render(request, 'result.html', {
                'optimal_point': optimal_point,
                'optimal_value': optimal_value,
                'graph_image': img_str,
            })

        except ValueError as e:
            return render(request, 'result.html', {
                'error_message': str(e),
            })

    return render(request, 'result.html')

def solve_linear_program(c, A, b):
    """Solves the linear programming problem and generates a graph."""
    bounds = [0, max(b)]  # Define a reasonable range for visualization
    constraints = list(zip(A, b))

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
                if all(np.dot(A, vertex) <= b) and all(vertex >= 0):  # Ensure non-negativity and feasibility
                    vertices.append(vertex)
            except np.linalg.LinAlgError:
                continue

    # Filter unique vertices
    feasible_vertices = np.unique(vertices, axis=0)

    # Evaluate the objective function at each vertex
    optimal_vertex = None
    optimal_value = None
    if len(feasible_vertices) > 0:
        z_values = [np.dot(c, v) for v in feasible_vertices]
        optimal_value = max(z_values)
        optimal_vertex = feasible_vertices[np.argmax(z_values)]

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
    for coeff, b in constraints:
        if coeff[1] != 0:  # Plot lines with a slope
            y = (b - coeff[0] * x) / coeff[1]
            plt.plot(x, y, label=f"{coeff[0]}x1 + {coeff[1]}x2 ≤ {b}")
        else:  # Vertical line
            x_val = b / coeff[0]
            plt.axvline(x_val, color='r', linestyle='--', label=f"x1 = {x_val}")

    # Highlight feasible region
    if feasible_region is not None and len(feasible_region) > 0:
        hull = ConvexHull(feasible_region)
        polygon = Polygon(feasible_region[hull.vertices], closed=True, color='lightgreen', alpha=0.5, label='Feasible Region')
        plt.gca().add_patch(polygon)

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
    plt.savefig('test_graph.png')

    return graph_image

def index(request):
    """Renders the index page to choose the method."""
    return render(request, 'index.html')
