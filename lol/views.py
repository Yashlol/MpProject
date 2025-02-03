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
            objective_function_input = request.POST.get('objective_function', '').strip()
            if not objective_function_input:
                raise ValueError("Objective function cannot be empty.")
            
            objective_coefficients = [float(x) for x in objective_function_input.split(',')]
            
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

def solve_linear_program(c, A, b, constraints, optimization_type):
    x_values = [abs(row[0]) for row in A] + [abs(row[1]) for row in A]
    bounds = [0, max(x_values) + 5]

    vertices = []
    num_constraints = len(A)
    for i in range(num_constraints):
        for j in range(i + 1, num_constraints):
            A_ = np.array([A[i], A[j]])
            b_ = np.array([b[i], b[j]])
            try:
                vertex = np.linalg.solve(A_, b_)
                
                feasible = True
                for k in range(num_constraints):
                    lhs_value = np.dot(A[k], vertex)
                    if constraints[k]['inequality_type'] == '<=' and lhs_value > b[k]:
                        feasible = False
                    elif constraints[k]['inequality_type'] == '>=' and lhs_value < b[k]:
                        feasible = False
                    elif constraints[k]['inequality_type'] == '=' and not np.isclose(lhs_value, b[k]):
                        feasible = False
                    if not feasible:
                        break
                
                if feasible and (vertex >= 0).all():
                    vertices.append(vertex)
            except np.linalg.LinAlgError:
                continue
    
    feasible_vertices = np.unique(vertices, axis=0)
    
    optimal_vertex, optimal_value = None, None
    if len(feasible_vertices) > 0:
        z_values = np.dot(feasible_vertices, c)
        optimal_index = np.argmax(z_values) if optimization_type == 'maximize' else np.argmin(z_values)
        optimal_value = z_values[optimal_index]
        optimal_vertex = feasible_vertices[optimal_index]
    
    graph_image = plot_constraints(constraints, bounds, feasible_vertices, optimal_vertex)
    return graph_image, optimal_vertex.tolist() if optimal_vertex is not None else None, optimal_value

def plot_constraints(constraints, bounds, feasible_region, optimal_vertex):
    plt.figure(figsize=(10, 8))
    plt.clf()  # Clear the current figure
    
    # Create a more granular x range
    x = np.linspace(bounds[0], bounds[1], 1000)
    
    # Plot each constraint line
    for constraint in constraints:
        coeff, b = constraint['coefficients'], constraint['rhs']
        if coeff[1] != 0:  # Normal case (not vertical line)
            y = (b - coeff[0] * x) / coeff[1]
            plt.plot(x, y, '-', label=f"{coeff[0]}x₁ + {coeff[1]}x₂ {constraint['inequality_type']} {b}")
        else:  # Vertical line case
            x_val = b / coeff[0] if coeff[0] != 0 else 0
            plt.axvline(x=x_val, color='r', linestyle='--', label=f"x₁ = {b}")

    
    # Plot feasible region if we have enough points
    if len(feasible_region) >= 3:
    try:
        hull = ConvexHull(feasible_region)
        polygon = Polygon(feasible_region[hull.vertices], alpha=0.2, color='lightblue', label='Feasible Region')
        plt.gca().add_patch(polygon)
    except Exception as e:
        print(f"Could not create convex hull: {e}")
        plt.fill(*zip(*feasible_region), alpha=0.2, color='lightblue', label='Feasible Region (approx)')

    
    # Plot corner points
    for point in feasible_region:
        plt.plot(point[0], point[1], 'bo', markersize=5)
    
    # Plot optimal solution with a different style
    if optimal_vertex is not None:
        plt.plot(optimal_vertex[0], optimal_vertex[1], 'ro', 
                markersize=10, 
                label='Optimal Solution')
        # Add coordinates annotation
        plt.annotate(f'({optimal_vertex[0]:.1f}, {optimal_vertex[1]:.1f})',
                    (optimal_vertex[0], optimal_vertex[1]),
                    xytext=(10, 10), 
                    textcoords='offset points')
    
    # Improve the plot appearance
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel("x₁", fontsize=12)
    plt.ylabel("x₂", fontsize=12)
    plt.title("Linear Programming: Graphical Method", fontsize=14, pad=20)
    
    # Set axis limits with some padding
    padding = (bounds[1] - bounds[0]) * 0.1
    plt.xlim(bounds[0] - padding, bounds[1] + padding)
    plt.ylim(bounds[0] - padding, bounds[1] + padding)
    
    # Improve legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Make the plot tight and save it
    plt.tight_layout()
    
    # Save to buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    return base64.b64encode(image_png).decode('utf-8')

def index(request):
    return render(request, 'index.html')
