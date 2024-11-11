import numpy as np
import matplotlib.pyplot as plt
from part_1 import random_points_generator, monte_carlo_area, test_convergence

def process_grid_results(file_name: str) -> dict:
    """
    Process the grid-based results from the text file.
    Returns dictionary with (grid_size, iterations) as key and area estimate as value.
    """
    results = {} 
    with open(file_name, 'r') as f:
        next(f)  # skip header
        for line in f:
            grid, iters, total, inside = map(int, line.strip().split(','))
            area = (inside / total) * 9
            results[(grid, iters)] = area  
    return results


def plot_comparison():
    # True value for reference
    true_area = 1.50659177
    results_grid = process_grid_results('question2.txt')
    available_grids = sorted(set([key[0] for key in results_grid.keys()]))
    iterations = sorted(set([key[1] for key in results_grid.keys()]))
    print(f"\nWill process {len(available_grids)} grid sizes and {len(iterations)} iteration counts")
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    # Left plot: Area estimation comparison
    print("\nGenerating area estimation comparison...")
    for i in iterations:
        valid_grids = [grid for grid in available_grids if (grid, i) in results_grid]
        if not valid_grids:
            continue
        print(f"\nProcessing iteration count {i}:")
        grid_areas = [results_grid[(grid, i)] for grid in valid_grids]
        random_areas = [monte_carlo_area(grid*grid, i) for grid in valid_grids]   # grid*grid for equal points

        # plot
        ax1.plot(valid_grids, grid_areas, 'o-', label=f'Grid -{i} iter')
        ax1.plot(valid_grids, random_areas, 's--', label=f'Random -{i} iter')

    ax1.axhline(y=true_area, color='r', linestyle=':', label='True Area')
    ax1.set_xlabel('Grid Size (N for NxN grid)')
    ax1.set_ylabel('Estimated Area')
    ax1.set_title('Area Estimate Comparison')
    ax1.grid(True)
    ax1.set_xscale('log')

    # Right plot: Relative Errors
    print("\nCalculating relative errors...")
    for i in iterations:
        valid_grids = [grid for grid in available_grids if (grid, i) in results_grid]
        if not valid_grids:
            continue
        print(f"Processing errors for iteration count {i}")
        grid_errors = [abs(results_grid[(grid, i)] - true_area) / true_area for grid in valid_grids]
        random_errors = [abs(monte_carlo_area(grid*grid, i) - true_area) / true_area for grid in valid_grids]
        # plot
        ax2.plot(valid_grids, grid_errors, 'o-', label=f'Grid - {i} iter')
        ax2.plot(valid_grids, random_errors, 's--', label=f'Random - {i} iter')
    
    ax2.set_xlabel('Grid Size (N for NxN grid)')
    ax2.set_ylabel('Relative Error')
    ax2.set_title('Relative Error Comparison')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    plt.tight_layout()

    # Print statistical summary
    print("\nStatistical Summary:")
    print("-" * 50)
    for i in iterations[-3:]:  # Look at the last few iteration counts
        valid_grids = [grid for grid in available_grids if (grid, i) in results_grid]
        if not valid_grids:
            continue
            
        print(f"\nFor {i} iterations:")
        for grid in valid_grids[-3:]:  # Look at the largest grid sizes
            grid_area = results_grid[(grid, i)]
            random_area = monte_carlo_area(grid*grid, i)
            
            grid_error = abs(grid_area - true_area)/true_area
            random_error = abs(random_area - true_area)/true_area
            
            print(f"\nGrid size {grid}x{grid} ({grid*grid} points):")
            print(f"  Grid method:   Area = {grid_area:.6f}, Error = {grid_error:.6%}")
            print(f"  Random method: Area = {random_area:.6f}, Error = {random_error:.6%}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        plot_comparison()
        plt.show()
    except Exception as e:
        print(f"An error occurred: {str(e)}")


