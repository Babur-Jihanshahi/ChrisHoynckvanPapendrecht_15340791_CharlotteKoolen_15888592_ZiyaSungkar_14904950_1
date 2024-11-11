import numpy as np
import matplotlib.pyplot as plt

def random_points_generator(num_samples: int) -> np.ndarray:
    real_parts = np.random.uniform(-2.0, 1.0, num_samples)
    imag_parts = np.random.uniform(-1.5, 1.5, num_samples)
    return real_parts + 1j * imag_parts

def mandelbrot(c_points, max_iter, escape_radius) -> tuple[np.array, np.array]:
    '''
    This function calculates the number of iterations until the magnitude of z escapes to infinity. 
    Within each iteration the z is updated with c, an imaginary number representing a grid point. 
    Ultimately, a mandelbrot is calculated. 
    '''
    iteration_count = np.zeros(c_points.shape)
    mandelbrot_set = np.zeros(c_points.shape, dtype=bool)
    for i in range(c_points.shape[0]):
        for j in range(c_points.shape[1]):

            # take a gridpoint
            c = c_points[i, j]
            z = 0
            for iteration in range(max_iter):
                # update of z
                z = z**2 + c
                if abs(z) > escape_radius:
                    iteration_count[i, j] = iteration
                    break
            else:
                iteration_count[i, j] = max_iter
    return (mandelbrot_set, iteration_count)

def monte_carlo_area(num_samples: int, max_iter: int) -> float:
    c_points = random_points_generator(num_samples)
    points_inside = 0
    for c in c_points: 
        z = 0
        for _ in range(max_iter):
            z = z*z + c
            if abs(z) > 2.0:
                break
        else:
            points_inside += 1
    
    total_area = (1-(-2)) * (1.5 - (-1.5))
    estimated_area = (points_inside / num_samples) * total_area
    return estimated_area

def test_convergence():
    iterations = [10, 20, 50, 100, 200]
    samples = [1000, 5000, 10000, 50000, 100000]
    results = {}
    for i in iterations:
        for s in samples:
            area = monte_carlo_area(s, i)
            results[(i, s)] = area
            print(f"i={i}, s={s}: Area ≈ {area:.6f}")
    return results
    

def plot_mandelbrot_visualization(grid_size=1000, max_iter=250, escape_radius=2):
    """
    Create visualization of the Mandelbrot set using a grid.
    """
    real = np.linspace(-2.0, 1.0, grid_size)
    imag = np.linspace(-1.5, 1.5, grid_size)
    real_grid, imag_grid = np.meshgrid(real, imag)
    c_points = real_grid + 1j * imag_grid
    
    _, iter_count = mandelbrot(c_points, max_iter, escape_radius)
    
    plt.figure(figsize=(10, 6), dpi=300)
    plt.imshow(iter_count, extent=(-2.0, 1.0, -1.5, 1.5), cmap='plasma', origin='lower')
    plt.xlabel("Re(c)")
    plt.ylabel("Im(c)")
    plt.title("Mandelbrot Set")
    plt.colorbar(label="Iteration Count")
    plt.show()


if __name__ == "__main__":
    # First create a visualization if desired
    # plot_mandelbrot_visualization()
    
    # Then test convergence
    results = test_convergence()
    # Plot results
    plt.figure(figsize=(12, 5))
   
    # Plot convergence with respect to iterations
    plt.subplot(1, 2, 1)
    for s in [1000, 10000, 100000]:
        areas = [results[(i, s)] for i in [10, 20, 50, 100, 200]]
        plt.plot([10, 20, 50, 100, 200], areas, 'o-', label=f's={s}')
    plt.xlabel('Number of iterations (i)')
    plt.ylabel('Estimated area')
    plt.title('Convergence with iterations')
    plt.legend()
    plt.grid(True)
   
    # Plot convergence with respect to samples
    plt.subplot(1, 2, 2)
    for i in [20, 50, 100]:
        areas = [results[(i, s)] for s in [1000, 5000, 10000, 50000, 100000]]
        plt.plot([1000, 5000, 10000, 50000, 100000], areas, 'o-', label=f'i={i}')
    plt.xlabel('Number of samples (s)')
    plt.ylabel('Estimated area')
    plt.title('Convergence with samples')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
   
    plt.tight_layout()
    plt.show()