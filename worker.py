import numpy as np
# import matplotlib.pyplot as plt


def mandelbrot(c_points, max_iter, escape_radius) -> tuple[int, int]:
    '''
    This function calculates the number of iterations until the magnitude of z escapes to infinity. 
    Within each iteration the z is updated with c, an imaginary number representing a grid point. 
    Ultimately, a mandelbrot is calculated. 
    '''
    # iteration_count = np.zeros(c_points.shape)
    # mandelbrot_set = np.zeros(c_points.shape, dtype=bool)
    number_outside = 0
    total_numbers = 0
    for i in range(c_points.shape[0]):
        for j in range(c_points.shape[1]):
            total_numbers +=1
            # take a gridpoint
            c = c_points[i, j]
            z = 0
            for iteration in range(max_iter):
                # update of z
                z = z**2 + c
                if abs(z) > escape_radius:
                    number_outside+=1
                    # mandelbrot_set[i, j] = False
                    # iteration_count[i, j] =iteration
                    break
            # else:
                # mandelbrot_set[i, j] = True
                # iteration_count[i, j] = max_iter
    return (total_numbers, number_outside)



def worker_function(xs):
    cpts, s = xs
    y = cpts.shape[0]
    y1 = cpts.shape[1]

    total_numbs, number_ins = mandelbrot(cpts,s, 2)
    return (total_numbs, number_ins)

def worker_function_sampling(pars):
    # add own sampling method instead of mandelbrot, and add sampling specific parameters to grid and bound. 
    # The function called (instead of mandelbrot) should essentially be the same as non-parallelized function. 
    grid, grid_size, bound = pars 
    total, out = mandelbrot(grid, bound, 2)

    # note here, the actual grid is not returned, only the grid size
    return (total, out), (grid_size, bound) 

def random_points_generator(num_samples: int) -> np.ndarray:
    """
    Generate random points for the Monte Carlo Integration
    """
    real_parts = np.random.uniform(-2.0, 1.0, num_samples)
    imag_parts = np.random.uniform(-1.5, 1.5, num_samples)
    return real_parts + 1j * imag_parts

def monte_carlo_calc(points: np.ndarray, max_iter: int) -> tuple[int, int]:
    points_inside = 0
    total_points = len(points)

    for c in points:
        z = 0
        for _ in range(max_iter):
            z = z*z + c
            if abs(z) > 2.0:
                break
        else:
            points_inside += 1

    return total_points, points_inside
    
def worker_pure(pars):
    """
    Woker function for the Monte Carlo sampling
    """
    _, grid_size, max_iter = pars
    points = random_points_generator(grid_size * grid_size)
    total_points, points_inside = monte_carlo_calc(points, max_iter)
    return (total_points, points_inside), (grid_size, max_iter)