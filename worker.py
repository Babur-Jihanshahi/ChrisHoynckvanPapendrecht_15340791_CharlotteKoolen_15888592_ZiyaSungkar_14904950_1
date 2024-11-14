import numpy as np
from multiprocessing import current_process
import logging
import scipy.stats.qmc as sampling
import orthogonal

def setup_logger(log_file='logs.txt'):
    # Set up basic configuration for logging
    logger = logging.getLogger(current_process().name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.handlers:
        # If the logger already has a handler, use the existing one
        file_handler = logger.handlers[0]
    else:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(file_handler)
    
    # Return both the logger and handler for later use
    return logger, file_handler

def mandelbrot_sampling(c_points, max_iter, escape_radius) -> tuple[int, int]:
    number_outside = 0
    total_numbers = len(c_points)
    
    for c in c_points:
        z = 0
        for iteration in range(max_iter):
            z = z**2 + c
            if abs(z) > escape_radius:
                number_outside+=1
                # If point escapes, break and count it as "outside"
                break
            
    return total_numbers, number_outside


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
            for _ in range(max_iter):
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

def hyper(gridsize, seed):
     orth_sampler = sampling.LatinHypercube(d=2, scramble=True, strength=1,seed=seed)
     orth_samples = orth_sampler.random(n=gridsize).astype(np.float32)
     return orth_samples

def worker_function_sampling(pars):
    # add own sampling method instead of mandelbrot, and add sampling specific parameters to grid and bound. 
    # The function called (instead of mandelbrot) should essentially be the same as non-parallelized function. 
    # logger = logging.getLogger()
    # logger.setLevel(logging.DEBUG)
    # logging.debug("test")

    grid_size, bound, run, random_seed = pars
    logger, file_handler = setup_logger()
    process_name = current_process().name

    logger.debug(f"I am {process_name} handling parameters: {grid_size, bound, run}")
    file_handler.flush()

    points = hyper(grid_size**2, random_seed) 

    
    min_x, max_x = -2, 1
    min_y, max_y = -1.5, 1.5
    real = min_x + points[:, 0]*3
    imag = min_y + points[:, 1]*3
    c_points = real + 1j * imag
    points = []

    logger.debug(f"I am {process_name} sample size: {c_points.size}")


    total, out = mandelbrot_sampling(c_points, bound, 2)

    # note here, the actual grid is not returned, only the grid size
    return (total, out), (grid_size, bound, run)

def worker_orthogonal(pars):
    grid_size, bound, run, random_seed = pars
    logger, file_handler = setup_logger()
    process_name = current_process().name

    logger.debug(f"I am {process_name} handling parameters: {grid_size, bound, run}")
    file_handler.flush()

    points = orthogonal.orthogonal_sample(grid_size, random_seed)
    c_points = points[:, 0] + 1j * points[:,1]

    logger.debug(f"I am {process_name} sample size: {c_points.size}")
    total, out = mandelbrot_sampling(c_points, bound, 2)

    # note here, the actual grid is not returned, only the grid size
    return (total, out), (grid_size, bound, run)

def random_points_generator(num_samples: int, rand) -> np.ndarray:
    """
    Generate random points for the Monte Carlo Integration
    """
    np.random.seed(rand)
    real_parts = np.random.uniform(-2.0, 1.0, num_samples)
    imag_parts = np.random.uniform(-1.5, 1.5, num_samples)
    return real_parts + 1j * imag_parts

# def monte_carlo_calc(points: np.ndarray, max_iter: int) -> tuple[int, int]:
#     points_inside = 0
#     total_points = len(points)

#     for c in points:
#         z = 0
#         for _ in range(max_iter):
#             z = z*z + c
#             if abs(z) > 2.0:
#                 break
#         else:
#             points_inside += 1

#     return total_points, points_inside
    
def worker_pure(pars):
    """
    Woker function for the Monte Carlo sampling
    """
    grid_size, max_iter, run, rand = pars
    logger, file_handler = setup_logger()
    process_name = current_process().name

    logger.debug(f"I am {process_name} handling parameters: {grid_size, max_iter, run}")
    file_handler.flush()
    points = random_points_generator(grid_size * grid_size, rand)
    logger.debug(f"I am {process_name} sample size: {points.size}")
    total_points, points_outside = mandelbrot_sampling(points, max_iter, 2)
    return (total_points, points_outside), (grid_size, max_iter, run)