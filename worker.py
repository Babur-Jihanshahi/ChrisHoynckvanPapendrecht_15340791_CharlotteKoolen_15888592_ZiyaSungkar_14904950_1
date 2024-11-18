import numpy as np
from multiprocessing import current_process
import logging
import scipy.stats.qmc as sampling
from scipy.spatial import cKDTree
from orthogonal import orthogonal_sample

def importance_random_points(num_samples, rand):
    """
    Random sampling improved using exact mathematical boundaries
    """
    np.random.seed(rand)

    # Split sampling between focused and pure random
    focused_samples = int(num_samples * 0.7)
    random_samples = num_samples - focused_samples
    
    points = []
    
    # Main cardioid sampling (60% of focused samples)
    cardioid_points = int(focused_samples * 0.6)
    t = np.linspace(0, 2*np.pi, cardioid_points)
    noise = np.random.normal(0, 0.1, cardioid_points)
    
    # Using the exact cardioid formula
    x = 0.25 * (2 * np.cos(t) - np.cos(2*t)) + noise * np.cos(t)
    y = 0.25 * (2 * np.sin(t) - np.sin(2*t)) + noise * np.sin(t)
    points.extend(x + 1j * y)

    # Period-2 bulb sampling (40% of focused samples)
    bulb_points = focused_samples - cardioid_points
    theta = np.linspace(0, 2*np.pi, bulb_points)
    noise = np.random.normal(0, 0.02, bulb_points)  # Smaller noise for the bulb
    
    # Using the exact period-2 bulb formula
    x = -1 + 0.25 * np.cos(theta) + noise * np.cos(theta)
    y = 0.25 * np.sin(theta) + noise * np.sin(theta)
    points.extend(x + 1j * y)

    # Pure random samples
    real_parts = np.random.uniform(-2.0, 1.0, random_samples)
    imag_parts = np.random.uniform(-1.5, 1.5, random_samples)
    points.extend(real_parts + 1j * imag_parts)
    
    return np.array(points)

def importance_orthogonal_sample(n, rand):
    """ 
    Orthogonal sampling with exact mathematical boundaries
    """
    MAJOR = n
    SAMPLES = MAJOR * MAJOR
    scale = 3 / SAMPLES
    np.random.seed(rand)

    xlist = np.array([[i * MAJOR + j for j in range(MAJOR)] for i in range(MAJOR)])
    ylist = xlist.copy()

    for i in range(MAJOR):
        np.random.shuffle(xlist[i])
        np.random.shuffle(ylist[i])

    coordinates = []
    for i in range(MAJOR):
        for j in range(MAJOR):
            if np.random.random() < 0.7:
                if np.random.random() < 0.6:
                    # Main cardioid using exact formula
                    theta = np.random.uniform(0, 2*np.pi)
                    noise = 0.1 * np.random.normal()
                    x = 0.25 * (2 * np.cos(theta) - np.cos(2*theta)) + noise * np.cos(theta)
                    y = 0.25 * (2 * np.sin(theta) - np.sin(2*theta)) + noise * np.sin(theta)
                else:
                    # Period-2 bulb using exact formula
                    theta = np.random.uniform(0, 2*np.pi)
                    noise = 0.1 * np.random.normal()
                    x = -1 + 0.25 * np.cos(theta) + noise * np.cos(theta)
                    y = 0.25 * np.sin(theta) + noise * np.sin(theta)
            else:
                # Original orthogonal sampling for coverage
                x = -2 + scale * (xlist[i][j] + np.random.uniform(0, 1))
                y = -1.5 + scale * (ylist[j][i] + np.random.uniform(0, 1))

            coordinates.append((x, y))
    return np.array(coordinates)

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

def worker_LHS(pars):
    grid_size, bound, run, random_seed = pars
    logger, file_handler = setup_logger()
    process_name = current_process().name

    logger.debug(f"I am {process_name} handling parameters: {grid_size, bound, run}")
    file_handler.flush()

    points = hyper(grid_size**2, random_seed) 

    real = -2 + points[:, 0]*3
    imag = -1.5 + points[:, 1]*3
    c_points = real + 1j * imag
    points = []

    logger.debug(f"I am {process_name}, handled sample size: {len(real)}")

    total, out = mandelbrot_sampling(c_points, bound, 2)

    # note here, the actual grid is not returned, only the grid size
    return (total, out), (grid_size, bound, run)

def worker_orthogonal(pars):
    grid_size, bound, run, random_seed = pars
    logger, file_handler = setup_logger()
    process_name = current_process().name

    logger.debug(f"I am {process_name} handling parameters: {grid_size, bound, run}")
    file_handler.flush()

    points = orthogonal_sample(grid_size, random_seed)
    c_points = points[:, 0] + 1j * points[:,1]

    logger.debug(f"I am {process_name} sample size: {c_points.size/2}")
    total, out = mandelbrot_sampling(c_points, bound, 2)

    # note here, the actual grid is not returned, only the grid size
    return (total, out), (grid_size, bound, run)

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
    logger.debug(f"I am {process_name}, handled sample size: {points.size/2}")
    total_points, points_outside = mandelbrot_sampling(points, max_iter, 2)
    return (total_points, points_outside), (grid_size, max_iter, run)

def random_points_generator(num_samples: int, rand) -> np.ndarray:
    """
    Generate random points for the Monte Carlo Integration
    """
    np.random.seed(rand)
    real_parts = np.random.uniform(-2.0, 1.0, num_samples)
    imag_parts = np.random.uniform(-1.5, 1.5, num_samples)
    return real_parts + 1j * imag_parts


def random_points_generator_importance(num_samples: int, rand) -> np.ndarray:
    """
    Generate random points for the Monte Carlo Integration
    """
    np.random.seed(rand)
    real_parts = np.random.uniform(-2.0, 1.0, num_samples)
    imag_parts = np.random.uniform(-1.5, 1.5, num_samples)
    return real_parts, imag_parts


def worker_importance_random(pars):
    """ 
    Worker function for improved random sampling using simple random point generation
    """
    grid_size, base_iter, run, rand, border_points, border_tree = pars
    logger, file_handler = setup_logger('logs.txt')
    process_name = current_process().name
    
    logger.debug(f"I am {process_name}, Method:Random, handling parameters: {grid_size, base_iter, run}")
    file_handler.flush()
    
    real, imag = random_points_generator_importance(grid_size**2, rand)
    
    rand+=331
    # Use pre-computed border points and tree
    total_points, points_outside = adaptive_mandelbrot_sampling(real, imag, border_points, rand, base_iter, border_tree)
    logger.debug(f"I am {process_name}, handled sample size: {total_points}")
    file_handler.flush()

    return (total_points, points_outside), (grid_size, base_iter, run)

def worker_importance_orthogonal(pars):
    """ 
    Worker function for improved orthogonal sampling using simple orthogonal sampling
    """
    grid_size, base_iter, run, rand, border_points, border_tree = pars
    logger, file_handler = setup_logger('logs.txt')
    
    process_name = current_process().name
    
    logger.debug(f"I am {process_name}, Method: Orthogonal, handling parameters: {grid_size, base_iter, run}")
    file_handler.flush()
    
    points = orthogonal_sample(grid_size, rand)
    
    rand+=331
    # Use pre-computed border points and tree
    total_points, points_outside = adaptive_mandelbrot_sampling(points[:, 0], points[:, 1], border_points, rand, base_iter, border_tree)

    logger.debug(f"I am {process_name}, handled sample size: {total_points}")
    file_handler.flush()

    return (total_points, points_outside), (grid_size, base_iter, run)

def worker_importance_LHS(pars):
    grid_size, base_iter, run, rand, border_points, border_tree = pars
    logger, file_handler = setup_logger('logs.txt')
    process_name = current_process().name

    logger.debug(f"I am {process_name}, Method: LHS, handling parameters: {grid_size, base_iter, run}")
    file_handler.flush()

    points = hyper(grid_size**2, rand) 

    real = -2 + points[:, 0]*3
    imag = -1.5 + points[:, 1]*3
    
    rand+=331
    total_points, points_outside = adaptive_mandelbrot_sampling(real, imag, border_points, rand, base_iter, border_tree)

    logger.debug(f"I am {process_name}, handled sample size: {total_points}")
    file_handler.flush()

    return (total_points, points_outside), (grid_size, base_iter, run)



def min_distance(reals, imags, border_points, base_iter, tree=None):
    """
    Calculate minimum distance using KD-tree if available
    """
    points = np.column_stack((reals, imags))
    if tree is not None:
        distances, _ = tree.query(points, k=1)
    else:
        distances = np.array([
            np.min(np.sqrt((reals[i] - border_points[:, 0])**2 + (imags[i] - border_points[:, 1])**2))
            for i in range(len(reals))
        ])
        # Fallback to original method
        # distances = np.sqrt((point[0] - border_points[:, 0])**2 + (point[1]-border_points[:, 1])**2)
        # return np.min(distances)
    # iters = np.where(distances < 3/100000, base_iter, 15)
    # is_inside = np.where(distances<3/100000, True, False)
    return distances

def weighting_scheme(reals, imags, border_points, base_iter, tree, rand):
    '''
    implements rejection scheme and weight of point that counters the bias. 
    Used for imprtance sampling
    '''
    np.random.seed(rand)
    # iterations, inside = min_distance(reals, imags, border_points, base_iter, tree)  # Boolean array of shape (N,)

    # # Create a random array for rejection sampling
    # random_vals = np.random.rand(len(reals))

    # # Accept points based on the mandelbrot border and rejection probability
    # accepted = inside | (random_vals > rejection_prob)  # Accept if inside or outside but not rejected

    # # Compute weights
    # weights = np.zeros(len(reals))
    # weights[inside] = 1  # Inside points are always accepted with weight 1
    # weights[~inside] = 1 / (1 - rejection_prob)  # Weight for outside points
    # weights[~accepted] = 0  # Rejected points have zero weight

    distances = min_distance(reals, imags, border_points, tree)
    
    # Define weighting function: closer points get higher weight
    max_distance = np.max(distances)
    weights = (1 - (distances / max_distance))**1.4  # Normalize distances and invert for weights
    weights = np.clip(weights, 0.01, 1.0)  # Avoid zero weights, clip to range [0.01, 1.0]
    
    # Iterations based on weights: higher weight means more iterations
    iterations = np.maximum((weights * base_iter).astype(int), 2)
    return reals, imags, iterations, weights

def resample_points(reals, imags, weights, num_samples, iters, rand):
    """
    Resample points based on weights to focus on more important regions.
    """
    np.random.seed(rand)
    probabilities = weights / np.sum(weights)  # Normalize weights to probabilities
    indices = np.random.choice(len(reals), size=num_samples, p=probabilities)
    return reals[indices], imags[indices], probabilities[indices], iters[indices]

def adaptive_mandelbrot_sampling(real, imag, border_points, rand, base_iter, tree=None):
    """ 
    Mandelbrot sampling with adaptive iterations based on border proximity
    """
    # waarde = np.sqrt(len(real))
    # rej = 0.05 if waarde > 1000 else 0.1 if waarde > 500 else 0.2 if waarde > 100 else 0.4
            
    # reals, imags, iters, weights = weighting_scheme(real, imag, border_points, base_iter, rand, tree, rej)
    # number_outside = 0
    # total_numbers = np.sum(weights)
    num_samples = len(real)
    reals, imags, iters, weights = weighting_scheme(real, imag, border_points, base_iter, tree, rand)
    
    # Resample points based on weights
    reals, imags,sampling_prob, iters = resample_points(reals, imags, weights, num_samples, iters, rand)
    normalized_weights = 1 / sampling_prob
    normalized_weights /= np.sum(normalized_weights)  # Normalize to ensure total contribution sums up correctly
    
    total_numbers = np.sum(normalized_weights)
    
    total_numbers = np.sum(normalized_weights)
    number_outside = 0
    c_points = reals + 1j * imags
    for i in range(len(reals)): 
        if weights[i] == 0:
            continue
        c = c_points[i]
        z = 0
        for _ in range(iters[i]):
            z = z**2 + c
            if abs(z) > 2:
                number_outside += normalized_weights[i]
                break
        
    return total_numbers, number_outside


def get_border_points(grid, maxiter, base):
    """
    Generates border points of the Mandelbrot set using high/low iteration comparison
    """
    real = np.linspace(-2.0, 1.0, grid)
    imag = np.linspace(-1.5, 1.5, grid)

    def mandelbrot_border(real_grid, imag_grid, max_iter):
        inside = []
        for i in real_grid:
            for j in imag_grid:
                c = i + 1j * j
                z = 0
                for _ in range(max_iter):
                    z = z**2 + c
                    if abs(z) > 2:
                        break
                else:
                    inside.append((i, j))
        return np.array(inside)

    # Get two sets with different iteration counts
    accurate_set = mandelbrot_border(real, imag, maxiter)
    quick_set = mandelbrot_border(real, imag, base)

    # Get border points through set difference
    set_accurate = set(map(tuple, accurate_set))
    set_quick = set(map(tuple, quick_set))
    border_points = np.array(list(set_quick - set_accurate))

    return border_points