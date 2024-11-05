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
    number_inside = 0
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
                    number_inside+=1
                    # mandelbrot_set[i, j] = False
                    # iteration_count[i, j] =iteration
                    break
            # else:
                # mandelbrot_set[i, j] = True
                # iteration_count[i, j] = max_iter
    return (total_numbers, number_inside)



def worker_function(xs):
    cpts, s = xs
    y = cpts.shape[0]
    y1 = cpts.shape[1]

    total_numbs, number_ins = mandelbrot(cpts,s, 2)
    return (total_numbs, number_ins)