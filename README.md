# Stochastic_Assignment_1
## Computing the Area of the Mandelbrot Set

This project investigates the area of the Mandelbrot set using Monte Carlo integration methods and compares various sampling techniques. It also explores importance sampling to improve the convergence rate.

---

## Features

1. **Monte Carlo Integration**:
   - Estimation of the Mandelbrot set area `A_M` by sampling and iterative calculation.
   - Investigation of convergence behavior as the number of iterations (`i`) and samples (`s`) increase.

2. **Sampling Techniques**:
   - **Pure random sampling**.
   - **Latin hypercube sampling**.
   - **Orthogonal sampling**.
   - **Importance sampling**: Uses a probability density function (PDF) based on the Mandelbrot set's structure to prioritize sampling in border regions, and can be applied all sampling methods.

3. **Parallelized Execution**:
   - Functions in `worker.py` enable efficient parallel processing of Monte Carlo simulations.

4. **Statistical Significance Testing**:
   - Conducted using `tests_significance.ipynb` to analyze and compare the accuracy of the different sampling methods.

---

## File Overview

- **`mandelbrot_sim.ipynb`**
  - Main script for computing the area of the Mandelbrot set.
  - Implements Monte Carlo integration with all sampling methods, including importance sampling.
  - Contains visualization code to plot differences in method, sample size and iteration bound. 

- **`worker.py`**:
  - Contains utility functions for parallelized execution of sampling tasks.
  - Provides tools for implementing importance sampling by resampling with the computed PDF based on the border region.

- **`tests_significance.py`**:
  - Performs statistical testing to compare the accuracy, convergence rates, and efficiency of the sampling methods.
  - Evaluates the impact of importance sampling on reducing variance.

- **`orthogonal.py`**:
  - samples according to the orthogonal structure, returns N samples ranging from 0 to 1. 
 

---

## Authors
 - Charlotte Koolen 15888592
 - Chris Hoynck van Papendrecht 15340791
 - Ziya Alim Sungkar 14904950