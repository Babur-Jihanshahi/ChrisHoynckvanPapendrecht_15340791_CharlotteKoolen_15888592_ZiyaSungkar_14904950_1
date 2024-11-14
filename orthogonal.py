import numpy as np
import matplotlib.pyplot as plt


def orthogonal_sample(n, rand):
    MAJOR = n
    SAMPLES = MAJOR * MAJOR
    scale = 3 / SAMPLES
    np.random.seed(rand)

    # Create 2D arrays for xlist and ylist with sequential indices
    xlist = np.array([[i*MAJOR+j for j in range(MAJOR)] for i in range(MAJOR)])
    ylist = xlist.copy()

    # Shuffle each row of xlist and ylist independently
    for i in range(MAJOR):
        np.random.shuffle(xlist[i])  # Shuffle each row of xlist
        np.random.shuffle(ylist[i])  # Shuffle each row of ylist
        
    # Calculate coordinates with jitter
    coordinates = []
    for i in range(MAJOR):
        for j in range(MAJOR):
            # Calculate x and y positions with added stochasticity
            x = -2 + scale * (xlist[i][j] + np.random.uniform(0, 1))
            y = -1.5 + scale * (ylist[j][i]+ np.random.uniform(0, 1))
            coordinates.append((x, y))

    coordinates = np.array(coordinates)
    return coordinates

def plot_orthogonal(coordinates, samples):
    plt.figure(figsize=(6, 8))
    plt.scatter(coordinates[:, 0], coordinates[:, 1])
    plt.xlim(-2, 1)
    plt.ylim(-1.5, 1.5)
    plt.grid()
    plt.xticks(np.linspace(-2, 1, samples+1))  
    plt.yticks(np.linspace(-1.5, 1.5, samples+1)) 
    plt.show()



