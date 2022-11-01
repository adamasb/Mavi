import matplotlib.pyplot as plt
import numpy as np
from irlc import savepdf

if __name__ == "__main__" and False:
    """
    Compute pi. Simplest idea is to generate 
    
    x = T x 2 
    
    matrix of uniform random numbers in the [-1,1]^2 cube and use them with the MC principle
    """
    T = 10000
    x = np.random.uniform(-1, 1, (T, 2))  #!b
    pih = 4*np.mean( np.sum(x ** 2,1)<1)  #!b
    print("True value of pi=", np.pi, "mcmc approximation", pih) #!o=a #!o
    x = x[np.sum(x ** 2,1)<1, :]
    plt.plot(x[:,0], x[:,1], 'k.')
    plt.axis('equal')
    plt.title('computing pi')
    savepdf("pi.pdf")
    plt.show()
    """
    Viral math puzzle problem from
    https://www.youtube.com/watch?v=cPNdvdYn05c
    """
    d = 10  # side length of square in problem
    x = np.random.uniform(0, d, (T, 2))  #!b
    def within(x, a, b, r):
        v = np.asarray([a,b])[np.newaxis]
        return np.sqrt(np.sum( (x - v)**2,1)) < r
    I = within(x,0,0,d) & within(x,d//2,d,d//2) & ~within(x,0,d//2,d//2)
    x = x[I,:]
    area = np.mean(I) * d**2 #!b your solution here
    print("Monte carlo solution to puzzle", area) #!o=puzzle #!o
    plt.plot(x[:, 0], x[:, 1], 'r.')
    plt.axis([0, 10, 0, 10])
    plt.gca().set_aspect('equal', 'box')
    plt.title('The viral puzzle')
    savepdf("viral.pdf")
    plt.show()
