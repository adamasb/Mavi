import numpy as np

if __name__ == "__main__":
    """
    Use np.random.randint to generate random values from a six-sided die. 
    Compute four quantities: 
        
    mu/var: True (exact) mean and variance of the die
    muh/varh: Estimated mean/variance based on T samples.
    """
    T = 10000  # number of samples to use
    #!b
    mu = np.sum( [k/6 for k in range(1, 7)] )
    var = np.sum( [1/6 * (k - mu)**2 for k in range(1, 7)] )
    rolls = [np.random.randint(1, 7) for _ in range(T)]
    muh = np.mean(rolls)
    varh = np.mean([(r-muh)**2 for r in rolls])
    #!b Compute true/estimated values here
    print("mean true value: ", mu, "estimated value", muh) #!o=die
    print("var true value: ", var, "estimated value", varh) #!o

    """
    Next, we will compute the fourth central moment of |x-mu|^p.    
    
    True value will be denoted E_true, sampled value E_mc. 
    
    In computing this value, we will assume mu=0 without loss of generality.    
    """
    p = 4
    sigma = 2
    from scipy.special import gamma  # gamma function
    E_true = sigma**p * (2 ** (p/2) * gamma( (p+1)/2 ) ) / np.sqrt( np.pi) #!b
    E_mc = np.mean( [ np.abs( np.random.randn()*sigma )**p for _ in range(T)] ) #!b Compute central moment here

    print("True value of E[|x|^p] using N(0,sigma)", E_true) #!o=moment
    print("MC estimate of E[|x|^p]", E_mc) #!o
