import numpy as np

"""
Test if we can import next weeks instructor. If this does not work, fix your PYTHONPATH to properly add irlc as a package
"""


def add(a,b):
    return a+b

def product(a,b): #!f Finish function body
    """
    This is the generic form of an exercise. Add code to return the product of a and b.
    """
    return a*b #!s=a #!s

if __name__ == "__main__":
    a, b = 4, 5 #!o
    print(f"Sum of {a} and {b} is", add(a,b))
    print(f"Product of {a} and {b} is", product(a, b))

    l = [0, 1, 1, 0, 4, 5, 6]  # make and show a simple list
    print(l)
    print([x ** 2 for x in l])  # computes a list of squares
    print("length of l", len(l))  # length function

    """ sets """
    s = set(l) # s is now a set.
    for x in s:  # iterate over a list
        print(x)
    print("len of s", len(s))
    s2 = {1, 5, 3, 2}  # alternative way to inialize a set

    """ dictionaries, used to store key:value pairs """
    d = {'mary': 4, 'john': 2, 'michael': 0}    # Explicitly initialize a dictionary
    print("number of children john has: ", d['john'])  # access an element
    for k in d:   # iterate over keys
        print(k, d[k] )  # show keys and values

    for k, item in d.items(): # alternative code for iterating over a dictionary
        print(k, item)

    print( {name: c+1 for name, c in d.items()} )  # dictionary comprehension

    print("a way to represent a function which maps from x in s to the cosine of x:")
    f = {x: np.cos(x) for x in s}  # f(x) = cos(x)
    print("cos(1):", f[1])  # using the function we just defined #!o
