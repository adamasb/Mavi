nums = [1, 2, 3, 4, 5, 6] #!s #!o
plusOneNums = [x + 1 for x in nums]

oddNums = [x for x in nums if x % 2 == 1]
print(oddNums)
oddNumsPlusOne = [x + 1 for x in nums if x % 2 == 1]
print(oddNumsPlusOne) #!s #!o

""" 
Dictionary comprehension. We make a new dictionary where both the keys and values may be changed
"""
toy_cost = {'toy car': 10, 'skipping rope': 6, 'toy train': 20} #!s=a #!o=a
print(toy_cost)

double_cost = {k: 2*v for k, v in toy_cost.items()}
print(double_cost)

bad_toys = {"broken "+k: v//2 for k, v in toy_cost.items()}
print(bad_toys)

expensive_toys = {k: v for k, v in toy_cost.items() if v >= 10}
print(expensive_toys) #!s #!o