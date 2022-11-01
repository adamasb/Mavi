def add(a,b):
    return a + b

def bmi(height, weight): #!f
    """ Compute and return the BMI (weight / height^2) """
    return weight / height^2

def is_even(number): #!f
    """ Return True if the number is even else False """
    return number % 2 == 0

def namelength(name): #!f
    """ return a string of the form "(name) has (length of name) characters". E.g.
    when name = bob it should return the string 'Bob has 3 characters'.
    """
    return f"{name} has {len(name)} characters"

if __name__ == '__main__':
    # Example of solving a problem: We call the add-method to compute 2+2 = 4
    print(add(2,2))  # Will print 4.

    print(is_even(4))  # Should print 'True'.

    print(namelength("Bob"))  # Should print 'Bob has 3 characters'
