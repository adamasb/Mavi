fruitPrices = {'apples': 2.00, 'oranges': 1.50, 'pears': 1.75} #!s #!o

def fruit_cost(fruitPrices, fruit, numPounds=1):
    if fruit in fruitPrices:
        return fruitPrices[fruit] * numPounds
    else:
        # Throw an exception and stop the program
        raise Exception("What do you want me to do? I don't know this", fruit)

def better_buy_fruit(fruitPrices, fruit, numPounds=1):
    if fruit not in fruitPrices:
        return f"Sorry we don't have {fruit}"
    else:
        cost = fruit_cost(fruitPrices, fruit, numPounds)
        return f"That'll be {cost} please"

if __name__ == '__main__':
    print(better_buy_fruit(fruitPrices, 'apples', 1))
    s = better_buy_fruit(fruitPrices, 'apples') # Use default value.
    print(s.upper()) #!s #!o
