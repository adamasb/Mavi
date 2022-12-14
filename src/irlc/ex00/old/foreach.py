# This is what a comment looks like #!s #!o
""" Technically, this is a multiline python string, however,
since it supports linebreaks it is also a useful way to write long
comments, although some people might frown upon it. """
fruits = ['apples', 'oranges', 'pears', 'bananas']
for fruit in fruits:
    print(fruit + ' for sale')

expensive_prices = {'apples': 3.40, 'oranges': 2.20, 'pears': 2.90}
for fruit, price in expensive_prices.items():
    if price < 2.00:
        print('%s cost %f a pound' % (fruit, price))
    else:
        print(fruit + ' are too expensive!') #!s  #!o

