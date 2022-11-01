from irlc.ex00.shop import FruitShop #!s #!o #!i

prices = {'apples': 1.00, 'oranges': 1.50, 'pears': 1.75}
shopA = FruitShop(name='the Berkeley Bowl', prices=prices)
applePrice = shopA.cost_per_pound('apples')
print(f'Apples cost {applePrice} at {shopA.name}')

other_prices = {'kiwis': 6.00, 'apples': 4.50, 'peaches': 8.75}
shopB = FruitShop(name='the Stanford Mall', prices=other_prices)
otherPrice = shopB.cost_per_pound('apples')
print(f'Apples cost {otherPrice} at {shopB.name}') #!s #!o #!i

prices = {'apples': 1.00, 'oranges': 1.50, 'pears': 1.75}  #!i=c1
shop = FruitShop(name='The brown banana', prices=prices)
shop.cost_per_pound("oranges")
FruitShop.cost_per_pound(shop, "oranges")  # Does exactly the same as the line above #!i

