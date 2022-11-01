from irlc.ex00.foreach import expensive_prices  #!s=a #!o=a
from irlc.ex00.fruit import fruitPrices, buyFruit
from irlc.ex00.fruit_args import better_buy_fruit

if __name__ == "__main__":
    print('>> We are now in __name__ == "__main__"')
    buyFruit(fruitPrices, 'apples', 2)
    print(better_buy_fruit(fruitPrices, 'apples', numPounds=2))
    print(better_buy_fruit(expensive_prices, 'apples', numPounds=2)) #!s #!o
