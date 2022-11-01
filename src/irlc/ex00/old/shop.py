class FruitShop: #!s
    def __init__(self, name, prices):
        """
            name: Name of the fruit shop

            prices: Dictionary with keys as fruit
            strings and prices for values e.g.
            {'apples': 2.00, 'oranges': 1.50, 'pears': 1.75}
        """
        self.prices = prices
        self.name = name
        print(f'Welcome to {name} fruit shop')

    def cost_per_pound(self, fruit):
        """
            fruit: Fruit string
        Returns cost of 'fruit', assuming 'fruit'
        is in our inventory or None otherwise
        """
        if fruit not in self.prices:
            raise Exception(f"Oy veh, we do not sell {fruit}")
        return self.prices[fruit]

    def get_name(self):
        return self.name #!s


if __name__ == "__main__": #!s=a #!o=a
    shopName = 'the Berkeley Bowl'
    fruitPrices = {'apples': 1.00, 'oranges': 1.50, 'pears': 1.75}
    berkeleyShop = FruitShop(shopName, fruitPrices)
    applePrice = berkeleyShop.cost_per_pound('apples')
    print(applePrice)
    print('Apples cost $%.2f at %s.' % (applePrice, shopName))

    otherName = 'the Stanford Mall'
    otherFruitPrices = {'kiwis': 6.00, 'apples': 4.50, 'peaches': 8.75}
    otherFruitShop = FruitShop(otherName, otherFruitPrices)
    otherPrice = otherFruitShop.cost_per_pound('apples')
    print(otherPrice)
    print('Apples cost $%.2f at %s.' % (otherPrice, otherName))
    print("My, that's expensive!") #!s #!o
