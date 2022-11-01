from irlc.ex00.shop import FruitShop  #!s=fancy
from irlc.ex00.shop_test import prices

class FancyFruitShop(FruitShop):
    def __init__(self, name, prices):
        super().__init__(name, prices)  # call super-class __init___
        print("(This is a fine establishment)")

    def cost_per_pound(self, fruit):  # Using inline if for brevity
        return 2 * self.prices[fruit] if fruit in self.prices else None

if __name__ == "__main__":
    shopA = FruitShop("The Brown Banana", prices) #!o
    shopB = FancyFruitShop("The Golden Delicious", prices)

    print("Shopping at: ", shopA.get_name(), "price of apples", shopA.cost_per_pound('apples'))
    print("Shopping at: ", shopB.get_name(), "price of apples", shopB.cost_per_pound('apples')) #!s #!o
