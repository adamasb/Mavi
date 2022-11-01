import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import irlc.ex00.fruit as fruit
sys.path.append( os.path.dirname(fruit.__file__) )


import fruit #!i=im1
fruit.fruitPrices #!i
from fruit import fruitPrices, buyFruit #!i=im2
buyFruit(fruitPrices, 'apples', 10) # Buy some fruit
from fruit import *  # Import everything.
from fruit import fruitPrices as prices
print(prices) #!i



# Import example from the fruit-package #!s=b
from irlc.ex00.fruit import fruitPrices
from irlc import VideoMonitor #!s
