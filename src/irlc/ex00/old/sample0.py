seq = "?0?"
k = seq.count("?")
seq = [int(s) if s != "?" else s for s in seq] #.split()
dx = [i for i in range(len(seq)) if seq[i] == "?"]
def swaps(ss):
    swaps = 0
    dx = len(ss)-2
    while True:
        if ss[dx] == ss[dx+1]:
            dx -= 1
        elif ss[dx] == 1 and ss[dx+1] == 0:
            swaps += 1
            ss[dx] = 0
            ss[dx+1] = 1
            dx += 1
            dx = min(len(ss)-2, dx)
        else:
            dx -= 1
        if dx == -1:
            break
    return swaps

nswaps = 0
for j in range(2**k):
    s = bin(j)[2:]
    s = (k-len(s))*"0" + s

    for i,dex in enumerate(dx):
        seq[dex] = int(s[i])
    nswaps += swaps(seq)
print(nswaps)


import numpy as np
np.ndarray

from pytorch import Tensor

from torch import Tensor


3 + 5  #!i=a
a = "hello there"
print(a)
def add(x,y):
    x = x + 3
    y = y - 2
    return x + y
print(add(5, 9))
a += "multifun" #!i

1 + 1 #!i=a1
2 * 3
7 / 2
7 // 2 #!i

1 == 0 #!i=a2
not (1 == 0)
(2 == 2) and (2 == 3)
(2 == 2) or (2 == 3) #!i

'artificial' + "intelligence" #!i=a3 #!i

'artificial'.upper() #!i=a4
'HELP'.lower()
len('Help')
"machine learning".find("earn") #!i

s = 'hello world' #!i=a5
print(s)
s.upper()
len(s.upper())
num = 8.0
num += 2.5
print(num) #!i

s = 'abc' #!i=a6
dir(s)
help(s.find) #!i

fruits = ['apple', 'orange', 'pear', 'banana'] #!i=b1
fruits[0] #!i

otherFruits = ['kiwi', 'strawberry'] #!i=b2
fruits + otherFruits
['apple', 'orange', 'pear', 'banana', 'kiwi', 'strawberry'] #!i

fruits[-2] #!i=b3
fruits.pop()
fruits
fruits.append('grapefruit')
fruits
fruits[-1] = 'pineapple'
fruits #!i

fruits[0:2] #!i=b4
fruits[:3]
fruits[2:]
len(fruits) #!i

lstOfLsts = [['a', 'b', 'c'], [1, 2, 3], ['one', 'two', 'three']] #!i=b5
lstOfLsts[1][2]
lstOfLsts[0].pop()
lstOfLsts #!i

lst = ['a', 'b', 'c'] #!i=b5
lst.reverse()
['c', 'b', 'a'] #!i

pair = (3, 5) #!i=b6
pair[0]
x, y = pair
x
pair[1] = 6 #!i

shapes = ['circle', 'square', 'triangle', 'circle'] #!i=b7
setOfShapes = set(shapes)
setOfShapes_alternative = {'circle', 'square', 'triangle', 'circle'} #!i

setOfShapes #!i=b8
setOfShapes.add('polygon')
setOfShapes
'circle' in setOfShapes
'rhombus' in setOfShapes
favoriteShapes = ['circle', 'triangle', 'hexagon']
setOfFavoriteShapes = set(favoriteShapes)
setOfShapes - setOfFavoriteShapes
setOfShapes & setOfFavoriteShapes
setOfShapes | setOfFavoriteShapes #!i

studentIds = {'knuth': 42.0, 'turing': 56.0, 'nash': 92.0} #!i=b9
studentIds['turing']
studentIds['nash'] = 'ninety-two'
studentIds
del studentIds['knuth']
studentIds
studentIds['knuth'] = [42.0, 'forty-two']
studentIds
studentIds.keys()
studentIds.values()
studentIds.items()
len(studentIds) #!i





