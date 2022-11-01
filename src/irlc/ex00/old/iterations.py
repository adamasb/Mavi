fruits = ['apples', 'oranges', 'pears'] #!s #!o

# Iterate i=0, 1, 2, ...
for i in range(len(fruits)):
    print(i, fruits[i])
print("---")
# Iterate over index and value using enumerate:
for i, fruit in enumerate(fruits):
    print(i, fruit) #!s #!o