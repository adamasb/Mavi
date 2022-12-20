
def ex_dist(l):
  # l = 8

  sum_distances = 0
  for x1 in range(1, l+1):
    for y1 in range(1, l+1):
      for x2 in range(1, l+1):
        for y2 in range(1, l+1):
          distance = abs(x1 - x2) + abs(y1 - y2)
          sum_distances += distance
    # print(x1)

  expected_distance = sum_distances / (l**4)
  # print(expected_distance)
  return expected_distance