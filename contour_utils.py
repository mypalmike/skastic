"""
Some opencv contour-related functions.

"""

from queue import Queue

import cv2


# Perimeter of contour below which it's considered a speck
PERIMETER_HEURISTIC = 40.0

# Ratio of area/perimeter of a contour beneath which it is assumed to be a line
LINEAR_HEURISTIC = 5.0

# Contour categories
CONTOUR_SPECK = 1
CONTOUR_LINE = 2
CONTOUR_BOX = 3


def offset_point(point, delta):
  return (point[0] + delta[0], point[1] + delta[1])


def categorize_contour(
    contour,
    perimeter_heuristic=PERIMETER_HEURISTIC,
    linear_heuristic=LINEAR_HEURISTIC):
  """
  Categorize a contour as a box, a line, or speck

  """
  area = cv2.contourArea(contour)
  perimeter = cv2.arcLength(contour, True)

  if perimeter <= perimeter_heuristic:
    return CONTOUR_SPECK

  ratio = area / perimeter
  if ratio < linear_heuristic:
    return CONTOUR_LINE
  else:
    return CONTOUR_BOX


def neighbors(img, point):
  """
  Get a list of neighboring points within the contour. Input point must be
  in the contour. 4-directions only (not diagonals). Contour must be greyscale
  format with white (255) contour.

  """
  results = []
  max_y, max_x = img.shape
  for delta in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
    candidate = offset_point(point, delta)
    if (candidate[0] >= 0 and
        candidate[0] <= max_x and
        candidate[1] >= 0 and
        candidate[1] <= max_y and
        img[candidate[1], candidate[0]] == 255):
      results.append(candidate)
  return results


def farthest_point(img, point):
  """
  Get the farthest connected white point in an image.

  """
  farthest_point = point
  farthest_dist = 0

  # Dict of point_in_contour -> distance (total dx + dy)
  searched = {}

  # Queue for breadth first search
  queue = Queue()

  searched[point] = 0
  queue.put(point)

  # Breadth first search, keeping track of max.
  while not queue.empty():
    curr_point = queue.get()
    curr_dist = searched.get(curr_point)
    for neighbor in neighbors(img, curr_point):
      neighbor_dist = searched.get(neighbor)
      if neighbor_dist is None:
        new_dist = curr_dist + 1
        if new_dist > farthest_dist:
          farthest_dist = new_dist
          farthest_point = neighbor
        searched[neighbor] = new_dist
        queue.put(neighbor)

  return farthest_point


def extreme_points(img, contour):
  """
  For a given contour, return the 2 points farthest apart.
  I believe this algorithm is general (I've yet to come up with edge cases
  which fail this approach), but I've only used it for line-following
  of narrow contours.

  Returns list (size 2) of (x, y) tuples.

  Approach:

  Pick a point of the contour (these are at the edge of the contour space).

  1. Perform a depth-first traversal of the image, assign total pixel distance
  to each pixel from the starting point in the contour. The most distant pixel
  is one endpoint.

  2. Then perform the same algorithm, using this endpoint as the
  starting point. The farthest point on this second pass is the other endpoint.

  Yes I know doing pixel access in python is slow. However, it's fast enough
  for my purposes. I may make a C python module for this some day since it
  seems to be of general use.

  """
  # Start point for search. Could be any point along contour but we pick first.
  x = contour[0][0][0]
  y = contour[0][0][1]
  point = (x, y)

  # Run algorithm to find each end.
  point1 = farthest_point(img, point)
  point2 = farthest_point(img, point1)

  return point1, point2
