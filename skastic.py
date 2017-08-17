#!/usr/bin/env python

import cv2
import numpy as np
from queue import Queue
import random
import sys


# Ratio of area/perimeter of a contour beneath which it is assumed to be a line
LINEAR_HEURISTIC = 5.0

# Perimeter of contour below which it's considered spurious
PERIMETER_HEURISTIC = 40.0

# Contour categories
CONTOUR_JUNK = 1
CONTOUR_LINE = 2
CONTOUR_BOX = 3

# Marker for edge node
EDGE_NODE = -1


class CompilerException(Exception):
  pass


# Categorize a contour as a box, a line, or junk (small spots in an image)
def categorize_contour(contour):
  area = cv2.contourArea(contour)
  perimeter = cv2.arcLength(contour, True)

  if perimeter < PERIMETER_HEURISTIC:
    return CONTOUR_JUNK

  ratio = area / perimeter
  if ratio < LINEAR_HEURISTIC:
    return CONTOUR_LINE
  else:
    return CONTOUR_BOX


# Get a random BGR tuple as a color. Can specify fixed value for any component.
def rand_color(blue=None, green=None, red=None):
  if blue is None:
    blue = random.randint(0, 255)
  if green is None:
    green = random.randint(0, 255)
  if red is None:
    red = random.randint(0, 255)
  return (blue, green, red)


def offset_point(point, delta):
  return (point[0] + delta[0], point[1] + delta[1])


class ContourNode:
  def __init__(self, img_contour, contour):
    self.img_contour = img_contour
    self.contour = contour
    self.children = []

    # Centroid computation from the opencv docs on contour attributes. (i.e. I have no clue)
    moments = cv2.moments(contour)
    self.centroid = (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))


class ContourLine:
  def __init__(self, img_contour, contour):
    self.img_contour = img_contour
    self.contour = contour
    self.endpoints = [None, None]  # (x, y) coordinates of the 2 endpoints
    self.nodes = [None, None]  # Connected nodes in graph

    self.find_endpoints()

  def neighbors(self, point):
    results = []
    max_y, max_x = self.img_contour.shape
    for delta in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
      candidate = offset_point(point, delta)
      if (candidate[0] >= 0 and
          candidate[0] <= max_x and
          candidate[1] >= 0 and
          candidate[1] <= max_y and
          self.img_contour[candidate[1], candidate[0]] == 255):
        results.append(candidate)
    return results

  # Use breadth-first search style shortest path algorithm to find farthest point.
  def farthest(self, point):
    farthest_point = point
    farthest_dist = 0

    searched = {}
    queue = Queue()

    searched[point] = 0
    queue.put(point)

    while not queue.empty():
      curr_point = queue.get()
      curr_dist = searched.get(curr_point)
      for neighbor in self.neighbors(curr_point):
        neighbor_dist = searched.get(neighbor)
        if neighbor_dist is None:
          new_dist = curr_dist + 1
          if new_dist > farthest_dist:
            farthest_dist = new_dist
            farthest_point = neighbor
          searched[neighbor] = new_dist
          queue.put(neighbor)

    return farthest_point

  def find_endpoints(self):
    # Current approach:
    # Pick a point of the contour (these are at the edge of the contour space).
    # 1. Perform a depth-first traversal of the image, assign total pixel distance
    # to each pixel from the starting point in the contour. The most distant pixel
    # is one endpoint. Then perform the same algorithm, using this endpoint as the
    # starting point. The farthest point is the other endpoint.
    # Yes I know doing pixel access in python is slow. I may make a C python module
    # for this since it seems to be of general use.

    # First, pick start point for search.isn't
    x = self.contour[0][0][0]
    y = self.contour[0][0][1]
    point = (x, y)

    self.endpoints[0] = self.farthest(point)
    self.endpoints[1] = self.farthest(self.endpoints[0])


def image_to_objects(filename):
  visual_analysis = VisualAnalysis(filename)


# Returns an ast.
def image_to_ast(filename):
  pass


class VisualAnalysis:
  def __init__(self, filename):
    # Load image, then do various conversions and thresholding.
    self.img_orig = cv2.imread(filename, cv2.IMREAD_COLOR)

    if self.img_orig is None:
      raise CompilerException("File '{}' not found".format(filename))

    self.img_grey = cv2.cvtColor(self.img_orig, cv2.COLOR_BGR2GRAY)
    # _, self.img_contour = cv2.threshold(self.img_grey, 240, 255, cv2.THRESH_BINARY)
    _, self.img_contour = cv2.threshold(self.img_grey, 250, 255, cv2.THRESH_BINARY_INV)
    # _, self.img_lines = cv2.threshold(self.img_grey, 100, 240, cv2.THRESH_BINARY)

    self.root_node = None

    self.contours = self.find_contours()
    # self.keypoints = self.detect_blobs()
    # self.lsd, self.lines = self.detect_lines()

    self.contour_lines, self.contour_nodes = self.categorize_contours()

    self.build_graph()
    self.build_parse_tree()


  def find_contours(self):
    _, contours, _ = cv2.findContours(
        self.img_contour,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )
    return contours

  def draw_contours(self, img):
    for contour_line in self.contour_lines:
      cv2.drawContours(img, [contour_line.contour], 0, rand_color(red=0), 3)

    # for index, contour in enumerate(self.contours):
    #   contour_category = categorize_contour(contour)

    #   if contour_category != CONTOUR_JUNK:
    #     if contour_category == CONTOUR_LINE:
    #       color = rand_color(red=0) # (0, 0, 255)
    #     else:
    #       color = (0, 0, 255)

    #     cv2.drawContours(img, self.contours, index, color, 3)

  def draw_lines(self, img):
    for contour_line in self.contour_lines:
      cv2.line(img, contour_line.endpoints[0], contour_line.endpoints[1], (255, 0, 255), 3)

  def draw_connections(self, img):
    for contour_line in self.contour_lines:
      if (contour_line.nodes[0] is not None and contour_line.nodes[0] != EDGE_NODE and
          contour_line.nodes[1] is not None and contour_line.nodes[1] != EDGE_NODE):
        pt1 = contour_line.nodes[0].centroid
        pt2 = contour_line.nodes[1].centroid
        cv2.line(img, pt1, pt2, (0, 255, 255), 4)

  def draw_parse_tree(self, img, node, depth):
    brightness = 50
    for child_node in node.children:
      pt1 = node.centroid
      pt2 = child_node.centroid
      color = (brightness, brightness, brightness)
      cv2.line(img, pt1, pt2, color, 2)
      self.draw_parse_tree(img, child_node, depth + 1)
      brightness += 50


  # Categorize contours.
  # Returns a tuple of [ContourLine...], [ContourNode...]
  def categorize_contours(self):
    print ("categorize_contours")
    contour_lines = []
    contour_nodes = []

    for index, contour in enumerate(self.contours):
      category = categorize_contour(contour)
      if category == CONTOUR_LINE:
        contour_lines.append(ContourLine(self.img_contour, contour))
      elif category == CONTOUR_BOX:
        contour_nodes.append(ContourNode(self.img_contour, contour))

    return contour_lines, contour_nodes

  def build_graph(self):
    # Currently O(n * m) which is sad. Spatial partitioning tree (kdtree or quadtree) on node
    # locations would make O(m * log n). M and N are small enough in most cases that this
    # is fast enough for now.
    for line in self.contour_lines:
      for index, endpoint in enumerate(line.endpoints):
        # Find node with centroid closest to this endpoint.
        closest_node = None
        closest_sq = sys.float_info.max
        for node in self.contour_nodes:
          dx = endpoint[0] - node.centroid[0]
          dy = endpoint[1] - node.centroid[1]
          dist_sq = dx * dx + dy * dy
          if dist_sq < closest_sq:
            closest_node = node
            closest_sq = dist_sq

        # Check for root node (closer to top edge of image than to any labeled node)
        edge_dist_sq = endpoint[1] * endpoint[1]
        if edge_dist_sq < closest_sq:
          closest_node = EDGE_NODE

        line.nodes[index] = closest_node

  def connected_nodes(self, match_node):
    # Linear search. Yuck.
    nodes = []
    for line in self.contour_lines:
      for index, node in enumerate(line.nodes):
        if match_node is node:
          nodes.append(line.nodes[1 - index])

    return nodes

  def build_parse_tree(self):
    # All sorts of linear digging here due to non-optimal graph rep. But I'm
    # cranking out code right now and will not be stopped. Refactor for speed later.
    self.root_node = None
    edge_line = None
    for line in self.contour_lines:
      for index, node in enumerate(line.nodes):
        if node == EDGE_NODE:
          edge_line = line
          self.root_node = line.nodes[1 - index]
          break
      if edge_line:
        break  # I don't like this control flow. Should break out a function.

    if not self.root_node:
      raise CompilerException("Root node not found")

    # remaining_lines = set(self.contour_lines)
    # remaining_lines.remove(edge_line)

    # remaining_nodes = set(self.contour_nodes)
    # remaining_nodes.remove(root_node)
    visited_nodes = set()

    stack = [self.root_node]

    while stack:
      print("Processing a node")
      curr_node = stack.pop()
      visited_nodes.add(curr_node)
      connected_nodes = self.connected_nodes(curr_node)
      for connected_node in connected_nodes:
        if connected_node != EDGE_NODE and not connected_node in visited_nodes:
          curr_node.children.append(connected_node)
          stack.append(connected_node)

    # We have parent hierarchy, but child order is currently arbitrary.
    # Sort by node centroid, left to right.
    # Ideally, we would use angle of connection, allowing for long lines which
    # are not right-to-left. But for now, this will do.
    for node in self.contour_nodes:
      node.children.sort(key=lambda x: x.centroid[0])

  def draw(self):
    img_out = self.img_orig.copy()
    self.draw_contours(img_out)
    self.draw_lines(img_out)
    self.draw_connections(img_out)
    self.draw_parse_tree(img_out, self.root_node, 0)
    cv2.imshow("Contours", img_out)
    cv2.imshow("Mask", self.img_contour)

    # im_out = cv2.drawKeypoints(self.img_orig, self.keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # self.lsd.drawSegments(im_out, self.lines)
    # cv2.imshow("Keypoints", im_out)


# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

def main(args=sys.argv[1:]):
  filename = args[0]
  visual_analysis = VisualAnalysis(filename)
  visual_analysis.draw()
  cv2.waitKey(0)

if __name__ == '__main__':
  main()


  # def detect_blobs(self):
  #   # Setup SimpleBlobDetector parameters.
  #   params = cv2.SimpleBlobDetector_Params()

  #   # Change thresholds
  #   # params.minThreshold = 100
  #   # params.maxThreshold = 250


  #   # Filter by Area.
  #   params.filterByArea = True
  #   params.minArea = 100
  #   params.maxArea = 100000

  #   # Filter by Circularity
  #   params.filterByCircularity = True
  #   params.minCircularity = 0.2

  #   # Filter by Convexity
  #   params.filterByConvexity = True
  #   params.minConvexity = 0.01

  #   # Filter by Inertia
  #   params.filterByInertia = True
  #   params.minInertiaRatio = 0.01

  #   ver = (cv2.__version__).split('.')
  #   if int(ver[0]) < 3 :
  #     detector = cv2.SimpleBlobDetector(params)
  #   else :
  #     detector = cv2.SimpleBlobDetector_create(params)

  #   keypoints = detector.detect(self.img_contour)

  #   print(keypoints[0])

  #   return keypoints

  # def detect_lines(self):
  #   lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
  #   lines, width, prec, nfa = lsd.detect(self.img_lines)

  #   return lsd, lines
