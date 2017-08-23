import sys

import cv2
import numpy as np

# from ast_utils import recognized
from contour_utils import categorize_contour, CONTOUR_BOX, CONTOUR_LINE
from graph import ContourLine, ContourNode, EDGE_NODE
from misc import CompilerException, rand_bgr

class Page:
  def __init__(self, filename, analyze_only):
    self.externs = set()
    self.analyze_only = analyze_only
    self.load(filename, analyze_only)

  def load(self, filename, analyze_only):
    # Load image, then do various conversions and thresholding.
    self.img_orig = cv2.imread(filename, cv2.IMREAD_COLOR)

    if self.img_orig is None:
      raise CompilerException("File '{}' not found".format(filename))

    self.img_grey = cv2.cvtColor(self.img_orig, cv2.COLOR_BGR2GRAY)
    _, self.img_contour = cv2.threshold(self.img_grey, 250, 255, cv2.THRESH_BINARY_INV)
    _, self.img_text = cv2.threshold(self.img_grey, 150, 255, cv2.THRESH_BINARY)
    self.root_node = None

    self.contours = self.find_contours()

    self.contour_lines, self.contour_nodes = self.categorize_contours()

    self.build_graph()
    self.build_parse_tree()

    self.parse_nodes()

    if not analyze_only:
      self.python_ast = self.root_node.to_python_ast()

  def find_contours(self):
    _, contours, _ = cv2.findContours(
        self.img_contour,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )
    return contours

  def text_mask_img(self):
    img = self.img_grey.copy()
    mask = np.ones(img.shape[:2], dtype="uint8") * 255
    for contour_node in self.contour_nodes:
      cv2.drawContours(mask, contour_node.contour, -1, 0, -1)
    image = cv2.bitwise_and(img, img, mask=mask)
    return img

  def draw_contours(self, img):
    for contour_node in self.contour_nodes:
      cv2.drawContours(img, [contour_node.contour], 0, (0, 0, 255), 3)
    for contour_line in self.contour_lines:
      cv2.drawContours(img, [contour_line.contour], 0, rand_bgr(red=0), 3)

  def draw_lines(self, img):
    for contour_line in self.contour_lines:
      cv2.line(img, contour_line.endpoints[0], contour_line.endpoints[1], (255, 0, 255), 3)

  def draw_connections(self, img):
    for contour_line in self.contour_lines:
      if (contour_line.nodes[0] is not None and contour_line.nodes[0] != EDGE_NODE and
          contour_line.nodes[1] is not None and contour_line.nodes[1] != EDGE_NODE):
        pt1 = contour_line.nodes[0].centroid
        pt2 = contour_line.nodes[1].centroid
        cv2.line(img, pt1, pt2, rand_bgr(), 4)

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

    visited_nodes = set()

    stack = [self.root_node]

    while stack:
      curr_node = stack.pop()
      visited_nodes.add(curr_node)
      connected_nodes = self.connected_nodes(curr_node)
      for connected_node in connected_nodes:
        if connected_node != EDGE_NODE and not connected_node in visited_nodes:
          curr_node.children.append(connected_node)
          stack.append(connected_node)

    # Now we have parent hierarchy, but child order is currently arbitrary.
    # Sort by x position of child node centroid, left to right.
    for node in self.contour_nodes:
      node.children.sort(key=lambda x: x.centroid[0])

  def parse_nodes(self):
    for node in self.contour_nodes:
      node.parse_text(self.img_text)
      # self.check_externs(node.text)

  # def check_externs(self, text):
  #   if not recognized(text):
  #     self.externs.add(text)

  # def dump_parse_tree_text(self, node):
  #   print ('{} : {}'.format(id(node), node.text))
  #   for child_node in node.children:
  #     self.dump_parse_tree_text(child_node)

  def draw(self, which):
    which = set(which)
    img_out = self.img_orig.copy()
    # if contours, lines, connections, parse-tree
    if 'contours' in which:
      self.draw_contours(img_out)
    if 'lines' in which:
      self.draw_lines(img_out)
    if 'connections' in which:
      self.draw_connections(img_out)
    if 'parse-tree' in which:
      self.draw_parse_tree(img_out, self.root_node, 0)
    cv2.imshow("Debug", img_out)

    # cv2.imshow("Mask", self.img_contour)
    # cv2.imshow("Text", self.img_text)

