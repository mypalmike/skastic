#!/usr/bin/env python

import ast
# from queue import Queue
import random
import subprocess
import sys

import cv2
import numpy as np

from contour_utils import categorize_contour, CONTOUR_BOX, CONTOUR_LINE, extreme_points


# Marker for edge node
EDGE_NODE = -1


class CompilerException(Exception):
  pass


def rand_bgr(blue=None, green=None, red=None):
  """
  Random BGR tuple as a color. Can specify fixed value for any component.

  """
  if blue is None:
    blue = random.randint(0, 255)
  if green is None:
    green = random.randint(0, 255)
  if red is None:
    red = random.randint(0, 255)
  return (blue, green, red)


class ContourNode:
  def __init__(self, img_contour, contour):
    self.img_contour = img_contour
    self.contour = contour
    self.children = []
    self.text = None

    # Centroid computation from the opencv docs on contour attributes. (i.e. I have no clue)
    moments = cv2.moments(contour)
    self.centroid = (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))

  def save_bounding_box(self, img, filename):
    x, y, width, height = cv2.boundingRect(self.contour)
    bounding_box = img[y:y+height, x:x+width]
    cv2.imwrite(filename, bounding_box)

  def parse_text(self, img):
    filename = "obj_{}.png".format(id(self))
    self.save_bounding_box(img, filename)
    cmd = [
        'tesseract',
        filename,
        'stdout',
        '--psm',
        '7',
        '-l',
        'eng+equ',
        '-c',
        'tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz0123456789=+-*/'
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    self.text = result.stdout.decode('ascii').strip().lower() or ''

  def to_python_ast(self):
    # print(self.text)
    if self.text == 'define':
      fn_node = self.children[0]
      fn_name = fn_node.text
      param_nodes = fn_node.children
      ast_args = ast.arguments(
          args=[ast.arg(arg=x.text, annotation=None) for x in param_nodes],
          vararg=None,
          kwonlyargs=[],
          kw_defaults=[],
          kwarg=None,
          defaults=[])
      return ast.FunctionDef(
          name=fn_name,
          args=ast_args,
          body=[ast.Return(value=self.children[1].to_python_ast())],
          decorator_list=[],
          returns=None)
    elif self.text == 'if':
      tst = self.children[0].to_python_ast()
      body = self.children[1].to_python_ast()
      if len(self.children) == 3:
        orelse = self.children[2].to_python_ast()
      else:
        orelse = ast.NameConstant(value=None)
      return ast.IfExp(
          test=tst,
          body=body,
          orelse=orelse)
    elif self.text == 'true':
      return ast.NameConstant(value=True)
    elif self.text == 'false':
      return ast.NameConstant(value=False)
    elif self.text.isdigit():
      return ast.Num(
          n=int(self.text))
    elif self.text == '=':
      left = self.children[0].to_python_ast()
      right = self.children[1].to_python_ast()
      return ast.Compare(
          left=left,
          ops=[ast.Eq()],
          comparators=[right])
    elif self.text == '+':
      left = self.children[0].to_python_ast()
      right = self.children[1].to_python_ast()
      return ast.BinOp(
          left=left,
          op=ast.Add(),
          right=right)
    elif self.text == '-':
      left = self.children[0].to_python_ast()
      right = self.children[1].to_python_ast()
      return ast.BinOp(
          left=left,
          op=ast.Sub(),
          right=right)
    elif self.text == '*':
      left = self.children[0].to_python_ast()
      right = self.children[1].to_python_ast()
      return ast.BinOp(
          left=left,
          op=ast.Mult(),
          right=right)
    else:
      if len(self.children) == 0:
        # Local var
        return ast.Name(
            id=self.text,
            ctx=ast.Load())
      else:
        # Function call
        if self.children[0].text == '':
          # Empty parameters if child is blank.
          args = []
        else:
          args = [x.to_python_ast() for x in self.children]
          return ast.Call(
              func=ast.Name(
                  id=self.text,
                  ctx=ast.Load()),
              args=args,
              keywords=[])


class ContourLine:
  def __init__(self, img_contour, contour):
    self.img_contour = img_contour
    self.contour = contour
    self.nodes = [None, None]  # Connected nodes in graph
    self.endpoints = extreme_points(self.img_contour, self.contour)


class VisualAnalysis:
  def __init__(self, filename):
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

    python_ast = self.root_node.to_python_ast()

    # HACK TO TEST FACTORIAL FUNCTION
    # test_ast = ast.Module(
    #     body=[
    #         python_ast,
    #         ast.Expr(
    #             value=ast.Call(
    #                 func=ast.Name(id='print', ctx=ast.Load()),
    #                 args=[
    #                     ast.Call(
    #                         func=ast.Name(id='add1', ctx=ast.Load()),
    #                         args=[ast.Num(n=1)],
    #                         keywords=[])],
    #                 keywords=[]))])

    test_ast = ast.Module(
        body=[
            ast.Expr(
                value=ast.Call(
                    func=ast.Name(id='print', ctx=ast.Load()),
                    args=[python_ast],
                    keywords=[]))])

    ast.fix_missing_locations(test_ast)
    exec(compile(test_ast, filename="<ast>", mode="exec"))

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

  def dump_parse_tree_text(self, node):
    print ('{} : {}'.format(id(node), node.text))
    for child_node in node.children:
      self.dump_parse_tree_text(child_node)

  def draw(self):
    img_out = self.img_orig.copy()
    self.draw_contours(img_out)
    self.draw_lines(img_out)
    self.draw_connections(img_out)
    self.draw_parse_tree(img_out, self.root_node, 0)
    cv2.imshow("Contours", img_out)
    cv2.imshow("Mask", self.img_contour)
    cv2.imshow("Text", self.img_text)


def main(args=sys.argv[1:]):
  filename = args[0]
  visual_analysis = VisualAnalysis(filename)
  visual_analysis.draw()
  cv2.waitKey(0)


if __name__ == '__main__':
  main()
