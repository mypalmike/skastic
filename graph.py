import ast
import subprocess

import cv2

from contour_utils import extreme_points


# Marker for edge node
EDGE_NODE = -1

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
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
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
          returns=None,
          lineno=self.centroid[1],
          col_offset=self.centroid[0])
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
          orelse=orelse,
          lineno=self.centroid[1],
          col_offset=self.centroid[0])
    elif self.text == 'true':
      return ast.NameConstant(
          value=True,
          lineno=self.centroid[1],
          col_offset=self.centroid[0])
    elif self.text == 'false':
      return ast.NameConstant(
          value=False,
          lineno=self.centroid[1],
          col_offset=self.centroid[0])
    elif self.text.isdigit():
      return ast.Num(
          n=int(self.text),
          lineno=self.centroid[1],
          col_offset=self.centroid[0])
    elif self.text == '=':
      left = self.children[0].to_python_ast()
      right = self.children[1].to_python_ast()
      return ast.Compare(
          left=left,
          ops=[ast.Eq()],
          comparators=[right],
          lineno=self.centroid[1],
          col_offset=self.centroid[0])
    elif self.text == '+':
      left = self.children[0].to_python_ast()
      right = self.children[1].to_python_ast()
      return ast.BinOp(
          left=left,
          op=ast.Add(),
          right=right,
          lineno=self.centroid[1],
          col_offset=self.centroid[0])
    elif self.text == '-':
      left = self.children[0].to_python_ast()
      right = self.children[1].to_python_ast()
      return ast.BinOp(
          left=left,
          op=ast.Sub(),
          right=right,
          lineno=self.centroid[1],
          col_offset=self.centroid[0])
    elif self.text == '*':
      left = self.children[0].to_python_ast()
      right = self.children[1].to_python_ast()
      return ast.BinOp(
          left=left,
          op=ast.Mult(),
          right=right,
          lineno=self.centroid[1],
          col_offset=self.centroid[0])
    else:
      if len(self.children) == 0:
        # Local var
        return ast.Name(
            id=self.text,
            ctx=ast.Load(),
            lineno=self.centroid[1],
            col_offset=self.centroid[0])
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
              keywords=[],
              lineno=self.centroid[1],
              col_offset=self.centroid[0])


class ContourLine:
  def __init__(self, img_contour, contour):
    self.img_contour = img_contour
    self.contour = contour
    self.nodes = [None, None]  # Connected nodes in graph
    self.endpoints = extreme_points(self.img_contour, self.contour)
