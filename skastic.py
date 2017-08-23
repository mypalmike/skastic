#!/usr/bin/env python

import argparse
import ast
import sys

import cv2

from ast_utils import make_module
from page import Page

class Skastic:
  def __init__(self, filename, args):
    self.function_pages = {}
    self.main_page = None
    self.args = args
    self.analyze_only = bool(args.draw)
    self.load(filename, args)
    if not self.analyze_only:
      self.exec()

  def load(self, filename, args):
    self.main_page = Page(filename, self.analyze_only)
    if self.analyze_only:
      self.main_page.draw(args.draw)
    else:
      self.load_includes(args)
      # self.load_externs()

  # def load_externs(self):
  #   externs = self.main_page.externs.copy()
  #   while externs:
  #     extern = externs.pop()
  #     extern_filename = '{}.ska.png'.format(extern)
  #     extern_page = Page(extern_filename, False)
  #     self.function_pages[extern] = extern_page
  #     # # There's probably a more functional way to do this:
  #     # for page_extern in extern_page.externs:
  #     #   if not recognized(page_extern):
  #     #     externs.add(page_extern)

  def load_includes(self, args):
    for filename in args.include:
      self.function_pages[filename.split('.')[0]] = Page(filename, False)

  def exec(self):
    function_asts = [x.python_ast for x in self.function_pages.values()]
    module_ast = make_module(function_asts, self.main_page.python_ast)

    if self.args.dump:
      print(ast.dump(module_ast, include_attributes=True))

    exec(compile(module_ast, filename="<ast>", mode="exec"))


def main(argv=sys.argv):
  parser = argparse.ArgumentParser(description='Execute a skastic image.')
  parser.add_argument('filename', help='Name of .ska.png file to execute')
  parser.add_argument('--include', '-i', action='append', help='Include file', default = [])
  parser.add_argument('--draw', action='append', help='Do not execute. For debugging an '
                      'image, display one of (contours, lines, connections, parse-tree)', default = [])
  parser.add_argument('--dump', action='store_true', help='Do not execute. For debugging, '
                      'dump ast')
  args = parser.parse_args(argv[1:])

  skastic = Skastic(args.filename, args)

  if args.draw:
    # Meh, probably better to wait for a key from the console?
    cv2.waitKey(0)


if __name__ == '__main__':
  main()
