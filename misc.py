import random


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
