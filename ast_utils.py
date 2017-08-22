import ast


SPECIAL_FORMS = ['if', 'define']

BINOPS = {
  '+': ast.Add,
  '-': ast.Sub,
  '/': ast.Div,
  '*': ast.Mult,
  '%': ast.Mod,
}

COMPARES = {
  '=': ast.Eq,
  '!=': ast.NotEq,
  '<': ast.Lt,
  '<=': ast.LtE,
  '>': ast.Gt,
  '>=': ast.GtE,
}

# def recognized(text):
#   return (
#       (text.isdigit()) or
#       (text in SPECIAL_FORMS) or
#       (text in BINOPS) or
#       (text in COMPARES) or
#       (text in dir(__builtins__)))

def make_module(function_asts, main_ast):
  module_body = function_asts
  module_body.append(ast.Expr(value=main_ast))

  module_ast = ast.Module(body=module_body)
  ast.fix_missing_locations(module_ast)

  # print (ast.dump(module_ast))
  return module_ast

      # HACK TO TEST FACTORIAL FUNCTION
    # test_ast =
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

    # test_ast = ast.Module(
    #     body=[
    #         ast.Expr(
    #             value=ast.Call(
    #                 func=ast.Name(id='print', ctx=ast.Load()),
    #                 args=[python_ast],
    #                 keywords=[]))])