

from pycparser import c_parser, c_generator
from pycparser.c_ast import Node


def translate_to_c():
    code = """
        void test_func(int c){
            c = c + 1; 
        }
    """

    parser = c_parser.CParser()
    ast = parser.parse(code, filename='<none>')
    ast.show(showcoord=True)
    print(isinstance(ast, Node))

    generator = c_generator.CGenerator()
    print(generator.visit(ast))

translate_to_c()
