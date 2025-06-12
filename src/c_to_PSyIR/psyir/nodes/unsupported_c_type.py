from collections import OrderedDict
from psyclone.psyir.symbols import DataType
import pycparser.c_ast as c_ast

class UnsupportedCType(DataType):
    '''
    PSyIR Datatype to handle unsupported C Types.

    :param declaration_ast: The c_ast Node that represents the
                            unsupported Type
    '''

    def __init__(self, declaration_ast: c_ast.Decl):
        self._declaration = declaration_ast

    @property
    def declaration(self) -> c_ast.Decl:
        '''
        :returns: The c_ast Decl corresponding to this Type.
        '''
        return self._declaration

    def __str__(self) -> str:
        '''
        :returns: A string representation of this node.
        '''
        return "UnsupportedCType<>"
