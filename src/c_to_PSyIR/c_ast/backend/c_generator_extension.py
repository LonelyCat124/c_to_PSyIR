from c_to_PSyIR.c_ast.nodes.comment_node import CommentNode
from c_to_PSyIR.psyir.nodes.unsupported_c_type import UnsupportedCType

from pycparser.c_generator import CGenerator

class CGeneratorExtension(CGenerator):
    '''
    An extension of the CGenerator implementation from pycparser to
    handle the new CommentNodes that can be added when passing through
    PSyclone.
    '''
    def visit_CommentNode(self, node: CommentNode) -> str:
        '''
        :returns: the C representation of the comment specified by node.
        '''
        return "/*" + node.message + "*/"
