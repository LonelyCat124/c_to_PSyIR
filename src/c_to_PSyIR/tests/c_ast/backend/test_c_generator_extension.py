from c_to_PSyIR.c_ast.nodes.comment_node import CommentNode
from c_to_PSyIR.c_ast.backend.c_generator_extension import CGeneratorExtension

def test_CommentNode_output():
    ''' Test that the CGeneratorExtension gives the correct output for a
    CommentNode.'''
    comment = CommentNode("Test comment")
    generator = CGeneratorExtension()
    out = generator.visit(comment)
    assert "/*Test comment*/" == out
