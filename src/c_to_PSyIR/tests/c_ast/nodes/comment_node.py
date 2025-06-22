from c_to_PSyIR.c_ast.nodes.comment_node import CommentNode

def test_CommentNode():
    ''' Test the methods of the CommentNode pycparser extension.'''
    a = CommentNode("Message")
    assert a.message == "Message"
    assert len(a.children()) == 0
    # Should be results from __iter__
    for child in a:
        assert False

    assert len(CommentNode.attr_names) == 0
