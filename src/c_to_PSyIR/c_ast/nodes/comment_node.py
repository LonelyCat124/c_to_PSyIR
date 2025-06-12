from pycparser.c_ast import  Node

class CommentNode(Node):
    '''
    Class to represent a Comment in the c_ast tree.

    :param message: The message contained in this comment.
    :param coord: No idea... Defaults to None.
    '''
    __slots__ = {'message', 'coord', '__weakref__'}

    def __init__(self, message: str, coord=None):
        self.message = message

    def children(self):
        return ()

    def __iter__(self):
        return
        yield

    attr_names = ()
