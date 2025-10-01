from psyclone.psyir.nodes import CodeBlock
import pycparser.c_ast as c_ast

class CMixin:
    pass
    # TODO Put stuff like C debug_string etc. here.

class C_CodeBlock(CodeBlock, CMixin):

    def __init__(self, c_nodes: list[c_ast.Node], structure: CodeBlock.Structure, parent=None, annotations=None):
        super().__init__([], structure, parent=parent, annotations=annotations)
        
        self._c_nodes = c_nodes


    def __eq__(self, other):
        is_eq = super().__eq__(other)
        is_eq = is_eq and self._c_nodes == other._c_nodes # TODO
        is_eq = is_eq and self.structure == other.structure
        return is_eq


    @property
    def get_ast_nodes(self) -> list[c_ast.Node]:
        return self._c_nodes


    def node_str(self, colour: bool = True) -> str:
        return (f"{self.coloured_name(colour)}["
                f"{list(map(type, self._c_nodes))}]")

    def get_symbol_names(self) -> list[str]:
        #assert False # TODO
        # This is very naughty this needs to be fixed >:(
        return []

    def __str__(self):
        return f"C_CodeBlock[{len(self._c_nodes())} nodes]"
