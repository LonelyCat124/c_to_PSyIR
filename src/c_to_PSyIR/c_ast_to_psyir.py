from c_to_PSyIR.C_CodeBlock import C_CodeBlock
import psyclone.psyir.nodes.node as PSynode
from psyclone.psyir.nodes import (
        CodeBlock, FileContainer, Routine, Reference, BinaryOperation, Literal
)
from psyclone.psyir.nodes import Assignment as PSyAssignment
from pycparser import c_parser, c_generator
from pycparser.c_ast import (
        Node, FileAST, NodeVisitor, FuncDef, Decl, Assignment, ID, BinaryOp, Constant, FuncDecl, TypeDecl, IdentifierType, Compound, ParamList,
        Struct
)
from psyclone.psyir.backend.visitor import PSyIRVisitor
from psyclone.psyir.symbols import SymbolTable
from psyclone.psyir.symbols import INTEGER_TYPE, DataSymbol, ScalarType, ArgumentInterface, ScalarType

type_map = {ScalarType.Intrinsic.INTEGER: {ScalarType.Precision.SINGLE: "int", ScalarType.Precision.DOUBLE: "long long int",
                                           ScalarType.Precision.UNDEFINED: "int", 32: "int32_t", 64: "int64_t", 8: "int8_t"},
                ScalarType.Intrinsic.REAL: {ScalarType.Precision.SINGLE: "float", ScalarType.Precision.DOUBLE: "double",
                                            },
#                ScalarType.Intrinsic.CHARACTER: {ScalarType.Precision.UNDEFINED: "char*"},
                ScalarType.Intrinsic.BOOLEAN: {ScalarType.Precision.UNDEFINED: "bool"}
               }

class CommentNode(Node):
    __slots__ = {'message', 'coord', '__weakref__'}
    def __init__(self, message: str, coord=None):
        self.message = message

    def children(self):
        return ()

    def __iter__(self):
        return
        yield

    attr_names = ()

class CGeneratorExtension(c_generator.CGenerator):
    def visit_CommentNode(self, node: CommentNode) -> str:
        return "/*" + node.message + "*/"


def create_str_to_type_map(type_map):
    str_to_type_map = {}
    for intrinsic in type_map.keys():
        for precision in type_map[intrinsic]:
            string = type_map[intrinsic][precision]
            if string not in str_to_type_map.keys():
                str_to_type_map[string] = (intrinsic, precision)
    return str_to_type_map

str_to_type_map = create_str_to_type_map(type_map)

class CNode_to_PSyIR_Visitor(NodeVisitor):
    # Based on pycparser generator visitor.

    def __init__(self):
        super().__init__()

        self._symbol_tables = []

    def visit(self, node: Node) -> PSynode.Node:
        # TODO Can maybe do better with mro ordering like PSyclone
        method = 'visit_' + node.__class__.__name__
        try:
            return getattr(self, method, self.generic_visit)(node)
        except Exception as e:
            # FIXME Try replacing this Error with a more precise Error.
            code_block = self.generic_visit(node)
            comment = f"Failed to handle input node of type '{(type(node))}'. "
            comment += f"Failed with error {str(e)}"
            code_block.preceding_comment = comment
            return code_block

    def generic_visit(self, node: Node) -> C_CodeBlock:
        # For something we don't handle explicitly we need a C_CodeBlock!
        # For now we're just making a CodeBlock for everything and not doing
        # anything smart like fparser2.py - probably better to keep it like
        # this for reverting to pycparser tree later.
        print(f"Found a node of type {node.__class__.__name__}")
        return C_CodeBlock([node], CodeBlock.Structure.STATEMENT)

    def visit_FileAST(self, node: FileAST) -> FileContainer:
        psyir_children = []
        # TODO Fix symbol tables (urgh)
        symbol_table = SymbolTable()
        self._symbol_tables.append(symbol_table)
        for child in node:
            res = self.visit(child)
            if res:
                psyir_children.append(res)
        self._symbol_tables.pop()
        return FileContainer.create("", symbol_table, psyir_children) 

    def visit_FuncDef(self, node: FuncDef) -> Routine:
        name = node.decl.name

        routine_sym_tab = SymbolTable()
        # Add it to the "stack" of symbol tables
        self._symbol_tables.append(routine_sym_tab)
        if node.decl.type.args and len(node.decl.type.args.params) > 0:
            # Time to get the arguments
            args = []
            for decl in node.decl.type.args.params:
                self.visit(decl)
                name = decl.name
                sym = routine_sym_tab.lookup(name)
                sym.interface = ArgumentInterface(
                        ArgumentInterface.Access.UNKNOWN)
                args.append(sym)
            routine_sym_tab.specify_argument_list(args)
        # Can't handle argument list for now
        args = []
        # Can't handle return type for now
        # This info is all contained in the FuncDef object of this node.decl.type
        # Probably this needs to be in some sort of visitor or something but seems
        # to be relatively complex so need to see what the cparser itself does with
        # visiting these objects.
        # Create the Routine's symbol table as it needs to be visible to the children.
        # For arguments we need to put them in the symbol table we pass to routine.create
        psyir_children = []
        for child in node.body:
            # If the body element returns a node then we add it to the list. 
            # Things like variable declarations become symbol table things in PSyIR
            result = self.visit(child)
            if result:
                psyir_children.append(result)

        # Now remove the symbol table
        self._symbol_tables.pop()

        return Routine.create(name, children=psyir_children, symbol_table=routine_sym_tab)

    def visit_Decl(self, node: Decl) -> None:
        name = node.name
        typedef = node.type
        # Structure declaration makes a CodeBlock for now.
        if isinstance(node.type, Struct):
            raise NotImplementedError("Structure declarations not yet implemented.")
        # Structure type declaration also makes a Codeblock for now.
        # This is a relatively easy fix though.
        if isinstance(node.type.type, Struct):
            raise NotImplementedError("Structure initialisation not yet implemented.")
        type_str = node.type.type.names
        if(len(type_str) > 1):
            assert False # Need to think what this means - maybe pointers? Or long long int etc.i
        else:
            type_str = type_str[0]
        # For now we assume its an integer <_<
        intrinsic, precision = str_to_type_map.get(type_str, (None, None))
        if intrinsic is None:
            assert False # There must be some unknown type object we can do for this like we do for CodeBlocks.
        # TODO Convert type_str to symbol type - see the mapping on SFP.
        # Get the current symbol table. Its always the lowest one.
        sym_tab = self._symbol_tables[-1]
        sym_tab.new_symbol(name, symbol_type=DataSymbol, datatype=ScalarType(intrinsic, precision))

    def visit_Assignment(self, node: Assignment) -> PSyAssignment:
        lhs = self.visit(node.lvalue)
        rhs = self.visit(node.rvalue)
        return PSyAssignment.create(lhs, rhs)

    def visit_ID(self, node: ID) -> Reference:
        # Get the symbol table.
        sym_tab = self._symbol_tables[-1]
        symbol = sym_tab.symbols_dict.get(node.name, None)
        if not symbol:
            assert False
        # We should probably do better since maybe this isn't just a basic symbol but oh well.
        # TODO
        return Reference(symbol)

    def visit_BinaryOp(self, node: BinaryOp) -> BinaryOperation:
        lhs = self.visit(node.left)
        # TODO Op needs a string to operator conversion. Think this is on SFP as well
        op = BinaryOperation.Operator.ADD 
        rhs = self.visit(node.right)
        return BinaryOperation.create(op, lhs, rhs)

    def visit_Constant(self, node: Constant) -> Literal:
        type_str = node.type
        value = node.value
        # For now we assume its an integer <_<
        if type_str != "int":
            assert False
        return Literal(value, INTEGER_TYPE)


class PSyIR_to_C_Visitor(PSyIRVisitor):
    # TODO
    # Goal of this is to recreate the pycparser tree (or similar enough).
    # This initial thing doesn't do that and is naught but it works to try stuff.

    def __call__(self, node: PSynode.Node) -> str:
        generator = CGeneratorExtension()
        # Maybe we want (likely) to call super.__call__ instead of just self._visit for
        # supporting lowering.
        return generator.visit(self._visit(node))

    def c_codeblock_node(self, node: C_CodeBlock) -> list[Node]:
        pre_comment = node.preceding_comment
        comment = CommentNode(pre_comment)
        node = node.get_ast_nodes[0]
        return [comment, node]

    def reference_node(self, node: Reference) -> ID:
        return ID(name=node.name)

    def binaryoperation_node(self, node: BinaryOperation) -> BinaryOp:
        lhs = self._visit(node.children[0])
        rhs = self._visit(node.children[1])
        # TODO This is bad lmao.
        op = "+"
        return BinaryOp(op, lhs, rhs)

    def assignment_node(self, node: PSyAssignment) -> Assignment:
        lhs = self._visit(node.lhs)
        rhs = self._visit(node.rhs)
        return Assignment("=", lhs, rhs)

    def literal_node(self, node: Literal) -> Constant:
        value = node.value
        dtype = type_map[node.datatype.intrinsic][node.datatype.precision]
        return Constant(dtype, value)

    def datasymbol_to_decl(self, symbol: DataSymbol) -> Decl:
        name = symbol.name
        dtype = [type_map[symbol.datatype.intrinsic][symbol.datatype.precision]]
        typedecl = TypeDecl(declname=name,quals=[], align=None, type=IdentifierType(names=dtype))
        # TODO I assume init is fine for us to enable.
        # NB We can do better with this for sure.
        # Maybe this isn't correct but we work with it for now.
        return Decl(name, [],[],[],[],typedecl,init=None,bitsize=None)

    def routine_node(self, node: Routine) -> FuncDef:
        # TODO This isn't how we want to actually do this - we want to recreate the pycparser stuff.
        # For next time.
        name = node.name
        # TODO Return type, arguments, symbol_table
#        rval = f"void {name}()" + "{\n"
        body = []
        arglist = []
        for symbol in node.symbol_table.symbols:
            if symbol.is_argument:
                arglist.append(self.datasymbol_to_decl(symbol))
            else:
                body.append(self.datasymbol_to_decl(symbol))
        for child in node.children:
            body.extend(self._visit(child))

        params = ParamList(arglist)
        # Create the Decl for the FuncDecl for the FuncDef
        # TODO At the moment arguments nor non-void types are supported
        # Most of Decl is unsupported also.
        dtype = FuncDecl(args=params, type=TypeDecl(declname=name,quals=[],align=None, type=IdentifierType(names=["void"])))
        decl = Decl(name=name, type=dtype, quals=None, align=None, storage=None, funcspec=None, init=None, bitsize=None) 
    
        # Create the FuncDef TODO Argument list.
        return FuncDef(decl=decl, param_decls=[], body=Compound(body))

    def filecontainer_node(self, node: FileContainer) -> FileAST:
        ext = []
        for symbol in node.symbol_table.symbols:
            if isinstance(symbol, DataSymbol):
                ext.append(self.datasymbol_to_decl(symbol))
        for child in node.children:
            res = self._visit(child)
            ext.extend(res)
        return FileAST(ext)
        
    def node_node(self, node: PSynode.Node) -> None:
        assert False

def translate_to_c():
    code = """
        int x;
        struct name{
            double f;
        };
        void test_func(int d, float e, double f){
            struct name thing;
            int c;
            int *a;
            c = c + 1;
        }
    """

    parser = c_parser.CParser()
    ast = parser.parse(code, filename='<none>')

    cnode_to_psyir = CNode_to_PSyIR_Visitor()
    psyir = cnode_to_psyir.visit(ast)

    psyir_to_c = PSyIR_to_C_Visitor()
    print(psyir_to_c(psyir))

translate_to_c()
