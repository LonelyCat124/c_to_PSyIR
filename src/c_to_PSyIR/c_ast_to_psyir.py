from collections import OrderedDict
from dataclasses import dataclass

from c_to_PSyIR.c_ast.nodes.comment_node import CommentNode
from c_to_PSyIR.c_ast.backend.c_generator_extension import CGeneratorExtension

from c_to_PSyIR.psyir.nodes.C_CodeBlock import C_CodeBlock
from c_to_PSyIR.psyir.nodes.unsupported_c_type import UnsupportedCType

from c_to_PSyIR.psyir.frontend.cnode_to_psyir_visitor import CNode_to_PSyIR_Visitor

from c_to_PSyIR.psyir.utils.operator_utils import (
        type_map, str_to_type_map, c_to_f_binop_operator_map,
        f_to_c_binop_operator_map, c_to_f_unary_operator_map,
        f_to_c_unary_operator_map,
)

import psyclone.psyir.nodes.node as PSynode
from psyclone.psyir.nodes import (
        CodeBlock, FileContainer, Routine, Reference, BinaryOperation, Literal, Loop, UnaryOperation, IfBlock,
        ArrayReference,
)
from psyclone.psyir.nodes import Assignment as PSyAssignment
from pycparser import c_parser, c_generator
from pycparser.c_ast import (
        Node, FileAST, NodeVisitor, FuncDef, Decl, Assignment, ID, BinaryOp, Constant, FuncDecl, TypeDecl, IdentifierType, Compound, ParamList,
        Struct, For, UnaryOp, If, PtrDecl, ArrayDecl, ArrayRef
)
from psyclone.psyir.backend.visitor import PSyIRVisitor
from psyclone.psyir.symbols import SymbolTable, StructureType, DataTypeSymbol
from psyclone.psyir.symbols import INTEGER_TYPE, DataSymbol, ScalarType, ArgumentInterface, ScalarType, Symbol, DataType, ArrayType, UnsupportedType

class PSyIR_to_C_Visitor(PSyIRVisitor):
    # TODO
    # Goal of this is to recreate the pycparser tree (or similar enough).
    # This initial thing doesn't do that and is naught but it works to try stuff.

    def __call__(self, node: PSynode.Node) -> str:
        generator = CGeneratorExtension()
        # Maybe we want (likely) to call super.__call__ instead of just self._visit for
        # supporting lowering.
        return generator.visit(self._visit(node))

    def strip_comments(self, nodes: list[Node] | Node):
         if isinstance(nodes, list):
            return nodes[1]
         else:
            return nodes

    def c_codeblock_node(self, node: C_CodeBlock) -> list[CommentNode, Node]:
        pre_comment = node.preceding_comment
        comment = CommentNode(pre_comment)
        node = node.get_ast_nodes[0]
        # TODO Handle returning comments again
        return [comment, node]

    def arrayreference_node(self, node: ArrayReference) -> ArrayRef:
        ref = ID(name=node.name)
        for index in node.indices:
           subscript = self._visit(index)
           ref = ArrayRef(name=ref, subscript=subscript)

        return ref
    
    def reference_node(self, node: Reference) -> ID:
        return ID(name=node.name)

    def binaryoperation_node(self, node: BinaryOperation) -> BinaryOp:
        lhs = self._visit(node.children[0])
        rhs = self._visit(node.children[1])
        # Remove comments in case of CodeBlocks
        lhs = self.strip_comments(lhs)
        rhs = self.strip_comments(rhs)
        op = f_to_c_binop_operator_map[node.operator]
        return BinaryOp(op, lhs, rhs)

    def assignment_node(self, node: PSyAssignment) -> Assignment:
        lhs = self._visit(node.lhs)
        rhs = self._visit(node.rhs)
        # Remove comments in case of CodeBlocks
        lhs = self.strip_comments(lhs)
        rhs = self.strip_comments(rhs)
        assign = Assignment("=", lhs, rhs)
        return assign

    def literal_node(self, node: Literal) -> Constant:
        value = node.value
        dtype = type_map[node.datatype.intrinsic][node.datatype.precision]
        return Constant(dtype, value)

    def datasymbol_to_decl(self, symbol: DataSymbol) -> Decl:
        name = symbol.name
        if isinstance(symbol.datatype, ScalarType):
            dtype = [type_map[symbol.datatype.intrinsic][symbol.datatype.precision]]
            typedecl = TypeDecl(declname=name,quals=[], align=None, type=IdentifierType(names=dtype))
        elif isinstance(symbol.datatype, ArrayType):
            dtype = symbol.datatype.datatype
            dtype = [type_map[dtype.intrinsic][dtype.precision]]
            typedecl = TypeDecl(declname=name, quals=[], align=None, type=IdentifierType(names=dtype))
            for ind in symbol.datatype.shape:
                if ind is not ArrayType.Extent.DEFERRED:
                    if ind.lower.value != "1":
                        assert False # Unsupported - this is set by PSyclone to 1 by default and this
                                     # parser doesn't touch it
                    typedecl = ArrayDecl(type=typedecl, dim=self._visit(ind.upper), dim_quals=[])
                    continue
                typedecl = PtrDecl(quals=[], type=typedecl)
        elif isinstance(symbol.datatype, DataTypeSymbol):
            typedecl = Struct(name=symbol.datatype.name, decls=None)
            typedecl = TypeDecl(declname=name, quals=[], align=None, type=typedecl)
        elif isinstance(symbol.datatype, UnsupportedCType):
            return symbol.datatype.declaration
        else:
            assert False
        # TODO I assume init is fine for us to enable.
        # NB We can do better with this for sure.
        # Maybe this isn't correct but we work with it for now.
        return Decl(name, [],[],[],[],typedecl,init=None,bitsize=None)

    def datatypesymbol_to_decl(self, symbol: DataTypeSymbol) -> Decl:
        dtype = symbol.datatype
        if isinstance(dtype, UnsupportedCType):
            return symbol.datatype.declaration
        components = []
        for component in dtype.components:
            if not isinstance(dtype.components[component].datatype, ScalarType):
                assert False
            subtype = type_map[dtype.components[component].datatype.intrinsic][dtype.components[component].datatype.precision]
            typedecl = TypeDecl(component, quals=[], align=None, type=IdentifierType(names=[subtype]))
            comp_decl = Decl(component, quals=[], align=[], storage=[], funcspec=[], type=typedecl, init=None, bitsize=None)
            components.append(comp_decl)

        struct_obj = Struct(name=symbol.name, decls=components)
        decl = Decl(symbol.name, quals=[], align=[], storage=[], funcspec=[],type=struct_obj, init=None, bitsize=None)
        return decl

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
            res = self._visit(child)
            if isinstance(res, list):
                body.extend(res)
            else:
                body.append(res)

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
            elif isinstance(symbol, DataTypeSymbol):
                ext.append(self.datatypesymbol_to_decl(symbol))
        for child in node.children:
            res = self._visit(child)
            if isinstance(res, list):
                ext.extend(res)
            else:
                ext.extend(res)
        return FileAST(ext)

    def loop_node(self, node: Loop) -> For:
        loop_body = node.children[3]
        body = []
        for child in loop_body:
            rest = self._visit(child)
            if isinstance(rest, list):
                body.extend(rest)
            else:
                body.append(rest)

        var = node.variable
        start = node.start_expr
        next = node.stop_expr
        cond = node.step_expr

        init_left = ID(name=node.variable.name)
        init_right = self._visit(node.start_expr)
        init_right = self.strip_comments(init_right)
        init = Assignment("=", init_left, init_right)

        step_val = int(node.step_expr.value)
        if(step_val > 0):
            stop_left = ID(name=node.variable.name)
            stop_right = self._visit(next)
            stop_right = self.strip_comments(stop_right)
            next = BinaryOp("<", stop_left, stop_right)

            if(step_val == 1):
                step = UnaryOp("p++", ID(name=node.variable.name))
            else:
                raise NotImplementedError("Don't yet support non unit loop strides")
        else:
            stop_left = ID(name=node.variable.name)
            stop_right = self._visit(next)
            stop_right = self.strip_comments(stop_right)
            next = BinaryOp(">", stop_left, stop_right)
            if(step_val == -1):
                step = UnaryOp("p--", ID(name=node.variable.name))
            else:
                raise NotImplementedError("Don't yet support non unit loop strides")

        # init, next, cond, stmt 
    
        return For(init, next, step, Compound(body))

    def unaryoperation_node(self, node: UnaryOperation) -> UnaryOp:
        op = f_to_c_unary_operator_map[node.operator]
        expr = self._visit(node.children[0])
        return UnaryOp(op, expr)

    def ifblock_node(self, node: IfBlock) -> If:
        cond = self._visit(node.condition)
        iftrue = []
        for child in node.if_body:
            res = self._visit(child)
            if isinstance(res, list):
                iftrue.extend(res)
            else:
                iftrue.append(res)
        if node.else_body:
            iffalse = []
            for child in node.else_body:
                res = self._visit(child)
                if isinstance(res, list):
                    iffalse.extend(res)
                else:
                    iffalse.append(res)
            # TODO Can't reproduce the original else if behaviour.
            #if 'was_elseif' in node.annotations:
            #    iffalse = iffalse[0]
            #else:
            iffalse = Compound(iffalse)
        else:
            iffalse = None
        return If(cond, Compound(iftrue), iffalse)
        
    def node_node(self, node: PSynode.Node) -> None:
        assert False

def translate_to_c():
     #   int x;
     #   struct y2{
     #       int jj;
     #       int kk;
     #   };
     #   struct name{
     #       double f;
     #       struct y2 a;
     #   };
#            struct y2 thing;
#                printf("Hello\\n");
    code = """
        void test_func(int d, float e, double f){
            int c;
            int *a;
            int **h;
            int g[50];
            int k[50][25];
            int i;
            int l;
            float j;
            c = c + 1;
            for(d = 0; d < f; d++){
                a[d] = 2;
                a[d] = c + -1;
                a[d] = a[d] - 1;
            }
            for(l = 0; l < f; l++){
                a[i] = 2;
            }
            j = 1.0;
            if(1){
            }
            if(e > f){
                a[0] = 1;
            }else if(e == f){
                a[1] = 1;
            }else{
                a[2] = 1;
                a[3] = k[25][13];
            }
        }
    """

    parser = c_parser.CParser()
    ast = parser.parse(code, filename='<none>')

    cnode_to_psyir = CNode_to_PSyIR_Visitor()
    psyir = cnode_to_psyir.visit(ast)
    print(psyir.debug_string())

    psyir_to_c = PSyIR_to_C_Visitor()
    print(psyir.view())
    print(psyir_to_c(psyir))

translate_to_c()
