from c_to_PSyIR.C_CodeBlock import C_CodeBlock
import psyclone.psyir.nodes.node as PSynode
from psyclone.psyir.nodes import (
        CodeBlock, FileContainer, Routine, Reference, BinaryOperation, Literal, Loop, UnaryOperation, IfBlock
)
from psyclone.psyir.nodes import Assignment as PSyAssignment
from pycparser import c_parser, c_generator
from pycparser.c_ast import (
        Node, FileAST, NodeVisitor, FuncDef, Decl, Assignment, ID, BinaryOp, Constant, FuncDecl, TypeDecl, IdentifierType, Compound, ParamList,
        Struct, For, UnaryOp, If
)
from psyclone.psyir.backend.visitor import PSyIRVisitor
from psyclone.psyir.symbols import SymbolTable
from psyclone.psyir.symbols import INTEGER_TYPE, DataSymbol, ScalarType, ArgumentInterface, ScalarType, Symbol

type_map = {ScalarType.Intrinsic.INTEGER: {ScalarType.Precision.SINGLE: "int", ScalarType.Precision.DOUBLE: "long long int",
                                           ScalarType.Precision.UNDEFINED: "int", 32: "int32_t", 64: "int64_t", 8: "int8_t"},
                ScalarType.Intrinsic.REAL: {ScalarType.Precision.SINGLE: "float", ScalarType.Precision.DOUBLE: "double",
                                            },
#                ScalarType.Intrinsic.CHARACTER: {ScalarType.Precision.UNDEFINED: "char*"},
                ScalarType.Intrinsic.BOOLEAN: {ScalarType.Precision.UNDEFINED: "bool"}
               }

def create_str_to_type_map(type_map: map) -> map:
    str_to_type_map = {}
    for intrinsic in type_map.keys():
        for precision in type_map[intrinsic]:
            string = type_map[intrinsic][precision]
            if string not in str_to_type_map.keys():
                str_to_type_map[string] = (intrinsic, precision)
    return str_to_type_map

str_to_type_map = create_str_to_type_map(type_map)

def invert_map(in_map: map) -> map:
    inverted_map = {}
    for name in in_map.keys():
        inverted_map[in_map[name]] = name
    return inverted_map

c_to_f_binop_operator_map = {
        "+": BinaryOperation.Operator.ADD,
        "-": BinaryOperation.Operator.SUB,
        "*": BinaryOperation.Operator.MUL,
        "/": BinaryOperation.Operator.DIV,
        "==": BinaryOperation.Operator.EQ, # No difference for logical operations.
        "!=": BinaryOperation.Operator.NE,
        ">": BinaryOperation.Operator.GT,
        "<": BinaryOperation.Operator.LT,
        ">=": BinaryOperation.Operator.GE,
        "<=": BinaryOperation.Operator.LE,
        "&&": BinaryOperation.Operator.AND,
        "&&": BinaryOperation.Operator.OR,
}

f_to_c_binop_operator_map = invert_map(c_to_f_binop_operator_map)

c_to_f_unary_operator_map = {
        "-": UnaryOperation.Operator.MINUS,
        "+": UnaryOperation.Operator.PLUS,
        "!": UnaryOperation.Operator.NOT,
}

f_to_c_unary_operator_map = invert_map(c_to_f_unary_operator_map)


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

class CNode_to_PSyIR_Visitor(NodeVisitor):
    # Based on pycparser generator visitor.

    def __init__(self):
        super().__init__()

        self._symbol_tables = []

    def visit(self, node: Node) -> PSynode.Node:
        # TODO Can maybe do better with mro ordering like PSyclone
        method = 'visit_' + node.__class__.__name__
        try:
            return getattr(self, method, None)(node)
            #return getattr(self, method, self.generic_visit)(node)
        except NotImplementedError as e:
            code_block = self.generic_visit(node)
            comment = f"Codeblock created - NYI error {str(e)}"
            code_block.preceding_comment = comment
            return code_block
        except Exception as e:
            # FIXME Try replacing this Error with a more precise Error.
            code_block = self.generic_visit(node)
            comment = "Codeblock created - unsupported code: "
            comment += f"Failed to handle input node of type '{(type(node))}'. "
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
        op = c_to_f_binop_operator_map.get(node.op, None)
        if not op:
            raise NotImplementedError("Unsupported BinaryOperation operator")
        rhs = self.visit(node.right)
        return BinaryOperation.create(op, lhs, rhs)

    def visit_Constant(self, node: Constant) -> Literal:
        type_str = node.type
        value = node.value
        # For now we assume its an integer <_<
        if type_str != "int":
            assert False
        return Literal(value, INTEGER_TYPE)

    def _check_loop_init_validity(self, loop_init: Node) -> (str, Node):
        # Op must be "="
        if loop_init.op != "=":
            raise NotImplementedError("Operations other than = aren't supported on loop init statements")
        if not isinstance(loop_init.lvalue, ID):
            raise NotImplementedError("Got a loop_init.lvalue that isn't an ID, don't understand.")
        loop_var_name = loop_init.lvalue.name
        loop_start = self.visit(loop_init.rvalue)
        return loop_var_name, loop_start

    def _check_loop_stop_validity(self, loop_stop: Node, loop_var: Symbol, loop_step: Literal) -> PSynode.Node:
        if not isinstance(loop_stop, BinaryOp):
            raise NotImplementedError("Only support BinaryOp loop_stop conditions")
        # Unpack the loop_stop
        op = loop_stop.op
        left = loop_stop.left
        right = loop_stop.right
        left_loop_var = False
        right_loop_var = False
        if isinstance(left, ID) and left.name == loop_var.name:
            left_loop_var = True
        if isinstance(right, ID) and right.name == loop_var.name:
            right_loop_var = True
        if not left_loop_var and not right_loop_var:
            raise NotImplementedError("The lhs or rhs of the loop_stop condition must be the loop variable")
        if left_loop_var and right_loop_var:
            raise NotImplementedError("Can't handle a loop_var comparison with itself")
        step_as_int = int(loop_step.value)
        if step_as_int > 0:
            if left_loop_var and op == "<":
                return self.visit(right)
            if left_loop_var and op == "<=":
                raise NotImplementedError("Support for <= in loop condition NYI")
            if right_loop_var and op == ">":
                return self.visit(left)
            if right_loop_var and op == ">=":
                raise NotImplementedError("Support for >= in loop condition NYI")
        if step_as_int < 0:
            if left_loop_var and op == ">":
                return self.visit(right)
            if left_loop_var and op == ">=":
                raise NotImplementedError("Support for >= in loop condition NYI")
            if right_loop_var and op == "<":
                return self.visit(left)
            if right_loop_var and op == "<=":
                raise NotImplementedError("Support for <= in loop condition NYI")

        raise NotImplementedError("Unsupported loop stop condition")


    def _check_loop_step_validity(self, loop_step: Node) -> PSynode.Node:
        if isinstance(loop_step, UnaryOp):
            if loop_step.op == "p++":
                return Literal("1", INTEGER_TYPE)
            elif loop_step.op == "p--":
                return Literal("-1", INTEGER_TYPE)
            else:
                raise NotImplementedError("Unsupported UnaryOp in loop_step")
        else:
            raise NotImplementedError("Non-unary op steps NYI")

    def visit_For(self, node: For) -> Loop:
        start = node.init
        if not isinstance(node.init, Assignment):
            raise NotImplementedError("For loops with declarations aren't supported.")
        # start needs to be an integer value set assignment.
        loop_var_name, start_cond = self._check_loop_init_validity(start)
        loop_var = None
        for symbol_table in self._symbol_tables[::-1]:
            var_symbol = symbol_table.lookup(loop_var_name, otherwise=None)
            if var_symbol is not None:
                loop_var = var_symbol
                break
        if not loop_var:
            raise NotImplementedError("Failed to find the symbol of the Loop")
        # TODO Move this thing up into _check_loop_init_validity and also check the
        # symbol is an integer.
        step = node.next
        step_cond = self._check_loop_step_validity(step)
        # Step condition needs to be a ++, --, +=, -= or equivalent statement.
        stop = node.cond
        stop_cond = self._check_loop_stop_validity(stop, loop_var, step_cond)
        # Stop condition needs to be a < statement if step is positive increment, or
        # a > statement if step is a negative increment and must be relative to the start
        # assignment
        # TODO Check the Loop condition is ok - needs to be basic for PSyclone
        # to parse it.
        body = []
        for child in node.stmt:
            body.append(self.visit(child))
        return Loop.create(loop_var, start_cond, stop_cond, step_cond, body)

    def visit_UnaryOp(self, node: UnaryOp) -> UnaryOperation:
        op = node.op
        expr = self.visit(node.expr)
        return UnaryOperation.create(c_to_f_unary_operator_map[op], expr)

    def visit_If(self, node: If) -> IfBlock:
        is_else_if = False
        cond = self.visit(node.cond)
        if_body = self.visit(node.iftrue)
        if not isinstance(if_body, list):
            if_body = [if_body]
        if node.iffalse:
            else_body = self.visit(node.iffalse)
            if isinstance(node.iffalse, If):
                is_else_if = True
            if not isinstance(else_body, list):
                else_body = [else_body]
        else:
            else_body = None
        ifblock = IfBlock.create(cond, if_body, else_body)
        ifblock.annotations.append('was_elseif')
        return ifblock

    def visit_Compound(self, node: Compound) -> list:
        result = []
        for child in node:
            result.append(self.visit(child))
        return result



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

    def reference_node(self, node: Reference) -> ID:
        return ID(name=node.name)

    def binaryoperation_node(self, node: BinaryOperation) -> BinaryOp:
        lhs = self._visit(node.children[0])
        rhs = self._visit(node.children[1])
        # Remove comments in case of CodeBlocks
        lhs = self.strip_comments(lhs)
        rhs = self.strip_comments(rhs)
        # TODO This is bad lmao.
        op = f_to_c_binop_operator_map[node.operator]
        return BinaryOp(op, lhs, rhs)

    def assignment_node(self, node: PSyAssignment) -> Assignment:
        lhs = self._visit(node.lhs)
        rhs = self._visit(node.rhs)
        # Remove comments in case of CodeBlocks
        lhs = self.strip_comments(lhs)
        rhs = self.strip_comments(rhs)
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
            if 'was_elseif' in node.annotations:
                iffalse = iffalse[0]
            else:
                iffalse = Compound(iffalse)
        else:
            iffalse = None
        return If(cond, Compound(iftrue), iffalse)
        
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
            for(d = 0; d < f; d++){
                a[d] = 2;
                a[d] = c + -1;
                a[d] = a[d] - 1;
            }
            for(int e = 0; e < e + 1; e++){
                a[i] = 2;
            }
            if(1){
                printf("Hello\\n");
            }
            if(e > f){
                a[0] = 1;
            }else if(e == f){
                a[1] = 1;
            }else{
                a[2] = 1;
            }
        }
    """

    parser = c_parser.CParser()
    ast = parser.parse(code, filename='<none>')

    cnode_to_psyir = CNode_to_PSyIR_Visitor()
    psyir = cnode_to_psyir.visit(ast)

    psyir_to_c = PSyIR_to_C_Visitor()
    print(psyir_to_c(psyir))

translate_to_c()
