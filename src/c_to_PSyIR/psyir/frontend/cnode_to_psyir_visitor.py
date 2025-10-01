from dataclasses import dataclass

from c_to_PSyIR.psyir.nodes.C_CodeBlock import C_CodeBlock
from c_to_PSyIR.psyir.nodes.unsupported_c_type import UnsupportedCType

from c_to_PSyIR.psyir.utils.operator_utils import (
        type_map, str_to_type_map, c_to_f_binop_operator_map,
        f_to_c_binop_operator_map, c_to_f_unary_operator_map,
        f_to_c_unary_operator_map,
)

import psyclone.psyir.nodes as PSyIR
import psyclone.psyir.symbols as PSySym

from pycparser.c_ast import (
        Node, FileAST, NodeVisitor, FuncDef, Decl, Assignment, ID, BinaryOp, Constant, FuncDecl, TypeDecl, IdentifierType, Compound, ParamList,
        Struct, For, UnaryOp, If, PtrDecl, ArrayDecl, ArrayRef
)


@dataclass
class PossibleArray():
    '''
    Dataclass to represent names of potential arrays.

    :param name: The name of the potential array.
    :param sym_table: The symbol table containing the symbol.
    :param dimensions: The number of dimensions of the array/pointer.
    '''
    name: str
    sym_table: PSySym.SymbolTable
    dimensions: int


class CNode_to_PSyIR_Visitor(NodeVisitor):
    '''
    A NodeVisitor subclass that takes a c_ast from the pycparser
    library, and converts it into a PSyIR tree for use with PSyclone.

    If an element of the input c_ast has functionality that is not
    supported by PSyIR (or extensions to PSyIR inside this library)
    then it is either lost (if they are properties of Nodes that can
    otherwise be supported by PSyIR - these may be readded by PSyIR
    extensions in the future) or results in a C_CodeBlock, which contains
    the c_ast nodes.
    '''
    # Based on pycparser generator visitor.

    def __init__(self):
        super().__init__()

        self._symbol_tables = []
        self._possible_arrays = []

    def visit(self, node: Node) -> PSyIR.Node:
        '''
        The entry point of the visitor. The specific visitor is found
        using the input node's class name.

        If there is no specific implementation for the input node or
        an Exception is raised while processing the tree, the generic_visit
        function is called, which results in a CodeBlock.

        Note that at current ALL exceptions will result in a CodeBlock,
        including errors in implementations. The latter is unintended
        behaviour, and specificity on what results in a Codeblock will
        be added during development.

        :param node: The input node to convert to PSyIR.

        :returns: The PSyIR node that represents the input node.
        '''
        method = 'visit_' + node.__class__.__name__
        try:
            return getattr(self, method, None)(node)
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
        '''
        Generic visitor for any c_ast node that can't be converted to
        a specific PSyIR node.

        :returns: a C_CodeBlock containing the input node.
        '''
        # For something we don't handle explicitly we need a C_CodeBlock!
        # For now we're just making a CodeBlock for everything and not doing
        # anything smart like fparser2.py - probably better to keep it like
        # this for reverting to pycparser tree later.
        return C_CodeBlock([node], PSyIR.CodeBlock.Structure.STATEMENT)

    def visit_FileAST(self, node: FileAST) -> PSyIR.FileContainer:
        '''
        Visitor for a c_ast FileAST node. This is converted into a
        PSyIR FileContainer containing the subtree of the input node.

        :returns: a FileContainer corresponding to the input node.
        '''
        psyir_children = []
        # TODO Fix symbol tables (urgh)
        symbol_table = PSySym.SymbolTable()
        self._symbol_tables.append(symbol_table)
        for child in node:
            res = self.visit(child)
            if res:
                psyir_children.append(res)
        self._symbol_tables.pop()
        return PSyIR.FileContainer.create("", symbol_table, psyir_children) 

    def visit_FuncDef(self, node: FuncDef) -> PSyIR.Routine:
        '''
        Visitor for a c_ast FuncDef node.

        :returns: a Routine corresponding to the input node.
        '''
        name = node.decl.name

        routine_sym_tab = PSySym.SymbolTable()
        # Add it to the "stack" of symbol tables
        self._symbol_tables.append(routine_sym_tab)
        if node.decl.type.args and len(node.decl.type.args.params) > 0:
            # Time to get the arguments
            args = []
            for decl in node.decl.type.args.params:
                self.visit(decl)
                name = decl.name
                sym = routine_sym_tab.lookup(name)
                sym.interface = PSySym.ArgumentInterface(
                        PSySym.ArgumentInterface.Access.UNKNOWN)
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

        return PSyIR.Routine.create(name, children=psyir_children, symbol_table=routine_sym_tab)

    def _get_struct_element(self, element: Node) -> (str, PSySym.DataType):
        '''
        Helper function to get the PSyIR symbol representation of an element
        of a C struct.

        :param element: An element of a structure definition.

        :returns: The name and datatype of the input element.

        :raises NotImplementedError: If the element contains a Structure.
        :raises NotImplementedError: If the element's type is not convertable
                                     to a PSyIR intrinsic type.
        '''
        if isinstance(element.type, Struct):
            raise NotImplementedError("Structure declaration inside Structure unsupported.")
        if isinstance(element.type.type, Struct):
            raise NotImplementedError("Structure declaration inside Structure unsupported.")
        type_str = element.type.type.names
        if(len(type_str) > 1):
            assert False # Need to think what this means - maybe pointers? Or long long int etc.i
        else:
            type_str = type_str[0]
        # Get the type.
        intrinsic, precision = str_to_type_map.get(type_str, (None, None))
        if intrinsic is None:
            raise NotImplementedError("Unknown intrinsic")
            # There must be some unknown type object we can do for this like we do for CodeBlocks.
        # Create a ScalarType for this.
        datatype = PSySym.ScalarType(intrinsic, precision)
        name = element.name
        return name, datatype


    def _unpack_struct(self, struct: Struct) -> (str, PSySym.StructureType):
        '''
        Helper function to convert a c_ast Struct into a PSyIR StructureType.

        :param struct: The input Struct definition.

        :returns: The name and StructureType corresponding to the input.

        :raises NotImplementedError: If provided an anonymous structure.
        '''
        struct_name = struct.name
        if not struct_name:
            raise NotImplementedError("Anonymous structure unsupported");
        decls = []
        for decl in struct.decls:
            name, element = self._get_struct_element(decl)
            decls.append((name, element, PSySym.Symbol.Visibility.PUBLIC, None))
        decl = PSySym.StructureType.create(decls) 
        return struct_name, decl

    def visit_IdentifierType(self, node: IdentifierType) -> PSySym.DataType:
        '''
        Visitor for a c_ast IdentifierType.

        :param node: Input IdentifierType node.
        
        :returns: A PSyIR DataType corresponding to the input node.

        :raises NotImplementedError: When receiving an unsupported type.
        '''
        type_str = node.names
        if(len(type_str) > 1):
            raise NotImplementedError(f"Unsure how to handle type_str array {type_str}")
        else:
            type_str = type_str[0]
        # Get the type.
        intrinsic, precision = str_to_type_map.get(type_str, (None, None))
        if intrinsic is None:
            raise NotImplementedError(f"Unsupported type {type_str}")
        # TODO Convert type_str to symbol type - see the mapping on SFP.
        # Get the current symbol table. Its always the lowest one.
        return PSySym.ScalarType(intrinsic, precision)

    def visit_Struct(self, node: Struct) -> PSySym.DataTypeSymbol:
        '''
        Visitor for a Struct object.

        :returns: The DataTypeSymbol corresponding to the declaration of this
                  structure's type.

        :raises NotImplementedError: If the structure is not already defined
                                     in a symbol table.
        :raises NotImplementedError: If the input node has a structure type
                                     definition.
        '''
        if node.decls is None:
            # instance of structure declaration without inline declaration.
            # Find the structure type
            for sym_tab in reversed(self._symbol_tables):
                struct = sym_tab.lookup(node.name, otherwise=None)
                if struct:
                    return struct
            # If we didn't find it then code block time.
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def visit_TypeDecl(self, node: TypeDecl) -> PSySym.DataType:
        '''
        Visitor for a c_ast TypeDecl.

        :returns: The PSyIR DataType corresponding to the input node.
        '''
        return self.visit(node.type)

    def visit_PtrDecl(self, node: PtrDecl) -> PSySym.ArrayType:
        '''
        Visitor for a c_ast PtrDecl.

        PSyclone doesn't contain a definition for a pointer symbol,
        so all pointers are converted into deferred extent ArrayTypes.

        :returns: An ArrayType representing this PtrDecl.
        '''
        subtype = self.visit(node.type)
        # Can't determine if this is an array or pointer until later.
        # This mostly doesn't matter for declaration and things, but does
        # matter for accesses in the tree (maybe?)
        # Make it an array with deferred extent.
        if isinstance(subtype, PSySym.ArrayType):
            # If its already defined as an array, we need to extend
            # the dimensions by 1.
            shape = subtype.shape.copy()
            # Reverse indexing, TBC this implementation is correct.
            shape.append(PSySym.ArrayType.Extent.DEFERRED)
            return PSySym.ArrayType(subtype.datatype, shape)
        return PSySym.ArrayType(subtype, [PSySym.ArrayType.Extent.DEFERRED])

    def visit_ArrayDecl(self, node: ArrayDecl) -> PSySym.ArrayType:
        '''
        Visitor for a c_ast ArrayDecl.

        PSyclone treats these as ArrayTypes with explicit shape values.

        :returns: The PSyIR ArrayType corresponding to node.
        '''
        subtype = self.visit(node.type)
        if isinstance(subtype, PSySym.ArrayType):
            # If its already defined as an array, we need to extend the
            # dimensions by 1.
            shape = subtype.shape.copy()
            dim = self.visit(node.dim)
            if isinstance(dim, PSyIR.Literal):
                dim = PSySym.ArrayType.ArrayBounds(
                        PSyIR.Literal("0", PSySym.INTEGER_TYPE),
                        PSyIR.Literal(
                            str(int(dim.value)-1),
                            PSySym.INTEGER_TYPE
                        )
                )
            shape.append(dim)
            return PSySym.ArrayType(subtype.datatype, shape)
        dim = self.visit(node.dim)
        if isinstance(dim, PSyIR.Literal):
            dim = PSySym.ArrayType.ArrayBounds(
                    PSyIR.Literal("0", PSySym.INTEGER_TYPE),
                    PSyIR.Literal(
                        str(int(dim.value)-1),
                        PSySym.INTEGER_TYPE
                    )
            )
        return PSySym.ArrayType(subtype, [dim])


    def visit_Decl(self, node: Decl) -> None:
        '''
        Visitor for a c_ast Decl.

        PSyIR handles declarations differently to c_ast, as each declaration
        is added into a Symbol Table, instead of being a standalone Node.

        This means that passing through PSyIR and back to c_ast can reorder
        variable declaration statements relative to the original code.

        There may be issues currently with Decl's that have initial values, as
        this reordering could result in incorrect values, and so this case
        should probably result in a CodeBlock AND a Symbol, but this is not
        straightforward to support.

        Supported datatypes will be converted into the relevant PSyIR
        declaration, while unsupported types will result in an
        UnsupportedCType being added to the SymbolTable.

        :param node: The input declaration.
        '''
        # FIXME We need to be cautious if the input declaration has an initial
        # value that isn't a constant...
        name = node.name
        typedef = node.type
        # Get the current symbol table. Its always the lowest one.
        sym_tab = self._symbol_tables[-1]
        # Structure declaration makes a CodeBlock for now.
        if isinstance(node.type, Struct):
            try:
                name, decl = self._unpack_struct(node.type)
                sym_tab.new_symbol(name, symbol_type=PSySym.DataTypeSymbol, datatype=decl)
            except NotImplementedError as err:
                # Unsupported structure definitions result in UnsupportedCType
                name = node.type.name
                decl = UnsupportedCType(node)
                sym_tab.new_symbol(name, symbol_type=PSySym.DataTypeSymbol, datatype=decl)
            return
        datatype = self.visit(node.type)
        if isinstance(datatype, PSyIR.CodeBlock):
            datatype = UnsupportedCType(node)
        sym_tab.new_symbol(name, symbol_type=PSySym.DataSymbol, datatype=datatype)
        # If we find a potential array, we need to keep track of it.
        if isinstance(datatype, PSySym.ArrayType):
            self._possible_arrays.append(PossibleArray(name, sym_tab, len(datatype.shape)))


    def visit_ArrayRef(self, node: ArrayRef) -> PSyIR.ArrayReference:
        '''
        A visitor for a c_ast ArrayRef node.

        Currently asserts False if there is no symbol in the symbol table
        corresponding to the name of the input node.

        :returns: An ArrayReference corresponding to the input node.
        '''
        name_node = node.name
        indices = []
        # TODO This is not totally resilient. Index order may be wrong.
        while not isinstance(name_node, ID):
            indices.insert(0, self.visit(name_node.subscript))
            name_node = name_node.name
        name = name_node.name
        sym_tab = self._symbol_tables[-1]
        symbol = sym_tab.symbols_dict.get(name, None)
        if not symbol:
            assert False
        index = self.visit(node.subscript)
        indices.append(index)
        return PSyIR.ArrayReference.create(symbol, indices)

    def visit_Assignment(self, node: Assignment) -> PSyIR.Assignment:
        '''
        Visitor for a c_ast Assignment.

        :returns: A PSyIR Assignment node corresponding to the input node.
        '''
        lhs = self.visit(node.lvalue)
        rhs = self.visit(node.rvalue)
        return PSyIR.Assignment.create(lhs, rhs)

    def visit_ID(self, node: ID) -> PSyIR.Reference:
        '''
        Visitor for a c_ast ID node.

        c_ast ID's don't map neatly to a PSyIR node, but the code uses
        other functionality to avoid hitting this visitor if it doesn't
        expect to receive a PSyIR Reference.

        :returns: A PSyIR Reference corresponding to the input ID.
        '''
        # Get the symbol table.
        sym_tab = self._symbol_tables[-1]
        symbol = sym_tab.symbols_dict.get(node.name, None)
        if not symbol:
            assert False
        # We should probably do better since maybe this isn't just a basic symbol but oh well.
        # TODO
        return PSyIR.Reference(symbol)

    def visit_BinaryOp(self, node: BinaryOp) -> PSyIR.BinaryOperation:
        '''
        Visitor for a c_ast BinaryOp node.

        :returns: A PSyIR BinaryOperation corresponding to the input node.

        :raises NotImplementedError: if the operation of the binary operation
                                     has no Fortran equivalent.
        '''
        lhs = self.visit(node.left)
        # TODO Op needs a string to operator conversion. Think this is on SFP as well
        op = c_to_f_binop_operator_map.get(node.op, None)
        if not op:
            raise NotImplementedError("Unsupported BinaryOperation operator")
        rhs = self.visit(node.right)
        return PSyIR.BinaryOperation.create(op, lhs, rhs)

    def visit_Constant(self, node: Constant) -> PSyIR.Literal:
        '''
        Visitor for a c_ast Constant node.

        :returns: A PsyIR Literal corresponding to the input node.

        :raises AssertionError: if the type of the input node is not
                                'int'.
        '''
        type_str = node.type
        value = node.value
        try:
            dtype = str_to_type_map[type_str]
        except ValueError:
            raise NotImplementedError("Unsupported Literal type")
        return PSyIR.Literal(value, PSySym.ScalarType(dtype[0], dtype[1]))

    def _check_loop_init_validity(self, loop_init: Node) -> (str, PSyIR.Node):
        '''
        Helper function to check the initialisation of a c_ast Loop node.

        :returns: the loop var name and a PSyIR node representing the value
                  of the initialisation.

        :raises NotImplementedError: if loop init operation isn't '='
        :raises NotImplementedError: if the lvalue of the loop init isn't an ID.
        '''
        # Op must be "="
        if loop_init.op != "=":
            raise NotImplementedError("Operations other than = aren't supported on loop init statements")
        if not isinstance(loop_init.lvalue, ID):
            raise NotImplementedError("Got a loop_init.lvalue that isn't an ID, don't understand.")
        loop_var_name = loop_init.lvalue.name
        loop_start = self.visit(loop_init.rvalue)
        return loop_var_name, loop_start

    def _check_loop_stop_validity(self, loop_stop: Node,
                                  loop_var: PSySym.Symbol,
                                  loop_step: PSyIR.Literal) -> PSyIR.Node:
        '''
        Helper function to check the loop stop condition can be converted to PSyIR.

        :param loop_stop: The loop stop c_ast Node.
        :param loop_var: The loop_var Symbol for PSyIR.
        :param loop_step: The loop_step Literal.

        :returns: a PSyIR Node representing the stop condition that can be
                  used in a PSyIR Loop node.
        
        :raises NotImplementedError: If the loop_stop isn't a BinaryOp
                                     node.
        :raises NotImplementedError: If neither the lhs or rhs of loop_stop
                                     are the loop variable.
        :raises NotImplementedError: If the operator of the loop_stop doesn't
                                     match a supported loop structure.
        '''
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


    def _check_loop_step_validity(self, loop_step: Node) -> PSyIR.Node:
        '''
        Helper function to check the loop_step can be represented in
        a PSyIR Loop node.
        Currently only ++ and -- are supported.

        :param loop_step: The loop_step to validate.

        :returns: The PSyIR representation of loop_step.
        
        :raises NotImplementedError: If the loop_step is not supported.
        '''
        if isinstance(loop_step, UnaryOp):
            if loop_step.op == "p++":
                return PSyIR.Literal("1", PSySym.INTEGER_TYPE)
            elif loop_step.op == "p--":
                return PSyIR.Literal("-1", PSySym.INTEGER_TYPE)
            else:
                raise NotImplementedError("Unsupported UnaryOp in loop_step")
        else:
            raise NotImplementedError("Non-unary op steps NYI")

    def visit_For(self, node: For) -> PSyIR.Loop:
        '''
        Visitor for a c_ast For node.

        :returns: a PSyIR Loop corresponding to the input node.

        :raises NotImplementedError: If the For loop has a variable
                                     declaration in its init child.
        :raises NotImplementedError: If the loop_var symbol isn't found in
                                     the symbol table.
        '''
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
        return PSyIR.Loop.create(loop_var, start_cond, stop_cond, step_cond, body)

    def visit_UnaryOp(self, node: UnaryOp) -> PSyIR.UnaryOperation:
        '''
        Visitor for a c_ast UnaryOp node.

        :returns: a UnaryOperation node corresponding to the input node.
        '''
        op = node.op
        expr = self.visit(node.expr)
        return PSyIR.UnaryOperation.create(c_to_f_unary_operator_map[op], expr)

    def visit_If(self, node: If) -> PSyIR.IfBlock:
        '''
        Visitor for a c_ast If node.

        Currently incorrectly annotates all ifblocks with 'was_elseif'

        :returns: a PSyIR IfBlock node corresponding to the input node.
        '''
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
        ifblock = PSyIR.IfBlock.create(cond, if_body, else_body)
        if is_else_if:
            ifblock.annotations.append('was_elseif') # Untested, may be wrong.
        return ifblock

    def visit_Compound(self, node: Compound) -> list:
        '''
        Visitor for a c_ast Compound.

        :returns: a list of nodes corresponding to the input node. This
                  is similar to a PSyIR Schedule, however Compounds can
                  contain Compound children, which PSyIR cannot currently
                  support.
        '''
        result = []
        for child in node:
            result.append(self.visit(child))
        return result


