from psyclone.psyir.nodes import BinaryOperation, UnaryOperation
from psyclone.psyir.symbols import ScalarType

type_map = {ScalarType.Intrinsic.INTEGER: {ScalarType.Precision.SINGLE: "int", ScalarType.Precision.DOUBLE: "long long int",
                                           ScalarType.Precision.UNDEFINED: "int", 32: "int32_t", 64: "int64_t", 8: "int8_t"},
                ScalarType.Intrinsic.REAL: {ScalarType.Precision.SINGLE: "float", ScalarType.Precision.DOUBLE: "double",
                                            },
#                ScalarType.Intrinsic.CHARACTER: {ScalarType.Precision.UNDEFINED: "char*"},
                ScalarType.Intrinsic.BOOLEAN: {ScalarType.Precision.UNDEFINED: "bool"}
               }

def _create_str_to_type_map(type_map: map) -> map:
    str_to_type_map = {}
    for intrinsic in type_map.keys():
        for precision in type_map[intrinsic]:
            string = type_map[intrinsic][precision]
            if string not in str_to_type_map.keys():
                str_to_type_map[string] = (intrinsic, precision)
    return str_to_type_map

str_to_type_map = _create_str_to_type_map(type_map)

def _invert_map(in_map: map) -> map:
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

f_to_c_binop_operator_map = _invert_map(c_to_f_binop_operator_map)

c_to_f_unary_operator_map = {
        "-": UnaryOperation.Operator.MINUS,
        "+": UnaryOperation.Operator.PLUS,
        "!": UnaryOperation.Operator.NOT,
}

f_to_c_unary_operator_map = _invert_map(c_to_f_unary_operator_map)
