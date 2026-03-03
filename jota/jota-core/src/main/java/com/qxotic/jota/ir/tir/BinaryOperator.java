package com.qxotic.jota.ir.tir;

/**
 * Binary operators for IR-T. Kept separate from the tensor package to maintain IR-T independence.
 */
public enum BinaryOperator {
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE,
    MIN,
    MAX,
    POW,

    LOGICAL_AND,
    LOGICAL_OR,
    LOGICAL_XOR,
    BITWISE_AND,
    BITWISE_OR,
    BITWISE_XOR,
    SHIFT_LEFT,
    SHIFT_RIGHT,
    SHIFT_RIGHT_UNSIGNED,

    EQUAL,
    LESS_THAN
}
