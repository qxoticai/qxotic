package com.qxotic.jota.ir.tir;

/**
 * Reduction operators for IR-T. Kept separate from the tensor package to maintain IR-T
 * independence.
 */
public enum ReductionOperator {
    SUM,
    PROD,
    MIN,
    MAX
}
