package ai.qxotic.jota.ir.irl;

/**
 * Sealed interface for scalar expressions in IR-L. Scalar expressions represent single-element
 * computations within loops.
 */
public sealed interface ScalarExpr extends IRLNode
        permits ScalarConst, ScalarUnary, ScalarBinary, ScalarTernary, ScalarCast, ScalarLoad {}
