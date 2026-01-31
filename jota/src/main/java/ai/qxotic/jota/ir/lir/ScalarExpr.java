package ai.qxotic.jota.ir.lir;

/**
 * Sealed interface for scalar expressions in IR-L. Scalar expressions represent single-element
 * computations within loops.
 */
public sealed interface ScalarExpr extends LIRNode
        permits ScalarConst,
                ScalarUnary,
                ScalarBinary,
                ScalarTernary,
                ScalarCast,
                ScalarLoad,
                AccumulatorRead,
                ScalarFromIndex {}
