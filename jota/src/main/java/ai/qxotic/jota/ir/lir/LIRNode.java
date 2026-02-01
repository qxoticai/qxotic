package ai.qxotic.jota.ir.lir;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.base.IRNode;

/**
 * Base interface for all IR-L (Loop-level IR) nodes. IR-L represents explicit loop structures,
 * index expressions, and memory accesses in a device-agnostic way.
 */
public sealed interface LIRNode extends IRNode
        permits IndexExpr,
                ScalarExpr,
                LIRInput,
                Load,
                Store,
                Loop,
                StructuredFor,
                TiledLoop,
                LoopNest,
                Block,
                Yield,
                ScalarLet {

    /** Accepts a visitor for traversing the IR-L graph. */
    default <T> T accept(LIRVisitor<T> visitor) {
        return switch (this) {
            case IndexVar n -> visitor.visitIndexVar(n);
            case IndexConst n -> visitor.visitIndexConst(n);
            case IndexBinary n -> visitor.visitIndexBinary(n);
            case ScalarLiteral n -> visitor.visitScalarLiteral(n);
            case ScalarUnary n -> visitor.visitScalarUnary(n);
            case ScalarBinary n -> visitor.visitScalarBinary(n);
            case ScalarTernary n -> visitor.visitScalarTernary(n);
            case ScalarCast n -> visitor.visitScalarCast(n);
            case ScalarLoad n -> visitor.visitScalarLoad(n);
            case ScalarInput n -> visitor.visitScalarInput(n);
            case ScalarFromIndex n -> visitor.visitScalarFromIndex(n);
            case ScalarRef n -> visitor.visitScalarRef(n);
            case BufferRef n -> visitor.visitBufferRef(n);
            case Load n -> visitor.visitLoad(n);
            case Store n -> visitor.visitStore(n);
            case Loop n -> visitor.visitLoop(n);
            case StructuredFor n -> visitor.visitStructuredFor(n);
            case TiledLoop n -> visitor.visitTiledLoop(n);
            case LoopNest n -> visitor.visitLoopNest(n);
            case Block n -> visitor.visitBlock(n);
            case Yield n -> visitor.visitYield(n);
            case ScalarLet n -> visitor.visitScalarLet(n);
        };
    }

    /** Returns the data type of this node, or null if not applicable (e.g., control flow). */
    @Override
    default DataType dataType() {
        return null;
    }
}
