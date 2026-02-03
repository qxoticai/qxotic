package ai.qxotic.jota.ir.lir;

/** Visitor interface for traversing IR-L graphs. */
public interface LIRVisitor<T> {

    // Index expressions
    T visitIndexVar(IndexVar node);

    T visitIndexConst(IndexConst node);

    T visitIndexBinary(IndexBinary node);

    // Scalar expressions
    T visitScalarLiteral(ScalarLiteral node);

    T visitScalarUnary(ScalarUnary node);

    T visitScalarBinary(ScalarBinary node);

    T visitScalarTernary(ScalarTernary node);

    T visitScalarCast(ScalarCast node);

    T visitScalarLoad(ScalarLoad node);

    T visitScalarInput(ScalarInput node);

    T visitScalarFromIndex(ScalarFromIndex node);

    T visitScalarRef(ScalarRef node);

    // Let binding (for hoisted values)
    T visitScalarLet(ScalarLet node);

    // Memory access
    T visitBufferRef(BufferRef node);

    T visitLoad(Load node);

    T visitStore(Store node);

    // Loops and control
    T visitStructuredFor(StructuredFor node);

    T visitTiledLoop(TiledLoop node);

    T visitBlock(Block node);

    T visitYield(Yield node);
}
