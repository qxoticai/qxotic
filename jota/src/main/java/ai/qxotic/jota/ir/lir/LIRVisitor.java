package ai.qxotic.jota.ir.lir;

/** Visitor interface for traversing IR-L graphs. */
public interface LIRVisitor<T> {

    // Index expressions
    T visitIndexVar(IndexVar node);

    T visitIndexConst(IndexConst node);

    T visitIndexBinary(IndexBinary node);

    // Scalar expressions
    T visitScalarConst(ScalarConst node);

    T visitScalarUnary(ScalarUnary node);

    T visitScalarBinary(ScalarBinary node);

    T visitScalarTernary(ScalarTernary node);

    T visitScalarCast(ScalarCast node);

    T visitScalarLoad(ScalarLoad node);

    T visitScalarFromIndex(ScalarFromIndex node);

    // Memory access
    T visitBufferRef(BufferRef node);

    T visitLoad(Load node);

    T visitStore(Store node);

    // Accumulators
    T visitAccumulator(Accumulator node);

    T visitAccumulatorRead(AccumulatorRead node);

    T visitAccumulatorUpdate(AccumulatorUpdate node);

    // Loops and control
    T visitLoop(Loop node);

    T visitTiledLoop(TiledLoop node);

    T visitLoopNest(LoopNest node);

    T visitBlock(Block node);
}
