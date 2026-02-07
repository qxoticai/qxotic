package ai.qxotic.jota.ir.lir;

import ai.qxotic.jota.DataType;

public final class SLoad extends ScalarNode {
    private final BufferRef buffer;

    SLoad(int id, BufferRef buffer, LIRExprNode offset, DataType dataType) {
        super(id, LIRExprKind.S_LOAD, dataType, new LIRExprNode[] {offset}, false, false);
        this.buffer = buffer;
    }

    public BufferRef buffer() {
        return buffer;
    }

    public LIRExprNode offset() {
        return inputs()[0];
    }

    @Override
    public LIRExprNode canonicalize(LIRExprGraph graph) {
        return this;
    }
}
