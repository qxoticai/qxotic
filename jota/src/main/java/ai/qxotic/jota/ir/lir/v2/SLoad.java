package ai.qxotic.jota.ir.lir.v2;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.lir.BufferRef;

public final class SLoad extends ScalarNode {
    private final BufferRef buffer;

    SLoad(int id, BufferRef buffer, V2Node offset, DataType dataType) {
        super(id, V2Kind.S_LOAD, dataType, new V2Node[] {offset}, false, false);
        this.buffer = buffer;
    }

    public BufferRef buffer() {
        return buffer;
    }

    public V2Node offset() {
        return inputs()[0];
    }

    @Override
    public V2Node canonicalize(LirV2Graph graph) {
        return this;
    }
}
