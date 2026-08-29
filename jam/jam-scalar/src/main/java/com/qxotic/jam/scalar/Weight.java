package com.qxotic.jam.scalar;

import com.qxotic.jam.internal.GGMLType;
import java.lang.foreign.MemorySegment;

/** One matmul's weight operand: the segment, its byte offset, dtype and element row stride. */
record Weight(MemorySegment seg, long off, GGMLType type, int ldw) {

    /** Byte offset of row {@code i}'s first block. */
    long row(int i) {
        return off + type.rowBytes((long) i * ldw);
    }

    int blockElems() {
        return type.elementsPerBlock();
    }
}
