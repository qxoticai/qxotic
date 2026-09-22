package com.qxotic.jinfer.models.parakeet;

import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.util.Map;

/** A checkpoint's tensors by name. */
record Tensors(Map<String, MemoryView<MemorySegment>> byName) {

    MemoryView<MemorySegment> require(String name) {
        MemoryView<MemorySegment> value = byName.get(name);
        if (value == null) throw new IllegalStateException("parakeet tensor missing: " + name);
        return value;
    }

    /** The named tensor, or null if the checkpoint has none. */
    MemoryView<MemorySegment> optional(String name) {
        return byName.get(name);
    }

    /**
     * A weight for {@code MatMul}, which takes its dimensions from the call, not the view:
     * quantized views report block counts, so only the leading (output) extent is checked.
     */
    MemoryView<MemorySegment> weight(String name, int out) {
        MemoryView<MemorySegment> value = require(name);
        if (value.shape().flatAt(0) != out)
            throw new IllegalArgumentException(
                    name + ": expected leading extent " + out + " but was " + value.shape());
        return value;
    }

    /** A tensor of exactly {@code length} elements. */
    MemoryView<MemorySegment> vector(String name, int length) {
        MemoryView<MemorySegment> value = require(name);
        if (value.shape().size() != length)
            throw new IllegalArgumentException(
                    name + ": expected " + length + " elements but was " + value.shape());
        return value;
    }

    /** A small F32 tensor, copied to the heap. */
    float[] floats(String name, int length) {
        return Views.toFloatArray(vector(name, length), name);
    }
}
