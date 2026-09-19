package com.qxotic.jinfer.models.parakeet;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Map;

/** Name-driven tensor access shared by the Parakeet loaders. */
final class Tensors {

    private Tensors() {}

    static MemoryView<MemorySegment> require(
            Map<String, MemoryView<MemorySegment>> tensors, String name) {
        MemoryView<MemorySegment> value = tensors.get(name);
        if (value == null) throw new IllegalStateException("parakeet tensor missing: " + name);
        return value;
    }

    /**
     * A weight for {@code MatMul.gemm}, which takes its dimensions from the call, not the view:
     * quantized views report block counts, so only the leading (output) extent is checked.
     */
    static MemoryView<MemorySegment> weight(
            Map<String, MemoryView<MemorySegment>> tensors, String name, int out) {
        MemoryView<MemorySegment> value = require(tensors, name);
        if (value.shape().flatAt(0) != out)
            throw new IllegalArgumentException(
                    name + ": expected leading extent " + out + " but was " + value.shape());
        return value;
    }

    /** A small tensor dequantized to a float array of the exact expected length. */
    static float[] floats(Map<String, MemoryView<MemorySegment>> tensors, String name, int length) {
        MemoryView<MemorySegment> value = require(tensors, name);
        if (value.shape().size() != length)
            throw new IllegalArgumentException(
                    name + ": expected " + length + " elements but was " + value.shape());
        // newCrossThread, not ofShared: a native image cannot close a shared arena
        Arena arena = Arenas.newCrossThread();
        try {
            MemoryView<MemorySegment> decoded =
                    Views.allocateF32(MemoryAllocators.ofArena(arena), length);
            Convert.copyToF32(value, 0, decoded, 0, length);
            return Views.toFloatArray(decoded, name);
        } finally {
            Arenas.close(arena);
        }
    }
}
