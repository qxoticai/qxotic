package com.qxotic.jinfer.x.llm;

import com.qxotic.jinfer.x.PanamaMemoryArena;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

final class TestLogits {
    private TestLogits() {}

    static MemoryView<MemorySegment> view(long size) {
        return Views.allocateF32(new PanamaMemoryArena(Arena.ofAuto()), size);
    }

    static MemoryView<MemorySegment> view(float... values) {
        return Views.fromFloatArray(new PanamaMemoryArena(Arena.ofAuto()), values);
    }

    static long size(MemoryView<?> view) {
        return view.shape().size();
    }

    static float get(MemoryView<?> view, long index) {
        MemoryView<MemorySegment> memory = Views.castToSegmentBacked(view, "logits");
        return memory.memory()
                .base()
                .get(ValueLayout.JAVA_FLOAT, memory.byteOffset() + index * Float.BYTES);
    }

    static void set(MemoryView<?> view, long index, float value) {
        MemoryView<MemorySegment> memory = Views.castToSegmentBacked(view, "logits");
        memory.memory()
                .base()
                .set(ValueLayout.JAVA_FLOAT, memory.byteOffset() + index * Float.BYTES, value);
    }
}
