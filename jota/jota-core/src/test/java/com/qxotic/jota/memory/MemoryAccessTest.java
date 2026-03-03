package com.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;
import com.qxotic.jota.memory.impl.DomainFactory;
import com.qxotic.jota.memory.impl.MemoryAccessFactory;
import com.qxotic.jota.memory.impl.MemoryFactory;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

public class MemoryAccessTest {

    @ParameterizedTest
    @MethodSource("com.qxotic.jota.memory.AbstractMemoryTest#domainsSupportingF32")
    <B> void testFloatAccess(MemoryDomain<B> domain) {
        try (domain) {
            var allocator = domain.memoryAllocator();
            Memory<B> memory = allocator.allocateMemory(DataType.FP32, 16);
            MemoryAccess<B> memoryAccess = domain.directAccess();
            for (int i = 0; i < 4; ++i) {
                memoryAccess.writeFloat(memory, i * Float.BYTES, i * (float) Math.PI);
            }
            for (int i = 0; i < 4; ++i) {
                assertEquals(i * (float) Math.PI, memoryAccess.readFloat(memory, i * Float.BYTES));
            }
        }
    }

    @Test
    void testFloatMemoryAccess() {
        Memory<float[]> memory = MemoryFactory.ofFloats(new float[16]);
        MemoryAccess<float[]> memoryAccess = MemoryAccessFactory.ofFloats();
        for (int i = 0; i < 4; ++i) {
            memoryAccess.writeFloat(memory, i * Float.BYTES, i * (float) Math.PI);
        }
        for (int i = 0; i < 4; ++i) {
            assertEquals(i * (float) Math.PI, memoryAccess.readFloat(memory, i * Float.BYTES));
        }
    }

    @Test
    void canary() {
        MemorySegment memorySegment = MemorySegment.ofArray(new long[] {Long.MAX_VALUE});
        MemoryDomain<MemorySegment> domain = DomainFactory.ofMemorySegment();
        Memory<MemorySegment> memory = MemoryFactory.ofMemorySegment(memorySegment);
        MemoryView<MemorySegment> view =
                MemoryView.of(memory, DataType.I64, Layout.of(Shape.of(10, 10), Stride.of(0, 0)));
        long l = domain.directAccess().readByte(view.memory(), view.byteOffset());
        System.out.println(l);
    }
}
