package com.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;
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
        Memory<float[]> memory = Memories.of(new float[16]);
        MemoryAccess<float[]> memoryAccess = MemoryDomains.floats().directAccess();
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
        MemoryDomain<MemorySegment> domain = MemoryDomains.of(MemoryAllocators.newScopedArena());
        Memory<MemorySegment> memory = Memories.of(memorySegment);
        MemoryView<MemorySegment> view =
                MemoryView.of(memory, DataType.I64, Layout.of(Shape.of(10, 10), Stride.of(0, 0)));
        long l = domain.directAccess().readByte(view.memory(), view.byteOffset());
    }

    @Test
    void booleanMemoryIsByteAddressableOnly() {
        Memory<boolean[]> memory = Memories.of(new boolean[4]);
        MemoryAccess<boolean[]> access = MemoryDomains.booleans().directAccess();
        access.writeByte(memory, 1, (byte) 7); // any non-zero byte is true, stored as exactly 1
        assertEquals(1, access.readByte(memory, 1));
        assertTrue(memory.base()[1]);
        assertThrows(
                UnsupportedOperationException.class, () -> access.writeInt(memory, 0, 0x01010101));
        assertThrows(
                UnsupportedOperationException.class, () -> access.writeShort(memory, 0, (short) 1));
        assertThrows(UnsupportedOperationException.class, () -> access.readLong(memory, 0));
        assertThrows(UnsupportedOperationException.class, () -> access.readFloat(memory, 0));
    }
}
