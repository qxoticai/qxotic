package ai.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.memory.impl.MemoryAccessFactory;
import ai.qxotic.jota.memory.impl.MemoryFactory;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

public class MemoryAccessTest {

    @ParameterizedTest
    @MethodSource("ai.qxotic.jota.memory.AbstractMemoryTest#contextsSupportingF32")
    <B> void testFloatAccess(MemoryContext<B> context) {
        try (context) {
            var allocator = context.memoryAllocator();
            Memory<B> memory = allocator.allocateMemory(DataType.FP32, 16);
            MemoryAccess<B> memoryAccess = context.memoryAccess();
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
}
