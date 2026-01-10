package com.qxotic.jota.memory;

import com.qxotic.jota.DataType;
import com.qxotic.jota.memory.impl.ContextFactory;
import com.qxotic.jota.memory.impl.MemoryAccessFactory;
import com.qxotic.jota.memory.impl.MemoryFactory;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.function.Supplier;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class MemoryAccessTest {

    public static Stream<Supplier<Context<?>>> contextProvider() {
        return Stream.of(
                ContextFactory::ofFloats,
                ContextFactory::ofMemorySegment
        );
    }

    @ParameterizedTest
    @MethodSource("contextProvider")
    <B> void testFloatAccess(Supplier<Context<B>> contextSupplier) {
        try (var context = contextSupplier.get()) {
            var allocator = context.memoryAllocator();
            Memory<B> memory = allocator.allocateMemory(DataType.F32, 16);
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
