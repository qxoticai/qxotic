package com.llm4j.jota.memory;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.function.Supplier;
import java.util.stream.Stream;

public class MemoryAccessTest {

//    static Stream<Supplier<Context<?>>> contextProvider() {
//        return Stream.of(ManagedContext::new, NativeContext::new);
//    }
//
//    @ParameterizedTest
//    @MethodSource("contextProvider")
//    <B> void testFloatAccess(Supplier<Context<B>> contextSupplier) {
//        try (var context = contextSupplier.get()) {
//            var allocator = context.memoryAllocator();
//            Memory<B> memory = allocator.allocateMemory(16 * Float.BYTES);
//            MemoryAccess<B> memoryAccess = context.memoryAccess();
//            for (int i = 0; i < 4; ++i) {
//                memoryAccess.writeFloat(memory, i * Float.BYTES, i * (float) Math.PI);
//            }
//            for (int i = 0; i < 4; ++i) {
//                Assertions.assertEquals(i * (float) Math.PI, memoryAccess.readFloat(memory, i * Float.BYTES));
//            }
//        }
//    }
//
//    @Test
//    void testFloatMemoryAccess() {
//        Memory<float[]> memory = new FloatArrayMemory(new float[16]);
//        MemoryAccess<float[]> memoryAccess = FloatArrayMemoryAccess.instance();
//        for (int i = 0; i < 4; ++i) {
//            memoryAccess.writeFloat(memory, i * Float.BYTES, i * (float) Math.PI);
//        }
//        for (int i = 0; i < 4; ++i) {
//            Assertions.assertEquals(i * (float) Math.PI, memoryAccess.readFloat(memory, i * Float.BYTES));
//        }
//    }
}
