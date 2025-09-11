package com.llm4j.jota.memory;

import com.llm4j.jota.memory.impl.MemoryAllocatorFactory;
import com.llm4j.jota.memory.impl.ContextFactory;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.function.Supplier;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MemoryOperationsTest {

    static Stream<Context<?>> contextProvider() {
        Stream<Supplier<Context<?>>> lazy = Stream.of(
                () -> ContextFactory.ofFloats(),
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(false)),
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(true)),
                () -> ContextFactory.ofMemorySegment()
        );
        // Lazy context creation.
        return lazy.map(Supplier::get);
    }

    @ParameterizedTest
    @MethodSource("contextProvider")
    <B> void copyShouldTransferDataBetweenBuffers(Context<B> context) {
        MemoryAllocator<B> memoryAllocator = context.memoryAllocator();
        MemoryOperations<B> memoryOperations = context.memoryOperations();
        MemoryAccess<B> memoryAccess = context.memoryAccess();

        Memory<B> src = memoryAllocator.allocateMemory(8);
        Memory<B> dst = memoryAllocator.allocateMemory(8);

        // Initialize source
        float value = (float) Math.PI;
        memoryAccess.writeFloat(src, 0, value);

        // Perform copy
        memoryOperations.copy(src, 0, dst, 0, 4);

        // Verify
        assertEquals(value, memoryAccess.readFloat(dst, 0));
    }

//    static float[] canary() {
//        return new float[]{1.2f, 3.4f, 4.5f, -1f, -0f, Float.NaN, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY, 0f, 1f, (float) Math.PI};
//    }
//
//    @ParameterizedTest
//    @MethodSource("contextProvider")
//    <B> void copyFromNativeShouldImportData(Context<B> context) {
//        MemoryAllocator<B> memoryAllocator = context.memoryAllocator();
//        MemoryOperations<B> memoryOperations = context.memoryOperations();
//        MemoryAccess<B> memoryAccess = context.memoryAccess();
//
//        Memory<MemorySegment> nativeMemory = MemoryFactory.ofMemorySegment(MemorySegment.ofBuffer(byteBuffer));
//        Memory<B> local = memoryAllocator.allocateMemory(DataType.F32, Shape.of(n));
//
//        memoryOperations.copyFromNative(nativeMemory, 0, local, 0, n * Float.BYTES);
//        for (int i = 0; i < n; ++i) {
//            assertEquals(canary()[i], memoryAccess.readFloat(local, i * Float.BYTES));
//        }
//    }
//
//    @ParameterizedTest
//    @MethodSource("contextProvider")
//    <B> void copyToNativeShouldExportData(Context<B> context) {
//        MemoryAllocator<B> memoryAllocator = context.memoryAllocator();
//        MemoryOperations<B> memoryOperations = context.memoryOperations();
//        MemoryAccess<B> memoryAccess = context.memoryAccess();
//
//        int n = canary().length;
//        Memory<MemorySegment> nativeMemory = MemoryFactory.ofMemorySegment(MemorySegment.ofArray(new float[n]));
//        Memory<B> local = memoryAllocator.allocateMemory(DataType.F32, Shape.of(n));
//
//        // Initialize local memory
//        ctx.access.writeInt(local, 0, 0xABCD1234);
//
//        memoryOperations.copyToNative(local, 0, nativeMemory, 0, n * Float.BYTES);
//    }
}
