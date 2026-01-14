package com.qxotic.jota.memory;

import com.qxotic.jota.DataType;
import com.qxotic.jota.memory.impl.ContextFactory;
import com.qxotic.jota.memory.impl.MemoryAccessFactory;
import com.qxotic.jota.memory.impl.MemoryAllocatorFactory;
import com.qxotic.jota.memory.impl.MemoryFactory;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MemoryOperationsComprehensiveTest {

    private static final int ELEMENT_COUNT = 8;
    private static final DataType[] DATA_TYPES = {
            DataType.I8,
            DataType.I16,
            DataType.I32,
            DataType.I64,
            DataType.F32,
            DataType.F64
    };

    static Stream<MemoryContext<?>> allContexts() {
        return Stream.concat(
                AbstractMemoryTest.onHeapContexts(),
                Stream.of(
                        ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(false)),
                        ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(true)),
                        ContextFactory.ofMemorySegment()
                )
        );
    }

    @ParameterizedTest
    @MethodSource("allContexts")
    <B> void copyCopiesValues(MemoryContext<B> context) {
        try (context) {
            MemoryAccess<B> memoryAccess = context.memoryAccess();
            Assumptions.assumeTrue(memoryAccess != null, "memory access required");

            MemoryAllocator<B> allocator = context.memoryAllocator();
            DataType dataType = dataTypeFor(allocator);
            long byteSize = dataType.byteSizeFor(ELEMENT_COUNT);

            Memory<B> src = allocator.allocateMemory(byteSize);
            Memory<B> dst = allocator.allocateMemory(byteSize);

            for (int i = 0; i < ELEMENT_COUNT; i++) {
                writeValue(memoryAccess, src, dataType, i);
            }

            context.memoryOperations().copy(src, 0, dst, 0, byteSize);

            for (int i = 0; i < ELEMENT_COUNT; i++) {
                assertValue(memoryAccess, dst, dataType, i);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("allContexts")
    <B> void fillWritesExpectedValues(MemoryContext<B> context) {
        try (context) {
            MemoryAccess<B> memoryAccess = context.memoryAccess();
            Assumptions.assumeTrue(memoryAccess != null, "memory access required");

            MemoryAllocator<B> allocator = context.memoryAllocator();
            MemoryOperations<B> memoryOperations = context.memoryOperations();

            for (DataType dataType : DATA_TYPES) {
                Memory<B> memory = allocateMemory(allocator, dataType);
                if (memory == null || !memory.supportsDataType(dataType)) {
                    continue;
                }
                try {
                    fill(memory, memoryOperations, dataType);
                } catch (UnsupportedOperationException ex) {
                    continue;
                }

                for (int i = 0; i < ELEMENT_COUNT; i++) {
                    assertFillValue(memoryAccess, memory, dataType, i);
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("allContexts")
    <B> void copyToNativeCopiesValues(MemoryContext<B> context) {
        try (context) {
            MemoryAccess<B> memoryAccess = context.memoryAccess();
            Assumptions.assumeTrue(memoryAccess != null, "memory access required");

            MemoryOperations<B> memoryOperations = context.memoryOperations();
            MemoryAllocator<B> allocator = context.memoryAllocator();
            MemoryAccess<MemorySegment> nativeAccess = MemoryAccessFactory.ofMemorySegment();

            for (DataType dataType : DATA_TYPES) {
                Memory<B> local = allocateMemory(allocator, dataType);
                if (local == null || !local.supportsDataType(dataType)) {
                    continue;
                }
                long byteSize = dataType.byteSizeFor(ELEMENT_COUNT);
                Memory<MemorySegment> nativeMemory = allocateNativeMemory(dataType);

                for (int i = 0; i < ELEMENT_COUNT; i++) {
                    writeValue(memoryAccess, local, dataType, i, 0);
                }

                memoryOperations.copyToNative(local, 0, nativeMemory, 0, byteSize);

                for (int i = 0; i < ELEMENT_COUNT; i++) {
                    assertValue(nativeAccess, nativeMemory, dataType, i, 0);
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("allContexts")
    <B> void copyFromNativeCopiesValues(MemoryContext<B> context) {
        try (context) {
            MemoryAccess<B> memoryAccess = context.memoryAccess();
            Assumptions.assumeTrue(memoryAccess != null, "memory access required");

            MemoryOperations<B> memoryOperations = context.memoryOperations();
            MemoryAllocator<B> allocator = context.memoryAllocator();
            MemoryAccess<MemorySegment> nativeAccess = MemoryAccessFactory.ofMemorySegment();

            for (DataType dataType : DATA_TYPES) {
                Memory<B> local = allocateMemory(allocator, dataType);
                if (local == null || !local.supportsDataType(dataType)) {
                    continue;
                }
                long byteSize = dataType.byteSizeFor(ELEMENT_COUNT);
                Memory<MemorySegment> nativeMemory = allocateNativeMemory(dataType);

                for (int i = 0; i < ELEMENT_COUNT; i++) {
                    writeValue(nativeAccess, nativeMemory, dataType, i, 50);
                }

                memoryOperations.copyFromNative(nativeMemory, 0, local, 0, byteSize);

                for (int i = 0; i < ELEMENT_COUNT; i++) {
                    assertValue(memoryAccess, local, dataType, i, 50);
                }
            }
        }
    }

    private static <B> DataType dataTypeFor(MemoryAllocator<B> allocator) {
        Memory<B> probe = allocator.allocateMemory(Long.BYTES * ELEMENT_COUNT);
        Object base = probe.base();
        if (base instanceof float[]) {
            return DataType.F32;
        }
        if (base instanceof double[]) {
            return DataType.F64;
        }
        if (base instanceof long[]) {
            return DataType.I64;
        }
        if (base instanceof int[]) {
            return DataType.I32;
        }
        if (base instanceof short[]) {
            return DataType.I16;
        }
        if (base instanceof byte[]) {
            return DataType.I8;
        }
        if (base instanceof ByteBuffer) {
            return DataType.F32;
        }
        if (base instanceof MemorySegment) {
            return DataType.I8;
        }
        throw new IllegalArgumentException("Unsupported memory base: " + base.getClass().getName());
    }

    private static <B> Memory<B> allocateMemory(MemoryAllocator<B> allocator, DataType dataType) {
        try {
            return allocator.allocateMemory(dataType, ELEMENT_COUNT);
        } catch (RuntimeException ex) {
            return null;
        }
    }

    private static Memory<MemorySegment> allocateNativeMemory(DataType dataType) {
        if (dataType == DataType.F32) {
            return MemoryFactory.ofMemorySegment(MemorySegment.ofArray(new float[ELEMENT_COUNT]));
        }
        if (dataType == DataType.F64) {
            return MemoryFactory.ofMemorySegment(MemorySegment.ofArray(new double[ELEMENT_COUNT]));
        }
        if (dataType == DataType.I64) {
            return MemoryFactory.ofMemorySegment(MemorySegment.ofArray(new long[ELEMENT_COUNT]));
        }
        if (dataType == DataType.I32) {
            return MemoryFactory.ofMemorySegment(MemorySegment.ofArray(new int[ELEMENT_COUNT]));
        }
        if (dataType == DataType.I16) {
            return MemoryFactory.ofMemorySegment(MemorySegment.ofArray(new short[ELEMENT_COUNT]));
        }
        if (dataType == DataType.I8) {
            return MemoryFactory.ofMemorySegment(MemorySegment.ofArray(new byte[ELEMENT_COUNT]));
        }
        throw new IllegalArgumentException("Unsupported data type: " + dataType);
    }

    private static <B> void writeValue(MemoryAccess<B> memoryAccess, Memory<B> memory, DataType dataType, int index) {
        writeValue(memoryAccess, memory, dataType, index, 0);
    }

    private static <B> void writeValue(MemoryAccess<B> memoryAccess, Memory<B> memory, DataType dataType, int index, int baseOffset) {
        long byteOffset = dataType.byteSize() * index;
        if (dataType == DataType.F32) {
            memoryAccess.writeFloat(memory, byteOffset, 1.5f + baseOffset + index);
        } else if (dataType == DataType.F64) {
            memoryAccess.writeDouble(memory, byteOffset, 2.5 + baseOffset + index);
        } else if (dataType == DataType.I64) {
            memoryAccess.writeLong(memory, byteOffset, 10000L + baseOffset + index);
        } else if (dataType == DataType.I32) {
            memoryAccess.writeInt(memory, byteOffset, 1000 + baseOffset + index);
        } else if (dataType == DataType.I16) {
            memoryAccess.writeShort(memory, byteOffset, (short) (100 + baseOffset + index));
        } else if (dataType == DataType.I8) {
            memoryAccess.writeByte(memory, byteOffset, (byte) (10 + baseOffset + index));
        } else {
            throw new IllegalArgumentException("Unsupported data type: " + dataType);
        }
    }

    private static <B> void assertValue(MemoryAccess<B> memoryAccess, Memory<B> memory, DataType dataType, int index) {
        assertValue(memoryAccess, memory, dataType, index, 0);
    }

    private static <B> void assertValue(MemoryAccess<B> memoryAccess, Memory<B> memory, DataType dataType, int index, int baseOffset) {
        long byteOffset = dataType.byteSize() * index;
        if (dataType == DataType.F32) {
            assertEquals(1.5f + baseOffset + index, memoryAccess.readFloat(memory, byteOffset), 0.0f);
        } else if (dataType == DataType.F64) {
            assertEquals(2.5 + baseOffset + index, memoryAccess.readDouble(memory, byteOffset), 0.0);
        } else if (dataType == DataType.I64) {
            assertEquals(10000L + baseOffset + index, memoryAccess.readLong(memory, byteOffset));
        } else if (dataType == DataType.I32) {
            assertEquals(1000 + baseOffset + index, memoryAccess.readInt(memory, byteOffset));
        } else if (dataType == DataType.I16) {
            assertEquals((short) (100 + baseOffset + index), memoryAccess.readShort(memory, byteOffset));
        } else if (dataType == DataType.I8) {
            assertEquals((byte) (10 + baseOffset + index), memoryAccess.readByte(memory, byteOffset));
        } else {
            throw new IllegalArgumentException("Unsupported data type: " + dataType);
        }
    }

    private static <B> void assertFillValue(MemoryAccess<B> memoryAccess, Memory<B> memory, DataType dataType, int index) {
        long byteOffset = dataType.byteSize() * index;
        if (dataType == DataType.F32) {
            assertEquals(1.5f, memoryAccess.readFloat(memory, byteOffset), 0.0f);
        } else if (dataType == DataType.F64) {
            assertEquals(2.5, memoryAccess.readDouble(memory, byteOffset), 0.0);
        } else if (dataType == DataType.I64) {
            assertEquals(10000L, memoryAccess.readLong(memory, byteOffset));
        } else if (dataType == DataType.I32) {
            assertEquals(1000, memoryAccess.readInt(memory, byteOffset));
        } else if (dataType == DataType.I16) {
            assertEquals((short) 100, memoryAccess.readShort(memory, byteOffset));
        } else if (dataType == DataType.I8) {
            assertEquals((byte) 10, memoryAccess.readByte(memory, byteOffset));
        } else {
            throw new IllegalArgumentException("Unsupported data type: " + dataType);
        }
    }

    private static <B> void fill(Memory<B> memory, MemoryOperations<B> memoryOperations, DataType dataType) {
        long byteSize = memory.byteSize();
        if (dataType == DataType.F32) {
            memoryOperations.fillFloat(memory, 0, byteSize, 1.5f);
        } else if (dataType == DataType.F64) {
            memoryOperations.fillDouble(memory, 0, byteSize, 2.5);
        } else if (dataType == DataType.I64) {
            memoryOperations.fillLong(memory, 0, byteSize, 10000L);
        } else if (dataType == DataType.I32) {
            memoryOperations.fillInt(memory, 0, byteSize, 1000);
        } else if (dataType == DataType.I16) {
            memoryOperations.fillShort(memory, 0, byteSize, (short) 100);
        } else if (dataType == DataType.I8) {
            memoryOperations.fillByte(memory, 0, byteSize, (byte) 10);
        } else {
            throw new IllegalArgumentException("Unsupported data type: " + dataType);
        }
    }
}
