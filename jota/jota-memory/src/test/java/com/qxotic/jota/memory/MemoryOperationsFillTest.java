package com.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import com.qxotic.jota.memory.impl.DomainFactory;
import com.qxotic.jota.memory.impl.MemoryAllocatorFactory;
import com.qxotic.jota.memory.impl.MemoryFactory;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

class MemoryOperationsFillTest {

    private static final int BYTE_SIZE = 16;

    static Stream<MemoryDomain<?>> domains() {
        return Stream.concat(
                AbstractMemoryTest.onHeapDomains(),
                Stream.of(
                        DomainFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(false)),
                        DomainFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(true)),
                        DomainFactory.ofMemorySegment()));
    }

    @ParameterizedTest
    @MethodSource("domains")
    <B> void everyFillWidthWritesTheExpectedBytes(MemoryDomain<B> domain) {
        try (domain) {
            for (Fill fill : Fill.values()) {
                Memory<B> memory = domain.memoryAllocator().allocateMemory(BYTE_SIZE);
                fill.apply(domain.memoryOperations(), memory);

                Memory<MemorySegment> nativeCopy =
                        MemoryFactory.ofMemorySegment(MemorySegment.ofArray(new byte[BYTE_SIZE]));
                domain.memoryOperations().copyToNative(memory, 0, nativeCopy, 0, BYTE_SIZE);

                assertArrayEquals(
                        fill.expectedBytes(),
                        nativeCopy.base().toArray(ValueLayout.JAVA_BYTE),
                        domain.memoryAllocator().getClass().getSimpleName() + " " + fill);
            }
        }
    }

    private enum Fill {
        BYTE(1),
        SHORT(Short.BYTES),
        INT(Integer.BYTES),
        LONG(Long.BYTES),
        FLOAT(Float.BYTES),
        DOUBLE(Double.BYTES);

        private final int width;

        Fill(int width) {
            this.width = width;
        }

        <B> void apply(MemoryOperations<B> operations, Memory<B> memory) {
            switch (this) {
                case BYTE -> operations.fillByte(memory, 0, BYTE_SIZE, (byte) 0x12);
                case SHORT -> operations.fillShort(memory, 0, BYTE_SIZE, (short) 0x1234);
                case INT -> operations.fillInt(memory, 0, BYTE_SIZE, 0x12345678);
                case LONG -> operations.fillLong(memory, 0, BYTE_SIZE, 0x0123456789ABCDEFL);
                case FLOAT -> operations.fillFloat(memory, 0, BYTE_SIZE, 1.5f);
                case DOUBLE -> operations.fillDouble(memory, 0, BYTE_SIZE, 1.5);
            }
        }

        byte[] expectedBytes() {
            ByteBuffer value = ByteBuffer.allocate(width).order(ByteOrder.nativeOrder());
            switch (this) {
                case BYTE -> value.put((byte) 0x12);
                case SHORT -> value.putShort((short) 0x1234);
                case INT -> value.putInt(0x12345678);
                case LONG -> value.putLong(0x0123456789ABCDEFL);
                case FLOAT -> value.putFloat(1.5f);
                case DOUBLE -> value.putDouble(1.5);
            }
            byte[] expected = new byte[BYTE_SIZE];
            for (int i = 0; i < expected.length; i++) {
                expected[i] = value.array()[i % width];
            }
            return expected;
        }
    }
}
