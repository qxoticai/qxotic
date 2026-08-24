package com.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;
import com.qxotic.jota.memory.impl.DomainFactory;
import com.qxotic.jota.memory.impl.MemoryFactory;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

class StridedCopyTest {

    private static final MemoryDomain<byte[]> OPAQUE = new OpaqueByteDomain();

    @Test
    void stagesAStridedSourceForAnOpaqueDomain() {
        MemoryView<byte[]> src = view(new byte[] {9, 1, 2, 9, 3, 4, 9}, 1, Stride.flat(3, 1));
        byte[] result = filled(8, (byte) 9);
        MemoryView<byte[]> dst =
                MemoryView.of(
                        MemoryFactory.ofBytes(result),
                        2,
                        DataType.I8,
                        Layout.rowMajor(Shape.flat(2, 2)));

        OPAQUE.copy(src, dst);

        assertArrayEquals(new byte[] {9, 9, 1, 2, 3, 4, 9, 9}, result);
    }

    @Test
    void stagesAStridedDestinationWithoutOverwritingItsGaps() {
        MemoryView<byte[]> src =
                MemoryView.of(
                        MemoryFactory.ofBytes(new byte[] {9, 1, 2, 3, 4, 9}),
                        1,
                        DataType.I8,
                        Layout.rowMajor(Shape.flat(2, 2)));
        byte[] result = filled(8, (byte) 9);
        MemoryView<byte[]> dst = view(result, 1, Stride.flat(3, 1));

        OPAQUE.copy(src, dst);

        assertArrayEquals(new byte[] {9, 1, 2, 9, 3, 4, 9, 9}, result);
    }

    @Test
    void rejectsIncompatibleViews() {
        MemoryView<byte[]> bytes =
                MemoryView.rowMajor(MemoryFactory.ofBytes(new byte[4]), DataType.I8, Shape.flat(4));
        MemoryView<byte[]> bools =
                MemoryView.rowMajor(
                        MemoryFactory.ofBytes(new byte[4]), DataType.BOOL, Shape.flat(4));
        MemoryView<byte[]> shorter =
                MemoryView.rowMajor(MemoryFactory.ofBytes(new byte[3]), DataType.I8, Shape.flat(3));

        assertThrows(IllegalArgumentException.class, () -> OPAQUE.copy(bytes, bools));
        assertThrows(IllegalArgumentException.class, () -> OPAQUE.copy(bytes, shorter));
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        OPAQUE.copy(
                                bytes,
                                MemoryView.rowMajor(
                                        otherDeviceMemory(4), DataType.I8, Shape.flat(4))));
    }

    @Test
    void emptyCopyIsANoOp() {
        MemoryView<byte[]> empty =
                MemoryView.rowMajor(MemoryFactory.ofBytes(new byte[0]), DataType.I8, Shape.flat(0));
        assertDoesNotThrow(() -> OPAQUE.copy(empty, empty));
    }

    private static MemoryView<byte[]> view(byte[] bytes, long byteOffset, Stride stride) {
        return MemoryView.of(
                MemoryFactory.ofBytes(bytes),
                byteOffset,
                DataType.I8,
                Layout.of(Shape.flat(2, 2), stride));
    }

    private static byte[] filled(int size, byte value) {
        byte[] bytes = new byte[size];
        Arrays.fill(bytes, value);
        return bytes;
    }

    private static Memory<byte[]> otherDeviceMemory(int size) {
        return new Memory<>() {
            private final byte[] bytes = new byte[size];

            @Override
            public long byteSize() {
                return bytes.length;
            }

            @Override
            public boolean isReadOnly() {
                return false;
            }

            @Override
            public Device device() {
                return DeviceType.PANAMA.deviceIndex(0);
            }

            @Override
            public byte[] base() {
                return bytes;
            }

            @Override
            public long memoryGranularity() {
                return 1;
            }
        };
    }

    private static final class OpaqueByteDomain implements MemoryDomain<byte[]> {

        private final MemoryDomain<byte[]> delegate = DomainFactory.ofBytes();

        @Override
        public Device device() {
            return delegate.device();
        }

        @Override
        public MemoryAllocator<byte[]> memoryAllocator() {
            return delegate.memoryAllocator();
        }

        @Override
        public MemoryAccess<byte[]> directAccess() {
            return null;
        }

        @Override
        public MemoryOperations<byte[]> memoryOperations() {
            return delegate.memoryOperations();
        }

        @Override
        public void close() {}
    }
}
