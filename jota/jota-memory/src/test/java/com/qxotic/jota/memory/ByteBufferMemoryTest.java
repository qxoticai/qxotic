package com.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * The contract of {@link Memories#of(ByteBuffer)}: whole buffer, the buffer's byte order, shared.
 */
class ByteBufferMemoryTest {

    private static ByteBuffer buffer(boolean direct, int capacity) {
        return direct ? ByteBuffer.allocateDirect(capacity) : ByteBuffer.allocate(capacity);
    }

    private static MemoryDomain<ByteBuffer> domain(boolean direct) {
        return MemoryDomains.ofByteBuffer(MemoryAllocators.newByteBuffer(direct));
    }

    // ---- whole buffer, cursor ignored -------------------------------------------------------

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void wrapsTheWholeCapacityAndIgnoresTheCursor(boolean direct) {
        ByteBuffer bb = buffer(direct, 16);
        bb.position(8).limit(12).mark();
        Memory<ByteBuffer> memory = Memories.of(bb);

        assertEquals(16, memory.byteSize());
        assertEquals(16, memory.base().limit()); // jota's view has no cursor of its own
        assertEquals(bb.order(), memory.base().order());
        if (!direct) assertSame(bb.array(), memory.base().array());
        MemoryAccess<ByteBuffer> access = domain(direct).directAccess();
        access.writeByte(memory, 0, (byte) 1); // below the position
        access.writeByte(memory, 15, (byte) 2); // beyond the limit
        assertEquals(1, bb.get(0));
        assertEquals(2, bb.duplicate().clear().get(15)); // bb.get(15) itself honours bb's limit
        // the cursor was never moved
        assertEquals(8, bb.position());
        assertEquals(12, bb.limit());
        bb.reset(); // the mark survives too
        assertEquals(8, bb.position());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void aSliceIsTheCallersWindow(boolean direct) {
        ByteBuffer bb = buffer(direct, 16);
        bb.position(8).limit(12);
        Memory<ByteBuffer> window = Memories.of(bb.slice());

        assertEquals(4, window.byteSize());
        domain(direct).directAccess().writeByte(window, 0, (byte) 7);
        assertEquals(7, bb.get(8)); // shared storage, offset by the old position
        assertThrows(
                IndexOutOfBoundsException.class,
                () -> domain(direct).directAccess().writeByte(window, 4, (byte) 1));
    }

    // ---- byte order follows the buffer ------------------------------------------------------

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void typedAccessFollowsTheBuffersByteOrder(boolean direct) {
        MemoryAccess<ByteBuffer> access = domain(direct).directAccess();
        for (ByteOrder order : new ByteOrder[] {ByteOrder.BIG_ENDIAN, ByteOrder.LITTLE_ENDIAN}) {
            ByteBuffer bb = buffer(direct, 16).order(order);
            Memory<ByteBuffer> memory = Memories.of(bb);
            access.writeInt(memory, 0, 0x01020304);
            access.writeShort(memory, 4, (short) 0x0A0B);
            access.writeLong(memory, 8, 0x1112131415161718L);
            // what the buffer itself sees, in its own order
            assertEquals(0x01020304, bb.getInt(0), order.toString());
            assertEquals((short) 0x0A0B, bb.getShort(4), order.toString());
            assertEquals(0x1112131415161718L, bb.getLong(8), order.toString());
            assertEquals(0x01020304, access.readInt(memory, 0), order.toString());
            assertEquals(0x1112131415161718L, access.readLong(memory, 8), order.toString());
            // the bytes in storage are laid out in that order
            byte first = bb.get(0);
            assertEquals(order == ByteOrder.BIG_ENDIAN ? 0x01 : 0x04, first, order.toString());
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void freshBuffersAreBigEndianByJdkDefault(boolean direct) {
        assertEquals(ByteOrder.BIG_ENDIAN, buffer(direct, 4).order());
        assertEquals(
                ByteOrder.BIG_ENDIAN,
                buffer(direct, 4).order(ByteOrder.nativeOrder()).slice().order());
        // jota's own allocator hands out native-order buffers
        Memory<ByteBuffer> allocated = MemoryAllocators.newByteBuffer(direct).allocateMemory(4);
        assertEquals(ByteOrder.nativeOrder(), allocated.base().order());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void bulkCopiesMoveBytesAsTheyAre(boolean direct) {
        MemoryDomain<ByteBuffer> domain = domain(direct);
        ByteBuffer big = buffer(direct, 4).order(ByteOrder.BIG_ENDIAN);
        ByteBuffer nat = buffer(direct, 4).order(ByteOrder.nativeOrder());
        Memory<ByteBuffer> bigMemory = Memories.of(big);
        Memory<ByteBuffer> natMemory = Memories.of(nat);
        domain.directAccess().writeInt(bigMemory, 0, 0x01020304);
        domain.directAccess().writeInt(natMemory, 0, 0x01020304);

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment fromBig = arena.allocate(4);
            MemorySegment fromNat = arena.allocate(4);
            domain.memoryOperations().copyToNative(bigMemory, 0, Memories.of(fromBig), 0, 4);
            domain.memoryOperations().copyToNative(natMemory, 0, Memories.of(fromNat), 0, 4);
            // the native-order buffer's bytes are the value; the big-endian buffer's bytes are
            // its big-endian encoding, seen swapped by a native reader
            assertEquals(0x01020304, fromNat.get(ValueLayout.JAVA_INT, 0));
            int expectedFromBig =
                    ByteOrder.nativeOrder() == ByteOrder.BIG_ENDIAN
                            ? 0x01020304
                            : Integer.reverseBytes(0x01020304);
            assertEquals(expectedFromBig, fromBig.get(ValueLayout.JAVA_INT, 0));
            assertArrayEquals(new byte[] {1, 2, 3, 4}, fromBig.toArray(ValueLayout.JAVA_BYTE));
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void nativeOrderBuffersRoundTripThroughOtherBackends(boolean direct) {
        MemoryDomain<ByteBuffer> domain = domain(direct);
        Memory<ByteBuffer> memory = Memories.of(buffer(direct, 8).order(ByteOrder.nativeOrder()));
        domain.directAccess().writeFloat(memory, 0, 1.5f);
        domain.directAccess().writeFloat(memory, 4, -2.25f);

        MemoryDomain<float[]> floats = MemoryDomains.floats();
        MemoryView<float[]> target = MemoryViews.zeros(floats, DataType.FP32, Shape.of(2));
        MemoryOperations.copy(
                domain.memoryOperations(),
                memory,
                0,
                floats.memoryOperations(),
                target.memory(),
                0,
                8);
        assertArrayEquals(new float[] {1.5f, -2.25f}, target.memory().base());
    }

    // ---- sharing and read-only --------------------------------------------------------------

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void storageIsSharedBothWays(boolean direct) {
        ByteBuffer bb = buffer(direct, 8).order(ByteOrder.nativeOrder());
        Memory<ByteBuffer> memory = Memories.of(bb);
        MemoryAccess<ByteBuffer> access = domain(direct).directAccess();
        bb.putInt(0, 42);
        assertEquals(42, access.readInt(memory, 0));
        access.writeInt(memory, 4, 43);
        assertEquals(43, bb.getInt(4));
        if (!direct) assertSame(bb.array(), memory.base().array());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void readOnlyBuffersYieldReadOnlyMemory(boolean direct) {
        ByteBuffer bb = buffer(direct, 8);
        Memory<ByteBuffer> writable = Memories.of(bb);
        Memory<ByteBuffer> readOnly = Memories.of(bb.asReadOnlyBuffer());
        assertFalse(writable.isReadOnly());
        assertTrue(readOnly.isReadOnly());
        MemoryAccess<ByteBuffer> access = domain(direct).directAccess();
        access.writeByte(writable, 0, (byte) 1);
        assertThrows(
                UnsupportedOperationException.class, () -> access.writeByte(readOnly, 0, (byte) 1));
        assertEquals(1, access.readByte(readOnly, 0)); // reads still see the shared storage
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void viewsOverTheBufferUseAbsoluteOffsets(boolean direct) {
        ByteBuffer bb = buffer(direct, 16).order(ByteOrder.nativeOrder());
        bb.position(4); // irrelevant to the view below
        MemoryView<ByteBuffer> view =
                MemoryViews.rowMajor(Memories.of(bb), DataType.I32, Shape.of(4));
        MemoryDomain<ByteBuffer> domain = domain(direct);
        for (int i = 0; i < 4; i++) domain.directAccess().writeInt(view.memory(), i * 4L, i * 10);
        assertEquals(0, bb.getInt(0));
        assertEquals(30, bb.getInt(12));
        assertThrows(
                IllegalArgumentException.class,
                () -> MemoryViews.rowMajor(Memories.of(bb), DataType.I32, Shape.of(5)));
    }
}
