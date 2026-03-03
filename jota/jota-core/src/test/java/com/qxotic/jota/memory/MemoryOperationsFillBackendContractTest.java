package com.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.Environment;
import com.qxotic.jota.memory.impl.MemoryAccessFactory;
import com.qxotic.jota.memory.impl.MemoryFactory;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class MemoryOperationsFillBackendContractTest {

    private static final MemoryAccess<MemorySegment> HOST_ACCESS =
            MemoryAccessFactory.ofMemorySegment();

    @Test
    <B> void fillByteSupportsFullAndPartialRanges() {
        MemoryDomain<B> domain = currentBackendDomain();
        Memory<B> memory = domain.memoryAllocator().allocateMemory(16);

        domain.memoryOperations().fillByte(memory, 0, 16, (byte) 0x11);
        domain.memoryOperations().fillByte(memory, 4, 6, (byte) 0x22);

        Memory<MemorySegment> host = copyToHost(domain, memory, 16);
        for (int i = 0; i < 16; i++) {
            byte expected = (i >= 4 && i < 10) ? (byte) 0x22 : (byte) 0x11;
            assertEquals(expected, HOST_ACCESS.readByte(host, i));
        }
    }

    @Test
    <B> void fillShortSupportsFullAndPartialRanges() {
        MemoryDomain<B> domain = currentBackendDomain();
        Memory<B> memory = domain.memoryAllocator().allocateMemory(8L * Short.BYTES);

        domain.memoryOperations().fillShort(memory, 0, 8L * Short.BYTES, (short) 0x1111);
        domain.memoryOperations()
                .fillShort(memory, 2L * Short.BYTES, 3L * Short.BYTES, (short) 0x2222);

        Memory<MemorySegment> host = copyToHost(domain, memory, 8L * Short.BYTES);
        for (int i = 0; i < 8; i++) {
            short expected = (i >= 2 && i < 5) ? (short) 0x2222 : (short) 0x1111;
            assertEquals(expected, HOST_ACCESS.readShort(host, (long) i * Short.BYTES));
        }
    }

    @Test
    <B> void fillIntSupportsFullAndPartialRanges() {
        MemoryDomain<B> domain = currentBackendDomain();
        Memory<B> memory = domain.memoryAllocator().allocateMemory(6L * Integer.BYTES);

        domain.memoryOperations().fillInt(memory, 0, 6L * Integer.BYTES, 0x11111111);
        domain.memoryOperations()
                .fillInt(memory, 2L * Integer.BYTES, 2L * Integer.BYTES, 0x22222222);

        Memory<MemorySegment> host = copyToHost(domain, memory, 6L * Integer.BYTES);
        for (int i = 0; i < 6; i++) {
            int expected = (i >= 2 && i < 4) ? 0x22222222 : 0x11111111;
            assertEquals(expected, HOST_ACCESS.readInt(host, (long) i * Integer.BYTES));
        }
    }

    @Test
    <B> void fillLongSupportsFullAndPartialRanges() {
        MemoryDomain<B> domain = currentBackendDomain();
        Memory<B> memory = domain.memoryAllocator().allocateMemory(5L * Long.BYTES);

        domain.memoryOperations().fillLong(memory, 0, 5L * Long.BYTES, 0x1111111111111111L);
        domain.memoryOperations()
                .fillLong(memory, Long.BYTES, 2L * Long.BYTES, 0x2222222222222222L);

        Memory<MemorySegment> host = copyToHost(domain, memory, 5L * Long.BYTES);
        for (int i = 0; i < 5; i++) {
            long expected = (i >= 1 && i < 3) ? 0x2222222222222222L : 0x1111111111111111L;
            assertEquals(expected, HOST_ACCESS.readLong(host, (long) i * Long.BYTES));
        }
    }

    @Test
    <B> void fillFloatAndFillDoubleUseExpectedBitPatterns() {
        MemoryDomain<B> domain = currentBackendDomain();

        Memory<B> floatMem = domain.memoryAllocator().allocateMemory(4L * Float.BYTES);
        domain.memoryOperations().fillFloat(floatMem, 0, 4L * Float.BYTES, -3.5f);
        Memory<MemorySegment> floatHost = copyToHost(domain, floatMem, 4L * Float.BYTES);
        for (int i = 0; i < 4; i++) {
            assertEquals(-3.5f, HOST_ACCESS.readFloat(floatHost, (long) i * Float.BYTES), 0.0f);
        }

        Memory<B> doubleMem = domain.memoryAllocator().allocateMemory(3L * Double.BYTES);
        domain.memoryOperations().fillDouble(doubleMem, 0, 3L * Double.BYTES, 0.125d);
        Memory<MemorySegment> doubleHost = copyToHost(domain, doubleMem, 3L * Double.BYTES);
        for (int i = 0; i < 3; i++) {
            assertEquals(0.125d, HOST_ACCESS.readDouble(doubleHost, (long) i * Double.BYTES), 0.0d);
        }
    }

    @Test
    <B> void fillThrowsForInvalidRanges() {
        MemoryDomain<B> domain = currentBackendDomain();
        Memory<B> memory = domain.memoryAllocator().allocateMemory(16);

        assertThrows(
                IndexOutOfBoundsException.class,
                () -> domain.memoryOperations().fillByte(memory, -1, 1, (byte) 1));
        assertThrows(
                IndexOutOfBoundsException.class,
                () -> domain.memoryOperations().fillByte(memory, 0, -1, (byte) 1));
        assertThrows(
                IndexOutOfBoundsException.class,
                () -> domain.memoryOperations().fillByte(memory, 10, 7, (byte) 1));
    }

    @Test
    <B> void fillThrowsForNonAlignedElementSizes() {
        MemoryDomain<B> domain = currentBackendDomain();
        Memory<B> memory = domain.memoryAllocator().allocateMemory(32);

        assertThrows(
                IllegalArgumentException.class,
                () -> domain.memoryOperations().fillShort(memory, 0, 3, (short) 7));
        assertThrows(
                IllegalArgumentException.class,
                () -> domain.memoryOperations().fillInt(memory, 0, 6, 7));
        assertThrows(
                IllegalArgumentException.class,
                () -> domain.memoryOperations().fillLong(memory, 0, 10, 7L));
        assertThrows(
                IllegalArgumentException.class,
                () -> domain.memoryOperations().fillFloat(memory, 0, 6, 1.0f));
        assertThrows(
                IllegalArgumentException.class,
                () -> domain.memoryOperations().fillDouble(memory, 0, 10, 1.0));
    }

    @Test
    <B> void zeroSizedFillIsNoOp() {
        MemoryDomain<B> domain = currentBackendDomain();
        Memory<B> memory = domain.memoryAllocator().allocateMemory(8);

        domain.memoryOperations().fillByte(memory, 0, 8, (byte) 0x11);
        domain.memoryOperations().fillInt(memory, 1, 0, 0x7F7F7F7F);

        Memory<MemorySegment> host = copyToHost(domain, memory, 8);
        for (int i = 0; i < 8; i++) {
            assertEquals((byte) 0x11, HOST_ACCESS.readByte(host, i));
        }
    }

    @SuppressWarnings("unchecked")
    private static <B> MemoryDomain<B> currentBackendDomain() {
        Environment env = Environment.current();
        return (MemoryDomain<B>) env.memoryDomainFor(env.defaultDevice());
    }

    private static <B> Memory<MemorySegment> copyToHost(
            MemoryDomain<B> domain, Memory<B> src, long byteSize) {
        Memory<MemorySegment> host =
                MemoryFactory.ofMemorySegment(MemorySegment.ofArray(new byte[(int) byteSize]));
        domain.memoryOperations().copyToNative(src, 0, host, 0, byteSize);
        return host;
    }
}
