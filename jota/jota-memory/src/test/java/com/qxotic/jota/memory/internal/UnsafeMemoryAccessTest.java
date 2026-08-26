package com.qxotic.jota.memory.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAllocators;
import org.junit.jupiter.api.Test;

class UnsafeMemoryAccessTest {

    @Test
    void accessesAbsoluteAddresses() {
        MemoryAccess<Void> access = UnsafeMemoryAccess.instance();
        try (var memory = MemoryAllocators.newScopedArena().allocateMemory(32)) {
            long address = memory.base().address();

            access.writeByte(null, address, (byte) 0x12);
            access.writeShort(null, address + 2, (short) 0x1234);
            access.writeInt(null, address + 4, 0x12345678);
            access.writeLong(null, address + 8, 0x0123456789ABCDEFL);
            access.writeFloat(null, address + 16, 1.5f);
            access.writeDouble(null, address + 24, 1.5);

            assertEquals((byte) 0x12, access.readByte(null, address));
            assertEquals((short) 0x1234, access.readShort(null, address + 2));
            assertEquals(0x12345678, access.readInt(null, address + 4));
            assertEquals(0x0123456789ABCDEFL, access.readLong(null, address + 8));
            assertEquals(1.5f, access.readFloat(null, address + 16));
            assertEquals(1.5, access.readDouble(null, address + 24));
        }
    }
}
