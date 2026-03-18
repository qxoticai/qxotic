package com.qxotic.jota.runtime.nativeimpl;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.ScopedMemory;
import com.qxotic.jota.memory.ScopedMemoryAllocator;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Field;
import java.util.concurrent.atomic.AtomicBoolean;
import sun.misc.Unsafe;

class NativeUnsafeAllocator implements ScopedMemoryAllocator<MemorySegment> {

    private static final Unsafe UNSAFE = initializeUnsafe();
    private static final NativeUnsafeAllocator INSTANCE = new NativeUnsafeAllocator();

    public static ScopedMemoryAllocator<MemorySegment> instance() {
        return INSTANCE;
    }

    private NativeUnsafeAllocator() {}

    @Override
    public Device device() {
        return new Device(DeviceType.PANAMA, 0);
    }

    @Override
    public long memoryGranularity() {
        return Byte.BYTES;
    }

    @Override
    public ScopedMemory<MemorySegment> allocateMemory(long byteSize, long byteAlignment) {
        if (!isPowerOf2(byteAlignment)) {
            throw new IllegalArgumentException("invalid byteAlignment, not a power of 2");
        }
        long mallocAddress = UNSAFE.allocateMemory(byteSize + byteAlignment - 1);
        long alignedAddress = mallocAddress;
        if (alignedAddress % byteAlignment != 0) {
            alignedAddress += byteAlignment - (alignedAddress % byteAlignment);
        }
        assert alignedAddress % byteAlignment == 0;
        return new RawScopedMemory(mallocAddress, alignedAddress, byteSize, true);
    }

    private static final class RawScopedMemory implements ScopedMemory<MemorySegment> {
        private final long mallocAddress;
        private final MemorySegment memorySegment;
        private final AtomicBoolean isMemoryFreed; // null if not owned by this instance

        public RawScopedMemory(
                long mallocAddress, long alignedAddress, long byteSize, boolean transferOwnership) {
            this.mallocAddress = mallocAddress;
            this.memorySegment = MemorySegment.ofAddress(alignedAddress).reinterpret(byteSize);
            this.isMemoryFreed = transferOwnership ? new AtomicBoolean(false) : null;
        }

        @Override
        public void close() {
            if (isMemoryFreed != null) {
                if (isMemoryFreed.compareAndSet(false, true)) {
                    UNSAFE.freeMemory(mallocAddress);
                } else {
                    throw new IllegalStateException("Memory already freed " + this);
                }
            }
        }

        @Override
        public long byteSize() {
            return memorySegment.byteSize();
        }

        @Override
        public boolean isReadOnly() {
            return memorySegment.isReadOnly();
        }

        @Override
        public Device device() {
            return new Device(DeviceType.PANAMA, 0);
        }

        @Override
        public MemorySegment base() {
            return memorySegment;
        }

        @Override
        public long memoryGranularity() {
            return Byte.BYTES;
        }

        @Override
        public String toString() {
            long address = memorySegment.address();
            StringBuilder sb = new StringBuilder("RawScopedMemory{address=0x");
            sb.append(Long.toHexString(address));
            if (mallocAddress != address) {
                sb.append(", mallocAddress=0x").append(Long.toHexString(mallocAddress));
            }
            sb.append(", byteSize=").append(byteSize());
            if (isMemoryFreed != null) {
                sb.append(", owned, freed=").append(isMemoryFreed.get());
            }
            if (isReadOnly()) {
                sb.append(", readOnly=true");
            }
            sb.append('}');
            return sb.toString();
        }
    }

    private static boolean isPowerOf2(long n) {
        return n > 0 && ((n & (n - 1)) == 0);
    }

    private static Unsafe initializeUnsafe() {
        try {
            Field theUnsafe = Unsafe.class.getDeclaredField("theUnsafe");
            theUnsafe.setAccessible(true);
            return (Unsafe) theUnsafe.get(null);
        } catch (IllegalAccessException | NoSuchFieldException e) {
            throw new RuntimeException(e);
        }
    }
}
