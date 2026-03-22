package com.qxotic.jota.testutil;

import com.qxotic.jota.BFloat16;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.tensor.Tensor;
import java.util.LinkedHashSet;

public final class TensorTestReads {

    private TensorTestReads() {}

    public static byte readByte(Tensor tensor, long linearIndex) {
        return ((Number) readValue(tensor, linearIndex, DataType.I8)).byteValue();
    }

    public static float readFloat(Tensor tensor, long linearIndex) {
        return ((Number) readValue(tensor, linearIndex, DataType.FP32)).floatValue();
    }

    public static long readLong(Tensor tensor, long linearIndex) {
        return ((Number) readValue(tensor, linearIndex, DataType.I64)).longValue();
    }

    public static Object readValue(Tensor tensor, long linearIndex, DataType dataType) {
        ReadContext rc = readContext(tensor.materialize());
        MemoryView<?> view = rc.view();
        long offset = Indexing.linearToOffset(view, linearIndex);
        MemoryAccess<Object> access = rc.access();
        Memory<Object> memory = memory(view);
        if (dataType == DataType.BOOL || dataType == DataType.I8) {
            return access.readByte(memory, offset);
        }
        if (dataType == DataType.I16) {
            return access.readShort(memory, offset);
        }
        if (dataType == DataType.I32) {
            return access.readInt(memory, offset);
        }
        if (dataType == DataType.I64) {
            return access.readLong(memory, offset);
        }
        if (dataType == DataType.FP16) {
            return Float.float16ToFloat(access.readShort(memory, offset));
        }
        if (dataType == DataType.BF16) {
            return BFloat16.toFloat(access.readShort(memory, offset));
        }
        if (dataType == DataType.FP32) {
            return access.readFloat(memory, offset);
        }
        if (dataType == DataType.FP64) {
            return access.readDouble(memory, offset);
        }
        throw new IllegalArgumentException("Unsupported data type: " + dataType);
    }

    public static ReadBuffer readBuffer(Tensor tensor) {
        return readBuffer(tensor.materialize());
    }

    public static ReadBuffer readBuffer(MemoryView<?> view) {
        ReadContext context = readContext(view);
        return new ReadBuffer(context.view(), context.access());
    }

    private static ReadContext readContext(MemoryView<?> view) {
        @SuppressWarnings("unchecked")
        MemoryDomain<Object> srcDomain =
                (MemoryDomain<Object>)
                        Environment.runtimeFor(view.memory().device()).memoryDomain();
        MemoryAccess<Object> srcAccess = srcDomain.directAccess();
        if (srcAccess != null) {
            return new ReadContext(view, srcAccess);
        }

        Environment environment = Environment.current();
        Device readableDevice = resolveReadableDevice(environment);
        if (readableDevice == null) {
            throw new IllegalStateException("No runtime with direct memory access is available");
        }

        @SuppressWarnings("unchecked")
        MemoryView<Object> hostView =
                (MemoryView<Object>) Tensor.of(view).to(readableDevice).materialize();
        @SuppressWarnings("unchecked")
        MemoryDomain<Object> hostDomain =
                (MemoryDomain<Object>) Environment.runtimeFor(readableDevice).memoryDomain();
        MemoryAccess<Object> hostAccess = hostDomain.directAccess();
        if (hostAccess == null) {
            throw new IllegalStateException(
                    "Selected readable runtime has no direct access: " + readableDevice);
        }
        return new ReadContext(hostView, hostAccess);
    }

    private static Device resolveReadableDevice(Environment environment) {
        LinkedHashSet<Device> candidates = new LinkedHashSet<>();
        candidates.add(environment.nativeDevice());
        candidates.add(environment.defaultDevice());
        candidates.add(DeviceType.PANAMA.deviceIndex(0));
        candidates.add(DeviceType.C.deviceIndex(0));
        candidates.add(DeviceType.JAVA.deviceIndex(0));
        candidates.addAll(environment.runtimes().devices());

        for (Device candidate : candidates) {
            if (!environment.runtimes().hasRuntimeFor(candidate)) {
                continue;
            }
            @SuppressWarnings("unchecked")
            MemoryDomain<Object> domain =
                    (MemoryDomain<Object>) Environment.runtimeFor(candidate).memoryDomain();
            if (domain.directAccess() != null) {
                return candidate;
            }
        }
        return null;
    }

    @SuppressWarnings("unchecked")
    private static Memory<Object> memory(MemoryView<?> view) {
        return (Memory<Object>) view.memory();
    }

    public record ReadBuffer(MemoryView<?> view, MemoryAccess<Object> access) {}

    private record ReadContext(MemoryView<?> view, MemoryAccess<Object> access) {}
}
