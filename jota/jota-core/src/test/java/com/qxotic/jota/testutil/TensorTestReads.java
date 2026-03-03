package com.qxotic.jota.testutil;

import com.qxotic.jota.BFloat16;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.tensor.Tensor;

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
        ReadContext rc = readContext(tensor);
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

    private static ReadContext readContext(Tensor tensor) {
        MemoryView<?> view = tensor.materialize();
        @SuppressWarnings("unchecked")
        MemoryDomain<Object> srcDomain =
                (MemoryDomain<Object>)
                        Environment.current().runtimeFor(view.memory().device()).memoryDomain();
        MemoryAccess<Object> srcAccess = srcDomain.directAccess();
        if (srcAccess != null) {
            return new ReadContext(view, srcAccess);
        }

        @SuppressWarnings("unchecked")
        MemoryView<Object> hostView =
                (MemoryView<Object>)
                        Tensor.of(view)
                                .to(Environment.current().nativeRuntime().device())
                                .materialize();
        @SuppressWarnings("unchecked")
        MemoryDomain<Object> hostDomain =
                (MemoryDomain<Object>) (MemoryDomain<?>) Environment.current().nativeMemoryDomain();
        MemoryAccess<Object> hostAccess = hostDomain.directAccess();
        if (hostAccess == null) {
            throw new IllegalStateException("Native memory domain has no direct access");
        }
        return new ReadContext(hostView, hostAccess);
    }

    @SuppressWarnings("unchecked")
    private static Memory<Object> memory(MemoryView<?> view) {
        return (Memory<Object>) view.memory();
    }

    private record ReadContext(MemoryView<?> view, MemoryAccess<Object> access) {}
}
