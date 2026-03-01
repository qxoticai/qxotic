package com.qxotic.jota.tensor;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

/**
 * A lazy computation that represents a constant scalar value broadcast to a shape.
 *
 * <p>This allows scalar tensors to be created without device allocation. The actual buffer is only
 * allocated when {@link #execute()} is called (i.e., when the tensor is materialized).
 *
 * <p>The constant value can be extracted without allocation, allowing the kernel compiler to emit
 * it as a literal in generated code.
 */
record ConstantComputation(long rawBits, DataType dataType, Shape shape, Device device)
        implements LazyComputation {

    public static ConstantComputation of(
            Number value, DataType dataType, Shape shape, Device device) {
        return new ConstantComputation(toBits(value, dataType), dataType, shape, device);
    }

    public Number value() {
        return fromBits(rawBits, dataType);
    }

    @Override
    public List<Tensor> inputs() {
        return List.of();
    }

    @Override
    public Map<String, Object> attributes() {
        return Map.of("value", value(), "dataType", dataType, "shape", shape);
    }

    @Override
    public MemoryView<?> execute() {
        Environment environment = Environment.current();
        MemoryDomain<?> memoryDomain = environment.memoryDomainFor(device);
        return allocateAndFill(memoryDomain);
    }

    @SuppressWarnings("unchecked")
    private <B> MemoryView<B> allocateAndFill(MemoryDomain<B> memoryDomain) {
        // Allocate single element
        Memory<B> memory = memoryDomain.memoryAllocator().allocateMemory(dataType, 1);

        // Write the scalar value
        MemoryAccess<B> access = memoryDomain.directAccess();
        if (access != null) {
            writeValue(access, memory, 0, rawBits, dataType);
        } else {
            @SuppressWarnings("unchecked")
            MemoryDomain<MemorySegment> hostDomain = Environment.current().nativeMemoryDomain();
            MemoryAccess<MemorySegment> hostAccess = hostDomain.directAccess();
            if (hostAccess == null) {
                throw new UnsupportedOperationException(
                        "Cannot materialize constant: no direct host MemoryAccess available");
            }
            Memory<MemorySegment> hostMemory =
                    hostDomain.memoryAllocator().allocateMemory(dataType, 1);
            writeValue(hostAccess, hostMemory, 0, rawBits, dataType);
            MemoryView<MemorySegment> hostScalar =
                    MemoryView.of(hostMemory, dataType, Layout.rowMajor(Shape.scalar()));
            MemoryView<B> deviceScalar =
                    MemoryView.of(memory, dataType, Layout.rowMajor(Shape.scalar()));
            MemoryDomain.copy(hostDomain, hostScalar, memoryDomain, deviceScalar);
        }

        // Create view with all-zero strides (broadcast)
        Layout layout = Layout.of(shape, Stride.zeros(shape));
        return MemoryView.of(memory, dataType, layout);
    }

    private static long toBits(Number value, DataType dataType) {
        if (dataType == DataType.BOOL) {
            return value.intValue() != 0 ? 1L : 0L;
        }
        if (dataType == DataType.I8) {
            return value.byteValue() & 0xFFL;
        }
        if (dataType == DataType.I16) {
            return value.shortValue() & 0xFFFFL;
        }
        if (dataType == DataType.I32) {
            return value.intValue() & 0xFFFFFFFFL;
        }
        if (dataType == DataType.I64) {
            return value.longValue();
        }
        if (dataType == DataType.FP16) {
            return Float.floatToFloat16(value.floatValue()) & 0xFFFFL;
        }
        if (dataType == DataType.BF16) {
            int floatBits = Float.floatToRawIntBits(value.floatValue());
            return (floatBits >>> 16) & 0xFFFFL;
        }
        if (dataType == DataType.FP32) {
            return Float.floatToRawIntBits(value.floatValue()) & 0xFFFFFFFFL;
        }
        if (dataType == DataType.FP64) {
            return Double.doubleToRawLongBits(value.doubleValue());
        }
        throw new IllegalArgumentException("Unsupported data type for constant: " + dataType);
    }

    private static Number fromBits(long bits, DataType dataType) {
        if (dataType == DataType.BOOL) {
            return bits != 0 ? 1 : 0;
        }
        if (dataType == DataType.I8) {
            return (byte) bits;
        }
        if (dataType == DataType.I16) {
            return (short) bits;
        }
        if (dataType == DataType.I32) {
            return (int) bits;
        }
        if (dataType == DataType.I64) {
            return bits;
        }
        if (dataType == DataType.FP16) {
            return Float.float16ToFloat((short) bits);
        }
        if (dataType == DataType.BF16) {
            int floatBits = ((int) bits) << 16;
            return Float.intBitsToFloat(floatBits);
        }
        if (dataType == DataType.FP32) {
            return Float.intBitsToFloat((int) bits);
        }
        if (dataType == DataType.FP64) {
            return Double.longBitsToDouble(bits);
        }
        throw new IllegalArgumentException("Unsupported data type for constant: " + dataType);
    }

    private static <B> void writeValue(
            MemoryAccess<B> access,
            Memory<B> memory,
            long offset,
            long rawBits,
            DataType dataType) {
        if (dataType == DataType.BOOL || dataType == DataType.I8) {
            access.writeByte(memory, offset, (byte) rawBits);
        } else if (dataType == DataType.I16
                || dataType == DataType.FP16
                || dataType == DataType.BF16) {
            access.writeShort(memory, offset, (short) rawBits);
        } else if (dataType == DataType.I32 || dataType == DataType.FP32) {
            access.writeInt(memory, offset, (int) rawBits);
        } else if (dataType == DataType.I64 || dataType == DataType.FP64) {
            access.writeLong(memory, offset, rawBits);
        } else {
            throw new IllegalArgumentException("Unsupported data type for constant: " + dataType);
        }
    }
}
