package com.qxotic.jota.runtime;

import com.qxotic.jota.BFloat16;
import com.qxotic.jota.DataType;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.tensor.Tensor;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class KernelArgs {

    public enum Kind {
        BUFFER,
        SCALAR,
        METADATA
    }

    public record Entry(Kind kind, Object value, DataType dataType) {}

    private final List<Entry> entries = new ArrayList<>();

    public KernelArgs addBuffer(MemoryView<?> view) {
        Objects.requireNonNull(view, "view");
        entries.add(new Entry(Kind.BUFFER, view, view.dataType()));
        return this;
    }

    public KernelArgs addScalar(Number value, DataType dataType) {
        Objects.requireNonNull(value, "value");
        Objects.requireNonNull(dataType, "dataType");
        long rawBits = toBits(value, dataType);
        entries.add(new Entry(Kind.SCALAR, rawBits, dataType));
        return this;
    }

    public KernelArgs addScalarBits(long rawBits, DataType dataType) {
        Objects.requireNonNull(dataType, "dataType");
        entries.add(new Entry(Kind.SCALAR, rawBits, dataType));
        return this;
    }

    public KernelArgs addMetadata(Object metadata) {
        Objects.requireNonNull(metadata, "metadata");
        entries.add(new Entry(Kind.METADATA, metadata, null));
        return this;
    }

    public List<Entry> entries() {
        return List.copyOf(entries);
    }

    public Entry entry(int index) {
        return entries.get(index);
    }

    public int size() {
        return entries.size();
    }

    public MemoryView<?> getBuffer(int index) {
        Entry entry = entry(index);
        if (entry.kind() != Kind.BUFFER) {
            throw new IllegalArgumentException("Entry " + index + " is not a buffer");
        }
        return (MemoryView<?>) entry.value();
    }

    public long getScalarBits(int index) {
        Entry entry = entry(index);
        if (entry.kind() != Kind.SCALAR) {
            throw new IllegalArgumentException("Entry " + index + " is not a scalar");
        }
        return (long) entry.value();
    }

    public DataType getScalarType(int index) {
        Entry entry = entry(index);
        if (entry.kind() != Kind.SCALAR) {
            throw new IllegalArgumentException("Entry " + index + " is not a scalar");
        }
        return entry.dataType();
    }

    public boolean getBoolean(int index) {
        return getScalarBits(index) != 0;
    }

    public byte getByte(int index) {
        return (byte) getScalarBits(index);
    }

    public short getShort(int index) {
        return (short) getScalarBits(index);
    }

    public int getInt(int index) {
        return (int) getScalarBits(index);
    }

    public long getLong(int index) {
        return getScalarBits(index);
    }

    public float getFloat(int index) {
        Entry entry = entry(index);
        if (entry.kind() != Kind.SCALAR) {
            throw new IllegalArgumentException("Entry " + index + " is not a scalar");
        }
        long bits = (long) entry.value();
        DataType type = entry.dataType();
        if (type == DataType.BOOL) {
            return bits != 0 ? 1.0f : 0.0f;
        }
        if (type == DataType.I8) {
            return (byte) bits;
        }
        if (type == DataType.I16) {
            return (short) bits;
        }
        if (type == DataType.I32) {
            return (int) bits;
        }
        if (type == DataType.I64) {
            return bits;
        }
        if (type == DataType.FP16) {
            return Float.float16ToFloat((short) bits);
        }
        if (type == DataType.BF16) {
            return BFloat16.toFloat((short) bits);
        }
        if (type == DataType.FP32) {
            return Float.intBitsToFloat((int) bits);
        }
        if (type == DataType.FP64) {
            return (float) Double.longBitsToDouble(bits);
        }
        throw new IllegalArgumentException("Unsupported scalar type: " + type);
    }

    public double getDouble(int index) {
        Entry entry = entry(index);
        if (entry.kind() != Kind.SCALAR) {
            throw new IllegalArgumentException("Entry " + index + " is not a scalar");
        }
        long bits = (long) entry.value();
        DataType type = entry.dataType();
        if (type == DataType.BOOL) {
            return bits != 0 ? 1.0 : 0.0;
        }
        if (type == DataType.I8) {
            return (byte) bits;
        }
        if (type == DataType.I16) {
            return (short) bits;
        }
        if (type == DataType.I32) {
            return (int) bits;
        }
        if (type == DataType.I64) {
            return bits;
        }
        if (type == DataType.FP16) {
            return Float.float16ToFloat((short) bits);
        }
        if (type == DataType.BF16) {
            return BFloat16.toFloat((short) bits);
        }
        if (type == DataType.FP32) {
            return Float.intBitsToFloat((int) bits);
        }
        if (type == DataType.FP64) {
            return Double.longBitsToDouble(bits);
        }
        throw new IllegalArgumentException("Unsupported scalar type: " + type);
    }

    public static KernelArgs fromVarargs(Object... args) {
        KernelArgs ka = new KernelArgs();
        for (int idx = 0; idx < args.length; idx++) {
            Object arg = Objects.requireNonNull(args[idx], "kernel arg at index " + idx);
            switch (arg) {
                case MemoryView<?> v -> ka.addBuffer(v);
                case Tensor t -> ka.addBuffer(materializeForKernel(t));
                case ScalarArg s -> ka.addScalarBits(s.rawBits(), s.dataType());
                case Integer i -> ka.addScalar(i, DataType.I32);
                case Long l -> ka.addScalar(l, DataType.I64);
                case Float f -> ka.addScalar(f, DataType.FP32);
                case Double d -> ka.addScalar(d, DataType.FP64);
                case Boolean b -> ka.addScalarBits(b ? 1L : 0L, DataType.BOOL);
                case Short s -> ka.addScalar(s, DataType.I16);
                case Byte b -> ka.addScalar(b, DataType.I8);
                default ->
                        throw new IllegalArgumentException(
                                "Unknown kernel arg type: "
                                        + arg.getClass().getName()
                                        + " — use MemoryView/Tensor for buffers,"
                                        + " boxed primitives or ScalarArg for scalars");
            }
        }
        return ka;
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static MemoryView<?> materializeForKernel(Tensor tensor) {
        MemoryView<?> view = tensor.materialize();
        if (view.layout().isSuffixContiguous(0)) {
            return view;
        }
        throw new IllegalArgumentException(
                "Kernel Tensor arg must be row-major contiguous; call tensor.contiguous() "
                        + "explicitly or pass MemoryView for custom strides.");
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
        throw new IllegalArgumentException("Unsupported scalar type: " + dataType);
    }
}
