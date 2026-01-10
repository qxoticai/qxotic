package com.qxotic.jota.memory;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Stride;
import com.qxotic.jota.memory.impl.MemoryAllocatorFactory;
import com.qxotic.jota.memory.impl.ContextFactory;
import com.qxotic.jota.memory.MemoryView;

import java.util.function.Supplier;
import java.util.stream.Stream;

public abstract class AbstractMemoryTest {

    public static Stream<Context<?>> contextProvider() {
        Stream<Supplier<Context<?>>> lazy = Stream.of(
                () -> ContextFactory.ofFloats(),
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(false)),
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(true)),
                () -> ContextFactory.ofMemorySegment()
        );
        // Lazy context creation.
        return lazy.map(Supplier::get);
    }

    public static <B> float readFloat(MemoryAccess<B> memoryAccess, MemoryView<B> view, long... indices) {
        return memoryAccess.readFloat(view.memory(), calculateByteOffset(view, indices));
    }

    public static <B> void writeFloat(MemoryAccess<B> memoryAccess, MemoryView<B> view, float floatValue, long... indices) {
        memoryAccess.writeFloat(view.memory(), calculateByteOffset(view, indices), floatValue);
    }

    public static <B> String toString(MemoryView<B> memoryView) {
        return toString(null, memoryView);
    }

    public static <B> String toString(MemoryAccess<B> memoryAccess, MemoryView<B> memoryView) {
        StringBuilder sb = new StringBuilder();
        // Basic metadata
        sb.append("MemoryView{")
                .append("layout=").append(memoryView.layout())
                .append(", dataType=").append(memoryView.dataType())
                .append(", memory=").append(memoryView.memory())
                .append(", offset=").append(memoryView.byteOffset())
                .append("}");

        // For small tensors, include values
        if (memoryAccess != null && memoryView.shape().size() <= 48) {
            sb.append(", values=");
            appendValues(sb, memoryView, memoryAccess);
        } else {
            sb.append(", size=").append(memoryView.shape().size());
        }

        sb.append("]");
        return sb.toString();
    }

    private static <B> void appendValues(StringBuilder sb, MemoryView<B> memoryView, MemoryAccess<B> memoryAccess) {
        if (memoryView.shape().isScalar()) {
            // Scalar case
            sb.append(readElement(new long[0], memoryView, memoryAccess));
            return;
        }

        sb.append("[");
        long[] indices = new long[memoryView.shape().rank()];
        appendValuesRecursive(sb, indices, 0, memoryView, memoryAccess);
        sb.append("]");
    }

    private static String formatFloatCompact(float value) {
        // Handle special cases
        if (Float.isNaN(value)) return "NaN";
        if (value == Float.POSITIVE_INFINITY) return "∞";
        if (value == Float.NEGATIVE_INFINITY) return "-∞";

        // Check if it's an integer value
        if (value == (int) value) {
            return String.format("%d.", (int) value);
        }

        // Default case - format with minimal decimal places
        String s = String.format("%.6f", value).replace(",", ".");
        s = s.replaceAll("0+$", ""); // Remove trailing zeros
        s = s.replaceAll("\\.$", "."); // Keep decimal point if ends with .
        return s;
    }

    private static <B> void appendValuesRecursive(StringBuilder sb, long[] indices, int dim, MemoryView<B> memoryView, MemoryAccess<B> memoryAccess) {
        if (dim == memoryView.shape().rank() - 1) {
            // Last dimension - print values
            sb.append("[");
            for (int i = 0; i < memoryView.shape().flatAt(dim); i++) {
                indices[dim] = i;
                sb.append(readElement(indices, memoryView, memoryAccess));
                if (i < memoryView.shape().flatAt(dim) - 1) {
                    sb.append(", ");
                }
            }
            sb.append("]");
        } else {
            // Higher dimensions
            sb.append("[");
            for (int i = 0; i < memoryView.shape().flatAt(dim); i++) {
                indices[dim] = i;
                appendValuesRecursive(sb, indices, dim + 1, memoryView, memoryAccess);
                if (i < memoryView.shape().flatAt(dim) - 1) {
                    sb.append(", ");
                }
            }
            sb.append("]");
        }
    }

    private static <B> String readElement(long[] indices, MemoryView<B> memoryView, MemoryAccess<B> memoryAccess) {

        long offset = memoryView.byteOffset();
        for (int i = 0; i < indices.length; i++) {
            offset += indices[i] * memoryView.byteStride().flatAt(i);
        }

        DataType dataType = memoryView.dataType();
        if (dataType == DataType.I8) return Byte.toString(memoryAccess.readByte(memoryView.memory(), offset));
        if (dataType == DataType.I16) return Short.toString(memoryAccess.readShort(memoryView.memory(), offset));
        if (dataType == DataType.I32) return Integer.toString(memoryAccess.readInt(memoryView.memory(), offset));
        if (dataType == DataType.I64) return Long.toString(memoryAccess.readLong(memoryView.memory(), offset));
        if (dataType == DataType.F32) return formatFloatCompact(memoryAccess.readFloat(memoryView.memory(), offset));
        if (dataType == DataType.F64) return String.format("%f", memoryAccess.readDouble(memoryView.memory(), offset));
        if (dataType == DataType.Q8_0) {
            short scale = memoryAccess.readShort(memoryView.memory(), offset);
            byte q = memoryAccess.readByte(memoryView.memory(), offset + 2); // First quantized value
            return String.format("%.2f", (q * scale) / 127.0f);
        }
        return "?";
    }

    public static long calculateByteOffset(MemoryView<?> view, long... indices) {
        if (indices.length != view.shape().rank()) {
            throw new IllegalArgumentException("Index count must match tensor rank");
        }

        long offset = view.byteOffset();
        Stride strides = view.byteStride();

        for (int i = 0; i < indices.length; i++) {
            offset += indices[i] * strides.flatAt(i);
        }

        return offset;
    }
}
