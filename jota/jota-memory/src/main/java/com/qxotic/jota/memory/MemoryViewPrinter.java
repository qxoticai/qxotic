package com.qxotic.jota.memory;

import com.qxotic.jota.BFloat16;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Indexing;

public final class MemoryViewPrinter {

    private MemoryViewPrinter() {}

    public static <B> String toString(MemoryView<B> view) {
        return toString(view, null, ViewPrintOptions.metaOnly());
    }

    public static <B> String toString(MemoryView<B> view, MemoryAccess<B> memoryAccess) {
        return toString(view, memoryAccess, ViewPrintOptions.compact());
    }

    public static <B> String toString(
            MemoryView<B> view, MemoryAccess<B> memoryAccess, ViewPrintOptions options) {
        StringBuilder sb = new StringBuilder();

        if (options.includeMetadata()) {
            sb.append("MemoryView{")
                    .append("layout=")
                    .append(view.layout())
                    .append(", dataType=")
                    .append(view.dataType())
                    .append(", memory=")
                    .append(view.memory());
            if (view.byteOffset() != 0) {
                sb.append(", offset=0x").append(Long.toHexString(view.byteOffset()));
            }
            sb.append("}");
        }

        if (options.includeValues()) {
            if (sb.length() > 0) {
                sb.append(System.lineSeparator());
            }
            if (memoryAccess == null) {
                sb.append("<unavailable>");
            } else {
                appendPrettyValues(sb, view, memoryAccess, options);
            }
        }

        return sb.toString();
    }

    private static <B> void appendPrettyValues(
            StringBuilder sb,
            MemoryView<B> memoryView,
            MemoryAccess<B> memoryAccess,
            ViewPrintOptions options) {
        if (memoryView.shape().isScalar()) {
            sb.append("[");
            sb.append(readElement(memoryAccess, memoryView, new long[0], options));
            sb.append("]");
            return;
        }

        long[] indices = new long[memoryView.shape().flatRank()];
        long[] dims = memoryView.shape().toArray();
        boolean elide = memoryView.shape().size() > options.maxElements();
        appendPrettyRecursive(sb, indices, dims, 0, memoryView, memoryAccess, options, elide, 0);
    }

    private static void appendIndent(StringBuilder sb, int spaces) {
        for (int i = 0; i < spaces; i++) {
            sb.append(' ');
        }
    }

    private static <B> void appendPrettyRecursive(
            StringBuilder sb,
            long[] indices,
            long[] dims,
            int dim,
            MemoryView<B> memoryView,
            MemoryAccess<B> memoryAccess,
            ViewPrintOptions options,
            boolean elide,
            int indent) {
        long dimSize = dims[dim];
        int edgeItems = options.edgeItems();
        boolean useElide = elide && edgeItems > 0 && dimSize > 2L * edgeItems;

        sb.append("[");
        if (dimSize == 0) {
            sb.append("]");
            return;
        }

        if (dim == dims.length - 1) {
            appendPrettyLineValues(
                    sb, indices, dims, dim, memoryView, memoryAccess, options, useElide, edgeItems);
            sb.append("]");
            return;
        }

        int nextIndent = indent + 2;
        boolean first = true;
        boolean addBlockSpacing = dims.length - dim >= 3;
        for (long i = 0; i < dimSize; i++) {
            if (useElide && i == edgeItems) {
                if (!first) {
                    sb.append(",");
                }
                sb.append(System.lineSeparator());
                if (addBlockSpacing && !first) {
                    sb.append(System.lineSeparator());
                }
                appendIndent(sb, nextIndent);
                sb.append("...");
                first = false;
                i = dimSize - edgeItems - 1;
                continue;
            }

            if (!first) {
                sb.append(",");
            }
            sb.append(System.lineSeparator());
            appendIndent(sb, nextIndent);
            indices[dim] = i;
            appendPrettyRecursive(
                    sb,
                    indices,
                    dims,
                    dim + 1,
                    memoryView,
                    memoryAccess,
                    options,
                    elide,
                    nextIndent);
            first = false;
        }
        sb.append(System.lineSeparator());
        appendIndent(sb, indent);
        sb.append("]");
    }

    private static <B> void appendPrettyLineValues(
            StringBuilder sb,
            long[] indices,
            long[] dims,
            int dim,
            MemoryView<B> memoryView,
            MemoryAccess<B> memoryAccess,
            ViewPrintOptions options,
            boolean useElide,
            int edgeItems) {
        long dimSize = dims[dim];
        long tailStart = Math.max(edgeItems, dimSize - edgeItems);
        int visibleCount = useElide ? edgeItems * 2 + 1 : (int) dimSize;
        String[] values = new String[visibleCount];
        int count = 0;
        int width = 0;

        if (!useElide) {
            for (long i = 0; i < dimSize; i++) {
                indices[dim] = i;
                String value = readElement(memoryAccess, memoryView, indices, options);
                values[count++] = value;
                width = Math.max(width, value.length());
            }
        } else {
            for (int i = 0; i < edgeItems; i++) {
                indices[dim] = i;
                String value = readElement(memoryAccess, memoryView, indices, options);
                values[count++] = value;
                width = Math.max(width, value.length());
            }
            values[count++] = "...";
            width = Math.max(width, 3);
            for (long i = tailStart; i < dimSize; i++) {
                indices[dim] = i;
                String value = readElement(memoryAccess, memoryView, indices, options);
                values[count++] = value;
                width = Math.max(width, value.length());
            }
        }

        for (int i = 0; i < count; i++) {
            String value = values[i];
            int padding = width - value.length();
            if (padding > 0) {
                sb.append(" ".repeat(padding));
            }
            sb.append(value);
            if (i < count - 1) {
                sb.append(", ");
            }
        }
    }

    private static <B> String readElement(
            MemoryAccess<B> memoryAccess,
            MemoryView<B> memoryView,
            long[] coords,
            ViewPrintOptions options) {
        long offset = Indexing.coordToOffset(memoryView, coords);
        DataType dataType = memoryView.dataType();
        Object value;
        if (dataType == DataType.BOOL) {
            value = memoryAccess.readByte(memoryView.memory(), offset) != 0;
        } else if (dataType == DataType.I8) {
            value = memoryAccess.readByte(memoryView.memory(), offset);
        } else if (dataType == DataType.I16) {
            value = memoryAccess.readShort(memoryView.memory(), offset);
        } else if (dataType == DataType.I32) {
            value = memoryAccess.readInt(memoryView.memory(), offset);
        } else if (dataType == DataType.I64) {
            value = memoryAccess.readLong(memoryView.memory(), offset);
        } else if (dataType == DataType.BF16) {
            value = BFloat16.toFloat(memoryAccess.readShort(memoryView.memory(), offset));
        } else if (dataType == DataType.FP16) {
            value = Float.float16ToFloat(memoryAccess.readShort(memoryView.memory(), offset));
        } else if (dataType == DataType.FP32) {
            value = memoryAccess.readFloat(memoryView.memory(), offset);
        } else if (dataType == DataType.FP64) {
            value = memoryAccess.readDouble(memoryView.memory(), offset);
        } else {
            value = "?";
        }
        return options.formatter().format(dataType, value);
    }
}
