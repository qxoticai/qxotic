package ai.qxotic.format.gguf.impl;

import ai.qxotic.format.gguf.GGUF;
import ai.qxotic.format.gguf.MetadataValueType;
import ai.qxotic.format.gguf.TensorInfo;
import java.lang.reflect.Array;

/** Provides formatting for GGUF instances with simple parameter-based control. */
class GGUFFormatter {

    /**
     * Formats the GGUF with control over what to display.
     *
     * @param gguf the GGUF instance to format
     * @param showKeys whether to display metadata keys and values
     * @param showTensors whether to display tensor information
     * @return formatted string representation
     */
    public static String toString(GGUF gguf, boolean showKeys, boolean showTensors) {
        return toString(gguf, showKeys, showTensors, 5, 50);
    }

    /**
     * Formats the GGUF with full control over display and elision.
     *
     * @param gguf the GGUF instance to format
     * @param showKeys whether to display metadata keys and values
     * @param showTensors whether to display tensor information
     * @param maxArrayElements maximum number of array elements to show before eliding
     * @param maxStringLength maximum string length before truncation
     * @return formatted string representation
     */
    public static String toString(
            GGUF gguf,
            boolean showKeys,
            boolean showTensors,
            int maxArrayElements,
            int maxStringLength) {
        StringBuilder sb = new StringBuilder("GGUF {\n");
        sb.append("  version: ").append(gguf.getVersion()).append('\n');
        sb.append("  alignment: 0x").append(Long.toHexString(gguf.getAlignment())).append('\n');
        sb.append("  tensorDataOffset: 0x")
                .append(Long.toHexString(gguf.getTensorDataOffset()))
                .append('\n');

        if (showKeys) {
            formatMetadata(sb, gguf, maxArrayElements, maxStringLength);
        } else {
            sb.append("  metadata: ").append(gguf.getMetadataKeys().size()).append(" keys\n");
        }

        if (showTensors) {
            formatTensors(sb, gguf);
        } else {
            sb.append("  tensors: ").append(gguf.getTensors().size()).append(" entries\n");
        }

        sb.append('}');
        return sb.toString();
    }

    private static void formatMetadata(
            StringBuilder sb, GGUF gguf, int maxArrayElements, int maxStringLength) {
        var keys = gguf.getMetadataKeys();
        sb.append("  metadata: {\n");
        for (String key : keys) {
            sb.append("    ").append(key).append(": ");

            MetadataValueType type = gguf.getType(key);
            Object value = gguf.getValue(Object.class, key);

            formatMetadataValue(sb, type, value, gguf, key, maxArrayElements, maxStringLength);
            sb.append('\n');
        }

        sb.append("  }\n");
    }

    private static void formatMetadataValue(
            StringBuilder sb,
            MetadataValueType type,
            Object value,
            GGUF gguf,
            String key,
            int maxArrayElements,
            int maxStringLength) {
        if (type == MetadataValueType.ARRAY) {
            MetadataValueType componentType = gguf.getComponentType(key);
            int arrayLength = Array.getLength(value);
            sb.append(componentType).append('[').append(arrayLength).append("] ");
            formatArrayValue(sb, value, arrayLength, maxArrayElements, maxStringLength);
        } else {
            sb.append(type).append(' ');
            formatScalarValue(sb, value, maxStringLength);
        }
    }

    private static void formatScalarValue(StringBuilder sb, Object value, int maxStringLength) {
        if (value instanceof String) {
            String str = escapeString((String) value);
            if (str.length() > maxStringLength) {
                sb.append('"').append(str, 0, maxStringLength).append("...\"");
            } else {
                sb.append('"').append(str).append('"');
            }
        } else {
            sb.append(value);
        }
    }

    private static String escapeString(String str) {
        return str.replace("\\", "\\\\")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\"", "\\\"");
    }

    private static void formatArrayValue(
            StringBuilder sb, Object array, int length, int maxArrayElements, int maxStringLength) {
        if (length == 0) {
            sb.append("[]");
            return;
        }

        if (length <= maxArrayElements) {
            // Show all elements
            sb.append('[');
            for (int i = 0; i < length; i++) {
                if (i > 0) sb.append(", ");
                Object elem = Array.get(array, i);
                formatScalarValue(sb, elem, maxStringLength);
            }
            sb.append(']');
        } else {
            // Show first few and last few elements
            int half = maxArrayElements / 2;
            sb.append('[');

            // First elements
            for (int i = 0; i < half; i++) {
                if (i > 0) sb.append(", ");
                Object elem = Array.get(array, i);
                formatScalarValue(sb, elem, maxStringLength);
            }

            sb.append(", ..., ");

            // Last elements
            for (int i = length - half; i < length; i++) {
                if (i > length - half) sb.append(", ");
                Object elem = Array.get(array, i);
                formatScalarValue(sb, elem, maxStringLength);
            }

            sb.append(']');
        }
    }

    private static void formatTensors(StringBuilder sb, GGUF gguf) {
        var tensors = gguf.getTensors();
        sb.append("  tensors: [\n");

        for (TensorInfo tensor : tensors) {
            sb.append("    ").append(tensor.name()).append(": ");
            sb.append(tensor.ggmlType()).append('[');

            // Format shape dimensions
            long[] shape = tensor.shape();
            for (int i = 0; i < shape.length; i++) {
                if (i > 0) sb.append(", ");
                sb.append(shape[i]);
            }

            sb.append("]\n");
        }

        sb.append("  ]\n");
    }
}
