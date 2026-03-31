package com.qxotic.format.safetensors.impl;

import com.qxotic.format.json.Json;
import com.qxotic.format.safetensors.Safetensors;
import com.qxotic.format.safetensors.TensorEntry;
import java.util.Map;

/** Provides formatting for Safetensors instances with simple parameter-based control. */
class SafetensorsFormatter {

    /**
     * Formats the Safetensors with full control over display and elision.
     *
     * @param safetensors the Safetensors instance to format
     * @param showMetadata whether to display metadata keys and values
     * @param showTensors whether to display tensor information
     * @return formatted string representation
     */
    public static String toString(
            Safetensors safetensors, boolean showMetadata, boolean showTensors) {
        StringBuilder sb = new StringBuilder("Safetensors {\n");
        sb.append("  alignment: 0x")
                .append(Long.toHexString(safetensors.getAlignment()))
                .append('\n');
        sb.append("  tensorDataOffset: 0x")
                .append(Long.toHexString(safetensors.getTensorDataOffset()))
                .append('\n');

        if (showMetadata) {
            formatMetadata(sb, safetensors);
        } else {
            sb.append("  \"__metadata__\" : { ")
                    .append(safetensors.getMetadata().size())
                    .append(" keys }\n");
        }

        if (showTensors) {
            formatTensors(sb, safetensors);
        } else {
            sb.append("  tensors: ").append(safetensors.getTensors().size()).append(" entries\n");
        }

        sb.append('}');
        return sb.toString();
    }

    private static void formatMetadata(StringBuilder sb, Safetensors safetensors) {
        sb.append("  \"").append(ReaderImpl.METADATA_KEY).append("\" : {\n");
        for (Map.Entry<String, String> entry : safetensors.getMetadata().entrySet()) {
            String key = entry.getKey();
            String value = entry.getValue();
            sb.append("    ")
                    .append(Json.stringify(key, false))
                    .append(" : ")
                    .append(Json.stringify(value, false));
            sb.append('\n');
        }

        sb.append("  }\n");
    }

    private static void formatTensors(StringBuilder sb, Safetensors safetensors) {
        var tensors = safetensors.getTensors();
        sb.append("  tensors: {\n");

        for (TensorEntry tensor : tensors) {
            sb.append("    ")
                    .append(tensor.name())
                    .append(": ")
                    .append(tensor.dtype())
                    .append('[')
                    .append(
                            java.util.Arrays.stream(tensor.shape())
                                    .mapToObj(Long::toString)
                                    .collect(java.util.stream.Collectors.joining(", ")))
                    .append("] @ offset=0x")
                    .append(Long.toHexString(tensor.byteOffset()))
                    .append("\n");
        }

        sb.append("  }\n");
    }
}
