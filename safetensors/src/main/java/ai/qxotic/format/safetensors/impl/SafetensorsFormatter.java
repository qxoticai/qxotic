package ai.qxotic.format.safetensors.impl;

import ai.qxotic.format.json.JSON;
import ai.qxotic.format.safetensors.Safetensors;
import ai.qxotic.format.safetensors.TensorEntry;

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
    public static String toString(Safetensors safetensors, boolean showMetadata, boolean showTensors) {
        StringBuilder sb = new StringBuilder("Safetensors {\n");
        sb.append("  alignment: 0x").append(Long.toHexString(safetensors.getAlignment())).append('\n');
        sb.append("  tensorDataOffset: 0x")
                .append(Long.toHexString(safetensors.getTensorDataOffset()))
                .append('\n');

        if (showMetadata) {
            formatMetadata(sb, safetensors);
        } else {
            sb.append("  \"__metadata__\" : { ").append(safetensors.getMetadata().size()).append(" keys }\n");
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
            sb.append("    ").append(JSON.stringify(key)).append(" : ").append(JSON.stringify(value));
            sb.append('\n');
        }

        sb.append("  }\n");
    }

    private static void formatTensors(StringBuilder sb, Safetensors safetensors) {
        var tensors = safetensors.getTensors();
        sb.append("  tensors: {\n");

        for (TensorEntry tensor : tensors) {
            sb.append("    ").append(tensor.name()).append(": ");
            sb.append(tensor.dtype()).append('[');

            // Format shape dimensions
            long[] shape = tensor.shape();
            for (int i = 0; i < shape.length; i++) {
                if (i > 0) sb.append(", ");
                sb.append(shape[i]);
            }
            sb.append("]");
            sb.append(" @ offset=0x").append(Long.toHexString(tensor.byteOffset()));
            sb.append("\n");
        }

        sb.append("  }\n");
    }
}
