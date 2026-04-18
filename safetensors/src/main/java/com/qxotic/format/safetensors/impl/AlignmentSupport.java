package com.qxotic.format.safetensors.impl;

import com.qxotic.format.safetensors.SafetensorsFormatException;
import java.util.Map;

final class AlignmentSupport {
    static final int DEFAULT_VALUE = 1;
    static final String KEY = "__alignment__";

    private AlignmentSupport() {}

    static boolean isValid(int alignment) {
        return alignment > 0 && Integer.bitCount(alignment) == 1;
    }

    static int parseMetadataAlignment(Map<String, String> metadata) {
        String raw = metadata.get(KEY);
        if (raw == null) {
            return DEFAULT_VALUE;
        }
        return parse(raw);
    }

    static int parse(String raw) {
        try {
            int alignment = Integer.parseInt(raw);
            if (!isValid(alignment)) {
                throw new IllegalArgumentException(
                        "alignment must be a positive power of 2 but was " + raw);
            }
            return alignment;
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("alignment must be an integer but was " + raw, e);
        }
    }

    static void validateHeaderMetadata(Map<String, String> metadata) {
        if (metadata.containsKey(KEY)) {
            String raw = metadata.get(KEY);
            try {
                parse(raw);
            } catch (IllegalArgumentException e) {
                throw new SafetensorsFormatException("Invalid __alignment__: " + e.getMessage(), e);
            }
        }
    }
}
