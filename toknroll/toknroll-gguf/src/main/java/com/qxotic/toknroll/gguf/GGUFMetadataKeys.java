package com.qxotic.toknroll.gguf;

import com.qxotic.format.gguf.GGUF;
import java.util.Locale;

final class GGUFMetadataKeys {
    static final String MODEL = "tokenizer.ggml.model";
    static final String PRE = "tokenizer.ggml.pre";
    static final String TOKENS = "tokenizer.ggml.tokens";
    static final String MERGES = "tokenizer.ggml.merges";
    static final String SCORES = "tokenizer.ggml.scores";
    static final String TOKEN_TYPE = "tokenizer.ggml.token_type";

    private GGUFMetadataKeys() {}

    /**
     * The tokenizer model, which every GGUF holding a tokenizer names. A GGUF without one is
     * another kind of file - a speech model, a vision projector - and says which in its
     * architecture.
     */
    static String requireModel(GGUF gguf) {
        String model = key(gguf, MODEL);
        if (model == null) {
            String architecture = key(gguf, "general.architecture");
            boolean projector = "clip".equals(architecture);
            throw new IllegalArgumentException(
                    (architecture == null ? "this GGUF" : "this " + architecture + " GGUF")
                            + (projector ? ", a vision projector," : "")
                            + " holds no tokenizer (no "
                            + MODEL
                            + ")"
                            + (projector ? ": its model's GGUF does" : ""));
        }
        return model;
    }

    static String key(GGUF gguf, String key) {
        String value = gguf.getValueOrDefault(String.class, key, null);
        if (value == null) {
            return null;
        }
        String trimmed = value.trim();
        if (trimmed.isEmpty()) {
            return null;
        }
        return normalizeKey(trimmed, key);
    }

    static String normalizeKey(String key, String fieldName) {
        if (key == null || key.isBlank()) {
            throw new IllegalArgumentException(fieldName + " must not be blank");
        }
        return key.trim().toLowerCase(Locale.ROOT);
    }
}
